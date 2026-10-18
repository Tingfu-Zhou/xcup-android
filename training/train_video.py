#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
training/train_video.py — 视频模型训练（默认用 P1 定下的 E1d 配方，见 training/RECIPE.md），替代 train_mobilenetv3_tap_3class.py

模型结构不变（MobileNetV3Small、12 帧时间平均、160×160、类别 [normal_plot, oral, sex]），换的是训练方法：
  · 坏片段：读 data_check.py scan 写的 bad_clips.txt 剔除；训练中读帧失败就跳过并报出来，不再用全黑帧顶替
  · 采样：类别 → 源视频 → 片段 均衡（源视频名从片段文件名取），不再用 class_weight
  · 增强：片段级（同一片段的所有帧用同一组参数）：裁剪面积 ≥0.7、水平翻转、轻度颜色、少量模糊 / JPEG
  · 选模型：验证集 12 帧的 p_oral+p_sex AUC（val_auc_do），不再用 val_acc
  · 导出：与 app 现用模型同结构的 12 帧 tflite，用真实验证片段核对 Keras / TFLite，并检查 bench 能直接用

预设（--preset；其余参数都可以单独覆盖，例如 --preset e1d --seed 7）。P1 比完（2026-10-18）定下的配方是 e1d，也是默认值，
配方的完整说明见 training/RECIPE.md
  e1d     （默认）e1a + 训练时每个片段只抽 4 帧（TSN：等分 4 段，每段随机一帧）；验证、选模型、导出都是 12 帧
  e1a     新基线：上面几条，微调设置不变（stage 2 解冻最后 30 层，BN 冻结，Adam lr 1e-5），训练用 12 帧
  e1b     e1a + 加深微调：从中段 expanded_conv_6 起解冻（BN 仍冻结），AdamW lr 1e-4，cosine，1 个 epoch warmup
          （P1 里 lr 1e-4 训练不稳，没有采用）
  e1c     e1a + label smoothing 0.05（P1 里端到端变差，不采用）
  legacy  旧脚本的做法（打乱采样 + class_weight、不增强、按 val_acc 选），只换了读帧。可选：量"同样的配方重训一次"的波动

用法（训练机，仓库根目录）
  python3 training/train_video.py --data_dir ~/ml/movinet/movinet_clips --out_dir ~/ml/movinet/runs/v2 --dry_run
  python3 training/train_video.py --data_dir ~/ml/movinet/movinet_clips --out_dir ~/ml/movinet/runs/v2   # 默认 e1d
  python3 training/train_video.py --export_only ~/ml/movinet/runs/v2          # 只从最好的权重重新导出 tflite

--dry_run：每个 stage 1 个 epoch × 5 步、验证 / 测试各取 60 个片段，几分钟内把读数据 → 训练 → 选模型 → 导出 → 核对整条走一遍
（输出写到 <out_dir>_dry），正式跑之前先用它确认环境没问题。

输出（out_dir）
  train.log                     本脚本的日志（每个 epoch 的验证指标、读帧失败、导出核对）
  config.json                   全部参数、版本、数据规模
  train_stage{1,2}_log.csv      每个 epoch：loss、acc、val_auc_do、val_acc、val_film_acc、val_loss、lr
  best_stage{1,2}.weights.h5    各 stage 里 val_auc_do 最好的一轮
  <name>.tflite / <name>.keras  导出的模型（与 app 现用的同结构，可直接 baseline.py rescore --model）
  summary.json                  最终的验证 / 测试指标、导出核对 —— 训练完把它和 train.log 发出来
"""
import argparse
import io
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import train_utils as U     # noqa: E402
import numpy as np          # noqa: E402
import keras                # noqa: E402

PRESETS = {
    'e1a': dict(sampler='film', film_alpha=0.5, augment='mild', select='auc_do', class_weight=False,
                train_frames=12, label_smoothing=0.0,
                stage1_epochs=10, stage1_lr=1e-3, stage1_patience=4,
                stage2_epochs=20, stage2_lr=1e-5, stage2_patience=6, unfreeze='last:30',
                optimizer='adam', schedule='constant', warmup_epochs=0.0, weight_decay=0.0, plateau=True),
}
PRESETS['e1b'] = dict(PRESETS['e1a'], unfreeze='from:expanded_conv_6', optimizer='adamw', stage2_lr=1e-4,
                      schedule='cosine', warmup_epochs=1.0, weight_decay=0.004, plateau=False)
PRESETS['e1c'] = dict(PRESETS['e1a'], label_smoothing=0.05)
PRESETS['e1d'] = dict(PRESETS['e1a'], train_frames=4)
PRESETS['legacy'] = dict(PRESETS['e1a'], sampler='shuffle', augment='none', select='acc', class_weight=True)


class Log(object):
    """打印并追加到 out_dir/train.log。"""

    def __init__(self, path):
        self.path = path

    def __call__(self, msg=''):
        print(msg, flush=True)
        with io.open(self.path, 'a', encoding='utf-8') as fh:
            fh.write(msg + '\n')


class EpochInfo(keras.callbacks.Callback):
    """每个 epoch 的用时和累计读帧失败数。"""

    def __init__(self, failures, log):
        super().__init__()
        self.failures, self.log, self.t0 = failures, log, None

    def on_epoch_begin(self, epoch, logs=None):
        self.t0 = time.time()

    def on_epoch_end(self, epoch, logs=None):
        if logs is not None:
            logs['epoch_min'] = (time.time() - self.t0) / 60.0
            logs['read_failures'] = self.failures.total()


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description='视频模型训练（P1，详见文件头）',
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data_dir', help='movinet_clips 目录（下面是 train/val/test/<类别>）')
    ap.add_argument('--out_dir', help='输出目录（每个实验一个，如 runs/e1a）')
    ap.add_argument('--preset', default='e1d', choices=sorted(PRESETS),
                    help='训练配方（默认 e1d = P1 定下的配方，见 training/RECIPE.md）')
    ap.add_argument('--name', help='导出的模型文件名（默认 = out_dir 的名字）')
    ap.add_argument('--bad_list', default='auto',
                    help='坏片段清单（默认 <data_dir>/bad_clips.txt，没有就先跑 data_check.py scan；none = 不剔除）')
    ap.add_argument('--dry_run', action='store_true', help='几分钟把整条流程走一遍（输出到 <out_dir>_dry）')
    ap.add_argument('--export_only', metavar='RUN_DIR', help='不训练，只从 RUN_DIR 里最好的权重重新导出、核对')
    ap.add_argument('--init', help='训练前先读这个权重（.weights.h5），例如复用 e1a 的 best_stage1.weights.h5 跳过 stage 1')
    ap.add_argument('--backbone', default='small', choices=sorted(U.BACKBONES))
    ap.add_argument('--weights', default='imagenet', help='backbone 的初始权重：imagenet（默认）/ none（随机，只用于测试）')
    ap.add_argument('--image_size', type=int, default=U.IMAGE_SIZE)
    ap.add_argument('--batch_size', type=int, default=8)
    ap.add_argument('--val_batch', type=int, default=16)
    ap.add_argument('--steps_per_epoch', type=int, help='默认 训练片段数 // batch_size（每个 epoch 的样本数与旧脚本相同）')
    ap.add_argument('--workers', type=int, default=max(1, min(8, (os.cpu_count() or 2) - 1)), help='读帧线程数')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--dropout', type=float, default=0.5)
    ap.add_argument('--export_check_clips', type=int, default=48, help='导出核对用的验证片段数')
    ap.add_argument('--verbose', type=int, default=1, help='Keras 的进度显示：1 进度条，2 每个 epoch 一行')
    # 预设里的项（不给就用预设的值）
    ap.add_argument('--sampler', choices=('film', 'shuffle'))
    ap.add_argument('--film_alpha', type=float, help='源视频的采样权重 = 片段数^alpha（0 每部片等权，1 按片段数）')
    ap.add_argument('--augment', choices=('mild', 'none'))
    ap.add_argument('--select', choices=('auc_do', 'acc'), help='按什么选 checkpoint / 早停')
    ap.add_argument('--class_weight', type=int, choices=(0, 1))
    ap.add_argument('--train_frames', type=int, help='训练时每个片段抽几帧（验证、导出始终 12 帧）')
    ap.add_argument('--label_smoothing', type=float)
    ap.add_argument('--stage1_epochs', type=int)
    ap.add_argument('--stage1_lr', type=float)
    ap.add_argument('--stage1_patience', type=int)
    ap.add_argument('--stage2_epochs', type=int)
    ap.add_argument('--stage2_lr', type=float)
    ap.add_argument('--stage2_patience', type=int)
    ap.add_argument('--unfreeze', help='stage 2 解冻哪些层：last:N / from:<层名前缀> / all')
    ap.add_argument('--optimizer', choices=('adam', 'adamw'))
    ap.add_argument('--schedule', choices=('constant', 'cosine'))
    ap.add_argument('--warmup_epochs', type=float)
    ap.add_argument('--weight_decay', type=float)
    ap.add_argument('--plateau', type=int, choices=(0, 1), help='stage 2 用 ReduceLROnPlateau（只在 constant 时有效）')
    a = ap.parse_args(argv)
    for k, v in PRESETS[a.preset].items():
        if getattr(a, k, None) is None:
            setattr(a, k, v)
    a.class_weight, a.plateau = bool(a.class_weight), bool(a.plateau)
    if not a.export_only and not (a.data_dir and a.out_dir):
        ap.error('训练要给 --data_dir 和 --out_dir')
    if a.export_only:
        return a
    a.data_dir = os.path.abspath(os.path.expanduser(a.data_dir))
    a.out_dir = os.path.abspath(os.path.expanduser(a.out_dir))
    if a.dry_run:
        a.out_dir = a.out_dir.rstrip('/\\') + '_dry'
        a.stage1_epochs = min(a.stage1_epochs, 1)
        a.stage2_epochs = min(a.stage2_epochs, 1)
        a.steps_per_epoch = 5
        a.export_check_clips = 12
    a.name = a.name or os.path.basename(a.out_dir.rstrip('/\\'))
    return a


def class_weights(labels):
    n = np.bincount(labels, minlength=U.NUM_CLASSES).astype(np.float64)
    return dict((i, float(len(labels) / (U.NUM_CLASSES * n[i]))) for i in range(U.NUM_CLASSES) if n[i])


def best_of(csv_path, key):
    """CSVLogger 的日志里 key 最大的一轮 → (epoch 从 1 起, 值)。"""
    if not os.path.exists(csv_path):
        return None, float('nan')
    import csv
    with io.open(csv_path, encoding='utf-8') as fh:
        rows = list(csv.DictReader(fh))
    vals = [(float(r[key]), int(r['epoch']) + 1) for r in rows if r.get(key) not in (None, '', 'nan')]
    if not vals:
        return None, float('nan')
    v, e = max(vals)
    return e, v


def film_overlap(samples):
    """各 split 之间同类别、同名的源视频（train/val/test 应按片子划分）。"""
    seen, out = {}, set()
    for s in samples:
        for k in set(s.film_keys()):
            if k in seen and seen[k] != s.split:
                out.add((k.split('|', 1)[1], seen[k], s.split))
            seen.setdefault(k, s.split)
    return sorted(out)


def run_stage(stage, a, net, m_train, m_eval, train_ds, steps, va, val_ds, failures, log, cw):
    monitor = 'val_auc_do' if a.select == 'auc_do' else 'val_acc'
    ck = os.path.join(a.out_dir, 'best_stage%d.weights.h5' % stage)
    if stage == 1:
        U.set_trainable(net.backbone, 'none')
        opt = U.make_optimizer('adam', a.stage1_lr)
        epochs, patience, plateau = a.stage1_epochs, a.stage1_patience, False
        what = '只训线性层（backbone 冻结），Adam lr %g' % a.stage1_lr
    else:
        start, first = U.set_trainable(net.backbone, a.unfreeze)
        epochs, patience = a.stage2_epochs, a.stage2_patience
        opt = U.make_optimizer(a.optimizer, a.stage2_lr, steps, epochs, a.schedule, a.warmup_epochs, a.weight_decay)
        plateau = a.plateau and a.schedule == 'constant'
        what = '解冻 %s（backbone 第 %d/%d 层 %s 起，BN 冻结），%s lr %g%s%s%s' % (
            a.unfreeze, start, len(net.backbone.layers), first, a.optimizer, a.stage2_lr,
            '，cosine + %g epoch warmup' % a.warmup_epochs if a.schedule == 'cosine' else '',
            '，weight decay %g' % a.weight_decay if a.optimizer == 'adamw' else '',
            '，ReduceLROnPlateau' if plateau else '')
    U.compile_model(m_train, opt, a.label_smoothing)
    log('\n══ Stage %d：%s ══' % (stage, what))
    log('  可训练参数 %d（backbone %d）；每个 epoch %d 步 × %d 个片段 × %d 帧；按 %s 选，早停 patience %d' % (
        U.n_params(m_train.trainable_weights), U.n_params(net.backbone.trainable_weights), steps, a.batch_size,
        a.train_frames, monitor, patience))
    cbs = [U.ValMetrics(m_eval, va, val_ds, log=log),
           keras.callbacks.ModelCheckpoint(ck, monitor=monitor, mode='max', save_best_only=True,
                                           save_weights_only=True, verbose=0),
           keras.callbacks.EarlyStopping(monitor=monitor, mode='max', patience=patience, verbose=1)]
    if plateau:
        cbs.append(keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=3, min_lr=1e-7,
                                                     verbose=1))
    cbs += [U.LrLogger(), EpochInfo(failures, log),
            keras.callbacks.CSVLogger(os.path.join(a.out_dir, 'train_stage%d_log.csv' % stage))]
    t0 = time.time()
    m_train.fit(train_ds, epochs=epochs, steps_per_epoch=steps, callbacks=cbs, class_weight=cw,
                verbose=a.verbose, shuffle=False)
    e, v = best_of(os.path.join(a.out_dir, 'train_stage%d_log.csv' % stage), monitor)
    log('  stage %d 用时 %.0f 分钟；最好的一轮：epoch %s，%s = %.4f' % (stage, (time.time() - t0) / 60.0, e, monitor, v))
    return ck, monitor, e, v


def export_and_check(net, out_dir, name, va, n_clips, log):
    tfl = os.path.join(out_dir, name + '.tflite')
    log('\n══ 导出 %s ══' % tfl)
    m_exp = U.export_tflite(net, tfl, log=log)
    try:
        m_exp.save(os.path.join(out_dir, name + '.keras'))
    except Exception as e:                           # noqa: BLE001  存不了 .keras 不影响 tflite
        log('  [WARN] 存 %s.keras 失败：%s' % (name, e))
    st = U.tflite_structure(tfl)
    log('  结构：%s；%d 个算子，%.1f MB；TF %s 转换，min_runtime_version %s' % (
        '✓ 与 app 现用模型相同（1 个 PACK × 12 + 1 个 FULLY_CONNECTED，bench 可直接用）' if st['ok']
        else '✗ ' + '；'.join(st['problems']), st['n_ops'], st['size'] / 1e6, st.get('converter_tf', '?'),
        st.get('min_runtime_version', '?')))
    idx = U.balanced_pick(va, n_clips, seed=1)
    clips = [U.read_clip(p) for p in va.paths[idx]]
    chk = U.check_tflite(m_exp, tfl, clips)
    log('  %s Keras（CPU float32）/ TFLite 一致性（%d 个真实验证片段）：概率最大差 %.1e，bench 的逐帧 logit 算法重算差 %.1e，'
        'argmax 一致 %.0f%%，app 判定一致 %.0f%%' % ('✓' if chk['ok'] else '✗', chk['n'], chk['max_diff'],
                                                   chk['bench_diff'], 100 * chk['argmax_agree'],
                                                   100 * chk['do_agree']))
    if 'gpu_diff' in chk:
        log('    （GPU 上的 Keras 与 TFLite 差 %.1e：GPU 默认用 TF32 算卷积，精度低，只作参考，不参与判断）' % chk['gpu_diff'])
    res = dict(tflite=tfl, structure=st, parity=chk)
    with io.open(os.path.join(out_dir, 'export_check.json'), 'w', encoding='utf-8') as fh:
        json.dump(res, fh, ensure_ascii=False, indent=1, default=str)
    return res


def export_only(run_dir):
    run_dir = os.path.abspath(os.path.expanduser(run_dir))
    with io.open(os.path.join(run_dir, 'config.json'), encoding='utf-8') as fh:
        cfg = json.load(fh)
    a = cfg['args']
    log = Log(os.path.join(run_dir, 'train.log'))
    net = U.VideoNet(a['backbone'], a['image_size'], a['dropout'], weights=None)
    ck = cfg.get('final_weights') or os.path.join(run_dir, 'best_stage2.weights.h5')
    U.load_weights(net.clip_model(a['train_frames']), ck)
    log('从 %s 重新导出' % ck)
    va = U.collect_samples(a['data_dir'], 'val', cfg.get('bad_list'))
    export_and_check(net, run_dir, a['name'], va, a['export_check_clips'], log)


def main(argv=None):
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(errors='replace')
        except Exception:                            # noqa: BLE001
            pass
    a = parse_args(argv)
    if a.export_only:
        return export_only(a.export_only)
    os.makedirs(a.out_dir, exist_ok=True)
    log = Log(os.path.join(a.out_dir, 'train.log'))
    t_start = time.time()
    log('\n████ train_video.py %s  preset=%s%s  → %s ████' % (time.strftime('%Y-%m-%d %H:%M'), a.preset,
                                                            '（dry run）' if a.dry_run else '', a.out_dir))
    keras.utils.set_random_seed(a.seed)
    U.setup_gpu(log)
    ver = U.versions()
    log('[INFO] 版本：%s' % '，'.join('%s %s' % kv for kv in ver.items()))
    if not ver['tensorflow'].startswith(U.REF_TF):
        log('[WARN] 现用模型是 TF %s.0 导出的（app 的运行时是 TFLite 2.13）；这次用 TF %s，导出后要在 app 里实际加载一次'
            % (U.REF_TF, ver['tensorflow']))

    # ---------------------------------------------------------------- 数据
    bad = a.bad_list
    if bad == 'auto':
        bad = os.path.join(a.data_dir, 'bad_clips.txt')
        if not os.path.exists(bad):
            sys.exit('没有 %s：先跑 python3 training/data_check.py scan --data_dir %s（或 --bad_list none 跳过）'
                     % (bad, a.data_dir))
    elif bad == 'none':
        bad = None
    tr, va, te = [U.collect_samples(a.data_dir, sp, bad) for sp in ('train', 'val', 'test')]
    for s in (tr, va, te):
        log('[数据] ' + s.describe())
    if not len(tr) or not len(va):
        sys.exit('训练集或验证集是空的：检查 --data_dir')
    ov = film_overlap([tr, va, te])
    if ov:
        log('[WARN] %d 个源视频（同类别同名）出现在不止一个 split 里，例如 %s；train/val/test 应按片子划分'
            % (len(ov), '，'.join('%s（%s/%s）' % x for x in ov[:3])))
    if a.dry_run:
        va = va.subset(U.balanced_pick(va, 60))
        te = te.subset(U.balanced_pick(te, 60)) if len(te) else te
    steps = a.steps_per_epoch or max(1, len(tr) // a.batch_size)

    # ---------------------------------------------------------------- 模型与数据流
    net = U.VideoNet(a.backbone, a.image_size, a.dropout, weights=None if a.weights == 'none' else a.weights)
    m_train, m_eval = net.clip_model(a.train_frames), net.clip_model(U.EVAL_FRAMES)
    if a.init:
        U.load_weights(m_train, os.path.expanduser(a.init))
        log('[INFO] 初始权重：%s' % a.init)
    if a.sampler == 'film':
        sampler = U.BalancedSampler(tr, a.film_alpha, seed=a.seed)
    else:
        sampler = U.ShuffleSampler(len(tr), seed=a.seed)
    log('[数据] ' + sampler.describe())
    failures = U.FailureLog(log)
    aug = U.AUG_MILD if a.augment == 'mild' else None
    train_ds, _ = U.train_dataset(tr, sampler, a.train_frames, a.batch_size, aug, a.image_size, a.seed,
                                  a.workers, failures=failures)
    val_ds = U.eval_dataset(va, U.EVAL_FRAMES, a.val_batch, a.image_size, a.workers)
    cw = class_weights(tr.labels) if a.class_weight else None
    if cw:
        log('[INFO] class_weight：%s' % cw)
    cfg = dict(args=vars(a), versions=ver, bad_list=bad, train=len(tr), val=len(va), test=len(te),
               train_films=len(set(tr.film_keys())), val_films=len(set(va.film_keys())),
               steps_per_epoch=steps, augment=U.AUG_MILD if aug else None, created=time.strftime('%Y-%m-%d %H:%M'))
    with io.open(os.path.join(a.out_dir, 'config.json'), 'w', encoding='utf-8') as fh:
        json.dump(cfg, fh, ensure_ascii=False, indent=1)
    with io.open(os.path.join(a.out_dir, 'class_names.json'), 'w', encoding='utf-8') as fh:
        json.dump(U.CLASS_NAMES, fh)                 # 与旧脚本相同（app 的类别顺序）

    # ---------------------------------------------------------------- 训练
    stages = {}
    for stage, epochs in ((1, a.stage1_epochs), (2, a.stage2_epochs)):
        if epochs <= 0:
            continue
        ck, monitor, e, v = run_stage(stage, a, net, m_train, m_eval, train_ds, steps, va, val_ds, failures, log, cw)
        if os.path.exists(ck):
            U.load_weights(m_train, ck)       # 接着用这个 stage 里最好的一轮（stage 2 从 stage 1 最好的权重开始）
        stages[stage] = dict(weights=ck, monitor=monitor, best_epoch=e, best=v)
    if not stages:
        sys.exit('stage1_epochs 和 stage2_epochs 都是 0，没有可训练的')
    # 两个 stage 里验证指标最好的那一轮（一样好取后面的 stage）
    final = max(stages.values(), key=lambda s: ((s['best'] if s['best'] == s['best'] else -1), s['weights']))
    if not os.path.exists(final['weights']):         # 验证指标一直是 nan（例如验证集缺一类）时一轮都没存
        sys.exit('没有存下任何 checkpoint（%s 一直是 nan？检查验证集三类是不是都有）' % final['monitor'])
    U.load_weights(m_train, final['weights'])
    cfg['final_weights'] = final['weights']
    with io.open(os.path.join(a.out_dir, 'config.json'), 'w', encoding='utf-8') as fh:
        json.dump(cfg, fh, ensure_ascii=False, indent=1)
    log('\n最终用 %s（%s = %.4f）' % (os.path.basename(final['weights']), final['monitor'], final['best']))

    # ---------------------------------------------------------------- 验证 / 测试（12 帧）
    summary = dict(preset=a.preset, name=a.name, stages=stages, final_weights=final['weights'],
                   read_failures=dict(failures.count))
    for split, s in (('val', va), ('test', te)):
        if not len(s):
            continue
        vm = U.ValMetrics(m_eval, s, U.eval_dataset(s, U.EVAL_FRAMES, a.val_batch, a.image_size, a.workers),
                          prefix=split, log=log)
        m, probs, ok = vm.evaluate()
        worst = U.worst_films(probs[ok], s.labels[ok], s.film_keys()[ok], 10)
        m['worst_films'] = worst
        summary[split] = m
        log('[%s] %s' % (split, U.fmt_metrics(m, split)))
        log('       混淆矩阵（行 = 真值 %s，列 = 预测）：%s' % ('/'.join(U.CLASS_NAMES), m['cm']))
        log(U.fmt_worst(worst, '%s 里最差的 10 部' % split))

    # ---------------------------------------------------------------- 导出
    summary['export'] = export_and_check(net, a.out_dir, a.name, va, a.export_check_clips, log)
    summary['minutes'] = (time.time() - t_start) / 60.0
    with io.open(os.path.join(a.out_dir, 'summary.json'), 'w', encoding='utf-8') as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=1, default=str)
    tfl = summary['export']['tflite']
    log('\n完成，用时 %.0f 分钟。下一步（仓库根目录，在放 13 部原片的电脑上）：' % summary['minutes'])
    log('  python3 bench/tools/baseline.py rescore --videos <原片目录> --model %s --name %s --jobs 4' % (tfl, a.name))
    log('  python3 bench/tools/baseline.py compare exact_crop %s' % a.name)
    log('  python3 bench/tools/frame_diag.py --name exact_crop --name %s' % a.name)
    ok = summary['export']['structure']['ok'] and summary['export']['parity']['ok']
    if not ok:
        log('⚠ 导出核对没过（见上面），先别拿去 rescore')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
