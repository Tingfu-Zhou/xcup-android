#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
bench/tools/train_check.py — 音频分类器训练数据与导出模型的体检

子命令:
  split   检查切好的训练片段（test.py 产出，按 000train/000val/000test/001train… 分目录）：
          · 同一个源视频的片段是否同时出现在 train / val / test（数据泄漏：相邻 2 秒片段几乎一样，
            测试准确率会虚高，早停也会挑错模型）
          · 片段格式是否都是 16kHz 单声道 16-bit、正好 32000 点（中断残留的 *_full.wav 会混进来）
          · 各类的静音片段比例、响度分布（某一类明显更响/更多静音，模型可能学的是响度而不是内容）
  export  检查导出是否一致（需要训练时的 TensorFlow 环境）：
          · 训练用的 YAMNet SavedModel 与 app 里的 yamnet.tflite，embedding 是否一致
          · 训练出的 Keras 分类器（.h5）与 app 里的 yamnet_finetuned.tflite，输出是否一致

用法（Windows 把 python3 换成 python）:
  python3 bench/tools/train_check.py split D:/yunpan/训练/音频
  python3 bench/tools/train_check.py export --saved-model C:/Users/tingf/Desktop/yamnet-tensorflow2-yamnet-v1 \\
          --h5 yamnet_finetuned.h5 --wavs D:/yunpan/训练/音频/000test

片段文件名按 test.py 的规则 "<源视频名>_part<序号>.wav" 还原源视频；对不上规则的文件单独列出。
"""
import argparse
import glob
import os
import unicodedata
import re
import sys
import wave
from collections import defaultdict

try:
    import numpy as np
except ImportError:
    np = None

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
ASSETS = os.path.join(ROOT, 'app', 'src', 'main', 'assets')
CLASS_NAMES = {'000': 'sex', '001': 'oral', '002': 'noise(剧情)'}
SPLITS = ('train', 'val', 'test')
SILENT_DBFS = -50.0
PART_RE = re.compile(r'^(.*)_part\d+$')


def source_of(path):
    stem = os.path.splitext(os.path.basename(path))[0]
    m = PART_RE.match(stem)
    return m.group(1) if m else None


def read_wav(path):
    with wave.open(path, 'rb') as w:
        sr, ch, sw, n = w.getframerate(), w.getnchannels(), w.getsampwidth(), w.getnframes()
        raw = w.readframes(n)
    x = None
    if sw == 2:
        x = np.frombuffer(raw, '<i2').astype(np.float32) / 32768.0
        if ch > 1:
            x = x.reshape(-1, ch).mean(axis=1)
    return sr, ch, sw, n, x


def _pad(s, n):
    w = sum(2 if unicodedata.east_asian_width(c) in 'WF' else 1 for c in s)
    return s + ' ' * max(0, n - w)


def _pct(a, b):
    return '%5.1f%%' % (100.0 * a / b) if b else '   - '


def cmd_split(a):
    if np is None:
        sys.exit('需要 numpy')
    dirs = {}
    for d in sorted(os.listdir(a.root)):
        m = re.match(r'^(\d{3})(train|val|test)$', d)
        if m and os.path.isdir(os.path.join(a.root, d)):
            dirs[(m.group(1), m.group(2))] = os.path.join(a.root, d)
    if not dirs:
        sys.exit('%s 下没找到 000train / 000val / 000test 这样的目录' % a.root)
    print('找到 %d 个目录：%s\n' % (len(dirs), ' '.join(os.path.basename(v) for v in dirs.values())))

    where = defaultdict(set)            # 源视频 -> {(类, split)}
    stats = {}
    bad_fmt, unnamed = [], []
    for (cls, sp), d in sorted(dirs.items()):
        files = sorted(glob.glob(os.path.join(d, '*.wav')))
        srcs, dbfs = set(), []
        for i, f in enumerate(files):
            if a.limit and i >= a.limit:
                break
            s = source_of(f)
            if s is None:
                unnamed.append(f)
            else:
                srcs.add(s)
                where[s].add((cls, sp))
            try:
                sr, ch, sw, n, x = read_wav(f)
            except (wave.Error, EOFError) as e:
                bad_fmt.append((f, '读不了: %s' % e))
                continue
            if (sr, ch, sw, n) != (16000, 1, 2, 32000):
                bad_fmt.append((f, '%dHz %d声道 %dbit %d点（%.2f 秒）' % (sr, ch, sw * 8, n, n / float(sr or 1))))
            if x is not None and len(x):
                rms = float(np.sqrt(np.mean(x.astype(np.float64) ** 2)))
                dbfs.append(20 * np.log10(max(rms, 1e-9)))
        stats[(cls, sp)] = (len(files), len(srcs), np.array(dbfs))
        sys.stdout.write('  读完 %-10s %6d 个片段\n' % (os.path.basename(d), len(files)))
    print()

    print('══ 各目录 ══')
    print('  ' + ''.join(_pad(h, w) for h, w in (('目录', 11), ('片段', 9), ('源视频', 9), ('静音片段', 11),
                                                    ('响度中位数', 15), ('响度 10%~90%', 16))))
    for (cls, sp), (nf, ns, db) in sorted(stats.items()):
        sil = int((db < SILENT_DBFS).sum()) if len(db) else 0
        q = np.percentile(db, [10, 50, 90]) if len(db) else [np.nan] * 3
        print('  %-11s%-9d%-9d%-11s%-15s%.1f ~ %.1f' % (cls + sp, nf, ns, _pct(sil, len(db)).strip(),
                                                       '%.1f dBFS' % q[1], q[0], q[2]))
    for cls in sorted(set(c for c, _ in stats)):
        n = sum(stats[k][0] for k in stats if k[0] == cls)
        print('  %s = %s：共 %d 个片段' % (cls, CLASS_NAMES.get(cls, '?'), n))
    total = sum(v[0] for v in stats.values())
    print('  合计 %d 个片段；各类占比 %s' % (total, '  '.join(
        '%s %s' % (CLASS_NAMES.get(c, c), _pct(sum(stats[k][0] for k in stats if k[0] == c), total))
        for c in sorted(set(c for c, _ in stats)))))

    print('\n══ 数据泄漏：同一源视频出现在多个 split ══')
    leak = dict((s, w) for s, w in where.items() if len(set(sp for _, sp in w)) > 1)
    if not leak:
        print('  ✓ 没有：每个源视频只在 train / val / test 之一里出现（按视频切分，正确）')
    else:
        n_src = len(where)
        print('  ✗ %d / %d 个源视频（%s）的片段同时出现在多个 split —— 很可能是按片段随机切分的。'
              % (len(leak), n_src, _pct(len(leak), n_src).strip()))
        print('    相邻 2 秒片段几乎一样，验证/测试准确率会虚高，早停与"最佳权重"也是按泄漏的验证集挑的。')
        print('    应该按源视频切：同一个视频的所有片段只进 train、val、test 之一。例子：')
        for s, w in sorted(leak.items())[:a.show]:
            print('      %-40s %s' % (s[:40], ' '.join(sorted(c + sp for c, sp in w))))
        by = defaultdict(int)
        for w in leak.values():
            for _, sp in w:
                by[sp] += 1
        print('    涉及：%s' % '  '.join('%s %d 个视频' % (sp, by[sp]) for sp in SPLITS if by[sp]))

    multi = [s for s, w in where.items() if len(set(c for c, _ in w)) > 1]
    print('\n══ 同一源视频贡献了几个类 ══')
    print('  %d / %d 个源视频同时有两类以上的片段。' % (len(multi), len(where)))
    print('  （剧情片段和动作片段来自同一批视频是好事：模型只能靠内容区分，不能靠"哪部片/哪种录音"区分。'
          '这个比例很低时要留意捷径学习。）')

    if bad_fmt:
        print('\n══ 格式不对的片段（%d 个）══' % len(bad_fmt))
        for f, why in bad_fmt[:a.show]:
            print('  %s  %s' % (f, why))
        print('  （*_full.wav 是 test.py 中断后残留的整段音频，会被训练脚本当成一个样本读进去，应删除）')
    else:
        print('\n  ✓ 所有片段都是 16kHz 单声道 16-bit、32000 点')
    if unnamed:
        print('\n  ⚠ %d 个文件名不符合 "<源视频>_part<序号>.wav"，无法判断来源，未计入泄漏检查，例如 %s'
              % (len(unnamed), os.path.basename(unnamed[0])))


def cmd_export(a):
    try:
        import tensorflow as tf
    except ImportError:
        sys.exit('export 需要 TensorFlow（在训练用的环境里跑）')
    files = sorted(glob.glob(os.path.join(a.wavs, '*.wav')))[:a.n]
    if not files:
        sys.exit('%s 下没有 wav' % a.wavs)
    serve = tf.saved_model.load(a.saved_model).signatures['serving_default']
    try:
        from ai_edge_litert.interpreter import Interpreter
    except ImportError:
        Interpreter = tf.lite.Interpreter
    yt = Interpreter(model_path=a.yamnet_tflite)
    yin = yt.get_input_details()[0]
    yt.resize_tensor_input(yin['index'], [32000])
    yt.allocate_tensors()
    emb_out = next(o['index'] for o in yt.get_output_details() if o['shape'][-1] == 1024)
    ct = Interpreter(model_path=a.classifier_tflite)
    ct.allocate_tensors()
    cin, cout = ct.get_input_details()[0]['index'], ct.get_output_details()[0]['index']
    keras = tf.keras.models.load_model(a.h5, compile=False) if a.h5 else None

    d_emb, d_cls_keras, agree = [], [], 0
    for f in files:
        wav, _ = tf.audio.decode_wav(tf.io.read_file(f), desired_channels=1)   # 与 FT.py 相同
        x = tf.squeeze(wav, axis=-1).numpy()
        if len(x) != 32000:
            continue
        e_train = serve(waveform=tf.constant(x))['output_1'].numpy()           # 训练时的 embedding
        yt.set_tensor(yin['index'], x.astype(np.float32))
        yt.invoke()
        e_app = yt.get_tensor(emb_out)                                         # app 里的 embedding
        d_emb.append(float(np.abs(e_train - e_app).max() / max(1e-6, np.abs(e_train).max())))   # 相对差
        m = e_app.mean(axis=0, keepdims=True).astype(np.float32)
        ct.set_tensor(cin, m)
        ct.invoke()
        p_app = ct.get_tensor(cout)[0]
        if keras is not None:                       # 同一个输入喂两边，只比分类器本身
            p_keras = keras.predict(m, verbose=0)[0]
            d_cls_keras.append(float(np.abs(p_app - p_keras).max()))
            agree += int(p_app.argmax() == p_keras.argmax())
    n = len(d_emb)
    print('比较了 %d 个片段' % n)
    print('  YAMNet：训练用 SavedModel vs app 的 yamnet.tflite，embedding 最大相对差 %.2e（中位 %.2e）'
          % (max(d_emb), float(np.median(d_emb))))
    print('    %s' % ('✓ 一致' if max(d_emb) < 1e-3 else '✗ 不一致：app 里的 YAMNet 和训练时用的不是同一个模型/版本'))
    if keras is not None:
        print('  分类器：%s vs app 的 yamnet_finetuned.tflite，概率最大差 %.2e，argmax 一致 %d/%d'
              % (os.path.basename(a.h5), max(d_cls_keras), agree, n))
        print('    %s' % ('✓ 一致' if max(d_cls_keras) < 1e-3 else
                         '✗ 不一致：app 里的分类器不是这个 .h5 导出的（或导出时出了问题）'))


def main():
    if hasattr(sys.stdout, 'reconfigure'):
        try:
            sys.stdout.reconfigure(errors='replace')
        except (AttributeError, ValueError):
            pass
    ap = argparse.ArgumentParser(description='音频分类器训练数据与导出模型的体检')
    sub = ap.add_subparsers(dest='cmd')
    p = sub.add_parser('split', help='检查训练/验证/测试片段：泄漏、格式、静音与响度')
    p.add_argument('root', help='包含 000train / 000val / 000test … 的目录')
    p.add_argument('--show', type=int, default=10)
    p.add_argument('--limit', type=int, default=0, help='每个目录最多读几个文件（调试用，默认全读）')
    p = sub.add_parser('export', help='检查 SavedModel / .h5 与 app 里的 tflite 是否一致（需要 TensorFlow）')
    p.add_argument('--saved-model', required=True, help='训练时用的 YAMNet SavedModel 目录')
    p.add_argument('--h5', help='训练脚本保存的 yamnet_finetuned.h5')
    p.add_argument('--wavs', required=True, help='拿来比对的 2 秒片段目录，如 000test')
    p.add_argument('--n', type=int, default=50)
    p.add_argument('--yamnet-tflite', default=os.path.join(ASSETS, 'yamnet.tflite'))
    p.add_argument('--classifier-tflite', default=os.path.join(ASSETS, 'yamnet_finetuned.tflite'))
    a = ap.parse_args()
    if a.cmd == 'split':
        cmd_split(a)
    elif a.cmd == 'export':
        cmd_export(a)
    else:
        ap.print_help()


if __name__ == '__main__':
    main()
