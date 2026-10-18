#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
bench/tools/frame_diag.py — 视频模型的逐帧诊断：同一套录像、同一批帧上，看模型错在哪、错得多确定

数据：baseline.py rescore（video_rescore.py）写的 bench/rescored/video_<名字>_runs_initial|_runs_test/ 下
      <片>_base.csv（每个视频结果的 12 帧时刻 v_fpts）和 <片>_base.vframes.npz（用到的每一帧的逐帧 logit）。
      模型是"逐帧 logit 平均 → softmax"，有了逐帧 logit，换窗口长度、换聚合方式都不用再跑模型。
      phone（手机原始 run）没有逐帧 logit，不支持。
口径：按视频结果算（每个结果一个窗口，约每 0.6 秒一个），标签取出结果那一刻（pos）的真值；other_adult 和 exclude
      不计（同 baseline.py 的 policy=exclude），片头 3 秒不计。p_do = p_oral + p_sex，用逐帧 logit 全精度重算
      （baseline.py 的视频 AUC 是按 tick、用 CSV 里 4 位小数的概率算的，两边会差一点）。

报告
  1. 判别力 AUC（p_do）：窗口长度（单帧 = 窗口最后一帧 / 3 秒 = 现在的 12 帧 / 6 秒 / 12 秒，后两个取 vframes 里
     落在这段时间内的全部帧）和聚合方式（3 秒窗口：logit 平均 = 现在的模型 / 概率平均 / 中位数 / 投票），
     各给 13 部合并、逐片平均、最差片
  2. 置信度分布：max(p_do, p_plot) 分档，每档占多少窗口、错多少、占全部错误多少（"错" = p_do ≥ 0.5 判转，判错）
  3. 成段的高置信度错误：连续 ≥8 秒、每个结果都以 ≥0.9 判错的段（误转 = 剧情判成转，漏转 = 动作判成剧情），
     合计秒数、占视频全部误转 / 漏转时长（按 app 阈值 0.65 / 0.60）的比例；最长的几段列出来
     （时间是窗口结尾 = 出结果时的播放位置，看片时往前多看 3 秒）
  4. 逐片：AUC、按 app 阈值的准确率 / 剧情判转 / 动作判不转、成段错误秒数
  几个变体一起给时并排比（第一个是参照），另外逐个窗口配对：判错 → 判对、判对 → 判错各多少秒，
  以及 v01~v07、v08~v13 两半各自的合并 AUC（只在一半上变好的改动要谨慎）。

用法（仓库根目录；Windows 把 python3 换成 python）
  python3 bench/tools/frame_diag.py --name exact_crop
  python3 bench/tools/frame_diag.py --name exact_crop --name e1d          # 并排比，第一个是参照
  python3 bench/tools/frame_diag.py --name exact_crop,e1d,new1 --segments 20
  python3 bench/tools/frame_diag.py --selftest
报告同时写进 bench/baseline/frame_diag_<名字>[_vs_<名字>…].txt。
"""
import argparse
import glob
import io
import os
import re
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import metrics as M          # noqa: E402
import video_rescore as V    # noqa: E402
import baseline as B         # noqa: E402

try:
    import numpy as np
except ImportError:
    np = None

FILMS = B.FILMS
POLICY = 'exclude'
WINDOWS = (('frame', '单帧', 0), ('3s', '3 秒（现在）', 3000), ('6s', '6 秒', 6000), ('12s', '12 秒', 12000))
AGGS = (('logit', 'logit 平均（现在）'), ('prob', '概率平均'), ('median', '中位数'), ('vote', '投票'))
CONF_BINS = (0.5, 0.6, 0.9, 0.99, 1.0000001)
SEG_CONF = 0.9           # 成段错误：每个结果都以 ≥ 0.9 判错
SEG_MIN_S = 8.0          # 持续 ≥ 8 秒
SEG_GAP_S = 3.0          # 中间夹着的不够格的结果累计不超过 3 秒就连成一段（在 exact_crop 上这样分出 19 段，与 2026-10-16 方案第 3 节一致）
MAX_AGE_MS = 2000        # 一个结果最多管 2 秒（融合循环的新鲜度窗）
HALVES = (('v01~v07', FILMS[:7]), ('v08~v13', FILMS[7:]))


def _need():
    if np is None:
        sys.exit('需要 numpy：pip install numpy')


# ======================================================================== 读数据
def auc(scores, positive):
    """Mann-Whitney AUC（并列算一半），与 metrics._auc 相同，numpy 实现。"""
    scores = np.asarray(scores, np.float64)
    positive = np.asarray(positive, bool)
    n1 = int(positive.sum())
    n0 = len(scores) - n1
    if n1 == 0 or n0 == 0:
        return float('nan')
    _, inv, cnt = np.unique(scores, return_inverse=True, return_counts=True)
    r = (np.cumsum(cnt) - (cnt - 1) / 2.0)[inv]
    return float((r[positive].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def _softmax(z):
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


class Film(object):
    """一部片在一个变体下的全部视频结果。数组按出结果的时刻排序：
       pos（播放位置 ms）、dt（这个结果管多久，秒）、y（1 = 该转，0 = 剧情，-1 = 不计）、lab（真值标签）、
       ts（墙钟，配对用）、p（窗口 p_do，dict：窗口 / 聚合方式 → 数组）、pplot（3 秒窗口的 p_plot）"""


def _run_path(dirs, vid):
    for d in dirs:
        hits = sorted(p for p in glob.glob(os.path.join(d, vid + '*.csv'))
                      if re.match(re.escape(vid) + r'([_.\-]|\.csv$)', os.path.basename(p)))
        base = [p for p in hits if os.path.basename(p) == vid + '_base.csv']
        if base or hits:
            return (base or hits)[0]
    return None


def load_film(vid, run_path, gt_path):
    meta, rs = V.load_results(run_path)
    npz = os.path.splitext(run_path)[0] + '.vframes.npz'
    if not os.path.exists(npz):
        raise RuntimeError('%s 没有逐帧 logit（%s）：用 baseline.py rescore 重算过的变体才有' % (vid, os.path.basename(npz)))
    z = np.load(npz)
    fms = (z['pts_us'] // 1000).astype(np.int64)
    lg = z['logits'].astype(np.float32)
    order = np.argsort(fms, kind='stable')
    fms, lg = fms[order], lg[order]
    idx = dict((int(m), i) for i, m in enumerate(fms))
    _, segs, _, _ = M.read_gt(gt_path)
    rows = []
    for r in rs:
        if r.fpts is None:
            continue
        try:
            ii = [idx[t] for t in r.fpts]
        except KeyError:
            continue
        rows.append((r.pos, r.ts, ii))
    rows.sort()
    f = Film()
    f.vid = vid
    n = len(rows)
    f.pos = np.array([x[0] for x in rows], np.int64)
    f.ts = np.array([x[1] for x in rows], np.int64)
    gap = np.diff(f.pos).astype(np.float64) if n > 1 else np.zeros(0)
    last = float(np.median(gap)) if len(gap) else 500.0
    f.dt = np.minimum(np.append(gap, last), MAX_AGE_MS) / 1000.0
    f.lab = []
    y = np.full(n, -1, np.int8)
    for k, (pos, _, _) in enumerate(rows):
        lab = M.gt_label_at(segs, pos)
        f.lab.append(lab)
        b = M.bin_of(lab, POLICY)
        if pos >= M.WARMUP_MS and b:
            y[k] = 1 if b == 'do' else 0
    f.y = y
    win = np.stack([lg[ii] for _, _, ii in rows]) if n else np.zeros((0, V.NUM_FRAMES, 3), np.float32)
    fp = _softmax(win)                                   # 逐帧概率 [n, 12, 3]
    fdo = fp[..., 1] + fp[..., 2]
    p3 = _softmax(win.mean(axis=1))
    f.p = {('3s', 'logit'): p3[:, 1] + p3[:, 2], ('frame', 'logit'): fdo[:, -1],
           ('3s', 'prob'): fp.mean(axis=1)[:, 1] + fp.mean(axis=1)[:, 2],
           ('3s', 'median'): np.median(fdo, axis=1), ('3s', 'vote'): (fdo >= 0.5).mean(axis=1)}
    f.pplot = p3[:, 0]
    # 长窗口：vframes 里落在 (最后一帧 - L, 最后一帧] 的全部帧，logit 平均
    cum = np.vstack([np.zeros((1, 3)), np.cumsum(lg.astype(np.float64), axis=0)])
    ends = np.array([ii[-1] for _, _, ii in rows], np.int64)
    for key, _, L in WINDOWS:
        if L <= 3000:
            continue
        starts = np.searchsorted(fms, fms[ends] - L, side='right')
        zz = (cum[ends + 1] - cum[starts]) / np.maximum(ends + 1 - starts, 1)[:, None]
        pp = _softmax(zz)
        f.p[(key, 'logit')] = pp[:, 1] + pp[:, 2]
    f.run_path = run_path
    return f


def load_variant(name, films=FILMS):
    if name == B.PHONE:
        sys.exit('phone（手机原始 run）没有逐帧 logit；frame_diag 只看 baseline.py rescore 重算过的变体（如 exact_crop）')
    dirs = B.variant_dirs(name)
    if not any(os.path.isdir(d) for d in dirs):
        sys.exit('找不到变体 %s 的数据：%s\n先运行：python3 bench/tools/baseline.py rescore --videos <原片目录>%s'
                 % (name, '、'.join(B.rel(d) for d in dirs),
                    '' if name == B.DEFAULT_PIPE else ' --model <模型文件> --name %s' % name))
    out, missing = {}, []
    for vid in films:
        rp = _run_path(dirs, vid)
        gp = os.path.join(B.GT_DIR, vid + '.csv')
        if not rp or not os.path.exists(gp):
            missing.append(vid)
            continue
        out[vid] = load_film(vid, rp, gp)
    return out, missing


# ======================================================================== 统计
def _cat(films, key):
    ps, ys, ws = [], [], []
    for f in films.values():
        m = f.y >= 0
        ps.append(f.p[key][m])
        ys.append(f.y[m])
        ws.append(f.dt[m])
    if not ps:
        return np.zeros(0), np.zeros(0, np.int8), np.zeros(0)
    return np.concatenate(ps), np.concatenate(ys), np.concatenate(ws)


def auc_stats(films, key):
    """→ (合并 AUC, 逐片平均, (最差 AUC, 片号))。"""
    p, y, _ = _cat(films, key)
    pooled = auc(p, y == 1)
    per = []
    for vid, f in films.items():
        m = f.y >= 0
        a = auc(f.p[key][m], f.y[m] == 1)
        if a == a:
            per.append((a, vid))
    return pooled, (float(np.mean([a for a, _ in per])) if per else float('nan')), (min(per) if per else (float('nan'), ''))


def app_decision(p_do, p_plot):
    """app 的判定：1 转、0 不转、-1 unclear。"""
    d = np.full(len(p_do), -1, np.int8)
    d[p_plot >= V.F.LEGACY_DEFAULTS['VIDEO_PLOT']] = 0
    d[p_do >= V.F.LEGACY_DEFAULTS['VIDEO_DO']] = 1
    return d


def conf_bins(films):
    """→ [(下限, 上限, 窗口占比, 错误率, 占全部错误)]，按 3 秒窗口、二分类（p_do ≥ 0.5 判转）。"""
    p, y, _ = _cat(films, ('3s', 'logit'))
    conf = np.maximum(p, 1 - p)
    wrong = (p >= 0.5).astype(np.int8) != y
    n, nw = len(p), max(int(wrong.sum()), 1)
    out = []
    for lo, hi in zip(CONF_BINS[:-1], CONF_BINS[1:]):
        m = (conf >= lo) & (conf < hi)
        k = int(m.sum())
        out.append((lo, min(hi, 1.0), k / float(max(n, 1)), float(wrong[m].mean()) if k else float('nan'),
                    float(wrong[m].sum()) / nw))
    return out


def segments(f, conf=SEG_CONF, min_s=SEG_MIN_S, gap_s=SEG_GAP_S):
    """一部片的成段高置信度错误 → [(起 pos ms, 止 pos ms, '误转'/'漏转', 段长秒, 其中高置信度判错的秒数, 平均置信度, 真值标签)]。
    以 ≥ conf 判错（剧情判成转 = 误转，动作判成剧情 = 漏转）的结果起头和收尾；中间夹着的不够格的结果（置信度掉到 conf 以下、
    甚至判对）累计不超过 gap_s 秒就连成一段；真值换段、结果中断（相邻结果隔 2 秒以上）都会断开。段长 ≥ min_s 秒才算。"""
    p = f.p[('3s', 'logit')]
    kind = np.zeros(len(p), np.int8)                     # 1 误转，2 漏转
    kind[(f.y == 0) & (p >= conf)] = 1
    kind[(f.y == 1) & (1 - p >= conf)] = 2
    out, i, n = [], 0, len(p)
    while i < n:
        if not kind[i]:
            i += 1
            continue
        k, j, last, gap = kind[i], i, i, 0.0
        while j + 1 < n and f.lab[j + 1] == f.lab[i] and f.pos[j + 1] - f.pos[j] <= MAX_AGE_MS:
            j += 1
            if kind[j] == k:
                last, gap = j, 0.0
            else:
                gap += f.dt[j]
                if gap > gap_s:
                    break
        sl = slice(i, last + 1)
        hit = kind[sl] == k
        span = (f.pos[last] - f.pos[i]) / 1000.0 + float(f.dt[last])
        if span >= min_s:
            c = p[sl][hit] if k == 1 else 1 - p[sl][hit]
            out.append((int(f.pos[i]), int(f.pos[last]), '误转' if k == 1 else '漏转', span,
                        float(f.dt[sl][hit].sum()), float(c.mean()), f.lab[i]))
        i = last + 1
    return out


def film_stats(f):
    m = f.y >= 0
    p = f.p[('3s', 'logit')]
    d = app_decision(p, f.pplot)
    plot, act = m & (f.y == 0), m & (f.y == 1)
    dec = m & (d >= 0)
    seg = segments(f)
    w = f.dt
    return dict(
        auc=auc(p[m], f.y[m] == 1), n=int(m.sum()),
        acc=float((d[dec] == f.y[dec]).mean()) if dec.any() else float('nan'),
        fp=float((d[plot] == 1).mean()) if plot.any() else float('nan'),
        fn=float((d[act] == 0).mean()) if act.any() else float('nan'),
        fp_s=float(w[plot & (d == 1)].sum()), fn_s=float(w[act & (d == 0)].sum()),
        seg_fp=sum(s[4] for s in seg if s[2] == '误转'), seg_fn=sum(s[4] for s in seg if s[2] == '漏转'),
        segs=seg)


def summarize(name, films, missing):
    r = dict(name=name, films=list(films), missing=missing, auc={}, per_film={})
    for wk, _, _ in WINDOWS:
        r['auc'][(wk, 'logit')] = auc_stats(films, (wk, 'logit'))
    for ak, _ in AGGS[1:]:
        r['auc'][('3s', ak)] = auc_stats(films, ('3s', ak))
    for half, vids in HALVES:
        sub = dict((v, f) for v, f in films.items() if v in vids)
        r['auc'][('half', half)] = auc_stats(sub, ('3s', 'logit'))
    r['bins'] = conf_bins(films)
    for vid, f in films.items():
        r['per_film'][vid] = film_stats(f)
    pf = r['per_film'].values()
    tot = lambda k: sum(x[k] for x in pf)                # noqa: E731
    r['fp_s'], r['fn_s'] = tot('fp_s'), tot('fn_s')
    r['seg_fp'], r['seg_fn'] = tot('seg_fp'), tot('seg_fn')
    r['segs'] = sorted(((vid,) + s for vid, x in r['per_film'].items() for s in x['segs']), key=lambda s: -s[4])  # 按段长
    p, y, w = _cat(films, ('3s', 'logit'))
    d = app_decision(p, np.concatenate([f.pplot[f.y >= 0] for f in films.values()]) if films else np.zeros(0))
    dec = d >= 0
    r['acc'] = float((d[dec] == y[dec]).mean()) if dec.any() else float('nan')
    r['fp'] = float((d[y == 0] == 1).mean()) if (y == 0).any() else float('nan')
    r['fn'] = float((d[y == 1] == 0).mean()) if (y == 1).any() else float('nan')
    r['n'] = len(p)
    return r


def pairing(fa, fb):
    """同一部片、同一个结果（墙钟 ts 相同）在两个变体下的二分类判定：→ dict(秒数)。"""
    out = dict(fix_fp=0.0, fix_fn=0.0, break_fp=0.0, break_fn=0.0, n=0)
    for vid in fa:
        if vid not in fb:
            continue
        a, b = fa[vid], fb[vid]
        common, ia, ib = np.intersect1d(a.ts, b.ts, return_indices=True)
        if not len(common):
            continue
        y = a.y[ia]
        m = y >= 0
        wa = (a.p[('3s', 'logit')][ia] >= 0.5).astype(np.int8) != y
        wb = (b.p[('3s', 'logit')][ib] >= 0.5).astype(np.int8) != y
        dt = a.dt[ia]
        out['n'] += int(m.sum())
        out['fix_fp'] += float(dt[m & wa & ~wb & (y == 0)].sum())
        out['fix_fn'] += float(dt[m & wa & ~wb & (y == 1)].sum())
        out['break_fp'] += float(dt[m & ~wa & wb & (y == 0)].sum())
        out['break_fn'] += float(dt[m & ~wa & wb & (y == 1)].sum())
    return out


# ======================================================================== 报告
def _f(x, fmt):
    return '—' if x is None or x != x else fmt % x


def _pct(x):
    return '—' if x is None or x != x else '%.1f%%' % (100.0 * x)


def report_one(r, n_seg=12):
    out = ['══ 逐帧诊断：%s ══' % r['name'],
           '（按视频结果算 %d 个窗口，标签取出结果那一刻的真值；other_adult 不计；p_do 用逐帧 logit 全精度重算）' % r['n']]
    if r['missing']:
        out.append('⚠ 缺这几部：%s' % ' '.join(r['missing']))
    out.append('')
    out.append('1. 判别力 AUC（p_oral+p_sex）           合并     逐片平均   最差片')
    rows = [('窗口 ' + lab, (wk, 'logit')) for wk, lab, _ in WINDOWS]
    rows += [('3 秒窗口 · ' + lab, ('3s', ak)) for ak, lab in AGGS[1:]]
    rows += [('3 秒窗口 · %s 合并' % h, ('half', h)) for h, _ in HALVES]
    for lab, key in rows:
        pooled, mean, (worst, wv) = r['auc'][key]
        out.append('   %s %8s %10s   %s %s' % (B.lpad(lab, 34), _f(pooled, '%.3f'), _f(mean, '%.3f'), wv,
                                             _f(worst, '%.3f')))
    out += ['', '2. 置信度分布（3 秒窗口，max(p_do, p_plot)；错 = p_do ≥ 0.5 判转、判错）',
            '   置信度         窗口占比   这一档的错误率   占全部错误']
    for lo, hi, share, err, of_err in r['bins']:
        out.append('   %.2f ~ %.2f  %9s %14s %12s' % (lo, hi, _pct(share), _pct(err), _pct(of_err)))
    tf, tn = max(r['fp_s'], 1e-9), max(r['fn_s'], 1e-9)
    out += ['', '3. 成段的高置信度错误（以 ≥%.1f 判错的结果连成 ≥%g 秒的段，中间夹着的其它结果累计 ≤%g 秒）：%d 段'
            % (SEG_CONF, SEG_MIN_S, SEG_GAP_S, len(r['segs'])),
            '   其中高置信度判错的时长：误转 %.0f 秒（占视频全部误转 %.0f 秒的 %s），漏转 %.0f 秒（占全部漏转 %.0f 秒的 %s）'
            % (r['seg_fp'], r['fp_s'], _pct(r['seg_fp'] / tf), r['seg_fn'], r['fn_s'], _pct(r['seg_fn'] / tn)),
            '   （"全部误转 / 漏转"按 app 阈值：剧情判成转 / 动作判成不转的结果所管的秒数；时间是窗口结尾，看片时往前多看 3 秒）']
    for vid, a, b, kind, span, sec, c, lab in r['segs'][:n_seg]:
        out.append('   %s  %6.0f ~ %6.0fs  段长 %5.1f 秒（高置信度判错 %5.1f 秒）  %s（真值 %s，平均置信度 %.3f）'
                   % (vid, a / 1000.0, b / 1000.0, span, sec, kind, lab, c))
    if len(r['segs']) > n_seg:
        out.append('   … 其余 %d 段（--segments N 多列几段）' % (len(r['segs']) - n_seg))
    out += ['', '4. 逐片（按 app 阈值 0.65 / 0.60；准确率只算给了判定的窗口）',
            '   %s %8s %9s %9s %9s %9s %9s %7s' % (B.lpad('片', 4), 'AUC', '准确率', '剧情判转', '动作判不转',
                                                 '成段误转', '成段漏转', '窗口')]
    for vid in sorted(r['per_film']):
        x = r['per_film'][vid]
        out.append('   %s %8s %9s %9s %9s %8.0fs %8.0fs %7d' % (B.lpad(vid, 4), _f(x['auc'], '%.3f'), _pct(x['acc']),
                                                            _pct(x['fp']), _pct(x['fn']), x['seg_fp'], x['seg_fn'],
                                                            x['n']))
    out.append('   合计 准确率 %s，剧情判成转 %s，动作判成不转 %s' % (_pct(r['acc']), _pct(r['fp']), _pct(r['fn'])))
    return '\n'.join(out) + '\n'


_CMP_ROWS = [
    ('AUC 3 秒 合并', lambda r: r['auc'][('3s', 'logit')][0], '%.3f', 1),
    ('AUC 3 秒 逐片平均', lambda r: r['auc'][('3s', 'logit')][1], '%.3f', 1),
    ('AUC 3 秒 最差片', lambda r: r['auc'][('3s', 'logit')][2][0], '%.3f', 1),
    ('AUC 3 秒 v01~v07 合并', lambda r: r['auc'][('half', 'v01~v07')][0], '%.3f', 1),
    ('AUC 3 秒 v08~v13 合并', lambda r: r['auc'][('half', 'v08~v13')][0], '%.3f', 1),
    ('AUC 单帧 合并', lambda r: r['auc'][('frame', 'logit')][0], '%.3f', 1),
    ('AUC 12 秒 合并', lambda r: r['auc'][('12s', 'logit')][0], '%.3f', 1),
    ('置信度 ≥0.99 的窗口占比', lambda r: r['bins'][-1][2], 'pct', 100),
    ('  其中判错的比例', lambda r: r['bins'][-1][3], 'pct', 100),
    ('  占全部错误', lambda r: r['bins'][-1][4], 'pct', 100),
    ('成段错误（段数）', lambda r: len(r['segs']), '%d', 1),
    ('成段误转（秒）', lambda r: r['seg_fp'], '%.0f', 1),
    ('成段漏转（秒）', lambda r: r['seg_fn'], '%.0f', 1),
    ('视频全部误转（秒）', lambda r: r['fp_s'], '%.0f', 1),
    ('视频全部漏转（秒）', lambda r: r['fn_s'], '%.0f', 1),
    ('准确率（app 阈值）', lambda r: r['acc'], 'pct', 100),
    ('剧情判成转', lambda r: r['fp'], 'pct', 100),
    ('动作判成不转', lambda r: r['fn'], 'pct', 100),
]


def _cell(x, fmt):
    if x is None or x != x:
        return '—'
    return '%.1f%%' % (100.0 * x) if fmt == 'pct' else fmt % x


def _delta(x, x0, fmt, mul):
    if x is None or x0 is None or x != x or x0 != x0:
        return ''
    d = (x - x0) * mul
    return '(%+.1f)' % d if fmt == 'pct' else ('(%+.3f)' % d if fmt == '%.3f' else '(%+.0f)' % d)


def report_compare(results, films_by_name):
    names = [r['name'] for r in results]
    w0, w = 26, max(16, max(B.width(n) for n in names) + 4)
    out = ['══ 逐帧诊断对比：%s ══' % ' / '.join(names),
           '括号里是相对 %s 的变化（百分比的变化是百分点）。AUC、准确率越高越好；错误秒数、占比越低越好' % names[0], '']
    out.append(B.lpad('', w0) + ''.join(B.rpad(n, w) for n in names))
    for lab, get, fmt, mul in _CMP_ROWS:
        x0 = get(results[0])
        cells = []
        for i, r in enumerate(results):
            x = get(r)
            cells.append(_cell(x, fmt) + (' ' + _delta(x, x0, fmt, mul) if i else ''))
        out.append(B.lpad(lab, w0) + ''.join(B.rpad(c, w) for c in cells))
    out += ['', '逐片 AUC（3 秒窗口）']
    vids = sorted(set(v for r in results for v in r['per_film']))
    for vid in vids:
        x0 = results[0]['per_film'].get(vid, {}).get('auc')
        cells = []
        for i, r in enumerate(results):
            x = r['per_film'].get(vid, {}).get('auc')
            cells.append(_cell(x, '%.3f') + (' ' + _delta(x, x0, '%.3f', 1) if i else ''))
        out.append(B.lpad('  ' + vid, w0) + ''.join(B.rpad(c, w) for c in cells))
    base = results[0]
    for r in results[1:]:
        better, worse = [], []
        for vid, x in r['per_film'].items():
            a0, a1 = base['per_film'].get(vid, {}).get('auc'), x['auc']
            if a0 is None or a0 != a0 or a1 != a1:
                continue
            if a1 - a0 > 0.005:
                better.append(vid)
            elif a0 - a1 > 0.005:
                worse.append(vid)
        w_old, w_new = base['auc'][('3s', 'logit')][2], r['auc'][('3s', 'logit')][2]
        pr = pairing(films_by_name[base['name']], films_by_name[r['name']])
        out += ['', '%s 相对 %s：' % (r['name'], base['name']),
                '  逐片 AUC 变好 %d 部（%s），变差 %d 部（%s）（变化不到 0.005 不算）' % (
                    len(better), ' '.join(sorted(better)) or '无', len(worse), ' '.join(sorted(worse)) or '无'),
                '  最差片：%s %.3f → %s %.3f%s' % (w_old[1], w_old[0], w_new[1], w_new[0],
                                                '' if w_new[0] >= w_old[0] - 0.005 else '  ⚠ 最差片退步了'),
                '  逐个窗口配对（%d 个窗口，按每个结果管的秒数）：判错 → 判对 %.0f 秒（剧情 %.0f、动作 %.0f），'
                '判对 → 判错 %.0f 秒（剧情 %.0f、动作 %.0f）' % (
                    pr['n'], pr['fix_fp'] + pr['fix_fn'], pr['fix_fp'], pr['fix_fn'],
                    pr['break_fp'] + pr['break_fn'], pr['break_fp'], pr['break_fn'])]
        # 两半各有几部变好 / 变差（按逐片 AUC；只看合并 AUC 不行：v08~v13 本来就接近 1，合并 AUC 动不了多少）
        halves = [(h, sum(1 for v in vids if v in better), sum(1 for v in vids if v in worse)) for h, vids in HALVES]
        out.append('  两半：%s' % '；'.join('%s 变好 %d 部、变差 %d 部' % x for x in halves))
        if (halves[0][1] > 0) != (halves[1][1] > 0) or any(b == 0 and w > 0 for _, b, w in halves):
            out.append('  ⚠ 只在一半片子上变好，要谨慎')
    return '\n'.join(out) + '\n'


# ======================================================================== 命令
def run(names, n_seg=12, write=True):
    _need()
    results, films_by_name = [], {}
    for n in names:
        films, missing = load_variant(n)
        films_by_name[n] = films
        results.append(summarize(n, films, missing))
    text = ''.join(report_one(r, n_seg) + '\n' for r in results)
    if len(results) > 1:
        text += report_compare(results, films_by_name)
    if write:
        path = os.path.join(B.OUT_DIR, 'frame_diag_%s.txt' % '_vs_'.join(names))
        B._write(path, text)
        text += '报告：%s\n' % B.rel(path)
    return text, results


# ======================================================================== 自检
def _write_synth(dirpath, vid, results, frames):
    """results: [(pos_ms, [12 帧 ms])]；frames: {帧 ms: logit[3]}。写成 rescore 的格式（csv + vframes.npz）。"""
    os.makedirs(dirpath, exist_ok=True)
    cols = ['pos_ms', 'wall_ms', 'v_action', 'v_conf', 'v_age', 'v_p_plot', 'v_p_oral', 'v_p_sex', 'v_pos', 'v_fpts']
    lines = ['# xcup-bench-run v1', '# params: VIDEO_DO=0.65 VIDEO_PLOT=0.60',
             '# video_rescore.py rescore: pipe=exact_crop matrix=auto delta=200ms model=synth.tflite', ','.join(cols)]
    for pos, fpts in results:
        z = np.mean([frames[t] for t in fpts], axis=0)
        p = _softmax(np.array(z, np.float64))
        lines.append('%d,%d,,0.0,100,%.4f,%.4f,%.4f,%d,%s' % (pos + 100, 1000000 + pos + 100, p[0], p[1], p[2],
                                                             fpts[-1], ';'.join(str(t) for t in fpts)))
    with io.open(os.path.join(dirpath, vid + '_base.csv'), 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(lines) + '\n')
    ks = sorted(frames)
    np.savez_compressed(os.path.join(dirpath, vid + '_base.vframes.npz'), pts_us=np.array(ks, np.int64) * 1000,
                        logits=np.array([frames[k] for k in ks], np.float32), pipe=np.array('exact_crop'),
                        matrix=np.array('auto'))


def selftest():
    _need()
    fails = []

    def check(ok, msg):
        print('  %s %s' % ('ok  ' if ok else 'FAIL', msg))
        if not ok:
            fails.append(msg)

    check(abs(auc([0.9, 0.8, 0.3, 0.7], [1, 1, 0, 0]) - 1.0) < 1e-12 and
          abs(auc([1.0, 0.8, 0.6, 0.7, 0.2], [1, 1, 1, 0, 0]) - M._auc([1.0, 0.8, 0.6], [0.7, 0.2])) < 1e-12,
          'AUC 与 metrics._auc 相同')
    # 合成：一部 120 秒的片，0~60 秒剧情、60~120 秒性爱；每 500ms 出一个结果，帧每 250ms 一帧。
    # 逐帧 logit：剧情帧大多判剧情，但 20~32 秒一段高置信度判成转（12 秒误转段）；性爱帧判转，90~93 秒一小段判剧情（3 秒，不算成段）
    keep = (B.RESCORED, B.GT_DIR, B.OUT_DIR)
    tmp = tempfile.mkdtemp(prefix='frame_diag_')
    try:
        B.RESCORED, B.GT_DIR, B.OUT_DIR = os.path.join(tmp, 'rescored'), os.path.join(tmp, 'gt'), os.path.join(tmp, 'out')
        os.makedirs(B.GT_DIR)
        with io.open(os.path.join(B.GT_DIR, 'v01.csv'), 'w', encoding='utf-8') as fh:
            fh.write('# duration_ms=120000\nstart_ms,end_ms,label,note\n0,60000,plot,\n60000,120000,sex,\n')
        rng = np.random.default_rng(0)

        def logit(t, model):
            if t < 60000:
                if 20000 <= t < 32000:
                    return [-4.0, 1.0, 3.0]                 # 判成转，p_do ≈ 0.999
                base = [2.0, 0.0, 0.0] if model == 'a' else [3.0, 0.0, 0.0]
            else:
                if 90000 <= t < 93000:
                    return [3.0, 0.0, -1.0]
                base = [0.0, 0.5, 2.0]
            return list(np.array(base) + rng.normal(0, 0.8 if model == 'a' else 0.3, 3))

        for model, name in (('a', 'synA'), ('b', 'synB')):
            frames = dict((t, logit(t, model)) for t in range(0, 120000, 250))
            res = [(pos, [pos - 200 - 250 * (11 - k) for k in range(12)]) for pos in range(3400, 119000, 500)]
            res = [(p, [t - t % 250 for t in fp]) for p, fp in res]
            for d in B.variant_dirs(name)[:1]:
                _write_synth(d, 'v01', res, frames)
        films, missing = load_variant('synA', films=('v01',))
        f = films['v01']
        p3 = f.p[('3s', 'logit')]
        _, rs = V.load_results(f.run_path)
        check(len(p3) == len(rs) and np.abs(np.array([r.probs[1] + r.probs[2] for r in rs]) - p3).max() < 2e-4,
              '3 秒窗口的 p_do = CSV 里的 v_p_oral+v_p_sex（逐帧 logit 和 v_fpts 对上了）')
        r = summarize('synA', films, missing)
        seg = r['segs']
        check(len(seg) == 1 and seg[0][3] == '误转' and 10.0 <= seg[0][4] <= 15.0 and seg[0][0] == 'v01'
              and seg[0][5] <= seg[0][4] + 1e-9,
              '成段错误：找到 20~32 秒那段误转（%s），90~93 秒的短段不算' % [(s[1], s[2], s[3], round(s[4], 1)) for s in seg])
        f2 = Film()
        f2.__dict__.update(f.__dict__)
        f2.p = dict(f.p)
        q = f.p[('3s', 'logit')].copy()
        mid = np.where((f.pos > 25000) & (f.pos < 27000))[0]
        q[mid] = 0.6                                    # 段中间 2 秒掉到 0.9 以下：仍连成一段
        f2.p[('3s', 'logit')] = q
        s2 = segments(f2)
        q[np.where((f.pos > 24000) & (f.pos < 28500))[0]] = 0.2   # 中间 4.5 秒判对：断成两段（都不到 8 秒就都不算）
        s3 = segments(f2)
        check(len(s2) == 1 and abs(s2[0][3] - seg[0][4]) < 1e-9 and s2[0][4] < seg[0][5] and len(s3) == 0,
              '成段错误：中间夹着 ≤3 秒不够格的结果仍算一段，夹着更长的断开')
        check(abs(r['seg_fp'] - seg[0][5]) < 1e-9 and r['seg_fn'] == 0 and r['fp_s'] >= r['seg_fp'],
              '成段误转秒数 ≤ 视频全部误转秒数')
        b = r['bins']
        check(abs(sum(x[2] for x in b) - 1.0) < 1e-9 and abs(sum(x[4] for x in b) - 1.0) < 1e-9,
              '置信度分档：窗口占比、错误占比各加起来 = 1')
        a3, a12 = r['auc'][('3s', 'logit')][0], r['auc'][('12s', 'logit')][0]
        check(0.5 < a3 < 1.0 and r['auc'][('frame', 'logit')][0] <= a3 + 1e-9,
              '单帧 AUC ≤ 3 秒窗口（单帧 %.3f、3 秒 %.3f、12 秒 %.3f）' % (r['auc'][('frame', 'logit')][0], a3, a12))
        text, res = run(['synA', 'synB'], write=True)
        check('逐帧诊断对比' in text and os.path.exists(os.path.join(B.OUT_DIR, 'frame_diag_synA_vs_synB.txt')),
              '对比报告写出来了')
        pr = pairing(*[load_variant(n, films=('v01',))[0] for n in ('synA', 'synB')])
        check(pr['n'] > 100 and pr['fix_fp'] > 0 and pr['break_fp'] + pr['break_fn'] < pr['fix_fp'] + pr['fix_fn'],
              '窗口配对：synB（剧情更确定、噪声小）修好的比弄坏的多（修 %.0f 秒，坏 %.0f 秒）'
              % (pr['fix_fp'] + pr['fix_fn'], pr['break_fp'] + pr['break_fn']))
    finally:
        B.RESCORED, B.GT_DIR, B.OUT_DIR = keep
        shutil.rmtree(tmp, ignore_errors=True)
    # 真实数据：exact_crop 在仓库里时，逐帧重算的概率应与 CSV 一致，合并 AUC 接近 baseline 的 0.935
    if all(os.path.isdir(d) for d in B.variant_dirs(B.DEFAULT_PIPE)):
        films, missing = load_variant(B.DEFAULT_PIPE)
        worst = 0.0
        for vid, f in films.items():
            _, rs = V.load_results(f.run_path)
            byts = dict((r.ts, r.probs[1] + r.probs[2]) for r in rs)
            p3 = f.p[('3s', 'logit')]
            worst = max(worst, max(abs(byts[t] - p) for t, p in zip(f.ts, p3)))
        r = summarize(B.DEFAULT_PIPE, films, missing)
        a = r['auc'][('3s', 'logit')][0]
        check(not missing and worst < 2e-4 and abs(a - 0.935) < 0.01,
              'exact_crop：13 部的逐帧 logit 与 CSV 对上（最大差 %.1e），合并 AUC %.3f（baseline.py 按 tick 算 0.935）'
              % (worst, a))
    else:
        print('  skip exact_crop（bench/rescored 下没有）')
    if fails:
        print('selftest: %d 项失败' % len(fails))
        return 1
    print('selftest: ALL PASS')
    return 0


def main():
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(errors='replace')
        except Exception:
            pass
    if '--selftest' in sys.argv:
        return selftest()
    ap = argparse.ArgumentParser(description='视频模型的逐帧诊断（详见文件头）',
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--name', action='append', required=True,
                    help='变体名（baseline.py rescore 时的 --name；exact_crop = 现用模型）。可多次或逗号分隔，第一个是参照')
    ap.add_argument('--segments', type=int, default=12, help='列出最长的几段成段错误（默认 12）')
    a = ap.parse_args()
    names = [x.strip() for n in a.name for x in n.split(',') if x.strip()]
    text, _ = run(names, a.segments)
    print(text, end='')
    return 0


if __name__ == '__main__':
    sys.exit(main())
