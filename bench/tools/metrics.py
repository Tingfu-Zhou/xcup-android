#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
bench/tools/metrics.py

把一次真机 dump（BenchLogger 产出的 run CSV）和人工真值对齐，输出两个核心指标：
  1. 逐秒判定准确率
  2. 切换延迟（真值切换点 -> 蓝牙指令发出的时间差）

两条时间线都用【视频播放位置 pos_ms】对齐，所以不需要任何标定。

用法:
  python3 bench/tools/metrics.py bench/gt/v01.csv                    # 只校验真值 + 打印统计
  python3 bench/tools/metrics.py bench/gt/v01.csv run_v01.csv        # 出指标
  python3 bench/tools/metrics.py gt.csv run.csv --policy do          # other_adult 按"该转"算
  python3 bench/tools/metrics.py gt.csv run.csv --append bench/summary.csv
  python3 bench/tools/metrics.py --selftest                          # 自检本脚本
"""
import csv, io, os, re, sys, argparse, statistics

GRID_MS   = 100     # 评测栅格；比 1 秒细，延迟才量得出来
GUARD_MS  = 500     # 真值边界护带，不计入准确率（规范 §4）
WARMUP_MS = 3000    # 视频窗 12x250ms 填满前系统结构上无法判定
LAT_WIN   = 15000   # 切换点之后多久内还算"切到了"
LAT_PRE   = 5000    # 允许提前多久（抢跑记为负延迟）

LABELS = ('plot', 'oral', 'sex', 'other_adult', 'exclude')
BIN    = {'plot': 'noise', 'oral': 'do', 'sex': 'do'}

# YAMNet output 0 是 AudioSet 521 类分数 [帧, 521]。按帧取平均后，挑这几类写进 run CSV 的 a_ys_* 列：
# (列名后缀, AudioSet 类别号, 英文类名)。只有 audio_rescore.py rescore 会写这些列（离线分析用），
# app 不算 AudioSet 分数。前 4 类是"剧情"候选证据（对白、旁白、配乐），其余是"动作"候选。
YS_CLASSES = [('speech', 0, 'Speech'), ('conversation', 2, 'Conversation'),
              ('narration', 3, 'Narration, monologue'), ('music', 132, 'Music'),
              ('silence', 494, 'Silence'), ('moan', 22, 'Wail, moan'), ('groan', 33, 'Groan'),
              ('breathing', 36, 'Breathing'), ('pant', 40, 'Pant'), ('slap', 461, 'Slap, smack')]
YS_COLS = ['a_ys_' + n for n, _, _ in YS_CLASSES]


# ----------------------------------------------------------------- 读文件
def _rows(path):
    meta = {}
    data = []
    header = None
    for raw in io.open(path, encoding='utf-8-sig'):
        line = raw.rstrip('\r\n')
        if not line.strip():
            continue
        if line.lstrip().startswith('#'):
            body = line.lstrip()[1:].strip()
            if body.startswith('params:'):
                meta['params'] = body[len('params:'):].strip()
            else:
                for kv in body.split():
                    if '=' in kv:
                        k, v = kv.split('=', 1)
                        meta[k] = v
            continue
        f = next(csv.reader([line]))
        if header is None:
            header = [c.strip() for c in f]
            continue
        data.append(dict(zip(header, f)))
    if header is None:
        sys.exit('%s: 找不到表头' % path)
    return meta, data


def read_gt(path):
    meta, data = _rows(path)
    segs = []
    for d in data:
        s, e, lab = int(d['start_ms']), int(d['end_ms']), d['label'].strip()
        if lab not in LABELS:
            sys.exit('%s: 非法标签 %r' % (path, lab))
        segs.append((s, e, lab))
    segs.sort()
    errs = []
    if not segs:
        sys.exit('%s: 没有数据行' % path)
    if segs[0][0] != 0:
        errs.append('首段不是从 0 开始（%d）' % segs[0][0])
    for i in range(1, len(segs)):
        if segs[i][0] != segs[i - 1][1]:
            errs.append('第 %d/%d 段首尾不相接: %d vs %d'
                        % (i, i + 1, segs[i - 1][1], segs[i][0]))
    for i, (s, e, lab) in enumerate(segs):
        if e - s < 1000:
            errs.append('第 %d 段仅 %dms < 1000ms（规范 §4）' % (i + 1, e - s))
    dur = int(meta.get('duration_ms') or segs[-1][1])
    if segs[-1][1] != dur:
        errs.append('末段结束 %d != duration_ms %d' % (segs[-1][1], dur))
    return meta, segs, dur, errs


def _norm(a):
    """run CSV 里的动作串 -> 'do' / 'noise' / ''(unclear)"""
    a = (a or '').strip()
    if a in ('do', 'oral', 'sex'):
        return 'do'
    if a.lower() == 'noise':
        return 'noise'
    return ''


def read_run(path):
    meta, data = _rows(path)
    rows = []
    for d in data:
        def i(k, dv=0):
            try: return int(float(d.get(k, '') or dv))
            except ValueError: return dv
        def f(k, dv=0.0):
            try: return float(d.get(k, '') or dv)
            except ValueError: return dv
        rows.append({
            'pos': i('pos_ms'), 'wall': i('wall_ms'),
            'v': _norm(d.get('v_action')), 'v_conf': f('v_conf'), 'v_age': i('v_age', -1),
            'a': _norm(d.get('a_action')), 'a_conf': f('a_conf'), 'a_age': i('a_age', -1),
            'vp': [f('v_p_plot', -1), f('v_p_oral', -1), f('v_p_sex', -1)],
            'ap': [f('a_p_sex', -1), f('a_p_oral', -1), f('a_p_noise', -1)],
            'ys': [f(c, -1) for c in YS_COLS],          # app 的 dump 没有这些列（只有 rescore 写）：全是 -1
            'loud_lv': i('loud_lv', -1), 'loud_conf': f('loud_conf'),
            'smoothed': _norm(d.get('smoothed')), 'gated': _norm(d.get('gated')),
            'ble_state': _norm(d.get('ble_state')), 'ble_level': i('ble_level'),
            'sent': i('sent') == 1, 'sent_action': _norm(d.get('sent_action')),
            'sent_level': i('sent_level', -1),
            # FusionArbiter 的输出原因（2026-10-07 起的 run 才有；旧 run 为空）
            'arb': (d.get('arb') or '').strip(),
        })
    rows.sort(key=lambda r: r['pos'])
    return meta, rows


# ----------------------------------------------------------------- 时间线
def bin_of(label, policy):
    if label in BIN:
        return BIN[label]
    if label == 'other_adult':
        return None if policy == 'exclude' else policy
    return None            # exclude


def gt_transitions(segs, policy):
    seq = [(s, bin_of(l, policy)) for s, e, l in segs]
    seq = [(s, b) for s, b in seq if b]
    out = []
    for i in range(1, len(seq)):
        if seq[i][1] != seq[i - 1][1]:
            out.append((seq[i][0], seq[i - 1][1], seq[i][1]))
    return out


def sys_transitions(rows):
    """只有真正改变马达状态的指令才算切换；同动作改档位的重发不算。"""
    out, cur = [], 'noise'          # 第一条指令之前马达不转
    for r in rows:
        if not r['sent'] or not r['sent_action']:
            continue
        if r['sent_action'] != cur:
            out.append((r['pos'], r['sent_action']))
            cur = r['sent_action']
    return out


def state_at(trans, pos):
    st = 'noise'
    for p, s in trans:
        if p <= pos:
            st = s
        else:
            break
    return st


# ----------------------------------------------------------------- 指标
def evaluate(segs, dur, rows, policy):
    n = dur // GRID_MS
    gt = [None] * n
    for s, e, lab in segs:
        b = bin_of(lab, policy)
        for i in range(max(0, s // GRID_MS), min(n, -(-e // GRID_MS))):
            gt[i] = b

    mask = [g is not None for g in gt]
    for s, e, lab in segs[1:]:                       # 边界护带
        for i in range(max(0, (s - GUARD_MS) // GRID_MS),
                       min(n, (s + GUARD_MS) // GRID_MS + 1)):
            mask[i] = False
    lo = max(WARMUP_MS, rows[0]['pos']) if rows else WARMUP_MS
    hi = rows[-1]['pos'] if rows else 0
    for i in range(n):
        if i * GRID_MS < lo or i * GRID_MS > hi:
            mask[i] = False

    strans = sys_transitions(rows)
    cm = {('noise', 'noise'): 0, ('noise', 'do'): 0, ('do', 'noise'): 0, ('do', 'do'): 0}
    for i in range(n):
        if not mask[i]:
            continue
        cm[(gt[i], state_at(strans, i * GRID_MS))] += 1
    tot = sum(cm.values())
    acc = (cm[('noise', 'noise')] + cm[('do', 'do')]) / tot if tot else float('nan')

    gt_noise = cm[('noise', 'noise')] + cm[('noise', 'do')]
    gt_do    = cm[('do', 'noise')] + cm[('do', 'do')]

    # 动作中误停：真值在"转"的时候，系统却发了停转
    false_stop = 0
    for p, s in strans:
        if s != 'noise':
            continue
        i = p // GRID_MS
        if 0 <= i < n and gt[i] == 'do' and mask[i]:
            false_stop += 1

    # 切换延迟 + 三段分解
    lat = {'noise->do': [], 'do->noise': []}
    dec = {'noise->do': [], 'do->noise': []}
    miss = {'noise->do': 0, 'do->noise': 0}
    used = -1
    for t, frm, to in gt_transitions(segs, policy):
        key = '%s->%s' % (frm, to)
        if t < lo or t > hi:
            continue
        hit = None
        for p, s in strans:
            if s == to and p > used and t - LAT_PRE <= p <= t + LAT_WIN:
                hit = p
                break
        if hit is None:
            miss[key] += 1
            continue
        used = hit
        lat[key].append(hit - t)
        tp = next((r['pos'] for r in rows if r['pos'] >= t and (r['v'] == to or r['a'] == to)), None)
        tf = next((r['pos'] for r in rows if r['pos'] >= t and r['gated'] == to), None)
        if tp is not None and tf is not None and tf >= tp:
            dec[key].append((tp - t, tf - tp, hit - tf))

    return dict(cm=cm, tot=tot, acc=acc, gt_noise=gt_noise, gt_do=gt_do,
                false_stop=false_stop, lat=lat, dec=dec, miss=miss,
                lo=lo, hi=hi, strans=strans, eval_ms=tot * GRID_MS)


# ----------------------------------------------------------------- 定位与归因
def gt_label_at(segs, pos):
    for a, b, lab in segs:
        if a <= pos < b:
            return lab
    return 'exclude'


def error_intervals(segs, rows, policy, strans):
    """把逐 tick 的错判合并成区间：[(起, 止, '误转'|'漏转', 该区间的 rows)]"""
    out, cur = [], None
    for r in rows:
        if r['pos'] < WARMUP_MS:
            continue
        g = bin_of(gt_label_at(segs, r['pos']), policy)
        if g is None:
            kind = None
        else:
            sysst = state_at(strans, r['pos'])
            kind = None if sysst == g else ('误转' if g == 'noise' else '漏转')
        if kind and cur and cur[2] == kind:
            cur[1] = r['pos']; cur[3].append(r)
        elif kind:
            if cur: out.append(cur)
            cur = [r['pos'], r['pos'], kind, [r]]
        elif cur:
            out.append(cur); cur = None
    if cur:
        out.append(cur)
    return out


def _dist(rs, key):
    c = {}
    for r in rs:
        k = r[key] or 'unclear'
        c[k] = c.get(k, 0) + 1
    n = max(1, len(rs))
    return ' '.join('%s%d%%' % (k, round(100.0 * v / n))
                    for k, v in sorted(c.items(), key=lambda x: -x[1]))


def report_errors(segs, rows, policy, strans, top):
    ivl = error_intervals(segs, rows, policy, strans)
    ivl = [x for x in ivl if x[1] - x[0] >= 1000]
    ivl.sort(key=lambda x: -(x[1] - x[0]))
    print('══ 误判区间（≥1s，按时长排序，共 %d 段）══' % len(ivl))
    print('  %-20s %7s %-5s %-8s %-26s %-26s %s'
          % ('时间段', '时长', '类型', '真值', '视频判定', '音频判定', '仲裁后'))
    for a, b, kind, rs in ivl[:top]:
        labs = sorted({gt_label_at(segs, r['pos']) for r in rs})
        print('  %7.1fs ~ %7.1fs %6.1fs %-5s %-8s %-26s %-26s %s'
              % (a / 1000.0, b / 1000.0, (b - a) / 1000.0, kind, '/'.join(labs),
                 _dist(rs, 'v'), _dist(rs, 'a'), _dist(rs, 'gated')))
    if len(ivl) > top:
        print('  … 其余 %d 段省略（--top N 可调）' % (len(ivl) - top))
    # 带小数点才按秒解析（整数按毫秒），所以示例里必须带 .1f
    print('\n  用 --at <时间> 看某一段的逐 tick 原始数据，例如 --at %.1f'
          % (ivl[0][0] / 1000.0 if ivl else 0))
    print()


def report_at(rows, t, half=5000):
    print('══ pos %s 附近的逐 tick 原始数据 ══' % sec(t))
    print('  %-9s %-22s %-7s %-24s %-7s %-6s %-6s %-13s %-6s %s'
          % ('pos', '视频 probs[plot,oral,sex]', 'V判定', '音频 probs[sex,oral,noise]',
             'A判定', '平滑', '仲裁', '仲裁原因', '蓝牙', '发送'))
    for r in rows:
        if not (t - half <= r['pos'] <= t + half):
            continue
        vp = '[%s]' % ','.join('%.2f' % x if x >= 0 else ' -- ' for x in r['vp'])
        ap = '[%s]' % ','.join('%.2f' % x if x >= 0 else ' -- ' for x in r['ap'])
        print('  %-9s %-22s %-7s %-24s %-7s %-6s %-6s %-13s %-6s %s'
              % (sec(r['pos']), vp, (r['v'] or 'unclear') + ('%.2f' % r['v_conf'] if r['v'] else ''),
                 ap, (r['a'] or 'unclear') + ('%.2f' % r['a_conf'] if r['a'] else ''),
                 r['smoothed'] or '-', r['gated'] or '-', r['arb'] or '-',
                 '%s/%d' % (r['ble_state'] or '-', r['ble_level']),
                 ('→ %s lv%d' % (r['sent_action'], r['sent_level'])) if r['sent'] else ''))
    print()


def _auc(pos, neg):
    """Mann-Whitney AUC：0.5 = 与随机猜测无异，1.0 = 完美可分。"""
    if not pos or not neg:
        return float('nan')
    allv = sorted([(v, 1) for v in pos] + [(v, 0) for v in neg])
    ranks = {}
    i = 0
    while i < len(allv):
        j = i
        while j + 1 < len(allv) and allv[j + 1][0] == allv[i][0]:
            j += 1
        for k in range(i, j + 1):
            ranks[k] = (i + j) / 2.0 + 1
        i = j + 1
    n1, n0 = len(pos), len(neg)
    ssum = sum(ranks[k] for k, (v, y) in enumerate(allv) if y == 1)
    return (ssum - n1 * (n1 + 1) / 2.0) / (n1 * n0)


def _split(segs, rows, policy, fn):
    pos, neg = [], []
    for r in rows:
        if r['pos'] < WARMUP_MS:
            continue
        g = bin_of(gt_label_at(segs, r['pos']), policy)
        v = fn(r)
        if g and v is not None and v >= 0:
            (pos if g == 'do' else neg).append(v)
    return pos, neg


def report_diag(segs, rows, policy):
    print('══ 模态诊断 ══')
    print('  单模态准确率（只统计该模态给出了明确判定的 tick）')
    for k, name in (('v', '视频 MobileNetV3'), ('a', '音频 YAMNet')):
        dec = []
        for r in rows:
            if r['pos'] < WARMUP_MS:
                continue
            g = bin_of(gt_label_at(segs, r['pos']), policy)
            if g and r[k]:
                dec.append((r, g))
        if not dec:
            continue
        nz = sum(1 for _, g in dec if g == 'noise') or 1
        dz = sum(1 for _, g in dec if g == 'do') or 1
        print('    %-18s 准确率 %5.1f%%   剧情误判为转 %5.1f%%   动作误判为不转 %5.1f%%'
              % (name, 100.0 * sum(1 for r, g in dec if r[k] == g) / len(dec),
                 100.0 * sum(1 for r, g in dec if g == 'noise' and r[k] == 'do') / nz,
                 100.0 * sum(1 for r, g in dec if g == 'do' and r[k] == 'noise') / dz))

    print('\n  各路"该转"证据的判别力 AUC（0.5 = 等于掷硬币）')
    feats = [
        ('视频 p_oral+p_sex  [现用]', lambda r: r['vp'][1] + r['vp'][2] if r['vp'][1] >= 0 else None),
        ('音频 p_sex+p_oral  [现用]', lambda r: r['ap'][0] + r['ap'][1] if r['ap'][0] >= 0 else None),
        ('音频 p_sex 单独', lambda r: r['ap'][0] if r['ap'][0] >= 0 else None),
        ('音频 p_oral 单独', lambda r: r['ap'][1] if r['ap'][1] >= 0 else None),
        ('音频 1-p_noise', lambda r: 1 - r['ap'][2] if r['ap'][2] >= 0 else None),
    ]
    for name, fn in feats:
        pos, neg = _split(segs, rows, policy, fn)
        print('    %-26s AUC = %.3f   (n=%d/%d)' % (name, _auc(pos, neg), len(pos), len(neg)))

    if any(r['ys'][0] >= 0 for r in rows):
        print('\n  YAMNet AudioSet 类别分数（a_ys_*）的 AUC：>0.5 越高越像动作，<0.5 越高越像剧情；'
              '判别力 = 离 0.5 多远')
        for k, (short, idx, en) in enumerate(YS_CLASSES):
            pos, neg = _split(segs, rows, policy, lambda r, k=k: r['ys'][k] if r['ys'][k] >= 0 else None)
            auc = _auc(pos, neg)
            if auc != auc:
                continue
            print('    %-26s AUC = %.3f   判别力 %.3f %s  (n=%d/%d)'
                  % ('%s #%d' % (en, idx), auc, max(auc, 1 - auc),
                     '↑转  ' if auc >= 0.5 else '↑剧情', len(pos), len(neg)))

    for name, fn, ths in (
            ('视频 pDo = p_oral+p_sex（现用阈值 0.65）',
             lambda r: r['vp'][1] + r['vp'][2] if r['vp'][1] >= 0 else None,
             (0.65, 0.50, 0.40, 0.30, 0.25, 0.20, 0.15, 0.10)),
            ('音频 pDo = p_sex+p_oral（现用阈值 0.55）',
             lambda r: r['ap'][0] + r['ap'][1] if r['ap'][0] >= 0 else None,
             (0.55, 0.70, 0.80, 0.90, 0.95, 0.98)),
            ('音频 p_sex 单独',
             lambda r: r['ap'][0] if r['ap'][0] >= 0 else None,
             (0.60, 0.50, 0.40, 0.30, 0.20, 0.10))):
        pos, neg = _split(segs, rows, policy, fn)
        if not pos or not neg:
            continue
        print('\n  阈值扫描：%s' % name)
        print('    阈值    剧情段判"转"(误转)   动作段判"转"(召回)')
        for th in ths:
            print('    %.2f        %5.1f%%               %5.1f%%'
                  % (th, 100.0 * sum(1 for x in neg if x >= th) / len(neg),
                     100.0 * sum(1 for x in pos if x >= th) / len(pos)))
    print('\n  注意：这些数字只来自当前这一个视频，不要据此直接改线上参数。\n')


# ----------------------------------------------------------------- 输出
def parse_time(t):
    """'2:07.8' / '1:02:07' -> 毫秒"""
    sec_ = 0.0
    for part in t.split(':'):
        sec_ = sec_ * 60 + float(part)
    return int(round(sec_ * 1000))


def pct(x):  return '%5.1f%%' % (100.0 * x)
def sec(ms): return '%.1fs' % (ms / 1000.0)


def report_gt(path, meta, segs, dur, errs, policy):
    per = {}
    for s, e, l in segs:
        per[l] = per.get(l, 0) + (e - s)
    print('══ 真值 %s ══' % path)
    print('  片长 %.1f min   段数 %d   标注者 %s   split %s   spec v%s'
          % (dur / 60000.0, len(segs), meta.get('annotator', '?'),
             meta.get('split', '?'), meta.get('spec_version', '?')))
    print('  ' + '  '.join('%s %s' % (l, pct(ms / dur))
                           for l, ms in sorted(per.items(), key=lambda x: -x[1])))
    oa = per.get('other_adult', 0)
    print('  other_adult 策略: %s（%s = %s%s）'
          % (policy, sec(oa), pct(oa / dur),
             '，不计入指标' if policy == 'exclude' else '，按 %s 计入' % policy))
    tr = gt_transitions(segs, policy)
    up = sum(1 for _, f, t in tr if t == 'do')
    dn = len(tr) - up
    print('  真值切换点 %d 个（noise→do %d，do→noise %d）' % (len(tr), up, dn))
    for e in errs:
        print('  ✗ %s' % e)
    if len(tr) < 5:
        print('  ⚠ 切换点仅 %d 个 —— 延迟指标样本不足（规范 §7 要求 ≥5）' % len(tr))
    if dn == 0:
        print('  ⚠ 无 do→noise 样本 —— 本片测不到「误停」和停转延迟')
    if not errs:
        print('  ✓ 格式自检通过')
    print()


def report_run(path, meta, rows, m):
    print('══ 回放 %s ══' % path)
    d = [rows[i]['pos'] - rows[i - 1]['pos'] for i in range(1, len(rows))]
    print('  tick %d 个   覆盖 pos %s~%s   中位间隔 %dms'
          % (len(rows), sec(rows[0]['pos']), sec(rows[-1]['pos']),
             statistics.median(d) if d else 0))
    back = [x for x in d if x < -1000]
    if back:
        print('  ⚠ 检测到 %d 处 pos 回退（seek/重播）—— 延迟指标不可靠，建议从头不间断重跑' % len(back))
    if meta.get('params'):
        print('  参数 %s' % meta['params'])
    print()

    print('══ 指标 1：逐秒判定准确率 ══')
    print('  准确率  %s   (%d/%d 格 @%dms，可评测 %.1f min)'
          % (pct(m['acc']), m['cm'][('noise', 'noise')] + m['cm'][('do', 'do')],
             m['tot'], GRID_MS, m['eval_ms'] / 60000.0))
    print('                 真值不转   真值转')
    print('     系统不转   %8d %8d' % (m['cm'][('noise', 'noise')], m['cm'][('do', 'noise')]))
    print('     系统转     %8d %8d' % (m['cm'][('noise', 'do')], m['cm'][('do', 'do')]))
    if m['gt_noise']:
        print('  误转（剧情中转）    %s 的"不转"时长' % pct(m['cm'][('noise', 'do')] / m['gt_noise']))
    if m['gt_do']:
        print('  漏转（动作中不转）  %s 的"转"时长' % pct(m['cm'][('do', 'noise')] / m['gt_do']))
    mins = m['eval_ms'] / 60000.0
    print('  动作中误停          %d 次（%.2f 次/分钟）'
          % (m['false_stop'], m['false_stop'] / mins if mins else 0))
    print()

    print('══ 指标 2：切换延迟（真值切换点 → 蓝牙指令）══')
    for key, name in (('noise->do', '启动 noise→do'), ('do->noise', '停转 do→noise')):
        v = m['lat'][key]
        if not v:
            print('  %-14s n=0   （本片无样本%s）'
                  % (name, '，%d 个漏切' % m['miss'][key] if m['miss'][key] else ''))
            continue
        sv = sorted(v)
        p90 = sv[min(len(sv) - 1, int(round(0.9 * (len(sv) - 1))))]
        print('  %-14s n=%d   中位 %s   p90 %s   最坏 %s   漏切 %d'
              % (name, len(v), sec(statistics.median(v)), sec(p90), sec(max(v)), m['miss'][key]))
        dd = m['dec'][key]
        if dd:
            print('  %-14s 分解(中位)  感知 %s  →  融合 %s  →  状态机 %s'
                  % ('', sec(statistics.median(x[0] for x in dd)),
                     sec(statistics.median(x[1] for x in dd)),
                     sec(statistics.median(x[2] for x in dd))))
    print()


def append_summary(path, gt_meta, run_path, m, policy):
    head = ['video_id', 'run', 'policy', 'acc', 'lat_up_med_ms', 'lat_up_p90_ms',
            'lat_down_med_ms', 'false_run', 'miss', 'false_stop_per_min', 'eval_min', 'params']
    up, dn = m['lat']['noise->do'], m['lat']['do->noise']
    sup = sorted(up)
    mins = m['eval_ms'] / 60000.0
    row = [gt_meta.get('video_id', '?'), os.path.basename(run_path), policy,
           '%.4f' % m['acc'],
           int(statistics.median(up)) if up else '',
           sup[min(len(sup) - 1, int(round(0.9 * (len(sup) - 1))))] if up else '',
           int(statistics.median(dn)) if dn else '',
           '%.4f' % (m['cm'][('noise', 'do')] / m['gt_noise']) if m['gt_noise'] else '',
           '%.4f' % (m['cm'][('do', 'noise')] / m['gt_do']) if m['gt_do'] else '',
           '%.3f' % (m['false_stop'] / mins) if mins else '',
           '%.2f' % mins, m.get('params', '')]
    new = not os.path.exists(path)
    with io.open(path, 'a', encoding='utf-8', newline='') as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(head)
        w.writerow(row)
    print('已追加一行到 %s' % path)


# ----------------------------------------------------------------- 自检
# metrics.py 读的列名和 BenchLogger.java 写的列名必须一致。这是跨语言的隐式契约，
# 一旦有人加列/改名而没同步，指标会静默算错——所以直接把 Java 源码里的表头抠出来比对。
RUN_COLS_NEEDED = ['pos_ms', 'wall_ms',
                   'v_action', 'v_conf', 'v_age', 'v_p_plot', 'v_p_oral', 'v_p_sex',
                   'a_action', 'a_conf', 'a_age', 'a_p_sex', 'a_p_oral', 'a_p_noise',
                   'loud_lv', 'loud_conf', 'loud_valid', 'loud_age',
                   'smoothed', 'gated', 'final_freq',
                   'ble_state', 'ble_level', 'sent', 'sent_action', 'sent_level',
                   # 音频窗口结尾的播放位置，给 audio_rescore.py 还原模型窗口用
                   'a_pos',
                   # FusionArbiter 本 tick 的输出原因（start_both / hold_audio / stop_neg …）
                   'arb',
                   # 这一行之前 FusionArbiter reset 的时刻（seek / 恢复播放；没有为空），回放时同样 reset
                   'arb_reset']


def _java_header():
    j = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     '..', '..', 'app', 'src', 'main', 'java', 'com', 'example',
                     'helloworld', 'BenchLogger.java')
    if not os.path.exists(j):
        return None
    txt = io.open(j, encoding='utf-8').read()
    i = txt.find('out.write("pos_ms,')
    if i < 0:
        return []
    k = txt.find(';', i)
    return ''.join(re.findall(r'"((?:[^"\\]|\\.)*)"', txt[i:k])).replace('\\n', '').strip().split(',')


def check_contract():
    cols = _java_header()
    if cols is None:
        print('  skip BenchLogger.java 不在当前路径，跳过列名契约检查')
        return []
    miss = [c for c in RUN_COLS_NEEDED if c not in cols]
    extra = [c for c in cols if c not in RUN_COLS_NEEDED]
    out = []
    if miss:
        out.append('BenchLogger.java 的表头缺少 metrics.py 需要的列: %s' % miss)
    if extra:
        out.append('BenchLogger.java 多了 metrics.py 不认识的列（请同步 RUN_COLS_NEEDED）: %s' % extra)
    if not out:
        print('  ok   列名契约一致（%d 列）' % len(cols))
    return out


def _write_run_csv(path, rows):
    """按 BenchLogger 的真实格式落盘，用来验证 read_run 的解析路径。"""
    with io.open(path, 'w', encoding='utf-8', newline='') as fh:
        fh.write('# xcup-bench-run v1\n# video_name=selftest\n')
        fh.write('# params: VIDEO_DO=0.65 AUDIO_DO=0.55\n')
        fh.write(','.join(RUN_COLS_NEEDED) + '\n')
        w = csv.writer(fh)
        for r in rows:
            w.writerow([r['pos'], r['wall'],
                        r['v'], '%.4f' % r['v_conf'], r['v_age'], '0.1000', '0.4000', '0.5000',
                        r['a'], '%.4f' % r['a_conf'], r['a_age'], '0.5000', '0.3000', '0.2000',
                        r['ble_level'], '0.8000', 1, 300,
                        r['smoothed'], r['gated'], r['ble_level'],
                        r['ble_state'], r['ble_level'],
                        1 if r['sent'] else 0, r['sent_action'], r['sent_level'],
                        -1, '', ''])


def selftest():
    """用真值合成一份"延迟固定 2000ms 的完美系统"的 run CSV，验证指标算得对。"""
    segs = [(0, 6900, 'exclude'), (6900, 214503, 'plot'), (214503, 318624, 'other_adult'),
            (318624, 393926, 'oral'), (393926, 407292, 'other_adult'),
            (407292, 643782, 'sex'), (643782, 652131, 'other_adult'),
            (652131, 805496, 'sex'), (805496, 827137, 'other_adult')]
    dur = 827137
    LAG = 2000
    fails = []

    def synth(lag, policy='exclude', inject_false_stop_at=None):
        """合成一个"感知完美、整体滞后 lag 毫秒"的系统。"""
        rows, cur = [], 'noise'
        for pos in range(0, dur, 500):
            want = None
            for s, e, l in segs:
                if s <= pos - lag < e:
                    want = bin_of(l, policy)
            if want is None:
                want = cur
            if inject_false_stop_at and inject_false_stop_at <= pos < inject_false_stop_at + 500:
                want = 'noise'
            sent = want != cur
            if sent:
                cur = want
            rows.append({'pos': pos, 'wall': pos, 'v': want, 'v_conf': .9, 'v_age': 300,
                         'a': want, 'a_conf': .9, 'a_age': 300, 'smoothed': want,
                         'gated': want, 'ble_state': cur, 'ble_level': 4,
                         'sent': sent, 'sent_action': want if sent else '', 'sent_level': 4})
        return rows

    m = evaluate(segs, dur, synth(LAG), 'exclude')
    up = m['lat']['noise->do']
    if len(up) != 1:
        fails.append('应找到 1 个 noise→do 样本，实际 %d' % len(up))
    elif not (1500 <= statistics.median(up) <= 2500):
        fails.append('注入延迟 2000ms，测出 %dms' % statistics.median(up))
    if not m['acc'] > 0.99:
        fails.append('完美系统的准确率应 >99%%，实际 %.3f' % m['acc'])
    if m['false_stop'] != 0:
        fails.append('完美系统不该有误停，实际 %d' % m['false_stop'])

    m2 = evaluate(segs, dur, synth(LAG, inject_false_stop_at=500000), 'exclude')
    if m2['false_stop'] != 1:
        fails.append('注入 1 次动作中误停，测出 %d' % m2['false_stop'])

    # policy=noise 下真值有 6 个切换点（3 启动 + 3 停转），系统也按同策略跟随 ->
    # 两个方向都应测到，且延迟都接近注入值。
    if len(gt_transitions(segs, 'noise')) != 6:
        fails.append('policy=noise 下应有 6 个切换点，实际 %d' % len(gt_transitions(segs, 'noise')))
    m3 = evaluate(segs, dur, synth(LAG, 'noise'), 'noise')
    if len(m3['lat']['noise->do']) != 3 or len(m3['lat']['do->noise']) != 3:
        fails.append('policy=noise 下应测到 3+3 个样本，实际 %d+%d'
                     % (len(m3['lat']['noise->do']), len(m3['lat']['do->noise'])))
    else:
        for k in ('noise->do', 'do->noise'):
            med = statistics.median(m3['lat'][k])
            if not (1500 <= med <= 2500):
                fails.append('%s 注入 2000ms，测出 %dms' % (k, med))

    # 系统从不停转时，停转切换点必须全部记成漏切，而不是悄悄不算
    m5 = evaluate(segs, dur, synth(LAG, 'exclude'), 'noise')
    if m5['miss']['do->noise'] != 3:
        fails.append('系统从不停转时应记 3 次停转漏切，实际 %d' % m5['miss']['do->noise'])

    nosend = [dict(r, sent=False, sent_action='') for r in synth(LAG)]
    m4 = evaluate(segs, dur, nosend, 'exclude')
    if m4['miss']['noise->do'] != 1:
        fails.append('一条指令都没发时应记 1 次漏切，实际 %d' % m4['miss']['noise->do'])

    # CSV 落盘 -> 解析 的往返，确保 Java 写出来的东西 Python 读得对
    fails += check_contract()
    import tempfile
    tmp = os.path.join(tempfile.gettempdir(), 'xcup_selftest_run.csv')
    base = synth(LAG)
    _write_run_csv(tmp, base)
    rmeta, rr = read_run(tmp)
    if rmeta.get('params') != 'VIDEO_DO=0.65 AUDIO_DO=0.55':
        fails.append('run CSV 的 params 头没解析出来: %r' % rmeta.get('params'))
    if len(rr) != len(base):
        fails.append('CSV 往返行数不一致: %d vs %d' % (len(rr), len(base)))
    else:
        for k in ('pos', 'v', 'a', 'smoothed', 'gated', 'sent', 'sent_action'):
            if [x[k] for x in rr] != [x[k] for x in base]:
                fails.append('CSV 往返列 %s 不一致' % k)
        if any(x['ys'] != [-1] * len(YS_COLS) for x in rr):
            fails.append('app 格式的 run 没有 a_ys_* 列，应读成 -1')
    m6 = evaluate(segs, dur, rr, 'exclude')
    if abs(m6['acc'] - m['acc']) > 1e-9 or m6['lat'] != m['lat']:
        fails.append('从 CSV 读回来算出的指标与内存版不一致')
    try: os.remove(tmp)
    except OSError: pass

    for f in fails:
        print('  FAIL ' + f)
    print('selftest: %s' % ('FAILED (%d)' % len(fails) if fails else 'ALL PASS'))
    return 1 if fails else 0


# ----------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(add_help=True)
    ap.add_argument('gt', nargs='?', help='bench/gt/<id>.csv')
    ap.add_argument('run', nargs='?', help='BenchLogger 产出的 run CSV')
    ap.add_argument('--policy', default='exclude', choices=('exclude', 'do', 'noise'),
                    help='other_adult 怎么算（默认 exclude：不计入）')
    ap.add_argument('--append', metavar='summary.csv', help='把本次结果追加一行，便于跨参数对比')
    ap.add_argument('--errors', action='store_true', help='列出误判区间及各层当时的判定')
    ap.add_argument('--top', type=int, default=10, help='--errors 显示前几段（默认 10）')
    ap.add_argument('--at', help='打印某时刻附近的逐 tick 原始数据。127.8 / 127.8s = 秒，127800 = 毫秒，2:07.8 也可')
    ap.add_argument('--diag', action='store_true', help='模态诊断：单模态准确率、AUC、阈值扫描')
    ap.add_argument('--selftest', action='store_true')
    a = ap.parse_args()

    if a.selftest:
        sys.exit(selftest())
    if not a.gt:
        ap.error('需要真值 CSV（或用 --selftest）')

    for label, path in (('真值 CSV', a.gt), ('run CSV', a.run)):
        if path and not os.path.exists(path):
            sys.exit('找不到%s: %s\n'
                     '  · 路径是相对当前目录的，用 "dir" / "ls" 确认文件名\n'
                     '  · 文件名含空格或中文时整个路径要加英文双引号\n'
                     '  · 建议把 run CSV 放到 bench/runs/ 下，命令会短很多'
                     % (label, path))

    gt_meta, segs, dur, errs = read_gt(a.gt)
    report_gt(a.gt, gt_meta, segs, dur, errs, a.policy)
    if not a.run:
        print('（没给 run CSV，只做了真值校验。拿到真机 dump 后再跑一次就能出指标。）')
        return

    run_meta, rows = read_run(a.run)
    if not rows:
        sys.exit('%s: 没有数据行' % a.run)
    m = evaluate(segs, dur, rows, a.policy)
    m['params'] = run_meta.get('params', '')
    report_run(a.run, run_meta, rows, m)
    if a.errors:
        report_errors(segs, rows, a.policy, m['strans'], a.top)
    if a.at:
        t = a.at.strip().rstrip('s')
        tms = parse_time(t) if ':' in t else int(round(float(t) * (1000 if '.' in t or a.at.strip().endswith('s') else 1)))
        report_at(rows, tms)
    if a.diag:
        report_diag(segs, rows, a.policy)
    if a.append:
        append_summary(a.append, gt_meta, a.run, m, a.policy)


if __name__ == '__main__':
    main()
