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
            'smoothed': _norm(d.get('smoothed')), 'gated': _norm(d.get('gated')),
            'ble_state': _norm(d.get('ble_state')), 'ble_level': i('ble_level'),
            'sent': i('sent') == 1, 'sent_action': _norm(d.get('sent_action')),
            'sent_level': i('sent_level', -1),
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


# ----------------------------------------------------------------- 输出
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
                   'ble_state', 'ble_level', 'sent', 'sent_action', 'sent_level']


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
                        1 if r['sent'] else 0, r['sent_action'], r['sent_level']])
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
    ap.add_argument('--selftest', action='store_true')
    a = ap.parse_args()

    if a.selftest:
        sys.exit(selftest())
    if not a.gt:
        ap.error('需要真值 CSV（或用 --selftest）')

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
    if a.append:
        append_summary(a.append, gt_meta, a.run, m, a.policy)


if __name__ == '__main__':
    main()
