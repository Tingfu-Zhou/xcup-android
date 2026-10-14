#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
bench/tools/baseline.py — 统一评测基准：比视频模型 / 视频管线时，其它全部固定，只换视频结果，在同一套录像上比。

标准（2026-10-14 定）
  片子 / 真值  bench/gt 的 v01~v13
  录像         runs_initial（v01~v07）+ runs_test（v08~v13）：app 1.2.5 从头播到尾录的，提供 tick 时间轴和音频结果。
               同一部片录两次结果就能差很多（v11 用同一套规则：runs_test 90.2%，runs_v2 82.5%），所以只用这一套录像，
               runs_v2 和以后的新录像都不混进来
  音频线索     rescored/audio_old、rescored/test_audio_old 里同名的 .yamnet.npz
  融合         app 现在的 FusionArbiter v2（= cue_lab 的 a1bvid），参数用 run 文件头里的（= app 现用的）
  代价         误转秒数 × 1 + 漏转秒数 × 1 + 动作中误停次数 × 3；other_adult 不计入
  视频结果     按"变体"的名字找：
                 phone       手机原始 run，旧视频管线（只取关键帧 + 整帧压扁）。README 里的数字都是这个口径
                 exact_crop  video_rescore.py 按修复后的 app 离线重算，现用视频模型：以后换模型都和它比
                 <名字>      rescore --model <新模型> --name <名字> 重算的
               phone 读 bench/runs_initial、bench/runs_test；其余读 bench/rescored/video_<名字>_runs_initial、_runs_test
  指标         端到端：代价、准确率、误转、漏转、误停、seek 进动作段多久启动（与 cue_lab.py eval 的"新："一段相同）；
               视频单独：p_oral+p_sex 的判别力 AUC（不受阈值影响，最能说明模型本身）和按现用阈值判定的准确率，
               逐片 + 13 部合并（逐 tick 统计，与 metrics.py --diag 相同）

用法（仓库根目录；Windows 把 python3 换成 python）
  python3 bench/tools/baseline.py eval phone                          # 旧管线，数字与 README 第 6 节一致
  python3 bench/tools/baseline.py check   --videos D:/xcup/videos     # 第一次 rescore 之前：核对原片，离线能复现手机
  python3 bench/tools/baseline.py rescore --videos D:/xcup/videos     # 修复后的管线 + 现用模型 → 变体 exact_crop
  python3 bench/tools/baseline.py eval exact_crop
  python3 bench/tools/baseline.py compare phone exact_crop            # 并排对比，括号里是相对第一个的变化
  python3 bench/tools/baseline.py rescore --videos D:/xcup/videos --model D:/models/new1.tflite --name new1
  python3 bench/tools/baseline.py compare exact_crop new1
  python3 bench/tools/baseline.py --selftest
check / rescore 包装 video_rescore.py，13 部（两个 runs 目录）一次做完；--jobs N 几部片并行，--only v01,v04 只做几部，
原片名字对不上时 --pair v01=D:/xcup/视频1.mp4（可多次）。
eval / compare 的报告同时写进 bench/baseline/（UTF-8 文本；eval 另写 JSON），不用自己重定向。
重算目录里的 variant.json 记下模型文件的指纹：换了模型还往同一个名字里写会被拒绝（防止 exact_crop 被悄悄换掉）。

不在这里做的：给新模型调阈值和融合参数（README 第 4 节）。最后决定换不换模型时，两边各自用同一套流程调好再比。
"""
import argparse
import contextlib
import glob
import hashlib
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import unicodedata

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fusion_lab as F        # noqa: E402
import arbiter_lab as A       # noqa: E402
import cue_lab as C           # noqa: E402
import metrics as M           # noqa: E402
import video_rescore as V     # noqa: E402

TOOLS = os.path.dirname(os.path.abspath(__file__))
ROOT = F.ROOT
GT_DIR = os.path.join(ROOT, 'bench', 'gt')
RESCORED = os.path.join(ROOT, 'bench', 'rescored')       # 视频变体的位置（自检时临时换掉）
OUT_DIR = os.path.join(ROOT, 'bench', 'baseline')
RUNS = ('runs_initial', 'runs_test')                       # 录像：v01~v07、v08~v13
NPZ_DIRS = tuple(os.path.join(ROOT, 'bench', 'rescored', x) for x in ('audio_old', 'test_audio_old'))
FILMS = tuple('v%02d' % i for i in range(1, 14))
PRESET = 'a1bvid'           # = app 的 FusionArbiter v2；app 的融合规则改了，这里跟着改
COST = 'fp=1,fn=1,stop=3'
POLICY = 'exclude'
PHONE = 'phone'
DEFAULT_PIPE = 'exact_crop'
MANIFEST = 'variant.json'
NAME_RE = re.compile(r'^[A-Za-z0-9][A-Za-z0-9_.-]*$')
MIN_HIT = 0.95              # rescore：换了概率的 tick 少于这个比例就提醒（正常只差片头几个）


# ======================================================================== 小工具
def rel(path):
    try:
        r = os.path.relpath(path, ROOT)
    except ValueError:      # Windows 上跨盘符
        return path
    return path if r.startswith('..') else r.replace('\\', '/')


def width(s):
    return sum(2 if unicodedata.east_asian_width(c) in 'WF' else 1 for c in s)


def lpad(s, n):
    return s + ' ' * max(0, n - width(s))


def rpad(s, n):
    return ' ' * max(0, n - width(s)) + s


def pct(x):
    return '—' if x is None or x != x else '%.1f%%' % (100.0 * x)


def num(x, fmt):
    return '—' if x is None or x != x else fmt % x


def tool_version():
    """当前提交，工具或 app 有没提交的改动时注明（数字可能和这个提交跑出来的不同）。"""
    try:
        h = subprocess.run(['git', '-C', ROOT, 'rev-parse', '--short', 'HEAD'],
                           capture_output=True, text=True, timeout=10).stdout.strip()
        dirty = subprocess.run(['git', '-C', ROOT, 'status', '--porcelain', '--', 'bench/tools', 'app/src/main'],
                               capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception:
        return '?'
    return (h or '?') + ('（bench/tools 或 app 有没提交的改动）' if dirty else '')


def model_info(path):
    h = hashlib.sha1()
    with open(path, 'rb') as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b''):
            h.update(chunk)
    return dict(file=os.path.basename(path), sha1=h.hexdigest()[:12], size=os.path.getsize(path))


def variant_dirs(name):
    if name == PHONE:
        return [os.path.join(ROOT, 'bench', r) for r in RUNS]
    return [os.path.join(RESCORED, 'video_%s_%s' % (name, r)) for r in RUNS]


def read_manifest(d):
    p = os.path.join(d, MANIFEST)
    if not os.path.exists(p):
        return None
    with io.open(p, encoding='utf-8') as fh:
        return json.load(fh)


def write_manifest(d, info):
    with io.open(os.path.join(d, MANIFEST), 'w', encoding='utf-8') as fh:
        json.dump(info, fh, ensure_ascii=False, indent=1, sort_keys=True)


def rescore_note(path):
    """video_rescore.py rescore 写在文件头的那一行 → {pipe, matrix, delta, model}；不是重算的 run 返回 None。"""
    with io.open(path, encoding='utf-8-sig') as fh:
        for line in fh:
            s = line.strip()
            if s and not s.startswith('#'):
                break
            m = re.match(r'#\s*video_rescore\.py rescore:\s*(.*)', s)
            if m:
                return dict(kv.split('=', 1) for kv in m.group(1).split() if '=' in kv)
    return None


def _write(path, text):
    d = os.path.dirname(path)
    if d and not os.path.isdir(d):
        os.makedirs(d)
    with io.open(path, 'w', encoding='utf-8', newline='\n') as fh:
        fh.write(text)


# ======================================================================== 评测
def _vscore(r):
    return r['vp'][1] + r['vp'][2] if r['vp'][1] >= 0 else None


def video_only(video):
    """视频单独：逐 tick，与 metrics.py --diag 相同（预热期不算，other_adult 按 POLICY）。"""
    _, rows = M.read_run(video.run_path)
    pos, neg = M._split(video.segs, rows, POLICY, _vscore)
    n = dict(ok=0, all=0, noise=0, fp=0, do=0, fn=0)
    for r in rows:
        if r['pos'] < M.WARMUP_MS:
            continue
        g = M.bin_of(M.gt_label_at(video.segs, r['pos']), POLICY)
        if not g or not r['v']:
            continue
        n['all'] += 1
        n['ok'] += r['v'] == g
        if g == 'noise':
            n['noise'] += 1
            n['fp'] += r['v'] == 'do'
        else:
            n['do'] += 1
            n['fn'] += r['v'] == 'noise'
    return pos, neg, n


def _rates(n):
    return dict(acc=n['ok'] / float(n['all']) if n['all'] else float('nan'),
                fp=n['fp'] / float(n['noise']) if n['noise'] else float('nan'),
                fn=n['fn'] / float(n['do']) if n['do'] else float('nan'))


def _source(name, videos, dirs):
    """视频结果的出处：一句话 + 警告。"""
    warns = []
    notes = dict((v.vid, rescore_note(v.run_path)) for v in videos)
    if name == PHONE:
        mixed = sorted(vid for vid, x in notes.items() if x)
        if mixed:
            warns.append('phone 的目录里混进了重算的 run：%s' % ' '.join(mixed))
        return '手机原始 run（旧视频管线：只取关键帧 + 整帧压扁）', warns
    bad = sorted(vid for vid, x in notes.items() if not x)
    if bad:
        warns.append('这几部不是 video_rescore.py 重算的 run：%s' % ' '.join(bad))
    kinds = sorted(set((x.get('pipe', '?'), x.get('model', '?'), x.get('delta', x.get('offset', '?')))
                       for x in notes.values() if x))
    if len(kinds) > 1:
        warns.append('13 部不是同一种重算：%s' % '；'.join('pipe=%s model=%s %s' % k for k in kinds))
    mans = [read_manifest(d) for d in dirs]
    shas = sorted(set(m['model']['sha1'] for m in mans if m and m.get('model')))
    if len(shas) > 1:
        warns.append('两个重算目录用的模型不同（sha1 %s），重新 rescore' % ' / '.join(shas))
    if not kinds:
        return '?', warns
    pipe, model, delta = kinds[0]
    return 'video_rescore.py 重算：pipe=%s，模型 %s%s，Δ=%s' % (
        pipe, model, '（sha1 %s）' % shas[0] if len(shas) == 1 else '', delta), warns


def evaluate(name):
    """name 变体在统一标准下的全部数字（dict，可写成 JSON）。"""
    dirs = variant_dirs(name)
    missing = [d for d in dirs if not os.path.isdir(d)]
    if missing:
        hint = 'python3 bench/tools/baseline.py rescore --videos <原片目录>' + (
            '' if name == DEFAULT_PIPE else ' --model <模型文件> --name %s' % name)
        sys.exit('找不到变体 %s 的数据：%s\n先运行：%s' % (name, '、'.join(rel(d) for d in missing), hint))
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        d = C.load(GT_DIR, ','.join(dirs), ','.join(NPZ_DIRS), seeks=True, only=','.join(FILMS))
    warns = [ln.strip() for ln in buf.getvalue().splitlines() if ln.strip()]
    have = [v.vid for v in d.videos]
    absent = [f for f in FILMS if f not in have]
    if absent:
        warns.append('缺这几部（标准是 13 部，数字不能和别的变体比）：%s' % ' '.join(absent))
    cue_bad = ['%s %d/%d' % (vid, h, t) for vid, (h, t) in sorted(d.coverage.items()) if t and h < 0.99 * t]
    if cue_bad:
        warns.append('音频线索没接全（YAMNet 分数对不上时间戳）：%s' % '，'.join(cue_bad))
    source, w2 = _source(name, d.videos, dirs)
    warns += w2
    ths = sorted(set((v.params['VIDEO_DO'], v.params['VIDEO_PLOT']) for v in d.videos))
    if len(ths) > 1:
        warns.append('各片的视频阈值不同：%s' % ths)

    cfg = C.make_cfg(**C.PRESETS[PRESET])
    cost = F.Cost.parse(COST)
    rep = A.evaluate(d, C.factory(cfg), cost)
    s = A.summary(rep)
    pv = C.per_video(d, cfg, cost)

    per, P, N, tot = {}, [], [], dict(ok=0, all=0, noise=0, fp=0, do=0, fn=0)
    for v in d.videos:
        pos, neg, n = video_only(v)
        P += pos
        N += neg
        for k in tot:
            tot[k] += n[k]
        x = rep['videos'][v.vid]
        vr = _rates(n)
        per[v.vid] = dict(cost=pv[v.vid]['cost'], acc=x['acc'], fp=x['fp'], fn=x['fn'], stops=x['stops'],
                          long_miss=x['long_miss'], long_false=x['long_false'],
                          v_auc=M._auc(pos, neg), v_n=[len(pos), len(neg)],
                          v_acc=vr['acc'], v_fp=vr['fp'], v_fn=vr['fn'])
    aucs = [(x['v_auc'], vid) for vid, x in per.items() if x['v_auc'] == x['v_auc']]
    worst = min(aucs) if aucs else (float('nan'), '')
    vt = _rates(tot)
    keys = ('cost', 'acc', 'fp', 'fn', 'stops', 'stops_pm', 'worst_fn', 'worst_long_miss', 'stop_med', 'stop_stuck',
            'start_med', 'start_stuck', 'seek_med', 'seek_p90', 'seek_never', 'seek_n', 'hard_med', 'hard_p90',
            'hard_never', 'hard_n', 'plot_on', 'plot_any', 'plot_n')
    return dict(
        name=name, created=time.strftime('%Y-%m-%d %H:%M'), tool=tool_version(),
        dirs=[rel(x) for x in dirs], source=source, warnings=warns, films=have,
        thresholds=list(ths[0]) if ths else None, preset=PRESET, cost_rule=COST, policy=POLICY,
        e2e=dict((k, s[k]) for k in keys), per_film=per,
        video=dict(auc=M._auc(P, N), auc_mean=sum(a for a, _ in aucs) / len(aucs) if aucs else float('nan'),
                   auc_worst=worst[0], auc_worst_vid=worst[1], n=[len(P), len(N)],
                   acc=vt['acc'], fp=vt['fp'], fn=vt['fn']))


# ======================================================================== 报告
def report(r):
    e, v = r['e2e'], r['video']
    th = r['thresholds'] or [None, None]
    out = ['══ 统一基准：%s ══' % r['name'],
           '视频结果  %s' % r['source'],
           'run 目录  %s（%d 部）；音频线索 %s' % (' + '.join(r['dirs']), len(r['films']),
                                              ' + '.join(rel(x) for x in NPZ_DIRS)),
           '融合      %s（= app 的 FusionArbiter v2）；视频阈值 转 ≥%s / 剧情 ≥%s；代价 %s；other_adult 不计入'
           % (r['preset'], num(th[0], '%.2f'), num(th[1], '%.2f'), r['cost_rule']),
           '工具      %s，%s' % (r['tool'], r['created'])]
    for w in r['warnings']:
        out.append('⚠ ' + w)
    out += ['',
            '端到端（13 部片 + 拼接样本 + seek 模拟）',
            '  代价 %.0f   准确率 %s   误转 %s   漏转 %s   误停 %d 次（%.2f 次/分）'
            % (e['cost'], pct(e['acc']), pct(e['fp']), pct(e['fn']), e['stops'], e['stops_pm']),
            '  最差片漏转 %s，最长连续漏转 %.1fs' % (pct(e['worst_fn']), e['worst_long_miss']),
            '  拼接样本：停止后还转 中位 %ss（≥40s %d 个）；启动后没转 中位 %ss（≥40s %d 个）'
            % (num(e['stop_med'], '%.1f'), e['stop_stuck'], num(e['start_med'], '%.1f'), e['start_stuck']),
            '  seek 进动作段：中位 %ss，p90 %ss，没启动 %d/%d；其中特写/低声音：p90 %ss，没启动 %d/%d'
            % (num(e['seek_med'], '%.1f'), num(e['seek_p90'], '%.1f'), e['seek_never'], e['seek_n'],
               num(e['hard_p90'], '%.1f'), e['hard_never'], e['hard_n']),
            '  seek 进剧情段：30 秒里平均误转 %ss，%d/%d 次转过' % (num(e['plot_on'], '%.1f'), e['plot_any'], e['plot_n']),
            '',
            '视频单独（p_oral+p_sex 的判别力 AUC 不受阈值影响；准确率按现用阈值，只算视频给了判定的 tick）',
            '  AUC：13 部合并 %s，逐片平均 %s，最差 %s %s'
            % (num(v['auc'], '%.3f'), num(v['auc_mean'], '%.3f'), v['auc_worst_vid'], num(v['auc_worst'], '%.3f')),
            '  准确率 %s，剧情判成转 %s，动作判成不转 %s' % (pct(v['acc']), pct(v['fp']), pct(v['fn'])),
            '']
    head = ['逐片', '代价', '准确率', '误转', '漏转', '误停', '最长漏转', '最长误转', '│', '视频AUC', '视频准确率']
    wid = [5, 8, 8, 8, 8, 5, 9, 9, 2, 9, 12]
    out.append(''.join(lpad(h, w) if i == 0 else rpad(h, w) for i, (h, w) in enumerate(zip(head, wid))))
    for vid in sorted(r['per_film']):
        x = r['per_film'][vid]
        cells = [vid, '%.1f' % x['cost'], pct(x['acc']), pct(x['fp']), pct(x['fn']), '%d' % x['stops'],
                 '%.1fs' % x['long_miss'], '%.1fs' % x['long_false'], '│', num(x['v_auc'], '%.3f'), pct(x['v_acc'])]
        out.append(''.join(lpad(c, w) if i == 0 else rpad(c, w) for i, (c, w) in enumerate(zip(cells, wid))))
    out.append('（逐片代价含这部片的拼接样本；视频 AUC 是"—"的片子没有剧情段）')
    return '\n'.join(out) + '\n'


# 对比表：(标签, 取值, 显示格式, 变化的格式, 乘数)
_ROWS = [
    ('代价', lambda r: r['e2e']['cost'], '%.0f', '%+.0f', 1),
    ('准确率', lambda r: r['e2e']['acc'], 'pct', '%+.1f', 100),
    ('误转', lambda r: r['e2e']['fp'], 'pct', '%+.1f', 100),
    ('漏转', lambda r: r['e2e']['fn'], 'pct', '%+.1f', 100),
    ('误停（次）', lambda r: r['e2e']['stops'], '%d', '%+d', 1),
    ('最差片漏转', lambda r: r['e2e']['worst_fn'], 'pct', '%+.1f', 100),
    ('最长连续漏转（秒）', lambda r: r['e2e']['worst_long_miss'], '%.1f', '%+.1f', 1),
    ('seek 进动作段 p90（秒）', lambda r: r['e2e']['seek_p90'], '%.1f', '%+.1f', 1),
    ('  其中特写/低声音 p90', lambda r: r['e2e']['hard_p90'], '%.1f', '%+.1f', 1),
    ('seek 后没启动（次）', lambda r: r['e2e']['seek_never'], '%d', '%+d', 1),
    ('seek 进剧情 平均误转（秒）', lambda r: r['e2e']['plot_on'], '%.1f', '%+.1f', 1),
    ('视频 AUC（13 部合并）', lambda r: r['video']['auc'], '%.3f', '%+.3f', 1),
    ('视频 AUC（逐片平均）', lambda r: r['video']['auc_mean'], '%.3f', '%+.3f', 1),
    ('视频 AUC（最差片）', lambda r: r['video']['auc_worst'], '%.3f', '%+.3f', 1),
    ('视频准确率（现用阈值）', lambda r: r['video']['acc'], 'pct', '%+.1f', 100),
]


def _cell(x, fmt):
    if x is None or x != x:
        return '—'
    return pct(x) if fmt == 'pct' else fmt % x


def _delta(x, x0, fmt, mul):
    if x is None or x0 is None or x != x or x0 != x0:
        return ''
    return '(' + fmt % ((x - x0) * mul) + ')'


def compare_text(results):
    names = [r['name'] for r in results]
    w0, w = 28, max(16, max(width(n) for n in names) + 4)
    out = ['══ 统一基准对比：%s ══' % ' / '.join(names),
           '括号里是相对 %s 的变化（百分比的变化是百分点）。代价、误转、漏转、误停、seek 的秒数越低越好；准确率、AUC 越高越好' % names[0]]
    for r in results:
        out.append('  %s：%s' % (r['name'], r['source']))
        for x in r['warnings']:
            out.append('    ⚠ ' + x)
    out.append('')
    out.append(lpad('', w0) + ''.join(rpad(n, w) for n in names))
    for label, get, fmt, dfmt, mul in _ROWS:
        x0 = get(results[0])
        cells = []
        for i, r in enumerate(results):
            x = get(r)
            c = _cell(x, fmt)
            if i:
                c += ' ' + _delta(x, x0, dfmt, mul)
            cells.append(c)
        out.append(lpad(label, w0) + ''.join(rpad(c, w) for c in cells))
    for title, key, fmt, dfmt in (('逐片代价（含拼接样本）', 'cost', '%.1f', '%+.1f'),
                                  ('逐片视频 AUC', 'v_auc', '%.3f', '%+.3f')):
        out += ['', title]
        vids = sorted(set(v for r in results for v in r['per_film']))
        for vid in vids:
            x0 = results[0]['per_film'].get(vid, {}).get(key)
            cells = []
            for i, r in enumerate(results):
                x = r['per_film'].get(vid, {}).get(key)
                c = _cell(x, fmt)
                if i:
                    c += ' ' + _delta(x, x0, dfmt, 1)
                cells.append(c)
            out.append(lpad('  ' + vid, w0) + ''.join(rpad(c, w) for c in cells))
    return '\n'.join(out) + '\n'


# ======================================================================== 包装 video_rescore.py
_HIDE = re.compile(r'^(下一步：|INFO: Created TensorFlow Lite)')    # video_rescore.py 自己的下一步提示（这里另给）、TFLite 的日志


def run_tool(args):
    """跑 video_rescore.py，输出转出来。子进程按 UTF-8 写、这边解码：Windows 控制台不是 UTF-8 时也不会因 ✓ ✗ 报错。"""
    env = dict(os.environ, PYTHONIOENCODING='utf-8', PYTHONUNBUFFERED='1')
    p = subprocess.Popen([sys.executable, os.path.join(TOOLS, 'video_rescore.py')] + args, cwd=ROOT, env=env,
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    lines = []
    try:
        for raw in p.stdout:
            line = raw.decode('utf-8', 'replace').rstrip('\r\n')
            if not _HIDE.match(line):
                print(line, flush=True)
            lines.append(line)
    except KeyboardInterrupt:
        p.kill()
        raise
    return p.wait(), lines


def _expected(a):
    return [x.strip() for x in a.only.split(',') if x.strip()] if a.only else list(FILMS)


def _films_in(runs):
    """bench/<runs> 里有 run 的片子（文件名以片号开头，同 fusion_lab._pick_run）。"""
    out = set()
    for p in glob.glob(os.path.join(ROOT, 'bench', runs, '*.csv')):
        m = re.match(r'(v\d+)([_.\-]|\.csv$)', os.path.basename(p))
        if m:
            out.add(m.group(1))
    return out


def _common_args(a, runs):
    """→ 传给 video_rescore.py 的参数；--only 里的片子这个目录一部都没有时返回 None（不用跑）。"""
    args = ['--gt', GT_DIR, '--runs', os.path.join(ROOT, 'bench', runs), '--videos', a.videos, '--jobs', str(a.jobs)]
    for p in a.pair or []:
        args += ['--pair', p]
    if a.only:
        mine = [v for v in _expected(a) if v in _films_in(runs)]
        if not mine:
            return None
        args += ['--only', ','.join(mine)]
    return args


_CHECK_HEAD = re.compile(r'^\s{2}(v\d+)\s{2}\S.*GOP')
_CHECK_GRADE = re.compile(r'MAE\s+[\d.]+.*?(✓ 复现|~ 接近|✗ 对不上)')
_CHECK_MISS = re.compile(r'^\s+-\s+(v\d+): 找不到视频')
_CHECK_NORES = re.compile(r'^\s+(v\d+): run 里没有可用的视频结果')
_SIZE = re.compile(r'⚠\s*(v\d+): .*大小')
_EDGE = '最佳时间差落在扫描范围的边上'


def parse_check(lines):
    """video_rescore.py check 的输出 → ({片: 结论}, {片: [另外的问题]})。结论取这部片第一条管线（= run 自己的管线）。"""
    res, flags, cur = {}, {}, None
    for ln in lines:
        m = _CHECK_HEAD.match(ln)
        if m:
            cur = m.group(1)
            continue
        if _EDGE in ln and cur:
            flags.setdefault(cur, []).append('时间差扫到了范围边上')
            continue
        m = _CHECK_MISS.match(ln)
        if m:
            res[m.group(1)] = '找不到视频'
            continue
        m = _CHECK_NORES.match(ln)
        if m:
            res[m.group(1)] = '没有视频结果'
            continue
        m = _SIZE.search(ln)
        if m:
            flags.setdefault(m.group(1), []).append('原片大小与真值记录的不同（不是录 run 时那个文件？）')
            continue
        m = _CHECK_GRADE.search(ln)
        if m and cur and cur not in res:
            res[cur] = m.group(1)
    return res, flags


def cmd_check(a):
    res, flags, failed = {}, {}, []
    for runs in RUNS:
        args = _common_args(a, runs)
        if args is None:
            continue
        print('══ %s ══' % runs, flush=True)
        rc, lines = run_tool(['check'] + args)
        if rc:
            failed.append(runs)
        r, fl = parse_check(lines)
        res.update(r)
        for v, x in fl.items():
            flags.setdefault(v, []).extend(x)
    want = _expected(a)
    good = [v for v in want if res.get(v) == '✓ 复现' and v not in flags]
    print('\n══ 核对结果 ══')
    print('  ✓ 复现（%d/%d）：%s' % (len(good), len(want), ' '.join(good) or '无'))
    groups = {}
    for v in want:
        if v in good:
            continue
        k = '，'.join([res.get(v, '没核对到（缺 run，或者上面出错了）')] + flags.get(v, []))
        groups.setdefault(k, []).append(v)
    for k, vs in sorted(groups.items()):
        print('  %s：%s' % (k, ' '.join(vs)))
    for runs in failed:
        print('  video_rescore.py check 在 %s 上出错退出，看上面的报错' % runs)
    if len(good) == len(want) and not failed:
        print('结论：离线能复现手机，可以 rescore：python3 bench/tools/baseline.py rescore --videos %s' % a.videos)
        return 0
    print('结论：先别 rescore，把上面的输出发出来看看。'
          '（原片放错文件夹就换 --videos，或用 --pair v01=<视频路径> 单独指定）')
    return 1


_RESCORE_LINE = re.compile(r'^\s+(v\d+) → .*?(\d+) 个视频结果（(\d+)/(\d+) 个 tick 换了概率）')


def parse_rescore(lines):
    """video_rescore.py rescore 的输出 → {片: (换了概率的 tick, 有视频结果的 tick)}。"""
    out = {}
    for ln in lines:
        m = _RESCORE_LINE.match(ln)
        if m:
            out[m.group(1)] = (int(m.group(3)), int(m.group(4)))
    return out


def rescore_plan(a):
    """→ (名字, 管线, 模型路径, 模型指纹)；参数不对时直接退出。"""
    V.parse_pipe(a.pipe)
    if a.model and not a.name:
        sys.exit('用 --model 换了模型时要给 --name（输出到 bench/rescored/video_<名字>_runs_initial / _runs_test），'
                 '免得覆盖 %s' % DEFAULT_PIPE)
    name = a.name or a.pipe
    if name == PHONE or not NAME_RE.match(name):
        sys.exit('--name 只能用字母、数字、_ . -，也不能叫 %s：%r' % (PHONE, name))
    model = os.path.abspath(a.model) if a.model else V.DEFAULT_MODEL
    if not os.path.exists(model):
        sys.exit('找不到模型文件：%s' % model)
    return name, a.pipe, model, model_info(model)


def guard(dirs, pipe, info, overwrite):
    """同一个名字只能装同一个模型 + 管线的结果。→ 冲突说明（没有冲突返回 None）。"""
    for d in dirs:
        old = read_manifest(d)
        if not old or overwrite:
            continue
        o = old.get('model') or {}
        if o.get('sha1') != info['sha1'] or old.get('pipe') != pipe:
            return ('%s 里已经是 %s（sha1 %s，pipe=%s）算的结果，这次是 %s（sha1 %s，pipe=%s）。\n'
                    '换一个 --name；确定要覆盖（原来的结果作废）就加 --overwrite'
                    % (rel(d), o.get('file'), o.get('sha1'), old.get('pipe'), info['file'], info['sha1'], pipe))
    return None


def cmd_rescore(a):
    name, pipe, model, info = rescore_plan(a)
    dirs = variant_dirs(name)
    msg = guard(dirs, pipe, info, a.overwrite)
    if msg:
        sys.exit(msg)
    print('变体 %s：管线 %s，模型 %s（sha1 %s）' % (name, pipe, rel(model), info['sha1']), flush=True)
    done, failed = {}, []
    for runs, d in zip(RUNS, dirs):
        args = _common_args(a, runs)
        if args is None:
            continue
        print('\n══ %s → %s ══' % (runs, rel(d)), flush=True)
        rc, lines = run_tool(['rescore'] + args + ['--pipe', pipe, '--model', model,
                                                   '--out', os.path.dirname(d), '--tag', os.path.basename(d)])
        if rc:
            failed.append(runs)
            continue
        got = parse_rescore(lines)
        done.update(got)
        if got and os.path.isdir(d):
            write_manifest(d, dict(name=name, pipe=pipe, model=dict(info, path=rel(model)),
                                   created=time.strftime('%Y-%m-%d %H:%M'), tool=tool_version()))
    want = _expected(a)
    print('\n══ 重算结果：%s ══' % name)
    ok = [v for v in want if v in done]
    print('  完成（%d/%d）：%s' % (len(ok), len(want), ' '.join(ok) or '无'))
    low = ['%s %d/%d' % (v, done[v][0], done[v][1]) for v in ok if done[v][1] and done[v][0] < MIN_HIT * done[v][1]]
    if low:
        print('  ⚠ 换了概率的 tick 偏少（正常只差片头几个）：%s' % '，'.join(low))
    miss = [v for v in want if v not in done]
    if miss or failed:
        print('  没做出来：%s%s' % (' '.join(miss), '（%s 上出错退出，看上面的报错）' % '、'.join(failed) if failed else ''))
        return 1
    print('下一步：python3 bench/tools/baseline.py eval %s' % name)
    if name != DEFAULT_PIPE and os.path.isdir(variant_dirs(DEFAULT_PIPE)[0]):
        print('        python3 bench/tools/baseline.py compare %s %s' % (DEFAULT_PIPE, name))
    return 0


# ======================================================================== 命令
def _result(name):
    """compare 用：变体的数据还在就现算（保证几个变体用的是同一版规则）；目录删了就读 eval 存下的 JSON。"""
    if all(os.path.isdir(d) for d in variant_dirs(name)):
        return evaluate(name)
    p = os.path.join(OUT_DIR, name + '.json')
    if os.path.exists(p):
        with io.open(p, encoding='utf-8') as fh:
            r = json.load(fh)
        r['warnings'] = r.get('warnings', []) + ['数据目录不在了，用的是 %s 里存的结果（%s，工具 %s）'
                                                 % (rel(p), r.get('created'), r.get('tool'))]
        return r
    return evaluate(name)       # 两样都没有：按 evaluate 的提示退出


def cmd_eval(a):
    r = evaluate(a.name)
    text = report(r)
    print(text, end='')
    base = os.path.join(OUT_DIR, a.name)
    _write(base + '.txt', text)
    _write(base + '.json', json.dumps(r, ensure_ascii=False, indent=1, sort_keys=True) + '\n')
    print('报告：%s（数字另存 %s）' % (rel(base + '.txt'), rel(base + '.json')))
    return 0


def cmd_compare(a):
    if len(a.names) < 2:
        sys.exit('至少给两个变体，如：compare phone exact_crop')
    text = compare_text([_result(n) for n in a.names])
    print(text, end='')
    path = os.path.join(OUT_DIR, 'compare_%s.txt' % '_vs_'.join(a.names))
    _write(path, text)
    print('对比表：%s' % rel(path))
    return 0


# ======================================================================== 自检
def selftest():
    fails = []

    def check(ok, msg):
        print('  %s %s' % ('ok  ' if ok else 'FAIL', msg))
        if not ok:
            fails.append(msg)

    # 1. phone 与 README 第 6 节（cue_lab.py eval --preset a1bvid，13 部）一致，视频 AUC 与 metrics.py --diag 一致
    r = evaluate(PHONE)
    e = r['e2e']
    check((round(e['cost']), pct(e['acc']), pct(e['fp']), pct(e['fn']), e['stops'], num(e['hard_p90'], '%.1f'),
           e['seek_never']) == (1164, '94.2%', '13.9%', '2.1%', 7, '7.2', 0),
          'phone 的端到端与 README 一致（代价 1164、94.2% / 13.9% / 2.1%、误停 7、特写 seek p90 7.2s、没启动 0）')
    pf = r['per_film']
    check([num(pf[v]['v_auc'], '%.3f') for v in ('v01', 'v02', 'v07', 'v09')] == ['0.871', '—', '0.467', '1.000'],
          'phone 的逐片视频 AUC 与 metrics.py --diag 一致（v01 0.871、v02 没有剧情段、v07 0.467、v09 1.000）')
    check(pct(pf['v01']['v_acc']) == '47.2%' and pct(pf['v07']['v_fp']) == '53.5%',
          'phone 的视频单独准确率与 metrics.py --diag 一致（v01 47.2%、v07 剧情判成转 53.5%）')
    check(len(r['films']) == 13 and not r['warnings'], '13 部齐全、音频线索全部接上、没有警告')
    pooled = M._auc([1.0, 0.8, 0.6], [0.7, 0.2]) == (5 / 6.0)
    check(pooled and r['video']['n'] == [sum(x['v_n'][0] for x in pf.values()), sum(x['v_n'][1] for x in pf.values())],
          '13 部合并 AUC 用的是全部片子的 tick')

    # 2. 重算格式：概率不动、按 video_rescore.py rescore 的格式写一遍，结果应与 phone 逐项相同
    keep = globals()['RESCORED']
    tmp = tempfile.mkdtemp(prefix='baseline_selftest_')
    globals()['RESCORED'] = tmp
    try:
        info = dict(file='selftest.tflite', sha1='0123456789ab', size=1)
        for runs, d in zip(RUNS, variant_dirs('selftest')):
            for src in sorted(glob.glob(os.path.join(ROOT, 'bench', runs, '*.csv'))):
                meta, results = V.load_results(src)
                comp = dict((x.ts, (x.probs, [(x.pos - V.NEW_APP_DELTA - V.FRAME_STEP_MS * (11 - i)) * 1000
                                              for i in range(V.NUM_FRAMES)])) for x in results)
                V.write_rescored(src, os.path.join(d, os.path.basename(src)), comp,
                                 'video_rescore.py rescore: pipe=exact_crop matrix=auto delta=200ms model=selftest.tflite',
                                 F.parse_params(meta.get('params', '')))
            write_manifest(d, dict(name='selftest', pipe='exact_crop', model=info))
        r2 = evaluate('selftest')
        same = all(str(r['e2e'][k]) == str(r2['e2e'][k]) for k in r['e2e'] if k != 'cost') \
            and abs(r['e2e']['cost'] - r2['e2e']['cost']) < 1e-6 \
            and json.dumps(r['video'], sort_keys=True) == json.dumps(r2['video'], sort_keys=True) \
            and all(abs(r['per_film'][v]['cost'] - r2['per_film'][v]['cost']) < 1e-6 for v in FILMS)
        check(same, '重算格式的 run（概率不变）与 phone 的数字逐项相同（目录、文件头、音频线索都接对了）')
        check('pipe=exact_crop' in r2['source'] and 'sha1 0123456789ab' in r2['source'] and not r2['warnings'],
              '读出重算的出处（管线、模型指纹）')
        txt = compare_text([r, r2])
        deltas = re.findall(r'\(([+-][\d.]+)\)', txt)
        check(len(deltas) >= len(_ROWS) and all(float(x) == 0 for x in deltas), 'compare：同样的数据，变化全是 0')
        d0 = variant_dirs('selftest')[0]
        other = dict(info, sha1='ffffffffffff')
        check(guard([d0], 'exact_crop', info, False) is None and guard([d0], 'exact_crop', other, False)
              and guard([d0], 'exact_squash', info, False) and guard([d0], 'exact_crop', other, True) is None,
              '模型指纹：同一个名字换了模型或管线会被拒绝，--overwrite 才放行')
        os.remove(os.path.join(variant_dirs('selftest')[1], 'v08_base.csv'))
        r3 = evaluate('selftest')
        check(any('缺这几部' in w and 'v08' in w for w in r3['warnings']), '少了片子会警告')
    finally:
        globals()['RESCORED'] = keep
        shutil.rmtree(tmp, ignore_errors=True)

    # 3. 包装 video_rescore.py：输出解析与命令行
    sample = ['  ⚠ v09: v09.mp4 的大小 1 与真值记录的 2 不同 —— 是不是同一个文件？',
              '  - v10: 找不到视频（file_name=视频10.mp4，run 里 video_name=x）。放进 --videos 目录，或用 --pair v10=<视频路径>',
              '  v08  v08.mp4  GOP 中位 12.0s / 最长 12.0s，1899 个视频结果，run 的视频管线 keyframe_legacy（没有 v_fpts：扫 Δ）',
              '    99 帧 × legacy 预处理 + 推理 1 秒（只解关键帧）',
              '    keyframe_legacy  Δ=875ms   MAE 0.0071  判定一致  98.4%  差≤0.05  94.6%   ✓ 复现   12 帧同一张图的窗口 1477 个：MAE 0.0039',
              '    exact_legacy     Δ=200ms   MAE 0.0900  判定一致  80.0%  差≤0.05  50.0%   ✗ 对不上',
              '  v09  v09.mp4  GOP 中位 12.0s / 最长 12.0s，1728 个视频结果，run 的视频管线 keyframe_legacy（没有 v_fpts：扫 Δ）',
              '    keyframe_legacy  Δ=1300ms  MAE 0.0500  判定一致  90.0%  差≤0.05  70.0%   ~ 接近']
    res, fl = parse_check(sample + ['    ⚠ keyframe_legacy 的最佳时间差落在扫描范围的边上，真实值可能更大，结果不可信'])
    check(res == {'v08': '✓ 复现', 'v09': '~ 接近', 'v10': '找不到视频'} and sorted(fl) == ['v09']
          and len(fl['v09']) == 2, 'check 的输出解析（每部片取 run 自己那条管线、找不到视频、文件大小不对、Δ 扫到边上）')
    got = parse_rescore(['  v08 → bench/rescored/video_exact_crop_runs_test/v08_base.csv：1899 个视频结果'
                         '（2355/2360 个 tick 换了概率），4700 帧，400 次推理'])
    check(got == {'v08': (2355, 2360)}, 'rescore 的输出解析')
    ns = argparse.Namespace(videos='D:/v', jobs=2, pair=['v01=D:/a.mp4'], only=None, pipe='exact_crop', model=None,
                            name=None, overwrite=False)
    args = _common_args(ns, 'runs_test')
    ns.only = 'v03,v09'
    a1, a2 = _common_args(ns, 'runs_initial'), _common_args(ns, 'runs_test')
    ns.only = 'v03'
    a3 = _common_args(ns, 'runs_test')
    ns.only = None
    check(args[args.index('--runs') + 1].endswith(os.path.join('bench', 'runs_test')) and '--pair' in args
          and args[args.index('--jobs') + 1] == '2' and '--only' not in args
          and a1[a1.index('--only') + 1] == 'v03' and a2[a2.index('--only') + 1] == 'v09' and a3 is None
          and _films_in('runs_initial') == set(FILMS[:7]) and _films_in('runs_test') == set(FILMS[7:]),
          'check / rescore 传给 video_rescore.py 的参数（--only 按片子所在的 runs 目录拆开，没有的目录不跑）')
    name, pipe, model, mi = rescore_plan(ns)
    check(name == 'exact_crop' and model == V.DEFAULT_MODEL and len(mi['sha1']) == 12
          and [os.path.basename(x) for x in variant_dirs(name)] == ['video_exact_crop_runs_initial',
                                                                   'video_exact_crop_runs_test'],
          '默认变体 exact_crop = video_rescore.py rescore 的默认输出目录、app 里的模型')
    ns.model = V.DEFAULT_MODEL
    try:
        rescore_plan(ns)
        refused = False
    except SystemExit:
        refused = True
    check(refused, '给了 --model 没给 --name 会被拒绝（不让新模型写进 exact_crop）')
    print('selftest: %s' % ('ALL PASS' if not fails else '%d FAIL' % len(fails)))
    return 0 if not fails else 1


def main():
    for s in (sys.stdout, sys.stderr):          # 输出被重定向到非 UTF-8 的地方时，✓ ✗ ⚠ 换成 ? 而不是报错
        try:
            s.reconfigure(errors='replace')
        except Exception:
            pass
    if '--selftest' in sys.argv:
        return selftest()
    ap = argparse.ArgumentParser(description='统一评测基准：只换视频结果，其它固定，在同一套录像上比（详见文件头）',
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd')
    p = sub.add_parser('eval', help='一个变体在统一标准下的报告')
    p.add_argument('name', help='phone / exact_crop / rescore 时起的名字')
    p = sub.add_parser('compare', help='几个变体并排对比（第一个是参照）')
    p.add_argument('names', nargs='+')
    for cmd, hlp in (('check', '核对原片：离线能复现手机（13 部）'), ('rescore', '离线重算视频结果（13 部）')):
        p = sub.add_parser(cmd, help=hlp)
        p.add_argument('--videos', default=V.DEFAULT_VIDEOS, help='原片所在目录（默认 bench/videos）')
        p.add_argument('--pair', action='append', help='逐个指定原片：v01=D:/xcup/视频1.mp4（可多次）')
        p.add_argument('--only', help='只做这几部，逗号分隔：v01,v04')
        p.add_argument('--jobs', type=int, default=1, help='几部片并行（每个进程各占 4 个线程）')
        if cmd == 'rescore':
            p.add_argument('--model', help='视频模型（默认 app 里的）；给了就要给 --name')
            p.add_argument('--name', help='变体名（默认 = 管线名，即 exact_crop）')
            p.add_argument('--pipe', default=DEFAULT_PIPE, help='管线（默认 exact_crop = 修复后的 app）')
            p.add_argument('--overwrite', action='store_true', help='允许用不同的模型 / 管线覆盖同名变体')
    a = ap.parse_args()
    fn = {'eval': cmd_eval, 'compare': cmd_compare, 'check': cmd_check, 'rescore': cmd_rescore}.get(a.cmd)
    if fn is None:
        ap.print_help()
        return 2
    return fn(a)


if __name__ == '__main__':
    sys.exit(main())
