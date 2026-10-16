#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
training/data_check.py — 视频训练素材检查（训练前跑一次）

素材是 make_movinet_clips.py 切出来的 movinet_clips/<split>/<类别>/<源视频名>_s<起点>_<类别>.mp4。

子命令
  scan   逐个打开片段、顺序解码全部帧。打不开、读不出帧、帧数不足的记进 bad_clips.txt（train_video.py 训练时剔除）；
         另写 clip_index.csv：每个片段的帧数、尺寸、帧率、亮度和 3 个画面指纹（leak 用）。默认都写在素材目录根下。
         几乎全黑的片段只报出来、不剔除（剧情里的黑场是正常素材）
  stats  按 split / 类别 / 源视频统计片段数；同一个源视频名出现在两个 split 里会报警（train/val/test 要按片子分）
  leak   查 bench 的 13 部测试片（v01~v13）有没有混进训练素材：测试片每 0.25 秒取一帧算画面指纹，和 clip_index.csv
         里每个片段的指纹比，同一个源视频有好几个片段对得上同一部测试片就报"疑似混入"。要先 scan

用法（训练机上，仓库根目录；Windows 把 python3 换成 python）
  python3 training/data_check.py scan  --data_dir ~/ml/movinet/movinet_clips --jobs 8
  python3 training/data_check.py stats --data_dir ~/ml/movinet/movinet_clips
  python3 training/data_check.py leak  --data_dir ~/ml/movinet/movinet_clips --videos <13 部原片所在目录>
  python3 training/data_check.py --selftest

leak 按 bench/gt/vNN.csv 文件头的 file_name 在 --videos 目录里找原片（与 baseline.py 相同），对不上时用
--pair v07=/path/视频7.mp4 逐个指定。

依赖：numpy、opencv-python（训练环境已有）；leak 和自检还要 ffmpeg 在 PATH 上。
"""
import argparse
import csv
import io
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import unicodedata
from collections import Counter, OrderedDict, defaultdict

try:
    import numpy as np
except ImportError:          # 只有 --help 能在没有 numpy 时跑
    np = None
try:
    import cv2
except ImportError:
    cv2 = None

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DEFAULT_GT = os.path.join(ROOT, 'bench', 'gt')

CLASS_NAMES = ('normal_plot', 'oral', 'sex')                          # 视频模型的类别顺序
ALL_CLASSES = ('normal_plot', 'foreplay', 'oral', 'sex', 'unclear')   # make_movinet_clips.py 的 CLASS_MAP
SPLITS = ('train', 'val', 'test')
VIDEO_EXTS = ('.mp4', '.mov', '.avi', '.mkv', '.webm', '.m4v')
# <源视频名>_s<起点秒>_<类别>：源视频名本身可能带下划线，贪婪匹配到最后一个 _s<数字>_<类别>
CLIP_RE = re.compile(r'^(?P<film>.+)_s(?P<start>\d+(?:\.\d+)?)_(?P<cls>%s)$' % '|'.join(ALL_CLASSES))

CLIP_FPS = 12            # make_movinet_clips.py：-r 12，3 秒 = 36 帧
MIN_FRAMES = 30          # 少于 2.5 秒的片段算帧数不足
DARK_MEAN = 12.0         # 每一帧的平均灰度都低于它 = 几乎全黑
HASH_FRAMES = (3, 18, 33)    # 指纹取第 3/18/33 帧（片段内 0.25/1.5/2.75 秒，起点是 1.5 秒的倍数，正好落在 0.25 秒的网格上）
HASH_STD_MIN = 6.0       # 32×32 灰度图的标准差低于它（纯色、黑场、字幕卡）不算指纹，免得到处对上
BENCH_FPS = 4            # 测试片每 0.25 秒取一帧
LEAK_DIST = 6            # 64 位指纹的汉明距离 ≤ 6 算同一画面（重新编码、缩放后一般 0~4；不同画面平均 32）
LEAK_MIN_HITS = 2        # 一个片段的 3 个指纹里至少 2 个对上，才算这个片段对上了
LEAK_MIN_CLIPS = 3       # 同一个源视频至少 3 个片段对上同一部测试片，才报"疑似混入"
BAD_LIST = 'bad_clips.txt'
INDEX = 'clip_index.csv'
INDEX_COLS = ('path', 'split', 'cls', 'film', 'start', 'ok', 'reason', 'frames', 'width', 'height', 'fps',
              'mean_gray', 'max_gray', 'hash1', 'hash2', 'hash3')


def width(s):
    """终端里的显示宽度（中文占两格）。"""
    return sum(2 if unicodedata.east_asian_width(c) in 'WF' else 1 for c in s)


def lpad(s, n):
    return s + ' ' * max(0, n - width(s))


def rpad(s, n):
    return ' ' * max(0, n - width(s)) + s


def _need(cv=True):
    if np is None:
        sys.exit('需要 numpy：pip install numpy opencv-python')
    if cv and cv2 is None:
        sys.exit('需要 OpenCV：pip install opencv-python')


# ======================================================================== 文件名与列表
def parse_clip_name(path):
    """片段文件名 → (源视频名, 起点秒, 类别)。命名不合规时整个文件名当源视频名，起点、类别为 None。"""
    stem = os.path.splitext(os.path.basename(path))[0]
    m = CLIP_RE.match(stem)
    if not m:
        return stem, None, None
    return m.group('film'), float(m.group('start')), m.group('cls')


def film_of(path):
    return parse_clip_name(path)[0]


def rel_key(path, data_dir):
    """bad_clips.txt / clip_index.csv 里的路径：相对素材目录、用 / 分隔（换机器、换系统也对得上）。"""
    return os.path.relpath(os.path.abspath(path), os.path.abspath(data_dir)).replace('\\', '/')


def list_clips(data_dir, splits=SPLITS, classes=CLASS_NAMES):
    """→ [(相对路径, split, 类别)]，按路径排序。缺的 split / 类别目录跳过（stats 会报出来）。"""
    out = []
    for sp in splits:
        for c in classes:
            d = os.path.join(data_dir, sp, c)
            if not os.path.isdir(d):
                continue
            for f in sorted(os.listdir(d)):
                if os.path.splitext(f)[1].lower() in VIDEO_EXTS:
                    out.append(('%s/%s/%s' % (sp, c, f), sp, c))
    return out


def read_bad_list(path):
    """bad_clips.txt → {相对路径: 原因}。# 开头的行是注释。"""
    out = OrderedDict()
    with io.open(path, encoding='utf-8') as fh:
        for line in fh:
            line = line.rstrip('\r\n')
            if not line.strip() or line.lstrip().startswith('#'):
                continue
            p, _, why = line.partition('\t')
            out[p.strip().replace('\\', '/')] = why.strip()
    return out


def read_index(path):
    with io.open(path, encoding='utf-8', newline='') as fh:
        return list(csv.DictReader(fh))


# ======================================================================== 画面指纹
def phash32(gray32):
    """32×32 灰度图 → 64 位感知哈希（DCT 左上 8×8，按除直流外的中位数二值化）。亮度整体偏移、缩放不影响。"""
    d = cv2.dct(np.asarray(gray32, np.float32))
    low = d[:8, :8].reshape(-1)
    bits = (low > np.median(low[1:])).astype(np.uint8)
    return int(np.packbits(bits).view('>u8')[0])


def frame_hash(bgr):
    """一帧（BGR，已是居中裁好的正方形）→ 指纹；画面太平返回 None。"""
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY) if bgr.ndim == 3 else bgr
    small = cv2.resize(g, (32, 32), interpolation=cv2.INTER_AREA)
    if float(small.std()) < HASH_STD_MIN:
        return None
    return phash32(small)


_POP8 = None


def popcount64(x):
    """uint64 数组逐元素数 1 的个数。"""
    if hasattr(np, 'bitwise_count'):
        return np.bitwise_count(x)
    global _POP8
    if _POP8 is None:
        _POP8 = np.array([bin(i).count('1') for i in range(256)], np.uint8)
    b = np.ascontiguousarray(x).view(np.uint8).reshape(x.shape + (8,))
    return _POP8[b].sum(axis=-1)


def min_dist(query, ref, chunk=2048):
    """query[n]、ref[m]（uint64）→ 每个 query 到 ref 的最小汉明距离和对应下标。"""
    best = np.full(len(query), 99, np.int32)
    arg = np.zeros(len(query), np.int64)
    if len(ref) == 0 or len(query) == 0:
        return best, arg
    for i in range(0, len(query), chunk):
        d = popcount64(query[i:i + chunk, None] ^ ref[None, :])
        best[i:i + chunk] = d.min(axis=1)
        arg[i:i + chunk] = d.argmin(axis=1)
    return best, arg


# ======================================================================== scan
def probe_clip(path, min_frames=MIN_FRAMES):
    """顺序解码一个片段的全部帧（与 train_utils.read_clip 相同的读法）→ 一行 clip_index。"""
    r = dict(ok=0, reason='', frames=0, width=0, height=0, fps=0.0, mean_gray=0.0, max_gray=0.0,
             hash1='', hash2='', hash3='')
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        r['reason'] = '打不开'
        return r
    r['fps'] = round(float(cap.get(cv2.CAP_PROP_FPS) or 0.0), 3)
    n_meta = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    grays, keep = [], {}
    while True:
        ok, f = cap.read()
        if not ok or f is None:
            break
        if not grays:
            r['height'], r['width'] = int(f.shape[0]), int(f.shape[1])
        g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        grays.append(float(g.mean()))
        if len(grays) - 1 in HASH_FRAMES:
            keep[len(grays) - 1] = f
    cap.release()
    n = len(grays)
    r['frames'] = n
    if n == 0:
        r['reason'] = '读不出帧'
        return r
    r['mean_gray'] = round(sum(grays) / n, 2)
    r['max_gray'] = round(max(grays), 2)
    if n < min_frames:
        r['reason'] = '帧数不足（%d < %d）' % (n, min_frames)
        if n_meta > n:
            r['reason'] += '，文件头说有 %d 帧（解码提前结束）' % n_meta
        return r
    r['ok'] = 1
    # 指纹：帧数够 36 用固定的 3 帧，不够时按比例取（再读一遍，只有少数片段会走到这里）
    want = list(HASH_FRAMES) if n > HASH_FRAMES[-1] else [int(round(x * (n - 1))) for x in (0.08, 0.5, 0.92)]
    if any(i not in keep for i in want):
        keep = {}
        cap = cv2.VideoCapture(path)
        i = 0
        while i <= max(want):
            ok, f = cap.read()
            if not ok:
                break
            if i in want:
                keep[i] = f
            i += 1
        cap.release()
    for k, i in enumerate(want):
        h = frame_hash(keep[i]) if i in keep else None
        r['hash%d' % (k + 1)] = '' if h is None else '%016x' % h
    return r


def _probe_job(job):
    data_dir, rel, split, cls, min_frames = job
    if cv2 is not None:
        cv2.setNumThreads(1)
    film, start, _ = parse_clip_name(rel)
    r = dict(path=rel, split=split, cls=cls, film=film, start='' if start is None else start)
    try:
        r.update(probe_clip(os.path.join(data_dir, rel), min_frames))
    except Exception as e:             # 解码器崩了也只算这一个片段坏
        r.update(ok=0, reason='读的时候出错：%s' % e)
    return r


def _map(fn, jobs, n, every=2000, label='片段'):
    t0 = time.time()
    out = []
    if n > 1 and len(jobs) > 1:
        import multiprocessing as mp
        with mp.Pool(n) as pool:
            for i, r in enumerate(pool.imap(fn, jobs, chunksize=16)):
                out.append(r)
                if (i + 1) % every == 0:
                    print('  … %d/%d 个%s（%.0f 秒）' % (i + 1, len(jobs), label, time.time() - t0), flush=True)
    else:
        for i, j in enumerate(jobs):
            out.append(fn(j))
            if (i + 1) % every == 0:
                print('  … %d/%d 个%s（%.0f 秒）' % (i + 1, len(jobs), label, time.time() - t0), flush=True)
    return out


def scan(data_dir, jobs=8, min_frames=MIN_FRAMES, out_dir=None, classes=CLASS_NAMES, quiet=False):
    """扫全部片段，写 clip_index.csv 和 bad_clips.txt（out_dir 默认 = 素材目录）。→ 每个片段一行 dict。"""
    _need()
    say = (lambda *a, **k: None) if quiet else print
    data_dir = os.path.abspath(data_dir)
    out_dir = out_dir or data_dir
    clips = list_clips(data_dir, classes=classes)
    if not clips:
        sys.exit('%s 下没找到片段（应为 <split>/<类别>/*.mp4，类别 %s）' % (data_dir, '/'.join(classes)))
    say('扫描 %d 个片段，%d 进程 …' % (len(clips), jobs), flush=True)
    t0 = time.time()
    rows = _map(_probe_job, [(data_dir, rel, sp, c, min_frames) for rel, sp, c in clips], jobs)
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    with io.open(os.path.join(out_dir, INDEX), 'w', encoding='utf-8', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=INDEX_COLS, lineterminator='\n')
        w.writeheader()
        for r in rows:
            w.writerow(dict((k, r.get(k, '')) for k in INDEX_COLS))
    bad = [r for r in rows if not int(r['ok'])]
    with io.open(os.path.join(out_dir, BAD_LIST), 'w', encoding='utf-8', newline='\n') as fh:
        fh.write('# data_check.py scan %s：%d 个片段里 %d 个坏的（训练时剔除）。路径相对素材目录\n'
                 % (time.strftime('%Y-%m-%d %H:%M'), len(rows), len(bad)))
        for r in bad:
            fh.write('%s\t%s\n' % (r['path'], r['reason']))
    if not quiet:
        report_scan(rows, min_frames)
        say('用时 %.0f 秒。已写 %s 和 %s' % (time.time() - t0, os.path.join(out_dir, INDEX),
                                         os.path.join(out_dir, BAD_LIST)))
    return rows


def report_scan(rows, min_frames=MIN_FRAMES):
    print('\n══ 扫描结果（坏片段训练时剔除；几乎全黑的只报不剔）══')
    print('  ' + lpad('split', 7) + lpad('类别', 12) + ''.join(rpad(h, 8) for h in ('片段', '坏的', '全黑', '非224')))
    for sp in SPLITS:
        for c in sorted(set(r['cls'] for r in rows), key=lambda x: ALL_CLASSES.index(x)):
            rs = [r for r in rows if r['split'] == sp and r['cls'] == c]
            if not rs:
                continue
            good = [r for r in rs if int(r['ok'])]
            dark = sum(1 for r in good if float(r['max_gray']) < DARK_MEAN)
            odd = sum(1 for r in good if (int(r['width']), int(r['height'])) != (224, 224))
            print('  ' + lpad(sp, 7) + lpad(c, 12) + ''.join(rpad(str(x), 8) for x in (len(rs), len(rs) - len(good),
                                                                                       dark, odd)))
    bad = [r for r in rows if not int(r['ok'])]
    if bad:
        why = Counter(re.sub(r'（.*', '', r['reason']) for r in bad)
        print('  坏片段原因：' + '，'.join('%s %d' % kv for kv in why.most_common()))
        for r in bad[:10]:
            print('    %s  %s' % (r['path'], r['reason']))
        if len(bad) > 10:
            print('    … 其余 %d 个见 bad_clips.txt' % (len(bad) - 10))
    nf = Counter(int(r['frames']) for r in rows if int(r['ok']))
    if nf:
        common = ', '.join('%d 帧 %d 个' % kv for kv in nf.most_common(4))
        print('  好片段的帧数：%s（12fps 的 3 秒片段应为 36 帧；少于 %d 帧算坏）' % (common, min_frames))
    fps = Counter(round(float(r['fps'])) for r in rows if int(r['ok']))
    if fps and set(fps) != {CLIP_FPS}:
        print('  ⚠ 帧率不全是 %d：%s' % (CLIP_FPS, dict(fps)))
    nohash = sum(1 for r in rows if int(r['ok']) and not (r['hash1'] or r['hash2'] or r['hash3']))
    if nohash:
        print('  %d 个片段画面太平（纯色 / 黑场），没有画面指纹，leak 查不到它们' % nohash)


# ======================================================================== stats
def stats(data_dir, quiet=False):
    """→ {'films': {(split, 类别): Counter(源视频: 片段数)}, 'overlap': [(源视频, [split...])], 'bad': n}。"""
    data_dir = os.path.abspath(data_dir)
    idx_path = os.path.join(data_dir, INDEX)
    if os.path.exists(idx_path):
        rows = [r for r in read_index(idx_path) if int(r['ok'])]
        src = '%s（坏片段已去掉）' % INDEX
        n_bad = sum(1 for r in read_index(idx_path) if not int(r['ok']))
    else:
        rows = [dict(path=rel, split=sp, cls=c, film=film_of(rel)) for rel, sp, c in list_clips(data_dir)]
        src = '文件名（还没 scan，坏片段没去掉）'
        n_bad = None
    films = defaultdict(Counter)
    odd = []
    for r in rows:
        films[(r['split'], r['cls'])][r['film']] += 1
        if parse_clip_name(r['path'])[2] is None:
            odd.append(r['path'])
    say = (lambda *a: None) if quiet else print
    say('══ 素材统计：%s，来源 %s ══' % (data_dir, src))
    say('  ' + lpad('split', 7) + lpad('类别', 12) + rpad('片段', 7) + rpad('源视频', 8) +
        rpad('每部的片段数 最少/中位/最多', 30) + '  片段最多的 3 部（占比）')
    for sp in SPLITS:
        for c in CLASS_NAMES:
            cnt = films.get((sp, c))
            if not cnt:
                say('  ' + lpad(sp, 7) + lpad(c, 12) + rpad('没有', 7))
                continue
            v = sorted(cnt.values())
            tot = sum(v)
            top = '，'.join('%s %.0f%%' % (f[:24], 100.0 * k / tot) for f, k in cnt.most_common(3))
            say('  ' + lpad(sp, 7) + lpad(c, 12) + rpad(str(tot), 7) + rpad(str(len(v)), 8) +
                rpad('%d / %d / %d' % (v[0], v[len(v) // 2], v[-1]), 30) + '  ' + top)
    # 同一个源视频名出现在不同 split：按片子划分的前提被破坏（同名不一定是同一部片，但要人工看一眼）
    where = defaultdict(set)
    for (sp, c), cnt in films.items():
        for f in cnt:
            where[f].add(sp)
    overlap = sorted((f, sorted(s, key=SPLITS.index)) for f, s in where.items() if len(s) > 1)
    if overlap:
        say('  ⚠ %d 个源视频名同时出现在几个 split 里（train/val/test 应按片子划分，请人工确认是不是同一部片）：' % len(overlap))
        for f, s in overlap[:15]:
            say('    %s：%s' % (f, ' / '.join(s)))
        if len(overlap) > 15:
            say('    … 其余 %d 个' % (len(overlap) - 15))
    else:
        say('  ✓ 没有源视频名同时出现在两个 split 里')
    if odd:
        say('  ⚠ %d 个片段的文件名不是 <源视频名>_s<起点>_<类别>，按整个文件名当源视频：%s'
            % (len(odd), ', '.join(odd[:3])))
    if n_bad:
        say('  （另有 %d 个坏片段，见 %s）' % (n_bad, BAD_LIST))
    return dict(films=films, overlap=overlap, bad=n_bad)


# ======================================================================== leak
def _gt_header(path):
    meta = {}
    with io.open(path, encoding='utf-8-sig') as fh:
        for line in fh:
            if not line.startswith('#'):
                break
            for kv in line[1:].split():
                if '=' in kv:
                    k, v = kv.split('=', 1)
                    meta[k] = v
    return meta


def bench_videos(gt_dir, videos_dir, pairs=()):
    """→ [(片号, 原片路径或 None, 真值里的 file_name)]。按真值文件头的 file_name 找，--pair 优先。"""
    explicit = dict(p.split('=', 1) for p in pairs or () if '=' in p)
    out = []
    for g in sorted(f for f in os.listdir(gt_dir) if f.endswith('.csv')):
        vid = os.path.splitext(g)[0]
        fn = _gt_header(os.path.join(gt_dir, g)).get('file_name', '')
        path = explicit.get(vid)
        if not path and videos_dir and fn and os.path.exists(os.path.join(videos_dir, fn)):
            path = os.path.join(videos_dir, fn)
        if not path and videos_dir and os.path.isdir(videos_dir):
            for f in sorted(os.listdir(videos_dir)):
                stem, ext = os.path.splitext(f)
                if ext.lower() in VIDEO_EXTS and stem in (vid, os.path.splitext(fn)[0]):
                    path = os.path.join(videos_dir, f)
                    break
        out.append((vid, path if path and os.path.exists(path) else None, fn))
    return out


def video_hashes(path, fps=BENCH_FPS, ffmpeg='ffmpeg'):
    """原片每 1/fps 秒一帧，按 make_movinet_clips.py 的方式居中裁成 224 再缩到 32×32 灰度 → (时刻秒[], 指纹 uint64[])。
    画面太平的帧不要。用 ffmpeg 解码（多线程，20 分钟的 1080p 约 1 分钟）。"""
    vf = ('fps=%g,scale=224:224:force_original_aspect_ratio=increase,crop=224:224,'
          'scale=32:32:flags=area,format=gray' % fps)
    cmd = [ffmpeg, '-v', 'error', '-i', path, '-an', '-sn', '-vf', vf, '-f', 'rawvideo', '-pix_fmt', 'gray', '-']
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    times, hashes, k = [], [], 0
    while True:
        buf = p.stdout.read(1024)
        if len(buf) < 1024:
            break
        g = np.frombuffer(buf, np.uint8).reshape(32, 32)
        if float(g.std()) >= HASH_STD_MIN:
            times.append(k / float(fps))
            hashes.append(phash32(g))
        k += 1
    err = p.stderr.read().decode('utf-8', 'replace')
    if p.wait() != 0 or k == 0:
        raise RuntimeError('ffmpeg 读不了 %s：%s' % (path, err.strip()[-300:]))
    return np.array(times), np.array(hashes, np.uint64)


def _clip_hash_table(rows):
    """clip_index 的行 → (片段下标[], 指纹 uint64[])：每个有效指纹一条。"""
    ci, hs = [], []
    for i, r in enumerate(rows):
        for k in ('hash1', 'hash2', 'hash3'):
            if r.get(k):
                ci.append(i)
                hs.append(int(r[k], 16))
    return np.array(ci, np.int64), np.array(hs, np.uint64)


def leak(data_dir, videos_dir=None, gt_dir=DEFAULT_GT, pairs=(), ffmpeg='ffmpeg', quiet=False):
    """→ {片号: [(split, 类别, 源视频, 对上的片段数, 例子)]}（只含达到 LEAK_MIN_CLIPS 的）。"""
    _need()
    say = (lambda *a, **k: None) if quiet else print
    idx_path = os.path.join(os.path.abspath(data_dir), INDEX)
    if not os.path.exists(idx_path):
        sys.exit('没有 %s：先跑 python3 training/data_check.py scan --data_dir %s' % (idx_path, data_dir))
    rows = [r for r in read_index(idx_path) if int(r['ok'])]
    ci, ch = _clip_hash_table(rows)
    say('训练素材：%d 个片段、%d 个画面指纹（%s）' % (len(rows), len(ch), idx_path))
    vids = bench_videos(gt_dir, videos_dir, pairs)
    missing = [(v, fn) for v, p, fn in vids if not p]
    if missing:
        say('  ⚠ 找不到原片：%s。放进 --videos 目录，或 --pair v01=<路径>'
            % '，'.join('%s（%s）' % (v, fn or '?') for v, fn in missing))
    found, checked = {}, []
    for vid, path, fn in vids:
        if not path:
            continue
        t0 = time.time()
        times, bh = video_hashes(path, ffmpeg=ffmpeg)
        dist, arg = min_dist(ch, bh)
        hit = dist <= LEAK_DIST
        per_clip = Counter(ci[hit].tolist())
        clips = [i for i, n in per_clip.items() if n >= LEAK_MIN_HITS]
        groups = defaultdict(list)
        for i in clips:
            r = rows[i]
            j = np.where((ci == i) & hit)[0]
            groups[(r['split'], r['cls'], r['film'])].append((r['path'], float(times[arg[j[0]]])))
        ranked = sorted(groups.items(), key=lambda kv: -len(kv[1]))
        sus = [(k[0], k[1], k[2], len(v), v[:3]) for k, v in ranked if len(v) >= LEAK_MIN_CLIPS]
        if sus:
            found[vid] = sus
        checked.append(vid)
        say('  %s  %-24s %5d 帧指纹，%5.0f 秒：对上的片段 %d 个%s' % (
            vid, os.path.basename(path)[:24], len(bh), time.time() - t0, len(clips),
            '，⚠ 疑似混入' if sus else ''), flush=True)
        for (sp, c, film), v in ranked[:5]:
            say('      %s %-6s %-11s %-30s %3d 个片段  例：%s ↔ 原片 %.1fs' % (
                '⚠' if len(v) >= LEAK_MIN_CLIPS else ' ', sp, c, film[:30], len(v), v[0][0], v[0][1]))
    say('\n══ 结论 ══')
    if found:
        for vid, sus in sorted(found.items()):
            for sp, c, film, n, ex in sus:
                say('  ⚠ %s 疑似混进了训练素材：%s/%s 的源视频 %s 有 %d 个片段对上（例：%s ↔ 原片 %.1f 秒）'
                    % (vid, sp, c, film, n, ex[0][0], ex[0][1]))
        say('  核实：并排看一下例子里的片段和原片那一刻；确认是同一部片就把这个源视频的片段全部移出素材目录，重新 scan')
    elif checked:
        say('  ✓ 查了 %d 部（%s），没有发现测试片混进训练素材%s' % (
            len(checked), ' '.join(checked), '' if not missing else '；还有 %d 部没找到原片，没查' % len(missing)))
    else:
        say('  一部测试片都没找到，没查成')
    return found


# ======================================================================== 自检
def _ffmpeg(args):
    r = subprocess.run(['ffmpeg', '-v', 'error', '-y'] + args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if r.returncode != 0:
        raise RuntimeError('ffmpeg 失败：%s\n%s' % (' '.join(args), r.stderr.decode('utf-8', 'replace')[-500:]))


SYNTH_SOURCES = {        # 合成"源视频"：每种画面不同、而且一直在动（只用 FFmpeg 4.2 也有的测试源）
    'filmA': 'testsrc2=size=1280x720:rate=30',
    'filmB': 'mandelbrot=size=1280x720:rate=30',
    'filmC': 'life=size=640x360:rate=30:mold=10:ratio=0.3:death_color=#C83232:life_color=#00ff00',
    'filmD': 'mandelbrot=size=1280x720:rate=30:start_x=-0.743643887037151:start_y=0.131825904205330:start_scale=0.3',
    'filmE': 'testsrc=size=960x540:rate=30',
    'filmF': 'smptehdbars=size=1280x720:rate=30,hue=H=2*PI*t/7:s=1.5,noise=alls=40:allf=t',
}


def make_source(path, spec, seconds):
    _ffmpeg(['-f', 'lavfi', '-i', spec, '-t', '%g' % seconds, '-c:v', 'libx264', '-preset', 'ultrafast',
             '-crf', '20', '-pix_fmt', 'yuv420p', path])


def cut_clip(src, out, start, seconds=3.0):
    """与 make_movinet_clips.py 的 cut_clip_ffmpeg 相同的切法。"""
    _ffmpeg(['-ss', '%.3f' % start, '-i', src, '-t', '%.3f' % seconds, '-an',
             '-vf', 'scale=224:224:force_original_aspect_ratio=increase,crop=224:224', '-r', '12',
             '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '23', '-pix_fmt', 'yuv420p', out])


def make_synth_dataset(root, layout=None, seconds=12, stride=1.5):
    """造一个小的 movinet_clips：layout = {(split, 类别): [源视频名, ...]}。→ (素材目录, {源视频名: 源文件})。
    片段从合成源视频按 stride 切 3 秒，命名同 make_movinet_clips.py。"""
    layout = layout or {('train', 'normal_plot'): ['filmA', 'filmB'], ('train', 'oral'): ['filmC'],
                        ('train', 'sex'): ['filmD'], ('val', 'normal_plot'): ['filmE'], ('val', 'oral'): ['filmF'],
                        ('val', 'sex'): ['filmB'], ('test', 'normal_plot'): ['filmE'], ('test', 'oral'): ['filmF'],
                        ('test', 'sex'): ['filmD']}
    data = os.path.join(root, 'movinet_clips')
    srcdir = os.path.join(root, 'sources')
    os.makedirs(srcdir, exist_ok=True)
    sources = {}
    for (sp, c), films in sorted(layout.items()):
        d = os.path.join(data, sp, c)
        os.makedirs(d, exist_ok=True)
        for f in films:
            if f not in sources:
                sources[f] = os.path.join(srcdir, f + '.mp4')
                make_source(sources[f], SYNTH_SOURCES[f], seconds)
            st = 0.0
            step = stride if sp == 'train' else 3.0
            while st + 3.0 <= seconds - 0.01:
                cut_clip(sources[f], os.path.join(d, '%s_s%.1f_%s.mp4' % (f, st, c)), st)
                st += step
    return data, sources


def selftest():
    _need()
    if not shutil.which('ffmpeg'):
        sys.exit('自检要 ffmpeg 在 PATH 上')
    fails = []

    def check(ok, msg):
        print('  %s %s' % ('ok  ' if ok else 'FAIL', msg))
        if not ok:
            fails.append(msg)

    check(parse_clip_name('a/b/My_Film_s12_x_s3.0_normal_plot.mp4') == ('My_Film_s12_x', 3.0, 'normal_plot')
          and parse_clip_name('x_s1.5_sex.mp4') == ('x', 1.5, 'sex')
          and parse_clip_name('weird name.mp4') == ('weird name', None, None),
          '文件名解析：源视频名带下划线 / _s 也能取对；不合规的整个文件名当源视频')
    a = np.array([0, 0xFFFFFFFFFFFFFFFF, 0x0F], np.uint64)
    check(list(popcount64(a)) == [0, 64, 4], 'popcount64')
    tmp = tempfile.mkdtemp(prefix='data_check_')
    try:
        data, src = make_synth_dataset(tmp)
        n_good = len(list_clips(data))
        # 坏片段：随便写的字节、只有 1 秒的片段
        with open(os.path.join(data, 'train', 'oral', 'broken_s0.0_oral.mp4'), 'wb') as fh:
            fh.write(os.urandom(5000))
        cut_clip(src['filmC'], os.path.join(data, 'train', 'sex', 'short_s0.0_sex.mp4'), 0.0, seconds=1.0)
        rows = scan(data, jobs=2, quiet=True)
        bad = read_bad_list(os.path.join(data, BAD_LIST))
        check(sorted(bad) == ['train/oral/broken_s0.0_oral.mp4', 'train/sex/short_s0.0_sex.mp4']
              and bad['train/oral/broken_s0.0_oral.mp4'] in ('打不开', '读不出帧')
              and bad['train/sex/short_s0.0_sex.mp4'].startswith('帧数不足'),
              'scan：坏文件、帧数不足的片段进 bad_clips.txt（%s）' % dict(bad))
        good = [r for r in rows if int(r['ok'])]
        check(len(good) == n_good and all(int(r['frames']) == 36 and (int(r['width']), int(r['height'])) == (224, 224)
                                          for r in good),
              'scan：%d 个好片段都是 36 帧 224×224' % n_good)
        check(sum(1 for r in good if r['hash1'] and r['hash2'] and r['hash3']) >= 0.9 * len(good),
              'scan：好片段都有画面指纹')
        idx = read_index(os.path.join(data, INDEX))
        check(len(idx) == len(rows) and idx[0]['path'].count('/') == 2, 'clip_index.csv 能读回，路径相对素材目录')
        st = stats(data, quiet=True)
        check(dict(st['overlap']).get('filmB') == ['train', 'val'] and dict(st['overlap']).get('filmE') == ['val', 'test'],
              'stats：同一个源视频出现在两个 split 会报出来（%s）' % st['overlap'])
        # leak：把 filmA、filmD 的源视频当成测试片，应该能找到；filmC 不在测试片里
        gt = os.path.join(tmp, 'gt')
        os.makedirs(gt)
        for vid, f in (('v01', 'filmA'), ('v02', 'filmD')):
            with io.open(os.path.join(gt, vid + '.csv'), 'w', encoding='utf-8') as fh:
                fh.write('# file_name=%s.mp4\nstart_ms,end_ms,label,note\n0,12000,plot,\n' % f)
        found = leak(data, os.path.dirname(src['filmA']), gt_dir=gt, quiet=True)
        films = dict((vid, set(x[2] for x in v)) for vid, v in found.items())
        check(films.get('v01') == {'filmA'} and films.get('v02') == {'filmD'},
              'leak：从测试片切出来的片段能找到，别的源视频不误报（%s）' % films)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    if fails:
        print('selftest: %d 项失败' % len(fails))
        return 1
    print('selftest: ALL PASS')
    return 0


# ======================================================================== main
def main():
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(errors='replace')
        except Exception:
            pass
    if '--selftest' in sys.argv:
        return selftest()
    ap = argparse.ArgumentParser(description='视频训练素材检查（详见文件头）',
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd')
    p = sub.add_parser('scan', help='扫坏片段，写 bad_clips.txt 和 clip_index.csv')
    p.add_argument('--data_dir', required=True, help='movinet_clips 目录（下面是 train/val/test）')
    p.add_argument('--jobs', type=int, default=max(1, min(8, (os.cpu_count() or 2) - 1)), help='进程数')
    p.add_argument('--min_frames', type=int, default=MIN_FRAMES, help='少于这么多帧算坏（默认 %d）' % MIN_FRAMES)
    p.add_argument('--out', help='结果写到哪（默认素材目录根下）')
    p = sub.add_parser('stats', help='按 split / 类别 / 源视频统计')
    p.add_argument('--data_dir', required=True)
    p = sub.add_parser('leak', help='查 13 部测试片有没有混进训练素材（要先 scan）')
    p.add_argument('--data_dir', required=True)
    p.add_argument('--videos', help='13 部原片所在目录')
    p.add_argument('--pair', action='append', help='逐个指定原片：v07=/path/视频7.mp4（可多次）')
    p.add_argument('--gt', default=DEFAULT_GT, help='真值目录（默认 bench/gt），用它文件头的 file_name 找原片')
    p.add_argument('--ffmpeg', default='ffmpeg')
    a = ap.parse_args()
    if a.cmd == 'scan':
        scan(a.data_dir, a.jobs, a.min_frames, a.out)
    elif a.cmd == 'stats':
        stats(a.data_dir)
    elif a.cmd == 'leak':
        leak(a.data_dir, a.videos, a.gt, a.pair, a.ffmpeg)
    else:
        ap.print_help()
        return 2
    return 0


if __name__ == '__main__':
    sys.exit(main())
