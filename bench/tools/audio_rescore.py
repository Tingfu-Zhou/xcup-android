#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
bench/tools/audio_rescore.py — 离线音频重打分

run CSV 记下了真机每个音频结果"是哪一刻出的"（wall_ms - a_age）以及当时的播放位置。有原视频，
就能在电脑上把那一刻送进模型的 2 秒音频还原出来、重新推理：换模型、换解码方式，都不用再上真机
从头播一遍。

子命令:
  check    验证对齐：用 app 里的旧模型，在 run CSV 记录的每个音频结果时刻重算 a_p_*，和真机记下的比。
           同时试两种"喂给模型的音频"，看哪种能把真机概率复现出来：
             phone  逐行复刻离线模式的 AudioDecoder + PcmCircularBuffer（真机现在实际喂的）
             clean  正确的做法：下混单声道、重采样到 16kHz、取 [T-2s, T]
           旧版 run 没记读窗位置，要扫时间偏移；真机解码流开头是否带 AAC priming（--lead）也自动试。
           结果（时间偏移、lead、复现误差）写进 bench/rescored/alignment.json，rescore 直接用。
  rescore  按指定管线和模型重算每个音频结果，写出新的 run CSV：a_p_* 换成新概率，a_action/a_conf
           按原阈值重判，追加 YAMNet AudioSet 类别分数列 a_ys_*。写到 bench/rescored/<tag>/，文件名
           不变，metrics.py / fusion_lab.py 直接能读（fusion_lab 的 check 除外：决策列还是原 run 的）。
  auc      跨片子的音频判别力：每部片和合并后的 AUC。结果级，只用整个 2 秒窗口都落在同一个真值段里的
           结果。--all-classes 给 AudioSet 全部 521 类排行（需要 rescore 产出的 .yamnet.npz）。

用法（Windows 把 python3 换成 python）:
  python3 bench/tools/audio_rescore.py check   --videos D:/xcup/videos
  python3 bench/tools/audio_rescore.py rescore --videos D:/xcup/videos --pipeline phone
  python3 bench/tools/audio_rescore.py rescore --videos D:/xcup/videos --pipeline clean
  python3 bench/tools/audio_rescore.py auc bench/gt bench/rescored/clean
  python3 bench/tools/audio_rescore.py --selftest

视频的查找：--videos 目录下按 真值文件头的 file_name（如 视频1.mp4）→ 真值 id（v01.*）→ run 文件头
的 video_name 依次找；也可以 --pair v01=D:/xcup/视频1.mp4 逐个指定。视频不进仓库。

依赖：numpy；TFLite 解释器（pip install numpy ai-edge-litert，Windows/macOS/Linux，Python 3.10+；
已装 tensorflow 也行）；ffmpeg 和 ffprobe 在 PATH 上（或 --ffmpeg 指定 ffmpeg.exe 的完整路径）。
"""
import argparse
import csv
import datetime
import glob
import io
import json
import os
import re
import shutil
import subprocess
import sys
import time
from collections import OrderedDict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import metrics as M        # noqa: E402
import fusion_lab as F     # noqa: E402

try:
    import numpy as np
except ImportError:        # 只有 --help 能在没有 numpy 时跑
    np = None

TOOLS = os.path.dirname(os.path.abspath(__file__))
ROOT = F.ROOT
ASSETS = os.path.join(ROOT, 'app', 'src', 'main', 'assets')
DEFAULT_YAMNET = os.path.join(ASSETS, 'yamnet.tflite')
DEFAULT_CLASSIFIER = os.path.join(ASSETS, 'yamnet_finetuned.tflite')
DEFAULT_OUT = os.path.join(ROOT, 'bench', 'rescored')
DEFAULT_VIDEOS = os.path.join(ROOT, 'bench', 'videos')
CLASS_MAP = os.path.join(TOOLS, 'yamnet_class_map.csv')

SR = 16000
WIN = 32000              # 模型输入 2 秒 @16kHz：readWindowRelaxed(currentMs, 32000)
N_AUDIOSET = 521
AUDIO_CLASSES = ('sex', 'oral', 'noise')     # yamnet_finetuned 的类目顺序（与 a_p_* 列一致）
VIDEO_EXTS = ('.mp4', '.mkv', '.mov', '.m4v', '.avi', '.webm', '.ts', '.flv', '.wmv', '.3gp')

# 对齐扫描。模型窗口结尾 T = ref - delta（ref 见 Result）。
# 模型输出对窗口位置非常敏感：窗口平移 1~5ms，三类概率的平均绝对差（MAE）就有 0.03~0.06。所以
# MAE 随 delta 的曲线是"一片噪声平台 + 真实偏移处一个只有几毫秒宽的深谷"，粗网格会漏掉。做法：
# 整个范围按 3ms 扫，每个偏移先只算 SCAN_FIRST 个结果；邻点平滑后挑出候选，再用 SCAN_N 个结果精选，
# 最后 1ms 精修。旧版 run 只能从"出结果时刻"反推窗口位置（真机推理耗时抖动约 ±3ms），所以即使复刻
# 完全正确 MAE 也不会是 0；判定主要看两种管线谁明显更好，绝对门槛只用来分档。
SCAN_RANGE = (-200, 400, 3)      # 旧版 run（没有 a_pos）：delta ≈ 推理耗时 + 固定差
SCAN_RANGE_POS = (-100, 100, 2)  # 新版 run 记了 a_pos（读窗时的播放位置）：delta 只剩固定差
SCAN_N = 60                      # 扫描用这么多个结果（均匀抽），定下偏移后再在全部结果上出统计
SCAN_FIRST = 16
GOOD_MAE, OK_MAE = 0.04, 0.10


def _pad(s, n):
    """按终端显示宽度补空格（中文占两格），表格才对得齐。"""
    import unicodedata
    w = sum(2 if unicodedata.east_asian_width(c) in 'WF' else 1 for c in s)
    return s + ' ' * max(0, n - w)


def _rel(path):
    """提示里的路径尽量写成相对当前目录的，命令短一些。"""
    try:
        r = os.path.relpath(path)
        return path if r.startswith('..') else r
    except ValueError:              # Windows 上跨盘符
        return path


def _need_numpy():
    if np is None:
        sys.exit('需要 numpy：pip install numpy ai-edge-litert')


# ======================================================================== 管线
def jdiv(a, b):
    """Java 的整数除法（向零截断）；Python 的 // 是向下取整，负数时结果不同。"""
    q = abs(a) // abs(b)
    return q if (a >= 0) == (b > 0) else -q


def _apply_lead(pcm, ch, lead):
    if lead > 0:
        return np.concatenate([np.zeros(lead * ch, np.int16), pcm])
    if lead < 0:
        return pcm[-lead * ch:]
    return pcm


class PhonePipeline(object):
    """逐行复刻离线模式的音频链路（AudioDecoder.startDecoding → PcmCircularBuffer.write →
    readWindowRelaxed）。输入是解码器输出的交错 int16 PCM（ffmpeg 解出来的，采样率/声道数不变）。

    AudioDecoder 没看声道数，把 MediaCodec 输出的交错 PCM（L R L R …）直接当单声道用：
      windowSize = sampleRate*4 个 short 凑一块；重采样按 i/sampleRate 算原始时间轴、只取前 2 秒，
      即前 2*sampleRate 个 short 拉成 32000 点，时间戳铺满 2000ms。所以
      · 双声道：每块只用到前 1 秒真实音频（L/R 交错），以 0.5 倍速、低八度送进模型，后 1 秒丢弃；
      · 单声道：速度对，但每 4 秒只留前 2 秒，中间 2 秒没有数据。
    另外凑满一块时，所在那个解码 buffer 剩下的部分直接丢掉（releaseOutputBuffer 后不再读）。

    frame：每个解码输出 buffer 的帧数（AAC-LC 1024，HE-AAC 2048）。块的周期由它决定（44.1k 双声道
    = 87 个 buffer = 2020ms），猜错的话相位会随时间漂移，整部片都对不上。
    lead：真机解码流比 ffmpeg 多出来的开头样本数（正数：编码器 priming，ffmpeg 按 MP4 edit list
    剪掉了，MediaCodec 可能不剪；负数：真机少了开头这么多样本）。只影响分块相位，默认 0；
    由此带来的固定时间差由对齐扫描的 delta 吸收。
    """
    CAPACITY = SR * 20              # new PcmCircularBuffer(16000, 20)
    MAX_ADVANCE_MS = 3000           # AudioDecoder.MAX_ADVANCE_NORMAL：解码最多领先播放 3 秒

    def __init__(self, pcm, sr, ch, frame=1024, lead=0, base_us=0):
        _need_numpy()
        pcm = _apply_lead(np.asarray(pcm, dtype=np.int16), ch, lead)
        self.sr, self.ch, self.frame, self.lead = sr, ch, frame, lead
        spb = frame * ch                          # 一个解码输出 buffer 的 short 数
        win = sr * 4                              # AudioDecoder: windowSize = sampleRate * 4
        bpw = -(-win // spb)                      # 凑满一块用掉几个 buffer

        def pts_ms(b):
            # MediaExtractor：样本时间 µs = 帧号*1e6/采样率（整数除）；AudioDecoder：presentationTimeUs/1000
            return jdiv(jdiv(b * frame * 1000000, sr) + base_us, 1000)

        if sr != SR:
            # AudioDecoder 里的线性插值，float 语义逐条照抄。凑满时 bufferPos == windowSize，
            # 所以每块的插值位置都一样，算一次。
            ot = np.arange(win, dtype=np.int64).astype(np.float32) / np.float32(sr)
            tt = np.arange(WIN, dtype=np.int64).astype(np.float32) / np.float32(SR)
            j = np.searchsorted(ot, tt, side='left') - 1
            j = np.clip(j, 0, win - 2)           # Java: while (j < bufferPos-1 && ot[j+1] < tt[i]) j++
            self._j = j
            self._ratio = (tt - ot[j]) / (ot[j + 1] - ot[j])
        n_out = WIN if sr != SR else win          # 16kHz 源不重采样，整块 windowSize 个点直接写
        nblk = 0 if len(pcm) < win else (len(pcm) - win) // (bpw * spb) + 1
        S = np.empty(nblk * n_out, np.float32)
        TS = np.empty(nblk * n_out, np.int32)
        starts, avail = [], []
        step = np.arange(n_out, dtype=np.int64) * 1000 // SR     # PcmCircularBuffer.write: i*1000L/16000
        for k in range(nblk):
            b0 = k * bpw
            # windowStartPtsMs 只在 <0 时赋值：开头若有负的 pts，会被后面第一个 >=0 的 buffer 覆盖
            p0 = pts_ms(b0)
            if p0 < 0:
                for b in range(b0, b0 + bpw):
                    p0 = pts_ms(b)
                    if p0 >= 0:
                        break
            a = pcm[b0 * spb: b0 * spb + win].astype(np.float32) / np.float32(32768)
            if sr != SR:
                v1 = a[self._j]
                out = v1 + self._ratio * (a[self._j + 1] - v1)
            else:
                out = a
            S[k * n_out:(k + 1) * n_out] = out
            TS[k * n_out:(k + 1) * n_out] = p0 + step
            starts.append(p0)
            # 解码线程只在 pts <= 播放位置+3 秒时才处理一个 buffer；一块在它最后一个 buffer 处理时写入
            avail.append(pts_ms(b0 + bpw - 1) - self.MAX_ADVANCE_MS)
        self.S, self.TS = S, TS
        self.n_out = n_out
        self.block_start_ms = np.asarray(starts, np.int64)
        self.avail = np.asarray(avail, np.int64)
        self.monotonic = bool(nblk == 0 or np.all(np.diff(TS.astype(np.int64)) >= 0))

    def window(self, T):
        """readWindowRelaxed(T, 32000)：从最新写入的样本往回收 32000 个 0 < ts <= T 的样本。"""
        T = int(T)
        K = int(np.searchsorted(self.avail, T, side='right'))     # 播放到 T 时已经写进缓冲的块数
        end = K * self.n_out
        beg = max(0, end - self.CAPACITY)                         # 环形缓冲只留最近 20 秒（按样本数）
        if end <= beg:
            return None
        if self.monotonic:
            seg = self.TS[beg:end]
            hi = beg + int(np.searchsorted(seg, T, side='right'))
            first = beg + int(np.searchsorted(seg, 0, side='right'))       # ts > 0 才算有效
            lo = max(first, hi - WIN)
            cnt = hi - lo
            src = self.S[lo:hi] if cnt > 0 else None
        else:                                                     # 时间戳有重叠（如 16kHz 双声道源）
            seg = self.TS[beg:end]
            idx = np.flatnonzero((seg > 0) & (seg <= T))[-WIN:] + beg
            cnt = len(idx)
            src = self.S[idx] if cnt > 0 else None
        if cnt < WIN * 0.9:                                       # Java: count < sampleCount * 0.9f → null
            return None
        out = np.zeros(WIN, np.float32)                           # 不足 32000 时前面补零
        out[WIN - cnt:] = src
        return out

    def describe(self):
        if self.sr == SR and self.ch == 1:
            return '16kHz 单声道源：与正确做法一致'
        if self.ch >= 2:
            real = 2.0 / self.ch if self.sr != SR else 4.0 / self.ch
            return ('AudioDecoder 把 %d 声道交错 PCM 当单声道：每块只用了 %.1f 秒真实音频，'
                    '拉长到 2 秒（%.2f 倍速）送进模型' % (self.ch, real, real / 2.0))
        return 'AudioDecoder：单声道源每 4 秒只用前 2 秒，速度正确，但中间 2 秒没有数据'


class CleanPipeline(object):
    """修好 AudioDecoder 后应有的输入：下混单声道 + 重采样到 16kHz（ffmpeg），取 [T-2s, T]。"""

    def __init__(self, mono16k, base_ms=0.0):
        _need_numpy()
        self.x = np.asarray(mono16k, dtype=np.float32)
        self.base_ms = float(base_ms)

    def window(self, T):
        end = int(round((T + 1 - self.base_ms) * SR / 1000.0))   # 与 phone 一致：含 ts == T 的那 1ms
        start = end - WIN
        s, e = max(0, start), min(end, len(self.x))
        if e - s < WIN * 0.9:
            return None
        out = np.zeros(WIN, np.float32)
        out[s - start:e - start] = self.x[s:e]
        return out

    def describe(self):
        return '下混单声道 + ffmpeg 重采样到 16kHz，窗口 [T-2s, T]'


# ======================================================================== ffmpeg
def find_ffmpeg(arg):
    ff = arg or shutil.which('ffmpeg')
    if ff and not os.path.exists(ff):
        ff = shutil.which(ff)
    if not ff:
        sys.exit('找不到 ffmpeg。\n'
                 '  · Windows：下载 ffmpeg（如 gyan.dev 的 release essentials），把 bin 目录加进 PATH，\n'
                 '    或者用 --ffmpeg C:/ffmpeg/bin/ffmpeg.exe 指定\n'
                 '  · macOS: brew install ffmpeg；Ubuntu: sudo apt install ffmpeg')
    d, base = os.path.split(ff)
    probe = os.path.join(d, base.lower().replace('ffmpeg', 'ffprobe')) if 'ffmpeg' in base.lower() else ''
    if not probe or not os.path.exists(probe):
        probe = shutil.which('ffprobe') or ''
    return ff, probe


def _run(cmd):
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if p.returncode != 0:
        sys.exit('命令失败: %s\n%s' % (' '.join(cmd[:6]) + ' …',
                                       p.stderr.decode('utf-8', 'replace')[-2000:]))
    return p.stdout


_LAYOUT_CH = {'mono': 1, 'stereo': 2, '2.1': 3, '3.0': 3, 'quad': 4, '4.0': 4, '5.0': 5,
              '5.1': 6, '6.1': 7, '7.1': 8}


def probe_audio(path, ffmpeg, ffprobe):
    """第一条音轨（MediaExtractor 也是选第一条 audio/* 轨）的采样率、声道数、每帧样本数。"""
    info = {'codec': '?', 'profile': '', 'sample_rate': None, 'channels': None,
            'start_time': 0.0, 'frame': None}
    if ffprobe:
        out = _run([ffprobe, '-v', 'error', '-select_streams', 'a:0', '-show_entries',
                    'stream=codec_name,profile,sample_rate,channels,start_time', '-of', 'json', path])
        st = (json.loads(out.decode('utf-8', 'replace')).get('streams') or [None])[0]
        if not st:
            sys.exit('%s: 没有音轨' % path)
        info['codec'] = st.get('codec_name', '?')
        info['profile'] = st.get('profile', '') or ''
        info['sample_rate'] = int(st['sample_rate'])
        info['channels'] = int(st['channels'])
        try:
            info['start_time'] = float(st.get('start_time') or 0.0)
        except ValueError:
            pass
        out = _run([ffprobe, '-v', 'error', '-select_streams', 'a:0', '-show_entries', 'frame=nb_samples',
                    '-read_intervals', '%+#12', '-of', 'json', path])
        ns = [int(f['nb_samples']) for f in json.loads(out.decode('utf-8', 'replace')).get('frames', [])
              if f.get('nb_samples')]
        if ns:
            info['frame'] = max(set(ns), key=ns.count)
    else:                    # 没有 ffprobe：从 ffmpeg -i 的输出里抠
        p = subprocess.run([ffmpeg, '-hide_banner', '-i', path], stdout=subprocess.PIPE,
                           stderr=subprocess.PIPE)
        txt = p.stderr.decode('utf-8', 'replace')
        m = re.search(r'Audio: (\w+)(?: \(([^)]*)\))?[^,]*, (\d+) Hz, ([^,]+),', txt)
        if not m:
            sys.exit('%s: 找不到音轨（也没有 ffprobe 可用）' % path)
        info['codec'], info['profile'] = m.group(1), m.group(2) or ''
        info['sample_rate'] = int(m.group(3))
        lay = m.group(4).strip()
        mc = re.match(r'(\d+) channels', lay)
        info['channels'] = int(mc.group(1)) if mc else _LAYOUT_CH.get(lay.split('(')[0], 2)
    if not info['frame']:
        info['frame'] = 2048 if 'HE' in info['profile'].upper() else 1024
    return info


def decode(path, ffmpeg, args, dtype):
    cmd = [ffmpeg, '-nostdin', '-hide_banner', '-v', 'error', '-i', path,
           '-map', '0:a:0', '-vn', '-sn', '-dn'] + args + ['pipe:1']
    return np.frombuffer(_run(cmd), dtype=dtype)


def build_pipeline(kind, path, info, ffmpeg, frame=None, lead=0):
    t0 = time.time()
    if kind == 'phone':
        pcm = decode(path, ffmpeg, ['-f', 's16le', '-acodec', 'pcm_s16le'], np.int16)
        base_us = int(round(max(0.0, info['start_time']) * 1e6))
        pipe = PhonePipeline(pcm, info['sample_rate'], info['channels'], frame or info['frame'],
                             lead, base_us)
        pipe.raw, pipe.base_us = pcm, base_us          # 换 lead 重建时不用再解码
    else:
        x = decode(path, ffmpeg, ['-ac', '1', '-ar', str(SR), '-f', 'f32le', '-acodec', 'pcm_f32le'],
                   np.float32)
        pipe = CleanPipeline(x, max(0.0, info['start_time']) * 1000.0)
    pipe.build_s = time.time() - t0
    return pipe


# ======================================================================== 模型
def _interpreter_class():
    try:
        from ai_edge_litert.interpreter import Interpreter
        return Interpreter
    except ImportError:
        pass
    try:
        from tflite_runtime.interpreter import Interpreter
        return Interpreter
    except ImportError:
        pass
    try:
        import tensorflow as tf
        return tf.lite.Interpreter
    except ImportError:
        pass
    sys.exit('需要 TFLite 解释器：pip install ai-edge-litert（Python 3.10+）；或者已装 tensorflow 也行')


class AudioModel(object):
    """与 AudioInferenceHelper.predict() 相同：YAMNet → 各帧 embedding 平均 → 分类器。
    另外留下 output 0 的 AudioSet 521 类分数（各帧平均，Java 端 a_ys_* 列同样按帧平均）。"""

    def __init__(self, yamnet=DEFAULT_YAMNET, classifier=DEFAULT_CLASSIFIER, threads=None):
        _need_numpy()
        for p in (yamnet, classifier):
            if not os.path.exists(p):
                sys.exit('找不到模型文件: %s' % p)
        Interp = _interpreter_class()
        threads = threads or min(4, os.cpu_count() or 1)
        self.yamnet_path, self.classifier_path = yamnet, classifier
        self.y = Interp(model_path=yamnet, num_threads=threads)
        inp = self.y.get_input_details()[0]
        self.y.resize_tensor_input(inp['index'], [WIN])
        self.y.allocate_tensors()
        self.y_in = inp['index']
        outs = self.y.get_output_details()
        try:   # 按形状认：[帧, 521] 是 AudioSet 分数（output 0），[帧, 1024] 是 embedding（output 1）
            self.y_scores = next(o['index'] for o in outs if o['shape'][-1] == N_AUDIOSET)
            self.y_emb = next(o['index'] for o in outs if o['shape'][-1] == 1024)
        except StopIteration:
            sys.exit('%s 不像 YAMNet：找不到 [N,521] / [N,1024] 输出' % yamnet)
        self.c = Interp(model_path=classifier, num_threads=1)
        self.c.allocate_tensors()
        ci, co = self.c.get_input_details()[0], self.c.get_output_details()[0]
        if ci['shape'][-1] != 1024:
            sys.exit('%s 的输入不是 1024 维 embedding' % classifier)
        self.c_in, self.c_out = ci['index'], co['index']
        self.n_classes = int(co['shape'][-1])
        if self.n_classes != len(AUDIO_CLASSES):
            sys.exit('%s 输出 %d 类；run CSV 的 a_p_* 是 [sex, oral, noise] 三类，'
                     '换了类目要先改 a_p_* 的约定' % (classifier, self.n_classes))
        self.calls = 0

    def run(self, w):
        """→ (probs float32[3], AudioSet 分数 float32[521])"""
        self.y.set_tensor(self.y_in, np.ascontiguousarray(w, dtype=np.float32))
        self.y.invoke()
        emb = self.y.get_tensor(self.y_emb)
        sc = self.y.get_tensor(self.y_scores)
        n = np.float32(emb.shape[0])
        m = emb[0].copy()                       # Java：逐帧累加再除帧数（float）
        for i in range(1, emb.shape[0]):
            m += emb[i]
        m /= n
        s = sc[0].copy()
        for i in range(1, sc.shape[0]):
            s += sc[i]
        s /= n
        self.c.set_tensor(self.c_in, m[None, :])
        self.c.invoke()
        self.calls += 1
        return self.c.get_tensor(self.c_out)[0].astype(np.float32), s


def audioset_names():
    names = ['class_%d' % i for i in range(N_AUDIOSET)]
    if os.path.exists(CLASS_MAP):
        with io.open(CLASS_MAP, encoding='utf-8') as fh:
            for row in csv.DictReader(fh):
                names[int(row['index'])] = row['display_name']
    return names


# ======================================================================== run CSV 里的音频结果
class Result(object):
    """一个音频推理结果（同一结果在相邻几个 tick 里重复出现，按出结果时刻去重）。

    ts：出结果的墙钟时刻 = wall_ms - a_age。
    ref：参考播放位置。新版 run 有 a_pos 列（音频线程读窗时的播放位置），ref 就是它，窗口正好是
         [ref-2s, ref]；旧版 run 没有，ref = 出结果那一刻的播放位置（用前后 tick 的 pos-wall 中位数插出来）。
         音频线程是先取播放位置、再读 PCM、推理（约 65ms）、最后才记 ts，所以旧版的 ref 比窗口结尾晚一个推理耗时。
    模型窗口结尾 T = ref - delta：delta 由 check 扫出来（旧版 ≈ 推理耗时，新版 ≈ 0）。"""
    __slots__ = ('ts', 'ref', 'has_apos', 'logged', 'consistent', 'rows')

    def __init__(self, ts, ref, logged, consistent, rows=(), has_apos=False):
        self.ts, self.ref, self.logged, self.consistent = ts, ref, logged, consistent
        self.rows, self.has_apos = rows, has_apos


def load_results(run_path):
    meta, data = M._rows(run_path)
    groups = OrderedDict()
    walls, offs = [], []
    for i, d in enumerate(data):
        wall, pos = F._inum(d.get('wall_ms'), 0), F._inum(d.get('pos_ms'), 0)
        walls.append(wall)
        offs.append(pos - wall)
        age = F._inum(d.get('a_age'), -1)
        if age >= 0:
            groups.setdefault(wall - age, []).append(i)
    walls = np.asarray(walls, np.int64)
    offs = np.asarray(offs, np.int64)
    order = np.argsort(walls, kind='mergesort')
    walls, offs = walls[order], offs[order]
    res = []
    for ts, idx in groups.items():
        probs, apos = [], []
        for i in idx:
            pr = tuple(F._fnum(data[i].get(c)) for c in ('a_p_sex', 'a_p_oral', 'a_p_noise'))
            if all(x is not None for x in pr):
                probs.append(pr)
            ap = F._inum(data[i].get('a_pos'), -1)
            if ap >= 0:
                apos.append(ap)
        if apos:
            ref, has = float(max(set(apos), key=apos.count)), True
        else:
            lo = int(np.searchsorted(walls, ts - 2500))
            hi = int(np.searchsorted(walls, ts + 2500, side='right'))
            if hi <= lo:
                k = min(len(walls) - 1, max(0, lo))
                lo, hi = k, k + 1
            ref, has = ts + float(np.median(offs[lo:hi])), False
        logged = max(set(probs), key=probs.count) if probs else None
        res.append(Result(ts, ref, logged, len(set(probs)) <= 1, idx, has))
    return meta, data, res


def _usable(results, min_pos=4000):
    """能拿来比对的结果：真机记了概率、同一结果的几行概率一致（旧版偶有竞态）、已过开头预热。"""
    return [r for r in results if r.logged is not None and r.consistent and r.ref >= min_pos]


def _subset(rs, n):
    if len(rs) <= n:
        return list(rs)
    step = len(rs) / float(n)
    return [rs[int(i * step)] for i in range(n)]


def compare(pipe, model, rs, delta):
    """在偏移 delta 下重算 rs，返回 (新概率 [n,3], 真机记录 [n,3], 用到的结果)。"""
    new, old, used = [], [], []
    for r in rs:
        w = pipe.window(r.ref - delta)
        if w is None:
            continue
        new.append(model.run(w)[0])
        old.append(r.logged)
        used.append(r)
    return np.asarray(new, np.float64).reshape(-1, 3), np.asarray(old, np.float64).reshape(-1, 3), used


def _mae(new, old):
    return float(np.abs(new - old).mean()) if len(new) else float('inf')


def default_scan(results):
    us = _usable(results)
    return SCAN_RANGE_POS if us and all(r.has_apos for r in us) else SCAN_RANGE


def scan_offset(pipe, model, results, rng=None, n=SCAN_N):
    """找让复刻结果最接近真机记录的 delta。返回 (delta, [(delta, MAE, 结果数)], {plateau, dip})，
    没有可比对的结果时 delta=None。"""
    us = _usable(results)
    if not us:
        return None, [], {}
    lo, hi, step = rng or default_scan(results)
    sub = _subset(us, n)
    sub = [sub[i] for i in np.random.RandomState(0).permutation(len(sub))]   # 每轮都铺满整部片
    offs = list(range(lo, hi + 1, step))
    errs = {}

    def run(ds, batch):
        for d in ds:
            new, old, _ = compare(pipe, model, batch, d)
            errs.setdefault(d, []).extend(np.abs(new - old).mean(axis=1).tolist())

    def mae(d):
        return float(np.mean(errs[d])) if errs.get(d) else float('inf')

    k = min(SCAN_FIRST, len(sub))
    run(offs, sub[:k])                                     # 第 1 轮：全部偏移 × 少量结果
    m = np.asarray([mae(d) for d in offs])
    sm = m.copy()
    if len(m) > 2:                                         # 真谷有几毫秒宽，邻点平滑压掉单点噪声低谷
        sm[1:-1] = (m[:-2] + 2 * m[1:-1] + m[2:]) / 4.0
    top = np.argsort(sm)[:10]
    cand = sorted(set(offs[j] for i in top for j in (i - 1, i, i + 1) if 0 <= j < len(offs)))
    run(cand, sub[k:])                                     # 第 2 轮：候选 × 其余结果
    best = min(cand, key=mae)
    fine = [d for d in range(best - step + 1, best + step) if d not in errs]
    run(fine, sub)                                         # 第 3 轮：1ms 精修
    full = [d for d in errs if len(errs[d]) >= len(errs[best])]
    best = min(full, key=mae)
    curve = sorted((d, mae(d), len(v)) for d, v in errs.items())
    # 谷有多深：第 1 轮里（所有偏移用的是同一批结果）最佳偏移 vs 中位数（平台）
    first = [float(np.mean(errs[d][:k])) for d in offs if errs.get(d)]
    info = {'plateau': float(np.median(first)) if first else float('nan'),
            'dip': float(np.mean(errs[best][:k])) if errs.get(best) else float('nan')}
    return best, curve, info


LEAD_CANDIDATES = (1024, 2048, 2112)      # 常见 AAC priming：ffmpeg 编码 1024，Apple/Nero/FDK 2112


def probe_lead(pipe0, best0, model, results, n=SCAN_N):
    """真机解码流开头是否多了 priming 样本（ffmpeg 按 edit list 剪掉了）。多了 L 个样本时，
    同样的内容在时间戳上晚 ch*L/sr 秒（双声道半速），扫出来的 delta 会偏大这么多，但块边界对不上，
    找不到尖谷。做法：每个候选 L 在预测的 delta 附近用少量结果粗筛，最好的那个再用全部扫描结果复核，
    比 lead=0 好 20% 以上才换（只用少量结果比，在几十次尝试里挑最小值，容易被噪声骗）。
    返回 (lead, delta, pipe)。"""
    rs = _subset(_usable(results), n)
    rs = [rs[i] for i in np.random.RandomState(0).permutation(len(rs))]
    few = rs[:SCAN_FIRST]

    def mae_at(pipe, d, batch):
        new, old, _ = compare(pipe, model, batch, d)
        return _mae(new, old)

    screen = []
    for L in LEAD_CANDIDATES:
        if L == pipe0.lead:
            continue
        p = PhonePipeline(pipe0.raw, pipe0.sr, pipe0.ch, pipe0.frame, L, pipe0.base_us)
        pred = best0 - int(round(pipe0.ch * (L - pipe0.lead) * 1000.0 / pipe0.sr))
        m, d = min((mae_at(p, d, few), d) for d in range(pred - 12, pred + 13, 2))
        screen.append((m, L, d))
        del p
    if not screen:
        return pipe0.lead, best0, pipe0
    _, L, d = min(screen)
    p = PhonePipeline(pipe0.raw, pipe0.sr, pipe0.ch, pipe0.frame, L, pipe0.base_us)
    m, d = min((mae_at(p, x, rs), x) for x in range(d - 2, d + 3))
    base = mae_at(pipe0, best0, rs)
    if m > 0.8 * base:
        return pipe0.lead, best0, pipe0
    p.raw, p.base_us = pipe0.raw, pipe0.base_us
    return L, d, p


def match_stats(new, old):
    if not len(new):
        return {'n': 0, 'mae': float('inf'), 'p90': float('nan'), 'corr_sex': float('nan'),
                'corr': float('nan'), 'within_005': 0.0, 'argmax': 0.0}
    d = np.abs(new - old)
    mx = d.max(axis=1)
    cors = []
    for c in range(new.shape[1]):
        a, b = new[:, c], old[:, c]
        cors.append(float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else float('nan'))
    ok = [x for x in cors if x == x]
    return {'n': int(len(new)), 'mae': float(d.mean()), 'p90': float(np.percentile(mx, 90)),
            'corr_sex': cors[0], 'corr': float(np.mean(ok)) if ok else float('nan'),
            'within_005': float((mx < 0.05).mean()),
            'argmax': float((new.argmax(1) == old.argmax(1)).mean())}


def _grade(st):
    if st['mae'] < GOOD_MAE:
        return '✓ 吻合'
    if st['mae'] < OK_MAE:
        return '~ 大致吻合'
    return '✗ 对不上'


# ======================================================================== 配对：真值 / run / 视频
def _gt_meta(path):
    meta, _ = M._rows(path)
    return meta


def find_video(videos_dir, vid, gt_meta, run_meta):
    if not videos_dir or not os.path.isdir(videos_dir):
        return None
    fn = gt_meta.get('file_name', '')
    if fn and os.path.exists(os.path.join(videos_dir, fn)):
        return os.path.join(videos_dir, fn)
    stems = [s for s in (vid, os.path.splitext(fn)[0] if fn else '',
                         gt_meta.get('video_id', '')) if s]
    vname = run_meta.get('video_name', '')
    for f in sorted(os.listdir(videos_dir)):
        stem, ext = os.path.splitext(f)
        if ext.lower() not in VIDEO_EXTS:
            continue
        if stem in stems or (vname and F._sanitize(stem) == vname):
            return os.path.join(videos_dir, f)
    return None


def resolve_pairs(a, need_video=True):
    """→ [(vid, gt_path, run_path, video_path)]。"""
    explicit = {}
    for p in a.pair or []:
        if '=' not in p:
            sys.exit('--pair 格式应为 v01=视频路径: %r' % p)
        k, v = p.split('=', 1)
        explicit[k.strip()] = v.strip()
    gts = sorted(glob.glob(os.path.join(a.gt, '*.csv')))
    runs = sorted(glob.glob(os.path.join(a.runs, '*.csv'))) if os.path.isdir(a.runs) else []
    if not gts:
        sys.exit('在 %s 下没找到真值 CSV' % a.gt)
    if not runs:
        sys.exit('在 %s 下没找到 run CSV' % a.runs)
    only = set(a.only.split(',')) if getattr(a, 'only', None) else None
    out = []
    for g in gts:
        vid = os.path.splitext(os.path.basename(g))[0]
        if only and vid not in only:
            continue
        gmeta = _gt_meta(g)
        vname = F._sanitize(os.path.splitext(gmeta.get('file_name', ''))[0])
        r, _ = F._pick_run(vid, runs, vname)
        if r is None:
            print('  - %s: 没有对应的 run CSV，跳过' % vid)
            continue
        if not need_video:
            out.append((vid, g, r, None))
            continue
        rmeta, _ = M._rows(r)
        v = explicit.get(vid) or find_video(a.videos, vid, gmeta, rmeta)
        if not v or not os.path.exists(v):
            print('  - %s: 找不到视频（file_name=%s，run 里 video_name=%s）。放进 --videos 目录，'
                  '或用 --pair %s=<视频路径>' % (vid, gmeta.get('file_name', '?'),
                                             rmeta.get('video_name', '?'), vid))
            continue
        size = gmeta.get('file_size')
        if size and size.isdigit() and int(size) != os.path.getsize(v):
            print('  ⚠ %s: %s 的大小 %d 与真值记录的 %s 不同 —— 是不是同一个文件？'
                  % (vid, os.path.basename(v), os.path.getsize(v), size))
        out.append((vid, g, r, v))
    if not out:
        sys.exit('没有可用的 真值 + run + 视频 组合')
    return out


# ======================================================================== alignment.json
def load_alignment(path):
    if path and os.path.exists(path):
        with io.open(path, encoding='utf-8') as fh:
            return json.load(fh)
    return {}


def save_alignment(path, data):
    d = os.path.dirname(path)
    if d and not os.path.isdir(d):
        os.makedirs(d)
    text = json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True)
    text = re.sub(r'\[\s+([^\[\]{}]*?)\s+\]', lambda m: '[' + ' '.join(m.group(1).split()) + ']', text)
    with io.open(path, 'w', encoding='utf-8', newline='\n') as fh:
        fh.write(text + '\n')


# ======================================================================== check
def _fmt_audio(info):
    return '%s%s %d Hz %d 声道，每帧 %d 样本，起点 %.3fs' % (
        info['codec'], (' (%s)' % info['profile']) if info['profile'] else '',
        info['sample_rate'], info['channels'], info['frame'], info['start_time'])


def cmd_check(a):
    _need_numpy()
    ffmpeg, ffprobe = find_ffmpeg(a.ffmpeg)
    pairs = resolve_pairs(a)
    model = AudioModel(a.yamnet, a.classifier, a.threads)
    align = load_alignment(a.alignment)
    rng = tuple(int(x) for x in a.scan.split(':')) if a.scan else None
    kinds = ('phone', 'clean') if a.pipeline == 'both' else (a.pipeline,)
    summary = []
    for vid, g, r, vpath in pairs:
        t0 = time.time()
        calls0 = model.calls
        info = probe_audio(vpath, ffmpeg, ffprobe)
        meta, data, res = load_results(r)
        usable = _usable(res)
        scan = rng or default_scan(res)
        print('══ %s  %s ══' % (vid, os.path.basename(vpath)))
        print('  音轨   %s' % _fmt_audio(info))
        print('  run    %s：%d 个音频结果，可比对 %d 个；%s，偏移在 %d~%dms 里扫'
              % (os.path.basename(r), len(res), len(usable),
                 '有 a_pos（读窗位置是记下来的）' if scan == SCAN_RANGE_POS
                 else '旧版 run，窗口位置从出结果时刻反推', scan[0], scan[1]))
        if 'HE' in info['profile'].upper():
            print('  ⚠ HE-AAC：Android 取到的 KEY_SAMPLE_RATE 可能是核心采样率（输出的一半），'
                  '复刻可能对不上；可试 --frame-size')
        rec = {'run': os.path.basename(r), 'video': os.path.basename(vpath),
               'checked_at': datetime.datetime.now().strftime('%Y-%m-%d %H:%M'),
               'audio': info, 'frame': a.frame_size or info['frame'],
               'model': [os.path.basename(model.yamnet_path), os.path.basename(model.classifier_path)]}
        rows = []
        dump = {}
        for kind in kinds:
            lead = 0 if a.lead == 'auto' else int(a.lead)
            pipe = build_pipeline(kind, vpath, info, ffmpeg, a.frame_size, lead)
            print('  %-6s %s' % (kind, pipe.describe()))
            best, curve, sinfo = scan_offset(pipe, model, res, scan, a.scan_n)
            if best is None:
                print('  ✗ 没有可比对的结果')
                continue
            if kind == 'phone' and a.lead == 'auto':
                lead, best2, pipe2 = probe_lead(pipe, best, model, res, a.scan_n)
                if pipe2 is not pipe:
                    print('         真机解码流开头多 %d 个样本（AAC priming）时对得更好：偏移 %+dms → %+dms'
                          % (lead, best, best2))
                    pipe, best = pipe2, best2
            new, old, used = compare(pipe, model, usable, best)
            st = match_stats(new, old)
            st['offset_ms'] = best
            st['lead'] = lead if kind == 'phone' else 0
            st.update(sinfo)
            nmax = max(c[2] for c in curve)
            st['top'] = [[d, round(e, 4)] for d, e, n in sorted(curve, key=lambda x: x[1]) if n == nmax][:6]
            rec[kind] = dict((k, round(v, 4) if isinstance(v, float) else v) for k, v in st.items())
            rows.append((kind, st))
            if a.dump:
                dump[kind] = dict((u.ts, p) for u, p in zip(used, new))
            if a.verbose:
                print('         候选偏移（MAE，%d 个扫描结果）：' % nmax
                      + '  '.join('%+d:%.3f' % (d, e) for d, e in st['top']))
            del pipe
        print('  ' + ''.join(_pad(h, w) for h, w in (('管线', 7), ('最佳偏移', 10), ('MAE', 8), ('p90|Δ|', 8),
                                                       ('相关', 8), ('|Δ|<0.05', 10), ('argmax一致', 12),
                                                       ('谷/平台', 14))))
        for kind, st in rows:
            print('  %-7s%+6dms   %-8.3f%-8.3f%-8.3f%-10s%-12s%-14s%s'
                  % (kind, st['offset_ms'], st['mae'], st['p90'], st['corr'],
                     '%.1f%%' % (100 * st['within_005']), '%.1f%%' % (100 * st['argmax']),
                     '%.3f/%.3f' % (st['dip'], st['plateau']), _grade(st)))
        verdict = None
        if rows:
            bk, bst = min(rows, key=lambda x: x[1]['mae'])
            others = [st['mae'] for k, st in rows if k != bk]
            clear = not others or min(others) >= 2 * bst['mae']
            if clear and bst['mae'] < 0.15:
                verdict = bk
                rec['device_pipeline'] = bk
                if bst['mae'] < GOOD_MAE:
                    print('  → ✓ 对齐验证通过：%s 管线在偏移 %+dms 处逐个复现了真机概率（MAE %.3f）'
                          % (bk, bst['offset_ms'], bst['mae']))
                else:
                    why = ('谷很深，多半是旧版 run 只能从出结果时刻反推窗口位置，几毫秒的抖动就让概率有出入；'
                           '新版 app 记了 a_pos 之后能逐个对上' if bst['dip'] < 0.6 * bst['plateau'] else
                           '没有明显的谷，可能是解码器差异或分块相位不对，试 --lead / --frame-size')
                    print('  → ✓ 真机喂给模型的是 %s 管线的音频%s，但逐个结果还有出入（MAE %.3f）：%s'
                          % (bk, '（比另一种接近得多）' if others else '', bst['mae'], why))
                if bk == 'phone' and info['channels'] >= 2:
                    print('    也就是说，真机送进模型的是 0.5 倍速、L/R 交错的音频（AudioDecoder 没按声道数处理）')
            else:
                print('  → ✗ 没能确认：%s' % ('两种管线差不多' if not clear else '最好的 MAE 也有 %.3f'
                                          % bst['mae']))
                print('    排查：视频文件对不对（大小）；--verbose 看有没有明显的谷；'
                      '试 --lead 1024 / 2112、--frame-size 2048')
        if a.dump and dump:
            path = os.path.join(a.out, 'check_%s.csv' % vid)
            _write_dump(path, res, dump, dict((k, st['offset_ms']) for k, st in rows))
            print('  逐结果对比已写到 %s' % path)
        align[vid] = rec
        summary.append((vid, rows, verdict))
        print('  （%.0f 秒，推理 %d 次）\n' % (time.time() - t0, model.calls - calls0))
    save_alignment(a.alignment, align)
    print('══ 汇总 ══')
    for vid, rows, verdict in summary:
        print('  %-5s %s  → %s' % (vid, '   '.join('%s MAE %.3f @%+dms' % (k, st['mae'], st['offset_ms'])
                                                 for k, st in rows), verdict or '未确认'))
    print('\n对齐结果已写进 %s（rescore 会用里面的偏移）' % _rel(a.alignment))


def _write_dump(path, res, dump, offsets):
    d = os.path.dirname(path)
    if d and not os.path.isdir(d):
        os.makedirs(d)
    kinds = sorted(dump)
    with io.open(path, 'w', encoding='utf-8', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['result_wall_ms', 'ref_pos_ms', 'logged_sex', 'logged_oral', 'logged_noise']
                   + ['%s_%s' % (k, c) for k in kinds for c in ('win_end_ms',) + AUDIO_CLASSES])
        for r in res:
            if r.logged is None:
                continue
            row = [r.ts, int(round(r.ref))] + ['%.4f' % x for x in r.logged]
            for k in kinds:
                p = dump[k].get(r.ts)
                row += [int(round(r.ref - offsets[k]))]
                row += ['%.4f' % x for x in p] if p is not None else ['', '', '']
            w.writerow(row)


# ======================================================================== rescore
def classify(probs, do_th, noise_th):
    """VideoProcessActivity 音频线程的分组概率判定（float 语义）。"""
    p_do = F.f32(F.f32(float(probs[0])) + F.f32(float(probs[1])))
    if p_do >= F.f32(do_th):
        return 'do', p_do
    if F.f32(float(probs[2])) >= F.f32(noise_th):
        return 'Noise', F.f32(float(probs[2]))
    return '', 0.0


def write_rescored(run_path, out_path, computed, note):
    """computed: {结果时刻 ts: (probs[3], ys[len(YS_COLS)])}。其余列原样保留。"""
    comments, header, rows = [], None, []
    with io.open(run_path, encoding='utf-8-sig') as fh:
        for raw in fh:
            line = raw.rstrip('\r\n')
            if not line.strip():
                continue
            if line.lstrip().startswith('#'):
                comments.append(line)
                continue
            f = next(csv.reader([line]))
            if header is None:
                header = [c.strip() for c in f]
            else:
                rows.append(f)
    meta, _ = M._rows(run_path)
    params = F.parse_params(meta.get('params', ''))
    cols = header + [c for c in M.YS_COLS if c not in header]
    ix = dict((c, i) for i, c in enumerate(cols))
    n_hit = n_tick = 0
    for f in rows:
        f += [''] * (len(cols) - len(f))
        age = F._inum(f[ix['a_age']], -1)
        wall = F._inum(f[ix['wall_ms']], 0)
        got = computed.get(wall - age) if age >= 0 else None
        if age >= 0:
            n_tick += 1
        if got is None:
            for c in ('a_p_sex', 'a_p_oral', 'a_p_noise'):
                f[ix[c]] = ''
            for c in M.YS_COLS:
                f[ix[c]] = ''
            f[ix['a_action']], f[ix['a_conf']] = '', '0.0000'
            continue
        n_hit += 1
        probs, ys = got
        for c, p in zip(('a_p_sex', 'a_p_oral', 'a_p_noise'), probs):
            f[ix[c]] = '%.4f' % p
        for c, v in zip(M.YS_COLS, ys):
            f[ix[c]] = '%.4f' % v
        if age > F.MAX_AGE_MS:              # 融合循环：结果超过 2 秒算过期，动作串清空
            act, conf = '', 0.0
        else:
            act, conf = classify(probs, params['AUDIO_DO'], params['AUDIO_NOISE'])
        f[ix['a_action']], f[ix['a_conf']] = act, '%.4f' % conf
    d = os.path.dirname(out_path)
    if d and not os.path.isdir(d):
        os.makedirs(d)
    with io.open(out_path, 'w', encoding='utf-8', newline='') as fh:
        for c in comments:
            fh.write(c + '\n')
        fh.write('# ' + note + '\n')
        fh.write('# 注意：a_p_* / a_action / a_conf / a_ys_* 是离线重算的；smoothed、gated、ble_*、sent* '
                 '仍是原 run 的。端到端指标请用 fusion_lab.py eval/oracle 重放\n')
        fh.write(','.join(cols) + '\n')
        w = csv.writer(fh, lineterminator='\n')
        for f in rows:
            w.writerow(f)
    return n_hit, n_tick


def _default_tag(a):
    tag = a.pipeline
    if os.path.abspath(a.classifier) != os.path.abspath(DEFAULT_CLASSIFIER):
        tag += '_' + os.path.splitext(os.path.basename(a.classifier))[0]
    if a.lead not in ('auto', '0'):
        tag += '_lead%s' % a.lead
    return tag


def cmd_rescore(a):
    _need_numpy()
    ffmpeg, ffprobe = find_ffmpeg(a.ffmpeg)
    pairs = resolve_pairs(a)
    model = AudioModel(a.yamnet, a.classifier, a.threads)
    align = load_alignment(a.alignment)
    tag = a.tag or _default_tag(a)
    out_dir = os.path.join(a.out, tag)
    ys_idx = [i for _, i, _ in M.YS_CLASSES]
    for vid, g, r, vpath in pairs:
        t0 = time.time()
        info = probe_audio(vpath, ffmpeg, ffprobe)
        meta, data, res = load_results(r)
        rec = align.get(vid, {})
        if a.offset is not None:
            delta, src = a.offset, '--offset'
        elif rec.get('device_pipeline') and rec.get(rec['device_pipeline'], {}).get('offset_ms') is not None:
            delta = rec[rec['device_pipeline']]['offset_ms']
            src = 'alignment.json（%s 管线对齐）' % rec['device_pipeline']
        else:
            print('  %s: alignment.json 里没有已确认的偏移，先用 phone 管线扫一遍' % vid)
            ph = build_pipeline('phone', vpath, info, ffmpeg, a.frame_size,
                                0 if a.lead == 'auto' else int(a.lead))
            delta, _, _ = scan_offset(ph, model, res, None, a.scan_n)
            del ph
            if delta is None:
                delta = 80
                src = '默认值（没有可比对的结果）'
            else:
                src = 'phone 管线现场扫描'
        if a.lead != 'auto':
            lead = int(a.lead)
        else:
            lead = int(rec.get('phone', {}).get('lead', 0)) if rec.get('device_pipeline') == 'phone' else 0
        pipe = build_pipeline(a.pipeline, vpath, info, ffmpeg, a.frame_size, lead)
        computed, ts_list, T_list, P, Y = {}, [], [], [], []
        for x in res:
            w = pipe.window(x.ref - delta)
            if w is None:
                continue
            p, s = model.run(w)
            computed[x.ts] = (p, s[ys_idx])
            ts_list.append(x.ts)
            T_list.append(int(round(x.ref - delta)))
            P.append(p)
            Y.append(s)
        out_path = os.path.join(out_dir, os.path.basename(r))
        note = ('rescored_by=audio_rescore.py pipeline=%s offset_ms=%d yamnet=%s classifier=%s lead=%d '
                'frame=%d at=%s' % (a.pipeline, delta, os.path.basename(model.yamnet_path),
                                    os.path.basename(model.classifier_path), lead,
                                    a.frame_size or info['frame'],
                                    datetime.datetime.now().strftime('%Y-%m-%dT%H:%M')))
        write_rescored(r, out_path, computed, note)
        np.savez_compressed(os.path.splitext(out_path)[0] + '.yamnet.npz',
                            ts=np.asarray(ts_list, np.int64), t_ms=np.asarray(T_list, np.int64),
                            probs=np.asarray(P, np.float32).reshape(-1, 3),
                            scores=np.asarray(Y, np.float16).reshape(-1, N_AUDIOSET))
        line = '  %-5s %s 管线，偏移 %+dms（%s）：%d/%d 个结果' % (
            vid, a.pipeline, delta, src, len(computed), len(res))
        if a.pipeline == 'phone':           # 复刻真机的管线 + 旧模型：应与真机记录一致
            pairs_ = [(computed[x.ts][0], x.logged) for x in _usable(res) if x.ts in computed]
            if pairs_:
                st = match_stats(np.asarray([p for p, _ in pairs_], np.float64),
                                 np.asarray([q for _, q in pairs_], np.float64))
                line += '，与真机记录 MAE %.3f' % st['mae']
        print(line + '  → %s（%.0f 秒）' % (out_path, time.time() - t0))
        del pipe
    print('\n下一步:\n'
          '  python bench/tools/audio_rescore.py auc %s %s\n'
          '  python bench/tools/fusion_lab.py eval %s %s'
          % (_rel(a.gt), _rel(out_dir), _rel(a.gt), _rel(out_dir)))


# ======================================================================== auc
def auc_np(pos, neg):
    """Mann-Whitney AUC（并列取平均秩），与 metrics._auc 一致，numpy 版给 521 类排行用。"""
    pos = np.asarray(pos, np.float64)
    neg = np.asarray(neg, np.float64)
    if not len(pos) or not len(neg):
        return float('nan')
    v = np.concatenate([pos, neg])
    order = np.argsort(v, kind='mergesort')
    sv = v[order]
    _, first, counts = np.unique(sv, return_index=True, return_counts=True)
    ranks = np.empty(len(v))
    ranks[order] = np.repeat(first + (counts + 1) / 2.0, counts)
    n1, n0 = len(pos), len(neg)
    return float((ranks[:n1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def result_rows(run_path, segs, policy):
    """逐个"新结果"取 (标签, a_p_*, a_ys_*, 结果时刻)。去重、过期、预热、跨边界的处理与
    fusion_lab.result_samples 相同，所以这里的 AUC 就是拟合证据表时面对的样本。"""
    meta, data = M._rows(run_path)
    out, last = [], None
    for d in data:
        age, wall, pos = F._inum(d.get('a_age'), -1), F._inum(d.get('wall_ms'), 0), F._inum(d.get('pos_ms'), 0)
        if age < 0:
            continue
        ts = wall - age
        if last is not None and ts <= last:
            continue
        last = ts
        ap = [F._fnum(d.get(c)) for c in ('a_p_sex', 'a_p_oral', 'a_p_noise')]
        if age > F.MAX_AGE_MS or any(x is None for x in ap) or sum(ap) <= 0.5:
            continue
        pr = pos - age
        if pr < M.WARMUP_MS:
            continue
        lab = F.clean_label(segs, pr - F.AUDIO_WIN_MS, pr, policy)
        if lab is None:
            continue
        ys = [F._fnum(d.get(c)) for c in M.YS_COLS]
        out.append({'lab': lab, 'ap': ap, 'ys': ys, 'ts': ts})
    return out


MIN_N = 5          # 每部片每类至少这么多个结果才算 AUC

BASE_FEATS = [('p_sex', '分类器 p_sex [证据融合现用]', lambda r: r['ap'][0]),
              ('p_sex+p_oral', '分类器 p_sex+p_oral', lambda r: r['ap'][0] + r['ap'][1]),
              ('p_oral', '分类器 p_oral', lambda r: r['ap'][1])]


def _feats(have_ys):
    fs = list(BASE_FEATS)
    if have_ys:
        for k, (short, idx, en) in enumerate(M.YS_CLASSES):
            fs.append((short, 'AudioSet %s #%d' % (en, idx),
                       lambda r, k=k: r['ys'][k]))
    return fs


def _auc_cell(x):
    return '  -  ' if x != x else '%.3f' % x


def cmd_auc(a):
    _need_numpy()
    pairs = resolve_pairs(a, need_video=False)
    per = OrderedDict()
    for vid, g, r, _ in pairs:
        gmeta, segs, dur, errs = M.read_gt(g)
        rows = result_rows(r, segs, a.policy)
        npz = os.path.splitext(r)[0] + '.yamnet.npz'
        if os.path.exists(npz):
            z = np.load(npz)
            idx = dict((int(t), i) for i, t in enumerate(z['ts']))
            sc = z['scores'].astype(np.float32)
            for x in rows:
                i = idx.get(x['ts'])
                x['all'] = sc[i] if i is not None else None
        per[vid] = rows
        meta, _ = M._rows(r)
        rs = meta.get('pipeline')
        print('  ✓ %-5s %-26s %4d 个结果（动作 %d / 剧情 %d）%s'
              % (vid, os.path.basename(r), len(rows), sum(1 for x in rows if x['lab'] == 'do'),
                 sum(1 for x in rows if x['lab'] == 'noise'),
                 ('  [重算: %s 管线]' % rs) if rs else '  [真机原始数据]'))
    vids = list(per)
    have_ys = any(x['ys'][0] is not None for rows in per.values() for x in rows)
    feats = _feats(have_ys)

    NAME_W = 38

    def table(title, cond=None):
        print('\n══ %s ══' % title)
        print('  AUC > 0.5：越高越像动作；< 0.5：越高越像剧情。判别力 = max(AUC, 1-AUC)；'
              '"最差片" = 各片里离 0.5 最近的那部；某部片任一类少于 %d 个样本时不算\n' % MIN_N)
        print('  ' + _pad('特征', NAME_W) + ''.join('%8s' % v for v in vids) + '    合并  判别力  方向   最差片')
        shown = 0
        for key, name, fn in feats:
            cells, P, N = [], [], []
            for v in vids:
                rows = [x for x in per[v] if cond is None or cond(x)]
                pos = [fn(x) for x in rows if x['lab'] == 'do' and fn(x) is not None]
                neg = [fn(x) for x in rows if x['lab'] == 'noise' and fn(x) is not None]
                ok = len(pos) >= MIN_N and len(neg) >= MIN_N
                cells.append(auc_np(pos, neg) if ok else float('nan'))
                if ok:
                    P += pos
                    N += neg
            if len(P) < MIN_N or len(N) < MIN_N:
                continue
            pooled = auc_np(P, N)
            power = max(pooled, 1 - pooled)
            same = [c if pooled >= 0.5 else 1 - c for c in cells if c == c]
            print('  ' + _pad(name, NAME_W) + ''.join('%8s' % _auc_cell(c) for c in cells)
                  + '   %.3f   %.3f  %s  %s' % (pooled, power, '↑转  ' if pooled >= 0.5 else '↑剧情',
                                               _auc_cell(min(same)) if same else '  -  '))
            shown += 1
        n_do = [sum(1 for x in per[v] if x['lab'] == 'do' and (cond is None or cond(x))) for v in vids]
        n_no = [sum(1 for x in per[v] if x['lab'] == 'noise' and (cond is None or cond(x))) for v in vids]
        print('  ' + _pad('样本数 动作/剧情', NAME_W) + ''.join('%8s' % ('%d/%d' % (d, n)) for d, n in zip(n_do, n_no)))
        if not shown:
            print('  （样本太少，算不出来）')

    table('音频特征的判别力（结果级）')
    table('只看分类器已经判"转"的结果（p_sex ≥ 0.5）：还能不能把剧情挑出来',
          cond=lambda x: x['ap'][0] >= 0.5)
    if not have_ys:
        print('\n  （这批 run 没有 a_ys_* 列：用 rescore 重算，或者用新版 app 重新采集）')

    for key in [k.strip() for k in (a.scan or '').split(',') if k.strip()]:
        fn = dict((k, f) for k, _, f in feats).get(key)
        if fn is None:
            continue
        pos = [fn(x) for v in vids for x in per[v] if x['lab'] == 'do' and fn(x) is not None]
        neg = [fn(x) for v in vids for x in per[v] if x['lab'] == 'noise' and fn(x) is not None]
        if not pos or not neg:
            continue
        print('\n══ 阈值扫描：%s ≥ 阈值 时当作"剧情"证据 ══' % key)
        print('  阈值   剧情里命中（越高越好）   动作里误伤（越低越好）')
        for th in (0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9):
            print('  %.2f        %5.1f%%                  %5.1f%%'
                  % (th, 100.0 * sum(1 for x in neg if x >= th) / len(neg),
                     100.0 * sum(1 for x in pos if x >= th) / len(pos)))

    if a.all_classes:
        have = [x for v in vids for x in per[v] if x.get('all') is not None]
        if not have:
            print('\n  --all-classes 需要 rescore 产出的 <run>.yamnet.npz（和 run CSV 放在一起）')
            return
        names = audioset_names()
        lab = np.asarray([x['lab'] == 'do' for x in have])
        S = np.stack([x['all'] for x in have])
        vid_of = np.asarray([v for v in vids for x in per[v] if x.get('all') is not None])
        ranks = []
        for c in range(N_AUDIOSET):
            pooled = auc_np(S[lab, c], S[~lab, c])
            if pooled != pooled:
                continue
            per_v = []
            for v in vids:
                m = vid_of == v
                per_v.append(auc_np(S[m & lab, c], S[m & ~lab, c]))
            same = [x if pooled >= 0.5 else 1 - x for x in per_v if x == x]
            ranks.append((min(same) if same else 0.5, pooled, c, per_v))
        print('\n══ AudioSet 521 类排行（按"最差片的判别力"排序，前 %d）══' % a.top)
        print('  只在某一两部片上有用的类会排在后面：要的是每部片都站得住的证据\n')
        print('  ' + _pad('类别', 40) + ''.join('%8s' % v for v in vids) + '    合并  方向')
        for worst, pooled, c, per_v in sorted(ranks, reverse=True)[:a.top]:
            print('  ' + _pad(('#%d %s' % (c, names[c]))[:38], 40) + ''.join('%8s' % _auc_cell(x) for x in per_v)
                  + '   %.3f  %s' % (pooled, '↑转' if pooled >= 0.5 else '↑剧情'))


# ======================================================================== selftest
def _java_reference(pcm, sr, ch, frame, Ts, lead=0, base_us=0):
    """把 AudioDecoder.startDecoding() 的循环和 PcmCircularBuffer 一行一行照抄成 Python（慢），
    用来校验 PhonePipeline 的向量化实现。Ts 必须递增：读之前先让"解码线程"跑到 T+3 秒。"""
    f32 = np.float32
    cap = SR * 20
    buf = np.zeros(cap, np.float32)
    tss = np.zeros(cap, np.int64)
    st = {'w': 0}

    def write(data, start_ms):
        for i in range(len(data)):
            idx = (st['w'] + i) % cap
            buf[idx] = data[i]
            tss[idx] = start_ms + (i * 1000 // SR)
        st['w'] = (st['w'] + len(data)) % cap

    def read(T, n):
        res = np.zeros(n, np.float32)
        cnt = 0
        i = 0
        while i < cap and cnt < n:
            idx = (st['w'] - 1 - i + cap) % cap
            ts = tss[idx]
            if 0 < ts <= T:
                res[n - cnt - 1] = buf[idx]
                cnt += 1
            i += 1
        return None if cnt < n * 0.9 else res

    pcm = _apply_lead(pcm, ch, lead)
    spb = frame * ch
    nbuf = -(-len(pcm) // spb)
    win = sr * 4
    abuf = np.zeros(win, np.float32)
    ot = [f32(i) / f32(sr) for i in range(win)]           # originalTime[i] = (float) i / sampleRate
    pos = 0
    wstart = -1
    b = 0
    out = []
    for T in Ts:
        while b < nbuf:
            pts_ms = jdiv(jdiv(b * frame * 1000000, sr) + base_us, 1000)
            if pts_ms > T + 3000:                         # 限速：解码器等播放追上来
                break
            if wstart < 0:
                wstart = pts_ms
            data = pcm[b * spb:(b + 1) * spb]
            # while (outputBuffer.remaining() >= 2 && bufferPos < windowSize) 逐个 getShort()；
            # 凑满后这个 buffer 剩下的部分随 releaseOutputBuffer 丢掉
            take = min(len(data), win - pos)
            abuf[pos:pos + take] = data[:take].astype(np.float32) / f32(32768)
            pos += take
            if pos == win:
                if sr != SR:
                    res = np.zeros(WIN, np.float32)
                    j = 0
                    for i in range(WIN):
                        t = f32(i) / f32(SR)
                        while j < pos - 1 and ot[j + 1] < t:
                            j += 1
                        if j < pos - 1:
                            ratio = (t - ot[j]) / (ot[j + 1] - ot[j])
                            res[i] = abuf[j] + ratio * (abuf[j + 1] - abuf[j])
                        else:
                            res[i] = abuf[j]
                    write(res, wstart)
                else:
                    write(abuf.copy(), wstart)
                pos = 0
                wstart = -1
            b += 1
        out.append(read(T, WIN))
    return out


def _synth_audio(seconds, sr, ch, seed=0):
    """带包络和音高变化的合成信号：两个声道不同，方便看出交错/半速的影响。"""
    rnd = np.random.RandomState(seed)
    n = int(seconds * sr)
    t = np.arange(n) / float(sr)
    f0 = 220 + 180 * np.sin(2 * np.pi * 0.13 * t)
    env = 0.5 + 0.5 * np.sin(2 * np.pi * 0.7 * t) ** 2
    chans = []
    for c in range(ch):
        ph = 2 * np.pi * np.cumsum(f0 * (1 + 0.5 * c)) / sr
        x = env * (0.4 * np.sin(ph) + 0.2 * np.sin(3 * ph)) + 0.05 * rnd.randn(n)
        bursts = (np.sin(2 * np.pi * (0.31 + 0.1 * c) * t) > 0.8)
        x = x + bursts * 0.3 * rnd.randn(n)
        chans.append(x)
    x = np.stack(chans, axis=1)
    return np.clip(np.round(x * 12000), -32768, 32767).astype(np.int16).reshape(-1)


def selftest():
    if np is None:
        print('selftest: 需要 numpy')
        return 1
    fails = []

    def check(cond, msg):
        if not cond:
            fails.append(msg)
            print('  FAIL ' + msg)
        return cond

    # 1. 向量化复刻 vs 逐行照抄的 Java（几种典型音轨，含负 pts 和时间戳重叠的情况）
    cases = [(44100, 2, 1024, 0, 0), (48000, 2, 1024, 0, 0), (44100, 1, 1024, 0, 0),
             (44100, 2, 1024, 2112, 0), (22050, 2, 2048, 0, 0), (16000, 2, 1024, 0, 0),
             (16000, 1, 1024, 0, 0), (44100, 2, 1024, 1024, 0)]
    n_windows = 0
    for sr, ch, frame, lead, base in cases:
        secs = 11
        pcm = _synth_audio(secs, sr, ch, seed=sr + ch)
        pipe = PhonePipeline(pcm, sr, ch, frame, lead, base)
        Ts = [1500, 1990, 2003, 2600, 4033, 5200, 6071, 8350, 9999, secs * 1000 + 5000]
        ref = _java_reference(pcm, sr, ch, frame, Ts, lead, base)
        bad = 0
        for T, rw in zip(Ts, ref):
            w = pipe.window(T)
            n_windows += rw is not None
            if (rw is None) != (w is None) or (w is not None and not np.array_equal(w, rw)):
                bad += 1
        check(bad == 0, 'PhonePipeline 与逐行 Java 不一致：%d Hz %d 声道 frame=%d lead=%d（%d/%d 个窗口）'
              % (sr, ch, frame, lead, bad, len(Ts)))
    if not fails:
        print('  ok   PhonePipeline 与逐行照抄的 Java 逐点一致（%d 种音轨，%d 个窗口）'
              % (len(cases), n_windows))

    # 2. 双声道 44.1k 的窗口确实是"半速"：每 2.02 秒一块，每块 32000 点 = 前 1 秒真实音频
    pcm = _synth_audio(12, 44100, 2, seed=1)
    p = PhonePipeline(pcm, 44100, 2)
    check(p.n_out == WIN and abs(p.block_start_ms[1] - p.block_start_ms[0] - 2020) <= 1,
          '双声道 44.1k 分块间隔应为 2020ms，实际 %s' % (p.block_start_ms[:3],))
    stereo = pcm.reshape(-1, 2).astype(np.float32) / 32768
    blk = p.S[:WIN]
    # 前 1 秒真实音频的交错序列按 44.1k 重采样到 16k，应与块内容一致（线性插值，误差很小）
    inter = stereo[:44100].reshape(-1)
    xs = np.arange(WIN) * 44100.0 / SR
    approx = np.interp(xs, np.arange(len(inter)), inter)
    # float32 时间轴带来的插值位置误差约 0.005 个样本，交错的 L/R 落差大，所以容差给 0.01
    check(np.abs(approx - blk).max() < 0.01 and np.corrcoef(approx, blk)[0, 1] > 0.9999,
          '双声道块内容不是"前 1 秒交错样本拉成 2 秒"')

    # 3. clean 管线取窗
    x = np.arange(SR * 10, dtype=np.float32)
    c = CleanPipeline(x)
    w = c.window(5000)
    check(w is not None and w[-1] == 5000 * 16 + 15 and w[0] == 5000 * 16 + 15 - WIN + 1,
          'CleanPipeline 窗口边界不对')
    check(c.window(1000) is None, 'CleanPipeline 样本不足 90% 时应返回 None')

    # 4. rescore 写出的 CSV：metrics / fusion_lab 都读得回来，阈值重判正确
    import tempfile
    tmpd = tempfile.mkdtemp(prefix='xcup_rescore_')
    src = os.path.join(tmpd, 'v01_base.csv')
    rows = []
    with io.open(src, 'w', encoding='utf-8', newline='') as fh:
        fh.write('# xcup-bench-run v1\n# video_name=selftest\n'
                 '# params: AUDIO_DO=0.55 AUDIO_NOISE=0.60\n')
        base_cols = [c for c in M.RUN_COLS_NEEDED if c not in M.YS_COLS]   # 旧版 dump 没有 a_ys_*
        fh.write(','.join(base_cols) + '\n')
        for k in range(40):
            wall = 1000000 + 500 * k
            age = (k % 2) * 500 + 30 if k > 1 else -1
            rows.append((wall, age))
            vals = dict((c, '') for c in base_cols)
            vals.update({'pos_ms': str(500 * k), 'wall_ms': str(wall), 'a_age': str(age),
                         'a_p_sex': '0.3000', 'a_p_oral': '0.3000', 'a_p_noise': '0.4000',
                         'a_action': 'do', 'a_conf': '0.6000', 'sent': '0', 'sent_level': '-1'})
            fh.write(','.join(vals[c] for c in base_cols) + '\n')
    computed = {}
    for k, (wall, age) in enumerate(rows):
        if age >= 0:
            ts = wall - age
            q = (k % 3) / 2.0
            computed[ts] = (np.asarray([0.9 * q, 0.05, 1 - 0.9 * q - 0.05], np.float32),
                            np.full(len(M.YS_COLS), 0.25, np.float32))
    dst = os.path.join(tmpd, 'out', 'v01_base.csv')
    write_rescored(src, dst, computed, 'rescored_by=selftest pipeline=clean')
    meta, rr = M.read_run(dst)
    check(meta.get('pipeline') == 'clean', 'rescore 头部没写 pipeline')
    ok = True
    for (wall, age), r in zip(rows, rr):
        if age < 0:
            ok &= r['ap'][0] == -1 and r['a'] == ''
            continue
        p = computed[wall - age][0]
        want = classify(p, 0.55, 0.60)[0]
        ok &= abs(r['ap'][0] - p[0]) < 1e-4 and r['a'] == M._norm(want) and r['ys'][0] == 0.25
    check(ok, 'rescore 的 CSV 读回来概率/动作/a_ys_* 不对')
    _, ticks = F.load_run(dst)
    check(len(ticks) == len(rows) and ticks[5].ap is not None, 'fusion_lab 读不了 rescore 的 CSV')
    shutil.rmtree(tmpd, ignore_errors=True)

    # 5. AUC：numpy 版与 metrics._auc 一致
    rnd = np.random.RandomState(3)
    pos = list(np.round(rnd.rand(200), 2))
    neg = list(np.round(rnd.rand(150) * 0.8, 2))
    check(abs(auc_np(pos, neg) - M._auc(pos, neg)) < 1e-12, 'auc_np 与 metrics._auc 不一致')

    # 6. 端到端（有 TFLite 解释器和模型时）：合成一份"真机" run，check 要能找回偏移、认出 phone 管线
    try:
        model = AudioModel()
    except SystemExit as e:
        print('  skip 端到端对齐测试（%s）' % str(e).splitlines()[0])
        model = None
    if model is not None:
        secs = 40
        rnd = np.random.RandomState(7)
        mono16 = _synth_audio(secs, SR, 1, seed=11).astype(np.float32) / 32768
        # 44.1k 双声道版本（两个声道相同 = 正常的立体声内容），供 phone 管线用
        t44 = np.arange(int(secs * 44100)) * (SR / 44100.0)
        x44 = np.interp(t44, np.arange(len(mono16)), mono16)
        pcm = np.clip(np.round(np.repeat(x44[:, None], 2, axis=1) * 32767), -32768, 32767).astype(np.int16)
        phone = PhonePipeline(pcm.reshape(-1), 44100, 2)
        clean = CleanPipeline(mono16)
        true_delta = 90
        res = []
        for k in range(4, secs - 1):
            T = k * 1000 + rnd.randint(0, 200)          # 真机读窗的播放位置
            w = phone.window(T)
            if w is None:
                continue
            pr = tuple(float('%.4f' % x) for x in model.run(w)[0])     # CSV 只存 4 位小数
            lat = true_delta + rnd.randint(-3, 4)       # 推理耗时的抖动（真机 run 里约 ±3ms）
            res.append(Result(1000000 + T + lat, T + lat, pr, True, []))
        best, _, _ = scan_offset(phone, model, res, (-60, 240, 3), 30)
        n1, o1, _ = compare(phone, model, res, best)
        bc, _, _ = scan_offset(clean, model, res, (-60, 240, 3), 30)
        n2, o2, _ = compare(clean, model, res, bc)
        check(abs(best - true_delta) <= 30, '端到端：找回的偏移 %d，应接近 %d' % (best, true_delta))
        check(_mae(n1, o1) < GOOD_MAE, '端到端：phone 管线复现误差 %.3f 偏大' % _mae(n1, o1))
        check(_mae(n2, o2) > 2 * _mae(n1, o1), '端到端：clean 管线不该和 phone 一样好（%.3f vs %.3f）'
              % (_mae(n2, o2), _mae(n1, o1)))
        if not fails:
            print('  ok   端到端：偏移 %+dms（真值 %+d），phone MAE %.4f，clean MAE %.4f'
                  % (best, true_delta, _mae(n1, o1), _mae(n2, o2)))

    print('selftest: %s' % ('FAILED (%d)' % len(fails) if fails else 'ALL PASS'))
    return 1 if fails else 0


# ======================================================================== main
def main():
    if hasattr(sys.stdout, 'reconfigure'):
        try:
            sys.stdout.reconfigure(errors='replace')      # 输出重定向到 GBK 文件时不因 ✓ 之类崩掉
        except (AttributeError, ValueError):
            pass
    ap = argparse.ArgumentParser(description='离线音频重打分：在 run CSV 的音频结果时刻重算 a_p_*')
    ap.add_argument('--selftest', action='store_true')
    sub = ap.add_subparsers(dest='cmd')

    def common(p, videos=True):
        p.add_argument('--gt', default=os.path.join(ROOT, 'bench', 'gt'), help='真值目录（默认 bench/gt）')
        p.add_argument('--runs', default=os.path.join(ROOT, 'bench', 'runs'),
                       help='run CSV 目录（默认 bench/runs，真机原始 dump）')
        p.add_argument('--only', help='只处理这些片子，如 v01,v04')
        if videos:
            p.add_argument('--videos', default=DEFAULT_VIDEOS, help='视频所在目录（默认 bench/videos）')
            p.add_argument('--pair', action='append', help='逐个指定视频：v01=D:/xcup/视频1.mp4（可重复）')
            p.add_argument('--ffmpeg', help='ffmpeg 可执行文件（默认从 PATH 找；ffprobe 取同目录）')
            p.add_argument('--yamnet', default=DEFAULT_YAMNET)
            p.add_argument('--classifier', default=DEFAULT_CLASSIFIER, help='分类器 tflite（默认 app 里的）')
            p.add_argument('--threads', type=int, default=0, help='推理线程数（默认 min(4, CPU 数)）')
            p.add_argument('--frame-size', type=int, default=0,
                           help='每个解码输出 buffer 的帧数（默认按 ffprobe：AAC-LC 1024）')
            p.add_argument('--lead', default='auto',
                           help='真机解码流比 ffmpeg 多出的开头样本数（AAC priming，如 1024/2112）。'
                                '默认 auto：check 自动试，rescore 用 check 的结果')
            p.add_argument('--alignment', default=os.path.join(DEFAULT_OUT, 'alignment.json'))
            p.add_argument('--scan-n', type=int, default=SCAN_N, help='扫描偏移时用多少个结果（默认 60）')
            p.add_argument('--out', default=DEFAULT_OUT, help='输出根目录（默认 bench/rescored）')

    p = sub.add_parser('check', help='用旧模型复现真机的 a_p_*，验证对齐，判断真机喂的是哪种音频')
    common(p)
    p.add_argument('--pipeline', default='both', choices=('both', 'phone', 'clean'))
    p.add_argument('--scan', help='偏移扫描范围 起:止:步长（ms）。默认旧版 run %d:%d:%d，有 a_pos 的 run %d:%d:%d'
                   % (SCAN_RANGE + SCAN_RANGE_POS))
    p.add_argument('--dump', action='store_true', help='把逐结果的对比写成 bench/rescored/check_<id>.csv')
    p.add_argument('--verbose', action='store_true', help='打印偏移扫描曲线')

    p = sub.add_parser('rescore', help='重算 a_p_* 和 a_ys_*，写出新的 run CSV')
    common(p)
    p.add_argument('--pipeline', default='clean', choices=('phone', 'clean'),
                   help='phone=复刻真机现在的解码；clean=正确的单声道 16kHz（默认）')
    p.add_argument('--offset', type=int, default=None,
                   help='时间偏移 ms（默认取 alignment.json 里 check 找到的）')
    p.add_argument('--tag', help='输出子目录名（默认按管线/模型自动起，如 clean）')

    p = sub.add_parser('auc', help='跨片子的音频判别力（AUC）')
    p.add_argument('gt', nargs='?', default=os.path.join(ROOT, 'bench', 'gt'))
    p.add_argument('runs', nargs='?', default=os.path.join(ROOT, 'bench', 'runs'))
    p.add_argument('--only', help='只看这些片子，如 v01,v04')
    p.add_argument('--policy', default='exclude', choices=('exclude', 'do', 'noise'))
    p.add_argument('--scan', default='speech', help='对这些特征做阈值扫描，逗号分隔（默认 speech）')
    p.add_argument('--all-classes', action='store_true', help='AudioSet 521 类排行（需要 .yamnet.npz）')
    p.add_argument('--top', type=int, default=20)

    a = ap.parse_args()
    if a.selftest:
        sys.exit(selftest())
    if a.cmd == 'check':
        a.pair = a.pair or []
        cmd_check(a)
    elif a.cmd == 'rescore':
        a.pair = a.pair or []
        cmd_rescore(a)
    elif a.cmd == 'auc':
        a.pair = []
        cmd_auc(a)
    else:
        ap.print_help()


if __name__ == '__main__':
    main()
