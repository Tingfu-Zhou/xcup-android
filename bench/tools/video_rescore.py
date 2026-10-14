#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
bench/tools/video_rescore.py — 离线视频重算

run CSV 记下了真机每个视频结果是哪一刻出的（wall_ms - v_age）。有原视频，就能在电脑上把那 12 帧还原出来、重新跑视频
模型：换抽帧方式、换预处理、换模型，都不用再上真机从头播一遍（音频那边的 audio_rescore.py 同理）。

“喂给模型的画面”由两部分组成，管线名 = 抽帧_预处理：
  抽帧
    keyframe  2026-10-14 之前的 VideoFrameExtractor：seekTo(T, SEEK_TO_CLOSEST_SYNC) 后直接取解码器吐出的第一帧，也就是离 T
              最近的关键帧（可能在 T 之后）。关键帧间隔 4~12 秒的片子，12 帧窗口里只有 1~2 张不同的图
    exact     修复后的 VideoFrameExtractor：从 T 之前的关键帧往后解，取 pts 不晚于 T + 半帧的最后一帧（离 T 最近的那一帧）
  预处理
    legacy    2026-10-14 之前的 EGLRenderer：整帧用 GPU 双线性（GL_LINEAR）一步采样到 160×160（16:9 被压扁，有混叠）
    crop      修复后的离线模式：居中裁正方形 → GPU 双线性画到 672×672 → 面积平均缩到 224 → 再缩到 160（与训练一致）
    squash    同 crop 但不裁、整帧压扁（FramePreprocessor.CENTER_CROP=false 时 app 的做法，A/B 对比用）
    train     训练素材的做法（make_movinet_clips.py + train_mobilenetv3_tap_3class.py）的近似：ffmpeg 同样的双三次缩放和
              居中裁剪到 224 → BT.601 转 RGB（OpenCV 读帧的默认）→ INTER_AREA 缩到 160。不含 x264 重编码。只作参照
  例：keyframe_legacy = 2026-10-14 之前的 app；exact_crop = 修复后的 app。

子命令:
  keyframes  列出每部片的关键帧间隔（GOP）。keyframe 管线下，视频结果每个 GOP 只变一次
  check      在 run 记录的每个视频结果上，按指定管线重算 v_p_*，和真机记下的比，看哪条管线能复现真机。
             旧 run（没有 v_fpts 列）默认比 keyframe_legacy：没记每帧的时刻，从出结果时刻往前推、扫一个时间差 Δ；
             新 run 默认比参数行 VIDEO_PIPE 那条，按 v_fpts 记下的 12 帧 pts 取帧（只扫 PyAV 与手机时间戳之间的常数差）。
             结果写进 bench/rescored/video_alignment.json，rescore 直接用其中的 Δ
  rescore    按指定管线（默认 exact_crop）重算每个视频结果，写出新的 run CSV：v_p_* 换成新概率，v_action / v_conf 按 run
             参数行的阈值重判，v_pos / v_fpts 写成重算用的帧时刻。写到 bench/rescored/<tag>/，文件名不变，metrics.py、
             fusion_lab.py eval/oracle、arbiter_lab.py 直接能读（fusion_lab check 除外：决策列还是原 run 的）。
             同时写 <run>.vframes.npz：用到的每一帧的 pts 和逐帧 logit（模型是“逐帧分类 + 12 帧 logit 平均”，
             有了逐帧 logit，换窗口长度、换聚合方式都不用再跑模型）
  golden     重新生成 Java 端 FramePreprocessorTest 的对拍数据（改了 FramePreprocessor 才需要）

用法（Windows 把 python3 换成 python）:
  python3 bench/tools/video_rescore.py keyframes --videos D:/xcup/videos
  python3 bench/tools/video_rescore.py check   --videos D:/xcup/videos --runs bench/runs_test --only v08
  python3 bench/tools/video_rescore.py rescore --videos D:/xcup/videos --runs bench/runs_initial                # exact_crop
  python3 bench/tools/video_rescore.py rescore --videos D:/xcup/videos --runs bench/runs_initial --pipe keyframe_crop
  python3 bench/tools/video_rescore.py --selftest

视频的查找与 audio_rescore.py 相同：--videos 目录下按 真值文件头的 file_name → 真值 id → run 的 video_name 找；
也可以 --pair v01=D:/xcup/视频1.mp4 逐个指定。视频不进仓库。

依赖：numpy；PyAV（pip install av，自带 FFmpeg）；TFLite 解释器（pip install ai-edge-litert，或已装的 tensorflow）。
建议再装 opencv-python-headless：GPU 双线性用 cv2.resize 算会快很多（没有就用 numpy，结果几乎一样）。
"""
import argparse
import bisect
import csv
import glob
import io
import json
import math
import os
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import metrics as M          # noqa: E402
import fusion_lab as F       # noqa: E402
import audio_rescore as A    # noqa: E402  视频查找、TFLite 解释器

try:
    import numpy as np
except ImportError:          # 只有 --help 能在没有 numpy 时跑
    np = None
try:
    import av
except ImportError:
    av = None
try:
    import cv2
except ImportError:
    cv2 = None

TOOLS = os.path.dirname(os.path.abspath(__file__))
ROOT = F.ROOT
ASSETS = os.path.join(ROOT, 'app', 'src', 'main', 'assets')
DEFAULT_MODEL = os.path.join(ASSETS, 'mobilenetv3small_tap_3class_float32.tflite')
DEFAULT_OUT = os.path.join(ROOT, 'bench', 'rescored')
DEFAULT_VIDEOS = os.path.join(ROOT, 'bench', 'videos')
DEFAULT_ALIGN = os.path.join(DEFAULT_OUT, 'video_alignment.json')
GOLDEN_PATH = os.path.join(ROOT, 'app', 'src', 'test', 'resources', 'video', 'preprocess_golden.txt')

MODEL_SIZE = 160         # FramePreprocessor.MODEL_SIZE / VideoClassifierHelper.INPUT_SIZE
CLIP_SIZE = 224          # FramePreprocessor.CLIP_SIZE：训练片段 224×224
STAGE_SIZE = 672         # VideoFrameExtractor.STAGE_SIZE：离线模式 GPU 先画到 672×672（= 3×224）
NUM_FRAMES = 12          # VideoClassifierHelper.NUM_FRAMES：12 帧、间隔 250ms
FRAME_STEP_MS = 250
FRAMES = ('keyframe', 'exact')
PREPS = ('legacy', 'crop', 'squash', 'train')
MATRICES = ('auto', 'bt601', 'bt709')

# 旧 run 没记每帧的时刻：最后一帧的取帧目标比出结果时的播放位置早 Δ 毫秒，往前每 250ms 一帧。
# 旧版抽帧除了只取关键帧，还会晚好几次请求：每次先 seek、只喂一帧就取最早的一个输出，解码器流水线里积压的是前几次请求的帧。
# runs_test 上从“冻住→变化”的时间点推出 Δ ≈ 0.5~1.3 秒（v10 525ms、v08 875ms、v11 约 900ms、v09 约 1.3 秒），所以扫到 3 秒
DELTA_SCAN = (0, 3000, 25)
# 修复后的 app：抽帧线程取“此刻的那一帧”，再加推理，出结果比最后一帧晚约 150~250ms。旧 run 按 exact 管线重算（rescore）时用它，
# 不用 check 扫出来的旧 Δ（那里面是旧解码器的积压，新 app 没有）
NEW_APP_DELTA = 200
# 新 run 的 v_fpts 是手机 MediaExtractor 的 pts；PyAV 的 pts 可能差一个常数（编辑列表、起始时间），按帧扫
OFFSET_FRAMES = (-2, -1, 0, 1, 2)
SCAN_N = 300             # 扫描时间差只用这么多个结果（均匀抽）
GOOD_MAE, OK_MAE = 0.03, 0.08


def _need(numpy=True, pyav=False):
    if numpy and np is None:
        sys.exit('需要 numpy：pip install numpy av ai-edge-litert')
    if pyav and av is None:
        sys.exit('需要 PyAV：pip install av')


# ======================================================================== 预处理（FramePreprocessor 的逐行移植）
# 像素用 uint8 [H, W, 3] RGB。面积缩小与 FramePreprocessor.java 同一套权重、同样的 float32 累加顺序，两边逐位相同
# （FramePreprocessorTest 用 golden 生成的对拍数据校验）；与 cv2.INTER_AREA 只在 2 倍整数缩小时有 ±1（OpenCV 那条路径
# 用 SIMD，舍入方式不同），其余逐位相同（--selftest 校验）。

def crop_rect(w, h, crop=True):
    """FramePreprocessor.cropRect：居中正方形 (x0, y0, w, h)；crop=False 时整帧。"""
    if not crop:
        return (0, 0, w, h)
    side = min(w, h)
    return ((w - side) // 2, (h - side) // 2, side, side)


def area_tab(ssize, dsize):
    """OpenCV computeResizeAreaTab（FramePreprocessor.areaTab）：第 k 项 dst[di] += src[si] * alpha。"""
    scale = 1.0 / (dsize / ssize)
    di, si, al = [], [], []
    for dx in range(dsize):
        fsx1 = dx * scale
        fsx2 = fsx1 + scale
        cell = min(scale, ssize - fsx1)
        sx1, sx2 = math.ceil(fsx1), math.floor(fsx2)
        sx2 = min(sx2, ssize - 1)
        sx1 = min(sx1, sx2)
        if sx1 - fsx1 > 1e-3:
            di.append(dx); si.append(sx1 - 1); al.append((sx1 - fsx1) / cell)
        for sx in range(sx1, sx2):
            di.append(dx); si.append(sx); al.append(1.0 / cell)
        if fsx2 - sx2 > 1e-3:
            di.append(dx); si.append(sx2); al.append(min(min(fsx2 - sx2, 1.0), cell) / cell)
    return di, si, [np.float32(a) for a in al]


def _sat(x):
    return np.clip(np.rint(x), 0, 255).astype(np.uint8)


def _area(img, dw, dh):
    sh, sw = img.shape[:2]
    xdi, xsi, xal = area_tab(sw, dw)
    ydi, ysi, yal = area_tab(sh, dh)
    src = img.astype(np.float32)
    buf = np.zeros((sh, dw, 3), np.float32)
    for d, s, a in zip(xdi, xsi, xal):              # 横向：每一项按顺序 float32 累加（与 Java 相同）
        buf[:, d, :] = buf[:, d, :] + src[:, s, :] * a
    out = np.empty((dh, dw, 3), np.float32)
    prev = None
    for d, s, b in zip(ydi, ysi, yal):              # 纵向：同一行的各项按顺序累加
        term = b * buf[s]
        if d != prev:
            out[d] = term
            prev = d
        else:
            out[d] = out[d] + term
    return _sat(out)


def _area_int(img, dw, dh):
    sh, sw = img.shape[:2]
    fx, fy = sw // dw, sh // dh
    s = img.reshape(dh, fy, dw, fx, 3).astype(np.int32).sum(axis=(1, 3))
    scale = np.float32(1.0) / np.float32(fx * fy)
    return _sat(s.astype(np.float32) * scale)


def _bilinear(img, dw, dh):
    sh, sw = img.shape[:2]
    fy = (np.arange(dh) + 0.5) * (sh / dh) - 0.5
    fx = (np.arange(dw) + 0.5) * (sw / dw) - 0.5
    ya, xa = np.floor(fy).astype(np.int64), np.floor(fx).astype(np.int64)
    wy, wx = (fy - ya)[:, None, None], (fx - xa)[None, :, None]
    y1, y2 = np.clip(ya, 0, sh - 1), np.clip(ya + 1, 0, sh - 1)
    x1, x2 = np.clip(xa, 0, sw - 1), np.clip(xa + 1, 0, sw - 1)
    p = img.astype(np.float64)
    top = (1 - wx) * p[y1][:, x1] + wx * p[y1][:, x2]
    bot = (1 - wx) * p[y2][:, x1] + wx * p[y2][:, x2]
    return _sat((1 - wy) * top + wy * bot)


def resize(img, dw, dh):
    """FramePreprocessor.resize：两个方向都缩小用面积平均（整数倍走整块求和），否则双线性。"""
    sh, sw = img.shape[:2]
    if (sw, sh) == (dw, dh):
        return img.copy()
    if sw >= dw and sh >= dh:
        if sw % dw == 0 and sh % dh == 0:
            return _area_int(img, dw, dh)
        return _area(img, dw, dh)
    return _bilinear(img, dw, dh)


def to_model_input(img, region=None):
    """FramePreprocessor.toModelInput：取 region (x0, y0, w, h) → 224×224 → 160×160。"""
    if region is not None:
        x0, y0, w, h = region
        img = img[y0:y0 + h, x0:x0 + w]
    return resize(resize(img, CLIP_SIZE, CLIP_SIZE), MODEL_SIZE, MODEL_SIZE)


# ======================================================================== 手机 GPU 那一步的模拟
def gl_linear(img, region, ow, oh):
    """GL_LINEAR 采样：输出像素中心映射回源图（像素中心对齐）、取 2×2 双线性，超出边界夹到边上。
    region=(x0, y0, w, h) 是画到输出上的那块。缩小倍数大时和 GPU 一样不抗混叠（只用到 4 个源像素）。
    有 cv2 时用 cv2.resize(INTER_LINEAR)（同一种采样，权重是定点数，个别像素差 1）。"""
    x0, y0, w, h = region
    if cv2 is not None:
        sub = np.ascontiguousarray(img[y0:y0 + h, x0:x0 + w])
        return cv2.resize(sub, (ow, oh), interpolation=cv2.INTER_LINEAR)
    H, W = img.shape[:2]
    ys = y0 + (np.arange(oh) + 0.5) * (h / oh) - 0.5
    xs = x0 + (np.arange(ow) + 0.5) * (w / ow) - 0.5
    ya, xa = np.floor(ys).astype(np.int64), np.floor(xs).astype(np.int64)
    wy = (ys - ya).astype(np.float32)[:, None, None]
    wx = (xs - xa).astype(np.float32)[None, :, None]
    y1, y2 = np.clip(ya, 0, H - 1), np.clip(ya + 1, 0, H - 1)
    x1, x2 = np.clip(xa, 0, W - 1), np.clip(xa + 1, 0, W - 1)
    f = img.astype(np.float32)
    top = f[y1][:, x1] * (1 - wx) + f[y1][:, x2] * wx
    bot = f[y2][:, x1] * (1 - wx) + f[y2][:, x2] * wx
    return _sat(top * (1 - wy) + bot * wy)


def _colorspace(frame, matrix):
    """YUV→RGB 用哪个矩阵。auto：码流标了就按标的；没标时与 Android 一样，高清（≥720 行）按 BT.709，标清按 BT.601。"""
    if matrix == 'bt601':
        return 'ITU601'
    if matrix == 'bt709':
        return 'ITU709'
    cs = int(getattr(frame, 'colorspace', 2) or 2)
    if cs == 1:
        return 'ITU709'
    if cs in (5, 6):
        return 'ITU601'
    return 'ITU709' if min(frame.width, frame.height) >= 720 else 'ITU601'


def _upright(a, rotation):
    """按显示矩阵把画面转正（手机上解码器按 KEY_ROTATION 转正，变换矩阵里已经包含）。"""
    k = int(round((rotation or 0) / 90.0)) % 4
    return np.ascontiguousarray(np.rot90(a, k)) if k else a


def ffmpeg_increase(w, h, size):
    """ffmpeg scale=size:size:force_original_aspect_ratio=increase 的输出尺寸（av_rescale 四舍五入）。"""
    tw = int(math.floor(size * w / h + 0.5))
    th = int(math.floor(size * h / w + 0.5))
    return max(size, tw), max(size, th)


def prep_frame(frame, preps, matrix):
    """一帧 → {预处理名: 160×160×3 uint8}。frame 是 PyAV 的 VideoFrame。"""
    out = {}
    rot = getattr(frame, 'rotation', 0) or 0
    gpu = [p for p in preps if p != 'train']
    if gpu:
        rgb = _upright(frame.to_ndarray(format='rgb24', src_colorspace=_colorspace(frame, matrix)), rot)
        H, W = rgb.shape[:2]
        for p in gpu:
            if p == 'legacy':
                out[p] = gl_linear(rgb, (0, 0, W, H), MODEL_SIZE, MODEL_SIZE)
            else:
                region = crop_rect(W, H, p == 'crop')
                out[p] = to_model_input(gl_linear(rgb, region, STAGE_SIZE, STAGE_SIZE))
    if 'train' in preps:
        # 训练：ffmpeg 在 YUV 上双三次缩放到短边 224 再居中裁（裁剪起点对齐到色度采样，取偶数），x264 编码后
        # OpenCV 读回来按 BT.601 转 BGR，再 INTER_AREA 缩到 160
        W, H = frame.width, frame.height
        if int(round(rot / 90.0)) % 2:
            W, H = H, W
        w2, h2 = ffmpeg_increase(W, H, CLIP_SIZE)
        cw, ch = (h2, w2) if int(round(rot / 90.0)) % 2 else (w2, h2)
        a = _upright(frame.to_ndarray(format='rgb24', width=cw, height=ch, interpolation='BICUBIC',
                                      src_colorspace='ITU601'), rot)
        x0 = ((w2 - CLIP_SIZE) // 2) & ~1
        y0 = ((h2 - CLIP_SIZE) // 2) & ~1
        out['train'] = resize(a[y0:y0 + CLIP_SIZE, x0:x0 + CLIP_SIZE], MODEL_SIZE, MODEL_SIZE)
    return out


# ======================================================================== 视频
class VideoSource(object):
    """一部片：帧时刻表（只解封装、不解码，很快）+ 按需解码。时间统一用整数微秒（与 MediaExtractor 相同）。"""

    def __init__(self, path):
        _need(pyav=True)
        self.path = path
        with av.open(path) as c:
            s = c.streams.video[0]
            tb = s.time_base
            self.width, self.height = s.codec_context.width, s.codec_context.height
            rate = s.average_rate or s.guessed_rate
            self.fps = float(rate) if rate else 30.0
            pts, keys = [], []
            for p in c.demux(s):
                t = p.pts if p.pts is not None else p.dts
                if t is None or p.size == 0:
                    continue
                us = int(round(float(t * tb) * 1e6))
                pts.append(us)
                if p.is_keyframe:
                    keys.append(us)
        self.frames_us = sorted(set(pts))
        self.keys_us = sorted(set(keys))
        self.key_set = set(self.keys_us)
        self.frame_dur_us = int(round(1e6 / self.fps)) if self.fps > 0 else 33333
        if not self.frames_us:
            raise RuntimeError('%s 里没有视频帧' % path)

    def gop_stats(self):
        k = self.keys_us
        gaps = np.diff(k) / 1e6 if len(k) > 1 else np.array([0.0])
        dur = (self.frames_us[-1] - self.frames_us[0]) / 1e6
        frozen = float(np.sum(np.maximum(gaps - NUM_FRAMES * FRAME_STEP_MS / 1000.0, 0))) / max(dur, 1e-9)
        return dict(n_keys=len(k), median=float(np.median(gaps)), p90=float(np.percentile(gaps, 90)),
                    max=float(gaps.max()), duration=dur, frozen=frozen)

    def keyframe_at(self, t_us):
        """旧 VideoFrameExtractor：离 t 最近的关键帧（一样近取前一个）。"""
        k = self.keys_us
        i = bisect.bisect_left(k, t_us)
        best = None
        for j in (i - 1, i):
            if 0 <= j < len(k) and (best is None or abs(k[j] - t_us) < abs(best - t_us)):
                best = k[j]
        return best

    def exact_at(self, t_us):
        """新 VideoFrameExtractor：pts 不晚于 t + 半帧的最后一帧；t 早于第一帧时取第一帧。"""
        i = bisect.bisect_right(self.frames_us, t_us + self.frame_dur_us // 2) - 1
        return self.frames_us[max(i, 0)]

    def decode(self, wanted_us, keyframes_only=False, progress=None):
        """按时间顺序解码，只把 pts 在 wanted_us 里的帧交出来：→ (pts_us, av.VideoFrame)。"""
        wanted = set(wanted_us)
        if not wanted:
            return
        last = max(wanted)
        n_total, n_done, t0 = len(self.frames_us), 0, time.time()
        with av.open(self.path) as c:
            s = c.streams.video[0]
            s.thread_type = 'AUTO'
            if keyframes_only:
                s.codec_context.skip_frame = 'NONKEY'
            for f in c.decode(s):
                n_done += 1
                if f.pts is None:
                    continue
                us = int(round(float(f.pts * f.time_base) * 1e6))
                if us in wanted:
                    yield us, f
                if us >= last:
                    break
                if progress and not keyframes_only and n_done % 3000 == 0:
                    progress('解码 %d/%d 帧（%.0f 帧/秒）' % (n_done, n_total, n_done / max(time.time() - t0, 1e-6)))


# ======================================================================== 模型
class VideoModel(object):
    """app 的视频模型。它其实是“12 帧各自过 MobileNetV3Small → 576 维特征取平均 → 线性层 → softmax”，平均和线性层可以交换，
    所以窗口输出 = softmax(12 帧各自 logit 的平均)。这里读出每帧的 576 维特征（模型里 PACK 算子的 12 个输入），
    自己乘线性层得到逐帧 logit：12 张不同的帧一次推理就拿到 12 个 logit，窗口随便怎么组合都不用再跑模型。"""

    def __init__(self, path=DEFAULT_MODEL, threads=None):
        _need()
        if not os.path.exists(path):
            sys.exit('找不到模型文件: %s' % path)
        Interp = A._interpreter_class()
        threads = threads or min(4, os.cpu_count() or 1)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            self.it = Interp(model_path=path, num_threads=threads, experimental_preserve_all_tensors=True)
        self.it.allocate_tensors()
        inp = self.it.get_input_details()[0]
        if list(inp['shape']) != [1, NUM_FRAMES, MODEL_SIZE, MODEL_SIZE, 3]:
            sys.exit('%s 的输入是 %s，不是 [1, 12, 160, 160, 3]' % (path, list(inp['shape'])))
        self.inp = inp['index']
        self.out = self.it.get_output_details()[0]['index']
        ops = self.it._get_ops_details()
        packs = [o for o in ops if o['op_name'] == 'PACK']
        fcs = [o for o in ops if o['op_name'] == 'FULLY_CONNECTED']
        if len(packs) != 1 or len(fcs) != 1 or len(packs[0]['inputs']) != NUM_FRAMES:
            sys.exit('%s 不是“逐帧特征 + 时间平均 + 线性层”的结构，逐帧 logit 读不出来' % path)
        self.feat_idx = [int(i) for i in packs[0]['inputs']]
        fc = fcs[0]['inputs']
        self.W = self.it.get_tensor(int(fc[1])).astype(np.float32)
        self.b = self.it.get_tensor(int(fc[2])).astype(np.float32) if len(fc) > 2 and fc[2] >= 0 \
            else np.zeros(self.W.shape[0], np.float32)
        self.calls = 0

    def run_window(self, frames):
        """12 帧（160×160×3 uint8）→ (窗口概率 float32[3], 逐帧 logit float32[12, 3])。"""
        x = np.stack(frames).astype(np.float32)[None]
        self.it.set_tensor(self.inp, x)
        self.it.invoke()
        self.calls += 1
        probs = self.it.get_tensor(self.out)[0].astype(np.float32)
        feats = np.stack([self.it.get_tensor(i)[0] for i in self.feat_idx]).astype(np.float32)
        return probs, feats @ self.W.T + self.b

    def frame_logits(self, frames):
        """任意张帧 → 逐帧 logit [n, 3]：每 12 张拼成一次推理（不足 12 张补最后一张）。"""
        out = []
        for i in range(0, len(frames), NUM_FRAMES):
            chunk = list(frames[i:i + NUM_FRAMES])
            n = len(chunk)
            chunk += [chunk[-1]] * (NUM_FRAMES - n)
            out.append(self.run_window(chunk)[1][:n])
        return np.concatenate(out) if out else np.zeros((0, 3), np.float32)


def _make_model(path):
    """check / rescore 用的模型（自检时换成按颜色出 logit 的替身，好检验对齐和取帧）。"""
    return VideoModel(path)


def window_probs(logits):
    """[12, 3] 逐帧 logit → 窗口概率（与模型里“特征平均 → 线性层 → softmax”相同，float32）。"""
    z = logits.astype(np.float32).mean(axis=0)
    e = np.exp(z - z.max())
    return (e / e.sum()).astype(np.float32)


def classify(probs, do_th, plot_th):
    """VideoProcessActivity.applyVideoResult 的分组概率判定（float 语义）。"""
    p_do = F.f32(F.f32(float(probs[1])) + F.f32(float(probs[2])))
    if p_do >= F.f32(do_th):
        return 'do', p_do
    if F.f32(float(probs[0])) >= F.f32(plot_th):
        return 'Noise', F.f32(float(probs[0]))
    return '', 0.0


class FrameCache(object):
    """(预处理, 帧 pts) → 逐帧 logit。同一帧只解码一次，几种预处理一起做。"""

    def __init__(self, src, model, matrix, log=None):
        self.src, self.model, self.matrix, self.log = src, model, matrix, log
        self.logits = {}

    def ensure(self, preps, pts_set):
        need = sorted(set(u for u in pts_set for p in preps if (p, u) not in self.logits))
        if not need:
            return
        only_keys = all(u in self.src.key_set for u in need)
        pend = dict((p, ([], [])) for p in preps)
        t0 = time.time()

        def flush(p, force=False):
            us, imgs = pend[p]
            if imgs and (force or len(imgs) >= NUM_FRAMES * 8):
                for u, lg in zip(us, self.model.frame_logits(imgs)):
                    self.logits[(p, u)] = lg
                del us[:], imgs[:]

        for u, f in self.src.decode(need, keyframes_only=only_keys, progress=self.log):
            imgs = prep_frame(f, preps, self.matrix)
            for p in preps:
                pend[p][0].append(u)
                pend[p][1].append(imgs[p])
                flush(p)
        for p in preps:
            flush(p, force=True)
        missing = [u for u in need for p in preps if (p, u) not in self.logits]
        if missing:
            raise RuntimeError('有 %d 帧没解出来（例如 %.3fs）：时间戳表和解码结果对不上' % (len(missing), missing[0] / 1e6))
        if self.log:
            self.log('%d 帧 × %s 预处理 + 推理 %.0f 秒%s' % (len(need), '/'.join(preps), time.time() - t0,
                                                      '（只解关键帧）' if only_keys else ''))

    def window(self, prep, pts_list):
        return window_probs(np.stack([self.logits[(prep, u)] for u in pts_list]))


# ======================================================================== run 里的视频结果
class VResult(object):
    __slots__ = ('ts', 'pos', 'age', 'probs', 'fpts')


def load_results(run_path):
    """run CSV → (meta, [VResult])。同一个结果在几个 tick 里重复出现，取年龄最小的那一行估“出结果时的播放位置”。"""
    meta, data = M._rows(run_path)
    by_ts = {}
    for d in data:
        age = F._inum(d.get('v_age'), -1)
        if age < 0:
            continue
        pp = [F._fnum(d.get(c)) for c in ('v_p_plot', 'v_p_oral', 'v_p_sex')]
        if any(x is None for x in pp):
            continue
        ts = F._inum(d.get('wall_ms'), 0) - age
        old = by_ts.get(ts)
        if old is not None and old.age <= age:
            continue
        r = VResult()
        r.ts, r.age, r.pos = ts, age, F._inum(d.get('pos_ms'), 0) - age
        r.probs = np.array(pp, np.float32)
        fp = (d.get('v_fpts') or '').strip()
        try:
            f = [int(x) for x in fp.split(';') if x.strip()] if fp else []
        except ValueError:
            f = []
        r.fpts = f if len(f) == NUM_FRAMES else None
        by_ts[ts] = r
    return meta, sorted(by_ts.values(), key=lambda r: r.ts)


def video_pipe_of(meta):
    """参数行里的 VIDEO_PIPE（2026-10-14 起的 app 才写）；旧 run 是 keyframe 抽帧 + 整帧压扁（legacy）。"""
    for kv in (meta.get('params') or '').split():
        if kv.startswith('VIDEO_PIPE='):
            v = kv.split('=', 1)[1]
            frames, _, geom = v.partition('_')
            if frames == 'web':
                return None           # 网页模式（TextureView 截帧）没有离线对应的管线
            return '%s_%s' % (frames, 'crop' if geom == 'crop' else 'squash')
    return 'keyframe_legacy'


def target_times_us(r, delta_ms, offset_us=0):
    """一个结果的 12 个取帧目标（微秒，从旧到新）。新 run 用 v_fpts，旧 run 从出结果时的播放位置往前推。"""
    if r.fpts is not None:
        return [t * 1000 + offset_us for t in r.fpts]
    end = r.pos - delta_ms
    return [(end - FRAME_STEP_MS * (NUM_FRAMES - 1 - k)) * 1000 for k in range(NUM_FRAMES)]


def pick_frames(src, frames, targets):
    pick = src.keyframe_at if frames == 'keyframe' else src.exact_at
    return [pick(max(0, t)) for t in targets]


def parse_pipe(pipe):
    frames, _, prep = pipe.partition('_')
    if frames not in FRAMES or prep not in PREPS:
        sys.exit('管线名应为 抽帧_预处理（抽帧：%s；预处理：%s），例如 exact_crop：%r'
                 % ('/'.join(FRAMES), '/'.join(PREPS), pipe))
    return frames, prep


def parity(pred, phone, params):
    """→ dict(mae, agree, close, n)：三类概率的平均绝对差、分组判定（转/不转/unclear）一致率、最大差 ≤ 0.05 的比例。"""
    pred, phone = np.asarray(pred), np.asarray(phone)
    if len(pred) == 0:
        return dict(mae=float('nan'), agree=float('nan'), close=float('nan'), n=0)
    d = np.abs(pred - phone)
    do_th, plot_th = params['VIDEO_DO'], params['VIDEO_PLOT']
    agree = np.mean([classify(a, do_th, plot_th)[0] == classify(b, do_th, plot_th)[0] for a, b in zip(pred, phone)])
    return dict(mae=float(d.mean()), agree=float(agree), close=float(np.mean(d.max(axis=1) <= 0.05)), n=len(pred))


def _subset(rs, n):
    if len(rs) <= n:
        return rs
    idx = np.linspace(0, len(rs) - 1, n).round().astype(int)
    return [rs[i] for i in sorted(set(idx))]


# ======================================================================== 一部片的 check / rescore
def _usable(results):
    return [r for r in results if r.pos - DELTA_SCAN[1] - FRAME_STEP_MS * NUM_FRAMES > 0 or r.fpts is not None]


def check_one(job):
    vid, gt_path, run_path, video_path, pipes, matrix, model_path = job
    lines = []

    def log(msg):
        lines.append('    ' + msg)

    meta, results = load_results(run_path)
    params = F.parse_params(meta.get('params', ''))
    results = _usable(results)
    if not results:
        return dict(vid=vid, lines=['  %s: run 里没有可用的视频结果' % vid], rows=[], align=None)
    src = VideoSource(video_path)
    model = _make_model(model_path)
    cache = FrameCache(src, model, matrix, log)
    run_pipe = video_pipe_of(meta)
    new_run = results[0].fpts is not None
    if not pipes:
        pipes = [run_pipe] if run_pipe else ['exact_crop']
    g = src.gop_stats()
    head = '  %s  %s  GOP 中位 %.1fs / 最长 %.1fs，%d 个视频结果，run 的视频管线 %s%s' % (
        vid, os.path.basename(video_path), g['median'], g['max'], len(results), run_pipe or 'web',
        '（有 v_fpts）' if new_run else '（没有 v_fpts：扫 Δ）')
    rows, align = [], None
    sub = _subset(results, SCAN_N)
    for pipe in pipes:
        frames, prep = parse_pipe(pipe)
        if new_run:
            cands = [o * src.frame_dur_us for o in OFFSET_FRAMES]
            label = lambda c: '%+d 帧' % int(round(c / src.frame_dur_us))      # noqa: E731
            tgt = lambda r, c: target_times_us(r, 0, c)                           # noqa: E731
        else:
            cands = list(range(DELTA_SCAN[0], DELTA_SCAN[1] + 1, DELTA_SCAN[2]))
            label = lambda c: 'Δ=%dms' % c                                         # noqa: E731
            tgt = lambda r, c: target_times_us(r, c)                               # noqa: E731
        if frames == 'exact' and not new_run:
            cands = [NEW_APP_DELTA]       # 旧 run 不是这条管线录的，扫不出有意义的 Δ：按修复后 app 的时序算一个对照
        need = set()
        for c in cands:
            for r in sub:
                need.update(pick_frames(src, frames, tgt(r, c)))
        cache.ensure([prep], need)
        scores = []
        for c in cands:
            pred = [cache.window(prep, pick_frames(src, frames, tgt(r, c))) for r in sub]
            scores.append((parity(pred, [r.probs for r in sub], params)['mae'], c))
        best_mae, best = min(scores)
        need = set()
        for r in results:
            need.update(pick_frames(src, frames, tgt(r, best)))
        cache.ensure([prep], need)
        pred, frozen = [], []
        for r in results:
            fs = pick_frames(src, frames, tgt(r, best))
            pred.append(cache.window(prep, fs))
            frozen.append(len(set(fs)) == 1)
        st = parity(pred, [r.probs for r in results], params)
        fz = np.array(frozen)
        st_f = parity([p for p, z in zip(pred, fz) if z], [r.probs for r, z in zip(results, fz) if z], params)
        edge = len(cands) > 1 and best == cands[-1]
        rows.append(dict(pipe=pipe, best=label(best), cand=best, st=st, frozen=st_f, n_frozen=int(fz.sum()), edge=edge))
        if pipe == (run_pipe or pipe) and (frames == 'keyframe' or new_run):
            align = dict(run=os.path.relpath(run_path, ROOT).replace('\\', '/'), pipe=pipe, matrix=matrix,
                         delta_ms=None if new_run else best, offset_us=best if new_run else 0,
                         mae=st['mae'], agree=st['agree'])
    out = [head] + lines
    for row in rows:
        st, fz = row['st'], row['frozen']
        grade = '✓ 复现' if st['mae'] <= GOOD_MAE else ('~ 接近' if st['mae'] <= OK_MAE else '✗ 对不上')
        out.append('    %-16s %-9s MAE %.4f  判定一致 %5.1f%%  差≤0.05 %5.1f%%   %s%s' % (
            row['pipe'], row['best'], st['mae'], 100 * st['agree'], 100 * st['close'], grade,
            ('   12 帧同一张图的窗口 %d 个：MAE %.4f' % (row['n_frozen'], fz['mae'])) if row['n_frozen'] else ''))
        if row['edge']:
            out.append('    ⚠ %s 的最佳时间差落在扫描范围的边上，真实值可能更大，结果不可信' % row['pipe'])
    return dict(vid=vid, lines=out, rows=rows, align=align)


def rescore_one(job):
    vid, gt_path, run_path, video_path, pipe, matrix, model_path, out_path, align, delta_arg = job
    lines = []

    def log(msg):
        lines.append('    ' + msg)

    meta, results = load_results(run_path)
    params = F.parse_params(meta.get('params', ''))
    frames, prep = parse_pipe(pipe)
    src = VideoSource(video_path)
    model = _make_model(model_path)
    cache = FrameCache(src, model, matrix, log)
    new_run = bool(results) and results[0].fpts is not None
    offset = (align or {}).get('offset_us') or 0
    if delta_arg is not None:
        delta = delta_arg
    elif frames == 'keyframe' and align and align.get('delta_ms') is not None:
        delta = align['delta_ms']              # 复现旧 app：用 check 扫出来的旧 Δ（含旧解码器的积压）
    else:
        delta = NEW_APP_DELTA                  # 模拟修复后的 app（或旧 Δ 没扫过）
    picks = {}
    for r in results:
        targets = target_times_us(r, delta, offset if new_run else 0)
        if not new_run and min(targets) < 0:
            continue
        picks[r.ts] = pick_frames(src, frames, targets)
    cache.ensure([prep], set(u for fs in picks.values() for u in fs))
    computed = dict((ts, (cache.window(prep, fs), fs)) for ts, fs in picks.items())
    note = ('video_rescore.py rescore: pipe=%s matrix=%s %s model=%s' % (
        pipe, matrix, ('offset=%dus' % offset) if new_run else ('delta=%dms' % delta),
        os.path.basename(model_path)))
    n_hit, n_tick = write_rescored(run_path, out_path, computed, note, params)
    used = sorted(set(u for fs in picks.values() for u in fs))
    np.savez_compressed(os.path.splitext(out_path)[0] + '.vframes.npz',
                        pts_us=np.array(used, np.int64),
                        logits=np.stack([cache.logits[(prep, u)] for u in used]) if used else np.zeros((0, 3), np.float32),
                        pipe=np.array(pipe), matrix=np.array(matrix))
    head = '  %s → %s：%d 个视频结果（%d/%d 个 tick 换了概率），%d 帧，%d 次推理' % (
        vid, os.path.relpath(out_path, ROOT), len(computed), n_hit, n_tick, len(used), model.calls)
    return dict(vid=vid, lines=[head] + lines)


def write_rescored(run_path, out_path, computed, note, params):
    """computed: {结果时刻 ts: (probs[3], 12 帧的 pts_us)}。v_p_* / v_action / v_conf / v_pos / v_fpts 换成重算的，其余列原样。"""
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
    cols = header + [c for c in ('v_pos', 'v_fpts') if c not in header]
    ix = dict((c, i) for i, c in enumerate(cols))
    n_hit = n_tick = 0
    for f in rows:
        f += [''] * (len(cols) - len(f))
        age = F._inum(f[ix['v_age']], -1)
        wall = F._inum(f[ix['wall_ms']], 0)
        if age < 0:
            continue
        n_tick += 1
        got = computed.get(wall - age)
        if got is None:
            for c in ('v_p_plot', 'v_p_oral', 'v_p_sex', 'v_fpts'):
                f[ix[c]] = ''
            f[ix['v_pos']] = '-1'
            f[ix['v_action']], f[ix['v_conf']] = '', '0.0'
            continue
        n_hit += 1
        probs, fs = got
        for c, p in zip(('v_p_plot', 'v_p_oral', 'v_p_sex'), probs):
            f[ix[c]] = '%.4f' % p
        if age > F.MAX_AGE_MS:              # 融合循环：结果超过 2 秒算过期，动作串清空
            act, conf = '', 0.0
        else:
            act, conf = classify(probs, params['VIDEO_DO'], params['VIDEO_PLOT'])
        f[ix['v_action']], f[ix['v_conf']] = act, F._gf(conf)
        f[ix['v_pos']] = str(fs[-1] // 1000)
        f[ix['v_fpts']] = ';'.join(str(u // 1000) for u in fs)
    d = os.path.dirname(out_path)
    if d and not os.path.isdir(d):
        os.makedirs(d)
    with io.open(out_path, 'w', encoding='utf-8', newline='') as fh:
        for c in comments:
            fh.write(c + '\n')
        fh.write('# ' + note + '\n')
        fh.write('# 注意：v_p_* / v_action / v_conf / v_pos / v_fpts 是离线重算的；gated、arb、ble_*、sent* 等决策列仍是原 run 的。'
                 '端到端指标请用 fusion_lab.py eval/oracle、arbiter_lab.py 重放\n')
        fh.write(','.join(cols) + '\n')
        w = csv.writer(fh, lineterminator='\n')
        for f in rows:
            w.writerow(f)
    return n_hit, n_tick


# ======================================================================== 子命令
def _jobs_map(fn, jobs, n):
    if n > 1 and len(jobs) > 1:
        with ProcessPoolExecutor(max_workers=n) as ex:
            for res in ex.map(fn, jobs):
                yield res
    else:
        for j in jobs:
            yield fn(j)


def load_alignment(path):
    if path and os.path.exists(path):
        with io.open(path, encoding='utf-8') as fh:
            return json.load(fh)
    return {}


def _align_key(run_path):
    return os.path.relpath(run_path, ROOT).replace('\\', '/')


def cmd_keyframes(a):
    _need(pyav=True)
    gts = sorted(glob.glob(os.path.join(a.gt, '*.csv')))
    only = set(a.only.split(',')) if a.only else None
    explicit = dict(p.split('=', 1) for p in (a.pair or []) if '=' in p)
    print('关键帧间隔（keyframe 管线下视频结果每个 GOP 只变一次；“冻住”= 12 帧窗口全是同一张关键帧的时长占比）')
    print('  %-5s %-28s %6s %8s %8s %8s %7s' % ('片子', '文件', '关键帧', 'GOP中位', 'GOP p90', 'GOP最长', '冻住'))
    for g in gts:
        vid = os.path.splitext(os.path.basename(g))[0]
        if only and vid not in only:
            continue
        gmeta = A._gt_meta(g)
        v = explicit.get(vid) or A.find_video(a.videos, vid, gmeta, {})
        if not v or not os.path.exists(v):
            print('  %-5s 找不到视频（file_name=%s）' % (vid, gmeta.get('file_name', '?')))
            continue
        st = VideoSource(v).gop_stats()
        print('  %-5s %-28s %6d %7.1fs %7.1fs %7.1fs %6.0f%%' % (vid, os.path.basename(v)[:28], st['n_keys'],
                                                              st['median'], st['p90'], st['max'], 100 * st['frozen']))


def cmd_check(a):
    _need(pyav=True)
    pairs = A.resolve_pairs(a)
    pipes = [p.strip() for p in a.pipes.split(',')] if a.pipes else None
    for p in pipes or []:
        parse_pipe(p)
    jobs = [(vid, g, r, v, pipes, a.matrix, a.model) for vid, g, r, v in pairs]
    print('视频重算 check：%d 部片，矩阵 %s%s' % (len(jobs), a.matrix, '' if cv2 is not None else '（没装 opencv，GPU 采样用 numpy 算，较慢）'))
    align = load_alignment(a.align)
    for res in _jobs_map(check_one, jobs, a.jobs):
        print('\n'.join(res['lines']))
        if res['align']:
            align[res['align']['run']] = res['align']
    if a.align:
        d = os.path.dirname(a.align)
        if d and not os.path.isdir(d):
            os.makedirs(d)
        with io.open(a.align, 'w', encoding='utf-8') as fh:
            json.dump(align, fh, ensure_ascii=False, indent=1, sort_keys=True)
        print('对齐结果已写入 %s（rescore 用其中的 Δ）' % os.path.relpath(a.align))
    print('\nMAE ≤ %.2f 算复现。keyframe_legacy 能复现旧 run，说明旧版抽帧确实只取到关键帧；'
          '新 run 用 exact_crop 复现，说明手机上跑的就是离线评估的那套。' % GOOD_MAE)


def cmd_rescore(a):
    _need(pyav=True)
    parse_pipe(a.pipe)
    pairs = A.resolve_pairs(a)
    tag = a.tag or 'video_%s_%s' % (a.pipe, os.path.basename(os.path.normpath(a.runs)))
    out_dir = os.path.join(a.out, tag)
    align = load_alignment(a.align)
    frames, _ = parse_pipe(a.pipe)
    jobs = []
    for vid, g, r, v in pairs:
        al = align.get(_align_key(r))
        if frames == 'keyframe' and a.delta is None and (al is None or al.get('delta_ms') is None):
            print('  %s: video_alignment.json 里没有这份 run 的 Δ，按 %dms 算（复现旧 app 前先跑 check）' % (vid, NEW_APP_DELTA))
        jobs.append((vid, g, r, v, a.pipe, a.matrix, a.model, os.path.join(out_dir, os.path.basename(r)), al, a.delta))
    print('视频重算 rescore：管线 %s，矩阵 %s → %s' % (a.pipe, a.matrix, os.path.relpath(out_dir)))
    for res in _jobs_map(rescore_one, jobs, a.jobs):
        print('\n'.join(res['lines']))
    print('下一步：python3 bench/tools/fusion_lab.py eval bench/gt %s（换了视频输入，概率分布变了，'
          '比较前先按 README 第 4 节重调阈值和融合参数）' % os.path.relpath(out_dir))


# ======================================================================== Java 对拍数据
GOLDEN_CASES = [
    # (种类, 图案, w, h, 区域 (x0, y0, cw, ch) 或 None, 目标宽, 目标高)
    ('resize', 'noise', 67, 45, None, 23, 15),            # 横向非整数倍、纵向整数倍 → 通用面积
    ('resize', 'noise', 56, 56, None, 40, 40),            # 1.4 倍，同 224 → 160
    ('resize', 'noise', 96, 72, None, 32, 24),            # 3 倍整数 → 整块求和
    ('resize', 'noise', 90, 60, None, 45, 30),            # 2 倍整数
    ('resize', 'pattern', 101, 77, None, 64, 64),         # 两个方向都非整数倍
    ('resize', 'noise', 120, 80, (20, 0, 80, 80), 30, 30),   # 先居中裁再缩
    ('resize', 'pattern', 40, 30, None, 56, 42),          # 放大 → 双线性
    ('resize', 'noise', 64, 48, None, 64, 48),            # 同尺寸 → 原样
    ('model', 'pattern', 672, 672, None, MODEL_SIZE, MODEL_SIZE),   # 离线模式：GPU 画好的 672 → 224 → 160
]
_LCG_A, _LCG_C, _MASK = 6364136223846793005, 1442695040888963407, (1 << 64) - 1


def golden_image(w, h, pattern, seed):
    """与 FramePreprocessorTest.goldenImage 相同的确定性测试图。"""
    if pattern == 'pattern':
        y, x = np.mgrid[0:h, 0:w]
        r = (x * 7 + y * 3) & 255
        g = (x * x + y) & 255
        b = ((x ^ y) * 5) & 255
        return np.stack([r, g, b], axis=-1).astype(np.uint8)
    s = seed
    out = np.empty(w * h * 3, np.uint8)
    for i in range(out.size):
        s = (s * _LCG_A + _LCG_C) & _MASK
        out[i] = s >> 56
    return out.reshape(h, w, 3)


def _golden_seed(i, w, h):
    return 1000003 * (i + 1) + 31 * w + h


def golden(path=GOLDEN_PATH, write=True):
    _need()
    lines = ['# FramePreprocessor 对拍数据：python3 bench/tools/video_rescore.py golden 生成，FramePreprocessorTest 校验。',
             '# 输入是 golden_image() 生成的测试图（Java 端同样生成）；每个 case 后面是期望输出，一行一行像素，每个像素 RRGGBB。',
             '# case <种类> <图案> <种子> <w> <h> <x0> <y0> <cw> <ch> <dw> <dh>']
    for i, (kind, pattern, w, h, region, dw, dh) in enumerate(GOLDEN_CASES):
        seed = _golden_seed(i, w, h)
        img = golden_image(w, h, pattern, seed)
        x0, y0, cw, ch = region or (0, 0, w, h)
        if kind == 'model':
            out = to_model_input(img, region)
        else:
            out = resize(img[y0:y0 + ch, x0:x0 + cw], dw, dh)
        lines.append('case %s %s %d %d %d %d %d %d %d %d %d' % (kind, pattern, seed, w, h, x0, y0, cw, ch, dw, dh))
        for row in out:
            lines.append(''.join('%02x%02x%02x' % tuple(int(v) for v in px) for px in row))
    txt = '\n'.join(lines) + '\n'
    if write:
        d = os.path.dirname(path)
        if not os.path.isdir(d):
            os.makedirs(d)
        with io.open(path, 'w', encoding='utf-8', newline='\n') as fh:
            fh.write(txt)
        print('已写 %s（%d 个 case）' % (os.path.relpath(path), len(GOLDEN_CASES)))
    return txt


# ======================================================================== 自检
def _synth_video(path, seconds=24, fps=30, gop=120, w=1280, h=720):
    """每帧都不一样的测试视频（红色的整体亮度逐帧变化），关键帧间隔 gop 帧，带 B 帧。"""
    with av.open(path, mode='w') as c:
        s = c.add_stream('libx264', rate=fps)
        s.width, s.height, s.pix_fmt = w, h, 'yuv420p'
        s.options = {'g': str(gop), 'keyint_min': str(gop), 'sc_threshold': '0', 'bf': '2',
                     'preset': 'ultrafast', 'crf': '20'}
        yy, xx = np.mgrid[0:h, 0:w]
        for i in range(seconds * fps):
            img = np.empty((h, w, 3), np.uint8)
            img[..., 0] = 30 + (11 * i) % 200
            img[..., 1] = xx * 255 // w
            img[..., 2] = ((xx // 80 + yy // 80 + i // 3) % 2) * 220
            fr = av.VideoFrame.from_ndarray(img, format='rgb24')
            fr.pts = i
            for p in s.encode(fr):
                c.mux(p)
        for p in s.encode():
            c.mux(p)


def _synth_run(path, src, cache, pipe, delta_ms, new_format, params_line):
    """按指定管线算出视频结果，写成 BenchLogger 格式的 run（新格式带 v_pos / v_fpts）。结果每 500ms 一个、年龄 120ms。"""
    frames, prep = parse_pipe(pipe)
    cols = list(M.RUN_COLS_NEEDED) if new_format else [c for c in M.RUN_COLS_NEEDED if c not in ('v_pos', 'v_fpts')]
    rows, end_ms = [], src.frames_us[-1] // 1000
    picks = []
    for pos in range(4000, end_ms - 200, 500):
        r = VResult()
        r.pos, r.fpts = pos, None
        targets = target_times_us(r, delta_ms)
        fs = pick_frames(src, frames, targets)
        picks.append((pos, fs, [t // 1000 for t in targets]))
    cache.ensure([prep], set(u for _, fs, _ in picks for u in fs))
    for pos, fs, tg in picks:
        p = cache.window(prep, fs)
        d = dict((c, '') for c in cols)
        d.update(pos_ms=pos + 120, wall_ms=1000000 + pos + 120, v_age=120, a_age=-1, a_pos=-1,
                 v_p_plot='%.4f' % p[0], v_p_oral='%.4f' % p[1], v_p_sex='%.4f' % p[2])
        if new_format:
            d.update(v_pos=fs[-1] // 1000, v_fpts=';'.join(str(u // 1000) for u in fs))
        rows.append(d)
    with io.open(path, 'w', encoding='utf-8', newline='') as fh:
        fh.write('# xcup-bench-run v1\n# video_name=synth\n# params: %s\n' % params_line)
        w = csv.DictWriter(fh, fieldnames=cols, lineterminator='\n')
        w.writeheader()
        for d in rows:
            w.writerow(d)


def selftest():
    import tempfile
    _need()
    fails = []

    def check(ok, msg):
        print('  %s %s' % ('ok  ' if ok else 'FAIL', msg))
        if not ok:
            fails.append(msg)

    # 1. 预处理
    check(golden(write=False) == io.open(GOLDEN_PATH, encoding='utf-8').read(),
          '对拍数据与 FramePreprocessor 的 Python 移植一致（改了算法要同步 Java 并跑 golden）')
    check(crop_rect(1920, 1080) == (420, 0, 1080, 1080) and crop_rect(1080, 1920) == (0, 420, 1080, 1080),
          '居中裁剪区域与 FramePreprocessor.cropRect 相同')
    check(ffmpeg_increase(1920, 1080, 224) == (398, 224) and ffmpeg_increase(720, 1280, 224) == (224, 398),
          'ffmpeg force_original_aspect_ratio=increase 的尺寸')
    if cv2 is not None:
        rng = np.random.default_rng(0)
        worst, exact = 0, True
        for sw, sh, dw, dh in [(672, 672, 224, 224), (224, 224, 160, 160), (67, 45, 23, 15), (101, 77, 64, 64),
                               (1080, 1080, 224, 224), (540, 540, 224, 224), (90, 60, 45, 30)]:
            img = rng.integers(0, 256, (sh, sw, 3), dtype=np.uint8)
            d = np.abs(resize(img, dw, dh).astype(int)
                       - cv2.resize(img, (dw, dh), interpolation=cv2.INTER_AREA).astype(int))
            worst = max(worst, int(d.max()))
            if not (sw == 2 * dw and sh == 2 * dh):
                exact &= int(d.max()) == 0
        check(worst <= 1 and exact, '面积缩小与 cv2.INTER_AREA 一致（2 倍整数缩小时差 ≤1，其余逐位相同）')
    else:
        print('  skip 与 cv2.INTER_AREA 对比（没装 opencv-python-headless）')
    img = np.random.default_rng(1).integers(0, 256, (90, 160, 3), dtype=np.uint8)
    if cv2 is not None:
        a = gl_linear(img, (20, 5, 100, 80), 50, 40)
        globals()['cv2'], keep = None, cv2
        try:
            b = gl_linear(img, (20, 5, 100, 80), 50, 40)
        finally:
            globals()['cv2'] = keep
        inner = np.abs(a.astype(int) - b.astype(int))[1:-1, 1:-1]
        check(inner.max() <= 1, 'GL 双线性：cv2 与 numpy 两种算法一致（内部差 ≤1）')

    # 2. 端到端：合成视频 + 模型
    if av is None:
        print('  skip 端到端（需要 PyAV：pip install av）')
        return _finish(fails)
    try:
        model = VideoModel()
    except SystemExit as e:
        print('  skip 端到端（%s）' % e)
        return _finish(fails)
    tmp = tempfile.mkdtemp(prefix='vrescore_')
    vpath = os.path.join(tmp, 'gop4.mp4')
    _synth_video(vpath)
    src = VideoSource(vpath)
    check(src.keys_us == [0, 4000000, 8000000, 12000000, 16000000, 20000000],
          '关键帧表（4 秒一个）：%s' % [k // 1000 for k in src.keys_us])
    check(src.keyframe_at(5900000) == 4000000 and src.keyframe_at(6100000) == 8000000,
          'keyframe 抽帧：取离目标最近的关键帧（可能在目标之后）')
    check(src.exact_at(1000000) == 1000000 and src.exact_at(1010000) == 1000000 and src.exact_at(1020000) == 1033333,
          'exact 抽帧：pts 不晚于目标 + 半帧的最后一帧')
    frames = {}
    for u, f in src.decode([0, 1000000, 4000000, 4033333]):
        frames[u] = f.to_ndarray(format='rgb24')
    check(sorted(frames) == [0, 1000000, 4000000, 4033333] and np.abs(frames[4000000].astype(int)
                                                                     - frames[4033333].astype(int)).mean() > 2,
          '按 pts 解码：要的帧都拿到了，相邻两帧内容不同')
    imgs = [prep_frame(f, ['crop'], 'auto')['crop'] for _, f in src.decode(src.frames_us[:12])]
    probs, lg = model.run_window(imgs)
    check(np.abs(window_probs(lg) - probs).max() < 1e-5, '逐帧 logit 求平均 = 模型的窗口输出（差 %.1e）'
          % np.abs(window_probs(lg) - probs).max())
    perm = list(range(12))
    perm[0], perm[7] = 7, 0
    _, lg2 = model.run_window([imgs[i] for i in perm])
    check(np.allclose(lg2[0], lg[7], atol=1e-4) and np.allclose(lg2[7], lg[0], atol=1e-4),
          '模型里 PACK 的输入顺序就是帧的顺序')

    # 3. check 能从旧格式 run 里找回 Δ、复现结果；新格式 run 按 v_fpts 逐帧对上。
    #    合成画面在真模型眼里全是“剧情”，概率几乎不动，分不出取了哪一帧；这里换成按画面颜色出 logit 的替身模型
    global _make_model
    keep_make = _make_model
    _make_model = lambda path: _ColorModel()            # noqa: E731
    try:
        cache = FrameCache(src, _ColorModel(), 'auto')
        params = 'VIDEO_DO=0.65 VIDEO_PLOT=0.60 AUDIO_DO=0.55 AUDIO_NOISE=0.60'
        old_run = os.path.join(tmp, 'old.csv')
        _synth_run(old_run, src, cache, 'keyframe_legacy', 200, False, params)
        res = check_one(('synth', None, old_run, vpath, ['keyframe_legacy', 'exact_legacy'], 'auto', DEFAULT_MODEL))
        kf = [r for r in res['rows'] if r['pipe'] == 'keyframe_legacy'][0]
        ex = [r for r in res['rows'] if r['pipe'] == 'exact_legacy'][0]
        # 结果每 500ms、帧每 250ms，取帧目标 ≡ -Δ (mod 250)，GOP 中点 ≡ 0：Δ 在 [0, 250) 里关键帧的分配都一样
        # （Δ=0 时目标正好落在中点上，一样近取前一个关键帧，和 Δ=200 时落在中点前 200ms 的分配相同），真值 200 也在其中
        check(kf['st']['mae'] < 1e-3 and 0 <= kf['cand'] < 250 and kf['n_frozen'] > 0,
              '旧格式 run：keyframe_legacy 复现（%s，MAE %.5f，%d 个窗口 12 帧同一张图）'
              % (kf['best'], kf['st']['mae'], kf['n_frozen']))
        check(ex['st']['mae'] > 0.05, '旧格式 run：exact_legacy 对不上（MAE %.4f）' % ex['st']['mae'])
        new_run = os.path.join(tmp, 'new.csv')
        _synth_run(new_run, src, cache, 'exact_crop', 150, True, params + ' VIDEO_PIPE=exact_crop')
        res = check_one(('synth', None, new_run, vpath, None, 'auto', DEFAULT_MODEL))
        row = res['rows'][0]
        check(row['pipe'] == 'exact_crop' and row['st']['mae'] < 1e-3 and row['cand'] == 0,
              '新格式 run：按参数行 VIDEO_PIPE=exact_crop、v_fpts 逐帧复现（%s，MAE %.5f）' % (row['best'], row['st']['mae']))
        out = os.path.join(tmp, 'rescored.csv')
        rescore_one(('synth', None, old_run, vpath, 'exact_crop', 'auto', DEFAULT_MODEL, out,
                     {'delta_ms': 900, 'offset_us': 0}, None))
        m2, rs2 = load_results(out)
        check(len(rs2) > 30 and all(r.fpts is not None for r in rs2), 'rescore 写出的 run 能读回，带 v_fpts')
        _, rs_old = load_results(old_run)
        lag = [r.pos - r2.fpts[-1] for r, r2 in zip(rs_old, rs2)]
        check(all(abs(x - NEW_APP_DELTA) <= 34 for x in lag),
              'rescore 的 exact 管线按修复后 app 的时序取帧（最后一帧比出结果早约 %dms），不用旧 Δ' % NEW_APP_DELTA)
        res = check_one(('synth', None, out, vpath, ['exact_crop'], 'auto', DEFAULT_MODEL))
        check(res['rows'][0]['st']['mae'] < 1e-3, 'rescore 写出的 run 再 check：exact_crop 逐帧复现（MAE %.5f）'
              % res['rows'][0]['st']['mae'])
        z = np.load(os.path.splitext(out)[0] + '.vframes.npz')
        check(z['logits'].shape == (len(z['pts_us']), 3), 'rescore 写出逐帧 logit（%d 帧）' % len(z['pts_us']))
    finally:
        _make_model = keep_make
    return _finish(fails)


class _ColorModel(object):
    """自检用的替身模型：logit 由画面平均颜色决定，帧与帧之间差别大，能检验对齐和取帧逻辑。"""
    calls = 0

    def frame_logits(self, frames):
        self.calls += 1
        return np.array([[f[..., c].mean() / 16.0 for c in range(3)] for f in frames], np.float32)


def _finish(fails):
    if fails:
        print('selftest: %d 项失败' % len(fails))
        sys.exit(1)
    print('selftest: ALL PASS')


# ======================================================================== main
def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[1],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--selftest', action='store_true', help='自检')
    sub = p.add_subparsers(dest='cmd')

    def common(sp, runs=True):
        sp.add_argument('--gt', default=os.path.join(ROOT, 'bench', 'gt'), help='真值目录（默认 bench/gt）')
        if runs:
            sp.add_argument('--runs', default=os.path.join(ROOT, 'bench', 'runs_initial'),
                            help='run CSV 目录（默认 bench/runs_initial）')
        sp.add_argument('--videos', default=DEFAULT_VIDEOS, help='原始视频目录（默认 bench/videos）')
        sp.add_argument('--pair', action='append', help='逐个指定视频：v01=D:/xcup/视频1.mp4（可多次）')
        sp.add_argument('--only', help='只跑这几部，逗号分隔：v01,v04')

    sp = sub.add_parser('keyframes', help='每部片的关键帧间隔')
    common(sp, runs=False)
    for name, helptext in (('check', '看哪条管线能复现真机的 v_p_*'), ('rescore', '按指定管线重算，写出新的 run CSV')):
        sp = sub.add_parser(name, help=helptext)
        common(sp)
        sp.add_argument('--model', default=DEFAULT_MODEL, help='视频模型（默认 app 里的）')
        sp.add_argument('--matrix', default='auto', choices=MATRICES,
                        help='YUV→RGB 矩阵：auto（码流标的；没标时高清 BT.709、标清 BT.601，同 Android）')
        sp.add_argument('--align', default=DEFAULT_ALIGN, help='对齐结果文件（默认 bench/rescored/video_alignment.json）')
        sp.add_argument('--jobs', type=int, default=1, help='几部片并行（每个进程各占 4 个线程）')
        if name == 'check':
            sp.add_argument('--pipes', help='要比的管线，逗号分隔（默认按 run 判断）：如 keyframe_legacy,exact_legacy')
        else:
            sp.add_argument('--pipe', default='exact_crop', help='管线（默认 exact_crop = 修复后的 app）')
            sp.add_argument('--out', default=DEFAULT_OUT, help='输出根目录（默认 bench/rescored）')
            sp.add_argument('--tag', help='输出子目录名（默认 video_<管线>_<runs 目录名>）')
            sp.add_argument('--delta', type=int, help='旧 run（没有 v_fpts）的时间差 Δ（ms）。默认：keyframe 管线用 check 扫出的值，'
                                                      'exact 管线用修复后 app 的 %dms' % NEW_APP_DELTA)
    sub.add_parser('golden', help='重新生成 Java 端 FramePreprocessorTest 的对拍数据（改了 FramePreprocessor 才需要）')
    a = p.parse_args()
    if a.selftest:
        selftest()
    elif a.cmd == 'golden':
        golden()
    elif a.cmd == 'keyframes':
        cmd_keyframes(a)
    elif a.cmd == 'check':
        cmd_check(a)
    elif a.cmd == 'rescore':
        cmd_rescore(a)
    else:
        p.print_help()


if __name__ == '__main__':
    main()
