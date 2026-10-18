#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
training/train_utils.py — 视频模型训练的积木（train_video.py、eval_clips.py 用）

模型结构与 app 现用的相同：每帧过 MobileNetV3（共享权重）→ 各帧特征取平均 → 线性层 → softmax，类别顺序
[normal_plot, oral, sex]，输入是原始像素 0~255（模型自带归一化）。

  读帧      read_clip：顺序解码全部帧再挑（不用 CAP_PROP_POS_FRAMES 跳帧：慢，h264 上也不一定准）。读不出来抛 ClipError，
            不再用全黑帧顶替。验证、评测固定 12 帧 np.linspace(0, n-1, 12)，与旧脚本相同（帧间隔约 250ms，同 app）；
            训练只取 4 帧时按 TSN 的做法：片段等分 4 段，每段随机取一帧
  增强      片段级：同一个片段的所有帧用同一组参数（裁剪面积 ≥0.7、水平翻转、轻度亮度 / 对比度 / 饱和度、少量模糊和 JPEG）
  采样      BalancedSampler：按 类别 → 源视频 → 片段 均衡。先等概率选类别，再按 片段数^alpha 选源视频（alpha=0 每部片
            等权，1 = 按片段数，默认 0.5），最后在这部片里等概率选片段
  选模型    ValMetrics：每个 epoch 在验证集上算 p_oral+p_sex 的 AUC（val_auc_do）、准确率、按片子平均的准确率、loss，
            写进 logs 给 ModelCheckpoint / EarlyStopping 用（要放在它们前面）
  导出      export_tflite：搭一个与现用模型同结构的 12 帧模型（TimeDistributed 展开 12 份 + 时间平均 + 线性层，与训练模型
            共享权重），转成只用内置算子的 float32 TFLite。tflite_structure 检查它是 1 个 PACK（12 个输入）+ 1 个
            FULLY_CONNECTED（bench 的 video_rescore.py / baseline.py rescore 靠这个读逐帧 logit）；check_tflite 用真实
            片段比 Keras 与 TFLite 的输出，并按 bench 的方法从逐帧特征重算窗口概率

训练用的模型把 [B, T, H, W, 3] 合成 [B*T, H, W, 3] 一次过 backbone（比 TimeDistributed 展开 T 份快），BN 始终冻结，
所以与逐帧过 backbone 的导出模型数值相同（check_tflite 会核对）。权重存成 Keras 的 .weights.h5，换帧数不影响。

自检：python3 training/train_utils.py --selftest（要 ffmpeg；不下载 ImageNet 权重，CPU 上约 2 分钟）
"""
import io
import math
import os
import re
import shutil
import sys
import tempfile
import time
import warnings
from collections import Counter, OrderedDict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')

import numpy as np          # noqa: E402
import cv2                  # noqa: E402
import tensorflow as tf     # noqa: E402
import keras                # noqa: E402

import data_check as D      # noqa: E402

CLASS_NAMES = list(D.CLASS_NAMES)          # ['normal_plot', 'oral', 'sex']
NUM_CLASSES = len(CLASS_NAMES)
IMAGE_SIZE = 160
EVAL_FRAMES = 12                           # app：12 帧、间隔 250ms
APP_DO, APP_PLOT = 0.65, 0.60              # app 的判定阈值（VideoProcessActivity）：p_do ≥ 0.65 转，p_plot ≥ 0.60 不转
REF_TF = '2.20'                            # 现用模型是 TF 2.20.0 导出的（tflite 的 CONVERSION_METADATA）；app 的运行时是 TFLite 2.13
APP_RUNTIME = (2, 13)

AUG_MILD = dict(min_area=0.7, ratio=(3 / 4.0, 4 / 3.0), flip=0.5, brightness=0.10, contrast=0.10, saturation=0.15,
                blur_p=0.10, blur_sigma=(0.4, 1.0), jpeg_p=0.15, jpeg_q=(40, 85))


class ClipError(Exception):
    pass


# ======================================================================== 读帧
def read_frames(path):
    """顺序解码全部帧 → [BGR uint8]。"""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ClipError('打不开：%s' % path)
    frames = []
    while True:
        ok, f = cap.read()
        if not ok or f is None:
            break
        frames.append(f)
    cap.release()
    if not frames:
        raise ClipError('读不出帧：%s' % path)
    return frames


def eval_indices(n, t=EVAL_FRAMES):
    """均匀取 t 帧（旧脚本的做法）。36 帧取 12 帧：0, 3, 6, 9, 12, 15, 19, 22, 25, 28, 31, 35。"""
    return np.linspace(0, n - 1, t).astype(np.int32)


def tsn_indices(n, t, rng):
    """片段等分 t 段，每段随机一帧（TSN）。"""
    edges = np.linspace(0, n, t + 1)
    lo = np.floor(edges[:-1]).astype(np.int64)
    hi = np.maximum(lo + 1, np.floor(edges[1:]).astype(np.int64))
    return np.array([rng.integers(a, b) for a, b in zip(lo, hi)], np.int32)


def to_model(rgb, size=IMAGE_SIZE):
    """缩到 size×size：缩小用 INTER_AREA（与旧脚本、app 的 FramePreprocessor 相同），放大用双线性。"""
    h, w = rgb.shape[:2]
    if (h, w) == (size, size):
        return rgb
    interp = cv2.INTER_AREA if (h >= size and w >= size) else cv2.INTER_LINEAR
    return cv2.resize(rgb, (size, size), interpolation=interp)


def sample_aug(rng, h, w, cfg=AUG_MILD):
    """一个片段的增强参数（所有帧共用）。裁剪：面积占比 [min_area, 1]、宽高比 ratio，位置随机。"""
    area = float(h * w)
    crop = (0, 0, w, h)
    for _ in range(10):
        a = rng.uniform(cfg['min_area'], 1.0) * area
        r = math.exp(rng.uniform(math.log(cfg['ratio'][0]), math.log(cfg['ratio'][1])))
        cw, ch = int(round(math.sqrt(a * r))), int(round(math.sqrt(a / r)))
        if 0 < cw <= w and 0 < ch <= h:
            crop = (int(rng.integers(0, w - cw + 1)), int(rng.integers(0, h - ch + 1)), cw, ch)
            break
    return dict(crop=crop, flip=bool(rng.random() < cfg['flip']),
                b=1.0 + rng.uniform(-cfg['brightness'], cfg['brightness']),
                c=1.0 + rng.uniform(-cfg['contrast'], cfg['contrast']),
                s=1.0 + rng.uniform(-cfg['saturation'], cfg['saturation']),
                blur=float(rng.uniform(*cfg['blur_sigma'])) if rng.random() < cfg['blur_p'] else 0.0,
                jpeg=int(rng.integers(cfg['jpeg_q'][0], cfg['jpeg_q'][1] + 1)) if rng.random() < cfg['jpeg_p'] else 0)


_LUMA = np.array([0.299, 0.587, 0.114], np.float32)


def apply_aug(rgb, p, size=IMAGE_SIZE):
    """一帧 RGB → 增强后的 size×size（裁剪 → 缩放 → 翻转 → 颜色 → 模糊 → JPEG）。"""
    x0, y0, cw, ch = p['crop']
    img = to_model(rgb[y0:y0 + ch, x0:x0 + cw], size)
    if p['flip']:
        img = img[:, ::-1]
    if (p['b'], p['c'], p['s']) != (1.0, 1.0, 1.0):
        f = img.astype(np.float32) * np.float32(p['b'])
        g = f @ _LUMA
        m = np.float32(g.mean())
        f = (f - m) * np.float32(p['c']) + m                    # 对比度：绕这一帧的平均亮度拉伸
        g = (f @ _LUMA)[..., None]
        f = g + (f - g) * np.float32(p['s'])                    # 饱和度：离灰度的距离
        img = np.clip(f + 0.5, 0, 255).astype(np.uint8)
    if p['blur'] > 0:
        img = cv2.GaussianBlur(np.ascontiguousarray(img), (0, 0), p['blur'])
    if p['jpeg']:
        ok, buf = cv2.imencode('.jpg', cv2.cvtColor(np.ascontiguousarray(img), cv2.COLOR_RGB2BGR),
                               [cv2.IMWRITE_JPEG_QUALITY, p['jpeg']])
        img = cv2.cvtColor(cv2.imdecode(buf, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    return np.ascontiguousarray(img)


def read_clip(path, n_frames=EVAL_FRAMES, train=False, seed=None, aug=None, size=IMAGE_SIZE, sampling=None):
    """一个片段 → uint8 [n_frames, size, size, 3]，RGB，0~255。
    train=False：均匀取帧、不增强（验证 / 评测）。train=True：sampling='uniform' 均匀取帧 / 'tsn' 每段随机一帧
    （默认：取 12 帧时 uniform，否则 tsn）；aug 给了就按片段级增强。seed 决定这个样本的随机数（多线程下也可复现）。"""
    frames = read_frames(path)
    n = len(frames)
    if n < n_frames:
        raise ClipError('只有 %d 帧，不够取 %d 帧：%s' % (n, n_frames, path))
    rng = np.random.default_rng(seed)
    if sampling is None:
        sampling = 'uniform' if n_frames == EVAL_FRAMES else 'tsn'
    idx = tsn_indices(n, n_frames, rng) if (train and sampling == 'tsn') else eval_indices(n, n_frames)
    p = sample_aug(rng, frames[0].shape[0], frames[0].shape[1], aug) if (train and aug) else None
    out = np.empty((n_frames, size, size, 3), np.uint8)
    for k, i in enumerate(idx):
        rgb = cv2.cvtColor(frames[i], cv2.COLOR_BGR2RGB)
        out[k] = apply_aug(rgb, p, size) if p else to_model(rgb, size)
    return out


# ======================================================================== 样本
class Samples(object):
    """一个 split 的片段：paths（绝对路径）、rel（相对素材目录）、labels（0/1/2）、films（源视频名）。"""

    def __init__(self, data_dir, split, exclude=(), classes=CLASS_NAMES):
        data_dir = os.path.abspath(data_dir)
        ex = set(exclude or ())
        rel, labels = [], []
        self.n_excluded = 0
        for r, sp, c in D.list_clips(data_dir, splits=(split,), classes=classes):
            if r in ex:
                self.n_excluded += 1
                continue
            rel.append(r)
            labels.append(classes.index(c))
        self.split = split
        self.data_dir = data_dir
        self.rel = np.array(rel)
        self.paths = np.array([os.path.join(data_dir, *r.split('/')) for r in rel])
        self.labels = np.array(labels, np.int32)
        self.films = np.array([D.film_of(r) for r in rel])
        self.classes = list(classes)

    def __len__(self):
        return len(self.paths)

    def subset(self, idx):
        s = Samples.__new__(Samples)
        s.__dict__.update(self.__dict__)
        idx = np.asarray(idx)
        for k in ('rel', 'paths', 'labels', 'films'):
            setattr(s, k, getattr(self, k)[idx])
        return s

    def film_keys(self):
        """(类别, 源视频) —— 采样和按片子统计都以它为单位（同名片子在不同类别下算两部）。"""
        return np.array(['%d|%s' % (c, f) for c, f in zip(self.labels, self.films)])

    def describe(self):
        out = []
        for c, name in enumerate(self.classes):
            m = self.labels == c
            out.append('%s %d 个 / %d 部' % (name, int(m.sum()), len(set(self.films[m]))))
        s = '%s：%d 个片段（%s）' % (self.split, len(self), '，'.join(out))
        if self.n_excluded:
            s += '，剔除坏片段 %d 个' % self.n_excluded
        return s


def collect_samples(data_dir, split, bad_list=None, classes=CLASS_NAMES):
    """bad_list：bad_clips.txt 的路径（None = 不剔除）。"""
    ex = D.read_bad_list(bad_list) if bad_list else {}
    return Samples(data_dir, split, ex, classes)


class BalancedSampler(object):
    """类别 → 源视频 → 片段 均衡采样（见文件头）。"""

    def __init__(self, samples, alpha=0.5, seed=0):
        self.rng = np.random.default_rng(seed)
        self.alpha = alpha
        self.groups = []
        for c in range(len(samples.classes)):
            idx = np.where(samples.labels == c)[0]
            if not len(idx):
                continue
            by_film = OrderedDict()
            for i in idx:
                by_film.setdefault(samples.films[i], []).append(int(i))
            names = sorted(by_film)
            sizes = np.array([len(by_film[f]) for f in names], np.float64)
            w = sizes ** alpha
            self.groups.append(dict(cls=c, names=names, clips=[np.array(by_film[f]) for f in names],
                                    p=w / w.sum(), share=sizes / sizes.sum()))

    def draw(self):
        g = self.groups[int(self.rng.integers(len(self.groups)))]
        f = int(self.rng.choice(len(g['clips']), p=g['p']))
        clips = g['clips'][f]
        return int(clips[int(self.rng.integers(len(clips)))])

    def describe(self, classes=CLASS_NAMES, top=3):
        lines = ['采样：类别等权；类别内按 片段数^%.2g 选源视频' % self.alpha]
        for g in self.groups:
            order = np.argsort(-g['share'])[:top]
            lines.append('  %-11s %3d 部；片段最多的 %s' % (classes[g['cls']], len(g['names']), '，'.join(
                '%s 占片段 %.0f%% → 采样 %.0f%%' % (g['names'][i][:20], 100 * g['share'][i], 100 * g['p'][i])
                for i in order)))
        return '\n'.join(lines)


class ShuffleSampler(object):
    """旧脚本的做法：每轮把全部片段打乱、不放回地过一遍（类别靠 class_weight 平衡）。"""

    def __init__(self, n, seed=0):
        self.rng = np.random.default_rng(seed)
        self.n, self.order, self.k = n, None, n

    def draw(self):
        if self.k >= self.n:
            self.order, self.k = self.rng.permutation(self.n), 0
        self.k += 1
        return int(self.order[self.k - 1])

    def describe(self, *a, **k):
        return '采样：全部片段打乱，按片段数（不均衡，靠 class_weight）'


# ======================================================================== tf.data
class FailureLog(object):
    """训练时读不出来的片段：跳过这个样本、记下来、每个片段只报一次。"""

    def __init__(self, log=print):
        self.count = Counter()
        self.log = log

    def add(self, path, err):
        self.count[path] += 1
        if self.count[path] == 1:
            self.log('[读帧失败，跳过] %s（%s）。确认是坏片段就加进 bad_clips.txt' % (path, err))

    def total(self):
        return sum(self.count.values())


def train_dataset(samples, sampler, n_frames, batch, aug=AUG_MILD, size=IMAGE_SIZE, seed=0, workers=8,
                  sampling=None, failures=None):
    """无限的训练流：(float32 [B, T, S, S, 3], one-hot [B, 3])。配 model.fit(steps_per_epoch=...)。"""
    failures = failures or FailureLog()
    paths, labels = samples.paths, samples.labels
    seeds = np.random.default_rng(seed + 7919)      # 放在外面：Keras 换 epoch 时重建迭代器，也不会重复同一串种子

    def gen():
        while True:
            yield sampler.draw(), int(seeds.integers(0, 2 ** 31 - 1))

    def load(i, s):
        i = int(i)
        try:
            return read_clip(paths[i], n_frames, True, int(s), aug, size, sampling), labels[i], True
        except Exception as e:                       # noqa: BLE001  cv2 的错误也不能让训练停下
            failures.add(paths[i], e)
            return np.zeros((n_frames, size, size, 3), np.uint8), labels[i], False

    def tf_load(i, s):
        x, y, ok = tf.numpy_function(load, [i, s], [tf.uint8, tf.int32, tf.bool])
        x.set_shape([n_frames, size, size, 3])
        y.set_shape([])
        ok.set_shape([])
        return x, y, ok

    ds = tf.data.Dataset.from_generator(gen, output_signature=(tf.TensorSpec((), tf.int64),
                                                               tf.TensorSpec((), tf.int64)))
    ds = ds.map(tf_load, num_parallel_calls=workers, deterministic=False)
    ds = ds.filter(lambda x, y, ok: ok)
    ds = ds.map(lambda x, y, ok: (x, tf.one_hot(y, NUM_CLASSES)))
    ds = ds.batch(batch, drop_remainder=True)
    ds = ds.map(lambda x, y: (tf.cast(x, tf.float32), y))
    return ds.prefetch(2), failures


def eval_dataset(samples, n_frames=EVAL_FRAMES, batch=16, size=IMAGE_SIZE, workers=8):
    """按顺序读全部片段（不增强）：(float32 [B, T, S, S, 3], ok [B])。读不出来的 ok=False，算指标时去掉。"""
    paths = samples.paths

    def load(i):
        try:
            return read_clip(paths[int(i)], n_frames, False, None, None, size), True
        except Exception:                            # noqa: BLE001
            return np.zeros((n_frames, size, size, 3), np.uint8), False

    def tf_load(i):
        x, ok = tf.numpy_function(load, [i], [tf.uint8, tf.bool])
        x.set_shape([n_frames, size, size, 3])
        ok.set_shape([])
        return x, ok

    ds = tf.data.Dataset.from_tensor_slices(np.arange(len(samples), dtype=np.int64))
    ds = ds.map(tf_load, num_parallel_calls=workers, deterministic=True)
    ds = ds.batch(batch).map(lambda x, ok: (tf.cast(x, tf.float32), ok))
    return ds.prefetch(2)


def predict(model, ds):
    """→ (probs [N, 3], ok [N])。model 可以是 Keras 模型，也可以是 batch → probs 的函数。"""
    ps, oks = [], []
    for x, ok in ds:
        p = model.predict_on_batch(x) if hasattr(model, 'predict_on_batch') else model(x)
        ps.append(np.asarray(p, np.float32))
        oks.append(ok.numpy())
    if not ps:
        return np.zeros((0, NUM_CLASSES), np.float32), np.zeros(0, bool)
    return np.concatenate(ps), np.concatenate(oks)


# ======================================================================== 指标
def auc(scores, positive):
    """Mann-Whitney AUC（并列算一半），与 bench 的 metrics._auc 相同。"""
    scores = np.asarray(scores, np.float64)
    positive = np.asarray(positive, bool)
    n1 = int(positive.sum())
    n0 = len(scores) - n1
    if n1 == 0 or n0 == 0:
        return float('nan')
    _, inv, cnt = np.unique(scores, return_inverse=True, return_counts=True)
    avg = np.cumsum(cnt) - (cnt - 1) / 2.0
    r = avg[inv]
    return float((r[positive].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def clip_metrics(probs, labels, film_keys=None):
    """验证 / 测试集上的指标。p_do = p_oral + p_sex；app 判定：p_do ≥ 0.65 转，否则 p_plot ≥ 0.60 不转，否则 unclear。"""
    probs = np.asarray(probs, np.float64)
    labels = np.asarray(labels)
    n = len(labels)
    p_do = probs[:, 1] + probs[:, 2]
    y_do = labels != 0
    pred = probs.argmax(axis=1)
    cm = np.zeros((NUM_CLASSES, NUM_CLASSES), np.int64)
    for t, p in zip(labels, pred):
        cm[t, p] += 1
    m = dict(n=n, auc_do=auc(p_do, y_do), acc=float(np.mean(pred == labels)) if n else float('nan'),
             bin_acc=float(np.mean((p_do >= 0.5) == y_do)) if n else float('nan'),
             loss=float(np.mean(-np.log(np.clip(probs[np.arange(n), labels], 1e-7, 1.0)))) if n else float('nan'),
             cm=cm.tolist())
    app_do, app_plot = p_do >= APP_DO, (p_do < APP_DO) & (probs[:, 0] >= APP_PLOT)
    plot, act = ~y_do, y_do
    m['app_fp'] = float(app_do[plot].mean()) if plot.any() else float('nan')       # 剧情判成转
    m['app_fn'] = float(app_plot[act].mean()) if act.any() else float('nan')       # 动作判成不转
    m['app_unclear'] = float(np.mean(~app_do & ~app_plot)) if n else float('nan')
    m['conf99'] = float(np.mean(np.maximum(p_do, probs[:, 0]) >= 0.99)) if n else float('nan')
    if film_keys is not None and n:
        fk = np.asarray(film_keys)
        accs = [float(np.mean(pred[fk == k] == labels[fk == k])) for k in sorted(set(fk))]
        m['film_acc'] = float(np.mean(accs))
        m['film_acc_worst'] = float(np.min(accs))
        m['n_films'] = len(accs)
    return m


def fmt_metrics(m, prefix='val'):
    s = '%s：AUC(p_do) %.4f  准确率 %.1f%%' % (prefix, m['auc_do'], 100 * m['acc'])
    if 'film_acc' in m:
        s += '  按片子 %.1f%%（最差 %.0f%%）' % (100 * m['film_acc'], 100 * m['film_acc_worst'])
    s += '  loss %.3f  app 阈值：剧情判转 %.1f%% 动作判不转 %.1f%% unclear %.1f%%  置信≥0.99 %.0f%%' % (
        m['loss'], 100 * m['app_fp'], 100 * m['app_fn'], 100 * m['app_unclear'], 100 * m['conf99'])
    return s


class ValMetrics(keras.callbacks.Callback):
    """每个 epoch 结束时在验证集上算指标，写进 logs（val_auc_do、val_acc、val_film_acc、val_loss、val_bin_acc）。
    放在 ModelCheckpoint / EarlyStopping / ReduceLROnPlateau / CSVLogger 前面，它们才看得到这些值。"""

    def __init__(self, eval_model, samples, ds, prefix='val', log=print):
        super().__init__()
        self.eval_model, self.samples, self.ds, self.prefix, self.log = eval_model, samples, ds, prefix, log
        self.history = []
        self.film_keys = samples.film_keys()

    def evaluate(self):
        probs, ok = predict(self.eval_model, self.ds)
        m = clip_metrics(probs[ok], self.samples.labels[ok], self.film_keys[ok])
        m['n_bad'] = int((~ok).sum())
        return m, probs, ok

    def on_epoch_end(self, epoch, logs=None):
        t0 = time.time()
        m, _, _ = self.evaluate()
        self.history.append(m)
        if logs is not None:
            for k in ('auc_do', 'acc', 'film_acc', 'loss', 'bin_acc'):
                if k in m:
                    logs['%s_%s' % (self.prefix, k)] = m[k]
        self.log('  [epoch %d] %s%s（%.0f 秒）' % (epoch + 1, fmt_metrics(m, self.prefix),
                                               '  ⚠ %d 个片段读不出' % m['n_bad'] if m['n_bad'] else '',
                                               time.time() - t0))


class LrLogger(keras.callbacks.Callback):
    """把当前学习率写进 logs['lr']（CSVLogger 记下来）。"""

    def on_epoch_end(self, epoch, logs=None):
        if logs is not None:
            try:
                logs['lr'] = float(keras.ops.convert_to_numpy(self.model.optimizer.learning_rate))
            except Exception:                        # noqa: BLE001
                pass


# ======================================================================== 模型
@keras.saving.register_keras_serializable(package='xcup')
class FoldTime(keras.layers.Layer):
    """[B, T, H, W, C] → [B*T, H, W, C]"""

    def call(self, x):
        return keras.ops.reshape(x, (-1,) + tuple(x.shape[2:]))

    def compute_output_shape(self, s):
        return (None,) + tuple(s[2:])


@keras.saving.register_keras_serializable(package='xcup')
class UnfoldTime(keras.layers.Layer):
    """[B*T, F] → [B, T, F]"""

    def __init__(self, frames, **kw):
        super().__init__(**kw)
        self.frames = int(frames)

    def call(self, x):
        return keras.ops.reshape(x, (-1, self.frames, x.shape[-1]))

    def compute_output_shape(self, s):
        return (None, self.frames, s[-1])

    def get_config(self):
        c = super().get_config()
        c['frames'] = self.frames
        return c


BACKBONES = {'small': ('MobileNetV3Small', 'mobilenetv3small'), 'large': ('MobileNetV3Large', 'mobilenetv3large')}


class VideoNet(object):
    """backbone + 线性层，可以搭成不同帧数的训练模型和 12 帧的导出模型（共享同一套权重）。"""

    def __init__(self, backbone='small', image_size=IMAGE_SIZE, dropout=0.5, weights='imagenet',
                 num_classes=NUM_CLASSES):
        if backbone not in BACKBONES:
            raise ValueError('backbone 只能是 %s' % '/'.join(BACKBONES))
        self.kind, self.size = backbone, image_size
        cls = getattr(keras.applications, BACKBONES[backbone][0])
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')        # 160 的输入用 224 的 ImageNet 权重：Keras 会提示，旧脚本也是这样
            self.backbone = cls(input_shape=(image_size, image_size, 3), include_top=False, weights=weights,
                                pooling='avg', include_preprocessing=True)
        self.feat_dim = int(self.backbone.output.shape[-1])
        self.dropout = keras.layers.Dropout(dropout, name='dropout')
        self.head = keras.layers.Dense(num_classes, activation='softmax', dtype='float32', name='class_probs')
        self._clip = {}

    def clip_model(self, frames):
        """训练 / 验证用：[B, T, S, S, 3] → [B, 3]。T 帧合成一批过 backbone。"""
        if frames not in self._clip:
            s = self.size
            inp = keras.Input((frames, s, s, 3), name='video_input')
            x = FoldTime(name='fold_time')(inp)
            x = self.backbone(x)
            x = UnfoldTime(frames, name='unfold_time')(x)
            x = keras.layers.GlobalAveragePooling1D(name='temporal_avg_pool')(x)
            x = self.dropout(x)
            self._clip[frames] = keras.Model(inp, self.head(x), name='clip_model_%d' % frames)
        return self._clip[frames]

    def export_model(self, frames=EVAL_FRAMES):
        """与 app 现用模型同结构：TimeDistributed（帧数固定时 Keras 展开成 12 份 → TFLite 里 1 个 PACK）→ 时间平均 →
        线性层 → softmax。输入名 video_input、[None, 12, S, S, 3] float32。"""
        s = self.size
        inp = keras.Input((frames, s, s, 3), name='video_input')
        x = keras.layers.TimeDistributed(self.backbone, name='td_backbone')(inp)
        x = keras.layers.GlobalAveragePooling1D(name='temporal_avg_pool')(x)
        return keras.Model(inp, self.head(x), name='%s_tap_3class' % BACKBONES[self.kind][1])

    def frame_logits(self, x):
        """[N, S, S, 3] → 逐帧 logit [N, 3]（bench 的口径：窗口概率 = softmax(各帧 logit 的平均)）。"""
        f = np.asarray(self.backbone.predict_on_batch(np.asarray(x, np.float32)), np.float32)
        w, b = [np.asarray(v, np.float32) for v in self.head.get_weights()]
        return f @ w + b


def load_weights(model, path):
    """读 .weights.h5。文件里还存着当时的优化器状态，和现在编译的优化器对不上时 Keras 会提示跳过——只是优化器，模型权重照常读进来。"""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        model.load_weights(path)


def set_trainable(backbone, spec):
    """解冻 backbone 的哪些层。spec：
         none              整个 backbone 冻结（stage 1，只训线性层）
         last:N            最后 N 层（旧脚本的 --unfreeze_last_n）
         from:<层名前缀>    从第一个名字以它开头的层起，如 from:expanded_conv_6（MobileNetV3Small 的中段）；也可以写下标
         all               全部
       BN 层一律冻结（推理模式，用 ImageNet 的统计量；小 batch 微调更稳，训练模型和导出模型的数值也才一致）。
       → (解冻起点下标, 起点层名)"""
    layers = backbone.layers
    if spec == 'none':
        backbone.trainable = False
        return len(layers), ''
    if spec == 'all':
        start = 0
    elif spec.startswith('last:'):
        start = max(0, len(layers) - int(spec[5:]))
    elif spec.startswith('from:'):
        key = spec[5:]
        if key.isdigit():
            start = int(key)
        else:
            start = next((i for i, l in enumerate(layers) if l.name.startswith(key)), None)
            if start is None:
                raise ValueError('backbone 里没有名字以 %r 开头的层' % key)
    else:
        raise ValueError('unfreeze 的写法：none / last:N / from:<层名前缀> / all，不认识 %r' % spec)
    backbone.trainable = True
    for i, l in enumerate(layers):
        l.trainable = i >= start and not isinstance(l, keras.layers.BatchNormalization)
    return start, layers[start].name if start < len(layers) else ''


def n_params(weights):
    return int(sum(int(np.prod(v.shape)) for v in weights))


def make_optimizer(kind, lr, steps_per_epoch=1, epochs=1, schedule='constant', warmup_epochs=0.0, weight_decay=0.0):
    """kind：adam / adamw；schedule：constant / cosine（warmup_epochs 个 epoch 从 lr/100 线性升到 lr，再余弦降到 lr/100）。"""
    if schedule == 'cosine':
        warm = int(round(steps_per_epoch * warmup_epochs))
        total = max(1, steps_per_epoch * epochs)
        lr_s = keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=lr * 0.01 if warm else lr, decay_steps=max(1, total - warm), alpha=0.01,
            warmup_target=lr if warm else None, warmup_steps=warm)
    elif schedule == 'constant':
        lr_s = lr
    else:
        raise ValueError('schedule 只能是 constant / cosine')
    if kind == 'adamw':
        opt = keras.optimizers.AdamW(learning_rate=lr_s, weight_decay=weight_decay)
        opt.exclude_from_weight_decay(var_names=['bias'])
        return opt
    if kind == 'adam':
        return keras.optimizers.Adam(learning_rate=lr_s)
    raise ValueError('optimizer 只能是 adam / adamw')


def compile_model(model, optimizer, label_smoothing=0.0):
    model.compile(optimizer=optimizer,
                  loss=keras.losses.CategoricalCrossentropy(label_smoothing=label_smoothing),
                  metrics=[keras.metrics.CategoricalAccuracy(name='acc')])


# ======================================================================== 导出与核对
def interpreter_class():
    """TFLite 解释器：ai_edge_litert → tflite_runtime → tf.lite（与 bench 的 audio_rescore._interpreter_class 相同顺序）。"""
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
    return tf.lite.Interpreter


def _interp(path, threads=4, preserve=False):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        it = interpreter_class()(model_path=path, num_threads=threads, experimental_preserve_all_tensors=preserve)
    it.allocate_tensors()
    return it


def export_tflite(net, path, frames=EVAL_FRAMES, log=print):
    """12 帧导出模型 → SavedModel → float32 TFLite（只用 TFLITE_BUILTINS，与旧脚本相同）。→ Keras 导出模型。"""
    m = net.export_model(frames)
    work = tempfile.mkdtemp(prefix='xcup_export_')
    try:
        sm = os.path.join(work, 'saved_model')
        try:
            try:
                m.export(sm, verbose=False)
            except TypeError:
                m.export(sm)
        except Exception as e:                       # noqa: BLE001  老版本 Keras 没有 export
            log('[导出] model.export 失败（%s），改用 tf.saved_model.save' % e)
            tf.saved_model.save(m, sm)
        conv = tf.lite.TFLiteConverter.from_saved_model(sm)
        conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
        conv.optimizations = []
        data = conv.convert()
    finally:
        shutil.rmtree(work, ignore_errors=True)
    d = os.path.dirname(os.path.abspath(path))
    if not os.path.isdir(d):
        os.makedirs(d)
    with open(path, 'wb') as fh:
        fh.write(data)
    return m


def tflite_metadata(path):
    """tflite 文件里的 min_runtime_version、转换用的 TensorFlow 版本（读不出来就空着）。"""
    out = {}
    try:
        from tensorflow.lite.python import schema_py_generated as S
        with open(path, 'rb') as fh:
            buf = fh.read()
        m = S.Model.GetRootAsModel(buf, 0)
        for i in range(m.MetadataLength()):
            md = m.Metadata(i)
            name = md.Name().decode('utf-8', 'replace')
            b = m.Buffers(md.Buffer())
            data = bytes(b.DataAsNumpy()) if b.DataLength() else b''
            if name == 'min_runtime_version':
                out['min_runtime_version'] = data.split(b'\0')[0].decode('ascii', 'replace')
            elif name == 'CONVERSION_METADATA':
                v = re.findall(rb'(\d+\.\d+\.\d+)', data)
                if v:
                    out['converter_tf'] = v[-1].decode()
    except Exception:                                # noqa: BLE001
        pass
    return out


def _ver(s):
    try:
        return tuple(int(x) for x in s.split('.')[:2])
    except ValueError:
        return None


def tflite_structure(path, frames=EVAL_FRAMES, size=IMAGE_SIZE):
    """检查 tflite 是不是 bench 能用的结构：输入 [1, 12, S, S, 3] float32、输出 [1, 3]、1 个 PACK（12 个输入）、
    1 个 FULLY_CONNECTED、没有 Flex（SELECT_TF_OPS）算子。→ dict（ok、problems、ops、版本信息）。"""
    it = _interp(path)
    ins, outs = it.get_input_details(), it.get_output_details()
    ops = it._get_ops_details()
    cnt = Counter(o['op_name'] for o in ops)
    packs = [o for o in ops if o['op_name'] == 'PACK']
    fcs = [o for o in ops if o['op_name'] == 'FULLY_CONNECTED']
    probs = []
    if len(ins) != 1 or list(ins[0]['shape']) != [1, frames, size, size, 3] or ins[0]['dtype'] != np.float32:
        probs.append('输入不是 [1, %d, %d, %d, 3] float32：%s' % (frames, size, size,
                                                                [(list(i['shape']), i['dtype'].__name__) for i in ins]))
    if len(outs) != 1 or list(outs[0]['shape']) != [1, NUM_CLASSES]:
        probs.append('输出不是 [1, 3]：%s' % [list(o['shape']) for o in outs])
    if len(packs) != 1 or len(packs[0]['inputs']) != frames:
        probs.append('PACK 应为 1 个、%d 个输入：%s' % (frames, [len(p['inputs']) for p in packs]))
    if len(fcs) != 1:
        probs.append('FULLY_CONNECTED 应为 1 个，实际 %d 个' % len(fcs))
    flex = [k for k in cnt if k.startswith('Flex') or 'CUSTOM' in k]
    if flex:
        probs.append('有非内置算子：%s' % flex)
    meta = tflite_metadata(path)
    mrv = _ver(meta.get('min_runtime_version', ''))
    if mrv and mrv > APP_RUNTIME:
        probs.append('min_runtime_version %s 高于 app 的 TFLite %d.%d' % ((meta['min_runtime_version'],) + APP_RUNTIME))
    return dict(ok=not probs, problems=probs, ops=dict(cnt), size=os.path.getsize(path),
                n_ops=len(ops), **meta)


def keras_probs_cpu(model, x, batch=8):
    """在 CPU 上逐批直接调用 Keras 模型（eager，不走 XLA），float32。导出核对用它当参照：GPU 上 Ampere 及以后的卡
    （RTX 30/40 系）TensorFlow 默认用 TF32 算卷积和矩阵乘（只有约 3 位有效数字），再加上 Keras 在 GPU 上默认 XLA 编译，
    和 TFLite（CPU float32）比概率能差到 1e-2，那是参照不准，不是导出有问题。"""
    out = []
    with tf.device('/CPU:0'):
        for i in range(0, len(x), batch):
            out.append(np.asarray(model(x[i:i + batch], training=False), np.float32))
    return np.concatenate(out)


def check_tflite(keras_model, path, clips, atol=1e-4, log=print):
    """真实片段（[12, S, S, 3] uint8 的列表）上比 Keras 导出模型（CPU float32）和 TFLite：概率最大差、argmax 一致；
    再按 bench 的 video_rescore.VideoModel 的方法（读 PACK 的 12 个输入 × 线性层 → 逐帧 logit → 平均 → softmax）重算，
    应与 TFLite 的输出相同。有 GPU 时另给 GPU 上 Keras 的差（gpu_diff，TF32，只作参考，不参与判断）。→ dict。"""
    x = np.stack(clips).astype(np.float32)
    pk = keras_probs_cpu(keras_model, x)
    it = _interp(path, preserve=True)
    inp, out = it.get_input_details()[0]['index'], it.get_output_details()[0]['index']
    ops = it._get_ops_details()
    pack = [o for o in ops if o['op_name'] == 'PACK'][0]
    fc = [o for o in ops if o['op_name'] == 'FULLY_CONNECTED'][0]['inputs']
    W = it.get_tensor(int(fc[1])).astype(np.float32)
    b = it.get_tensor(int(fc[2])).astype(np.float32) if len(fc) > 2 and fc[2] >= 0 else np.zeros(W.shape[0], np.float32)
    pt, pd = [], []
    for xi in x:
        it.set_tensor(inp, xi[None])
        it.invoke()
        pt.append(it.get_tensor(out)[0].astype(np.float32))
        feats = np.stack([it.get_tensor(int(i))[0] for i in pack['inputs']]).astype(np.float32)
        z = (feats @ W.T + b).mean(axis=0)
        e = np.exp(z - z.max())
        pd.append(e / e.sum())
    pt, pd = np.stack(pt), np.stack(pd)
    r = dict(n=len(x), max_diff=float(np.abs(pk - pt).max()), bench_diff=float(np.abs(pd - pt).max()),
             argmax_agree=float(np.mean(pk.argmax(1) == pt.argmax(1))),
             do_agree=float(np.mean(((pk[:, 1] + pk[:, 2]) >= APP_DO) == ((pt[:, 1] + pt[:, 2]) >= APP_DO))),
             reference='keras_cpu_float32')
    if tf.config.list_physical_devices('GPU'):
        pg = np.asarray(keras_model.predict(x, batch_size=8, verbose=0), np.float32)
        r['gpu_diff'] = float(np.abs(pg - pt).max())
    r['ok'] = r['max_diff'] <= atol and r['bench_diff'] <= atol and r['argmax_agree'] == 1.0
    return r


def worst_films(probs, labels, film_keys, n=10, classes=CLASS_NAMES):
    """按片子（类别 + 源视频）的准确率从低到高 → [(类别名, 源视频, 片段数, 准确率, 平均 p_do, 判错时多判成哪类)]。
    整部片几乎全错的，先人工看一眼是不是标错 / 混进了别的内容。"""
    probs, labels, keys = np.asarray(probs), np.asarray(labels), np.asarray(film_keys)
    rows = []
    for k in sorted(set(keys)):
        m = keys == k
        c = int(labels[m][0])
        pred = probs[m].argmax(axis=1)
        wrong = Counter(int(x) for x in pred[pred != c])
        rows.append((float(np.mean(pred == c)), -int(m.sum()), classes[c], k.split('|', 1)[1], int(m.sum()),
                     float(np.mean(probs[m, 1] + probs[m, 2])),
                     classes[wrong.most_common(1)[0][0]] if wrong else ''))
    rows.sort()
    return [(cn, film, cnt, acc, pdo, w) for acc, _, cn, film, cnt, pdo, w in rows[:n]]


def fmt_worst(rows, title):
    out = ['  %s（按片子的准确率从低到高；整部几乎全错的先人工看一眼是不是标错）：' % title,
           '    ' + D.lpad('真值', 12) + D.lpad('源视频', 42) + D.rpad('片段', 6) + D.rpad('准确率', 8) +
           D.rpad('平均 p_do', 10) + '  多判成']
    for cn, film, cnt, acc, pdo, w in rows:
        out.append('    ' + D.lpad(cn, 12) + D.lpad(film[:40], 42) + D.rpad(str(cnt), 6) +
                   D.rpad('%.0f%%' % (100 * acc), 8) + D.rpad('%.3f' % pdo, 10) + '  ' + w)
    return '\n'.join(out)


def balanced_pick(samples, n, seed=0):
    """每类各取 n/3 个片段的下标（导出核对用）。"""
    rng = np.random.default_rng(seed)
    out = []
    for c in range(len(samples.classes)):
        idx = np.where(samples.labels == c)[0]
        out += list(rng.choice(idx, size=min(len(idx), max(1, n // len(samples.classes))), replace=False))
    return sorted(int(i) for i in out)


def versions():
    return OrderedDict([('python', sys.version.split()[0]), ('tensorflow', tf.__version__),
                        ('keras', keras.__version__), ('numpy', np.__version__), ('opencv', cv2.__version__)])


def setup_gpu(log=print):
    gpus = tf.config.list_physical_devices('GPU')
    for g in gpus:
        try:
            tf.config.experimental.set_memory_growth(g, True)
        except Exception as e:                       # noqa: BLE001
            log('[WARN] set_memory_growth：%s' % e)
    log('[INFO] GPU：%s' % (', '.join(g.name for g in gpus) if gpus else '没有，用 CPU 训练（会很慢）'))
    return gpus


# ======================================================================== 自检
def selftest():
    fails = []

    def check(ok, msg):
        print('  %s %s' % ('ok  ' if ok else 'FAIL', msg), flush=True)
        if not ok:
            fails.append(msg)

    keras.utils.set_random_seed(0)
    print('版本：%s' % ', '.join('%s %s' % kv for kv in versions().items()))
    # 1. 读帧、取帧、增强
    rng = np.random.default_rng(0)
    check(list(eval_indices(36)) == [0, 3, 6, 9, 12, 15, 19, 22, 25, 28, 31, 35], '12 帧均匀取帧与旧脚本相同')
    t = [tsn_indices(36, 4, rng) for _ in range(200)]
    check(all(9 * k <= x[k] < 9 * (k + 1) for x in t for k in range(4)) and len(set(int(x[0]) for x in t)) == 9,
          'TSN 取 4 帧：每段一帧，段内随机')
    wf = worst_films(np.array([[.9, .05, .05], [.2, .1, .7], [.1, .1, .8], [.1, .8, .1], [.6, .2, .2]]),
                     np.array([0, 0, 0, 1, 1]), np.array(['0|a', '0|b', '0|b', '1|c', '1|d']), 3)
    check([(r[1], r[2], r[3], r[5]) for r in wf] == [('b', 2, 0.0, 'sex'), ('d', 1, 0.0, 'normal_plot'),
                                                    ('a', 1, 1.0, '')],
          '按片子列出最差的几部（全错的在前，同样全错时片段多的在前）')
    check(abs(auc([0.9, 0.8, 0.7, 0.6], [1, 1, 0, 0]) - 1.0) < 1e-12 and abs(auc([0.5, 0.5], [1, 0]) - 0.5) < 1e-12
          and abs(auc([1.0, 0.8, 0.6, 0.7, 0.2], [1, 1, 1, 0, 0]) - 5 / 6.0) < 1e-12, 'AUC（与 metrics._auc 同口径）')
    tmp = tempfile.mkdtemp(prefix='train_utils_')
    try:
        data, src = D.make_synth_dataset(tmp)
        with open(os.path.join(data, 'train', 'oral', 'broken_s0.0_oral.mp4'), 'wb') as fh:
            fh.write(os.urandom(4000))
        D.scan(data, jobs=2, quiet=True)
        bad = os.path.join(data, D.BAD_LIST)
        tr, va = collect_samples(data, 'train', bad), collect_samples(data, 'val', bad)
        check(tr.n_excluded == 1 and len(tr) == 24 and len(va) == 9 and sorted(set(tr.films)) == ['filmA', 'filmB',
                                                                                                'filmC', 'filmD'],
              'collect_samples：剔除 bad_clips.txt 里的片段，源视频名从文件名取（%s；%s）' % (tr.describe(), va.describe()))
        p0 = tr.paths[0]
        x1 = read_clip(p0)
        x2 = read_clip(p0)
        check(x1.shape == (12, 160, 160, 3) and x1.dtype == np.uint8 and np.array_equal(x1, x2), '验证读帧：确定、形状对')
        # 旧脚本的读法（CAP_PROP_POS_FRAMES 跳帧 + INTER_AREA），对照一下
        cap = cv2.VideoCapture(p0)
        old = []
        for i in eval_indices(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
            ok, f = cap.read()
            old.append(cv2.resize(cv2.cvtColor(f, cv2.COLOR_BGR2RGB), (160, 160), interpolation=cv2.INTER_AREA))
        cap.release()
        print('  info 顺序读帧与旧脚本跳帧读的差：平均 %.2f（0 = 跳帧是准的）' % np.abs(x1.astype(int) - np.stack(old)).mean())
        flip = dict(crop=(0, 0, 224, 224), flip=True, b=1.0, c=1.0, s=1.0, blur=0.0, jpeg=0)
        frames = read_frames(p0)
        rgb = cv2.cvtColor(frames[0], cv2.COLOR_BGR2RGB)
        check(np.array_equal(apply_aug(rgb, flip), to_model(rgb)[:, ::-1]), '增强：只翻转时 = 缩放后左右翻转')
        same = [np.ascontiguousarray(rgb)] * 5
        consistent = True
        for s in range(20):
            p = sample_aug(np.random.default_rng(s), 224, 224)
            outs = [apply_aug(f, p) for f in same]
            consistent &= all(np.array_equal(o, outs[0]) for o in outs)
        check(consistent, '增强：同一片段的帧用同一组参数（相同的帧增强后仍相同）')
        crops = [sample_aug(np.random.default_rng(s), 224, 224)['crop'] for s in range(500)]
        areas = [cw * ch / (224.0 * 224) for _, _, cw, ch in crops]
        ratios = [cw / float(ch) for _, _, cw, ch in crops]
        check(min(areas) >= 0.69 and min(ratios) >= 0.74 and max(ratios) <= 1.34 and max(areas) > 0.97,
              '增强：裁剪面积 %.2f~%.2f（≥0.7），宽高比 %.2f~%.2f' % (min(areas), max(areas), min(ratios), max(ratios)))
        xa = read_clip(p0, 12, True, 5, AUG_MILD)
        xb = read_clip(p0, 12, True, 5, AUG_MILD)
        xc = read_clip(p0, 4, True, 6, AUG_MILD)
        check(np.array_equal(xa, xb) and not np.array_equal(xa, x1) and xc.shape == (4, 160, 160, 3),
              '训练读帧：同一个 seed 结果相同（多线程下可复现），4 帧形状对')
        # 2. 采样
        s0 = BalancedSampler(tr, alpha=0.0, seed=1)
        draws = [s0.draw() for _ in range(30000)]
        cls = Counter(int(tr.labels[i]) for i in draws)
        film = Counter(tr.films[i] for i in draws if tr.labels[i] == 0)
        check(all(abs(cls[c] / 30000.0 - 1 / 3.0) < 0.02 for c in range(3))
              and abs(film['filmA'] / float(sum(film.values())) - 0.5) < 0.03,
              '均衡采样：类别各 1/3，alpha=0 时类别内每部片等权（%s；剧情 %s）' % (dict(cls), dict(film)))
        sub = tr.subset([i for i in range(len(tr)) if tr.films[i] != 'filmB'      # filmB 只留 2 个片段，filmA 7 个
                         or tr.rel[i].endswith(('_s0.0_normal_plot.mp4', '_s1.5_normal_plot.mp4'))])
        s1, s5 = BalancedSampler(sub, alpha=1.0, seed=2), BalancedSampler(sub, alpha=0.5, seed=2)
        fa = Counter(sub.films[s1.draw()] for _ in range(30000))
        n_a, n_b = int(np.sum(sub.films == 'filmA')), int(np.sum(sub.films == 'filmB'))
        ga = [g for g in s5.groups if g['cls'] == 0][0]
        check(abs(fa['filmA'] / float(fa['filmA'] + fa['filmB']) - n_a / float(n_a + n_b)) < 0.02
              and abs(ga['p'][ga['names'].index('filmA')] - math.sqrt(n_a) / (math.sqrt(n_a) + math.sqrt(n_b))) < 1e-9,
              '均衡采样：alpha=1 按片段数，alpha=0.5 按片段数开方（filmA %d 个、filmB %d 个）' % (n_a, n_b))
        # 3. tf.data：坏片段跳过、不停
        fake = tr.subset(np.arange(len(tr)))
        fake.paths = fake.paths.copy()
        fake.paths[0] = os.path.join(data, 'train', 'oral', 'broken_s0.0_oral.mp4')
        msgs = []
        ds, fl = train_dataset(fake, ShuffleSampler(len(fake), 0), 4, 4, AUG_MILD, seed=0, workers=2,
                               failures=FailureLog(msgs.append))
        shapes = [(tuple(x.shape), tuple(y.shape)) for x, y in ds.take(10)]
        check(shapes == [((4, 4, 160, 160, 3), (4, 3))] * 10 and fl.total() >= 1 and len(msgs) == 1,
              '训练流：形状对、one-hot；读不出的片段跳过并只报一次（%s）' % (msgs[:1],))
        # 4. 模型：训练模型（合成一批）与导出模型（TimeDistributed）共享权重、数值一致
        net = VideoNet('small', weights=None)
        m4, m12, mex = net.clip_model(4), net.clip_model(12), net.export_model()
        xv = np.stack([read_clip(p) for p in va.paths[:3]]).astype(np.float32)
        p12 = m12.predict(xv, verbose=0)
        pex = mex.predict(xv, verbose=0)
        rep = np.repeat(xv[:, :1], 4, axis=1)
        check(np.abs(p12 - pex).max() < 1e-5 and np.abs(m4.predict(rep, verbose=0)
                                                       - m12.predict(np.repeat(xv[:, :1], 12, axis=1), verbose=0)).max()
              < 1e-5, '训练模型（合成一批过 backbone）= 导出模型（逐帧），4 帧 / 12 帧共享权重（差 %.1e）'
              % np.abs(p12 - pex).max())
        lg = net.frame_logits(xv[0])
        z = lg.mean(0)
        check(np.abs(np.exp(z - z.max()) / np.exp(z - z.max()).sum() - p12[0]).max() < 1e-5,
              '窗口概率 = softmax(逐帧 logit 的平均)（bench 的口径）')
        n_bb = len(net.backbone.layers)
        st, name = set_trainable(net.backbone, 'last:30')
        tw = [l for l in net.backbone.layers if l.trainable and l.weights]
        check(st == n_bb - 30 and all(not isinstance(l, keras.layers.BatchNormalization) for l in tw) and tw,
              'set_trainable last:30：从第 %d 层（%s）起，BN 冻结' % (st, name))
        st, name = set_trainable(net.backbone, 'from:expanded_conv_6')
        check(name.startswith('expanded_conv_6') and not any(l.trainable for l in net.backbone.layers[:st]),
              'set_trainable from:expanded_conv_6：从第 %d 层（%s）起' % (st, name))
        set_trainable(net.backbone, 'all')
        n_all = n_params(net.backbone.trainable_weights)
        set_trainable(net.backbone, 'none')
        check(n_params(net.backbone.trainable_weights) == 0 and n_all > 0, 'set_trainable none / all')
        # 5. 训练几步：ValMetrics 写进 logs，ModelCheckpoint 按 val_auc_do 存、权重能读回
        out = os.path.join(tmp, 'run')
        os.makedirs(out)
        set_trainable(net.backbone, 'last:30')
        tds, _ = train_dataset(tr, BalancedSampler(tr, 0.5, 0), 4, 2, AUG_MILD, seed=0, workers=2)
        vds = eval_dataset(va, batch=4, workers=2)
        compile_model(m4, make_optimizer('adamw', 1e-3, 3, 2, 'cosine', 1.0, 0.004), label_smoothing=0.05)
        ck = os.path.join(out, 'best.weights.h5')
        vm = ValMetrics(m12, va, vds, log=lambda *a: None)
        hist = m4.fit(tds, epochs=2, steps_per_epoch=3, verbose=0, shuffle=False, callbacks=[
            vm, keras.callbacks.ModelCheckpoint(ck, monitor='val_auc_do', mode='max', save_best_only=True,
                                                save_weights_only=True),
            keras.callbacks.EarlyStopping(monitor='val_auc_do', mode='max', patience=5, restore_best_weights=True),
            LrLogger(), keras.callbacks.CSVLogger(os.path.join(out, 'log.csv'))])
        head = io.open(os.path.join(out, 'log.csv'), encoding='utf-8').readline()
        check(os.path.exists(ck) and 'val_auc_do' in head and 'lr' in head and len(vm.history) == 2
              and 'val_auc_do' in hist.history, 'fit：val_auc_do 进了 logs / CSV，按它存了最好的权重')
        net2 = VideoNet('small', weights=None)
        net2.clip_model(4).load_weights(ck)
        m4.load_weights(ck)
        check(np.abs(net2.clip_model(12).predict(xv, verbose=0) - m12.predict(xv, verbose=0)).max() < 1e-6,
              '4 帧模型存的权重读进新建的模型，12 帧输出相同')
        # 6. 导出 TFLite：结构、与 Keras 一致、bench 的逐帧 logit 算法
        tfl = os.path.join(out, 'm.tflite')
        mex = export_tflite(net, tfl, log=lambda *a: None)
        st = tflite_structure(tfl)
        check(st['ok'], 'TFLite 结构：输入 [1,12,160,160,3]、1 个 PACK（12 个输入）、1 个 FULLY_CONNECTED（%s）'
              % (st['problems'] or 'TF %s 转换，min_runtime %s' % (st.get('converter_tf', '?'),
                                                               st.get('min_runtime_version', '?'))))
        clips = [read_clip(p) for p in va.paths[balanced_pick(va, 6)]]
        r = check_tflite(mex, tfl, clips)
        check(r['ok'], 'Keras / TFLite 一致：最大差 %.1e，bench 算法重算差 %.1e，argmax 一致 %.0f%%'
              % (r['max_diff'], r['bench_diff'], 100 * r['argmax_agree']))
        bench = os.path.join(os.path.dirname(HERE), 'bench', 'tools')
        if os.path.exists(os.path.join(bench, 'video_rescore.py')):
            sys.path.insert(0, bench)
            import video_rescore as V
            vm2 = V.VideoModel(tfl)
            probs, lg2 = vm2.run_window(list(clips[0]))
            check(np.abs(probs - V.window_probs(lg2)).max() < 1e-5 and np.abs(probs - mex.predict(
                clips[0][None].astype(np.float32), verbose=0)[0]).max() < 1e-4,
                  'bench 的 video_rescore.VideoModel 能直接读这个 tflite（baseline.py rescore --model 可用）')
        else:
            print('  skip bench 兼容性（没找到 bench/tools/video_rescore.py）')
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    if fails:
        print('selftest: %d 项失败' % len(fails))
        return 1
    print('selftest: ALL PASS')
    return 0


if __name__ == '__main__':
    if '--selftest' in sys.argv:
        sys.exit(selftest())
    print(__doc__)
