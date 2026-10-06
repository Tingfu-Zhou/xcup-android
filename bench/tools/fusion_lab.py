#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
bench/tools/fusion_lab.py — 融合层离线实验台

run CSV 里已经存了两个模型每个 tick 的原始概率（v_p_* / a_p_*），融合层的全部输入都在。
这里把融合层整个搬到 Python 里离线重放：改融合不用再实时重跑视频，几秒一部片。

子命令:
  check   复刻线上 Java 融合（平滑投票 + 启动仲裁 + 蓝牙状态机），和 run CSV 里记下的
          逐 tick 输出比对。一致率 >=99% 才说明回放器可信，后面的对比才有意义。
  eval    留一视频交叉验证：证据融合 vs 现有融合 vs 三条基线（一直转/只用视频/只用音频）。
          每部片的参数只用其余几部拟合，数字代表"没见过的片子"上的表现。
  oracle  假设某个模态完美（概率换成真值），端到端最多能到多少——决定先改哪个模型。

用法:
  python3 bench/tools/fusion_lab.py check  bench/gt bench/runs
  python3 bench/tools/fusion_lab.py eval   bench/gt bench/runs
  python3 bench/tools/fusion_lab.py oracle bench/gt bench/runs
  python3 bench/tools/fusion_lab.py --selftest

真值与 run 的配对：bench/gt/v01.csv 配 bench/runs/ 下以 "v01" 开头的文件（如 v01_base.csv）；
同一部片有多份时优先 <id>_base.csv。也可以用 --pair gt.csv=run.csv 显式指定（可重复）。

证据融合（只在这里离线实验、和现有融合对比。app 里的 EvidenceFusion.java 与 fusion_config.json
已于 2026-10-06 删除，app 只用现有融合；要重新接入见 git 历史）:
  证据：每个模型每出一个新结果，取特征 s（视频 p_oral+p_sex，音频 p_sex+p_oral 或 p_sex），
        查表得 LLR = log P(s 落在该区间 | 动作) / P(s 落在该区间 | 剧情)。表从 bench 统计。
  累积：L = clamp(L + w_v*LLR_v + w_a*LLR_a, -clamp_lo, +clamp_hi)
  决策：不转 -> 转：L >= theta_on；转 -> 不转：L <= theta_off；之后照旧走蓝牙状态机。
  上下界分开：clamp_lo 管"剧情里信念压多深"（越小启动越快），clamp_hi 管"动作里攒多少余量"
  （越大越扛得住遮挡/镜头掉线，但动作结束后停得越慢）。
  每个模态"说话算多少"由统计出来的 LLR 决定，不再手写"音频权重 1.2"。换了模型重新拟合即可。
"""
import argparse
import glob
import io
import json
import math
import os
import random
import re
import statistics
import struct
import sys
import tempfile
import time
from collections import namedtuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import metrics as M  # noqa: E402

GRID_MS = M.GRID_MS
MAX_AGE_MS = 2000          # 融合循环的新鲜度窗（VideoProcessActivity 融合循环里的 MAX_AGE）
VIDEO_WIN_MS = 3000        # 视频模型窗口：12 帧 x 250ms
AUDIO_WIN_MS = 2000        # 音频模型窗口：32000 点 @16kHz

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

_F32 = struct.Struct('<f')


def f32(x):
    """舍入成 Java float。现有融合全是 float 运算，阈值比较要按单精度复刻才能逐 tick 对得上。"""
    return _F32.unpack(_F32.pack(x))[0]


F01, F07, F08, F12 = f32(0.1), f32(0.7), f32(0.8), f32(1.2)


# ======================================================================== 读数据
class Tick(object):
    """run CSV 的一行。v/a 是融合循环实际用到的动作串（已过新鲜度过滤），vp/ap 是原始概率。"""
    __slots__ = ('pos', 'wall', 'v', 'vc', 'v_age', 'vp', 'a', 'ac', 'a_age', 'ap',
                 'loud_lv', 'loud_conf', 'log')

    def copy(self, **kw):
        t = Tick()
        for k in Tick.__slots__:
            setattr(t, k, kw[k] if k in kw else getattr(self, k))
        return t


LOG_COLS = ('smoothed', 'gated', 'final_freq', 'ble_state', 'ble_level', 'sent',
            'sent_action', 'sent_level')


def _fnum(s):
    s = (s or '').strip()
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _inum(s, dv):
    v = _fnum(s)
    return dv if v is None else int(v)


def _probs(d, keys):
    vals = [_fnum(d.get(k)) for k in keys]
    if any(v is None for v in vals):
        return None
    return tuple(f32(v) for v in vals)


def load_run(path):
    meta, data = M._rows(path)
    ticks = []
    for d in data:
        t = Tick()
        t.pos = _inum(d.get('pos_ms'), 0)
        t.wall = _inum(d.get('wall_ms'), 0)
        t.v = (d.get('v_action') or '').strip()
        t.vc = f32(_fnum(d.get('v_conf')) or 0.0)
        t.v_age = _inum(d.get('v_age'), -1)
        t.vp = _probs(d, ('v_p_plot', 'v_p_oral', 'v_p_sex'))
        t.a = (d.get('a_action') or '').strip()
        t.ac = f32(_fnum(d.get('a_conf')) or 0.0)
        t.a_age = _inum(d.get('a_age'), -1)
        # 与 app 音频模型的输出一致：只有 [sex, oral, noise]（rescore 写的 a_ys_* 列只做离线分析）
        t.ap = _probs(d, ('a_p_sex', 'a_p_oral', 'a_p_noise'))
        t.loud_lv = _inum(d.get('loud_lv'), -1)
        t.loud_conf = f32(_fnum(d.get('loud_conf')) or 0.0)
        t.log = dict((k, (d.get(k) or '').strip()) for k in LOG_COLS)
        ticks.append(t)
    return meta, ticks       # 保持文件顺序（= 时间顺序），回放必须按真实先后


class Video(object):
    def __init__(self, vid, segs, dur, ticks, params, gt_path='', run_path='',
                 synthetic=False, policy='exclude', splice=None):
        self.vid = vid
        self.splice = splice            # None=真实片子；'stop'/'start'=拼接样本
        self.fusion = ''                # run 参数行里的 FUSION=<mode>/<配置 id>（只有 app 带证据融合那几版录的 run 有）
        self.audio_pipe = 'legacy'      # 这份 run 的音频是哪条管线出的（audio_pipe_of）
        self.rescored = False           # audio_rescore.py 离线重算出来的 run
        self.segs = segs
        self.dur = dur
        self.ticks = ticks
        self.params = params
        self.gt_path = gt_path
        self.run_path = run_path
        self.synthetic = synthetic
        self.policy = policy
        ps = sorted(t.pos for t in ticks)
        self.lo = max(M.WARMUP_MS, ps[0]) if ps else M.WARMUP_MS
        self.hi = ps[-1] if ps else 0
        self.gti = GtIndex(segs, dur, self.lo, self.hi, policy)

    def with_ticks(self, ticks, vid=None):
        return Video(vid or self.vid, self.segs, self.dur, ticks, self.params, self.gt_path,
                     self.run_path, self.synthetic, self.policy)


def _pick_run(stem, runs, video_name):
    def ok(r):
        b = os.path.basename(r)
        return re.match(re.escape(stem) + r'([_.\-]|$)', b) is not None
    cands = [r for r in runs if ok(r)]
    if not cands and video_name:            # 没改名的原始 dump：按文件头里的 video_name 配
        cands = [r for r in runs if _run_video_name(r) == video_name]
    if not cands:
        return None, ''
    base = [r for r in cands if os.path.basename(r) == stem + '_base.csv']
    if base:
        return base[0], ''
    cands.sort()
    note = '（%d 份候选，取 %s；可用 --pair 指定）' % (len(cands), os.path.basename(cands[-1])) \
        if len(cands) > 1 else ''
    return cands[-1], note


def _run_video_name(path):
    try:
        with io.open(path, encoding='utf-8-sig') as fh:
            for _ in range(10):
                line = fh.readline()
                if line.startswith('# video_name='):
                    return line.split('=', 1)[1].strip()
    except (IOError, OSError):
        pass
    return None


def _sanitize(s):
    """与 BenchLogger.sanitize 一致：中文片名保留，其余非字母数字换成 _。"""
    out = ''.join(c if (c.isalnum() or c in '-_') else '_' for c in s)[:48]
    return out or 'run'


def audio_pipe_of(meta):
    """run 的音频是哪条管线出的：app 参数行的 AUDIO_PIPE（legacy/fixed/web），audio_rescore.py 重算的看
    头部 pipeline（phone 就是旧版 app 的 legacy），都没有就是旧版 app（legacy）。"""
    m = re.search(r'\bAUDIO_PIPE=(\S+)', meta.get('params', ''))
    if m:
        return m.group(1)
    p = meta.get('pipeline', '')
    return 'legacy' if p in ('', 'phone') else p


def load_videos(gt_dir, runs_dir, pairs=(), policy='exclude', quiet=False):
    say = (lambda *a: None) if quiet else print
    todo = []
    if pairs:
        for p in pairs:
            if '=' not in p:
                sys.exit('--pair 格式应为 gt.csv=run.csv: %r' % p)
            g, r = p.split('=', 1)
            todo.append((g, r, ''))
    else:
        if not os.path.isdir(gt_dir):
            sys.exit('找不到真值目录: %s' % gt_dir)
        gts = sorted(glob.glob(os.path.join(gt_dir, '*.csv')))
        runs = sorted(glob.glob(os.path.join(runs_dir, '*.csv'))) if os.path.isdir(runs_dir) else []
        if not runs:
            sys.exit('在 %s 下没找到 run CSV。\n'
                     '  · 把手机里导出的 run_*.csv 拷到 bench/runs/，文件名以真值 id 开头（如 v01_base.csv）\n'
                     '  · bench/runs/ 已允许提交进 git：离线实验的结论要能从仓库里的数据复现'
                     % runs_dir)
        for g in gts:
            gmeta, _, _, _ = M.read_gt(g)
            stem = os.path.splitext(os.path.basename(g))[0]
            vname = _sanitize(os.path.splitext(gmeta.get('file_name', ''))[0])
            r, note = _pick_run(stem, runs, vname)
            if r is None:
                say('  - %s: 没有对应的 run CSV，跳过' % stem)
                continue
            todo.append((g, r, note))
    videos = []
    for g, r, note in todo:
        gmeta, segs, dur, errs = M.read_gt(g)
        if errs:
            say('  ✗ %s 真值格式有误，跳过: %s' % (g, '; '.join(errs)))
            continue
        rmeta, ticks = load_run(r)
        if len(ticks) < 20:
            say('  ✗ %s 数据太少，跳过' % r)
            continue
        back = sum(1 for i in range(1, len(ticks)) if ticks[i].pos - ticks[i - 1].pos < -1000)
        if back:
            say('  ✗ %s 有 %d 处 pos 回退（seek/重播），回放不可信，跳过' % (r, back))
            continue
        vid = os.path.splitext(os.path.basename(g))[0]
        videos.append(Video(vid, segs, dur, ticks, parse_params(rmeta.get('params', '')),
                            g, r, policy=policy))
        m = re.search(r'\bFUSION=(\S+)', rmeta.get('params', ''))
        videos[-1].fusion = m.group(1) if m else ''
        videos[-1].audio_pipe = audio_pipe_of(rmeta)
        videos[-1].rescored = 'rescored_by' in rmeta
        say('  ✓ %-5s %-28s %5d tick%s%s' % (vid, os.path.basename(r), len(ticks), note,
                                            '' if videos[-1].audio_pipe == 'legacy'
                                            else '  [音频管线 %s]' % videos[-1].audio_pipe))
    if not videos:
        sys.exit('没有可用的 真值+run 配对')
    return videos


# ======================================================================== 现有融合（逐行复刻）
LEGACY_DEFAULTS = {
    'VIDEO_DO': 0.65, 'VIDEO_PLOT': 0.60, 'AUDIO_DO': 0.55, 'AUDIO_NOISE': 0.60,
    'SMOOTH_WIN': 6, 'SMOOTH_MIN': 3, 'FASTPATH_V': 0.70, 'FASTPATH_A': 0.70,
    'PLOT_CONFLICT': 0.80, 'CONFLICT_MARGIN': 0.15, 'CONFLICT_HOLD': 4,
    'AUDIO_STRONG': 0.75, 'AUDIO_ONLY_TICKS': 2,
    'BLE_MIN_DUR': 2000, 'BLE_GAP': 1600, 'LEVEL_STABLE': 0, 'LEVEL_MIN_DUR': 0,
}


def parse_params(line):
    """run CSV 文件头的参数行 -> dict。缺的键用当前 Java 默认值。"""
    p = dict(LEGACY_DEFAULTS)
    for kv in (line or '').split():
        if '=' not in kv:
            continue
        k, v = kv.split('=', 1)
        if k in p:
            try:
                p[k] = int(float(v)) if isinstance(LEGACY_DEFAULTS[k], int) else float(v)
            except ValueError:
                pass
    return p


def _is_sex(a):
    return a.startswith('do') or a.startswith('oral')


def _jhash(s):
    h = 0
    for ch in s:
        h = (31 * h + ord(ch)) & 0xFFFFFFFF
    return h


def _java_hashmap_order(keys):
    """Java HashMap<String,?>（容量 16）的遍历顺序。平局时谁先被遍历谁赢，所以顺序要一致。"""
    def bucket(k):
        h = _jhash(k)
        return (h ^ (h >> 16)) & 15
    return sorted(keys, key=bucket)      # sorted 稳定：同桶保持插入顺序


class LegacyCore(object):
    """VideoProcessActivity 的 smoothedFusion + applyDoStartGate + selectBestAction（float 语义）。"""

    def __init__(self, p):
        self.win = p['SMOOTH_WIN']
        self.min_count = p['SMOOTH_MIN']
        self.fast_v = f32(p['FASTPATH_V'])
        self.fast_a = f32(p['FASTPATH_A'])
        self.plot_conflict = f32(p['PLOT_CONFLICT'])
        self.margin = f32(p['CONFLICT_MARGIN'])
        self.hold_max = p['CONFLICT_HOLD']
        self.audio_strong = f32(p['AUDIO_STRONG'])
        self.audio_only_ticks = p['AUDIO_ONLY_TICKS']
        self.hist = []
        self.streak = 0
        self.hold = 0

    @staticmethod
    def select_best(v, a, vc, ac):
        if a != '':
            if a != 'Noise' or ac > F07:
                return a
        if v != '' and v != 'Background':
            if v != 'Noise' or vc > F07:
                return v
        if a == 'Noise' or v == 'Noise':
            return 'Noise'
        return ''

    def smoothed(self, v, a, vc, ac):
        self.hist.append((v, a, vc, ac))
        while len(self.hist) > self.win:
            self.hist.pop(0)
        if v != '' and v == a and vc >= self.fast_v and ac >= self.fast_a:
            return v
        if len(self.hist) < 3:
            return self.select_best(v, a, vc, ac)
        scores, counts = {}, {}
        n = len(self.hist)
        for idx, (rv, ra, rvc, rac) in enumerate(self.hist):
            w = f32((idx + 1) / float(n))
            if rv != '' and rv != 'Background':
                scores[rv] = f32(scores.get(rv, 0.0) + f32(f32(rvc * w) * F08))
                counts[rv] = counts.get(rv, 0) + 1
            if ra != '':
                scores[ra] = f32(scores.get(ra, 0.0) + f32(f32(rac * w) * F12))
                counts[ra] = counts.get(ra, 0) + 1
        best, best_score = '', 0.0
        for k in _java_hashmap_order(list(scores.keys())):
            if counts[k] >= self.min_count and scores[k] > best_score:
                best_score, best = scores[k], k
        if best == '':
            lv, la, lvc, lac = self.hist[-1]
            best = self.select_best(lv, la, lvc, lac)
        return best

    def gate(self, final, v, vc, a, ac, ble_state):
        if final != 'do':
            self.streak = 0
            self.hold = 0
            return final
        if _is_sex(ble_state):
            self.streak = 0
            self.hold = 0
            return final
        a_do = ac if a == 'do' else 0.0
        strong = v == 'Noise' and vc >= self.plot_conflict and f32(vc - a_do) >= self.margin
        if strong:
            self.hold += 1
            if self.hold <= self.hold_max:
                return ''
        else:
            self.hold = 0
        if a_do >= self.audio_strong:
            self.streak = 0
            return final
        for rv, _, _, _ in self.hist:
            if rv == 'do':
                self.streak = 0
                return final
        self.streak += 1
        if self.streak >= self.audio_only_ticks:
            return final
        return ''


class Ble(object):
    """updateBluetoothState 的复刻：待确认 / 最短驻留 / 发送间隔 / 档位重发（时间轴用墙钟）。"""
    STOP_STABLE_MS = 1000

    def __init__(self, p):
        self.min_dur = p['BLE_MIN_DUR']
        self.gap = p['BLE_GAP']
        self.level_stable = p['LEVEL_STABLE']
        self.level_min_dur = p['LEVEL_MIN_DUR']
        self.last_send = 0
        self.cur = ''
        self.cur_start = 0
        self.pend = ''
        self.pend_start = 0
        self.level = 1
        self.level_since = 0
        self.pend_level = None
        self.pend_level_since = 0
        self.last_sent_level = 0

    def update(self, new, freq, now):
        """返回本次发出的 (动作, 档位)，没发返回 None。"""
        supports = _is_sex(new)
        if supports:
            latest = max(0, min(8, freq))
            worth = latest >= min(10, self.level + 1) or latest <= max(0, self.level - 1)
            if worth:
                if self.level_stable <= 0 and now - self.level_since >= self.level_min_dur:
                    self.level = latest
                    self.level_since = now
                    self.pend_level = None
                    self.pend_level_since = 0
                elif self.pend_level is None or self.pend_level != latest:
                    self.pend_level = latest
                    self.pend_level_since = now
                elif (now - self.pend_level_since >= self.level_stable
                      and now - self.level_since >= self.level_min_dur):
                    self.level = self.pend_level
                    self.level_since = now
                    self.pend_level = None
                    self.pend_level_since = 0
            else:
                self.pend_level = None
                self.pend_level_since = 0
        else:
            self.pend_level = None
            self.pend_level_since = 0

        if new != self.pend:
            self.pend = new
            self.pend_start = now
        stable = self.STOP_STABLE_MS if (_is_sex(self.cur) and new == 'Noise') else 0
        if now - self.pend_start < stable:
            return None

        if self.pend != self.cur:
            if not (self.cur == '' or now - self.cur_start >= self.min_dur):
                return None
            if now - self.last_send < self.gap:
                return None
            lvl = self.level if _is_sex(self.pend) else 0
            self.cur = self.pend
            self.cur_start = now
            self.last_send = now
            if _is_sex(self.cur):
                self.level = lvl
                self.level_since = now
                self.last_sent_level = lvl
            else:
                self.last_sent_level = 0
            return (self.cur, lvl)

        if supports and self.level != self.last_sent_level and now - self.last_send >= self.gap:
            self.last_sent_level = self.level
            self.last_send = now
            return (self.cur, self.level)
        return None


def final_freq(loud_lv, loud_conf, cur_level):
    """computeFinalFreq 的复刻。run CSV 里 loud_lv=-1 表示 NaN（过期/无效）。"""
    nan = loud_lv < 0
    ff = 0 if nan else max(0, min(8, loud_lv))
    if nan or loud_conf < F01:
        return cur_level
    if ff > cur_level and loud_conf < F01:
        return cur_level
    if ff < cur_level and loud_conf < F01:
        return cur_level
    return ff


Step = namedtuple('Step', 'smoothed gated final_freq ble_state ble_level sent ev_l ev_llr_v ev_llr_a')


def replay_legacy(ticks, params, actions=None):
    """actions: 可选，逐 tick 替换 (v, vc, a, ac)。"""
    core, ble, out = LegacyCore(params), Ble(params), []
    for i, t in enumerate(ticks):
        v, vc, a, ac = actions[i] if actions else (t.v, t.vc, t.a, t.ac)
        sm = core.smoothed(v, a, vc, ac)
        g = core.gate(sm, v, vc, a, ac, ble.cur)
        ff = final_freq(t.loud_lv, t.loud_conf, ble.level)
        sent = ble.update(g, ff, t.wall) if g else None
        out.append(Step(sm, g, ff, ble.cur, ble.level, sent, None, None, None))
    return out


# ======================================================================== 证据融合
class Modality(object):
    """特征 s = sum(weights[i] * probs[i])，裁到 [0,1]，按 edges 分箱查 llr。"""

    def __init__(self, weights, edges, llr, feature=''):
        self.weights = [float(x) for x in weights]
        self.edges = [float(x) for x in edges]
        self.llr = [float(x) for x in llr]
        self.feature_name = feature
        if len(self.edges) < 2 or len(self.llr) != len(self.edges) - 1:
            raise ValueError('edges 与 llr 长度不匹配: %d vs %d' % (len(self.edges), len(self.llr)))
        if any(self.edges[i] >= self.edges[i + 1] for i in range(len(self.edges) - 1)):
            raise ValueError('edges 必须严格递增')

    def feature(self, probs):
        s = 0.0
        for w, p in zip(self.weights, probs):
            s += w * p
        return 0.0 if s < 0.0 else (1.0 if s > 1.0 else s)

    def lookup(self, s):
        i = 0
        for k in range(1, len(self.edges) - 1):
            if s >= self.edges[k]:
                i = k
            else:
                break
        return self.llr[i]

    def to_dict(self):
        return {'feature': self.feature_name, 'weights': self.weights,
                'edges': self.edges, 'llr': self.llr}

    @classmethod
    def from_dict(cls, d):
        return cls(d['weights'], d['edges'], d['llr'], d.get('feature', ''))


class EvidenceConfig(object):
    FORMAT = 'xcup-fusion-v1'

    def __init__(self, video, audio, w_v=1.0, w_a=0.5, clamp_lo=4.0, clamp_hi=4.0, theta_on=1.5,
                 theta_off=-1.0, max_age_ms=MAX_AGE_MS, mode='shadow', meta=None):
        self.video, self.audio = video, audio
        self.w_v, self.w_a = float(w_v), float(w_a)
        self.clamp_lo, self.clamp_hi = float(clamp_lo), float(clamp_hi)
        self.theta_on, self.theta_off = float(theta_on), float(theta_off)
        self.max_age_ms = int(max_age_ms)
        self.mode = mode
        self.meta = meta or {}
        if not (self.clamp_lo > 0 and self.clamp_hi > 0
                and -self.clamp_lo < self.theta_off < self.theta_on <= self.clamp_hi):
            raise ValueError('需满足 -clamp_lo < theta_off < theta_on <= clamp_hi')

    def with_params(self, w_v, w_a, clamp_lo, clamp_hi, theta_on, theta_off):
        return EvidenceConfig(self.video, self.audio, w_v, w_a, clamp_lo, clamp_hi, theta_on,
                              theta_off, self.max_age_ms, self.mode, dict(self.meta))

    def params(self):
        return (self.w_v, self.w_a, self.clamp_lo, self.clamp_hi, self.theta_on, self.theta_off)

    def to_json(self):
        d = {'format': self.FORMAT, 'mode': self.mode,
             'id': self.meta.get('id', 'unnamed'),
             'video': self.video.to_dict(), 'audio': self.audio.to_dict(),
             'w_v': self.w_v, 'w_a': self.w_a,
             'clamp_lo': self.clamp_lo, 'clamp_hi': self.clamp_hi,
             'theta_on': self.theta_on, 'theta_off': self.theta_off,
             'max_age_ms': self.max_age_ms}
        for k, v in self.meta.items():
            if k != 'id':
                d.setdefault(k, v)
        text = json.dumps(d, ensure_ascii=False, indent=2)
        # 数组压成一行，证据表一眼能看全，diff 也短
        text = re.sub(r'\[\s+([^\[\]{}]*?)\s+\]',
                      lambda m: '[' + ' '.join(m.group(1).split()) + ']', text)
        return text + '\n'

    @classmethod
    def from_json(cls, text):
        d = json.loads(text)
        if d.get('format') != cls.FORMAT:
            raise ValueError('不认识的配置格式: %r' % d.get('format'))
        known = ('format', 'mode', 'video', 'audio', 'w_v', 'w_a', 'clamp', 'clamp_lo',
                 'clamp_hi', 'theta_on', 'theta_off', 'max_age_ms')
        meta = dict((k, v) for k, v in d.items() if k not in known)
        lo = d['clamp_lo'] if 'clamp_lo' in d else d['clamp']      # 兼容只有对称 clamp 的旧配置
        hi = d['clamp_hi'] if 'clamp_hi' in d else d['clamp']
        return cls(Modality.from_dict(d['video']), Modality.from_dict(d['audio']),
                   d['w_v'], d['w_a'], lo, hi, d['theta_on'], d['theta_off'],
                   d.get('max_age_ms', MAX_AGE_MS), d.get('mode', 'shadow'), meta)


def _valid(probs, n):
    if probs is None or len(probs) < n:
        return False
    s = 0.0
    for p in probs:
        if p != p or p in (float('inf'), float('-inf')):
            return False
        s += p
    return s > 0.5          # 推理失败时 Java 端给的是全 0 数组


class EvidenceFusion(object):
    """证据融合（double 运算）。app 里原有一份逐 tick 对拍的 Java 版，2026-10-06 随证据融合一起删除。"""
    NONE, NOISE, DO = 0, 1, 2

    def __init__(self, cfg):
        self.cfg = cfg
        self.reset()

    def reset(self):
        self.L = 0.0
        self.state = self.NONE
        self.v_ts = None
        self.a_ts = None

    def _consume(self, mod, probs, ts, last, now):
        """每个模型结果只用一次（按结果时间戳去重）；过期或无效的结果算"用过"但不给证据。"""
        if ts is None or (last is not None and ts <= last):
            return None, last
        if now - ts > self.cfg.max_age_ms or not _valid(probs, len(mod.weights)):
            return None, ts
        return mod.lookup(mod.feature(probs)), ts

    def update(self, vprobs, v_ts, aprobs, a_ts, now):
        cfg = self.cfg
        llr_v, self.v_ts = self._consume(cfg.video, vprobs, v_ts, self.v_ts, now)
        llr_a, self.a_ts = self._consume(cfg.audio, aprobs, a_ts, self.a_ts, now)
        if llr_v is not None or llr_a is not None:
            d = 0.0
            if llr_v is not None:
                d += cfg.w_v * llr_v
            if llr_a is not None:
                d += cfg.w_a * llr_a
            x = self.L + d
            self.L = cfg.clamp_hi if x > cfg.clamp_hi else (-cfg.clamp_lo if x < -cfg.clamp_lo else x)
            if self.state == self.DO:
                if self.L <= cfg.theta_off:
                    self.state = self.NOISE
            else:
                self.state = self.DO if self.L >= cfg.theta_on else self.NOISE
        act = '' if self.state == self.NONE else ('do' if self.state == self.DO else 'Noise')
        return act, llr_v, llr_a


def _ts(t, which):
    age = t.v_age if which == 'v' else t.a_age
    return None if age is None or age < 0 else t.wall - age


def replay_evidence(ticks, cfg, params):
    ef, ble, out = EvidenceFusion(cfg), Ble(params), []
    for t in ticks:
        act, lv, la = ef.update(t.vp, _ts(t, 'v'), t.ap, _ts(t, 'a'), t.wall)
        ff = final_freq(t.loud_lv, t.loud_conf, ble.level)
        sent = ble.update(act, ff, t.wall) if act else None
        out.append(Step(act, act, ff, ble.cur, ble.level, sent, ef.L, lv, la))
    return out


def prepare(video, cfg):
    """扫参用：把查表结果预先算好，(pos, wall, llr_v|None, llr_a|None, loud_lv, loud_conf)。"""
    ef = EvidenceFusion(cfg)
    out = []
    for t in video.ticks:
        lv, ef.v_ts = ef._consume(cfg.video, t.vp, _ts(t, 'v'), ef.v_ts, t.wall)
        la, ef.a_ts = ef._consume(cfg.audio, t.ap, _ts(t, 'a'), ef.a_ts, t.wall)
        out.append((t.pos, t.wall, lv, la, t.loud_lv, t.loud_conf))
    return out


def simulate(prep, w_v, w_a, lo, hi, on, off, params):
    """replay_evidence 的快速版（只要状态切换点）。selftest 保证二者结果一致。"""
    L, state, cur = 0.0, 0, 'noise'
    ble = Ble(params)
    trans = []
    for pos, wall, lv, la, loud_lv, loud_conf in prep:
        if lv is not None or la is not None:
            d = 0.0
            if lv is not None:
                d += w_v * lv
            if la is not None:
                d += w_a * la
            x = L + d
            L = hi if x > hi else (-lo if x < -lo else x)
            if state == 2:
                if L <= off:
                    state = 1
            else:
                state = 2 if L >= on else 1
        if state:
            s = ble.update('do' if state == 2 else 'Noise',
                           final_freq(loud_lv, loud_conf, ble.level), wall)
            if s is not None:
                a = 'do' if _is_sex(s[0]) else 'noise'
                if a != cur:
                    trans.append((pos, a))
                    cur = a
    return trans


# ======================================================================== 评测
class GtIndex(object):
    """把 metrics.evaluate 的真值栅格（护带/预热/覆盖范围）压成区间，扫参时 O(区间数) 出混淆矩阵。"""

    def __init__(self, segs, dur, lo, hi, policy='exclude'):
        n = dur // GRID_MS
        gt = [None] * n
        for s, e, lab in segs:
            b = M.bin_of(lab, policy)
            for i in range(max(0, s // GRID_MS), min(n, -(-e // GRID_MS))):
                gt[i] = b
        mask = [g is not None for g in gt]
        for s, e, lab in segs[1:]:
            for i in range(max(0, (s - M.GUARD_MS) // GRID_MS),
                           min(n, (s + M.GUARD_MS) // GRID_MS + 1)):
                mask[i] = False
        for i in range(n):
            if i * GRID_MS < lo or i * GRID_MS > hi:
                mask[i] = False
        runs, i = [], 0
        while i < n:
            if not mask[i]:
                i += 1
                continue
            j = i
            while j < n and mask[j] and gt[j] == gt[i]:
                j += 1
            runs.append((i, j, gt[i]))
            i = j
        self.n, self.gt, self.mask, self.runs = n, gt, mask, runs

    def score(self, trans):
        """trans: [(pos, 'do'|'noise')]（只含真正改变马达状态的指令）-> (混淆矩阵, 动作中误停次数)"""
        cm = {('noise', 'noise'): 0, ('noise', 'do'): 0, ('do', 'noise'): 0, ('do', 'do'): 0}
        cps = [(-(-p // GRID_MS), s) for p, s in trans]
        k, state = 0, 'noise'
        for i0, i1, g in self.runs:
            i = i0
            while i < i1:
                while k < len(cps) and cps[k][0] <= i:
                    state = cps[k][1]
                    k += 1
                j = min(i1, cps[k][0]) if k < len(cps) else i1
                cm[(g, state)] += j - i
                i = j
        fs = 0
        for p, s in trans:
            if s != 'noise':
                continue
            i = p // GRID_MS
            if 0 <= i < self.n and self.gt[i] == 'do' and self.mask[i]:
                fs += 1
        return cm, fs


class Cost(object):
    """扫参目标：误转秒数 x fp + 漏转秒数 x fn + 动作中误停次数 x stop（秒）。这是产品取舍，不随模型变。"""

    def __init__(self, fp=1.0, fn=1.0, stop=3.0):
        self.fp, self.fn, self.stop = float(fp), float(fn), float(stop)

    def __call__(self, cm, fs):
        return ((self.fp * cm[('noise', 'do')] + self.fn * cm[('do', 'noise')]) * GRID_MS / 1000.0
                + self.stop * fs)

    def __str__(self):
        return 'fp=%g,fn=%g,stop=%g' % (self.fp, self.fn, self.stop)

    @classmethod
    def parse(cls, s):
        kw = {}
        for part in (s or '').split(','):
            if part.strip():
                k, v = part.split('=', 1)
                if k.strip() not in ('fp', 'fn', 'stop'):
                    raise ValueError('未知的代价项 %r（可用 fp/fn/stop）' % k)
                kw[k.strip()] = float(v)
        return cls(**kw)


def transitions_from_steps(ticks, steps):
    out, cur = [], 'noise'
    for t, s in zip(ticks, steps):
        if s.sent is None:
            continue
        a = 'do' if _is_sex(s.sent[0]) else 'noise'
        if a != cur:
            out.append((t.pos, a))
            cur = a
    return out


def transitions_from_log(ticks):
    out, cur = [], 'noise'
    for t in ticks:
        if t.log.get('sent') != '1' or not t.log.get('sent_action'):
            continue
        a = 'do' if _is_sex(t.log['sent_action']) else 'noise'
        if a != cur:
            out.append((t.pos, a))
            cur = a
    return out


def latency(video, trans):
    """复刻 metrics.evaluate 的切换延迟匹配规则。"""
    lat = {'noise->do': [], 'do->noise': []}
    miss = {'noise->do': 0, 'do->noise': 0}
    used = -1
    for t, frm, to in M.gt_transitions(video.segs, video.policy):
        key = '%s->%s' % (frm, to)
        if t < video.lo or t > video.hi:
            continue
        hit = None
        for p, s in trans:
            if s == to and p > used and t - M.LAT_PRE <= p <= t + M.LAT_WIN:
                hit = p
                break
        if hit is None:
            miss[key] += 1
            continue
        used = hit
        lat[key].append(hit - t)
    return lat, miss


class Score(object):
    def __init__(self, cm=None, fs=0, lat=None, miss=None, cost=0.0):
        self.cm = cm or {('noise', 'noise'): 0, ('noise', 'do'): 0, ('do', 'noise'): 0, ('do', 'do'): 0}
        self.fs = fs
        self.lat = lat or {'noise->do': [], 'do->noise': []}
        self.miss = miss or {'noise->do': 0, 'do->noise': 0}
        self.cost = cost

    @classmethod
    def of(cls, video, trans, cost):
        cm, fs = video.gti.score(trans)
        lat, miss = latency(video, trans)
        return cls(cm, fs, lat, miss, cost(cm, fs))

    def add(self, o):
        for k in self.cm:
            self.cm[k] += o.cm[k]
        self.fs += o.fs
        for k in self.lat:
            self.lat[k] = self.lat[k] + o.lat[k]
            self.miss[k] += o.miss[k]
        self.cost += o.cost
        return self

    @property
    def tot(self):
        return sum(self.cm.values())

    def acc(self):
        t = self.tot
        return (self.cm[('noise', 'noise')] + self.cm[('do', 'do')]) / float(t) if t else float('nan')

    def fp(self):
        n = self.cm[('noise', 'noise')] + self.cm[('noise', 'do')]
        return self.cm[('noise', 'do')] / float(n) if n else float('nan')

    def fn(self):
        n = self.cm[('do', 'do')] + self.cm[('do', 'noise')]
        return self.cm[('do', 'noise')] / float(n) if n else float('nan')

    def stops_per_min(self):
        mins = self.tot * GRID_MS / 60000.0
        return self.fs / mins if mins else float('nan')


def _pct(x):
    return '   —  ' if x != x else '%5.1f%%' % (100.0 * x)


# ======================================================================== 拼接样本
def make_splices(video, n=3, act_ms=60000, plot_ms=45000, margin=3000):
    """bench 里的片子都是"剧情 -> 动作"，而且动作前几乎都隔着一段 other_adult（不计分）：
    停转切换点一个都没有，干净的启动切换点也只有一两个。不补的话扫参会挑出"永远不停"
    或"启动很慢"的参数。

    这里用同一部片的动作段和剧情段首尾拼接，造两类样本：
      停转样本 'stop'：动作 60s + 剧情 45s，看动作结束后马达多转几秒
      启动样本 'start'：剧情 45s + 动作 60s，看动作开始后马达多久才转
    拼接处感知是瞬间切换的，真实片子里模型窗口要 2~3 秒才翻过来，所以两种延迟都偏乐观。
    """
    if video.synthetic:
        return []
    pol = video.policy
    dos = [(s, e) for s, e, l in video.segs
           if M.bin_of(l, pol) == 'do' and e - s >= act_ms + 2 * margin]
    plots = [(s, e) for s, e, l in video.segs
             if M.bin_of(l, pol) == 'noise' and e - s >= plot_ms + 2 * margin
             and s >= M.WARMUP_MS]
    if not dos or not plots:
        return []
    out = []
    for kind in ('stop', 'start'):
        for k in range(n):
            frac = (k + 1) / float(n + 1)
            ds, de = dos[k % len(dos)]
            ps, pe = plots[k % len(plots)]
            a1 = int(ds + margin + act_ms + frac * (de - ds - 2 * margin - act_ms))
            a0 = a1 - act_ms
            p0 = int(ps + margin + frac * (pe - ps - 2 * margin - plot_ms))
            p1 = p0 + plot_ms
            part_a = [t for t in video.ticks if a0 <= t.pos < a1]
            part_p = [t for t in video.ticks if p0 <= t.pos < p1]
            if len(part_a) < 20 or len(part_p) < 20:
                continue
            first, second = (part_a, part_p) if kind == 'stop' else (part_p, part_a)
            # 墙钟保持原来的量级（蓝牙状态机拿"距上次发送"算间隔，不能从 0 开始）；
            # 结果时间戳 = wall - age，age 不变，所以去重/新鲜度在拼接后照样成立。
            ticks = [t.copy(pos=t.pos - first[0].pos) for t in first]
            b_pos = ticks[-1].pos + 500
            b_wall = ticks[-1].wall + 500
            for t in second:
                ticks.append(t.copy(pos=b_pos + (t.pos - second[0].pos),
                                    wall=b_wall + (t.wall - second[0].wall)))
            dur = ticks[-1].pos + 500
            labs = ('sex', 'plot') if kind == 'stop' else ('plot', 'sex')
            segs = [(0, b_pos, labs[0]), (b_pos, dur, labs[1])]
            out.append(Video('%s#%s%d' % (video.vid, '停' if kind == 'stop' else '启', k + 1),
                             segs, dur, ticks, video.params, video.gt_path, video.run_path,
                             synthetic=True, policy=pol, splice=kind))
    return out


def splice_seconds(sp, trans):
    """停转样本：拼接点之后马达还在转的秒数；启动样本：拼接点之后马达没转的秒数。"""
    b, e = sp.segs[1][0], sp.segs[1][1]
    want = 'noise' if sp.splice == 'stop' else 'do'
    bad = sum(1 for i in range(b // GRID_MS, e // GRID_MS) if M.state_at(trans, i * GRID_MS) != want)
    return bad * GRID_MS / 1000.0


# ======================================================================== 拟合
EDGES = [0.0, 0.02, 0.05, 0.1, 0.2, 0.35, 0.5, 0.65, 0.8, 0.9, 0.95, 0.98, 1.0]
VIDEO_FEATURE = ('p_oral+p_sex', [0.0, 1.0, 1.0])            # 视频类目 [plot, oral, sex]
AUDIO_FEATURES = [('p_sex+p_oral', [1.0, 1.0, 0.0]),        # 音频类目 [sex, oral, noise]
                  ('p_sex', [1.0, 0.0, 0.0])]
LLR_CLIP = 3.0       # 单个结果最多贡献的证据，防止某个极端区间的小样本一锤定音
ALPHA = 0.5          # 每格的平滑伪计数

GRID = {
    # 视频证据的权重也要搜：音频修好之后，"少信视频、甚至只用音频"在剧情有裸露的片子上更好，
    # 固定 w_v=1 就搜不到这一片（w_v=0 即只用音频）
    'w_v': [1.0, 0.5, 0.25, 0.0],
    'w_a': [0.0, 0.5, 1.0, 2.0, 3.0],
    'clamp_lo': [1.0, 2.0, 3.0, 4.0, 6.0],
    'clamp_hi': [3.0, 4.0, 6.0, 8.0, 12.0, 16.0],
    'theta_on': [-0.5, 0.0, 0.5, 1.0, 1.5, 2.0],
    'theta_off': [-4.0, -2.0, -1.0, -0.5, 0.0, 0.5],
}
QUICK_GRID = {
    'w_v': [1.0, 0.0],
    'w_a': [0.0, 1.0],
    'clamp_lo': [2.0, 4.0],
    'clamp_hi': [4.0, 8.0],
    'theta_on': [0.5, 1.5],
    'theta_off': [-1.0, 0.0],
}
# 视频"像剧情"那一半证据的软化系数，拟合时一起搜，结果直接乘进证据表。
# 动机：遮挡/特写时视频模型在动作段里也判"剧情"，原样的负证据会把马达压停。
NEG_SCALES = [1.0, 0.6, 0.35]


def bin_index(edges, s):
    i = 0
    for k in range(1, len(edges) - 1):
        if s >= edges[k]:
            i = k
        else:
            break
    return i


def clean_label(segs, a, b, policy):
    """[a,b] 整个落在同一个真值段内才给标签，模型窗口跨边界的结果不拿来统计。"""
    for s, e, lab in segs:
        if s <= b < e:
            return M.bin_of(lab, policy) if s <= a else None
    return None


def result_samples(video, which, weights):
    """逐个"新结果"取 (特征值, 真值)。"""
    W = VIDEO_WIN_MS if which == 'v' else AUDIO_WIN_MS
    mod = Modality(weights, [0.0, 1.0], [0.0])
    out, last = [], None
    for t in video.ticks:
        age = t.v_age if which == 'v' else t.a_age
        probs = t.vp if which == 'v' else t.ap
        if age is None or age < 0:
            continue
        ts = t.wall - age
        if last is not None and ts <= last:
            continue
        last = ts
        if age > MAX_AGE_MS or not _valid(probs, len(weights)):
            continue
        pr = t.pos - age
        if pr < M.WARMUP_MS:
            continue
        lab = clean_label(video.segs, pr - W, pr, video.policy)
        if lab is None:
            continue
        out.append((mod.feature(probs), lab))
    return out


def pav(y, w):
    """加权保序回归（单调不减）：LLR 应随"像动作"的概率单调上升，消掉小样本区间的锯齿。"""
    blocks = []
    for yi, wi in zip(y, w):
        blocks.append([yi * wi, wi, 1])
        while len(blocks) > 1 and blocks[-2][0] / blocks[-2][1] > blocks[-1][0] / blocks[-1][1]:
            b = blocks.pop()
            blocks[-1][0] += b[0]
            blocks[-1][1] += b[1]
            blocks[-1][2] += b[2]
    out = []
    for s, ws, c in blocks:
        out += [s / ws] * c
    return out


def fit_table(samples, edges=EDGES, alpha=ALPHA, clip=LLR_CLIP):
    K = len(edges) - 1
    nd, nn = [0] * K, [0] * K
    for s, lab in samples:
        i = bin_index(edges, s)
        if lab == 'do':
            nd[i] += 1
        else:
            nn[i] += 1
    Nd, Nn = sum(nd), sum(nn)
    if Nd == 0 or Nn == 0:
        return [0.0] * K, (nd, nn)      # 缺一类样本：这个模态不提供证据
    llr = [math.log((nd[i] + alpha) / (Nd + K * alpha))
           - math.log((nn[i] + alpha) / (Nn + K * alpha)) for i in range(K)]
    llr = pav(llr, [nd[i] + nn[i] + 2 * alpha for i in range(K)])
    llr = [round(max(-clip, min(clip, x)), 4) for x in llr]
    return llr, (nd, nn)


GRID_KEYS = ('w_v', 'w_a', 'clamp_lo', 'clamp_hi', 'theta_on', 'theta_off')


def _space(variant, grid):
    sp = dict((k, list(grid.get(k, [1.0]))) for k in GRID_KEYS)
    if variant == 'video_only':
        sp['w_v'], sp['w_a'] = [1.0], [0.0]
    elif variant == 'audio_only':
        sp['w_v'], sp['w_a'] = [0.0], [1.0]
    return sp


def _combo(space, ix):
    return tuple(space[k][i] for k, i in zip(GRID_KEYS, ix))


def _combo_ok(combo):
    w_v, w_a, lo, hi, on, off = combo
    return (w_v > 0 or w_a > 0) and -lo < off < on <= hi


def param_combos(variant, grid):
    """全网格（--exhaustive 用）。"""
    space = _space(variant, grid)
    for ix in _product([range(len(space[k])) for k in GRID_KEYS]):
        c = _combo(space, ix)
        if _combo_ok(c):
            yield c


def _product(ranges):
    if not ranges:
        yield ()
        return
    for i in ranges[0]:
        for rest in _product(ranges[1:]):
            yield (i,) + rest


def fit(train, cost, variant='evidence', grid=GRID, exhaustive=False, top_k=10):
    """证据表只用真实片子统计；动态参数在 真实片子 + 拼接样本 上按代价搜。

    默认先粗后细：粗扫（每维隔一个取值）留下最好的 top_k 组，再在它们的邻域（每维 ±1 档）
    里扫全网格，比全网格快约 10 倍，结果与全网格一致或几乎一致（--exhaustive 可对照）。
    """
    real = [v for v in train if not v.synthetic]
    vs = []
    for v in real:
        vs += result_samples(v, 'v', VIDEO_FEATURE[1])
    vt0, vcounts = fit_table(vs)
    afeats = AUDIO_FEATURES if variant != 'video_only' else AUDIO_FEATURES[:1]
    negs = NEG_SCALES if variant != 'audio_only' else [1.0]
    space = _space(variant, grid)
    atables = {}
    contexts = []
    for ns in negs:
        vt = [round(x * ns, 4) if x < 0 else x for x in vt0]
        for aname, aw in afeats:
            if aname not in atables:
                as_ = []
                for v in real:
                    as_ += result_samples(v, 'a', aw)
                atables[aname] = fit_table(as_)
            at, acounts = atables[aname]
            base = EvidenceConfig(Modality(VIDEO_FEATURE[1], EDGES, vt, VIDEO_FEATURE[0]),
                                  Modality(aw, EDGES, at, aname))
            preps = sorted([(prepare(v, base), v) for v in train], key=lambda x: -len(x[0]))
            info = {'audio_feature': aname, 'neg_scale': ns,
                    'video_counts': vcounts, 'audio_counts': acounts}
            contexts.append((base, preps, info))

    best = [None]

    def redundant(ctx, combo):
        """w_v=0 时视频负证据折扣无关、w_a=0 时音频特征无关：这些组合只在第一个对应的 context 里算。"""
        info = contexts[ctx][2]
        return ((combo[0] == 0 and info['neg_scale'] != negs[0])
                or (combo[1] == 0 and info['audio_feature'] != afeats[0][0]))

    def evaluate(ctx, combo, bound):
        """总代价；超过 bound 就提前放弃（返回 None）。"""
        total = 0.0
        for prep, v in contexts[ctx][1]:
            cm, fs = v.gti.score(simulate(prep, *combo, params=v.params))
            total += cost(cm, fs)
            if bound is not None and total >= bound:
                return None
        return total

    def consider(ctx, combo):
        b = best[0][0] if best[0] else None
        t = evaluate(ctx, combo, b)
        if t is not None and (b is None or t < b):
            base, _, info = contexts[ctx]
            best[0] = (t, base.with_params(*combo), info)
        return t

    if exhaustive:
        for ctx in range(len(contexts)):
            for combo in param_combos(variant, grid):
                if not redundant(ctx, combo):
                    consider(ctx, combo)
        return best[0]

    coarse = dict((k, sorted(set(range(0, len(space[k]), 2)) | {len(space[k]) - 1}))
                  for k in GRID_KEYS)
    scored = []
    for ctx in range(len(contexts)):
        for ix in _product([coarse[k] for k in GRID_KEYS]):
            combo = _combo(space, ix)
            if _combo_ok(combo) and not redundant(ctx, combo):
                t = evaluate(ctx, combo, None)
                scored.append((t, ctx, ix))
                if best[0] is None or t < best[0][0]:
                    base, _, info = contexts[ctx]
                    best[0] = (t, base.with_params(*combo), info)
    scored.sort()
    seen = set()
    where = {}                                     # 当前最优在哪个 (ctx, ix)

    def neighbourhood(ctx, ix):
        ranges = [range(max(0, i - 1), min(len(space[k]), i + 2)) for k, i in zip(GRID_KEYS, ix)]
        for nix in _product(ranges):
            if (ctx, nix) in seen:
                continue
            seen.add((ctx, nix))
            combo = _combo(space, nix)
            if _combo_ok(combo) and not redundant(ctx, combo):
                before = best[0][0]
                consider(ctx, combo)
                if best[0][0] < before:
                    where['at'] = (ctx, nix)

    for _, ctx, ix in scored[:top_k]:
        neighbourhood(ctx, ix)
    for _ in range(4):                             # 围着当前最优继续爬，直到不再变好
        if 'at' not in where:
            break
        at = where.pop('at')
        neighbourhood(*at)
    return best[0]


# ======================================================================== 各策略跑一遍
def run_policy(video, policy, cfg=None):
    """返回 (状态切换点, 逐 tick Step)。"""
    if policy == 'legacy':
        steps = replay_legacy(video.ticks, video.params)
    elif policy == 'always':
        steps = []
        ble = Ble(video.params)
        for t in video.ticks:
            sent = ble.update('do', final_freq(t.loud_lv, t.loud_conf, ble.level), t.wall)
            steps.append(Step('do', 'do', 0, ble.cur, ble.level, sent, None, None, None))
    else:
        steps = replay_evidence(video.ticks, cfg, video.params)
    return transitions_from_steps(video.ticks, steps), steps


POLICIES = [('legacy', '现有融合'), ('always', '一直转'), ('video_only', '只用视频'),
            ('audio_only', '只用音频'), ('evidence', '证据融合')]


_W = {}          # 留一拟合的共享数据；并行时每个子进程由 _lovo_init 填一次


def _lovo_init(videos, splices, cost, grid, exhaustive):
    _W.update(videos=videos, splices=splices, cost=cost, grid=grid, exhaustive=exhaustive)


def _lovo_task(held_vid, var):
    train = [v for v in _W['videos'] if v.vid != held_vid]
    train_all = list(train)
    for v in train:
        train_all += _W['splices'][v.vid]
    t0 = time.time()
    best = fit(train_all, _W['cost'], var, _W['grid'], _W['exhaustive'])
    return held_vid, var, best, time.time() - t0


def lovo(videos, cost, grid, variants=('video_only', 'audio_only', 'evidence'), log=print,
         exhaustive=False, jobs=1):
    """留一视频交叉验证：拟合时连同被留出那部片的拼接样本一起排除。
    各折互相独立，jobs>1 时多进程并行（spawn，Windows/Linux 行为一致），结果与串行完全相同。"""
    splices = dict((v.vid, make_splices(v)) for v in videos)
    tasks = [(held.vid, var) for held in videos for var in variants]
    init = (videos, splices, cost, grid, exhaustive)
    folds = {}

    def done(held_vid, var, best, dt):
        folds[(held_vid, var)] = best
        log('  · 留出 %-5s %-8s 拟合完成（%.0fs）  %s' % (
            held_vid, dict(POLICIES)[var], dt, fmt_params(best[1], best[2])))

    jobs = max(1, min(jobs or (os.cpu_count() or 1), len(tasks)))
    if jobs == 1:
        _lovo_init(*init)
        for t in tasks:
            done(*_lovo_task(*t))
    else:
        import concurrent.futures as cf
        import multiprocessing as mp
        log('  （%d 个进程并行）' % jobs)
        order = sorted(tasks, key=lambda t: t[1] != 'evidence')      # 证据融合最慢，先排上
        with cf.ProcessPoolExecutor(jobs, mp_context=mp.get_context('spawn'),
                                    initializer=_lovo_init, initargs=init) as ex:
            futs = [ex.submit(_lovo_task, *t) for t in order]
            for f in cf.as_completed(futs):
                done(*f.result())
    return folds, splices


def fmt_params(cfg, info=None):
    s = 'w_v=%g w_a=%g C=-%g/+%g on=%g off=%g' % (cfg.w_v, cfg.w_a, cfg.clamp_lo, cfg.clamp_hi,
                                                 cfg.theta_on, cfg.theta_off)
    if info and cfg.w_a:
        s += ' 音频特征=%s' % info.get('audio_feature', '?')
    if info and cfg.w_v and info.get('neg_scale', 1.0) != 1.0:
        s += ' 视频负证据x%g' % info['neg_scale']
    return s


def evaluate_all(videos, cost, grid, log=print, variants=('video_only', 'audio_only', 'evidence'),
                 exhaustive=False, jobs=1):
    folds, splices = lovo(videos, cost, grid, variants, log, exhaustive, jobs)
    per = dict((p, {}) for p, _ in POLICIES)
    sp = dict((p, []) for p, _ in POLICIES)       # [(kind, Score, 秒数)]
    for v in videos:
        for p, _ in POLICIES:
            if p in ('legacy', 'always'):
                cfg = None
            elif p in variants:
                cfg = folds[(v.vid, p)][1]
            else:
                continue
            per[p][v.vid] = Score.of(v, run_policy(v, p, cfg)[0], cost)
            for s in splices[v.vid]:
                tr = run_policy(s, p, cfg)[0]
                sp[p].append((s.splice, Score.of(s, tr, cost), splice_seconds(s, tr)))
    return per, sp, folds, splices


def _splice_col(rows, kind):
    xs = [sec for k, _, sec in rows if k == kind]
    if not xs:
        return '—'
    stuck = sum(1 for x in xs if x >= 40)
    return '%.1fs%s' % (statistics.median(xs), '(%d个≥40s)' % stuck if stuck else '')


def print_table(title, videos, per, sp, policies=POLICIES, show_videos=True):
    print('══ %s ══' % title)
    print('  %-10s %7s %7s %7s %8s %14s %14s %8s' % (
        '策略', '准确率', '误转', '漏转', '误停/分', '启动后没转', '停止后还转', '代价'))
    for p, name in policies:
        if not per.get(p):
            continue
        pooled = Score()
        for v in videos:
            if v.vid in per[p]:
                pooled.add(per[p][v.vid])
        rows = sp.get(p) or []
        print('  %-10s %7s %7s %7s %8.2f %14s %14s %8.0f' % (
            name, _pct(pooled.acc()), _pct(pooled.fp()), _pct(pooled.fn()),
            pooled.stops_per_min(), _splice_col(rows, 'start'), _splice_col(rows, 'stop'),
            pooled.cost + sum(sc.cost for _, sc, _ in rows)))
    if show_videos:
        print('\n  逐片（准确率 / 误转 / 漏转 / 误停次数）')
        print('  %-6s' % '' + ''.join('%-24s' % name for p, name in policies if per.get(p)))
        for v in videos:
            cells = []
            for p, name in policies:
                if not per.get(p):
                    continue
                s = per[p].get(v.vid)
                cells.append('%-24s' % ('—' if s is None else '%s/%s/%s/%d' % (
                    _pct(s.acc()).strip(), _pct(s.fp()).strip(), _pct(s.fn()).strip(), s.fs)))
            print('  %-6s' % v.vid + ''.join(cells))
    print()


# ======================================================================== 子命令
def _load_cfg(path):
    return EvidenceConfig.from_json(io.open(path, encoding='utf-8').read())


def cmd_check(args):
    videos = load_videos(args.gt_dir, args.runs_dir, args.pair, args.policy)
    print()
    if any(v.rescored for v in videos):
        print('⚠ 这里有 audio_rescore.py 离线重算的 run：它们的决策列还是原 run 的，check 对不上是正常的。\n')
    cfg = _load_cfg(args.config) if args.config else None
    cols = ('smoothed', 'gated', 'final_freq', 'ble_state', 'ble_level', 'sent')
    worst = 1.0
    for v in videos:
        # live 模式的 run（app 带证据融合那几版、由它驱动蓝牙时录的）要用证据融合回放来对；配置 id 记在参数行里
        if v.fusion.startswith('live'):
            if cfg is None:
                print('══ %s：live 模式采集（%s），需要用 --config 给出当时的配置文件（已从 app 删除，'
                      '按配置 id 在 git log -p -- app/src/main/assets/fusion_config.json 里找）══\n'
                      % (v.vid, v.fusion))
                continue
            steps = replay_evidence(v.ticks, cfg, v.params)
            what = '证据融合（%s）' % cfg.meta.get('id', '?')
        else:
            steps = replay_legacy(v.ticks, v.params)
            what = '现有融合'
        agree = dict((c, 0) for c in cols)
        divs = []
        for t, s in zip(v.ticks, steps):
            lg = t.log
            got = {'smoothed': s.smoothed, 'gated': s.gated, 'final_freq': str(s.final_freq),
                   'ble_state': s.ble_state, 'ble_level': str(s.ble_level),
                   'sent': ('%s/%d' % s.sent) if s.sent else ''}
            want = dict((c, lg.get(c, '')) for c in cols[:-1])
            want['sent'] = ('%s/%s' % (lg.get('sent_action'), lg.get('sent_level'))) \
                if lg.get('sent') == '1' else ''
            for c in cols:
                if got[c] == want[c]:
                    agree[c] += 1
                elif len(divs) < args.show:
                    divs.append((t.pos, c, want[c], got[c]))
        n = float(len(v.ticks))
        tl, tr = transitions_from_log(v.ticks), transitions_from_steps(v.ticks, steps)
        same = cells = 0
        for i0, i1, g in v.gti.runs:
            for i in range(i0, i1):
                cells += 1
                same += M.state_at(tl, i * GRID_MS) == M.state_at(tr, i * GRID_MS)
        tl_agree = same / float(cells) if cells else 1.0
        worst = min(worst, tl_agree, agree['gated'] / n)
        sl, sr = Score.of(v, tl, Cost()), Score.of(v, tr, Cost())
        print('══ %s  %s  回放：%s ══' % (v.vid, os.path.basename(v.run_path), what))
        print('  逐 tick 一致率  ' + '  '.join('%s %.2f%%' % (c, 100.0 * agree[c] / n) for c in cols))
        print('  马达状态时间线一致率 %.2f%%   准确率 实测 %s / 回放 %s'
              % (100.0 * tl_agree, _pct(sl.acc()).strip(), _pct(sr.acc()).strip()))
        for pos, c, w, g in divs:
            print('    pos %8.1fs  %-10s 实测 %-10r 回放 %r' % (pos / 1000.0, c, w, g))
        print()
    ok = worst >= 0.99
    print('结论: %s（最差一致率 %.2f%%，阈值 99%%）' % ('回放器可信' if ok else '回放与实测不一致，先查清再比较',
                                                  100.0 * worst))
    return 0 if ok else 1


def cmd_eval(args):
    cost = Cost.parse(args.cost)
    grid = QUICK_GRID if args.quick else GRID
    print('加载数据（other_adult 策略 %s，代价 %s）' % (args.policy, cost))
    videos = load_videos(args.gt_dir, args.runs_dir, args.pair, args.policy)
    if len(videos) < 2:
        sys.exit('留一交叉验证至少要 2 部片')
    print('\n留一视频交叉验证（每部片只用其余 %d 部拟合）…' % (len(videos) - 1))
    t0 = time.time()
    per, sp, folds, splices = evaluate_all(videos, cost, grid, exhaustive=args.exhaustive, jobs=args.jobs)
    print('  共用时 %.0fs\n' % (time.time() - t0))
    nst = sum(1 for x in splices.values() for s in x if s.splice == 'start')
    nsp = sum(1 for x in splices.values() for s in x if s.splice == 'stop')
    print_table('留一视频交叉验证（拼接样本：启动 %d 个，停转 %d 个）' % (nst, nsp), videos, per, sp)
    print('  · 准确率/误转/漏转/误停只算真实片子：误转 = 剧情时长里马达在转的比例，漏转 = 动作时长里马达没转的比例。')
    print('  · 启动后没转 / 停止后还转：拼接样本里，切换点之后马达"该转没转" / "该停还转"的秒数（中位数，括号里是')
    print('    ≥40 秒、基本等于没切过去的个数）。拼接处感知是瞬间切换的，比真实片子乐观约 1~2 秒。')
    print('  · 代价 = 真实片子 + 拼接样本（误转秒×fp + 漏转秒×fn + 误停次数×stop）。\n')
    return 0


def oracle_video(video, which):
    """把 which（'v'/'a'）模态在可评测段落里的输出换成真值。"""
    ticks = []
    for t in video.ticks:
        kw = {}
        for m in which:
            age = t.v_age if m == 'v' else t.a_age
            if age is None or age < 0 or age > MAX_AGE_MS:
                continue
            lab = M.bin_of(M.gt_label_at(video.segs, t.pos - age), video.policy)
            if lab is None:
                continue
            if m == 'v':
                kw['vp'] = (0.0, 0.0, 1.0) if lab == 'do' else (1.0, 0.0, 0.0)
                kw['v'], kw['vc'] = ('do', 1.0) if lab == 'do' else ('Noise', 1.0)
            else:
                kw['ap'] = (1.0, 0.0, 0.0) if lab == 'do' else (0.0, 0.0, 1.0)
                kw['a'], kw['ac'] = ('do', 1.0) if lab == 'do' else ('Noise', 1.0)
        ticks.append(t.copy(**kw))
    return video.with_ticks(ticks)


def cmd_oracle(args):
    cost = Cost.parse(args.cost)
    grid = QUICK_GRID if args.quick else GRID
    videos = load_videos(args.gt_dir, args.runs_dir, args.pair, args.policy)
    print('\n模态上限：把某个模态的输出换成真值（窗口末端的真值，无感知延迟），看端到端最多能到多少。')
    print('现有融合不用拟合；证据融合按留一视频交叉验证重新拟合。\n')
    scen = [('实际模型', ''), ('视频完美', 'v'), ('音频完美', 'a')]
    pols = [('legacy', '现有融合'), ('evidence', '证据融合')]
    for name, which in scen:
        vs = [oracle_video(v, which) if which else v for v in videos]
        per, sp, _, _ = evaluate_all(vs, cost, grid, log=lambda *a: None, variants=('evidence',),
                                     exhaustive=args.exhaustive, jobs=args.jobs)
        print_table(name, vs, per, sp, pols, show_videos=False)
    print('  · 如果"视频完美"在现有融合下几乎不涨、在证据融合下大涨，说明现有融合挡住了视频侧的改进。')
    return 0


# ======================================================================== 自检
def _synth_video(seed, layout, vid='syn', policy='exclude'):
    """合成一部"视频准而召回低、音频召回高而剧情里乱报"的片子（和 bench 实测的形态一致）。"""
    rnd = random.Random(seed)
    segs, t = [], 0
    for lab, d in layout:
        segs.append((t, t + d, lab))
        t += d
    dur = t
    p = dict(LEGACY_DEFAULTS)

    def lab_at(pos):
        return M.bin_of(M.gt_label_at(segs, pos), 'noise') or 'noise'

    ticks = []
    wall0 = 1760000000000
    v_res = a_res = None
    next_v, next_a = 3000, 2000
    v_drop = False
    pos = 0
    while pos < dur:
        while next_v <= pos:
            lab = lab_at(next_v)
            if lab == 'do':
                v_drop = (rnd.random() < 0.15) if not v_drop else (rnd.random() < 0.8)
                pdo = rnd.uniform(0.7, 1.0) if not v_drop else rnd.uniform(0.0, 0.5)
            else:
                pdo = rnd.uniform(0.0, 0.08) if rnd.random() > 0.03 else rnd.uniform(0.5, 0.9)
            o = pdo * rnd.random()
            v_res = (wall0 + next_v + rnd.randint(0, 40), (f32(1 - pdo), f32(o), f32(pdo - o)))
            next_v += 500
        while next_a <= pos:
            lab = lab_at(next_a)
            hit = rnd.random() < (0.95 if lab == 'do' else 0.8)
            pdo = rnd.uniform(0.6, 1.0) if hit else rnd.uniform(0.0, 0.5)
            s = pdo * rnd.uniform(0.3, 1.0)
            a_res = (wall0 + next_a + rnd.randint(0, 60), (f32(s), f32(pdo - s), f32(1 - pdo)))
            next_a += 1000
        wall = wall0 + pos + rnd.randint(41, 90)
        tk = Tick()
        tk.pos, tk.wall = pos, wall
        for m, res, do_th, no_th, idx in (('v', v_res, p['VIDEO_DO'], p['VIDEO_PLOT'], (1, 2, 0)),
                                          ('a', a_res, p['AUDIO_DO'], p['AUDIO_NOISE'], (0, 1, 2))):
            if res is None:
                age, probs, act, conf = -1, None, '', f32(0.0)
            else:
                age, probs = wall - res[0], res[1]
                pdo = f32(probs[idx[0]] + probs[idx[1]])
                pno = probs[idx[2]]
                if age > MAX_AGE_MS:
                    act, conf = '', f32(0.0)
                elif pdo >= f32(do_th):
                    act, conf = 'do', pdo
                elif pno >= f32(no_th):
                    act, conf = 'Noise', pno
                else:
                    act, conf = '', f32(0.0)
            if m == 'v':
                tk.v, tk.vc, tk.v_age, tk.vp = act, f32(float('%.4f' % conf)), age, probs
            else:
                tk.a, tk.ac, tk.a_age, tk.ap = act, f32(float('%.4f' % conf)), age, probs
        tk.loud_lv = rnd.randint(2, 7) if rnd.random() < 0.9 else -1
        tk.loud_conf = f32(float('%.4f' % rnd.random())) if tk.loud_lv >= 0 else f32(0.0)
        tk.log = {}
        ticks.append(tk)
        pos += 500
    steps = replay_legacy(ticks, p)
    for tk, s in zip(ticks, steps):     # 让"实测列"等于现有融合的输出，模拟真机 dump
        tk.log = {'smoothed': s.smoothed, 'gated': s.gated, 'final_freq': str(s.final_freq),
                  'ble_state': s.ble_state, 'ble_level': str(s.ble_level),
                  'sent': '1' if s.sent else '0', 'sent_action': s.sent[0] if s.sent else '',
                  'sent_level': str(s.sent[1]) if s.sent else '-1'}
    return Video(vid, segs, dur, ticks, p, policy=policy)


def _write_synth(video, gt_path, run_path, fusion=''):
    with io.open(gt_path, 'w', encoding='utf-8', newline='\n') as fh:
        fh.write('# xcup-gt spec_version=1\n# video_id=%s\n# duration_ms=%d\n' % (video.vid, video.dur))
        fh.write('start_ms,end_ms,label,note\n')
        for s, e, l in video.segs:
            fh.write('%d,%d,%s,\n' % (s, e, l))
    with io.open(run_path, 'w', encoding='utf-8', newline='\n') as fh:
        fh.write('# xcup-bench-run v1\n# video_name=%s\n# params: %s%s\n' % (
            video.vid, ' '.join('%s=%s' % kv for kv in sorted(LEGACY_DEFAULTS.items())),
            (' FUSION=' + fusion) if fusion else ''))
        fh.write(','.join(M.RUN_COLS_NEEDED) + '\n')
        for t in video.ticks:
            def pr(ps, i):
                return '' if ps is None else '%.4f' % ps[i]
            lg = t.log
            row = [t.pos, t.wall, t.v, '%.4f' % t.vc, t.v_age, pr(t.vp, 0), pr(t.vp, 1), pr(t.vp, 2),
                   t.a, '%.4f' % t.ac, t.a_age, pr(t.ap, 0), pr(t.ap, 1), pr(t.ap, 2),
                   t.loud_lv, '%.4f' % t.loud_conf, 1, 100,
                   lg['smoothed'], lg['gated'], lg['final_freq'], lg['ble_state'], lg['ble_level'],
                   lg['sent'], lg['sent_action'], lg['sent_level']]
            row += [''] * (len(M.RUN_COLS_NEEDED) - len(row))      # 后加的列（a_pos）留空
            fh.write(','.join(str(x) for x in row) + '\n')


def selftest():
    fails = []

    def check(cond, msg):
        if not cond:
            fails.append(msg)

    layout_a = [('exclude', 4000), ('plot', 150000), ('other_adult', 40000), ('sex', 200000),
                ('plot', 70000), ('oral', 90000)]
    layout_b = [('exclude', 3000), ('plot', 120000), ('sex', 260000), ('other_adult', 30000)]
    syn = [_synth_video(1, layout_a, 's01'), _synth_video(2, layout_b, 's02'),
           _synth_video(3, layout_a, 's03')]

    # 1. CSV 落盘 -> 读回 -> 回放，与"实测列"逐 tick 一致（验证解析与回放链路）
    tmp = tempfile.mkdtemp(prefix='fusion_lab_')
    gdir, rdir = os.path.join(tmp, 'gt'), os.path.join(tmp, 'runs')
    os.makedirs(gdir)
    os.makedirs(rdir)
    for v in syn:
        _write_synth(v, os.path.join(gdir, v.vid + '.csv'), os.path.join(rdir, v.vid + '_base.csv'))
    loaded = load_videos(gdir, rdir, quiet=True)
    check(len(loaded) == 3, '应配对出 3 部合成片，实际 %d' % len(loaded))
    for v in loaded:
        steps = replay_legacy(v.ticks, v.params)
        bad = sum(1 for t, s in zip(v.ticks, steps)
                  if (t.log['gated'], t.log['ble_state'], t.log['sent'] == '1')
                  != (s.gated, s.ble_state, s.sent is not None))
        check(bad == 0, '%s: 读回后回放与实测列不一致 %d tick' % (v.vid, bad))

    # 2. 快速评测 == metrics.evaluate
    rnd = random.Random(7)
    for v in loaded:
        for trial in range(6):
            trans, cur = [], 'noise'
            for _ in range(rnd.randint(0, 12)):
                cur = 'do' if cur == 'noise' else 'noise'
                trans.append((rnd.randint(0, v.hi), cur))
            trans.sort()
            dedup, c = [], 'noise'
            for p_, s_ in trans:
                if s_ != c:
                    dedup.append((p_, s_))
                    c = s_
            rows = [{'pos': p_, 'sent': True, 'sent_action': s_, 'v': '', 'a': '', 'gated': s_}
                    for p_, s_ in dedup]
            rows.append({'pos': v.ticks[0].pos, 'sent': False, 'sent_action': '', 'v': '', 'a': '',
                         'gated': ''})
            rows.append({'pos': v.ticks[-1].pos, 'sent': False, 'sent_action': '', 'v': '', 'a': '',
                         'gated': ''})
            rows.sort(key=lambda r: r['pos'])
            m = M.evaluate(v.segs, v.dur, rows, v.policy)
            cm, fs = v.gti.score(dedup)
            lat, miss = latency(v, dedup)
            check(cm == m['cm'] and fs == m['false_stop'],
                  '%s: 快速评测与 metrics.evaluate 不一致 %r/%d vs %r/%d'
                  % (v.vid, cm, fs, m['cm'], m['false_stop']))
            check(lat == m['lat'] and miss == m['miss'], '%s: 延迟匹配与 metrics.evaluate 不一致' % v.vid)

    # 3. 证据融合语义
    cfg = EvidenceConfig(Modality([0, 1, 1], [0, 0.5, 1], [-1.0, 1.0]),
                         Modality([1, 1, 0], [0, 0.5, 1], [-0.5, 0.5]),
                         1.0, 0.5, 2.0, 2.0, 1.5, -0.5)
    ef = EvidenceFusion(cfg)
    do_p, no_p = (0.1, 0.2, 0.7), (0.9, 0.05, 0.05)
    check(ef.update(None, None, None, None, 1000)[0] == '', '没有任何结果时应输出空')
    check(ef.update(do_p, 900, None, None, 1000)[0] == 'Noise' and ef.L == 1.0, '一次视频 do 证据 L=1')
    check(ef.update(do_p, 900, None, None, 1500)[0] == 'Noise' and ef.L == 1.0, '同一结果不应重复计入')
    check(ef.update(do_p, 1400, None, None, 1500)[0] == 'do' and ef.L == 2.0, 'L 越过 theta_on 应启动并被 clamp')
    check(ef.update(do_p, 1900, None, None, 2000)[0] == 'do' and ef.L == 2.0, 'clamp 上限')
    check(ef.update(no_p, 2400, None, None, 2500)[0] == 'do' and ef.L == 1.0, '迟滞：L=1 仍在 do')
    check(ef.update(no_p, 2900, None, None, 3000)[0] == 'do' and ef.L == 0.0, '迟滞：L=0 仍在 do')
    check(ef.update(no_p, 3400, None, None, 3500)[0] == 'Noise', 'L<=theta_off 应停')
    check(ef.update(do_p, 100, None, None, 4000)[1] is None, '比已用结果更旧的结果应忽略')
    check(ef.update(do_p, 3500, None, None, 6000)[1] is None, '过期结果（age>2000）不给证据')
    check(ef.update((0.0, 0.0, 0.0), 5900, None, None, 6000)[1] is None, '全 0 概率视为推理失败')
    check(ef.update(None, 5950, None, None, 6000)[1] is None, '概率缺失视为推理失败')
    check(ef.update(None, None, (0.05, 0.05, 0.9), 5950, 6000)[2] == -0.5, '音频证据查表')
    ef.reset()
    check(ef.update(None, None, None, None, 5000)[0] == '' and ef.L == 0.0, 'reset 后回到无判定')
    rt = EvidenceConfig.from_json(cfg.to_json())
    check(rt.to_json() == cfg.to_json() and rt.params() == cfg.params(), '配置 JSON 往返不一致')
    try:
        EvidenceConfig(cfg.video, cfg.audio, 1, 1, 2.0, 2.0, 0.5, 1.0)
        fails.append('theta_off >= theta_on 应报错')
    except ValueError:
        pass
    try:
        EvidenceConfig(cfg.video, cfg.audio, 1, 1, 1.0, 4.0, 1.5, -1.0)
        fails.append('theta_off <= -clamp_lo 应报错')
    except ValueError:
        pass
    # 上下界分开：L 的下界是 -clamp_lo、上界是 +clamp_hi
    ef2 = EvidenceFusion(EvidenceConfig(cfg.video, cfg.audio, 1.0, 0.0, 1.0, 3.0, 0.5, -0.5))
    for i in range(6):
        ef2.update(no_p, 100 + i, None, None, 200 + i)
    check(ef2.L == -1.0, '下界应为 -clamp_lo，实际 %r' % ef2.L)
    for i in range(6):
        ef2.update(do_p, 300 + i, None, None, 400 + i)
    check(ef2.L == 3.0, '上界应为 +clamp_hi，实际 %r' % ef2.L)
    old = json.loads(cfg.to_json())
    del old['clamp_lo'], old['clamp_hi']
    old['clamp'] = 2.0
    c_old = EvidenceConfig.from_json(json.dumps(old))
    check((c_old.clamp_lo, c_old.clamp_hi) == (2.0, 2.0), '旧配置的对称 clamp 应同时作为上下界')

    # 4. simulate == replay_evidence（扫参用的快速版不能和精确版有出入）
    for v in loaded:
        for combo in [(1.0, 0.5, 4.0, 4.0, 1.5, -1.0), (1.0, 0.0, 2.0, 2.0, 0.5, -0.5),
                      (0.0, 1.0, 6.0, 6.0, 3.0, 0.5), (1.0, 1.0, 2.0, 8.0, 1.0, -1.5)]:
            c = EvidenceConfig(Modality(VIDEO_FEATURE[1], EDGES, [x * 0.4 - 2 for x in range(12)]),
                               Modality(AUDIO_FEATURES[0][1], EDGES, [x * 0.1 - 0.5 for x in range(12)]),
                               *combo)
            a = transitions_from_steps(v.ticks, replay_evidence(v.ticks, c, v.params))
            b = simulate(prepare(v, c), *combo, params=v.params)
            check(a == b, '%s %r: simulate 与 replay_evidence 结果不同' % (v.vid, combo))

    # 5. 保序回归 + 证据表
    y = pav([0.0, 2.0, 1.0, 3.0, -1.0], [1, 1, 1, 1, 4])
    check(all(y[i] <= y[i + 1] + 1e-12 for i in range(len(y) - 1)), 'pav 结果应单调')
    llr, _ = fit_table([(0.9, 'do')] * 50 + [(0.05, 'noise')] * 50 + [(0.05, 'do')] * 10)
    check(llr[bin_index(EDGES, 0.9)] > 0 > llr[bin_index(EDGES, 0.05)], '证据表方向错误')

    # 6. 拼接样本：停转 3 个 + 启动 3 个
    sps = make_splices(loaded[0])
    kinds = [s.splice for s in sps]
    check(kinds.count('stop') == 3 and kinds.count('start') == 3,
          '应拼出停转/启动各 3 个，实际 %r' % kinds)
    for s in sps:
        tr = M.gt_transitions(s.segs, s.policy)
        want = ('do', 'noise') if s.splice == 'stop' else ('noise', 'do')
        check(len(tr) == 1 and tr[0][1:] == want, '%s 样本应恰有一个 %s 切换' % (s.splice, want))
        check(all(s.ticks[i].pos < s.ticks[i + 1].pos and s.ticks[i].wall < s.ticks[i + 1].wall
                  for i in range(len(s.ticks) - 1)), '拼接后 pos/wall 应单调')
        b = s.segs[1][0]
        check(splice_seconds(s, []) == (0.0 if s.splice == 'stop' else (s.segs[1][1] // GRID_MS - b // GRID_MS) / 10.0),
              '一直不转时：停转样本应记 0 秒、启动样本应记整段')

    # 7. 拟合：合成数据上证据融合的代价应低于现有融合
    cost = Cost()
    train = loaded[:2] + make_splices(loaded[0]) + make_splices(loaded[1])
    best = fit(train, cost, 'evidence', QUICK_GRID)
    held = loaded[2]
    ce = Score.of(held, run_policy(held, 'evidence', best[1])[0], cost)
    cl = Score.of(held, run_policy(held, 'legacy')[0], cost)
    check(ce.cost < cl.cost, '合成数据上证据融合应优于现有融合（%.0f vs %.0f）' % (ce.cost, cl.cost))
    check(ce.fp() < cl.fp(), '合成数据上证据融合的误转应更少')

    # 8. live 模式采集的 run（app 带证据融合那几版录的）：check 要用 --config 改用证据融合回放
    live_cfg = EvidenceConfig(Modality(VIDEO_FEATURE[1], EDGES, [x * 0.4 - 2 for x in range(12)]),
                              Modality(AUDIO_FEATURES[1][1], EDGES, [x * 0.1 - 0.5 for x in range(12)]),
                              1.0, 1.0, 2.0, 8.0, 1.0, -1.5, meta={'id': 'selftest-live'})
    lv = loaded[0]
    steps = replay_evidence(lv.ticks, live_cfg, lv.params)
    for t, st in zip(lv.ticks, steps):
        t.log = dict(t.log, smoothed=st.smoothed, gated=st.gated, final_freq=str(st.final_freq),
                     ble_state=st.ble_state, ble_level=str(st.ble_level),
                     sent='1' if st.sent else '0', sent_action=st.sent[0] if st.sent else '',
                     sent_level=str(st.sent[1]) if st.sent else '-1')
    ldir = os.path.join(tmp, 'live')
    os.makedirs(os.path.join(ldir, 'gt'))
    os.makedirs(os.path.join(ldir, 'runs'))
    _write_synth(lv, os.path.join(ldir, 'gt', 'l01.csv'), os.path.join(ldir, 'runs', 'l01_base.csv'),
                 fusion='live/selftest-live')
    cpath = os.path.join(ldir, 'cfg.json')
    with io.open(cpath, 'w', encoding='utf-8') as fh:
        fh.write(live_cfg.to_json())
    ns = argparse.Namespace(gt_dir=os.path.join(ldir, 'gt'), runs_dir=os.path.join(ldir, 'runs'),
                            pair=[], policy='exclude', show=0, config=cpath)
    import contextlib
    with contextlib.redirect_stdout(io.StringIO()) as out:
        rc = cmd_check(ns)
    check(rc == 0 and '证据融合（selftest-live）' in out.getvalue(),
          'live 模式的 run 应该用证据融合回放并完全一致:\n' + out.getvalue()[-400:])
    ns.config = None
    with contextlib.redirect_stdout(io.StringIO()) as out:
        cmd_check(ns)
    check('需要用 --config' in out.getvalue() and '回放：' not in out.getvalue(),
          'live 模式的 run 没给 --config 时应跳过并提示:\n' + out.getvalue()[-400:])

    # 9. 音频管线标记；rescore 写的 run 带 a_ys_* 列，读出来的音频向量仍是 [sex, oral, noise]（与 app 一致）
    check(audio_pipe_of({'params': 'FUSION=live/x AUDIO_PIPE=fixed'}) == 'fixed'
          and audio_pipe_of({'pipeline': 'phone'}) == 'legacy' and audio_pipe_of({}) == 'legacy'
          and audio_pipe_of({'pipeline': 'clean'}) == 'clean', 'audio_pipe_of 映射不对')
    ypath = os.path.join(tmp, 'ys_run.csv')
    ycols = M.RUN_COLS_NEEDED + M.YS_COLS
    with io.open(ypath, 'w', encoding='utf-8', newline='\n') as fh:
        fh.write('# xcup-bench-run v1\n# params: AUDIO_PIPE=fixed\n' + ','.join(ycols) + '\n')
        row = dict((c, '') for c in ycols)
        row.update({'pos_ms': '1000', 'wall_ms': '5000', 'a_age': '30', 'a_p_sex': '0.6000',
                    'a_p_oral': '0.1000', 'a_p_noise': '0.3000', 'a_pos': '970'})
        row.update(dict((c, '0.2500') for c in M.YS_COLS))
        fh.write(','.join(row[c] for c in ycols) + '\n')
    _, yt = load_run(ypath)
    check(yt[0].ap == (f32(0.6), f32(0.1), f32(0.3)),
          '带 a_ys_* 的 run 应读成 3 维音频向量 [sex, oral, noise]，实际 %r' % (yt[0].ap,))

    for f in fails:
        print('  FAIL ' + f)
    print('selftest: %s' % ('FAILED (%d)' % len(fails) if fails else 'ALL PASS'))
    return 1 if fails else 0


# ======================================================================== main
def main():
    ap = argparse.ArgumentParser(description='Xcup 融合层离线实验台（详见文件头）')
    ap.add_argument('--selftest', action='store_true')
    sub = ap.add_subparsers(dest='cmd')

    def common(p):
        p.add_argument('gt_dir', nargs='?', default=os.path.join(ROOT, 'bench', 'gt'))
        p.add_argument('runs_dir', nargs='?', default=os.path.join(ROOT, 'bench', 'runs'))
        p.add_argument('--pair', action='append', default=[], metavar='GT=RUN',
                       help='显式指定 真值=run CSV，可重复')
        p.add_argument('--policy', default='exclude', choices=('exclude', 'do', 'noise'),
                       help='other_adult 怎么算（默认 exclude，与 metrics.py 一致）')

    p = sub.add_parser('check', help='回放现有融合，与 run CSV 逐 tick 比对')
    common(p)
    p.add_argument('--show', type=int, default=8, help='每部片列出前几处不一致')
    p.add_argument('--config', help='live 模式的 run（app 带证据融合那几版录的）采集时用的配置，用它回放'
                                    '（配置已从 app 删除，到 git 历史里按参数行的配置 id 找）')
    for name, hlp in (('eval', '留一视频交叉验证：各融合策略对比'),
                      ('oracle', '模态上限：假设某个模型完美')):
        p = sub.add_parser(name, help=hlp)
        common(p)
        p.add_argument('--cost', default='fp=1,fn=1,stop=3',
                       help='扫参目标：误转秒 x fp + 漏转秒 x fn + 误停次数 x stop（默认 fp=1,fn=1,stop=3）')
        p.add_argument('--quick', action='store_true', help='用小网格（调试用）')
        p.add_argument('--exhaustive', action='store_true',
                       help='扫全网格（默认先粗后细，快约 10 倍，结果基本一致）')
        p.add_argument('--jobs', type=int, default=0,
                       help='留一拟合用几个进程并行（默认 0 = CPU 核数；1 = 串行）')

    a = ap.parse_args()
    if a.selftest:
        sys.exit(selftest())
    fn = {'check': cmd_check, 'eval': cmd_eval, 'oracle': cmd_oracle}.get(a.cmd)
    if fn is None:
        ap.print_help()
        sys.exit(2)
    sys.exit(fn(a))


if __name__ == '__main__':
    main()
