#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
bench/tools/fusion_lab.py — 融合层离线实验台

run CSV 里已经存了两个模型每个 tick 的原始概率（v_p_* / a_p_*），融合层的全部输入都在。
这里把融合层整个搬到 Python 里离线重放：改融合不用再实时重跑视频，几秒一部片。

子命令:
  check   复刻线上 Java 融合，和 run CSV 里记下的逐 tick 输出比对。参数行有 ARB=（2026-10-07 起的 app）
          的 run 用融合仲裁 FusionArbiter 回放（还比对 arb 列），没有的用旧的平滑投票 + 启动仲裁；
          之后都接蓝牙状态机。一致率 >=99% 才说明回放器可信，后面的对比才有意义。
  eval    留一视频交叉验证：旧融合 / 融合仲裁 / 证据融合 / 三条基线（一直转/只用视频/只用音频）。
          证据融合与单模态基线的参数只用其余几部拟合；旧融合与融合仲裁用固定参数。
  oracle  假设某个模态完美（概率换成真值），端到端最多能到多少——决定先改哪个模型。
  golden  生成 FusionArbiterTest 的对拍数据（改了融合仲裁的规则 / 参数后要重新生成）。

用法:
  python3 bench/tools/fusion_lab.py check  bench/gt bench/runs
  python3 bench/tools/fusion_lab.py eval   bench/gt bench/runs
  python3 bench/tools/fusion_lab.py oracle bench/gt bench/runs
  python3 bench/tools/fusion_lab.py golden bench/gt bench/runs_initial
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
import contextlib
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
    """run CSV 的一行。v/a 是融合循环实际用到的动作串（已过新鲜度过滤），vp/ap 是原始概率。
    ays：这一行的音频结果对应的 YAMNet AudioSet 分组分数（cue_lab.py 接上去的，见那里的 FEAT_*），没有为 None。"""
    __slots__ = ('pos', 'wall', 'v', 'vc', 'v_age', 'vp', 'a', 'ac', 'a_age', 'ap',
                 'loud_lv', 'loud_conf', 'log', 'ays')

    def copy(self, **kw):
        t = Tick()
        for k in Tick.__slots__:
            setattr(t, k, kw[k] if k in kw else getattr(self, k, None))
        return t


LOG_COLS = ('smoothed', 'gated', 'final_freq', 'ble_state', 'ble_level', 'sent',
            'sent_action', 'sent_level', 'arb', 'arb_reset')


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
        t.ays = None
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
    """run CSV 文件头的参数行 -> dict。缺的键用当前 Java 默认值。

    参数行里有 ARB=<版本> 说明这份 run 是 FusionArbiter（融合仲裁）驱动蓝牙的：p['ARB'] 记版本，
    ARB_* 记它的参数（缺的用 ARB_DEFAULTS）。没有 ARB= 的是旧版（applyDoStartGate）录的 run。"""
    p = dict(LEGACY_DEFAULTS)
    p.update(ARB_DEFAULTS)
    p['ARB'] = ''
    for kv in (line or '').split():
        if '=' not in kv:
            continue
        k, v = kv.split('=', 1)
        if k == 'ARB':
            p['ARB'] = v
        elif k in p:
            ref = LEGACY_DEFAULTS.get(k, ARB_DEFAULTS.get(k))
            try:
                p[k] = int(float(v)) if isinstance(ref, int) else float(v)
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
    """旧融合：原 VideoProcessActivity 的 smoothedFusion + applyDoStartGate + selectBestAction（float 语义）。
    app 里 2026-10-07 已删除，这里留作"旧融合"对照，并用来回放旧融合时期录的 run。"""

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


# ======================================================================== 融合仲裁（FusionArbiter.java 的逐行移植）
# 2026-10-07 起 app 用它替代 applyDoStartGate：启动 / 维持 / 停止 分开判断，证据按"新推理结果"计数。
# 规则与取舍见 FusionArbiter.java 的类注释和 bench/README.md。改这里要同步改 Java，并重新生成对拍数据
# （python3 bench/tools/fusion_lab.py golden）。
ARB_DEFAULTS = {
    'ARB_V_FLOOR': 0.85, 'ARB_A_FLOOR': 0.90,
    'ARB_C_WIN': 3000, 'ARB_C_V': 2, 'ARB_C_A': 1,
    'ARB_Z_WIN': 9000, 'ARB_Z_N': 4,
    'ARB_MA_N': 4, 'ARB_MA_K': 3, 'ARB_MV_N': 4, 'ARB_MV_K': 2, 'ARB_M_MAX': 12000,
    'ARB_CONS_WIN': 6000, 'ARB_V_CONS': 70,
    'ARB_NEG_WIN': 4000, 'ARB_UNCLEAR_MAX': 6000, 'ARB_MAX_AGE': 2000, 'ARB_KEEP': 20000,
    'ARB_COLD': 6000, 'ARB_START_V_CONS': 80, 'ARB_START_MA_K': 4,
}
ARB_CAP = 128            # Java 账本的环形缓冲容量（正常用不满：每秒最多约 2 个结果 × 20 秒）
ARB_VERSIONS = ('v1',)   # 这份移植对应的 FusionArbiter.java 规则版本


def _arb_cls(action):
    return 1 if action == 'do' else (-1 if action == 'Noise' else 0)


class ArbLedger(object):
    """FusionArbiter.Ledger：一个模态的"新结果"账本，按结果时间戳去重。"""

    def __init__(self, max_age=MAX_AGE_MS, keep=20000):
        self.max_age, self.keep = max_age, keep
        self.reset(None)

    def reset(self, min_ts):
        self.items = []          # [(ts, cls, conf)]，旧的在前
        self.last_ts = None
        self.min_ts = min_ts

    def feed(self, action, conf, ts, now):
        if ts is None or ts <= 0 or ts > now or (self.last_ts is not None and ts <= self.last_ts):
            return False
        self.last_ts = ts
        if (self.min_ts is not None and ts < self.min_ts) or now - ts > self.max_age:
            return False
        if len(self.items) == ARB_CAP:
            self.items.pop(0)
        self.items.append((ts, _arb_cls(action), conf))
        lim = now - self.keep
        while self.items and self.items[0][0] <= lim:
            self.items.pop(0)
        return True

    def recent(self, now, n, max_win):
        lo = now - max_win
        got = 0
        for ts, _, _ in list(reversed(self.items))[:n]:
            if ts <= lo:
                break
            got += 1
        return got

    def count(self, now, win, c, floor=0.0):
        lo = now - win
        n = 0
        for ts, k, cf in reversed(self.items):
            if ts <= lo:
                break
            if k == c and cf >= floor:
                n += 1
        return n

    def total(self, now, win):
        lo = now - win
        n = 0
        for ts, _, _ in reversed(self.items):
            if ts <= lo:
                break
            n += 1
        return n

    def last_n_pos(self, now, n, max_win):
        lo = now - max_win
        pos = 0
        for ts, k, _ in list(reversed(self.items))[:n]:
            if ts <= lo:
                break
            if k == 1:
                pos += 1
        return pos


class FusionArbiter(object):
    """FusionArbiter.java 的移植。decide() 返回 'do' / 'Noise' / ''，reason 是写进 CSV arb 列的原因。"""

    def __init__(self, params=None):
        p = dict(ARB_DEFAULTS)
        for k in ARB_DEFAULTS:
            if params and k in params:
                p[k] = params[k]
        self.p = p
        self.v_floor = f32(p['ARB_V_FLOOR'])      # Java 里是 float 常量，边界比较按单精度
        self.a_floor = f32(p['ARB_A_FLOOR'])
        self.video = ArbLedger(p['ARB_MAX_AGE'], p['ARB_KEEP'])
        self.audio = ArbLedger(p['ARB_MAX_AGE'], p['ARB_KEEP'])
        self.last_support = -1
        self.prev_now = None
        self.reset_at = None
        self.reason = ''

    def reset(self, now, warmup=0):
        self.video.reset(now + warmup)
        self.audio.reset(now + warmup)
        self.last_support = -1
        self.prev_now = None
        self.reset_at = now
        self.reason = ''

    def _has_neg(self, now):
        w = self.p['ARB_NEG_WIN']
        return self.video.count(now, w, -1) + self.audio.count(now, w, -1) > 0

    def _video_support(self, now, cons_pct):
        p = self.p
        if self.video.last_n_pos(now, p['ARB_MV_N'], p['ARB_M_MAX']) < p['ARB_MV_K']:
            return False
        n = self.video.total(now, p['ARB_CONS_WIN'])
        k = self.video.count(now, p['ARB_CONS_WIN'], 1)
        return n > 0 and k * 100 >= cons_pct * n

    def decide(self, now, v, vc, v_ts, a, ac, a_ts, motor_on):
        p = self.p
        if self.prev_now is not None and now < self.prev_now:
            self.reset(now)                          # 墙钟往回跳
        self.prev_now = now
        self.video.feed(v, vc, v_ts, now)
        self.audio.feed(a, ac, a_ts, now)
        if not motor_on:
            self.last_support = -1
            if self.video.count(now, p['ARB_C_WIN'], 1, self.v_floor) >= p['ARB_C_V'] and \
                    self.audio.count(now, p['ARB_C_WIN'], 1, self.a_floor) >= p['ARB_C_A'] and \
                    (self._video_support(now, p['ARB_START_V_CONS'])
                     or self.audio.last_n_pos(now, p['ARB_MA_N'], p['ARB_M_MAX']) >= p['ARB_START_MA_K']):
                self.reason = 'start_both'
                return 'do'
            n = self.audio.total(now, p['ARB_Z_WIN'])
            if n >= p['ARB_Z_N'] and self.audio.count(now, p['ARB_Z_WIN'], 1) == n:
                self.reason = 'start_audio'
                return 'do'
            if self._has_neg(now):
                self.reason = 'off_neg'
                return 'Noise'
            self.reason = 'off_wait'
            return ''
        if self.last_support < 0:
            self.last_support = now
        sa = self.audio.last_n_pos(now, p['ARB_MA_N'], p['ARB_M_MAX']) >= p['ARB_MA_K']
        sv = self._video_support(now, p['ARB_V_CONS'])
        if sv or sa:
            self.last_support = now
            self.reason = 'hold_both' if (sv and sa) else ('hold_video' if sv else 'hold_audio')
            return 'do'
        waiting = now - self.last_support < p['ARB_UNCLEAR_MAX']
        cold = self.reset_at is not None and now - self.reset_at < p['ARB_COLD']
        if waiting and cold and (self.audio.recent(now, p['ARB_MA_N'], p['ARB_M_MAX']) < p['ARB_MA_N']
                                 or self.video.recent(now, p['ARB_MV_N'], p['ARB_M_MAX']) < p['ARB_MV_N']):
            self.reason = 'wait_sparse'
            return 'do'
        if self._has_neg(now):
            self.reason = 'stop_neg'
            return 'Noise'
        if not waiting:
            self.reason = 'stop_unclear'
            return 'Noise'
        self.reason = 'wait_unclear'
        return 'do'


def parse_arb_reset(text):
    """run CSV 的 arb_reset 列 '<ms>[+<预热ms>][:seek]' -> (ms, warmup, seek)；空串返回 None。"""
    text = (text or '').strip()
    if not text:
        return None
    m = re.match(r'^(-?\d+)(?:\+(\d+))?(:seek)?$', text)
    if not m:
        raise ValueError('arb_reset 列格式不对: %r' % text)
    return int(m.group(1)), int(m.group(2) or 0), bool(m.group(3))


def ble_seek_reset(ble):
    """seek 时 app 对蓝牙状态机做的清空（VideoProcessActivity 的 seek 处理；发送间隔与上次发出的档位不清）。"""
    ble.cur = ''
    ble.cur_start = 0
    ble.pend = ''
    ble.pend_start = 0
    ble.level = 1
    ble.level_since = 0
    ble.pend_level = None
    ble.pend_level_since = 0


def _tick_ts(t, which):
    age = t.v_age if which == 'v' else t.a_age
    return None if age is None or age < 0 else t.wall - age


def replay_legacy(ticks, params, actions=None, arbiter=None, reasons=None):
    """现有融合的回放。actions: 可选，逐 tick 替换 (v, vc, a, ac)。

    arbiter: None = 按 run 的参数行（有 ARB= 用融合仲裁，否则用旧的 applyDoStartGate）；
             True / False = 强制用融合仲裁 / 旧启动仲裁（eval 里两者并排比较）。
    reasons: 可选 list，融合仲裁每个 tick 的原因追加进去（check 比对 arb 列用）。"""
    use_arb = bool(params.get('ARB')) if arbiter is None else arbiter
    if use_arb and params.get('ARB') and params['ARB'] not in ARB_VERSIONS:
        print('⚠ run 的参数行是 ARB=%s，这份回放只实现了 %s：结果可能对不上' % (params['ARB'], '/'.join(ARB_VERSIONS)))
    core, ble, out = LegacyCore(params), Ble(params), []
    arb = FusionArbiter(params) if use_arb else None
    for i, t in enumerate(ticks):
        v, vc, a, ac = actions[i] if actions else (t.v, t.vc, t.a, t.ac)
        rs = parse_arb_reset((getattr(t, 'log', None) or {}).get('arb_reset', ''))
        if rs:                                       # app 在这个 tick 之前 reset 过（seek / 恢复播放）
            ms, warm, seek = rs
            if seek:                                 # seek 还清空了投票窗和蓝牙状态（旧融合回放也一样，与 arbiter_lab 一致）
                core = LegacyCore(params)
                ble_seek_reset(ble)
            if arb is not None:
                arb.reset(ms, warm)
        sm = core.smoothed(v, a, vc, ac) if arb is None else ''
        if arb is not None:
            g = arb.decide(t.wall, v, vc, _tick_ts(t, 'v'), a, ac, _tick_ts(t, 'a'), _is_sex(ble.cur))
            if reasons is not None:
                reasons.append(arb.reason)
        else:
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
        steps = replay_legacy(video.ticks, video.params, arbiter=False)
    elif policy == 'arbiter':
        steps = replay_legacy(video.ticks, video.params, arbiter=True)
    elif policy == 'always':
        steps = []
        ble = Ble(video.params)
        for t in video.ticks:
            sent = ble.update('do', final_freq(t.loud_lv, t.loud_conf, ble.level), t.wall)
            steps.append(Step('do', 'do', 0, ble.cur, ble.level, sent, None, None, None))
    else:
        steps = replay_evidence(video.ticks, cfg, video.params)
    return transitions_from_steps(video.ticks, steps), steps


POLICIES = [('legacy', '旧融合'), ('arbiter', '融合仲裁'), ('always', '一直转'), ('video_only', '只用视频'),
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
            if p in ('legacy', 'arbiter', 'always'):
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
    cols = ('gated', 'final_freq', 'ble_state', 'ble_level', 'sent')
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
            reasons = []
            steps = replay_legacy(v.ticks, v.params, reasons=reasons)
            what = '融合仲裁（%s）' % v.params['ARB'] if v.params.get('ARB') else '旧融合（applyDoStartGate）'
        # 融合仲裁的 run 比对原因列 arb；旧融合的 run 还有投票窗的 smoothed 列（2026-10-07 起 app 不再写）
        vcols = cols + ('arb',) if v.params.get('ARB') else ('smoothed',) + cols
        agree = dict((c, 0) for c in vcols)
        divs = []
        for k, (t, s) in enumerate(zip(v.ticks, steps)):
            lg = t.log
            got = {'smoothed': s.smoothed, 'gated': s.gated, 'final_freq': str(s.final_freq),
                   'ble_state': s.ble_state, 'ble_level': str(s.ble_level),
                   'sent': ('%s/%d' % s.sent) if s.sent else ''}
            want = dict((c, lg.get(c, '')) for c in vcols if c not in ('sent', 'arb'))
            want['sent'] = ('%s/%s' % (lg.get('sent_action'), lg.get('sent_level'))) \
                if lg.get('sent') == '1' else ''
            if 'arb' in vcols:
                got['arb'] = reasons[k]
                want['arb'] = lg.get('arb', '')
            for c in vcols:
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
        if 'arb' in vcols:                            # 原因也要对上：输出相同但走的规则不同，说明两边实现已经分叉
            worst = min(worst, agree['arb'] / n)
        sl, sr = Score.of(v, tl, Cost()), Score.of(v, tr, Cost())
        print('══ %s  %s  回放：%s ══' % (v.vid, os.path.basename(v.run_path), what))
        print('  逐 tick 一致率  ' + '  '.join('%s %.2f%%' % (c, 100.0 * agree[c] / n) for c in vcols))
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
    print('  · 代价 = 真实片子 + 拼接样本（误转秒×fp + 漏转秒×fn + 误停次数×stop）。')
    print('  · 旧融合与融合仲裁不拟合：旧融合是手调的固定参数；融合仲裁的参数是在这几部片上定的（样本内），')
    print('    它的留一数字看 bench/tools/arbiter_lab.py lovo。\n')
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
    print('旧融合 / 融合仲裁不用拟合；证据融合按留一视频交叉验证重新拟合。\n')
    scen = [('实际模型', ''), ('视频完美', 'v'), ('音频完美', 'a')]
    pols = [('legacy', '旧融合'), ('arbiter', '融合仲裁'), ('evidence', '证据融合')]
    for name, which in scen:
        vs = [oracle_video(v, which) if which else v for v in videos]
        per, sp, _, _ = evaluate_all(vs, cost, grid, log=lambda *a: None, variants=('evidence',),
                                     exhaustive=args.exhaustive, jobs=args.jobs)
        print_table(name, vs, per, sp, pols, show_videos=False)
    print('  · 如果"视频完美"在某种融合下几乎不涨，说明这种融合挡住了视频侧的改进（旧融合基本只听音频）。')
    print('  · 融合仲裁里视频单独就能维持（音频听不到目标声音时靠它），所以"音频完美"也压不住视频在剧情里的长时间误判。')
    return 0


# ======================================================================== 对拍数据（Java 单元测试用）
GOLDEN_PATH = os.path.join(ROOT, 'app', 'src', 'test', 'resources', 'fusion', 'arbiter_golden.csv')
GOLDEN_HEADER = 'reset,now,v_action,v_conf,v_ts,a_action,a_conf,a_ts,motor_on,exp_action,exp_reason'


def _gf(x):
    """float 写成 Java Float.parseFloat 能还原成同一个单精度值的十进制串。"""
    return '%.9g' % f32(x)


def _golden_rows_from_video(video, lo, hi):
    """真实 run 的一段：按 app 的方式回放（融合仲裁 + 蓝牙状态机），逐 tick 记下输入和期望输出。"""
    arb, ble, rows = FusionArbiter(), Ble(video.params), []
    first = True
    for t in video.ticks:
        if not (lo <= t.pos < hi):
            continue
        vts, ats = _tick_ts(t, 'v'), _tick_ts(t, 'a')
        on = _is_sex(ble.cur)
        if first:
            arb.reset(t.wall)        # 这一段从 seek 开始：之前产出的结果不算（Java 端同样先 reset）
        g = arb.decide(t.wall, t.v, t.vc, vts, t.a, t.ac, ats, on)
        rows.append((t.wall if first else '', t.wall, t.v, t.vc, -1 if vts is None else vts,
                     t.a, t.ac, -1 if ats is None else ats, on, g, arb.reason))
        first = False
        if g:
            ble.update(g, final_freq(t.loud_lv, t.loud_conf, ble.level), t.wall)
    return rows


def _golden_rows_synthetic():
    """边界用例：每一条都让被测的边界真正决定输出（评审用变异测试查过：边界写错一个符号就会对不上）。"""
    rows = []
    T = 1700000000000
    arb = FusionArbiter()
    W = ARB_DEFAULTS

    def tick(now, v='', vc=0.0, vts=-1, a='', ac=0.0, ats=-1, on=False, reset=None):
        if reset is not None:
            ms, warm = (int(x) for x in str(reset).split('+')) if '+' in str(reset) else (int(reset), 0)
            arb.reset(ms, warm)
        g = arb.decide(now, v, f32(vc), vts, a, f32(ac), ats, on)
        rows.append(('' if reset is None else reset, now, v, f32(vc), vts, a, f32(ac), ats, on, g, arb.reason))
        return g

    # 1) 两路佐证：置信度恰好等于下限（0.85f / 0.90f）算"强"，再低一点不算
    t = T
    tick(t, reset=t)
    for k in range(6):                                       # 先攒够维持条件（视频一致性）
        tick(t + 500 * (k + 1), 'do', 0.85, t + 500 * (k + 1) - 10)
    tick(t + 3500, 'do', 0.85, t + 3490, 'do', 0.90, t + 3495)                       # start_both
    t = T + 50000
    tick(t, reset=t)
    for k in range(6):
        tick(t + 500 * (k + 1), 'do', 0.8499999, t + 500 * (k + 1) - 10)
    tick(t + 3500, 'do', 0.8499999, t + 3490, 'do', 0.90, t + 3495)                  # 视频不够强：不启动
    # 2) 窗口边界：ts == now - win 不在窗口内（NEG_WIN）
    t = T + 100000
    tick(t, reset=t)
    tick(t + 200, 'Noise', 0.9, t + 100)                                             # off_neg
    tick(t + 100 + W['ARB_NEG_WIN'] - 1, 'Noise', 0.9, t + 100)                      # 还在窗口内
    tick(t + 100 + W['ARB_NEG_WIN'], 'Noise', 0.9, t + 100)                          # 恰好出窗：off_wait
    # 3) 过期边界：产出 2000ms 的结果还记，2001ms 的不记
    for age, base in ((W['ARB_MAX_AGE'], T + 150000), (W['ARB_MAX_AGE'] + 1, T + 160000)):
        tick(base, reset=base)
        tick(base + 10 + age, 'Noise', 0.9, base + 10)
    # 4) reset 边界：产出时刻 == reset 时刻的结果算，早 1ms 的不算
    for off, base in ((0, T + 170000), (-1, T + 180000)):
        tick(base + 100, 'Noise', 0.9, base + off, reset=base)
    # 5) 同一个结果重复出现只算一次：一个音频结果撑不起"4 个里 3 个"
    t = T + 200000
    tick(t, on=True, reset=t)
    for k in range(6):
        tick(t + 500 * (k + 1), a='do', ac=0.95, ats=t + 100, on=True)
    # 6) 音频兜底：9 秒内 >=4 个结果全是"转"才启动；夹一个 unclear 就要等它滑出窗口
    t = T + 300000
    tick(t, reset=t)
    seq = ['do', 'do', '', 'do', 'do', 'do', 'do', 'do', 'do', 'do', 'do', 'do', 'do', 'do']
    for i, x in enumerate(seq):
        now = t + 1000 * (i + 1)
        tick(now, 'Noise', 0.99, now - 300, x, 0.80 if x else 0.0, now - 50)
    # 7) 维持：音频 4 个里 3 个"转"撑得住，2 个撑不住；有"不转"就停，看不清有上限
    t = T + 400000
    tick(t, on=True, reset=t)
    pattern = ['do', 'do', 'do', 'do', 'Noise', 'do', 'Noise', 'Noise', '', '', '', '', '', '', '', '', '']
    for i, x in enumerate(pattern):
        now = t + 1000 * (i + 1)
        for d in (0, 500):
            tick(now + d, 'Noise', 0.9, now + d - 100, x, 0.9 if x else 0.0, now - 100, True)
    # 8) 维持·视频一致性：恰好 70%（7/10）算支持，6/10 不算
    for pos_n in (7, 6):
        t = T + 500000 + pos_n * 100000
        tick(t, on=True, reset=t)
        for i in range(10):
            now = t + 500 * (i + 1)
            x = 'do' if i >= 10 - pos_n else 'Noise'
            tick(now, x, 0.9, now - 50, 'do', 0.9, now - 30 if i % 2 == 0 else now - 530, True)
    # 9) 刚恢复（reset 后马达仍在转）：音频结果还不够 4 个时，视频一个"剧情"不能把马达停掉
    t = T + 1300000
    tick(t, on=True, reset=t)
    for k in range(12):
        now = t + 500 * (k + 1)
        tick(now, 'Noise', 0.9, now - 100, 'do', 0.95, t + 1000 * (k // 2) + 900 if k >= 2 else -1, True)
    # 10) 两路都是"转""不转"交替的强证据：不满足维持条件就不启动（防止每 2 秒开关一次）
    t = T + 1400000
    tick(t, reset=t)
    for k in range(20):
        now = t + 500 * (k + 1)
        tick(now, 'do' if k % 2 else 'Noise', 0.95, now - 50, 'do' if (k // 2) % 2 else 'Noise', 0.95,
             now - 100 if k % 2 == 0 else now - 600)
    # 11) 停转后不转（马达不转时 lastSupport 必须清零）：再转起来时从头计"看不清"的等待
    t = T + 1500000
    tick(t, on=True, reset=t)
    for k in range(4):
        tick(t + 1000 * (k + 1), a='do', ac=0.95, ats=t + 1000 * (k + 1) - 10, on=True)
    for k in range(20):
        tick(t + 5000 + 500 * k)
    tick(t + 15500, on=True)
    tick(t + 16000, on=True)
    # 12) 墙钟往回跳：自动 reset；"未来"的旧结果不记，也不挡住之后的新结果
    t = T + 1600000
    tick(t, on=True, reset=t)
    for k in range(4):
        tick(t + 1000 * (k + 1), a='do', ac=0.95, ats=t + 1000 * (k + 1) - 10, on=True)
    back = t + 4000 - 60000
    tick(back, a='do', ac=0.95, ats=t + 3990, on=True)
    for k in range(4):
        tick(back + 1000 * (k + 1), 'Noise', 0.9, back + 1000 * (k + 1) - 20, 'Noise', 0.9,
             back + 1000 * (k + 1) - 10, True)
    # 14) 两路都只有 unclear：结果够了但判断不出，最多再转 UNCLEAR_MAX 然后停
    t = T + 1650000
    tick(t, on=True, reset=t)
    for k in range(16):
        now = t + 500 * (k + 1)
        tick(now, '', 0.0, now - 20, '', 0.0, t + 1000 * (k // 2) + 400, True)
    # 15) 置信度比下限低 1 ulp：视频 / 音频都不算"强"（各自只差这一点，就启动不了）
    def below(x):
        return struct.unpack('<f', struct.pack('<I', struct.unpack('<I', struct.pack('<f', x))[0] - 1))[0]
    for vfl, afl, base in ((below(f32(0.85)), 0.95, T + 2000000), (0.95, below(f32(0.90)), T + 2010000),
                           (f32(0.85), f32(0.90), T + 2020000)):
        tick(base, reset=base)
        for k in range(10):
            tick(base + 400 * (k + 1), 'do', vfl, base + 400 * (k + 1) - 5, 'Noise', 0.9, base + 400 * (k + 1) - 300)
        tick(base + 4400, 'do', vfl, base + 4395, 'do', afl, base + 4390)
    # 16) 音频兜底窗口的下沿：9 秒前那个"不转"恰好出窗才启动
    for edge, base in ((1, T + 2100000), (0, T + 2120000)):
        tick(base, reset=base)
        tick(base + 10, a='Noise', ac=0.9, ats=base)
        for k in range(1, 9):
            tick(base + 1000 * k + 10, a='do', ac=0.7, ats=base + 1000 * k)
        tick(base + W['ARB_Z_WIN'] - edge, a='do', ac=0.7, ats=base + 8000)
    # 17) 环形缓冲容量 128：一个"不转"之后又来了 127 个"转"时它还在，第 128 个"转"把它挤掉才启动
    base = T + 2200000
    tick(base, reset=base)
    tick(base + 1, a='Noise', ac=0.9, ats=base + 1)
    for k in range(ARB_CAP):
        tick(base + 2 + k, a='do', ac=0.7, ats=base + 2 + k)
    # 18) reset 之后不能沿用 reset 之前的"上一个 now"去判断墙钟往回跳
    base = T + 2300000
    tick(base + 5000, reset=base + 4000)
    tick(base + 3000, 'Noise', 0.9, base + 2500, reset=base)
    # 19) 静音恢复的预热：reset(T, 2000) 之后，T+2000 之前产出的结果不算
    for off, base in ((1999, T + 2400000), (2000, T + 2410000)):
        tick(base, reset='%d+%d' % (base, 2000))
        tick(base + off + 100, 'Noise', 0.9, base + off)
    # 19b) 预热对音频同样生效
    for off, base in ((1999, T + 2420000), (2000, T + 2430000)):
        tick(base, reset='%d+%d' % (base, 2000))
        tick(base + off + 100, a='Noise', ac=0.9, ats=base + off)
    # 19c) 冷启动窗口从 reset 时刻算，不从预热结束算：预热 2 秒，6.2 秒时已经不等了
    base = T + 2440000
    tick(base, on=True, reset='%d+%d' % (base, 2000))
    for k, (x, ts) in enumerate((('do', 2100), ('do', 3100), ('do', 4100), ('Noise', 5100), ('Noise', 6100))):
        tick(base + ts + 100, a=x, ac=0.9, ats=base + ts, on=True)
    # 20) 冷启动窗口的上沿：reset 后 6 秒内结果不够判断的一路可以让马达先等，6 秒整就不等了
    base = T + 2500000
    tick(base, on=True, reset=base)
    for k in range(1, 12):
        tick(base + 500 * k, 'Noise', 0.9, base + 500 * k - 10, '', 0.0, -1, True)
    tick(base + W['ARB_COLD'] - 1, 'Noise', 0.9, base + 5490, '', 0.0, -1, True)
    tick(base + W['ARB_COLD'], 'Noise', 0.9, base + 5490, '', 0.0, -1, True)
    # 21) 启动迟滞：两路佐证要视频"转"占比 >= 80%（维持只要 70%），或音频最近 4 个全是"转"（维持只要 3 个）
    for vpos, apat, base in ((8, 'NDDD', T + 2600000), (7, 'NDDD', T + 2610000), (5, 'DDDD', T + 2620000),
                             (5, 'NDDD', T + 2630000)):
        tick(base, reset=base)
        for k in range(10):
            now = base + 500 * (k + 1)
            vv = 'do' if k >= 10 - vpos else 'Noise'
            aa = {'N': 'Noise', 'D': 'do'}[apat[(k - 6) // 1]] if k >= 6 else ''
            tick(now, vv, 0.95, now - 5, aa, 0.95 if aa else 0.0, now - 10 if aa else -1)
    # 13) 两路佐证的窗口边界：第二个强视频结果恰好 C_WIN 前产出就不算
    for edge, base in ((0, T + 1700000), (1, T + 1710000)):
        tick(base, reset=base)
        for k in range(8):
            tick(base + 400 * (k + 1), 'do', 0.95, base + 400 * (k + 1) - 5)
        last = base + 3200 - 5
        tick(last - 400 + W['ARB_C_WIN'] - edge, 'do', 0.5, last - 400 + W['ARB_C_WIN'] - edge - 1,
             'do', 0.95, last - 400 + W['ARB_C_WIN'] - edge - 1)
    return rows


def _golden_rows_random(seed=20261007, n=3000):
    """固定种子的随机序列：间隔、重复 / 倒序 / 过期时间戳、恰好在下限上的置信度、马达状态翻转、中途 reset。"""
    import random
    rnd = random.Random(seed)
    arb, rows = FusionArbiter(), []
    now = 1750000000000
    on = False
    last = {'v': None, 'a': None}
    confs = [0.5, 0.849999, 0.85, 0.850001, 0.899999, 0.9, 0.900001, 0.95, 0.99, 1.0]
    acts = ['do', 'do', 'do', 'Noise', 'Noise', '']
    reset = now
    arb.reset(now)
    for i in range(n):
        now += rnd.choice([0, 1, 499, 500, 500, 500, 501, 999, 1000, 2000, 2001, 7000])
        if rnd.random() < 0.01:
            reset = now - rnd.choice([0, 1, 500])
            arb.reset(reset)
        else:
            reset = ''
        for m in ('v', 'a'):
            r = rnd.random()
            if r < 0.45 or last[m] is None:
                last[m] = (rnd.choice(acts), f32(rnd.choice(confs)), now - rnd.choice([0, 1, 50, 300, 1999, 2000, 2001, 2500]))
            elif r < 0.5:
                last[m] = (last[m][0], last[m][1], last[m][2] - rnd.choice([1, 100]))      # 倒序时间戳
        if rnd.random() < 0.08:
            on = not on
        v, a = last['v'], last['a']
        g = arb.decide(now, v[0], v[1], v[2], a[0], a[1], a[2], on)
        rows.append((reset, now, v[0], v[1], v[2], a[0], a[1], a[2], on, g, arb.reason))
    return rows


def cmd_golden(args):
    """生成 FusionArbiterTest 的对拍数据：真实 run 的几段 + 合成边界用例。"""
    videos = load_videos(args.gt_dir, args.runs_dir, args.pair, args.policy, quiet=True)
    rows = []
    for v in videos:
        if v.vid not in ('v01', 'v03', 'v04', 'v07'):
            continue
        trans = [s for s, e, lab in v.segs[1:]]
        for c in trans[:3]:
            rows += _golden_rows_from_video(v, max(0, c - 60000), c + 60000)
    rows += _golden_rows_synthetic()
    rows += _golden_rows_random()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with io.open(args.out, 'w', encoding='utf-8', newline='\n') as fh:
        fh.write(GOLDEN_HEADER + '\n')
        for r in rows:
            reset, now, v, vc, vts, a, ac, ats, on, g, why = r
            fh.write(','.join([str(reset), str(now), v, _gf(vc), str(vts), a, _gf(ac), str(ats),
                               '1' if on else '0', g, why]) + '\n')
    reasons = {}
    for r in rows:
        reasons[r[-1]] = reasons.get(r[-1], 0) + 1
    print('写入 %s：%d 行；原因覆盖 %s' % (args.out, len(rows), sorted(reasons.items())))
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


def _write_synth(video, gt_path, run_path, fusion='', extra_params='', corrupt_arb_every=0):
    with io.open(gt_path, 'w', encoding='utf-8', newline='\n') as fh:
        fh.write('# xcup-gt spec_version=1\n# video_id=%s\n# duration_ms=%d\n' % (video.vid, video.dur))
        fh.write('start_ms,end_ms,label,note\n')
        for s, e, l in video.segs:
            fh.write('%d,%d,%s,\n' % (s, e, l))
    with io.open(run_path, 'w', encoding='utf-8', newline='\n') as fh:
        fh.write('# xcup-bench-run v1\n# video_name=%s\n# params: %s%s%s\n' % (
            video.vid, ' '.join('%s=%s' % kv for kv in sorted(LEGACY_DEFAULTS.items())),
            (' FUSION=' + fusion) if fusion else '', (' ' + extra_params) if extra_params else ''))
        cols = list(M.RUN_COLS_NEEDED)
        if 'ARB=' not in extra_params:              # 旧融合的 run（2026-10-07 之前的 app）还有投票窗的 smoothed 列
            cols.insert(cols.index('gated'), 'smoothed')
        fh.write(','.join(cols) + '\n')
        for k, t in enumerate(video.ticks):
            def pr(ps, i):
                return '' if ps is None else '%.4f' % ps[i]
            lg = t.log
            val = {'pos_ms': t.pos, 'wall_ms': t.wall,
                   'v_action': t.v, 'v_conf': '%.6f' % t.vc, 'v_age': t.v_age,
                   'v_p_plot': pr(t.vp, 0), 'v_p_oral': pr(t.vp, 1), 'v_p_sex': pr(t.vp, 2),
                   'a_action': t.a, 'a_conf': '%.6f' % t.ac, 'a_age': t.a_age,
                   'a_p_sex': pr(t.ap, 0), 'a_p_oral': pr(t.ap, 1), 'a_p_noise': pr(t.ap, 2),
                   'loud_lv': t.loud_lv, 'loud_conf': '%.4f' % t.loud_conf, 'loud_valid': 1, 'loud_age': 100,
                   'smoothed': lg.get('smoothed', ''), 'gated': lg['gated'], 'final_freq': lg['final_freq'],
                   'ble_state': lg['ble_state'], 'ble_level': lg['ble_level'], 'sent': lg['sent'],
                   'sent_action': lg['sent_action'], 'sent_level': lg['sent_level'], 'a_pos': -1,
                   'arb': 'bogus' if corrupt_arb_every and k % corrupt_arb_every == 0 else lg.get('arb', ''),
                   'arb_reset': lg.get('arb_reset', '')}
            fh.write(','.join(str(val.get(c, '')) for c in cols) + '\n')


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

    # 10. 维持时交 'do' 会取消状态机里没执行的停转计时；交 '' 不会（旧计时保留，下一次 Noise 立刻停）
    params0 = dict(LEGACY_DEFAULTS)
    T0 = 100000
    for hold, want_stop_at in (('do', T0 + 4500), ('', T0 + 3500)):
        ble = Ble(params0)
        check(ble.update('do', 1, T0) is not None, '状态机应在 T0 启动')
        ble.update('Noise', 1, T0 + 2500)            # 出现 Noise，开始停转计时（需稳定 1 秒）
        if hold:
            ble.update(hold, 1, T0 + 3000)           # 维持仲裁交 'do'；交 '' 时融合循环根本不调状态机
        stops = [t for t in (T0 + 3500, T0 + 4000, T0 + 4500) if ble.update('Noise', 1, t) is not None]
        check(stops[:1] == [want_stop_at], '维持时交 %r：期望 %dms 停转，实际 %r' % (hold, want_stop_at, stops))
    # 11. 融合仲裁：参数行解析 + 按 ARB= 自动切换回放方式 + 同一结果不重复计数
    pp = parse_params('VIDEO_DO=0.65 AUDIO_PIPE=fixed ' + 'ARB=v1 ARB_V_FLOOR=0.85 ARB_Z_WIN=9000 ARB_MV_N=4')
    check(pp['ARB'] == 'v1' and pp['ARB_V_FLOOR'] == 0.85 and pp['ARB_Z_WIN'] == 9000
          and isinstance(pp['ARB_Z_WIN'], int) and parse_params('VIDEO_DO=0.65')['ARB'] == '',
          'ARB 参数行解析不对: %r' % ({k: pp[k] for k in pp if k.startswith('ARB')},))
    arb = FusionArbiter()
    arb.reset(1000)
    for k in range(6):                               # 同一个音频结果在 6 个 tick 里重复出现
        g = arb.decide(1500 + 500 * k, '', 0.0, None, 'do', f32(0.95), 1400, True)
    check(g == 'do' and arb.reason == 'wait_sparse',
          '同一个结果只该算一次（1 个结果不构成支持、音频还判断不了），实际 %r/%r' % (g, arb.reason))

    # 11b. 提交进仓库的 Java 对拍数据，Python 移植逐行重放也必须一致（只改 Python 不改 Java 时这里会报）
    if os.path.exists(GOLDEN_PATH):
        ga = FusionArbiter()
        bad = []
        with io.open(GOLDEN_PATH, encoding='utf-8') as fh:
            head = fh.readline().strip()
            check(head == GOLDEN_HEADER, '对拍数据表头不对: %r' % head)
            for ln, line in enumerate(fh, 2):
                f = line.rstrip('\n').split(',')
                if f[0]:
                    ms, _, warm = f[0].partition('+')
                    ga.reset(int(ms), int(warm or 0))
                g = ga.decide(int(f[1]), f[2], f32(float(f[3])), int(f[4]), f[5], f32(float(f[6])), int(f[7]),
                              f[8] == '1')
                if (g, ga.reason) != (f[9], f[10]) and len(bad) < 3:
                    bad.append('第 %d 行 期望 %s/%s 实际 %s/%s' % (ln, f[9], f[10], g, ga.reason))
        check(not bad, 'Python 移植重放 %s 对不上（改了规则就重新 golden）: %s' % (GOLDEN_PATH, bad))

    # 12. 新格式 run（ARB=v1 + arb / arb_reset 列，中途 reset 过）：check 逐 tick 对得上；原因列错 5% 时 check 判不可信
    arb_params = 'ARB=v1 ' + ' '.join('%s=%s' % kv for kv in sorted(ARB_DEFAULTS.items()))
    v0 = syn[0]
    phone = [t.copy(log=dict(t.log)) for t in v0.ticks]
    for i in (len(phone) // 3, 2 * len(phone) // 3):
        phone[i].log['arb_reset'] = str(phone[i].wall - 200)
    pp = parse_params(' '.join('%s=%s' % kv for kv in sorted(LEGACY_DEFAULTS.items())) + ' ' + arb_params)
    why = []
    for t, st in zip(phone, replay_legacy(phone, pp, reasons=why)):
        t.log.update({'gated': st.gated, 'final_freq': str(st.final_freq),
                      'ble_state': st.ble_state, 'ble_level': str(st.ble_level),
                      'sent': '1' if st.sent else '0', 'sent_action': st.sent[0] if st.sent else '',
                      'sent_level': str(st.sent[1]) if st.sent else '-1'})
    for t, r in zip(phone, why):
        t.log['arb'] = r
    adir = os.path.join(tmp, 'arb')
    os.makedirs(os.path.join(adir, 'gt'))
    os.makedirs(os.path.join(adir, 'runs'))
    for corrupt in (0, 20):
        _write_synth(v0.with_ticks(phone), os.path.join(adir, 'gt', v0.vid + '.csv'),
                     os.path.join(adir, 'runs', v0.vid + '_base.csv'), extra_params=arb_params,
                     corrupt_arb_every=corrupt)
        ns = argparse.Namespace(gt_dir=os.path.join(adir, 'gt'), runs_dir=os.path.join(adir, 'runs'), pair=[],
                                policy='exclude', show=0, config=None)
        with contextlib.redirect_stdout(io.StringIO()) as out:
            rc = cmd_check(ns)
        txt = out.getvalue()
        if corrupt:
            check(rc == 1, '原因列 5%% 对不上时 check 应判不可信:\n%s' % txt)
        else:
            check(rc == 0 and 'arb 100.00%' in txt and 'gated 100.00%' in txt,
                  '新格式 run（含 reset 记号）回放应逐 tick 一致:\n%s' % txt)

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

    p = sub.add_parser('golden', help='生成 FusionArbiterTest 的对拍数据（改了融合仲裁的规则/参数后要重新生成）')
    common(p)
    p.set_defaults(runs_dir=os.path.join(ROOT, 'bench', 'runs_initial'))
    p.add_argument('--out', default=GOLDEN_PATH)

    a = ap.parse_args()
    if a.selftest:
        sys.exit(selftest())
    fn = {'check': cmd_check, 'eval': cmd_eval, 'oracle': cmd_oracle, 'golden': cmd_golden}.get(a.cmd)
    if fn is None:
        ap.print_help()
        sys.exit(2)
    sys.exit(fn(a))


if __name__ == '__main__':
    main()
