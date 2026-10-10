#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
bench/tools/cue_lab.py — 用 YAMNet 自带的 AudioSet 521 类分数（tflite 第 0 个输出）判断"音频这一刻靠不靠得住"，
离线评估它对融合仲裁的改进。这里是实验台（各种开关组合）；选定的 a1bvid 2026-10-11 做进了 app（AudioCue.java +
FusionArbiter v2，Python 移植在 fusion_lab.py），--selftest 核对两边逐 tick 一致。

数据：真机 run（默认 bench/runs_initial）+ 同名的 <run>.yamnet.npz（默认 bench/rescored/audio_old，audio_rescore.py
rescore 写的，每个音频结果的 521 类分数，按结果时间戳 wall-a_age 对齐）。

每个音频结果判一个"线索"（AudioCue，见 cue_of），按顺序判：
  dialog 对白   人声组很低、Speech 高：分类头这时判"转"多半是把说话当成了动作声
  music  音乐   没有人声、没有撞击/口腔声，Music 高而且不低于 Speech（带人声的歌也算）
  quiet  安静   没有人声、没有对白、没有撞击/口腔声，Silence 高
  blind  听不出 同上，但也不是音乐 / 安静
  vocal  人声   人声组明显
  impact 拍打   撞击/口腔声明显（拍打、敲击、咀嚼……）
  other  其它   以上都不是
  None   未知   没有分数（老 run 对不上时间戳等）：所有规则都不生效，行为与 v1 完全一样

在融合仲裁 v1（fusion_lab.FusionArbiter，与 FusionArbiter.java 逐 tick 对拍的那份）上加几类开关，全关时与 v1 逐 tick 一致：
  A1 "听不出"时放行视频单独启动（新原因 start_video），条件比两路佐证更严
  A2 "听不出"时音频的"不转"不算结束证据
  A3 "听不出"时音频的"转"不算启动证据：1 = 只在音频兜底（"9 秒内全部是转"）里；2 = 所有启动规则。
     A3 的"听不出"单独由 a3_set 定（A1 放行视频时安静的风险大，A3 不让安静里的"转"启动则正需要安静）
  B  对白时音频的"转"按 unclear 记：start = 只在启动判断里；all = 启动和维持都算；
     vid = 只在启动判断里、而且只在视频最近 bvn 个结果都是置信度 ≥ bvc 的"剧情"时（2026-10-11 看了 v09/v11 之后加的）
  B2 最近 bv 个音频结果都是对白时，视频不能单独维持
  P1 最近 p_n 个音频结果都是"确认的转"（分类头判转 + 人声 / 拍打）时直接启动（新原因 start_cue）
  P2 维持时，最近 4 个音频结果里有确认的转，音频支持就从 3 个转降到 2 个（新原因 hold_cue）；
     最多续 p2_max（从最后一次正常支持算起），零星的"转+人声"不能无限续持
"听不出"包括哪些线索由 uninf 决定：m = 音乐，mq = 音乐 + 安静，mqb = 音乐 + 安静 + 听不出。

用法（默认 bench/gt + bench/runs_initial + bench/rescored/audio_old）:
  python3 bench/tools/cue_lab.py stats              # 每部片动作 / 剧情里各线索的占比，以及分类头在各线索下判转的比例
  python3 bench/tools/cue_lab.py blame              # 误判归因：基线的误转 / 漏转 / 误停落在哪种线索下（各规则的上限）
  python3 bench/tools/cue_lab.py stress [--cfg ..]  # 压力场景（带线索）：v1 对照某个配置
  python3 bench/tools/cue_lab.py eval   [--cfg ..]  # 某个配置的完整报告（同 arbiter_lab report）+ 逐片对比基线
  python3 bench/tools/cue_lab.py scan               # 单条规则扫阈值（样本内，只看方向，不拿来选参）
  python3 bench/tools/cue_lab.py lovo   [--jobs 4]  # 留一视频选组 / 阈值 / 规则组合，按验收线判定
  python3 bench/tools/cue_lab.py sens   [--cfg ..]  # 阈值敏感曲线：一次动一个参数
  python3 bench/tools/cue_lab.py --selftest
--cfg 用 k=v 覆盖默认配置，如 --cfg a1=1 b=start d_sp=0.8；可用的键见 CFG0。--preset 用 PRESETS 里起好名字的候选。
其他片子：在命令后面给 <gt 目录> <runs 目录>，--npz 指定 .yamnet.npz 所在目录（都可以逗号分隔给几个），例如全部 13 部：
  python3 bench/tools/cue_lab.py eval bench/gt bench/runs_initial,bench/runs_test \
      --npz bench/rescored/audio_old,bench/rescored/test_audio_old --preset a1bvid

结论（详见 bench/README.md 第 6 节）：2026-10-10 在 7 部片上，A2、A3、B=all、B2、P 留一视频都不成立；B=start 留一成立，
但口交时 YAMNet 常把口腔声判成 Speech，会推迟启动。2026-10-11 加 6 部新片：B=start 让 v11（边打电话边口交）漏转翻倍；
改成 bvid（只在视频确定是剧情时否决）后 v10/v12/v13 独立检验没有坏处。13 部片上推荐 a1bvid。候选见 PRESETS。
"""
import argparse
import contextlib
import io
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fusion_lab as F     # noqa: E402
import arbiter_lab as A    # noqa: E402
import metrics as M        # noqa: E402

ROOT = F.ROOT
GRID_MS = F.GRID_MS
DEF_GT = os.path.join(ROOT, 'bench', 'gt')
DEF_RUNS = os.path.join(ROOT, 'bench', 'runs_initial')
DEF_NPZ = os.path.join(ROOT, 'bench', 'rescored', 'audio_old')


# ======================================================================== 类别组（按含义定，下标见 yamnet_class_map.csv）
GROUPS = [
    # 非语言人声。核心组：呜咽 21、呻吟 22、叹气 23、哼 33、咕哝 34、呼吸 36、倒吸气 39、喘 40
    ('voc_core', (21, 22, 23, 33, 34, 36, 39, 40)),
    # 扩展组 = 核心组 + 尖叫 11、笑 13/15/16/17/18、哭 19（上一轮按判别力挑出来的那组）
    ('voc_ext', (11, 13, 15, 16, 17, 18, 19, 21, 22, 23, 33, 34, 36, 39, 40)),
    ('speech', (0,)),                  # Speech
    ('speech_ext', (0, 2, 3)),         # + Conversation、Narration
    ('music', (132,)),                 # Music
    ('silence', (494,)),               # Silence
    # 撞击声和口腔声：咀嚼 49、敲 353、轻敲 354、挤压水声 441、拍打 461、重击 462。动作里常有，有它们就不算"听不出"
    ('impact', (49, 353, 354, 441, 461, 462)),
]
FEAT = dict((name, i) for i, (name, _) in enumerate(GROUPS))

DIAL, MUSIC, QUIET, BLIND, VOCAL, IMPACT, OTHER = 'dialog', 'music', 'quiet', 'blind', 'vocal', 'impact', 'other'
CUES = (DIAL, MUSIC, QUIET, BLIND, VOCAL, IMPACT, OTHER, None)
CUE_ZH = {DIAL: '对白', MUSIC: '音乐', QUIET: '安静', BLIND: '听不出', VOCAL: '人声', IMPACT: '拍打', OTHER: '其它',
          None: '未知'}
CONFIRM = frozenset([VOCAL, IMPACT])
UNINF_SETS = {'m': frozenset([MUSIC]), 'mq': frozenset([MUSIC, QUIET]), 'mqb': frozenset([MUSIC, QUIET, BLIND])}

CFG0 = dict(
    voc='ext',          # 人声组：core / ext
    sp='speech',        # 对白组：speech / ext
    d_voc=0.02,         # 对白：人声 < d_voc 且 Speech ≥ d_sp
    d_sp=0.7,
    b_voc=0.05,         # 听不出：人声 < b_voc、Speech < b_sp、撞击/口腔声 < b_imp（b_imp=0 表示不看撞击声）
    b_sp=0.2,
    b_imp=0.1,
    b_mus=0.3,          # Music ≥ b_mus 且 ≥ Speech 为音乐；否则 Speech < b_sp 时 Silence ≥ b_sil 为安静，都不是为 blind
    b_sil=0.3,
    p_voc=0.3,          # 人声组 ≥ p_voc 为 vocal；撞击/口腔声 ≥ p_imp 为 impact
    p_imp=0.2,
    uninf='mq',         # A1 / A2 的"听不出"：m / mq / mqb
    a3_set='mqb',       # A3 的"听不出"：m / mq / mqb
    a1=0, a1_n=2, a1_cv=3, a1_cons=90,   # A1：音频最近 a1_n 个都听不出 + 3 秒内视频强转 ≥ a1_cv + 视频占比 ≥ a1_cons%
    a2=0,
    a3=0,               # 0 / 1（只管音频兜底）/ 2（所有启动规则）
    b='off',            # off / start / all / vid
    bvn=4, bvc=0.9,     # b=vid：视频最近 bvn 个结果都是置信度 ≥ bvc 的"剧情"时才否决
    bv=0,               # B2：最近 bv 个音频结果都是对白时视频不能单独维持（0 = 关）
    p1=0, p_n=3,        # P1：最近 p_n 个音频结果都是确认的转 → 启动
    p2=0,               # P2：确认的转让音频维持少要一个转
    p2_max=6000,        # P2 最多续多久（ms，从最后一次正常支持算起）
)
_INT_KEYS = ('a1', 'a1_n', 'a1_cv', 'a1_cons', 'a2', 'a3', 'bv', 'p1', 'p_n', 'p2', 'p2_max', 'bvn')

# 2026-10-10 留一视频 + 敏感曲线之后剩下的候选（阈值都在平台中间，前后挪一档结果几乎不变）
PRESETS = {
    # 音频最近 4 个结果都听不出（音乐 / 安静 / 听不出）时，视频 3 秒内强转 ≥3 个、6 秒内占比 ≥90% 就单独启动
    'a1': dict(a1=1, uninf='mqb', a1_n=4, a1_cons=90),
    # 对白（人声 < 0.01、Speech ≥ 0.7）时分类头的"转"不算启动证据；维持不受影响
    'bstart': dict(b='start', d_voc=0.01, d_sp=0.7),
    'a1b': dict(a1=1, uninf='mqb', a1_n=4, a1_cons=90, b='start', d_voc=0.01, d_sp=0.7),
    # 2026-10-11 看了 v09（剧情对白、视频很确定是剧情）和 v11（边打电话边口交、视频不确定）之后加的：
    # 对白的"转"只在视频最近 4 个结果都是 ≥0.9 的剧情时才不算启动证据。是事后设计的，要靠 v10/v12/v13 和新片检验
    'bvid': dict(b='vid', d_voc=0.01, d_sp=0.7, bvn=4, bvc=0.9),
    # 2026-10-11 在 13 部片上的推荐组合（见 bench/README.md 第 6 节）
    'a1bvid': dict(a1=1, uninf='mqb', a1_n=4, a1_cons=90, b='vid', d_voc=0.01, d_sp=0.7, bvn=4, bvc=0.9),
}
_STR_KEYS = ('voc', 'sp', 'uninf', 'a3_set', 'b')


def make_cfg(**kw):
    c = dict(CFG0)
    for k, v in kw.items():
        if k not in c:
            raise KeyError('未知的配置项 %r（可用：%s）' % (k, ', '.join(sorted(CFG0))))
        c[k] = v
    return c


def parse_cfg(items, preset=''):
    kw = {}
    if preset:
        if preset not in PRESETS:
            sys.exit('没有这个 preset：%s（可用：%s）' % (preset, ', '.join(sorted(PRESETS))))
        kw.update(PRESETS[preset])
    for it in items or ():
        for part in it.replace(',', ' ').split():
            k, v = part.split('=', 1)
            if k in _STR_KEYS:
                kw[k] = v
            elif k in _INT_KEYS:
                kw[k] = int(v)
            else:
                kw[k] = float(v)
    return make_cfg(**kw)


def cfg_str(c, full=False):
    """只写和 v1 不同、且起作用的部分。"""
    parts = []
    if c['a1'] or c['a2'] or c['a3']:
        on = [n for n in ('a1', 'a2', 'a3') if c[n]]
        parts.append('+'.join(x.upper() + ('(兜底)' if x == 'a3' and c['a3'] == 1 else '') for x in on))
        sets = []
        if c['a1'] or c['a2']:
            sets.append(c['uninf'])
        if c['a3']:
            sets.append('A3:' + c['a3_set'])
        parts.append('听不出=%s(人声<%g,Speech<%g%s)' % ('/'.join(sets), c['b_voc'], c['b_sp'],
                                                     ',撞击<%g' % c['b_imp'] if c['b_imp'] else ''))
        if c['a1']:
            parts.append('A1[n=%d,强转≥%d,占比≥%d%%]' % (c['a1_n'], c['a1_cv'], c['a1_cons']))
    if c['b'] != 'off' or c['bv']:
        parts.append('%s(人声<%g,Speech≥%g%s)' % ('+'.join(([('B=' + c['b'])] if c['b'] != 'off' else [])
                                                           + (['B2[%d]' % c['bv']] if c['bv'] else [])),
                                                 c['d_voc'], c['d_sp'],
                                                 ',视频最近%d个都是≥%g的剧情' % (c['bvn'], c['bvc']) if c['b'] == 'vid' else ''))
    if c['p1'] or c['p2']:
        parts.append('%s(人声≥%g或拍打≥%g%s%s)' % ('+'.join(x.upper() for x in ('p1', 'p2') if c[x]), c['p_voc'],
                                                c['p_imp'], ',n=%d' % c['p_n'] if c['p1'] else '',
                                                ',续%gs' % (c['p2_max'] / 1000.0) if c['p2'] else ''))
    if parts or full:
        parts.append('人声组=%s' % c['voc'])
        if c['sp'] != 'speech':
            parts.append('对白组=%s' % c['sp'])
    return ' '.join(parts) if parts else 'v1（不用线索）'


def cue_of(f, c):
    """一个音频结果的线索。f：分组分数（GROUPS 的顺序），None = 没有分数。"""
    if f is None:
        return None
    voc = f[1] if c['voc'] == 'ext' else f[0]
    sp = f[3] if c['sp'] == 'ext' else f[2]
    if voc < c['d_voc'] and sp >= c['d_sp']:
        return DIAL
    if voc < c['b_voc'] and (not c['b_imp'] or f[6] < c['b_imp']):
        if f[4] >= c['b_mus'] and f[4] >= sp:
            return MUSIC
        if sp < c['b_sp']:
            return QUIET if f[5] >= c['b_sil'] else BLIND
    if voc >= c['p_voc']:
        return VOCAL
    if f[6] >= c['p_imp']:
        return IMPACT
    return OTHER


def group_features(scores):
    """[N, 521] → [N, len(GROUPS)] float32，按 GROUPS 里的下标顺序累加（GROUPS 与 fusion_lab.CUE_GROUPS 相同）。"""
    return F.cue_features_np(scores)


# ======================================================================== 带线索的融合仲裁
# 账本一条：(ts, 置信度, 原始类别, 启动用类别, 兜底用类别, 维持用类别, 算不算结束证据, 听不出, 确认的转, 对白,
#           b=vid 否决生效时的启动用类别, b=vid 否决生效时的兜底用类别)
_CONF, _K0, _KS, _KZ, _KH, _NEG, _UN, _CF, _DL, _KSV, _KZV = 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11


class _Ledger(object):
    """fusion_lab.ArbLedger 加上每条结果的派生类别（规则按阶段取不同的列）。去重、过期、容量与 v1 相同。"""
    __slots__ = ('max_age', 'keep', 'items', 'last_ts', 'min_ts')

    def __init__(self, max_age, keep):
        self.max_age, self.keep = max_age, keep
        self.reset(None)

    def reset(self, min_ts):
        self.items = []
        self.last_ts = None
        self.min_ts = min_ts

    def feed(self, ts, rec, now):
        if ts is None or ts <= 0 or ts > now or (self.last_ts is not None and ts <= self.last_ts):
            return False
        self.last_ts = ts
        if (self.min_ts is not None and ts < self.min_ts) or now - ts > self.max_age:
            return False
        if len(self.items) == F.ARB_CAP:
            self.items.pop(0)
        self.items.append((ts,) + rec)
        lim = now - self.keep
        while self.items and self.items[0][0] <= lim:
            self.items.pop(0)
        return True

    def recent(self, now, n, max_win):
        lo = now - max_win
        got = 0
        for it in reversed(self.items[-n:] if n > 0 else []):
            if it[0] <= lo:
                break
            got += 1
        return got

    def count(self, now, win, col, c, floor=0.0):
        lo = now - win
        n = 0
        for it in reversed(self.items):
            if it[0] <= lo:
                break
            if it[col] == c and it[_CONF] >= floor:
                n += 1
        return n

    def total(self, now, win):
        lo = now - win
        n = 0
        for it in reversed(self.items):
            if it[0] <= lo:
                break
            n += 1
        return n

    def last_n_pos(self, now, n, max_win, col):
        lo = now - max_win
        pos = 0
        for it in reversed(self.items[-n:] if n > 0 else []):
            if it[0] <= lo:
                break
            if it[col] == 1:
                pos += 1
        return pos

    def last_n_all(self, now, n, max_win, col):
        """最近 n 个结果都在窗口内，而且 col 全为真。"""
        lo = now - max_win
        if len(self.items) < n:
            return False
        for it in self.items[-n:]:
            if it[0] <= lo or not it[col]:
                return False
        return True

    def last_n_any(self, now, n, max_win, col):
        """最近 n 个结果（窗口内的）里有 col 为真的。"""
        lo = now - max_win
        for it in reversed(self.items[-n:] if n > 0 else []):
            if it[0] <= lo:
                break
            if it[col]:
                return True
        return False


class CueArbiter(object):
    """fusion_lab.FusionArbiter（v1）+ 线索开关。decide() 多一个参数 cue；开关全关或 cue=None 时与 v1 逐 tick 相同。"""

    def __init__(self, params=None, cfg=None):
        p = dict(F.ARB_DEFAULTS)
        for k in F.ARB_DEFAULTS:
            if params and k in params:
                p[k] = params[k]
        self.p = p
        self.c = c = cfg or CFG0
        self.v_floor = F.f32(p['ARB_V_FLOOR'])
        self.a_floor = F.f32(p['ARB_A_FLOOR'])
        self.uninf = UNINF_SETS[c['uninf']] if (c['a1'] or c['a2']) else frozenset()
        self.uninf3 = UNINF_SETS[c['a3_set']] if c['a3'] else frozenset()
        self.confirm = CONFIRM if (c['p1'] or c['p2']) else frozenset()
        self.b_start = c['b'] in ('start', 'all')
        self.b_hold = c['b'] == 'all'
        self.video = _Ledger(p['ARB_MAX_AGE'], p['ARB_KEEP'])
        self.audio = _Ledger(p['ARB_MAX_AGE'], p['ARB_KEEP'])
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

    def _arec(self, a, ac, cue):
        c = self.c
        k = F._arb_cls(a)
        un = cue in self.uninf
        un3 = cue in self.uninf3
        dial = cue == DIAL
        ks = 0 if (k == 1 and ((dial and self.b_start) or (un3 and c['a3'] == 2))) else k
        kz = 0 if (k == 1 and un3) else ks
        kh = 0 if (k == 1 and dial and self.b_hold) else k
        neg = k == -1 and not (un and c['a2'])
        ksv = 0 if (k == 1 and dial) else ks
        kzv = 0 if (k == 1 and dial) else kz
        return (ac, k, ks, kz, kh, neg, un, k == 1 and cue in self.confirm, dial, ksv, kzv)

    def _has_neg(self, now):
        w = self.p['ARB_NEG_WIN']
        return self.video.count(now, w, _NEG, True) + self.audio.count(now, w, _NEG, True) > 0

    def _video_says_plot(self, now):
        """b=vid：视频最近 bvn 个结果（都在 M_MAX 内）全是置信度 ≥ bvc 的"剧情"。"""
        c, p = self.c, self.p
        its = self.video.items[-c['bvn']:]
        if len(its) < c['bvn']:
            return False
        lo = now - p['ARB_M_MAX']
        return all(it[0] > lo and it[_K0] == -1 and it[_CONF] >= F.f32(c['bvc']) for it in its)

    def _video_support(self, now, cons_pct):
        p = self.p
        if self.video.last_n_pos(now, p['ARB_MV_N'], p['ARB_M_MAX'], _K0) < p['ARB_MV_K']:
            return False
        n = self.video.total(now, p['ARB_CONS_WIN'])
        k = self.video.count(now, p['ARB_CONS_WIN'], _K0, 1)
        return n > 0 and k * 100 >= cons_pct * n

    def decide(self, now, v, vc, v_ts, a, ac, a_ts, motor_on, cue=None):
        p, c = self.p, self.c
        if self.prev_now is not None and now < self.prev_now:
            self.reset(now)
        self.prev_now = now
        kv = F._arb_cls(v)
        self.video.feed(v_ts, (vc, kv, kv, kv, kv, kv == -1, False, False, False, kv, kv), now)
        self.audio.feed(a_ts, self._arec(a, ac, cue), now)
        if not motor_on:
            self.last_support = -1
            KS, KZ = _KS, _KZ
            if c['b'] == 'vid' and self._video_says_plot(now):
                KS, KZ = _KSV, _KZV
            if self.video.count(now, p['ARB_C_WIN'], _K0, 1, self.v_floor) >= p['ARB_C_V'] and \
                    self.audio.count(now, p['ARB_C_WIN'], KS, 1, self.a_floor) >= p['ARB_C_A'] and \
                    (self._video_support(now, p['ARB_START_V_CONS'])
                     or self.audio.last_n_pos(now, p['ARB_MA_N'], p['ARB_M_MAX'], KS) >= p['ARB_START_MA_K']):
                self.reason = 'start_both'
                return 'do'
            n = self.audio.total(now, p['ARB_Z_WIN'])
            if n >= p['ARB_Z_N'] and self.audio.count(now, p['ARB_Z_WIN'], KZ, 1) == n:
                self.reason = 'start_audio'
                return 'do'
            if c['p1'] and self.audio.last_n_all(now, c['p_n'], p['ARB_M_MAX'], _CF):   # 不看视频：特写时也能启动
                self.reason = 'start_cue'
                return 'do'
            if c['a1'] and self.audio.last_n_all(now, c['a1_n'], p['ARB_M_MAX'], _UN) and \
                    self.video.count(now, p['ARB_C_WIN'], _K0, 1, self.v_floor) >= c['a1_cv'] and \
                    self._video_support(now, c['a1_cons']):
                self.reason = 'start_video'
                return 'do'
            if self._has_neg(now):
                self.reason = 'off_neg'
                return 'Noise'
            self.reason = 'off_wait'
            return ''
        if self.last_support < 0:
            self.last_support = now
        sa = self.audio.last_n_pos(now, p['ARB_MA_N'], p['ARB_M_MAX'], _KH) >= p['ARB_MA_K']
        sv = self._video_support(now, p['ARB_V_CONS'])
        if sv and c['bv'] and self.audio.last_n_all(now, c['bv'], p['ARB_M_MAX'], _DL):
            sv = False
        if sv or sa:
            self.last_support = now
            self.reason = 'hold_both' if (sv and sa) else ('hold_video' if sv else 'hold_audio')
            return 'do'
        waiting = now - self.last_support < p['ARB_UNCLEAR_MAX']
        if c['p2'] and now - self.last_support < c['p2_max'] and \
                self.audio.last_n_pos(now, p['ARB_MA_N'], p['ARB_M_MAX'], _KH) >= p['ARB_MA_K'] - 1 and \
                self.audio.last_n_any(now, p['ARB_MA_N'], p['ARB_M_MAX'], _CF):
            self.reason = 'hold_cue'           # 不更新 last_support：从最后一次正常支持算起最多续 p2_max
            return 'do'
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


class CuePolicy(object):
    """arbiter_lab.replay 用的策略：每个 tick 按 Prep.aext 算线索，交给 CueArbiter。"""

    def __init__(self, params, cfg, log=None):
        self.cfg = cfg
        self.arb = CueArbiter(params, cfg)
        self.log = log          # 可选 list：每个 tick 追加 (原因, 线索)

    def on_reset(self, ms, warm, seek):
        self.arb.reset(ms, warm)

    def step(self, p, i, ble):
        cue = cue_of(p.aext[i], self.cfg)
        act = self.arb.decide(p.wall[i], p.v[i], p.vc[i], p.vts[i], p.a[i], p.ac[i], p.ats[i],
                              F._is_sex(ble.cur), cue)
        if self.log is not None:
            self.log.append((self.arb.reason, cue))
        return act


def factory(cfg):
    return lambda params: CuePolicy(params, cfg)


# ======================================================================== 数据
def attach_features(videos, npz_dir, say=print):
    """把 <npz_dir>/<run 文件名>.yamnet.npz 的分数按结果时间戳接到每个 tick（Tick.ays）。返回 {vid: (接上, 总数)}。
    npz_dir 可以是逗号分隔的几个目录，按顺序找。实现在 fusion_lab.attach_cue_features（golden 也用它）。"""
    return F.attach_cue_features(videos, npz_dir, say)


def load(gt_dir=DEF_GT, runs_dir=DEF_RUNS, npz_dir=DEF_NPZ, seeks=True, quiet=False, only=None):
    d = A.Data()
    d.videos = []
    for rd in [x for x in runs_dir.split(',') if x]:      # 可以给几个 run 目录（逗号分隔），同一部片取第一个
        with contextlib.redirect_stdout(io.StringIO()):
            try:
                vs = F.load_videos(gt_dir, rd, quiet=True)
            except SystemExit:
                vs = []
        have = set(v.vid for v in d.videos)
        d.videos += [v for v in vs if v.vid not in have]
    d.videos.sort(key=lambda v: v.vid)
    if not d.videos:
        sys.exit('在 %s 下没找到能和真值配对的 run' % runs_dir)
    if only:
        keep = set(x.strip() for x in only.split(',') if x.strip())
        d.videos = [v for v in d.videos if v.vid in keep]
        if not d.videos:
            sys.exit('--only %s：一部片都没选中' % only)
    d.coverage = attach_features(d.videos, npz_dir, say=(lambda *a: None) if quiet else print)
    d.splices = dict((v.vid, F.make_splices(v)) for v in d.videos)
    d.prep = dict((v.vid, A.prepare(v)) for v in d.videos)
    d.sprep = dict((v.vid, [A.prepare(s) for s in d.splices[v.vid]]) for v in d.videos)
    d.seeks = dict((v.vid, A.make_seeks(v) if seeks else []) for v in d.videos)
    return d


def _fresh(t):
    """这一行带着一个没过期的音频结果（unclear 的动作串也是空的，不能用 t.a 判断）。"""
    return t.a_age is not None and 0 <= t.a_age <= F.MAX_AGE_MS


def _gt_bin(video, pos_ms):
    i = pos_ms // GRID_MS
    g = video.gti
    if 0 <= i < g.n and g.mask[i]:
        return g.gt[i]
    return None


# ======================================================================== stats / blame
def cmd_stats(a):
    d = load(a.gt_dir, a.runs_dir, a.npz, only=a.only, seeks=False)
    c = parse_cfg(a.cfg, a.preset)
    print('线索定义：%s' % cfg_str(c, full=True))
    print('（只取 2 秒窗整段落在同一真值段内、且在评测范围里的音频结果；"判转%" 是分类头在该线索下判"转"的比例）\n')
    order = (VOCAL, IMPACT, OTHER, DIAL, MUSIC, QUIET, BLIND)
    print('%-4s %-5s %5s  ' % ('片', '真值', '结果') + '  '.join('%-11s' % CUE_ZH[k] for k in order))
    print('%-4s %-5s %5s  ' % ('', '', '') + '  '.join('%-11s' % '占比/判转%' for _ in order))
    for v in d.videos:
        for lab in ('do', 'noise'):
            cnt, pos, seen = {}, {}, set()
            for t in v.ticks:
                if not _fresh(t):
                    continue
                ts = t.wall - t.a_age
                if ts in seen:
                    continue
                seen.add(ts)
                end = t.pos - t.a_age
                if any(_gt_bin(v, end - dt) != lab for dt in (0, 1000, 1999)):
                    continue
                k = cue_of(t.ays, c)
                cnt[k] = cnt.get(k, 0) + 1
                pos[k] = pos.get(k, 0) + (t.a == 'do')
            n = sum(cnt.values())
            if not n:
                continue
            cells = []
            for k in order:
                x = cnt.get(k, 0)
                cells.append('%4.1f%%/%3s' % (100.0 * x / n, ('%d' % round(100.0 * pos.get(k, 0) / x)) if x else ' — '))
            print('%-4s %-5s %5d  ' % (v.vid, '动作' if lab == 'do' else '剧情', n) + '  '.join('%-11s' % s for s in cells))
    return 0


def _replay_log(prep, cfg):
    log = []
    pol = CuePolicy(prep.video.params, cfg, log=log)
    trans, states = A.replay(prep, pol, want_states=True)
    return trans, states, log


def cmd_blame(a):
    d = load(a.gt_dir, a.runs_dir, a.npz, only=a.only, seeks=False)
    c = parse_cfg(a.cfg, a.preset)
    v1 = make_cfg()
    print('基线（v1）回放，按 500ms tick 统计（与 100ms 栅格的官方指标有细微差别）。线索定义：%s\n' % cfg_str(c, full=True))
    fn, fp = {}, {}
    fn_tot = fp_tot = 0.0
    stops = []
    for v in d.videos:
        prep = d.prep[v.vid]
        trans, states, log = _replay_log(prep, v1)
        for i, t in enumerate(v.ticks):
            g = _gt_bin(v, t.pos)
            if g is None:
                continue
            cue = cue_of(t.ays, c) if _fresh(t) else '(无音频结果)'
            key = (v.vid, cue, t.a or '-', t.v or '-')
            if g == 'do' and states[i] == 'noise':
                fn[key] = fn.get(key, 0.0) + 0.5
                fn_tot += 0.5
            elif g == 'noise' and states[i] == 'do':
                fp[key] = fp.get(key, 0.0) + 0.5
                fp_tot += 0.5
        for pos, s in trans:
            if s == 'noise' and _gt_bin(v, pos) == 'do':
                i = next(k for k, t in enumerate(v.ticks) if t.pos >= pos)
                lo = max(0, i - 8)
                cues = [cue_of(v.ticks[k].ays, c) for k in range(lo, i + 1) if _fresh(v.ticks[k])]
                why = [log[k][0] for k in range(lo, i + 1)]
                stops.append((v.vid, pos, why[-1], cues))

    def table(title, dd, tot):
        print('══ %s：共 %.0f 秒 ══' % (title, tot))
        by_cue = {}
        for (vid, cue, aa, vv), s in dd.items():
            by_cue.setdefault(cue, []).append((vid, aa, vv, s))
        rows = sorted(by_cue.items(), key=lambda kv: -sum(x[3] for x in kv[1]))
        for cue, xs in rows:
            s = sum(x[3] for x in xs)
            name = CUE_ZH.get(cue, cue)
            by_a = {}
            by_v = {}
            by_vid = {}
            for vid, aa, vv, sec in xs:
                by_a[aa] = by_a.get(aa, 0) + sec
                by_v[vv] = by_v.get(vv, 0) + sec
                by_vid[vid] = by_vid.get(vid, 0) + sec
            fa = ' '.join('%s %.0f' % (k if k != '-' else 'unclear', x) for k, x in sorted(by_a.items(), key=lambda kv: -kv[1]))
            fv = ' '.join('%s %.0f' % (k if k != '-' else 'unclear', x) for k, x in sorted(by_v.items(), key=lambda kv: -kv[1]))
            fvid = ' '.join('%s %.0f' % kv for kv in sorted(by_vid.items(), key=lambda kv: -kv[1]))
            print('  %-8s %5.0fs %4.0f%% | 音频判 %-26s | 视频判 %-26s | %s' % (name, s, 100 * s / tot if tot else 0, fa, fv, fvid))
        print()

    table('漏转（真值动作、马达没转）', fn, fn_tot)
    table('误转（真值剧情、马达在转）', fp, fp_tot)
    print('══ 动作中误停：%d 次 ══' % len(stops))
    for vid, pos, why, cues in stops:
        print('  %s %6.1fs  原因 %-12s 之前 4 秒的音频线索 %s' % (vid, pos / 1000.0, why,
                                                         ' '.join(CUE_ZH.get(x, x) for x in cues)))
    print()
    # 每条规则最多能救多少（按 tick 粗算，不考虑连锁效应）
    un = UNINF_SETS[c['uninf']]
    a1 = sum(s for (vid, cue, aa, vv), s in fn.items() if cue in un and vv == 'do')
    b_fp = sum(s for (vid, cue, aa, vv), s in fp.items() if cue == DIAL and aa == 'do')
    a3 = sum(s for (vid, cue, aa, vv), s in fp.items() if cue in un and aa == 'do')
    print('上限粗估（只看这一刻的线索和判定，不考虑启动 / 维持的连锁）：')
    print('  A1 视频单独启动：漏转里"听不出 + 视频判转" %.0f 秒' % a1)
    print('  B  对白否决：    误转里"对白 + 分类头判转" %.0f 秒' % b_fp)
    print('  A3 听不出不兜底：误转里"听不出 + 分类头判转" %.0f 秒' % a3)
    return 0


# ======================================================================== 压力场景（带线索）
SYN = {   # 合成场景用的分组分数（GROUPS 顺序：voc_core, voc_ext, speech, speech_ext, music, silence, impact）
    VOCAL: (0.6, 0.8, 0.05, 0.06, 0.0, 0.0, 0.0),
    DIAL: (0.0, 0.0, 0.95, 0.97, 0.0, 0.0, 0.0),
    MUSIC: (0.0, 0.0, 0.02, 0.02, 0.8, 0.0, 0.0),
    QUIET: (0.0, 0.0, 0.02, 0.02, 0.0, 0.9, 0.0),
    OTHER: (0.15, 0.2, 0.3, 0.35, 0.1, 0.0, 0.0),
    IMPACT: (0.02, 0.03, 0.05, 0.05, 0.0, 0.0, 0.5),
}


def _aP(c=0.95, cue=VOCAL):
    return ('do', c, SYN[cue])


def _aN(c=0.8, cue=OTHER):
    return ('Noise', c, SYN[cue])


def _aU(cue=OTHER):
    return ('', 0.0, SYN[cue])


def _vP(c=0.95):
    return ('do', c)


def _vN(c=0.9):
    return ('Noise', c)


_vU = ('', 0.0)
SW = A.SW
START_WITHIN_S = 10


def _alt(t, period, x, y):
    return x if (t // period) % 2 == 0 else y


# (名称, 用途, 视频, 音频, 时长, 期望)
# 用途：safety = 每个候选配置都必须通过（新规则不能把它做坏）；goal = 新规则想救的场景，只报告
# 期望：nostart = 全程不该转；start = 应在 START_WITHIN_S 秒内启动；hold = SW 之后不该停；
#       stop = SW 之后 20 秒内停，而且停了不能再自己启动
CUE_SCENARIOS = [
    # —— A1 的风险：剧情里有背景音乐 / 很安静，视频又误判
    ('剧情+音乐：视频 +/- 交替，音频判不转', 'safety',
     lambda t: _alt(t, 625, _vP(0.9), _vN()), lambda t: _aN(0.8, MUSIC), 60000, 'nostart'),
    ('剧情+音乐：视频 3 个里 2 个 +', 'safety',
     lambda t: _vN() if (t // 625) % 3 == 0 else _vP(0.9), lambda t: _aN(0.8, MUSIC), 60000, 'nostart'),
    ('剧情+音乐：视频 4 个里 1 个 +', 'safety',
     lambda t: _vP(0.9) if (t // 625) % 4 == 0 else _vN(), lambda t: _aN(0.8, MUSIC), 60000, 'nostart'),
    ('剧情+安静：视频 +/- 交替，音频 unclear', 'safety',
     lambda t: _alt(t, 625, _vP(0.9), _vN()), lambda t: _aU(QUIET), 60000, 'nostart'),
    ('剧情+安静：视频 3 个里 2 个 +，音频判不转', 'safety',
     lambda t: _vN() if (t // 625) % 3 == 0 else _vP(0.9), lambda t: _aN(0.7, QUIET), 60000, 'nostart'),
    # —— B / B2 的风险：动作里说话
    ('动作中每 4 个音频有 1 个对白（仍判转），视频特写判剧情 40 秒', 'safety',
     lambda t: _vP() if t < SW else _vN(),
     lambda t: _aP(0.95, DIAL) if t >= SW and (t // 1000) % 4 == 1 else _aP(), SW + 40000, 'hold'),
    ('动作中每 5 秒连着 2 个对白（判不转），视频判转', 'safety',
     lambda t: _vP(),
     lambda t: _aN(0.9, DIAL) if t >= SW and (t // 1000) % 5 in (1, 2) else _aP(), SW + 40000, 'hold'),
    # —— A2 的风险：动作里夹音乐
    ('动作：音频人声 + 每 5 秒夹 1 秒音乐 -，视频特写判剧情 40 秒', 'safety',
     lambda t: _vP() if t < SW else _vN(),
     lambda t: _aN(0.7, MUSIC) if t >= SW and (t // 1000) % 5 == 2 else _aP(), SW + 40000, 'hold'),
    # —— P1 / P2 的风险：剧情里偶尔有"转+人声"
    ('剧情：每 15 秒出现一对"转+人声"，其余判不转，视频 -', 'safety',
     lambda t: _vN(), lambda t: _aP(0.95, VOCAL) if (t // 1000) % 15 in (5, 6) else _aN(), 60000, 'nostart'),
    ('剧情（已在转）："转+人声"与"不转"交替，视频 -', 'safety',
     lambda t: _vP() if t < SW else _vN(),
     lambda t: _aP() if t < SW else _alt(t, 1000, _aP(0.95, VOCAL), _aN()), SW + 45000, 'stop'),
    ('剧情（已在转）：一直有拍打 / 敲击声，分类头判不转，视频 -', 'safety',
     lambda t: _vP() if t < SW else _vN(), lambda t: _aP() if t < SW else _aN(0.8, IMPACT), SW + 45000, 'stop'),
    # —— 目标场景
    ('动作+音乐：音频判不转，视频强 +', 'goal', lambda t: _vP(), lambda t: _aN(0.8, MUSIC), 60000, 'start'),
    ('动作+安静：音频 unclear，视频强 +', 'goal', lambda t: _vP(), lambda t: _aU(QUIET), 60000, 'start'),
    ('动作+音乐（已在转）：视频特写 5 秒判 unclear', 'goal',
     lambda t: _vU if SW <= t < SW + 5000 else _vP(), lambda t: _aP() if t < SW else _aN(0.8, MUSIC), SW + 30000, 'hold'),
    ('剧情对白：分类头判转，视频 -', 'goal', lambda t: _vN(), lambda t: _aP(0.95, DIAL), 60000, 'nostart'),
    ('剧情对白（已在转）：分类头判转，视频 -', 'goal',
     lambda t: _vP() if t < SW else _vN(), lambda t: _aP() if t < SW else _aP(0.95, DIAL), SW + 45000, 'stop'),
    ('剧情对白（已在转）：分类头判不转，视频误判转', 'goal',
     lambda t: _vP(), lambda t: _aP() if t < SW else _aN(0.9, DIAL), SW + 45000, 'stop'),
    ('剧情安静：分类头判转，视频 -', 'goal', lambda t: _vN(), lambda t: _aP(0.9, QUIET), 60000, 'nostart'),
    ('剧情安静：分类头判转，视频误判转', 'goal', lambda t: _vP(0.9), lambda t: _aP(0.9, QUIET), 60000, 'nostart'),
    ('动作特写（已在转）：音频"转+拍打"每 8 个夹 1 个不转 1 个 unclear，视频 -', 'goal',
     lambda t: _vP() if t < SW else _vN(),
     lambda t: _aP() if t < SW else (_aN(0.8, IMPACT) if (t // 1000) % 8 == 3 else
                                     (_aU(IMPACT) if (t // 1000) % 8 == 4 else _aP(0.95, IMPACT))),
     SW + 40000, 'hold'),
    ('动作特写（没在转）：音频"转+人声"每 4 个夹 1 个不转，视频 -', 'goal',
     lambda t: _vN(), lambda t: _aN(0.8, OTHER) if (t // 1000) % 4 == 0 else _aP(0.95, VOCAL), 60000, 'start'),
]

CUE_FLAP = [
    ('视频 5 个里 4 个 +，音频音乐 -', lambda t: _vN() if (t // 625) % 5 == 0 else _vP(0.9),
     lambda t: _aN(0.8, MUSIC)),
    ('视频 3 个里 2 个 +，音频人声 + / 音乐 - 每秒交替', lambda t: _vN() if (t // 625) % 3 == 0 else _vP(0.9),
     lambda t: _alt(t, 1000, _aP(), _aN(0.8, MUSIC))),
    ('视频 +，音频音乐 - 与人声 + 每 3 秒交替', lambda t: _vP(0.9),
     lambda t: _alt(t, 3000, _aN(0.8, MUSIC), _aP())),
    ('视频 -，音频"转+人声" 3 个 / 不转 2 个循环', lambda t: _vN(),
     lambda t: _aP(0.95, VOCAL) if (t // 1000) % 5 < 3 else _aN()),
]
FLAP_MAX_PER_MIN = 2.0

# 原有压力场景（arbiter_lab.SCENARIOS / FLAP_SCENARIOS）给音频结果补上线索再跑一遍，期望与 v1 相同：
# "不转"/unclear 分别当作 其它 / 音乐 / 安静；"转"在 hold（动作）场景里当作人声，在 stop（剧情）场景里当作其它
# （剧情里分类头误判的"转"很少带人声，实测 0~5%；带人声的情况另有专门的场景）
_OLD_CUE_VARIANTS = (OTHER, MUSIC, QUIET)


def _with_cue(afun, neg_cue, pos_cue=VOCAL):
    def f(t):
        r = afun(t)
        if r is None:
            return None
        return (r[0], r[1], SYN[pos_cue if r[0] == 'do' else neg_cue])
    return f


def _run_one(make, params, vf, af, dur, want):
    ticks = A._synth_ticks(vf, af, dur)
    v = F.Video('syn', [(0, dur, 'sex')], dur, ticks, params, synthetic=True)
    tr, st = A.replay(A.prepare(v), make(params), want_states=True)
    if want in ('nostart', 'start'):
        on = sum(1 for x in st if x == 'do') * 0.5
        first = next((pos / 1000.0 for pos, x in tr if x == 'do'), None)
        ok = first is None if want == 'nostart' else (first is not None and first <= START_WITHIN_S)
        return ok, on, first, tr
    on = sum(1 for t, x in zip(ticks, st) if t.pos >= SW and x == 'do') * 0.5
    off = next(((pos - SW) / 1000.0 for pos, x in tr if x == 'noise' and pos >= SW), None)
    if want == 'stop':      # 比 arbiter_lab 多一条：停了之后不能再自己启动（新规则可以只靠视频启动）
        again = off is not None and any(x == 'do' and pos > SW + off * 1000 for pos, x in tr)
        ok = off is not None and off <= A.STOP_WITHIN_S and not again
    else:
        ok = off is None
    return ok, on, off, tr


def run_stress(make, params):
    """→ dict：safety_ok（全部安全约束都过）、rows（逐条结果）"""
    rows = []
    for name, use, vf, af, dur, want in CUE_SCENARIOS:
        ok, on, at, _ = _run_one(make, params, vf, af, dur, want)
        rows.append((use, name, want, ok, on, at))
    for name, vf, af in CUE_FLAP:
        ticks = A._synth_ticks(vf, af, 120000)
        v = F.Video('syn', [(0, 120000, 'sex')], 120000, ticks, params, synthetic=True)
        tr, _ = A.replay(A.prepare(v), make(params))
        rate = len(tr) / 2.0
        rows.append(('safety', name, '不乱开关', rate <= FLAP_MAX_PER_MIN, rate, None))
    for neg_cue in _OLD_CUE_VARIANTS:
        for name, vf, af, dur, want in A.SCENARIOS:
            pos_cue = OTHER if want == 'stop' else VOCAL
            ok, on, at, _ = _run_one(make, params, vf, _with_cue(af, neg_cue, pos_cue), dur, want)
            rows.append(('safety', '[原有·不转=%s] %s' % (CUE_ZH[neg_cue], name), want, ok, on, at))
        for name, vf, af in A.FLAP_SCENARIOS:
            ticks = A._synth_ticks(vf, _with_cue(af, neg_cue), 120000)
            v = F.Video('syn', [(0, 120000, 'sex')], 120000, ticks, params, synthetic=True)
            tr, _ = A.replay(A.prepare(v), make(params))
            rate = len(tr) / 2.0
            rows.append(('safety', '[原有·不转=%s] %s' % (CUE_ZH[neg_cue], name), '不乱开关',
                         rate <= FLAP_MAX_PER_MIN, rate, None))
    return dict(safety_ok=all(ok for use, _, _, ok, _, _ in rows if use == 'safety'),
                goals=sum(1 for use, _, _, ok, _, _ in rows if use == 'goal' and ok), rows=rows)


def _fmt_row(want, ok, on, at):
    mark = '✓' if ok else '✗'
    if want == '不乱开关':
        return '%s 开关 %.1f 次/分' % (mark, on)
    if want in ('nostart', 'start'):
        return '%s %s，共转 %.1fs' % (mark, '没启动' if at is None else '%.1fs 启动' % at, on)
    return '%s 转 %4.1fs，%s' % (mark, on, '没停' if at is None else '%.1fs 停' % at)


def cmd_stress(a):
    d = load(a.gt_dir, a.runs_dir, a.npz, only=a.only, seeks=False, quiet=True)
    params = d.videos[0].params
    c = parse_cfg(a.cfg, a.preset)
    r0 = run_stress(factory(make_cfg()), params)
    r1 = run_stress(factory(c), params)
    print('压力场景：v1 对照 %s' % cfg_str(c))
    print('（nostart = 全程不该转；start = %d 秒内启动；hold = 第 30 秒之后不该停；stop = 第 30 秒之后 %d 秒内停）\n'
          % (START_WITHIN_S, A.STOP_WITHIN_S))
    for use in ('safety', 'goal'):
        print('══ %s ══' % ('安全约束（每个候选配置都必须通过）' if use == 'safety' else '目标场景（新规则想救的，只报告）'))
        for x, y in zip(r0['rows'], r1['rows']):
            if x[0] != use:
                continue
            if use == 'safety' and a.brief and x[3] and y[3]:
                continue
            print('  %-56s %-8s v1 %-26s 新 %s' % (x[1], x[2], _fmt_row(x[2], x[3], x[4], x[5]),
                                                  _fmt_row(y[2], y[3], y[4], y[5])))
        print()
    n_safe = sum(1 for x in r1['rows'] if x[0] == 'safety')
    print('安全约束：v1 %s，新配置 %s（共 %d 条）' % ('全过' if r0['safety_ok'] else '有不过的',
                                            '全过' if r1['safety_ok'] else '有不过的', n_safe))
    return 0


# ======================================================================== 评测
def per_video(data, cfg, cost, seeks=True):
    """每部片单独评测（含它的拼接样本、seek 片段），供留一视频选参。"""
    mk = factory(cfg)
    out = {}
    for v in data.videos:
        rep = A.evaluate(data, mk, cost, seeks=seeks, only={v.vid})
        P = rep['pooled']
        out[v.vid] = dict(
            cost=rep['cost'], cm=dict(P.cm), fs=P.fs, lat=dict((k, list(x)) for k, x in P.lat.items()),
            miss=dict(P.miss), videos=rep['videos'], splice=[(k, vid, sc.cost, sec) for k, vid, sc, sec in rep['splice']],
            seek_do=rep['seek_do'], seek_plot=rep['seek_plot'], lat_stop=rep['lat_stop'])
    return out


def combine(picks):
    """picks: {vid: 结果 dict}（每部片取自己那一折选中的配置）→ arbiter_lab.evaluate 同格式的 rep。"""
    rep = {'videos': {}, 'pooled': F.Score(), 'splice': [], 'seek_do': [], 'seek_plot': [], 'lat_stop': [], 'cost': 0.0}
    for vid, r in sorted(picks.items()):
        s = F.Score(dict(r['cm']), r['fs'], dict((k, list(x)) for k, x in r['lat'].items()), dict(r['miss']), 0.0)
        rep['pooled'].add(s)
        rep['videos'].update(r['videos'])
        for k, v2, c, sec in r['splice']:
            sc = F.Score()
            sc.cost = c
            rep['splice'].append((k, v2, sc, sec))
        rep['seek_do'] += r['seek_do']
        rep['seek_plot'] += r['seek_plot']
        rep['lat_stop'] += r['lat_stop']
        rep['cost'] += r['cost']
    return rep


def fn_seconds(r):
    return r['cm'][('do', 'noise')] * GRID_MS / 1000.0


def fp_seconds(r):
    return r['cm'][('noise', 'do')] * GRID_MS / 1000.0


def cmd_eval(a):
    d = load(a.gt_dir, a.runs_dir, a.npz, only=a.only)
    c = parse_cfg(a.cfg, a.preset)
    cost = F.Cost.parse(a.cost)
    # 基线一律是 v1（开关全关，与 v1 逐 tick 一致）。不用 A.factory('arbiter')：它按 run 的 ARB= 回放，
    # 2026-10-11 起的 run 是 v2，那样"基线"就成了 a1bvid 本身
    base = A.evaluate(d, factory(make_cfg()), cost)
    new = A.evaluate(d, factory(c), cost)
    A.print_report(base, '基线：融合仲裁 v1（不用线索）')
    A.print_report(new, '新：%s' % cfg_str(c))
    pv0 = per_video(d, make_cfg(), cost)
    pv1 = per_video(d, c, cost)
    print('逐片代价（片子本身 + 它的拼接样本；seek 片段不计入代价）')
    for v in d.videos:
        x, y = pv0[v.vid], pv1[v.vid]
        print('  %s  %7.1f → %7.1f  (%+.1f)   漏转 %5.1fs → %5.1fs   误转 %5.1fs → %5.1fs' % (
            v.vid, x['cost'], y['cost'], y['cost'] - x['cost'], fn_seconds(x), fn_seconds(y), fp_seconds(x), fp_seconds(y)))
    r = run_stress(factory(c), d.videos[0].params)
    print('\n压力场景安全约束：%s；目标场景通过 %d 个' % ('全过' if r['safety_ok'] else '有不过的（用 stress 看）', r['goals']))
    if a.slow:
        for k, which, label in ((2, 'v', '视频结果间隔 ×2'), (3, 'v', '视频结果间隔 ×3'),
                                (2, 'a', '音频结果间隔 ×2'), (3, 'a', '音频结果间隔 ×3')):
            sd = A.slow_data(d, k, which)
            for name, mk in (('v1', factory(make_cfg())), ('新', factory(c))):
                s = A.summary(A.evaluate(sd, mk, cost, seeks=False))
                print('  慢手机 %-14s %-3s 准确率 %s 误转 %s 漏转 %s 误停 %d' % (
                    label, name, A._pct(s['acc']), A._pct(s['fp']), A._pct(s['fn']), s['stops']))
    return 0


# ======================================================================== 扫参（样本内，只看方向）
def cmd_scan(a):
    d = load(a.gt_dir, a.runs_dir, a.npz, only=a.only, seeks=False, quiet=True)
    cost = F.Cost.parse(a.cost)
    vids = [v.vid for v in d.videos]

    def row(c):
        pv = per_video(d, c, cost, seeks=False)
        rep = combine(pv)
        return A.summary(rep), rep['cost'], dict((vid, r['cost']) for vid, r in pv.items())

    base_s, base_c, base_pv = row(make_cfg())
    params = d.videos[0].params
    print('样本内扫描（只看方向，不拿来选参）。代价含拼接样本，不含 seek。基线代价 %.0f，漏转 %s，误转 %s，误停 %d\n'
          % (base_c, A._pct(base_s['fn']).strip(), A._pct(base_s['fp']).strip(), base_s['stops']))
    head = '  %-62s %6s %6s %6s %6s %3s %2s  ' % ('配置', '代价', '准确率', '误转', '漏转', '误停', '安全') + \
        ' '.join('%6s' % x for x in vids)
    for title, fam in scan_families():
        print('══ %s（%d 组，按代价排前 %d）══' % (title, len(fam), a.top))
        print(head)
        res = []
        for c in fam:
            s, cc, pv = row(c)
            ok = run_stress(factory(c), params)['safety_ok']
            res.append((cc, c, s, pv, ok))
        res.sort(key=lambda x: x[0])
        for cc, c, s, pv, ok in res[:a.top]:
            print('  %-62s %6.0f %6s %6s %6s %3d %2s  ' % (cfg_str(c)[:62], cc, A._pct(s['acc']), A._pct(s['fp']),
                                                         A._pct(s['fn']), s['stops'], '✓' if ok else '✗')
                  + ' '.join('%+6.0f' % (pv[x] - base_pv[x]) for x in vids))
        print()
    return 0


def scan_families():
    fams = []
    fam = []
    for voc in ('ext', 'core'):
        for mode in ('start', 'all'):
            for dv in (0.01, 0.02, 0.05):
                for ds in (0.5, 0.6, 0.7, 0.8, 0.9):
                    fam.append(make_cfg(voc=voc, b=mode, d_voc=dv, d_sp=ds))
    fams.append(('B 对白否决', fam))
    fam = []
    for b in ('off', 'all'):
        for bv in (2, 3):
            for dv in (0.02, 0.05):
                for ds in (0.6, 0.7, 0.8):
                    fam.append(make_cfg(b=b, bv=bv, d_voc=dv, d_sp=ds))
    fams.append(('B2 对白时视频不能单独维持（可叠加 B=all）', fam))
    fam = []
    for a3 in (1, 2):
        for st in ('m', 'mq', 'mqb'):
            for bv in (0.05, 0.1):
                for imp in (0.0, 0.1):
                    fam.append(make_cfg(a3=a3, a3_set=st, b_voc=bv, b_imp=imp))
    fams.append(('A3 听不出时的转不算启动证据', fam))
    fam = []
    for un in ('m', 'mq', 'mqb'):
        for bv in (0.05, 0.1):
            for n in (2, 3):
                for cons in (80, 90):
                    fam.append(make_cfg(a1=1, uninf=un, b_voc=bv, a1_n=n, a1_cons=cons))
    fams.append(('A1 听不出时视频单独启动', fam))
    fam = []
    for un in ('m', 'mq', 'mqb'):
        for bv in (0.05, 0.1):
            fam.append(make_cfg(a2=1, uninf=un, b_voc=bv))
    fams.append(('A2 听不出时的不转不算结束证据', fam))
    fam = []
    for p1, p2 in ((1, 0), (0, 1), (1, 1)):
        for pv in (0.2, 0.3, 0.5):
            for pi in (0.1, 0.2, 0.3):
                for n in ((2, 3) if p1 else (3,)):
                    fam.append(make_cfg(p1=p1, p2=p2, p_voc=pv, p_imp=pi, p_n=n))
    fams.append(('P 人声 / 拍打确认的转', fam))
    return fams


# ======================================================================== 留一视频
def lovo_grid():
    """候选配置（全组合）。grid[0] 是 v1；其余按"从简单到复杂"排，同代价时先出现的赢。"""
    b_opts = [dict(b='off')]
    b_opts += [dict(b='start', d_voc=dv, d_sp=0.7) for dv in (0.01, 0.02)]
    b_opts += [dict(b='all', d_voc=dv, d_sp=ds, bv=bv) for bv in (0, 2, 3) for dv in (0.01, 0.02) for ds in (0.6, 0.7, 0.8)]
    a1_opts = [dict()] + [dict(a1=1, uninf=u, a1_cons=cons) for u in ('m', 'mq', 'mqb') for cons in (90, 80)]
    a3_opts = [dict(), dict(a3=1, a3_set='mqb'), dict(a3=2, a3_set='mq'), dict(a3=2, a3_set='mqb')]
    p_opts = [dict(), dict(p2=1, p2_max=3000), dict(p2=1, p2_max=6000), dict(p1=1, p2=1, p2_max=3000),
              dict(p1=1, p2=1, p2_max=6000), dict(p1=1, p2=1, p2_max=6000, p_voc=0.5, p_imp=0.3)]
    out = [make_cfg()]
    seen = set()
    for po in p_opts:
        for a3 in a3_opts:
            for a1 in a1_opts:
                for bo in b_opts:
                    for bvoc in ((0.05, 0.1) if (a1 or a3) else (0.05,)):
                        for voc in ('ext', 'core'):
                            kw = dict(bo)
                            kw.update(a1)
                            kw.update(a3)
                            kw.update(po)
                            if kw == dict(b='off'):
                                continue
                            c = make_cfg(voc=voc, b_voc=bvoc, **kw)
                            key = tuple(sorted(c.items()))
                            if key not in seen:
                                seen.add(key)
                                out.append(c)
    return out


_G = {}


def _lovo_init(gt_dir, runs_dir, npz, cost):
    _G['d'] = load(gt_dir, runs_dir, npz, seeks=False, quiet=True)
    _G['cost'] = cost


def _lovo_task(c):
    """→ None（没过压力场景安全约束）或 {vid: (代价, 漏转秒, 误转秒, 误停次数)}，不含 seek。"""
    d, cost = _G['d'], _G['cost']
    mk = factory(c)
    if not run_stress(mk, d.videos[0].params)['safety_ok']:
        return None
    out = {}
    for v in d.videos:
        rep = A.evaluate(d, mk, cost, seeks=False, only={v.vid})
        P = rep['pooled']
        out[v.vid] = (rep['cost'], P.cm[('do', 'noise')] * GRID_MS / 1000.0,
                      P.cm[('noise', 'do')] * GRID_MS / 1000.0, P.fs)
    return out


def lovo_eval(a, grid, cost, log=print):
    t0 = time.time()
    res = [None] * len(grid)
    step = max(500, len(grid) // 10)
    if a.jobs > 1:
        import multiprocessing as mp
        with mp.Pool(a.jobs, initializer=_lovo_init, initargs=(a.gt_dir, a.runs_dir, a.npz, cost)) as pool:
            for i, r in enumerate(pool.imap(_lovo_task, grid, chunksize=8)):
                res[i] = r
                if (i + 1) % step == 0:
                    log('  … %d/%d（%.0f 秒）' % (i + 1, len(grid), time.time() - t0))
    else:
        _lovo_init(a.gt_dir, a.runs_dir, a.npz, cost)
        for i, c in enumerate(grid):
            res[i] = _lovo_task(c)
            if (i + 1) % step == 0:
                log('  … %d/%d（%.0f 秒）' % (i + 1, len(grid), time.time() - t0))
    return res


class SeekCache(object):
    """按需算某个配置在每部片上的 seek 结果（seek 片段多，只给每折排在前面的候选算）。"""

    def __init__(self, d, grid, cost):
        self.d, self.grid, self.cost, self.cache = d, grid, cost, {}

    def get(self, i):
        if i not in self.cache:
            pv = per_video(self.d, self.grid[i], self.cost, seeks=True)
            self.cache[i] = pv
        return self.cache[i]


def _seek_stats(pvs, vids):
    xs = [dt for v in vids for (_, _, _, dt, _, _) in pvs[v]['seek_do']]
    never = sum(1 for x in xs if x is None)
    done = [x for x in xs if x is not None]
    return never, (A._q(done, 0.9) if done else float('nan'))


def select_fold(grid, res, train, seeks, fn_w=1.0, max_check=400):
    """一折：在训练片上选代价最低的配置。约束：
      · 压力场景安全约束全过（res[i] 不为 None）
      · 训练片误停不比基线多 1 次以上，训练片漏转秒数不比基线多（漏转优先）
      · 训练片 seek 进动作段：没启动的次数不比基线多，p90 不超过 max(10 秒, 基线)"""
    base = res[0]
    b_stops = sum(base[v][3] for v in train)
    b_fn = sum(base[v][1] for v in train)
    b_never, b_p90 = _seek_stats(seeks.get(0), train)
    cands = []
    for i, r in enumerate(res):
        if r is None:
            continue
        if sum(r[v][3] for v in train) > b_stops + 1:
            continue
        fn = sum(r[v][1] for v in train)
        if fn > b_fn + 1e-9:
            continue
        cands.append((sum(r[v][0] for v in train) + (fn_w - 1.0) * fn, i))
    cands.sort()
    for cst, i in cands[:max_check]:
        never, p90 = _seek_stats(seeks.get(i), train)
        if never <= b_never and p90 <= max(10.0, b_p90) + 1e-9:
            return i, cst
    return 0, None


def cmd_lovo(a):
    cost = F.Cost.parse(a.cost)
    d = load(a.gt_dir, a.runs_dir, a.npz, only=a.only, quiet=True)
    vids = [v.vid for v in d.videos]
    print('YAMNet 分数接上的音频结果：%s' % ', '.join('%s %d/%d' % (v, d.coverage[v][0], d.coverage[v][1]) for v in vids))
    # 基线留一：arbiter_lab lovo（NARROW 网格选融合仲裁参数），逐片代价单独算出来作对照
    brep, bpicks = A.lovo(d, A.NARROW, cost, log=lambda *x: None)
    base_pv = dict((vid, A.evaluate(d, A.factory('arbiter', bpicks[vid]), cost, seeks=True, only={vid})['cost'])
                   for vid in vids)
    base_sum = A.summary(brep)
    print('基线（v1，留一视频选参，同 arbiter_lab lovo）：代价 %.0f，准确率 %s，误转 %s，漏转 %s，误停 %d' % (
        brep['cost'], A._pct(base_sum['acc']).strip(), A._pct(base_sum['fp']).strip(),
        A._pct(base_sum['fn']).strip(), base_sum['stops']))
    grid = lovo_grid()
    res = None
    if a.cache and os.path.exists(a.cache):
        import pickle
        with open(a.cache, 'rb') as fh:
            saved = pickle.load(fh)
        if saved.get('grid') == grid and saved.get('cost') == str(cost) and saved.get('vids') == vids:
            res = saved['res']
            print('候选配置 %d 组：读缓存 %s' % (len(grid), a.cache))
    if res is None:
        print('候选配置 %d 组，%d 进程…' % (len(grid), a.jobs))
        res = lovo_eval(a, grid, cost)
        if a.cache:
            import pickle
            with open(a.cache, 'wb') as fh:
                pickle.dump(dict(grid=grid, cost=str(cost), vids=vids, res=res), fh)
    print('  %d 组通过压力场景安全约束' % sum(1 for r in res if r is not None))
    if res[0] is None:
        sys.exit('v1 自己没过压力场景安全约束，场景写错了')
    if a.where:
        conds = []
        for part in a.where.replace(',', ' ').split():
            neg = '!=' in part
            k, v = part.split('!=' if neg else '=', 1)
            if k not in CFG0:
                sys.exit('--where 里的 %r 不是配置项' % k)
            conds.append((k, neg, type(CFG0[k])(v) if not isinstance(CFG0[k], str) else v))
        keep = lambda c: all((c[k] != v) if neg else (c[k] == v) for k, neg, v in conds)  # noqa: E731
        res = [r if (i == 0 or keep(grid[i])) else None for i, r in enumerate(res)]
        print('  --where %s：保留 %d 组' % (a.where, sum(1 for r in res if r is not None)))
    seeks = SeekCache(d, grid, cost)
    chosen = {}
    for vid in vids:
        train = [v for v in vids if v != vid]
        i, cst = select_fold(grid, res, train, seeks, fn_w=a.fn_w)
        chosen[vid] = i
        print('  留出 %s：%s' % (vid, cfg_str(grid[i])))
    cnt = {}
    for i in chosen.values():
        cnt[i] = cnt.get(i, 0) + 1
    print('\n  各折选中次数：' + '；'.join('%d 折 · %s' % (n, cfg_str(grid[i]))
                                   for i, n in sorted(cnt.items(), key=lambda kv: -kv[1])))
    picks = dict((vid, seeks.get(chosen[vid])[vid]) for vid in vids)
    rep = combine(picks)
    print()
    A.print_report(rep, '新：留一视频选组 / 阈值 / 规则%s' % ('（选参时漏转权重 %g）' % a.fn_w if a.fn_w != 1.0 else ''))
    new_pv = dict((vid, picks[vid]['cost']) for vid in vids)
    print('逐片代价（片子本身 + 它的拼接样本；seek 片段不计入代价）：基线留一 → 新留一')
    for vid in vids:
        print('  %s  %7.1f → %7.1f  (%+.1f)   漏转 %5.1fs   误转 %5.1fs   误停 %d' % (
            vid, base_pv[vid], new_pv[vid], new_pv[vid] - base_pv[vid], fn_seconds(picks[vid]),
            fp_seconds(picks[vid]), picks[vid]['fs']))
    ok, lines = acceptance(new_pv, base_pv, brep['cost'], rep['cost'], A.summary(rep), base_sum)
    print('\n验收线：')
    for ln in lines:
        print('  ' + ln)
    print('\n  结论：%s' % ('全部通过' if ok else '没过验收线'))
    top = max(cnt.items(), key=lambda kv: (kv[1], -kv[0]))[0]
    allpv = res[top]
    print('\n  最常被选中的配置在全部 %d 部上（样本内，不含 seek）：代价 %.0f（v1 %.0f）；--cfg %s' % (
        len(vids), sum(allpv[v][0] for v in vids), sum(res[0][v][0] for v in vids), cfg_args(grid[top])))
    if a.save:
        with io.open(a.save, 'w', encoding='utf-8') as fh:
            for vid in vids:
                fh.write('%s\t%s\n' % (vid, cfg_args(grid[chosen[vid]])))
        print('  各折选中的配置写到 %s' % a.save)
    return 0


def cfg_args(c):
    """和默认值不同的项，写成 --cfg 能吃的形式。"""
    return ' '.join('%s=%s' % (k, c[k]) for k in sorted(c) if c[k] != CFG0[k]) or '(v1)'


def acceptance(new_pv, base_pv, base_total, new_total, new_sum, base_sum):
    """验收线（2026-10-10 与用户确认）。"""
    lines, ok = [], True
    th = 0.95 * base_total
    c1 = new_total <= th
    ok &= c1
    lines.append('%s 留一代价 %.0f ≤ 基线留一 %.0f × 95%% = %.0f' % ('✓' if c1 else '✗', new_total, base_total, th))
    worse = []
    for vid in sorted(base_pv):
        b, n = base_pv[vid], new_pv[vid]
        if n - b > max(0.05 * b, 3.0):
            worse.append((vid, b, n))
    c2 = len(worse) <= 1 and all(n <= 1.05 * b for _, b, n in worse)
    ok &= c2
    lines.append('%s 变差的片子 %d 部（容差：多 5%% 且多 3 以上才算变差）%s' % (
        '✓' if c2 else '✗', len(worse), '：' + '，'.join('%s %.0f→%.0f' % w for w in worse) if worse else ''))
    c3 = new_sum['stops'] <= 6
    ok &= c3
    lines.append('%s 误停 %d 次 ≤ 6（基线 %d）' % ('✓' if c3 else '✗', new_sum['stops'], base_sum['stops']))
    c4 = (new_sum['seek_med'] <= base_sum['seek_med'] + 1e-9 and new_sum['seek_p90'] <= base_sum['seek_p90'] + 1e-9
          and new_sum['seek_never'] <= base_sum['seek_never'] and new_sum['hard_p90'] <= base_sum['hard_p90'] + 1e-9)
    ok &= c4
    lines.append('%s seek 进动作段启动不变慢：中位 %.1f→%.1fs，p90 %.1f→%.1fs，没启动 %d→%d；特写/低声音 p90 %.1f→%.1fs' % (
        '✓' if c4 else '✗', base_sum['seek_med'], new_sum['seek_med'], base_sum['seek_p90'], new_sum['seek_p90'],
        base_sum['seek_never'], new_sum['seek_never'], base_sum['hard_p90'], new_sum['hard_p90']))
    c5 = new_sum['fn'] <= base_sum['fn'] + 1e-9 and new_sum['acc'] >= base_sum['acc'] - 1e-9
    ok &= c5
    lines.append('%s 漏转不升、准确率不降：漏转 %s→%s，准确率 %s→%s' % (
        '✓' if c5 else '✗', A._pct(base_sum['fn']).strip(), A._pct(new_sum['fn']).strip(),
        A._pct(base_sum['acc']).strip(), A._pct(new_sum['acc']).strip()))
    return ok, lines


# ======================================================================== 敏感曲线
SENS_AXES = [
    ('d_sp', (0.5, 0.6, 0.7, 0.8, 0.9)),
    ('d_voc', (0.01, 0.02, 0.03, 0.05, 0.1)),
    ('b_voc', (0.03, 0.05, 0.1, 0.15, 0.2)),
    ('b_sp', (0.1, 0.2, 0.3, 0.4)),
    ('b_imp', (0.0, 0.05, 0.1, 0.2)),
    ('b_mus', (0.1, 0.2, 0.3, 0.5)),
    ('b_sil', (0.1, 0.2, 0.3, 0.5)),
    ('a1_n', (1, 2, 3, 4)),
    ('a1_cv', (2, 3, 4)),
    ('a1_cons', (70, 80, 90, 100)),
]


def cmd_sens(a):
    d = load(a.gt_dir, a.runs_dir, a.npz, only=a.only, quiet=True)
    cost = F.Cost.parse(a.cost)
    c0 = parse_cfg(a.cfg, a.preset)
    vids = [v.vid for v in d.videos]
    base = per_video(d, make_cfg(), cost)
    print('敏感曲线（样本内，代价 = 片子 + 拼接样本）：%s' % cfg_str(c0))
    print('  基线 v1 各片代价：' + ' '.join('%s %.0f' % (v, base[v]['cost']) for v in vids) + '，合计 %.0f\n'
          % sum(base[v]['cost'] for v in vids))
    for key, vals in SENS_AXES:
        if key.startswith('d_') and c0['b'] == 'off':
            continue
        if key.startswith('b_') and not (c0['a1'] or c0['a2'] or c0['a3']):
            continue
        if key.startswith('a1') and not c0['a1']:
            continue
        print('══ %s ══' % key)
        for x in vals:
            c = dict(c0)
            c[key] = x
            pv = per_video(d, c, cost)
            tot = sum(pv[v]['cost'] for v in vids)
            st = run_stress(factory(c), d.videos[0].params)
            stops = sum(pv[v]['fs'] for v in vids)
            fn = sum(fn_seconds(pv[v]) for v in vids)
            fp = sum(fp_seconds(pv[v]) for v in vids)
            print('  %-6s %s 代价 %6.0f  漏转 %5.0fs 误转 %5.0fs 误停 %2d  %s | %s' % (
                x, '*' if x == c0[key] else ' ', tot, fn, fp, stops, '安全✓' if st['safety_ok'] else '安全✗',
                ' '.join('%s %+.0f' % (v, pv[v]['cost'] - base[v]['cost']) for v in vids)))
        print()
    return 0


# ======================================================================== 自检
def selftest():
    fails = []

    def check(ok, msg):
        if not ok:
            fails.append(msg)
        print('  %s %s' % ('ok  ' if ok else 'FAIL', msg))

    # 1. 线索判定
    c = make_cfg()
    check(cue_of(None, c) is None, '没有分数时线索为未知')
    check(cue_of(SYN[DIAL], c) == DIAL, '对白')
    check(cue_of(SYN[MUSIC], c) == MUSIC and cue_of(SYN[QUIET], c) == QUIET, '音乐 / 安静')
    check(cue_of(SYN[VOCAL], c) == VOCAL and cue_of(SYN[OTHER], c) == OTHER, '人声 / 其它')
    check(cue_of(SYN[IMPACT], c) == IMPACT and cue_of(SYN[IMPACT], make_cfg(b_imp=0.0)) == BLIND,
          '撞击声排除出"听不出"（b_imp=0 时不看）')
    check(cue_of((0.0, 0.0, 0.45, 0.5, 0.6, 0.0, 0.0), c) == MUSIC, '带人声的歌（Music ≥ Speech）算音乐')
    # 2. 开关全关 / 线索未知时与 v1 逐 tick 一致：真实片子 + 拼接样本 + seek 片段
    d = load(quiet=True)
    full = make_cfg(a1=1, a2=1, a3=2, b='all', bv=2, p1=1, p2=1, uninf='mqb', a3_set='mqb')
    full_vid = make_cfg(a1=1, a3=2, b='vid', p1=1, p2=1, uninf='mqb', a3_set='mqb')
    n, bad = 0, []
    for v in d.videos:
        preps = [d.prep[v.vid]] + d.sprep[v.vid] + [pp for _, _, pp, _, _ in d.seeks[v.vid]]
        for pp in preps:
            # 开关全关 + 真实线索；开关全开 + 线索未知：两种都必须与 v1 相同
            for cfg, use_cue in ((make_cfg(), True), (full, False), (full_vid, False)):
                ref = F.FusionArbiter(pp.video.params)
                arb = CueArbiter(pp.video.params, cfg)
                for i in range(len(pp.pos)):
                    if pp.reset[i]:
                        ms, warm, _ = pp.reset[i]
                        ref.reset(ms, warm)
                        arb.reset(ms, warm)
                    motor = (i * 7919 + len(pp.pos)) % 3 != 0        # 马达状态随便给，两边一样即可
                    cue = cue_of(pp.aext[i], cfg) if use_cue else None
                    x = ref.decide(pp.wall[i], pp.v[i], pp.vc[i], pp.vts[i], pp.a[i], pp.ac[i], pp.ats[i], motor)
                    y = arb.decide(pp.wall[i], pp.v[i], pp.vc[i], pp.vts[i], pp.a[i], pp.ac[i], pp.ats[i], motor, cue)
                    n += 1
                    if (x, ref.reason) != (y, arb.reason):
                        bad.append((v.vid, i, x, ref.reason, y, arb.reason))
    check(not bad, '开关全关、或线索未知（开关全开）时与 v1 逐 tick 一致（%d tick）%s' % (n, '' if not bad else '：%r' % bad[:3]))
    # 2b. app 的移植（fusion_lab.FusionArbiter v2 + audio_cue）与这里的 a1bvid 逐 tick 一致：13 部片 + 拼接 + seek
    d13 = load(DEF_GT, DEF_RUNS + ',' + os.path.join(ROOT, 'bench', 'runs_test'),
               DEF_NPZ + ',' + os.path.join(ROOT, 'bench', 'rescored', 'test_audio_old'), quiet=True)
    cfg13 = make_cfg(**PRESETS['a1bvid'])
    n2, bad2, cue_bad, cues = 0, [], 0, 0
    for v in d13.videos:
        params = dict(v.params, ARB='v2')
        preps = [d13.prep[v.vid]] + d13.sprep[v.vid] + [pp for _, _, pp, _, _ in d13.seeks[v.vid]]
        for pp in preps:
            ref = F.FusionArbiter(params)
            arb = CueArbiter(params, cfg13)
            for i in range(len(pp.pos)):
                if pp.reset[i]:
                    ms, warm, _ = pp.reset[i]
                    ref.reset(ms, warm)
                    arb.reset(ms, warm)
                motor = (i * 7919 + len(pp.pos)) % 3 != 0
                c1 = cue_of(pp.aext[i], cfg13)
                c2 = F.audio_cue(pp.aext[i])
                cues += 1
                cue_bad += F.CUE_NAMES[c2] != (c1 or '')
                x = ref.decide(pp.wall[i], pp.v[i], pp.vc[i], pp.vts[i], pp.a[i], pp.ac[i], pp.ats[i], motor, c2)
                y = arb.decide(pp.wall[i], pp.v[i], pp.vc[i], pp.vts[i], pp.a[i], pp.ac[i], pp.ats[i], motor, c1)
                n2 += 1
                if (x, ref.reason) != (y, arb.reason):
                    bad2.append((v.vid, i, x, ref.reason, y, arb.reason))
    check(cue_bad == 0, 'fusion_lab.audio_cue 与 cue_of(a1bvid) 一致（%d 个 tick，%d 处不同）' % (cues, cue_bad))
    check(not bad2 and len(d13.videos) == 13,
          'fusion_lab.FusionArbiter v2（app 的移植）与 a1bvid 逐 tick 一致（%d 部片，%d tick）%s'
          % (len(d13.videos), n2, '' if not bad2 else '：%r' % bad2[:3]))
    # 3. 对拍数据（Java 侧的期望，带线索）用这里的 a1bvid 重放也逐行一致
    rows = 0
    mism = []
    with io.open(F.GOLDEN_PATH, encoding='utf-8') as fh:
        check(fh.readline().strip() == F.GOLDEN_HEADER, 'FusionArbiterTest 对拍数据的表头是 v2 格式（带 a_cue）')
        arb = CueArbiter(None, cfg13)
        for line in fh:
            f = line.rstrip('\n').split(',')
            if not f or len(f) < 12:
                continue
            if f[0]:
                if '+' in f[0]:
                    ms, w = f[0].split('+')
                    arb.reset(int(ms), int(w))
                else:
                    arb.reset(int(f[0]))
            got = arb.decide(int(f[1]), f[2], F.f32(float(f[3])), int(f[4]), f[5], F.f32(float(f[6])), int(f[7]),
                             f[9] == '1', f[8] or None)
            rows += 1
            if (got, arb.reason) != (f[10], f[11]):
                mism.append(rows)
    check(rows >= 10000 and not mism, 'FusionArbiterTest 的对拍数据（%d 行，带线索）用 a1bvid 重放逐行一致%s'
          % (rows, '' if not mism else '：第 %r 行' % mism[:5]))
    # 4. 线索接上的比例
    for vid, (hit, tot) in sorted(d.coverage.items()):
        check(tot > 0 and hit >= 0.99 * tot, '%s 的音频结果接上 YAMNet 分数 %d/%d' % (vid, hit, tot))
    # 5. 合成场景能把线索带进去，A1 / B 在目标场景里起作用
    params = d.videos[0].params
    ok1, _, at, _ = _run_one(factory(make_cfg(a1=1, uninf='m')), params, lambda t: _vP(), lambda t: _aN(0.8, MUSIC), 60000, 'start')
    ok0, _, _, _ = _run_one(factory(make_cfg()), params, lambda t: _vP(), lambda t: _aN(0.8, MUSIC), 60000, 'start')
    check(ok1 and not ok0, 'A1：动作+音乐时 v1 不启动、A1 在 %s 秒启动' % at)
    okb, _, _, _ = _run_one(factory(make_cfg(b='start')), params, lambda t: _vN(), lambda t: _aP(0.95, DIAL), 60000, 'nostart')
    okb0, _, _, _ = _run_one(factory(make_cfg()), params, lambda t: _vN(), lambda t: _aP(0.95, DIAL), 60000, 'nostart')
    check(okb and not okb0, 'B：剧情对白时 v1 会启动、B=start 不启动')
    for name, kw in sorted(PRESETS.items()):
        check(run_stress(factory(make_cfg(**kw)), params)['safety_ok'], '候选 %s 通过全部压力场景安全约束' % name)
    # 6. 慢手机抽稀时线索跟着保留下来的结果走
    sd = A.slow_data(d, 2, 'a')
    moved = 0
    for v in sd.videos:
        for t in v.ticks:
            if t.ays is not None and t.a_age is not None and t.a_age >= 0:
                moved += 1
    check(moved > 1000, '慢手机抽稀后线索仍在（%d tick 带分数）' % moved)
    print('selftest: %s' % ('ALL PASS' if not fails else '%d FAIL' % len(fails)))
    return 0 if not fails else 1


# ======================================================================== main
def main():
    if '--selftest' in sys.argv:
        return selftest()
    ap = argparse.ArgumentParser(description='YAMNet 线索 + 融合仲裁离线实验（详见文件头）')
    sub = ap.add_subparsers(dest='cmd')
    for name, hlp in (('stats', '各线索的占比'), ('blame', '误判归因'), ('stress', '压力场景'),
                      ('eval', '某个配置的完整报告'), ('scan', '单条规则扫阈值'), ('lovo', '留一视频'),
                      ('sens', '阈值敏感曲线')):
        p = sub.add_parser(name, help=hlp)
        p.add_argument('gt_dir', nargs='?', default=DEF_GT)
        p.add_argument('runs_dir', nargs='?', default=DEF_RUNS)
        p.add_argument('--npz', default=DEF_NPZ, help='.yamnet.npz 所在目录（默认 bench/rescored/audio_old）')
        p.add_argument('--cfg', nargs='*', default=[], help='覆盖配置，如 a1=1 b=start d_sp=0.8')
        p.add_argument('--preset', default='', help='候选配置：%s（--cfg 可以再覆盖）' % ' / '.join(sorted(PRESETS)))
        p.add_argument('--cost', default='fp=1,fn=1,stop=3')
        p.add_argument('--only', default='', help='只看这几部片，如 v08,v09,v11')
        if name == 'stress':
            p.add_argument('--brief', action='store_true', help='安全约束只列两边不同或没过的')
        if name == 'eval':
            p.add_argument('--slow', action='store_true', help='再跑模拟慢手机')
        if name == 'scan':
            p.add_argument('--top', type=int, default=6)
        if name == 'lovo':
            p.add_argument('--jobs', type=int, default=max(1, min(4, os.cpu_count() or 1)))
            p.add_argument('--fn-w', type=float, default=1.0,
                           help='选参时漏转秒数的权重（默认 1，与代价相同；验收仍按默认代价）')
            p.add_argument('--save', default='', help='把各折选中的配置写到文件')
            p.add_argument('--cache', default='', help='候选配置的评测结果缓存（pickle）；网格和代价不变时直接读')
            p.add_argument('--where', default='', help='只在满足条件的配置里选，如 "b!=all,bv=0"（不在维持阶段否决）')
    a = ap.parse_args()
    fn = {'stats': cmd_stats, 'blame': cmd_blame, 'stress': cmd_stress, 'eval': cmd_eval, 'scan': cmd_scan,
          'lovo': cmd_lovo, 'sens': cmd_sens}.get(a.cmd)
    if fn is None:
        ap.print_help()
        return 2
    return fn(a)


if __name__ == '__main__':
    sys.exit(main())
