#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
bench/tools/arbiter_lab.py — 融合仲裁（FusionArbiter）的离线评测：旧融合 vs 融合仲裁，按"最差的片子"看。

fusion_lab.py 的 eval 只给总准确率 / 误转 / 漏转 / 误停和拼接样本中位数。这里补上评审要求看的几项：
  · 逐片的漏转率和最长连续漏转（总漏转 1.5% 会掩盖 v03 那种"六分之一的动作时间不转"）
  · 动作中误停次数、真实结束动作后的停转延迟、拼接样本里"基本不停"（≥40 秒）的个数
  · seek 直接进动作段（特写 / 低声音）之后多久才启动；seek 进剧情段之后误转多久
  · 压力测试：交替出现的零星假"转"会不会无限续持、看不清 / 结果过期时能不能停、手机推理慢一倍时还灵不灵
仲裁器用的是 fusion_lab.FusionArbiter（与 FusionArbiter.java 逐 tick 对拍过的那份移植）。

用法:
  python3 bench/tools/arbiter_lab.py report [gt_dir] [runs_dir]   # 旧融合 vs 融合仲裁，逐片 + 拼接 + seek
  python3 bench/tools/arbiter_lab.py stress [gt_dir] [runs_dir]   # 合成压力场景 + 模拟慢手机
  python3 bench/tools/arbiter_lab.py lovo   [gt_dir] [runs_dir]   # 留一视频选参（--wide 连音频兜底窗口一起搜）
默认 gt_dir = bench/gt，runs_dir = bench/runs_initial。
"""
import argparse
import contextlib
import io
import itertools
import math
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fusion_lab as F  # noqa: E402
import metrics as M     # noqa: E402

ROOT = F.ROOT
GRID_MS = F.GRID_MS


# ======================================================================== 数据
class Prep(object):
    """一部片（或拼接样本 / seek 片段）的逐 tick 输入。aext：音频结果的 YAMNet 分组分数（Tick.ays，cue_lab.py 用）。"""
    __slots__ = ('video', 'pos', 'wall', 'v', 'vc', 'vts', 'a', 'ac', 'ats', 'aext', 'loud_lv', 'loud_conf',
                 'reset')


def prepare(video, mask=None):
    """mask=(seek 时刻的墙钟, 视频就绪 ms, 音频就绪 ms)：seek 之后、模型窗口还没重新攒满之前产出的结果一律丢弃
    （真机 seek 会清空帧窗与 PCM：视频要重新攒 12 帧≈2.75s，音频有 2s 死区）。"""
    p = Prep()
    p.video = video
    n = len(video.ticks)
    p.pos, p.wall = [0] * n, [0] * n
    p.v, p.vc, p.vts = [''] * n, [0.0] * n, [None] * n
    p.a, p.ac, p.ats, p.aext = [''] * n, [0.0] * n, [None] * n, [None] * n
    p.loud_lv, p.loud_conf = [0] * n, [0.0] * n
    p.reset = [None] * n                      # app 在这个 tick 之前 reset 过：(ms, 预热 ms, 是否 seek)
    for i, t in enumerate(video.ticks):
        if mask is None:
            p.reset[i] = F.parse_arb_reset((getattr(t, 'log', None) or {}).get('arb_reset', ''))
        v, vc, a, ac = t.v, t.vc, t.a, t.ac
        vts, ats = F._tick_ts(t, 'v'), F._tick_ts(t, 'a')
        aext = getattr(t, 'ays', None)
        if mask is not None:
            sw, vr, ar = mask
            if vts is None or vts < sw + vr:
                v, vc, vts = '', 0.0, None
            if ats is None or ats < sw + ar:
                a, ac, ats, aext = '', 0.0, None, None
        p.pos[i], p.wall[i] = t.pos, t.wall
        p.v[i], p.vc[i], p.vts[i] = v, vc, vts
        p.a[i], p.ac[i], p.ats[i], p.aext[i] = a, ac, ats, aext
        p.loud_lv[i], p.loud_conf[i] = t.loud_lv, t.loud_conf
    return p


SEEK_STEP_MS, SEEK_WIN_MS, V_READY_MS, A_READY_MS = 15000, 30000, 2750, 2000


def make_seeks(video):
    """每个够长的真值段里每 15 秒一个 seek 点，从该点起以全新状态回放 30 秒（或到段尾）。
    hard_v / hard_a：seek 后最先出的 3 个视频 / 2 个音频结果多数不是"转"——特写遮挡 / 低声音的近似。"""
    out = []
    for s, e, lab in video.segs:
        b = M.bin_of(lab, video.policy)
        if b not in ('do', 'noise') or e - s < 12000:
            continue
        k = 0
        while True:
            sp = s + 1000 + k * SEEK_STEP_MS
            if sp + 10000 > e:
                break
            k += 1
            if sp < M.WARMUP_MS:
                continue
            ticks = [t for t in video.ticks if sp <= t.pos < min(e - 500, sp + SEEK_WIN_MS)]
            if len(ticks) < 15:
                continue
            pp = prepare(video.with_ticks(ticks), mask=(ticks[0].wall, V_READY_MS, A_READY_MS))
            vs = _first(pp.v, pp.vts, 3)
            as_ = _first(pp.a, pp.ats, 2)
            hard_v = sum(1 for x in vs if x != 'do') >= 2
            hard_a = sum(1 for x in as_ if x != 'do') >= 1
            out.append(('do' if b == 'do' else 'plot', lab, pp, hard_v, hard_a))
    return out


def _first(acts, tss, n):
    seen, out = None, []
    for act, ts in zip(acts, tss):
        if ts is None or ts == seen:
            continue
        seen = ts
        out.append(act)
        if len(out) >= n:
            break
    return out


class Data(object):
    pass


def load(gt_dir, runs_dir, seeks=True):
    d = Data()
    with contextlib.redirect_stdout(io.StringIO()):
        d.videos = F.load_videos(gt_dir, runs_dir, quiet=True)
    d.splices = dict((v.vid, F.make_splices(v)) for v in d.videos)
    d.prep = dict((v.vid, prepare(v)) for v in d.videos)
    d.sprep = dict((v.vid, [prepare(s) for s in d.splices[v.vid]]) for v in d.videos)
    d.seeks = dict((v.vid, make_seeks(v) if seeks else []) for v in d.videos)
    return d


# ======================================================================== 策略
class Legacy(object):
    """旧融合：投票窗 + applyDoStartGate（与 fusion_lab 的 replay_legacy(arbiter=False) 相同）。"""

    def __init__(self, params):
        self.params = params
        self.core = F.LegacyCore(params)

    def on_reset(self, ms, warm, seek):
        if seek:
            self.core = F.LegacyCore(self.params)

    def step(self, p, i, ble):
        v, vc, a, ac = p.v[i], p.vc[i], p.a[i], p.ac[i]
        sm = self.core.smoothed(v, a, vc, ac)
        return self.core.gate(sm, v, vc, a, ac, ble.cur)


class Arbiter(object):
    """融合仲裁：fusion_lab.FusionArbiter；overrides 覆盖 ARB_* 参数（扫参用）。"""

    def __init__(self, params, overrides=None):
        q = dict(params)
        if overrides:
            q.update(overrides)
        self.arb = F.FusionArbiter(q)

    def on_reset(self, ms, warm, seek):
        self.arb.reset(ms, warm)

    def step(self, p, i, ble):
        return self.arb.decide(p.wall[i], p.v[i], p.vc[i], p.vts[i], p.a[i], p.ac[i], p.ats[i],
                               F._is_sex(ble.cur), F.audio_cue(p.aext[i]))


def replay(p, policy, want_states=False):
    ble = F.Ble(p.video.params)
    cur, trans, states = 'noise', [], []
    for i in range(len(p.pos)):
        if p.reset[i]:                        # 与 fusion_lab.replay_legacy 相同：seek 还清空蓝牙状态与投票窗
            ms, warm, seek = p.reset[i]
            if seek:
                F.ble_seek_reset(ble)
            policy.on_reset(ms, warm, seek)
        act = policy.step(p, i, ble)
        if act:
            s = ble.update(act, F.final_freq(p.loud_lv[i], p.loud_conf[i], ble.level), p.wall[i])
            if s is not None:
                st = 'do' if F._is_sex(s[0]) else 'noise'
                if st != cur:
                    trans.append((p.pos[i], st))
                    cur = st
        if want_states:
            states.append(cur)
    return trans, states


def factory(name, overrides=None):
    if name == 'legacy':
        return lambda params: Legacy(params)
    return lambda params: Arbiter(params, overrides)


# ======================================================================== 指标
def longest_run(video, trans, want_gt, bad_state):
    """真值为 want_gt 的可评测格里，马达状态连续为 bad_state 的最长时长（秒）。"""
    gti, best, cur = video.gti, 0, 0
    for i in range(gti.n):
        if not gti.mask[i] or gti.gt[i] != want_gt:
            cur = 0
            continue
        if M.state_at(trans, i * GRID_MS) == bad_state:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return best * GRID_MS / 1000.0


def _q(xs, q):
    if not xs:
        return float('nan')
    xs = sorted(xs)
    return xs[min(len(xs) - 1, max(0, int(math.ceil(q * len(xs))) - 1))]


def evaluate(data, make, cost=None, seeks=True, only=None):
    cost = cost or F.Cost()
    rep = {'videos': {}, 'pooled': F.Score(), 'splice': [], 'seek_do': [], 'seek_plot': [], 'lat_stop': []}
    for v in data.videos:
        if only and v.vid not in only:
            continue
        tr, _ = replay(data.prep[v.vid], make(v.params))
        s = F.Score.of(v, tr, cost)
        rep['pooled'].add(s)
        rep['videos'][v.vid] = dict(acc=s.acc(), fp=s.fp(), fn=s.fn(), stops=s.fs,
                                    long_miss=longest_run(v, tr, 'do', 'noise'),
                                    long_false=longest_run(v, tr, 'noise', 'do'))
        rep['lat_stop'] += s.lat['do->noise']
        for sp in data.sprep[v.vid]:
            t2, _ = replay(sp, make(v.params))
            rep['splice'].append((sp.video.splice, sp.video.vid, F.Score.of(sp.video, t2, cost),
                                  F.splice_seconds(sp.video, t2)))
        if seeks:
            for kind, lab, pp, hv, ha in data.seeks[v.vid]:
                t3, st = replay(pp, make(v.params), want_states=True)
                win = (pp.pos[-1] - pp.pos[0]) / 1000.0
                if kind == 'do':
                    first = next((pos for pos, x in t3 if x == 'do'), None)
                    dt = (first - pp.pos[0]) / 1000.0 if first is not None else None
                    rep['seek_do'].append((v.vid, lab, pp.pos[0], dt, win, hv or ha))
                else:
                    rep['seek_plot'].append((v.vid, pp.pos[0], sum(1 for x in st if x == 'do') * 0.5, win))
    rep['cost'] = rep['pooled'].cost + sum(sc.cost for _, _, sc, _ in rep['splice'])
    return rep


def summary(rep):
    P = rep['pooled']
    stop = [x for k, _, _, x in rep['splice'] if k == 'stop']
    start = [x for k, _, _, x in rep['splice'] if k == 'start']
    sd = [(dt if dt is not None else win) for _, _, _, dt, win, _ in rep['seek_do']]
    hard = [(dt if dt is not None else win) for _, _, _, dt, win, h in rep['seek_do'] if h]
    vv = rep['videos']
    return dict(
        acc=P.acc(), fp=P.fp(), fn=P.fn(), stops_pm=P.stops_per_min(), stops=P.fs, cost=rep['cost'],
        worst_fn=max((x['fn'] for x in vv.values() if x['fn'] == x['fn']), default=float('nan')),
        worst_long_miss=max((x['long_miss'] for x in vv.values()), default=0.0),
        stop_med=statistics.median(stop) if stop else float('nan'), stop_stuck=sum(1 for x in stop if x >= 40),
        start_med=statistics.median(start) if start else float('nan'), start_stuck=sum(1 for x in start if x >= 40),
        seek_med=statistics.median(sd) if sd else float('nan'), seek_p90=_q(sd, 0.9),
        seek_never=sum(1 for x in rep['seek_do'] if x[3] is None), seek_n=len(sd),
        hard_med=statistics.median(hard) if hard else float('nan'), hard_p90=_q(hard, 0.9),
        hard_never=sum(1 for x in rep['seek_do'] if x[3] is None and x[5]), hard_n=len(hard),
        plot_on=(sum(x[2] for x in rep['seek_plot']) / len(rep['seek_plot'])) if rep['seek_plot'] else float('nan'),
        plot_any=sum(1 for x in rep['seek_plot'] if x[2] > 0), plot_n=len(rep['seek_plot']),
        lat_stop=[round(x / 1000.0, 1) for x in rep['lat_stop']])


def _pct(x):
    return '  —  ' if x != x else '%5.1f%%' % (100.0 * x)


def print_report(rep, title):
    s = summary(rep)
    print('══ %s ══' % title)
    print('  准确率 %s  误转 %s  漏转 %s  误停 %.2f 次/分（%d 次）  代价 %.0f'
          % (_pct(s['acc']), _pct(s['fp']), _pct(s['fn']), s['stops_pm'], s['stops'], s['cost']))
    print('  最差片漏转 %s，最长连续漏转 %.1fs；真实停转延迟 %s 秒'
          % (_pct(s['worst_fn']).strip(), s['worst_long_miss'], s['lat_stop']))
    print('  拼接样本：停止后还转 中位 %.1fs，≥40s %d 个；启动后没转 中位 %.1fs，≥40s %d 个'
          % (s['stop_med'], s['stop_stuck'], s['start_med'], s['start_stuck']))
    print('  seek 进动作段：中位 %.1fs，p90 %.1fs，没启动 %d/%d；其中特写/低声音：中位 %.1fs，p90 %.1fs，没启动 %d/%d'
          % (s['seek_med'], s['seek_p90'], s['seek_never'], s['seek_n'],
             s['hard_med'], s['hard_p90'], s['hard_never'], s['hard_n']))
    print('  seek 进剧情段：30 秒里平均误转 %.1fs，%d/%d 次转过' % (s['plot_on'], s['plot_any'], s['plot_n']))
    print('  逐片  准确率 / 误转 / 漏转 / 误停 / 最长连续漏转 / 最长连续误转')
    for vid in sorted(rep['videos']):
        x = rep['videos'][vid]
        print('    %s  %s / %s / %s / %d / %5.1fs / %5.1fs' % (
            vid, _pct(x['acc']), _pct(x['fp']), _pct(x['fn']), x['stops'], x['long_miss'], x['long_false']))
    stuck = ['%s %.0fs' % (vid, sec) for k, vid, _, sec in rep['splice'] if sec >= 40]
    if stuck:
        print('  拼接样本 ≥40s：' + '，'.join(stuck))
    print()


# ======================================================================== 压力测试
def _synth_ticks(vfun, afun, dur_ms, v_period=625, a_period=1000):
    """vfun/afun(t) -> (动作, 置信度) 或 None（这一刻不出新结果，用来造"结果过期"）。
    afun 也可以返回 (动作, 置信度, YAMNet 分组分数)，分数写进 Tick.ays（cue_lab.py 的压力场景用）。"""
    T0, ticks, last_v, last_a = 1700000000000, [], None, None
    tv, ta, t = 300, 700, 0
    while t < dur_ms:
        while tv <= t:
            r = vfun(tv)
            if r is not None:
                last_v = (T0 + tv, r[0], r[1])
            tv += v_period
        while ta <= t:
            r = afun(ta)
            if r is not None:
                last_a = (T0 + ta, r[0], r[1], r[2] if len(r) > 2 else None)
            ta += a_period
        k = F.Tick()
        k.pos, k.wall = t, T0 + t
        k.ays = None
        for which, last in (('v', last_v), ('a', last_a)):
            if last is None:
                act, conf, age = '', 0.0, -1
            else:
                age = k.wall - last[0]
                act, conf = (last[1], F.f32(last[2])) if age <= F.MAX_AGE_MS else ('', 0.0)
                if which == 'a' and age <= F.MAX_AGE_MS:
                    k.ays = last[3]
            setattr(k, which, act)
            setattr(k, which + 'c', conf)
            setattr(k, which + '_age', age)
        k.vp = k.ap = None
        k.loud_lv, k.loud_conf, k.log = 3, F.f32(0.5), {}
        ticks.append(k)
        t += 500
    return ticks


_P = lambda c=0.95: ('do', c)          # noqa: E731
_N = lambda c=0.9: ('Noise', c)        # noqa: E731
_U = ('', 0.0)
SW = 30000                              # 先 30 秒干净的动作，然后进入场景
STOP_WITHIN_S = 20                      # "应该停"的场景：20 秒内停下就算有限（结果完全中断时实际约 16.5 秒）


def _alt(t, period, a, b):
    return a if (t // period) % 2 == 0 else b


SCENARIOS = [
    # 名称, 视频, 音频, 总时长, 期望（stop = 应该停，hold = 应该一直转）
    ('剧情：音频 +/- 交替，视频 -', lambda t: _P() if t < SW else _N(),
     lambda t: _P() if t < SW else _alt(t, 1000, _P(0.8), _N(0.7)), SW + 45000, 'stop'),
    ('剧情：音频 3 个里 1 个 +，视频 -', lambda t: _P() if t < SW else _N(),
     lambda t: _P() if t < SW else (_P(0.8) if (t // 1000) % 3 == 0 else _N(0.7)), SW + 45000, 'stop'),
    ('剧情：视频 +/- 交替，音频 -', lambda t: _P() if t < SW else _alt(t, 625, _P(0.8), _N(0.8)),
     lambda t: _P() if t < SW else _N(0.8), SW + 45000, 'stop'),
    ('剧情：两路都 -（正常结束）', lambda t: _P() if t < SW else _N(), lambda t: _P() if t < SW else _N(0.8),
     SW + 45000, 'stop'),
    ('结果过期：两路都不再出新结果', lambda t: _P() if t < SW else None, lambda t: _P() if t < SW else None,
     SW + 45000, 'stop'),
    ('看不清听不清：两路都 unclear', lambda t: _P() if t < SW else _U, lambda t: _P() if t < SW else _U,
     SW + 45000, 'stop'),
    ('遮挡：视频 - 5 秒，音频 + 夹一个 -', lambda t: _N() if SW <= t < SW + 5000 else _P(),
     lambda t: _N(0.7) if SW + 2000 <= t < SW + 3000 else _P(), SW + 20000, 'hold'),
    ('特写：视频 - 40 秒，音频 + 每 12 秒夹一个 -', lambda t: _P() if t < SW else _N(),
     lambda t: _N(0.7) if t >= SW and (t // 1000) % 12 == 5 else _P(), SW + 45000, 'hold'),
    ('音频听不到：视频 + 45 秒，音频 -', lambda t: _P(), lambda t: _P() if t < SW else _N(0.8), SW + 45000, 'hold'),
    ('音频听不到 + 视频每 4 个夹一个 unclear', lambda t: _P() if t < SW or (t // 625) % 4 else _U,
     lambda t: _P() if t < SW else _N(0.8), SW + 45000, 'hold'),
    ('视频断了（不再出结果），音频 5 个里 3 个 +', lambda t: _P() if t < SW else None,
     lambda t: _P() if t < SW else (_P(0.9) if (t // 1000) % 5 < 3 else _N(0.8)), SW + 60000, 'stop'),
    ('视频每 2.5 秒才出一个 -，音频 5 个里 3 个 +', lambda t: _P() if t < SW else (_N(0.9) if (t // 2500) % 4 == 0 else None),
     lambda t: _P() if t < SW else (_P(0.9) if (t // 1000) % 5 < 3 else _N(0.8)), SW + 60000, 'stop'),
    # 视频 20 秒前就断了（最近的结果早已滑出维持窗口），靠音频维持；之后音频变成"5 个里 3 个 +"。
    # "某一路结果不够就先等"如果不限于刚 reset，这里会一直等下去：音频每 5 秒有 2 秒的支持，等待永远不超时
    ('视频 20 秒前就断了，之后音频 5 个里 3 个 +', lambda t: _P() if t < SW - 20000 else None,
     lambda t: _P() if t < SW else (_P(0.9) if (t // 1000) % 5 < 3 else _N(0.8)), SW + 60000, 'stop'),
]


FLAP_SCENARIOS = [
    # 两路都是强证据、"转""不转"交替：内容模棱两可，转或不转都行，最不能接受的是马达每 2 秒开关一次
    ('两路强证据 +/- 交替（视频 600ms、音频 1s）',
     lambda t: _P() if (t // 600) % 2 == 0 else _N(0.95), lambda t: _P() if (t // 1000) % 2 == 0 else _N(0.95)),
    ('两路强证据，每 3 个里 1 个 -',
     lambda t: _N(0.95) if (t // 600) % 3 == 0 else _P(), lambda t: _N(0.95) if (t // 1000) % 3 == 0 else _P()),
    ('证据在维持门槛上下晃（视频 2/3 转，音频 1/3 转其余 unclear）',
     lambda t: _N(0.8) if (t // 625) % 3 == 0 else _P(), lambda t: _P() if (t // 1000) % 3 == 0 else _U),
]


def run_flap(make, params, dur=120000):
    """返回 [(名称, 每分钟开关次数)]。"""
    out = []
    for name, vf, af in FLAP_SCENARIOS:
        ticks = _synth_ticks(vf, af, dur)
        v = F.Video('syn', [(0, dur, 'sex')], dur, ticks, params, synthetic=True)
        tr, _ = replay(prepare(v), make(params))
        out.append((name, len(tr) / (dur / 60000.0)))
    return out


def run_scenarios(make, params):
    """返回 [(名称, 期望, 场景开始后转了几秒, 场景开始后第一次停在几秒（None=没停）, 是否符合期望)]。"""
    out = []
    for name, vf, af, dur, want in SCENARIOS:
        ticks = _synth_ticks(vf, af, dur)
        v = F.Video('syn', [(0, dur, 'sex')], dur, ticks, params, synthetic=True)
        tr, st = replay(prepare(v), make(params), want_states=True)
        on = sum(1 for t, x in zip(ticks, st) if t.pos >= SW and x == 'do') * 0.5
        off = next(((pos - SW) / 1000.0 for pos, x in tr if x == 'noise' and pos >= SW), None)
        ok = (off is not None and off <= STOP_WITHIN_S) if want == 'stop' else off is None
        out.append((name, want, on, off, ok))
    return out


def decimate(video, keep_every, which):
    """模拟慢手机：某个模态只保留每 keep_every 个新结果中的一个（结果间隔变成 keep_every 倍）。"""
    ticks, kept, n, last_seen = [], None, 0, None
    for t in video.ticks:
        ts = F._tick_ts(t, which)
        if ts is not None and ts != last_seen:
            last_seen = ts
            if n % keep_every == 0:
                kept = (ts, t.v if which == 'v' else t.a, t.vc if which == 'v' else t.ac, getattr(t, 'ays', None))
            n += 1
        if kept is None:
            kw = {which: '', which + 'c': 0.0, which + '_age': -1}
        else:
            ag = t.wall - kept[0]
            kw = ({which: '', which + 'c': 0.0, which + '_age': ag} if ag > F.MAX_AGE_MS
                  else {which: kept[1], which + 'c': kept[2], which + '_age': ag})
        if which == 'a':                       # YAMNet 分数跟着保留下来的那个音频结果走
            kw['ays'] = kept[3] if kept is not None and kw['a_age'] <= F.MAX_AGE_MS else None
        ticks.append(t.copy(**kw))
    nv = video.with_ticks(ticks)
    nv.splice = video.splice
    return nv


def slow_data(data, keep_every, which):
    d = Data()
    d.videos = [decimate(v, keep_every, which) for v in data.videos]
    d.prep = dict((v.vid, prepare(v)) for v in d.videos)
    d.sprep = dict((v.vid, [prepare(decimate(s, keep_every, which)) for s in data.splices[v.vid]])
                   for v in data.videos)
    d.seeks = dict((v.vid, []) for v in d.videos)
    return d


# ======================================================================== 留一视频选参
NARROW = {'ARB_MV_N': [4, 5], 'ARB_V_CONS': [60, 70], 'ARB_CONS_WIN': [6000, 8000], 'ARB_V_FLOOR': [0.85, 0.90]}
WIDE = dict(NARROW, ARB_Z_WIN=[8000, 9000, 10000, 12000])


def lovo(data, grid, cost=None, log=print):
    """每部片只用其余几部选参。约束：压力场景全部符合期望；训练片里 seek 进动作段从不失败、p90 ≤ 10 秒。"""
    cost = cost or F.Cost()
    keys = sorted(grid)
    configs = [dict(zip(keys, vals)) for vals in itertools.product(*[grid[k] for k in keys])]
    params0 = data.videos[0].params
    per = {}
    for ci, cfg in enumerate(configs):
        mk = factory('arbiter', cfg)
        if not all(ok for _, _, _, _, ok in run_scenarios(mk, params0)):
            continue
        for v in data.videos:
            sub = evaluate(data, mk, cost, seeks=True, only={v.vid})
            per[(ci, v.vid)] = (sub['cost'], [(dt if dt is not None else None) for _, _, _, dt, _, _ in sub['seek_do']])
    ok_cfgs = sorted({ci for ci, _ in per})
    log('  %d 组参数，%d 组通过压力场景' % (len(configs), len(ok_cfgs)))
    rep = {'videos': {}, 'pooled': F.Score(), 'splice': [], 'seek_do': [], 'seek_plot': [], 'lat_stop': []}
    picks = {}
    for v in data.videos:
        train = [u.vid for u in data.videos if u.vid != v.vid]

        def seek_ok(ci):
            xs = [x for u in train for x in per[(ci, u)][1]]
            return all(x is not None for x in xs) and _q(xs, 0.9) <= 10.0
        cand = [ci for ci in ok_cfgs if seek_ok(ci)] or ok_cfgs
        best = min(cand, key=lambda ci: sum(per[(ci, u)][0] for u in train))
        picks[v.vid] = configs[best]
        sub = evaluate(data, factory('arbiter', configs[best]), cost, seeks=True, only={v.vid})
        rep['videos'].update(sub['videos'])
        rep['pooled'].add(sub['pooled'])
        for k in ('splice', 'seek_do', 'seek_plot', 'lat_stop'):
            rep[k] += sub[k]
    rep['cost'] = rep['pooled'].cost + sum(sc.cost for _, _, sc, _ in rep['splice'])
    return rep, picks


# ======================================================================== 子命令
def cmd_report(a):
    d = load(a.gt_dir, a.runs_dir)
    print_report(evaluate(d, factory('legacy')), '旧融合（投票窗 + applyDoStartGate）')
    print_report(evaluate(d, factory('arbiter')), '融合仲裁（FusionArbiter，app 当前参数）')
    print('  · 准确率 / 误转 / 漏转 / 误停只算真实片子，参数是在这几部片上定的（样本内）；留一视频的数字看 lovo。')
    print('  · seek 模拟丢掉了 seek 后视频 2.75 秒、音频 2 秒内产出的结果；"特写/低声音"是按 seek 后最先几个结果近似挑的。')
    print('  · 拼接样本在拼接处瞬间切换，停转 / 启动延迟比真实片子乐观约 1~2 秒。')
    return 0


def cmd_stress(a):
    d = load(a.gt_dir, a.runs_dir, seeks=False)
    params = d.videos[0].params
    print('══ 合成压力场景（30 秒动作之后进入场景；期望 stop = %d 秒内停，hold = 一直转）══' % STOP_WITHIN_S)
    res = dict((n, run_scenarios(factory(n), params)) for n in ('legacy', 'arbiter'))
    print('  %-36s %-5s %-24s %s' % ('场景', '期望', '旧融合', '融合仲裁'))
    for i, (name, _, _, _, want) in enumerate(SCENARIOS):
        cells = []
        for n in ('legacy', 'arbiter'):
            _, _, on, off, ok = res[n][i]
            cells.append('%s 转 %4.1fs，%s' % ('✓' if ok else '✗', on, '没停' if off is None else '%.1fs 停' % off))
        print('  %-36s %-5s %-24s %s' % (name, want, cells[0], cells[1]))
    flap = dict((n, run_flap(factory(n), params)) for n in ('legacy', 'arbiter'))
    for i, (name, _, _) in enumerate(FLAP_SCENARIOS):
        print('  %-36s %-5s %-24s %s' % (name, '不乱开关', '开关 %.1f 次/分' % flap['legacy'][i][1],
                                          '开关 %.1f 次/分' % flap['arbiter'][i][1]))
    print()
    for k, which, label in ((2, 'v', '视频结果间隔 ×2'), (3, 'v', '视频结果间隔 ×3'), (4, 'v', '视频结果间隔 ×4'),
                            (2, 'a', '音频结果间隔 ×2'), (3, 'a', '音频结果间隔 ×3')):
        sd = slow_data(d, k, which)
        print('══ 模拟慢手机：%s ══' % label)
        for n, title in (('legacy', '旧融合'), ('arbiter', '融合仲裁')):
            s = summary(evaluate(sd, factory(n), seeks=False))
            print('  %-8s 准确率 %s 误转 %s 漏转 %s 误停 %.2f/分 最差片漏转 %s 拼接≥40s 停 %d / 启 %d'
                  % (title, _pct(s['acc']), _pct(s['fp']), _pct(s['fn']), s['stops_pm'],
                     _pct(s['worst_fn']).strip(), s['stop_stuck'], s['start_stuck']))
        print()
    return 0


def cmd_lovo(a):
    d = load(a.gt_dir, a.runs_dir)
    grid = WIDE if a.wide else NARROW
    print('留一视频选参（网格 %s）…' % ', '.join('%s∈%s' % (k, v) for k, v in sorted(grid.items())))
    rep, picks = lovo(d, grid)
    for vid in sorted(picks):
        print('  留出 %s：%s' % (vid, ' '.join('%s=%s' % (k[4:], v) for k, v in sorted(picks[vid].items()))))
    print()
    print_report(rep, '融合仲裁（留一视频选参）')
    return 0


def main():
    ap = argparse.ArgumentParser(description='融合仲裁离线评测（详见文件头）')
    sub = ap.add_subparsers(dest='cmd')
    for name, hlp in (('report', '旧融合 vs 融合仲裁'), ('stress', '合成压力场景 + 模拟慢手机'),
                      ('lovo', '留一视频选参')):
        p = sub.add_parser(name, help=hlp)
        p.add_argument('gt_dir', nargs='?', default=os.path.join(ROOT, 'bench', 'gt'))
        p.add_argument('runs_dir', nargs='?', default=os.path.join(ROOT, 'bench', 'runs_initial'))
        if name == 'lovo':
            p.add_argument('--wide', action='store_true', help='连音频兜底窗口 ARB_Z_WIN 一起搜')
    a = ap.parse_args()
    fn = {'report': cmd_report, 'stress': cmd_stress, 'lovo': cmd_lovo}.get(a.cmd)
    if fn is None:
        ap.print_help()
        return 2
    return fn(a)


if __name__ == '__main__':
    sys.exit(main())
