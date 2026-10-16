#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
bench/tools/retune.py — 给一个视频变体重调视频阈值和与视频置信度有关的融合参数（留一视频选参），按 README 验收线检查

什么时候用：新模型的概率分布变了（例如 label smoothing 把最高概率压到约 0.97），只用现用参数比不公平。这时两边
（exact_crop 和新模型）各自用同一套流程调好再比（compare）。其余情况先用 baseline.py compare（现用参数）比。

调哪些（只调和视频置信度有关的 4 个；其余融合参数、音频阈值、代价不动）
  VIDEO_DO / VIDEO_PLOT   视频判"转" / "不转"的阈值（VideoProcessActivity / OnlineAnalysisActivity：0.65 / 0.60）
  ARB_V_FLOOR             两路佐证启动、视频单独启动里"视频强转"的置信度下限（FusionArbiter.V_FLOOR：0.85）
  ARB_BV_CONF             对白否决里"视频确定是剧情"的置信度下限（FusionArbiter.BV_CONF：0.90）
  网格：--grid video 只调前两个（4×3 = 12 组）；--grid conf（默认）四个一起（4×3×3×3 = 108 组）。第 0 组是现用参数。

怎么选（留一视频，同 cue_lab.py lovo）：轮流留出一部片，在其余 12 部上选代价最低的一组，约束：
  · 训练片误停次数不比现用参数多 1 次以上，训练片漏转秒数不比现用参数多
  · 训练片 seek 进动作段：没启动的次数不比现用参数多，p90 不超过 max(10 秒, 现用参数)
留出的片子用这一折选中的参数评测，13 折拼起来就是"留一视频选参"的结果。另给"13 部一起选"（样本内）作参照：
它总比留一好看，差得越多越说明在过拟合。

用法（仓库根目录；Windows 把 python3 换成 python）
  python3 bench/tools/retune.py run exact_crop                # 现用参数 vs 留一选参 vs 样本内
  python3 bench/tools/retune.py run e1c --grid video --jobs 4
  python3 bench/tools/retune.py compare exact_crop e1c        # 两边各自留一选参后并排比（第一个是参照），按验收线检查
  python3 bench/tools/retune.py --selftest
报告写进 bench/baseline/retune_<名字>_<网格>.txt / retune_compare_<名字>_vs_<名字>_<网格>.txt。
数据与 baseline.py 相同：13 部片、runs_initial + runs_test 的录像、音频线索、融合 a1bvid（= app 的 FusionArbiter v2）、
代价 fp=1,fn=1,stop=3；视频结果读 bench/rescored/video_<名字>_runs_*（baseline.py rescore 写的）。
视频判定按 run 里的 v_p_*（4 位小数）用新阈值重判；阈值与 run 文件头相同时直接用 run 里记下的判定。
"""
import argparse
import contextlib
import io
import itertools
import os
import sys
import time
from collections import OrderedDict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fusion_lab as F       # noqa: E402
import arbiter_lab as A      # noqa: E402
import cue_lab as C          # noqa: E402
import video_rescore as V    # noqa: E402
import baseline as B         # noqa: E402

CURRENT = OrderedDict([('VIDEO_DO', 0.65), ('VIDEO_PLOT', 0.60), ('ARB_V_FLOOR', 0.85), ('ARB_BV_CONF', 0.90)])
GRIDS = OrderedDict([
    ('video', OrderedDict([('VIDEO_DO', (0.65, 0.5, 0.8, 0.9)), ('VIDEO_PLOT', (0.60, 0.5, 0.75))])),
    ('conf', OrderedDict([('VIDEO_DO', (0.65, 0.5, 0.8, 0.9)), ('VIDEO_PLOT', (0.60, 0.5, 0.75)),
                          ('ARB_V_FLOOR', (0.85, 0.75, 0.92)), ('ARB_BV_CONF', (0.90, 0.80, 0.95))])),
])
SHORT = OrderedDict([('VIDEO_DO', '转'), ('VIDEO_PLOT', '不转'), ('ARB_V_FLOOR', 'V_FLOOR'), ('ARB_BV_CONF', 'BV_CONF')])


def combos(grid):
    """网格 → [参数 dict]，第 0 组是现用参数。"""
    keys = list(GRIDS[grid])
    return [OrderedDict(CURRENT, **dict(zip(keys, c))) for c in itertools.product(*[GRIDS[grid][k] for k in keys])]


def combo_str(c):
    return ' '.join('%s %.2f' % (SHORT[k], c[k]) for k in CURRENT)


# ======================================================================== 数据
def load(name, seeks=False):
    """name 变体在统一标准下的数据（同 baseline.evaluate）：A.Data，带音频线索。"""
    dirs = B.variant_dirs(name)
    missing = [d for d in dirs if not os.path.isdir(d)]
    if missing:
        sys.exit('找不到变体 %s 的数据：%s（先 baseline.py rescore）' % (name, '、'.join(B.rel(d) for d in missing)))
    with contextlib.redirect_stdout(io.StringIO()):
        d = C.load(B.GT_DIR, ','.join(dirs), ','.join(B.NPZ_DIRS), seeks=seeks, quiet=True, only=','.join(B.FILMS))
    d.has_seeks = seeks
    d.name = name
    return d


def rethreshold(d0, do_th, plot_th, seeks=False, force=False):
    """按新的视频阈值重判每个 tick 的视频动作和置信度（结果年龄 > 2 秒的照旧为空），重建拼接样本和 seek 片段。
    阈值与 run 文件头相同（且 force=False）时沿用 run 里记下的判定。"""
    same = not force and all(abs(v.params['VIDEO_DO'] - do_th) < 1e-9 and abs(v.params['VIDEO_PLOT'] - plot_th) < 1e-9
                             for v in d0.videos)
    if same and d0.has_seeks == seeks:
        return d0
    d = A.Data()
    d.videos, d.coverage, d.has_seeks, d.name = [], getattr(d0, 'coverage', {}), seeks, getattr(d0, 'name', '')
    for v in d0.videos:
        if same:
            nv = v
        else:
            ticks = []
            for t in v.ticks:
                if t.vp is None or t.v_age is None or t.v_age < 0:
                    ticks.append(t)
                    continue
                act, conf = ('', 0.0) if t.v_age > F.MAX_AGE_MS else V.classify(t.vp, do_th, plot_th)
                ticks.append(t.copy(v=act, vc=F.f32(conf)))
            nv = F.Video(v.vid, v.segs, v.dur, ticks, dict(v.params, VIDEO_DO=do_th, VIDEO_PLOT=plot_th),
                         v.gt_path, v.run_path, policy=v.policy)
            for k in ('fusion', 'audio_pipe', 'rescored'):
                setattr(nv, k, getattr(v, k))
        d.videos.append(nv)
    d.splices = dict((v.vid, F.make_splices(v)) for v in d.videos)
    d.prep = dict((v.vid, A.prepare(v)) for v in d.videos)
    d.sprep = dict((v.vid, [A.prepare(s) for s in d.splices[v.vid]]) for v in d.videos)
    d.seeks = dict((v.vid, A.make_seeks(v) if seeks else []) for v in d.videos)
    return d


def make_policy(combo):
    """融合 = a1bvid（app 的 FusionArbiter v2），换上这组的 V_FLOOR / BV_CONF。"""
    cfg = C.make_cfg(**dict(C.PRESETS[B.PRESET], bvc=combo['ARB_BV_CONF']))
    vf = combo['ARB_V_FLOOR']
    return lambda params: C.CuePolicy(dict(params, ARB_V_FLOOR=vf), cfg)


def per_video(d, combo, cost, seeks):
    """每部片单独评测（含它的拼接样本、seek 片段），格式同 cue_lab.per_video（cue_lab.combine 能拼）。"""
    mk = make_policy(combo)
    out = {}
    for v in d.videos:
        rep = A.evaluate(d, mk, cost, seeks=seeks, only={v.vid})
        P = rep['pooled']
        out[v.vid] = dict(
            cost=rep['cost'], cm=dict(P.cm), fs=P.fs, lat=dict((k, list(x)) for k, x in P.lat.items()),
            miss=dict(P.miss), videos=rep['videos'], splice=[(k, vid, sc.cost, sec) for k, vid, sc, sec in rep['splice']],
            seek_do=rep['seek_do'], seek_plot=rep['seek_plot'], lat_stop=rep['lat_stop'])
    return out


def _brief(pv):
    """per_video → {片: (代价, 漏转秒, 误转秒, 误停)}（cue_lab.select_fold 要的格式）。"""
    return dict((vid, (r['cost'], C.fn_seconds(r), C.fp_seconds(r), r['fs'])) for vid, r in pv.items())


_G = {}


def _init(name, grid, cost):
    _G['d0'] = load(name, seeks=False)
    _G['combos'] = combos(grid)
    _G['cost'] = cost


def _task(pair):
    """一对视频阈值：重判一次，把这对阈值下的所有融合参数组合都评测一遍。→ [(组号, {片: 简要结果})]"""
    do_th, plot_th = pair
    d = rethreshold(_G['d0'], do_th, plot_th)
    out = []
    for i, c in enumerate(_G['combos']):
        if (c['VIDEO_DO'], c['VIDEO_PLOT']) == pair:
            out.append((i, _brief(per_video(d, c, _G['cost'], seeks=False))))
    return out


def grid_eval(name, grid, cost, jobs=1, log=print):
    cs = combos(grid)
    pairs = list(OrderedDict(((c['VIDEO_DO'], c['VIDEO_PLOT']), 1) for c in cs))
    res = [None] * len(cs)
    t0 = time.time()
    if jobs > 1:
        import multiprocessing as mp
        with mp.Pool(min(jobs, len(pairs)), initializer=_init, initargs=(name, grid, cost)) as pool:
            for k, out in enumerate(pool.imap_unordered(_task, pairs)):
                for i, r in out:
                    res[i] = r
                log('  … %d/%d 对视频阈值（%.0f 秒）' % (k + 1, len(pairs), time.time() - t0))
    else:
        _init(name, grid, cost)
        for k, pair in enumerate(pairs):
            for i, r in _task(pair):
                res[i] = r
            log('  … %d/%d 对视频阈值（%.0f 秒）' % (k + 1, len(pairs), time.time() - t0))
    return cs, res


class SeekCache(object):
    """按需算某一组参数在每部片上带 seek 片段的完整结果（只给每折排在前面的候选算）。"""

    def __init__(self, d0, cs, cost):
        self.d0, self.cs, self.cost = d0, cs, cost
        self.cache, self.data = {}, {}

    def get(self, i):
        if i not in self.cache:
            c = self.cs[i]
            key = (c['VIDEO_DO'], c['VIDEO_PLOT'])
            if key not in self.data:
                self.data[key] = rethreshold(self.d0, key[0], key[1], seeks=True)
            self.cache[i] = per_video(self.data[key], c, self.cost, seeks=True)
        return self.cache[i]


# ======================================================================== 留一视频
def retune(name, grid='conf', cost=None, jobs=1, log=print):
    cost = cost or F.Cost.parse(B.COST)
    log('重调 %s：网格 %s（%d 组），%d 进程 …' % (name, grid, len(combos(grid)), jobs))
    cs, res = grid_eval(name, grid, cost, jobs, log)
    d0 = load(name, seeks=False)
    vids = [v.vid for v in d0.videos]
    seeks = SeekCache(d0, cs, cost)
    chosen = {}
    for vid in vids:
        train = [v for v in vids if v != vid]
        i, _ = C.select_fold(cs, res, train, seeks)
        chosen[vid] = i
    picks = dict((vid, seeks.get(chosen[vid])[vid]) for vid in vids)
    rep = C.combine(picks)
    base = seeks.get(0)
    brep = C.combine(base)
    totals = [sum(r[v][0] for v in vids) if r else float('inf') for r in res]
    best = min(range(len(cs)), key=lambda i: (totals[i], i))
    return dict(name=name, grid=grid, combos=cs, res=res, vids=vids, chosen=chosen,
                rep=rep, sum=A.summary(rep), pv=dict((v, picks[v]['cost']) for v in vids),
                base_rep=brep, base_sum=A.summary(brep), base_pv=dict((v, base[v]['cost']) for v in vids),
                insample=best, insample_cost=totals[best], current_cost=totals[0])


def _line(s):
    return ('代价 %.0f，准确率 %s，误转 %s，漏转 %s，误停 %d 次，seek 进动作段 p90 %.1fs（特写/低声音 %.1fs），没启动 %d'
            % (s['cost'], A._pct(s['acc']).strip(), A._pct(s['fp']).strip(), A._pct(s['fn']).strip(), s['stops'],
               s['seek_p90'], s['hard_p90'], s['seek_never']))


def report_run(r):
    cs = r['combos']
    out = ['══ 重调：%s（网格 %s，%d 组；%s）══' % (r['name'], r['grid'], len(cs), '、'.join(GRIDS[r['grid']])),
           '  现用参数（%s）：%s' % (combo_str(cs[0]), _line(r['base_sum'])),
           '  留一视频选参：%s' % _line(r['sum'])]
    cnt = {}
    for i in r['chosen'].values():
        cnt[i] = cnt.get(i, 0) + 1
    out.append('    各折选中：' + '；'.join('%d 折 · %s' % (n, '现用参数' if i == 0 else combo_str(cs[i]))
                                     for i, n in sorted(cnt.items(), key=lambda kv: -kv[1])))
    out.append('  13 部一起选（样本内，只作参照，不含 seek 约束）：代价 %.0f（现用 %.0f），%s'
               % (r['insample_cost'], r['current_cost'], combo_str(cs[r['insample']])))
    out += ['', '  逐片代价（片子本身 + 它的拼接样本）：现用参数 → 留一选参']
    for vid in r['vids']:
        b, n = r['base_pv'][vid], r['pv'][vid]
        out.append('    %s  %7.1f → %7.1f  (%+.1f)%s' % (vid, b, n, n - b,
                                                       '' if r['chosen'][vid] == 0 else '   ' + combo_str(cs[r['chosen'][vid]])))
    ok, lines = C.acceptance(r['pv'], r['base_pv'], r['base_rep']['cost'], r['rep']['cost'], r['sum'], r['base_sum'])
    out += ['', '  验收线（留一选参 vs 现用参数，同一个模型）：'] + ['    ' + x for x in lines]
    out.append('    结论：%s' % ('重调有用（全部通过）' if ok else '重调不值得（没过验收线），用现用参数'))
    return '\n'.join(out) + '\n'


def report_compare(ra, rb):
    out = ['══ 各自留一选参后比较：%s（参照）vs %s（网格 %s）══' % (ra['name'], rb['name'], ra['grid']),
           '  %s 现用参数：%s' % (ra['name'], _line(ra['base_sum'])),
           '  %s 现用参数：%s' % (rb['name'], _line(rb['base_sum'])),
           '  %s 留一选参：%s' % (ra['name'], _line(ra['sum'])),
           '  %s 留一选参：%s' % (rb['name'], _line(rb['sum'])),
           '', '  逐片代价（各自留一选参）：%s → %s' % (ra['name'], rb['name'])]
    for vid in ra['vids']:
        if vid in rb['pv']:
            out.append('    %s  %7.1f → %7.1f  (%+.1f)' % (vid, ra['pv'][vid], rb['pv'][vid], rb['pv'][vid] - ra['pv'][vid]))
    ok, lines = C.acceptance(rb['pv'], ra['pv'], ra['rep']['cost'], rb['rep']['cost'], rb['sum'], ra['sum'])
    out += ['', '  验收线（%s 对 %s，两边都是留一选参）：' % (rb['name'], ra['name'])] + ['    ' + x for x in lines]
    out.append('    结论：%s' % ('%s 全部通过' % rb['name'] if ok else '%s 没过验收线' % rb['name']))
    return '\n'.join(out) + '\n'


def cmd_run(a):
    r = retune(a.name, a.grid, jobs=a.jobs)
    text = report_run(r)
    print(text, end='')
    path = os.path.join(B.OUT_DIR, 'retune_%s_%s.txt' % (a.name, a.grid))
    B._write(path, text)
    print('报告：%s' % B.rel(path))
    return 0


def cmd_compare(a):
    ra = retune(a.names[0], a.grid, jobs=a.jobs)
    rb = retune(a.names[1], a.grid, jobs=a.jobs)
    text = report_run(ra) + '\n' + report_run(rb) + '\n' + report_compare(ra, rb)
    print(text, end='')
    path = os.path.join(B.OUT_DIR, 'retune_compare_%s_vs_%s_%s.txt' % (a.names[0], a.names[1], a.grid))
    B._write(path, text)
    print('报告：%s' % B.rel(path))
    return 0


# ======================================================================== 自检
def selftest():
    fails = []

    def check(ok, msg):
        print('  %s %s' % ('ok  ' if ok else 'FAIL', msg), flush=True)
        if not ok:
            fails.append(msg)

    cv, cc = combos('video'), combos('conf')
    check(len(cv) == 12 and len(cc) == 108 and cv[0] == CURRENT and cc[0] == CURRENT
          and len(set(combo_str(c) for c in cc)) == 108, '网格：12 / 108 组，第 0 组是现用参数')
    if not all(os.path.isdir(d) for d in B.variant_dirs(B.DEFAULT_PIPE)):
        print('  skip 真实数据（bench/rescored 下没有 exact_crop）')
    else:
        cost = F.Cost.parse(B.COST)
        d0 = load(B.DEFAULT_PIPE)
        ref = B.evaluate(B.DEFAULT_PIPE)['e2e']['cost']
        tot = sum(r['cost'] for r in per_video(d0, CURRENT, cost, seeks=False).values())
        check(abs(tot - ref) < 1e-6, '现用参数的回放与 baseline.py eval exact_crop 相同（代价 %.1f / %.1f）' % (tot, ref))
        d1 = rethreshold(d0, 0.65, 0.60, force=True)
        n = diff = 0
        for v0, v1 in zip(d0.videos, d1.videos):
            for t0, t1 in zip(v0.ticks, v1.ticks):
                n += 1
                diff += t0.v != t1.v
        tot1 = sum(r['cost'] for r in per_video(d1, CURRENT, cost, seeks=False).values())
        check(diff <= 0.001 * n and abs(tot1 - ref) <= 5,
              '按 v_p_*（4 位小数）重判视频动作：与 run 里记下的判定差 %d/%d 个 tick，代价 %.1f（%.1f）' % (diff, n, tot1, ref))
        d2 = rethreshold(d0, 0.9, 0.5)
        dd = sum(1 for v in d2.videos for t in v.ticks if t.v == 'do')
        d00 = sum(1 for v in d0.videos for t in v.ticks if t.v == 'do')
        check(dd < d00, '提高转的阈值，判"转"的 tick 变少（%d → %d）' % (d00, dd))
        t0 = time.time()
        r = retune(B.DEFAULT_PIPE, 'video', cost, jobs=min(4, os.cpu_count() or 1), log=lambda *a: None)
        text = report_run(r)
        check(abs(r['base_rep']['cost'] - ref) < 1e-6 and r['rep']['cost'] == r['rep']['cost'] and '验收线' in text,
              'exact_crop 只调视频阈值：现用 %.0f，留一选参 %.0f，样本内 %.0f（%.0f 秒；2026-10-15 的结果是 1219 / 1219 / 1218）'
              % (r['base_rep']['cost'], r['rep']['cost'], r['insample_cost'], time.time() - t0))
    if fails:
        print('selftest: %d 项失败' % len(fails))
        return 1
    print('selftest: ALL PASS')
    return 0


def main():
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(errors='replace')
        except Exception:
            pass
    if '--selftest' in sys.argv:
        return selftest()
    ap = argparse.ArgumentParser(description='重调视频阈值和与视频置信度有关的融合参数（详见文件头）',
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd')
    p = sub.add_parser('run', help='一个变体：现用参数 vs 留一选参')
    p.add_argument('name')
    p2 = sub.add_parser('compare', help='两个变体各自留一选参后比较（第一个是参照）')
    p2.add_argument('names', nargs=2)
    for q in (p, p2):
        q.add_argument('--grid', default='conf', choices=list(GRIDS))
        q.add_argument('--jobs', type=int, default=max(1, min(4, (os.cpu_count() or 2) - 1)), help='进程数')
    a = ap.parse_args()
    if a.cmd == 'run':
        return cmd_run(a)
    if a.cmd == 'compare':
        return cmd_compare(a)
    ap.print_help()
    return 2


if __name__ == '__main__':
    sys.exit(main())
