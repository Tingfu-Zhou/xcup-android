#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
training/eval_clips.py — 在训练素材的某个 split 上评测 12 帧视频模型（tflite），几个模型并排

和 train_video.py 用同一套读帧（12 帧均匀取、不增强）和指标，给新模型的 val / test 数字一个参照，最常用的是把
app 现用的模型也评一遍。注意：现用模型当初是按这份 val 集选的 checkpoint，它在 val 上的数字偏乐观，比 test。
这些数字只是训练素材上的参考，比模型以 bench 的 13 部片为准（baseline.py / frame_diag.py）。

另外按片子列出准确率最低的几部（--worst，默认 10）：整部几乎全错的，先人工看一眼是不是标错、混进了别的内容。

用法（仓库根目录）
  python3 training/eval_clips.py --data_dir ~/ml/movinet/movinet_clips --split test \\
      --model app/src/main/assets/mobilenetv3small_tap_3class_float32.tflite --model runs/new1/new1.tflite
"""
import argparse
import io
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import train_utils as U     # noqa: E402
import numpy as np          # noqa: E402


class TFLiteModel(object):
    """12 帧 tflite → predict_on_batch（逐个片段推理）。"""

    def __init__(self, path, threads=4):
        self.it = U._interp(path, threads)
        self.inp = self.it.get_input_details()[0]['index']
        self.out = self.it.get_output_details()[0]['index']

    def predict_on_batch(self, x):
        x = np.asarray(x, np.float32)
        out = []
        for xi in x:
            self.it.set_tensor(self.inp, xi[None])
            self.it.invoke()
            out.append(self.it.get_tensor(self.out)[0].copy())
        return np.stack(out)


def main(argv=None):
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(errors='replace')
        except Exception:                            # noqa: BLE001
            pass
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data_dir', required=True)
    ap.add_argument('--split', default='test', choices=('train', 'val', 'test'))
    ap.add_argument('--model', action='append', required=True, help='12 帧 tflite（可多次）')
    ap.add_argument('--bad_list', default='auto', help='默认 <data_dir>/bad_clips.txt；none = 不剔除')
    ap.add_argument('--max_clips', type=int, help='每类最多评多少个片段（快速看一眼用）')
    ap.add_argument('--workers', type=int, default=max(1, min(8, (os.cpu_count() or 2) - 1)))
    ap.add_argument('--threads', type=int, default=4, help='TFLite 线程数')
    ap.add_argument('--worst', type=int, default=10, help='每个模型列出准确率最低的几部片（默认 10，0 = 不列）')
    ap.add_argument('--out', help='结果另存为 JSON')
    a = ap.parse_args(argv)
    data_dir = os.path.abspath(os.path.expanduser(a.data_dir))
    bad = os.path.join(data_dir, 'bad_clips.txt') if a.bad_list == 'auto' else (None if a.bad_list == 'none'
                                                                                 else a.bad_list)
    if bad and not os.path.exists(bad):
        print('[WARN] 没有 %s：坏片段没剔除（先跑 data_check.py scan）' % bad)
        bad = None
    s = U.collect_samples(data_dir, a.split, bad)
    if a.max_clips:
        s = s.subset(U.balanced_pick(s, a.max_clips * len(s.classes)))
    print('[数据] ' + s.describe())
    keys = s.film_keys()
    results = []
    for path in a.model:
        t0 = time.time()
        st = U.tflite_structure(path)
        if any('输入' in p for p in st['problems']):
            sys.exit('%s：%s' % (path, '；'.join(st['problems'])))
        probs, ok = U.predict(TFLiteModel(path, a.threads), U.eval_dataset(s, U.EVAL_FRAMES, 16, U.IMAGE_SIZE,
                                                                            a.workers))
        m = U.clip_metrics(probs[ok], s.labels[ok], keys[ok])
        m['n_bad'] = int((~ok).sum())
        m['model'] = path
        results.append(m)
        print('\n%s（%.0f 秒）\n  %s' % (path, time.time() - t0, U.fmt_metrics(m, a.split)))
        print('  混淆矩阵（行 = 真值 %s，列 = 预测）：%s%s' % ('/'.join(U.CLASS_NAMES), m['cm'],
                                                    '  ⚠ %d 个片段读不出' % m['n_bad'] if m['n_bad'] else ''))
        if a.worst:
            m['worst_films'] = U.worst_films(probs[ok], s.labels[ok], keys[ok], a.worst)
            print(U.fmt_worst(m['worst_films'], '%s 里最差的 %d 部' % (a.split, a.worst)))
    if len(results) > 1:
        print('\n══ 并排（%s，%d 个片段）══' % (a.split, results[0]['n']))
        rows = [('AUC(p_do)', 'auc_do', '%.4f'), ('准确率', 'acc', 'pct'), ('按片子平均的准确率', 'film_acc', 'pct'),
                ('最差的片子', 'film_acc_worst', 'pct'), ('loss', 'loss', '%.3f'), ('剧情判成转（app 阈值）', 'app_fp', 'pct'),
                ('动作判成不转', 'app_fn', 'pct'), ('unclear', 'app_unclear', 'pct'), ('置信度 ≥0.99', 'conf99', 'pct')]
        names = [os.path.splitext(os.path.basename(m['model']))[0] for m in results]
        w = max(14, max(U.D.width(n) for n in names) + 3)
        print('  ' + U.D.lpad('', 24) + ''.join(U.D.rpad(n, w) for n in names))
        for lab, k, fmt in rows:
            cells = [('%.1f%%' % (100 * m[k]) if fmt == 'pct' else fmt % m[k]) if k in m else '—' for m in results]
            print('  ' + U.D.lpad(lab, 24) + ''.join(U.D.rpad(c, w) for c in cells))
    if a.out:
        with io.open(a.out, 'w', encoding='utf-8') as fh:
            json.dump(results, fh, ensure_ascii=False, indent=1, default=str)
    return 0


if __name__ == '__main__':
    sys.exit(main())
