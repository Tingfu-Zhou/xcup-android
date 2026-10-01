# bench/ — Xcup 评测基准

离线回放评测流水线。目标：每改一个参数都能量化对比，而不是凭感觉调。

## 目录

```
bench/
  ANNOTATION_SPEC.md      人工真值标注规范（改动需递增 spec_version）
  gt/<video_id>.csv       人工真值（提交进 git）
  tools/
    labeler.html          真值标注器，双击即用，纯本地无上传
    test_labeler.js       标注器核心逻辑测试：node bench/tools/test_labeler.js
```

## 标注

1. 浏览器打开 `bench/tools/labeler.html`（`file://` 直接打开即可，无需起服务）
2. 打开视频（或直接拖进窗口）
3. 按 `1`~`5` 在播放头处打点，按三遍法标注，详见
   [ANNOTATION_SPEC.md](ANNOTATION_SPEC.md) §6
4. 右侧「自检 / 告警」全部清空后，导出 CSV 放进 `bench/gt/`

视频**不上传**任何地方：标注器用 `URL.createObjectURL` 在本地读取，
标注进度暂存在浏览器 localStorage（按文件名 + 文件大小作 key），刷新不丢。
换机器前记得导出 CSV。

浏览器放不出来（H.265/HEVC 等）时先转码，**不要裁剪**：

```bash
ffmpeg -i in.mp4 -c:v libx264 -crf 20 -c:a aac -movflags +faststart out.mp4
```

## 采集 + 出指标

不另做测试 app。采集直接复用线上代码：`BenchLogger` 把每个融合 tick 写成一行 CSV，
`metrics.py` 和真值对齐算指标。两条时间线都用**视频播放位置 pos_ms**，不需要任何标定。

### 1. 采集（手机）

装 **debug 包**（`BenchLogger` 自动开启；release 包把 `FORCE_ENABLE` 改成 true）。

1. 进入离线视频分析，选同一个视频文件
2. **从头不间断播到尾**：不要 seek、不要暂停、不要切后台
3. 退出页面时会弹 Toast 告知文件路径

蓝牙不用连也能采：dump 挂在"决定下发指令"处，不在"BLE 已连接"分支里。

输出：`/sdcard/Android/data/com.example.helloworld/files/bench/run_<视频名>_<时间>.csv`
（应用专属外部目录，无需任何存储权限，数据线/文件管理器直接拷）

> seek 会清空滑窗与音频缓冲并重置状态机，这一轮的延迟数字就没意义了。
> `metrics.py` 会检测 pos 回退并告警。

### 2. 出指标（电脑）

```bash
python3 bench/tools/metrics.py bench/gt/v01.csv run_v01_20261001_143022.csv
```

只给真值则仅做格式校验与统计：

```bash
python3 bench/tools/metrics.py bench/gt/v01.csv
```

常用参数：

| 参数 | 作用 |
|---|---|
| `--policy exclude\|do\|noise` | `other_adult` 怎么算，默认 `exclude`（不计入） |
| `--append bench/summary.csv` | 追加一行，便于跨参数/跨视频对比 |
| `--selftest` | 自检脚本本身（含与 BenchLogger 的 CSV 列名契约检查） |

run CSV 的文件头记录了当次的全部可调参数（`benchParamsLine()`），所以任何一份 dump
都能追溯到是哪组参数跑出来的。

### 关于 dump 存的是原始概率

`v_p_*` / `a_p_*` 存的是模型原始 softmax，不是判定结果。这样以后想做「秒级回放 + 参数
扫描」时，这些 CSV 就是现成的输入，不用重新跑模型——第一步不是弯路。

## 尚未实现（等真有需要再做）

- `DecisionCore.java` — 把融合/仲裁/蓝牙状态机抽成纯 Java（注入时钟、参数化），
  配一个 JVM 回放器。做完才能秒级重放 + 批量扫参，代价是一次重构。
  现在每改一个参数要重跑一遍视频（实时），片子不多时可以接受。
