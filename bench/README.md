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

## 尚未实现（后续）

- `tools/dump_perception.py` — ffmpeg 抽帧/抽音 + 两个 TFLite 模型，产出 `traces/<id>/perception.csv`（原始 probs）
- `DecisionCore.java` — 从 `VideoProcessActivity` / `OnlineAnalysisActivity` 抽出的纯 Java 决策逻辑（注入时钟，参数化）
- `ReplayMain.java` — JVM 回放器，`perception.csv` + PCM → `decisions.csv` / `ble_events.csv`
- `tools/metrics.py` — 逐秒准确率、切换延迟（感知/融合/状态机三段分解）、抖动
- `tools/compare.py` / `tools/sweep.py` — 回归对比与参数扫描
