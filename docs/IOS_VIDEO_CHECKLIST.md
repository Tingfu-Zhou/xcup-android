# iOS 视频输入检查清单

对照 Android 端 2026-10-14 的修复（分支 `claude/gallant-dijkstra-tii123`）。iOS 版是从 Android 移植的，没有网页视频模式；离线模式、在线模式、视频模型和融合应当与 Android 一致，所以 Android 上的问题 iOS 很可能也有。

逐条查完，把结果填进第 8 节的表。最快的判断方法在 3.2 节：加两行日志，播 30 秒就能看出来。

---

## 1. Android 发现并修复的问题

| # | 问题 | 影响的模式 | 证据 | 修复 |
|---|---|---|---|---|
| A | **只取到关键帧**：`seekTo(T, SEEK_TO_CLOSEST_SYNC)` 之后直接拿解码器吐出的第一帧，拿到的是离 T 最近的关键帧（可能在 T 之后） | 离线 | bench 的 v08–v13：视频输出每个 GOP（8–12 秒）才变一次，每次冻住 4.5–8.5 秒。用“最近关键帧”模拟能复现真机的 `v_p_*`（6 部片 MAE 0.007–0.021，判定一致 95–99%） | 顺序解码，按 pts 取离播放位置最近的那一帧 |
| B | **比画面晚 0.5–1.3 秒**：每次先 seek、只喂一帧就取最早的一个输出，解码器里积压的是前几次请求的帧 | 离线 | 从“冻住→变化”的时间点推出：v10 0.5 秒、v08/v11 0.9 秒、v09 1.3 秒 | 同上：不再每次 seek，按 pts 取帧 |
| C | **预处理与训练不一致**：整帧压扁成 160×160（16:9 横向压扁 1.78 倍，训练时裁掉的左右两边也送进了模型）；离线模式 GPU 从原分辨率一步缩到 160，没有抗混叠 | 离线、网页、在线 | 对照训练脚本 `make_movinet_clips.py` / `train_mobilenetv3_tap_3class.py` | 新增 `FramePreprocessor`：居中裁正方形 → 面积平均缩到 224 → 再缩到 160 |
| D | **抽帧在主线程**：每 250ms 占约 43ms，渲染后 `sleep(20)` 赌帧已经到了 | 离线 | 代码 | 单独的抽帧线程；等帧到达并核对时间戳 |

A、B 只影响离线模式。网页模式截的是 TextureView 当前显示的帧，在线模式截的是屏幕，取到的都是当前画面。C 三种模式都有。

Android 端改动的文件：

- `VideoFrameExtractor.java`（重写）：顺序解码。只在第一次、seek 之后、播放位置往回跳或往前跳太远时，才 `SEEK_TO_PREVIOUS_SYNC` + flush。pts 不到目标的帧丢掉、不渲染。每次最多花 150ms，seek 进长 GOP 时分几次追上，追赶中不拿离目标太远的帧凑数。
- `EGLRenderer.java`（重写）：用 SurfaceTexture 的变换矩阵（含旋转、裁边），在 GPU 上只画中间的正方形，画成 672×672；等 `onFrameAvailable` 并核对时间戳。
- `FramePreprocessor.java`（新增，纯 Java）：居中裁剪、面积平均缩小（与 `cv2.INTER_AREA` 同一算法），三种模式共用。`CENTER_CROP=false` 可改回整帧压扁，做 A/B 对比用。
- `VideoProcessActivity.java`：抽帧线程；seek 时给帧编号加一，作废在途的帧；同一帧不重复放进窗口；网页模式改成截短边 448 的图再裁剪缩放。
- `OnlineAnalysisActivity.java`：截屏帧改用 `FramePreprocessor`（居中裁剪 + 面积缩小）。
- `BenchLogger.java`：参数行加 `VIDEO_PIPE`；新增 `v_pos`、`v_fpts` 两列（窗口里 12 帧各自的 pts）。
- `bench/tools/video_rescore.py`（新增）：离线视频重算，能判断某份 run 用的是哪种抽帧和预处理。

---

## 2. 模型要的输入（以训练为准）

训练素材的处理（金标准）：

1. 每个 3 秒片段由 ffmpeg 处理：`scale=224:224:force_original_aspect_ratio=increase,crop=224:224`，即**短边缩到 224，居中裁 224×224**；输出 12fps、x264 编码。
2. 训练读帧：OpenCV 读出 BGR → 转 RGB → `cv2.resize(..., (160,160), interpolation=cv2.INTER_AREA)`；在 3 秒里均匀取 12 帧（相邻约 0.27 秒）。

所以 app 喂给模型的应当是：

| 项 | 要求 |
|---|---|
| 张量 | `[1, 12, 160, 160, 3]`，float32，NHWC |
| 12 帧 | 最近 3 秒、每 250ms 一帧、**12 个不同时刻**，从旧到新 |
| 每帧画面 | 先转正（应用旋转），再**居中裁正方形**（边长 = 短边），**抗混叠缩小**到 160×160（面积平均最接近训练） |
| 通道顺序 | **RGB** |
| 数值 | **0–255 原始像素**，不除以 255、不减均值（模型里自带 Rescaling 层） |
| 输出 | softmax `[plot, oral, sex]` |

模型结构：12 帧各自过 MobileNetV3Small，得到 576 维特征后取平均，再接一个线性层。所以帧的顺序目前不影响结果，但 12 帧必须是 12 个不同时刻：12 帧都是同一张图，就等于只看了一帧。

---

## 3. 离线模式抽帧（对应问题 A、B、D）

### 3.1 先看 iOS 用的是哪种 API

| API | 风险 | 要查的 |
|---|---|---|
| `AVAssetImageGenerator` | **很可能是问题 A**。`requestedTimeToleranceBefore/After` 默认是 `.positiveInfinity`，这时取到的是离请求时刻最近的关键帧，和 Android 的 `SEEK_TO_CLOSEST_SYNC` 一样 | 两个容差有没有设成 `.zero`（或很小，如 1/60 秒）？返回的 `actualTime` 与请求时刻差多少？`appliesPreferredTrackTransform` 是否为 `true`（默认 false，竖拍视频会横着）？`maximumSize` 之后又怎么缩放的？ |
| `AVPlayerItemVideoOutput` | 推荐。`copyPixelBuffer(forItemTime:itemTimeForDisplay:)` 拿到的就是该时刻显示的那一帧，等同 Android 网页模式的 TextureView | 时刻是否用 `itemTime(forHostTime: CACurrentMediaTime())`；`hasNewPixelBuffer` 为 false 时（暂停、卡顿）不要把同一帧重复放进窗口；`pixelBufferAttributes` 里不要指定宽高（让 AVFoundation 缩放就不知道用了什么滤波），拿原尺寸自己裁剪缩放 |
| `AVAssetReader` | 顺序解码，帧是准的 | 是否每次取帧都重建 reader（很慢）；reader 是否与播放同步（跑太快或落后）；用的是不是 sample 的 PTS |
| 自己用 VideoToolbox 解码 | 若写成“先 seek 再取第一帧”，就是 Android 的老问题（A + B） | 同 Android 的修法：顺序解码，按 PTS 选帧 |

如果用的是 `AVAssetImageGenerator` 且容差是默认值，问题 A 基本可以确定。把容差设成 `.zero` 能拿到准确的帧，但每次都要从前一个关键帧解起：12 秒的 GOP 每次最多要解 360 帧，实时每秒 4 帧大概跟不上。更好的办法是换成 `AVPlayerItemVideoOutput`。

### 3.2 加日志验证（10 分钟）

1. 每取一帧，记下**请求时刻**（`player.currentTime()`，毫秒）和**这一帧的实际时刻**（`actualTime` / `itemTimeForDisplay` / sample PTS）：
   - 正常：两者相差不超过 1 帧（约 33ms），相邻两帧的实际时刻相差约 250ms。
   - 问题 A：实际时刻一段时间不变，然后跳一个 GOP（4–12 秒）。
   - 问题 B：实际时刻始终比请求时刻晚 0.5 秒以上。
2. 每个视频结果把三类概率按**全精度**记下来（`String(format: "%.9g", p)`），数一数“和上一个结果完全相同”的比例：

```python
# log.csv 至少包含 pos_ms, p_plot, p_oral, p_sex 四列，每个视频结果一行
import pandas as pd, numpy as np
d = pd.read_csv('log.csv')
p = d[['p_plot', 'p_oral', 'p_sex']].to_numpy()
same = np.all(p[1:] == p[:-1], axis=1)
unsat = p[1:].max(axis=1) < 0.99            # 饱和的输出（≈1.0）本来就可能相同，不算
print('没饱和的结果里和上一个完全相同的比例: %.1f%%' % (100 * same[unsat].mean()))
```

Android 旧版在 GOP 长的片子上这个比例是 48–72%，正常应接近 0%。

### 3.3 线程和 seek

- 抽帧、解码、预处理不要在主线程。
- seek 之后清空帧窗口；seek 之前请求、之后才送到的帧要丢掉（Android 用一个 seek 时加一的编号）。
- 同一帧（时刻相同）不要重复放进窗口。

---

## 4. 预处理（对应问题 C；离线和在线模式都要查）

逐项对照第 2 节：

1. **几何**：有没有先居中裁正方形？
   - 错误写法（压扁）：直接 `UIGraphicsImageRenderer(size: 160×160)` + `draw(in: 160×160)`；`CIImage.transformed(by: scale(160/w, 160/h))`；`vImageScale` 直接缩到 160×160。
   - 错误写法（留黑边）：`AVMakeRect(aspectRatio:insideRect:)` 等比缩小放进 160×160。
   - 正确写法：先 `cropped(to:)` 取中间 `min(w,h)` 的正方形，再等比缩放。裁剪区域 = `x0 = (w - side) / 2, y0 = (h - side) / 2`。
2. **缩小滤波**要抗混叠：1080p → 160 缩小了 6.75 倍，只取几个像素插值会混叠。
   - 推荐：分两步，先缩到 224 再缩到 160，都用面积平均；或用 `vImageScale_ARGB8888`（Lanczos，抗混叠）、Core Image 的 `CILanczosScaleTransform`。
   - 慎用：单纯的双线性或仿射变换一步缩到 160。
3. **通道顺序**：AVFoundation 和 ReplayKit 给的 `CVPixelBuffer` 常是 `kCVPixelFormatType_32BGRA`（内存顺序 B、G、R、A）。填张量时必须按 **R、G、B** 顺序取。R/B 对调是移植时最常见的错误，要专门查。
4. **数值**：0–255 的 float，不除以 255、不做 `(x-127.5)/127.5`。
5. **旋转**：竖拍视频要转正（`preferredTransform` / `appliesPreferredTrackTransform`）。
6. **颜色**：
   - YUV→RGB 的矩阵：训练时 OpenCV 用 BT.601；AVFoundation 对高清视频一般用 BT.709。Android 也是这样，这次没改。差别不大，后面会用 `video_rescore.py` 量一下。
   - 全范围（full range）/ 视频范围（video range）不要弄错：`420f` 是全范围，`420v` 是视频范围，弄错了画面对比度会明显不对。
7. **如果 iOS 用的是 Core ML 而不是 TFLite**：查转换时 `ImageType` 的 `color_layout`（RGB/BGR）以及 `scale`/`bias`。模型里已经有 Rescaling，`scale` 应为 1、`bias` 应为 0。

### 4.1 用 Android 的对拍数据验证

`app/src/test/resources/video/preprocess_golden.txt` 是 `FramePreprocessor` 的标准答案，Java 和 Python 两版都与它逐位相同。iOS 照 `FramePreprocessor.java`（约 150 行纯计算）移植一份，再用这个文件写单元测试：

- 每个 `case` 行：`case <种类> <图案> <种子> <w> <h> <x0> <y0> <cw> <ch> <dw> <dh>`，后面跟 dh 行期望输出，每个像素 6 位十六进制 `RRGGBB`。
- 输入图由 `FramePreprocessorTest.goldenImage()` 生成：`pattern` 是 `r=(x*7+y*3)&255, g=(x*x+y)&255, b=((x^y)*5)&255`；`noise` 是 64 位 LCG（`s = s*6364136223846793005 + 1442695040888963407`，取 `s >>> 56`，每个像素依次取 R、G、B）。
- `resize` 种类：调用 `resize(src, stride=w, x0, y0, cw, ch, dw, dh)`；`model` 种类：调用 `toModelInput(src, w, h, 区域)`，整张图时区域为空。

再用一帧真实画面对比：同一部片、同一时刻，导出 iOS 的 160×160 输入（PNG），和 Android 的对比（或用 `video_rescore.py` 里的 `prep_frame` 算出期望值）。应当一样是画面中间的正方形、不变形、颜色一致。

---

## 5. 在线模式（如果 iOS 用 ReplayKit 广播扩展截屏）

- 截到的是**整个屏幕**（含状态栏、播放器控件、黑边）。整屏压扁成 160×160 是错的；至少要居中裁正方形（Android 现在就是这么做的）。更好的做法是检测视频画面所在的区域，只裁那一块，这一点 Android 也还没做。
- 视频帧通常是 YUV（`420f`/`420v`），转 RGB 时注意矩阵和范围（见 4.6）。
- 用 sample buffer 的 PTS 作为帧时刻；每秒取 4 帧即可，多余的丢掉。
- 广播扩展有内存上限（约 50MB）：缓冲区要复用，不要每帧分配大图。

---

## 6. 让 iOS 也能用 bench 工具（建议）

如果 iOS 在调试包里写出与 Android `BenchLogger` **同格式**的 CSV，bench 的工具（`metrics.py`、`fusion_lab.py`、`video_rescore.py`）就能直接拿来评 iOS：

- 文件头、表头、列的顺序和格式照 `BenchLogger.java`（当前 38 列）。列名清单见 `bench/tools/metrics.py` 的 `RUN_COLS_NEEDED`。
- 参数行写清楚 iOS 用的抽帧和预处理，例如 `VIDEO_PIPE=exact_crop`；`v_fpts` 写窗口里 12 帧各自的时刻（ms，分号隔开）。
- 写出来之后，`python bench/tools/video_rescore.py check --runs <iOS 的 run 目录>` 会告诉你 iOS 和哪条管线对得上：`keyframe_legacy` 说明有问题 A，`exact_crop` 说明对了。

---

## 7. 顺便复查：之前音频那边的修复

2026-10-05 的音频修复 iOS 也要对齐（Android 现在固定走 `PcmResampler`，评测 CSV 参数行是 `AUDIO_PIPE=fixed`），复查一遍：

- 双声道 PCM 先下混成单声道（左右取平均），不能把交错立体声当单声道（会变成半速、低八度）；
- 重采样到 16kHz 要抗混叠，跨块连续处理，块边界不丢样本；
- 节律、线索、融合参数与 Android 一致。

---

## 8. 检查结果

| 项 | 结果 |
|---|---|
| 离线抽帧用的 API |  |
| 容差 / 是否精确到帧 |  |
| 帧的实际时刻 − 请求时刻（中位 / 最大） |  |
| 没饱和的结果里与上一个完全相同的比例 |  |
| 抽帧、预处理所在线程 |  |
| seek 后是否清窗口、丢在途帧 |  |
| 几何：居中裁剪 / 压扁 / 留黑边 |  |
| 缩小滤波 |  |
| 通道顺序：RGB / BGR |  |
| 数值范围 |  |
| 旋转 |  |
| YUV→RGB 矩阵、范围 |  |
| 在线模式的几何与像素格式 |  |
| 对拍数据（preprocess_golden.txt）是否通过 |  |
