# bench/ — Xcup 评测基准

离线回放评测流水线。目标：每改一个参数都能量化对比，而不是凭感觉调。

## 目录

```
bench/
  ANNOTATION_SPEC.md      人工真值标注规范（改动需递增 spec_version）
  gt/<video_id>.csv       人工真值（提交进 git）
  runs/<video_id>_*.csv   真机 dump（提交进 git：融合配置是从它们拟合出来的，要能复现）
  rescored/               audio_rescore.py 的输出：对齐结果 alignment.json + 离线重算的 run CSV
  videos/                 原始视频（不进 git，放这里命令最短；也可以放别处用 --videos 指定）
  tools/
    labeler.html          真值标注器，双击即用，纯本地无上传
    test_labeler.js       标注器核心逻辑测试：node bench/tools/test_labeler.js
    metrics.py            单部片的指标 / 误判定位 / 模态诊断
    fusion_lab.py         融合层离线实验台：回放、交叉验证、模态上限、拟合配置
    audio_rescore.py      音频离线重打分：从原视频还原真机送进模型的 2 秒音频，换模型/换解码方式重算
    yamnet_class_map.csv  YAMNet 的 AudioSet 521 类名（来自 tensorflow/models，Apache-2.0）
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

### 定位错误

```bash
# 列出误判区间（按时长排序），每段显示当时各层的判定分布
python3 bench/tools/metrics.py gt.csv run.csv --errors --top 15

# 看某个时刻附近的逐 tick 原始数据：两个模型的概率、各决策层输出、蓝牙动作
python3 bench/tools/metrics.py gt.csv run.csv --at 127.8

# 模态诊断：单模态准确率、各路证据的判别力 AUC、阈值扫描
python3 bench/tools/metrics.py gt.csv run.csv --diag
```

`--diag` 的 AUC 是关键：**AUC ≈ 0.5 说明该证据等于掷硬币，调阈值救不了**，必须动模型或
换判据；AUC 高但阈值不合适，才是调参能解决的。

run CSV 的文件头记录了当次的全部可调参数（`benchParamsLine()`），所以任何一份 dump
都能追溯到是哪组参数跑出来的。

### 关于 dump 存的是原始概率

`v_p_*` / `a_p_*` 存的是模型原始 softmax，不是判定结果。这样以后想做「秒级回放 + 参数
扫描」时，这些 CSV 就是现成的输入，不用重新跑模型——第一步不是弯路。

## 融合层实验台 fusion_lab.py

run CSV 里存着两个模型每个 tick 的原始概率，融合层的输入全在。`fusion_lab.py` 把融合层整个
搬到 Python 里重放：改融合/扫参数不用再实时重跑视频，一部片几秒。

先把每部片的 run CSV 拷进 `bench/runs/`，文件名以真值 id 开头（`v01_base.csv` 配 `gt/v01.csv`）。

```bash
# 1. 回放器可信吗：复刻线上 Java 融合（投票窗 + 启动仲裁 + 蓝牙状态机），与 dump 逐 tick 比对
python3 bench/tools/fusion_lab.py check

# 2. 留一视频交叉验证：证据融合 vs 现有融合 vs 一直转 / 只用视频 / 只用音频
python3 bench/tools/fusion_lab.py eval

# 3. 模态上限：假设视频（或音频）完美，端到端最多能到多少 —— 决定先改哪个模型
python3 bench/tools/fusion_lab.py oracle

# 4. 用全部片子拟合，写出 app/src/main/assets/fusion_config.json（默认 mode=shadow）
python3 bench/tools/fusion_lab.py fit
python3 bench/tools/fusion_lab.py fit --mode live      # 确认 eval 结果后，让证据融合驱动蓝牙
```

Windows 下把 `python3` 换成 `python`（或 `py`）、路径用 `\` 也可以。`eval` / `oracle` 的留一拟合默认按 CPU 核数
多进程并行（`--jobs 1` 改回串行，结果完全相同）；参数搜索默认"先粗后细"，`--exhaustive` 改扫全网格
（慢约 9 倍，结果基本一致）。

视频证据的权重 `w_v` 也在搜索范围里（1 / 0.5 / 0.25 / 0，0 = 只用音频）：音频修好之后，剧情里有裸露的片子
上"少信视频"明显更好，固定 `w_v=1` 搜不到。旧数据上它的留一代价 1448，略好于固定 `w_v=1` 的 1492。

`check` 一致率要 ≥99% 才说明回放器可信，后面的对比才有意义。少量不一致来自 CSV 把置信度
存成 4 位小数、以及蓝牙状态机内部取时间比 `wall_ms` 晚几毫秒，正好卡在阈值边界上时会翻转。
live 模式采集的 run（参数行里 `FUSION=live/<配置 id>`）会自动改用证据融合回放来比对。

### 证据融合是什么

原融合把"音频更准、音频权重 1.2、音频可直通启动"写死在代码里；bench 显示这个前提不成立。
证据融合（`EvidenceFusion.java`，Python 端同名类逐 tick 对拍）不写死谁更可信：

```
证据：模型每出一个新结果，取特征 s（视频 p_oral+p_sex；音频 p_sex+p_oral 或 p_sex），
      查表 LLR = log P(s 落在该区间 | 动作) / P(s 落在该区间 | 剧情)       ← 从 bench 统计
累积：L = clamp(L + w_v·LLR_v + w_a·LLR_a, -clamp_lo, +clamp_hi)       ← 每个结果只计一次
决策：不转→转：L ≥ theta_on；转→不转：L ≤ theta_off；之后照旧走蓝牙状态机
```

- 证据表用真实片子统计：模型窗口跨真值边界的结果不用；分 12 格，保序回归后裁到 ±3。
  视频"像剧情"那一半证据可以整体打折（`video_neg_scale`，拟合时在 1 / 0.6 / 0.35 里选）：
  遮挡、特写时视频模型在动作里也判"剧情"，原样的负证据会把马达压停。
- 上下界分开：`clamp_lo` 管剧情里信念压多深（越小，动作开始后启动越快）；`clamp_hi` 管动作里
  攒多少余量（越大越扛得住遮挡/镜头掉线，但动作结束后停得越慢）。
- `w_a`、`clamp_lo/hi`、`theta_on/off`、音频特征、视频负证据折扣按代价一起搜：
  `代价 = 误转秒数×fp + 漏转秒数×fn + 动作中误停次数×stop`，默认 `--cost fp=1,fn=1,stop=3`。
  这是产品取舍，不随模型变；换了模型只需重新 `fit`。
- **拼接样本**：现有片子全是"剧情→动作"，而且动作前几乎都隔着一段 other_adult（不计分）：
  停转切换点一个都没有，干净的启动切换点也只有一两个，不补的话扫参会挑出"永远不停"或
  "启动很慢"的参数。所以用同一部片的动作段和剧情段首尾拼接，造"停转样本"（动作 60s +
  剧情 45s）和"启动样本"（剧情 45s + 动作 60s），一起算进代价。`eval` 报表里的
  "启动后没转 / 停止后还转"就是拼接点之后马达该转没转 / 该停还转的秒数（中位数）。
  拼接处感知是瞬间切换的，比真实片子乐观约 1~2 秒。

### 配置的三种模式（`fusion_config.json` 的 `mode`）

| mode | 行为 |
|---|---|
| `legacy` | 不运行证据融合 |
| `shadow` | 每 tick 计算并写进 run CSV 的 `ev_*` 列，**蓝牙仍由原融合驱动**（默认） |
| `live` | 证据融合驱动蓝牙，原投票窗与启动仲裁不再参与；蓝牙状态机照旧 |

配置缺失或解析失败时 app 自动回退到原融合。想退回原融合，把 `fusion_config.json` 的
`"mode"` 改成 `"shadow"`（照样记证据融合的影子列）或 `"legacy"`，重新编译即可。

配置里的 `"audio_pipeline"` 记录证据表是在哪种音频上拟合的，由 `fit` 按训练数据自动写入，不要手改。
离线模式现在只有修复后的音频（`PcmResampler`，见下文"音频解码修复"），旧的半速处理已删除，所以 app 只接受
`"fixed"` 的配置；没有这一项的旧配置（在半速音频上拟合的）会被拒绝，app 回退到原融合（logcat 里有
`[证据融合]` 的报错）。`EvidenceFusionTest` 和 `fusion_lab.py --selftest` 也会检查打包的配置。

### 当前结果（2026-10-05，6 部片，留一视频交叉验证）

| 策略 | 准确率 | 误转 | 漏转 | 误停/分 | 启动后没转 | 停止后还转 |
|---|---|---|---|---|---|---|
| 现有融合 | 79.2% | 63.2% | 6.3% | 0.63 | 0.0s（剧情里本来就在转） | 40.1s（15 个里 8 个基本不停） |
| 只用视频 | 85.6% | 31.5% | 8.5% | 0.20 | 0.6s | 16.7s |
| 只用音频 | 83.3% | 35.3% | 10.4% | 0.04 | 20.0s | 28.2s |
| **证据融合** | **85.5%** | **31.3%** | 8.7% | **0.14** | 1.1s | **10.5s** |

6 部片逐片都比现有融合好（v01 73.8→75.1，v02 96.8→100，v04 77.8→80.5，v05 85.5→93.4，
v06 85.2→100，v07 62.6→69.1）。剩下的误转主要在 v04/v07：剧情里有裸露，视频模型自己也判
"转"，融合层救不了，要靠改视频模型。仓库里的配置是用全部 6 部片拟合的（`mode: live`）。

改了证据融合算法时：Python 和 Java 两边一起改，然后 `fusion_lab.py golden` 重新生成
`app/src/test/resources/fusion/` 下的对拍数据，`EvidenceFusionTest` 会逐 tick 校验两边一致。

在线模式（`OnlineAnalysisActivity`，抓屏）仍是原融合：抓屏画面和离线解码的输入分布不同，
等有了在线模式的 bench 数据再接。

### 自检

```bash
python3 bench/tools/fusion_lab.py --selftest
python3 bench/tools/metrics.py --selftest
python3 bench/tools/audio_rescore.py --selftest   # 需要 numpy；有 TFLite 解释器时多跑一个端到端对齐测试
```

## 音频离线重打分 audio_rescore.py

run CSV 记下了真机每个音频结果"是哪一刻出的"（`wall_ms - a_age`）和当时的播放位置。有原视频，就能在
电脑上把那一刻送进模型的 2 秒音频还原出来、重新推理：换分类器、换解码方式、看 YAMNet 自带的 AudioSet
类别分数，都不用再上真机从头播一遍。

依赖（Windows 也一样）：

```bash
pip install numpy ai-edge-litert     # TFLite 解释器，Python 3.10+；已经装了 tensorflow 也行
# ffmpeg / ffprobe 放进 PATH，或者每条命令加 --ffmpeg C:/ffmpeg/bin/ffmpeg.exe
```

视频按真值文件头的 `file_name`（如 `视频1.mp4`）在 `--videos` 目录里找（默认 `bench/videos/`），
找不到再按真值 id（`v01.mp4`）和 run 文件头的 `video_name` 找；也可以 `--pair v01=D:/xcup/视频1.mp4`。
文件大小和真值里记的 `file_size` 对不上会提示。`--only v01,v04` 只跑其中几部。

### 1. 先验证对齐：check

```bash
python bench/tools/audio_rescore.py check --videos D:/xcup/videos
```

用 app 里的旧模型在每个音频结果时刻重算 `a_p_*`，和真机记下的比，看哪种"喂给模型的音频"能把真机
概率复现出来：

| 管线 | 是什么 |
|---|---|
| `phone` | 逐行复刻旧版 `AudioDecoder` + `PcmCircularBuffer`（与真正的 Java 代码逐点一致） |
| `fixed` | 修复后的 app：下混 + `PcmResampler` 重采样到 16kHz，取 `[T-2s, T]`（与 Java 逐点一致，`PcmResamplerTest` 对拍） |
| `clean` | 与训练数据相同：`ffmpeg -ac 1 -ar 16000` 转 16-bit（下混 = (L+R)/2），取 `[T-2s, T]` |

默认：旧版 app 录的 run 比 `phone`/`clean`；修复后的 app 录的 run（参数行 `AUDIO_PIPE=fixed`）比
`fixed`/`phone`。`--pipeline phone,fixed,clean` 可以自己指定。每部片每种管线约 1 分多钟。看输出的最后一行：

- `✓ 对齐验证通过`：逐个结果都对上了（MAE < 0.04），后面的重算可以直接信。
- `✓ 真机喂给模型的是 X 管线的音频……但逐个结果还有出入`：管线认对了，只是旧版 run 没记读窗位置，
  只能从出结果时刻反推，几毫秒的抖动就会让概率有出入（模型对窗口位置非常敏感：平移 1~5ms，三类概率的
  平均差就有 0.03~0.06）。不影响用重算结果做 AUC / 融合评估。
- `✗ 没能确认`：先确认视频文件对不对，再用 `--verbose` 看候选偏移，试 `--frame-size 2048`。

时间偏移（旧版 run 约等于推理耗时 65ms）、真机解码流开头有没有 AAC priming（`--lead`，自动试 1024/2048/2112）
都是扫出来的，写进 `bench/rescored/alignment.json`，`rescore` 直接用。`--dump` 把逐个结果的对比写成
`bench/rescored/check_<id>.csv`。

### 音频解码修复：旧版离线模式喂给模型的是半速音频（已证实）

`AudioDecoder` 没看声道数，把 MediaCodec 输出的交错 PCM（L R L R …）直接当单声道用：
`windowSize = sampleRate*4` 个 short 凑一块，重采样时按 `i/sampleRate` 算时间轴、只取前 2 秒。对双声道
片源（绝大多数 MP4）这意味着：

- 每 2 秒的块只用到**前 1 秒**真实音频，L/R 交错着拉长成 2 秒送进模型 —— **0.5 倍速、低八度**；后 1 秒丢弃；
- 模型窗口 `[T-2s, T]` 由前后两块的半截拼成，听到的内容比播放位置晚 0.5~2.5 秒。

单声道片源速度是对的，但每 4 秒只留前 2 秒。分类器是用正常 16kHz 音频训练的，线上输入和训练分布对不上，
这可能是音频在剧情段大量判"转"、`p_oral` AUC < 0.5 的原因之一。网页模式（`PcmCaptureAudioProcessor`）
有正确下混，不受影响。

**2026-10-05 用 6 部 bench 片证实**（全是 48kHz 双声道 AAC）：

| | v01 | v02 | v04 | v05 | v06 | v07 |
|---|---|---|---|---|---|---|
| `phone` 复现误差 MAE | 0.070 | 0.040 | **0.027** | 0.043 | **0.021** | 0.049 |
| `clean` 复现误差 MAE | 0.284 | 0.275 | 0.253 | 0.237 | 0.248 | 0.300 |

偏移都在 +25~27ms，v04/v06 逐个结果对上（✓）。同样的旧分类器，喂正确的音频（`clean` 重算）后结果级 AUC：

| 特征 | 旧版 app（真机） | 正确的音频 |
|---|---|---|
| `p_sex`（合并 / 最差片） | 0.693 / 0.618 | **0.833** / 0.638 |
| `p_sex+p_oral`（合并 / 最差片） | 0.641 / 0.479 | **0.901** / 0.829 |
| AudioSet Speech（判别力，↑剧情） | — | 0.707 / 0.622 |

**修复**：`PcmResampler`（纯 Java）先按声道平均下混，再用 Blackman 窗 sinc 低通插值重采样到 16kHz
（截止 7.6kHz：7kHz 内 −0.1dB，8.5kHz 以上衰减 >76dB），流式处理、跨 buffer 连续，不丢样本；`AudioDecoder`
把每个解码 buffer 交给它，按"起头 pts + 样本序号/16000"连续写入 `PcmCircularBuffer`。
`audio_rescore.py` 的 `fixed` 管线是它的逐行移植（`PcmResamplerTest` 用 `audio_rescore.py golden` 的
对拍数据校验），所以离线重算出的就是手机会喂给模型的音频。

修复最初由 `fusion_config.json` 的 `audio_pipeline` 开关打开（2026-10-05）；2026-10-06 起旧的半速处理已从
`AudioDecoder` 删除，离线模式只有修复后的音频。证据表要在 `fixed` 音频上拟合：

```bash
python bench/tools/audio_rescore.py rescore --videos D:/xcup/videos            # 默认 --pipeline fixed
python bench/tools/fusion_lab.py eval bench/gt bench/rescored/fixed             # 与 bench/runs 的 eval 对比
python bench/tools/fusion_lab.py eval bench/gt bench/rescored/fixed --speech    # 加上 Speech 证据是否更好
python bench/tools/fusion_lab.py fit  bench/gt bench/rescored/fixed --mode live [--speech]
```

`fit` 会检查所有 run 是同一种音频（重算头部的 `pipeline=` / app 参数行的 `AUDIO_PIPE=`），写进配置的
`audio_pipeline`，并且只接受 `fixed`：`clean` 是 app 产生不了的音频，`legacy`（旧版 app 录的、或 `phone`
重算的）是 app 已经删掉的音频，都拒绝拟合。app 采集的 run 参数行记 `AUDIO_PIPE=fixed`（网页视频模式记
`web`）；用新包重录 bench 后，`check` 就是 `fixed` 对 `phone`，并且有 `a_pos`，能逐个精确对上。

### 2. 重算：rescore

```bash
python bench/tools/audio_rescore.py rescore --videos D:/xcup/videos                    # fixed：修复后的 app
python bench/tools/audio_rescore.py rescore --videos D:/xcup/videos --pipeline phone   # 旧版 app 的半速音频
python bench/tools/audio_rescore.py rescore --videos D:/xcup/videos --classifier new.tflite   # 换分类器
```

写到 `bench/rescored/<管线>[_<分类器>]/`，**文件名与 `bench/runs/` 里的一样**，所以 metrics.py /
fusion_lab.py 指向这个目录就能用：

- `a_p_*` 换成重算的概率，`a_action` / `a_conf` 按原阈值重判（过期规则同融合循环）；
- 追加 `a_ys_*`：YAMNet AudioSet 类别分数（见下表）；
- 旁边的 `<run>.yamnet.npz` 存了每个结果的全部 521 类分数，给 `auc --all-classes` 用；
- 决策列（`smoothed`、`gated`、`ble_*`、`sent*`）仍是原 run 的：端到端指标请用 `fusion_lab.py eval/oracle`
  重放，`fusion_lab.py check` 对重算目录没有意义。

自洽检查：`fusion_lab.py eval bench/gt bench/rescored/phone` 应该和 `bench/runs` 的结果基本一样
（同一管线、同一模型）。`fixed` 和 `phone` 的 eval 一比，就是"修好解码"在端到端上值多少。

### 3. 看判别力：auc

```bash
python bench/tools/audio_rescore.py auc bench/gt bench/rescored/fixed
python bench/tools/audio_rescore.py auc bench/gt bench/rescored/fixed --all-classes   # 521 类排行
python bench/tools/metrics.py bench/gt/v01.csv bench/rescored/fixed/v01_base.csv --diag
```

`auc` 按结果算（每个音频结果一次，只用整个 2 秒窗口都落在同一真值段里的结果，与拟合证据表时用的样本
一致），给出每部片和合并后的 AUC：> 0.5 越高越像动作，< 0.5 越高越像剧情。重点看两张表：

1. 全部结果：某个类（如 Speech）是不是在**每部片**上都明显偏离 0.5（"最差片"那一列）；
2. 只看分类器已经判"转"（`p_sex ≥ 0.5`）的结果：它还能不能把剧情挑出来 —— 这才是"拿它当不转证据、
   压剧情误转"真正需要的。`--scan speech,music` 打印阈值扫描（剧情里命中多少、动作里误伤多少）。

AUC 好看不等于端到端有用：要不要把 Speech 加进融合，以 `fusion_lab.py eval --speech` 的留一结果为准。
`--speech` 在音频特征候选里加上 `p_sex+p_oral-speech/2`、`p_sex-speech/2`（app 送进证据融合的音频向量是
`[sex, oral, noise] + a_ys_*`，Java 端不用改），由拟合按代价挑。

### run CSV 新增的列

新版 app（`BenchLogger`）和 `rescore` 都会写这些列，旧版 dump 没有也不影响其他工具：

| 列 | 含义 |
|---|---|
| `a_pos` | 该音频结果读 PCM 时的播放位置：模型窗口 = `[a_pos-2s, a_pos]`。有它，`check` 不用再反推窗口位置（只有 app 写） |
| `a_ys_speech` / `conversation` / `narration` / `music` | AudioSet 0 / 2 / 3 / 132 类，"剧情"候选证据 |
| `a_ys_silence` / `moan` / `groan` / `breathing` / `pant` / `slap` | AudioSet 494 / 22 / 33 / 36 / 40 / 461 类，"动作"候选 |

分数是 YAMNet output 0（`[帧, 521]`，sigmoid，多标签）按帧取平均；output 1 才是分类器用的 embedding。
挑哪几类在 `AudioInferenceHelper.AUDIOSET_LOG_CLASSES` 和 `metrics.YS_CLASSES` 两处，`metrics.py --selftest`
会核对两边一致。
