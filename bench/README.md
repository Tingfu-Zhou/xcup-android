# bench/ — Xcup 评测基准

离线回放评测流水线。目标：每改一个参数都能量化对比，而不是凭感觉调。

## 目录

```
bench/
  ANNOTATION_SPEC.md      人工真值标注规范（改动需递增 spec_version）
  gt/<video_id>.csv       人工真值（提交进 git）
  runs/<video_id>_*.csv   真机 dump（提交进 git：融合配置是从它们拟合出来的，要能复现）
  tools/
    labeler.html          真值标注器，双击即用，纯本地无上传
    test_labeler.js       标注器核心逻辑测试：node bench/tools/test_labeler.js
    metrics.py            单部片的指标 / 误判定位 / 模态诊断
    fusion_lab.py         融合层离线实验台：回放、交叉验证、模态上限、拟合配置
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

Windows 下把 `python3` 换成 `python`（或 `py`）、路径用 `\` 也可以。`eval` 约 4 分钟，`oracle`
约 10 分钟；参数搜索默认"先粗后细"，`--exhaustive` 改扫全网格（慢约 9 倍，结果基本一致）。

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
```
