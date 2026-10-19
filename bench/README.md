# bench/ — Xcup 评测基准

离线回放评测：每改一个参数或一个模型，都在同一批片子上量化对比，而不是凭感觉调。

## 目录

```
bench/
  ANNOTATION_SPEC.md      人工真值标注规范（改动需递增 spec_version）
  gt/<id>.csv             人工真值（v01~v13）
  runs_initial/<id>_*.csv 真机 dump：app 1.2.5、融合仲裁 v1、现用音频分类头，7 部片（调参用的基线）
  runs_test/<id>_*.csv    同一个包录的新片 v08~v13（不参与调参，只做检验）
  runs_v2/<id>_*.csv      v2 包（加音频线索）重录的 v03/v09/v11：验证过真机 v2 与离线回放逐 tick 一致，不进统一基准
  runs_v3/<id>_*.csv      修复视频抽帧后的包录的 v08：验证过真机与离线重算的 exact_crop 一致（第 7 节），不进统一基准
  rescored/               audio_rescore.py 的输出：audio_old / test_audio_old = runs_initial / runs_test 用现用分类头离线重算
                          （与真机一致，另带 YAMNet 521 类分数 .yamnet.npz）；audio_new = 2026-10-09 评估过的新分类头（未采用）；
                          video_<名字>_<runs 目录名>/ 是视频离线重算的结果（video_exact_crop_* 是统一基准的对照、
                          video_e1d_* 是 2026-10-18 起 app 用的模型，第 5 节），video_alignment.json 是 video_rescore.py check 定下的时间差
  videos/                 原始视频（不进 git；也可以放别处，用 --videos 指定）
  summary.csv             metrics.py --append 的汇总
  baseline/               baseline.py / frame_diag.py / retune.py 的报告（phone、exact_crop、e1d 是现在的基准，第 5 节）
  tools/
    labeler.html          真值标注器，双击即用，纯本地无上传
    metrics.py            单部片：指标、误判定位、单模态诊断
    fusion_lab.py         融合层回放：check（复现真机）、eval（策略对比）、oracle（模态上限）、golden（Java 对拍数据）；
                          FusionArbiter / AudioCue 的 Python 移植
    arbiter_lab.py        融合仲裁评测：逐片最差情况、seek 模拟、压力场景、慢手机、留一选参
    audio_rescore.py      音频离线重算：换音频模型不用上真机
    cue_lab.py            YAMNet 线索实验（第 6 节）：线索统计、误判归因、压力场景、留一视频选规则（app 用了其中的 a1bvid）
    video_rescore.py      视频离线重算（第 7 节）：换抽帧方式 / 预处理 / 视频模型不用上真机；复现真机的 v_p_*；
                          模拟在线模式（录屏）截到的画面
    baseline.py           统一评测基准：换视频模型 / 视频管线时只换视频结果，13 部片按同一口径比（check / rescore / eval / compare）
    frame_diag.py         视频模型的逐帧诊断：窗口长度 / 聚合方式的 AUC、置信度分布、成段的高置信度错误，几个变体并排
    retune.py             给一个视频变体留一视频重调视频阈值和与视频置信度有关的融合参数（label smoothing 之类改了概率分布时用）
    test_labeler.js       标注器测试：node bench/tools/test_labeler.js
    yamnet_class_map.csv  YAMNet 的 AudioSet 521 类名（来自 tensorflow/models，Apache-2.0）
```

run 文件名要以真值 id 开头（`v01_base.csv` 配 `gt/v01.csv`）。命令都在仓库根目录运行；Windows 把 `python3` 换成 `python`。

## 1. 标注

1. 浏览器打开 `bench/tools/labeler.html`（`file://` 直接打开，无需起服务），打开或拖入视频
2. 按 `1`~`5` 在播放头处打点，按三遍法标注，详见 [ANNOTATION_SPEC.md](ANNOTATION_SPEC.md) §6
3. 右侧「自检 / 告警」全部清空后，导出 CSV 放进 `bench/gt/`

视频不上传任何地方；进度暂存在浏览器 localStorage，换机器前记得导出。浏览器放不出来（H.265 等）时先转码，**不要裁剪**：
`ffmpeg -i in.mp4 -c:v libx264 -crf 20 -c:a aac -movflags +faststart out.mp4`

## 2. 采集 + 出指标

`BenchLogger` 把每个融合 tick 写成一行 CSV，`metrics.py` 按视频播放位置 `pos_ms` 和真值对齐，不需要任何标定。

**采集（手机）**：装 debug 包（`BenchLogger` 自动开启；release 包把 `FORCE_ENABLE` 改成 true），进入离线视频分析，
**从头不间断播到尾**（不要 seek、暂停、切后台），退出页面时 Toast 会提示文件路径：
`/sdcard/Android/data/com.example.helloworld/files/bench/run_<视频名>_<时间>.csv`。蓝牙不连也能采。

run 的文件头记录了当次的全部参数（含融合仲裁的 `ARB=v1 …`），任何一份 dump 都能追溯参数。
`v_p_*` / `a_p_*` 是模型的原始概率，离线回放和扫参都靠它们。

**出指标（电脑）**：

```bash
python3 bench/tools/metrics.py bench/gt/v01.csv bench/runs_initial/v01_base.csv              # 准确率 / 误转 / 漏转 / 误停 / 延迟
python3 bench/tools/metrics.py bench/gt/v01.csv bench/runs_initial/v01_base.csv --errors --top 15   # 误判区间，按时长排序
python3 bench/tools/metrics.py bench/gt/v01.csv bench/runs_initial/v01_base.csv --at 127.8   # 某时刻附近逐 tick：概率、仲裁原因、蓝牙动作
python3 bench/tools/metrics.py bench/gt/v01.csv bench/runs_initial/v01_base.csv --diag       # 单模态准确率、判别力 AUC、阈值扫描
```

`--policy exclude|do|noise` 决定 other_adult 怎么算（默认不计入）；`--append bench/summary.csv` 追加一行汇总。
`--diag` 的 AUC ≈ 0.5 说明这一路证据等于掷硬币，调阈值救不了，只能改模型。

## 3. 融合仲裁 FusionArbiter（app 当前的融合层）

`FusionArbiter.java`（纯 Java），离线 / 网页视频 / 在线三种模式共用。融合循环每 500ms 调一次 `decide()`：输入是两个
分析线程判出的动作（`"do"` / `"Noise"` / `""`）、置信度、结果产出时刻和音频线索（v2，见下），输出交给 `updateBluetoothState`
（蓝牙状态机没有改）。
旧融合（投票窗 `smoothedFusion` + 快通道 + `selectBestAction`、启动仲裁 `applyDoStartGate`）已从 app 删除，
`fusion_lab.py` 里留有 Python 版，作"旧融合"对照、回放旧 run。证据按"新推理结果"计数：同一个结果在几个 tick 里
重复出现只算一次，产出超过 2 秒的不记。

| 阶段 | 规则 |
|---|---|
| 启动 | 满足任一条：<br>• 两路佐证：3 秒内视频强"转"（≥0.85）≥2 个、音频强"转"（≥0.90）≥1 个，并满足比维持更严一档的条件（视频 6 秒内"转"占比 ≥80%，或音频最近 4 个全是"转"），防止开关抖动；<br>• 音频兜底：9 秒内音频结果 ≥4 个且**全部**是"转"（不看视频，特写时也能启动；零容忍挡住剧情里"转"中夹"不转"的音频）；<br>• 视频单独启动（v2）：音频最近 4 个结果的线索都是音乐 / 安静 / 听不出，且 3 秒内视频强"转"≥3 个、6 秒内占比 ≥90%（动作被背景音乐盖住时分类头一直判"不转"）。<br>对白否决（v2）：视频最近 4 个都是 ≥0.90 的"剧情"时，线索为对白的音频"转"在上面三条里按 unclear 算（剧情对白被分类头判成"转"）。 |
| 维持 | 任一路有支持：音频最近 4 个里 ≥3 个"转"；或视频最近 4 个里 ≥2 个"转"且 6 秒内占比 ≥70%。维持时主动交 `"do"`，取消状态机里待执行的停转。 |
| 停止 | 两路都没有支持时：刚 reset 的 6 秒内某一路结果还不到 4 个，先等；4 秒内有"不转"就交 Noise（之后照旧要状态机 1 秒稳定 + 2 秒最短驻留）；看不清 / 听不清 / 过期最多再等 6 秒（两路结果完全中断时约 16.5 秒停）。 |

- **reset**：seek、暂停后恢复时 `reset(now)`；静音后恢复（网页视频模式的 VAD、在线模式）`reset(now, 2000)`，恢复后 2 秒内的
  结果也不算（音频窗口里还混着静音）；墙钟回跳自动 reset。run 的 `arb_reset` 列记下每次 reset，离线回放照做。
- **暂停**：离线 / 网页视频模式下，只有"有证据撑着"的 do（原因 `start_*` / `hold_*`）才挂起、恢复后接着转，只是在等结果时
  同步成 Noise（否则网页模式下"静音暂停 → 一声短响恢复 → 又静音暂停"会让马达跟着每声短响一直转）。在线模式暂停一律同步成 Noise。
- 每个 tick 的原因写进 `arb` 列：`start_both` / `start_audio` / `start_video` / `hold_audio` / `hold_video` / `hold_both` /
  `stop_neg` / `stop_unclear` / `wait_sparse` / `wait_unclear` / `off_neg` / `off_wait`。
- **音频线索**（v2，2026-10-11，`AudioCue.java`）：`AudioInferenceHelper` 同一次推理多取 YAMNet 的第 0 个输出（AudioSet 521 类
  分数，4 帧平均），按含义分组求和后给每个音频结果判一个线索（对白 / 音乐 / 安静 / 听不出 / 人声 / 拍打 / 其它，判法见第 6 节），
  界面的音频行后面会显示。只用在启动的两条规则上，维持和停止不看；拿不到分数时线索为未知，行为与 v1 完全相同。
  参数行多了 `ARB=v2 … ARB_A1_* ARB_BV_*` 和 `CUE=v1 CUE_*`，CSV 多了 `a_cue`、`a_cue_*` 列（第 5 节）。

### v1 的效果（runs_initial 7 部片，离线回放）

| | 准确率 | 误转 | 漏转 | 动作中误停 | v03 漏转 | v04 / v07 误转 | 停止后还转 | 启动后没转 |
|---|---|---|---|---|---|---|---|---|
| 旧融合 | 87.8% | 34.4% | 3.7% | 24 次 | 37.1% | 74.2% / 68.6% | 13.5s | 0.1s |
| 融合仲裁 | 92.6% | 18.7% | 3.0% | 4 次 | 26.5% | 43.6% / 55.8% | 3.2s | 2.6s |

- 本节和第 6 节的数字都是旧视频管线（`phone`：只取关键帧 + 整帧压扁）上的；修复后的对照见第 5 节 baseline.py。
- 规则和参数是看着这 7 部片定的（样本内）；留一视频选参后是 92.6% / 19.0% / 3.0%。真正的验收要靠新片子。
- v2（加音频线索）在全部 13 部片上：准确率 93.3% → 94.2%，误转 16.0% → 13.9%，漏转 2.4% → 2.1%，误停 6 → 7 次（第 6 节）。
- 误转 = 剧情时长里马达在转的比例，漏转 = 动作时长里马达没转的比例。"停止后还转 / 启动后没转"取自拼接样本（同一部片的
  动作段和剧情段首尾拼接，造出切换点）：切换点之后该停还转 / 该转没转的秒数中位数，比真实片子乐观 1~2 秒。

**已知代价和局限**：
- 启动更慢：seek 进特写或低声音段（seek 后最先几个结果不是"转"）中位 6 秒、p90 9 秒，81 次里 2 次 30 秒内没启动
  （旧融合 p90 4.5 秒）。音频一直判不出"转"（听不到目标声音、只有背景音乐，如 v03 415~484s、516~600s）时不会启动；
  已在转的马达可以只靠视频维持。
- v07 最长连续漏转 2.5s → 12s，最长连续误转 74s → 104s。
- 剩下的误判多是两个模型同时错：v07 剧情 93~228s 两个都判"转"；v03 口交 415~437s（有背景音乐）两个都判剧情。
- 慢手机：视频结果慢 2~3 倍基本不受影响，慢 4 倍时漏转升到 4%；音频结果慢到 2 秒一个时误转升到 25%，3 秒一个时漏转升到 6%，
  要按实际间隔重新调参。
- 蓝牙状态机的计时用墙钟，墙钟大幅往回跳时会卡住相应时长（原有问题）。

### 上真机验证

1. 用新包按第 2 节重录（最好加几部规则没见过的新片）。
2. `python3 bench/tools/fusion_lab.py check bench/gt <新 runs 目录>`：`gated` 和 `arb` 的一致率都 ≥99%，才说明手机上跑的
   就是离线评估的那套。v2 的 run 还会打出"音频线索 真机与重判一致 x/y"（要 100%）和线索分布；出现"没有音频线索"
   说明手机上没读到 YAMNet 的第 0 个输出，线索规则没生效。
3. `python3 bench/tools/arbiter_lab.py report bench/gt <新 runs 目录>` 看逐片结果；`metrics.py --at` 的"仲裁原因"列能看出
   某次停转是哪条规则触发的。

## 4. 换模型后怎么调参

先改哪个：`fusion_lab.py oracle`（把一路换成真值）在融合仲裁下，音频完美时准确率 92.6% → 99.4%（代价 985 → 175），
视频完美时 → 96.3%（代价 572）。改音频收益最大；但视频单独就能维持马达，剧情里视频的长时间误判（v07）只能靠改视频模型。
音频接近完美后，启动规则（约 3 秒）会变成瓶颈，那时要放宽启动参数。

1. **拿到新模型的数据**
   - 音频模型：`audio_rescore.py rescore --classifier new.tflite` 在电脑上重算，输出目录直接给下面的工具用，不用上真机。
   - 视频模型：`baseline.py rescore --model new.tflite --name <名字>` 在电脑上重算 13 部，再 `baseline.py compare exact_crop <名字>`
     和现在的基准比（第 5 节）。目前只支持与现用模型同一结构的模型（逐帧特征 → 时间平均 → 线性层，输入 [1,12,160,160,3]）；
     换了时序头、帧数、分辨率或训练时的裁剪方式，要先扩展 `video_rescore.py`（和 app 的 `FramePreprocessor`）。
   - 类别顺序（视频 `[plot, oral, sex]`、音频 `[sex, oral, noise]`）或类别数变了，要同时改两个 Activity 里的判定、`BenchLogger`
     的列和 bench 工具。
   - **别拿 bench 这几部片训练**，否则这里的数字就不再能代表新片子。
   - 新旧模型要在各自调好的参数下比：换了模型，概率分布就变了（阈值、`A_FLOOR` 等都要跟着变），只用现有参数比不公平。
     rescore 时给每个模型起单独的名字（音频 `--tag`、视频 `--name`），免得互相覆盖。
2. **先定各路的阈值**：`metrics.py --diag` 看每一路的 AUC 和阈值扫描，定 `VIDEO_DO_PROB_THRESHOLD` / `VIDEO_PLOT_PROB_THRESHOLD` /
   `AUDIO_DO_PROB_THRESHOLD` / `AUDIO_NOISE_PROB_THRESHOLD`（`VideoProcessActivity` 和 `OnlineAnalysisActivity` 里各一份）。
   这些阈值在采集时就判定好了，改了要重录（音频也可以改 run 文件头里的 `AUDIO_DO` / `AUDIO_NOISE` 再 rescore）。
   视频离线重算时按原 run 文件头的 `VIDEO_DO` / `VIDEO_PLOT`（0.65 / 0.60）判定，离线试别的视频阈值要先给工具加参数。
3. **看融合仲裁哪里不行**：`arbiter_lab.py report` 和 `stress`，按最差的片子看，不只看平均：

   | 现象 | 先看这些参数（`FusionArbiter.java` 顶部） |
   |---|---|
   | 启动慢 | `V_FLOOR` / `A_FLOOR`、`C_V` / `C_A`、`Z_WIN_MS` / `Z_N`、`START_V_CONS_PCT` / `START_MA_K` |
   | 剧情误转 | 上面几个往严调；维持的 `MA_K`、`MV_K`、`V_CONS_PCT` |
   | 动作中误停 | `MA_K`（音频容忍几个"不转"）、`MV_K` / `V_CONS_PCT`、`NEG_WIN_MS` |
   | 动作结束后停得慢 | `NEG_WIN_MS`、`MA_K` / `MV_K`、`UNCLEAR_MAX_MS` |
   | 推理变慢（结果间隔变大） | `M_MAX_MS`、`Z_WIN_MS`、`CONS_WIN_MS` |

4. **选参**：把候选值写进 `arbiter_lab.py` 的 `NARROW` / `WIDE` 网格（用 Python 侧的名字，如 `ARB_Z_WIN`，对照 `fusion_lab.py`
   的 `ARB_DEFAULTS`），跑 `lovo`（留一视频选参；压力场景必须全过、seek 进动作段不能有不启动的）。优先选多数折都选中的值。
5. **改代码**：`FusionArbiter.java` 的常量和 `fusion_lab.py` 的 `ARB_DEFAULTS` 一起改（规则本身要改时，两边的 `FusionArbiter`
   逐行一起改；线索的分组和阈值在 `AudioCue.java` 和 `fusion_lab.py` 的 `CUE_*` / `audio_cue`），然后
   `python3 bench/tools/fusion_lab.py golden` 重新生成 `app/src/test/resources/fusion/` 下的两份对拍数据，
   跑 `FusionArbiterTest`、`AudioCueTest` 和 `fusion_lab.py --selftest`。
6. **真机复验**：同第 3 节。

`FusionArbiterTest` / `AudioCueTest` 在 Android Studio 里直接跑，或 `gradlew :app:testDebugUnitTest`。没有 Android SDK 时用 JDK + JUnit：

```bash
CP=junit-4.13.2.jar:hamcrest-core-1.3.jar
J=app/src/main/java/com/example/helloworld; T=app/src/test/java/com/example/helloworld
javac --release 11 -encoding UTF-8 -cp $CP -d /tmp/arbtest \
  $J/AudioCue.java $J/FusionArbiter.java $J/FramePreprocessor.java \
  $T/FusionArbiterTest.java $T/AudioCueTest.java $T/FramePreprocessorTest.java
java -cp /tmp/arbtest:app/src/test/resources:$CP org.junit.runner.JUnitCore \
  com.example.helloworld.FusionArbiterTest com.example.helloworld.AudioCueTest com.example.helloworld.FramePreprocessorTest
```

## 5. 工具参考

### fusion_lab.py

```bash
python3 bench/tools/fusion_lab.py check  bench/gt bench/runs_initial   # 回放能否逐 tick 复现真机（≥99% 才可信）
python3 bench/tools/fusion_lab.py eval   bench/gt bench/runs_initial   # 策略对比：旧融合 / 融合仲裁 / 只用视频 / 只用音频 / 一直转 / 证据融合
python3 bench/tools/fusion_lab.py oracle bench/gt bench/runs_initial   # 模态上限：假设视频（或音频）完美，最多能到多少
python3 bench/tools/fusion_lab.py golden                               # 生成 Java 对拍数据（FusionArbiterTest、AudioCueTest）
```

- 回放按 run 的参数行选融合：有 `ARB=` 的用融合仲裁（`ARB=v1` 不看线索），否则用旧融合；之后都接蓝牙状态机的复刻。
  音频线索优先用 run 里记下的 `a_cue`，没有时按 `a_cue_*` 现判；老 run 两样都没有，线索为未知（要线索就用 `cue_lab.py`，
  它从 `.yamnet.npz` 接上分数）。
- `golden` 默认读 `bench/gt` + `runs_initial,runs_test`，真实片段的线索来自 `rescored/audio_old,test_audio_old` 的
  `.yamnet.npz`；对拍数据里线索规则真正决定输出的行数会打出来（`--selftest` 要求视频单独启动 ≥20 行、对白否决 ≥50 行）。
- 少量不一致来自蓝牙状态机取时间比 `wall_ms` 晚几毫秒、以及旧 run 把置信度存成 4 位小数（新 run 存完整 float），
  正好卡在阈值边界时会翻转。
- 代价 = 误转秒数 × fp + 漏转秒数 × fn + 动作中误停次数 × stop，默认 `--cost fp=1,fn=1,stop=3`（产品取舍，不随模型变）。
- "证据融合"是 2026-10-06 从 app 删掉的方案（按证据表累积对数似然比），只作参照。留一拟合默认多进程（`--jobs 1` 串行）。

### arbiter_lab.py

```bash
python3 bench/tools/arbiter_lab.py report   # 旧融合 vs 融合仲裁：逐片、最长连续漏转、拼接样本、seek 模拟、真实停转延迟
python3 bench/tools/arbiter_lab.py stress   # 合成压力场景（交替假"转"、结果过期、遮挡、听不到、视频断了）+ 模拟慢手机
python3 bench/tools/arbiter_lab.py lovo     # 留一视频选参（--wide 连音频兜底窗口一起选）
```

默认 `bench/gt` + `bench/runs_initial`，也可以在命令后面给 `<gt 目录> <runs 目录>`。seek 模拟：每个够长的真值段里每 15 秒
seek 一次，丢掉模型窗口还没重新攒满的结果（视频约 2.75 秒、音频 2 秒），再从全新状态回放 30 秒。

### baseline.py（统一评测基准，2026-10-14 起）

```bash
python3 bench/tools/baseline.py eval phone                          # 手机原始 run（旧视频管线），与第 6 节 13 部的 a1bvid 一致
python3 bench/tools/baseline.py check   --videos D:/xcup/videos --model old.tflite   # 换一处原片后，rescore 前先核对（13 部）
python3 bench/tools/baseline.py rescore --videos D:/xcup/videos     # 修复后的管线 + app 里的模型 → exact_crop（10-15 做的）
python3 bench/tools/baseline.py rescore --videos D:/xcup/videos --model new.tflite --name new1
python3 bench/tools/baseline.py compare phone exact_crop new1       # 并排对比，第一个是参照
```

- **2026-10-18 起 app 里的视频模型是 E1d**（P1 定的，sha1 63f3e033cf38，`training/RECIPE.md`），阈值和融合参数没改。
  - `exact_crop` 还是固定对照：P1 之前的模型（sha1 93851823e4a4）+ 修复后的管线，也就是录 `runs_initial` / `runs_test` 的那个模型。
  - `e1d` 是 app 现在的水平。新模型两个都比：`compare exact_crop e1d new1`。
  - 现在不带 `--model` 跑 `rescore`，会因为模型指纹不同被拒绝写进 `exact_crop`。
- **`check` 要用录 run 时的旧模型。** 它核对的是离线能不能复现 `runs_initial` / `runs_test`，不带 `--model` 会提醒"对不上"。
  旧模型从 git 历史取：
  `git show 11bf834:app/src/main/assets/mobilenetv3small_tap_3class_float32.tflite > old.tflite`。
  `runs_v2`、`runs_v3` 也是旧模型录的，对它们跑 `video_rescore.py check` 同样要加 `--model old.tflite`。

- 固定的部分：13 部片（`gt` v01~v13）、录像 `runs_initial` + `runs_test`、音频线索 `audio_old` + `test_audio_old`、
  融合 `a1bvid`（= app 的 FusionArbiter v2）、代价 fp=1,fn=1,stop=3。变的只有视频结果（"变体"）：`phone` 读原始 run，
  其余读 `rescored/video_<名字>_runs_initial`、`_runs_test`。不混用别的录像：同一部片两次录像差别就很大（v11：90.2% / 82.5%）。
- 视频单独的指标（p_oral+p_sex 的 AUC、按现用阈值的准确率）逐 tick 算，与 `metrics.py --diag` 相同，另给 13 部合并。
- 重算目录里的 `variant.json` 记下模型指纹，换了模型还写同一个名字会被拒绝（`--overwrite` 放行）。
- 不含调参：给新模型调阈值和融合参数按第 4 节，最终比较时两边用同一套流程各自调好。

**现在的基准**（2026-10-15，报告在 `bench/baseline/`；13 部片，app 现用的阈值和融合参数）：

| 视频结果 | 代价 | 准确率 | 误转 | 漏转 | 误停 | seek 进特写 p90 / 没启动 | 视频 AUC 合并 / 最差片 | 视频准确率 |
|---|---|---|---|---|---|---|---|---|
| `phone`：手机原始 run（旧管线） | 1164 | 94.2% | 13.9% | 2.1% | 7 | 7.2s / 0 | 0.913 / 0.467（v07） | 83.0% |
| `exact_crop`：修复后的管线 + P1 之前的模型 | 1219 | 93.6% | 16.5% | 1.7% | 5 | 13.1s / 2 | 0.935 / 0.671（v07） | 87.9% |
| `e1d`：修复后的管线 + E1d（2026-10-18 起 app 用） | 1099 | 94.3% | 15.0% | 1.4% | 5 | 9.1s / 0 | 0.948 / 0.715（v07） | 89.1% |

- 修复后视频本身更准：AUC 0.913 → 0.935，v07 0.467 → 0.671、v01 0.871 → 0.979，按现用阈值的准确率 83.0% → 87.9%；
  v12、v13 的 AUC 略降（0.919 → 0.870、0.950 → 0.874）。
- 端到端反而略差（代价 +55：v04 +63、v03 +47、v09 +28，v11 −98），seek 进特写/低声音时启动变慢（这类 seek 按当时的
  模型结果挑，两边样本不同：104 / 64 个）：融合仲裁的参数是在旧输入（同一张关键帧的一串重复结果）上调出来的。
- `exact_crop` 是以后换视频模型的对照，`phone` 只作历史参照（README 里其余的数字都是它的口径）。
- `e1d` 一行是 2026-10-18 补的（`bench/baseline/compare_exact_crop_vs_e1d.txt`）。两边各自留一视频重调后：exact_crop 1219、e1d 1073
  （`retune_compare_exact_crop_vs_e1d_conf.txt`）。

**换视频模型后怎么比**：

```bash
python3 bench/tools/baseline.py rescore --videos D:/xcup/videos --model D:/models/new1.tflite --name new1 --jobs 4
python3 bench/tools/baseline.py compare exact_crop e1d new1
```

1. 先看视频 AUC（13 部合并、逐片平均、最差片）：只反映模型本身，不受阈值和融合参数影响，是比模型的主要依据。
   `python3 bench/tools/frame_diag.py --name exact_crop --name new1` 再看错在哪：成段的高置信度错误、置信度分布、逐个窗口判错 → 判对多少秒。
2. 再看端到端：两边都用现有参数，只能看方向。
3. 决定换不换之前，按第 4 节给新模型调阈值和融合参数，`exact_crop` 也用同一套流程调一次，两边各自调好再比。
   只是视频的概率分布变了（例如 label smoothing）时，`python3 bench/tools/retune.py compare exact_crop new1` 就够：
   两边各自留一视频重调 `VIDEO_DO` / `VIDEO_PLOT` / `ARB_V_FLOOR` / `ARB_BV_CONF`，再按验收线比。
4. 前提：新模型没用这 13 部片的任何片段训练或挑模型；结构与现用模型相同（见第 4 节）。
   视频模型的训练见 `training/RECIPE.md`（配方）和 `training/README.md`（脚本怎么用）。`train_video.py` 导出的 tflite 与现用模型
   同结构，可以直接 rescore。

### frame_diag.py（视频模型的逐帧诊断，2026-10-16）

```bash
python3 bench/tools/frame_diag.py --name exact_crop                 # 一个变体
python3 bench/tools/frame_diag.py --name exact_crop,e1d,new1        # 并排，第一个是参照
python3 bench/tools/frame_diag.py --selftest
```

- 读 rescore 写的逐帧 logit（`<run>.vframes.npz`）和每个结果的 12 帧时刻（`v_fpts`），按视频结果（约每 0.6 秒一个）算；
  标签取出结果那一刻的真值，other_adult 不计。`phone` 没有逐帧 logit，不支持。
- 报告：窗口长度（单帧 / 3 秒 / 6 秒 / 12 秒）和聚合方式（logit 平均 / 概率平均 / 中位数 / 投票）的 AUC；置信度分档的错误率；
  成段的高置信度错误（以 ≥0.9 判错的结果连成 ≥8 秒的段，中间夹着的其它结果累计 ≤3 秒）；逐片 AUC 和按 app 阈值的误判；
  并排时另给逐片 AUC 变好 / 变差几部、最差片、逐个窗口配对（判错 → 判对、判对 → 判错各多少秒）、两半片子各自的合并 AUC。
- `exact_crop` 的结果（`bench/baseline/frame_diag_exact_crop.txt`）：单帧 / 3 秒 / 12 秒 AUC 0.924 / 0.935 / 0.942；置信度 ≥0.99
  的窗口占 65%，其中 6.2% 判错，占全部错误的 31%；成段错误 19 段，误转 180 秒、漏转 310 秒（各占视频全部误转 / 漏转的一半多）。
- `e1d` 对 `exact_crop`（`bench/baseline/frame_diag_exact_crop_vs_e1d.txt`）：成段错误 18 段，误转 120 秒、漏转 271 秒；
  逐片 AUC 5 部变好（v01 v03 v07 v12 v13）、0 部变差。

### retune.py（重调视频阈值，2026-10-16）

```bash
python3 bench/tools/retune.py run exact_crop [--grid video|conf]    # 现用参数 vs 留一视频选参 vs 样本内
python3 bench/tools/retune.py compare exact_crop new1               # 两边各自留一选参后按验收线比
python3 bench/tools/retune.py --selftest
```

- 只调与视频置信度有关的 4 个：`VIDEO_DO`、`VIDEO_PLOT`（视频判定阈值）、`ARB_V_FLOOR`（视频强转）、`ARB_BV_CONF`（对白否决）；
  `--grid video` 只调前两个（12 组），默认 `conf` 四个一起（108 组）。视频判定按 run 里的 `v_p_*` 用新阈值重判。
- 留一视频选参的约束同 `cue_lab.py lovo`（训练片误停最多 +1、漏转不升、seek 启动不变慢），验收线同第 6 节。
- `exact_crop`（`bench/baseline/retune_exact_crop_conf.txt`）：现用 1219，留一选参 1219，样本内 1198 —— 阈值已在平台上，
  重调只会过拟合，与 2026-10-15 的结论相同。
- `e1d`（`bench/baseline/retune_compare_exact_crop_vs_e1d_conf.txt`）：
  - 现用参数 1099，留一选参 1073，12/13 折选了 `VIDEO_PLOT 0.75`、`ARB_BV_CONF 0.80`。
  - 对它自己的现用参数只降 2.4%，特写 seek p90 还慢 1.5 秒，没过验收线，所以 app 先不改参数。
  - 两边各自留一选参后，e1d 对 exact_crop（1219 → 1073）验收线全过。

### audio_rescore.py

依赖：`pip install numpy ai-edge-litert`（或已装的 tensorflow），ffmpeg / ffprobe 在 PATH 上（或 `--ffmpeg` 指定）。

```bash
python3 bench/tools/audio_rescore.py check   --videos D:/xcup/videos --runs bench/runs_initial   # 验证离线还原的音频和真机一致
python3 bench/tools/audio_rescore.py rescore --videos D:/xcup/videos --runs bench/runs_initial [--classifier new.tflite] [--tag 目录名]
python3 bench/tools/audio_rescore.py auc bench/gt bench/rescored/audio_old [--all-classes]  # 音频判别力 AUC
```

- `check` 用 app 里的模型在每个音频结果时刻重算概率、和真机比，对齐结果写进 `bench/rescored/alignment.json`，`rescore` 直接用。
- `rescore` 写到 `bench/rescored/<管线>[_<分类器>]/`（或 `--tag` 指定的目录），文件名不变：`a_p_*` 换成新概率，`a_action` /
  `a_conf` 按 run 文件头的阈值重判。决策列还是原 run 的，所以只能用 `eval` / `arbiter_lab` 重放，`check` 对它没有意义。
  另写 YAMNet 自带的 AudioSet 类别分数：`a_ys_*` 列（10 类）和同名 `.yamnet.npz`（全部 521 类，按结果时间戳对齐），只做离线分析；
  音频线索 `a_cue` / `a_cue_*` 也按重算的分数重写（新 app 的 run 里原有的那份不留）。
- 管线 `fixed` = 现在 app 喂给模型的音频（下混 + `PcmResampler`，与 Java 逐点对拍）；`phone` = 2026-10-05 之前的半速音频；
  `clean` = 训练数据的处理方式。
- 视频按真值文件头的 `file_name` 在 `--videos`（默认 `bench/videos/`）里找，或 `--pair v01=D:/xcup/视频1.mp4`；`--only v01,v04`
  只跑几部。
- 音轨时间戳不连续的片子（多半是分片下载再拼起来的；表现是 run 里播放位置比墙钟快 0.5% 以上，`check` 报"✗ 没能确认：两种管线
  差不多"）：手机的 `AudioDecoder` 按每帧的 pts 打时间戳、偏差超过 100ms 就重新起头，`fixed` 管线现在用 ffprobe 读每帧 pts
  照做（时间戳连续的片子结果逐位不变）。2026-10-11 之前 rescore 的这类片子要重跑 `check` 和 `rescore`。

### 自检

```bash
python3 bench/tools/fusion_lab.py --selftest
python3 bench/tools/metrics.py --selftest      # 含与 BenchLogger 的 CSV 列名契约检查
python3 bench/tools/audio_rescore.py --selftest
python3 bench/tools/cue_lab.py --selftest      # 开关全关时与 v1 逐 tick 一致、对拍数据、线索接上的比例
python3 bench/tools/video_rescore.py --selftest  # 预处理 / ScreenCropper 对拍数据、与 cv2.INTER_AREA 一致、在线模式截图的几何、
                                                # 合成视频上的取帧与对齐
python3 bench/tools/baseline.py --selftest       # phone 与第 6 节一致、重算格式读对、模型指纹、check / rescore 的包装
python3 bench/tools/frame_diag.py --selftest     # 合成数据上的 AUC / 成段错误 / 配对；exact_crop 的逐帧 logit 与 CSV 对上
python3 bench/tools/retune.py --selftest         # 现用参数的回放与 baseline.py 相同；exact_crop 只调视频阈值的留一结果
node bench/tools/test_labeler.js
```

### run CSV 里后来加的列

| 列 | 含义 |
|---|---|
| `a_pos` | 该音频结果读 PCM 时的播放位置，模型窗口 = `[a_pos-2s, a_pos]` |
| `arb` | 融合仲裁本 tick 的原因（见第 3 节）。`gated` 是融合仲裁的输出。旧融合时期的 run 还有 `smoothed` 列（投票窗），新 run 没有 |
| `arb_reset` | 这一行之前的 reset：`<ms>[+<预热ms>][:seek]`。回放时同样 reset，带 `:seek` 的还清空蓝牙状态（旧融合回放时也清空投票窗） |
| `a_ys_*` | AudioSet 类别分数（Speech、Music、Moan 等，见 `metrics.YS_CLASSES`），只有 `rescore` 写，4 位小数 |
| `a_cue` | 该音频结果的线索（`dialog` / `music` / `quiet` / `blind` / `vocal` / `impact` / `other`，没有为空），2026-10-11 起的 app 写 |
| `a_cue_*` | 线索用的 YAMNet 分组分数（`voc_core` `voc_ext` `speech` `speech_ext` `music` `silence` `impact`，全精度），同上 |
| `v_pos` | 该视频结果窗口里最后一帧的时刻（离线模式是帧的 pts，网页模式是抓帧时的播放位置），2026-10-14 起的 app 写 |
| `v_fpts` | 窗口里 12 帧各自的时刻（ms，分号隔开，从旧到新），同上。`video_rescore.py` 靠它逐帧还原模型输入 |

参数行里的 `BOOST=`（2026-10-18 起的 app）记手动电子菜单里的「定时加档」设置：`BOOST=off`，或 `BOOST=+<档数>@<分钟>min`
（规则见 `LevelBoost.java`）。打开时，到点后 `sent_level` 是加档后的档位，到点那一刻还会多一次补发（`final_freq` / `ble_level`
仍是分析档位），`fusion_lab.py check` 的 `sent` 列从那之后会对不上。录评测数据时请关掉定时加档（默认就是关的）。

## 6. YAMNet 线索（cue_lab.py；app 2026-10-11 起用 `a1bvid`）

YAMNet 的第 0 个输出（AudioSet 521 类分数）以前 app 没读。`cue_lab.py` 把 `rescored/audio_old/*.yamnet.npz` 按结果时间戳
接到 `runs_initial` 的每个音频结果上（7 部片 100% 接上），给每个结果判一个"线索"，再在融合仲裁上加开关，看能不能改进。
开关全关、或者线索未知时与 v1 逐 tick 一致（`--selftest` 校验，含 FusionArbiterTest 的对拍数据）。最后选定的 `a1bvid`
已经做进 app（`AudioCue.java` + `FusionArbiter` v2，第 3 节）；`--selftest` 也核对 `fusion_lab.py` 里 app 的移植与
`cue_lab.py` 的 `a1bvid` 在 13 部片上逐 tick 一致。

| 线索 | 判定（4 帧平均的分数） |
|---|---|
| 对白 | 人声组 < 0.01 且 Speech ≥ 0.7 |
| 音乐 | 人声组 < 0.05、撞击/口腔声 < 0.1，Music ≥ 0.3 且不低于 Speech（带人声的歌也算） |
| 安静 / 听不出 | 人声组 < 0.05、Speech < 0.2、撞击/口腔声 < 0.1；其中 Silence ≥ 0.3 的是安静 |
| 人声 / 拍打 | 人声组 ≥ 0.3 / 撞击/口腔声 ≥ 0.2 |

人声组 = 尖叫 11、笑 13/15~18、哭 19、呜咽 21、呻吟 22、叹气 23、哼 33、咕哝 34、呼吸 36、倒吸气 39、喘 40；
撞击/口腔声 = 咀嚼 49、敲 353、轻敲 354、挤压水声 441、拍打 461、重击 462（下标见 `yamnet_class_map.csv`）。

**基线的误判落在哪**（`cue_lab.py blame`）：
- 漏转 104 秒，v03 占 70 秒：435~454 秒视频判转、音频被背景音乐盖住（分类头判不转），线索能救；416~434 秒两路都判剧情、
  473~483 秒和 559~568 秒视频特写判剧情，线索救不了。
- 误转 249 秒：对白 104 秒（v07 76、v04 29），但其中 82 秒视频也判转（多是视频在维持）；安静 42 秒（v04、v06：Silence≈1 时分类头判转）；
  其余主要是两路同时判错。
- 动作中误停 4 次：v03 两次（音乐 + 视频特写）、v05 一次（对白）、v07 一次（特写，拍打声里夹了一个不转）。

**规则和留一视频**（`cue_lab.py lovo`，13859 组全组合；约束：压力场景全过、训练片误停最多 +1、训练片漏转不升、
seek 启动不变慢；基线留一代价 996）：

| 规则 | 做什么 | 留一视频 | 结论 |
|---|---|---|---|
| A1 | 音频最近 4 个都听不出时，视频（3 秒内强转 ≥3、6 秒内占比 ≥90%）单独启动 | 只有 v03 有这种情况，留出 v03 时选不出来；其余 6 部完全不变 | 候选 `a1` |
| A2 | 听不出时音频的"不转"不算结束证据 | 违反"视频断了、音频时有时无的支持不能一直转"的压力场景 | 不要 |
| A3 | 听不出（含安静）时音频的"转"不算启动证据 | 989；漏转 3.0→3.6%，seek 启动变慢 | 不要 |
| B=start | 对白时分类头的"转"不算启动证据，维持不受影响 | 891；7 折都选同一组阈值 | 候选 `bstart` |
| B=all、B2 | 对白时的"转"也不算维持证据 / 对白时视频不能单独维持 | 908；v01 口交时口腔声被判成 Speech，多一次误停 | 不要 |
| P1、P2 | 人声 / 拍打确认的"转"启动更快、维持时多容忍一个不转 | 1010；动作结束后停得慢（v06） | 不要 |

候选在 7 部片上（固定参数，`cue_lab.py eval --preset <名字>`；敏感曲线 `sens` 显示阈值前后挪一档结果几乎不变）：

| | 代价 | 准确率 | 误转 | 漏转 | 误停 | 最差片漏转 / 最长连续漏转 | seek 进特写/低声音 p90、没启动 | 逐片代价变化 |
|---|---|---|---|---|---|---|---|---|
| v1（现在） | 985 | 92.6% | 18.7% | 3.0% | 4 | 26.5% / 51.1s | 9.0s，2/81 | |
| `a1` | 910 | 93.0% | 18.7% | 2.5% | 5 | 19.8% / 23.5s | 7.0s，0/81 | 只有 v03 变：−75 |
| `bstart` | 891 | 93.4% | 16.1% | 3.0% | 4 | 26.5% / 51.1s | 11.5s，2/81 | v04 −88、v07 −22、v01 +14、v05 +1 |
| `a1b` | 815 | 93.7% | 16.1% | 2.5% | 5 | 19.8% / 23.5s | 7.0s，0/81 | v03 −75、v04 −88、v07 −22、v01 +14、v05 +1 |

- 按 2026-10-10 定的验收线（留一代价降 ≥5%、最多 1 部变差且 ≤5%、误停 ≤6、seek 启动不变慢、漏转不升）：`a1` 全过，但收益只来自
  v03 一部片，留一验证不了；`bstart` 和 `a1b` 卡在 v01 变差（+14，全在"剧情 → 口交"的拼接样本里：口交的口腔声被 YAMNet 判成
  Speech，启动晚了几秒），`bstart` 还让 seek 进特写的 p90 从 9.0 秒变成 11.5 秒。YAMNet 的 521 类里找不到能把口腔声和真对白
  分开的类（两边 Speech 0.85 / 0.87，其余接近 0）。
**新片 v08~v13**（2026-10-11，`runs_test`，不参与调参；v11 是边打电话边口交）：
- v08、v09、v11 上（候选事先定好）：`a1` 没有作用；`bstart` / `a1b` 误转 9.1%→1.9%，但 v11 漏转 22→43 秒、seek 进特写的
  启动 p90 7→15.5 秒，不能用。于是加了 `bvid`：对白的"转"只在视频最近 4 个结果都是 ≥0.9 的剧情时才不算启动证据
  （v09 剧情里视频 1.00 确定是剧情；v11 口交时视频拿不准）。v08/v09/v11 上误转 9.1%→1.9%、漏转 3.5%→3.6%。
- `bvid` 是看了 v09/v11 之后设计的，用 v10、v12、v13 独立检验：误转 16.7%→15.2%（v12 −7 秒），漏转 0→0，误停 0→0，
  seek 不变；`bstart` 误转降得一样多，但 seek 进特写的启动 p90 7.1→12.1 秒。
- 6 部新片里都没有"动作时被背景音乐盖住"，`a1` 在 12 部片上完全不变，只在 v03 上起作用。

全部 13 部（固定参数，旧视频管线 `phone`）：

| | 代价 | 准确率 | 误转 | 漏转 | 误停 | seek 进特写 p90 | 逐片代价变化 |
|---|---|---|---|---|---|---|---|
| v1（现在） | 1407 | 93.3% | 16.0% | 2.4% | 6 | 9.0s | |
| `a1` | 1332 | 93.5% | 16.0% | 2.1% | 7 | 7.0s | v03 −75 |
| `bvid` | 1239 | 93.9% | 13.9% | 2.5% | 6 | 9.5s | v09 −151、v08 −11、v12 −7、v11 −2、v01 +3 |
| **`a1bvid`** | **1164** | **94.2%** | **13.9%** | **2.1%** | 7 | **7.2s** | 上面两行合起来 |
| `bstart` | 1204 | 94.1% | 12.5% | 2.8% | 6 | 13.0s | v11 +53、v01 +14 …… |

- 选定 `a1bvid`（`a1` + `bvid`）：代价 −17%，漏转 126→109 秒、误转 384→334 秒，准确率 +0.9 个百分点，seek 进动作段从不启动
  2→0 次。代价：v03 多一次误停（以前那段根本没转）；v01 +3 全在"剧情 → 口交"的拼接样本里（真实片子不变），按验收线的
  "最多 1 部变差且 ≤5%"算是擦边超了（+12%，绝对值 3 秒）。
- `bvid` 的收益主要来自 v09 一部；`a1` 的收益只有 v03 一部。两条在其余片子上都几乎不改变结果，风险小，但收益要靠以后的新片继续确认。
- 新片验证：标注 + 真机录 run（第 1、2 节），放进单独的目录（如 `bench/runs_test/`，不参与调参），用
  `audio_rescore.py check / rescore --runs bench/runs_test --tag test_audio_old` 算出 `.yamnet.npz`，然后
  `python3 bench/tools/cue_lab.py eval bench/gt bench/runs_test --npz bench/rescored/test_audio_old --preset a1b`。

## 7. 视频抽帧与预处理（video_rescore.py；app 2026-10-14 起 `VIDEO_PIPE=exact_crop`）

**2026-10-14 之前的两个问题**（视频模型的输入和训练时不一样，和 10-05 音频半速是同一类问题）：

1. **离线模式只取到关键帧**。旧 `VideoFrameExtractor.getFrameAt(T)` 每次 `seekTo(T, SEEK_TO_CLOSEST_SYNC)` 后直接拿解码器吐出的
   第一帧，也就是离 T 最近的关键帧（可能在 T 之后，最多差半个 GOP）。12 帧窗口里只有 1~2 张不同的图，视频结果每个 GOP 才变一次：
   v01/v02/v04/v06/v07 每 ≈4.2 秒冻住 1~1.5 秒，v10/v12/v13 每 ≈8.3 秒（x264 默认 keyint=250）冻住 4.5~4.8 秒，
   v08/v09/v11 每 ≈12 秒冻住 ≈8.5 秒（ffprobe 已确认 v08 的关键帧间隔是 12 秒）。而且比播放位置晚 0.5~1.3 秒：每次先 seek、
   只喂一帧就取最早的一个输出，解码器流水线里积压的是前几次请求的帧，换关键帧要等积压排空（从 runs_test“冻住→变化”的时间点推出：
   v10 525ms、v08 875ms、v11 ≈900ms、v09 ≈1.3 秒，各片不同，跟解码器要攒几帧才出帧有关）。融合仲裁按“新结果个数”计数，同一张
   关键帧的 9~17 个结果被当成了独立证据；现在的仲裁参数就是在这种输入上调出来的。
   网页模式（TextureView 截当前显示的帧）和在线模式（截屏）没有这个问题；下面第 2 个问题三种模式都有。
2. **三种模式都把整帧压扁成 160×160**。训练素材是 ffmpeg 短边缩到 224、居中裁 224×224，读帧时再 `cv2.INTER_AREA` 缩到 160；
   app 却把 16:9 的整帧横向压扁 1.78 倍（训练时裁掉的左右两边也送进了模型），离线模式还是 GPU 一步采样到 160（1080p 时每个输出
   像素只用到对应约 80 个原像素里的 4 个，有混叠）。

**修复**（2026-10-14 起的 app，参数行带 `VIDEO_PIPE=`）：
- `VideoFrameExtractor`：只在第一次、seek 之后、播放位置往回跳或往前跳太远时才 `seekTo(T, SEEK_TO_PREVIOUS_SYNC)` + flush；
  平时顺着往后解，pts 不到 T 的帧丢掉不渲染，取 pts 不晚于 T + 半帧的最后一帧（离 T 最近的那一帧）。每次最多 150ms，seek 进
  长 GOP 时分几次追上，追赶中不拿离目标太远的帧凑数。VideoView 的 seek 本身落在前一个关键帧上，所以通常不用追。
- 抽帧挪到单独的抽帧线程（以前在主线程，每 250ms 占约 43ms）；`EGLRenderer` 用 SurfaceTexture 的变换矩阵（含裁边、旋转），
  等 onFrameAvailable 并核对时间戳（以前 `sleep(20)` 赌帧已经到了）。
- `FramePreprocessor`（纯 Java，三种模式共用）：居中裁正方形 → 面积平均缩到 224 → 再缩到 160，与训练相同；离线模式先在 GPU 上
  把裁好的正方形画成 672×672（1080p 只缩 1.6 倍，双线性基本不混叠）。`FramePreprocessor.CENTER_CROP=false` 可改回整帧压扁
  （A/B 对比用）。面积缩小与 `cv2.INTER_AREA` 逐位相同（2 倍整数缩小时差 ≤1），`FramePreprocessorTest` 与 Python 版对拍。
- 评测 CSV：参数行加 `VIDEO_PIPE=exact_crop`（网页模式 `web_crop`；没有这一项的旧 run = keyframe 抽帧 + 整帧压扁），
  列加 `v_pos`、`v_fpts`（第 5 节）。

**视频模型的结构**（从 tflite 计算图读出来的）：12 份共享权重的 MobileNetV3Small（每帧 576 维）→ 时间平均 → 576×3 线性层 →
softmax。平均和线性层可以交换，所以窗口输出 = softmax(12 帧各自 logit 的平均)，与帧的顺序无关。`video_rescore.py` 直接读出
每帧的 576 维特征（模型里 PACK 的 12 个输入）算逐帧 logit：12 张不同的帧一次推理，窗口怎么组合都不用再跑模型（与模型输出差 < 1e-8）。

**video_rescore.py**（依赖：`pip install numpy av ai-edge-litert opencv-python-headless`，opencv 可选，只为快）：

```bash
python3 bench/tools/video_rescore.py keyframes --videos D:/xcup/videos                      # 每部片的 GOP、冻住的时长占比
python3 bench/tools/video_rescore.py check   --videos D:/xcup/videos --runs bench/runs_test  # 哪条管线能复现真机
python3 bench/tools/video_rescore.py rescore --videos D:/xcup/videos --runs bench/runs_test  # 按 exact_crop 重算，写新 run
python3 bench/tools/video_rescore.py --selftest
```

- 管线名 = 抽帧_预处理。抽帧：`keyframe`（旧）/ `exact`（新）；预处理：`legacy`（旧：整帧 GPU 一步到 160）/ `crop`（新）/
  `squash`（不裁，A/B）/ `train`（训练素材做法的近似：ffmpeg 同样的双三次缩放 + 居中裁、BT.601 转 RGB，不含 x264 重编码；只作参照）。
- `check`：旧 run（没有 `v_fpts`）默认比 `keyframe_legacy`，从出结果时刻往前推 12 帧（每 250ms 一帧），扫“最后一帧的取帧目标比出结果
  早 Δ”（0~3000ms；旧版因为解码器积压，Δ 在 0.5~1.3 秒，最佳值落在扫描边上会报警）；新 run 默认比参数行 `VIDEO_PIPE` 那条，按 `v_fpts`
  逐帧取（只扫 PyAV 与手机时间戳之间 ±2 帧的常数差）。
  2026-10-15 实测：13 部旧 run 的 `keyframe_legacy` 全部复现（MAE 0.003~0.016，Δ 0.6~1.4 秒），确认旧版只取到关键帧、而且滞后。
  真机验收（`runs_v3/v08_check.csv`，修复后的包把 v08 整部播完）：`exact_crop` MAE 0.0024；视频结果每分钟 95 个（与旧版相同），
  窗口里 12 帧全都不同，出结果比最后一帧晚中位 161ms（rescore 按 200ms 算）；`fusion_lab.py check` 逐 tick 100% 一致。
  打出 MAE（三类概率平均绝对差）、分组判定一致率，以及“12 帧全是同一张图”的窗口单独的 MAE（不受时间差影响，最能说明像素管线对不对）。
  定下的 Δ 写进 `bench/rescored/video_alignment.json`，`rescore` 直接用。`--pipes keyframe_legacy,exact_legacy` 可以加对照。
  `--matrix bt601|bt709` 试 YUV→RGB 矩阵（默认 auto：码流标了按标的，没标时高清按 BT.709，同 Android）。
- `rescore`：默认 `exact_crop`（修复后的 app），写到 `bench/rescored/video_<管线>_<runs 目录名>/`，同时写 `<run>.vframes.npz`
  （用到的每一帧的 pts 和逐帧 logit）。旧 run 的取帧时刻：`exact` 管线按修复后 app 的时序（最后一帧比出结果早 200ms，没有旧解码器的
  积压）；`keyframe` 管线（复现旧 app）用 check 扫出的 Δ；`--delta` 可以手动指定。之后用 `baseline.py` 按统一口径评测、对比（第 5 节；它也能把两个 runs 目录一次重算完）；换了视频输入，概率分布变了，
  比较前先按第 4 节重调阈值和融合参数。`--jobs N` 几部片并行；1080p 每个要用的帧约 30ms，一部 20 分钟的片子约 3 分钟。
- 局限：训练素材经过 x264 重编码（crf 23、224×224），`train` 管线没有模拟；`--model` 只支持与现用模型同一结构的模型；
  网页模式（TextureView 截帧）没有对应的离线管线。

### 在线模式（录屏）的画面（2026-10-19）

**问题**：训练素材是视频文件本身的居中正方形；在线模式截的是整个屏幕，取的是截图的居中正方形。只有横屏全屏播放、截图也是横屏
尺寸时两者才相同。2026-10-19 之前 `OnlineAnalysisService` 只在启动时读一次屏幕尺寸（截图是屏幕的一半），之后不跟着转屏：竖屏
打开在线分析、再转横屏全屏播放（最常见的用法）时，系统把横屏画面等比缩进竖屏尺寸的截图、上下补黑。2400×1080 的手机看 16:9
的视频，模型输入（160×160）里是这样的：

| 场景（管线） | 视频在模型输入里 | 与训练比 |
|---|---|---|
| 训练 / 离线模式（`exact_crop`） | 视频的居中正方形，占满 160×160 | — |
| 横屏全屏，截图跟着转屏（`exact_scrland`） | 同上 | 一致（只是截图是屏幕一半的分辨率） |
| 竖屏开始、转横屏全屏，截图还是竖屏尺寸（`exact_scrrot`，2026-10-19 之前） | 整幅 16:9 画面（含训练时裁掉的左右两边）缩成中间约 128×72，其余是黑的 | 画面缩到 0.45 倍，64% 是黑边 |
| 竖屏全屏播放（`exact_scrport`） | 整幅画面缩成中间 160×90，上下是黑的 | 缩到 0.56 倍，44% 是黑边 |

**修复**（app 2026-10-19）：
1. `OnlineAnalysisService`：截图尺寸跟着被录的画面变。Android 14+ 用 `MediaProjection.Callback.onCapturedContentResize`
   （"只共享一个应用"时也对），Android 13 及以下用 `DisplayListener` + `onConfigurationChanged`；先 `VirtualDisplay.setSurface`
   换成新尺寸的 `ImageReader` 再 `resize`，都在截图线程上做；初始尺寸用屏幕的实际尺寸（含导航栏）。`OnlineAnalysisActivity`
   看到截图尺寸变了就清空 12 帧窗口。
2. `ScreenCropper`（纯 Java）：先在截图里找出视频画面（去掉对称的纯黑边），再取它的居中正方形。竖屏全屏、横屏看竖屏视频、
   以及截图仍没跟上转屏的设备，都取回视频自己的居中正方形；横屏全屏看 16:9 时结果与不去黑边逐位相同。黑边不对称（画面上沿暗、
   贴着视频的控制条）不裁；亮着的范围没超出上次裁剩的范围（暗场景、黑边里的字幕）就沿用；片源自己带的对称黑边（宽银幕压在 16:9
   里）也会被裁掉。`ScreenCropper.ENABLED=false` 可关掉做 A/B。`ScreenCropperTest` 与 `video_rescore.py` 里的 Python 版逐帧对拍。

**离线量化**：`video_rescore.py` 的 `scr*` 预处理在电脑上模拟在线模式的截图（见文件头；屏幕默认 2400×1080，`--screen` 改），
只换视频结果，录像的 tick 时间轴、音频、融合都与统一基准相同，所以直接和 `e1d`（离线模式、app 现用模型）比：

```bash
for p in scrland scrrot scrport scrrotbar scrportbar; do
  python3 bench/tools/baseline.py rescore --videos D:/xcup/videos --pipe exact_$p --jobs 4    # 每个和重算 e1d 差不多久
done
python3 bench/tools/baseline.py compare e1d exact_scrland exact_scrrot exact_scrrotbar exact_scrport exact_scrportbar
python3 bench/tools/frame_diag.py --name e1d,exact_scrland,exact_scrrot,exact_scrport,exact_scrportbar
```

- 不带 `--model` 就是 app 里的模型（E1d），变体名默认就是管线名。PowerShell 里循环写成
  `foreach ($p in 'scrland','scrrot','scrport','scrrotbar','scrportbar') { python bench/tools/baseline.py rescore --videos D:/xcup/videos --pipe "exact_$p" --jobs 4 }`。
- 预期：`exact_scrland` ≈ `e1d`；`exact_scrrot` / `exact_scrport` 量出缩小加黑边让视频 AUC 和端到端差了多少；`…bar` 取回多少
  （截图分辨率低、缩得多时 GPU 双线性有混叠，可能比 `exact_scrland` 略差）。`exact_scrlandbar` 与 `exact_scrland` 逐位相同，不用算。
- 局限：没有模拟播放器的界面（状态栏、控制条、竖屏不全屏时周围的页面），视频以外全是纯黑；视频按 SurfaceFlinger 一次双线性合成进
  截图（`gl_linear`），用 TextureView 的播放器中间多一次缩放没有模拟。在线模式不写评测 CSV，这些数字还没有真机对照；真机上用 logcat
  看 `[录屏]`（转屏后的截图尺寸）和 `[视频线程] 截图 … 里的视频画面`（去黑边的结果）。

## 时间线

- 2026-10-05：发现离线模式喂给音频模型的是半速音频（`AudioDecoder` 没按声道下混），用 `PcmResampler` 修复；之前录的 run 的音频结论作废。
  证据（6 部片的复现误差、修复前后的 AUC）见提交 `20f08f9` 里本文件的"音频解码修复"一节。
- 2026-10-06：删除证据融合 `EvidenceFusion`，app 回到旧融合；`fusion_lab.py` 里保留作离线参照。
- 2026-10-07：旧融合（投票窗 + `applyDoStartGate`）换成融合仲裁 `FusionArbiter`，旧代码从 app 删除；run CSV 不再有 `smoothed` 列。
- 2026-10-08：用融合仲裁的包（app 1.2.5）重录 7 部片，`runs_initial` 成为基线；旧融合时期的 `runs_fixed` 删除。
- 2026-10-09：评估新音频分类头（训练数据更平衡）。转 vs 剧情 AUC 0.888 → 0.921，剧情判对 59% → 82%，但动作中有对白或
  背景音乐时更容易判成剧情；融合参数各自重调后代价只低约 5%（7 部片里 4 部好、3 部差），不显著，暂不采用，数据留在
  `rescored/audio_new`。`rescored` 下 10 月 5 日排查解码时的 `clean` / `fixed` / `phone` 删除（git 历史里可找回）。
- 2026-10-10：YAMNet 线索离线实验（第 6 节，`cue_lab.py`）。A2、A3、B=all、B2、P 不成立；剩下候选 `a1`、`bstart`、`a1b`，
  app 还没改，等新片验证后再定上哪条。`fusion_lab.Tick` 多了 `ays`（音频结果的 YAMNet 分组分数），`arbiter_lab` 的 seek /
  慢手机 / 合成场景会把它带上；没有分数时一切照旧。
- 2026-10-11：加新片 v08~v13（`runs_test`）。v10/v12/v13 的音轨时间戳比样本数快 0.5~0.9%，离线重算越播越偏（到片尾差约 6 秒），
  `audio_rescore.py` 的 `fixed` 管线改成按每帧 pts 分段（与手机 `AudioDecoder` 一致）。`cue_lab.py` 加 `--only`、候选 `bvid` /
  `a1bvid`，run / npz 目录可以逗号分隔给几个。13 部片上推荐 `a1bvid`（第 6 节），等确认后改 app（阶段 2）。
- 2026-10-11：app 上 `a1bvid`（阶段 2）。`AudioInferenceHelper` 同一次推理多取 YAMNet 第 0 个输出；新增 `AudioCue`；
  `FusionArbiter` v2（`start_video`、对白否决，线索未知时与 v1 相同）；两个 Activity 把线索传给仲裁、显示在音频行；
  `BenchLogger` 加 `a_cue`、`a_cue_*` 列和 `CUE=` 参数。`fusion_lab.py` 同步（check 核对线索、golden 带线索）、
  `audio_rescore.py` rescore 时重写线索列；对拍数据重新生成，新增 `AudioCueTest`。要用新包重录验证（第 3 节"上真机验证"）。
- 2026-10-14：发现离线模式视频模型只看到关键帧（`VideoFrameExtractor` 用 `SEEK_TO_CLOSEST_SYNC` 取帧），以及三种模式都把整帧压扁
  （训练是居中裁剪）。app 修复：顺序解码取当前帧、抽帧挪到单独线程、`FramePreprocessor` 与训练一致（第 7 节）；评测 CSV 加
  `VIDEO_PIPE` 参数和 `v_pos`、`v_fpts` 列。新增 `video_rescore.py`（离线视频重算）和 `FramePreprocessorTest`。之前录的 run 的视频
  结论（含融合仲裁的参数）都是在只看到关键帧的输入上得到的。
- 2026-10-15：统一评测基准 `baseline.py`（第 5 节）：13 部片和录像、音频线索、融合规则、代价固定，只换视频结果。13 部按修复后的
  管线离线重算（`rescored/video_exact_crop_*`）：视频 AUC 0.913 → 0.935，现有参数下端到端代价 1164 → 1219。真机验收通过（`runs_v3`）。
- 2026-10-16：视频模型 P1（训练方法）的工具：`training/`（素材检查、新的训练脚本 `train_video.py`，见 `training/README.md`），
  bench 端 `frame_diag.py`（逐帧诊断）和 `retune.py`（留一视频重调视频阈值）。`exact_crop` 的参照报告在 `bench/baseline/`
  （`frame_diag_exact_crop.txt`、`retune_exact_crop_conf.txt`）。
- 2026-10-18：P1 的 4 个候选在 13 部片上比完（`training/RECIPE.md`）：配方定为 E1d。`retune.py` 的验收线
  对 seek 启动留容差（中位数 / p90 0.1 秒、特写 0.5 秒）：不同模型的结果时刻差几毫秒，零容差会误判"变慢"；`frame_diag.py`
  按逐片 AUC 数两半各有几部变好 / 变差（只看合并 AUC 时，v08~v13 本来接近 1，会误报"只在一半上变好"）。
  只留 `exact_crop` 和 `e1d` 的重算数据（E1a~E1c 的在 git 历史 `aa9defc` 里）。
- 2026-10-18：app 的视频模型换成 E1d，阈值、融合参数不变（第 5 节）。`baseline.py check` 加 `--model`：`runs_initial` / `runs_test`
  是旧模型录的，核对要用旧模型。P1 用完的文件删除（git 历史 `e8848b0` 里可找回）：E1a~E1c 的训练记录、报告和
  `bench/baseline/e1{a,b,c}.json`；P1 的对比报告只留 exact_crop 对 e1d 的（`compare_` / `frame_diag_` / `retune_compare_exact_crop_vs_e1d*`）。
- 2026-10-19：在线模式（录屏）送进视频模型的画面与训练不一致：截图尺寸只在启动时定一次，竖屏开始、转横屏后系统把横屏画面缩进
  竖屏尺寸的截图、上下补黑，模型看到的是缩成中间一条的整幅画面；竖屏全屏播放时也是上下黑边（第 7 节"在线模式（录屏）的画面"）。
  app：截图尺寸跟着转屏（`OnlineAnalysisService`），新增 `ScreenCropper` 去黑边后再居中裁；`video_rescore.py` 加在线模式的
  `scr*` 预处理，`golden` 同时生成 `ScreenCropperTest` 的对拍数据；`baseline.py rescore` 加 `--screen`。13 部片上的量化和真机验证待做。
