# bench/ — Xcup 评测基准

离线回放评测：每改一个参数或一个模型，都在同一批片子上量化对比，而不是凭感觉调。

## 目录

```
bench/
  ANNOTATION_SPEC.md      人工真值标注规范（改动需递增 spec_version）
  gt/<id>.csv             人工真值（v01~v07）
  runs_initial/<id>_*.csv 真机 dump：app 1.2.5、融合仲裁、现用音频分类头，7 部片（当前基线）
  rescored/               audio_rescore.py 的输出：audio_old = runs_initial 用现用分类头离线重算（与真机一致，
                          另带 YAMNet 521 类分数 .yamnet.npz）；audio_new = 2026-10-09 评估过的新分类头（未采用）
  videos/                 原始视频（不进 git；也可以放别处，用 --videos 指定）
  summary.csv             metrics.py --append 的汇总
  tools/
    labeler.html          真值标注器，双击即用，纯本地无上传
    metrics.py            单部片：指标、误判定位、单模态诊断
    fusion_lab.py         融合层回放：check（复现真机）、eval（策略对比）、oracle（模态上限）、golden（Java 对拍数据）
    arbiter_lab.py        融合仲裁评测：逐片最差情况、seek 模拟、压力场景、慢手机、留一选参
    audio_rescore.py      音频离线重算：换音频模型不用上真机
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
分析线程判出的动作（`"do"` / `"Noise"` / `""`）、置信度和结果产出时刻，输出交给 `updateBluetoothState`（蓝牙状态机没有改）。
旧融合（投票窗 `smoothedFusion` + 快通道 + `selectBestAction`、启动仲裁 `applyDoStartGate`）已从 app 删除，
`fusion_lab.py` 里留有 Python 版，作"旧融合"对照、回放旧 run。证据按"新推理结果"计数：同一个结果在几个 tick 里
重复出现只算一次，产出超过 2 秒的不记。

| 阶段 | 规则 |
|---|---|
| 启动 | 满足任一条：<br>• 两路佐证：3 秒内视频强"转"（≥0.85）≥2 个、音频强"转"（≥0.90）≥1 个，并满足比维持更严一档的条件（视频 6 秒内"转"占比 ≥80%，或音频最近 4 个全是"转"），防止开关抖动；<br>• 音频兜底：9 秒内音频结果 ≥4 个且**全部**是"转"（不看视频，特写时也能启动；零容忍挡住剧情里"转"中夹"不转"的音频）。 |
| 维持 | 任一路有支持：音频最近 4 个里 ≥3 个"转"；或视频最近 4 个里 ≥2 个"转"且 6 秒内占比 ≥70%。维持时主动交 `"do"`，取消状态机里待执行的停转。 |
| 停止 | 两路都没有支持时：刚 reset 的 6 秒内某一路结果还不到 4 个，先等；4 秒内有"不转"就交 Noise（之后照旧要状态机 1 秒稳定 + 2 秒最短驻留）；看不清 / 听不清 / 过期最多再等 6 秒（两路结果完全中断时约 16.5 秒停）。 |

- **reset**：seek、暂停后恢复时 `reset(now)`；静音后恢复（网页视频模式的 VAD、在线模式）`reset(now, 2000)`，恢复后 2 秒内的
  结果也不算（音频窗口里还混着静音）；墙钟回跳自动 reset。run 的 `arb_reset` 列记下每次 reset，离线回放照做。
- **暂停**：离线 / 网页视频模式下，只有"有证据撑着"的 do（原因 `start_*` / `hold_*`）才挂起、恢复后接着转，只是在等结果时
  同步成 Noise（否则网页模式下"静音暂停 → 一声短响恢复 → 又静音暂停"会让马达跟着每声短响一直转）。在线模式暂停一律同步成 Noise。
- 每个 tick 的原因写进 `arb` 列：`start_both` / `start_audio` / `hold_audio` / `hold_video` / `hold_both` / `stop_neg` /
  `stop_unclear` / `wait_sparse` / `wait_unclear` / `off_neg` / `off_wait`。

### 当前效果（runs_initial 7 部片，离线回放）

| | 准确率 | 误转 | 漏转 | 动作中误停 | v03 漏转 | v04 / v07 误转 | 停止后还转 | 启动后没转 |
|---|---|---|---|---|---|---|---|---|
| 旧融合 | 87.8% | 34.4% | 3.7% | 24 次 | 37.1% | 74.2% / 68.6% | 13.5s | 0.1s |
| 融合仲裁 | 92.6% | 18.7% | 3.0% | 4 次 | 26.5% | 43.6% / 55.8% | 3.2s | 2.6s |

- 规则和参数是看着这 7 部片定的（样本内）；留一视频选参后是 92.6% / 19.0% / 3.0%。真正的验收要靠新片子。
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
   就是离线评估的那套。
3. `python3 bench/tools/arbiter_lab.py report bench/gt <新 runs 目录>` 看逐片结果；`metrics.py --at` 的"仲裁原因"列能看出
   某次停转是哪条规则触发的。

## 4. 换模型后怎么调参

先改哪个：`fusion_lab.py oracle`（把一路换成真值）在融合仲裁下，音频完美时准确率 92.6% → 99.4%（代价 985 → 175），
视频完美时 → 96.3%（代价 572）。改音频收益最大；但视频单独就能维持马达，剧情里视频的长时间误判（v07）只能靠改视频模型。
音频接近完美后，启动规则（约 3 秒）会变成瓶颈，那时要放宽启动参数。

1. **拿到新模型的数据**
   - 音频模型：`audio_rescore.py rescore --classifier new.tflite` 在电脑上重算，输出目录直接给下面的工具用，不用上真机。
   - 视频模型：装新包在真机上重录（目前没有离线重算视频的工具）。
   - 类别顺序（视频 `[plot, oral, sex]`、音频 `[sex, oral, noise]`）或类别数变了，要同时改两个 Activity 里的判定、`BenchLogger`
     的列和 bench 工具。
   - **别拿 bench 这几部片训练**，否则这里的数字就不再能代表新片子。
   - 新旧模型要在各自调好的参数下比：换了模型，概率分布就变了（阈值、`A_FLOOR` 等都要跟着变），只用现有参数比不公平。
     rescore 时用 `--tag` 给每个模型起单独的目录，免得互相覆盖。
2. **先定各路的阈值**：`metrics.py --diag` 看每一路的 AUC 和阈值扫描，定 `VIDEO_DO_PROB_THRESHOLD` / `VIDEO_PLOT_PROB_THRESHOLD` /
   `AUDIO_DO_PROB_THRESHOLD` / `AUDIO_NOISE_PROB_THRESHOLD`（`VideoProcessActivity` 和 `OnlineAnalysisActivity` 里各一份）。
   这些阈值在采集时就判定好了，改了要重录（音频也可以改 run 文件头里的 `AUDIO_DO` / `AUDIO_NOISE` 再 rescore）。
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
   逐行一起改），然后 `python3 bench/tools/fusion_lab.py golden bench/gt <runs 目录>` 重新生成
   `app/src/test/resources/fusion/arbiter_golden.csv`，跑 `FusionArbiterTest` 和 `fusion_lab.py --selftest`。
6. **真机复验**：同第 3 节。

`FusionArbiterTest` 在 Android Studio 里直接跑，或 `gradlew :app:testDebugUnitTest`。没有 Android SDK 时用 JDK + JUnit：

```bash
CP=junit-4.13.2.jar:hamcrest-core-1.3.jar
javac --release 11 -encoding UTF-8 -cp $CP -d /tmp/arbtest \
  app/src/main/java/com/example/helloworld/FusionArbiter.java app/src/test/java/com/example/helloworld/FusionArbiterTest.java
java -cp /tmp/arbtest:app/src/test/resources:$CP org.junit.runner.JUnitCore com.example.helloworld.FusionArbiterTest
```

## 5. 工具参考

### fusion_lab.py

```bash
python3 bench/tools/fusion_lab.py check  bench/gt bench/runs_initial   # 回放能否逐 tick 复现真机（≥99% 才可信）
python3 bench/tools/fusion_lab.py eval   bench/gt bench/runs_initial   # 策略对比：旧融合 / 融合仲裁 / 只用视频 / 只用音频 / 一直转 / 证据融合
python3 bench/tools/fusion_lab.py oracle bench/gt bench/runs_initial   # 模态上限：假设视频（或音频）完美，最多能到多少
python3 bench/tools/fusion_lab.py golden bench/gt bench/runs_initial   # 生成 Java 对拍数据
```

- 回放按 run 的参数行选融合：有 `ARB=` 的用融合仲裁，否则用旧融合；之后都接蓝牙状态机的复刻。
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
  另写 YAMNet 自带的 AudioSet 类别分数：`a_ys_*` 列（10 类）和同名 `.yamnet.npz`（全部 521 类，按结果时间戳对齐），只做离线分析。
- 管线 `fixed` = 现在 app 喂给模型的音频（下混 + `PcmResampler`，与 Java 逐点对拍）；`phone` = 2026-10-05 之前的半速音频；
  `clean` = 训练数据的处理方式。
- 视频按真值文件头的 `file_name` 在 `--videos`（默认 `bench/videos/`）里找，或 `--pair v01=D:/xcup/视频1.mp4`；`--only v01,v04`
  只跑几部。

### 自检

```bash
python3 bench/tools/fusion_lab.py --selftest
python3 bench/tools/metrics.py --selftest      # 含与 BenchLogger 的 CSV 列名契约检查
python3 bench/tools/audio_rescore.py --selftest
node bench/tools/test_labeler.js
```

### run CSV 里后来加的列

| 列 | 含义 |
|---|---|
| `a_pos` | 该音频结果读 PCM 时的播放位置，模型窗口 = `[a_pos-2s, a_pos]` |
| `arb` | 融合仲裁本 tick 的原因（见第 3 节）。`gated` 是融合仲裁的输出。旧融合时期的 run 还有 `smoothed` 列（投票窗），新 run 没有 |
| `arb_reset` | 这一行之前的 reset：`<ms>[+<预热ms>][:seek]`。回放时同样 reset，带 `:seek` 的还清空蓝牙状态（旧融合回放时也清空投票窗） |
| `a_ys_*` | AudioSet 类别分数（Speech、Music、Moan 等，见 `metrics.YS_CLASSES`），只有 `rescore` 写 |

## 时间线

- 2026-10-05：发现离线模式喂给音频模型的是半速音频（`AudioDecoder` 没按声道下混），用 `PcmResampler` 修复；之前录的 run 的音频结论作废。
  证据（6 部片的复现误差、修复前后的 AUC）见提交 `20f08f9` 里本文件的"音频解码修复"一节。
- 2026-10-06：删除证据融合 `EvidenceFusion`，app 回到旧融合；`fusion_lab.py` 里保留作离线参照。
- 2026-10-07：旧融合（投票窗 + `applyDoStartGate`）换成融合仲裁 `FusionArbiter`，旧代码从 app 删除；run CSV 不再有 `smoothed` 列。
- 2026-10-08：用融合仲裁的包（app 1.2.5）重录 7 部片，`runs_initial` 成为基线；旧融合时期的 `runs_fixed` 删除。
- 2026-10-09：评估新音频分类头（训练数据更平衡）。转 vs 剧情 AUC 0.888 → 0.921，剧情判对 59% → 82%，但动作中有对白或
  背景音乐时更容易判成剧情；融合参数各自重调后代价只低约 5%（7 部片里 4 部好、3 部差），不显著，暂不采用，数据留在
  `rescored/audio_new`。`rescored` 下 10 月 5 日排查解码时的 `clean` / `fixed` / `phone` 删除（git 历史里可找回）。
