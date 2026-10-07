# bench/ — Xcup 评测基准

离线回放评测流水线。目标：每改一个参数都能量化对比，而不是凭感觉调。

## 目录

```
bench/
  ANNOTATION_SPEC.md      人工真值标注规范（改动需递增 spec_version）
  gt/<video_id>.csv       人工真值（提交进 git）
  runs/<video_id>_*.csv   真机 dump（提交进 git：离线实验的结论要能从仓库复现）
  rescored/               audio_rescore.py 的输出：对齐结果 alignment.json + 离线重算的 run CSV
  videos/                 原始视频（不进 git，放这里命令最短；也可以放别处用 --videos 指定）
  tools/
    labeler.html          真值标注器，双击即用，纯本地无上传
    test_labeler.js       标注器核心逻辑测试：node bench/tools/test_labeler.js
    metrics.py            单部片的指标 / 误判定位 / 模态诊断
    fusion_lab.py         融合层离线实验台：回放、交叉验证、模态上限、融合仲裁的 Java 对拍数据
    arbiter_lab.py        融合仲裁的离线评测：逐片最差情况、seek 模拟、压力场景、慢手机、留一选参
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

## 融合仲裁 FusionArbiter（2026-10-07 起 app 在用）

`applyDoStartGate` 换成了 `FusionArbiter.java`（离线 / 网页视频 / 在线三种模式共用），把"启动 / 维持 / 停止"分开判断。
旧的投票窗 `smoothedFusion` 还在算，但只写进 run CSV 的 `smoothed` 列作对照，不再参与决策。

**为什么改。** 7 部片上，动作中 22 次误停全部是"只有一个模态说剧情"：
- 13 次是视频遇到特写 / 遮挡 / 3P 判剧情，音频一个 1~2 秒的 noise 就把马达停了；
- 9 次是视频稳稳判动作，音频听不到目标声音（v03）。

两个模态同时判剧情的一次都没有。剧情里的误转主要是音频单独判"转"：剧情 tick 里音频判"转" 40%，视频 15%。
旧融合把"音频更可信"写死（投票权重 1.2 / 0.8，音频 ≥0.75 直接启动），两头都吃亏。

**规则。** 证据一律按"新推理结果"计数：同一个结果在几个 500ms tick 里重复出现只算一次（按结果产出时刻去重），
产出时已超过 2 秒的不记。维持按"最近 n 个结果"计数（这几个结果都要在 12 秒内），手机推理慢一些也不会失效。

| 阶段 | 规则（括号里是参数） |
|---|---|
| 启动（马达不转） | 满足任一条：<br>• 两路佐证：3 秒内视频强"转"（≥0.85）≥2 个、音频强"转"（≥0.90）≥1 个，**并且此刻满足比维持更严一档的条件**：视频 6 秒内"转"占比 ≥80%，或音频最近 4 个全是"转"。启动比维持严（迟滞），防止"一启动就停"：证据在维持门槛上下晃时，不加它马达会每几秒开关一次。<br>• 音频兜底：9 秒内音频结果 ≥4 个且**全部**是"转"（夹一个"不转"或 unclear 都不算）。兜底不看视频（特写时视频判剧情，不能永远不启动），但零容忍，正好挡住剧情里"转"中夹"不转"的音频。 |
| 维持（马达在转） | 满足任一条：<br>• 音频最近 4 个结果里 ≥3 个"转"；<br>• 视频最近 4 个里 ≥2 个"转"，且 6 秒内视频"转"占比 ≥70%。<br>一个孤立结果撑不住；交替出现的零星"转"也撑不住。 |
| 停止 | 两路都没有支持时：<br>• 刚 reset（seek / 恢复播放）的 6 秒内，某一路的新结果还不够 4 个，先等，不停。否则恢复播放时恰好在特写，视频一个"剧情"就会在音频攒够结果之前把马达停掉。只在刚 reset 时这样等：平时某一路变慢或断掉，不能让另一路时有时无的支持把马达一直留着；<br>• 4 秒内有任一"不转"结果就交 Noise（状态机照旧要 1 秒稳定 + 2 秒最短驻留）；<br>• 看不清 / 听不清 / 结果过期，最多再等 6 秒（从最后一次有支持算起），然后停。最近的支持要先滑出窗口，所以两路结果完全中断时实际约 16.5 秒才停；旧融合在这种情况下会一直转。 |

另外几条：
- 维持时主动交 `"do"` 而不是 `""`：这样 `updateBluetoothState` 里尚未执行的停转计时会被取消，之后的 Noise
  要重新稳定 1 秒才停。交 `""` 会保留旧计时，下一次 Noise 立刻停转（`fusion_lab.py --selftest` 第 10 项）。
- reset：seek、网页模式 seek、暂停后恢复时 `reset(now)`，此前产出的结果一律不算。静音暂停后恢复（网页视频模式的 VAD、
  在线模式）用 `reset(now, 2000)`：恢复后 2 秒内产出的结果也不算，那时 2 秒的音频窗口里还混着暂停前的静音。
  上面"刚 reset 的 6 秒"从 reset 时刻算，不从预热结束算。
- 离线 / 网页视频模式暂停时，马达只有在仲裁器上一个 tick 有证据（`start_*` / `hold_*`）时才挂起、恢复时接着转；
  只是在"等结果"（`wait_sparse` / `wait_unclear`）时直接同步成 Noise，恢复后从头判断。否则网页视频模式下
  "静音暂停 → 一声短响恢复 → 还在等结果时又静音暂停"会让马达跟着每一声短响一直转（模拟：剧情里每 9 秒一声短响，
  120 秒里转 68 秒 → 5 秒）。动作中暂停、定格画面仍判"转"时照常挂起，恢复那一刻马达就转。
- run CSV 的 `arb_reset` 列记下 reset：`<ms>[+<预热ms>][:seek]`。离线回放同样 reset；带 `:seek` 时还会清空蓝牙状态机和
  旧投票窗（app 在 seek 时也是这样做的）。
- 墙钟往回跳时自动 reset。注意只有仲裁器会 reset：蓝牙状态机的计时也用墙钟，往回跳一大步时会卡住相应的时长，
  那是原有的问题，这次没改。
- 动作串、置信度和产出时刻作为一个对象原子发布（`ResultSnap` / `FusionArbiter.Result`）。否则融合线程可能读到
  "旧动作 + 新时间戳"，去重时把真正的新结果丢掉。
- 在线模式暂停时，如果马达在转，先发一次 Noise 并把状态机同步成 Noise，恢复后从头开始判断。

**效果**（`bench/runs_fixed` 7 部片，`arbiter_lab.py report` / `lovo` / `fusion_lab.py eval`）：

| | 准确率 | 误转 | 漏转 | 误停/分 | v03 漏转 | v04 误转 | v07 误转 | 停转样本 ≥40s | 停止后还转（中位） | 启动后没转（中位） | seek 进特写/低声音 p90 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 旧融合 | 88.1% | 32.9% | 3.9% | 0.28（22 次） | 39.2% | 70.5% | 65.7% | 2 | 13.0s | 0.6s | 5.0s |
| 融合仲裁（app 参数，样本内） | 93.2% | 17.8% | 2.6% | 0.10（8 次） | 19.9% | 36.7% | 54.3% | 2 | 2.9s | 3.4s | 12.0s |
| 留一视频，只重选 4 个参数 | 93.1% | 18.1% | 2.6% | 0.10（8 次） | 19.9% | 37.0% | 55.6% | 2 | 3.1s | 3.7s | 12.0s |
| 留一视频，连音频兜底窗口一起选 | 93.3% | 17.1% | 2.7% | 0.09（7 次） | 19.9% | 33.7% | 53.7% | 2 | 3.1s | 3.7s | 12.5s |

数字的口径：
- 规则结构（有哪几条、按结果计数、零容忍兜底、迟滞等）是看着这 7 部片定的，所以**没有一行是真正的样本外**；
  留一视频只能说明参数不是硬凑的。真正的验收要靠没参与过的新片子。
- 留一选参时每个参数选中 app 取值的折数：`CONS_WIN=6000` 7/7，`V_CONS=70` 6/7，`V_FLOOR=0.85` 6/7，`MV_N=4` 5/7。
  音频兜底窗口单独放开选时，5 折选 12 秒、2 折选 10 秒，没有一折选 app 用的 9 秒（原因见下）。

**代价和没解决的。**
- 启动要"持续的新证据"，比旧融合慢：
  - 拼接样本里"启动后没转"中位数 0.6s → 3.4s；
  - seek 进特写 / 低声音的动作段，p90 5s → 12s（这时只能靠 9 秒音频兜底）。
- 逐片看有变差的：
  - v07 漏转 1.8% → 3.8%，最长连续漏转 2.5s → 12.0s（1003~1026s：视频判剧情、音频"转"里夹"不转"，和 v06 剧情的特征一样，融合分不开）；
  - v04 最长连续漏转 2.0s → 4.5s，v05 2.3s → 5.3s（漏转 1.4% → 1.9%）；
  - v07 最长连续误转 73s → 104s。
- 真实停转延迟（只有 v03 的 3 个样本）：[3.0, 5.0, −1.3]s → [3.0, 6.0, −0.8]s。
- 剩下的 8 次误停：3 次在 v03，两个模型最近的结果都已经判"不转"；其余 5 次（v04 / v05 各 1 次、v07 3 次）是视频判剧情
  或看不清，同时音频最近 4 个结果里只剩 2 个"转"（另外两个是"不转"或 unclear）。再放宽就会把 v06 / v07 的剧情一起留住。
- 两个 ≥40s 的停转样本（v07#停2 / #停3）来自 v07 剧情 93~228s：两个模型都判"转"，融合救不了，旧融合也一样。
  v03 剩下的漏转（415~437s 口交等）两个模型都判剧情，同样是模型问题。
- 视频单独就能维持（音频听不到目标声音时靠它），所以 `fusion_lab.py oracle` 里"音频完美"也压不住视频在剧情里的
  长时间误判（还剩 1 个 ≥40s 的停转样本）。融合仲裁下，视频模型的剧情误判比以前更值得修。
- 模拟慢手机（把结果抽稀）：视频结果间隔 ×2 / ×3 / ×4 时准确率 93.3% / 93.2% / 93.4%，最差片漏转 19~21%，基本不受影响。
  音频间隔 ×2 时 91.2%；×3 时总体 93.2%，但 9 秒里凑不齐 4 个音频结果，兜底基本不触发，v03 漏转回到 36%（旧融合 32%）。
  目标手机上音频结果如果真慢到 3 秒一个，兜底窗口要按实际结果间隔放宽。

**怎么定的。** 三种结构做过对比：
- 在旧投票上加门：在慢手机上失效，视频结果间隔 ×2 时 v03 漏转 58%；
- 纯证据计数、按时间窗计数：遇到交替出现的假"转"永远不停，音频间隔 ×2 时漏转 30%；
- 一致性检查：误转较高。

现在这版取"纯证据 + 按最近 n 个结果计数 + 视频一致性"，之后按评审意见加了三处：
- 启动迟滞：启动不加额外条件时，两路强"转""不转"交替会让马达每分钟开关约 30 次；只要求"此刻满足维持条件"时，
  证据在维持门槛上下晃仍会每分钟开关 16 次（`stress` 最后三个场景）；
- "等结果"只在刚 reset 的 6 秒内：否则视频早就断了时，音频时有时无的支持会把马达一直留着（`stress` 的
  "视频 20 秒前就断了"场景：不限时永远不停，现在 6 秒停）；
- 维持窗口 8 秒 → 12 秒：音频间隔 ×3 时准确率 82.9% → 93.2%、最差片漏转 47% → 36%，正常速度下数字完全一样；
  代价是两路结果完全中断时，停转从 12.5 秒变成 16.5 秒。

音频兜底窗口取 9 秒而不是代价略低的 10 / 12 秒：12 秒能把 v06 误转压到 7%，但 v07 最长连续漏转会从 12s 变成 26s（10 秒也有 24s），
而"动作中不转"是第三方测试最在意的问题。

```bash
python3 bench/tools/arbiter_lab.py report    # 旧融合 vs 融合仲裁：逐片、最长连续漏转、拼接样本、seek 模拟、真实停转延迟
python3 bench/tools/arbiter_lab.py stress    # 合成压力场景（交替假"转"、结果过期、遮挡、音频听不到、视频断了）+ 模拟慢手机
python3 bench/tools/arbiter_lab.py lovo      # 留一视频选参（--wide 连音频兜底窗口一起选）
```

**上真机验证。** 用新包按"采集"一节重录这 7 部片（最好再加几部新片，规则和参数都没在它们上面调过），然后：
1. `fusion_lab.py check bench/gt <新 runs 目录>`。参数行带 `ARB=v1` 的 run 自动用融合仲裁回放（遇到 `arb_reset` 同样 reset），
   并比对新增的 `arb` 列。`arb` 列是每个 tick 的原因：`start_both` / `start_audio` / `hold_audio` / `hold_video` /
   `hold_both` / `stop_neg` / `stop_unclear` / `wait_sparse` / `wait_unclear` / `off_neg` / `off_wait`。
   `gated` 和 `arb` 的一致率都 ≥99% 才说明手机上跑的就是离线评估的那套。新 run 的置信度按完整 float 写出，
   不再有 4 位小数造成的边界翻转。
2. `metrics.py ... --errors` / `--at`：`--at` 多了一列"仲裁原因"，能直接看出某次停转是哪条规则触发的。

改规则或参数时，Java（`FusionArbiter.java`）和 Python（`fusion_lab.py` 的 `FusionArbiter` / `ARB_DEFAULTS`）要一起改，
然后 `python3 bench/tools/fusion_lab.py golden bench/gt bench/runs_fixed` 重新生成
`app/src/test/resources/fusion/arbiter_golden.csv`（真实片段 + 边界用例 + 固定种子的随机序列），
跑 `FusionArbiterTest` 确认两边逐 tick 一致（`fusion_lab.py --selftest` 也会用 Python 回放一遍这份数据）。
没有 Android SDK 时可以直接用 JDK + JUnit 跑：

```bash
CP=junit-4.13.2.jar:hamcrest-core-1.3.jar
javac --release 11 -encoding UTF-8 -cp $CP -d /tmp/arbtest \
  app/src/main/java/com/example/helloworld/FusionArbiter.java app/src/test/java/com/example/helloworld/FusionArbiterTest.java
java -cp /tmp/arbtest:app/src/test/resources:$CP org.junit.runner.JUnitCore com.example.helloworld.FusionArbiterTest
```

## 融合层实验台 fusion_lab.py

run CSV 里存着两个模型每个 tick 的原始概率，融合层的输入全在。`fusion_lab.py` 把融合层整个
搬到 Python 里重放：改融合/扫参数不用再实时重跑视频，一部片几秒。

先把每部片的 run CSV 拷进 `bench/runs/`，文件名以真值 id 开头（`v01_base.csv` 配 `gt/v01.csv`）。

```bash
# 1. 回放器可信吗：复刻线上 Java 融合（参数行有 ARB= 的用融合仲裁，否则用旧的投票窗 + 启动仲裁；
#    之后都接蓝牙状态机），与 dump 逐 tick 比对
python3 bench/tools/fusion_lab.py check

# 2. 留一视频交叉验证：旧融合 vs 融合仲裁 vs 证据融合 vs 一直转 / 只用视频 / 只用音频
python3 bench/tools/fusion_lab.py eval

# 3. 模态上限：假设视频（或音频）完美，端到端最多能到多少 —— 决定先改哪个模型
python3 bench/tools/fusion_lab.py oracle

# 4. 改了融合仲裁的规则 / 参数后，重新生成 Java 对拍数据（FusionArbiterTest 用）
python3 bench/tools/fusion_lab.py golden bench/gt bench/runs_fixed
```

Windows 下把 `python3` 换成 `python`（或 `py`）、路径用 `\` 也可以。`eval` / `oracle` 的留一拟合默认按 CPU 核数
多进程并行（`--jobs 1` 改回串行，结果完全相同）；参数搜索默认"先粗后细"，`--exhaustive` 改扫全网格
（慢约 9 倍，结果基本一致）。

视频证据的权重 `w_v` 也在搜索范围里（1 / 0.5 / 0.25 / 0，0 = 只用音频）：音频修好之后，剧情里有裸露的片子
上"少信视频"明显更好，固定 `w_v=1` 搜不到。旧数据上它的留一代价 1448，略好于固定 `w_v=1` 的 1492。

`check` 一致率要 ≥99% 才说明回放器可信，后面的对比才有意义。少量不一致来自旧 run 把置信度
存成 4 位小数（2026-10-07 起改存完整 float）、以及蓝牙状态机内部取时间比 `wall_ms` 晚几毫秒，正好卡在阈值边界上时会翻转。
app 带证据融合那几版在 live 模式下录的 run（参数行里 `FUSION=live/<配置 id>`）是证据融合驱动的蓝牙，要加
`--config <当时的配置>` 才能比对；配置已从 app 删除，按配置 id 在 `git log -p -- app/src/main/assets/fusion_config.json`
里找。

### 证据融合是什么

原融合把"音频更准、音频权重 1.2、音频可直通启动"写死在代码里；bench 显示这个前提不成立。
证据融合（现在只在 `fusion_lab.py` 里离线实验，app 里的实现已删除，见下文）不写死谁更可信：

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
  这是产品取舍，不随模型变；换了模型只需重新拟合。
- **拼接样本**：现有片子全是"剧情→动作"，而且动作前几乎都隔着一段 other_adult（不计分）：
  停转切换点一个都没有，干净的启动切换点也只有一两个，不补的话扫参会挑出"永远不停"或
  "启动很慢"的参数。所以用同一部片的动作段和剧情段首尾拼接，造"停转样本"（动作 60s +
  剧情 45s）和"启动样本"（剧情 45s + 动作 60s），一起算进代价。`eval` 报表里的
  "启动后没转 / 停止后还转"就是拼接点之后马达该转没转 / 该停还转的秒数（中位数）。
  拼接处感知是瞬间切换的，比真实片子乐观约 1~2 秒。

### app 里的证据融合已删除（2026-10-06）

（2026-10-07 起启动仲裁 `applyDoStartGate` 又换成了融合仲裁 `FusionArbiter`，见上文；下面是 10-06 当时的状态。）

app 只保留原融合：离线模式、网页视频模式、在线模式（抓屏）都是投票窗（`smoothedFusion`）+ 快通道 +
启动仲裁（`applyDoStartGate`）+ `selectBestAction`，之后照旧走蓝牙状态机。一起删掉的有：`EvidenceFusion.java`、
`assets/fusion_config.json`（`mode` / `audio_pipeline` 开关随之没了）、`EvidenceFusionTest` 与
`app/src/test/resources/fusion/` 下的对拍数据，以及 `fusion_lab.py` 的 `fit`（写 app 的配置）和 `golden`
（生成 Java 对拍数据）。run CSV 不再有 `ev_*` 列，参数行不再有 `FUSION=`。

`fusion_lab.py` 里的证据融合留作离线实验，`eval` / `oracle` 照样拿它和现有融合、几条基线对比。以后要重新接入 app，
用 `git log -- app/src/main/java/com/example/helloworld/EvidenceFusion.java` 找到删除它的提交，从上一个版本取回。

### 当前结果（2026-10-06，修复后的音频，6 部片，留一视频交叉验证）

`fusion_lab.py eval bench/gt bench/rescored/fixed`（音频是 `audio_rescore.py` 按修复后的处理离线重算的，
视频和时序来自 1.2.4 真机录的 run）：

| 策略 | 准确率 | 误转 | 漏转 | 误停/分 | 启动后没转 | 停止后还转 |
|---|---|---|---|---|---|---|
| 现有融合 | 89.8% | 37.3% | 0.8% | 0.14 | 0.0s | 18.5s（15 个里 2 个基本不停） |
| 只用视频 | 85.6% | 31.5% | 8.5% | 0.20 | 0.6s | 16.7s（5 个基本不停） |
| **只用音频** | **94.3%** | **10.0%** | 4.2% | 0.14 | 2.6s | **4.1s** |
| 证据融合 | 92.5% | 17.4% | 4.0% | 0.10 | 2.6s | 4.6s（2 个基本不停） |

音频修好后，拟合几乎总是选 `w_v=0`（只用音频）：6 折里 5 折如此，只有留出 v07 那一折选了 `w_v=0.25`，
v07 从 89.7% 掉到 80.8%，所以"证据融合"一行比"只用音频"差。当时打包进 app 的配置 `fit-20261006-074840`
（已随证据融合删除）是用全部 6 部片拟合的，同样是 `w_v=0`、音频特征 `p_sex+p_oral`，泛化表现看"只用音频"一行。
剩下的误转主要在 v04（24%）、v07（19%），漏转主要在 v05（15%）。视频模型改进之后重新拟合，视频才会重新参与。

**2026-10-06 起 app 只用原融合**（先把 `mode` 改成 `legacy` 停用了证据融合，随后整个删除，见上）。
注意原融合现在拿到的是修复后的音频，它的阈值是在半速音频时代手调的；
上表"现有融合"一行（89.8% / 误转 37.3% / 漏转 0.8% / 停止后还转 18.5s）就是它在修复后音频上的回放结果
（原融合没有拟合参数，不受留一影响）。

### 旧结果（2026-10-05，旧的半速音频，留一视频交叉验证）

| 策略 | 准确率 | 误转 | 漏转 | 误停/分 | 启动后没转 | 停止后还转 |
|---|---|---|---|---|---|---|
| 现有融合 | 79.2% | 63.2% | 6.3% | 0.63 | 0.0s（剧情里本来就在转） | 40.1s（15 个里 8 个基本不停） |
| 只用视频 | 85.6% | 31.5% | 8.5% | 0.20 | 0.6s | 16.7s |
| 只用音频 | 83.3% | 35.3% | 10.4% | 0.04 | 20.0s | 28.2s |
| **证据融合** | **85.5%** | **31.3%** | 8.7% | **0.14** | 1.1s | **10.5s** |

当时 6 部片逐片都比现有融合好（v01 73.8→75.1，v02 96.8→100，v04 77.8→80.5，v05 85.5→93.4，
v06 85.2→100，v07 62.6→69.1），用的配置是 `fit-20261005-060733`（音频特征 `p_sex`，`w_v=1`）。

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
`AudioDecoder` 删除，离线模式只有修复后的音频（这个配置文件后来也随证据融合删了）。离线评估用 `fixed` 重算的结果：

```bash
python bench/tools/audio_rescore.py rescore --videos D:/xcup/videos            # 默认 --pipeline fixed
python bench/tools/fusion_lab.py eval bench/gt bench/rescored/fixed             # 与 bench/runs 的 eval 对比
```

app 采集的 run 参数行记 `AUDIO_PIPE=fixed`（网页视频模式记 `web`）；用新包重录 bench 后，`check` 就是
`fixed` 对 `phone`，并且有 `a_pos`，能逐个精确对上。

### 2. 重算：rescore

```bash
python bench/tools/audio_rescore.py rescore --videos D:/xcup/videos                    # fixed：修复后的 app
python bench/tools/audio_rescore.py rescore --videos D:/xcup/videos --pipeline phone   # 旧版 app 的半速音频
python bench/tools/audio_rescore.py rescore --videos D:/xcup/videos --classifier new.tflite   # 换分类器
```

写到 `bench/rescored/<管线>[_<分类器>]/`，**文件名与 `bench/runs/` 里的一样**，所以 metrics.py /
fusion_lab.py 指向这个目录就能用：

- `a_p_*` 换成重算的概率，`a_action` / `a_conf` 按原阈值重判（过期规则同融合循环）；
- 追加 `a_ys_*`：YAMNet AudioSet 类别分数（见下表，只做离线分析）；
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

AUC 好看不等于端到端有用。2026-10-06 起 app 不再算 AudioSet 分数（`AudioInferenceHelper` 只取 YAMNet 的
embedding，与 main 分支一致），`fusion_lab.py` 里证据融合的音频向量只有 `[sex, oral, noise]`，`--speech`
也一起撤掉了；`a_ys_*` 现在只用于离线分析。以后要把 AudioSet 类别用进融合，app（`AudioInferenceHelper`
取 output 0、融合层、`BenchLogger` 的列）和 `fusion_lab.py` 要一起改，可参考提交 `a3efcf5`、`20f08f9`。

### run CSV 新增的列

旧版 dump 没有这些列也不影响其他工具：

| 列 | 含义 |
|---|---|
| `a_pos` | 该音频结果读 PCM 时的播放位置：模型窗口 = `[a_pos-2s, a_pos]`。有它，`check` 不用再反推窗口位置（只有 app 写） |
| `arb` | 融合仲裁本 tick 的输出原因（`start_both` / `hold_audio` / `stop_neg` …，2026-10-07 起的 run 才有）。`gated` 列是融合仲裁的输出，`smoothed` 列是旧投票（只作对照） |
| `arb_reset` | 这一行之前融合仲裁 reset 的时刻：`<ms>[+<预热ms>][:seek]`（seek / 恢复播放；没有为空）。回放时同样 reset，带 `:seek` 的还清空蓝牙状态机和旧投票窗 |
| `a_ys_speech` / `conversation` / `narration` / `music` | AudioSet 0 / 2 / 3 / 132 类，"剧情"候选证据（只有 `rescore` 写） |
| `a_ys_silence` / `moan` / `groan` / `breathing` / `pant` / `slap` | AudioSet 494 / 22 / 33 / 36 / 40 / 461 类，"动作"候选（只有 `rescore` 写） |

分数是 YAMNet output 0（`[帧, 521]`，sigmoid，多标签）按帧取平均；output 1 才是分类器用的 embedding。
挑哪几类在 `metrics.YS_CLASSES`；app 不算这些分数，`BenchLogger` 也不写这些列。
