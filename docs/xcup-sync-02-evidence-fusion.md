# Xcup 同步单 02 · 证据融合（Android → iOS 移植对照）

> **⚠ 2026-10-06 更新：本单作废，iOS 不需要移植。** Android 已删除证据融合（`EvidenceFusion.java`、
> `assets/fusion_config.json`、`EvidenceFusionTest` 与对拍数据），离线 / 网页视频 / 在线模式都只用旧融合
> （投票窗 + 快通道 + 启动仲裁 + `selectBestAction`，之后照旧走蓝牙状态机）。本文只留作记录。

> 本阶段只改融合层，不动模型。旧融合（投票窗 + 启动仲裁）的代码保留作后备，由配置文件切换。
> 本单按移植顺序列出每处改动的规则、位置与 iOS 落地要点，末尾附「保持不变」的部分与验证清单。

| 项目 | 值 |
|---|---|
| 仓库 | `Tingfu-Zhou/xcup-android` |
| 分支 | `claude/video-annotation-benchmark-automation-85f1ju`（尚未合并到 main） |
| 提交 | `c78b5ec`（证据融合，影子模式接入）→ `86bdca5`（上下界分开）→ `47bcc58`（上线 live） |
| 日期 | 2026-10-05 |
| 需要移植 | `EvidenceFusion.java`（新增）、`assets/fusion_config.json`（新增）、`VideoProcessActivity.java`（接入） |
| 可选移植 | `BenchLogger.java`（只有 iOS 也做评测采集时才需要） |
| 不需要移植 | `bench/tools/*.py`（Python 离线工具，与平台无关） |
| 适用范围 | **离线模式**（VideoProcessActivity，含网页视频模式）。**在线模式未改**，仍是旧融合 |
| 前置 | 基于同步单 01（第一层延迟优化：融合循环 500ms、分组概率判定、启动仲裁等）之后的代码 |

---

## 背景（供不了解本项目的人阅读）

视频模型（MobileNetV3Small，类目 `[plot, oral, sex]`）和音频模型（YAMNet + 分类器，类目
`[sex, oral, noise]`）各自输出概率，融合层决定马达「转 / 不转」，再交给蓝牙状态机下发。

用 6 部完整视频按 100ms 标注真值做评测后发现：旧融合把「音频更准、音频权重 1.2、音频可单独
启动」写死在代码里，但音频在剧情段里也大量判「转」（多数片子里音频判「转」的似然比 ≈ 1，
等于没有信息），结果是剧情时长里有 63% 在误转；而且动作结束后旧融合基本停不下来（中位多转 40 秒）。

证据融合不再手写谁更可信：每个模型结果按概率查一张「证据表」（对数似然比，从评测数据统计），
随时间累积，过阈值才切换。证据表和参数都在配置文件里；换了模型，用 Python 工具重新拟合一份
配置即可，两端代码都不用改。

### 效果（留一视频交叉验证：每部片的参数只用其余 5 部拟合）

| | 准确率 | 剧情里误转 | 动作里漏转 | 误停（次/分） | 动作结束后还转（中位） |
|---|---|---|---|---|---|
| 旧融合 | 79.2% | 63.2% | 6.3% | 0.63 | 40.1 秒 |
| **证据融合** | **85.5%** | **31.3%** | 8.7% | **0.14** | **10.5 秒** |

6 部片逐片都比旧融合好。剩下的误转主要是剧情里有裸露、视频模型自己也判「转」，要靠改模型。

---

## 参数速查（当前配置 `fit-20261005-060733`）

| 参数 | 值 | 说明 |
|---|---|---|
| `mode` | `live` | `legacy` 不运行 / `shadow` 只记日志 / `live` 驱动蓝牙 |
| 视频特征 | `p_oral + p_sex`（weights `[0, 1, 1]`） | 按视频类目顺序 `[plot, oral, sex]` |
| 音频特征 | `p_sex`（weights `[1, 0, 0]`） | 按音频类目顺序 `[sex, oral, noise]` |
| `w_v` / `w_a` | 1.0 / 1.0 | 两个模态证据的权重 |
| `clamp_lo` / `clamp_hi` | 2.0 / 8.0 | 累积量 L 的范围是 [−2, +8] |
| `theta_on` / `theta_off` | 1.5 / 0.0 | L ≥ 1.5 启动；L ≤ 0 停转（都含等号） |
| `max_age_ms` | 2000 | 结果超过 2 秒不给证据 |
| `edges`（两个模态相同） | `[0, 0.02, 0.05, 0.1, 0.2, 0.35, 0.5, 0.65, 0.8, 0.9, 0.95, 0.98, 1.0]` | 12 格 |
| 视频 `llr` | `[-0.6913, -0.2649, -0.1063, -0.1063, 0.3012, 1.1594, 1.1594, 1.1594, 1.5054, 1.5054, 1.5054, 1.5054]` | |
| 音频 `llr` | `[-1.5088, -0.6825, -0.2958, -0.0088, 0.3458, 0.4764, 0.4764, 0.5965, 0.8231, 0.8231, 0.8231, 1.8531]` | |

直观理解：视频高概率是「动作」的主要证据（最多 +1.5）；音频 `p_sex` 很低是「剧情」的主要证据
（最多 −1.5）；视频判「剧情」的证据已打 0.35 折（遮挡、特写时视频模型在动作里也会判剧情）。

---

## 逐项改动

按此顺序移植；01 是其余改动的前提。

### 01 · 模型结果快照

**位置**：`applyVideoResult()`；音频循环里写入推理结果处。

**旧**：动作串、置信度、时间戳三个原子变量分开写；评测日志另存一份概率。三者可能来自不同次推理。

**新**：每次推理后，把「概率拷贝 + 产出时刻」打包成一个不可变快照，一次性原子发布：

```
resultTs = 当前墙钟毫秒
probsOk  = 结果有效 && probs.length >= 3
latestVideoSnap = Snapshot(probsOk ? copy(probs) : null, resultTs)
latestVideoTimestamp = resultTs          // 与快照同一个时间戳（旧融合仍用这个变量）
// 音频同理 → latestAudioSnap
```

**iOS 要点**：
- Snapshot 用不可变 struct（`probs: [Float]?` + `tsMs: Int64`），用一个锁或原子引用整体替换。
- 推理失败也要发布快照（`probs = nil`），它会被标记为「用过」但不给证据。
- `tsMs` 必须和融合循环里的 `now` 用同一个时钟（毫秒墙钟）。

### 02 · 证据融合算法（核心，必须逐字对齐）

**位置**：新文件 `EvidenceFusion.java`（纯逻辑，除 JSON 解析外不依赖平台）。

```
状态：L = 0.0（double），state = NONE，lastVideoTs = lastAudioTs = −∞

update(videoSnap, audioSnap, now):
    llrV = 无, llrA = 无
    if videoSnap != null && videoSnap.ts > lastVideoTs:        // 每个结果只计一次
        lastVideoTs = videoSnap.ts                             // 先标记为用过
        if now − videoSnap.ts <= maxAgeMs && valid(video, videoSnap.probs):
            llrV = lookup(video, feature(video, videoSnap.probs))
    （音频同理 → llrA）

    if llrV 或 llrA 有值:                                       // 没有新证据：L 和状态都不变
        d = 0.0
        if llrV 有值: d += wV * llrV                            // 加法顺序固定：先视频后音频
        if llrA 有值: d += wA * llrA
        x = L + d
        L = x > clampHi ? clampHi : (x < −clampLo ? −clampLo : x)
        if state == DO:
            if L <= thetaOff: state = NOISE
        else:                                                  // NONE 或 NOISE
            state = (L >= thetaOn) ? DO : NOISE

    return action = (NONE → "")  (DO → "do")  (NOISE → "Noise"),  以及 L、llrV、llrA

valid(m, probs):   probs 非空 && probs.count >= m.weights.count && 没有 NaN/Inf
                   && 全部元素之和 > 0.5                      // 推理失败时是空数组或全 0
feature(m, probs): s = Σ_{i < weights.count} weights[i] * probs[i]（double），再裁到 [0, 1]
lookup(m, s):      i = 0
                   for k in 1 ..< edges.count − 1:  if s >= edges[k] { i = k } else { break }
                   return llr[i]                               // 左闭右开，最后一格含 1.0

reset():           L = 0, state = NONE, lastVideoTs = lastAudioTs = −∞
```

**iOS 要点**：
- 全部用 `Double`；概率从 `Float` 转 `Double` 再参与运算；加法顺序照上面写，对拍要求误差 ≤ 1e-12。
- `""` 表示本 tick 不驱动蓝牙：reset 之后、还没收到任何有效结果时就是这个状态。

### 03 · 配置文件 `fusion_config.json`

**位置**：`app/src/main/assets/fusion_config.json`（与 Android 共用同一份，直接拷进 iOS bundle）。

```json
{
  "format": "xcup-fusion-v1",
  "mode": "live",
  "id": "fit-20261005-060733",
  "video": { "feature": "p_oral+p_sex", "weights": [0.0, 1.0, 1.0], "edges": [...], "llr": [...] },
  "audio": { "feature": "p_sex", "weights": [1.0, 0.0, 0.0], "edges": [...], "llr": [...] },
  "w_v": 1.0, "w_a": 1.0, "clamp_lo": 2.0, "clamp_hi": 8.0,
  "theta_on": 1.5, "theta_off": 0.0, "max_age_ms": 2000,
  "fitted_at": "...", "provisional": false, "note": "...", "fit": { ... }
}
```

- 只需要读上面列出的字段，其余字段（`fitted_at`、`fit` 等，仅作记录）忽略。
- `format` 必须是 `xcup-fusion-v1`。
- 兼容旧格式：只有 `"clamp"`、没有 `clamp_lo/clamp_hi` 时，上下界都取 `clamp`。
- 校验：`edges` 严格递增、`llr.count == edges.count − 1`、`−clamp_lo < theta_off < theta_on ≤ clamp_hi`、
  `mode` 只能是三者之一。**文件缺失、解析失败、校验不过 → 一律回退旧融合**，不能影响正常使用。
- **类目顺序**：`weights` 是按模型输出的类目顺序写的。iOS 的模型（尤其是转换过的 Core ML 模型）
  输出顺序必须和 Android 一致（视频 `[plot, oral, sex]`，音频 `[sex, oral, noise]`），否则要改 `weights`。

### 04 · 融合循环接入

**位置**：`startMainFusionLoop()`（500ms 一次）。

```
（原有）读动作串 / 置信度 / 时间戳 / 响度档位
vSnap = latestVideoSnap; aSnap = latestAudioSnap               // 新增
currentTime = 当前墙钟毫秒
（原有）2 秒新鲜度过滤 —— 只作用于旧融合的动作串

ev = evidenceFusion?.update(vSnap, aSnap, currentTime)         // 新增：每 tick 都算
if evidenceLive && ev != null:
    finalAction = ev.action                                    // 不再调用投票窗和启动仲裁
else:
    finalAction = smoothedFusion(...)                          // 旧融合（shadow / legacy / 配置无效）
    finalAction = applyDoStartGate(finalAction, ...)

（原有）finalFreq = computeFinalFreq(...)
（原有）if finalAction 非空 && 未被本地按键暂停: updateBluetoothState(finalAction, finalFreq)
```

### 05 · 重置点

**位置**：离线模式 seek（防抖后的处理）与网页视频模式 seek，和旧的「清空动作历史」在同一处：

```
latestVideoSnap = null; latestAudioSnap = null; evidenceFusion?.reset()
```

暂停 / 恢复不重置：恢复后旧结果会因为时间戳去重或过期自然不再计入。

### 06 · 加载与模式

**位置**：`onCreate()`，在评测日志初始化之前（参数行要记下融合模式）。

- `legacy`：不创建证据融合；`shadow`：创建、每 tick 计算，但蓝牙由旧融合驱动；`live`：由它驱动蓝牙。
- 成功时打印日志 `[证据融合] 已加载 <id>，mode=<mode>`，失败时打印回退原因。

### 07 · 评测日志（可选，iOS 也做评测采集时才需要）

**位置**：`BenchLogger.java`。

- 删除 `setVideoProbs / setAudioProbs`，改为 `tick()` 直接传入快照里的概率。
- `v_age / a_age` 改为「当前时刻 − 快照时间戳」（离线回放靠 `wall_ms − age` 还原结果时刻做去重）。
- 末尾新增 4 列：`ev_action, ev_l, ev_llr_v, ev_llr_a`（未启用时留空）。
- 参数行末尾新增 `FUSION=<mode>/<配置 id>`。

iOS 如果按同样的列格式写 CSV，Android 的 Python 工具（`metrics.py`、`fusion_lab.py check`）可以直接用。

---

## 保持不变（iOS 不要动）

- 两个模型、抽帧、音频解码、2 秒音频死区。
- 分析线程里的阈值判定（视频 0.65 / 0.60，音频 0.55 / 0.60）：仍然计算，供旧融合和界面上的 V:/A: 文字。
- 蓝牙状态机：停转需稳定 1000ms、最短驻留 2000ms、发送间隔 1600ms、档位迟滞与重发逻辑。
- 响度档位 `computeFinalFreq`。
- 旧融合代码（`smoothedFusion` / `applyDoStartGate` / `selectBestAction`）：保留作后备，不要删。
- 在线模式（抓屏）：本阶段未改，仍是旧融合。

---

## 验证清单

1. **对拍（最重要）**：用 `app/src/test/resources/fusion/golden_config.json` 构造配置，逐行喂
   `golden.csv`，480 行必须全部一致。列格式：

   | 列 | 含义 |
   |---|---|
   | `reset` | 1 = 先调用 `reset()` 再处理本行 |
   | `now` | 当前墙钟毫秒 |
   | `v_ts` / `a_ts` | 快照时间戳；空 = 快照为 null |
   | `v_probs` / `a_probs` | `|` 分隔的概率；空 = probs 为 null；`-` = 空数组 |
   | `exp_action` | 期望输出 `do` / `Noise` / 空 |
   | `exp_l` | 期望的 L（误差 ≤ 1e-12） |
   | `exp_llr_v` / `exp_llr_a` | 期望的证据值；空 = 本行没有用到该模态 |

   Android 端对应的测试是 `EvidenceFusionTest.matchesPythonGolden`，可以照着写一个 XCTest。
2. 打包的配置能解析，日志显示 `mode=live`。
3. 行为抽查：剧情对话段不启动；动作开始后几秒内启动；短暂遮挡 / 镜头切走不停；
   动作结束后十几秒内停（拼接样本里中位 10.5 秒，真实片子会再慢一两秒）。
4. 如果 iOS 也写评测 CSV：用同一批视频采集，`fusion_lab.py check` 应与离线回放一致，
   `metrics.py` 的指标应与 Android 接近。

---

## 附：Python 离线工具（不需要移植）

`bench/tools/fusion_lab.py`：`check`（回放校验）、`eval`（留一交叉验证）、`oracle`（模态上限）、
`fit`（拟合并写出配置）、`golden`（重新生成对拍数据）。详见 `bench/README.md`。

以后换模型（音频或视频）后的流程：重新采集评测 CSV → `fusion_lab.py fit --mode live` → 把新的
`fusion_config.json` 同时放进 Android 和 iOS。改了证据融合**算法**本身才需要动两端代码，
并用 `fusion_lab.py golden` 重新生成对拍数据。
