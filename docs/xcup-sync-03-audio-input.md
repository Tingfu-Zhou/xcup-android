# Xcup 同步单 03 · 音频模型的输入管线（Android 排查结论 → iOS 自查）

> **⚠ 2026-10-06 更新：** 模型该吃什么音频、Android 错在哪、`PcmResampler` 怎么修（§1、§2、§4 的修复本身）、
> iOS 自查（§5）仍然有效。以下内容已过时，iOS 跳过即可：`fusion_config.json` 的 `audio_pipeline` 开关
> （Android 离线模式现在**只有**修复后的音频，旧的半速处理已删除）、10 个 AudioSet 分数（YAMNet 只取 embedding，
> 与 main 分支一致）、一切和证据融合有关的部分（证据融合已从 Android 删除，只用旧融合，见同步单 02 的更新）。

> Android 离线模式送进音频模型的音频是错的（半速、L/R 交错），已用 6 部评测片证实并修好。
> 本单写清楚：模型到底应该吃什么样的音频、Android 错在哪、怎么修的，以及 iOS 该怎么自查。
> 可以直接把本单交给一个只看得到 iOS 代码的新对话。

| 项目 | 值 |
|---|---|
| 仓库 | `Tingfu-Zhou/xcup-android` |
| 分支 | `claude/video-annotation-benchmark-automation-85f1ju` |
| 提交 | `a3efcf5`（离线重打分工具 + 评测日志）→ `20f08f9`（修复，配置开关）→ `4a1c696`（参考管线对齐训练脚本） |
| 日期 | 2026-10-05 |
| 状态 | Android 修复已合入代码，但**默认关闭**：要用修复后的音频重新拟合 `fusion_config.json` 才打开（见 §4） |
| iOS 要做 | 先按 §5 自查；有问题再按 §6 移植 |

---

## 1. 模型要吃的音频（以训练为准）

音频模型 = YAMNet（`yamnet.tflite`）+ 自己训练的分类器（`yamnet_finetuned.tflite`），类目 `[sex, oral, noise]`。
训练数据是这样做的（训练脚本原文）：

```python
command = ["ffmpeg", "-i", input_path, "-ac", "1", "-ar", str(SAMPLE_RATE), "-y", temp_wav_path]   # 16-bit WAV
```

由此得到推理时的输入契约，**每一条都要满足**：

| 项 | 要求 | 说明 |
|---|---|---|
| 采样率 | 16000 Hz | 用带抗混叠低通的重采样（ffmpeg 的 swr），不能简单线性插值/抽点 |
| 声道 | 单声道，**(L+R)/2** | 实测：上面这条命令写 16-bit WAV 时下混是 (L+R)/2。注意 ffmpeg 输出浮点时是 0.707·(L+R)，响 3dB——增益不同模型输出就不同 |
| 数值 | float32，int16/32768，范围 [-1, 1] | 不要再做归一化/AGC |
| 长度 | 32000 点 = 2 秒 | 当前播放位置 T 往前**连续**的 2 秒 `[T-2s, T]`，中间不能有缺口、不能重复 |
| 速度 | 1 倍速 | 时间轴必须按"帧"（每帧含全部声道）算，不能按 short 个数算 |

YAMNet tflite：输入是一维 waveform `[32000]`；输出 0 = AudioSet 521 类分数 `[4, 521]`，输出 1 = embedding
`[4, 1024]`，输出 2 = log-mel。分类器吃 **4 帧 embedding 的平均** `[1, 1024]`，输出 softmax `[sex, oral, noise]`。
（转成 Core ML 等格式时输出顺序可能变，按形状认：最后一维 1024 的才是 embedding。）

---

## 2. Android 发现的问题

### P1（严重）交错立体声被当成单声道 → 半速、低八度

`AudioDecoder` 从 MediaCodec 拿到的是交错 PCM（`L R L R …`），代码没看声道数，直接当单声道：

```java
final int windowSize = sampleRate * 4;          // 按 short 计：双声道时 = 2 秒，不是 4 秒
audioBuffer[bufferPos++] = s / 32768f;           // L、R 交替塞进同一个数组
originalTime[i] = (float) i / sampleRate;        // 时间轴按 short 个数算：双声道时快了一倍
// 重采样只取前 2"秒" = 前 2*sampleRate 个 short = 真实的前 1 秒，拉成 32000 点，时间戳铺满 2000ms
```

后果（48kHz/44.1kHz 双声道，即绝大多数 MP4）：

- 每 2 秒的块只用到**前 1 秒**真实音频，L/R 交错着拉长成 2 秒送进模型：**0.5 倍速、低一个八度**；后 1 秒直接丢掉。
- 模型窗口 `[T-2s, T]` 由前后两块的半截拼成，听到的内容比播放位置晚 0.5~2.5 秒。
- 单声道片源：速度对，但每 4 秒只留前 2 秒，中间 2 秒没有数据。

### P2 没有抗混叠

即使声道对了，Android 的重采样（`AudioDecoder` 和 `PcmCircularBuffer.resample`）都是逐块线性插值：
48k→16k 实际上等于每 3 个点取 1 个，8~24kHz 的内容全部混叠进 0~8kHz；每块重新从 0 相位开始，块与块之间
还会丢零点几个样本。

### P3 块边界丢样本

凑满一块时，所在那个解码 buffer 剩下的部分直接 `releaseOutputBuffer` 丢掉（约 20ms）。

### 证据（6 部评测片，全是 48kHz 双声道 AAC）

用离线工具在真机每个音频结果的时刻重建模型输入、用旧模型重算概率，和真机记下的比（MAE = 三类概率的平均绝对差）：

| | v01 | v02 | v04 | v05 | v06 | v07 |
|---|---|---|---|---|---|---|
| 复刻 Android 旧管线 | 0.070 | 0.040 | 0.027 | 0.043 | 0.021 | 0.049 |
| 训练同款处理 | 0.284 | 0.275 | 0.253 | 0.237 | 0.248 | 0.300 |

只有"复刻旧管线"能复现真机输出 → 真机喂的就是半速音频。同一个分类器，喂正确的音频后判别力（结果级 AUC，
剧情 vs 动作）大幅上升：

| 特征 | Android 旧管线（真机） | 正确的音频 |
|---|---|---|
| `p_sex+p_oral`（6 部合并 / 最差的一部） | 0.641 / 0.479 | **0.901 / 0.829** |
| `p_sex`（合并 / 最差） | 0.693 / 0.618 | 0.833 / 0.638 |

也就是说，之前"音频在剧情段大量判转、`p_oral` 像人声检测器"很大程度是输入错了，不全是模型的问题。

### Android 其他模式（未改）

- **网页视频模式**（ExoPlayer，`PcmCaptureAudioProcessor`）：下混正确（各声道求和 / 声道数），但仍走
  `PcmCircularBuffer.resample` 的逐块线性插值（P2），时间戳取写入时的播放位置（近似）。
- **在线模式**（抓屏，`AudioPlaybackCapture`）：系统直接给 16kHz 单声道，下混方式由系统决定，未验证。

---

## 3. 附带的发现：YAMNet 自带的 AudioSet 类别

YAMNet 输出 0（521 类分数）以前算了就丢。现在挑了 10 类写进评测 CSV（`a_ys_*`），按 4 帧取平均：

| 列 | AudioSet 类别号 | 用途 |
|---|---|---|
| speech / conversation / narration / music | 0 / 2 / 3 / 132 | "剧情"候选证据 |
| silence / moan / groan / breathing / pant / slap | 494 / 22 / 33 / 36 / 40 / 461 | "动作"候选 |

正确音频下：Speech 判别力 0.71（越高越像剧情），呻吟/喘息类 0.80~0.85；Music 没用。要不要把 Speech 加进融合，
以 `fusion_lab.py eval --speech` 的留一交叉验证结果为准（待跑）。

---

## 4. Android 的修复（iOS 移植以此为准）

新增 `PcmResampler.java`（纯 Java，不依赖 Android），`AudioDecoder` 在 `fixed` 模式下把**每个解码 buffer**交给它：

1. **下混**：每帧各声道整数相加，`mono = (float) sum / (channels * 32768f)`，即 (L+R)/2，与训练一致。
2. **重采样**到 16kHz：Blackman 窗 sinc 低通插值。
   - `g = gcd(inRate, 16000)`，`num = inRate/g`，`den = 16000/g`；第 n 个输出在输入里的位置 `p = n*num/den`
     （`i0 = n*num / den` 整除，相位 `k = n*num % den`）。
   - `cutoff = min(7600, 0.475*min(inRate,16000))` Hz，`fc = cutoff/inRate`，窗半宽 `W = 24/(2fc)`，`H = ceil(W)`。
   - 相位 k 的 2H 个系数：`t_j = k/den + H - 1 - j`（j = 0..2H-1），`|t|<W` 时
     `h = 2fc·sinc(2fc·t)·(0.42 + 0.5cos(πt/W) + 0.08cos(2πt/W))`，否则 0；`sinc(x)=sin(πx)/(πx)`；
     再除以系数和（按 j 顺序累加），使直流增益为 1。
   - `y[n] = Σ_j x[i0-H+1+j] * h_k[j]`（double，按 j 顺序累加），流开头之前的输入当 0。
   - 输入本来就是 16kHz 时只下混、不滤波。
   - 频响：7kHz 内 -0.1dB，7.5kHz -4dB，8.5kHz 以上衰减 >76dB。与训练用的 ffmpeg 处理比，6kHz 以下 SNR 78dB，
     分类器概率差 MAE ≈ 0.01。
3. **流式、连续**：跨 buffer 保留历史，不丢样本、不重置相位；只在第 n 个输出所需的输入（到 `i0+H`）都到齐后
   才产出，而且一次只产出 16 的整数倍个（16 个 = 1ms）。
4. **时间戳**：第 n 个输出的时间 = 起头 buffer 的 pts + n/16000 秒（写入环形缓冲时 `startMs = basePtsMs + n0/16`）。
   相邻 buffer 的 pts 与按样本数推算的时间差超过 100ms（seek/不连续）时，重采样器重新起头。
5. 读窗不变：`readWindowRelaxed(T, 32000)` 取时间戳 ≤ T 的最后 32000 个样本，修复后就是连续的 `[T-2s, T]`。

对拍：`app/src/test/resources/audio/resampler_golden.txt`（`python bench/tools/audio_rescore.py golden` 生成），
`PcmResamplerTest` 校验。输入不存文件，用 64 位 LCG 现场生成（与 Python 端相同）：

```
s = inRate*31 + channels
每个 short：s = s*6364136223846793005 + 1442695040888963407（64 位溢出回绕）；value = (s >>> 48) - 32768
```

文件里每个 `case <采样率> <声道> <帧数> <输出个数>` 后面是期望的 16kHz 输出（误差 ≤ 1e-6 算通过）。iOS 可以照着写
一个 XCTest。

### 开关：`fusion_config.json` 的 `audio_pipeline`

证据融合的证据表只对拟合时那种音频有效。所以修复由配置打开：`"audio_pipeline": "fixed"` → 用修复后的管线；
没有这一项（现有配置）或其他值 → 旧管线，行为不变。`fusion_lab.py fit` 会按训练数据自动写入这一项。

**同一份配置给 Android 和 iOS 用，iOS 也必须遵守这个开关**：拿到 `audio_pipeline: fixed` 的配置时，iOS 喂给模型的
音频必须是 §1 的正确音频；否则就继续用旧配置。

### 其他改动

- 证据融合收到的音频向量改为 `[sex, oral, noise] + 上表 10 个 AudioSet 分数`（只用前 3 项的配置不受影响；以后
  拟合出带 Speech 的配置时会用到第 4 项）。
- 评测 CSV 新增 `a_pos`（该音频结果读 PCM 时的播放位置）和 `a_ys_*` 列；参数行新增 `AUDIO_PIPE=legacy/fixed/web`。

---

## 5. iOS 自查清单

对 iOS 的**每一条音频来源**（本地视频离线模式、网页视频、在线/抓屏模式，如果有）逐项看：

1. **解码输出格式**：采样率、声道数、是否交错（interleaved）、Int16 还是 Float32。代码里哪里假设了"单声道"？
   有没有照搬 Android 的 `windowSize = sampleRate * 4`、`i / sampleRate` 这类按 short 个数算时间的写法（→ P1）？
2. **下混**：是不是 (L+R)/2？常见的坑：
   - 交错数据直接当单声道（P1）；
   - 只取左声道（例如 `AVAudioConverter` 转单声道而没设 `downmix = true`）；
   - 让系统下混（`AVAssetReader` 输出设置 `AVNumberOfChannelsKey: 1`、`AVAudioConverter.downmix = true`）——系数
     未必是 0.5，要用 §5.1 的测试片实测。
3. **重采样**：是否用 `AVAudioConverter` / AVAssetReader 的采样率转换（有抗混叠），还是手写线性插值（→ P2）？
   转换器是不是跨 buffer 连续使用（每块新建一个会在块边界产生伪影）？
4. **窗口**：送进模型的 32000 点是不是播放位置往前连续的 2 秒？有没有丢块、补零、重复？时间戳怎么和播放位置对齐？
5. **数值**：Int16/32768（或系统给的 Float32 [-1,1]），没有额外增益、归一化、AGC。
6. **模型调用**：YAMNet 取的是 1024 维 embedding（不是 521 维分数），4 帧取平均；分类器输出按 `[sex, oral, noise]` 解释。
7. **采样率来源**：HE-AAC 等格式里容器报的采样率可能是核心采样率（输出的一半），以解码器实际输出格式为准。

### 5.1 用测试片实测（最快、最确定）

做一个左右声道不同音调的测试片（48kHz 双声道 AAC，30 秒；`sine` 默认幅度 0.125）：

```bash
ffmpeg -f lavfi -i "color=c=black:s=320x240:r=30:d=30" \
  -f lavfi -i "sine=frequency=440:sample_rate=48000:duration=30" \
  -f lavfi -i "sine=frequency=1000:sample_rate=48000:duration=30" \
  -filter_complex "[1:a][2:a]join=inputs=2:channel_layout=stereo[a]" \
  -map 0:v -map "[a]" -c:v libx264 -c:a aac -b:a 192k -shortest tone_test.mp4
```

在 iOS 里播放它，把送进 YAMNet 的 32000 个 float 原样存成文件（如 `win_<播放毫秒>.f32`，小端 float32），
用下面的脚本看：

```python
import numpy as np, sys          # 用法：python peaks.py win_15000.f32
x = np.fromfile(sys.argv[1], dtype='<f4')
w = np.hanning(len(x)); amp = 2 * np.abs(np.fft.rfft(x * w)) / w.sum(); f = np.fft.rfftfreq(len(x), 1 / 16000)
pk = [i for i in range(1, len(amp) - 1) if amp[i] > amp[i - 1] and amp[i] >= amp[i + 1] and amp[i] > 0.005]
for i in sorted(pk, key=lambda i: -amp[i])[:6]: print('%7.1f Hz  幅度 %.4f' % (f[i], amp[i]))
```

| 看到的 | 说明 |
|---|---|
| 440Hz 和 1000Hz，幅度都约 **0.0625** | 正确：(L+R)/2，1 倍速 |
| 440Hz 和 1000Hz，幅度约 0.088 | 下混系数是 0.707，比训练响 3dB，要改成 /2 |
| 只有 440Hz，幅度约 0.125 | 只取了左声道 |
| 220Hz 和 500Hz（约 0.05），7.5kHz 附近还有镜像峰 | **P1 半速 bug**，和 Android 旧版一样 |
| 8kHz 以上的音调在 0~8kHz 出现镜像峰 | 没有抗混叠（可另做一个 10kHz 的测试片：正确时应几乎为 0） |

再用一部真实片子对比"iOS 实际送进模型的窗口"与训练同款处理：

```bash
ffmpeg -i real.mp4 -ac 1 -ar 16000 -f s16le ref.s16      # 与训练脚本相同的处理（16-bit）
```

```python
import numpy as np
ref = np.fromfile('ref.s16', dtype='<i2') / 32768.0
win = np.fromfile('win_123456.f32', dtype='<f4')          # 播放到 123456ms 时送进模型的窗口
T = 123456 * 16
best = max(range(-8000, 8001, 4), key=lambda d: np.dot(win, ref[T + d - 32000:T + d]))   # ±0.5 秒找对齐
seg = ref[T + best - 32000:T + best]
print('窗口比播放位置偏 %+d ms，SNR %.1f dB' % (best / 16, 10 * np.log10(np.sum(seg**2) / np.sum((win - seg)**2))))
```

正确时 SNR 应在 20dB 以上（差别只来自重采样器）、偏移是几十毫秒以内的固定值；SNR 接近 0 或为负说明输入不对。

---

## 6. iOS 若有问题：怎么改

- **推荐**：照 §4 逐行移植 `PcmResampler`，用 `resampler_golden.txt` 写 XCTest 对拍。这样两端喂给模型的音频
  逐样本一致，同一份 `fusion_config.json` 在两端效果相同。
- 也可以用系统转换器（`AVAudioConverter`），但要保证：(L+R)/2 的增益（用 §5.1 测）、跨 buffer 连续、输出连续的
  `[T-2s, T]`。与 Android 的差别会很小（分类器概率差约 0.01），但不是逐样本一致。
- 实现 `fusion_config.json` 的 `audio_pipeline` 开关（§4）：旧配置 → 保持现状；`fixed` 配置 → 用正确音频。
- 证据融合的音频向量补上 10 个 AudioSet 分数（顺序见 §3），为以后带 Speech 的配置做准备。

## 保持不变

模型文件、2 秒窗口与 1 秒推理间隔、分析线程的阈值判定、证据融合算法本身、蓝牙状态机、响度档位。

## 附：离线工具（Python，不需要移植）

`bench/tools/audio_rescore.py`：`check`（判断真机喂的是哪种音频、验证对齐）、`rescore`（按 phone/fixed/clean
管线重算评测 CSV）、`auc`（判别力）、`golden`（重采样对拍数据）。详见 `bench/README.md`。
