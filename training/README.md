# training/ — 视频模型训练（P1：训练方法）

对应《Xcup 视频动作识别：精度提升方案（2026-10-16）》的 Phase 1：模型结构不变（MobileNetV3Small、12 帧时间平均、
160×160、类别 `[normal_plot, oral, sex]`），训练素材不变（`movinet_clips`），只改训练方法；13 部测试片（v01~v13）只用来
比做完的候选，选 checkpoint 只看训练素材自己的验证集。

| 文件 | 作用 |
|---|---|
| `data_check.py` | 素材检查：`scan` 找坏片段 → `bad_clips.txt`；`stats` 按 split / 类别 / 源视频统计；`leak` 查 13 部测试片有没有混进训练素材 |
| `train_utils.py` | 训练积木：顺序读帧、片段级增强、按源视频均衡采样、按 p_do AUC 选模型、12 帧导出与 Keras / TFLite 一致性检查 |
| `train_video.py` | 训练入口，替代 `train_mobilenetv3_tap_3class.py`；`--preset e1a / e1b / e1c / e1d / legacy` |
| `eval_clips.py` | 在训练素材的 val / test 上评任意 12 帧 tflite（包括 app 现用的），给新模型的数字一个参照 |

bench 端新增：`bench/tools/frame_diag.py`（逐帧诊断：窗口长度、置信度分布、成段的高置信度错误，几个变体并排）、
`bench/tools/retune.py`（给新模型留一视频重调视频阈值，E1c 用）。见 `bench/README.md` 第 5 节。

## 和旧脚本比改了什么（E1a 一次打包）

| | 旧 `train_mobilenetv3_tap_3class.py` | 新 `train_video.py --preset e1a` |
|---|---|---|
| 坏片段 | 打不开 / 读不出的片段返回全黑帧，带着原标签进训练 | 训练前 `data_check.py scan` 剔除；训练中读帧失败跳过并报出来 |
| 读帧 | 每帧 `CAP_PROP_POS_FRAMES` 跳一次（每个片段从关键帧解十几次） | 顺序解码一遍再挑帧；取哪 12 帧不变（`np.linspace(0, n-1, 12)`），合成片段上与旧读法逐像素相同 |
| 采样 | 全部片段打乱 + `class_weight` | 类别等权 → 类别内按 片段数^0.5 选源视频 → 源视频内等概率选片段（源视频名从文件名 `<源视频名>_s<起点>_<类别>.mp4` 取） |
| 增强 | 无 | 片段级（12 帧同一组参数）：裁剪面积 0.7~1、宽高比 3/4~4/3、水平翻转、亮度 / 对比度 ±10%、饱和度 ±15%、10% 概率轻微模糊、15% 概率 JPEG（质量 40~85） |
| 选 checkpoint / 早停 | `val_acc` | 验证集 12 帧的 p_oral+p_sex AUC（`val_auc_do`）；同时记准确率、按片子平均的准确率、loss |
| 导出后验证 | 全零输入跑一次 | 用真实验证片段比 Keras 与 TFLite（概率差 ≤1e-4），检查结构是 1 个 PACK × 12 + 1 个 FULLY_CONNECTED（bench 能直接用） |
| 微调 | stage 2 解冻最后 30 层，BN 冻结，Adam lr 1e-5 | 不变 |

训练用的模型把 12 帧合成一批过 backbone（比 TimeDistributed 展开 12 份快），导出时再搭成和现用模型一模一样的
TimeDistributed 结构（共享权重）；BN 始终冻结，两者数值相同（导出核对会查）。

## 在哪个系统里跑

| 步骤 | 在哪跑 |
|---|---|
| 0~2、4 的训练（`data_check.py scan/stats`、`train_video.py`） | Ubuntu（训练用的 venv 和 GPU 在那边，素材在 `~/ml/movinet/movinet_clips`） |
| 3 的 bench（`baseline.py`、`frame_diag.py`、`retune.py`）和 `data_check.py leak` | 哪边都行，只要能读到 13 部原片和导出的 `.tflite` |

13 部原片不用挪，`--videos` 指向它们现在的位置就行；按 `bench/gt/vNN.csv` 文件头的 `file_name`（`视频1.mp4` ~ `视频13.mp4`）
找，文件名不同就用 `--pair v07=<路径>` 逐个指定。

- Ubuntu 是 WSL2（Ubuntu 里 `uname -r` 带 `microsoft`）：Windows 的 D 盘就是 `/mnt/d`，全部在 Ubuntu 里做最省事，
  `--videos "/mnt/d/视频/测试视频"`。bench 要多装一个包：`pip install av`；`data_check.py` 的自检和 `leak` 要 `sudo apt install ffmpeg`。
- Ubuntu 是双系统：在 Ubuntu 的文件管理器里点一下 Windows 的分区就会挂到 `/media/<用户名>/<分区名>/`，只读也够用；
  或者 bench 回到 Windows 的 PowerShell 里跑（下一条）。
- 在 Windows PowerShell 里跑 bench：用 Windows 上的仓库（同样切到这个分支），`python3` 换成 `python`，路径用 Windows 的写法，例如
  `python bench\tools\baseline.py rescore --videos "D:\视频\测试视频" --model <e1a.tflite 的路径> --name e1a --jobs 4`。
  依赖：`pip install numpy av ai-edge-litert opencv-python-headless`。WSL 里训练出的模型在 Windows 下的路径是
  `\\wsl$\Ubuntu\home\<用户名>\ml\movinet\runs\e1a\e1a.tflite`。
- 第一次用某处的原片之前，先 `baseline.py check --videos <原片目录>`：确认 13 部都找得到、文件大小和录 run 时一样、离线能复现手机。

## 0. 准备（训练机，一次）

```bash
cd ~/ml/movinet && source .venv/bin/activate
# 代码：在训练机上 clone 仓库（已有就 git fetch），切到这个分支
git clone https://github.com/Tingfu-Zhou/xcup-android.git ~/ml/xcup-android
cd ~/ml/xcup-android && git checkout claude/serene-gauss-o8s5df
python3 -c "import tensorflow as tf, keras; print(tf.__version__, keras.__version__)"
python3 training/data_check.py --selftest      # 约 1 分钟，要 ffmpeg
python3 training/train_utils.py --selftest     # 约 3 分钟；最后一项确认导出的 tflite 能被 bench 的 video_rescore.py 直接读
```

- TensorFlow 保持 **2.20**：现用模型是 TF 2.20.0 导出的（tflite 里记着），app 的运行时是 TFLite 2.13。换了版本脚本会提醒，
  导出的模型要在 app 里实际加载一次。
- 依赖就是旧脚本那些（tensorflow、opencv-python、numpy；h5py 随 Keras 装好），不再需要 scikit-learn。
- 下面的命令都在仓库根目录跑。训练输出放仓库外面（例子里是 `~/ml/movinet/runs`），不要提交权重。

## 1. 检查素材（约 10~20 分钟）

```bash
D=~/ml/movinet/movinet_clips
python3 training/data_check.py scan  --data_dir $D --jobs 8      # → $D/bad_clips.txt、$D/clip_index.csv
python3 training/data_check.py stats --data_dir $D
python3 training/data_check.py leak  --data_dir $D --videos <13 部原片目录>
```

- `scan`：顺序解码每个片段。打不开、读不出帧、少于 30 帧（12fps 的 3 秒应为 36 帧）的记进 `bad_clips.txt`，
  `train_video.py` 自动剔除；几乎全黑的只报不剔（剧情里的黑场是正常素材）。
- `stats`：每个 split / 类别有多少片段、多少部源视频、片段最多的几部占多少。同一个源视频名出现在两个 split 里会报警
  （train/val/test 应按片子划分），人工确认一下是不是同一部片。同一部片常被切成几段、文件名带时间（`SDDE-697-00.10.47.97`、
  `SDDE-697-00.30.55.17`），所以还会按番号（从文件名里猜的）归并再查一遍跨 split 的，并数一下同一番号既有剧情又有动作的有几部。
- `leak`（方案"开始前确认"的第 2 条）：测试片每 0.25 秒取一帧算画面指纹，和每个训练片段的 3 个指纹比，同一个源视频
  ≥3 个片段对上同一部测试片就报"疑似混入"。按 `bench/gt/vNN.csv` 文件头的 `file_name` 在 `--videos` 里找原片，名字对不上用
  `--pair v07=/path/视频7.mp4`。原片不在训练机上时，把 `$D/clip_index.csv` 拷到原片那台电脑，放进一个空目录当 `--data_dir` 就能查。

## 2. E1a：新基线

```bash
R=~/ml/movinet/runs
python3 training/train_video.py --preset e1a --data_dir $D --out_dir $R/e1a --dry_run   # 几分钟走通全流程（写到 $R/e1a_dry）
python3 training/train_video.py --preset e1a --data_dir $D --out_dir $R/e1a
```

- 正式跑：stage 1（只训线性层，≤10 个 epoch）+ stage 2（微调，≤20 个 epoch），各自按 `val_auc_do` 早停。每个 epoch 的
  样本数与旧脚本相同（训练片段数 ÷ batch）。第一个 epoch 结束会打出用时，按它估总时间（早停前最多 30 个 epoch）。
- 每个 epoch 结束打一行验证指标：`AUC(p_do)`、准确率、按片子平均的准确率（最差那部）、loss、按 app 阈值的误判率、置信度 ≥0.99 的占比。
  出现 `[读帧失败，跳过]` 说明有 scan 没查出来的坏片段，把它加进 `bad_clips.txt`。
- 结束时：在 val / test（12 帧）上的最终指标和各自最差的 10 部片（整部几乎全错的先人工看一眼是不是标错），导出
  `$R/e1a/e1a.tflite`，最后两行导出核对都要是 ✓。
- 导出核对拿 CPU 上的 Keras（float32）当参照。GPU 上 RTX 30/40 系默认用 TF32 算卷积（只有约 3 位有效数字），GPU 上的 Keras 和
  TFLite 能差到 1e-2，这一行只作参考。2026-10-17 之前的版本拿 GPU 结果当参照，会误报 ✗：`git pull` 后
  `--export_only $R/e1a` 重新核对（几分钟，不用重训）。
- 输出：`train.log`、`config.json`、`train_stage{1,2}_log.csv`、`best_stage{1,2}.weights.h5`、`e1a.tflite`、`e1a.keras`、`summary.json`。
  中断了重跑即可；只想重新导出用 `--export_only $R/e1a`。
- 可选参照：`python3 training/eval_clips.py --data_dir $D --split test --model <仓库>/app/src/main/assets/mobilenetv3small_tap_3class_float32.tflite --model $R/e1a/e1a.tflite`
  （现用模型当初按这份 val 集选的 checkpoint，所以比 test；`--split val` 也能跑，看两个模型是不是错在同几部片上）。
  每个模型后面列出最差的 10 部片（`--worst N` 改个数）。这些只是训练素材上的数字，比模型以 13 部片为准。

## 3. 在 13 部测试片上比（放原片的电脑上，仓库根目录）

```bash
python3 bench/tools/baseline.py rescore --videos <原片目录> --model <路径>/e1a.tflite --name e1a --jobs 4
python3 bench/tools/baseline.py compare exact_crop e1a
python3 bench/tools/frame_diag.py --name exact_crop --name e1a
```

- `rescore` 每部片约 3 分钟（要 numpy、av、ai-edge-litert 或 tensorflow，同 2026-10-15 做 exact_crop 时）。
- 先看模型级（`frame_diag` 的前半和 `compare` 的视频 AUC 几行），再看端到端：
  - 合并 AUC、逐片平均 AUC 上升，最差片（现在 v07 0.671）不下降，多数片变好；
  - 成段的高置信度错误（现在 19 段：误转 180 秒、漏转 310 秒）和置信度 ≥0.99 的错误占比（现在 31%）下降；
  - 只在 v01~v07 或只在 v08~v13 一半上变好的改动要谨慎（`frame_diag` 会提醒）；
  - 端到端先用现用参数比（README 第 5 节：代价降 ≥5%、最多 1 部片变差且 ≤5%、误停不增加、seek 进动作段启动不变慢、漏转不上升）。
- `exact_crop` 的参照报告已经在 `bench/baseline/`：`exact_crop.txt`、`frame_diag_exact_crop.txt`、`retune_exact_crop_conf.txt`。

## 4. E1b / E1c / E1d：各在 E1a 上只改一样

```bash
for p in e1b e1c e1d; do
  python3 training/train_video.py --preset $p --data_dir $D --out_dir $R/$p
done                                                    # 可以挂一晚上
```

| 实验 | 在 E1a 上改什么 | 注意 |
|---|---|---|
| E1b | stage 2 从中段 `expanded_conv_6` 起解冻（BN 仍冻结），AdamW lr 1e-4（weight decay 0.004，bias 不衰减），cosine 衰减，1 个 epoch warmup | stage 1 与 E1a 完全相同，想省半小时可以 `--init $R/e1a/best_stage1.weights.h5 --stage1_epochs 0`；全部解冻再加 `--unfreeze all`（另起名字，按验证集 AUC 二选一，不要两个都拿去 13 部片上挑） |
| E1c | label smoothing ε=0.05 | 最高概率会压到约 0.97，端到端要两边各自重调门槛再比：`python3 bench/tools/retune.py compare exact_crop e1c` |
| E1d | 训练时每个片段只抽 4 帧（等分 4 段、每段随机一帧）；验证、选 checkpoint、导出都是 12 帧 | 只为省显存、提速；跟 E1a 比的是同样的 12 帧评测 |

每个做完都按第 3 节 rescore，然后一起比：

```bash
python3 bench/tools/baseline.py compare exact_crop e1a e1b e1c e1d
python3 bench/tools/frame_diag.py --name exact_crop,e1a,e1b,e1c,e1d
python3 bench/tools/retune.py compare exact_crop e1c          # 只有 E1c 需要
```

## 5. 组合成 P1 的最佳配方

- 每个改动单独和 E1a 比；模型级指标变好、端到端不变差的才保留。
- 把保留的改动叠在一起再训一次，例如 E1b、E1c 都有用：
  `python3 training/train_video.py --preset e1b --label_smoothing 0.05 --data_dir $D --out_dir $R/p1`，再按第 3 节和 exact_crop、单项实验比。
- 差别很小时（合并 AUC 差不到 0.005、代价差不到 3%）当作噪声：同一个配方重训一次本身就会有这么大的波动。想量一下，可以
  `--preset legacy`（旧配方，只换了读帧）重训一次和 exact_crop 比，或者对最好的候选换个 `--seed` 再训一次。
- P1 的结果决定 Phase 2（补素材）重训时用的配方；Phase 4 的逐帧特征缓存先用现模型做。

## 6. 做完发回来什么

每个实验：`$R/<名字>/summary.json`、`train.log`、`train_stage1_log.csv`、`train_stage2_log.csv`（可以像 E1a 那样拷进
`training/results/<名字>/` 提交，权重和 tflite 不提交），以及 bench 的报告
`bench/baseline/compare_*.txt`、`frame_diag_*.txt`（E1c 还有 `retune_compare_*.txt`）。也可以把
`bench/rescored/video_<名字>_runs_initial|_runs_test`（每个变体约 7 MB）和 `bench/baseline/` 的报告提交到这个分支，
这样逐帧诊断、重调都能直接复现。

## 参数和取舍

- **源视频的采样权重**（`--film_alpha`，默认 0.5）：0 = 每部源视频等权（片段少的片子会被反复抽到），1 = 按片段数（= 不按片子
  均衡）。0.5 是两者之间的常用折中；训练日志开头会打出每个类别里片段最多的几部"占片段多少 → 采样多少"。
- **增强**都是温和的、对 app 输入合理的变化（app 的取帧是居中裁剪，不会有大幅裁剪和变形）；不做色相变化（肤色是线索）。
  参数在 `train_utils.AUG_MILD`。
- **BN 一律冻结**（推理模式），与旧脚本相同；这也保证训练模型和导出模型数值一致。
- 选 checkpoint 只用训练素材自己的 val；val 是按片子划分的，和 train 的片子不重叠。test 只看、不参与选择。
- 其余参数都能单独覆盖，见 `python3 training/train_video.py --help`；常用的：`--batch_size`（显存不够改 4）、`--workers`
  （GPU 利用率低、数据跟不上时调大）、`--seed`。
