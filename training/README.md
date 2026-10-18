# training/ — 视频模型训练

> 训练配方（P1 定的 E1d）、每一项设定、P1 的结果和没采用的候选见 [`RECIPE.md`](RECIPE.md)。
> 这里是各个脚本怎么用，以及新模型怎么在 13 部测试片上和现有模型比。

模型结构：MobileNetV3Small、12 帧时间平均、160×160、类别 `[normal_plot, oral, sex]`（与 app 相同）。
- 13 部测试片（v01~v13）只用来比做完的候选。
- 选 checkpoint 只看训练素材自己的验证集。
- v01~v13 的任何片段都不能进训练素材，新增的素材也一样。

| 文件 | 作用 |
|---|---|
| `RECIPE.md` | 训练配方 E1d：怎么训、每一项设定、P1 的结果、已知的问题 |
| `train_video.py` | 训练入口，默认就是 E1d：两阶段训练 → 按验证集 AUC 选模型 → 导出 12 帧 tflite → 导出核对。`--preset e1a / e1b / e1c / legacy` 是 P1 的其它候选和旧配方 |
| `train_utils.py` | 训练积木：顺序读帧、片段级增强、按源视频均衡采样、按 p_do AUC 选模型、12 帧导出与 Keras / TFLite 一致性检查 |
| `data_check.py` | 素材检查：`scan` 找坏片段 → `bad_clips.txt`；`stats` 按 split / 类别 / 源视频统计；`leak` 查 13 部测试片有没有混进素材 |
| `eval_clips.py` | 在训练素材的 val / test 上评一个或几个 12 帧 tflite，列出最差的片子 |
| `results/e1d/` | E1d 那次训练的记录（`summary.json`、`train.log`、每个 epoch 的指标） |

bench 端比模型用 `bench/tools/baseline.py`（端到端 + 视频 AUC）、`frame_diag.py`（逐帧诊断）、`retune.py`（留一视频重调视频阈值），
见 `bench/README.md` 第 5 节。

## 在哪个系统里跑

| 步骤 | 在哪跑 |
|---|---|
| 准备、检查素材（`scan` / `stats`）、训练 | Ubuntu（训练用的 venv 和 GPU 在那边，素材在 `~/ml/movinet/movinet_clips`） |
| 在 13 部测试片上比（`baseline.py`、`frame_diag.py`、`retune.py`）和 `data_check.py leak` | 哪边都行，只要能读到 13 部原片和导出的 `.tflite` |

13 部原片不用挪，`--videos` 指向它们现在的位置就行；按 `bench/gt/vNN.csv` 文件头的 `file_name`（`视频1.mp4` ~ `视频13.mp4`）
找，文件名不同就用 `--pair v07=<路径>` 逐个指定。

- **Ubuntu 是 WSL2**（`uname -r` 带 `microsoft`）：Windows 的 D 盘就是 `/mnt/d`，全部在 Ubuntu 里做最省事，例如
  `--videos "/mnt/d/视频/测试视频"`。
  - bench 要多装一个包：`pip install av`。
  - `data_check.py` 的自检和 `leak` 要 ffmpeg：`sudo apt install ffmpeg`。
- **Ubuntu 是双系统**：在 Ubuntu 的文件管理器里点一下 Windows 的分区，它就会挂到 `/media/<用户名>/<分区名>/`，只读也够用。
  也可以回到 Windows 的 PowerShell 里跑 bench（下一条）。
- **在 Windows PowerShell 里跑 bench**：
  - 用 Windows 上的仓库；`python3` 换成 `python`；路径用 Windows 的写法。
  - 依赖：`pip install numpy av ai-edge-litert opencv-python-headless`。
  - WSL 里训练出的模型，在 Windows 下的路径是 `\\wsl$\Ubuntu\home\<用户名>\ml\movinet\runs\<名字>\<名字>.tflite`。

## 准备（训练机，一次）

```bash
cd ~/ml/movinet && source .venv/bin/activate
git clone https://github.com/Tingfu-Zhou/xcup-android.git ~/ml/xcup-android   # 已经有了就 cd 进去 git pull
cd ~/ml/xcup-android
python3 -c "import tensorflow as tf, keras; print(tf.__version__, keras.__version__)"
python3 training/data_check.py --selftest      # 约 1 分钟，要 ffmpeg
python3 training/train_utils.py --selftest     # 约 3 分钟；最后一项确认导出的 tflite 能被 bench 的 video_rescore.py 直接读
```

- **TensorFlow 保持 2.20。** app 里的模型是 TF 2.20.0 导出的（tflite 里记着），app 的运行时是 TFLite 2.13。
  换了版本脚本会提醒；导出的模型要在 app 里实际加载一次。
- **依赖：** tensorflow、opencv-python、numpy（h5py 随 Keras 装好）。
- **路径：** 下面的命令都在仓库根目录跑。训练输出放仓库外面（例子里是 `~/ml/movinet/runs`），权重和 tflite 不提交。

## 检查素材（约 10~20 分钟；素材有增减就重跑）

```bash
D=~/ml/movinet/movinet_clips
python3 training/data_check.py scan  --data_dir $D --jobs 8      # → $D/bad_clips.txt、$D/clip_index.csv
python3 training/data_check.py stats --data_dir $D
python3 training/data_check.py leak  --data_dir $D --videos <13 部原片目录>
```

- **`scan`：** 顺序解码每个片段，把有问题的记进 `bad_clips.txt`，`train_video.py` 会自动剔除。
  - 剔除：打不开、读不出帧、少于 30 帧（12fps 的 3 秒应为 36 帧）。
  - 几乎全黑的只报不剔：剧情里的黑场是正常素材。
- **`stats`：** 每个 split / 类别有多少片段、多少部源视频，片段最多的几部各占多少。
  - 同一个源视频名出现在两个 split 里会报警（train/val/test 应按片子划分），人工确认一下是不是同一部片。
  - 同一部片常被切成几段，文件名带时间（如 `SDDE-697-00.10.47.97`、`SDDE-697-00.30.55.17`）。所以 `stats` 还会按番号
    （从文件名里猜）归并，再查一遍跨 split 的，并数一下同一番号既有剧情又有动作的有几部。
- **`leak`：** 查 13 部测试片有没有混进素材。
  - 测试片每 0.25 秒取一帧算画面指纹，和每个训练片段的 3 个指纹比；同一个源视频有 ≥3 个片段对上同一部测试片，就报"疑似混入"。
  - 按 `bench/gt/vNN.csv` 文件头的 `file_name` 在 `--videos` 里找原片，名字对不上用 `--pair v07=/path/视频7.mp4`。
  - 原片不在训练机上时，把 `$D/clip_index.csv` 拷到原片那台电脑，放进一个空目录当 `--data_dir` 就能查。

## 训练

配方和每一项设定见 `RECIPE.md`。

```bash
R=~/ml/movinet/runs
python3 training/train_video.py --data_dir $D --out_dir $R/<名字> --dry_run   # 几分钟走通全流程（写到 $R/<名字>_dry）
python3 training/train_video.py --data_dir $D --out_dir $R/<名字>             # 约 2 小时（RTX 4070 笔记本）
```

- **每个 epoch：** 打一行验证指标：`AUC(p_do)`、准确率、按片子平均的准确率（最差那部）、loss、按 app 阈值的误判率、
  置信度 ≥0.99 的占比。第一个 epoch 结束会打出用时，可以按它估总时间。
- **`[读帧失败，跳过]`：** 说明有 `scan` 没查出来的坏片段，把它加进 `bad_clips.txt`。
- **结束时：**
  - 打出 val / test（12 帧）上的最终指标，以及各自最差的 10 部片。整部几乎全错的，先人工看一眼是不是标错。
  - 导出 `$R/<名字>/<名字>.tflite`，最后两行"导出核对"都要是 ✓。
- **导出核对的参照：** 拿 CPU 上的 Keras（float32）当参照。RTX 30/40 系的 GPU 默认用 TF32 算卷积，GPU 上的 Keras 和 TFLite
  能差到 1e-2，那一行只作参考。
- **输出：** `train.log`、`config.json`、`train_stage{1,2}_log.csv`、`best_stage{1,2}.weights.h5`、`<名字>.tflite`、`<名字>.keras`、
  `summary.json`。
- **中断、重新导出：** 中断了重跑即可；只想重新导出用 `--export_only $R/<名字>`。
- **实现上的一点：** 训练用的模型把 12 帧合成一批过 backbone（比 TimeDistributed 展开 12 份快），导出时再搭成和 app 里的模型
  一模一样的 TimeDistributed 结构（共享权重）。BN 始终冻结，两者数值相同，导出核对会查。
- **可选参照：** 在训练素材上和 app 现用模型比，每个模型后面列出最差的 10 部片（`--worst N` 改个数）。这只是训练素材上的数字，
  比模型以 13 部片为准。
  ```bash
  python3 training/eval_clips.py --data_dir $D --split val \
      --model app/src/main/assets/mobilenetv3small_tap_3class_float32.tflite --model $R/<名字>/<名字>.tflite
  ```
- **记录：** 训练记录（`summary.json`、`train.log`、`train_stage{1,2}_log.csv`）可以拷进 `training/results/<名字>/` 提交。

## 在 13 部测试片上比（放原片的电脑上，仓库根目录）

```bash
python3 bench/tools/baseline.py rescore --videos <原片目录> --model <路径>/<名字>.tflite --name <名字> --jobs 4
python3 bench/tools/baseline.py compare exact_crop e1d <名字>
python3 bench/tools/frame_diag.py --name exact_crop,e1d,<名字>
python3 bench/tools/retune.py compare exact_crop <名字>     # 两边各自留一视频重调后，按验收线比
```

**两个现成的对照**（重算数据在 `bench/rescored/`，报告在 `bench/baseline/`）：
- `exact_crop`：P1 之前的模型，固定对照。报告：`exact_crop.txt`、`frame_diag_exact_crop.txt`、`retune_exact_crop_conf.txt`。
- `e1d`：2026-10-18 起 app 用的模型。报告：`compare_exact_crop_vs_e1d.txt`、`frame_diag_exact_crop_vs_e1d.txt`、
  `retune_compare_exact_crop_vs_e1d_conf.txt`。

**`rescore`：**
- 每部片约 3 分钟，要 numpy、av、ai-edge-litert 或 tensorflow。
- 每个模型起单独的 `--name`。同一个名字换了模型会被拒绝（重算目录里的 `variant.json` 记着模型指纹）。
- 换一处原片目录第一次用之前，先 `baseline.py check --videos <原片目录> --model old.tflite`，确认 13 部都找得到、
  离线能复现手机。
  - `runs_initial` / `runs_test` 是用 P1 之前的模型录的，所以 `check` 要用那个旧模型。
  - 旧模型从 git 历史取：`git show 11bf834:app/src/main/assets/mobilenetv3small_tap_3class_float32.tflite > old.tflite`。

**先看模型级**（`frame_diag` 的前半，和 `compare` 的视频 AUC 几行）：
- 合并 AUC、逐片平均 AUC 上升，最差片不下降，多数片变好。
- 成段的高置信度错误和置信度 ≥0.99 的错误占比下降。
- 只在 v01~v07 或只在 v08~v13 一半上变好的改动要谨慎（`frame_diag` 会数两半各有几部变好 / 变差）。

**再看端到端**（`bench/README.md` 第 5、6 节）：
- 先用现用参数比。
- 再用 `retune.py compare` 两边各自留一视频重调。概率分布变了时（例如 label smoothing）必须看这一步。
- 验收线：代价降 ≥5%；最多 1 部片变差且 ≤5%；误停不增加；seek 进动作段启动不变慢；漏转不上升。

**噪声：** 差别很小时（合并 AUC 差不到 0.005、代价差不到 3%）当作噪声，同一个配方重训一次本身就会有这么大的波动。
想量一下，可以对最好的候选换个 `--seed` 再训一次。

## 参数和取舍

- **源视频的采样权重**（`--film_alpha`，默认 0.5）：
  - 0 = 每部源视频等权（片段少的片子会被反复抽到）；1 = 按片段数（= 不按片子均衡）；0.5 是两者之间的常用折中。
  - 训练日志开头会打出每个类别里片段最多的几部"占片段多少 → 采样多少"。
- **增强**都是温和的、对 app 输入合理的变化：app 的取帧是居中裁剪，不会有大幅裁剪和变形。
  - 不做色相变化（肤色是线索）。
  - 参数在 `train_utils.AUG_MILD`。
- **BN 一律冻结**（推理模式），这也保证训练模型和导出模型数值一致。
- **选 checkpoint** 只用训练素材自己的 val。val 按片子划分，和 train 的片子不重叠；test 只看、不参与选择。
- **其余参数**都能单独覆盖，见 `python3 training/train_video.py --help`。常用的：
  - `--batch_size`：显存不够改 4。
  - `--workers`：GPU 利用率低、数据跟不上时调大。
  - `--seed`。
