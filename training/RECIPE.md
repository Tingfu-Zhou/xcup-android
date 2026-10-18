# 视频模型训练配方：E1d（P1 定稿，2026-10-18）

**一句话**：模型结构不变（MobileNetV3Small、12 帧时间平均、160×160），训练方法换成 E1a 的一组改动（剔除坏片段、
按源视频均衡采样、片段级增强、按验证集 AUC 选模型、导出后真实片段核对），训练时每个片段只抽 4 帧。
脚本是 `training/train_video.py`，**默认就是这个配方**；旧的 `train_mobilenetv3_tap_3class.py` 不再用。

## 怎么训

```bash
cd ~/ml/xcup-android && git pull
source ~/ml/movinet/.venv/bin/activate
D=~/ml/movinet/movinet_clips
R=~/ml/movinet/runs

python3 training/data_check.py scan --data_dir $D --jobs 8          # 素材有增减时重跑：更新 $D/bad_clips.txt
python3 training/train_video.py --data_dir $D --out_dir $R/v2 --dry_run   # 几分钟走通全流程
python3 training/train_video.py --data_dir $D --out_dir $R/v2             # 正式训练，约 2 小时（RTX 4070 笔记本）
```

- 输出在 `$R/v2/`：`v2.tflite`（与 app 现用模型同结构、同输入输出，可以直接放进 `app/src/main/assets/`）、
  `best_stage{1,2}.weights.h5`、`train.log`、`summary.json` 等。最后两行"导出核对"都应是 ✓。
- 换素材后（Phase 2）也用同一条命令，只改 `--out_dir`。在 13 部测试片上和 `e1d`（app 现用）、`exact_crop`（P1 之前的模型）比，
  按 `training/README.md` 的"在 13 部测试片上比"（`baseline.py rescore` → `compare exact_crop e1d <名字>`）。
- TensorFlow 用 2.20（现用模型就是 2.20.0 导出的；app 的运行时是 TFLite 2.13）。

## 配方

| 项 | 设定 |
|---|---|
| 素材 | `movinet_clips/<train,val,test>/<normal_plot,oral,sex>/`：`make_movinet_clips.py` 切的 3 秒片段，短边缩到 224、居中裁 224×224、12fps、x264 crf 23；train/val/test 按片子分。`data_check.py scan` 查出的坏片段（打不开、读不出帧、少于 30 帧）从 `bad_clips.txt` 剔除 |
| 类别 | `[normal_plot, oral, sex]`（与 app 相同）；p_do = p_oral + p_sex |
| 读帧 | 顺序解码全部帧再挑，读不出来的样本跳过并报出（不用全黑帧顶替） |
| 训练取帧 | **4 帧**：片段等分 4 段，每段随机一帧（TSN） |
| 验证 / 选模型 / 导出 | 12 帧，`np.linspace(0, n−1, 12)` 均匀取（帧间隔约 250ms，与 app 相同） |
| 缩放 | `cv2.INTER_AREA` 缩到 160×160（与 app 的 FramePreprocessor 相同） |
| 采样 | 类别等权 → 类别内按 片段数^0.5 选源视频 → 源视频内等概率选片段；不用 class_weight。每个 epoch = 训练片段数 ÷ 8 步（现在 3805 步） |
| 增强（同一片段的所有帧用同一组参数） | 随机裁剪：面积 0.7~1、宽高比 3/4~4/3；水平翻转 50%；亮度 ±10%、对比度 ±10%、饱和度 ±15%；10% 概率高斯模糊 σ 0.4~1.0；15% 概率 JPEG 质量 40~85；不改色相 |
| 模型 | 每帧过 MobileNetV3Small（ImageNet 权重，自带归一化，输入 0~255）→ 576 维 → 帧间平均 → Dropout 0.5 → Dense(3, softmax) |
| Stage 1 | backbone 全部冻结，只训线性层；Adam lr 1e-3；最多 10 个 epoch，按 `val_auc_do` 早停（patience 4），取最好的一轮 |
| Stage 2 | 从 stage 1 最好的权重开始；解冻 backbone 最后 30 层（第 128 层起），**BN 一律冻结**；Adam lr 1e-5，`ReduceLROnPlateau`（val_loss，×0.3，patience 3，最低 1e-7）；最多 20 个 epoch，按 `val_auc_do` 早停（patience 6），取最好的一轮 |
| 损失 | categorical cross-entropy，不做 label smoothing |
| batch | 8 个片段（× 4 帧） |
| 选模型 | 验证集（12 帧）的 p_oral+p_sex AUC（`val_auc_do`）；两个 stage 里最好的那一轮 |
| 导出 | 搭成与现用模型相同的 12 帧结构（TimeDistributed + 时间平均 + 线性层，共享权重）→ SavedModel → float32 TFLite（只用内置算子）。核对：结构是 1 个 PACK × 12 + 1 个 FULLY_CONNECTED（bench 能读逐帧 logit）；48 个真实验证片段上与 CPU 上的 Keras 概率差 ≤ 1e-4 |
| 随机种子 | 42 |

预设里的完整参数见 `train_video.py` 的 `PRESETS['e1d']`（= `PRESETS['e1a']` + `train_frames=4`）；增强参数是 `train_utils.AUG_MILD`。

## P1 里 E1d 那一次的结果

- 训练（2026-10-18，118 分钟）：stage 1 第 8 个 epoch 最好（val AUC 0.9410），stage 2 第 11 个 epoch 最好（0.9537）；
  test AUC 0.9826、准确率 88.1%。记录在 `training/results/e1d/`。
- 13 部测试片，对 P1 之前的模型 `exact_crop`（报告：`bench/baseline/compare_exact_crop_vs_e1d.txt`、
  `frame_diag_exact_crop_vs_e1d.txt`、`retune_compare_exact_crop_vs_e1d_conf.txt`）：

| | exact_crop（P1 之前的模型） | E1d |
|---|---|---|
| 视频 AUC：合并 / 逐片平均 / 最差片 v07 | 0.935 / 0.930 / 0.671 | 0.948 / 0.943 / 0.715 |
| 逐片 AUC 变好 / 变差 | | 5 部（v01 v03 v07 v12 v13）/ 0 部 |
| 成段的高置信度错误：误转 / 漏转 | 180 / 310 秒 | 120 / 271 秒 |
| 端到端代价（现用参数） | 1219 | 1099（−10%；v09 +25，是对白否决阈值边上的偶然波动） |
| 端到端代价（两边各自留一视频重调参数） | 1219 | 1073（−12%），验收线全部通过 |

- 2026-10-18 起 app 用的就是 E1d（`app/src/main/assets/mobilenetv3small_tap_3class_float32.tflite`，sha1 63f3e033cf38），
  阈值和融合参数没改，即上表"现用参数"一行。
  - 留一重调里 12/13 折选了 `BV_CONF 0.80`（对白否决，FusionArbiter）和 `VIDEO_PLOT 0.75`（视频"不转"阈值，两个 Activity）。
  - 但对 E1d 自己的现用参数只降 2.4%（1099 → 1073），特写 seek p90 还慢 1.5 秒，没过验收线，所以先不改。
  - Phase 4 换上最终模型时按同样流程再定。要改就按 `bench/README.md` 第 4 节改代码、重新生成对拍数据，再上真机复验。

## 试过、没采用的

| | 改了什么 | 为什么不用 |
|---|---|---|
| E1a | 训练用 12 帧 | 与 E1d 效果相当（两次独立训练提升一致，说明这组改动有效），但训练慢 37% |
| E1b | 从中段解冻 + AdamW lr 1e-4 + cosine | 训练不稳（val AUC 在 0.92~0.96 之间跳，最好的是第 1 个 epoch）；最差片 v07 好得最多，但动作判成剧情变多、v05 多一次误停、v04 变差。lr 降到 3e-5 可以再试 |
| E1c | label smoothing 0.05 | 置信度被压低（≥0.99 的窗口 65% → 10%），融合里按置信度的规则失效，端到端变差（1375），重调也救不回来 |

这三个的训练记录和 13 部片上的报告 2026-10-18 从仓库删了，要看就从 git 历史取回：记录和报告在提交 `e8848b0`
（如 `git checkout e8848b0 -- training/results/e1b`），重算数据（`bench/rescored/video_e1{a,b,c}_*`）在 `aa9defc`。

## 已知的问题（留给 Phase 2）

- 训练素材的验证集里约 22% 的剧情片段被判成"转"（所有候选都一样，stage 1 只训线性层时就这样），有整部片全错：
  `eval_clips.py --split val` 列出最差的片子，先人工看是不是标错。
- 同一部片常被切成几段（如 `SDDE-697-00.10.47.97` / `SDDE-697-00.30.55.17`）：`data_check.py stats` 按番号查有没有跨 split。
