# GearXAI 实验记录

## 任务理解

GearXAI 是解释型齿轮箱故障诊断任务。输入是 8 通道振动时间序列窗口，每个窗口长度为 100；输出是 9 类故障诊断概率，同时需要输出同尺寸 relevance map，说明模型关注的通道和时间区域。最终模型需要导出为 CPU 可运行的 ONNX 文件。

官方规则里先用 macro-F1 做 0.80 eligibility gate；通过后解释性排名由 faithfulness、mechanical alignment、simplicity 组合。公开 devkit 不提供 private mechanical band config，所以本地只能完整计算分类、faithfulness 和 simplicity，official hidden score 需要平台重算。

## 文献与可借鉴做法

调研结论：

- 旋转机械故障诊断常用 raw vibration 1D CNN/TCN，而不是先手工抽特征。Ince 等和 Eren 等的 1D CNN 工作说明，短时振动窗口中的局部冲击形态可直接由卷积核学习，适合本题 `[8, 100]` 输入。
- Zhang 等的 noisy/different-load bearing fault diagnosis 工作强调负载/转速变化下要用正则、扰动增强和更稳健的训练策略，避免模型只记住单一工况幅值。
- 多尺度 CNN/TCN 类方法通常同时看短周期冲击和更长时间上下文；本项目已有膨胀 depthwise conv，保留这一思路。
- 齿轮/轴承故障具有频率特征。文献中常见做法是 STFT/CWT/spectrum 与时域分支融合；但官方 ONNX 输入只能是 `[N, 8, 100]`，所以本次采用固定 DFT basis 的 ONNX 友好频谱分支，把频域幅值作为模型内部特征，不改变提交接口。
- XAI 评分采用 deletion/insertion faithfulness，本地 relevance map 不能只做漂亮热图。当前 relevance 继续融合 learned gate、输入能量和 decoder 输出，并保留稀疏/TV 正则，保证解释图集中且不剧烈噪声化。

因此第二版主要改动是：时域 TCN + 固定 DFT 频谱分支、EMA 权重验证/保存、数据审计脚本、NumPy 2.x 兼容的 devkit package wrapper。

## 当前方法

第一版采用 PyTorch 1D CNN/TCN 模型：

- 输入先经过 relevance gate，得到 `[N, 8, 100]` 的初始关注图。
- 分类主干使用膨胀残差卷积扩大时间感受野，用 depthwise temporal conv 保留每个传感器通道的局部故障模式。
- 残差块中加入 ChannelSE 通道注意力，让模型自动调节不同通道的重要性。
- 最终 relevance map 融合 learned gate、局部振动能量和特征解码器输出，范围约束到 `[0, 1]`。

训练损失包括：

- 交叉熵分类损失；
- relevance 稀疏正则，鼓励解释区域集中；
- relevance 时间/通道 total variation 正则，鼓励解释图平滑连续。

## 工程实现

已完成：

- 数据加载与 label 解析：`src/gearxai_project/data.py`
- 模型定义：`src/gearxai_project/model.py`
- 训练与验证：`src/gearxai_project/train.py`
- 评估脚本：`src/gearxai_project/evaluate.py`
- ONNX 导出与 onnxruntime 验证：`src/gearxai_project/export_onnx.py`
- smoke test：`scripts/smoke_test.py`
- 数据 schema 探查：`src/gearxai_project/inspect_data.py`

## 已验证结果与训练记录

本地已完成不依赖数据集的 smoke test：

- PyTorch 前向输出概率形状为 `[2, 9]`；
- relevance map 形状为 `[2, 8, 100]`；
- 概率和为 1；
- relevance map 数值在 `[0, 1]`；
- ONNX 导出成功；
- ONNXRuntime CPU 前向验证通过。

数据集读取问题已经通过 Hugging Face 镜像和项目内缓存解决：

- 在导入 `datasets` 之前设置 `HF_ENDPOINT=https://hf-mirror.com`；
- 数据缓存放在 `data/hf_cache`；
- `windows_100` 配置已缓存，train split 为 737352 条，validation split 为 83790 条。

已完成两组有代表性的训练：

1. `large_v1`：30000 train / 6000 validation，8 epoch。最好第 8 轮，validation `accuracy=0.8545`，`macro_f1=0.8530`。
2. `large_v2_100k`：100000 train / 20000 validation，8 epoch。最好第 7 轮，validation `accuracy=0.8781`，`macro_f1=0.8772`。
3. `large_v3_reg_100k`：100000 train / 20000 validation，8 epoch。将 dropout 从 0.12 提到 0.18、label smoothing 从 0.03 提到 0.05、weight decay 从 0.01 提到 0.02。最好第 8 轮，validation `accuracy=0.8846`，`macro_f1=0.8842`。

当前推荐使用 `large_v3_reg_100k` 的 best checkpoint：

- PyTorch checkpoint：`runs/large_v3_reg_100k/best.pt`
- ONNX：`runs/large_v3_reg_100k/model.onnx`

第二版 `spectral_v1_100k`：

- 训练命令：`python -m gearxai_project.train --config configs/default.yaml --max-train-samples 100000 --max-val-samples 20000 --epochs 8 --batch-size 256 --num-workers 0 --output-dir runs/spectral_v1_100k`
- 新模型：时域 CNN/TCN + 固定 DFT 频谱分支，EMA decay 0.995。
- 训练在第 5 轮发生 PyTorch CPU segfault，前 4 轮 checkpoint 正常保存；best 为第 4 轮。
- 20k validation：accuracy 0.9139，macro F1 0.9138。
- 完整 validation：accuracy 0.9114，macro F1 0.9118。
- ONNX 完整 validation raw input 复评：accuracy 0.9114，macro F1 0.9118。
- 5k devkit public validation：macro F1 0.9129，faith_score 0.5753，simplicity_score 0.6370，eligible true。
- 提交包：`runs/spectral_v1_100k/submission_normalized.zip`，package inspect valid。

用完整 validation split 复评 `large_v3_reg_100k` best checkpoint，结果为：

- accuracy：0.8803
- macro F1：0.8803
- per-class F1：HEA 0.7901，CTF 0.9143，MTF 0.9212，RCF 0.7908，SWF 0.7078，BWF 0.9165，CWF 0.9685，IRF 0.9575，ORF 0.9561

## 指标分析

从 `large_v1` 到 `large_v2_100k`，主要提升来自扩大训练样本：2 万验证子集上的 macro F1 从约 0.847 提高到 0.877。`large_v3_reg_100k` 进一步说明更强正则有效，2 万验证子集 macro F1 提高到 0.884，完整 validation split macro F1 提高到 0.880。

类别贡献上，CTF、MTF、BWF、CWF、IRF、ORF 已经比较稳定，多数 F1 在 0.91 以上；限制总分的主要仍是 HEA、RCF、SWF。完整 validation 上最大混淆集中在：

- RCF -> SWF：1358
- SWF -> HEA：1302
- HEA -> SWF：1288
- SWF -> RCF：1081
- HEA -> RCF：654
- RCF -> HEA：524

这说明后续调参应优先改善 HEA/RCF/SWF 的边界，而不是单纯增加模型宽度。

`spectral_v1_100k` 相比 `large_v3_reg_100k` 的完整 validation macro F1 从 0.8803 提高到 0.9118。主要收益：

- RCF F1 从约 0.7908 提高到 0.9230，频谱分支明显改善了裂纹类与健康/磨损类的边界。
- SWF F1 从约 0.7078 提高到 0.7933，但仍是最低类别。
- HEA F1 从约 0.7901 提高到 0.8194，仍和 SWF 互相混淆。
- 完整 validation 上最大残余混淆：HEA -> SWF 1743，SWF -> HEA 1220，SWF -> RCF 232，MTF -> CTF 622，BWF -> CWF/IRF/ORF 共 828。

数据审计结论：

- train split 737352 条，validation split 83790 条；完整 validation 每类 9310 条，严格均衡。
- validation 有 19 个 condition_id，每个 4410 条；fixed_speed_load 39690 条，variable_speed 44100 条。
- variable_speed 样本的 `speed_hz/load_nm` 为 `unknown`，不是异常；固定工况中 speed_hz 为 20/30/40/50，load_nm 为 0-5。
- 原始信号没有 non-finite 值，没有近零标准差窗口；但部分通道/窗口 std 很低，必须保留 per-window channel normalization。
- validation raw channel std 范围约 0.799-1.314，channel 4 方差最大；train 100k 随机审计 raw channel std 约 0.937-1.021，说明不同 split/工况分布存在幅值差异，嵌入 ONNX 的输入标准化是必要的。

当前最值得继续优化的是 SWF/HEA 边界，可以尝试：更强的工况均衡采样、针对 HEA/RCF/SWF 的 hard-example mining、或更细的 class-conditional relevance/auxiliary loss。

## 下一步调参方向

- 正则细调：当前 0.18 dropout、0.05 label smoothing、0.02 weight decay 已优于旧配置；下一步可尝试 dropout 0.16/0.20 与 label smoothing 0.04/0.06 的小网格，观察 validation loss 和 SWF/RCF F1。
- 数据增强：只对训练集加入轻微幅值缩放、时间平移和小噪声，目标是让模型减少对局部幅值偶然模式的过拟合。
- 更大数据：如果时间允许，用 200k 或完整 train split 训练，但需要关注训练准确率过快到 1.0 时是否仍能提升验证 F1。
- 针对性评估：每次实验都重点记录 HEA、RCF、SWF 的 F1 和三者之间的混淆数，避免只看总体 accuracy。
