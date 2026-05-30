# GearXAI Fault Diagnosis Starter

这是 GearXAI 解释型齿轮箱故障诊断作业的初始代码仓库。目标是训练一个 PyTorch 模型，输入 8 通道、长度 100 的振动窗口，输出 9 类故障概率和同尺寸 relevance map，并导出为官方平台需要的 ONNX 模型。

官方资料：

- 赛题主页：https://gearxai-ijcai-ecai2026.pages.dev
- 数据集：https://huggingface.co/datasets/edi45/gearxai-dds-seu

## 0. 当前第一版思路

本仓库实现的是一个可训练、可导出、可验证的第一版方案：

- 输入使用官方 `windows_100` 数据配置，窗口为 8 通道、100 个时间点。
- 模型是轻量 1D CNN/TCN：先生成输入级 relevance gate，再用膨胀残差卷积、depthwise temporal conv 和通道注意力提取故障特征。
- relevance map 输出尺寸固定为 `[N, 8, 100]`，由 learned gate、局部振动能量和特征解码器融合得到，范围约束在 `[0, 1]`。
- 训练目标是交叉熵分类损失，加 relevance 稀疏性和时间/通道平滑正则，避免解释图全亮或剧烈噪声。
- 导出 ONNX 后会默认运行 `onnx.checker` 和 `onnxruntime` CPU 前向检查，减少提交包接口错误。

## 1. 环境安装

建议在服务器上使用 Python 3.10+。

```bash
git clone <你的 GitHub 仓库地址>
cd <仓库目录>
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
```

如果只需要 CPU 训练/导出，建议先安装 CPU 版 PyTorch，再装项目依赖，避免默认 PyPI 拉取大型 CUDA wheel：

```bash
pip install --index-url https://download.pytorch.org/whl/cpu torch
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -e .
```

如果 Hugging Face 数据集需要登录：

```bash
hf auth login
```

还需要在 Hugging Face 数据集页面接受访问条款，否则 `datasets.load_dataset` 可能无法下载数据文件。

国内网络环境下，本项目已在数据加载代码导入 Hugging Face 库之前强制设置：

```python
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
```

默认数据缓存位置是项目内的 `data/hf_cache`，避免在受限环境里写用户目录缓存失败。

## 2. 快速 smoke test

不下载数据，只检查模型前向、ONNX 导出和 onnxruntime CPU 推理：

```bash
python scripts/smoke_test.py
```

## 3. 数据检查

先用 streaming 方式只读取少量样本，确认字段、shape 和 label 解析：

```bash
python -m gearxai_project.inspect_data --max-samples 4 --count-labels
```

如果当前网络到 Hugging Face 较慢，可以用 `--timeout-seconds 30` 快速失败，避免命令长时间卡住。

如果想统计已下载数据集中的完整 split，可以关闭 streaming：

```bash
python -m gearxai_project.inspect_data --no-streaming --split train --count-labels
```

## 4. 训练

先用小样本确认流程：

```bash
python -m gearxai_project.train \
  --config configs/default.yaml \
  --max-train-samples 2000 \
  --max-val-samples 500 \
  --epochs 2 \
  --patience 2
```

正式训练：

```bash
python -m gearxai_project.train --config configs/default.yaml
```

训练产物默认保存在 `runs/gearxai_cnn_gate/`：

- `best.pt`：验证集 macro F1 最好的 checkpoint
- `last.pt`：最后一个 epoch 的 checkpoint
- `metrics.jsonl`：每个 epoch 的指标
- `best_metrics.json`：最佳 epoch 的 loss、accuracy、macro F1、per-class F1 和 confusion matrix
- `last_metrics.json`：最近一个 epoch 的完整指标

默认配置使用 cosine learning-rate schedule。若显存不足，优先把 `configs/default.yaml` 里的 `training.batch_size` 从 384 下调到 256 或 128。

当前较好的本地第二版训练命令：

```bash
python -m gearxai_project.train \
  --config configs/default.yaml \
  --max-train-samples 100000 \
  --max-val-samples 20000 \
  --epochs 4 \
  --batch-size 256 \
  --num-workers 0 \
  --output-dir runs/spectral_v1_100k
```

这个版本在模型内部加入固定 DFT 频谱分支，并使用 EMA 权重做验证和保存。当前 best checkpoint 在第 4 轮，2 万验证子集上 `accuracy=0.9139`、`macro_f1=0.9138`。用完整 validation split 复评后，PyTorch 和 ONNX 均为 `accuracy=0.9114`、`macro_f1=0.9118`。

备注：一次 8 epoch CPU 训练在第 5 轮发生 PyTorch 进程级 segfault，但前 4 轮 checkpoint 已正常保存，且第 4 轮是当前 best。后续在 CPU 环境建议先跑 4 epoch 或设置 patience；GPU/Kaggle 上可继续延长训练。

## 5. 导出 ONNX

训练后可以先复测一次 checkpoint：

```bash
python -m gearxai_project.evaluate \
  --checkpoint runs/gearxai_cnn_gate/best.pt
```

如果 checkpoint 训练时用了 `--max-val-samples`，但想复评完整 validation split：

```bash
python -m gearxai_project.evaluate \
  --checkpoint runs/spectral_v1_100k/best.pt \
  --full-val \
  --batch-size 512 \
  --num-workers 0
```

```bash
python -m gearxai_project.export_onnx \
  --checkpoint runs/spectral_v1_100k/best.pt \
  --output runs/spectral_v1_100k/model_normalized.onnx
```

导出脚本默认会做 ONNX checker 和 onnxruntime CPU 推理检查。若只想导出文件，可加 `--skip-verify`。

导出的 ONNX 接口：

- 输入：`input`，形状 `[N, 8, 100]`
- 输出 1：`probabilities`，形状 `[N, 9]`
- 输出 2：`relevance_map`，形状 `[N, 8, 100]`

也可以直接评估导出的 ONNX 文件，确认提交文件和 PyTorch checkpoint 的分数一致：

```bash
python -m gearxai_project.evaluate_onnx \
  --model runs/spectral_v1_100k/model_normalized.onnx \
  --config runs/spectral_v1_100k/config.resolved.json \
  --full-val \
  --batch-size 512 \
  --num-workers 0 \
  --no-normalize
```

当前 ONNX 在完整 validation split 上为 `accuracy=0.9114`、`macro_f1=0.9118`。

## 6. 官方提交包

官方 devkit 的 README 里通常会提供类似下面的命令，具体参数以官方仓库为准：

```bash
gearxai package --model runs/gearxai_cnn_gate/model.onnx --output submission.zip
```

如果官方命令要求额外 metadata 或示例输入，请按 devkit 说明补齐。

当前 NumPy 2.x 环境下官方 devkit v1.0.1 的 `gearxai package` 可能因为 `np.trapz` 被移除而失败。本项目提供了等价 wrapper：

```bash
python -m gearxai_project.package_devkit package \
  --model runs/spectral_v1_100k/model_normalized.onnx \
  --data-dir prepared_hf_val5k \
  --split validation \
  --out runs/spectral_v1_100k/submission_normalized.zip \
  --batch-size 512
```

本地 5k devkit validation 报告：`macro_f1=0.9129`、`faith_score=0.5753`、`simplicity_score=0.6370`、`eligible=true`。生成的 `runs/spectral_v1_100k/submission_normalized.zip` 已通过 package inspect。

## 7. 汇报可以讲的点

- 数据特点：这是多通道短窗口振动信号，模型需要同时判断故障类别和给出通道-时间二维解释图。
- baseline 改动：没有只输出分类结果，而是在分类主干前后都引入 relevance 分支，并把 relevance 用于输入 gating。
- 正则设计：稀疏正则让解释图更集中，TV 平滑正则让时间轴解释连续，避免提交图过于噪声化。
- 工程可靠性：加入 smoke test、数据探查脚本、best/last 指标文件和 ONNXRuntime 验证，便于定位数据、训练或提交问题。
- 后续实验：可以比较 `base_channels=64/96/128`、`depth=4/5/6`、`sparse_weight/tv_weight` 和 batch size 对 macro F1 与解释图质量的影响。

## 8. 推到 GitHub

我已经把本地项目整理成可以直接建仓上传的结构，并初始化了本地 Git 仓库。你在 GitHub 新建一个空仓库后，在本机运行：

```bash
git remote add origin git@github.com:<你的用户名>/<仓库名>.git
git push -u origin main
```

之后服务器上就可以 `git clone` 下载。
