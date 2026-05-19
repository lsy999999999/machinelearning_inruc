# GearXAI Fault Diagnosis Starter

这是 GearXAI 解释型齿轮箱故障诊断作业的初始代码仓库。目标是训练一个 PyTorch 模型，输入 8 通道、长度 100 的振动窗口，输出 9 类故障概率和同尺寸 relevance map，并导出为官方平台需要的 ONNX 模型。

官方资料：

- 赛题主页：https://gearxai-ijcai-ecai2026.pages.dev
- 数据集：https://huggingface.co/datasets/edi45/gearxai-dds-seu

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

如果 Hugging Face 数据集需要登录：

```bash
huggingface-cli login
```

## 2. 快速 smoke test

不下载数据，只检查模型前向和 ONNX 导出逻辑：

```bash
python scripts/smoke_test.py
```

## 3. 训练

先用小样本确认流程：

```bash
python -m gearxai_project.train \
  --config configs/default.yaml \
  --max-train-samples 2000 \
  --max-val-samples 500 \
  --epochs 2
```

正式训练：

```bash
python -m gearxai_project.train --config configs/default.yaml
```

训练产物默认保存在 `runs/gearxai_cnn_gate/`：

- `best.pt`：验证集 macro F1 最好的 checkpoint
- `last.pt`：最后一个 epoch 的 checkpoint
- `metrics.jsonl`：每个 epoch 的指标

## 4. 导出 ONNX

训练后可以先复测一次 checkpoint：

```bash
python -m gearxai_project.evaluate \
  --checkpoint runs/gearxai_cnn_gate/best.pt
```

```bash
python -m gearxai_project.export_onnx \
  --checkpoint runs/gearxai_cnn_gate/best.pt \
  --output runs/gearxai_cnn_gate/model.onnx
```

导出的 ONNX 接口：

- 输入：`input`，形状 `[N, 8, 100]`
- 输出 1：`probabilities`，形状 `[N, 9]`
- 输出 2：`relevance_map`，形状 `[N, 8, 100]`

## 5. 官方提交包

官方 devkit 的 README 里通常会提供类似下面的命令，具体参数以官方仓库为准：

```bash
gearxai package --model runs/gearxai_cnn_gate/model.onnx --output submission.zip
```

如果官方命令要求额外 metadata 或示例输入，请按 devkit 说明补齐。

## 6. 推到 GitHub

我已经把本地项目整理成可以直接建仓上传的结构。你在 GitHub 新建一个空仓库后，在本机运行：

```bash
git init
git add .
git commit -m "Initial GearXAI starter"
git branch -M main
git remote add origin git@github.com:<你的用户名>/<仓库名>.git
git push -u origin main
```

之后服务器上就可以 `git clone` 下载。
