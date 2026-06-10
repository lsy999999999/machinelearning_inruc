# 对 Claude 建议的代码复核修正版

日期：2026-06-10

结论先写清楚：Claude 的大方向“不要重训大模型，继续做局部精修”是对的，但它没有看到当前代码和历史实验，所以有几处需要修正。

  注意取舍：如果按老师口径 0.6*F1 + 0.3*faith + 0.1*simplicity，新包优先。如果更担心官方 hidden explainability proxy，原
  来的 logic_timereise_row_coordinate_ext_bestproxy_submission.zip public proxy 仍更高、更稳。

当前主线不是重新训练，而是：

```text
固定 LogicLSTM 主分类器
优先优化 relevance map
必要时只做很小的分类输出校准
每个候选都必须完整 devkit 复评
```

当前稳定 best 仍然是：

```text
runs/final_candidates/logic_timereise_row_coordinate_ext_bestproxy_submission.zip
```

本地 devkit：

```text
macro_f1     = 0.983716900
faith        = 0.795904409
deletion_auc = 0.122045794
insertion_auc= 0.713854611
simplicity   = 0.901619365
teacher_score= 0.919163399
```

其中：

```text
teacher_score = 0.6 * macro_f1 + 0.3 * faith + 0.1 * simplicity
```

注意：devkit 自带的官方 explainability proxy 不直接奖励 F1，只把 macro-F1 当 eligibility gate；如果最终按老师口径算分，F1 校准版有价值。如果按官方 explainability proxy 排名，原 row-coordinate 版本更稳。

---

## 1. Claude 建议哪些适合，哪些要调整

### 方法 1：logit bias / class calibration

判断：可以用，但 Claude 给的参数范围太小，而且不能假设“几乎无副作用”。

实际代码情况：

```text
ONNX 内部是 Gemm -> Softmax -> probabilities
relevance 又使用 probabilities @ weights_flat 做软类别加权
```

所以 logit bias 不只是改分类结果，也会连带改变 relevance 和 faith。必须完整 devkit 复评。

我已经执行：

新增脚本：

```text
tools/run_logic_logit_bias_calibration.py
```

先跑 Claude 风格小网格：

```text
ORF 降一点，CWF/IRF 升一点，RCF/HEA 小幅调整
```

结果：

```text
base macro_f1 = 0.983716900
best macro_f1 = 0.983716900
changed_predictions = 0
```

原因不是方向错，而是当前错例 margin 太大。关键错例 log-prob margin：

```text
CWF -> ORF: n=22, min=0.276, median=7.203, max=24.667
IRF -> ORF: n=7,  min=1.189, median=4.344, max=11.361
HEA -> RCF: n=10, min=3.808, median=15.622, max=27.614
```

这说明 `0.05/0.10/0.20` 级别 bias 只能碰到极少数边界样本，大部分错误不是简单校准能修。

随后扩大网格并做完整 devkit 复评。当前老师口径最好的校准候选是：

```text
b_ORF = -2.0
b_CWF = +1.5
其他类别 bias = 0
```

生成包：

```text
runs/final_candidates/logic_timereise_row_coordinate_ext_logit_bias_teacherbest_submission.zip
```

本地 devkit：

```text
macro_f1     = 0.985095578
faith        = 0.795498763
deletion_auc = 0.122422480
insertion_auc= 0.713420007
simplicity   = 0.900796797
teacher_score= 0.919786655
```

相对当前稳定 best：

```text
macro_f1     +0.001378677
faith        -0.000405645
simplicity   -0.000822568
teacher_score+0.000623256
```

混淆变化主要是：

```text
CWF -> ORF: 22 -> 17
IRF -> ORF: 7  -> 5
IRF -> CWF: 6  -> 7
```

判断：

```text
如果最终按老师口径提交：teacherbest 可以作为新冲分候选。
如果更看重官方 explainability proxy / hidden 稳定性：原 row_coordinate_ext 更稳。
```

---

### 方法 2：hard-example fine-tuning

判断：现在不建议作为主线。

原因：

```text
当前 macro_f1 已经 0.9837+
重训会同时影响分类、faith、simplicity 和 ONNX 稳定性
现有错误里很多 margin 极大，不是最后一层轻微 fine-tuning 一定能可靠修复
```

如果后面一定要做，只能作为备选实验：

```text
冻结大部分层
只调分类头或最后一层
训练 1-3 epoch
只接受完整 devkit teacher_score 上升且 faith 不明显下降的候选
```

不要把它排在 relevance / calibration 前面。

---

### 方法 3：condition-aware validation

判断：思路有价值，但当前导出的 devkit 数据不能直接做。

Claude 提到 `condition_id/speed_hz/load_nm/regime`，但当前本地文件：

```text
prepared_hf_val5k/validation_metadata.jsonl
```

实际只有：

```text
index
label
source_split
```

因此现在不能直接按工况分组计算 CWF recall、ORF precision 或 faith。要做这个分析，需要重新从 Hugging Face 原始 dataset 导出 metadata，并保留这些字段。当前建议改成：

```text
短期：继续做类别/混淆/margin 诊断
中期：重新导出带 condition 字段的 validation metadata
然后再做 condition-wise table
```

不要在报告里声称已经做了工况分析。

---

### 方法 4：relevance gamma sharpening

判断：适合，但已经基本做过，而且最终不是用单一全局 gamma。

历史结果已经覆盖：

```text
weights = normalize(weights ** p)
全局最佳 p ≈ 1.26
faith = 0.790501
```

后面又做了：

```text
class-specific sharpen
class row selection
row-coordinate full devkit search
```

最终到：

```text
faith = 0.795904
```

所以 Claude 建议的：

```text
gamma = 0.7, 0.9, 1.0, 1.2, 1.5, 2.0
```

已经太粗，不适合作为下一轮主实验。若继续做，只应该做类别级小范围 coordinate search，而不是重新跑全局 gamma。

---

### 方法 5：时间块大小搜索

判断：方向对，但 Claude 的写法和当前代码不完全一致。

当前代码主要用的是：

```text
num_bins
```

不是直接传 `block_size`。历史已经比较过：

```text
b10 / b20 / b25 / b50
```

结论：

```text
b10 最稳
b25/b50 变差
```

原因也和 devkit 一致：

```text
faith top-k mask 是 0%,10%,...,100%
b10 的解释粒度和评估粒度更匹配
更细 bin 会让扰动统计更噪
```

因此不建议再按 Claude 的 `{2,4,5,10}` 盲跑。若要补报告，可以把已有 b10/b25/b50 作为粒度消融。

---

### 方法 6：class-specific relevance smoothing

判断：调整后可以做，但优先级低于当前已经完成的 row-coordinate。

可行版本应该是：

```text
对 9x8x100 离线 weights 做时间方向平滑
生成若干 smoothed row candidate
再丢给 row-coordinate full devkit search
```

不要在 ONNX 里新增 AveragePool/Conv 去在线平滑 relevance，因为那会增加 operator_count，伤 simplicity。

建议只尝试很小网格：

```text
sigma ∈ {0.5, 1.0}
只对 CWF/ORF/IRF/RCF 生成候选行
只接受 full devkit teacher_score 或 faith 上升
```

---

### 方法 7：insertion-oriented relevance mix

判断：概念可以，但不是 Claude 写的那种直接混两个未知 map。

当前已有脚本已经用过 deletion / insertion marginal gain：

```text
drop_gain
insert_gain
drop_weight
penalty
power ensemble
```

最终高分来自这些候选的类别级选择，而不是简单：

```text
R_final = alpha * R_deletion + (1-alpha) * R_insertion
```

如果继续做，应复用已有 marginal gains 和候选权重，不要另起一套定义不清的 R_deletion/R_insertion。

---

### 方法 8：simplicity 微调

判断：只做防守，不主攻。

当前稳定版：

```text
operator_count  = 24
parameter_count = 79020
simplicity      = 0.901619365
```

logit-bias 版会变成：

```text
operator_count  = 25
parameter_count = 79029
simplicity      = 0.900796797
```

也就是说分类校准虽然提高老师口径总分，但会降低 simplicity 和官方 explainability proxy。后续所有 relevance 优化仍应优先离线折叠进常量权重，避免新增在线算子。

---

## 2. 我已经执行的计划

### 2.1 核对 ONNX 结构

当前 best 模型输出：

```text
probabilities
timereise_coord_step1_relevance
```

核心尾部结构：

```text
Gemm -> Softmax -> probabilities
Abs/Sqrt(input)
MatMul(probabilities, weights_flat)
Reshape
Mul -> relevance
```

这解释了为什么改 probabilities 会影响 relevance。

### 2.2 跑 logit-bias calibration

新增：

```text
tools/run_logic_logit_bias_calibration.py
```

执行了三层排查：

```text
1. Claude 小网格：无预测变化，拒绝
2. margin 诊断：确认多数错例不是小 bias 可修
3. 扩大 ORF/CWF/IRF 搜索并完整 devkit 复评
```

最终保留两个有意义候选：

稳定解释版：

```text
runs/final_candidates/logic_timereise_row_coordinate_ext_bestproxy_submission.zip
teacher_score = 0.919163399
public_proxy  = 0.498685636
```

老师口径 F1 冲分版：

```text
runs/final_candidates/logic_timereise_row_coordinate_ext_logit_bias_teacherbest_submission.zip
teacher_score = 0.919786655
public_proxy  = 0.498358865
```

其中 teacherbest 包检查通过：

```text
valid=true
eligible=true
model_sha256=11d7989ec5e64ec390ff0e2332a34131a261acc7b8788f40ef8af6845db6bec4
zip_sha256=fffd0d8e84c0a48cdbc44655ec87655fa29870362d076228be6c68413ffda5fe
```

---

## 3. 下一步建议

按当前结果，我建议最终保留三类候选：

```text
A. 稳定解释版：
   runs/final_candidates/logic_timereise_row_coordinate_ext_bestproxy_submission.zip

B. 老师口径 F1 冲分版：
   runs/final_candidates/logic_timereise_row_coordinate_ext_logit_bias_teacherbest_submission.zip

C. 保守备份版：
   runs/final_candidates/logic_timereise_class_candidate_selection_bestproxy_submission.zip
```

提交选择：

```text
如果老师最终明确用 0.6*F1 + 0.3*faith + 0.1*simplicity：
    优先考虑 B

如果更接近官方 explainability 排名，或担心 hidden distribution：
    优先考虑 A
```

后续还值得做但不要乱扩大的实验：

```text
1. 离线 class-specific smoothing rows -> row-coordinate full devkit
2. 重新导出带 condition_id/speed/load 的 metadata -> 只做分析和报告
3. 对 teacherbest 做 hidden-risk 说明：它是分类校准候选，不是 relevance 最优候选
```

不建议继续做：

```text
1. 大模型重训
2. 全局粗 gamma 网格
3. b25/b50 细粒度 TimeREISE 重跑
4. 在线 smoothing/复杂 relevance 算子
5. 没有完整 devkit 的单指标局部替换
```

---

## 4. 追加升级执行结果

在 `teacherbest` 基础上继续做了两步升级：

```text
Step 1: 用 teacherbest 的校准概率作为 base，重新做 full-devkit row-coordinate relevance 搜索
Step 2: 针对 class 8 做 folded weight morph，小范围搜索 physical proxy blend
```

### 4.1 修复 row-coordinate 二次生成问题

当 base model 本身已经是 TimeREISE 变体时，旧脚本生成 `coord_step1` 会和已有 initializer 重名：

```text
timereise_coord_step1_weights_flat initializer name is not unique
```

已修复：

```text
tools/run_logic_timereise_row_coordinate_search.py
```

把新 step tag 改为：

```text
coord_update1, coord_update2, ...
```

这样以后可以安全地以已有 TimeREISE/校准模型作为 base model 做二次 row-coordinate。

### 4.2 teacherbest + row-coordinate upgrade

执行：

```text
base-model  = runs/logic_logit_bias_calibration_aggressive_d/logic_timereise_row_coordinate_ext_logit_bias.onnx
start-model = runs/logic_logit_bias_calibration_aggressive_d/logic_timereise_row_coordinate_ext_logit_bias.onnx
候选池覆盖 stable / classsel / phys / marginal / robust / sharp / power ensemble
```

最终只接受一个替换：

```text
class 8 -> phys_all
```

生成包：

```text
runs/final_candidates/logic_timereise_teacherbest_rowcoord_upgrade_bestproxy_submission.zip
```

本地 devkit：

```text
macro_f1     = 0.985095578
faith        = 0.795792735
deletion_auc = 0.122746494
insertion_auc= 0.714331964
simplicity   = 0.900796209
teacher_score= 0.919874788
```

包检查：

```text
valid=true
eligible=true
model_sha256=eecfb8555e80667e755137d97dc4884173b1c591353f1a54ef4fed62cc3f0dbd
zip_sha256=d94163b1e7820034d0fb46ea3d69929968903f5fa5507451fd81aafea90c9dd1
```

### 4.3 folded weight morph upgrade

新增脚本：

```text
tools/run_logic_timereise_weight_morph_search.py
```

这个脚本只做离线 `weights_9x8x100` 变形，不新增在线复杂算子：

```text
temporal smoothing
class-row power
class-row physical proxy blend
可选 combo
```

小网格最优：

```text
tag = blend_phys_top_a035_c8
含义：在当前最高分 start 基础上，class 8 行再混入 35% phys_top row
```

生成包：

```text
runs/final_candidates/logic_timereise_weight_morph_bestproxy_submission.zip
```

本地 devkit：

```text
macro_f1     = 0.985095578
faith        = 0.795800691
deletion_auc = 0.122723194
insertion_auc= 0.714324576
simplicity   = 0.900793562
teacher_score= 0.919876910
```

包检查：

```text
valid=true
eligible=true
model_sha256=5f159da8bffe9bce2683b19fe061b1863f45b67a2dd2accbc0f3d4937ca6d691
zip_sha256=c2c6fdee68d1b99242a11cf2dccff15ff97593a4e17c2f474c1c8fabb8ce4ac4
```

当前老师口径最高候选变为：

```text
runs/final_candidates/logic_timereise_weight_morph_bestproxy_submission.zip
```

相对原稳定版：

```text
teacher_score: 0.919163399 -> 0.919876910
delta = +0.000713511
```

代价：

```text
official/public explainability proxy 低于原稳定 row-coordinate
operator_count 从 24 到 25
```

因此提交选择仍然是：

```text
老师口径优先：logic_timereise_weight_morph_bestproxy_submission.zip
hidden/official proxy 稳定优先：logic_timereise_row_coordinate_ext_bestproxy_submission.zip
```

---

## 5. 后续长命令

### 5.1 推荐先跑：长时间 folded weight morph 搜索

这个命令已用 smoke 确认能跑通。它会继续围绕当前最高分候选扩展 class 8 的 physical blend，同时细化 smoothing/power：

```bash
.venv/bin/python tools/run_logic_timereise_weight_morph_search.py \
  --output-dir runs/logic_timereise_weight_morph_long \
  --copy-prefix logic_timereise_weight_morph_long \
  --blend-alpha 0.05 0.10 0.15 0.20 0.25 0.30 0.35 0.40 0.45 0.50 0.60 0.70 0.80 0.90 \
  --blend-classes 8 \
  --smooth-passes 1 2 3 4 5 6 \
  --smooth-classes 6,7,8 \
  --power 0.90 0.95 1.00 1.03 1.05 1.08 1.10 1.12 1.15 1.20 1.25 \
  --power-classes 6,7,8
```

看结果：

```bash
.venv/bin/python -c "import json; d=json.load(open('runs/logic_timereise_weight_morph_long/summary_top.json')); print(json.dumps(d['best_proxy'][:10], indent=2)); print(json.dumps(d['best_faith'][:10], indent=2))"
```

如果它超过当前：

```text
teacher_score > 0.919876910
faith > 0.795800691
```

就继续用它作为 start 做 row-coordinate。

### 5.2 更长：对 morph long 的 top 候选做 row-coordinate

这个会比上面慢很多。建议在 morph long 有明显收益后再跑：

```bash
.venv/bin/python tools/run_logic_timereise_row_coordinate_search.py \
  --base-model runs/candidates/logic_timereise_weight_morph_long_bestproxy.onnx \
  --start-model runs/candidates/logic_timereise_weight_morph_long_bestproxy.onnx \
  --output-dir runs/logic_timereise_weight_morph_long_rowcoord_full \
  --copy-prefix logic_timereise_weight_morph_long_rowcoord_full \
  --passes 2 \
  --min-delta 1e-7 \
  --candidate \
  current=runs/candidates/logic_timereise_weight_morph_long_bestproxy.onnx \
  upgrade=runs/candidates/logic_timereise_weight_morph_bestproxy.onnx \
  phys_all=runs/logic_timereise_classsel_phys_proxy_all/logic_timereise_classsel_pred.onnx \
  phys_top=runs/logic_timereise_classsel_phys_proxy_top/logic_timereise_classsel_pred.onnx \
  pow047=runs/logic_timereise_power_ensemble_robust_fine/logic_timereise_b047.onnx \
  pow048=runs/logic_timereise_power_ensemble_robust_fine/logic_timereise_b048.onnx \
  sharp126=runs/logic_timereise_classsel_all_sharp_ultrafine/logic_timereise_sharp126.onnx
```

注意：`--candidate` 只能写一次，然后后面接完整 `name=path` 列表。重复写多个 `--candidate` 时，argparse 会只保留最后一组，导致候选池不完整。

### 5.3 不优先但可跑：长训练

训练 smoke 已确认能跑通：

```text
.venv/bin/python -m gearxai_project.train --config configs/spectral_lite_c.yaml --epochs 1 --max-train-samples 512 --max-val-samples 256 --batch-size 128 --output-dir runs/train_smoke_spectral_lite_c_512 --num-workers 0
```

但训练分支历史上明显低于官方 LogicLSTM 的 `macro_f1=0.9837+`，所以这是高风险备选。若一定要跑，建议：

```bash
.venv/bin/python -m gearxai_project.train \
  --config configs/spectral_lite_c.yaml \
  --epochs 20 \
  --max-train-samples 300000 \
  --max-val-samples 50000 \
  --batch-size 384 \
  --lr 8e-4 \
  --dropout 0.16 \
  --weight-decay 0.02 \
  --label-smoothing 0.04 \
  --augment \
  --noise-std 0.01 \
  --scale-range 0.05 \
  --time-shift 2 \
  --output-dir runs/spectral_lite_c_long_boundary \
  --num-workers 2
```

训练后导出和评估：

```bash
.venv/bin/python -m gearxai_project.export_onnx \
  --checkpoint runs/spectral_lite_c_long_boundary/best.pt \
  --output runs/spectral_lite_c_long_boundary/model.onnx \
  --relevance-mode abs

.venv/bin/python -m gearxai_project.evaluate_devkit \
  --model runs/spectral_lite_c_long_boundary/model.onnx \
  --data-dir prepared_hf_val5k \
  --split validation \
  --batch-size 512 \
  --output runs/spectral_lite_c_long_boundary/devkit_abs.json
```

只有满足下面条件才值得继续：

```text
macro_f1 >= 0.985
teacher_score > 0.919876910
```

否则不要用训练模型替换当前 folded TimeREISE 主线。

---

## 6. 5.1 / 5.2 长搜索实跑结果

你已经跑完 5.1 和 5.2。复核后结论如下。

### 6.1 5.1 morph long

5.1 的长网格最优不是之前的小网格 `blend_phys_top_a035_c8`，而是：

```text
tag = smooth6_c678
```

生成包：

```text
runs/final_candidates/logic_timereise_weight_morph_long_bestproxy_submission.zip
```

本地 devkit：

```text
macro_f1      = 0.985095578
faith         = 0.795821464
deletion_auc  = 0.122659853
insertion_auc = 0.714302781
simplicity    = 0.900796503
teacher_score = 0.919883436
public_proxy  = 0.498487886
```

相对上一版 `logic_timereise_weight_morph_bestproxy_submission.zip`：

```text
teacher_score: 0.919876910 -> 0.919883436
delta = +0.000006526
```

### 6.2 原 5.2 命令的问题

最开始按文档里重复写多个 `--candidate` 的方式跑出来的：

```text
runs/final_candidates/logic_timereise_weight_morph_long_rowcoord_bestproxy_submission.zip
```

实际候选池不完整，只测到了 `start` 和 `sharp126`，没有测完整候选列表。这个命令写法已经在上面修正。

### 6.3 修正版 5.2 full row-coordinate

用正确候选池重跑后，full devkit coordinate 接受了两个替换：

```text
class 6 -> upgrade
class 7 -> upgrade
```

生成包：

```text
runs/final_candidates/logic_timereise_weight_morph_long_rowcoord_full_bestproxy_submission.zip
```

本地 devkit：

```text
macro_f1      = 0.985095578
faith         = 0.795931709
deletion_auc  = 0.122692754
insertion_auc = 0.714556172
simplicity    = 0.900796209
teacher_score = 0.919916480
public_proxy  = 0.498531925
```

包检查：

```text
valid=true
eligible=true
model_sha256=3b61e3ec936495d2fd6fd4d18ae202a7f59e70a23824d863eb181dbca40c5dfa
zip_sha256=20500b7053e8b798006e3765747a3338b72371edd3e35ed02f80ce06422cd242
```

当前老师口径最高候选变为：

```text
runs/final_candidates/logic_timereise_weight_morph_long_rowcoord_full_bestproxy_submission.zip
```

相对原稳定版：

```text
teacher_score: 0.919163399 -> 0.919916480
delta = +0.000753081
```

相对 5.1 morph long：

```text
teacher_score: 0.919883436 -> 0.919916480
delta = +0.000033044
```

提交判断：

```text
老师口径优先：logic_timereise_weight_morph_long_rowcoord_full_bestproxy_submission.zip
hidden/official proxy 稳定优先：logic_timereise_row_coordinate_ext_bestproxy_submission.zip
```
