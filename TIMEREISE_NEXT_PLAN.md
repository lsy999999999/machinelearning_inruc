
# TimeREISE 后续优化计划

## 1. 当前结论

当前 public-faith 最优主提交更新为：

```text
runs/final_candidates/logic_timereise_marginal_val5k_b10_bestproxy_submission.zip
```

当前最优配置：

```text
tag: marg_dw050_p010_tb015
faith=0.770607
deletion_auc=0.146884
insertion_auc=0.688098
macro_f1=0.983717
simplicity=0.901616
public_proxy=0.488566
```

Package inspect：

```text
valid=true
eligible=true
model_sha256=5d6b744cc549550f89168b28a719a6a0ded8545d2d44a272347a3ac070a865cc
zip_sha256=571196dc84c725c7530ff4b9a0162d36e09337f8a175016304798d844b1d6636
```

当前路线的优势很明确：

- 不改分类器，`macro_f1` 稳定在很高水平。
- 解释图直接对齐官方 deletion / insertion faithfulness。
- ONNX 结构很轻，`simplicity` 不会明显受损。
- marginal b10 直接针对 devkit faith 公式优化，把 `deletion_auc` 从 `0.186942` 压到 `0.146884`，同时 `insertion_auc` 仍保持 `0.688098`。

但短板也明确：

- 新候选使用 public validation 做 faith-targeted 校准，public proxy 很强，但 hidden split 过拟合风险高于 robust-only 版本。
- hidden mechanical score 本地不可见，只追 public proxy 有过拟合风险。
- mechanical alignment 仍然没有本地可计算的官方 band config。

因此，接下来策略是：

```text
public 提交优先使用 marginal_val5k_b10。
robust_offline_contrast_l050 保留为 hidden-stability 备选。
不再扩大 train50k raw marginal b20，因为它已被验证会明显损伤 insertion AUC。
```

## 2. 对 Claude 建议的判断

Claude 没有看到完整代码，所以它的方向只能作为启发，不能直接照搬。

合理的部分：

- 继续围绕 TimeREISE 做解释图增强是对的。
- 引入工况稳健性是合理的，因为 hidden split 可能和 public validation 分布不同。
- 引入轻量机械先验有价值，因为官方 mechanical score 占 40%。
- 混淆类对比解释有诊断意义，可以作为副线尝试。

需要修正的部分：

- 当前 metadata 没有直接可用的 speed/load condition 字段，不能直接做真实工况分组。
- 官方 mechanical band config 是隐藏的，不能强行假设具体频带。
- 频域因子不能权重过大，否则可能损坏当前 faithfulness 排名。
- 多尺度 bins 更像参数扩展，创新性和收益都不如 robust / mechanical-aware 副线。

## 3. 优先级

### 优先级 1：Robust TimeREISE

目标：

```text
降低 public 5k 过拟合风险，让 relevance factor 对不同输入分布更稳定。
```

当前数据没有明确 condition id，所以采用伪工况分组：

- 按窗口 RMS 分为低 / 中 / 高。
- 按 dominant frequency 分为低 / 中 / 高。
- 按 crest factor 或峰均比分为低 / 中 / 高。

每个组分别统计 TimeREISE 的 drop / keep factor，然后聚合：

```text
group_factor_g = TimeREISE(group_g)
final_factor = mean / median / trimmed_mean(group_factor_g)
```

第一轮只做小搜索：

```text
mode = mix50
num_bins = 20
beta = 0.34, 0.35, 0.36
aggregate = mean, median, trimmed_mean
```

候选名称：

```text
robust_mix50_tb034
robust_mix50_tb035
robust_mix50_tb036
```

评估标准：

- `public_proxy` 是否超过 `0.481032`。
- `faith` 是否接近或超过 `0.751771`。
- deletion / insertion 曲线是否正常。
- `simplicity` 是否保持在 `0.90` 附近。

这条是最值得先做的方向，因为它不依赖隐藏机械频带假设，风险最低。

## 4. 优先级 2：Mechanical-Aware TimeREISE

目标：

```text
在不破坏当前 faithfulness 的前提下，让 relevance map 更符合机械诊断直觉。
```

不要直接大幅修改 relevance，而是在当前最优基础上加入轻量机械 proxy：

```text
relevance =
    sqrt(abs(input))
    * timereise_factor
    * (1 + gamma * mechanical_proxy)
```

`gamma` 只做小范围：

```text
gamma = 0.05, 0.10, 0.15
```

mechanical proxy 不使用隐藏频带假设，优先使用可解释、ONNX 友好的信号统计：

- 局部冲击能量：短窗口局部峰值或局部均值差。
- 通道稳定响应：类别通道统计强度，但权重要轻。
- 高频 proxy：相邻差分能量 `abs(x[t] - x[t-1])`。
- 周期/局部结构 proxy：短窗口 average pool 后的残差峰值。

候选名称：

```text
mech_mix50_tb035_g005
mech_mix50_tb035_g010
mech_mix50_tb035_g015
```

保留标准：

```text
public_proxy >= 0.4800
faith 不明显低于当前最优
simplicity 不明显下降
```

如果 public proxy 没超过当前最优，但只小幅下降，可以作为汇报中的 mechanical-risk mitigation 副线，不替换主提交。

## 5. 优先级 3：Contrastive TimeREISE

目标：

```text
让解释图突出区分类别的证据，而不只是突出预测类别的高能量区域。
```

当前 TimeREISE 是：

```text
factor = sum_c p_c * factor_c
```

可以改成：

```text
factor = sum_c p_c * factor_c - lambda * contrast_factor
```

contrast_factor 可以先用简单版本：

```text
contrast_factor = factor_top2
```

也就是削弱第二高概率混淆类也认为重要的位置，让 relevance 更偏向 top1 类的判别证据。

搜索范围：

```text
lambda = 0.05, 0.10, 0.15
base = mix50_tb035
```

候选名称：

```text
contrast_mix50_tb035_l005
contrast_mix50_tb035_l010
contrast_mix50_tb035_l015
```

风险：

- 如果 top1 和 top2 共享真实故障证据，contrast 可能误伤 faithfulness。
- 因此这条优先级低于 robust 和 mechanical-aware。

## 6. 暂不优先：Multi-Scale Bins

多尺度 TimeREISE 有一定意义，例如：

```text
bins = 10, 20, 25, 50
factor = w10 * factor10 + w20 * factor20 + w25 * factor25 + w50 * factor50
```

但当前不优先，原因是：

- 它更像参数扩展，容易继续变成 public proxy 搜索。
- 搜索空间变大，过拟合风险更高。
- 机械解释性提升不如 robust / mechanical-aware 明确。

如果前三条没有收益，再考虑做一个很小的多尺度验证：

```text
0.2 * b10 + 0.5 * b20 + 0.2 * b25 + 0.1 * b50
beta = 0.35
```

## 7. 决策规则

任何副线候选只有满足以下条件才考虑替换主提交：

- `public_proxy > 0.481032`。
- 或者 `public_proxy` 基本持平，但解释曲线更稳、理论上 hidden mechanical 风险更低。
- `macro_f1` 保持在 `0.9837` 附近。
- `simplicity` 保持在 `0.90` 附近。
- deletion / insertion 曲线没有异常。
- 生成的 ONNX 能通过 devkit validation 和 package inspect。

否则：

```text
主提交不换。
副线作为报告中的创新尝试和消融实验。
```

## 8. 接下来具体执行顺序

### Step 1：补 TimeREISE 报告

新增或完善：

```text
TIMEREISE_REPORT.md
```

内容包括：

- 为什么选择不重训分类器。
- drop / keep 的定义。
- `mix50` 为什么优于单独 drop / keep。
- `beta=0.35` 的细搜结果。
- soft vs hard 的对比。
- 当前 hidden mechanical 风险。
- 后续 robust / mechanical-aware / contrastive 设计。

### Step 2：实现 Robust TimeREISE

基于 `tools/run_logic_timereise_search.py` 新增脚本或参数：

```text
tools/run_logic_timereise_robust_search.py
```

先实现：

- RMS 分组。
- 每组独立统计 drop / keep。
- mean / median / trimmed_mean 聚合。
- 只搜索 `mix50` 和 `beta=0.34, 0.35, 0.36`。

### Step 3：实现 Mechanical-Aware 小搜索

基于当前 `mix50_tb035` 的 factor 加轻量 proxy。

优先实现 ONNX 简单算子：

- `Abs`
- `Sqrt`
- `Sub`
- `AveragePool`
- `Relu`
- `Mul`
- `Add`

避免引入复杂频域 ONNX 图，先保证 simplicity 不崩。

### Step 4：实现 Contrastive 小搜索

在 soft class weighting 里加入 top2 / non-top1 抑制项。

只做小范围：

```text
lambda = 0.05, 0.10, 0.15
```

### Step 5：统一评估和记录

每条副线都记录：

```text
tag
faith
deletion_auc
insertion_auc
macro_f1
simplicity
public_proxy
是否 package 成功
是否值得替换主提交
```

最终形成一个表：

```text
current_mix50_tb035
robust_best
mechanical_best
contrastive_best
```

## 9. 汇报包装

不要把当前工作描述成：

```text
在 baseline 上调 relevance map 和 beta。
```

应该描述成：

```text
我们把诊断准确性和解释生成解耦：分类器使用高 F1 的轻量 LogicLSTM 保持稳定诊断性能；解释模块采用 evaluation-aware perturbation distillation，通过 deletion 和 insertion 统计每个类别、通道、时间块对模型置信度的因果贡献。进一步地，我们设计了工况稳健聚合、轻量机械先验和混淆类对比解释，以降低只优化 public faithfulness 的风险，并提升 hidden mechanical relevance 的可信度。
```

## 10. 最终原则

```text
当前最优主提交不轻易替换。
后续实验服务于两个目标：
1. 尝试找到比 mix50_tb035 更强的候选。
2. 即使没有超过，也要补足方法创新性和汇报说服力。
```
