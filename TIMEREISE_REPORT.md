# TimeREISE Report

## Executive Summary

当前主提交保持为：

```text
runs/final_candidates/logic_timereise_50k_b20_mix50_tb035_bestproxy_submission.zip
```

对应模型来自 `mix50_tb035`，本地 5k devkit validation 指标为：

| tag | faith | deletion_auc | insertion_auc | macro_f1 | simplicity | public_proxy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| mix50_tb035 | 0.751771 | 0.186616 | 0.690158 | 0.983717 | 0.901619 | 0.481032 |

本阶段不重训分类器。诊断模型继续使用官方轻量 LogicLSTM 的高 F1 分类输出，解释模块独立替换为 TimeREISE-style perturbation distillation，避免分类性能和解释图搜索互相牵制。

## Why We Do Not Retrain the Classifier

GearXAI 评分同时包含诊断准确性、faithfulness、mechanical relevance 和 simplicity。当前 LogicLSTM 在本地 devkit validation 上 `macro_f1=0.983717`，已经明显高于此前自训 spectral/causal 分支。重训分类器的收益不确定，但会引入三类风险：

- `macro_f1` 回落会直接损伤诊断部分，并影响 deletion / insertion 使用的预测类别置信度。
- 更复杂的分类器通常增加 ONNX 图复杂度，压低 `simplicity`。
- 解释图和分类器共同训练容易把优化目标混在一起，难以判断 faithfulness 提升来自模型行为还是 relevance 后处理。

因此当前方案把诊断和解释解耦：分类器保持稳定，解释图通过扰动统计对齐官方 faithfulness。

## Drop / Keep Definitions

TimeREISE 统计以类别、通道和时间块为单位。对每个样本先运行基础模型，取预测类别 `y_hat` 和原始置信度 `p_base = p(y_hat | x)`。

`drop` 定义为删除一个通道时间块后的置信度下降：

```text
drop(c, b) = max(p_base - p(y_hat | x with block(c,b)=0), 0)
```

它对应 deletion faithfulness：如果某个区域是真正重要证据，删除后预测置信度应该下降。

`keep` 定义为只保留一个通道时间块时的预测类别置信度：

```text
keep(c, b) = p(y_hat | only block(c,b) kept)
```

它对应 insertion faithfulness：如果某个区域本身携带足够诊断证据，只插入该区域时置信度应该较高。

统计时按预测类别聚合，得到 `factor[class, channel, time_bin]`。ONNX 推理阶段使用 soft class weighting：

```text
factor(x) = sum_k p_k(x) * factor_k
relevance = sqrt(abs(x)) * factor(x)
```

soft weighting 比 hard argmax 更平滑，也避免类别边界附近 relevance 突变。

## Why mix50 Works Best

单独 `drop` 更偏向删除敏感区域，容易强调模型脆弱点；单独 `keep` 更偏向局部可识别片段，可能忽略需要组合证据的故障形态。`mix50` 使用二者等权融合：

```text
mix50 = 0.5 * normalize(drop) + 0.5 * normalize(keep)
```

它同时对齐 deletion 和 insertion，避免只优化单侧曲线。50k / 20 bins 细搜显示，`mix50` 的最优 public proxy 明显高于加入 ratio channel prior 的版本，也高于单独 drop / keep 的主力候选。

## Beta Fine Search

`beta` 控制时间块扰动 factor 偏离 1 的强度：

```text
time_factor = 1 + beta * (expanded_factor - 1)
```

50k / 20 bins refine 的局部结果如下：

| tag | faith | deletion_auc | insertion_auc | macro_f1 | simplicity | public_proxy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| mix50_tb034 | 0.751516 | 0.187126 | 0.690157 | 0.983717 | 0.901619 | 0.480930 |
| mix50_tb035 | 0.751771 | 0.186616 | 0.690158 | 0.983717 | 0.901619 | 0.481032 |
| mix50_tb036 | 0.751366 | 0.187288 | 0.690019 | 0.983717 | 0.901619 | 0.480870 |

`beta=0.35` 是当前局部峰值。更大的 beta 会让解释图过度集中，deletion 有时改善但 insertion 或综合 faithfulness 回落；更小的 beta 则区分度不足。

## Soft vs Hard

hard 版本使用 `argmax(probabilities)` 选择单个类别 factor，soft 版本使用概率加权。当前最佳附近：

| tag | faith | deletion_auc | insertion_auc | macro_f1 | simplicity | public_proxy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| mix50_tb035 | 0.751771 | 0.186616 | 0.690158 | 0.983717 | 0.901619 | 0.481032 |
| mix50_tb035_hard | 0.751741 | 0.186723 | 0.690205 | 0.983717 | 0.901622 | 0.481021 |

hard 的 simplicity 轻微更高，但 faith 和 proxy 略低。主提交保留 soft `mix50_tb035`。

## Hidden Mechanical Risk

当前 local proxy 只显式包含 devkit 可见的 faithfulness、classification 和 simplicity：

```text
public_proxy = 0.4 * faith_score + 0.2 * simplicity_score
```

隐藏 leaderboard 还包含 mechanical relevance。官方 mechanical band config 不公开，当前 metadata 也没有直接可用的 speed/load condition 字段。因此不能强行假设具体机械频带；过度追 public proxy 可能让 relevance 变成 evaluation-aware heatmap，而不是机械诊断上稳定的证据图。

## Follow-up Branches

新增三条副线脚本，均不替换当前主提交，除非明确超过 `public_proxy=0.481032` 或 public 持平且理论 hidden 风险更低。

| branch | script | idea | search |
| --- | --- | --- | --- |
| Robust TimeREISE | `tools/run_logic_timereise_robust_search.py` | 用 RMS、dominant frequency、crest factor 伪工况分组，分别统计再聚合 | `beta=0.34/0.35/0.36`, `mean/median/trimmed_mean` |
| Mechanical-aware TimeREISE | `tools/run_logic_timereise_mech_search.py` | 在 `mix50_tb035` 上轻量加入相邻差分和局部峰值 proxy | `gamma=0.05/0.10/0.15` |
| Contrastive TimeREISE | `tools/run_logic_timereise_contrastive_search.py` | 抑制 top2 混淆类也认为重要的位置 | `lambda=0.05/0.10/0.15` |

保留规则：

- `public_proxy > 0.481032` 才优先考虑替换主提交。
- 若 public proxy 基本持平，优先选择 mechanical-aware 或 robust 这种 hidden 更稳的设计。
- `macro_f1` 应保持在 `0.9837` 附近。
- `simplicity` 应保持在 `0.90` 附近。
- deletion / insertion 不应出现单侧异常。
- 最终候选必须能通过 devkit evaluation、package 和 inspect。

## Branch Results

三条副线均已在本地 5k devkit validation 上完成评估。`--no-package` 用于初筛，只有超过当前主提交的 robust bestproxy 额外执行了 package 和 inspect。

| candidate | faith | deletion_auc | insertion_auc | macro_f1 | simplicity | public_proxy | decision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| current_mix50_tb035 | 0.751771 | 0.186616 | 0.690158 | 0.983717 | 0.901619 | 0.481032 | current main |
| robust_mean_mix50_tb035 | 0.751933 | 0.186384 | 0.690251 | 0.983717 | 0.901616 | 0.481097 | new packaged candidate |
| mech_mix50_tb035_g005 | 0.750203 | 0.188436 | 0.688842 | 0.983717 | 0.890313 | 0.478144 | do not replace |
| contrast_mix50_tb035_l005 | 0.752718 | 0.185510 | 0.690947 | 0.983717 | 0.894312 | 0.479950 | faith improves, proxy loses |

Robust 分支的最佳候选是：

```text
runs/final_candidates/logic_timereise_robust_mean_mix50_tb035_bestproxy_submission.zip
```

Package inspect 已通过：

```text
valid=true
eligible=true
model_sha256=6f107b447e854eb821568d9daae9a48f6ff03a843b6ae2f775dd746d97dd4ec9
zip_sha256=42e6438a9fdb1afcdd1508cb0db5d8aeed0366efa9d9c7cc7969cb0558ea4513
```

这个候选的 public proxy 比当前主提交高 `0.0000649`，提升很小，但它来自伪工况 robust 聚合，理论上比单一全局 TimeREISE factor 更能降低 public split 过拟合风险。因此它可以作为新的替换候选；若策略偏保守，仍可保留原 `mix50_tb035` 主提交，把 robust 作为消融和 hidden-risk mitigation 结果汇报。

Mechanical-aware 第一版不保留为替换候选，原因是轻量 proxy 虽然符合机械直觉，但 ONNX 节点和 relevance 分布改变导致 simplicity 明显下降，faith 也同步下降。Contrastive 第一版把 faith 提到 `0.752718`，说明 top2 抑制确实能强化判别解释；但 `TopK` 等算子增加复杂度，public proxy 低于主提交，适合作为报告中的诊断性消融，不适合作为最终提交。

## Low-complexity Offline Innovation

第一版 mechanical-aware 和 contrastive 的共同问题是在线 ONNX 算子增加了复杂度。第二轮把创新全部折叠到离线 `weights_9x8x100`，ONNX 结构保持为 TimeREISE 的轻量形式：

```text
relevance = sqrt(abs(input)) * soft_class_weighted_factor
```

新增脚本：

```text
tools/run_logic_timereise_offline_innovation_search.py
```

离线 mechanical prior 使用相邻差分能量、局部峰值、绝对幅值和 crest factor，在训练样本上按预测类别、通道、时间块聚合，然后直接融合进 factor。离线 contrastive 不再使用 `TopK`，而是在 factor 级别抑制其他类别共享的高响应区域：

```text
factor_c = factor_c - lambda * max(mean_factor_other_classes - 1, 0)
```

小搜索结果：

| candidate | faith | deletion_auc | insertion_auc | macro_f1 | simplicity | public_proxy | decision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| robust_mean_mix50_tb035 | 0.751933 | 0.186384 | 0.690251 | 0.983717 | 0.901616 | 0.481097 | previous robust best |
| robust_offline_contrast_mix50_tb035_l050 | 0.752061 | 0.186942 | 0.691064 | 0.983717 | 0.901611 | 0.481146 | new best |
| robust_offline_contrast_mix50_tb035_l080 | 0.752011 | 0.186864 | 0.690886 | 0.983717 | 0.901611 | 0.481126 | backup |
| robust_offline_mech_mix50_tb035_g020 | 0.751253 | 0.187447 | 0.689954 | 0.983717 | 0.901612 | 0.480824 | do not replace |
| robust_offline_combo_mix50_tb035_g020_l050 | 0.751452 | 0.187331 | 0.690234 | 0.983717 | 0.901610 | 0.480903 | do not replace |

新的最佳候选是：

```text
runs/final_candidates/logic_timereise_robust_offline_contrast_l050_bestproxy_submission.zip
```

Package inspect 已通过：

```text
valid=true
eligible=true
model_sha256=20c3337f0c56f580bcbdb5b4ef27b2b3996b6ffbef6082789d26be2510cb554b
zip_sha256=c75ff97fa37cc63182ab55e6c9b4626936b2e4d987b0f3b19f5199a3bc98334b
```

这个候选比原 `mix50_tb035` 提升 `0.000114` public proxy，比 robust-only 候选提升 `0.000049`。提升仍然很小，但它同时满足两个条件：ONNX complexity 基本不变，faithfulness 和 insertion 曲线有增益。因此当前替换优先级更新为：

```text
1. logic_timereise_robust_offline_contrast_l050_bestproxy_submission.zip
2. logic_timereise_robust_mean_mix50_tb035_bestproxy_submission.zip
3. logic_timereise_50k_b20_mix50_tb035_bestproxy_submission.zip
```

Mechanical offline 版本保留为消融，不作为提交候选。它证明了低复杂度实现可以把 simplicity 拉回 `0.9016` 附近，但本轮 mechanical prior 与 public faithfulness 不同向；除非后续有 hidden mechanical 反馈，否则不再扩大这条搜索。
