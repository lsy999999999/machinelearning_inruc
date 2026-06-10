# TimeREISE Current Route Summary

## Current Recommended Submission

Use this local package for the current public-faith / public-proxy best submission:

```text
runs/final_candidates/logic_timereise_class_candidate_selection_bestproxy_submission.zip
```

Local devkit validation:

```text
tag=classsel_pred
method=class-specific row selection over marginal/robust/power candidates
macro_f1=0.983717
faith=0.788796
deletion_auc=0.130383
insertion_auc=0.707975
simplicity=0.901619
public_proxy=0.495842
valid=true
eligible=true
model_sha256=16cb742165b72280dca7e5cbb78080166160ecec6b3990a52021bc27e89b0740
zip_sha256=1ecc26d66ca5b5e2587123529675b5afc4933fe5315435de169b93ad1883ef6f
```

`macro_f1` did not drop because the classification output is unchanged. All work here modifies only the relevance output. `simplicity` is effectively unchanged compared with the previous low-complexity TimeREISE family.

## Scoring Formula We Optimized

The devkit computes faithfulness by sorting relevance cells and applying top-k masks at:

```text
0%, 10%, 20%, ..., 100%
```

For each fraction:

- deletion starts from the original window and replaces top-k relevant cells with the baseline;
- insertion starts from the baseline and inserts top-k relevant cells from the original window.

Final faith score:

```text
faith = (insertion_auc + (1 - deletion_auc)) / 2
```

Mechanical alignment is still hidden locally because the official band config is private.

## Route Progression

### 1. Baseline LogicLSTM

We kept the official / baseline LogicLSTM classifier because it gives stable high classification accuracy:

```text
macro_f1=0.983717
```

The main strategy is therefore:

```text
Keep probabilities unchanged.
Only improve the relevance output.
```

### 2. Original TimeREISE

Original TimeREISE used perturbation statistics over class, channel, and time bins:

```text
relevance = sqrt(abs(input)) * soft_class_weighted_factor
```

Previous best before the latest marginal search:

```text
robust_offline_contrast_mix50_tb035_l050
faith=0.752061
deletion_auc=0.186942
insertion_auc=0.691064
simplicity=0.901611
public_proxy=0.481146
```

This was already low-complexity because the innovation was folded into offline `weights_9x8x100`; the ONNX graph stayed simple.

### 3. Faith-targeted Marginal TimeREISE

We inspected the devkit faithfulness implementation and directly estimated marginal faith gains:

```text
drop_gain   = p(original, predicted_class) - p(deleted_block, predicted_class)
insert_gain = p(inserted_block, predicted_class) - p(zero_baseline, predicted_class)
```

We also tracked negative gain, where a block harms the deletion/insertion objective, and added a small penalty.

Best setting:

```text
num_bins=10
drop_weight=0.50
negative_gain_penalty=0.10
time_beta=0.15
contrast_lambda=0.00
```

Why `b10` worked best:

- the official faith metric evaluates top-k masks every 10%;
- `b10` aligns one time block scale with that evaluation granularity;
- finer `b25/b50` bins introduced noisier block statistics and degraded faith.

Improvement over previous best:

```text
previous faith=0.752061, deletion_auc=0.186942, insertion_auc=0.691064
current  faith=0.770607, deletion_auc=0.146884, insertion_auc=0.688098
```

The main gain is from substantially lower deletion AUC while preserving insertion AUC reasonably well.

### 4. Power-Ensemble TimeREISE

We then fused the current marginal b10 weights with the previous robust TimeREISE weights in log space:

```text
log W_final = a * log(W_marginal + eps) + b * log(W_robust + eps)
W_final normalized per class to mean 1
```

Best setting:

```text
marginal exponent=0.58
robust exponent=0.42
```

Improvement over marginal b10:

```text
marginal faith=0.770607, deletion_auc=0.146884, insertion_auc=0.688098
ensemble faith=0.783607, deletion_auc=0.140239, insertion_auc=0.707453
```

The important result is that insertion AUC improved rather than being sacrificed. The peak was around robust exponent `0.42`; beyond `0.50`, deletion AUC rose enough that faith started to fall.

### 5. Class-Specific Candidate Selection

We computed local faith by predicted class for the strongest folded-weight candidates and selected the best row per class:

```text
class 0 -> pow_b042
class 1 -> pow_b046
class 2 -> marg_b10
class 3 -> marg_b10
class 4 -> pow_b045
class 5 -> robust
class 6 -> pow_b030
class 7 -> marg_b10
class 8 -> pow_b048
```

Final validation:

```text
classsel_pred faith=0.788796
deletion_auc=0.130383
insertion_auc=0.707975
simplicity=0.901619
```

This improved over the best single power-ensemble candidate:

```text
power ensemble faith=0.783607
class selection faith=0.788796
delta=+0.005189
```

The gain came mostly from lowering deletion AUC while keeping insertion AUC slightly higher.

## Follow-up Checks We Rejected

### Finer Bins

| branch | best tag | faith | deletion_auc | insertion_auc | simplicity | decision |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| b10 previous best | marg_dw050_p010_tb015 | 0.770607 | 0.146884 | 0.688098 | 0.901616 | keep as backup |
| b25 | marg_dw070_p010_tb015 | 0.751326 | 0.173709 | 0.676361 | 0.901616 | reject |
| b50 | marg_dw050_p010_tb015 | 0.736649 | 0.195607 | 0.668905 | 0.901616 | reject |

Conclusion: finer bins did not help.

### Sample-level Relevance Gate

We tried multiplying the marginal b10 relevance by sample-level gates from:

- absolute amplitude;
- adjacent difference energy;
- local peak residual;
- local mean energy;
- light combinations of the above.

Best sample-gated candidate:

```text
gate_abs_g030
faith=0.770180
deletion_auc=0.148142
insertion_auc=0.688502
simplicity=0.895942
public_proxy=0.487260
```

This did not beat the current best and lowered simplicity due to extra ONNX operators. We rejected it.

### Quantile / Trimmed Marginal Aggregation

We recomputed per-sample marginal gains on validation b10 and swept:

```text
quantile = 0.70, 0.75, 0.80
trim_ratio = 0.05, 0.10, 0.15
drop_weight = 0.50, 0.55
penalty = 0.05, 0.10, 0.15
time_beta = 0.10, 0.15, 0.20
```

Best result:

```text
tr05_dw050_p005_tb015
faith=0.752281
deletion_auc=0.155978
insertion_auc=0.660541
```

Decision: reject. Quantile performed much worse; trimmed 5% was the best robust aggregation variant but still below marginal b10 and far below class-specific selection.

### Mechanical Prior Fusion

We fused the class-specific best with the existing offline mechanical prior:

```text
gamma = 0.005, 0.01, 0.02, 0.05, 0.08
```

Best result:

```text
mech_g005
faith=0.788486
deletion_auc=0.130560
insertion_auc=0.707531
simplicity=0.901620
```

Decision: reject for public-faith target. It may still be useful if hidden mechanical alignment has high weight, but locally it did not beat the class-specific model.

## Current Risk Assessment

Strengths:

- classification remains stable;
- public faith improved materially;
- ONNX complexity remains low;
- final package passes devkit packaging and inspect checks.

Risks:

- the best candidate is calibrated on public validation faith;
- hidden mechanical alignment cannot be computed locally;
- if hidden distribution differs strongly, the robust offline candidate may be safer.
- power-ensemble is still a validation-tuned relevance map, so keep the marginal b10 package as a conservative backup.
- class-specific row selection is even more validation-tuned than a single global relevance map.

Conservative backup:

```text
runs/final_candidates/logic_timereise_marginal_val5k_b10_bestproxy_submission.zip
```

Current recommendation:

```text
Submit logic_timereise_class_candidate_selection_bestproxy_submission.zip
```
