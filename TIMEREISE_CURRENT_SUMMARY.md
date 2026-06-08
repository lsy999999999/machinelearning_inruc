# TimeREISE Current Route Summary

## Current Recommended Submission

Use this local package for the current public-faith / public-proxy best submission:

```text
runs/final_candidates/logic_timereise_marginal_val5k_b10_bestproxy_submission.zip
```

Local devkit validation:

```text
tag=marg_dw050_p010_tb015
macro_f1=0.983717
faith=0.770607
deletion_auc=0.146884
insertion_auc=0.688098
simplicity=0.901616
public_proxy=0.488566
valid=true
eligible=true
model_sha256=5d6b744cc549550f89168b28a719a6a0ded8545d2d44a272347a3ac070a865cc
zip_sha256=571196dc84c725c7530ff4b9a0162d36e09337f8a175016304798d844b1d6636
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

## Follow-up Checks We Rejected

### Finer Bins

| branch | best tag | faith | deletion_auc | insertion_auc | simplicity | decision |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| b10 current | marg_dw050_p010_tb015 | 0.770607 | 0.146884 | 0.688098 | 0.901616 | keep |
| b25 | marg_dw070_p010_tb015 | 0.751326 | 0.173709 | 0.676361 | 0.901616 | reject |
| b50 | marg_dw050_p010_tb015 | 0.736649 | 0.195607 | 0.668905 | 0.901616 | reject |

Conclusion: finer bins did not help.

### Sample-level Relevance Gate

We tried multiplying the current best relevance by sample-level gates from:

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

Conservative backup:

```text
runs/final_candidates/logic_timereise_robust_offline_contrast_l050_bestproxy_submission.zip
```

Current recommendation:

```text
Submit logic_timereise_marginal_val5k_b10_bestproxy_submission.zip
```

