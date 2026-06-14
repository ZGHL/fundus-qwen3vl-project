# Stage-2 Faithful Triage: Final Training and Checkpoint Evaluation

Date: 2026-06-14

## 1. Objective

Stage-2 trains a Qwen3-VL model to perform auditable diabetic-retinopathy triage from directly visible lesion evidence. The model first audits MA, HE, EX, and SE, always abstains on visually unreliable IRMA and NV, and then maps the visible lesion pattern to one of five faithful triage tiers:

1. `No-DR`
2. `Mild`
3. `Moderate`
4. `Mod-or-Severe-indeterminate`
5. `Severe`

These five tiers are the project's faithful-triage target, not a direct ICDR Grade 0-4 classifier. Clinical grades are retained separately for referable-DR and severe-case safety evaluation.

## 2. Final Training Method

### Initialization and model

- Base model: `Qwen3-VL-8B-Instruct`
- Warm start: Stage-1.5 v3 `checkpoint-400`
- Fine-tuning method: continued LoRA SFT using the existing adapter
- LoRA rank / alpha / dropout: `16 / 32 / 0.05`
- LoRA targets: all supported target modules
- Vision tower: trainable
- Multimodal projector: frozen
- Language model: trainable through LoRA
- Attention implementation: SDPA
- Template: `qwen3_vl_nothink`

### Optimization

| Setting | Value |
|---|---:|
| Epochs | 2.0 |
| Optimizer steps | 512 |
| Per-device batch size | 2 |
| Gradient accumulation | 8 |
| Effective batch size | 16 |
| Learning rate | `5e-6` |
| Scheduler | cosine |
| Warmup ratio | 0.1 |
| Precision | BF16 |
| Max gradient norm | 1.0 |
| Optimizer | AdamW Torch |
| Gradient checkpointing | enabled |
| Cutoff length | 2304 |
| Image pixels | 65,536 to 589,824 |
| Save interval | 60 optimizer steps |

### Training outcome

| Result | Value |
|---|---:|
| Runtime | 5,526 seconds / 1:32:06 |
| Final epoch | 2.0 |
| Aggregate training loss | 0.24827 |
| Samples per second | 1.477 |
| Steps per second | 0.093 |

The logged ten-step training loss reached approximately `0.006-0.008` after the first epoch and remained stable. No in-training validation was configured, so checkpoint selection was performed post hoc.

## 3. Training and Internal Evaluation Data

### Faithful lesion-pattern mapping

The target tier is fitted from verifiable MA/HE/EX/SE presence patterns. Examples:

- No visible trusted lesion -> `No-DR`
- MA only -> `Mild`
- HE, HE+SE, MA+HE, and similar patterns -> `Moderate`
- HE+EX and HE+EX+SE patterns -> `Severe`
- Patterns that cannot reliably separate moderate from severe, such as MA+HE+EX without SE -> `Mod-or-Severe-indeterminate`

IRMA and NV are always abstained and are never used as claimed visible evidence.

### Training set

The training set contains 4,081 rows:

| Faithful tier | Rows |
|---|---:|
| No-DR | 1,400 |
| Mild | 1,000 |
| Moderate | 318 |
| Severe | 633 |
| Mod-or-Severe-indeterminate | 730 |

Source composition:

- Grounded mask-based rows: 2,077
- Grade-derived audited rows: 2,004
- Faithful referable targets: 1,681
- Faithful non-referable targets: 2,400

### Internal checkpoint-selection set

The internal set contains 300 image-disjoint rows, balanced across faithful tiers:

| Faithful tier | Rows |
|---|---:|
| No-DR | 60 |
| Mild | 60 |
| Moderate | 60 |
| Severe | 60 |
| Mod-or-Severe-indeterminate | 60 |

This set is used for checkpoint selection and should be reported as an internal model-selection evaluation, not as an untouched final generalization estimate. Messidor-2 is reserved as the external grade-based evaluation set.

## 4. Checkpoint Evaluation Method

The following candidates were evaluated on the same 300-row internal set:

`checkpoint-120`, `checkpoint-180`, `checkpoint-240`, `checkpoint-300`, `checkpoint-360`, `checkpoint-420`, `checkpoint-480`, `checkpoint-512`, and `final`.

Uniform inference settings:

- Temperature: 0
- Top-p: 1
- Top-k: -1
- Seed: 20260614
- Max new tokens: 512
- Cutoff length: 2304
- Image pixels: 65,536 to 589,824
- Same five-tier prompt and template for every candidate

Before inference, every adapter was checked for non-finite tensors and merged into the same base model. Predictions and labels were verified to align row-by-row for all 300 samples, and all 300 image files were present and passed into multimodal inference.

### Output normalization

The scorer normalizes only unambiguous aliases. Examples:

- `No DR`, `No_DR`, `NoDR`, `0`, and `DR0` -> `No-DR`
- `mild NPDR` and `1` -> `Mild`
- `moderate NPDR`, `DR2`, and `2` -> `Moderate`
- `severe NPDR`, `3`, and `DR3` -> `Severe`
- Explicit moderate/severe-indeterminate variants -> `Mod-or-Severe-indeterminate`

Ambiguous outputs such as `referable`, `ungradable`, `N/A`, or missing JSON remain invalid and are not silently assigned a faithful tier.

## 5. Metric Definitions

### Valid tier rate

The fraction of all 300 predictions that can be parsed and normalized to exactly one of the five faithful tiers:

`valid tier rate = valid faithful-tier predictions / 300`

This measures deployable output-format coverage. It is reported separately from QWK.

### Project five-tier QWK

QWK is computed only on valid faithful-tier predictions using the project ordering:

`No-DR < Mild < Moderate < Mod-or-Severe-indeterminate < Severe`

For true tier index `i` and predicted tier index `j`, the quadratic disagreement weight is:

`w(i,j) = ((i - j) / 4)^2`

The reported QWK is Cohen's quadratic weighted kappa over valid predictions. Invalid outputs are excluded from QWK and separately penalized by the valid tier rate. This is the project's faithful-triage QWK, not standard ICDR Grade 0-4 QWK.

### Macro F1

For each of the five faithful tiers:

- `precision = TP / (TP + FP)`
- `recall = TP / (TP + FN)`
- `F1 = 2 * precision * recall / (precision + recall)`

Macro F1 is the unweighted mean of the five tier-specific F1 scores. Invalid outputs count as failures for the true class.

### MAE

Faithful tiers use the same ordered indices `0-4`. For each sample:

`absolute error = |true tier index - predicted tier index|`

MAE is the average absolute error over all samples. Invalid outputs receive the maximum ordinal penalty of 4.

### Referable sensitivity and specificity

Clinical ground truth defines referable DR as clinical grade `>= 2`.

Predicted faithful tiers are grouped as:

- Predicted referable: `Moderate`, `Mod-or-Severe-indeterminate`, or `Severe`
- Predicted non-referable: `No-DR` or `Mild`

Metrics:

- `referable sensitivity = TP / (TP + FN)`
- `referable specificity = TN / (TN + FP)`

Invalid predictions are treated as non-referable for safety accounting.

### Severe safety recall

The fraction of clinical Grade 3/4 samples that are at least predicted referable:

`severe safety recall = clinical Grade 3/4 predicted referable / all clinical Grade 3/4`

This does not require exact severe-tier classification; it measures whether severe clinical cases are missed by triage.

### Faithfulness to own audit

For each JSON-parsed prediction, the scorer takes the model's own reported `lesions_present`, applies the exact fitted MA/HE/EX/SE pattern-to-tier map, and checks whether that mapped tier equals the model's reported `dr_tier`:

`faithfulness = tier consistent with own lesion audit / JSON-parsed predictions`

This tests internal explanation-decision consistency, not whether the lesion audit itself is visually correct.

### NV/IRMA fabrication rate

The fraction of JSON-parsed predictions that incorrectly claim NV or IRMA in `lesions_present`:

`fabrication rate = predictions claiming visible NV/IRMA / JSON-parsed predictions`

The desired value is zero.

### Abstention rate

The fraction of all predictions assigned to `Mod-or-Severe-indeterminate`:

`abstention rate = indeterminate predictions / 300`

## 6. Checkpoint Sweep Results

| Candidate | Valid tier | QWK valid | Macro F1 | MAE | Ref sens | Ref spec | Severe recall | Faithful | Fabrication | Abstain |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| checkpoint-360 | 0.6700 | **0.6779** | 0.2763 | 1.9200 | 0.6250 | 0.8485 | 0.7303 | 0.3165 | 0.0000 | 0.0500 |
| checkpoint-480 | 0.6667 | 0.6576 | 0.2508 | 1.9667 | 0.5833 | 0.8712 | 0.7303 | 0.3463 | 0.0000 | 0.0700 |
| checkpoint-420 | 0.6867 | 0.6539 | 0.2714 | 1.8967 | 0.6310 | 0.8258 | 0.7079 | 0.3404 | 0.0000 | 0.0633 |
| **checkpoint-240** | **0.7533** | 0.6453 | **0.2769** | **1.7033** | **0.6488** | 0.8333 | 0.7416 | **0.4312** | 0.0000 | **0.0867** |
| checkpoint-300 | 0.7400 | 0.6199 | 0.2734 | 1.7700 | 0.6429 | 0.7879 | **0.7640** | 0.3614 | **0.0000** | 0.0633 |
| checkpoint-512 | 0.7167 | 0.6188 | 0.2625 | 1.8533 | 0.6131 | 0.8409 | 0.7079 | 0.3718 | 0.0000 | 0.0767 |
| final | 0.7133 | 0.6161 | 0.2606 | 1.8667 | 0.6131 | 0.8409 | 0.7079 | 0.3682 | 0.0000 | 0.0767 |
| checkpoint-180 | 0.7567 | 0.5862 | 0.2180 | 1.7833 | 0.6250 | 0.8712 | 0.7416 | 0.4667 | 0.0000 | 0.0433 |
| checkpoint-120 | 0.8233 | 0.3691 | 0.1758 | 1.7733 | 0.3571 | **0.9318** | 0.4270 | **0.6558** | **0.0000** | 0.0133 |

## 7. Interpretation and Checkpoint Recommendation

### Pure project-QWK winner: checkpoint-360

`checkpoint-360` has the highest valid-output five-tier QWK (`0.6779`). However, only `67.0%` of its outputs normalize to a valid faithful tier. Its MAE, referable sensitivity, and explanation-decision faithfulness are worse than checkpoint-240.

### Balanced recommended candidate: checkpoint-240

`checkpoint-240` is the recommended balanced candidate for subsequent validation because it provides:

- Highest Macro F1: `0.2769`
- Lowest MAE: `1.7033`
- Higher valid tier coverage than later checkpoints: `75.3%`
- Highest referable sensitivity among the main candidates: `0.6488`
- Severe safety recall: `0.7416`
- Better own-audit faithfulness than checkpoints 300-512: `0.4312`
- Zero measured NV/IRMA fabrication: `0.0000`

Checkpoint-300 is a reasonable safety-oriented alternative because it has the highest severe safety recall (`0.7640`), but its referable specificity is lower (`0.7879`). No candidate claimed NV/IRMA in `lesions_present` after case-normalized scoring.

### Checkpoint-240 faithful-tier confusion matrix

Rows are true faithful tiers; columns are normalized predicted faithful tiers. `Invalid` means the output could not be normalized to a valid tier.

| True / Pred | No-DR | Mild | Moderate | Indeterminate | Severe | Invalid |
|---|---:|---:|---:|---:|---:|---:|
| No-DR | 54 | 0 | 0 | 0 | 2 | 4 |
| Mild | 23 | 0 | 2 | 3 | 10 | 22 |
| Moderate | 11 | 0 | 4 | 5 | 20 | 20 |
| Indeterminate | 4 | 0 | 8 | 8 | 23 | 17 |
| Severe | 3 | 0 | 7 | 10 | 29 | 11 |

The main observed failures are:

1. The model does not correctly emit `Mild` for any of the 60 Mild targets.
2. Moderate and indeterminate cases are frequently overcalled as Severe.
3. Approximately one quarter of checkpoint-240 outputs are invalid or non-standard tier strings.
4. The model often produces semantically related but schema-invalid outputs such as `referable`, `ungradable`, `N/A`, or incomplete JSON.

## 8. Limitations and Next Evaluation

- The 300-row set is used for checkpoint selection, so its results are internal model-selection results.
- Messidor-2 should be used as the external evaluation. It provides grade labels but no lesion ground truth, so it can evaluate clinical grade/referable behavior but not lesion-audit faithfulness.
- The current maximum generation length is 512 tokens. Some responses are truncated before complete JSON output; a controlled re-evaluation with a longer generation limit may improve valid tier coverage.
- Stage-1.5 lesion-perception regression should be checked before declaring checkpoint-240 final.
- A stronger output-constrained decoding or schema-focused calibration stage is needed before deployment.

## 9. Reproducibility References

- Training config: `experiments/stage1_5_mask_grounded/configs/stage2_grade_warmstart.yaml`
- Stage-2 design and faithful-tier definition: `experiments/stage1_5_mask_grounded/STAGE2_README.md`
- Stage-1.5 v3 initialization decision: `experiments/stage1_5_mask_grounded/STAGE1_5_V3_HANDOFF.md`

The metric formulas, output-normalization policy, inference settings, complete checkpoint table, and selected confusion matrix are recorded directly in this report.
