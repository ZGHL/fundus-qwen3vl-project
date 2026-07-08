# Stage-1.5 v3 Checkpoint Selection

Date: 2026-06-13

## Decision

Select **checkpoint-400** as the Stage-1.5 v3 handoff checkpoint.

Selection rule:

1. Require parse coverage >= 0.95 and Macro recall >= 0.80.
2. Maximize Macro balanced accuracy for the specificity-correction objective.
3. Use Macro F1, Micro balanced accuracy, and Micro F1 as tie-breakers.

Checkpoint-400 passes the guardrails and is best on every primary/tie-break metric.

## Uniform Evaluation Setup

All checkpoints were merged into the same Qwen3-VL-8B-Instruct base and evaluated
on the same image-disjoint `stage1_5_v3_test` set with:

- template: `qwen3_vl_nothink`
- cutoff length: 2304
- max new tokens: 256
- image pixels: 65536 to 262144
- batch size: 24
- temperature: 0
- top-p: 1
- top-k: -1
- seed: 20260613
- parse failures counted as classification errors

## Aggregate Comparison

| Model | Parse | Macro F1 | Macro Recall | Macro Spec | Macro Balanced | Micro F1 | Micro Recall | Micro Spec | Micro Balanced |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Adapter1 baseline | 0.992 | 0.528 | 0.965 | 0.213 | 0.589 | 0.528 | 0.958 | 0.198 | 0.578 |
| checkpoint-80 | 0.979 | 0.557 | 0.908 | 0.370 | 0.639 | 0.555 | 0.894 | 0.363 | 0.629 |
| checkpoint-160 | 0.979 | 0.575 | 0.892 | 0.442 | 0.667 | 0.570 | 0.872 | 0.431 | 0.652 |
| checkpoint-240 | 0.985 | 0.594 | 0.877 | 0.501 | 0.689 | 0.597 | 0.869 | 0.501 | 0.685 |
| checkpoint-320 | 0.969 | 0.592 | 0.871 | 0.505 | 0.688 | 0.590 | 0.852 | 0.502 | 0.677 |
| **checkpoint-400** | **0.986** | **0.616** | **0.844** | **0.606** | **0.725** | **0.619** | **0.827** | **0.594** | **0.711** |

## Checkpoint-400 Per-Lesion Results

| Lesion | Parse | F1 | Recall | Specificity | Balanced Accuracy |
|---|---:|---:|---:|---:|---:|
| MA | 0.949 | 0.673 | 0.908 | 0.405 | 0.656 |
| HE | 1.000 | 0.706 | 0.706 | 0.810 | 0.758 |
| EX | 0.993 | 0.740 | 0.813 | 0.812 | 0.813 |
| SE | 1.000 | 0.344 | 0.950 | 0.397 | 0.673 |

## Interpretation

The sweep shows the intended specificity correction developing with training.
Macro specificity rises from 0.370 at step 80 to 0.606 at step 400, while Macro
recall declines from 0.908 to 0.844. Checkpoint-400 gives the best balance.

MA remains the largest weakness: its specificity is only 0.405 and parse coverage
is 0.949. SE retains high recall but low precision because positives are rare, so
its F1 remains 0.344 despite balanced accuracy of 0.673.

This clean v3 test set has now been used for checkpoint selection. Treat these
numbers as model-selection results, not as an untouched final generalization
estimate. Stage-2 should reserve a new image-disjoint holdout.
