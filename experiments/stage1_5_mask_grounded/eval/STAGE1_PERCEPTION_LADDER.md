# Stage-1.5 perception ladder (same present/absent test)

test = stage1_5_v3_test_sft.jsonl  (n=1108 single-lesion rows)　models: base, adapter1, stage1_best_ckpt400

Lesion-perception capability head-to-head on one identical, image-disjoint test.
Recall = sensitivity to a present lesion; Spec = not over-reporting on absent; BalAcc = (Recall+Spec)/2. This is the NEW-mainline foundation (the faithful grader is bounded by this audit quality).

### RECALL by lesion
| lesion | base | adapter1 | stage1_best_ckpt400 |
|---|---|---|---|
| MA | 0.000 | 1.000 | 0.955 |
| HE | 0.000 | 0.881 | 0.706 |
| EX | 0.000 | 0.989 | 0.822 |
| SE | 0.000 | 1.000 | 0.950 |

### SPEC by lesion
| lesion | base | adapter1 | stage1_best_ckpt400 |
|---|---|---|---|
| MA | 1.000 | 0.038 | 0.481 |
| HE | 1.000 | 0.595 | 0.810 |
| EX | 0.000 | 0.210 | 0.816 |
| SE | 1.000 | 0.017 | 0.397 |

### F1 by lesion
| lesion | base | adapter1 | stage1_best_ckpt400 |
|---|---|---|---|
| MA | 0.000 | 0.629 | 0.705 |
| HE | 0.000 | 0.703 | 0.706 |
| EX | 0.000 | 0.549 | 0.747 |
| SE | 0.000 | 0.256 | 0.344 |

### Summary (macro)
| metric | base | adapter1 | stage1_best_ckpt400 |
|---|---|---|---|
| macro recall | 0.000 | 0.967 | 0.858 |
| macro spec | 0.750 | 0.215 | 0.626 |
| macro f1 | 0.000 | 0.534 | 0.626 |
| **macro BalAcc** | **0.375** | **0.591** | **0.742** |
| parse_fail (total) | 824 | 39 | 61 |

Reading: a trained perception model should lift Recall AND Spec over base zero-shot (which typically over-reports -> low Spec, or fails to emit the schema -> parse_fail). v3 already fixed Spec (0.20->0.59 vs Adapter1); v4 should additionally lift Recall (esp. MA/HE/EX) without losing that Spec.
