# Stage1 Adapter 1 vs Qwen3-VL Base Model

Same prompt, image preprocessing, decoding settings, scorer, and held-out rows are used for each comparison.

- Adapter 1: `saves/qwen3-vl-8b-fundus/lora/stage1_en_cot`
- Base model: `models/Qwen3-VL-8B-Instruct` without an adapter
- Selection policy: this report does not use Gold-test to select a model.

## Gold-dev (596 rows)

| Model | Macro F1 | Recall | Specificity | Balanced Acc | JSON Parse | Target Consistency |
|---|---:|---:|---:|---:|---:|---:|
| Qwen3-VL-8B base | 0.00% | 0.00% | 0.00% | 0.00% | 100.00% | 0.00% |
| Adapter 1: stage1_en_cot | 70.81% | 82.40% | 15.95% | 49.17% | 100.00% | 100.00% |

| Lesion | Base F1 | Adapter 1 F1 | Delta | Base Recall | Adapter Recall | Base Specificity | Adapter Specificity |
|---|---:|---:|---:|---:|---:|---:|---:|
| MA | 0.00% | 63.30% | 63.30% | 0.00% | 52.27% | 0.00% | 0.00% |
| HE | 0.00% | 81.65% | 81.65% | 0.00% | 78.76% | 0.00% | 55.56% |
| EX | 0.00% | 64.49% | 64.49% | 0.00% | 98.57% | 0.00% | 5.06% |
| SE | 0.00% | 73.82% | 73.82% | 0.00% | 100.00% | 0.00% | 3.17% |

## Gold-test (900 rows)

| Model | Macro F1 | Recall | Specificity | Balanced Acc | JSON Parse | Target Consistency |
|---|---:|---:|---:|---:|---:|---:|
| Qwen3-VL-8B base | 0.00% | 0.00% | 0.00% | 0.00% | 100.00% | 0.00% |
| Adapter 1: stage1_en_cot | 65.24% | 87.58% | 17.93% | 52.76% | 100.00% | 100.00% |

| Lesion | Base F1 | Adapter 1 F1 | Delta | Base Recall | Adapter Recall | Base Specificity | Adapter Specificity |
|---|---:|---:|---:|---:|---:|---:|---:|
| MA | 0.00% | 52.49% | 52.49% | 0.00% | 63.71% | 0.00% | 2.97% |
| HE | 0.00% | 90.08% | 90.08% | 0.00% | 86.60% | 0.00% | 64.52% |
| EX | 0.00% | 86.80% | 86.80% | 0.00% | 100.00% | 0.00% | 3.70% |
| SE | 0.00% | 31.58% | 31.58% | 0.00% | 100.00% | 0.00% | 0.55% |

## Conclusion

- Adapter 1 Gold-test Macro F1 improvement over base: **65.24%**.
- Adapter 1 remains the default Stage2 starting point unless a calibration candidate passes the committed preservation guardrails.
