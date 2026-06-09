# Stage1 Overnight Experiment Results

Date: 2026-06-09

## Recommendation

- Recommended balanced Stage2 starting adapter: `/workspace/LLaMA-Factory/saves/qwen3-vl-8b-fundus/lora/stage1_en_cot_gentle_calibrated/checkpoint-20`
- Adapter SHA256: `9d35140057356c4fd07bca095b969ac4e80d86faf65134d38c1c37a8130b4976`
- Selected only on Gold-dev using preservation guardrails; Gold-test was evaluated once after selection.
- Adapter 1 remains the conservative fallback when any MA regression is unacceptable.

## Gold-test Summary

| Model | Macro F1 | Recall | Specificity | Balanced Accuracy |
|---|---:|---:|---:|---:|
| Qwen3-VL-8B base (relaxed semantic scorer) | 32.37% | 25.32% | 80.11% | 52.72% |
| Adapter 1: stage1_en_cot | 65.24% | 87.58% | 17.93% | 52.76% |
| Balanced checkpoint-20 | 65.95% | 81.64% | 38.46% | 60.05% |

| Lesion | Base F1 | Adapter 1 F1 | Balanced F1 | Balanced vs Adapter 1 |
|---|---:|---:|---:|---:|
| MA | 0.00% | 52.49% | 51.61% | -0.88% |
| HE | 51.28% | 90.08% | 89.73% | -0.35% |
| EX | 63.60% | 86.80% | 86.52% | -0.29% |
| SE | 14.58% | 31.58% | 35.94% | 4.37% |

## Interpretation

- Balanced candidate improves Gold-test Macro F1 by **0.71%** over Adapter 1.
- Balanced candidate improves specificity by **20.53%** and balanced accuracy by **7.29%**.
- Balanced candidate improves semantic Macro F1 over the base model by **33.58%**.
- HE and EX are effectively preserved; SE improves; MA decreases slightly on Gold-test despite improving on Gold-dev.
- The base model strict task score is 0 because it does not follow the required target-lesion schema. The relaxed semantic score is reported to separate formatting adaptation from lesion-presence performance.

## Artifacts

- Adapter 1 vs base report: `reports/stage1_adapter1_vs_base_model_20260609.md`
- Recommended adapter manifest: `reports/metrics/stage1_stage2_recommended_adapter.json`
- Base predictions and metrics: `saves/qwen3-vl-8b-fundus/lora/stage1_eval/base_model/`
- Balanced candidate selection and test: `saves/qwen3-vl-8b-fundus/lora/stage1_eval/gentle_calibration/`
- Overnight logs: `logs/stage1_overnight_20260609/`
