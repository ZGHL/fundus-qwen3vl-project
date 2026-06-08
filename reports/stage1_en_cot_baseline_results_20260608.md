# Stage1 English CoT Baseline Results

Date: 2026-06-08

## Scope

This report records the completed Stage1 English CoT baseline before hard-negative calibration. Large artifacts are not committed:

- Adapter: `saves/qwen3-vl-8b-fundus/lora/stage1_en_cot`
- Predictions: `saves/qwen3-vl-8b-fundus/lora/stage1_eval/*/generated_predictions.jsonl`
- Training checkpoints: `saves/qwen3-vl-8b-fundus/lora/stage1_en_cot/checkpoint-*`

Committed metric JSON files:

- `reports/metrics/stage1_en_cot_gold_test_metrics.json`
- `reports/metrics/stage1_en_cot_gold_dev_metrics.json`
- `reports/metrics/stage1_en_cot_nv_locked_metrics.json`
- `reports/metrics/stage1_en_cot_irma_locked_metrics.json`

## Training

- Base model: `Qwen/Qwen3-VL-8B-Instruct`
- Adapter output: `saves/qwen3-vl-8b-fundus/lora/stage1_en_cot`
- Training rows: 12,650
- Epochs: 1.0
- Steps: 791
- Runtime: 7,401.1893 s
- Train loss: 0.26299
- Final built-in eval loss: 0.01916
- LoRA rank: 16
- Vision tower LoRA: enabled
- Projector: frozen
- Image upper bound: 768 x 768 equivalent (`image_max_pixels=589824`)

## Gold Test Summary

900 DDR strong-mask rows.

| Lesion | n | Pos | Neg | Precision | Recall | Specificity | F1 | Balanced Acc |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| MA | 225 | 124 | 101 | 0.446 | 0.637 | 0.030 | 0.525 | 0.333 |
| HE | 225 | 194 | 31 | 0.939 | 0.866 | 0.645 | 0.901 | 0.756 |
| EX | 225 | 171 | 54 | 0.767 | 1.000 | 0.037 | 0.868 | 0.519 |
| SE | 225 | 42 | 183 | 0.188 | 1.000 | 0.005 | 0.316 | 0.503 |
| Macro | 900 | - | - | - | 0.876 | 0.179 | 0.652 | 0.528 |

Format metrics:

- JSON parse success: 1.000
- Target consistency: 1.000
- No grade output rate: 1.000

## Rare Locked Summary

| Lesion | n | Pos | Neg | Precision | Recall | Specificity | F1 | Balanced Acc |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| NV | 88 | 8 | 80 | 0.116 | 1.000 | 0.238 | 0.208 | 0.619 |
| IRMA | 100 | 20 | 80 | 0.212 | 0.700 | 0.350 | 0.326 | 0.525 |

## Interpretation

The baseline learned the strict English output format very well and has high sensitivity, but it is strongly positive-biased. HE is already strong. MA, EX, SE, NV, and IRMA need specificity repair.

Primary failure modes:

- MA: insufficient strong/hard negatives in training; gold-test specificity is 0.030.
- SE: training positives were too RetSAM-heavy; gold-test specificity is 0.005.
- EX: recall is perfect but specificity is only 0.037.
- Rare lesions: recall improved over old English baselines, but false positives are high.

## Next Step

Run the committed hard-negative calibration pipeline:

```bash
cd /workspace/fundus-qwen3vl-project
./scripts/run_stage1_en_cot_calibration.sh
```

This builds a unique-image calibration set, continues from `stage1_en_cot`, and automatically evaluates `gold_test` into:

```text
saves/qwen3-vl-8b-fundus/lora/stage1_eval/calibrated_gold_test/stage1_metrics.json
```
