# Lesion Perception English CoT Full Run Results

Date: 2026-05-21

## Training

- Model: `Qwen/Qwen3-VL-8B-Instruct`
- Method: LoRA SFT
- Dataset: `fundus_lesion_perception_en_cot_full_train`
- Train rows: 12,408
- Epochs: 1.0
- Optimization steps: 776
- Trainable parameters: 52,493,824
- Final train loss: 0.343852
- Runtime: 8,141.7338 seconds
- Samples/sec: 1.524
- Steps/sec: 0.095
- Adapter output: `saves/qwen3-vl-8b-fundus/lora/lesion_perception_en_cot_full`

Large artifacts such as adapter weights, checkpoints, base model files, image data,
and full prediction JSONL files are intentionally not committed to GitHub.

## Dataset Split

The train, validation, and NV locked evaluation image sets were checked for
overlap before training.

| Split | Rows | Unique images | Missing images |
|---|---:|---:|---:|
| Train | 12,408 | 5,458 | 0 |
| Validation | 5,335 | 1,259 | 0 |
| NV locked eval | 105 | 105 | 0 |

Image overlap:

| Pair | Overlap |
|---|---:|
| Train vs validation | 0 |
| Train vs NV locked eval | 0 |
| Validation vs NV locked eval | 0 |

## Evaluation Summary

Metrics are computed from the structured JSON output only. AUC is not reported
because the model is not trained to emit calibrated probability scores.

| Evaluation set | n | JSON parse | Target consistency | Macro F1 | Rare F1 |
|---|---:|---:|---:|---:|---:|
| Balanced eval | 200 | 99.5% | 99.0% | 0.6243 | n/a |
| Validation subset | 1,230 | 98.94% | 98.54% | 0.5699 | 0.0800 |
| IRMA locked eval | 100 | 99.0% | 96.0% | n/a | n/a |
| NV locked eval | 105 | 100.0% | 100.0% | n/a | n/a |

## Per-Lesion Results

Balanced eval:

| Lesion | n | Positive | Negative | F1 | Recall | Specificity |
|---|---:|---:|---:|---:|---:|---:|
| HE | 40 | 20 | 20 | 0.6786 | 0.9500 | 0.1500 |
| EX | 40 | 20 | 20 | 0.8444 | 0.9500 | 0.7000 |
| MA | 40 | 20 | 20 | 0.5854 | 0.6000 | 0.5500 |
| SE | 39 | 19 | 20 | 0.3889 | 0.3684 | 0.5000 |
| IRMA | 40 | 20 | 20 | n/a | 0.0000 | 0.9500 |

Validation subset:

| Lesion | n | Positive | Negative | F1 | Recall | Specificity |
|---|---:|---:|---:|---:|---:|---:|
| HE | 250 | 199 | 51 | 0.8308 | 0.8392 | 0.2941 |
| EX | 249 | 194 | 55 | 0.8483 | 0.8505 | 0.4545 |
| MA | 229 | 177 | 52 | 0.6855 | 0.6158 | 0.3846 |
| SE | 249 | 60 | 189 | 0.4048 | 0.5667 | 0.6085 |
| IRMA | 240 | 20 | 220 | 0.0800 | 0.0500 | 0.9818 |

Locked rare-lesion eval:

| Lesion | n | Positive | Negative | F1 | Recall | Specificity |
|---|---:|---:|---:|---:|---:|---:|
| IRMA | 99 | 20 | 79 | n/a | 0.0000 | 1.0000 |
| NV | 105 | 5 | 100 | n/a | 0.0000 | 0.9900 |

## Interpretation

The run learns HE, EX, and MA reasonably well under the current structured-output
format. SE remains weaker, and the rare vascular lesions are still the main
failure mode. IRMA and NV locked evaluations show near-complete negative bias:
specificity is high, but recall is effectively zero.

The next experiment should target rare-lesion recall rather than general format
following. Candidate changes are rare-lesion oversampling, lower negative pressure
for NV/IRMA, and an evaluation-triggered checkpoint selection strategy that uses
rare-lesion recall instead of train loss.
