# Fundus Qwen3-VL Project Mainline Inventory

This inventory records the assets that should be preserved when moving the
project to a clean GitHub/cloud-GPU environment. Large image datasets, base
models, and LoRA checkpoints should not be committed to GitHub; this document
only records their roles and current paths.

## Main Objective

Train Qwen3-VL-8B for a three-level fundus workflow:

1. L2 anatomical perception: laterality and cup-to-disc ratio.
2. L3 lesion perception: MA, HE, EX, SE, IRMA, and NV.
3. L4 DR grading: grade 0-4 from lesion evidence.

The current publication-facing direction is to move from the validated Chinese
CoT baseline to an English CoT format aligned with the report Figure 3/4 style.

## Naming Convention

Use semantic experiment names in new environments. Keep legacy names only as
lineage references.

| Semantic name | Legacy path/name | Role |
|---|---|---|
| `l3_zh_cot_baseline_step1` | `l3_targeted_calib_v3_full` | Chinese CoT L3 first step, MA/HE/EX/SE targeted calibration. |
| `l3_zh_cot_baseline_step2` | `l3_six_lesion_calib_pilot` | Chinese CoT L3 second step, six-lesion balanced calibration. |
| `l3_en_cot_incomplete_step2_only` | `v8_l3_v3flow_v7cot` | Incomplete English CoT control: only rewrote step2 data; do not use as final English L3 baseline. |
| `l4_zh_cot_pipeline_baseline` | `l4_unified_lesion_cot_v3` | Current strongest L4 pipeline result based on the Chinese L3 baseline. |

## Valid Data Backbone

### Original images and labels

Keep as external data assets, not Git files:

| Asset | Current path | Role |
|---|---|---|
| APTOS/DDR/IDRiD/FGADR/Messidor images | `data/FGADR`, `data/DDR-dataset`, `data/idrid`, `data/messidor-2`, `data/cropped`, `data/processed_images` | Image pool for training/evaluation. |
| DR labels | `data/DR_grading.csv`, `data/messidor_data.csv`, `data/idrid_old/idrid_labels.csv` | Grade labels and external evaluation labels. |

### RetSAM and validated evidence

Preserve these metadata files or regenerate them from scripts:

| Asset | Current path | Role |
|---|---|---|
| Unified RetSAM/strong-label evidence | `data/fundus_validated/validated.jsonl` | Raw unified evidence after merging RetSAM and strong labels. |
| Cleaned evidence | `data/fundus_validated/validated_clean.jsonl` | Main cleaned data source for L2/L3/L4 construction. |
| Cleaned stats | `data/fundus_validated/validated_clean.stats.json` | Required for reporting sample counts and lesion availability. |
| RetSAM run reports | `reports/retsam_*` | Required for explaining pseudo-label filtering and quality control. |

Important cleaned-data counts:

| Field | Count |
|---|---:|
| L2 usable | 9875 |
| L3 usable | 7083 |
| L4 usable | 9493 |
| RetSAM L3 | 5713 |
| Strong-label L3 | 1370 |

Lesion-positive availability:

| Lesion | Positive count |
|---|---:|
| HE | 5794 |
| EX | 5268 |
| SE | 1690 |
| MA | 1052 |
| IRMA | 98 |
| NV | 27 |

## Chinese CoT Baseline Path

This is the current validated L3/L4 baseline and should be kept as the
reference path.

### Stage 0/1 anatomical and early perception warmup

| Item | Path |
|---|---|
| Config | `examples/train_lora/qwen3vl_fundus_stage1_pilot.yaml` |
| Output | `saves/qwen3-vl-8b-fundus/lora/stage1_pilot` |
| Runtime | 49m17s |
| Train loss | 0.2211 |

### L3 Chinese CoT step 1: targeted calibration

| Item | Path |
|---|---|
| Data | `data/annotation/fundus_l3_targeted_calib_v3_full_sft.jsonl` |
| Stats | `data/annotation/fundus_l3_targeted_calib_v3_full_stats.json` |
| Config | `examples/train_lora/qwen3vl_fundus_l3_targeted_calib_v3_full.yaml` |
| Output | `saves/qwen3-vl-8b-fundus/lora/l3_targeted_calib_v3_full` |
| Samples | 23253 |
| Runtime | 6h24m30s |
| Train loss | 0.0371 |

This step includes MA/HE/EX/SE targeted lesion calibration and is required.
The English CoT main experiment must reproduce this step in English before
running six-lesion calibration.

### L3 Chinese CoT step 2: six-lesion calibration

| Item | Path |
|---|---|
| Data | `data/annotation/fundus_l3_six_lesion_calib_pilot_sft.jsonl` |
| Stats | `data/annotation/fundus_l3_six_lesion_calib_pilot_stats.json` |
| Config | `examples/train_lora/qwen3vl_fundus_l3_six_lesion_calib_pilot.yaml` |
| Output | `saves/qwen3-vl-8b-fundus/lora/l3_six_lesion_calib_pilot` |
| Samples | 7200 |
| Distribution | 6 lesions x 600 positive + 600 negative |
| Runtime | 2h05m28s |
| Train loss | 0.0597 |

Reference L3 metrics:

| Lesion/set | Precision | Recall | FPR | F1 |
|---|---:|---:|---:|---:|
| HE | 0.900 | 0.900 | 0.100 | 0.900 |
| EX | 0.769 | 1.000 | 0.300 | 0.870 |
| MA | 0.667 | 0.800 | 0.400 | 0.727 |
| SE | 0.750 | 0.600 | 0.200 | 0.667 |
| holdout80 micro | 0.767 | 0.825 | 0.250 | 0.795 |
| NV holdout | 0.743 | 0.724 | 0.250 | 0.733 |
| IRMA holdout | 0.917 | 0.286 | 0.026 | 0.436 |

## English CoT Work Status

The current `v8_l3_v3flow_v7cot` run is not a complete English L3 main
experiment. It starts from `l3_targeted_calib_v3_full` and trains only the
rewritten six-lesion step. That skips the English version of targeted
calibration, so it should be treated as an incomplete step2-only control.

Observed partial evaluation:

| Set | Main issue |
|---|---|
| MA/HE/EX/SE holdout80 | Micro F1 0.693; EX recall dropped; MA/SE false positives remain high. |
| NV holdout | F1 0.0; model predicts almost all NV samples as negative. |

Correct next English CoT path:

1. Rewrite `fundus_l3_targeted_calib_v3_full_sft.jsonl` into the English Figure 3/4-style CoT format.
2. Train from `stage1_pilot` using the same hyperparameters as `l3_targeted_calib_v3_full`.
3. Rewrite `fundus_l3_six_lesion_calib_pilot_sft.jsonl` into the same English CoT format.
4. Continue from the English targeted-calibration checkpoint using the same hyperparameters as `l3_six_lesion_calib_pilot`.
5. Evaluate on the same L3 holdouts: holdout80, NV holdout, IRMA holdout.

## L4 Baseline

| Item | Path |
|---|---|
| Data | `data/annotation/fundus_l4_unified_lesion_cot_v3_sft.jsonl` |
| Stats | `data/annotation/fundus_l4_unified_lesion_cot_v3_stats.json` |
| Config | `examples/train_lora/qwen3vl_fundus_l4_unified_lesion_cot_v3.yaml` |
| Output | `saves/qwen3-vl-8b-fundus/lora/l4_unified_lesion_cot_v3` |
| Base adapter | `l3_six_lesion_calib_pilot` |
| Runtime | 13h09m57s |
| Train loss | 0.1690 |

Reference L4 metrics:

| Eval set | Accuracy | Macro F1 | QWK |
|---|---:|---:|---:|
| holdout150 | 0.500 | 0.498 | 0.735 |
| Messidor2 | 0.467 | 0.442 | 0.695 |
| FunBench L4C generative | 0.417 | 0.423 | 0.657 |
| FunBench L4C MCQ | 0.280 | 0.182 | 0.261 |

Base-model comparison:

| Eval set | Model | Accuracy | Macro F1 | QWK |
|---|---|---:|---:|---:|
| holdout150 | Qwen3-VL-8B base | 0.393 | 0.283 | 0.594 |
| holdout150 | L4 baseline | 0.500 | 0.498 | 0.735 |

## Code To Preserve

Project-specific scripts:

| Directory | Role |
|---|---|
| `scripts/fundus` | Data construction, L3/L4 scoring, FunBench scoring, monitor, report generation. |
| `scripts/fundus_v4` | English CoT/v4-v7 construction utilities; keep but clean names before publication. |
| `scripts/retsam_pseudo` | RetSAM parsing, filtering, reporting, and pseudo-CoT utilities. |
| `scripts/stage1_easy` | Strong-mask-derived stage1 data generation. |

LLaMA-Factory patches:

| File | Role |
|---|---|
| `src/llamafactory/model/model_utils/qwen3_vl_blackwell.py` | GB10/Blackwell Qwen3-VL CUDA compatibility patch. |
| `src/llamafactory/model/patcher.py` | Calls the Qwen3-VL Blackwell patch. |
| `src/llamafactory/train/sft/metric.py` and `trainer.py` | Custom parsing/metric extensions; review before publishing. |
| `src/llamafactory/data/mm_plugin.py` | Optional torchaudio guard. |

## Do Not Commit Large Assets

Keep these out of GitHub:

| Path | Reason |
|---|---|
| `data/FGADR`, `data/DDR-dataset`, `data/idrid`, `data/messidor-2`, `data/cropped`, `data/processed_images` | Image data. |
| `models/` | Base and comparison models. |
| `saves/` | LoRA checkpoints and predictions. |
| `outputs/` | RetSAM raw outputs. |
| `logs/` | Runtime logs; keep selected summaries only. |

Use external storage or Hugging Face Hub for datasets and adapters.

