# Fundus Qwen3-VL Fine-tuning

Lightweight project repository for fundus fine-tuning experiments with Qwen3-VL-8B and LLaMA-Factory.

This repository stores project code, training/evaluation configs, LLaMA-Factory patches, experiment manifests, and summary metrics. It intentionally does not store image datasets, base models, LoRA checkpoints, RetSAM raw outputs, or large generated JSONL files.

## Mainline

The project targets a three-level fundus workflow:

1. L2 anatomical perception: laterality and cup-to-disc ratio.
2. L3 lesion perception: MA, HE, EX, SE, IRMA, and NV.
3. L4 DR grading: grade 0-4 from lesion evidence.

Current validated baseline:

- `l3_zh_cot_baseline_step1_targeted_calib`
- `l3_zh_cot_baseline_step2_six_lesion`
- `l4_zh_cot_pipeline_baseline`

The next main experiment should rebuild the L3 path in English CoT format from step 1, not only rewrite the six-lesion step.

See `reports/PROJECT_MAINLINE_INVENTORY.md` for the full lineage.

## Artifact Policy

- Existing 2026-05-21 HF/R2 tar files are frozen restore snapshots.
- Future annotation/CoT/SFT updates should be uploaded to Hugging Face Dataset as native JSONL/directories.
- Future image updates should use R2 directory/object sync rather than rebuilding a monolithic image tar.
- LLaMA-Factory is pinned to commit `f80e15dbb41cafc3a6f662aa520f40e596a41997` before project patches are applied.
- See `docs/ENVIRONMENT.md`, `docs/data_and_artifact_transfer.md`, and `reports/REPOSITORY_AND_ARTIFACT_WORKFLOW.md`.


## Cloud Setup

On a new GPU server:

```bash
cd /workspace
git clone https://github.com/hiyouga/LLaMA-Factory.git
git clone <your-repo-url> fundus-qwen3vl-project
cd fundus-qwen3vl-project
bash scripts/setup/apply_llamafactory_patches.sh /workspace/LLaMA-Factory
bash scripts/setup/sync_project_files.sh /workspace/LLaMA-Factory
```

Then download or mount `models/Qwen3-VL-8B-Instruct`, fundus datasets, generated annotation JSONL files, and optional LoRA checkpoints.

## What Not To Commit

Do not commit `data/`, `models/`, `saves/`, `outputs/`, `logs/`, model weights, or large JSONL/CSV files.
