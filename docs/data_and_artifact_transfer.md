# Data and Artifact Transfer Plan

This project should not store training data, images, RetSAM outputs, base models, or LoRA checkpoints in Git. Use Git only for code/configs/patches/manifests. Use Hugging Face Dataset for generated annotation JSONL files, Cloudflare R2 for images and large binary artifacts, and Hugging Face Model repos or R2 for adapters.

## Current Storage Policy

| Artifact | Preferred storage | Reason |
|---|---|---|
| Code, configs, patches, manifests, metrics summaries | GitHub | Small, reviewable, versioned. |
| Generated annotations, CoT/SFT JSONL, holdout JSONL, stats | Hugging Face Dataset | Versioned data commits, easy download, supports native JSONL layout. |
| Raw and processed fundus images | Cloudflare R2 object tree | Large, incrementally syncable, can pull only needed subsets. |
| LoRA adapters and checkpoints | Hugging Face Model repo or R2 | Too large for GitHub; should be referenced by manifest. |
| Full prediction dumps | Hugging Face Dataset or R2 | Keep only summaries in GitHub. |

## Artifact Layers

| Layer | Examples | Current size | Recommended handling |
|---|---|---:|---|
| L0 original images and labels | `data/FGADR`, `data/DDR-dataset`, `data/idrid`, `data/messidor-2`, label CSVs | about 21G for listed raw datasets | Keep external. Store as R2 directory objects, not a repeatedly rebuilt tar. |
| L0 cropped and APTOS processed images | `data/cropped`, `data/processed_images` | about 43G | `data/processed_images` is the processed APTOS image folder; store as R2 directory objects. |
| L1 RetSAM raw outputs | `outputs/retsam_*` | about 160M currently visible | Optional. Keep if auditing pseudo-labels; otherwise regenerate from RetSAM. |
| L2 validated evidence | `data/fundus_validated/*.jsonl`, stats | about 55M | Upload as native JSONL/JSON files to Hugging Face Dataset. |
| L3/L4 generated SFT JSONL | `data/annotation`, `data/annotation_v4` | about 683M | Upload as native JSONL/JSONL.GZ files to Hugging Face Dataset. |
| Evaluation JSONL | L3 holdouts, L4 holdouts, FunBench/Messidor2 SFT files | included in annotation dirs | Upload with generated SFT JSONL. |
| Base model | `models/Qwen3-VL-8B-Instruct` | about 17G | Download from Hugging Face on the server; record repo and revision in `manifests/models/base_models.yaml`. |
| LoRA adapters | selected `saves/qwen3-vl-8b-fundus/lora/*` | variable | Upload only important adapters to HF Model repo or R2; record URI and checksum. |
| Full experiment outputs | old prediction dumps, long logs | large/noisy | Do not transfer wholesale. Keep selected metrics in GitHub. |

## Current Snapshot Artifacts

The following existing files are valid frozen restore snapshots and should be kept:

```text
HF Dataset: Guohou/fundusAnnotationsV1/fundus_generated_annotations_20260521.tar.gz
R2: images/fundus_image_dataset_20260521.tar
```

They are useful for restoring the exact 2026-05-21 state. They should not be the default format for future updates.

## R2 Image Strategy: Directory Sync

Future image updates should use R2 as an object tree. This avoids rebuilding and re-uploading a 63 GiB tar file after every small change.

Recommended R2 layout:

```text
r2:fundusv1/images/FGADR/
r2:fundusv1/images/DDR-dataset/
r2:fundusv1/images/idrid/
r2:fundusv1/images/messidor-2/
r2:fundusv1/images/cropped/
r2:fundusv1/images/aptos_processed/
r2:fundusv1/labels/DR_grading.csv
r2:fundusv1/labels/messidor_data.csv
r2:fundusv1/labels/idrid_old/idrid_labels.csv
```

Example upload from the LLaMA-Factory root:

```bash
rclone sync data/FGADR/ r2:fundusv1/images/FGADR/ -P --transfers 32
rclone sync data/DDR-dataset/ r2:fundusv1/images/DDR-dataset/ -P --transfers 32
rclone sync data/idrid/ r2:fundusv1/images/idrid/ -P --transfers 32
rclone sync data/messidor-2/ r2:fundusv1/images/messidor-2/ -P --transfers 32
rclone sync data/cropped/ r2:fundusv1/images/cropped/ -P --transfers 32
rclone sync data/processed_images/ r2:fundusv1/images/aptos_processed/ -P --transfers 32
```

Example partial restore on a VM:

```bash
mkdir -p /workspace/LLaMA-Factory/data
rclone copy r2:fundusv1/images/idrid/ /workspace/LLaMA-Factory/data/idrid/ -P
rclone copy r2:fundusv1/images/FGADR/ /workspace/LLaMA-Factory/data/FGADR/ -P
```

Use the 63 GiB tar only when a full cold restore is simpler than subset sync.

## Hugging Face Annotation Strategy: Native Files

`fundus_generated_annotations_20260521.tar.gz` remains a valid frozen snapshot. New CoT/SFT/evaluation files should be uploaded directly to the Hugging Face Dataset repository as JSONL/JSONL.GZ files or directories.

Recommended layout inside the dataset repo:

```text
data/fundus_validated/
data/annotation/
data/annotation_v4/
eval/
stats/
reports/
```

Example upload:

```bash
hf upload Guohou/fundusAnnotationsV1 data/annotation_v4/ data/annotation_v4/ --repo-type dataset --commit-message "Add English L3 CoT annotation files"
hf upload Guohou/fundusAnnotationsV1 data/fundus_validated/ data/fundus_validated/ --repo-type dataset --commit-message "Update validated fundus evidence"
```

For a new experiment, update GitHub manifests with the HF file path, sample count, lesion distribution, and expected image roots.

## Returning Results From A VM

After training, keep GitHub small and push only reproducibility metadata:

```bash
cd /workspace/fundus-qwen3vl-project
git pull
git add configs manifests reports scripts
git commit -m "Record <experiment_name> results"
git push
```

Upload large adapters to R2 or a Hugging Face Model repo. Example R2 path:

```bash
EXP=<experiment_name>
rclone copy /workspace/LLaMA-Factory/saves/qwen3-vl-8b-fundus/lora/$EXP/   r2:fundusv1/adapters/$EXP/ -P --transfers 16
```

Upload large prediction dumps separately:

```bash
EXP=<experiment_name>
rclone copy /workspace/LLaMA-Factory/saves/qwen3-vl-8b-fundus/lora/$EXP/generated_predictions.jsonl   r2:fundusv1/predictions/$EXP/ -P
```

Then record the adapter/prediction URI and checksum in `manifests/experiments/experiment_registry.yaml` or a dedicated experiment manifest.

## Minimal Cloud Package

For continuing the English CoT L3 work, the minimal external annotation package is:

```text
data/fundus_validated/validated_clean.jsonl
data/fundus_validated/validated_clean.stats.json
data/annotation/fundus_l3_targeted_calib_v3_full_sft.jsonl
data/annotation/fundus_l3_targeted_calib_v3_full_stats.json
data/annotation/fundus_l3_six_lesion_calib_pilot_sft.jsonl
data/annotation/fundus_l3_six_lesion_calib_pilot_stats.json
data/annotation/fundus_l3_presence_holdout80_sft.jsonl
data/annotation/fundus_l3_nv_single_holdout_sft.jsonl
data/annotation/fundus_l3_irma_single_holdout_sft.jsonl
data/annotation/dataset_info.json
```

Also provide the image roots referenced by those JSONL files. If paths are unchanged under LLaMA-Factory, configs run without edits.

## Storage Choices

Recommended order:

1. GitHub for lightweight project state.
2. Hugging Face private Dataset repo for annotation JSONL, evaluation JSONL, and stats.
3. Cloudflare R2 object tree for raw and processed images.
4. Hugging Face private Model repo or R2 for LoRA adapters.
5. rsync/scp only for one-off migration.

Do not use GitHub Releases for the full image dataset unless files are small and stable.
