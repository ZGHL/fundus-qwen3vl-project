# Data and Artifact Transfer Plan

This project should not store training data, images, RetSAM outputs, base models, or LoRA checkpoints in Git. Use Git only for code/configs/patches/manifests. Use object storage, Hugging Face Hub, an external disk, or rsync/scp for large artifacts.

## Artifact Layers

| Layer | Examples | Current size | Recommended handling |
|---|---|---:|---|
| L0 original images and labels | `data/FGADR`, `data/DDR-dataset`, `data/idrid`, `data/messidor-2`, label CSVs | about 21G for listed raw datasets | Keep external. Download from source or upload to private storage. |
| L0 cropped and APTOS processed images | `data/cropped`, `data/processed_images` | about 43G | `data/processed_images` is the processed APTOS image folder; `data/cropped` stores cropped fundus images. Transfer only if you do not want to regenerate preprocessing. |
| L1 RetSAM raw outputs | `outputs/retsam_*` | about 160M currently visible | Optional. Keep if you need to audit pseudo-labels; otherwise regenerate from RetSAM. |
| L2 validated evidence | `data/fundus_validated/*.jsonl`, stats | about 55M | Upload directly. This is the key compact evidence layer built from RetSAM plus strong labels. |
| L3/L4 generated SFT JSONL | `data/annotation`, `data/annotation_v4` | about 683M | Upload directly or regenerate from L2 evidence. These are the actual ShareGPT/SFT samples, including CoT answers. |
| Evaluation JSONL | L3 holdouts, L4 holdouts, FunBench/Messidor2 SFT files | included in annotation dirs | Upload directly with generated SFT JSONL. |
| Base model | `models/Qwen3-VL-8B-Instruct` | about 17G | Download from Hugging Face on the server. |
| LoRA adapters | `stage1_pilot`, `l3_targeted_calib_v3_full`, `l3_six_lesion_calib_pilot`, `l4_unified_lesion_cot_v3` | about 15.3G total | Upload only if you need to continue/evaluate old baselines. Otherwise retrain. |
| Full experiment outputs | old `saves/*predict*`, logs | large/noisy | Do not transfer wholesale. Keep selected metrics in Git. |

## What Is `fundus_generated_annotations`?

`fundus_generated_annotations` is the compact data package containing generated annotation assets, not raw images. It includes:

- cleaned RetSAM/strong-label evidence: `data/fundus_validated/`
- generated SFT/ShareGPT JSONL files: `data/annotation/`, `data/annotation_v4/`
- CoT training samples and evaluation samples
- dataset stats and RetSAM filtering stats

This package is what lets a new server train/evaluate without regenerating every JSONL file. It still requires the referenced image files to exist at the same relative paths.

## Minimal Cloud Package

For continuing the new English CoT L3 work, the minimal external package is:

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

Also provide image roots referenced by those JSONL files. If paths are unchanged under LLaMA-Factory, configs run without edits.

## Recommended Full Generated-Annotation Package

For complete reproducibility without regenerating JSONL/CoT samples. This package does not mean raw images; it contains compact generated annotation files, cleaned evidence, SFT JSONL, evaluation JSONL, and RetSAM stats:

```text
data/fundus_validated/
data/annotation/
data/annotation_v4/
reports/retsam_*_stats/
reports/retsam_*_run_report/*.stats.json
```

This is under 1G excluding images and checkpoints, so it is reasonable to archive and upload.

## Suggested Archive Commands

From the current LLaMA-Factory root:

```bash
cd /workspace/LLaMA-Factory
mkdir -p /workspace/artifacts

tar -czf /workspace/artifacts/fundus_generated_annotations_$(date +%Y%m%d).tar.gz \
  data/fundus_validated \
  data/annotation \
  data/annotation_v4 \
  reports/retsam_aptos_stats \
  reports/retsam_ddr_subset789_stats \
  reports/retsam_aptos_run_report/*.stats.json \
  reports/retsam_ddr_subset789_run_report/*.stats.json
```

Images can be packaged separately:

```bash
tar -czf /workspace/artifacts/fundus_images_raw_$(date +%Y%m%d).tar.gz \
  data/FGADR data/DDR-dataset data/idrid data/messidor-2 \
  data/DR_grading.csv data/messidor_data.csv data/idrid_old/idrid_labels.csv
```

Cropped images and processed APTOS images are optional:

```bash
tar -czf /workspace/artifacts/fundus_cropped_and_aptos_processed_images_$(date +%Y%m%d).tar.gz \
  data/cropped data/processed_images
```

Important adapters can be packaged separately:

```bash
tar -czf /workspace/artifacts/fundus_key_adapters_$(date +%Y%m%d).tar.gz \
  saves/qwen3-vl-8b-fundus/lora/stage1_pilot \
  saves/qwen3-vl-8b-fundus/lora/l3_targeted_calib_v3_full \
  saves/qwen3-vl-8b-fundus/lora/l3_six_lesion_calib_pilot \
  saves/qwen3-vl-8b-fundus/lora/l4_unified_lesion_cot_v3
```

## Pulling On A New Server

After downloading archives to the server:

```bash
cd /workspace/LLaMA-Factory
tar -xzf /workspace/artifacts/fundus_generated_annotations_YYYYMMDD.tar.gz
tar -xzf /workspace/artifacts/fundus_images_raw_YYYYMMDD.tar.gz
# optional
tar -xzf /workspace/artifacts/fundus_cropped_and_aptos_processed_images_YYYYMMDD.tar.gz
# optional
tar -xzf /workspace/artifacts/fundus_key_adapters_YYYYMMDD.tar.gz
```

Then sync project files:

```bash
cd /workspace/fundus-qwen3vl-project
bash scripts/setup/apply_llamafactory_patches.sh /workspace/LLaMA-Factory
bash scripts/setup/sync_project_files.sh /workspace/LLaMA-Factory
```

## Storage Choices

Recommended order:

1. Hugging Face private Dataset repo for annotation JSONL and small stats.
2. Hugging Face private Model repo or object storage for LoRA adapters.
3. Object storage, NAS, or external disk for image archives.
4. rsync/scp only for one-off migration.

Do not use GitHub Releases for the full image dataset unless the files are small and stable; it is less convenient for repeated training.
