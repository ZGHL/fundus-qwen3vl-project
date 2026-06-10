# Current Progress and New VM Restore

Date: 2026-06-10

## Restore readiness

The current Stage 1 and Stage 1.5 work can be continued on a new VM.

The required components are backed up separately:

| Component | Source |
|---|---|
| Project code, configs, patches, reports | GitHub `ZGHL/fundus-qwen3vl-project`, branch `main` |
| Base model | Hugging Face `Qwen/Qwen3-VL-8B-Instruct` |
| Raw images and labels | R2 `s3://fundusv1/images/fundus_image_dataset_20260521.tar` |
| Current generated annotations | R2 `s3://fundusv1/annotations/fundus_annotations_current_20260610.tar.zst` |
| Current Stage 1 baseline and fallback | R2 `s3://fundusv1/models/stage1/stage1_stage2_handoff_20260609.tar.zst` |
| Full original Adapter 1 checkpoints | R2 `s3://fundusv1/models/stage1/stage1_en_cot_20260608_full.tar.gz` |
| Stage 1.5 research checkpoints | R2 `s3://fundusv1/models/stage1_5/stage1_5_six_lesion_specificity_20260610.tar.zst` |

The current recommended Stage 2 initialization is:

```text
/workspace/LLaMA-Factory/saves/qwen3-vl-8b-fundus/lora/stage1_en_cot_gentle_calibrated/checkpoint-20
```

Stage 1.5 checkpoints `40`, `80`, and `160` are preserved as complete resumable
checkpoints.

## Pinned software

Use the following controlled baseline:

| Component | Version |
|---|---|
| LLaMA-Factory commit | `f80e15dbb41cafc3a6f662aa520f40e596a41997` |
| Python | `3.12.3` |
| PyTorch | `2.11.0+cu128` |
| CUDA reported by PyTorch | `12.8` |
| Transformers | `5.0.0` |
| Datasets | `4.0.0` |
| Accelerate | `1.11.0` |
| PEFT | `0.18.1` |
| Pillow | `11.3.0` |

The project stores the required local LLaMA-Factory changes under
`patches/llama_factory/`. They cover the tracked modifications currently used
by the VM, including Qwen3-VL Blackwell compatibility, multimodal collation,
evaluation metrics, ordinal-loss support, and accelerated inference changes.

## Restore procedure

### 1. Clone code

```bash
cd /workspace
git clone https://github.com/ZGHL/fundus-qwen3vl-project.git
git clone https://github.com/hiyouga/LLaMA-Factory.git
cd /workspace/LLaMA-Factory
git checkout f80e15dbb41cafc3a6f662aa520f40e596a41997
```

### 2. Recreate the Python environment

Create `/workspace/qwen3vl-env` and install a CUDA/PyTorch stack compatible with
the target GPU. Match the pinned versions above when exact reproducibility is
required. Then install LLaMA-Factory in editable mode.

The existing environment is not archived as a portable binary environment,
because CUDA and GPU-specific packages must match the new VM.

### 3. Apply project patches and sync files

```bash
cd /workspace/fundus-qwen3vl-project
bash scripts/setup/apply_llamafactory_patches.sh /workspace/LLaMA-Factory
bash scripts/setup/sync_project_files.sh /workspace/LLaMA-Factory
source /workspace/qwen3vl-env/bin/activate
bash scripts/setup/check_env.sh
```

### 4. Restore images

```bash
mkdir -p /workspace/artifacts/fundus_images
cd /workspace/artifacts/fundus_images
aws s3 cp s3://fundusv1/images/fundus_image_dataset_20260521.tar . --endpoint-url "$R2_ENDPOINT"
aws s3 cp s3://fundusv1/images/fundus_image_dataset_20260521.tar.sha256 . --endpoint-url "$R2_ENDPOINT"
sha256sum -c fundus_image_dataset_20260521.tar.sha256
tar -xf fundus_image_dataset_20260521.tar -C /workspace/LLaMA-Factory
```

### 5. Restore exact current annotations

```bash
mkdir -p /workspace/artifacts/annotations
cd /workspace/artifacts/annotations
aws s3 cp s3://fundusv1/annotations/fundus_annotations_current_20260610.tar.zst . --endpoint-url "$R2_ENDPOINT"
aws s3 cp s3://fundusv1/annotations/fundus_annotations_current_20260610.tar.zst.sha256 . --endpoint-url "$R2_ENDPOINT"
sha256sum -c fundus_annotations_current_20260610.tar.zst.sha256
tar --zstd -xf fundus_annotations_current_20260610.tar.zst -C /workspace/LLaMA-Factory
```

### 6. Restore the current Stage 1 baseline

```bash
mkdir -p /workspace/artifacts/stage1
cd /workspace/artifacts/stage1
aws s3 cp s3://fundusv1/models/stage1/stage1_stage2_handoff_20260609.tar.zst . --endpoint-url "$R2_ENDPOINT"
aws s3 cp s3://fundusv1/models/stage1/stage1_stage2_handoff_20260609.tar.zst.sha256 . --endpoint-url "$R2_ENDPOINT"
sha256sum -c stage1_stage2_handoff_20260609.tar.zst.sha256
tar --zstd -xf stage1_stage2_handoff_20260609.tar.zst
```

The handoff archive extracts into its own bundle directory. Follow its
`RESTORE.md` to place the baseline checkpoint and fallback adapter at their
expected LLaMA-Factory paths.

### 7. Restore Stage 1.5 when needed

```bash
mkdir -p /workspace/artifacts/stage1_5
cd /workspace/artifacts/stage1_5
aws s3 cp s3://fundusv1/models/stage1_5/stage1_5_six_lesion_specificity_20260610.tar.zst . --endpoint-url "$R2_ENDPOINT"
aws s3 cp s3://fundusv1/models/stage1_5/stage1_5_six_lesion_specificity_20260610.tar.zst.sha256 . --endpoint-url "$R2_ENDPOINT"
sha256sum -c stage1_5_six_lesion_specificity_20260610.tar.zst.sha256
tar --zstd -xf stage1_5_six_lesion_specificity_20260610.tar.zst -C /workspace
```

## What can be continued

After restoration:

- Stage 2 can initialize from the current Stage 1 baseline.
- Stage 1.5 can continue training from checkpoints `40`, `80`, or `160`
  because optimizer, scheduler, RNG, and trainer state are included.
- Stage 1 and Stage 1.5 evaluations can be reproduced using the exact Gold
  Dev/Test JSONL files in the current annotation snapshot.
- Dataset builders can create revised calibration and Stage 2 datasets from
  the restored validated facts.

## Remaining reproducibility limits

- The Python virtual environment itself is not a portable snapshot. A new VM
  must install GPU-compatible dependencies.
- Exact throughput depends on GPU model, driver, storage, and CPU.
- Gold Test and Locked Test should not be repeatedly used for tuning after
  restoration.
- Credentials are intentionally excluded from Git and archives.

