# Next VM Handoff: Fundus Qwen3-VL Stage1 Calibration

This document restores the current project state on a new Vast.ai machine and continues from the latest stopped point. It intentionally does not store secrets. Paste the tokens from the private handoff/chat into environment variables before running downloads.

## Current State

- Main repo: `/workspace/fundus-qwen3vl-project`
- LLaMA-Factory: `/workspace/LLaMA-Factory`
- Python env: `/workspace/qwen3vl-env`
- Artifacts: `/workspace/artifacts`
- Latest committed project work before shutdown:
  - Stage1 English CoT data/training pipeline.
  - Stage1 hard-negative calibration builder.
  - Calibration training config.
  - Automatic `gold_test` evaluation config.
  - One-command calibration pipeline.

Training status:

- Completed baseline adapter: `/workspace/LLaMA-Factory/saves/qwen3-vl-8b-fundus/lora/stage1_en_cot`
- Baseline checkpoint evaluated: `checkpoint-791`
- Calibration data was generated once on the old VM but should be regenerated on the new VM.
- Calibration training was not run after the user paused.

## Required Secrets

Set these in the shell. Do not commit them.

```bash
export HF_TOKEN='<paste Hugging Face token>'
export AWS_ACCESS_KEY_ID='<paste R2 S3 access key id>'
export AWS_SECRET_ACCESS_KEY='<paste R2 S3 secret access key>'
export R2_ENDPOINT='https://4ff11044d39e473b1c3f56367f45fe71.r2.cloudflarestorage.com'
export GITHUB_TOKEN='<paste GitHub token if pushing is needed>'
```

Verify without printing values:

```bash
python3 - <<'PY'
import os
for k in ['HF_TOKEN','AWS_ACCESS_KEY_ID','AWS_SECRET_ACCESS_KEY','R2_ENDPOINT','GITHUB_TOKEN']:
    print(k, 'OK' if os.environ.get(k) else 'MISSING')
PY
```

## Base Environment

```bash
cd /workspace
nvidia-smi
df -h /workspace
free -h
python3 --version
which python3
uname -a
```

For RTX 5090 / Blackwell, use CUDA 12.8 PyTorch:

```bash
python3 -m venv /workspace/qwen3vl-env
source /workspace/qwen3vl-env/bin/activate
pip install -U pip setuptools wheel
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

Check CUDA:

```bash
python - <<'PY'
import torch
print('torch', torch.__version__)
print('cuda', torch.version.cuda)
print('cuda_available', torch.cuda.is_available())
print('device', torch.cuda.get_device_name(0))
print('capability', torch.cuda.get_device_capability(0))
print('bf16', torch.cuda.is_bf16_supported())
PY
```

Install system tools:

```bash
apt-get update
apt-get install -y git git-lfs curl wget rsync rclone unzip tmux htop nvtop build-essential python3-dev python3-venv
git lfs install
```

## Clone Code

```bash
cd /workspace
git clone https://github.com/ZGHL/fundus-qwen3vl-project.git
cd /workspace/fundus-qwen3vl-project
git checkout main
git pull --ff-only
```

Clone and patch LLaMA-Factory:

```bash
cd /workspace
git clone https://github.com/hiyouga/LLaMA-Factory.git
cd /workspace/fundus-qwen3vl-project
bash scripts/setup/apply_llamafactory_patches.sh /workspace/LLaMA-Factory
bash scripts/setup/sync_project_files.sh /workspace/LLaMA-Factory
```

Expected LLaMA-Factory commit:

```bash
cd /workspace/LLaMA-Factory
git rev-parse HEAD
# expected: f80e15dbb41cafc3a6f662aa520f40e596a41997
```

Install Python dependencies:

```bash
source /workspace/qwen3vl-env/bin/activate
cd /workspace/LLaMA-Factory
pip install -e '.[torch,metrics]'
pip install -r /workspace/fundus-qwen3vl-project/requirements/base.txt || true
pip install -U awscli
bash /workspace/fundus-qwen3vl-project/scripts/setup/check_env.sh
```

## Restore Base Model and Data

Download Qwen3-VL base model:

```bash
source /workspace/qwen3vl-env/bin/activate
cd /workspace/LLaMA-Factory
mkdir -p models
python - <<'PY'
import os
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='Qwen/Qwen3-VL-8B-Instruct',
    local_dir='/workspace/LLaMA-Factory/models/Qwen3-VL-8B-Instruct',
    local_dir_use_symlinks=False,
    token=os.environ.get('HF_TOKEN'),
)
print('base model downloaded')
PY
```

Restore annotations:

```bash
source /workspace/qwen3vl-env/bin/activate
mkdir -p /workspace/artifacts/fundus_annotations
python - <<'PY'
import os
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='Guohou/fundusAnnotationsV1',
    repo_type='dataset',
    local_dir='/workspace/artifacts/fundus_annotations',
    local_dir_use_symlinks=False,
    token=os.environ.get('HF_TOKEN'),
)
print('HF annotations downloaded')
PY
cd /workspace/LLaMA-Factory
if [ -f /workspace/artifacts/fundus_annotations/fundus_generated_annotations_20260521.tar.gz ]; then
  tar -xzf /workspace/artifacts/fundus_annotations/fundus_generated_annotations_20260521.tar.gz
fi
```

Restore images from R2:

```bash
source /workspace/qwen3vl-env/bin/activate
mkdir -p /workspace/artifacts/fundus_images
cd /workspace/artifacts/fundus_images
aws s3 cp s3://fundusv1/images/fundus_image_dataset_20260521.tar.sha256 . --endpoint-url "$R2_ENDPOINT"
aws s3 cp s3://fundusv1/images/fundus_image_dataset_20260521.tar . --endpoint-url "$R2_ENDPOINT"
sha256sum -c fundus_image_dataset_20260521.tar.sha256
cd /workspace/LLaMA-Factory
tar -xf /workspace/artifacts/fundus_images/fundus_image_dataset_20260521.tar
```

Restore the baseline adapter if it is stored externally. Required path before calibration:

```text
/workspace/LLaMA-Factory/saves/qwen3-vl-8b-fundus/lora/stage1_en_cot
```

If the adapter is not available from artifact storage, rerun the Stage1 English baseline first:

```bash
source /workspace/qwen3vl-env/bin/activate
cd /workspace/LLaMA-Factory
llamafactory-cli train /workspace/fundus-qwen3vl-project/configs/train/stage1_en_cot.yaml
```

## Current Evaluation Baseline

Baseline `stage1_en_cot/checkpoint-791` on `gold_test`:

| Lesion | F1 | Recall | Specificity | Balanced Acc |
|---|---:|---:|---:|---:|
| MA | 0.525 | 0.637 | 0.030 | 0.333 |
| HE | 0.901 | 0.866 | 0.645 | 0.756 |
| EX | 0.868 | 1.000 | 0.037 | 0.519 |
| SE | 0.316 | 1.000 | 0.005 | 0.503 |
| Macro | 0.652 | 0.876 | 0.179 | 0.528 |

Rare locked baseline:

| Lesion | F1 | Recall | Specificity |
|---|---:|---:|---:|
| NV | 0.208 | 1.000 | 0.238 |
| IRMA | 0.326 | 0.700 | 0.350 |

Interpretation: the English Stage1 model learned output format and high recall, but MA/SE and overall specificity are too low. HE is already strong.

## Next Training Plan

Run a low-risk calibration baseline before any full retraining.

Goal:

- Start from `/workspace/LLaMA-Factory/saves/qwen3-vl-8b-fundus/lora/stage1_en_cot`.
- Build a unique-image hard-negative calibration set.
- Train a new adapter at `/workspace/LLaMA-Factory/saves/qwen3-vl-8b-fundus/lora/stage1_en_cot_calibrated`.
- Automatically evaluate on `fundus_stage1_en_cot_gold_test`.

Calibration set target layout:

| Lesion | Positive | Negative |
|---|---:|---:|
| MA | 600 | 680 |
| HE | 1000 | 1000 |
| EX | 1000 | 1200 |
| SE | 800 | 1400 |
| IRMA | 136 | 272 |
| NV | 37 | 74 |

Important design choices:

- No `gold_dev`, `gold_test`, or locked eval rows are used as training input.
- No duplicate lesion-image pairs in calibration training.
- MA negatives prioritize HE-present/MA-absent hard negatives.
- SE negatives prioritize EX-present/SE-absent hard negatives.
- SE RetSAM-positive pressure is reduced compared with the baseline data.
- NV/IRMA positives use unique available positives without aggressive repeat cycling.

Training config:

- Adapter start: `stage1_en_cot`
- Output: `stage1_en_cot_calibrated`
- LR: `2e-6`
- Epochs: `1`
- Gradient accumulation: `16`
- Vision tower LoRA: on
- Projector: frozen
- Language LoRA: on

Run calibration and automatic `gold_test`:

```bash
source /workspace/qwen3vl-env/bin/activate
cd /workspace/fundus-qwen3vl-project
./scripts/run_stage1_en_cot_calibration.sh
```

Expected generated files:

```text
/workspace/LLaMA-Factory/data/annotation_v4/fundus_stage1_en_cot_calibration_train_sft.jsonl
/workspace/LLaMA-Factory/data/annotation_v4/fundus_stage1_en_cot_calibration_stats.json
/workspace/LLaMA-Factory/saves/qwen3-vl-8b-fundus/lora/stage1_en_cot_calibrated
/workspace/LLaMA-Factory/saves/qwen3-vl-8b-fundus/lora/stage1_eval/calibrated_gold_test/stage1_metrics.json
/workspace/LLaMA-Factory/logs/stage1_en_cot_calibration/
```

After training, compare against the baseline above. Primary acceptance criteria:

- JSON parse success remains 100%.
- Target consistency remains 100%.
- Main4 specificity improves materially from 0.179.
- MA specificity improves from 0.030 without collapsing recall.
- SE specificity improves from 0.005 without collapsing recall.
- HE F1 does not regress severely.

If calibration improves specificity enough, continue from calibrated adapter. If it fails, the next step is a full English two-stage rebuild: English main-4 full checkpoint followed by six-lesion balanced calibration.

## Restoring Stage1 Baseline Adapter From R2

The completed Stage1 English CoT adapter/checkpoints were archived to R2 after the baseline evaluation.

R2 objects:

```text
s3://fundusv1/models/stage1/stage1_en_cot_20260608_full.tar.gz
s3://fundusv1/models/stage1/stage1_en_cot_20260608_full.tar.gz.sha256
s3://fundusv1/models/stage1/stage1_en_cot_20260608_run_artifacts.tar.gz
s3://fundusv1/models/stage1/stage1_en_cot_20260608_run_artifacts.tar.gz.sha256
```

Restore the full adapter/checkpoints:

```bash
source /workspace/qwen3vl-env/bin/activate
mkdir -p /workspace/artifacts/model_checkpoints
cd /workspace/artifacts/model_checkpoints
aws s3 cp s3://fundusv1/models/stage1/stage1_en_cot_20260608_full.tar.gz . --endpoint-url "$R2_ENDPOINT"
aws s3 cp s3://fundusv1/models/stage1/stage1_en_cot_20260608_full.tar.gz.sha256 . --endpoint-url "$R2_ENDPOINT"
sha256sum -c stage1_en_cot_20260608_full.tar.gz.sha256
mkdir -p /workspace/LLaMA-Factory/saves/qwen3-vl-8b-fundus/lora
cd /workspace/LLaMA-Factory/saves/qwen3-vl-8b-fundus/lora
tar -xzf /workspace/artifacts/model_checkpoints/stage1_en_cot_20260608_full.tar.gz
```

Restore the small run-artifacts bundle if you want the exact generated calibration set and metrics without regenerating them:

```bash
source /workspace/qwen3vl-env/bin/activate
cd /workspace/artifacts
aws s3 cp s3://fundusv1/models/stage1/stage1_en_cot_20260608_run_artifacts.tar.gz . --endpoint-url "$R2_ENDPOINT"
aws s3 cp s3://fundusv1/models/stage1/stage1_en_cot_20260608_run_artifacts.tar.gz.sha256 . --endpoint-url "$R2_ENDPOINT"
sha256sum -c stage1_en_cot_20260608_run_artifacts.tar.gz.sha256
tar -xzf stage1_en_cot_20260608_run_artifacts.tar.gz
```

The full checkpoint archive SHA256 is:

```text
6bd9be5084ca967d4bff6f19c03184c66812a96d0c7c9906cb312fcc537f39cf  stage1_en_cot_20260608_full.tar.gz
```
