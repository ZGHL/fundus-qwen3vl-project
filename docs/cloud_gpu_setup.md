# Cloud GPU Setup

Recommended layout:

```text
/workspace/
  LLaMA-Factory/
  fundus-qwen3vl-project/
  models/
  artifacts/
```

## Source Checkout

Use the project setup script to pin LLaMA-Factory to the verified commit before applying local patches.

```bash
cd /workspace
git clone https://github.com/hiyouga/LLaMA-Factory.git
git clone https://github.com/ZGHL/fundus-qwen3vl-project.git
cd fundus-qwen3vl-project
bash scripts/setup/apply_llamafactory_patches.sh /workspace/LLaMA-Factory
bash scripts/setup/sync_project_files.sh /workspace/LLaMA-Factory
```

Pinned LLaMA-Factory commit:

```text
f80e15dbb41cafc3a6f662aa520f40e596a41997
```

If you intentionally use an already patched local LLaMA-Factory tree:

```bash
SKIP_LLAMA_FACTORY_CHECKOUT=1 bash scripts/setup/apply_llamafactory_patches.sh /workspace/LLaMA-Factory
```

## Environment Check

```bash
cd /workspace/LLaMA-Factory
bash /workspace/fundus-qwen3vl-project/scripts/setup/check_env.sh
```

See `docs/ENVIRONMENT.md` for the currently observed working package versions.

## Required External Assets

- `models/Qwen3-VL-8B-Instruct`, from the source recorded in `manifests/models/base_models.yaml`
- image datasets listed in `manifests/datasets/dataset_manifest.yaml`
- generated JSONL annotation files from Hugging Face Dataset
- optional LoRA adapters listed in `manifests/models/model_manifest.yaml`

## Data Restore Strategy

For the exact current snapshot, use the Hugging Face annotation snapshot and R2 image snapshot recorded in `reports/UPLOAD_RECORD_20260521.md`.

For future experiments, prefer:

```text
HF Dataset native JSONL/directories for CoT/SFT/evaluation data
R2 directory sync for image roots
R2 or HF Model repo for adapters/checkpoints
GitHub for configs/manifests/metrics only
```

Do not rely on files that only exist on a rented VM disk. Upload important outputs before releasing the VM.
