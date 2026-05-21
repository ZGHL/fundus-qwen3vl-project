# Cloud GPU Setup

Recommended layout:

```text
/workspace/
  LLaMA-Factory/
  fundus-qwen3vl-project/
  models/
  data/
  saves/
```

Setup:

```bash
cd /workspace
git clone https://github.com/hiyouga/LLaMA-Factory.git
git clone <your-repo-url> fundus-qwen3vl-project
cd fundus-qwen3vl-project
bash scripts/setup/apply_llamafactory_patches.sh /workspace/LLaMA-Factory
bash scripts/setup/sync_project_files.sh /workspace/LLaMA-Factory
```

Check environment:

```bash
cd /workspace/LLaMA-Factory
bash /workspace/fundus-qwen3vl-project/scripts/setup/check_env.sh
```

Required external assets:

- `models/Qwen3-VL-8B-Instruct`
- image datasets listed in `manifests/datasets/dataset_manifest.yaml`
- generated JSONL annotation files, or enough source data to regenerate them
- optional LoRA adapters listed in `manifests/models/model_manifest.yaml`
