# Environment Reproducibility

This project should pin the LLaMA-Factory source commit and record the Python/CUDA stack for each runnable environment. Qwen3-VL multimodal loading and vLLM behavior are sensitive to `transformers`, `torch`, `accelerate`, and LLaMA-Factory changes.

## Verified Local Environment

Current checked environment in `gb10_pytorch_zgh`:

```text
container: gb10_pytorch_zgh
workspace: /workspace/LLaMA-Factory
LLaMA-Factory commit: f80e15dbb41cafc3a6f662aa520f40e596a41997
python package observations:
  torch: 2.6.0a0+ecf3bae40a.nv25.01
  transformers: 5.0.0
  peft: 0.18.1
  accelerate: 1.11.0
  datasets: 4.0.0
  trl: 0.24.0
  vllm: 0.14.1
  flash_attn: not installed in this container
```

Use `scripts/setup/check_env.sh` after restoring a VM to confirm the active Python environment before training.

## LLaMA-Factory Pinning

The setup script pins LLaMA-Factory before applying project patches:

```text
LLAMA_FACTORY_COMMIT=f80e15dbb41cafc3a6f662aa520f40e596a41997
```

Default setup:

```bash
cd /workspace
git clone https://github.com/hiyouga/LLaMA-Factory.git
git clone https://github.com/ZGHL/fundus-qwen3vl-project.git
cd fundus-qwen3vl-project
bash scripts/setup/apply_llamafactory_patches.sh /workspace/LLaMA-Factory
bash scripts/setup/sync_project_files.sh /workspace/LLaMA-Factory
```

If you are intentionally working on an already patched local LLaMA-Factory tree, set:

```bash
SKIP_LLAMA_FACTORY_CHECKOUT=1 bash scripts/setup/apply_llamafactory_patches.sh /workspace/LLaMA-Factory
```

Only use this when you understand the compatibility risk.

## VM Rule

For rented GPU VMs, treat the VM as disposable:

1. Rebuild from GitHub + Hugging Face + R2.
2. Do not rely on files that only exist on the VM disk.
3. Push lightweight code/config/metrics back to GitHub.
4. Upload checkpoints and large prediction artifacts to R2 or Hugging Face before releasing the VM.
