# LLaMA-Factory Patches

A patch is a text diff that records how our local LLaMA-Factory differs from the clean upstream repository.

Why we use patches:

- Keep this project repository lightweight.
- Avoid forking and uploading the full LLaMA-Factory working directory.
- Reproduce the same runtime behavior on a clean cloud GPU server.
- Separate project code/configs from upstream framework code.

Current patches:

| Patch | Purpose |
|---|---|
| `0001-local-llamafactory-tracked-changes.patch` | Existing tracked local changes in LLaMA-Factory, including vLLM inference, custom metrics/trainer behavior, dependency range, and multimodal handling changes. Review before publication. |
| `0002-add-qwen3-vl-blackwell-patch.patch` | Adds `qwen3_vl_blackwell.py`, used to avoid a Qwen3-VL CUDA/NVRTC issue on GB10/Blackwell. |
| `0003-tolerate-truncated-pillow-images.patch` | Sets Pillow `LOAD_TRUNCATED_IMAGES` in LLaMA-Factory multimodal image preprocessing so a partially truncated image stream does not abort Stage1 training. |

Apply them on a clean LLaMA-Factory checkout:

```bash
cd fundus-qwen3vl-project
bash scripts/setup/apply_llamafactory_patches.sh /workspace/LLaMA-Factory
```

If a patch fails, the upstream LLaMA-Factory version likely changed. In that case, inspect the failed hunk and either pin the LLaMA-Factory commit or regenerate the patch from a compatible local checkout.
