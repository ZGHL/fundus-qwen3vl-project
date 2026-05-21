#!/usr/bin/env bash
set -euo pipefail

STAGE="${1:-stage1_smoke}"

case "${STAGE}" in
  stage1_smoke)
    CONFIG="examples/train_lora/qwen3vl_fundus_stage1_smoke.yaml"
    ;;
  stage1)
    CONFIG="examples/train_lora/qwen3vl_fundus_stage1_train.yaml"
    ;;
  stage2)
    CONFIG="examples/train_lora/qwen3vl_fundus_stage2_train.yaml"
    ;;
  stage3)
    CONFIG="examples/train_lora/qwen3vl_fundus_stage3_train.yaml"
    ;;
  *)
    echo "Usage: $0 {stage1_smoke|stage1|stage2|stage3}" >&2
    exit 2
    ;;
esac

docker exec gb10_pytorch bash -lc "
  cd /workspace/LLaMA-Factory &&
  PYTHONPATH=src /workspace/miniconda3/bin/python -m llamafactory.cli train ${CONFIG}
"
