#!/usr/bin/env bash
set -euo pipefail

python - <<PY
import sys
print("python", sys.executable)
try:
    import torch
    print("torch", torch.__version__)
    print("cuda_available", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("device_count", torch.cuda.device_count())
        print("bf16_supported", torch.cuda.is_bf16_supported())
except Exception as exc:
    print("torch_check_error", repr(exc))
PY

command -v llamafactory-cli >/dev/null && llamafactory-cli version || true
