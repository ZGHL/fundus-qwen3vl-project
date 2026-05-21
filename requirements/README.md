# Requirements

`base.txt` records the package versions observed in the working `gb10_pytorch_zgh` container. It is mainly a reproducibility reference.

For a fresh rented VM, prefer a CUDA/PyTorch image that already matches the GPU and driver stack, then install or verify the remaining Python packages. Do not blindly reinstall PyTorch over a vendor-tuned image unless necessary.
