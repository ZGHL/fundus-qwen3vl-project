#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 1 ]; then
  echo "Usage: bash scripts/setup/apply_llamafactory_patches.sh /path/to/LLaMA-Factory" >&2
  exit 2
fi

LLAMA_FACTORY_DIR="$1"
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

cd "$LLAMA_FACTORY_DIR"

git apply --check "$PROJECT_DIR/patches/llama_factory/0001-local-llamafactory-tracked-changes.patch"
git apply "$PROJECT_DIR/patches/llama_factory/0001-local-llamafactory-tracked-changes.patch"

git apply --check "$PROJECT_DIR/patches/llama_factory/0002-add-qwen3-vl-blackwell-patch.patch"
git apply "$PROJECT_DIR/patches/llama_factory/0002-add-qwen3-vl-blackwell-patch.patch"

echo "Applied LLaMA-Factory patches to $LLAMA_FACTORY_DIR"
