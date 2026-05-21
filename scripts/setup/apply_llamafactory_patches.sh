#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 1 ]; then
  echo "Usage: bash scripts/setup/apply_llamafactory_patches.sh /path/to/LLaMA-Factory" >&2
  exit 2
fi

LLAMA_FACTORY_DIR="$1"
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LLAMA_FACTORY_COMMIT="${LLAMA_FACTORY_COMMIT:-f80e15dbb41cafc3a6f662aa520f40e596a41997}"
SKIP_CHECKOUT="${SKIP_LLAMA_FACTORY_CHECKOUT:-0}"

cd "$LLAMA_FACTORY_DIR"

if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  current_commit="$(git rev-parse HEAD)"
  if [ "$SKIP_CHECKOUT" != "1" ] && [ "$current_commit" != "$LLAMA_FACTORY_COMMIT" ]; then
    if [ -n "$(git status --porcelain)" ]; then
      echo "LLaMA-Factory worktree is dirty and is not at the pinned commit." >&2
      echo "Current: $current_commit" >&2
      echo "Expected: $LLAMA_FACTORY_COMMIT" >&2
      echo "Commit/stash local changes, or set SKIP_LLAMA_FACTORY_CHECKOUT=1 if you know this tree is already compatible." >&2
      exit 1
    fi
    git fetch origin "$LLAMA_FACTORY_COMMIT"
    git checkout "$LLAMA_FACTORY_COMMIT"
  else
    echo "Using LLaMA-Factory commit: $current_commit"
  fi
else
  echo "$LLAMA_FACTORY_DIR is not a git checkout; cannot verify LLaMA-Factory commit." >&2
  exit 1
fi

apply_patch_once() {
  local patch_file="$1"
  if git apply --check "$patch_file" >/dev/null 2>&1; then
    git apply "$patch_file"
    echo "Applied $(basename "$patch_file")"
  elif git apply --reverse --check "$patch_file" >/dev/null 2>&1; then
    echo "Already applied $(basename "$patch_file")"
  else
    echo "Patch does not apply cleanly: $patch_file" >&2
    echo "Check that LLaMA-Factory is at commit $LLAMA_FACTORY_COMMIT." >&2
    exit 1
  fi
}

apply_patch_once "$PROJECT_DIR/patches/llama_factory/0001-local-llamafactory-tracked-changes.patch"
apply_patch_once "$PROJECT_DIR/patches/llama_factory/0002-add-qwen3-vl-blackwell-patch.patch"

echo "LLaMA-Factory patches are ready in $LLAMA_FACTORY_DIR"
