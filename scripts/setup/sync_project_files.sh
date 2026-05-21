#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 1 ]; then
  echo "Usage: bash scripts/setup/sync_project_files.sh /path/to/LLaMA-Factory" >&2
  exit 2
fi

LLAMA_FACTORY_DIR="$1"
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

mkdir -p "$LLAMA_FACTORY_DIR/examples/train_lora"
mkdir -p "$LLAMA_FACTORY_DIR/examples/eval"
mkdir -p "$LLAMA_FACTORY_DIR/scripts"
mkdir -p "$LLAMA_FACTORY_DIR/reports"

rsync -a "$PROJECT_DIR/scripts/fundus" "$LLAMA_FACTORY_DIR/scripts/"
rsync -a "$PROJECT_DIR/scripts/fundus_v4" "$LLAMA_FACTORY_DIR/scripts/"
rsync -a "$PROJECT_DIR/scripts/retsam_pseudo" "$LLAMA_FACTORY_DIR/scripts/"
rsync -a "$PROJECT_DIR/scripts/stage1_easy" "$LLAMA_FACTORY_DIR/scripts/"
rsync -a "$PROJECT_DIR/configs/train/" "$LLAMA_FACTORY_DIR/examples/train_lora/"
rsync -a "$PROJECT_DIR/configs/eval/" "$LLAMA_FACTORY_DIR/examples/eval/"
rsync -a "$PROJECT_DIR/reports/" "$LLAMA_FACTORY_DIR/reports/"

echo "Synced project scripts/configs/reports into $LLAMA_FACTORY_DIR"
