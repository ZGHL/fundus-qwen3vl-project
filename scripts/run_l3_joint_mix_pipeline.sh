#!/usr/bin/env bash
set -euo pipefail

export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

ROOT=/workspace/LLaMA-Factory
PROJECT=/workspace/fundus-qwen3vl-project
ENV=/workspace/qwen3vl-env
LOG_DIR="$ROOT/logs/l3_joint_mix_full"

mkdir -p "$LOG_DIR"
cd "$ROOT"
source "$ENV/bin/activate"

echo "[$(date -Is)] L3 joint mix pipeline start"
echo "[$(date -Is)] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

echo "[$(date -Is)] training start"
llamafactory-cli train examples/train_lora/l3_joint_mix_full.yaml 2>&1 | tee "$LOG_DIR/train.log"
echo "[$(date -Is)] training done"

run_eval() {
  local name="$1"
  local cfg="$2"
  local out_dir="$3"
  echo "[$(date -Is)] eval $name start"
  llamafactory-cli train "$cfg" 2>&1 | tee "$LOG_DIR/eval_${name}.log"
  python "$PROJECT/scripts/fundus/score_l3_joint_mix_predictions.py" \
    "$out_dir/generated_predictions.jsonl" \
    --json-out "$out_dir/l3_joint_mix_score.json" \
    2>&1 | tee "$LOG_DIR/score_${name}.log"
  echo "[$(date -Is)] eval $name done"
}

run_eval "val_subset" \
  "examples/train_lora/l3_joint_mix_predict_val_subset.yaml" \
  "saves/qwen3-vl-8b-fundus/lora/l3_joint_mix_full_predict_val_subset"

run_eval "balanced" \
  "examples/train_lora/l3_joint_mix_predict_balanced.yaml" \
  "saves/qwen3-vl-8b-fundus/lora/l3_joint_mix_full_predict_balanced"

run_eval "irma_locked" \
  "examples/train_lora/l3_joint_mix_predict_irma_locked.yaml" \
  "saves/qwen3-vl-8b-fundus/lora/l3_joint_mix_full_predict_irma_locked"

run_eval "nv_locked" \
  "examples/train_lora/l3_joint_mix_predict_nv_locked.yaml" \
  "saves/qwen3-vl-8b-fundus/lora/l3_joint_mix_full_predict_nv_locked"

echo "[$(date -Is)] L3 joint mix pipeline complete"
