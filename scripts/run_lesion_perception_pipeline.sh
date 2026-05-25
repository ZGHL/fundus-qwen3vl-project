#!/usr/bin/env bash
set -euo pipefail

export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

ROOT=/workspace/LLaMA-Factory
PROJECT=/workspace/fundus-qwen3vl-project
ENV=/workspace/qwen3vl-env
LOG_DIR="$ROOT/logs/lesion_perception_en_cot_full"

mkdir -p "$LOG_DIR"
cd "$ROOT"
source "$ENV/bin/activate"

echo "[$(date -Is)] pipeline start"
echo "[$(date -Is)] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "[$(date -Is)] python=$(which python)"
echo "[$(date -Is)] llamafactory-cli=$(which llamafactory-cli || true)"

echo "[$(date -Is)] SFT start"
llamafactory-cli train examples/train_lora/lesion_perception_en_cot_full.yaml 2>&1 | tee "$LOG_DIR/train.log"
echo "[$(date -Is)] SFT done"

echo "[$(date -Is)] internal validation prediction start"
llamafactory-cli train examples/eval/lesion_perception_en_cot_full_val.yaml 2>&1 | tee "$LOG_DIR/eval_val.log"
python "$PROJECT/scripts/fundus/score_lesion_perception_predictions.py" \
  saves/qwen3-vl-8b-fundus/lora/lesion_perception_en_cot_full_predict_val/generated_predictions.jsonl \
  --json-out saves/qwen3-vl-8b-fundus/lora/lesion_perception_en_cot_full_predict_val/lesion_perception_val_score.json \
  2>&1 | tee "$LOG_DIR/score_val.log"
echo "[$(date -Is)] internal validation scoring done"

echo "[$(date -Is)] NV locked prediction start"
llamafactory-cli train examples/eval/lesion_perception_en_cot_nv_locked_eval.yaml 2>&1 | tee "$LOG_DIR/eval_nv_locked.log"
python "$PROJECT/scripts/fundus/score_lesion_perception_predictions.py" \
  saves/qwen3-vl-8b-fundus/lora/lesion_perception_en_cot_full_predict_nv_locked/generated_predictions.jsonl \
  --json-out saves/qwen3-vl-8b-fundus/lora/lesion_perception_en_cot_full_predict_nv_locked/lesion_perception_nv_locked_score.json \
  2>&1 | tee "$LOG_DIR/score_nv_locked.log"
echo "[$(date -Is)] NV locked scoring done"

echo "[$(date -Is)] pipeline complete"
