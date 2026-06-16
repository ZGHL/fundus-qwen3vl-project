#!/usr/bin/env bash
# Stage-1.5 v5 — sweep saved checkpoints on the DEV set to pick the best one.
#
# This is the checkpoint-SELECTION pass: every checkpoint-<step> in OUTPUT_DIR is run on the
# DEV dataset via vLLM (predictions row-aligned), scored with the proof scorer, and turned into a
# step-vs-macro-BalAcc curve. The peak is the selected checkpoint; you then run that ONE checkpoint
# on the TEST set for the final number (TEST is never touched here).
#
# Single training GPU? Run this AFTER training finishes (it sweeps all saved checkpoints at once).
# A second GPU free? Set CUDA_VISIBLE_DEVICES and `--poll` to evaluate checkpoints as they land
# (near-real-time DEV curve), so you can early-stop once the curve plateaus.
#
# Matches the working VM eval flow (run_eval_only.sh): vllm_infer.py loads base + adapter directly
# (NO separate merge needed), --enforce_eager, --max_lora_rank 32.
set -uo pipefail

LF=${LF:-/workspace/LLaMA-Factory}
EXP=${EXP:-/workspace/stage1_5_experiment}
OUTPUT_DIR=${OUTPUT_DIR:-saves/qwen3-vl-8b-fundus/lora/stage1_5_v5}   # relative to $LF
BASE=${BASE:-models/Qwen3-VL-8B-Instruct}
DEV_DATASET=${DEV_DATASET:-stage1_5_v5_dev}                            # registered in dataset_info.json
DEV_SFT=${DEV_SFT:-data/annotation/stage1_5_v5_dev_sft.jsonl}         # relative to $LF
PREDS=${PREDS:-$EXP/eval/v5_dev}
POLL=${POLL:-0}            # 1 = keep polling for new checkpoints (use only if a GPU is free)
mkdir -p "$PREDS"

infer_one() {  # $1 = step
  local step="$1" ckpt="$LF/$OUTPUT_DIR/checkpoint-$1" out="$PREDS/dev_ckpt-$1.jsonl"
  [ -f "$out" ] && { echo "  [skip] step $step already scored"; return 0; }
  [ -d "$ckpt" ] || { echo "  [warn] missing $ckpt"; return 1; }
  echo "  [infer] checkpoint-$step -> $out"
  ( cd "$LF" && python scripts/vllm_infer.py \
      --model_name_or_path "$BASE" --adapter_name_or_path "$OUTPUT_DIR/checkpoint-$step" \
      --dataset "$DEV_DATASET" --dataset_dir data/annotation --media_dir data \
      --template qwen3_vl_nothink --cutoff_len 2304 --max_new_tokens 512 \
      --image_max_pixels 589824 --image_min_pixels 65536 --batch_size 16 --enforce_eager true \
      --max_lora_rank 32 --gpu_memory_utilization 0.80 --save_name "$out" )
}

sweep_once() {
  for d in "$LF/$OUTPUT_DIR"/checkpoint-*; do
    [ -d "$d" ] || continue
    infer_one "$(basename "$d" | sed 's/checkpoint-//')"
  done
  echo "[score] building DEV curve + selecting best checkpoint"
  ( cd "$LF" && python "$EXP/scripts/pick_best_dev_ckpt.py" "$DEV_SFT" "$PREDS" "$PREDS/dev_curve.csv" )
}

echo "=== v5 DEV sweep START $(date) ==="
if [ "$POLL" = "1" ]; then
  seen=""
  while true; do
    sweep_once
    now=$(ls -d "$LF/$OUTPUT_DIR"/checkpoint-* 2>/dev/null | wc -l)
    [ "$now" = "$seen" ] && { echo "no new checkpoints; sleeping 120s"; sleep 120; } || seen="$now"
  done
else
  sweep_once
fi
echo "=== v5 DEV sweep DONE $(date) ==="
