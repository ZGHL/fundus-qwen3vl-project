#!/usr/bin/env bash
# Stage-1.5 proof — overnight orchestration (host-side; survives container restarts).
# train (watchdog+resume) -> eval Adapter1 baseline + trained on FGADR held-out -> score -> report.
set -uo pipefail
CTR=gb10_pytorch_zgh
LF=/sda/zgh/LLaMA-Factory
EXP=/sda/zgh/stage1_5_experiment
LOG="$EXP/results/run.log"
mkdir -p "$EXP/results" "$EXP/eval"
exec >>"$LOG" 2>&1
echo "===== START $(date) ====="

dexec(){ docker exec -e PYTHONPATH=/workspace/LLaMA-Factory/src -e DISABLE_VERSION_CHECK=1 "$CTR" bash -lc "$1"; }
ensure_ctr(){ if [ "$(docker inspect -f '{{.State.Running}}' "$CTR" 2>/dev/null)" != "true" ]; then echo "[wd] restarting container $(date)"; docker start "$CTR"; sleep 15; fi; }

OUT=saves/qwen3-vl-8b-fundus/lora/stage1_5_fgadr_proof
FINAL="$LF/$OUT/adapter_model.safetensors"
DEADLINE=$(( $(date +%s) + 8*3600 ))   # train must finish within 8h

# build a resume config (no overwrite, resume from latest ckpt)
sed -e 's/^overwrite_output_dir: true/overwrite_output_dir: false/' \
    "$LF/examples/train_lora/stage1_5_fgadr_proof.yaml" > "$LF/examples/train_lora/_stage1_5_resume.yaml"
echo "resume_from_checkpoint: true" >> "$LF/examples/train_lora/_stage1_5_resume.yaml"

attempt=0
while [ ! -f "$FINAL" ] && [ "$attempt" -lt 8 ] && [ "$(date +%s)" -lt "$DEADLINE" ]; do
  attempt=$((attempt+1))
  ensure_ctr
  if ls "$LF/$OUT"/checkpoint-* >/dev/null 2>&1; then CFG=examples/train_lora/_stage1_5_resume.yaml; else CFG=examples/train_lora/stage1_5_fgadr_proof.yaml; fi
  echo "[train] attempt=$attempt cfg=$CFG $(date)"
  dexec "cd /workspace/LLaMA-Factory && llamafactory-cli train $CFG"; rc=$?
  echo "[train] rc=$rc $(date)"
  [ -f "$FINAL" ] && break
  sleep 30
done

if [ ! -f "$FINAL" ]; then echo "[FAIL] training did not finish; latest ckpt:"; ls "$LF/$OUT" 2>/dev/null;
  # fall back: use latest checkpoint dir as adapter for eval
  LASTCK=$(ls -d "$LF/$OUT"/checkpoint-* 2>/dev/null | sort -V | tail -1)
  if [ -n "$LASTCK" ]; then TRAINED_ADP="${LASTCK#$LF/}"; else echo "[FATAL] no checkpoint; abort"; exit 1; fi
else TRAINED_ADP="$OUT"; fi
echo "[eval] trained adapter = $TRAINED_ADP"

run_infer(){ # $1 adapter(rel) $2 outname
  ensure_ctr
  dexec "cd /workspace/LLaMA-Factory && python scripts/vllm_infer.py \
    --model_name_or_path models/Qwen3-VL-8B-Instruct --adapter_name_or_path '$1' \
    --dataset fgadr_main4_proof_test --dataset_dir data/annotation --media_dir data \
    --template qwen3_vl_nothink --cutoff_len 2304 --max_new_tokens 512 \
    --image_max_pixels 262144 --image_min_pixels 65536 --batch_size 16 --enforce_eager true \
    --max_lora_rank 32 --gpu_memory_utilization 0.80 \
    --save_name /workspace/stage1_5_experiment/eval/$2"
}
echo "[eval] baseline Adapter1 $(date)"; run_infer "saves/qwen3-vl-8b-fundus/lora/stage1_en_cot" baseline_adapter1.jsonl
echo "[eval] trained $(date)"; run_infer "$TRAINED_ADP" trained_stage1_5.jsonl

echo "[score] $(date)"
dexec "cd /workspace/LLaMA-Factory && python /workspace/stage1_5_experiment/scripts/score_proof.py \
  data/annotation/fgadr_main4_proof_test_sft.jsonl \
  /workspace/stage1_5_experiment/eval/baseline_adapter1.jsonl \
  /workspace/stage1_5_experiment/eval/trained_stage1_5.jsonl \
  /workspace/stage1_5_experiment/results/PROOF_RESULTS.md"
echo "===== DONE $(date) ====="
echo "RESULTS: $EXP/results/PROOF_RESULTS.md"
