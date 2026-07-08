set -uo pipefail
CTR=gb10_pytorch_zgh
dexec(){ docker exec -e PYTHONPATH=/workspace/LLaMA-Factory/src -e DISABLE_VERSION_CHECK=1 "$CTR" bash -lc "$1"; }
LOGD=/workspace/stage1_5_experiment
run_infer(){ dexec "cd /workspace/LLaMA-Factory && python scripts/vllm_infer.py \
  --model_name_or_path models/Qwen3-VL-8B-Instruct --adapter_name_or_path '$1' \
  --dataset fgadr_main4_proof_test --dataset_dir data/annotation --media_dir data \
  --template qwen3_vl_nothink --cutoff_len 2304 --max_new_tokens 512 \
  --image_max_pixels 262144 --image_min_pixels 65536 --batch_size 16 --enforce_eager true \
  --max_lora_rank 32 --gpu_memory_utilization 0.80 --save_name $LOGD/eval/$2"; }
echo "=== EVAL START $(date) ==="
echo "[baseline]"; run_infer saves/qwen3-vl-8b-fundus/lora/stage1_en_cot baseline_adapter1.jsonl; echo "rc_base=$?"
echo "[trained]"; run_infer saves/qwen3-vl-8b-fundus/lora/stage1_5_fgadr_proof trained_stage1_5.jsonl; echo "rc_trn=$?"
echo "[score]"; dexec "cd /workspace/LLaMA-Factory && python $LOGD/scripts/score_proof.py \
  data/annotation/fgadr_main4_proof_test_sft.jsonl \
  $LOGD/eval/baseline_adapter1.jsonl $LOGD/eval/trained_stage1_5.jsonl \
  $LOGD/results/PROOF_RESULTS.md"
echo "=== EVAL DONE $(date) ==="
