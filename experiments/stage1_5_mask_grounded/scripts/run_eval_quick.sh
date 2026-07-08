set -uo pipefail
CTR=gb10_pytorch_zgh
dexec(){ docker exec -e PYTHONPATH=/workspace/LLaMA-Factory/src -e DISABLE_VERSION_CHECK=1 "$CTR" bash -lc "$1"; }
LOGD=/workspace/stage1_5_experiment
ri(){ dexec "cd /workspace/LLaMA-Factory && python scripts/vllm_infer.py \
  --model_name_or_path models/Qwen3-VL-8B-Instruct --adapter_name_or_path '$1' \
  --dataset fgadr_main4_proof_testq --dataset_dir data/annotation --media_dir data \
  --template qwen3_vl_nothink --cutoff_len 2304 --max_new_tokens 256 \
  --image_max_pixels 262144 --image_min_pixels 65536 --batch_size 24 --enforce_eager true \
  --max_lora_rank 32 --gpu_memory_utilization 0.85 --save_name $LOGD/eval/$2"; }
echo "=== EVALQ START $(date) ==="
echo "[baseline]"; ri saves/qwen3-vl-8b-fundus/lora/stage1_en_cot baseline_adapter1_q.jsonl; echo "rcbase=$?"
echo "[trained]"; ri saves/qwen3-vl-8b-fundus/lora/stage1_5_fgadr_proof trained_stage1_5_q.jsonl; echo "rctrn=$?"
echo "[score]"; dexec "cd /workspace/LLaMA-Factory && python $LOGD/scripts/score_proof.py \
  data/annotation/fgadr_main4_proof_testq_sft.jsonl \
  $LOGD/eval/baseline_adapter1_q.jsonl $LOGD/eval/trained_stage1_5_q.jsonl \
  $LOGD/results/PROOF_RESULTS.md"
echo "=== EVALQ DONE $(date) ==="
