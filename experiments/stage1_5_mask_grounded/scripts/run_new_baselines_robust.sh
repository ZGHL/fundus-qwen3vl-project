#!/bin/bash
# HOST-side robust runner for the inference container: clean restart per model + GPU gate + retry.
set -u
CT=gb10_vllm_infer
WLF=/workspace/LLaMA-Factory
OUT=/workspace/stage1_5_experiment/preds
HPREDS=/sda/zgh/stage1_5_experiment/preds
LOG=$HPREDS/new_baselines_robust.log
say(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }
ensure_gpu(){ for t in 1 2 3 4 5; do
  docker exec $CT python3 -c "import torch,sys;sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null && return 0
  say "GPU down -> restart $CT (try $t)"; docker restart $CT >/dev/null 2>&1; sleep 22; done; return 1; }
declare -A F=( [stage1]=data/stage1_5_v3_test_baseprompt_sft.jsonl [s2int]=data/stage2_audit_queries_sft.jsonl [s2msd]=data/messidor2_audit_queries_sft.jsonl )
declare -A N=( [stage1]=1108 [s2int]=1200 [s2msd]=600 )
for nm in QoQ-Med-VL-7B MedGemma-4B-IT; do
  say "clean restart before $nm"; docker restart $CT >/dev/null 2>&1; sleep 22; ensure_gpu || exit 1
  for s in stage1 s2int s2msd; do
    of=$HPREDS/${nm}__${s}.jsonl
    for try in 1 2; do
      ensure_gpu || exit 1
      say "run $nm / $s (try $try)"
      docker exec $CT bash -lc "cd $WLF && python3 scripts/run_vlm_perception.py models/$nm ${F[$s]} data $OUT/${nm}__${s}.jsonl" >/dev/null 2>&1
      got=$(wc -l < "$of" 2>/dev/null || echo 0)
      [ "$got" -ge "$(( ${N[$s]} - 20 ))" ] && { say "$nm/$s OK ($got)"; break; }
      say "$nm/$s incomplete ($got) -> restart+retry"; docker restart $CT >/dev/null 2>&1; sleep 22
    done
  done
done
say "=== NEW_BASELINES_ROBUST_DONE ==="
