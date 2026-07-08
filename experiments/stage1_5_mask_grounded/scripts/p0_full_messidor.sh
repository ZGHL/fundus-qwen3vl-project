#!/bin/bash
# P0 full Messidor-2 (1744). our model=vllm_infer+adapter (NO merge), qwen3_vl_nothink; baselines=run_vlm_perception.
# FIX: container mount is /sda/zgh -> /workspace, so docker-exec output paths MUST be /workspace/... (host reads /sda/zgh/...).
set -u
CT_OLD=gb10_pytorch_zgh; CT_NEW=gb10_vllm_infer
WLF=/workspace/LLaMA-Factory; HLF=/sda/zgh/LLaMA-Factory
HEXP=/sda/zgh/stage1_5_experiment; PREDS=$HEXP/preds          # host preds
CPREDS=/workspace/stage1_5_experiment/preds                    # SAME dir inside container
LOG=$PREDS/p0_full.log; CSZ=500
ADP=saves/qwen3-vl-8b-fundus/lora/stage1_5_v3se/checkpoint-270
say(){ echo "[$(date '+%m-%d %H:%M:%S')] $*"|tee -a "$LOG"; }
gpu(){ local ct=$1; for t in 1 2 3 4 5 6; do docker exec $ct python3 -c "import torch,sys;sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null && return 0; say "$ct GPU down restart $t"; docker restart $ct>/dev/null 2>&1; sleep 25; done; return 1; }

our_infer(){ # qf(rel data) out(host) maxtok
  local qf=$1 out=$2 maxtok=$3 fn=$(basename "$2")
  local total=$(wc -l < $HLF/data/$qf); local nshard=$(( (total+CSZ-1)/CSZ ))
  say "=== OUR $fn: $total q, $nshard shards (vllm_infer+adapter) ==="
  for ((s=0;s<nshard;s++)); do
    local osh="$PREDS/${fn}.sh${s}"; local csave="$CPREDS/${fn}.sh${s}.tmp"; local want
    sed -n "$((s*CSZ+1)),$(((s+1)*CSZ))p" $HLF/data/$qf > /tmp/_psh.jsonl; want=$(wc -l < /tmp/_psh.jsonl)
    [ "$(wc -l < $osh 2>/dev/null||echo 0)" -ge "$want" ] && { say "  shard$s skip"; continue; }
    for try in 1 2 3; do
      gpu $CT_OLD||return 1
      cp /tmp/_psh.jsonl $HLF/data/p0_shard.jsonl
      say "  shard$s/$nshard try$try ($want q)"
      docker exec $CT_OLD bash -lc "cd $WLF && python scripts/vllm_infer.py --model_name_or_path ./models/Qwen3-VL-8B-Instruct --adapter_name_or_path $ADP --max_lora_rank 16 --enforce_eager true --dataset p0_shard --dataset_dir data --media_dir data --template qwen3_vl_nothink --cutoff_len 2304 --temperature 0 --top_p 1 --top_k -1 --max_new_tokens $maxtok --image_max_pixels 262144 --image_min_pixels 65536 --enable_thinking false --seed 20260613 --save_name $csave" >/dev/null 2>&1
      mv "${osh}.tmp" "$osh" 2>/dev/null
      [ "$(wc -l < $osh 2>/dev/null||echo 0)" -ge "$want" ] && { say "  shard$s ok ($(wc -l <$osh))"; break; }
      say "  shard$s incomplete restart"; docker restart $CT_OLD>/dev/null 2>&1; sleep 20
      [ $try -eq 3 ] && say "  WARN shard$s failed 3x"
    done
  done
  cat ${PREDS}/${fn}.sh* > "$out" 2>/dev/null; say "$fn assembled $(wc -l < $out 2>/dev/null||echo 0)/$total"
}
bb_infer(){ # model ct qf out maxtok
  local model=$1 ct=$2 qf=$3 out=$4 maxtok=$5 fn=$(basename "$4")
  local total=$(wc -l < $HLF/data/$qf); local nshard=$(( (total+CSZ-1)/CSZ ))
  say "=== BB $fn: $total q, $nshard shards (run_vlm_perception) ==="
  for ((s=0;s<nshard;s++)); do
    local sf="${qf%.jsonl}.sh${s}.jsonl"
    [ -f $HLF/data/$sf ] || sed -n "$((s*CSZ+1)),$(((s+1)*CSZ))p" $HLF/data/$qf > $HLF/data/$sf
    local want=$(wc -l < $HLF/data/$sf); local osh="$PREDS/${fn}.sh${s}"; local csave="$CPREDS/${fn}.sh${s}.tmp"
    [ "$(wc -l < $osh 2>/dev/null||echo 0)" -ge "$want" ] && { say "  shard$s skip"; continue; }
    for try in 1 2 3; do
      gpu $ct||return 1
      say "  shard$s/$nshard try$try ($want q)"
      docker exec -e MAX_TOKENS=$maxtok -e MAX_PIXELS=262144 $ct bash -lc "cd $WLF && python scripts/run_vlm_perception.py models/$model data/$sf data $csave" >/dev/null 2>&1
      mv "${osh}.tmp" "$osh" 2>/dev/null
      [ "$(wc -l < $osh 2>/dev/null||echo 0)" -ge "$want" ] && { say "  shard$s ok ($(wc -l <$osh))"; break; }
      say "  shard$s incomplete restart"; docker restart $ct>/dev/null 2>&1; sleep 20
      [ $try -eq 3 ] && say "  WARN shard$s failed 3x"
    done
  done
  cat ${PREDS}/${fn}.sh* > "$out" 2>/dev/null; say "$fn assembled $(wc -l < $out 2>/dev/null||echo 0)/$total"
}

docker start $CT_OLD >/dev/null 2>&1; sleep 8; gpu $CT_OLD||exit 1
our_infer msd1744_audit_queries.jsonl $PREDS/msd1744_v3se270_audit.jsonl 256
bb_infer Qwen3-VL-8B-Instruct $CT_OLD msd1744_blackbox.jsonl $PREDS/msd1744_bb_Qwen.jsonl 400
docker start $CT_NEW >/dev/null 2>&1; sleep 10; gpu $CT_NEW||exit 1
bb_infer Lingshu-I-8B $CT_NEW msd1744_blackbox.jsonl $PREDS/msd1744_bb_Lingshu.jsonl 400
bb_infer InternVL3_5-8B-HF $CT_NEW msd1744_blackbox.jsonl $PREDS/msd1744_bb_InternVL.jsonl 400

say "=== P0_FULL_INFER_DONE ==="
rm -f $HLF/data/msd1744_blackbox.sh*.jsonl 2>/dev/null
