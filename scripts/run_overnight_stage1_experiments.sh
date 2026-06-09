#!/usr/bin/env bash
set -euo pipefail
ROOT=/workspace/LLaMA-Factory
PROJECT=/workspace/fundus-qwen3vl-project
PY=/workspace/qwen3vl-env/bin/python
CLI=/workspace/qwen3vl-env/bin/llamafactory-cli
LOG="$ROOT/logs/stage1_overnight_20260609"
BASE="$ROOT/saves/qwen3-vl-8b-fundus/lora/stage1_eval/base_model"
mkdir -p "$LOG" "$BASE/gold_dev" "$BASE/gold_test"
cd "$ROOT"

run_base_eval() {
  local split="$1"
  local config="$PROJECT/configs/eval/stage1_base_gold_${split}_fast.yaml"
  local output="$BASE/gold_${split}"
  if [[ ! -f "$output/stage1_metrics.json" ]]; then
    "$CLI" train "$config" 2>&1 | tee "$LOG/base_gold_${split}.log"
    "$PY" "$PROJECT/scripts/fundus_v4/score_stage1_en_cot.py" \
      "$output/generated_predictions.jsonl" --json-out "$output/stage1_metrics.json" \
      2>&1 | tee "$LOG/base_gold_${split}_score.log"
  fi
}

run_base_eval dev
run_base_eval test
"$PY" "$PROJECT/scripts/fundus_v4/make_stage1_base_comparison.py" 2>&1 | tee "$LOG/comparison_report.log"

# Continue with the gentle search only after the required baseline report exists.
"$PY" "$PROJECT/scripts/fundus_v4/build_stage1_gentle_calibration.py" 2>&1 | tee "$LOG/gentle_build.log"
"$PY" - <<'PYREG'
import json
from pathlib import Path
path=Path('data/annotation_v4/dataset_info.json'); data=json.loads(path.read_text(encoding='utf-8'))
data['fundus_stage1_en_cot_gentle_calibration_train']={'file_name':'fundus_stage1_en_cot_gentle_calibration_train_sft.jsonl','formatting':'sharegpt','columns':{'messages':'messages','images':'images'},'tags':{'role_tag':'role','content_tag':'content','user_tag':'user','assistant_tag':'assistant','system_tag':'system'}}
path.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
PYREG
"$CLI" train "$PROJECT/configs/train/stage1_en_cot_gentle_calibration.yaml" 2>&1 | tee "$LOG/gentle_train.log"

ADAPTER="$ROOT/saves/qwen3-vl-8b-fundus/lora/stage1_en_cot_gentle_calibrated"
EVAL="$ROOT/saves/qwen3-vl-8b-fundus/lora/stage1_eval/gentle_calibration"
mkdir -p "$EVAL/gold_dev" "$EVAL/gold_test"
for step in 10 20 30 40; do
  adapter="$ADAPTER/checkpoint-$step"; [[ -d "$adapter" ]] || continue
  cfg="$LOG/gentle_gold_dev_$step.yaml"; out="$EVAL/gold_dev/checkpoint-$step"
  "$PY" - "$PROJECT/configs/eval/stage1_en_cot_replay_gold_dev_fast.yaml" "$cfg" "$adapter" "$out" <<'PYCFG'
import sys,yaml
src,dst,adapter,out=sys.argv[1:]; c=yaml.safe_load(open(src)); c['adapter_name_or_path']=adapter; c['output_dir']=out; c['overwrite_cache']=False; yaml.safe_dump(c,open(dst,'w'),sort_keys=False)
PYCFG
  "$CLI" train "$cfg" 2>&1 | tee "$LOG/gentle_gold_dev_$step.log"
  "$PY" "$PROJECT/scripts/fundus_v4/score_stage1_en_cot.py" "$out/generated_predictions.jsonl" --json-out "$out/stage1_metrics.json" 2>&1 | tee "$LOG/gentle_gold_dev_${step}_score.log"
done

if "$PY" "$PROJECT/scripts/fundus_v4/select_stage1_targeted_candidate.py" --baseline "$PROJECT/reports/metrics/stage1_en_cot_gold_dev_metrics.json" --metrics-root "$EVAL/gold_dev" --adapter-root "$ADAPTER" --json-out "$EVAL/selection.json" 2>&1 | tee "$LOG/gentle_selection.log"; then
  selected="$($PY -c "import json; print(json.load(open('$EVAL/selection.json'))['selected']['adapter'])")"
  cfg="$LOG/gentle_gold_test_selected.yaml"
  "$PY" - "$PROJECT/configs/eval/stage1_en_cot_replay_gold_test_fast.yaml" "$cfg" "$selected" "$EVAL/gold_test" <<'PYTEST'
import sys,yaml
src,dst,adapter,out=sys.argv[1:]; c=yaml.safe_load(open(src)); c['adapter_name_or_path']=adapter; c['output_dir']=out; c['overwrite_cache']=False; yaml.safe_dump(c,open(dst,'w'),sort_keys=False)
PYTEST
  "$CLI" train "$cfg" 2>&1 | tee "$LOG/gentle_gold_test.log"
  "$PY" "$PROJECT/scripts/fundus_v4/score_stage1_en_cot.py" "$EVAL/gold_test/generated_predictions.jsonl" --json-out "$EVAL/gold_test/stage1_metrics.json" 2>&1 | tee "$LOG/gentle_gold_test_score.log"
else
  echo 'No gentle calibration candidate passed strict guardrails; retaining Adapter 1.' | tee "$LOG/gentle_result.txt"
fi
