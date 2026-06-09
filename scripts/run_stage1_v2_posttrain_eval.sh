#!/usr/bin/env bash
set -euo pipefail

ROOT=/workspace/LLaMA-Factory
PROJECT=/workspace/fundus-qwen3vl-project
PYTHON=/workspace/qwen3vl-env/bin/python
CLI=/workspace/qwen3vl-env/bin/llamafactory-cli
ADAPTER_DIR="$ROOT/saves/qwen3-vl-8b-fundus/lora/stage1_en_cot_v2_single_stage_projector"
EVAL_ROOT="$ROOT/saves/qwen3-vl-8b-fundus/lora/stage1_eval/v2_single_stage"
LOG_DIR="$ROOT/logs/stage1_en_cot_v2/eval"
BASELINE="$PROJECT/reports/metrics/stage1_en_cot_gold_dev_metrics.json"

mkdir -p "$EVAL_ROOT/gold_dev" "$EVAL_ROOT/gold_test" "$LOG_DIR"
cd "$ROOT"

declare -A seen_adapter_hashes=()
for adapter in "$ADAPTER_DIR"/checkpoint-* "$ADAPTER_DIR"; do
  [[ -d "$adapter" ]] || continue
  adapter_file="$adapter/adapter_model.safetensors"
  if [[ -f "$adapter_file" ]]; then
    adapter_hash="$(sha256sum "$adapter_file" | awk '{print $1}')"
    if [[ -n "${seen_adapter_hashes[$adapter_hash]:-}" ]]; then
      echo "Skipping duplicate adapter $adapter (same weights as ${seen_adapter_hashes[$adapter_hash]})"
      continue
    fi
    seen_adapter_hashes[$adapter_hash]="$adapter"
  fi
  name="$(basename "$adapter")"
  [[ "$adapter" == "$ADAPTER_DIR" ]] && name=final
  output="$EVAL_ROOT/gold_dev/$name"
  config="$LOG_DIR/gold_dev_${name}.yaml"
  "$PYTHON" - "$PROJECT/configs/eval/stage1_en_cot_replay_gold_dev_fast.yaml" "$config" "$adapter" "$output" <<'PY'
import sys, yaml
src, dst, adapter, output = sys.argv[1:]
cfg = yaml.safe_load(open(src, encoding="utf-8"))
cfg["adapter_name_or_path"] = adapter
cfg["output_dir"] = output
cfg["overwrite_cache"] = False
with open(dst, "w", encoding="utf-8") as handle:
    yaml.safe_dump(cfg, handle, sort_keys=False)
PY
  "$CLI" train "$config" 2>&1 | tee "$LOG_DIR/gold_dev_${name}.log"
  "$PYTHON" "$PROJECT/scripts/fundus_v4/score_stage1_en_cot.py" \
    "$output/generated_predictions.jsonl" --json-out "$output/stage1_metrics.json" \
    2>&1 | tee "$LOG_DIR/gold_dev_${name}_score.log"
done

"$PYTHON" "$PROJECT/scripts/fundus_v4/select_stage1_candidate.py" \
  --baseline "$BASELINE" --metrics-root "$EVAL_ROOT/gold_dev" \
  --adapter-root "$ADAPTER_DIR" --json-out "$EVAL_ROOT/selection.json" \
  2>&1 | tee "$LOG_DIR/selection.log"

selected="$("$PYTHON" -c "import json; print(json.load(open('$EVAL_ROOT/selection.json'))['selected']['adapter'])")"
test_config="$LOG_DIR/gold_test_selected.yaml"
"$PYTHON" - "$PROJECT/configs/eval/stage1_en_cot_replay_gold_test_fast.yaml" "$test_config" "$selected" "$EVAL_ROOT/gold_test" <<'PY'
import sys, yaml
src, dst, adapter, output = sys.argv[1:]
cfg = yaml.safe_load(open(src, encoding="utf-8"))
cfg["adapter_name_or_path"] = adapter
cfg["output_dir"] = output
cfg["overwrite_cache"] = False
with open(dst, "w", encoding="utf-8") as handle:
    yaml.safe_dump(cfg, handle, sort_keys=False)
PY
"$CLI" train "$test_config" 2>&1 | tee "$LOG_DIR/gold_test.log"
"$PYTHON" "$PROJECT/scripts/fundus_v4/score_stage1_en_cot.py" \
  "$EVAL_ROOT/gold_test/generated_predictions.jsonl" \
  --json-out "$EVAL_ROOT/gold_test/stage1_metrics.json" \
  2>&1 | tee "$LOG_DIR/gold_test_score.log"
