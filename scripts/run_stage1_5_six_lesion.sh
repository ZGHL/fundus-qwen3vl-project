#!/usr/bin/env bash
set -euo pipefail
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false

ROOT=/workspace/LLaMA-Factory
PROJECT=/workspace/fundus-qwen3vl-project
PYTHON=/workspace/qwen3vl-env/bin/python
CLI=/workspace/qwen3vl-env/bin/llamafactory-cli
TRAIN_CONFIG="$PROJECT/configs/train/stage1_5_six_lesion_limited.yaml"
ADAPTER_DIR="$ROOT/saves/qwen3-vl-8b-fundus/lora/stage1_5_six_lesion_limited"
EVAL_ROOT="$ROOT/saves/qwen3-vl-8b-fundus/lora/stage1_eval/stage1_5_six_lesion"
LOG_DIR="$ROOT/logs/stage1_5_six_lesion"
BASELINE="$PROJECT/reports/metrics/stage1_overnight_balanced.json"

mkdir -p "$LOG_DIR/eval" "$EVAL_ROOT/gold_dev" "$EVAL_ROOT/gold_test" "$EVAL_ROOT/irma_locked" "$EVAL_ROOT/nv_locked"
cd "$ROOT"

"$CLI" train "$TRAIN_CONFIG" 2>&1 | tee "$LOG_DIR/train.log"

for step in 40 80 120 160 180; do
  adapter="$ADAPTER_DIR/checkpoint-$step"
  [[ -d "$adapter" ]] || continue
  output="$EVAL_ROOT/gold_dev/checkpoint-$step"
  config="$LOG_DIR/eval/gold_dev_checkpoint-$step.yaml"
  "$PYTHON" - "$PROJECT/configs/eval/stage1_en_cot_replay_gold_dev_fast.yaml" "$config" "$adapter" "$output" <<'PY'
import sys, yaml
src, dst, adapter, output = sys.argv[1:]
cfg = yaml.safe_load(open(src, encoding="utf-8"))
cfg["adapter_name_or_path"] = adapter
cfg["output_dir"] = output
cfg["overwrite_cache"] = False
with open(dst, "w", encoding="utf-8") as f:
    yaml.safe_dump(cfg, f, sort_keys=False)
PY
  "$CLI" train "$config" 2>&1 | tee "$LOG_DIR/eval/gold_dev_checkpoint-$step.log"
  "$PYTHON" "$PROJECT/scripts/fundus_v4/score_stage1_en_cot.py" \
    "$output/generated_predictions.jsonl" --json-out "$output/stage1_metrics.json" \
    2>&1 | tee "$LOG_DIR/eval/gold_dev_checkpoint-${step}_score.log"
done

"$PYTHON" - "$EVAL_ROOT/gold_dev" "$ADAPTER_DIR" "$BASELINE" "$EVAL_ROOT/selection.json" <<'PY'
import json, sys
from pathlib import Path
metrics_root, adapter_root, baseline_path, out = map(Path, sys.argv[1:])
baseline = json.loads(baseline_path.read_text())
candidates = []
for p in sorted(metrics_root.glob("checkpoint-*/stage1_metrics.json")):
    m = json.loads(p.read_text())
    macro = m["main4_macro"]
    passed = (
        macro["f1"] >= baseline["main4_macro"]["f1"] - 0.01
        and macro["balanced_accuracy"] >= 0.56
        and m["by_lesion"]["HE"]["f1"] >= 0.80
        and m["by_lesion"]["EX"]["f1"] >= 0.66
    )
    candidates.append({"name": p.parent.name, "adapter": str(adapter_root / p.parent.name), "passed": passed, "metrics": m})
eligible = [x for x in candidates if x["passed"]]
if not eligible:
    raise SystemExit("No Stage1.5 checkpoint passed common-lesion guardrails")
selected = max(eligible, key=lambda x: (x["metrics"]["main4_macro"]["balanced_accuracy"], x["metrics"]["main4_macro"]["f1"]))
out.write_text(json.dumps({"selected": selected, "candidates": candidates}, indent=2))
print(json.dumps({"selected": selected}, indent=2))
PY

selected="$("$PYTHON" -c "import json; print(json.load(open('$EVAL_ROOT/selection.json'))['selected']['adapter'])")"

run_eval () {
  local name="$1" dataset="$2" template="$3"
  local output="$EVAL_ROOT/$name" config="$LOG_DIR/eval/${name}_selected.yaml"
  "$PYTHON" - "$template" "$config" "$selected" "$output" "$dataset" <<'PY'
import sys, yaml
src, dst, adapter, output, dataset = sys.argv[1:]
cfg = yaml.safe_load(open(src, encoding="utf-8"))
cfg["adapter_name_or_path"] = adapter
cfg["output_dir"] = output
cfg["eval_dataset"] = dataset
cfg["overwrite_cache"] = False
with open(dst, "w", encoding="utf-8") as f:
    yaml.safe_dump(cfg, f, sort_keys=False)
PY
  "$CLI" train "$config" 2>&1 | tee "$LOG_DIR/eval/${name}_selected.log"
  "$PYTHON" "$PROJECT/scripts/fundus_v4/score_stage1_en_cot.py" \
    "$output/generated_predictions.jsonl" --json-out "$output/stage1_metrics.json" \
    2>&1 | tee "$LOG_DIR/eval/${name}_selected_score.log"
}

run_eval gold_test fundus_stage1_en_cot_gold_test "$PROJECT/configs/eval/stage1_en_cot_replay_gold_test_fast.yaml"
run_eval irma_locked fundus_stage1_en_cot_irma_locked "$PROJECT/configs/eval/stage1_en_cot_replay_gold_dev_fast.yaml"
run_eval nv_locked fundus_stage1_en_cot_nv_locked "$PROJECT/configs/eval/stage1_en_cot_replay_gold_dev_fast.yaml"
