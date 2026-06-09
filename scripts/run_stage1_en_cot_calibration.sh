#!/usr/bin/env bash
set -euo pipefail

ROOT=/workspace/LLaMA-Factory
PROJECT=/workspace/fundus-qwen3vl-project
PYTHON=/workspace/qwen3vl-env/bin/python
CLI=/workspace/qwen3vl-env/bin/llamafactory-cli
LOG_DIR="$ROOT/logs/stage1_en_cot_calibration"
OUTPUT_DIR="$ROOT/saves/qwen3-vl-8b-fundus/lora/stage1_eval/calibrated_gold_test"
DATASET_INFO="$ROOT/data/annotation_v4/dataset_info.json"

mkdir -p "$LOG_DIR"
cd "$ROOT"

"$PYTHON" - <<'PY'
import json
from pathlib import Path

path = Path("data/annotation_v4/dataset_info.json")
entry = {
    "file_name": "fundus_stage1_en_cot_calibration_train_sft.jsonl",
    "formatting": "sharegpt",
    "columns": {"messages": "messages", "images": "images"},
    "tags": {
        "role_tag": "role",
        "content_tag": "content",
        "user_tag": "user",
        "assistant_tag": "assistant",
        "system_tag": "system",
    },
}
if path.exists():
    data = json.loads(path.read_text(encoding="utf-8"))
else:
    data = {}
data["fundus_stage1_en_cot_calibration_train"] = entry
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(f"registered fundus_stage1_en_cot_calibration_train in {path}")
PY

"$PYTHON" "$PROJECT/scripts/fundus_v4/build_stage1_en_cot_calibration.py" \
  2>&1 | tee "$LOG_DIR/build.log"

"$CLI" train "$PROJECT/configs/train/stage1_en_cot_calibration.yaml" \
  2>&1 | tee "$LOG_DIR/train.log"

"$CLI" train "$PROJECT/configs/eval/stage1_en_cot_calibrated_gold_test_fast.yaml" \
  2>&1 | tee "$LOG_DIR/gold_test.log"

"$PYTHON" "$PROJECT/scripts/fundus_v4/score_stage1_en_cot.py" \
  "$OUTPUT_DIR/generated_predictions.jsonl" \
  --json-out "$OUTPUT_DIR/stage1_metrics.json" \
  2>&1 | tee "$LOG_DIR/gold_test_score.log"
