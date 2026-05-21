#!/usr/bin/env python3
"""Re-run eval stages only (Arm A + Arm B) after launcher fixed the
dataset->eval_dataset bug. Adapters are already trained.

Updates _v4_exp_status.json so the dashboard continues to track.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path("/workspace/LLaMA-Factory")
STATUS = ROOT / "_v4_exp_status.json"
LOG_DIR = ROOT / "_v4_exp_logs"
LOG_DIR.mkdir(exist_ok=True)

EVAL_STAGES = [
    {
        "name": "eval_arm_a",
        "label": "Eval Arm A — quick eval (112 samples, ~95min)",
        "adapter": "saves/qwen3-vl-8b-fundus/lora/v4_arm_a_mixed_5h",
        "out_dir": "saves/qwen3-vl-8b-fundus/lora/v4_arm_a_mixed_5h_predict",
        "stage_idx_in_status": 4,
    },
    {
        "name": "eval_arm_b",
        "label": "Eval Arm B — quick eval (112 samples, ~95min)",
        "adapter": "saves/qwen3-vl-8b-fundus/lora/v4_arm_b3_l4_final",
        "out_dir": "saves/qwen3-vl-8b-fundus/lora/v4_arm_b3_l4_final_predict",
        "stage_idx_in_status": 5,
    },
]

EVAL_YAML = """### v4 quick eval ({tag}) — 112 samples
model_name_or_path: ./models/Qwen3-VL-8B-Instruct
adapter_name_or_path: {adapter}
trust_remote_code: true
flash_attn: sdpa

stage: sft
do_predict: true
finetuning_type: lora

dataset_dir: data/annotation
media_dir: data
eval_dataset: fundus_v4_quick_eval
template: qwen3_vl_nothink
cutoff_len: 1280
max_samples: 112
preprocessing_num_workers: 4
overwrite_cache: true

image_max_pixels: 262144
image_min_pixels: 65536

output_dir: {out_dir}
per_device_eval_batch_size: 1
predict_with_generate: true
overwrite_output_dir: true
report_to: none
"""

state_lock = threading.Lock()


def load_status():
    return json.loads(STATUS.read_text())


def save_status(s):
    with state_lock:
        STATUS.write_text(json.dumps(s, indent=2))


def reset_eval_stages():
    s = load_status()
    for stage in EVAL_STAGES:
        idx = stage["stage_idx_in_status"]
        s["stages"][idx]["status"] = "pending"
        s["stages"][idx]["started_at"] = None
        s["stages"][idx]["completed_at"] = None
        s["stages"][idx]["current_step"] = 0
        s["stages"][idx]["loss_history"] = []
        s["stages"][idx]["last_log_line"] = ""
        s["stages"][idx]["elapsed_seconds"] = 0
        s["stages"][idx]["fail_reason"] = None
    save_status(s)


FAIL_PAT = re.compile(r"Traceback|RuntimeError|ValueError|FAILED|CUDA out|OutOfMemoryError")
STEP_PAT = re.compile(r"(\d+)/(\d+)\s+\[.*?(\d+\.\d+)s/it")


def parse_loop(stage_idx_in_status: int, log_path: Path):
    last_size = 0
    while True:
        s = load_status()
        if s["stages"][stage_idx_in_status]["status"] != "running":
            return
        try:
            if log_path.exists():
                size = log_path.stat().st_size
                if size > last_size:
                    with log_path.open("rb") as f:
                        f.seek(last_size)
                        chunk = f.read().decode(errors="ignore")
                    last_size = size
                    s = load_status()
                    st = s["stages"][stage_idx_in_status]
                    step_matches = STEP_PAT.findall(chunk)
                    if step_matches:
                        try:
                            st["current_step"] = int(step_matches[-1][0])
                            st["expected_steps"] = int(step_matches[-1][1])
                        except: pass
                    last_line = chunk.strip().split("\n")[-1][:200]
                    st["last_log_line"] = last_line
                    if FAIL_PAT.search(chunk):
                        st["status"] = "failed"
                        st["fail_reason"] = last_line
                    started = st.get("started_at")
                    if started:
                        st["elapsed_seconds"] = int(
                            time.time() - time.mktime(time.strptime(started, "%Y-%m-%d %H:%M:%S"))
                        )
                    save_status(s)
        except Exception:
            pass
        time.sleep(3)


def run(stage):
    idx = stage["stage_idx_in_status"]
    log_path = LOG_DIR / f"{stage['name']}.log"
    eval_yaml = LOG_DIR / f"{stage['name']}.yaml"
    eval_yaml.write_text(EVAL_YAML.format(
        tag=stage["name"], adapter=stage["adapter"], out_dir=stage["out_dir"]))

    s = load_status()
    s["current_stage_idx"] = idx
    s["stages"][idx]["status"] = "running"
    s["stages"][idx]["started_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    save_status(s)

    parser_t = threading.Thread(target=parse_loop, args=(idx, log_path), daemon=True)
    parser_t.start()

    cmd = ["llamafactory-cli", "train", str(eval_yaml)]
    proc = subprocess.Popen(
        cmd, cwd=str(ROOT),
        stdout=log_path.open("w"), stderr=subprocess.STDOUT,
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
    )
    rc = proc.wait()

    s = load_status()
    st = s["stages"][idx]
    st["completed_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    if rc != 0 and st["status"] != "failed":
        st["status"] = "failed"
        st["fail_reason"] = f"exit code {rc}; check {log_path}"
    elif st["status"] != "failed":
        st["status"] = "done"
    save_status(s)
    return st["status"] == "done"


def main():
    reset_eval_stages()
    for stage in EVAL_STAGES:
        ok = run(stage)
        if not ok:
            print(f"[FAIL] {stage['name']}")
            sys.exit(1)
        print(f"[DONE] {stage['name']}")
    s = load_status()
    s["current_stage_idx"] = -1
    save_status(s)


if __name__ == "__main__":
    main()
