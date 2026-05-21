#!/usr/bin/env python3
"""v4 Arm A vs Arm B sequential launcher with live status export.

Runs in the gb10_pytorch container. Sequentially launches all training and
eval stages, parsing each stage's log in real time to push step/loss/GPU
updates into a shared status JSON that the host dashboard reads.

Usage (inside container):
  cd /workspace/LLaMA-Factory
  python3 scripts/fundus_v4/launcher.py
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

ROOT = Path("/workspace/LLaMA-Factory")
STATUS = ROOT / "_v4_exp_status.json"
LOG_DIR = ROOT / "_v4_exp_logs"
LOG_DIR.mkdir(exist_ok=True)

# ----------------------------- stage plan -----------------------------

STAGES: list[dict[str, Any]] = [
    {
        "name": "arm_a_mixed",
        "label": "Arm A — Mixed (5h, 4k samples)",
        "kind": "train",
        "yaml": "examples/train_lora/qwen3vl_fundus_v4_mixed_5h.yaml",
        "expected_steps": 250,
        "expected_minutes": 310,
        "adapter_out": "saves/qwen3-vl-8b-fundus/lora/v4_arm_a_mixed_5h",
    },
    {
        "name": "arm_b1_l2",
        "label": "Arm B-1 — L2 only (1.5h, 1.3k samples)",
        "kind": "train",
        "yaml": "examples/train_lora/qwen3vl_fundus_v4_arm_b1_l2.yaml",
        "expected_steps": 81,
        "expected_minutes": 100,
        "adapter_out": "saves/qwen3-vl-8b-fundus/lora/v4_arm_b1_l2",
    },
    {
        "name": "arm_b2_l3",
        "label": "Arm B-2 — L3 continued from B-1 (1.5h, 1.3k samples)",
        "kind": "train",
        "yaml": "examples/train_lora/qwen3vl_fundus_v4_arm_b2_l3.yaml",
        "expected_steps": 81,
        "expected_minutes": 100,
        "adapter_out": "saves/qwen3-vl-8b-fundus/lora/v4_arm_b2_l3",
    },
    {
        "name": "arm_b3_l4",
        "label": "Arm B-3 — L4 continued from B-2 (2h, 1.4k samples)",
        "kind": "train",
        "yaml": "examples/train_lora/qwen3vl_fundus_v4_arm_b3_l4.yaml",
        "expected_steps": 87,
        "expected_minutes": 130,
        "adapter_out": "saves/qwen3-vl-8b-fundus/lora/v4_arm_b3_l4_final",
    },
    {
        "name": "eval_arm_a",
        "label": "Eval Arm A — quick eval (112 samples, ~95min)",
        "kind": "eval",
        "adapter": "saves/qwen3-vl-8b-fundus/lora/v4_arm_a_mixed_5h",
        "out_dir": "saves/qwen3-vl-8b-fundus/lora/v4_arm_a_mixed_5h_predict",
        "expected_minutes": 95,
    },
    {
        "name": "eval_arm_b",
        "label": "Eval Arm B — quick eval (112 samples, ~95min)",
        "kind": "eval",
        "adapter": "saves/qwen3-vl-8b-fundus/lora/v4_arm_b3_l4_final",
        "out_dir": "saves/qwen3-vl-8b-fundus/lora/v4_arm_b3_l4_final_predict",
        "expected_minutes": 95,
    },
]

EVAL_YAML_TEMPLATE = """### v4 quick eval ({tag}) — 112 samples
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

# ----------------------------- state -----------------------------

state: dict[str, Any] = {
    "experiment": "v4 Arm A (mixed) vs Arm B (staged) — 5h each",
    "started_at": None,
    "current_stage_idx": -1,
    "stages": [
        {**s, "status": "pending", "started_at": None, "completed_at": None,
         "current_step": 0, "loss_history": [], "last_log_line": "",
         "elapsed_seconds": 0, "fail_reason": None}
        for s in STAGES
    ],
    "gpu": {"util_pct": None, "memory_mb": None, "updated_at": None},
    "last_updated": None,
}

state_lock = threading.Lock()


def save_state():
    with state_lock:
        STATUS.write_text(json.dumps(state, indent=2))


# ----------------------------- GPU poller -----------------------------

def gpu_poller():
    while True:
        try:
            r = subprocess.run(
                ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5,
            )
            line = r.stdout.strip().split("\n")[0]
            util, mem = [x.strip() for x in line.split(",")]
            with state_lock:
                state["gpu"] = {
                    "util_pct": int(util) if util.isdigit() else None,
                    "memory_mb": int(mem) if mem.isdigit() else None,
                    "updated_at": time.strftime("%H:%M:%S"),
                }
            save_state()
        except Exception:
            pass
        time.sleep(5)


# ----------------------------- log parser -----------------------------

STEP_PAT = re.compile(r"(\d+)/(\d+)\s+\[")
LOSS_PAT = re.compile(r"'loss':\s*'?([\d.]+)'?")
EPOCH_PAT = re.compile(r"'epoch':\s*'?([\d.]+)'?")
DONE_PAT = re.compile(r"train_runtime|Predict")
FAIL_PAT = re.compile(r"Traceback|RuntimeError|FAILED|CUDA out|OutOfMemoryError")


def parse_loop(stage_idx: int, log_path: Path):
    """Tail the log file, push state updates while stage is running."""
    last_size = 0
    while True:
        with state_lock:
            cur_status = state["stages"][stage_idx]["status"]
        if cur_status not in {"running"}:
            break
        try:
            if log_path.exists():
                size = log_path.stat().st_size
                if size > last_size:
                    with log_path.open("rb") as f:
                        f.seek(last_size)
                        chunk = f.read().decode(errors="ignore")
                    last_size = size
                    # Find last step
                    step_matches = STEP_PAT.findall(chunk)
                    loss_matches = LOSS_PAT.findall(chunk)
                    last_line = chunk.strip().split("\n")[-1][:200]
                    with state_lock:
                        st = state["stages"][stage_idx]
                        if step_matches:
                            try:
                                st["current_step"] = int(step_matches[-1][0])
                                st["expected_steps"] = max(st["expected_steps"],
                                                           int(step_matches[-1][1]))
                            except: pass
                        for lv in loss_matches:
                            try:
                                st["loss_history"].append({
                                    "step": st["current_step"],
                                    "loss": float(lv),
                                    "at": time.strftime("%H:%M:%S"),
                                })
                            except: pass
                        st["last_log_line"] = last_line
                        if FAIL_PAT.search(chunk):
                            st["status"] = "failed"
                            st["fail_reason"] = last_line
                        st["elapsed_seconds"] = int(
                            time.time() - time.mktime(time.strptime(
                                st["started_at"], "%Y-%m-%d %H:%M:%S"))
                        ) if st["started_at"] else 0
                    save_state()
        except Exception:
            pass
        time.sleep(3)


# ----------------------------- stage runners -----------------------------

def run_stage(stage_idx: int):
    with state_lock:
        st = state["stages"][stage_idx]
        st["status"] = "running"
        st["started_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    state["current_stage_idx"] = stage_idx
    save_state()

    s = STAGES[stage_idx]
    log_path = LOG_DIR / f"{s['name']}.log"

    # Build command per kind
    if s["kind"] == "train":
        cmd = ["llamafactory-cli", "train", s["yaml"]]
    elif s["kind"] == "eval":
        # Generate eval yaml on the fly
        eval_yaml = LOG_DIR / f"{s['name']}.yaml"
        eval_yaml.write_text(EVAL_YAML_TEMPLATE.format(
            tag=s["name"], adapter=s["adapter"], out_dir=s["out_dir"]))
        cmd = ["llamafactory-cli", "train", str(eval_yaml)]
    else:
        raise ValueError(f"unknown kind {s['kind']}")

    # Start parser thread
    parser_t = threading.Thread(target=parse_loop, args=(stage_idx, log_path), daemon=True)
    parser_t.start()

    # Run subprocess
    proc = subprocess.Popen(
        cmd, cwd=str(ROOT),
        stdout=log_path.open("w"), stderr=subprocess.STDOUT,
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
    )
    rc = proc.wait()

    with state_lock:
        st = state["stages"][stage_idx]
        st["completed_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        if rc != 0 and st["status"] != "failed":
            st["status"] = "failed"
            st["fail_reason"] = f"exit code {rc}; check {log_path}"
        elif st["status"] != "failed":
            st["status"] = "done"
    save_state()
    return rc == 0 and state["stages"][stage_idx]["status"] == "done"


# ----------------------------- main -----------------------------

def main():
    state["started_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    save_state()

    # Start GPU poller
    threading.Thread(target=gpu_poller, daemon=True).start()

    for i, s in enumerate(STAGES):
        ok = run_stage(i)
        if not ok:
            print(f"[FAIL] stage {s['name']} failed, halting.")
            with state_lock:
                state["current_stage_idx"] = -1
            save_state()
            return 1
        print(f"[DONE] stage {s['name']}")

    print("All stages complete.")
    with state_lock:
        state["current_stage_idx"] = -1
    save_state()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
