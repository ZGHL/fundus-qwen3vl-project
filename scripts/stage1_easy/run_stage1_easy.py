#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

# Allow running this file directly: add repo root to sys.path.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.stage1_easy.progress import default_progress_path, mark_done, mark_error, mark_running, update_progress


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(time.time()))


def _log_dir() -> Path:
    return _repo_root() / "outputs" / "stage1_easy" / "monitor"


def _run(cmd: list[str], log_path: Path, cwd: Path | None = None, env: dict[str, str] | None = None) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    # Overwrite per run so the dashboard log tail always reflects the latest attempt
    # (append mode makes old Tracebacks look like "current" failures).
    with log_path.open("w", encoding="utf-8") as f:
        f.write(f"\n===== {_now_iso()} RUN =====\n")
        f.write("CMD: " + " ".join(cmd) + "\n")
        f.flush()
        p = subprocess.Popen(cmd, cwd=str(cwd or _repo_root()), env=env, stdout=f, stderr=subprocess.STDOUT, text=True)
        return p.wait()


def _yaml_get(path: Path, key: str) -> str | None:
    # minimal YAML key: value reader (no quotes, no nesting).
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        if s.startswith(key + ":"):
            return s.split(":", 1)[1].strip()
    return None


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Run Stage1 Easy pipeline with progress + logs.")
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--fgadr-per-grade", type=int, default=300)
    ap.add_argument("--train-config", default="examples/train_lora/qwen3vl_stage1_easy_lora.yaml")
    ap.add_argument("--skip-preprocess", action="store_true")
    ap.add_argument("--skip-preprocess-idrid", action="store_true")
    ap.add_argument("--skip-preprocess-fgadr", action="store_true")
    ap.add_argument("--skip-build", action="store_true")
    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument("--skip-vllm", action="store_true")
    ap.add_argument("--skip-eval", action="store_true")
    ap.add_argument(
        "--resume-from",
        choices=[
            "preprocess_idrid",
            "preprocess_fgadr",
            "build_dataset",
            "train",
            "vllm_infer",
            "eval_metrics",
        ],
        default=None,
        help="Skip all steps before this one (useful after fixing config / restarting container).",
    )
    ap.add_argument("--monitor-state", default=str(default_progress_path()))
    ap.add_argument("--pred-jsonl", default="outputs/stage1_easy_idrid_test_preds.vllm.jsonl")
    ap.add_argument("--metrics-out", default="reports/stage1_easy_idrid_metrics.json")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    root = _repo_root()
    os.chdir(root)

    prog = Path(args.monitor_state)
    if not prog.is_absolute():
        prog = root / prog

    logs = _log_dir()

    # Resume helper: skip everything strictly before the chosen step.
    # Note: preprocess_idrid/preprocess_fgadr share the same CLI block; resuming fgadr will rerun idrid too.
    resume = args.resume_from
    if resume is not None:
        order = ["preprocess_idrid", "preprocess_fgadr", "build_dataset", "train", "vllm_infer", "eval_metrics"]
        idx = order.index(resume)
        # default: run all
        args.skip_preprocess = False
        args.skip_preprocess_idrid = False
        args.skip_preprocess_fgadr = False
        args.skip_build = False
        args.skip_train = False
        args.skip_vllm = False
        args.skip_eval = False

        if idx > order.index("preprocess_idrid"):
            args.skip_preprocess_idrid = True
        if idx > order.index("preprocess_fgadr"):
            args.skip_preprocess_fgadr = True
        if idx > order.index("build_dataset"):
            args.skip_build = True
        if idx > order.index("train"):
            args.skip_train = True
        if idx > order.index("vllm_infer"):
            args.skip_vllm = True
        if idx > order.index("eval_metrics"):
            args.skip_eval = True

        args.skip_preprocess = bool(args.skip_preprocess_idrid and args.skip_preprocess_fgadr)

    # --- preprocess ---
    if not args.skip_preprocess:
        # IDRiD
        if not args.skip_preprocess_idrid:
            step = "preprocess_idrid"
            log_path = logs / f"{step}.log"
            mark_running(prog, step, log_path=str(log_path))
            rc = _run([sys.executable, "scripts/stage1_easy/preprocess.py", "--only", "idrid", "--data-root", args.data_root], log_path)
            if rc != 0:
                mark_error(prog, step, f"exit_code={rc}", log_path=str(log_path))
                return rc
            mark_done(prog, step, log_path=str(log_path))

        # FGADR
        if not args.skip_preprocess_fgadr:
            step = "preprocess_fgadr"
            log_path = logs / f"{step}.log"
            mark_running(prog, step, log_path=str(log_path))
            rc = _run([sys.executable, "scripts/stage1_easy/preprocess.py", "--only", "fgadr", "--data-root", args.data_root], log_path)
            if rc != 0:
                mark_error(prog, step, f"exit_code={rc}", log_path=str(log_path))
                return rc
            mark_done(prog, step, log_path=str(log_path))

    # --- build dataset ---
    if not args.skip_build:
        step = "build_dataset"
        log_path = logs / f"{step}.log"
        mark_running(prog, step, log_path=str(log_path), fgadr_per_grade=args.fgadr_per_grade)
        rc = _run(
            [sys.executable, "scripts/stage1_easy/build_dataset.py", "--data-root", args.data_root, "--fgadr-per-grade", str(args.fgadr_per_grade)],
            log_path,
        )
        if rc != 0:
            mark_error(prog, step, f"exit_code={rc}", log_path=str(log_path))
            return rc
        mark_done(prog, step, log_path=str(log_path))

    train_cfg = root / args.train_config
    adapter_out = _yaml_get(train_cfg, "output_dir") or "saves/qwen3-vl-8b-aptos/lora/stage1_easy"
    model_path = _yaml_get(train_cfg, "model_name_or_path") or "./models/Qwen3-VL-8B-Instruct"
    template = _yaml_get(train_cfg, "template") or "qwen3_vl_nothink"
    cutoff_len = _yaml_get(train_cfg, "cutoff_len") or "4096"
    img_max = _yaml_get(train_cfg, "image_max_pixels") or "1003520"
    img_min = _yaml_get(train_cfg, "image_min_pixels") or "200704"

    # --- train ---
    if not args.skip_train:
        step = "train"
        log_path = logs / f"{step}.log"
        mark_running(prog, step, log_path=str(log_path), output_dir=adapter_out, config=str(train_cfg))
        rc = _run(["llamafactory-cli", "train", str(train_cfg)], log_path)
        if rc != 0:
            mark_error(prog, step, f"exit_code={rc}", log_path=str(log_path))
            return rc
        mark_done(prog, step, log_path=str(log_path), output_dir=adapter_out)

    # --- vLLM infer ---
    if not args.skip_vllm:
        step = "vllm_infer"
        log_path = logs / f"{step}.log"
        pred_jsonl = args.pred_jsonl
        mark_running(prog, step, log_path=str(log_path), pred_jsonl=pred_jsonl, adapter=adapter_out)
        rc = _run(
            [
                sys.executable,
                "scripts/vllm_infer.py",
                "--model_name_or_path",
                model_path,
                "--adapter_name_or_path",
                adapter_out,
                "--dataset",
                "idrid_stage1_easy_test",
                "--dataset_dir",
                "data/annotation",
                "--media_dir",
                "data",
                "--template",
                template,
                "--cutoff_len",
                str(cutoff_len),
                "--max_new_tokens",
                "1024",
                "--image_max_pixels",
                str(img_max),
                "--image_min_pixels",
                str(img_min),
                "--save_name",
                pred_jsonl,
            ],
            log_path,
        )
        if rc != 0:
            mark_error(prog, step, f"exit_code={rc}", log_path=str(log_path))
            return rc
        mark_done(prog, step, log_path=str(log_path), pred_jsonl=pred_jsonl)

    # --- eval metrics ---
    if not args.skip_eval:
        step = "eval_metrics"
        log_path = logs / f"{step}.log"
        mark_running(prog, step, log_path=str(log_path), pred_jsonl=args.pred_jsonl, out=args.metrics_out)
        rc = _run(
            [
                sys.executable,
                "scripts/stage1_easy/eval_metrics.py",
                "--pred-jsonl",
                args.pred_jsonl,
                "--test-json",
                "data/annotation/idrid_stage1_easy_test.json",
                "--mask-dir",
                "data/idrid/segmentation/test",
                "--out",
                args.metrics_out,
            ],
            log_path,
        )
        if rc != 0:
            mark_error(prog, step, f"exit_code={rc}", log_path=str(log_path))
            return rc
        mark_done(prog, step, log_path=str(log_path), out=args.metrics_out)

    update_progress(prog, "pipeline", {"status": "done", "ended_at": _now_iso()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

