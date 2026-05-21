#!/usr/bin/env python3
"""Build balanced six-lesion L3 calibration data.

This stage stays at lesion sensing only.  It mixes the calibrated MA/HE/EX/SE
single-lesion data with NV/IRMA single-lesion data, without DR grading samples.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import build_stage2_lite as stage2


BASE = Path("data/annotation")
L3_FULL = BASE / "fundus_l3_targeted_calib_v3_full_sft.jsonl"
NV_TRAIN = BASE / "fundus_l3_nv_single_train_sft.jsonl"
IRMA_TRAIN = BASE / "fundus_l3_irma_single_train_sft.jsonl"

TASKS = [
    "L3_MA_single",
    "L3_HE_single",
    "L3_EX_single",
    "L3_SE_single",
    "L3_NV_single",
    "L3_IRMA_single",
]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def clone(row: dict[str, Any]) -> dict[str, Any]:
    return json.loads(json.dumps(row, ensure_ascii=False))


def answer_obj(row: dict[str, Any]) -> dict[str, Any]:
    return json.loads(row["messages"][-1]["content"].split("【JSON】", 1)[-1].strip())


def take_cycle(rows: list[dict[str, Any]], n: int) -> list[dict[str, Any]]:
    if n <= 0 or not rows:
        return []
    out: list[dict[str, Any]] = []
    while len(out) < n:
        out.extend(rows[: min(len(rows), n - len(out))])
    return out


def normalize(row: dict[str, Any], stage: str, component: str) -> dict[str, Any]:
    item = clone(row)
    obj = answer_obj(item)
    meta = dict(item.get("meta", {}))
    meta["task"] = obj.get("task", meta.get("task"))
    meta["lesion"] = obj.get("lesion", meta.get("lesion"))
    meta["present"] = bool(obj.get("present"))
    meta["stage_mix"] = stage
    meta["mix_component"] = component
    item["meta"] = meta
    return item


def collect_sources() -> dict[str, list[dict[str, Any]]]:
    sources = read_jsonl(L3_FULL) + read_jsonl(NV_TRAIN) + read_jsonl(IRMA_TRAIN)
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in sources:
        try:
            obj = answer_obj(row)
        except Exception:
            continue
        task = obj.get("task")
        if task in TASKS:
            buckets[task].append(row)
    return buckets


def balanced_task(rows: list[dict[str, Any]], task: str, n: int, stage: str) -> list[dict[str, Any]]:
    buckets = {True: [], False: []}
    for row in rows:
        obj = answer_obj(row)
        if obj.get("task") == task:
            buckets[bool(obj.get("present"))].append(row)
    selected = take_cycle(stage2.stable_shuffle(buckets[True], f"{stage}_{task}_pos"), n // 2 + n % 2)
    selected += take_cycle(stage2.stable_shuffle(buckets[False], f"{stage}_{task}_neg"), n // 2)
    return [normalize(row, stage, task) for row in selected[:n]]


def summarize(rows: list[dict[str, Any]], source_buckets: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    tasks = Counter()
    present = Counter()
    components = Counter()
    missing = 0
    source_counts = {}
    for task, src_rows in source_buckets.items():
        c = Counter(bool(answer_obj(row).get("present")) for row in src_rows)
        source_counts[task] = {"total": len(src_rows), "positive": c[True], "negative": c[False]}
    for row in rows:
        meta = row.get("meta", {})
        task = meta.get("task", "unknown")
        tasks[task] += 1
        present[(task, bool(meta.get("present")))] += 1
        components[meta.get("mix_component", "unknown")] += 1
        for image in row.get("images", []):
            if not Path("data", image).exists():
                missing += 1
    return {
        "n": len(rows),
        "tasks": dict(tasks),
        "present": {str(k): v for k, v in sorted(present.items(), key=lambda x: str(x[0]))},
        "components": dict(components),
        "source_counts": source_counts,
        "missing_images": missing,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-task", type=int, default=1200)
    parser.add_argument("--output", default="data/annotation/fundus_l3_six_lesion_calib_pilot_sft.jsonl")
    args = parser.parse_args()

    stage = "l3_six_lesion_calib_pilot"
    sources = collect_sources()
    rows: list[dict[str, Any]] = []
    for task in TASKS:
        rows.extend(balanced_task(sources.get(task, []), task, args.per_task, stage))
    rows = stage2.stable_shuffle(rows, f"{stage}_final")

    out_path = Path(args.output)
    write_jsonl(out_path, rows)
    stats = {
        "output": str(out_path),
        "per_task": args.per_task,
        "requested_total": args.per_task * len(TASKS),
        "summary": summarize(rows, sources),
    }
    stats_path = BASE / "fundus_l3_six_lesion_calib_pilot_stats.json"
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
