#!/usr/bin/env python3
"""Build L3 NV/IRMA lesion-sensing pilot data with single-lesion replay."""

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


def take(rows: list[dict[str, Any]], n: int) -> list[dict[str, Any]]:
    if n <= 0 or not rows:
        return []
    if n <= len(rows):
        return rows[:n]
    out: list[dict[str, Any]] = []
    while len(out) < n:
        out.extend(rows[: min(len(rows), n - len(out))])
    return out


def normalize_meta(row: dict[str, Any], stage: str, component: str) -> dict[str, Any]:
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


def balanced_single(rows: list[dict[str, Any]], task: str, n: int, stage: str, component: str) -> list[dict[str, Any]]:
    buckets = {True: [], False: []}
    for row in rows:
        obj = answer_obj(row)
        if obj.get("task") == task:
            buckets[bool(obj.get("present"))].append(row)
    half = n // 2
    selected = take(buckets[True], half + n % 2) + take(buckets[False], half)
    return [normalize_meta(row, stage, component) for row in selected[:n]]


def l3_replay(rows: list[dict[str, Any]], n: int, stage: str) -> list[dict[str, Any]]:
    buckets: dict[tuple[str, bool], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        obj = answer_obj(row)
        task = obj.get("task")
        if task in {"L3_MA_single", "L3_HE_single", "L3_EX_single", "L3_SE_single"}:
            buckets[(task, bool(obj.get("present")))].append(row)
    out: list[dict[str, Any]] = []
    tasks = ["L3_MA_single", "L3_HE_single", "L3_EX_single", "L3_SE_single"]
    base = n // len(tasks)
    rem = n % len(tasks)
    for i, task in enumerate(tasks):
        budget = base + (1 if i < rem else 0)
        out.extend(take(buckets[(task, True)], budget // 2 + budget % 2))
        out.extend(take(buckets[(task, False)], budget // 2))
    return [normalize_meta(row, stage, "l3_replay") for row in out[:n]]


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    tasks = Counter()
    present = Counter()
    components = Counter()
    for row in rows:
        meta = row.get("meta", {})
        task = meta.get("task", "unknown")
        tasks[task] += 1
        present[(task, bool(meta.get("present")))] += 1
        components[meta.get("mix_component", "unknown")] += 1
    return {
        "n": len(rows),
        "tasks": dict(tasks),
        "present": {str(k): v for k, v in sorted(present.items(), key=lambda x: str(x[0]))},
        "components": dict(components),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--total", type=int, default=3200)
    parser.add_argument("--output", default="data/annotation/fundus_l3_nv_irma_pilot_sft.jsonl")
    args = parser.parse_args()

    stage = "l3_nv_irma_pilot"
    nv = stage2.stable_shuffle(read_jsonl(NV_TRAIN), "l3_nv_irma_nv")
    irma = stage2.stable_shuffle(read_jsonl(IRMA_TRAIN), "l3_nv_irma_irma")
    replay_src = stage2.stable_shuffle(read_jsonl(L3_FULL), "l3_nv_irma_replay")

    nv_n = round(args.total * 0.28)
    irma_n = round(args.total * 0.32)
    replay_n = args.total - nv_n - irma_n

    rows: list[dict[str, Any]] = []
    rows.extend(balanced_single(nv, "L3_NV_single", nv_n, stage, "nv_single"))
    rows.extend(balanced_single(irma, "L3_IRMA_single", irma_n, stage, "irma_single"))
    rows.extend(l3_replay(replay_src, replay_n, stage))
    rows = stage2.stable_shuffle(rows, "l3_nv_irma_final")

    out_path = Path(args.output)
    write_jsonl(out_path, rows)
    stats = {
        "output": str(out_path),
        "requested_total": args.total,
        "budgets": {"nv_single": nv_n, "irma_single": irma_n, "l3_replay": replay_n},
        "summary": summarize(rows),
    }
    stats_path = BASE / "fundus_l3_nv_irma_pilot_stats.json"
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
