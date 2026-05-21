#!/usr/bin/env python3
"""Build ratio-controlled fundus SFT stage mixtures."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


BASE = Path("data/annotation")
FILES = {
    "l2_laterality": "fundus_l2_laterality_sft.jsonl",
    "l2_cdr": "fundus_l2_cdr_sft.jsonl",
    "l2_vessel": "fundus_l2_vessel_abstain_sft.jsonl",
    "l3_single": "fundus_l3_single_lesion_sft.jsonl",
    "l3_lesion_only": "fundus_l3_lesion_only_sft.jsonl",
    "l3_burden": "fundus_l3_burden_sft.jsonl",
    "l4_grade": "fundus_l4_evidence_grading_sft.jsonl",
    "l4_conflict": "fundus_l4_conflict_review_sft.jsonl",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def stable_hash(text: str) -> str:
    import hashlib

    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def stable_shuffle(rows: list[dict[str, Any]], salt: str) -> list[dict[str, Any]]:
    return sorted(
        rows,
        key=lambda row: stable_hash(f"{salt}::{row.get('meta', {}).get('record_id', '')}::{row.get('meta', {}).get('task', '')}"),
    )


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def take(rows: list[dict[str, Any]], n: int) -> list[dict[str, Any]]:
    if n <= 0 or not rows:
        return []
    if n <= len(rows):
        return rows[:n]
    out = []
    while len(out) < n:
        out.extend(rows[: min(len(rows), n - len(out))])
    return out


def by_task(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        out.setdefault(row.get("meta", {}).get("task", "unknown"), []).append(row)
    return out


def mix_l3_single_presence(rows: list[dict[str, Any]], n: int) -> list[dict[str, Any]]:
    tasks = ["L3_MA_single", "L3_HE_single", "L3_EX_single", "L3_SE_single"]
    by_presence: dict[tuple[str, bool], list[dict[str, Any]]] = {}
    for row in rows:
        meta = row.get("meta", {})
        task = meta.get("task")
        if task not in tasks:
            continue
        present = bool(meta.get("present"))
        by_presence.setdefault((task, present), []).append(row)

    out: list[dict[str, Any]] = []
    base = n // len(tasks)
    rem = n % len(tasks)
    for i, task in enumerate(tasks):
        budget = base + (1 if i < rem else 0)
        pos_n = budget // 2 + budget % 2
        neg_n = budget // 2
        out.extend(take(by_presence.get((task, True), []), pos_n))
        out.extend(take(by_presence.get((task, False), []), neg_n))

    if len(out) < n:
        selected = {(r.get("meta", {}).get("record_id"), r.get("meta", {}).get("task"), r.get("meta", {}).get("present")) for r in out}
        fallback = [
            r
            for r in rows
            if r.get("meta", {}).get("task") in tasks
            and (r.get("meta", {}).get("record_id"), r.get("meta", {}).get("task"), r.get("meta", {}).get("present"))
            not in selected
        ]
        out.extend(take(fallback, n - len(out)))
    return out[:n]


def with_stage(rows: list[dict[str, Any]], stage: str) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        item = dict(row)
        meta = dict(item.get("meta", {}))
        meta["stage_mix"] = stage
        item["meta"] = meta
        out.append(item)
    return out


def load_sources() -> dict[str, list[dict[str, Any]]]:
    data = {}
    for key, name in FILES.items():
        rows = read_jsonl(BASE / name)
        for row in rows:
            row.setdefault("meta", {})["source_file"] = name
        data[key] = stable_shuffle(rows, name)
    return data


def mix_l2(data: dict[str, list[dict[str, Any]]], n: int) -> list[dict[str, Any]]:
    n_lat = round(n * 0.40)
    n_cdr = round(n * 0.40)
    return take(data["l2_laterality"], n_lat) + take(data["l2_cdr"], n_cdr) + take(data["l2_vessel"], n - n_lat - n_cdr)


def mix_l3(data: dict[str, list[dict[str, Any]]], n: int) -> list[dict[str, Any]]:
    single_by_task = by_task(data["l3_single"])
    abstain = single_by_task.get("L3_SE_abstain", [])
    n_explicit = round(n * 0.60)
    n_lesion_only = round(n * 0.25)
    n_burden = round(n * 0.10)
    n_abstain = n - n_explicit - n_lesion_only - n_burden
    return (
        mix_l3_single_presence(data["l3_single"], n_explicit)
        + take(data["l3_lesion_only"], n_lesion_only)
        + take(data["l3_burden"], n_burden)
        + take(abstain, n_abstain)
    )


def mix_l4(data: dict[str, list[dict[str, Any]]], n: int) -> list[dict[str, Any]]:
    grade_by_task = by_task(data["l4_grade"])
    n0 = round(n * 0.15)
    n1 = round(n * 0.20)
    ne = round(n * 0.55)
    nc = n - n0 - n1 - ne
    # Conflict samples are important for rule learning, but only 52 unique
    # records exist. Cap repetition to reduce memorization, and shift the
    # leftover budget to evidence-bound grading.
    nc_cap = min(nc, len(data["l4_conflict"]) * 3)
    ne += nc - nc_cap
    nc = nc_cap
    return (
        take(grade_by_task.get("L4_grade0_no_reliable_dr", []), n0)
        + take(grade_by_task.get("L4_grade1_template", []), n1)
        + take(grade_by_task.get("L4_evidence_bound_grading", []), ne)
        + take(data["l4_conflict"], nc)
    )


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    tasks = Counter()
    source_files = Counter()
    datasets = Counter()
    for row in rows:
        meta = row.get("meta", {})
        tasks[meta.get("task", "unknown")] += 1
        source_files[meta.get("source_file", "unknown")] += 1
        rid = meta.get("record_id", "unknown")
        datasets["::".join(rid.split("::")[:2])] += 1
    return {"n": len(rows), "tasks": dict(tasks), "source_files": dict(source_files), "datasets": dict(datasets)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-n", type=int, default=600)
    parser.add_argument("--pilot-n", type=int, default=3000)
    parser.add_argument("--stage1-n", type=int, default=45000)
    args = parser.parse_args()
    data = load_sources()

    l3_total = len(data["l3_single"]) + len(data["l3_lesion_only"]) + len(data["l3_burden"])
    l4_total = len(data["l4_grade"]) + len(data["l4_conflict"])

    mixes = {
        "fundus_stage1_smoke_sft.jsonl": with_stage(
            mix_l2(data, round(args.smoke_n * 0.30)) + mix_l3(data, args.smoke_n - round(args.smoke_n * 0.30)),
            "stage1_smoke",
        ),
        "fundus_stage1_pilot_sft.jsonl": with_stage(
            mix_l2(data, round(args.pilot_n * 0.30)) + mix_l3(data, args.pilot_n - round(args.pilot_n * 0.30)),
            "stage1_pilot",
        ),
        "fundus_stage1_train_sft.jsonl": with_stage(
            mix_l2(data, round(args.stage1_n * 0.30)) + mix_l3(data, args.stage1_n - round(args.stage1_n * 0.30)),
            "stage1_train",
        ),
        "fundus_stage2_train_sft.jsonl": with_stage(
            mix_l2(data, round((l4_total / 0.30) * 0.20))
            + mix_l3(data, round((l4_total / 0.30) * 0.50))
            + mix_l4(data, l4_total),
            "stage2_train",
        ),
        "fundus_stage3_train_sft.jsonl": with_stage(
            mix_l2(data, round((l4_total / 0.25) * 0.20))
            + mix_l3(data, round((l4_total / 0.25) * 0.55))
            + mix_l4(data, l4_total),
            "stage3_train",
        ),
    }

    stats = {}
    for name, rows in mixes.items():
        write_jsonl(BASE / name, rows)
        stats[name] = summarize(rows)

    with (BASE / "fundus_stage_mix_stats.json").open("w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
