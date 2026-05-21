#!/usr/bin/env python3
"""Build a pilot mixture for NV/IRMA-grounded Grade 4 training.

This combines:
* L4 grade4 augmented supervision
* L3 NV single-lesion supervision
* L3 IRMA single-lesion supervision

The pilot is intentionally compact and balanced so it can be trained quickly
before promoting to a larger run.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


BASE = Path("data/annotation")
SRC = {
    "l3_nv_train": BASE / "fundus_l3_nv_single_train_sft.jsonl",
    "l3_nv_holdout": BASE / "fundus_l3_nv_single_holdout_sft.jsonl",
    "l3_irma_train": BASE / "fundus_l3_irma_single_train_sft.jsonl",
    "l3_irma_holdout": BASE / "fundus_l3_irma_single_holdout_sft.jsonl",
    "l4_train": BASE / "fundus_l4_grade4_augmented_train_sft.jsonl",
    "l4_holdout": BASE / "fundus_l4_grade4_augmented_holdout_sft.jsonl",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def stable_hash(text: str) -> str:
    import hashlib

    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def stable_shuffle(rows: list[dict[str, Any]], salt: str) -> list[dict[str, Any]]:
    return sorted(
        rows,
        key=lambda row: stable_hash(
            f"{salt}::{row.get('meta', {}).get('record_id', '')}::{row.get('meta', {}).get('task', '')}"
        ),
    )


def take(rows: list[dict[str, Any]], n: int) -> list[dict[str, Any]]:
    if n <= 0 or not rows:
        return []
    if n <= len(rows):
        return rows[:n]
    out: list[dict[str, Any]] = []
    while len(out) < n:
        out.extend(rows[: min(len(rows), n - len(out))])
    return out[:n]


def extract_json(row: dict[str, Any]) -> dict[str, Any]:
    text = row["messages"][-1]["content"]
    tail = text.split("【JSON】", 1)[-1] if "【JSON】" in text else text
    return json.loads(tail.strip())


def tag_stage(rows: list[dict[str, Any]], stage: str) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        item = json.loads(json.dumps(row, ensure_ascii=False))
        meta = dict(item.get("meta", {}))
        meta["stage_mix"] = stage
        item["meta"] = meta
        out.append(item)
    return out


def balance_presence(rows: list[dict[str, Any]], task: str, key: str, n: int) -> list[dict[str, Any]]:
    pos: list[dict[str, Any]] = []
    neg: list[dict[str, Any]] = []
    for row in rows:
        meta = row.get("meta", {})
        if meta.get("task") != task:
            continue
        obj = extract_json(row)
        present = bool(obj.get(key) == "present" or obj.get("present"))
        if present:
            pos.append(row)
        else:
            neg.append(row)
    pos = stable_shuffle(pos, f"{task}::pos")
    neg = stable_shuffle(neg, f"{task}::neg")
    half = n // 2
    return take(pos, half + (n % 2)) + take(neg, half)


def balance_grade(rows: list[dict[str, Any]], n: int) -> list[dict[str, Any]]:
    by_grade: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        obj = extract_json(row)
        grade = obj.get("dr_grade")
        if isinstance(grade, int):
            by_grade[grade].append(row)
    out: list[dict[str, Any]] = []
    # This source only contains Grade 4 and Grade 3. Keep the split balanced.
    quotas = {4: n // 2, 3: n - (n // 2)}
    for grade in [4, 3]:
        out.extend(take(stable_shuffle(by_grade.get(grade, []), f"grade::{grade}"), quotas[grade]))
    return out[:n]


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    tasks = Counter()
    grades = Counter()
    for row in rows:
        meta = row.get("meta", {})
        tasks[meta.get("task", "unknown")] += 1
        if meta.get("task") == "L4_grade4_augmented":
            grades[extract_json(row).get("dr_grade")] += 1
    return {"n": len(rows), "tasks": dict(tasks), "l4_grades": dict(grades)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-n", type=int, default=900)
    parser.add_argument("--holdout-n", type=int, default=240)
    args = parser.parse_args()

    src = {k: stable_shuffle(read_jsonl(v), k) for k, v in SRC.items()}
    for rows, source in src.items():
        for row in source:
            row.setdefault("meta", {})["source_file"] = SRC[rows].name

    train_l4 = balance_grade(src["l4_train"], round(args.train_n * 0.50))
    train_nv = balance_presence(src["l3_nv_train"], "L3_NV_single", "present", round(args.train_n * 0.25))
    train_irma = balance_presence(src["l3_irma_train"], "L3_IRMA_single", "present", args.train_n - len(train_l4) - len(train_nv))
    train_rows = tag_stage(train_l4 + train_nv + train_irma, "nv_irma_grade4_pilot")

    hold_l4 = balance_grade(src["l4_holdout"], round(args.holdout_n * 0.50))
    hold_nv = balance_presence(src["l3_nv_holdout"], "L3_NV_single", "present", round(args.holdout_n * 0.25))
    hold_irma = balance_presence(src["l3_irma_holdout"], "L3_IRMA_single", "present", args.holdout_n - len(hold_l4) - len(hold_nv))
    hold_rows = tag_stage(hold_l4 + hold_nv + hold_irma, "nv_irma_grade4_pilot_holdout")

    train_path = BASE / "fundus_nv_irma_grade4_pilot_sft.jsonl"
    hold_path = BASE / "fundus_nv_irma_grade4_pilot_holdout_sft.jsonl"
    write_jsonl(train_path, train_rows)
    write_jsonl(hold_path, hold_rows)

    stats = {
        "train": summarize(train_rows),
        "holdout": summarize(hold_rows),
        "sources": {k: len(v) for k, v in src.items()},
    }
    (BASE / "fundus_nv_irma_grade4_pilot_stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
