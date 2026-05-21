#!/usr/bin/env python3
"""Build a strict global L4 mix with Grade4 augmentation.

This variant is intentionally narrower than stage2_lite_grade4_augmented:
it trains only the global DR Grade 0-4 task and keeps every answer in the same
L4_evidence_bound_grading JSON schema. It avoids L3 replay and abstention-style
phrases so the model cannot learn to answer L3_DR_review/unknown on L4 prompts.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import build_stage2_lite as stage2


BASE = Path("data/annotation")
L4_GRADE = BASE / "fundus_l4_evidence_grading_sft.jsonl"
L4_CONFLICT = BASE / "fundus_l4_conflict_review_sft.jsonl"
L4_GRADE4_AUG = BASE / "fundus_l4_grade4_augmented_train_sft.jsonl"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
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


def clone(row: dict[str, Any]) -> dict[str, Any]:
    return json.loads(json.dumps(row, ensure_ascii=False))


def take(rows: list[dict[str, Any]], n: int) -> list[dict[str, Any]]:
    if n <= 0 or not rows:
        return []
    if n <= len(rows):
        return rows[:n]
    out: list[dict[str, Any]] = []
    while len(out) < n:
        out.extend(rows[: min(len(rows), n - len(out))])
    return out


def answer(observation: str, evidence: str, conclusion: str, obj: dict[str, Any]) -> str:
    return (
        f"【观察】{observation}\n\n"
        f"【证据】{evidence}\n\n"
        f"【结论】{conclusion}\n\n"
        "【JSON】\n"
        + json.dumps(obj, ensure_ascii=False, separators=(",", ":"))
    )


def grade_of(row: dict[str, Any]) -> int | None:
    try:
        obj = stage2.extract_json(row["messages"][-1]["content"])
        grade = obj.get("dr_grade", obj.get("grade"))
        return int(grade) if grade is not None else None
    except Exception:
        return None


def evidence_of(obj: dict[str, Any], grade: int) -> list[str]:
    evidence = [str(x).upper() for x in (obj.get("evidence") or []) if str(x).strip()]
    if evidence:
        return evidence
    if grade == 0:
        return []
    if grade == 1:
        return ["MA"]
    if grade == 2:
        return ["MA", "HE"]
    if grade == 3:
        return ["HE", "EX", "SE"]
    return ["NV", "IRMA"]


def normalize_l4(row: dict[str, Any], component: str, stage: str) -> dict[str, Any]:
    item = clone(row)
    src_obj = stage2.extract_json(item["messages"][-1]["content"])
    grade = int(src_obj.get("dr_grade", src_obj.get("grade", 0)))
    evidence = evidence_of(src_obj, grade)
    referable = grade >= 2
    task = "L4_evidence_bound_grading"

    item["messages"][0]["content"] = (
        "你是眼底 DR 分级助手。本题只输出全局 L4_evidence_bound_grading。"
        "必须给出 dr_grade 0-4，不输出 L3 任务名、不输出 unknown、不输出 needs_review。"
    )
    item["messages"][1]["content"] = (
        "<image>\n请根据可见 DR 病灶证据完成 Grade 0-4 分级，并只输出固定 JSON。"
    )

    evidence_text = ",".join(evidence) if evidence else "none"
    obj = {
        "task": task,
        "dr_grade": grade,
        "referable_dr": referable,
        "evidence": evidence,
        "component": component,
    }
    item["messages"][-1]["content"] = answer(
        "按 MA、HE、EX、SE、NV、IRMA 顺序核查病灶，再把证据映射到 DR Grade 0-4。",
        f"visible_lesions={evidence_text}; dr_grade={grade}; referable_dr={str(referable).lower()}",
        f"DR Grade {grade}; referable_dr={str(referable).lower()}。",
        obj,
    )

    meta = dict(item.get("meta", {}))
    meta["task"] = task
    meta["dr_grade"] = grade
    meta["mix_component"] = component
    meta["stage_mix"] = stage
    item["meta"] = meta
    return item


def by_grade(rows: list[dict[str, Any]]) -> dict[int, list[dict[str, Any]]]:
    out = {i: [] for i in range(5)}
    for row in rows:
        grade = grade_of(row)
        if grade in out:
            out[grade].append(row)
    return out


def mix_global_l4(rows: list[dict[str, Any]], n: int, stage: str) -> list[dict[str, Any]]:
    buckets = by_grade(rows)
    budgets = {0: round(n * 0.19), 1: round(n * 0.19), 2: round(n * 0.22), 3: round(n * 0.22)}
    budgets[4] = n - sum(budgets.values())
    out: list[dict[str, Any]] = []
    for grade, budget in budgets.items():
        out.extend(normalize_l4(r, "global_l4", stage) for r in take(buckets[grade], budget))
    return out


def mix_grade4_aug(rows: list[dict[str, Any]], n: int, stage: str) -> list[dict[str, Any]]:
    return [normalize_l4(row, "grade4_aug", stage) for row in take(rows, n)]


def mix_boundary(l4_rows: list[dict[str, Any]], grade4_rows: list[dict[str, Any]], n: int, stage: str) -> list[dict[str, Any]]:
    g3 = [r for r in l4_rows if grade_of(r) == 3]
    g4 = [r for r in l4_rows if grade_of(r) == 4]
    half = n // 2
    out = [normalize_l4(r, "grade3_grade4_boundary", stage) for r in take(g3, half)]
    out.extend(normalize_l4(r, "grade3_grade4_boundary", stage) for r in take(g4 + grade4_rows, n - len(out)))
    return out[:n]


def mix_format_anchor(rows: list[dict[str, Any]], n: int, stage: str) -> list[dict[str, Any]]:
    selected = []
    for row in rows:
        grade = grade_of(row)
        if grade in {0, 1, 2, 3, 4}:
            selected.append(row)
    return [normalize_l4(row, "format_anchor", stage) for row in take(selected, n)]


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    tasks = Counter()
    components = Counter()
    grades = Counter()
    invalid_terms = Counter()
    for row in rows:
        meta = row.get("meta", {})
        tasks[meta.get("task", "unknown")] += 1
        components[meta.get("mix_component", "unknown")] += 1
        grade = grade_of(row)
        if grade is not None:
            grades[str(grade)] += 1
        text = row["messages"][-1]["content"].lower()
        for term in ("unknown", "needs_review", "l3_dr", "l3 dr", "dr_level"):
            if term in text:
                invalid_terms[term] += 1
    return {
        "n": len(rows),
        "tasks": dict(tasks),
        "components": dict(components),
        "l4_grades": dict(sorted(grades.items())),
        "invalid_terms": dict(invalid_terms),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--total", type=int, default=5000)
    parser.add_argument("--output", default="data/annotation/fundus_stage2_lite_grade4_strict_pilot_sft.jsonl")
    args = parser.parse_args()

    stage = "stage2_lite_grade4_strict_pilot"
    l4 = [stage2.enhance_l4(r) for r in stage2.stable_shuffle(read_jsonl(L4_GRADE), "strict_l4")]
    conflicts = [stage2.enhance_l4(r) for r in stage2.stable_shuffle(read_jsonl(L4_CONFLICT), "strict_conflict")]
    grade4_aug = stage2.stable_shuffle(read_jsonl(L4_GRADE4_AUG), "strict_grade4_aug")
    all_l4 = l4 + conflicts

    global_n = round(args.total * 0.75)
    grade4_n = round(args.total * 0.15)
    boundary_n = round(args.total * 0.07)
    anchor_n = args.total - global_n - grade4_n - boundary_n

    rows: list[dict[str, Any]] = []
    rows.extend(mix_global_l4(all_l4, global_n, stage))
    rows.extend(mix_grade4_aug(grade4_aug, grade4_n, stage))
    rows.extend(mix_boundary(all_l4, grade4_aug, boundary_n, stage))
    rows.extend(mix_format_anchor(l4, anchor_n, stage))
    rows = stage2.stable_shuffle(rows, "strict_final")

    out_path = Path(args.output)
    write_jsonl(out_path, rows)
    stats = {
        "output": str(out_path),
        "requested_total": args.total,
        "budgets": {
            "global_l4": global_n,
            "grade4_aug": grade4_n,
            "grade3_grade4_boundary": boundary_n,
            "format_anchor": anchor_n,
        },
        "summary": summarize(rows),
    }
    stats_path = BASE / "fundus_stage2_lite_grade4_strict_pilot_stats.json"
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
