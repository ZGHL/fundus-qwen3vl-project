#!/usr/bin/env python3
"""Build a global Stage2-Lite mix with controlled Grade4 augmentation."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import build_stage2_lite as stage2


BASE = Path("data/annotation")
L3_FULL = BASE / "fundus_l3_targeted_calib_v3_full_sft.jsonl"
L4_GRADE = BASE / "fundus_l4_evidence_grading_sft.jsonl"
L4_CONFLICT = BASE / "fundus_l4_conflict_review_sft.jsonl"
L4_GRADE4_AUG = BASE / "fundus_l4_grade4_augmented_train_sft.jsonl"
L3_NV = BASE / "fundus_l3_nv_single_train_sft.jsonl"
L3_IRMA = BASE / "fundus_l3_irma_single_train_sft.jsonl"


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


def take(rows: list[dict[str, Any]], n: int) -> list[dict[str, Any]]:
    if n <= 0 or not rows:
        return []
    if n <= len(rows):
        return rows[:n]
    out: list[dict[str, Any]] = []
    while len(out) < n:
        out.extend(rows[: min(len(rows), n - len(out))])
    return out


def clone(row: dict[str, Any]) -> dict[str, Any]:
    return json.loads(json.dumps(row, ensure_ascii=False))


def set_stage(rows: list[dict[str, Any]], stage: str, component: str) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        item = clone(row)
        meta = dict(item.get("meta", {}))
        meta["stage_mix"] = stage
        meta["mix_component"] = component
        item["meta"] = meta
        out.append(item)
    return out


def set_stage_only(rows: list[dict[str, Any]], stage: str) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        item = clone(row)
        meta = dict(item.get("meta", {}))
        meta["stage_mix"] = stage
        item["meta"] = meta
        out.append(item)
    return out


def normalize_l4_format(row: dict[str, Any], component: str) -> dict[str, Any]:
    item = clone(row)
    obj = stage2.extract_json(item["messages"][-1]["content"])
    grade = int(obj.get("dr_grade", 0))
    evidence = list(obj.get("evidence") or [])
    referable = grade >= 2
    task = "L4_evidence_bound_grading"
    ma_state = obj.get("MA", "unknown")
    nv_state = obj.get("NV", obj.get("neovascular_sign", "unknown"))
    irma_state = obj.get("IRMA", "unknown")

    item["messages"][0]["content"] = (
        "你是眼底 DR 分级助手。必须输出 L4_evidence_bound_grading JSON；"
        "先列出可靠病灶证据，再给出 dr_grade。不得输出 L3 任务名或 unknown grade。"
    )
    item["messages"][1]["content"] = "<image>\n请基于可见病灶证据完成 DR Grade 0-4 分级，并输出固定 JSON。"
    visible = ",".join(evidence) if evidence else "none"
    caveat = ""
    if grade == 4:
        caveat = "；若 NV/IRMA 未在事实层出现，不得编造，但仍需按监督标签输出 dr_grade=4"
    new_obj = {
        "task": task,
        "dr_grade": grade,
        "referable_dr": referable,
        "evidence": evidence,
        "MA": ma_state,
        "NV": nv_state,
        "IRMA": irma_state,
        "component": component,
    }
    item["messages"][-1]["content"] = stage2.answer(
        "逐项核查 MA、HE、EX、SE、NV、IRMA；只把事实层支持的病灶写入 evidence。",
        f"visible_lesions={visible}; MA={ma_state}; NV={nv_state}; IRMA={irma_state}; dr_grade={grade}",
        f"DR 分级为 Grade {grade}；referable_dr={str(referable).lower()}{caveat}。",
        new_obj,
    )
    meta = dict(item.get("meta", {}))
    meta["task"] = task
    meta["dr_grade"] = grade
    meta["mix_component"] = component
    item["meta"] = meta
    return item


def grade_of(row: dict[str, Any]) -> int | None:
    try:
        obj = stage2.extract_json(row["messages"][-1]["content"])
        grade = obj.get("dr_grade")
        return int(grade) if grade is not None else None
    except Exception:
        return None


def mix_l4_global(rows: list[dict[str, Any]], conflicts: list[dict[str, Any]], n: int) -> list[dict[str, Any]]:
    by_grade: dict[int, list[dict[str, Any]]] = {i: [] for i in range(5)}
    for row in rows:
        g = grade_of(row)
        if g in by_grade:
            by_grade[g].append(row)
    budgets = {0: round(n * 0.18), 1: round(n * 0.18), 2: round(n * 0.22), 3: round(n * 0.22)}
    budgets[4] = n - sum(budgets.values())
    out: list[dict[str, Any]] = []
    for grade, budget in budgets.items():
        selected = [normalize_l4_format(r, "global_l4") for r in take(by_grade[grade], budget)]
        out.extend(selected)
    return out


def mix_l3_replay(rows: list[dict[str, Any]], n: int) -> list[dict[str, Any]]:
    return stage2.mix_l3_replay(rows, n)


def mix_grade4_aug(rows: list[dict[str, Any]], n: int) -> list[dict[str, Any]]:
    selected = [normalize_l4_format(r, "grade4_aug") for r in take(rows, n)]
    return selected


def mix_boundary(l4_rows: list[dict[str, Any]], grade4_rows: list[dict[str, Any]], n: int) -> list[dict[str, Any]]:
    g3 = [r for r in l4_rows if grade_of(r) == 3]
    g4 = [r for r in l4_rows if grade_of(r) == 4]
    half = n // 2
    out = [normalize_l4_format(r, "grade3_grade4_boundary") for r in take(g3, half)]
    out.extend(normalize_l4_format(r, "grade3_grade4_boundary") for r in take(g4 + grade4_rows, n - len(out)))
    return out[:n]


def mix_format_repair(l4_rows: list[dict[str, Any]], conflicts: list[dict[str, Any]], n: int) -> list[dict[str, Any]]:
    conflict_n = min(len(conflicts), n // 3)
    out = [normalize_l4_format(r, "format_repair") for r in take(conflicts, conflict_n)]
    remaining = n - len(out)
    if remaining:
        out.extend(normalize_l4_format(r, "format_repair") for r in take(l4_rows, remaining))
    return out


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    tasks = Counter()
    components = Counter()
    grades = Counter()
    datasets = Counter()
    l3_present = Counter()
    for row in rows:
        meta = row.get("meta", {})
        tasks[meta.get("task", "unknown")] += 1
        components[meta.get("mix_component", "unknown")] += 1
        rid = meta.get("record_id", "unknown")
        datasets["::".join(rid.split("::")[:2])] += 1
        grade = grade_of(row)
        if grade is not None:
            grades[str(grade)] += 1
        if str(meta.get("task", "")).startswith("L3_"):
            l3_present[(meta.get("lesion"), bool(meta.get("present")))] += 1
    return {
        "n": len(rows),
        "tasks": dict(tasks),
        "components": dict(components),
        "l4_grades": dict(sorted(grades.items())),
        "l3_present": {str(k): v for k, v in sorted(l3_present.items(), key=lambda x: str(x[0]))},
        "datasets": dict(datasets),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--total", type=int, default=6000)
    parser.add_argument("--output", default="data/annotation/fundus_stage2_lite_grade4_aug_pilot_sft.jsonl")
    args = parser.parse_args()

    l3 = stage2.stable_shuffle(read_jsonl(L3_FULL), "stage2_global_grade4_l3")
    l4 = [stage2.enhance_l4(r) for r in stage2.stable_shuffle(read_jsonl(L4_GRADE), "stage2_global_grade4_l4")]
    conflicts = [stage2.enhance_l4(r) for r in stage2.stable_shuffle(read_jsonl(L4_CONFLICT), "stage2_global_grade4_conflict")]
    grade4_aug = stage2.stable_shuffle(read_jsonl(L4_GRADE4_AUG), "stage2_global_grade4_aug")

    for row in l3:
        row.setdefault("meta", {})["source_file"] = L3_FULL.name
    for row in l4:
        row.setdefault("meta", {})["source_file"] = L4_GRADE.name
    for row in conflicts:
        row.setdefault("meta", {})["source_file"] = L4_CONFLICT.name
    for row in grade4_aug:
        row.setdefault("meta", {})["source_file"] = L4_GRADE4_AUG.name

    l4_n = round(args.total * 0.55)
    l3_n = round(args.total * 0.20)
    grade4_n = round(args.total * 0.15)
    boundary_n = round(args.total * 0.07)
    repair_n = args.total - l4_n - l3_n - grade4_n - boundary_n

    rows: list[dict[str, Any]] = []
    rows.extend(mix_l4_global(l4, conflicts, l4_n))
    rows.extend(set_stage(mix_l3_replay(l3, l3_n), "stage2_lite_grade4_aug_pilot", "l3_replay"))
    rows.extend(mix_grade4_aug(grade4_aug, grade4_n))
    rows.extend(mix_boundary(l4, grade4_aug, boundary_n))
    rows.extend(mix_format_repair(l4, conflicts, repair_n))
    rows = set_stage_only(stage2.stable_shuffle(rows, "stage2_lite_grade4_aug_pilot_final"), "stage2_lite_grade4_aug_pilot")

    out_path = Path(args.output)
    write_jsonl(out_path, rows)
    stats = {
        "output": str(out_path),
        "requested_total": args.total,
        "budgets": {
            "global_l4": l4_n,
            "l3_replay": l3_n,
            "grade4_aug": grade4_n,
            "grade3_grade4_boundary": boundary_n,
            "format_repair": repair_n,
        },
        "summary": summarize(rows),
    }
    stats_path = BASE / "fundus_stage2_lite_grade4_aug_pilot_stats.json"
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
