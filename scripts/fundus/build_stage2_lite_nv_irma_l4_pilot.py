#!/usr/bin/env python3
"""Build a two-step L4 fusion mix after NV/IRMA L3 tuning.

This mix intentionally contains only L4 grading questions.  The previous
Grade4 augmentation run mixed L3 single-lesion prompts into the global grading
stage, which improved local NV/IRMA sensing but caused global holdout answers
to drift toward L3 task names and unknown grades.
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


def grade_of(row: dict[str, Any]) -> int | None:
    try:
        obj = stage2.extract_json(row["messages"][-1]["content"])
        grade = obj.get("dr_grade")
        return int(grade) if grade is not None else None
    except Exception:
        return None


def norm_state(value: Any) -> str:
    text = str(value).strip().lower()
    if text in {"true", "present", "positive", "1", "yes"}:
        return "present"
    if text in {"false", "absent", "negative", "0", "no"}:
        return "absent"
    return "unknown"


def normalize_l4(row: dict[str, Any], component: str) -> dict[str, Any]:
    item = clone(row)
    obj = stage2.extract_json(item["messages"][-1]["content"])
    grade = int(obj.get("dr_grade", 0))
    referable = grade >= 2
    evidence = list(obj.get("evidence") or [])
    ma_state = norm_state(obj.get("MA", "unknown"))
    nv_state = norm_state(obj.get("NV", obj.get("neovascular_sign", "unknown")))
    irma_state = norm_state(obj.get("IRMA", "unknown"))

    item["messages"][0]["content"] = (
        "你是眼底分级助手。DR 分级必须先引用 L3 病灶证据；"
        "MA unknown 时不得编造 MA，NV unknown 时不得编造增殖证据。"
    )
    item["messages"][1]["content"] = "<image>\n请先列出支持分级的可见病灶，再给出 DR 分级和是否 referable。"

    visible = ",".join(evidence) if evidence else "none"
    new_obj = {
        "task": "L4_evidence_bound_grading",
        "dr_grade": grade,
        "referable_dr": referable,
        "evidence": evidence,
        "MA": ma_state,
        "NV": nv_state,
        "IRMA": irma_state,
        "component": component,
    }
    caveat = ""
    if grade == 4:
        caveat = "；Grade 4 优先学习 NV/IRMA 等增殖性或重度缺血相关证据与分级的对应关系"
    item["messages"][-1]["content"] = stage2.answer(
        "先判断 L3 病灶是否可靠可见，再把病灶证据映射到监督分级；不把 unknown 的 MA 或 NV 写成可见证据。",
        f"visible_lesions={visible}; MA={ma_state}; NV={nv_state}; IRMA={irma_state}; dr_grade={grade}; referable_dr={str(referable).lower()}",
        f"监督分级为 DR Grade {grade}，主要依据为 {visible}；referable_dr={str(referable).lower()}{caveat}。",
        new_obj,
    )

    meta = dict(item.get("meta", {}))
    meta["task"] = "L4_evidence_bound_grading"
    meta["dr_grade"] = grade
    meta["stage_mix"] = "stage2_lite_nv_irma_l4_pilot"
    meta["mix_component"] = component
    item["meta"] = meta
    return item


def mix_global_l4(l4_rows: list[dict[str, Any]], conflicts: list[dict[str, Any]], n: int) -> list[dict[str, Any]]:
    mixed = stage2.mix_l4(l4_rows, conflicts, n)
    return [normalize_l4(row, "global_l4") for row in mixed]


def mix_grade4_aug(rows: list[dict[str, Any]], n: int) -> list[dict[str, Any]]:
    return [normalize_l4(row, "grade4_aug") for row in take(rows, n)]


def mix_boundary(l4_rows: list[dict[str, Any]], grade4_rows: list[dict[str, Any]], n: int) -> list[dict[str, Any]]:
    g3 = [row for row in l4_rows if grade_of(row) == 3]
    g4 = [row for row in l4_rows if grade_of(row) == 4]
    half = n // 2
    out = [normalize_l4(row, "grade3_grade4_boundary") for row in take(g3, half)]
    out.extend(normalize_l4(row, "grade3_grade4_boundary") for row in take(g4 + grade4_rows, n - len(out)))
    return out[:n]


def mix_format_anchor(l4_rows: list[dict[str, Any]], n: int) -> list[dict[str, Any]]:
    by_grade: dict[int, list[dict[str, Any]]] = {i: [] for i in range(5)}
    for row in l4_rows:
        grade = grade_of(row)
        if grade in by_grade:
            by_grade[grade].append(row)
    out: list[dict[str, Any]] = []
    base = n // 5
    rem = n % 5
    for grade in range(5):
        budget = base + (1 if grade < rem else 0)
        out.extend(normalize_l4(row, "format_anchor") for row in take(by_grade[grade], budget))
    return out[:n]


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    components = Counter()
    grades = Counter()
    tasks = Counter()
    datasets = Counter()
    states = Counter()
    missing_images = 0
    for row in rows:
        meta = row.get("meta", {})
        components[meta.get("mix_component", "unknown")] += 1
        tasks[meta.get("task", "unknown")] += 1
        grade = grade_of(row)
        if grade is not None:
            grades[str(grade)] += 1
        rid = meta.get("record_id") or (row.get("images") or ["unknown"])[0]
        datasets["::".join(str(rid).split("::")[:2])] += 1
        obj = stage2.extract_json(row["messages"][-1]["content"])
        states[(obj.get("NV", "unknown"), obj.get("IRMA", "unknown"))] += 1
        for image in row.get("images", []):
            if not Path("data", image).exists():
                missing_images += 1
    return {
        "n": len(rows),
        "tasks": dict(tasks),
        "components": dict(components),
        "l4_grades": dict(sorted(grades.items())),
        "nv_irma_states": {str(k): v for k, v in sorted(states.items(), key=lambda x: str(x[0]))},
        "datasets": dict(datasets),
        "missing_images": missing_images,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--total", type=int, default=6000)
    parser.add_argument("--output", default="data/annotation/fundus_stage2_lite_nv_irma_l4_pilot_sft.jsonl")
    args = parser.parse_args()

    l4 = [stage2.enhance_l4(row) for row in stage2.stable_shuffle(read_jsonl(L4_GRADE), "nv_irma_l4_global")]
    conflicts = [stage2.enhance_l4(row) for row in stage2.stable_shuffle(read_jsonl(L4_CONFLICT), "nv_irma_l4_conflict")]
    grade4_aug = stage2.stable_shuffle(read_jsonl(L4_GRADE4_AUG), "nv_irma_l4_grade4_aug")

    for row in l4:
        row.setdefault("meta", {})["source_file"] = L4_GRADE.name
    for row in conflicts:
        row.setdefault("meta", {})["source_file"] = L4_CONFLICT.name
    for row in grade4_aug:
        row.setdefault("meta", {})["source_file"] = L4_GRADE4_AUG.name

    global_n = round(args.total * 0.75)
    grade4_n = round(args.total * 0.10)
    boundary_n = round(args.total * 0.10)
    anchor_n = args.total - global_n - grade4_n - boundary_n

    rows: list[dict[str, Any]] = []
    rows.extend(mix_global_l4(l4, conflicts, global_n))
    rows.extend(mix_grade4_aug(grade4_aug, grade4_n))
    rows.extend(mix_boundary(l4, grade4_aug, boundary_n))
    rows.extend(mix_format_anchor(l4, anchor_n))
    rows = stage2.stable_shuffle(rows, "stage2_lite_nv_irma_l4_pilot_final")

    output = Path(args.output)
    write_jsonl(output, rows)
    stats = {
        "output": str(output),
        "requested_total": args.total,
        "budgets": {
            "global_l4": global_n,
            "grade4_aug": grade4_n,
            "grade3_grade4_boundary": boundary_n,
            "format_anchor": anchor_n,
        },
        "summary": summarize(rows),
    }
    stats_path = BASE / "fundus_stage2_lite_nv_irma_l4_pilot_stats.json"
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
