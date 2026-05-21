#!/usr/bin/env python3
"""Build L3-grounded DR grading mixtures.

Stage2-Lite continues from the calibrated L3 adapter and teaches evidence-bound
DR grading while replaying L3 single-lesion questions to reduce forgetting.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


BASE = Path("data/annotation")
VALIDATED = Path("data/fundus_validated/validated_clean.jsonl")
L3_FULL = BASE / "fundus_l3_targeted_calib_v3_full_sft.jsonl"
L4_GRADE = BASE / "fundus_l4_evidence_grading_sft.jsonl"
L4_CONFLICT = BASE / "fundus_l4_conflict_review_sft.jsonl"


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
    return out


def extract_json(text: str) -> dict[str, Any]:
    tail = text.split("【JSON】", 1)[-1] if "【JSON】" in text else text
    return json.loads(tail.strip())


def answer(observation: str, evidence: str, conclusion: str, obj: dict[str, Any]) -> str:
    return (
        f"【观察】{observation}\n\n"
        f"【证据】{evidence}\n\n"
        f"【结论】{conclusion}\n\n"
        "【JSON】\n"
        + json.dumps(obj, ensure_ascii=False, separators=(",", ":"))
    )


def enhance_l4(row: dict[str, Any]) -> dict[str, Any]:
    item = json.loads(json.dumps(row, ensure_ascii=False))
    obj = extract_json(item["messages"][-1]["content"])
    task = obj.get("task")
    grade = obj.get("dr_grade")
    referable = isinstance(grade, int) and grade >= 2

    if task == "L4_grade0_no_reliable_dr":
        item["messages"][0]["content"] = "你是眼底分级助手。先核查 MA/HE/EX/SE 证据，再判断 DR 分级；Grade 0 不得编造病灶。"
        item["messages"][1]["content"] = "<image>\n请基于可见病灶证据判断是否支持 DR Grade 0。"
        new_obj = {"task": task, "dr_grade": 0, "referable_dr": False, "evidence": [], "source": "label+validated_clean"}
        item["messages"][-1]["content"] = answer(
            "先按病灶顺序检查：微小红点样 MA、暗红点片状 HE、亮黄色边界清楚 EX、灰白棉絮样 SE。清洗事实层未保留可靠阳性病灶。",
            "reliable_DR_lesions=false; MA/HE/EX/SE not supported; dr_grade=0",
            "支持 DR Grade 0；该结论来自无可靠 DR 病灶证据，而不是忽略图像直接输出标签。",
            new_obj,
        )
    elif task == "L4_grade1_template":
        item["messages"][0]["content"] = "你是眼底分级助手。Grade 1 可由 MA-only 规则解释，但 template_only 不能写成图像直接检出。"
        item["messages"][1]["content"] = "<image>\n该图标注为轻度 DR，请说明证据边界并给出分级。"
        new_obj = {
            "task": task,
            "dr_grade": 1,
            "referable_dr": False,
            "MA": "template_only",
            "ma_source": "grade_rule",
            "forbid": "RetSAM_detected_MA",
        }
        item["messages"][-1]["content"] = answer(
            "先排除 HE/EX/SE 可靠阳性；当前事实层没有可作为 L3 强监督的可见 MA。Grade 1 只能按 MA-only 标签规则解释。",
            "dr_grade=1; MA=template_only; MA source=grade_rule; RetSAM_MA=unsupported",
            "可输出轻度 DR，但必须说明 MA 是 template_only，不得声称 RetSAM 或图像直接检出 MA。",
            new_obj,
        )
    elif task == "L4_evidence_bound_grading":
        ev = list(obj.get("evidence") or [])
        ma_state = obj.get("MA", "unknown")
        item["messages"][0]["content"] = "你是眼底分级助手。DR 分级必须先引用 L3 病灶证据；MA unknown 时不得编造 MA，NV unknown 时不得编造增殖证据。"
        item["messages"][1]["content"] = "<image>\n请先列出支持分级的可见病灶，再给出 DR 分级和是否 referable。"
        evidence_text = ",".join(ev) if ev else "none"
        new_obj = {
            "task": task,
            "dr_grade": grade,
            "referable_dr": referable,
            "evidence": ev,
            "MA": ma_state,
            "source": "validated_clean",
        }
        caveat = "；Grade 4 标签不等同于本题可见 NV，若 NV 未在事实层出现则不得编造 NV" if grade == 4 else ""
        item["messages"][-1]["content"] = answer(
            "先判断 L3 病灶是否可靠可见，再把病灶证据映射到监督分级；不把 unknown 的 MA 或 NV 写成可见证据。",
            f"visible_lesions={evidence_text}; MA={ma_state}; dr_grade={grade}; referable_dr={str(referable).lower()}",
            f"监督分级为 DR Grade {grade}，主要依据为 {evidence_text}；referable_dr={str(referable).lower()}{caveat}。",
            new_obj,
        )
    elif task == "L4_conflict_review":
        item["messages"][0]["content"] = "你是眼底分级质控助手。若病灶证据与 grade 规则冲突，应输出 needs_review，不要强行合理化。"
        flags = obj.get("flags") or []
        new_obj = {
            "task": task,
            "dr_grade": grade,
            "needs_review": True,
            "flags": flags,
        }
        item["messages"][-1]["content"] = answer(
            "先核对强标注病灶与 grade 规则是否一致；冲突样本用于训练复核能力，不作为普通分级样本。",
            f"cleaning_flags={','.join(flags)}; dr_grade={grade}",
            "该样本应标记 needs_review=true，不能为了匹配标签而编造或忽略病灶证据。",
            new_obj,
        )
    return item


def with_stage(rows: list[dict[str, Any]], stage: str) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        item = dict(row)
        meta = dict(item.get("meta", {}))
        meta["stage_mix"] = stage
        item["meta"] = meta
        out.append(item)
    return out


def by_l4_grade(rows: list[dict[str, Any]]) -> dict[Any, list[dict[str, Any]]]:
    out: dict[Any, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        obj = extract_json(row["messages"][-1]["content"])
        out[obj.get("dr_grade")].append(row)
    return out


def mix_l4(rows: list[dict[str, Any]], conflicts: list[dict[str, Any]], n: int) -> list[dict[str, Any]]:
    grade = by_l4_grade(rows)
    n0 = round(n * 0.20)
    n1 = round(n * 0.20)
    nc = min(round(n * 0.05), len(conflicts) * 3)
    ne = n - n0 - n1 - nc
    n2 = ne // 3
    n3 = ne // 3
    n4 = ne - n2 - n3
    return (
        take(grade.get(0, []), n0)
        + take(grade.get(1, []), n1)
        + take(grade.get(2, []), n2)
        + take(grade.get(3, []), n3)
        + take(grade.get(4, []), n4)
        + take(conflicts, nc)
    )


def mix_l3_replay(rows: list[dict[str, Any]], n: int) -> list[dict[str, Any]]:
    buckets: dict[tuple[str, bool], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        meta = row.get("meta", {})
        task = meta.get("task")
        if task in {"L3_MA_single", "L3_HE_single", "L3_EX_single", "L3_SE_single"}:
            buckets[(task, bool(meta.get("present")))].append(row)

    out: list[dict[str, Any]] = []
    tasks = ["L3_MA_single", "L3_HE_single", "L3_EX_single", "L3_SE_single"]
    base = n // len(tasks)
    rem = n % len(tasks)
    for i, task in enumerate(tasks):
        budget = base + (1 if i < rem else 0)
        pos = budget // 2 + budget % 2
        neg = budget // 2
        out.extend(take(buckets.get((task, True), []), pos))
        out.extend(take(buckets.get((task, False), []), neg))
    return out[:n]


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    tasks = Counter()
    grades = Counter()
    datasets = Counter()
    source_files = Counter()
    l3_present = Counter()
    for row in rows:
        meta = row.get("meta", {})
        task = meta.get("task", "unknown")
        tasks[task] += 1
        rid = meta.get("record_id", "unknown")
        datasets["::".join(rid.split("::")[:2])] += 1
        source_files[meta.get("source_file", "unknown")] += 1
        if task.startswith("L4_"):
            obj = extract_json(row["messages"][-1]["content"])
            grades[obj.get("dr_grade")] += 1
        if task.startswith("L3_") and task.endswith("_single"):
            l3_present[(meta.get("lesion"), bool(meta.get("present")))] += 1
    return {
        "n": len(rows),
        "tasks": dict(tasks),
        "l4_grades": {str(k): v for k, v in sorted(grades.items(), key=lambda x: str(x[0]))},
        "l3_present": {str(k): v for k, v in sorted(l3_present.items(), key=lambda x: str(x[0]))},
        "source_files": dict(source_files),
        "datasets": dict(datasets),
    }


def load_build_sft_module():
    path = Path("scripts/fundus/build_fundus_sft.py")
    spec = importlib.util.spec_from_file_location("build_fundus_sft", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_l4_holdout(n: int) -> list[dict[str, Any]]:
    mod = load_build_sft_module()
    rows = []
    for record in read_jsonl(VALIDATED):
        sp, _, _ = mod.split_of(record, 10)
        if sp == "train":
            continue
        item = mod.l4_evidence(record, sp)
        if item:
            item.setdefault("meta", {})["source_file"] = "validated_clean_holdout"
            rows.append(enhance_l4(item))
        item = mod.l4_conflict(record, sp)
        if item:
            item.setdefault("meta", {})["source_file"] = "validated_clean_holdout"
            rows.append(enhance_l4(item))
    rows = stable_shuffle(rows, "stage2_lite_l4_holdout")

    by_grade = by_l4_grade(rows)
    conflict = [r for r in rows if r.get("meta", {}).get("task") == "L4_conflict_review"]
    grade_rows = [r for r in rows if r.get("meta", {}).get("task") != "L4_conflict_review"]
    selected = mix_l4(grade_rows, conflict, n)
    return with_stage(stable_shuffle(selected, "stage2_lite_l4_holdout_final"), "stage2_lite_eval")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot-n", type=int, default=5000)
    parser.add_argument("--full-n", type=int, default=16000)
    parser.add_argument("--holdout-n", type=int, default=120)
    args = parser.parse_args()

    l3 = stable_shuffle(read_jsonl(L3_FULL), "stage2_lite_l3_replay")
    for row in l3:
        row.setdefault("meta", {})["source_file"] = L3_FULL.name
    l4_grade = [enhance_l4(r) for r in stable_shuffle(read_jsonl(L4_GRADE), "stage2_lite_l4_grade")]
    l4_conflict = [enhance_l4(r) for r in stable_shuffle(read_jsonl(L4_CONFLICT), "stage2_lite_l4_conflict")]
    for row in l4_grade:
        row.setdefault("meta", {})["source_file"] = L4_GRADE.name
    for row in l4_conflict:
        row.setdefault("meta", {})["source_file"] = L4_CONFLICT.name

    outputs: dict[str, list[dict[str, Any]]] = {}
    for name, total in [("pilot", args.pilot_n), ("full", args.full_n)]:
        l4_n = round(total * 0.70)
        l3_n = total - l4_n
        rows = mix_l4(l4_grade, l4_conflict, l4_n) + mix_l3_replay(l3, l3_n)
        outputs[f"fundus_stage2_lite_{name}_sft.jsonl"] = with_stage(
            stable_shuffle(rows, f"stage2_lite_{name}_final"), f"stage2_lite_{name}"
        )

    outputs["fundus_stage2_lite_l4_holdout120_sft.jsonl"] = build_l4_holdout(args.holdout_n)

    stats = {}
    for filename, rows in outputs.items():
        write_jsonl(BASE / filename, rows)
        stats[filename] = summarize(rows)

    stats_path = BASE / "fundus_stage2_lite_stats.json"
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
