#!/usr/bin/env python3
"""Build joint L3+L4 lesion-to-grade CoT data.

This is route A: every L4 sample contains an explicit six-lesion L3-style
visual audit, followed by L4 grading evidence and a fixed JSON answer.
The input data is the already-cleaned L4 v3 selection, so grade distribution
and NV augmentation remain comparable to the current L4 baseline.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


BASE = Path("data/annotation")
VALIDATED = Path("data/fundus_validated/validated_clean.jsonl")
L4_V3_TRAIN = BASE / "fundus_l4_unified_lesion_cot_v3_sft.jsonl"
L4_V3_HOLDOUT = BASE / "fundus_l4_unified_lesion_cot_v3_holdout_sft.jsonl"
LESIONS = ["MA", "HE", "EX", "SE", "IRMA", "NV"]

GRADE_CAPS = {0: 1300, 1: 743, 2: 1600, 3: 878, 4: 1090}

OBS_PRESENT = {
    "MA": "可见微小红点样改变，形态符合微动脉瘤；需结合强标注来源判断其可靠性。",
    "HE": "可见暗红色点片状或斑块状出血样改变，提示视网膜出血证据。",
    "EX": "可见亮黄色、边界较清楚的沉积样病灶，符合硬性渗出表现。",
    "SE": "可见灰白色、边界相对模糊的棉絮样病灶，符合软性渗出表现。",
    "IRMA": "可见疑似视网膜内异常微血管样改变，可作为重度 NPDR 边界证据。",
    "NV": "可见异常新生血管样血管团或异常血管增殖表现，可作为 PDR 直接证据。",
}

OBS_ABSENT = {
    "MA": "未见可可靠确认的微小红点样微动脉瘤证据。",
    "HE": "未见明确暗红色出血样病灶。",
    "EX": "未见明确亮黄色硬性渗出样沉积灶。",
    "SE": "未见明确灰白棉絮样软性渗出灶。",
    "IRMA": "未见可可靠确认的视网膜内异常微血管证据。",
    "NV": "未见明确新生血管样异常血管团。",
}

OBS_UNKNOWN = {
    "MA": "当前样本缺少可靠 MA 强标注或可用候选，不能把未知写成阳性。",
    "HE": "当前样本缺少可靠 HE 病灶证据。",
    "EX": "当前样本缺少可靠 EX 病灶证据。",
    "SE": "当前样本缺少可靠 SE 病灶证据。",
    "IRMA": "当前样本缺少可靠 IRMA 病灶证据。",
    "NV": "当前样本缺少可靠 NV 病灶证据，不能据此编造 PDR。",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def load_stage2() -> Any:
    path = Path("scripts/fundus/build_stage2_lite.py")
    spec = importlib.util.spec_from_file_location("stage2_lite", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def extract_json(text: str) -> dict[str, Any]:
    if "【JSON】" in text:
        text = text.split("【JSON】", 1)[1]
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return {}
    try:
        return json.loads(m.group(0))
    except Exception:
        return {}


def norm_record_id(record_id: str) -> str:
    return record_id.split("__aug", 1)[0]


def load_validated_by_id() -> dict[str, dict[str, Any]]:
    out = {}
    if not VALIDATED.exists():
        return out
    for row in read_jsonl(VALIDATED):
        out[row["record_id"]] = row
    return out


def lesion_meta_from_validated(record: dict[str, Any] | None, lesion: str) -> dict[str, Any]:
    if not record:
        return {}
    return (record.get("lesions") or {}).get(lesion) or {}


def state_from_obj(obj: dict[str, Any], lesion: str) -> str:
    value = str(obj.get(lesion, "unknown")).lower()
    if value in {"present", "present_strong", "strong"}:
        return "present_strong"
    if value in {"weak", "present_weak"}:
        return "present_weak"
    if value in {"absent", "false"}:
        return "absent"
    if value in {"template"}:
        return "template"
    return "unknown"


def evidence_line(meta: dict[str, Any], state: str) -> str:
    if meta:
        parts = [f"source={meta.get('source', 'unknown')}"]
        if "present" in meta:
            parts.append(f"present={meta.get('present')}")
        if meta.get("count_bucket") is not None:
            parts.append(f"count={meta.get('count_bucket')}")
        elif meta.get("count") is not None:
            parts.append(f"count={meta.get('count')}")
        if meta.get("area_bucket") is not None:
            parts.append(f"area={meta.get('area_bucket')}")
        elif meta.get("area") is not None:
            parts.append(f"area={meta.get('area')}")
        if meta.get("confidence") is not None:
            parts.append(f"confidence={meta.get('confidence')}")
        if meta.get("suppressed_reason"):
            parts.append(f"suppressed_reason={meta.get('suppressed_reason')}")
        return "，".join(parts)
    if state == "absent":
        return "source=grade_rule_or_cleaning_rule，present=false"
    if state == "template":
        return "source=grade_rule_override，present=template"
    return "source=missing_or_unreliable，present=unknown"


def observation(lesion: str, state: str, meta: dict[str, Any]) -> str:
    if state in {"present_strong", "present_weak"}:
        return OBS_PRESENT[lesion]
    if state == "absent":
        if meta.get("suppressed_reason"):
            return f"{OBS_ABSENT[lesion]} 原始候选已因 {meta.get('suppressed_reason')} 被清洗压制。"
        return OBS_ABSENT[lesion]
    if state == "template":
        return f"{OBS_UNKNOWN[lesion]} 当前仅保留规则模板证据，不能当作直接视觉阳性。"
    return OBS_UNKNOWN[lesion]


def lesion_block(lesion: str, state: str, meta: dict[str, Any]) -> str:
    return (
        f"▸ {lesion}：\n"
        f"观察：{observation(lesion, state, meta)}\n"
        f"证据：{evidence_line(meta, state)}\n"
        f"verdict：{state}"
    )


def referable(grade: int) -> str:
    return "true" if grade >= 2 else "false"


def evidence_tier(meta: dict[str, Any]) -> str:
    if meta.get("evidence_limited"):
        return "supervised_evidence_limited"
    return "direct"


def selected_path_text(meta: dict[str, Any], obj: dict[str, Any]) -> str:
    grade = int(meta["dr_grade"])
    step = meta.get("selected_step")
    rule = meta.get("decision_rule")
    nv = obj.get("NV", "unknown")
    irma = obj.get("IRMA", "unknown")
    burden = meta.get("burden", obj.get("burden", "unknown"))
    lines = [
        "Step1：先核查 NV；NV present_strong 是 Grade4/PDR 的直接证据，不能用 HE/EX/SE 替代。",
        "Step2：若仅有 Grade4 标签或 NV template 而缺少直接 NV 证据，则标记 evidence_limited。",
        "Step3：再核查 IRMA 与重度 NPDR 负担；IRMA 只作为 Grade3 边界证据，不直接推出 Grade4。",
        "Step4：若存在可靠非增殖性病灶且负担为 light/moderate，倾向 Grade2。",
        "Step5：若主要为 MA-only 或轻微模板证据，倾向 Grade1。",
        "Step6：若无可靠 DR 病灶，判定 Grade0。",
        f"本例：NV={nv}，IRMA={irma}，NPDR_burden={burden}，选中路径 {step} → Grade {grade}（rule={rule}）。",
    ]
    if meta.get("evidence_limited"):
        lines.append("注意：本例可见病灶证据不足以完全独立支持该 grade，按监督标签学习分级边界，不得把 unknown 病灶写成直接阳性。")
    return "\n".join(lines)


def convert_row(row: dict[str, Any], validated: dict[str, dict[str, Any]]) -> dict[str, Any]:
    meta = dict(row.get("meta") or {})
    obj = extract_json(row["messages"][-1]["content"])
    rid = norm_record_id(str(meta.get("record_id", "")))
    record = validated.get(rid)
    lesion_states = {lesion: state_from_obj(obj, lesion) for lesion in LESIONS}
    strong = [k for k, v in lesion_states.items() if v == "present_strong"]
    weak = [k for k, v in lesion_states.items() if v == "present_weak"]
    unknown = [k for k, v in lesion_states.items() if v in {"unknown", "template"}]
    absent = [k for k, v in lesion_states.items() if v == "absent"]
    grade = int(meta["dr_grade"])
    burden = meta.get("burden", obj.get("burden", "unknown"))
    proliferative = meta.get("proliferative_evidence", obj.get("proliferative_evidence", "none"))
    boundary = meta.get("boundary_evidence", obj.get("boundary_evidence", "none"))

    blocks = [
        lesion_block(lesion, lesion_states[lesion], lesion_meta_from_validated(record, lesion))
        for lesion in LESIONS
    ]
    json_obj = {
        "task": "L3_L4_joint_lesion_grade_cot_v1",
        "dr_grade": grade,
        "referable_dr": grade >= 2,
        "MA": "present" if lesion_states["MA"].startswith("present") else lesion_states["MA"],
        "HE": "present" if lesion_states["HE"].startswith("present") else lesion_states["HE"],
        "EX": "present" if lesion_states["EX"].startswith("present") else lesion_states["EX"],
        "SE": "present" if lesion_states["SE"].startswith("present") else lesion_states["SE"],
        "IRMA": "present" if lesion_states["IRMA"].startswith("present") else lesion_states["IRMA"],
        "NV": "present" if lesion_states["NV"].startswith("present") else lesion_states["NV"],
        "evidence_strong": strong,
        "evidence_weak": weak,
        "NPDR_burden": burden,
        "proliferative_evidence": proliferative,
        "boundary_evidence": boundary,
        "selected_step": meta.get("selected_step"),
        "decision_rule": meta.get("decision_rule"),
        "evidence_tier": evidence_tier(meta),
        "evidence_limited": bool(meta.get("evidence_limited")),
    }
    content = (
        "【逐项核查】\n"
        + "\n\n".join(blocks)
        + "\n\n【证据强度归类】\n"
        + f"strong_present={strong if strong else []}\n"
        + f"weak_present={weak if weak else []}\n"
        + f"unknown_or_template={unknown if unknown else []}\n"
        + f"absent={absent if absent else []}\n\n"
        + "【病灶负担】\n"
        + f"NPDR_burden={burden}\n"
        + f"proliferative_evidence={proliferative}\n"
        + f"boundary_evidence={boundary}\n\n"
        + "【分级路径】\n"
        + selected_path_text(meta, obj)
        + "\n\n【结论】\n"
        + f"DR Grade {grade}，referable_dr={referable(grade)}，evidence_tier={evidence_tier(meta)}。\n\n"
        + "【JSON】\n"
        + json.dumps(json_obj, ensure_ascii=False, separators=(",", ":"))
    )
    new_row = {
        "messages": [
            {
                "role": "system",
                "content": (
                    "你是眼底 DR 分级助手。必须先按 MA/HE/EX/SE/IRMA/NV 顺序完成六病灶视觉核查，"
                    "每个病灶都输出观察、证据和 verdict；再汇总证据强度、NPDR 负担、增殖性/边界证据，"
                    "最后按有序规则判断 DR Grade 0-4 并输出固定 JSON。NV 是 Grade4/PDR 的直接证据；"
                    "IRMA 仅作为 Grade3 边界；不能把 HE/EX/SE 编造成 PDR 证据。"
                ),
            },
            {
                "role": "user",
                "content": (
                    "<image>\n"
                    "请逐项核查 MA/HE/EX/SE/IRMA/NV 六类病灶证据，并基于这些病灶证据判断 DR 分级。"
                    "请按【逐项核查】→【证据强度归类】→【病灶负担】→【分级路径】→【结论】→【JSON】输出。"
                ),
            },
            {"role": "assistant", "content": content},
        ],
        "images": row.get("images") or [],
        "meta": {
            **meta,
            "task": "L3_L4_joint_lesion_grade_cot_v1",
            "source_task": "L4_unified_lesion_cot_v3",
        },
    }
    return new_row


def select_train(rows: list[dict[str, Any]], stage2: Any) -> list[dict[str, Any]]:
    by_grade: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_grade[int(row["meta"]["dr_grade"])].append(row)
    out: list[dict[str, Any]] = []
    for grade in range(5):
        items = stage2.stable_shuffle(by_grade[grade], f"joint_v1_grade_{grade}")
        out.extend(items[: min(GRADE_CAPS[grade], len(items))])
    return stage2.stable_shuffle(out, "joint_v1_train_final")


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grades = Counter()
    steps = Counter()
    rules = Counter()
    evidence_limited = Counter()
    lesion_states = Counter()
    missing_images = 0
    for row in rows:
        meta = row.get("meta") or {}
        grades[str(meta.get("dr_grade"))] += 1
        steps[str(meta.get("selected_step"))] += 1
        rules[str(meta.get("decision_rule"))] += 1
        evidence_limited[str(bool(meta.get("evidence_limited")))] += 1
        obj = extract_json(row["messages"][-1]["content"])
        for lesion in LESIONS:
            lesion_states[f"{lesion}:{obj.get(lesion, 'unknown')}"] += 1
        for image in row.get("images") or []:
            if not Path("data", image).exists():
                missing_images += 1
    return {
        "n": len(rows),
        "grades": dict(sorted(grades.items())),
        "selected_step": dict(steps),
        "decision_rules": dict(rules),
        "evidence_limited": dict(evidence_limited),
        "lesion_states": dict(sorted(lesion_states.items())),
        "missing_images": missing_images,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-output", default="data/annotation/fundus_l3_l4_joint_lesion_grade_cot_v1_sft.jsonl")
    ap.add_argument("--holdout-output", default="data/annotation/fundus_l3_l4_joint_lesion_grade_cot_v1_holdout_sft.jsonl")
    args = ap.parse_args()

    stage2 = load_stage2()
    validated = load_validated_by_id()
    train_raw = [convert_row(row, validated) for row in read_jsonl(L4_V3_TRAIN)]
    holdout = [convert_row(row, validated) for row in read_jsonl(L4_V3_HOLDOUT)]
    train = select_train(train_raw, stage2)

    write_jsonl(Path(args.train_output), train)
    write_jsonl(Path(args.holdout_output), holdout)
    stats = {
        "name": "fundus_l3_l4_joint_lesion_grade_cot_v1",
        "design": "Route A: unified L3 six-lesion audit embedded in L4 DR grading CoT.",
        "grade_caps": GRADE_CAPS,
        "source_train": str(L4_V3_TRAIN),
        "source_holdout": str(L4_V3_HOLDOUT),
        "train_output": args.train_output,
        "holdout_output": args.holdout_output,
        "selected": {"train": summarize(train), "holdout": summarize(holdout)},
    }
    stats_path = BASE / "fundus_l3_l4_joint_lesion_grade_cot_v1_stats.json"
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
