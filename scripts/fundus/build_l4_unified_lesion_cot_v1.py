#!/usr/bin/env python3
"""Build unified lesion-driven L4 DR grading CoT data.

This dataset matches the current L3 foundation: the model can sense
MA/HE/EX/SE/NV with useful but imperfect reliability, while IRMA is conservative
and should be used mainly as a G3/G4 boundary cue.  Every L4 answer follows one
path: inspect six lesions, structure the lesion states, then map evidence to
DR Grade 0-4.  Samples whose grade label is not fully explained by direct lesion
facts are kept as evidence_limited instead of forcing a wrong lesion rationale.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import build_stage2_lite as stage2


BASE = Path("data/annotation")
VALIDATED = Path("data/fundus_validated/validated_clean.jsonl")
GRADE4_AUG = BASE / "fundus_l4_grade4_augmented_train_sft.jsonl"
LESIONS = ["MA", "HE", "EX", "SE", "IRMA", "NV"]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def load_build_sft_module():
    path = Path("scripts/fundus/build_fundus_sft.py")
    spec = importlib.util.spec_from_file_location("build_fundus_sft", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def norm_state(value: Any) -> str:
    text = str(value).strip().lower()
    if text in {"true", "present", "positive", "1", "yes"}:
        return "present"
    if text in {"false", "absent", "negative", "0", "no"}:
        return "absent"
    if text in {"template_only"}:
        return "template_only"
    if text in {"possible_by_grade_template"}:
        return "possible_by_grade_template"
    return "unknown"


def lesion_state_from_validated(record: dict[str, Any], key: str) -> tuple[str, dict[str, Any]]:
    d = record.get("lesions", {}).get(key)
    if not isinstance(d, dict):
        return "unknown", {"source": "missing"}
    state = norm_state(d.get("present"))
    source = d.get("source", "unknown")
    if key == "IRMA":
        return "unknown", {"source": "not_in_validated_clean"}
    if key == "NV" and state == "present" and source == "grade_rule":
        return "possible_by_grade_template", d
    return state, d


def visible_from_states(states: dict[str, str]) -> list[str]:
    out = []
    for key in LESIONS:
        if states.get(key) == "present":
            out.append(key)
    return out


def burden_desc(record: dict[str, Any], key: str) -> str:
    d = record.get("lesions", {}).get(key)
    if not isinstance(d, dict):
        return "unknown"
    parts = []
    if d.get("count_bucket") not in {None, "unknown"}:
        parts.append(f"count={d.get('count_bucket')}")
    if d.get("area_bucket") not in {None, "unknown"}:
        parts.append(f"area={d.get('area_bucket')}")
    return ",".join(parts) if parts else "unknown"


def decision_rule(grade: int, states: dict[str, str], evidence: list[str], evidence_limited: bool) -> tuple[str, str]:
    nv = states.get("NV", "unknown")
    irma = states.get("IRMA", "unknown")
    non_prolif = [x for x in ["MA", "HE", "EX", "SE"] if states.get(x) == "present"]
    if grade == 0:
        return "no_reliable_dr_lesion_grade0", "未见可靠 DR 病灶，支持 Grade 0。"
    if grade == 1:
        return "mild_or_template_ma_grade1", "轻度 DR 主要对应 MA-only 或极轻微病灶；若 MA 来自模板，只能作为标签规则解释。"
    if grade == 2:
        return "nonproliferative_lesions_without_severe_boundary_grade2", "存在明确非增殖性病灶，但未见可靠 NV，也缺少 IRMA/重度边界证据，更支持 Grade 2。"
    if grade == 3:
        if irma == "present":
            return "irma_without_nv_grade3_boundary", "IRMA 提示重度 NPDR/G3 边界；未见可靠 NV 时，不能单独升为 Grade 4。"
        return "heavy_nonproliferative_without_nv_grade3", "HE/EX/SE 或病灶负担较重时支持重度 NPDR；未见可靠 NV，因此不作为 PDR/Grade 4。"
    if grade == 4:
        if nv == "present":
            return "nv_present_grade4_pdr", "可见 NV，属于 PDR/Grade 4 的直接证据；其他病灶为辅助证据。"
        if nv == "possible_by_grade_template":
            return "grade4_label_with_possible_nv_template", "标签为 Grade 4，但当前事实层只有 possible_by_grade_template 的 NV；按监督标签输出，同时标记证据受限。"
        if evidence_limited:
            return "supervised_grade4_without_direct_nv_evidence_limited", "监督标签为 Grade 4，但当前事实层缺少直接 NV；不得把 HE/EX/SE 编造成 PDR 证据。"
        if irma == "present" or non_prolif:
            return "grade4_label_with_nonproliferative_evidence_limited", "监督标签为 Grade 4，但可见证据主要是非增殖性病灶；需标记 evidence_limited。"
    return "label_supervision_evidence_limited", "按监督标签输出，但病灶证据不能完全解释该等级。"


def answer_text(grade: int, states: dict[str, str], evidence: list[str], rule: str, rationale: str, evidence_limited: bool) -> str:
    obs = (
        "按固定顺序逐项核查六类 DR 相关病灶：MA 微小红点，HE 暗红点片状出血，"
        "EX 亮黄色硬性渗出，SE 灰白棉絮样软性渗出，IRMA 视网膜内异常迂曲/旁路样血管，"
        "NV 盘面或视网膜表面新生血管。"
    )
    ev = "; ".join(f"{k}={states.get(k, 'unknown')}" for k in LESIONS)
    ev += f"; visible_lesions={','.join(evidence) if evidence else 'none'}"
    grading = (
        "Grade0=无可靠 DR 病灶；Grade1=MA-only/轻微病灶；Grade2=明确 NPDR 但无重度或增殖证据；"
        "Grade3=重度 NPDR 或 IRMA/较重非增殖性负担且 NV absent/unknown；"
        "Grade4=NV 或其他增殖性证据。"
    )
    conclusion = f"{rationale} 因此输出 DR Grade {grade}。"
    if evidence_limited:
        conclusion += " 该样本 evidence_limited=true，表示分级标签强于当前可见病灶事实，不能把 unknown 病灶写成阳性。"
    payload = {
        "task": "L4_unified_lesion_cot",
        "dr_grade": grade,
        "referable_dr": grade >= 2,
        "evidence": evidence,
        "MA": states.get("MA", "unknown"),
        "HE": states.get("HE", "unknown"),
        "EX": states.get("EX", "unknown"),
        "SE": states.get("SE", "unknown"),
        "IRMA": states.get("IRMA", "unknown"),
        "NV": states.get("NV", "unknown"),
        "decision_rule": rule,
        "evidence_limited": evidence_limited,
    }
    return (
        f"【观察】{obs}\n\n"
        f"【证据】{ev}\n\n"
        f"【分级依据】{grading} {rationale}\n\n"
        f"【结论】{conclusion}\n\n"
        "【JSON】\n"
        + json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    )


def sft(system: str, user: str, assistant: str, image: str, meta: dict[str, Any]) -> dict[str, Any]:
    if image.startswith("data/"):
        image = image[len("data/") :]
    return {
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": "<image>\n" + user},
            {"role": "assistant", "content": assistant},
        ],
        "images": [image],
        "meta": meta,
    }


def build_from_validated(record: dict[str, Any], split: str) -> dict[str, Any] | None:
    grade = record.get("grade")
    if not isinstance(grade, int) or grade < 0 or grade > 4:
        return None
    if not record.get("usable_for", {}).get("L4"):
        return None
    states: dict[str, str] = {}
    for key in LESIONS:
        states[key], _ = lesion_state_from_validated(record, key)
    evidence = visible_from_states(states)

    evidence_limited = False
    if grade == 0:
        evidence = []
        states = {k: ("absent" if states.get(k) != "template_only" else states[k]) for k in LESIONS}
    elif grade == 1 and states.get("MA") == "template_only":
        evidence = ["MA"]
        evidence_limited = True
    elif grade == 4 and states.get("NV") != "present":
        evidence_limited = True
    elif not evidence:
        evidence_limited = True

    rule, rationale = decision_rule(grade, states, evidence, evidence_limited)
    system = (
        "你是眼底 DR 分级助手。必须先逐项核查 MA/HE/EX/SE/IRMA/NV，再根据病灶证据输出 DR Grade 0-4；"
        "NV 是 Grade4/PDR 的直接证据，IRMA 是 G3/G4 边界证据但不能单独当作 Grade4。"
    )
    user = "请先逐项核查六类 DR 病灶，再基于病灶证据给出 DR Grade 0-4、referable_dr 和固定 JSON。"
    return sft(
        system,
        user,
        answer_text(grade, states, evidence, rule, rationale, evidence_limited),
        record["image_path"],
        {
            "record_id": record.get("record_id"),
            "task": "L4_unified_lesion_cot",
            "split": split,
            "dr_grade": grade,
            "source_file": VALIDATED.name,
            "evidence_limited": evidence_limited,
            "decision_rule": rule,
        },
    )


def build_from_grade4_aug(row: dict[str, Any], split: str) -> dict[str, Any] | None:
    obj = stage2.extract_json(row["messages"][-1]["content"])
    grade = int(obj.get("dr_grade", -1))
    if grade not in {3, 4}:
        return None
    states = {k: "unknown" for k in LESIONS}
    for key in ["MA", "HE", "EX", "SE", "IRMA", "NV"]:
        if key in obj:
            states[key] = norm_state(obj.get(key))
    for key in obj.get("evidence") or []:
        if key in states and states[key] == "unknown":
            states[key] = "present"
    evidence = visible_from_states(states)

    # NV-present Grade3 rows are contradictory for the current PDR rule. Keep
    # them out of ordinary training instead of teaching "NV present but not G4".
    if grade == 3 and states.get("NV") == "present":
        return None
    evidence_limited = grade == 4 and states.get("NV") != "present"
    rule, rationale = decision_rule(grade, states, evidence, evidence_limited)
    system = (
        "你是眼底 DR 分级助手。必须先逐项核查 MA/HE/EX/SE/IRMA/NV，再根据病灶证据输出 DR Grade 0-4；"
        "NV present 直接支持 Grade4/PDR，IRMA present 但 NV absent 更支持 Grade3 边界。"
    )
    user = "请先逐项核查六类 DR 病灶，再基于病灶证据给出 DR Grade 0-4、referable_dr 和固定 JSON。"
    meta = dict(row.get("meta", {}))
    meta.update(
        {
            "task": "L4_unified_lesion_cot",
            "split": split,
            "dr_grade": grade,
            "source_file": GRADE4_AUG.name,
            "evidence_limited": evidence_limited,
            "decision_rule": rule,
        }
    )
    return sft(system, user, answer_text(grade, states, evidence, rule, rationale, evidence_limited), row["images"][0], meta)


def split_rows() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    mod = load_build_sft_module()
    train: list[dict[str, Any]] = []
    holdout: list[dict[str, Any]] = []
    for record in read_jsonl(VALIDATED):
        split, _, _ = mod.split_of(record, 10)
        item = build_from_validated(record, split)
        if item is None:
            continue
        if split == "train":
            train.append(item)
        else:
            holdout.append(item)

    for row in stage2.stable_shuffle(read_jsonl(GRADE4_AUG), "l4_unified_grade4_aug"):
        item = build_from_grade4_aug(row, "train")
        if item is not None:
            train.append(item)
    return train, holdout


def balance_train(rows: list[dict[str, Any]], n: int) -> list[dict[str, Any]]:
    by_grade: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_grade[int(row["meta"]["dr_grade"])].append(row)
    budgets = {0: round(n * 0.18), 1: round(n * 0.18), 2: round(n * 0.22), 3: round(n * 0.22)}
    budgets[4] = n - sum(budgets.values())
    out: list[dict[str, Any]] = []
    for grade in range(5):
        out.extend(stage2.take(stage2.stable_shuffle(by_grade[grade], f"l4_unified_g{grade}"), budgets[grade]))
    return stage2.stable_shuffle(out, "l4_unified_train_final")


def balance_holdout(rows: list[dict[str, Any]], n: int) -> list[dict[str, Any]]:
    by_grade: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_grade[int(row["meta"]["dr_grade"])].append(row)
    base = n // 5
    rem = n % 5
    out = []
    for grade in range(5):
        out.extend(stage2.take(stage2.stable_shuffle(by_grade[grade], f"l4_unified_holdout_g{grade}"), base + (1 if grade < rem else 0)))
    return stage2.stable_shuffle(out, "l4_unified_holdout_final")


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grades = Counter()
    rules = Counter()
    evidence_limited = Counter()
    lesion_states = Counter()
    source_files = Counter()
    missing = 0
    for row in rows:
        meta = row.get("meta", {})
        grades[str(meta.get("dr_grade"))] += 1
        rules[meta.get("decision_rule", "unknown")] += 1
        evidence_limited[str(bool(meta.get("evidence_limited")))] += 1
        source_files[meta.get("source_file", "unknown")] += 1
        obj = stage2.extract_json(row["messages"][-1]["content"])
        for key in LESIONS:
            lesion_states[(key, obj.get(key, "unknown"))] += 1
        for image in row.get("images", []):
            if not Path("data", image).exists():
                missing += 1
    return {
        "n": len(rows),
        "grades": dict(sorted(grades.items())),
        "decision_rules": dict(rules),
        "evidence_limited": dict(evidence_limited),
        "lesion_states": {str(k): v for k, v in sorted(lesion_states.items(), key=lambda x: str(x[0]))},
        "source_files": dict(source_files),
        "missing_images": missing,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-n", type=int, default=10000)
    parser.add_argument("--holdout-n", type=int, default=150)
    parser.add_argument("--train-output", default="data/annotation/fundus_l4_unified_lesion_cot_v1_sft.jsonl")
    parser.add_argument("--holdout-output", default="data/annotation/fundus_l4_unified_lesion_cot_v1_holdout_sft.jsonl")
    args = parser.parse_args()

    train_raw, holdout_raw = split_rows()
    train = balance_train(train_raw, args.train_n)
    holdout = balance_holdout(holdout_raw, args.holdout_n)
    write_jsonl(Path(args.train_output), train)
    write_jsonl(Path(args.holdout_output), holdout)
    stats = {
        "train_output": args.train_output,
        "holdout_output": args.holdout_output,
        "raw": {"train": summarize(train_raw), "holdout": summarize(holdout_raw)},
        "selected": {"train": summarize(train), "holdout": summarize(holdout)},
    }
    stats_path = BASE / "fundus_l4_unified_lesion_cot_v1_stats.json"
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
