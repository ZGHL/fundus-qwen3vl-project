#!/usr/bin/env python3
"""Build v2 of the unified lesion-driven L4 DR grading CoT.

v2 改进相对 v1：
- 每个病灶在【逐项核查】里给出 per-sample verdict_line，包含 present 状态
  + 来源强弱 + count/area 形态（仅在数据真有时填，否则诚实说 unknown）。
- 新增【证据强度归类】把 6 个 verdict 折叠到 strong_present/weak_present/
  unknown_or_template/absent 四类。
- 新增【病灶负担】基于 HE/EX/SE/MA 的 area_bucket 计算确定性 burden score，
  用来支撑 G2 vs G3 边界。
- 【分级路径】按 Step1..Step6 有序判断，可审计。
- evidence_limited 样本在【结论】里显式承认证据不足以独立解释 grade。

数据事实（来自实测 validated_clean.jsonl 11783 行）：
- HE/EX/SE 大约 10494 行有 count_bucket+area_bucket。
- MA 仅 81 行有 count/area；present 大多数没定量信息。
- IRMA 仅 1289 行（FGADR）有 present True/False，其余 10494 行 IRMA 字段为 None。
- NV present=True 仅 27 行；possible_by_grade_template 1152 行；其余 absent。
- usable_for.L4=True 只有 9493 行，其余被排除。
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

# verdict_line 中使用的固定状态枚举
STRONG_SOURCES = {"validated_retsam", "strong_mask_stage1_easy", "fgadr_lesion_only_sft_v3"}
WEAK_SOURCES = {"grade_rule_override"}


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
    if key == "IRMA" and not d:
        return "unknown", {"source": "not_assessed"}
    if key == "NV" and state == "present" and source == "grade_rule":
        # grade_rule 派生的 NV present 不能当强证据
        return "possible_by_grade_template", d
    return state, d


def classify_evidence_tier(state: str, lesion_dict: dict[str, Any]) -> str:
    """Map (present_state, source, area_bucket, conf) to one of:
    strong_present / weak_present / template / unknown / absent
    """
    if state == "absent":
        return "absent"
    if state == "unknown":
        return "unknown"
    if state == "template_only":
        return "template"
    if state == "possible_by_grade_template":
        return "template"
    if state != "present":
        return "unknown"

    source = lesion_dict.get("source", "")
    area = lesion_dict.get("area_bucket")
    conf = lesion_dict.get("confidence")

    if source in WEAK_SOURCES:
        return "weak_present"
    if source in STRONG_SOURCES:
        # 强来源里再用 area+conf 区分强弱
        if area in {"medium", "large"} and (conf is None or conf >= 0.5):
            return "strong_present"
        if area == "small" or (isinstance(conf, (int, float)) and conf < 0.5):
            return "weak_present"
        # 强来源但缺定量字段（如 MA FGADR / NV strong_mask）→ 仍记 strong
        return "strong_present"
    return "weak_present"


def verdict_line(key: str, state: str, tier: str, d: dict[str, Any]) -> str:
    """One line per lesion in 【逐项核查】."""
    cb = d.get("count_bucket") if isinstance(d, dict) else None
    ab = d.get("area_bucket") if isinstance(d, dict) else None
    src = d.get("source") if isinstance(d, dict) else None

    def has_morph() -> bool:
        return (cb in {"few", "some", "many"}) or (ab in {"small", "medium", "large"})

    if state == "absent":
        # 区分被清洗压制 vs 单纯 retsam_negative
        if isinstance(d, dict) and d.get("raw_present") is True:
            return f"{key}: absent（原始信号弱，已按清洗规则压制）"
        return f"{key}: absent"

    if state == "unknown":
        return f"{key}: unknown（当前来源不足以判定）"

    if state == "template_only":
        return f"{key}: template_only（标签提示 MA-only，无强病灶来源）"

    if state == "possible_by_grade_template":
        return f"{key}: possible_by_grade_template（分级标签为 PDR，但当前来源未直接观察到 NV）"

    # state == "present"
    if tier == "strong_present":
        if has_morph():
            return f"{key}: present_strong（数量={cb or 'unknown'}，面积={ab or 'unknown'}，来源={src}）"
        return f"{key}: present_strong（强标注来源={src}，无定量形态信息）"
    # weak_present
    if has_morph():
        return f"{key}: present_weak（数量={cb or 'unknown'}，面积={ab or 'unknown'}，来源={src}）"
    return f"{key}: present_weak（来源={src}，无定量形态信息）"


def compute_burden(states: dict[str, str], tiers: dict[str, str], lesions: dict[str, dict[str, Any]]) -> str:
    """NPDR 负担打分 → none/light/moderate/heavy.

    HE/EX/SE/MA 任一 strong_present：按 area_bucket 加权 (large=3, medium=2, small=1, 缺=1)。
    weak_present：固定 +1。其余 0。
    """
    score = 0
    for k in ("HE", "EX", "SE", "MA"):
        tier = tiers.get(k)
        if tier == "strong_present":
            ab = (lesions.get(k) or {}).get("area_bucket")
            score += {"large": 3, "medium": 2, "small": 1}.get(ab, 1)
        elif tier == "weak_present":
            score += 1
    if score == 0:
        return "none"
    if score <= 2:
        return "light"
    if score <= 5:
        return "moderate"
    return "heavy"


def proliferative_evidence(states: dict[str, str], tiers: dict[str, str]) -> str:
    if tiers.get("NV") == "strong_present":
        return "direct"
    if states.get("NV") == "possible_by_grade_template":
        return "possible_template"
    return "none"


def boundary_evidence(states: dict[str, str], tiers: dict[str, str]) -> str:
    if tiers.get("IRMA") == "strong_present":
        return "irma_present"
    return "none"


def select_step(
    grade: int,
    states: dict[str, str],
    tiers: dict[str, str],
    burden: str,
    prolif: str,
    boundary: str,
) -> tuple[str, str, bool]:
    """按 Step1..Step6 顺序选路径，返回 (step_id, decision_rule, evidence_limited)."""
    # 训练目标按 supervised label 对齐 grade，但 step 选哪条要尊重证据
    if grade == 4:
        if prolif == "direct":
            return "Step1", "nv_present_grade4_pdr", False
        if prolif == "possible_template":
            return "Step2a", "grade4_label_with_possible_nv_template", True
        # G4 标签但既无 direct 也无 template：纯 supervised，evidence 不足
        return "Step2b", "supervised_grade4_without_direct_nv_evidence_limited", True
    if grade == 3:
        if boundary == "irma_present":
            return "Step3a", "irma_without_nv_grade3_boundary", False
        if burden == "heavy":
            return "Step3b", "heavy_nonproliferative_without_nv_grade3", False
        # G3 标签但 burden 不到 heavy 也无 IRMA：evidence 不足
        return "Step3c", "supervised_grade3_evidence_limited", True
    if grade == 2:
        if burden in {"light", "moderate"}:
            return "Step4", "nonproliferative_lesions_without_severe_boundary_grade2", False
        # G2 标签但负担不足
        return "Step4b", "supervised_grade2_evidence_limited", True
    if grade == 1:
        if states.get("MA") == "template_only":
            return "Step5a", "mild_or_template_ma_grade1", True
        if tiers.get("MA") in {"strong_present", "weak_present"} and burden in {"none", "light"}:
            return "Step5b", "ma_only_grade1", False
        return "Step5c", "supervised_grade1_evidence_limited", True
    if grade == 0:
        if burden == "none" and prolif == "none" and boundary == "none":
            return "Step6", "no_reliable_dr_lesion_grade0", False
        return "Step6b", "supervised_grade0_unexpected_lesion", True
    return "StepX", "label_supervision_evidence_limited", True


def build_evidence_groups(states: dict[str, str], tiers: dict[str, str]) -> dict[str, list[str]]:
    out = {"strong_present": [], "weak_present": [], "unknown_or_template": [], "absent": []}
    for k in LESIONS:
        t = tiers.get(k, "unknown")
        if t == "strong_present":
            out["strong_present"].append(k)
        elif t == "weak_present":
            out["weak_present"].append(k)
        elif t in {"unknown", "template"}:
            out["unknown_or_template"].append(k)
        else:
            out["absent"].append(k)
    return out


GRADING_RULES_TEXT = (
    "Step1 PDR：NV strong_present → Grade 4；"
    "Step2 G4 标签但仅 NV template/缺证据 → Grade 4 + evidence_limited；"
    "Step3 G3：IRMA strong_present 或 NPDR_burden=heavy 且 NV absent → Grade 3；"
    "Step4 G2：NPDR_burden in {light,moderate} 且 NV/IRMA absent → Grade 2；"
    "Step5 G1：仅 MA-only / template_only_MA → Grade 1；"
    "Step6 G0：无可靠 DR 病灶 → Grade 0。"
)


def answer_text(
    grade: int,
    verdicts: dict[str, str],
    groups: dict[str, list[str]],
    burden: str,
    prolif: str,
    boundary: str,
    step: str,
    rule: str,
    evidence_limited: bool,
    states: dict[str, str],
) -> str:
    insp = "\n".join(verdicts[k] for k in LESIONS)
    grp = (
        f"strong_present={groups['strong_present'] or 'none'}; "
        f"weak_present={groups['weak_present'] or 'none'}; "
        f"unknown_or_template={groups['unknown_or_template'] or 'none'}; "
        f"absent={groups['absent'] or 'none'}"
    )
    burden_line = (
        f"NPDR_burden={burden}; proliferative_evidence={prolif}; boundary_evidence={boundary}"
    )
    path_line = f"选中路径：{step} → Grade {grade}（rule={rule}）"
    if evidence_limited:
        concl = (
            f"DR Grade {grade}，referable_dr={'true' if grade >= 2 else 'false'}，"
            f"evidence_tier=supervised_evidence_limited。当前可见病灶证据不足以独立支持该 grade，"
            f"按训练标签输出，不得把 unknown 病灶或弱证据写成 PDR/重度证据。"
        )
    else:
        concl = (
            f"DR Grade {grade}，referable_dr={'true' if grade >= 2 else 'false'}，"
            f"evidence_tier=direct。"
        )
    payload = {
        "task": "L4_unified_lesion_cot_v2",
        "dr_grade": grade,
        "referable_dr": grade >= 2,
        "MA": states.get("MA", "unknown"),
        "HE": states.get("HE", "unknown"),
        "EX": states.get("EX", "unknown"),
        "SE": states.get("SE", "unknown"),
        "IRMA": states.get("IRMA", "unknown"),
        "NV": states.get("NV", "unknown"),
        "evidence_strong": groups["strong_present"],
        "evidence_weak": groups["weak_present"],
        "burden": burden,
        "proliferative_evidence": prolif,
        "boundary_evidence": boundary,
        "selected_step": step,
        "decision_rule": rule,
        "evidence_limited": evidence_limited,
    }
    return (
        "【逐项核查】按 MA/HE/EX/SE/IRMA/NV 顺序逐项核查：\n"
        + insp
        + "\n\n【证据强度归类】"
        + grp
        + "\n\n【病灶负担】"
        + burden_line
        + "\n\n【分级路径】"
        + GRADING_RULES_TEXT
        + " "
        + path_line
        + "\n\n【结论】"
        + concl
        + "\n\n【JSON】\n"
        + json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    )


def sft(system: str, user: str, assistant: str, image: str, meta: dict[str, Any]) -> dict[str, Any]:
    if image.startswith("data/"):
        image = image[len("data/"):]
    return {
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": "<image>\n" + user},
            {"role": "assistant", "content": assistant},
        ],
        "images": [image],
        "meta": meta,
    }


SYSTEM_PROMPT = (
    "你是眼底 DR 分级助手。必须按 MA/HE/EX/SE/IRMA/NV 顺序逐项核查每个病灶（present_strong/"
    "present_weak/absent/unknown/template），再聚合为 NPDR_burden 和 proliferative/boundary"
    " 证据，最后按 Step1..Step6 有序规则输出 DR Grade 0-4 和固定 JSON。NV 是 Grade4/PDR 的"
    "直接证据；IRMA 仅作为 Grade3 边界；不能把 HE/EX/SE 编造成 PDR 证据。"
)
USER_PROMPT = (
    "请按【逐项核查】→【证据强度归类】→【病灶负担】→【分级路径】→【结论】→【JSON】的顺序输出。"
)


def build_from_validated(record: dict[str, Any], split: str) -> dict[str, Any] | None:
    grade = record.get("grade")
    if not isinstance(grade, int) or grade < 0 or grade > 4:
        return None
    if not record.get("usable_for", {}).get("L4"):
        return None
    states: dict[str, str] = {}
    lesion_meta: dict[str, dict[str, Any]] = {}
    tiers: dict[str, str] = {}
    verdicts: dict[str, str] = {}
    for key in LESIONS:
        state, d = lesion_state_from_validated(record, key)
        states[key] = state
        lesion_meta[key] = d if isinstance(d, dict) else {}
        tier = classify_evidence_tier(state, lesion_meta[key])
        tiers[key] = tier
        verdicts[key] = verdict_line(key, state, tier, lesion_meta[key])

    burden = compute_burden(states, tiers, lesion_meta)
    prolif = proliferative_evidence(states, tiers)
    boundary = boundary_evidence(states, tiers)
    step, rule, evidence_limited = select_step(grade, states, tiers, burden, prolif, boundary)
    groups = build_evidence_groups(states, tiers)

    answer = answer_text(grade, verdicts, groups, burden, prolif, boundary, step, rule, evidence_limited, states)
    return sft(
        SYSTEM_PROMPT,
        USER_PROMPT,
        answer,
        record["image_path"],
        {
            "record_id": record.get("record_id"),
            "task": "L4_unified_lesion_cot_v2",
            "split": split,
            "dr_grade": grade,
            "source_file": VALIDATED.name,
            "evidence_limited": evidence_limited,
            "decision_rule": rule,
            "selected_step": step,
            "burden": burden,
            "proliferative_evidence": prolif,
            "boundary_evidence": boundary,
        },
    )


def build_from_grade4_aug(row: dict[str, Any], split: str) -> dict[str, Any] | None:
    """Grade4 augmented samples 没有 retsam 形态字段。
    把它们的 lesion 状态映射到 weak_present / unknown，evidence_limited 大概率 True。
    """
    obj = stage2.extract_json(row["messages"][-1]["content"])
    grade = int(obj.get("dr_grade", -1))
    if grade not in {3, 4}:
        return None
    states = {k: "unknown" for k in LESIONS}
    for key in LESIONS:
        if key in obj:
            states[key] = norm_state(obj.get(key))
    for key in obj.get("evidence") or []:
        if key in states and states[key] == "unknown":
            states[key] = "present"
    # NV present 在 grade=3 与规则矛盾，跳过
    if grade == 3 and states.get("NV") == "present":
        return None
    # aug 来源没有 retsam 形态：present 全部记 weak_present，NV present 只能是 strong_mask 等价
    lesion_meta: dict[str, dict[str, Any]] = {k: {"source": "fundus_l4_grade4_augmented"} for k in LESIONS}
    if states.get("NV") == "present":
        # G4 NV present 直接证据：当 strong 处理（数据来源是早期 augmented，应当尊重）
        lesion_meta["NV"]["source"] = "fgadr_lesion_only_sft_v3"
    tiers: dict[str, str] = {}
    verdicts: dict[str, str] = {}
    for key in LESIONS:
        tiers[key] = classify_evidence_tier(states[key], lesion_meta[key])
        verdicts[key] = verdict_line(key, states[key], tiers[key], lesion_meta[key])

    burden = compute_burden(states, tiers, lesion_meta)
    prolif = proliferative_evidence(states, tiers)
    boundary = boundary_evidence(states, tiers)
    step, rule, evidence_limited = select_step(grade, states, tiers, burden, prolif, boundary)
    groups = build_evidence_groups(states, tiers)

    answer = answer_text(grade, verdicts, groups, burden, prolif, boundary, step, rule, evidence_limited, states)
    meta = dict(row.get("meta", {}))
    meta.update(
        {
            "task": "L4_unified_lesion_cot_v2",
            "split": split,
            "dr_grade": grade,
            "source_file": GRADE4_AUG.name,
            "evidence_limited": evidence_limited,
            "decision_rule": rule,
            "selected_step": step,
            "burden": burden,
            "proliferative_evidence": prolif,
            "boundary_evidence": boundary,
        }
    )
    return sft(SYSTEM_PROMPT, USER_PROMPT, answer, row["images"][0], meta)


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

    if GRADE4_AUG.exists():
        for row in stage2.stable_shuffle(read_jsonl(GRADE4_AUG), "l4_unified_v2_grade4_aug"):
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
        out.extend(stage2.take(stage2.stable_shuffle(by_grade[grade], f"l4_unified_v2_g{grade}"), budgets[grade]))
    return stage2.stable_shuffle(out, "l4_unified_v2_train_final")


def balance_holdout(rows: list[dict[str, Any]], n: int) -> list[dict[str, Any]]:
    by_grade: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_grade[int(row["meta"]["dr_grade"])].append(row)
    base = n // 5
    rem = n % 5
    out = []
    for grade in range(5):
        out.extend(stage2.take(stage2.stable_shuffle(by_grade[grade], f"l4_unified_v2_holdout_g{grade}"), base + (1 if grade < rem else 0)))
    return stage2.stable_shuffle(out, "l4_unified_v2_holdout_final")


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grades = Counter()
    rules = Counter()
    steps = Counter()
    burdens = Counter()
    evidence_limited = Counter()
    tier_counts = Counter()
    source_files = Counter()
    missing = 0
    for row in rows:
        meta = row.get("meta", {})
        grades[str(meta.get("dr_grade"))] += 1
        rules[meta.get("decision_rule", "unknown")] += 1
        steps[meta.get("selected_step", "unknown")] += 1
        burdens[meta.get("burden", "unknown")] += 1
        evidence_limited[str(bool(meta.get("evidence_limited")))] += 1
        source_files[meta.get("source_file", "unknown")] += 1
        obj = stage2.extract_json(row["messages"][-1]["content"])
        for key in LESIONS:
            tier_counts[(key, obj.get(key, "unknown"))] += 1
        for image in row.get("images", []):
            if not Path("data", image).exists():
                missing += 1
    return {
        "n": len(rows),
        "grades": dict(sorted(grades.items())),
        "decision_rules": dict(rules),
        "selected_step": dict(steps),
        "burden": dict(burdens),
        "evidence_limited": dict(evidence_limited),
        "lesion_states": {str(k): v for k, v in sorted(tier_counts.items(), key=lambda x: str(x[0]))},
        "source_files": dict(source_files),
        "missing_images": missing,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-n", type=int, default=10000)
    parser.add_argument("--holdout-n", type=int, default=150)
    parser.add_argument("--train-output", default="data/annotation/fundus_l4_unified_lesion_cot_v2_sft.jsonl")
    parser.add_argument("--holdout-output", default="data/annotation/fundus_l4_unified_lesion_cot_v2_holdout_sft.jsonl")
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
    stats_path = BASE / "fundus_l4_unified_lesion_cot_v2_stats.json"
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
