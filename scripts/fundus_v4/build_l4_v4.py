#!/usr/bin/env python3
"""Build L4 v4: full-pool, English, concise CoT for DR grading.

Design (per v4 spec, user-confirmed):
- Same image_id-level 8:2 split as L2/L3 v4
- Reuses v3 logic verbatim for: evidence-tier classification, burden score,
  step selection (Step1..Step6b), evidence_limited flag, Step2b/Step6b drop.
  These functions are imported from build_l4_unified_lesion_cot_v2.py.
- English concise CoT, ~290 tok median (vs v3 ~835 tok):
    [Lesion Audit]  — 6 verdict lines in one row (| separated)
    [Reasoning]     — doctor-narrative: lesion → grade implication
    [Burden]        — NPDR / PDR / boundary one-liner
    [Decision]      — selected step + rule (no rule recital)
    [Conclusion]    — grade + referable + evidence_tier
    [JSON]          — 4 fields: grade, step, rule, el
- Sample budget: ~8k after step budgets (mixed-adapter balance with L2/L3)

Output:
  data/annotation_v4/fundus_l4_v4_train_sft.jsonl   (~8k)
  data/annotation_v4/fundus_l4_v4_val_sft.jsonl     (natural distribution)
  data/annotation_v4/fundus_l4_v4_stats.json
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "fundus"))
from build_fundus_sft import hbucket, read_jsonl, sft, write_jsonl  # noqa: E402

VALIDATED = Path("/home/aim_lab/LLaMA-Factory/data/fundus_validated/validated_clean.jsonl")
GRADE4_AUG = Path("/home/aim_lab/LLaMA-Factory/data/annotation/fundus_l4_grade4_augmented_train_sft.jsonl")
OUT_DIR = Path("/home/aim_lab/LLaMA-Factory/data/annotation_v4")
LESIONS = ("MA", "HE", "EX", "SE", "IRMA", "NV")
VAL_PCT = 20

# v4 step budgets — drops Step2b/6b (validated in v3 to give QWK +0.044)
# Tighter caps to keep L4 ~8k for mixed training balance.
V4_STEP_BUDGETS: dict[str, int | None] = {
    "Step1":  None,   # all NV-direct
    "Step2a": 900,    # G4 NV template
    "Step2b": 0,      # DROPPED (validated)
    "Step3a": 200,    # G3 IRMA
    "Step3b": 500,    # G3 heavy NPDR
    "Step3c": 400,    # G3 EL
    "Step4":  2400,   # G2 main
    "Step4b": 100,    # G2 EL
    "Step5a": 700,    # G1 template
    "Step5b": 400,    # G1 MA-only
    "Step5c": 100,    # G1 EL
    "Step6":  2100,   # G0
    "Step6b": 0,      # DROPPED
}


# ---------------------------- import v3 logic verbatim ----------------------------

def _load_v3():
    """Load build_l4_unified_lesion_cot_v2.py for its verdict-tier / burden / step
    selection logic. We reuse classify_evidence_tier, compute_burden, etc."""
    p = Path("/home/aim_lab/LLaMA-Factory/scripts/fundus/build_l4_unified_lesion_cot_v2.py")
    spec = importlib.util.spec_from_file_location("build_l4_v2", p)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["build_l4_v2"] = mod
    spec.loader.exec_module(mod)
    return mod


_v2 = _load_v3()
classify_evidence_tier = _v2.classify_evidence_tier
compute_burden = _v2.compute_burden
proliferative_evidence = _v2.proliferative_evidence
boundary_evidence = _v2.boundary_evidence
select_step = _v2.select_step
lesion_state_from_validated = _v2.lesion_state_from_validated
build_evidence_groups = _v2.build_evidence_groups


# ---------------------------- split (shared logic) ----------------------------

def assign_splits(records: list[dict]) -> dict[str, str]:
    eval_iids = set()
    for r in records:
        if r.get("dataset") == "idrid" and r.get("split") == "test":
            eval_iids.add(r["image_id"])
        if r.get("dataset") == "ddr_seg" and r.get("split") in {"valid", "test"}:
            eval_iids.add(r["image_id"])
    iid_split: dict[str, str] = {}
    for r in records:
        iid = r["image_id"]
        if iid in eval_iids:
            iid_split[iid] = "eval"
        elif iid not in iid_split:
            iid_split[iid] = "val" if hbucket(iid) < VAL_PCT else "train"
    return iid_split


# ---------------------------- English render helpers ----------------------------

SRC_TAG = {
    "validated_retsam": "retsam",
    "strong_mask_stage1_easy": "strong_mask",
    "fgadr_lesion_only_sft_v3": "strong_mask",
    "grade_rule_override": "grade_rule",
    "grade_rule": "grade_rule",
    "cleaning_rule": "cleaning_rule",
}


def short_verdict(k: str, state: str, tier: str, lesion: dict) -> str:
    """One-line concise verdict per lesion for the [Lesion Audit] row.
    Examples:
      MA: unknown
      HE: strong(many,large,retsam)
      EX: weak(some,small,retsam)
      SE: absent
      IRMA: unknown
      NV: present(grade_template)
    """
    if state == "absent":
        return f"{k}: absent"
    if state == "unknown":
        return f"{k}: unknown"
    if state == "template_only":
        return f"{k}: template_only"
    if state == "possible_by_grade_template":
        return f"{k}: grade_template_only"
    src = SRC_TAG.get(lesion.get("source", "unknown"), lesion.get("source", "unknown"))
    if tier == "strong_present":
        cb = lesion.get("count_bucket") or "?"
        ab = lesion.get("area_bucket") or "?"
        return f"{k}: strong({cb},{ab},{src})"
    if tier == "weak_present":
        cb = lesion.get("count_bucket") or "?"
        ab = lesion.get("area_bucket") or "?"
        return f"{k}: weak({cb},{ab},{src})"
    return f"{k}: {state}"


def doctor_reasoning(states: dict[str, str], tiers: dict[str, str], burden: str,
                     prolif: str, boundary: str, grade: int, step: str) -> str:
    """Doctor-narrative reasoning: walk through lesions and connect to grade.

    Deterministic template — model learns to imitate, not induce.
    """
    lines = []

    # NV (proliferative)
    nv_state = states.get("NV")
    nv_tier = tiers.get("NV")
    if nv_tier == "strong_present":
        lines.append("- NV present (strong evidence) → direct proliferative finding (Grade 4 / PDR).")
    elif nv_state == "possible_by_grade_template":
        lines.append("- NV indicated by grade template only, no direct visual confirmation → supervised G4 (evidence_limited).")
    else:
        lines.append("- NV absent / not observed → no proliferative (PDR) evidence.")

    # IRMA (G3 boundary)
    irma_tier = tiers.get("IRMA")
    if irma_tier == "strong_present":
        lines.append("- IRMA present → severe-NPDR boundary marker (Grade 3).")
    else:
        lines.append("- IRMA absent / unknown → no severe-boundary evidence.")

    # NPDR lesions (HE / EX / SE / MA)
    npdr_present = [k for k in ("HE", "EX", "SE", "MA")
                    if tiers.get(k) in {"strong_present", "weak_present"}]
    if npdr_present:
        descriptors = ", ".join(
            f"{k}={tiers.get(k).replace('_present', '')}" for k in npdr_present
        )
        lines.append(f"- NPDR lesions ({descriptors}) → burden={burden}.")
    else:
        lines.append("- No NPDR lesions visible → burden=none.")

    # MA template only path
    if states.get("MA") == "template_only" and burden == "none":
        lines.append("- MA suggested by Grade-1 template only, no visual confirmation.")

    # Final implication
    el_phrase = " (label-driven; evidence is insufficient)" if step in {
        "Step2a", "Step2b", "Step3c", "Step4b", "Step5a", "Step5c", "Step6b"
    } else ""
    grade_desc = {
        0: "no DR",
        1: "mild NPDR",
        2: "moderate NPDR",
        3: "severe NPDR",
        4: "proliferative DR",
    }.get(grade, f"DR Grade {grade}")
    lines.append(f"Therefore: {grade_desc} → Grade {grade}{el_phrase}.")
    return "\n".join(lines)


def render_assistant(grade: int, states: dict, tiers: dict, lesion_meta: dict,
                     burden: str, prolif: str, boundary: str,
                     step: str, rule: str, evidence_limited: bool) -> str:
    audit = " | ".join(
        short_verdict(k, states[k], tiers[k], lesion_meta[k]) for k in LESIONS
    )
    reasoning = doctor_reasoning(states, tiers, burden, prolif, boundary, grade, step)
    el_str = "true" if evidence_limited else "false"
    et = "supervised_evidence_limited" if evidence_limited else "direct"
    ref = "true" if grade >= 2 else "false"
    payload = {"grade": grade, "step": step, "rule": rule, "el": evidence_limited}
    return (
        f"[Lesion Audit] {audit}\n\n"
        f"[Reasoning]\n{reasoning}\n\n"
        f"[Burden] NPDR={burden}; PDR_evidence={prolif}; boundary={boundary}\n\n"
        f"[Decision] {step} -> Grade {grade} (rule={rule})\n\n"
        f"[Conclusion] DR Grade {grade}, referable={ref}, evidence={et}.\n\n"
        f"[JSON] {json.dumps(payload, ensure_ascii=False, separators=(',', ':'))}"
    )


SYSTEM_PROMPT = (
    "You are a fundus DR grading assistant. First audit each of MA/HE/EX/SE/IRMA/NV "
    "in order (strong / weak / absent / unknown / template). Walk through how each "
    "lesion implies a DR severity (NV->PDR/G4; IRMA->severe-boundary/G3; NPDR burden "
    "drives G2-G3). Then output the chosen Step rule and the JSON payload. "
    "Do not fabricate evidence — when an unknown lesion appears, say unknown."
)

USER_PROMPT = (
    "Analyse this fundus image and produce a DR grade. "
    "Output in the order: [Lesion Audit] -> [Reasoning] -> [Burden] -> [Decision] "
    "-> [Conclusion] -> [JSON]."
)


def build_item(record: dict, split: str) -> dict | None:
    grade = record.get("grade")
    if not isinstance(grade, int) or grade < 0 or grade > 4:
        return None
    if not record.get("usable_for", {}).get("L4"):
        return None

    states: dict[str, str] = {}
    lesion_meta: dict[str, dict] = {}
    tiers: dict[str, str] = {}
    for k in LESIONS:
        state, d = lesion_state_from_validated(record, k)
        states[k] = state
        lesion_meta[k] = d if isinstance(d, dict) else {}
        tiers[k] = classify_evidence_tier(state, lesion_meta[k])

    burden = compute_burden(states, tiers, lesion_meta)
    prolif = proliferative_evidence(states, tiers)
    boundary = boundary_evidence(states, tiers)
    step, rule, el = select_step(grade, states, tiers, burden, prolif, boundary)

    assistant = render_assistant(grade, states, tiers, lesion_meta,
                                 burden, prolif, boundary, step, rule, el)
    meta = {
        "record_id": record["record_id"],
        "image_id": record["image_id"],
        "dataset": record["dataset"],
        "task": "L4_dr_grading_v4",
        "split": split,
        "dr_grade": grade,
        "decision_rule": rule,
        "selected_step": step,
        "burden": burden,
        "proliferative_evidence": prolif,
        "boundary_evidence": boundary,
        "evidence_limited": el,
    }
    return sft(SYSTEM_PROMPT, USER_PROMPT, assistant, record["image_path"], meta)


# ---------------------------- step budgeting ----------------------------

def apply_budgets(items: list[dict], budgets: dict[str, int | None], seed: int) -> list[dict]:
    rng = random.Random(seed)
    by_step: dict[str, list] = defaultdict(list)
    for it in items:
        by_step[it["meta"]["selected_step"]].append(it)

    kept = []
    summary = {}
    for step, pool in by_step.items():
        budget = budgets.get(step, None)
        if budget == 0:
            summary[step] = {"available": len(pool), "kept": 0, "action": "DROPPED"}
            continue
        rng.shuffle(pool)
        if budget is None or budget >= len(pool):
            kept.extend(pool)
            summary[step] = {"available": len(pool), "kept": len(pool), "action": "use_all"}
        else:
            kept.extend(pool[:budget])
            summary[step] = {"available": len(pool), "kept": budget, "action": f"capped_{budget}"}
    for step in sorted(summary):
        s = summary[step]
        print(f"  {step:<7}  avail={s['available']:>5}  kept={s['kept']:>5}  ({s['action']})")
    return kept


# ---------------------------- main ----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    records = list(read_jsonl(VALIDATED))
    iid_split = assign_splits(records)
    train_recs = [r for r in records if iid_split[r["image_id"]] == "train"]
    val_recs = [r for r in records if iid_split[r["image_id"]] == "val"]

    train_pool = [it for r in train_recs if (it := build_item(r, "train"))]
    val_pool = [it for r in val_recs if (it := build_item(r, "val"))]

    print("=== train step distribution before budget ===")
    print(Counter(it["meta"]["selected_step"] for it in train_pool))

    print("=== applying budgets ===")
    train_items = apply_budgets(train_pool, V4_STEP_BUDGETS, args.seed)
    # val: no budget, keep natural distribution
    val_items = val_pool

    # Shuffle train so SGD sees mixed steps
    rng = random.Random(args.seed + 1)
    rng.shuffle(train_items)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    train_path = args.out_dir / "fundus_l4_v4_train_sft.jsonl"
    val_path = args.out_dir / "fundus_l4_v4_val_sft.jsonl"
    stats_path = args.out_dir / "fundus_l4_v4_stats.json"

    if not args.dry_run:
        write_jsonl(train_path, train_items)
        write_jsonl(val_path, val_items)

    grades_train = Counter(it["meta"]["dr_grade"] for it in train_items)
    grades_val = Counter(it["meta"]["dr_grade"] for it in val_items)
    el_train = Counter(it["meta"]["evidence_limited"] for it in train_items)

    stats = {
        "v4_step_budgets": V4_STEP_BUDGETS,
        "train_total": len(train_items),
        "val_total": len(val_items),
        "train_grades": dict(sorted(grades_train.items())),
        "val_grades": dict(sorted(grades_val.items())),
        "train_evidence_limited": {str(k): v for k, v in el_train.items()},
        "train_steps": dict(Counter(it["meta"]["selected_step"] for it in train_items)),
        "train_path": str(train_path),
        "val_path": str(val_path),
    }
    if not args.dry_run:
        stats_path.write_text(json.dumps(stats, indent=2, ensure_ascii=False))

    print()
    print("=== L4 v4 build summary ===")
    print(f"train: {len(train_items)} items")
    print(f"val:   {len(val_items)} items")
    print(f"train grade dist: {dict(sorted(grades_train.items()))}")
    print(f"train EL: {dict(el_train)}")


if __name__ == "__main__":
    main()
