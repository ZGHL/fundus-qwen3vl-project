#!/usr/bin/env python3
"""Build L4 v5: DR grading CoT with quadrant LOCATION (#1) + sequential conditional reasoning (#3).

Changes from L4 v4:
  - [Lesion Audit]: HE/EX/SE include quadrant info from quadrant_index when available,
    e.g. `HE: strong(many,large,4Q,retsam)`.
  - [Reasoning]: rewritten as sequential conditional flow (Step 1 → Step 2 → ...),
    each step asks one question, answers it, and decides whether to continue.
    Includes explicit 4-2-1 ETDRS check when quadrant data is available.
  - Decision logic (Step1-6) UNCHANGED — we still use v3's select_step verbatim.
  - JSON: gain `quadrant_summary` field showing where HE/EX/SE were observed.

Output:
  data/annotation_v4/fundus_l4_v5_train_sft.jsonl
  data/annotation_v4/fundus_l4_v5_val_sft.jsonl
  data/annotation_v4/fundus_l4_v5_stats.json
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "fundus"))
from build_fundus_sft import hbucket, read_jsonl, sft, write_jsonl  # noqa: E402

VALIDATED = Path("/home/aim_lab/LLaMA-Factory/data/fundus_validated/validated_clean.jsonl")
QUAD_INDEX = Path("/home/aim_lab/LLaMA-Factory/data/fundus_validated/quadrant_index.jsonl")
OUT_DIR = Path("/home/aim_lab/LLaMA-Factory/data/annotation_v4")
LESIONS = ("MA", "HE", "EX", "SE", "IRMA", "NV")
VAL_PCT = 20

# Same step budgets as v4
V5_STEP_BUDGETS = {
    "Step1":  None,  "Step2a": 1100, "Step2b": 0,
    "Step3a": 200,   "Step3b": 700,  "Step3c": 500,
    "Step4":  2400,  "Step4b": 100,
    "Step5a": 700,   "Step5b": 400,  "Step5c": 100,
    "Step6":  2100,  "Step6b": 0,
}

Q_NAMES = {
    "ST": "superior-temporal",
    "SN": "superior-nasal",
    "IT": "inferior-temporal",
    "IN": "inferior-nasal",
}


# ---------------------------- v3 logic import ----------------------------

def _load_v2():
    p = Path("/home/aim_lab/LLaMA-Factory/scripts/fundus/build_l4_unified_lesion_cot_v2.py")
    spec = importlib.util.spec_from_file_location("build_l4_v2", p)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["build_l4_v2"] = mod
    spec.loader.exec_module(mod)
    return mod

_v2 = _load_v2()
classify_evidence_tier = _v2.classify_evidence_tier
compute_burden = _v2.compute_burden
proliferative_evidence = _v2.proliferative_evidence
boundary_evidence = _v2.boundary_evidence
select_step = _v2.select_step
lesion_state_from_validated = _v2.lesion_state_from_validated


# ---------------------------- split + quadrant ----------------------------

def assign_splits(records):
    eval_iids = set()
    for r in records:
        if r.get("dataset") == "idrid" and r.get("split") == "test":
            eval_iids.add(r["image_id"])
        if r.get("dataset") == "ddr_seg" and r.get("split") in {"valid", "test"}:
            eval_iids.add(r["image_id"])
    iid_split = {}
    for r in records:
        iid = r["image_id"]
        if iid in eval_iids: iid_split[iid] = "eval"
        elif iid not in iid_split:
            iid_split[iid] = "val" if hbucket(iid) < VAL_PCT else "train"
    return iid_split


def load_quadrant_index():
    out = {}
    if not QUAD_INDEX.exists(): return out
    for row in read_jsonl(QUAD_INDEX):
        out[(row["dataset"], row["image_id"])] = row["quadrants"]
    return out


def quadrant_summary_short(quad):
    """Short qualitative summary: '4Q', 'inf-temp', 'sup-only', etc."""
    if not quad or quad.get("total", 0) == 0:
        return None
    total = quad["total"]
    q = {k: quad.get(k, 0) for k in ("ST", "SN", "IT", "IN")}
    non_zero = sum(1 for v in q.values() if v > 0)
    if non_zero == 4 and min(q.values()) >= max(1, total // 6):
        return "4Q"
    if non_zero == 1:
        only = max(q, key=q.get)
        return Q_NAMES[only].replace("-", "-")
    top = sorted(q.items(), key=lambda x: -x[1])
    top2 = [k for k, v in top if v > 0][:2]
    if len(top2) == 1:
        return Q_NAMES[top2[0]]
    return "+".join(k for k in top2)


def quadrant_summary_long(quad):
    """Full prose for [Reasoning]."""
    if not quad or quad.get("total", 0) == 0:
        return None
    total = quad["total"]
    q = {k: quad.get(k, 0) for k in ("ST", "SN", "IT", "IN")}
    non_zero = sum(1 for v in q.values() if v > 0)
    if non_zero == 4 and min(q.values()) >= max(1, total // 6):
        return f"{total} total, distributed throughout all 4 quadrants"
    if non_zero == 1:
        only = max(q, key=q.get)
        return f"{total} total, concentrated in {Q_NAMES[only]}"
    nums = ", ".join(f"{Q_NAMES[k]}:{v}" for k, v in q.items() if v > 0)
    return f"{total} total ({nums})"


# ---------------------------- short verdict line (with location) ----------------------------

SRC_TAG = {
    "validated_retsam":          "retsam",
    "strong_mask_stage1_easy":   "strong_mask",
    "fgadr_lesion_only_sft_v3":  "strong_mask",
    "grade_rule_override":       "grade_rule",
    "grade_rule":                "grade_rule",
    "cleaning_rule":             "cleaning_rule",
}


def short_verdict(k, state, tier, lesion, quad_short):
    if state == "absent":           return f"{k}: absent"
    if state == "unknown":          return f"{k}: unknown"
    if state == "template_only":    return f"{k}: template_only"
    if state == "possible_by_grade_template":
        return f"{k}: grade_template_only"
    src = SRC_TAG.get(lesion.get("source", "unknown"), lesion.get("source", "unknown"))
    cb = lesion.get("count_bucket") or "?"
    ab = lesion.get("area_bucket") or "?"
    tier_word = "strong" if tier == "strong_present" else "weak"
    if quad_short:
        return f"{k}: {tier_word}({cb},{ab},{quad_short},{src})"
    return f"{k}: {tier_word}({cb},{ab},{src})"


# ---------------------------- sequential conditional reasoning ----------------------------

GRADE_DESC = {
    0: "no DR", 1: "mild NPDR", 2: "moderate NPDR",
    3: "severe NPDR", 4: "proliferative DR",
}

EL_STEPS = {"Step2a", "Step2b", "Step3c", "Step4b", "Step5a", "Step5c", "Step6b"}

# v5.1: rule name → short code (enum) for JSON
RULE_CODE = {
    "nv_present_grade4_pdr":                                "nv_g4",
    "grade4_label_with_possible_nv_template":               "g4_template",
    "supervised_grade4_without_direct_nv_evidence_limited": "g4_el",
    "irma_without_nv_grade3_boundary":                      "irma_g3",
    "heavy_nonproliferative_without_nv_grade3":             "heavy_g3",
    "supervised_grade3_evidence_limited":                   "g3_el",
    "nonproliferative_lesions_without_severe_boundary_grade2": "npdr_g2",
    "supervised_grade2_evidence_limited":                   "g2_el",
    "mild_or_template_ma_grade1":                            "g1_template",
    "ma_only_grade1":                                        "ma_g1",
    "supervised_grade1_evidence_limited":                    "g1_el",
    "no_reliable_dr_lesion_grade0":                          "g0",
    "supervised_grade0_unexpected_lesion":                   "g0_unexpected",
}


def quadrants_count_to_enum(quad):
    """Map quadrant counts → enum: 4Q | ST_only | ST_pred | ST+IT | absent etc."""
    if not quad or quad.get("total", 0) == 0:
        return "absent"
    total = quad["total"]
    q = {k: quad.get(k, 0) for k in ("ST", "SN", "IT", "IN")}
    non_zero = sum(1 for v in q.values() if v > 0)
    if non_zero == 4 and min(q.values()) >= max(1, total // 6):
        return "4Q"
    if non_zero == 1:
        return max(q, key=q.get) + "_only"
    sorted_q = sorted(q.items(), key=lambda x: -x[1])
    top = [k for k, v in sorted_q if v >= max(1, total // 4)]
    if len(top) == 1: return f"{top[0]}_pred"
    if len(top) == 2: return f"{top[0]}+{top[1]}"
    return f"{top[0]}+{top[1]}+{top[2]}"


def sequential_reasoning(states, tiers, lesion_meta, burden, prolif, boundary,
                       grade, step, quad_data):
    """v5 #3: write [Reasoning] as a sequential conditional walk."""
    lines = []

    # === Step 1: Proliferative check ===
    nv_state = states.get("NV")
    nv_tier = tiers.get("NV")
    if nv_tier == "strong_present":
        lines.append("Step 1 (Proliferative check): NV present (strong evidence) "
                     "→ direct PDR → exit at Grade 4.")
        lines.append(f"\nTherefore: {GRADE_DESC[grade]} → Grade {grade}.")
        return "\n".join(lines)
    if nv_state == "possible_by_grade_template":
        lines.append("Step 1 (Proliferative check): NV indicated by grade template "
                     "only; no direct visual confirmation. Tentatively flag as "
                     "supervised Grade 4 (evidence-limited).")
    else:
        lines.append("Step 1 (Proliferative check): NV absent / not observed "
                     "→ no PDR. Continue.")

    # === Step 2: ETDRS 4-2-1 check (severe NPDR boundary) ===
    if quad_data:
        # Apply 4-2-1 ETDRS rule: ≥1 HE in all 4 quadrants → severe NPDR criterion 1
        he_q = quad_data.get("HE", {})
        he_n4 = sum(1 for k in ("ST","SN","IT","IN") if he_q.get(k, 0) >= 1)
        # Severe NPDR criterion 2: venous beading in ≥2 quadrants (we don't track) — skip
        # Severe NPDR criterion 3: IRMA in ≥1 quadrant (already in boundary_evidence)
        if he_n4 == 4:
            lines.append(f"Step 2 (ETDRS 4-2-1 check): HE present in all 4 quadrants "
                         f"({quadrant_summary_long(he_q)}) "
                         f"→ severe-NPDR criterion satisfied.")
        else:
            lines.append(f"Step 2 (ETDRS 4-2-1 check): HE distributed in {he_n4} quadrants "
                         f"({quadrant_summary_long(he_q) or 'none'}). "
                         f"4-2-1 rule not fully met.")

    # === Step 3: IRMA boundary ===
    if tiers.get("IRMA") == "strong_present":
        lines.append("Step 3 (Boundary check): IRMA present → severe-NPDR boundary "
                     "marker (Grade 3).")
    else:
        lines.append("Step 3 (Boundary check): IRMA absent / unknown → no severe-boundary marker.")

    # === Step 4: NPDR burden ===
    npdr_present = [k for k in ("HE","EX","SE","MA")
                    if tiers.get(k) in {"strong_present", "weak_present"}]
    if npdr_present:
        parts = []
        for k in npdr_present:
            t_short = "strong" if tiers[k] == "strong_present" else "weak"
            q_short = None
            if quad_data and k in quad_data:
                q_short = quadrant_summary_short(quad_data[k])
            if q_short:
                parts.append(f"{k}={t_short}@{q_short}")
            else:
                parts.append(f"{k}={t_short}")
        lines.append(f"Step 4 (NPDR burden): observed {', '.join(parts)} → burden={burden}.")
    else:
        lines.append("Step 4 (NPDR burden): no NPDR lesions visible → burden=none.")

    # === Step 5: MA template-only branch ===
    if states.get("MA") == "template_only" and burden == "none":
        lines.append("Step 5 (G1 template): MA suggested by Grade-1 template only; no visual MA detected.")

    # === Final ===
    el_phrase = " (label-driven; visible evidence is insufficient)" if step in EL_STEPS else ""
    lines.append(f"\nTherefore: {GRADE_DESC[grade]} → Grade {grade}{el_phrase}.")
    return "\n".join(lines)


def render_assistant(grade, states, tiers, lesion_meta, burden, prolif, boundary,
                     step, rule, evidence_limited, quad_data):
    # Build [Lesion Audit] with quadrant info
    audit_parts = []
    for k in LESIONS:
        q_short = None
        if quad_data and k in quad_data and states[k] == "present":
            q_short = quadrant_summary_short(quad_data[k])
        audit_parts.append(short_verdict(k, states[k], tiers[k], lesion_meta[k], q_short))
    audit = " | ".join(audit_parts)

    reasoning = sequential_reasoning(
        states, tiers, lesion_meta, burden, prolif, boundary, grade, step, quad_data)

    el_str = "true" if evidence_limited else "false"
    et = "supervised_evidence_limited" if evidence_limited else "direct"
    ref = "true" if grade >= 2 else "false"

    # v5.1: quadrant per-lesion enum (4Q / ST_only / ST+IT / absent)
    q_he = quadrants_count_to_enum(quad_data.get("HE") if quad_data else None) if quad_data else None
    q_ex = quadrants_count_to_enum(quad_data.get("EX") if quad_data else None) if quad_data else None
    q_se = quadrants_count_to_enum(quad_data.get("SE") if quad_data else None) if quad_data else None
    rule_code = RULE_CODE.get(rule, rule)

    payload = {
        "grade": grade,
        "step": step,
        "rule": rule_code,
        "el": evidence_limited,
        "q_HE": q_he,
        "q_EX": q_ex,
        "q_SE": q_se,
    }

    # v5.2: Findings (bulleted per-lesion) / Impression (bulleted reasoning) / Result (pipe-separated)
    # [Findings] — each lesion on its own line as bullet
    findings_lines = []
    for k in LESIONS:
        q_short = None
        if quad_data and k in quad_data and states[k] == "present":
            q_short = quadrant_summary_short(quad_data[k])
        findings_lines.append("- " + short_verdict(k, states[k], tiers[k], lesion_meta[k], q_short))
    findings = "\n".join(findings_lines)

    # [Impression] — same logic as sequential_reasoning but as compact bullets
    imp_lines = []
    if tiers.get("NV") == "strong_present":
        imp_lines.append("- NV present (strong) → direct PDR (Grade 4).")
    elif states.get("NV") == "possible_by_grade_template":
        imp_lines.append("- NV indicated by grade template only; supervised G4 (evidence-limited).")
    else:
        imp_lines.append("- NV absent → no PDR.")

    if quad_data:
        he_q = quad_data.get("HE", {})
        he_n4 = sum(1 for k in ("ST","SN","IT","IN") if he_q.get(k, 0) >= 1)
        if he_n4 == 4:
            imp_lines.append(f"- ETDRS 4-2-1: HE in all 4 quadrants → severe-NPDR criterion met.")
        elif he_n4 > 0:
            imp_lines.append(f"- ETDRS 4-2-1: HE in {he_n4}/4 quadrants → criterion not met.")

    if tiers.get("IRMA") == "strong_present":
        imp_lines.append("- IRMA present → severe-NPDR boundary marker.")
    else:
        imp_lines.append("- IRMA absent / unknown → no severe-boundary marker.")

    npdr_present = [k for k in ("HE","EX","SE","MA")
                    if tiers.get(k) in {"strong_present", "weak_present"}]
    if npdr_present:
        parts = [f"{k}={('strong' if tiers[k]=='strong_present' else 'weak')}" for k in npdr_present]
        imp_lines.append(f"- NPDR burden: {burden} ({', '.join(parts)}).")
    else:
        imp_lines.append("- NPDR burden: none.")

    if states.get("MA") == "template_only" and burden == "none":
        imp_lines.append("- MA suggested by Grade-1 template only.")

    el_note = " (label-driven, evidence-limited)" if evidence_limited else ""
    imp_lines.append(f"- Final: {GRADE_DESC[grade]} → Grade {grade}{el_note}.")
    impression = "\n".join(imp_lines)

    # [Result] — pipe-separated, no JSON
    result = (f"grade={grade} | step={step} | rule={rule_code} | el={el_str} | "
              f"q_HE={q_he} | q_EX={q_ex} | q_SE={q_se}")

    return (
        f"[Findings]\n{findings}\n\n"
        f"[Impression]\n{impression}\n\n"
        f"[Result] {result}"
    )


SYSTEM_PROMPT = (
    "You are a fundus DR grading assistant. First audit each of MA/HE/EX/SE/IRMA/NV "
    "with their spatial distribution. Then walk through a step-by-step clinical "
    "reasoning: (1) check NV for PDR; (2) apply ETDRS 4-2-1 rule with quadrant counts; "
    "(3) check IRMA for severe-NPDR boundary; (4) assess NPDR burden. "
    "Output the chosen Step rule and the JSON payload. "
    "Do not fabricate evidence — when an unknown lesion appears, say unknown."
)

USER_PROMPT = (
    "Analyse this fundus image and produce a DR grade. "
    "Output in the order: [Lesion Audit] -> [Reasoning] -> [Burden] -> [Decision] "
    "-> [Conclusion] -> [JSON]."
)


def build_item(record, split, quad_idx):
    grade = record.get("grade")
    if not isinstance(grade, int) or grade < 0 or grade > 4: return None
    if not record.get("usable_for", {}).get("L4"): return None

    states, lesion_meta, tiers = {}, {}, {}
    for k in LESIONS:
        state, d = lesion_state_from_validated(record, k)
        states[k] = state
        lesion_meta[k] = d if isinstance(d, dict) else {}
        tiers[k] = classify_evidence_tier(state, lesion_meta[k])

    burden = compute_burden(states, tiers, lesion_meta)
    prolif = proliferative_evidence(states, tiers)
    boundary = boundary_evidence(states, tiers)
    step, rule, el = select_step(grade, states, tiers, burden, prolif, boundary)

    quad_key = (record.get("dataset"), record.get("image_id"))
    quad_data = quad_idx.get(quad_key)

    assistant = render_assistant(
        grade, states, tiers, lesion_meta, burden, prolif, boundary,
        step, rule, el, quad_data)

    meta = {
        "record_id": record["record_id"],
        "image_id": record["image_id"],
        "dataset": record["dataset"],
        "task": "L4_dr_grading_v5",
        "split": split,
        "dr_grade": grade,
        "decision_rule": rule,
        "selected_step": step,
        "burden": burden,
        "proliferative_evidence": prolif,
        "boundary_evidence": boundary,
        "evidence_limited": el,
        "has_quadrant_data": quad_data is not None,
    }
    return sft(SYSTEM_PROMPT, USER_PROMPT, assistant, record["image_path"], meta)


def apply_budgets(items, budgets, seed):
    rng = random.Random(seed)
    by_step = defaultdict(list)
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
            kept.extend(pool); summary[step] = {"available": len(pool), "kept": len(pool), "action": "use_all"}
        else:
            kept.extend(pool[:budget]); summary[step] = {"available": len(pool), "kept": budget, "action": f"capped_{budget}"}
    for s in sorted(summary):
        ss = summary[s]
        print(f"  {s:<7}  avail={ss['available']:>5}  kept={ss['kept']:>5}  ({ss['action']})")
    return kept


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

    quad_idx = load_quadrant_index()
    print(f"[quad] loaded {len(quad_idx)} entries")

    train_pool = [it for r in train_recs if (it := build_item(r, "train", quad_idx))]
    val_pool = [it for r in val_recs if (it := build_item(r, "val", quad_idx))]

    print("=== applying budgets ===")
    train_items = apply_budgets(train_pool, V5_STEP_BUDGETS, args.seed)
    val_items = val_pool

    rng = random.Random(args.seed + 1)
    rng.shuffle(train_items)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    train_path = args.out_dir / "fundus_l4_v5_train_sft.jsonl"
    val_path = args.out_dir / "fundus_l4_v5_val_sft.jsonl"
    stats_path = args.out_dir / "fundus_l4_v5_stats.json"

    if not args.dry_run:
        write_jsonl(train_path, train_items)
        write_jsonl(val_path, val_items)

    grades_train = Counter(it["meta"]["dr_grade"] for it in train_items)
    with_quad = sum(1 for it in train_items if it["meta"].get("has_quadrant_data"))
    el_train = Counter(it["meta"]["evidence_limited"] for it in train_items)

    stats = {
        "v5_changes": ["sequential_conditional_reasoning", "quadrant_location", "etdrs_check"],
        "v5_step_budgets": V5_STEP_BUDGETS,
        "train_total": len(train_items),
        "val_total": len(val_items),
        "train_with_quadrant_data": with_quad,
        "train_grades": dict(sorted(grades_train.items())),
        "train_evidence_limited": {str(k): v for k, v in el_train.items()},
    }
    if not args.dry_run:
        stats_path.write_text(json.dumps(stats, indent=2, ensure_ascii=False))

    print()
    print("=== L4 v5 build summary ===")
    print(f"train: {len(train_items)}  val: {len(val_items)}")
    print(f"train with quadrant data: {with_quad} ({with_quad/max(len(train_items),1)*100:.1f}%)")
    print(f"train grade dist: {dict(sorted(grades_train.items()))}")
    print(f"train EL: {dict(el_train)}")


if __name__ == "__main__":
    main()
