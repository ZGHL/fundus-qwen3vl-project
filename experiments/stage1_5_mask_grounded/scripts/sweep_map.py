#!/usr/bin/env python3
"""Sweep the presence->tier DECISION MAP on a FIXED set of predictions (no retraining).

Because tier = map(audit) is computed at scoring time, the abstention policy is a post-hoc
knob: we can re-fit the map at several severe-share thresholds and re-score the SAME
checkpoint predictions to trace the accuracy <-> abstention <-> safety tradeoff, then pick an
operating point. The training run is untouched.

Usage:
  sweep_map.py <test_jsonl> <pred_jsonl> <train_jsonl> [out_md]

<train_jsonl> = stage2_grade_train_sft.jsonl (grounded rows define the fit, exactly as
build_stage2_grade.fit_map). <pred_jsonl> = one checkpoint's vLLM predictions (e.g. ckpt-420).
"""
import json, sys
from collections import defaultdict, Counter
from score_stage2 import evaluate

LES4 = ["MA", "HE", "EX", "SE"]
CLIN = {0: "No-DR", 1: "Mild", 2: "Moderate", 3: "Severe", 4: "Severe"}


def grounded_counts(train_path):
    """pattern-tuple -> Counter(tier) over grounded rows (mirrors build_stage2_grade)."""
    pc = defaultdict(Counter)
    for l in open(train_path):
        if not l.strip():
            continue
        m = json.loads(l)["meta"]
        if m.get("tier_src") != "grounded":
            continue
        p = m.get("pattern")
        if not p:
            continue
        pc[tuple(bool(x) for x in p)][CLIN[m["clinical_grade"]]] += 1
    return pc


def fit_map(pc, sev_thr, min_n=8, allow_abstain=True):
    """Mirror of build_stage2_grade.fit_map with tunable threshold / abstention.
    Returns a string-keyed map ('MAHEEX', 'none', ...) for score_stage2."""
    m = {}
    for pat, c in pc.items():
        MA, HE, EX, SE = pat
        n = sum(c.values())
        key = "".join(code for code, b in zip(LES4, pat) if b) or "none"
        if n < min_n:
            tier = ("No-DR" if not any(pat) else "Mild" if (MA and not HE and not EX and not SE)
                    else ("Mod-or-Severe-indeterminate" if allow_abstain and (HE or EX) else "Moderate"))
            m[key] = tier
            continue
        top = max(c, key=c.get)
        if top in ("No-DR", "Mild"):
            m[key] = top
        else:
            mod, sev = c.get("Moderate", 0), c.get("Severe", 0)
            ms = (mod + sev) or 1
            if sev / ms >= sev_thr:
                m[key] = "Severe"
            elif mod / ms >= sev_thr:
                m[key] = "Moderate"
            elif allow_abstain:
                m[key] = "Mod-or-Severe-indeterminate"
            else:                                  # commit to the majority, never abstain
                m[key] = "Severe" if sev >= mod else "Moderate"
    return m


def main():
    test_path, pred_path, train_path = sys.argv[1], sys.argv[2], sys.argv[3]
    out_md = sys.argv[4] if len(sys.argv) > 4 else None
    test = [json.loads(l) for l in open(test_path) if l.strip()]
    preds = [json.loads(l).get("predict", "") for l in open(pred_path) if l.strip()]
    pc = grounded_counts(train_path)

    variants = [("abstain@0.70", 0.70, True), ("abstain@0.65", 0.65, True),
                ("abstain@0.60 (current)", 0.60, True), ("abstain@0.55", 0.55, True),
                ("abstain@0.50", 0.50, True), ("commit (no abstain)", 0.50, False)]

    L = ["# Stage-2 decision-map sweep on a FIXED checkpoint (no retraining)", "",
         f"test = n{len(test)}   pred = {pred_path.split('/')[-1]}", "",
         "Each row re-fits the presence->tier map at a different severe-share threshold and",
         "re-scores the SAME predictions in from-audit mode (faithful by construction).",
         "Lowering the threshold resolves borderline patterns (e.g. MA+HE+EX) into a committed",
         "tier instead of abstaining -> abstention falls, QWK/Macro-F1 rise, at the cost of",
         "committing on the empirically ~50/50 G2/G3 boundary.", "",
         "| map variant | abstain | QWK | MacroF1 | MAE | RefSens | RefSpec | SevRecall | Faithful |",
         "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name, thr, ab in variants:
        fmap = fit_map(pc, thr, allow_abstain=ab)
        m, _, _, _ = evaluate(test, preds, True, fmap, true_map=fmap)  # truth re-mapped under same rule
        L.append(f"| {name} | {m['abstain_rate']:.3f} | {m['qwk']:.4f} | {m['macro_f1']:.4f} | "
                 f"{m['mae']:.4f} | {m['sens']:.3f} | {m['spec']:.3f} | {m['sev_recall']:.3f} | "
                 f"{m['faith']:.3f} |")
    L += ["", "Referable sens/spec and severe-safety recall are the clinical headlines; abstention",
          "is reported as a transparency statistic. Faithfulness is 1.000 in every variant",
          "(tier is always a function of the stated audit)."]
    txt = "\n".join(L)
    print(txt)
    if out_md:
        open(out_md, "w").write(txt + "\n")


if __name__ == "__main__":
    main()
