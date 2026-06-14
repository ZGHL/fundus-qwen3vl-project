#!/usr/bin/env python3
"""Score Stage-2 faithful triage. Usage: score_stage2.py <test_jsonl> <pred_jsonl> [out_md]
Aligns by order. test meta has clinical_grade/clinical_tier/dr_tier(GT-map)/pattern."""
import json, re, sys
from collections import Counter

TIERS = ["No-DR", "Mild", "Moderate", "Severe", "Mod-or-Severe-indeterminate"]
REFER = {"Moderate", "Severe", "Mod-or-Severe-indeterminate"}

def jget(t):
    objs, d, st = [], 0, None
    for i, c in enumerate(t):
        if c == "{":
            if d == 0: st = i
            d += 1
        elif c == "}":
            d -= 1
            if d == 0 and st is not None: objs.append(t[st:i+1]); st = None
    for m in reversed(objs):
        try: return json.loads(m)
        except Exception: pass
    return None

def map_tier(present):  # faithful rule used to check consistency (mirrors fitted map's logic, coarse)
    MA, HE, EX, SE = (l in present for l in ["MA", "HE", "EX", "SE"])
    if not any([MA, HE, EX, SE]): return "No-DR"
    if MA and not HE and not EX and not SE: return "Mild"
    if (HE and EX): return "Severe" if not (MA and not SE) else "Severe"
    return "Moderate"  # coarse; exact map in distribution json

def main():
    test = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
    preds = [json.loads(l).get("predict", "") for l in open(sys.argv[2]) if l.strip()]
    n = min(len(test), len(preds))
    tier_pred = Counter(); parse_fail = 0
    ref_tp = ref_fp = ref_fn = ref_tn = 0
    sev_total = sev_referred = 0
    faith_ok = faith_n = nv_irma_claim = 0
    cm = Counter()  # clinical_tier -> pred_tier
    for r, p in zip(test[:n], preds[:n]):
        j = jget(p)
        cg = r["meta"]["clinical_grade"]; ctier = r["meta"]["clinical_tier"]
        clin_ref = cg >= 2
        if not j or "dr_tier" not in j:
            parse_fail += 1; pt = None
        else:
            pt = j.get("dr_tier"); tier_pred[pt] += 1
            pres = j.get("lesions_present") or []
            # faithfulness: tier consistent with its own audit (coarse) + no NV/IRMA claimed
            faith_n += 1
            if pt in (map_tier(pres), "Mod-or-Severe-indeterminate"): faith_ok += 1
            if any(x in ("NV", "IRMA") for x in pres) or "NV" in str(j.get("present", "")): nv_irma_claim += 1
        pred_ref = (pt in REFER) if pt else False
        if clin_ref and pred_ref: ref_tp += 1
        elif clin_ref and not pred_ref: ref_fn += 1
        elif (not clin_ref) and pred_ref: ref_fp += 1
        else: ref_tn += 1
        if cg in (3, 4):
            sev_total += 1; sev_referred += int(pred_ref)
        cm[(ctier, pt)] += 1
    sens = ref_tp / (ref_tp + ref_fn) if ref_tp + ref_fn else 0
    spec = ref_tn / (ref_tn + ref_fp) if ref_tn + ref_fp else 0
    ppv = ref_tp / (ref_tp + ref_fp) if ref_tp + ref_fp else 0
    out = [f"# Stage-2 results (n={n})", "",
           f"parse_fail={parse_fail}  abstain(Indeterminate)={tier_pred.get('Mod-or-Severe-indeterminate',0)}",
           f"pred tier dist: {dict(tier_pred)}", "",
           "## Referable (>=Moderate, vs clinical grade>=2)",
           f"sensitivity={sens:.3f}  specificity={spec:.3f}  PPV={ppv:.3f}  (TP{ref_tp} FP{ref_fp} FN{ref_fn} TN{ref_tn})",
           f"severe-safety recall (true g3/g4 -> referable) = {sev_referred}/{sev_total} = {sev_referred/sev_total if sev_total else 0:.3f}", "",
           "## Faithfulness",
           f"tier consistent with own audit = {faith_ok}/{faith_n} = {faith_ok/faith_n if faith_n else 0:.3f}",
           f"NV/IRMA fabrication = {nv_irma_claim} (want 0)", "",
           "## clinical_tier -> predicted_tier",
           "| clinical \\ pred | " + " | ".join(TIERS) + " | None |", "|" + "---|" * (len(TIERS)+2)]
    for ct in TIERS[:4]:
        row = [str(cm.get((ct, pt), 0)) for pt in TIERS] + [str(cm.get((ct, None), 0))]
        out.append(f"| {ct} | " + " | ".join(row) + " |")
    out.append("\n(reference: faithful ceiling = 0.688 4-tier on GT presence)")
    txt = "\n".join(out); print(txt)
    if len(sys.argv) > 3: open(sys.argv[3], "w").write(txt + "\n")

if __name__ == "__main__":
    main()
