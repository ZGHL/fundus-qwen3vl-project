#!/usr/bin/env python3
"""Score Stage-2 faithful triage from a PER-LESION audit prediction file (1200 rows = 300 img x4).

Assembles each image's MA/HE/EX/SE audit (lenient Present/Absent parse), applies the fitted
presence->tier map, and scores referable/severe/QWK vs the per-image clinical grade & GT tier.
Used to compare any model (zero-shot baselines OR ours) under the identical from-audit protocol.

Usage: score_stage2_from_audit.py <audit_queries.jsonl> <pred.jsonl> <dist.json> [label]
  pred rows align by order to the query set (4 per image: MA,HE,EX,SE).
"""
import json, re, sys
from collections import defaultdict

LES = ["MA", "HE", "EX", "SE"]
ORDER = ["No-DR", "Mild", "Moderate", "Mod-or-Severe-indeterminate", "Severe"]
IDX = {t: i for i, t in enumerate(ORDER)}
REFER = {"Moderate", "Mod-or-Severe-indeterminate", "Severe"}

def pa(t):
    # schema output (ours): JSON {"present": true/false}; fall back to lenient Present/Absent
    m = re.search(r'"present"\s*:\s*(true|false)', str(t))
    if m:
        return m.group(1) == "true"
    s = str(t).strip().lower()
    m = re.search(r"\b(present|absent|yes|no)\b", s[:80]) or re.search(r"\b(present|absent|yes|no)\b", s)
    return None if not m else m.group(1) in ("present", "yes")

def pattern_key(present):
    k = "".join(c for c in LES if c in present)
    return k if k else "none"

def qwk(ti, pj, k=5):
    n = len(ti)
    if not n: return 0.0
    O = [[0]*k for _ in range(k)]
    for i, j in zip(ti, pj): O[i][j] += 1
    rt = [sum(O[i]) for i in range(k)]; ct = [sum(O[i][j] for i in range(k)) for j in range(k)]
    num = den = 0.0
    for i in range(k):
        for j in range(k):
            w = ((i-j)/(k-1))**2; num += w*O[i][j]; den += w*rt[i]*ct[j]/n
    return 1 - num/den if den else 0.0

def main():
    qf, pf, distf = sys.argv[1], sys.argv[2], sys.argv[3]
    label = sys.argv[4] if len(sys.argv) > 4 else pf.split("/")[-1]
    fmap = json.load(open(distf)).get("fitted_map(pattern->tier)")
    queries = [json.loads(l) for l in open(qf) if l.strip()]
    preds = [json.loads(l).get("predict", "") for l in open(pf) if l.strip()]
    n = min(len(queries), len(preds))

    # assemble per-image audit
    img = {}  # image -> {lesion: present_bool}, meta
    for q, p in zip(queries[:n], preds[:n]):
        m = q["meta"]; key = m["image"]
        d = img.setdefault(key, {"present": set(), "meta": m})
        if pa(p):
            d["present"].add(m["lesion"])

    ref_tp = ref_fp = ref_fn = ref_tn = 0
    sev_tot = sev_ref = 0
    abstain = 0
    ti, pj = [], []
    for key, d in img.items():
        m = d["meta"]; cg = m["clinical_grade"]; true_tier = m["dr_tier"]
        pred = fmap.get(pattern_key(d["present"]), "No-DR")
        clin_ref = cg >= 2; pred_ref = pred in REFER
        if clin_ref and pred_ref: ref_tp += 1
        elif clin_ref: ref_fn += 1
        elif pred_ref: ref_fp += 1
        else: ref_tn += 1
        if cg in (3, 4):
            sev_tot += 1; sev_ref += int(pred_ref)
        if pred == "Mod-or-Severe-indeterminate": abstain += 1
        ti.append(IDX[true_tier]); pj.append(IDX[pred])

    N = len(img)
    sens = ref_tp/(ref_tp+ref_fn) if ref_tp+ref_fn else 0
    spec = ref_tn/(ref_tn+ref_fp) if ref_tn+ref_fp else 0
    print(f"{label}: images={N}")
    print(f"  Referable  sens={sens:.3f}  spec={spec:.3f}  (TP{ref_tp} FP{ref_fp} FN{ref_fn} TN{ref_tn})")
    print(f"  Severe safety recall = {sev_ref}/{sev_tot} = {sev_ref/sev_tot if sev_tot else 0:.3f}")
    print(f"  5-tier QWK = {qwk(ti,pj):.3f}   Abstain = {abstain}/{N} = {abstain/N:.3f}")
    print(f"  Faithfulness = 1.000 (from-audit, tier=map(audit) by construction)")

if __name__ == "__main__":
    main()
