#!/usr/bin/env python3
"""Stage-2 grade data: FAITHFUL triage via data-fit presence->tier map + calibrated abstention.

- Audit = MA/HE/EX/SE present/absent (verifiable); IRMA/NV ABSTAINED (never claimed).
- tier = f(audit pattern), where f is FIT FROM DATA (grounded clinical grades), not hand-coded:
    * patterns dominated by No-DR / Mild -> that tier
    * referable patterns: Severe-share>=0.6 -> Severe ; Moderate-share>=0.6 -> Moderate ;
      else (G2/G3 not separable from visible evidence, e.g. MA+HE+EX no-SE ~50/50)
      -> "Mod-or-Severe-indeterminate" (CALIBRATED ABSTENTION on the empirically undecidable boundary).
  -> 100% faithful (tier is a function of stated, verifiable presence), data-optimal, interpretable.
- clinical grade kept in meta = EVAL reference (faithful-ceiling, referable, vs learned grader).
- Grounded audit = FGADR real masks + IDRiD validated_clean present. aptos/ddr via --v3-preds (VM).
- g3+g4 share "Severe" semantics (severe-NPDR-or-PDR; NV/IRMA unseeable -> not sub-graded).
"""
from __future__ import annotations
import json, os, argparse, hashlib
from collections import defaultdict, Counter
import cv2

ROOT = "/workspace/LLaMA-Factory"
VAL = f"{ROOT}/data/fundus_validated/validated_clean.jsonl"
OUT = f"{ROOT}/data/annotation"
EXP = "/workspace/stage1_5_experiment"
A1_TRAIN = "/workspace/_anno_current/data/annotation_v4/fundus_stage1_en_cot_train_sft.jsonl"
LI = json.load(open(f"{EXP}/scripts/lesion_info.json"))
FG = {"MA": "Microaneurysms_Masks", "HE": "Hemohedge_Masks", "EX": "HardExudate_Masks", "SE": "SoftExudate_Masks"}
LES4 = ["MA", "HE", "EX", "SE"]
CLIN = {0: "No-DR", 1: "Mild", 2: "Moderate", 3: "Severe", 4: "Severe"}
TIERS = ["No-DR", "Mild", "Moderate", "Severe", "Mod-or-Severe-indeterminate"]
REFER = {"Moderate", "Severe", "Mod-or-Severe-indeterminate"}
CAPS = {"No-DR": 1400, "Mild": 1000, "Moderate": 1400, "Severe": 1400, "Mod-or-Severe-indeterminate": 1400}
TEST_PER_TIER = 60
SEV_THR = 0.60   # severe-share to confidently call Severe / moderate-share for Moderate

def h(s): return hashlib.md5(s.encode()).hexdigest()
def stem(p): return os.path.splitext(os.path.basename(p))[0] if p else ""
def g(r):
    x = r.get("grade"); return int(x) if x is not None and str(x).lstrip("-").isdigit() and 0 <= int(x) <= 4 else None
def mask_present(p):
    im = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
    return (im is not None) and bool((im >= 128).any())

def fit_map(pattern_counts, min_n=8):
    m = {}
    for pat, c in pattern_counts.items():
        MA, HE, EX, SE = pat
        n = sum(c.values())
        if n < min_n:
            m[pat] = ("No-DR" if not any(pat) else "Mild" if (MA and not HE and not EX and not SE)
                      else "Mod-or-Severe-indeterminate" if (HE or EX) else "Moderate")
            continue
        top = max(c, key=c.get)
        if top in ("No-DR", "Mild"):
            m[pat] = top
        else:
            mod, sev = c.get("Moderate", 0), c.get("Severe", 0); ms = (mod + sev) or 1
            m[pat] = ("Severe" if sev / ms >= SEV_THR else "Moderate" if mod / ms >= SEV_THR
                      else "Mod-or-Severe-indeterminate")
    return m

def audit_block(pa):
    lines = [(f"- {l}: present — {LI[l]['visual']}." if pa[l] else f"- {l}: absent.") for l in LES4]
    lines += ["- IRMA: not visually assessable — abstained.", "- NV: not visually assessable — abstained."]
    return "\n".join(lines)

def decision_path(pa, tier):
    present = [l for l in LES4 if pa[l]]
    s = []
    if tier == "No-DR":
        return "Step1 no reliable DR lesion (MA/HE/EX/SE) visible -> No-DR."
    s.append("Step1 reliable DR lesion present: " + ", ".join(present) + ".")
    if tier == "Mild":
        return "\n".join(s + ["Step2 only MA, no HE/EX -> Mild."])
    s.append("Step2 HE and/or EX present -> referable.")
    if tier == "Moderate":
        s.append("Step3 this lesion pattern is consistent with moderate NPDR -> Moderate.")
    elif tier == "Severe":
        s.append("Step3 broad lesion co-occurrence (MA/HE/EX" + (" + SE" if pa["SE"] else "") +
                 ") -> Severe (severe-NPDR-or-PDR).")
    else:  # indeterminate
        s.append("Step3 this lesion pattern does not separate moderate from severe NPDR on visible evidence "
                 "(distinguishing them needs per-quadrant hemorrhage counts / IRMA / NV, which are not visually "
                 "assessable) -> referable, severity indeterminate.")
    return "\n".join(s)

def make_row(img, pa, grade, tier, src):
    clin = CLIN[grade]; referable = tier in REFER
    sev_indet = tier == "Mod-or-Severe-indeterminate"
    body = ("[Lesion Audit]\n" + audit_block(pa) + "\n\n[Decision Path]\n" + decision_path(pa, tier) +
            "\n\n[Conclusion]\n" + f"DR tier = {tier}; referable_dr = {'yes' if referable else 'no'}; " +
            ("severity indeterminate (moderate vs severe / PDR not visually separable); evidence_limited."
             if (sev_indet or tier == "Severe") else "evidence is directly visible.") +
            "\n\n[JSON]\n" + json.dumps({
                "task": "stage2_grade", "dr_tier": tier, "referable_dr": referable,
                "lesions_present": [l for l in LES4 if pa[l]], "abstained": ["IRMA", "NV"],
                "severity_indeterminate": sev_indet, "evidence_limited": sev_indet or tier == "Severe",
            }, ensure_ascii=False, separators=(",", ":")))
    return {"messages": [
                {"role": "system", "content": ("You are a diabetic retinopathy triage assistant. Audit MA/HE/EX/SE from "
                    "directly visible evidence; IRMA and NV are not visually reliable and must be abstained (never claim to "
                    "see them). Using only verifiable present/absent evidence, assign a DR tier; when visible evidence "
                    "cannot separate moderate from severe NPDR, say so (severity indeterminate) rather than guessing. Do not "
                    "use lesion counts.")},
                {"role": "user", "content": ("<image>\nAudit MA/HE/EX/SE (abstain on IRMA/NV), then assign the DR tier. "
                    "Output: [Lesion Audit] -> [Decision Path] -> [Conclusion] -> [JSON].")},
                {"role": "assistant", "content": body}],
            "images": [img],
            "meta": {"record_id": stem(img), "clinical_grade": grade, "clinical_tier": clin, "dr_tier": tier,
                     "referable": referable, "tier_src": src, "pattern": [int(pa[l]) for l in LES4],
                     "seen_by_adapter1": stem(img) in A1SEEN}}

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--v3-preds", default=None); args = ap.parse_args()
    global A1SEEN
    A1SEEN = set()
    for l in open(A1_TRAIN):
        mm = json.loads(l)["meta"]; rid = str(mm.get("image_group") or mm.get("record_id", ""))
        A1SEEN.add(rid.split("::")[-1] if "::" in rid else rid)
    recs = [json.loads(l) for l in open(VAL) if l.strip()]

    # collect (img, pattern, grade, src) for grounded
    items = []
    for r in recs:
        if r["dataset"] != "fgadr_seg": continue
        gg = g(r)
        if gg is None: continue
        iid = stem(r["cropped_path"])
        pa = {les: mask_present(f"{ROOT}/data/FGADR/Seg-set/{fo}/{iid}.png") for les, fo in FG.items()}
        items.append((r["cropped_path"], pa, gg, "grounded"))
    for r in recs:
        if r["dataset"] != "idrid": continue
        gg = g(r)
        if gg is None or not r.get("cropped_path"): continue
        les = r.get("lesions") or {}
        pa = {l: bool((les.get(l) or {}).get("present")) for l in LES4}
        items.append((r["cropped_path"], pa, gg, "grounded"))
    # aptos/ddr g0/g1: audit derived from grade (clinical: g0=no lesion, g1=MA-only) -> fills No-DR/Mild, no v3 needed
    for r in recs:
        if r["dataset"] not in ("aptos", "ddr_grading") or not r.get("cropped_path"): continue
        gg = g(r)
        if gg == 0:
            items.append((r["cropped_path"], {l: False for l in LES4}, 0, "grade_derived"))
        elif gg == 1:
            items.append((r["cropped_path"], {"MA": True, "HE": False, "EX": False, "SE": False}, 1, "grade_derived"))
    # pseudo g2+ (optional, needs v3)
    if args.v3_preds and os.path.exists(args.v3_preds):
        preds = {json.loads(l)["image_id"]: json.loads(l) for l in open(args.v3_preds)}
        for r in recs:
            if r["dataset"] not in ("aptos", "ddr_grading"): continue
            gg = g(r); iid = stem(r["cropped_path"])
            if gg is None or gg < 2 or iid not in preds: continue   # g0/g1 already added via grade_derived
            pa = {l: bool(preds[iid].get(l)) for l in LES4}
            items.append((r["cropped_path"], pa, gg, "pseudo"))

    # PASS1: fit presence->tier map on GROUNDED clinical grades
    pat_counts = defaultdict(Counter)
    for img, pa, gg, src in items:
        if src == "grounded":
            pat_counts[tuple(pa[l] for l in LES4)][CLIN[gg]] += 1
    fmap = fit_map(pat_counts)

    # PASS2: assign tier = map(pattern)
    rows = [make_row(img, pa, gg, fmap[tuple(pa[l] for l in LES4)], src) for img, pa, gg, src in items]
    by_tier = defaultdict(list)
    for r in rows: by_tier[r["meta"]["dr_tier"]].append(r)
    test, train = [], []
    for tier, lst in by_tier.items():
        unseen = sorted([r for r in lst if not r["meta"]["seen_by_adapter1"]], key=lambda r: h("te" + r["meta"]["record_id"]))
        test += unseen[:TEST_PER_TIER]
        rest = unseen[TEST_PER_TIER:] + [r for r in lst if r["meta"]["seen_by_adapter1"]]
        train += sorted(rest, key=lambda r: h(r["meta"]["record_id"]))[:CAPS.get(tier, 1400)]
    train = sorted(train, key=lambda r: h(r["meta"]["record_id"] + r["meta"]["dr_tier"]))

    for name, data in [("stage2_grade_train", train), ("stage2_grade_test", test)]:
        with open(f"{OUT}/{name}_sft.jsonl", "w") as fo:
            for r in data: fo.write(json.dumps(r, ensure_ascii=False) + "\n")
    def st(data):
        cross = Counter((r["meta"]["clinical_tier"], r["meta"]["dr_tier"]) for r in data)
        return {"n": len(data), "tier": {t: sum(1 for r in data if r["meta"]["dr_tier"] == t) for t in TIERS},
                "src": dict(Counter(r["meta"]["tier_src"] for r in data)),
                "referable": dict(Counter(r["meta"]["referable"] for r in data)),
                "clinical_vs_tier": {f"{c}->{t}": n for (c, t), n in sorted(cross.items())}}
    fmap_readable = {"".join(k for k, v in zip("MA/HE/EX/SE".split("/"), pat) if v) or "none": t
                     for pat, t in sorted(fmap.items())}
    meta = {"fitted_map(pattern->tier)": fmap_readable, "sev_thr": SEV_THR,
            "train": st(train), "test": st(test), "caps": CAPS,
            "note": "faithful: tier=data-fit map over verifiable presence; abstain on undecidable Mod/Severe; "
                    "clinical grade = eval ref; grounded FGADR(mask)+IDRiD; aptos/ddr via --v3-preds."}
    json.dump(meta, open(f"{EXP}/data/stage2_grade_distribution.json", "w"), ensure_ascii=False, indent=1)
    print(json.dumps(meta, ensure_ascii=False, indent=1))

if __name__ == "__main__":
    main()
