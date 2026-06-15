#!/usr/bin/env python3
"""Stage-1.5 v4 — RECALL-REBALANCED present/absent perception (keeps v3's format & test).

Why v4: under the decoupled from-audit grader, the remaining clinical weaknesses are
(a) referable sensitivity 0.71 / severe recall 0.84 — ~49/168 referable cases are missed
because the AUDIT under-detects HE/EX on those images; and (b) Mild F1 = 0 — the model
cannot report MA on the aptos domain (v3 had NO aptos MA positives). Both are recall
problems created by v3's specificity-first balance:
  - v3 was absent-leaning (present:absent = 1000:1300) PLUS ~2055 grade-0 all-absent images
    -> heavy 'absent' pressure -> conservative recall (good spec, missed lesions).
  - v3 MA-present came only from FGADR/DDR masks -> zero aptos-domain MA -> Mild collapse.

v4 changes (ONLY the data balance; same single-lesion CoT format, same warm-start = Adapter1,
same LoRA recipe, SAME test set so recall/spec are directly comparable to v3):
  1. Recall-leaning caps: PRES_CAP 1000->1300, ABS_CAP 1300->1100. Absent is still
     hard-negative-first (empty-mask on DR images -> precise discrimination kept); the cut
     trims the bulk easy grade-0 negatives that were suppressing recall.
  2. Add aptos/ddr_grading g1 -> MA-present (grade-derived, clinically mild = MA-only),
     capped MA_DERIVED_CAP, marked source 'g1_ma_derived' -> teaches MA on the aptos domain
     -> targets the Mild root cause. (Same weak-label convention Stage-2 already uses.)

Evaluate v4 on the IDENTICAL v3 test (1108): per-lesion recall MUST rise (esp. MA/HE/EX)
while specificity stays acceptable. Then re-warm-start Stage-2 from v4 and re-run the
from-audit grader: referable sensitivity and Mild should improve, map unchanged.
"""
from __future__ import annotations
import json, os, hashlib
from collections import defaultdict, Counter
import numpy as np, cv2

ROOT = "/workspace/LLaMA-Factory"
VAL = f"{ROOT}/data/fundus_validated/validated_clean.jsonl"
OUT = f"{ROOT}/data/annotation"
EXP = "/workspace/stage1_5_experiment"
A1_TRAIN = "/workspace/_anno_current/data/annotation_v4/fundus_stage1_en_cot_train_sft.jsonl"
LI = json.load(open(f"{EXP}/scripts/lesion_info.json"))
MAIN4 = ["MA", "HE", "EX", "SE"]
FG = {"MA": "Microaneurysms_Masks", "HE": "Hemohedge_Masks", "EX": "HardExudate_Masks", "SE": "SoftExudate_Masks"}
MIN_PX = 5
PRES_CAP = 1300          # v3 1000 -> 1300 (more positives -> recall)
ABS_CAP = 1100           # v3 1300 -> 1100 (less 'absent' pressure; hard-neg kept first)
MA_DERIVED_CAP = 600     # aptos/ddr_grading g1 -> MA present (grade-derived), to fix Mild
N_TEST_MASK_IMG = 150    # SAME as v3 -> identical test images -> direct recall/spec comparison
N_TEST_G0_IMG = 120      # SAME as v3

def h(s): return hashlib.md5(s.encode()).hexdigest()
def stem(p): return os.path.splitext(os.path.basename(p))[0] if p else ""
def mask_present(p):
    im = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
    if im is None: return None
    bw = (im >= 128).astype(np.uint8)
    if int(bw.sum()) == 0: return False
    n, _, st, _ = cv2.connectedComponentsWithStats(bw, 8)
    return (int((st[1:, cv2.CC_STAT_AREA] >= MIN_PX).sum()) if n > 1 else 0) > 0
def fgm(img, les): return f"{ROOT}/data/FGADR/Seg-set/{FG[les]}/{stem(img)}.png"
def ddrm(img, les):
    p = img.split("/"); sp = p[2]; lbl = "segmentation label" if sp == "valid" else "label"
    return f"{ROOT}/data/DDR-dataset/lesion_segmentation/{sp}/{lbl}/{les}/{stem(img)}.tif"

def sys_p(les):
    info = LI[les]
    return ("You are a fundus lesion perception specialist.\n\nThis is a strictly single-lesion perception task. "
            "Inspect the image only for the specified target lesion. Do not assign a diabetic retinopathy grade, "
            "diagnose a disease stage, or report non-target lesions.\n\nTarget lesion:\n"
            f"- Name: {info['name']}\n- Abbreviation: {les}\n- Typical visual evidence: {info['visual']}.\n"
            f"- Important exclusions: {info['exclude']}.\n\nBase the decision only on directly visible image evidence. "
            "Do not infer lesion presence or absence from a DR grade.")
def usr_p(les):
    return (f"<image>\n\nInspect this fundus image for {LI[les]['name']} ({les}) only.\n\nDetermine whether directly "
            "visible evidence of the target lesion is present. Briefly describe the relevant visual evidence, exclude "
            "plausible confounders when applicable, and return the structured result.")
def assistant(les, present):
    info = LI[les]; name = info["name"]
    if present:
        tgt = f"Visible findings are consistent with {info['visual']}."; conf = info["positive_confounder"]
        concl = f"Directly visible evidence supports the presence of {name}."; state = "present"
    else:
        tgt = f"No reliable directly visible evidence consistent with {info['visual']} is identified."; conf = info["negative_confounder"]
        concl = f"No reliable visual evidence supports the presence of {name}."; state = "absent"
    at = "No additional target-lesion attributes are reported." if present else "No target-lesion attributes are reported because no reliable target evidence is present."
    payload = {"task": "stage1_single_lesion_perception", "target_lesion": {"name": name, "abbreviation": les},
               "image_quality": "adequate", "evidence_state": state, "present": present, "attributes": {}}
    return ("[Target Evidence]\n" + tgt + "\n\n[Confounder Assessment]\n" + conf + "\n\n[Attribute Summary]\n" + at +
            "\n\n[Conclusion]\n" + concl + "\n\n[Structured Output]\n" + json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
def row(img, les, present, source):
    return {"messages": [{"role": "system", "content": sys_p(les)}, {"role": "user", "content": usr_p(les)},
                         {"role": "assistant", "content": assistant(les, present)}],
            "images": [img],
            "meta": {"record_id": stem(img), "lesion": les, "present_state": "present" if present else "absent",
                     "evidence_source": source, "seen_by_adapter1": stem(img) in A1SEEN}}

def main():
    global A1SEEN
    A1SEEN = set()
    for l in open(A1_TRAIN):
        m = json.loads(l)["meta"]; rid = str(m.get("image_group") or m.get("record_id", ""))
        A1SEEN.add(rid.split("::")[-1] if "::" in rid else rid)
    recs = [json.loads(l) for l in open(VAL) if l.strip()]
    def g(r):
        x = r.get("grade"); return int(x) if x is not None and str(x).lstrip("-").isdigit() else None
    fg = [r["cropped_path"] for r in recs if r["dataset"] == "fgadr_seg"]
    ddr = [r["cropped_path"] for r in recs if r["dataset"] == "ddr_seg"]
    g0_imgs = [r["cropped_path"] for r in recs if g(r) == 0 and r.get("cropped_path")]
    # NEW: aptos/ddr_grading g1 -> MA-present derived (clinically mild = MA-only); no masks -> weak label
    g1_ma_imgs = [r["cropped_path"] for r in recs
                  if r.get("dataset") in ("aptos", "ddr_grading") and g(r) == 1 and r.get("cropped_path")]

    # mask present/absent per (img,lesion)
    mask = {}
    for lst, mf in [(fg, fgm), (ddr, ddrm)]:
        for img in lst:
            for les in MAIN4:
                p = mask_present(mf(img, les))
                if p is not None: mask[(img, les)] = p
    idrid = []
    for l in open(A1_TRAIN):
        r = json.loads(l); m = r["meta"]
        if m.get("evidence_source") == "strong_mask_stage1_easy" and m.get("present_state") == "present":
            idrid.append((m["lesion"], r["images"][0]))

    # TEST images — IDENTICAL selection to v3 (same params + hash keys) -> comparable
    unseen_mask_imgs = sorted({img for (img, _) in mask if stem(img) not in A1SEEN}, key=lambda x: h("tm" + x))
    unseen_g0_imgs = sorted({i for i in g0_imgs if stem(i) not in A1SEEN}, key=lambda x: h("tg" + x))
    test_imgs = set(unseen_mask_imgs[:N_TEST_MASK_IMG]) | set(unseen_g0_imgs[:N_TEST_G0_IMG])

    test, pres_pool, hard_neg, clean_neg = [], defaultdict(list), defaultdict(list), defaultdict(list)
    ma_derived = []
    for (img, les), p in mask.items():
        r = row(img, les, p, ("fgadr_mask" if "FGADR" in img else "ddr_mask"))
        if img in test_imgs: test.append(r)
        elif p: pres_pool[les].append(r)
        else: hard_neg[les].append(r)
    for les, img in idrid:
        if img not in test_imgs: pres_pool[les].append(row(img, les, True, "strong_mask"))
    for img in g0_imgs:
        for les in MAIN4:
            r = row(img, les, False, "grade0_neg")
            if img in test_imgs: test.append(r)
            else: clean_neg[les].append(r)
    for img in g1_ma_imgs:                              # NEW MA positives (kept out of test)
        if img not in test_imgs:
            ma_derived.append(row(img, "MA", True, "g1_ma_derived"))

    # TRAIN: present cap (MA reserves room for derived); absent = hard-neg first then clean-neg, cap
    train = []
    for les in MAIN4:
        if les == "MA":
            der = sorted(ma_derived, key=lambda x: h(x["meta"]["record_id"] + "mader"))[:MA_DERIVED_CAP]
            msk = sorted(pres_pool["MA"], key=lambda x: h(x["meta"]["record_id"] + "MA"))[:max(0, PRES_CAP - len(der))]
            pres = msk + der
        else:
            pres = sorted(pres_pool[les], key=lambda x: h(x["meta"]["record_id"] + les))[:PRES_CAP]
        hn = sorted(hard_neg[les], key=lambda x: h(x["meta"]["record_id"] + "hn" + les))
        cn = sorted(clean_neg[les], key=lambda x: h(x["meta"]["record_id"] + "cn" + les))
        absn = (hn + cn)[:ABS_CAP]
        train += pres + absn
    train = sorted(train, key=lambda x: h(x["meta"]["record_id"] + x["meta"]["lesion"] + x["meta"]["present_state"]))

    for name, rows in [("stage1_5_v4_train", train), ("stage1_5_v4_test", test)]:
        with open(f"{OUT}/{name}_sft.jsonl", "w") as fo:
            for r in rows: fo.write(json.dumps(r, ensure_ascii=False) + "\n")
    def st(rows):
        ls = Counter((r["meta"]["lesion"], r["meta"]["present_state"]) for r in rows)
        sr = Counter(r["meta"]["evidence_source"] for r in rows)
        return {"n": len(rows), "lesion_state": {f"{a}/{b}": n for (a, b), n in sorted(ls.items())},
                "by_source": dict(sr), "all_unseen": all(not r["meta"].get("seen_by_adapter1") for r in rows) if rows else None}
    meta = {"train": st(train), "test": st(test), "pres_cap": PRES_CAP, "abs_cap": ABS_CAP,
            "ma_derived_cap": MA_DERIVED_CAP, "n_test_mask_img": N_TEST_MASK_IMG, "n_test_g0_img": N_TEST_G0_IMG,
            "note": "v4 recall-rebalanced: present-leaning caps + aptos/ddr_grading g1 MA-present (fix Mild); "
                    "same single-lesion format + Adapter1 warm-start; test identical to v3 for direct comparison"}
    json.dump(meta, open(f"{EXP}/data/stage1_5_v4_distribution.json", "w"), ensure_ascii=False, indent=1)
    print(json.dumps(meta, ensure_ascii=False, indent=1))

if __name__ == "__main__":
    main()
