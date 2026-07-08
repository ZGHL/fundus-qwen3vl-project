#!/usr/bin/env python3
"""Stage-1.5 v4 — RECALL-REBALANCED present/absent perception, FAITHFUL (real masks only).

Why v4: under the decoupled from-audit grader the only clinically-valuable weakness is recall
(referable sensitivity 0.71 / severe recall 0.84 — the audit under-detects HE/EX). v3 was
specificity-first (present:absent = 1000:1300) so recall was conservative.

v4 = recall rebalance using ONLY mask-grounded positives (faithful; no grade-derived weak
labels). We have plenty of real positives on disk: MA ~2075, HE ~2137, EX ~1846, SE ~906.
v3 left them on the table by capping present at 1000.

Changes vs v3 (data only):
  - PRES_CAP 1000 -> 2000: use ALL available mask-grounded positives (MA/HE/EX ~1.6-1.8k each).
  - ABS_CAP unchanged at 1300 (same proven negative budget -> specificity preserved). Net per
    lesion flips from absent-leaning 0.77:1 to present-leaning ~1.4:1 -> recall rises, spec held.
  - NO g1_ma_derived. The earlier v4 draft inferred MA from grade-1 labels (aptos/ddr_grading),
    which (a) violates the single-lesion rule "do not infer presence from a DR grade", (b) risks
    teaching MA hallucination, and (c) actually REDUCED grounded MA to 700 (<v3's 1000). Removed.

Leakage fix (the critical one): the Stage-2 grading test (297 images) overlapped the v3/v4
Stage-1.5 training pool by 172, so a Stage-2 grader warm-started from v3/v4 was evaluated partly
on images seen during Stage-1.5 training. v4 train now EXCLUDES every Stage-2-test image stem
(data/stage2_test_heldout_stems.txt) -> warm-starting Stage-2 from v4 and evaluating on the
Stage-2 test is leak-free for those 297. (The v3 referable/Mild numbers were leak-affected and
must not be cited as generalization.)

Mild / aptos domain: there is NO faithful MA label for aptos (no masks; RetSAM weak labels have
no MA). So v4 does not touch aptos. Mild is represented faithfully by real MA-only mask images,
and MA recall is measured in-domain on the test (which contains real MA positives). Whether mask-
domain MA transfers to aptos is an honest, separate limitation, not something to fake with
grade-derived labels.

Same single-lesion CoT format, same Adapter1 warm-start, same test selection as v3 (so per-lesion
recall/spec are directly comparable). TRAINING CONFIG must also match v3 exactly — see
configs/stage1_5_v4_warmstart.yaml.
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
HELDOUT = f"{EXP}/data/stage2_test_heldout_stems.txt"   # Stage-2 test image stems -> excluded from train
LI = json.load(open(f"{EXP}/scripts/lesion_info.json"))
MAIN4 = ["MA", "HE", "EX", "SE"]
FG = {"MA": "Microaneurysms_Masks", "HE": "Hemohedge_Masks", "EX": "HardExudate_Masks", "SE": "SoftExudate_Masks"}
MIN_PX = 5
PRES_CAP = 2000          # v3 1000 -> 2000: use all mask-grounded positives (recall)
ABS_CAP = 1300           # = v3 (preserve specificity budget); net ratio flips present-leaning
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
    EXCLUDE = set(l.strip() for l in open(HELDOUT) if l.strip())   # Stage-2 test stems -> never in train
    recs = [json.loads(l) for l in open(VAL) if l.strip()]
    def g(r):
        x = r.get("grade"); return int(x) if x is not None and str(x).lstrip("-").isdigit() else None
    fg = [r["cropped_path"] for r in recs if r["dataset"] == "fgadr_seg"]
    ddr = [r["cropped_path"] for r in recs if r["dataset"] == "ddr_seg"]
    g0_imgs = [r["cropped_path"] for r in recs if g(r) == 0 and r.get("cropped_path")]

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

    def excluded(img):                       # held out from TRAIN: own test OR Stage-2 test stem
        return img in test_imgs or stem(img) in EXCLUDE

    test, pres_pool, hard_neg, clean_neg = [], defaultdict(list), defaultdict(list), defaultdict(list)
    n_skip = Counter()
    for (img, les), p in mask.items():
        r = row(img, les, p, ("fgadr_mask" if "FGADR" in img else "ddr_mask"))
        if img in test_imgs: test.append(r)
        elif stem(img) in EXCLUDE: n_skip["mask"] += 1          # Stage-2 test stem -> drop from train
        elif p: pres_pool[les].append(r)
        else: hard_neg[les].append(r)
    for les, img in idrid:
        if not excluded(img): pres_pool[les].append(row(img, les, True, "strong_mask"))
        elif stem(img) in EXCLUDE: n_skip["idrid"] += 1
    for img in g0_imgs:
        for les in MAIN4:
            r = row(img, les, False, "grade0_neg")
            if img in test_imgs: test.append(r)
            elif stem(img) in EXCLUDE: n_skip["g0"] += 1
            else: clean_neg[les].append(r)

    # TRAIN: present cap (all real mask positives); absent = hard-neg first then clean-neg, cap
    train = []
    for les in MAIN4:
        pres = sorted(pres_pool[les], key=lambda x: h(x["meta"]["record_id"] + les))[:PRES_CAP]
        hn = sorted(hard_neg[les], key=lambda x: h(x["meta"]["record_id"] + "hn" + les))
        cn = sorted(clean_neg[les], key=lambda x: h(x["meta"]["record_id"] + "cn" + les))
        absn = (hn + cn)[:ABS_CAP]
        train += pres + absn
    train = sorted(train, key=lambda x: h(x["meta"]["record_id"] + x["meta"]["lesion"] + x["meta"]["present_state"]))

    # leak assertion: no train image stem may be a Stage-2 test stem
    leaked = sorted({r["meta"]["record_id"] for r in train} & EXCLUDE)
    assert not leaked, f"LEAK: {len(leaked)} Stage-2 test stems present in train, e.g. {leaked[:5]}"

    for name, rows in [("stage1_5_v4_train", train), ("stage1_5_v4_test", test)]:
        with open(f"{OUT}/{name}_sft.jsonl", "w") as fo:
            for r in rows: fo.write(json.dumps(r, ensure_ascii=False) + "\n")
    def st(rows):
        ls = Counter((r["meta"]["lesion"], r["meta"]["present_state"]) for r in rows)
        sr = Counter(r["meta"]["evidence_source"] for r in rows)
        return {"n": len(rows), "lesion_state": {f"{a}/{b}": n for (a, b), n in sorted(ls.items())},
                "by_source": dict(sr), "all_unseen": all(not r["meta"].get("seen_by_adapter1") for r in rows) if rows else None}
    meta = {"train": st(train), "test": st(test), "pres_cap": PRES_CAP, "abs_cap": ABS_CAP,
            "n_test_mask_img": N_TEST_MASK_IMG, "n_test_g0_img": N_TEST_G0_IMG,
            "stage2_test_stems_excluded_from_train": len(EXCLUDE), "train_skipped_as_heldout": dict(n_skip),
            "note": "v4 faithful recall-rebalance: real mask positives only (PRES_CAP 2000), ABS_CAP=v3 1300, "
                    "NO grade-derived labels; Stage-2 test stems excluded from train (leak-free v4->Stage-2 eval); "
                    "same single-lesion format + Adapter1 warm-start + v3-identical test"}
    json.dump(meta, open(f"{EXP}/data/stage1_5_v4_distribution.json", "w"), ensure_ascii=False, indent=1)
    print(json.dumps(meta, ensure_ascii=False, indent=1))

if __name__ == "__main__":
    main()
