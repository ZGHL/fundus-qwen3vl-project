#!/usr/bin/env python3
"""Stage-1.5 v5 — PER-LESION differentiated rebalance, FAITHFUL (real masks only).

Why v5 (supersedes v4's blunt uniform PRES_CAP=2000):
  v3 ckpt-400 per-lesion profile shows the four lesions are NOT bottlenecked on the same axis:
    - HE  recall 0.706 / spec 0.810  -> RECALL-bound (this under-detection is what caps the
                                        decoupled grader's referable sensitivity at 0.71).
    - EX  recall 0.822 / spec 0.816  -> already balanced; leave it alone.
    - MA  recall 0.955 / spec 0.481  -> OVER-REPORTS (recall already maxed); needs SPECIFICITY.
    - SE  recall 0.950 / spec 0.397  -> OVER-REPORTS; needs SPECIFICITY (precision ~0.21).
  v4 raised PRES_CAP for ALL four. That is the right medicine for HE, pointless for EX, and the
  WRONG direction for MA and SE (pushing the present:absent ratio up makes the two over-reporters
  report even more -> spec drops further). v5 treats each lesion on its own bottleneck.

Changes vs v4 (DATA ONLY — training config still mirrors v3 exactly, see configs/stage1_5_v5_warmstart.yaml):
  - PRES_CAP / ABS_CAP are now PER-LESION:
        HE: present 1800 / absent 1300  (recall push; HE spec 0.810 has headroom to give a little)
        EX: present 1000 / absent 1300  (unchanged = v3; already balanced)
        MA: present 1000 / absent 2000  (spec push; keep present moderate, recall is maxed)
        SE: present 1000 / absent 2000  (spec push; SE has only ~906 real positives so present
                                         resolves to ~860 regardless -> we do NOT add SE positives,
                                         which would worsen over-reporting; we add SE NEGATIVES)
  - SE/MA absent draw HARD negatives FIRST (empty-mask lesions on DR images), then grade-0 clean.
    For SE the hard-neg pool (~1733) alone nearly fills ABS_CAP=2000, and those DR images are
    confounder-rich (they contain EX / HE / glare) -> exactly the bright-region confounders the
    SE [Confounder Assessment] step must learn to reject. This is the targeted fix for SE
    over-reporting, and it is purely real-mask-grounded (no RetSAM weak SE, which is low-confidence
    and was deliberately suppressed in cleaning; adding SE positives there would be both noisy AND
    the wrong direction).

KEPT IDENTICAL to v4 (so the ladder base->Adapter1->v3->v4->v5 stays directly comparable):
  - Single-lesion present/absent CoT format (no count/area), Adapter1 warm-start.
  - TEST selection (N_TEST_MASK_IMG=150, N_TEST_G0_IMG=120, same hash keys) -> stage1_5_v5_test
    is byte-identical in content to v3/v4 test. Per-lesion recall/spec are directly comparable.
  - Stage-2 test stems excluded from TRAIN (data/stage2_test_heldout_stems.txt) + leak assertion
    -> warm-starting Stage-2 from v5 and evaluating on the Stage-2 test stays leak-free.

Availability check (real masks on disk; v5 caps are all within budget):
  positives: MA ~2075, HE ~2137, EX ~1846, SE ~906
  negatives per lesion = hard-neg (empty mask on DR img) + grade-0 clean:
    MA ~605+2055, HE ~542+2055, EX ~834+2055, SE ~1733+2055
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
# PER-LESION caps (v5): HE recall-push, MA/SE spec-push, EX unchanged. See module docstring.
PRES_CAP = {"HE": 1800, "EX": 1000, "MA": 1000, "SE": 1000}   # SE resolves to ~860 (availability)
ABS_CAP  = {"HE": 1300, "EX": 1300, "MA": 2000, "SE": 2000}
N_TEST_MASK_IMG = 150    # SAME as v3/v4 -> identical TEST images -> direct recall/spec comparison
N_TEST_G0_IMG = 120      # SAME as v3/v4
N_DEV_MASK_IMG = 150     # NEW: separate checkpoint-SELECTION (DEV) images, disjoint from TEST and TRAIN
N_DEV_G0_IMG = 120       # NEW: 3-way split so we never select the checkpoint on the final TEST set

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

    # Held-out image pools (Adapter1-unseen, deterministic hash order).
    # TEST = first slice -> IDENTICAL to v3/v4 (ladder-comparable); NEVER touched during selection.
    # DEV  = next, disjoint slice -> used ONLY to pick the checkpoint; never reported as final.
    unseen_mask_imgs = sorted({img for (img, _) in mask if stem(img) not in A1SEEN}, key=lambda x: h("tm" + x))
    unseen_g0_imgs = sorted({i for i in g0_imgs if stem(i) not in A1SEEN}, key=lambda x: h("tg" + x))
    assert len(unseen_mask_imgs) >= N_TEST_MASK_IMG + N_DEV_MASK_IMG, \
        f"not enough unseen mask images for disjoint TEST+DEV: have {len(unseen_mask_imgs)}"
    assert len(unseen_g0_imgs) >= N_TEST_G0_IMG + N_DEV_G0_IMG, \
        f"not enough unseen grade-0 images for disjoint TEST+DEV: have {len(unseen_g0_imgs)}"
    test_imgs = set(unseen_mask_imgs[:N_TEST_MASK_IMG]) | set(unseen_g0_imgs[:N_TEST_G0_IMG])
    dev_imgs = set(unseen_mask_imgs[N_TEST_MASK_IMG:N_TEST_MASK_IMG + N_DEV_MASK_IMG]) | \
               set(unseen_g0_imgs[N_TEST_G0_IMG:N_TEST_G0_IMG + N_DEV_G0_IMG])
    assert not (test_imgs & dev_imgs), "TEST/DEV image overlap"

    def excluded(img):                       # held out from TRAIN: TEST or DEV image, or Stage-2 test stem
        return img in test_imgs or img in dev_imgs or stem(img) in EXCLUDE

    # has_other_lesion[img] = any of MA/HE/EX/SE mask present on this image (for confounder ranking)
    has_other = defaultdict(set)
    for (img, les), p in mask.items():
        if p: has_other[img].add(les)

    test, dev = [], []
    pres_pool, hard_neg, clean_neg = defaultdict(list), defaultdict(list), defaultdict(list)
    n_skip = Counter()
    for (img, les), p in mask.items():
        r = row(img, les, p, ("fgadr_mask" if "FGADR" in img else "ddr_mask"))
        if img in test_imgs: test.append(r)
        elif img in dev_imgs: dev.append(r)
        elif stem(img) in EXCLUDE: n_skip["mask"] += 1          # Stage-2 test stem -> drop from train
        elif p: pres_pool[les].append(r)
        else: hard_neg[les].append((img, r))                    # keep img for confounder ranking
    for les, img in idrid:
        if not excluded(img): pres_pool[les].append(row(img, les, True, "strong_mask"))
        elif stem(img) in EXCLUDE: n_skip["idrid"] += 1
    for img in g0_imgs:
        for les in MAIN4:
            r = row(img, les, False, "grade0_neg")
            if img in test_imgs: test.append(r)
            elif img in dev_imgs: dev.append(r)
            elif stem(img) in EXCLUDE: n_skip["g0"] += 1
            else: clean_neg[les].append(r)

    # TRAIN: per-lesion present cap; absent = hard-neg first then clean-neg, per-lesion cap.
    # For absent hard-negatives we additionally rank CONFOUNDER-rich images first (DR images that
    # carry a bright/other lesion the model tends to confuse with the target) -> sharper specificity.
    CONFOUNDERS = {"SE": {"EX", "HE"}, "MA": {"HE"}, "HE": {"EX"}, "EX": {"SE", "HE"}}
    train = []
    for les in MAIN4:
        pres = sorted(pres_pool[les], key=lambda x: h(x["meta"]["record_id"] + les))[:PRES_CAP[les]]
        conf = CONFOUNDERS.get(les, set())
        # rank: images carrying a known confounder lesion come first (rank 0), then the rest;
        # within each rank, deterministic hash ordering (reproducible, no Math.random equivalent).
        def hn_key(item):
            img, r = item
            has_conf = 0 if (has_other.get(img, set()) & conf) else 1
            return (has_conf, h(r["meta"]["record_id"] + "hn" + les))
        hn = [r for _, r in sorted(hard_neg[les], key=hn_key)]
        cn = sorted(clean_neg[les], key=lambda x: h(x["meta"]["record_id"] + "cn" + les))
        absn = (hn + cn)[:ABS_CAP[les]]
        train += pres + absn
    train = sorted(train, key=lambda x: h(x["meta"]["record_id"] + x["meta"]["lesion"] + x["meta"]["present_state"]))

    # leak assertion: no train image stem may be a Stage-2 test stem
    leaked = sorted({r["meta"]["record_id"] for r in train} & EXCLUDE)
    assert not leaked, f"LEAK: {len(leaked)} Stage-2 test stems present in train, e.g. {leaked[:5]}"

    for name, rows in [("stage1_5_v5_train", train), ("stage1_5_v5_dev", dev), ("stage1_5_v5_test", test)]:
        with open(f"{OUT}/{name}_sft.jsonl", "w") as fo:
            for r in rows: fo.write(json.dumps(r, ensure_ascii=False) + "\n")
    def st(rows):
        ls = Counter((r["meta"]["lesion"], r["meta"]["present_state"]) for r in rows)
        sr = Counter(r["meta"]["evidence_source"] for r in rows)
        return {"n": len(rows), "lesion_state": {f"{a}/{b}": n for (a, b), n in sorted(ls.items())},
                "by_source": dict(sr), "all_unseen": all(not r["meta"].get("seen_by_adapter1") for r in rows) if rows else None}
    meta = {"train": st(train), "dev": st(dev), "test": st(test), "pres_cap": PRES_CAP, "abs_cap": ABS_CAP,
            "n_test_mask_img": N_TEST_MASK_IMG, "n_test_g0_img": N_TEST_G0_IMG,
            "n_dev_mask_img": N_DEV_MASK_IMG, "n_dev_g0_img": N_DEV_G0_IMG,
            "stage2_test_stems_excluded_from_train": len(EXCLUDE), "train_skipped_as_heldout": dict(n_skip),
            "note": "v5 per-lesion rebalance + 3-way split. HE present-push (recall), MA/SE absent-push "
                    "(specificity), EX unchanged; SE/MA absent rank confounder-rich DR images first; real "
                    "masks only, NO grade-derived/RetSAM-SE weak labels; Stage-2 test stems excluded "
                    "(leak-free). TRAIN/DEV/TEST image-disjoint: DEV = checkpoint selection only, TEST = "
                    "v3/v4-identical, never touched during selection. Same single-lesion format + Adapter1 "
                    "warm-start."}
    json.dump(meta, open(f"{EXP}/data/stage1_5_v5_distribution.json", "w"), ensure_ascii=False, indent=1)
    print(json.dumps(meta, ensure_ascii=False, indent=1))

if __name__ == "__main__":
    main()
