#!/usr/bin/env python3
"""Stage-1.5 v2 dataset for WARM-START (Adapter 1) with clean eval.

- TEST: carved from Adapter1-UNSEEN images only (whole-image disjoint) -> clean
  for both Adapter1 and the warm-started model (no memorization). Small (~N_TEST_IMG images).
- TRAIN: all other images = Adapter1-seen FGADR+DDR + remaining unseen + IDRiD(reused);
  per-lesion negatives capped at NEG_RATIO x present to boost weak lesions (MA/SE) by not
  drowning their scarce present samples.
- MAIN4 only; count_bucket single/few(2-5)/many(>5); area_bucket per-(dataset,lesion) tertiles.
- FGADR/DDR-seg masks read fresh (zero-crop); IDRiD reused from existing Stage-1 train.
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
N_TEST_IMG = 150          # unseen images held out for clean test (whole-image)
NEG_RATIO = 1.5           # per-lesion train negatives <= NEG_RATIO x present

def h(s): return hashlib.md5(s.encode()).hexdigest()
def stem(p): return os.path.splitext(os.path.basename(p))[0]
def bucket_count(n):
    if n <= 0: return None
    return "single" if n == 1 else "few" if n <= 5 else "many"
def bucket_area(v, lo, hi):
    if v <= 0: return None
    return "small" if v <= lo else "medium" if v <= hi else "large"
def mask_fact(p):
    im = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
    if im is None: return None
    bw = (im >= 128).astype(np.uint8); nz = int(bw.sum())
    if nz == 0: return {"present": False, "count": 0, "frac": 0.0}
    n, _, st, _ = cv2.connectedComponentsWithStats(bw, 8)
    cc = int((st[1:, cv2.CC_STAT_AREA] >= MIN_PX).sum()) if n > 1 else 0
    return {"present": True, "count": cc, "frac": nz / im.size}
def fgm(img, les): return f"{ROOT}/data/FGADR/Seg-set/{FG[les]}/{stem(img)}.png"
def ddrm(img, les):
    parts = img.split("/"); sp = parts[2]; lbl = "segmentation label" if sp == "valid" else "label"
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
def assistant(les, present, attrs):
    info = LI[les]; name = info["name"]
    if present:
        tgt = f"Visible findings are consistent with {info['visual']}."; conf = info["positive_confounder"]
        at = ("Reliable coarse attributes: " + ", ".join(f"{k}={v}" for k, v in attrs.items()) + ".") if attrs else "No additional target-lesion attributes are reported."
        concl = f"Directly visible evidence supports the presence of {name}."; state = "present"
    else:
        tgt = f"No reliable directly visible evidence consistent with {info['visual']} is identified."; conf = info["negative_confounder"]
        at = "No target-lesion attributes are reported because no reliable target evidence is present."
        concl = f"No reliable visual evidence supports the presence of {name}."; state = "absent"
    payload = {"task": "stage1_single_lesion_perception", "target_lesion": {"name": name, "abbreviation": les},
               "image_quality": "adequate", "evidence_state": state, "present": present, "attributes": attrs}
    return ("[Target Evidence]\n" + tgt + "\n\n[Confounder Assessment]\n" + conf + "\n\n[Attribute Summary]\n" + at +
            "\n\n[Conclusion]\n" + concl + "\n\n[Structured Output]\n" + json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
def row(img, les, f, tert, source):
    present = f["present"]; attrs = {}
    if present:
        cb = bucket_count(f["count"]); ab = bucket_area(f["frac"], *tert)
        if cb: attrs["count_bucket"] = cb
        if ab: attrs["area_bucket"] = ab
    return {"messages": [{"role": "system", "content": sys_p(les)}, {"role": "user", "content": usr_p(les)},
                         {"role": "assistant", "content": assistant(les, present, attrs)}],
            "images": [img],
            "meta": {"record_id": stem(img), "lesion": les, "present_state": "present" if present else "absent",
                     "evidence_level": "S0", "evidence_source": source, "dataset": source.replace("_mask", ""),
                     "count": f["count"], "count_bucket": attrs.get("count_bucket"), "area_bucket": attrs.get("area_bucket"),
                     "area_frac": round(f["frac"], 6), "seen_by_adapter1": stem(img) in A1SEEN}}

def main():
    global A1SEEN
    A1SEEN = set()
    for l in open(A1_TRAIN):
        m = json.loads(l)["meta"]; rid = str(m.get("image_group") or m.get("record_id", ""))
        A1SEEN.add(rid.split("::")[-1] if "::" in rid else rid)
    recs = [json.loads(l) for l in open(VAL) if l.strip()]
    imgs = {"fgadr": [r["cropped_path"] for r in recs if r["dataset"] == "fgadr_seg"],
            "ddr":   [r["cropped_path"] for r in recs if r["dataset"] == "ddr_seg"]}
    maskfn = {"fgadr": fgm, "ddr": ddrm}; source = {"fgadr": "fgadr_mask", "ddr": "ddr_mask"}
    # facts + tertiles
    facts = {}; fracs = defaultdict(list)
    for ds, lst in imgs.items():
        for img in lst:
            for les in MAIN4:
                f = mask_fact(maskfn[ds](img, les))
                if f is None: continue
                facts[(ds, img, les)] = f
                if f["present"]: fracs[(ds, les)].append(f["frac"])
    tert = {k: (float(np.percentile(v, 33)), float(np.percentile(v, 66))) if len(v) >= 3 else (0.0, 1.0) for k, v in fracs.items()}
    # choose TEST images from Adapter1-unseen (whole-image disjoint), deterministic
    unseen = sorted({(ds, img) for (ds, img, _) in facts if stem(img) not in A1SEEN}, key=lambda x: h("t" + x[1]))
    test_imgs = set(unseen[:N_TEST_IMG])
    # build
    test, train_pool = [], defaultdict(lambda: defaultdict(list))  # train_pool[les][state]
    for (ds, img, les), f in facts.items():
        r = row(img, les, f, tert.get((ds, les), (0.0, 1.0)), source[ds])
        if (ds, img) in test_imgs:
            test.append(r)
        else:
            train_pool[les]["present" if f["present"] else "absent"].append(r)
    # IDRiD reused -> train (present only)
    idrid = 0
    for l in open(A1_TRAIN):
        r = json.loads(l); m = r["meta"]
        if m.get("evidence_source") == "strong_mask_stage1_easy":
            m["evidence_level"] = "S0"; m.setdefault("dataset", "idrid"); m["seen_by_adapter1"] = True
            train_pool[m["lesion"]][m["present_state"]].append(r); idrid += 1
    # negative cap per lesion
    train = []
    for les in MAIN4 + ["IRMA", "NV"]:
        pres = train_pool[les]["present"]; absn = train_pool[les]["absent"]
        absn = sorted(absn, key=lambda x: h(x["meta"]["record_id"] + les))[: int(NEG_RATIO * len(pres)) if pres else len(absn)]
        train += pres + absn
    train = sorted(train, key=lambda x: h(x["meta"]["record_id"] + x["meta"]["lesion"] + x["meta"]["present_state"]))
    for name, rows in [("stage1_5_v2_train", train), ("stage1_5_v2_test", test)]:
        with open(f"{OUT}/{name}_sft.jsonl", "w") as fo:
            for r in rows: fo.write(json.dumps(r, ensure_ascii=False) + "\n")
    def st(rows):
        ls = Counter((r["meta"]["lesion"], r["meta"]["present_state"]) for r in rows)
        return {"n": len(rows), "by_dataset": dict(Counter(r["meta"].get("dataset") for r in rows)),
                "lesion_state": {f"{a}/{b}": n for (a, b), n in sorted(ls.items())},
                "all_unseen": all(not r["meta"].get("seen_by_adapter1") for r in rows) if rows else None}
    meta = {"train": st(train), "test": st(test), "idrid_reused": idrid, "n_test_images": len(test_imgs),
            "neg_ratio": NEG_RATIO, "note": "warm-start; TEST=Adapter1-unseen whole-image disjoint; train neg capped to boost MA/SE"}
    json.dump(meta, open(f"{EXP}/data/stage1_5_v2_distribution.json", "w"), ensure_ascii=False, indent=1)
    print(json.dumps(meta, ensure_ascii=False, indent=1))

if __name__ == "__main__":
    main()
