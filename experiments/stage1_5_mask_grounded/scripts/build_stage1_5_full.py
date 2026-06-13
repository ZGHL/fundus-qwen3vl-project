#!/usr/bin/env python3
"""Build FULL Stage-1.5 single-lesion structured perception dataset (all negatives).

Strong pixel masks, MAIN4 (MA/HE/EX/SE), ALL present + ALL negatives:
  - FGADR Seg-set  (fresh; zero-crop; mask=Seg-set/<Folder>/<id>.png)
  - DDR-seg        (fresh; zero-crop; mask=lesion_segmentation/<split>/<labeldir>/<LES>/<id>.tif)
  - IDRiD          (REUSED from existing fundus_stage1_en_cot_train S0 strong_mask samples;
                    image id<->mask name digit mismatch, so reuse proven-correct samples)
Area buckets use per-(dataset,lesion) tertiles (cross-dataset area not comparable -> relative).
count bucket single/few(2-5)/many(>5). Image-disjoint 15% held-out test per fresh dataset.
English single-lesion CoT schema (warm-start compatible with Adapter 1).
"""
from __future__ import annotations
import json, glob, hashlib, os
from collections import defaultdict, Counter
import numpy as np, cv2

ROOT = "/workspace/LLaMA-Factory"
VAL = f"{ROOT}/data/fundus_validated/validated_clean.jsonl"
OUT = f"{ROOT}/data/annotation"
EXP = "/workspace/stage1_5_experiment"
TRAIN_S1 = "/workspace/_anno_current/data/annotation_v4/fundus_stage1_en_cot_train_sft.jsonl"
LI = json.load(open(f"{EXP}/scripts/lesion_info.json"))
MAIN4 = ["MA", "HE", "EX", "SE"]
FG_FOLDER = {"MA": "Microaneurysms_Masks", "HE": "Hemohedge_Masks", "EX": "HardExudate_Masks", "SE": "SoftExudate_Masks"}
MIN_PX = 5
TEST_FRAC = 0.15

def h(s): return hashlib.md5(s.encode()).hexdigest()
def bucket_count(n):
    if n <= 0: return None
    if n == 1: return "single"
    if n <= 5: return "few"
    return "many"
def bucket_area(v, lo, hi):
    if v <= 0: return None
    return "small" if v <= lo else "medium" if v <= hi else "large"

def mask_fact(path):
    im = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if im is None: return None
    bw = (im >= 128).astype(np.uint8); nz = int(bw.sum()); tot = im.size
    if nz == 0: return {"present": False, "count": 0, "frac": 0.0}
    n, _, st, _ = cv2.connectedComponentsWithStats(bw, connectivity=8)
    cc = int((st[1:, cv2.CC_STAT_AREA] >= MIN_PX).sum()) if n > 1 else 0
    return {"present": True, "count": cc, "frac": nz / tot}

def fg_mask(img, les):  # img = FGADR/Seg-set/Original_Images/<id>.png
    iid = os.path.splitext(os.path.basename(img))[0]
    return f"{ROOT}/data/FGADR/Seg-set/{FG_FOLDER[les]}/{iid}.png"
def ddr_mask(img, les):  # img = DDR-dataset/lesion_segmentation/<split>/image/<id>.jpg
    parts = img.split("/"); split = parts[2]; iid = os.path.splitext(parts[-1])[0]
    lbldir = "segmentation label" if split == "valid" else "label"
    return f"{ROOT}/data/DDR-dataset/lesion_segmentation/{split}/{lbldir}/{les}/{iid}.tif"

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

def make_row(img, les, fact, tert, source):
    present = fact["present"]; attrs = {}
    if present:
        cb = bucket_count(fact["count"]); ab = bucket_area(fact["frac"], *tert)
        if cb: attrs["count_bucket"] = cb
        if ab: attrs["area_bucket"] = ab
    return {"messages": [{"role": "system", "content": sys_p(les)}, {"role": "user", "content": usr_p(les)},
                         {"role": "assistant", "content": assistant(les, present, attrs)}],
            "images": [img],
            "meta": {"record_id": os.path.splitext(os.path.basename(img))[0], "lesion": les,
                     "present_state": "present" if present else "absent", "evidence_level": "S0",
                     "evidence_source": source, "dataset": source.replace("_mask", ""),
                     "count": fact["count"], "count_bucket": attrs.get("count_bucket"),
                     "area_bucket": attrs.get("area_bucket"), "area_frac": round(fact["frac"], 6)}}

EXCLUDE = set()
_exf = f"{EXP}/data/exclude_ids.txt"
if os.path.exists(_exf):
    EXCLUDE = {l.strip() for l in open(_exf) if l.strip()}

def build_fresh(records, maskfn, source, train, test):
    # pass1 facts + tertiles
    facts = {}; fracs = defaultdict(list)
    for img in records:
        iid0 = os.path.splitext(os.path.basename(img))[0]
        if iid0 in EXCLUDE:  # drop Gold-Dev/Test + locked image_groups (benchmark validity)
            continue
        for les in MAIN4:
            f = mask_fact(maskfn(img, les))
            if f is None: continue
            facts[(img, les)] = f
            if f["present"]: fracs[les].append(f["frac"])
    tert = {les: (float(np.percentile(v, 33)), float(np.percentile(v, 66))) if len(v) >= 3 else (0.0, 1.0) for les, v in fracs.items()}
    for (img, les), f in facts.items():
        iid = os.path.splitext(os.path.basename(img))[0]
        dest = test if int(h("split" + iid)[:8], 16) % 100 < TEST_FRAC * 100 else train
        dest.append(make_row(img, les, f, tert.get(les, (0.0, 1.0)), source))
    return tert

def main():
    recs = [json.loads(l) for l in open(VAL) if l.strip()]
    fg_imgs = [r["cropped_path"] for r in recs if r.get("dataset") == "fgadr_seg"]
    ddr_imgs = [r["cropped_path"] for r in recs if r.get("dataset") == "ddr_seg"]
    train, test = [], []
    tert_fg = build_fresh(fg_imgs, fg_mask, "fgadr_mask", train, test)
    tert_ddr = build_fresh(ddr_imgs, ddr_mask, "ddr_mask", train, test)
    # IDRiD S0 from existing stage1 train -> held out ENTIRELY as external cross-dataset test
    idrid_ext = []
    if os.path.exists(TRAIN_S1):
        for l in open(TRAIN_S1):
            r = json.loads(l); m = r.get("meta", {})
            if m.get("evidence_source") == "strong_mask_stage1_easy":
                r["meta"]["evidence_level"] = "S0"; r["meta"].setdefault("dataset", "idrid")
                idrid_ext.append(r)
    idrid_reused = len(idrid_ext)
    with open(f"{OUT}/stage1_5_idrid_external_sft.jsonl", "w") as fo:
        for r in idrid_ext: fo.write(json.dumps(r, ensure_ascii=False) + "\n")
    train = sorted(train, key=lambda x: h(x["meta"]["record_id"] + x["meta"]["lesion"] + x["meta"]["present_state"]))
    def stats(rows):
        ls = Counter((r["meta"]["lesion"], r["meta"]["present_state"]) for r in rows)
        ds = Counter(r["meta"].get("dataset") for r in rows)
        return {"n": len(rows), "by_dataset": dict(ds), "lesion_state": {f"{a}/{b}": n for (a, b), n in sorted(ls.items())}}
    for name, rows in [("stage1_5_full_train", train), ("stage1_5_full_test", test)]:
        with open(f"{OUT}/{name}_sft.jsonl", "w") as fo:
            for r in rows: fo.write(json.dumps(r, ensure_ascii=False) + "\n")
    meta = {"train": stats(train), "test": stats(test), "idrid_reused": idrid_reused,
            "area_tertiles": {"fgadr": tert_fg, "ddr_seg": tert_ddr},
            "min_px": MIN_PX, "count_buckets": "single/few(2-5)/many(>5)",
            "area_buckets": "per-(dataset,lesion) tertiles small/medium/large",
            "note": "FGADR+DDR-seg fresh (all present+negatives), IDRiD reused S0; zero-crop all."}
    json.dump(meta, open(f"{EXP}/data/stage1_5_full_distribution.json", "w"), ensure_ascii=False, indent=1)
    print(json.dumps(meta, ensure_ascii=False, indent=1))

if __name__ == "__main__":
    main()
