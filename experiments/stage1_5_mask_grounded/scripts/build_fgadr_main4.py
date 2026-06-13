#!/usr/bin/env python3
"""Build FGADR MAIN4 (MA/HE/EX/SE) single-lesion structured perception data.

Proof-of-effect for Stage-1.5: uses the FGADR pixel masks that the current
Stage-1 WASTED (entered only as presence). Adds count_bucket + area_bucket
from the masks, plus reliable negatives (empty masks). Same English single-
lesion CoT schema as Adapter 1, so it is a warm-start continuation.

FGADR uses the original image directly (crop_box=[0,0,0,0]) so mask and image
are pixel-aligned -> no crop handling needed. Area buckets use FGADR-internal
per-lesion tertiles (cross-dataset area is not comparable). Image-disjoint
train/test split by image_id.
"""
from __future__ import annotations
import json, glob, hashlib, os
from collections import defaultdict, Counter
import numpy as np, cv2

ROOT = "/workspace/LLaMA-Factory"
VAL = f"{ROOT}/data/fundus_validated/validated_clean.jsonl"
FG = f"{ROOT}/data/FGADR/Seg-set"
OUT = f"{ROOT}/data/annotation"
EXP = "/workspace/stage1_5_experiment"  # =/sda/zgh/stage1_5_experiment
LI = json.load(open(f"{EXP}/scripts/lesion_info.json"))
MASKDIR = {"MA": "Microaneurysms_Masks", "HE": "Hemohedge_Masks", "EX": "HardExudate_Masks", "SE": "SoftExudate_Masks"}
MAIN4 = ["MA", "HE", "EX", "SE"]
MIN_PX = 5
TEST_FRAC = 0.15
# per-lesion train/test caps (present, absent)
TRAIN_CAP = {"present": 320, "absent": 160}
TEST_CAP = {"present": 100, "absent": 50}

def h(s):  # deterministic shuffle key
    return hashlib.md5(s.encode()).hexdigest()

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
    bw = (im >= 128).astype(np.uint8)  # strict binarize, drop antialias
    nz = int(bw.sum()); tot = im.size
    if nz == 0: return {"present": False, "count": 0, "frac": 0.0}
    n, _, stats, _ = cv2.connectedComponentsWithStats(bw, connectivity=8)
    cc = int((stats[1:, cv2.CC_STAT_AREA] >= MIN_PX).sum()) if n > 1 else 0
    return {"present": True, "count": cc, "frac": nz / tot}

def assistant(les, present, attrs):
    info = LI[les]; name = info["name"]
    if present:
        tgt = f"Visible findings are consistent with {info['visual']}."
        conf = info["positive_confounder"]
        at = ("Reliable coarse attributes: " + ", ".join(f"{k}={v}" for k, v in attrs.items()) + ".") if attrs else "No additional target-lesion attributes are reported."
        concl = f"Directly visible evidence supports the presence of {name}."
        state = "present"
    else:
        tgt = f"No reliable directly visible evidence consistent with {info['visual']} is identified."
        conf = info["negative_confounder"]; at = "No target-lesion attributes are reported because no reliable target evidence is present."
        concl = f"No reliable visual evidence supports the presence of {name}."; state = "absent"
    payload = {"task": "stage1_single_lesion_perception", "target_lesion": {"name": name, "abbreviation": les},
               "image_quality": "adequate", "evidence_state": state, "present": present, "attributes": attrs}
    return ("[Target Evidence]\n" + tgt + "\n\n[Confounder Assessment]\n" + conf +
            "\n\n[Attribute Summary]\n" + at + "\n\n[Conclusion]\n" + concl +
            "\n\n[Structured Output]\n" + json.dumps(payload, ensure_ascii=False, separators=(",", ":")))

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

def main():
    recs = [json.loads(l) for l in open(VAL) if l.strip()]
    fg = [r for r in recs if r.get("dataset") == "fgadr_seg"]
    # pass1: facts per (image,lesion)
    facts = {}
    fracs = defaultdict(list)
    for r in fg:
        iid = r["image_id"]; img = r["cropped_path"]
        for les in MAIN4:
            mp = f"{FG}/{MASKDIR[les]}/{iid}.png"
            f = mask_fact(mp)
            if f is None: continue
            facts[(iid, les)] = (img, f)
            if f["present"]: fracs[les].append(f["frac"])
    tert = {les: (float(np.percentile(v, 33)), float(np.percentile(v, 66))) if len(v) >= 3 else (0.0, 1.0) for les, v in fracs.items()}
    # split image_ids
    iids = sorted({iid for (iid, _) in facts})
    test_ids = set(i for i in iids if int(h("split" + i)[:8], 16) % 100 < TEST_FRAC * 100)
    # build samples
    def build(split_ids, caps):
        per = defaultdict(lambda: defaultdict(list))  # lesion->state->samples
        for (iid, les), (img, f) in facts.items():
            in_test = iid in test_ids
            if (split_ids == "test") != in_test: continue
            present = f["present"]
            attrs = {}
            if present:
                cb = bucket_count(f["count"]); ab = bucket_area(f["frac"], *tert[les])
                if cb: attrs["count_bucket"] = cb
                if ab: attrs["area_bucket"] = ab
            row = {"messages": [{"role": "system", "content": sys_p(les)}, {"role": "user", "content": usr_p(les)},
                                {"role": "assistant", "content": assistant(les, present, attrs)}],
                   "images": [img],
                   "meta": {"record_id": iid, "lesion": les, "present_state": "present" if present else "absent",
                            "evidence_level": "S0", "evidence_source": "fgadr_mask", "dataset": "fgadr_seg",
                            "count": f["count"], "count_bucket": attrs.get("count_bucket"),
                            "area_bucket": attrs.get("area_bucket"), "area_frac": round(f["frac"], 6)}}
            per[les]["present" if present else "absent"].append(row)
        out = []
        for les in MAIN4:
            for st in ("present", "absent"):
                pool = sorted(per[les][st], key=lambda x: h(x["meta"]["record_id"] + les + st))
                out.extend(pool[:caps[st]])
        return out
    train = sorted(build("train", TRAIN_CAP), key=lambda x: h(x["meta"]["record_id"] + x["meta"]["lesion"]))
    test = build("test", TEST_CAP)
    def stats(rows):
        c = Counter((r["meta"]["lesion"], r["meta"]["present_state"]) for r in rows)
        cb = Counter((r["meta"]["lesion"], r["meta"]["count_bucket"]) for r in rows if r["meta"]["present_state"] == "present")
        ab = Counter((r["meta"]["lesion"], r["meta"]["area_bucket"]) for r in rows if r["meta"]["present_state"] == "present")
        return {"n": len(rows), "lesion_state": {f"{a}/{b}": n for (a, b), n in sorted(c.items())},
                "count_bucket": {f"{a}/{b}": n for (a, b), n in sorted(cb.items())},
                "area_bucket": {f"{a}/{b}": n for (a, b), n in sorted(ab.items())}}
    for name, rows in [("fgadr_main4_proof_train", train), ("fgadr_main4_proof_test", test)]:
        with open(f"{OUT}/{name}_sft.jsonl", "w") as fo:
            for r in rows: fo.write(json.dumps(r, ensure_ascii=False) + "\n")
    meta = {"area_tertiles": tert, "n_images_total": len(iids), "n_test_images": len(test_ids),
            "train": stats(train), "test": stats(test), "min_px": MIN_PX, "count_buckets": "single/few(2-5)/many(>5)",
            "area_buckets": "FGADR per-lesion tertiles small/medium/large"}
    json.dump(meta, open(f"{EXP}/data/fgadr_main4_distribution.json", "w"), ensure_ascii=False, indent=1)
    print(json.dumps(meta, ensure_ascii=False, indent=1))

if __name__ == "__main__":
    main()
