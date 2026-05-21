#!/usr/bin/env python3
"""Build L2 v5: anatomy CoT with qualitative visual prose (#2) + abstain phenomenology (#4).

Changes from L2 v4:
  - CDR: replace `CDR ≈ 0.66` raw-float prose with cup-rim visual descriptor per bucket;
    raw value moves to JSON only.
  - A/V ratio: replace `A/V ratio=0.93` with comparative artery-vein calibre phrasing.
  - Tortuosity: replace `tortuosity=0.109` with vessel-curve descriptor.
  - Vessel abstain: replace metadata-only "vessel_qc=failed (vessel_qc_failed_or_missing)"
    with 5-option visual phenomenology, deterministically chosen per record_id.
  - Laterality: minor — add the "disc is nasal to fovea" rule sentence to [Observe].
  - JSON schema: unchanged (downstream parsers keep working).

Output:
  data/annotation_v4/fundus_l2_v5_train_sft.jsonl
  data/annotation_v4/fundus_l2_v5_val_sft.jsonl
  data/annotation_v4/fundus_l2_v5_stats.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "fundus"))
from build_fundus_sft import hbucket, read_jsonl, sft, write_jsonl  # noqa: E402

VALIDATED = Path("/home/aim_lab/LLaMA-Factory/data/fundus_validated/validated_clean.jsonl")
OUT_DIR = Path("/home/aim_lab/LLaMA-Factory/data/annotation_v4")
VAL_PCT = 20


# ---------------------------- v5 visual prose maps ----------------------------

CUP_PROSE = {
    "normal":              "cup occupies clearly less than half the disc; rim is thick",
    "mild_elevation":      "cup occupies about half the disc; rim still well-preserved",
    "moderate_elevation":  "cup occupies more than half the disc; rim is thinning",
    "glaucoma_suspicion":  "cup occupies most of the disc; rim is markedly thin",
}

AV_PROSE = {
    "low":      "arteries appear visibly narrower than adjacent veins",
    "normal":   "artery and vein calibres look balanced",
    "elevated": "arteries appear nearly as wide as (or wider than) veins",
}

TORT_PROSE = {
    "normal":   "vessels follow gentle smooth curves",
    "mild":     "occasional vessel windings visible",
    "moderate": "several vessel segments show notable twisting",
    "severe":   "many vessels are highly tortuous",
}

# v5 #4: 5 abstention prose, deterministic selection per record
ABSTAIN_PROSE = [
    "Image appears blurry — vessel boundaries cannot be traced reliably",
    "Illumination is uneven (one side underexposed) — fine vessel structure not visible",
    "Image is heavily peripheral-cropped — central vessels not adequately captured",
    "Reflections or media opacity obscure the retinal vessels",
    "Vessel branches are too small to resolve at this image resolution",
]


def cdr_bucket(v: float) -> str:
    if v < 0.40: return "normal"
    if v < 0.50: return "mild_elevation"
    if v < 0.65: return "moderate_elevation"
    return "glaucoma_suspicion"

def av_bucket(v: float) -> str:
    if v < 0.65: return "low"
    if v < 0.85: return "normal"
    return "elevated"

def tort_bucket(v: float) -> str:
    if v < 0.20: return "normal"
    if v < 0.40: return "mild"
    if v < 0.60: return "moderate"
    return "severe"

def src_tag(s: str) -> str:
    return {
        "validated_retsam": "retsam",
        "strong_mask_stage1_easy": "strong_mask",
        "fgadr_lesion_only_sft_v3": "strong_mask",
    }.get(s, s)

def abstain_choice(record_id: str) -> str:
    h = int(hashlib.sha1(record_id.encode()).hexdigest()[:8], 16)
    return ABSTAIN_PROSE[h % len(ABSTAIN_PROSE)]


# ---------------------------- shared split ----------------------------

def assign_splits(records):
    eval_iids = set()
    for r in records:
        if r.get("dataset") == "idrid" and r.get("split") == "test":
            eval_iids.add(r["image_id"])
        if r.get("dataset") == "ddr_seg" and r.get("split") in {"valid", "test"}:
            eval_iids.add(r["image_id"])
    iid_split = {}
    for r in records:
        iid = r["image_id"]
        if iid in eval_iids:
            iid_split[iid] = "eval"
        elif iid not in iid_split:
            iid_split[iid] = "val" if hbucket(iid) < VAL_PCT else "train"
    return iid_split


# ---------------------------- L2-laterality ----------------------------

LATERALITY_SYSTEM = (
    "You are a fundus image analyst. Determine the eye laterality (left vs right) only. "
    "Base your judgement on the spatial relation between optic disc and fovea. "
    "Do not output a DR grade or mention lesions."
)
LATERALITY_USER = (
    "Examine this fundus image and decide if it is from the left or right eye. "
    "Output in the order: [Observe] -> [Evidence] -> [Conclusion] -> [JSON]."
)


def build_laterality(record, split):
    bm = record.get("biomarkers", {}).get("eye_side")
    if not isinstance(bm, dict) or not bm.get("valid"): return None
    side = bm.get("value")
    if side not in {"left", "right"}: return None
    nasal_side = "left" if side == "right" else "right"
    temporal_side = "right" if side == "right" else "left"
    src = src_tag(bm.get("source", "unknown"))

    # v5.2: Findings / Impression / Result bullet format
    assistant = (
        "[Findings]\n"
        f"- Optic disc: located on the {nasal_side.upper()} side of the image\n"
        f"- Fovea: located on the {temporal_side.upper()} side\n"
        "- Spatial rule: optic disc is normally NASAL to the fovea\n\n"
        "[Impression]\n"
        f"- Disc-nasal in the {nasal_side} field indicates a {side} eye.\n\n"
        f"[Result] task=L2_laterality | eye_side={side} | source={src}"
    )
    meta = {"record_id": record["record_id"], "image_id": record["image_id"],
            "dataset": record["dataset"], "task": "L2_laterality", "split": split,
            "eye_side": side, "source": bm.get("source")}
    return sft(LATERALITY_SYSTEM, LATERALITY_USER, assistant, record["image_path"], meta)


# ---------------------------- L2-CDR (v5 #2: qualitative prose) ----------------------------

CDR_SYSTEM = (
    "You are a fundus image analyst. Estimate the cup-disc ratio (CDR) only. "
    "Identify the optic disc rim and the central cup; describe what fraction of the disc the cup occupies. "
    "Do not output a DR grade or mention lesions."
)
CDR_USER = (
    "Examine the optic disc in this fundus image and estimate the cup-disc ratio category. "
    "Output in the order: [Observe] -> [Evidence] -> [Conclusion] -> [JSON]."
)


def build_cdr(record, split):
    bm = record.get("biomarkers", {}).get("cdr")
    if not isinstance(bm, dict) or not bm.get("valid"): return None
    v = bm.get("value")
    if v is None: return None
    bucket = cdr_bucket(v)
    src = src_tag(bm.get("source", "unknown"))
    od_qc = record.get("biomarkers", {}).get("od_qc_flag", True)
    qc_phrase = "optic disc clearly visible" if od_qc else "optic disc partially visible"

    # v5.2: Findings / Impression / Result bullet format
    assistant = (
        "[Findings]\n"
        f"- Optic disc: {qc_phrase}\n"
        f"- Cup: {CUP_PROSE[bucket]}\n\n"
        "[Impression]\n"
        f"- CDR is in the {bucket.replace('_',' ')} range.\n\n"
        f"[Result] task=L2_cdr | bucket={bucket} | source={src}"
    )
    meta = {"record_id": record["record_id"], "image_id": record["image_id"],
            "dataset": record["dataset"], "task": "L2_cdr", "split": split,
            "cdr_value": round(v, 4), "cdr_bucket": bucket, "source": bm.get("source")}
    return sft(CDR_SYSTEM, CDR_USER, assistant, record["image_path"], meta)


# ---------------------------- L2-vessel (v5 #2 + #4) ----------------------------

VESSEL_SYSTEM = (
    "You are a fundus image analyst. Assess vessel A/V calibre ratio and tortuosity. "
    "Compare artery and vein widths qualitatively; note vessel curvature. "
    "If vessel structures are not adequately resolvable, abstain and output unknown. "
    "Do not output a DR grade or mention lesions."
)
VESSEL_USER = (
    "Examine the retinal vessels and report A/V calibre category and tortuosity, "
    "or abstain if vessel structures are not adequately resolvable. "
    "Output in the order: [Observe] -> [Evidence] -> [Conclusion] -> [JSON]."
)


def build_vessel(record, split):
    bm = record.get("biomarkers", {})
    av = bm.get("av_ratio")
    tort = bm.get("tortuosity")
    if not isinstance(av, dict) or not isinstance(tort, dict): return None

    if av.get("valid") and tort.get("valid"):
        av_v = av.get("value"); tort_v = tort.get("value")
        if av_v is None or tort_v is None: return None
        av_b = av_bucket(av_v); tort_b = tort_bucket(tort_v)
        src = src_tag(av.get("source", "unknown"))

        # v5.2: Findings / Impression / Result bullet format
        assistant = (
            "[Findings]\n"
            "- Vessel quality: ok\n"
            f"- A/V calibre: {AV_PROSE[av_b]}\n"
            f"- Tortuosity: {TORT_PROSE[tort_b]}\n\n"
            "[Impression]\n"
            f"- A/V ratio category: {av_b}.\n"
            f"- Tortuosity: {tort_b}.\n\n"
            f"[Result] task=L2_vessel | av={av_b} | tort={tort_b} | source={src}"
        )
        meta = {"record_id": record["record_id"], "image_id": record["image_id"],
                "dataset": record["dataset"], "task": "L2_vessel", "split": split,
                "vessel_state": "valid", "av_value": round(av_v, 4), "av_bucket": av_b,
                "tortuosity_value": round(tort_v, 4), "tortuosity_bucket": tort_b}
    else:
        # v5.2: abstain with phenomenological cue + bullet
        reason = av.get("cleaned_reason") or "vessel_qc_failed_or_missing"
        abstain_observation = abstain_choice(record["record_id"])
        assistant = (
            "[Findings]\n"
            f"- Vessel quality: poor — {abstain_observation}\n\n"
            "[Impression]\n"
            "- Vessel measurements abstained due to inadequate image quality.\n\n"
            "[Result] task=L2_vessel | av=unknown | tort=unknown | reason=image_quality_poor"
        )
        meta = {"record_id": record["record_id"], "image_id": record["image_id"],
                "dataset": record["dataset"], "task": "L2_vessel", "split": split,
                "vessel_state": "abstain", "reason": reason,
                "abstain_prose": abstain_observation[:40]}
    return sft(VESSEL_SYSTEM, VESSEL_USER, assistant, record["image_path"], meta)


# ---------------------------- per-task sampling (same as v4) ----------------------------

def sample_laterality(items, seed, target):
    rng = random.Random(seed)
    by_side = defaultdict(list)
    for it in items: by_side[it["meta"]["eye_side"]].append(it)
    for s in by_side.values(): rng.shuffle(s)
    n_per = min(target // 2, min(len(by_side["left"]), len(by_side["right"])))
    return by_side["left"][:n_per] + by_side["right"][:n_per]

def sample_cdr(items, seed, target):
    rng = random.Random(seed)
    by_b = defaultdict(list)
    for it in items: by_b[it["meta"]["cdr_bucket"]].append(it)
    for b in by_b.values(): rng.shuffle(b)
    per = target // 4
    out = []
    for b in ("normal","mild_elevation","moderate_elevation","glaucoma_suspicion"):
        out.extend(by_b[b][:per])
    return out

def sample_vessel(items, seed, target):
    rng = random.Random(seed)
    valid = [it for it in items if it["meta"]["vessel_state"] == "valid"]
    abstain = [it for it in items if it["meta"]["vessel_state"] == "abstain"]
    rng.shuffle(valid); rng.shuffle(abstain)
    half = target // 2
    return valid[:half] + abstain[:half]


# ---------------------------- main ----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--laterality-target", type=int, default=2400)
    ap.add_argument("--cdr-target", type=int, default=2000)
    ap.add_argument("--vessel-target", type=int, default=2000)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    records = list(read_jsonl(VALIDATED))
    iid_split = assign_splits(records)
    train_recs = [r for r in records if iid_split[r["image_id"]] == "train"]
    val_recs = [r for r in records if iid_split[r["image_id"]] == "val"]

    builders = [
        ("L2_laterality", build_laterality, sample_laterality, args.laterality_target),
        ("L2_cdr",         build_cdr,        sample_cdr,        args.cdr_target),
        ("L2_vessel",      build_vessel,     sample_vessel,     args.vessel_target),
    ]

    train_items = []; val_items = []
    stats = {"v5_changes": ["qualitative_prose", "abstain_phenomenology"],
             "val_pct": VAL_PCT, "per_task": {}}
    for tname, builder, sampler, target in builders:
        train_pool = [it for r in train_recs if (it := builder(r, "train"))]
        val_pool = [it for r in val_recs if (it := builder(r, "val"))]
        sampled = sampler(train_pool, args.seed + hash(tname) % 100000, target)
        train_items.extend(sampled)
        val_items.extend(val_pool)
        stats["per_task"][tname] = {
            "train_pool_size": len(train_pool),
            "train_sampled":   len(sampled),
            "val_pool_size":   len(val_pool),
        }

    rng = random.Random(args.seed)
    rng.shuffle(train_items); rng.shuffle(val_items)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    train_path = args.out_dir / "fundus_l2_v5_train_sft.jsonl"
    val_path   = args.out_dir / "fundus_l2_v5_val_sft.jsonl"
    stats_path = args.out_dir / "fundus_l2_v5_stats.json"

    if not args.dry_run:
        write_jsonl(train_path, train_items)
        write_jsonl(val_path, val_items)

    stats["train_total"] = len(train_items)
    stats["val_total"] = len(val_items)
    stats["train_path"] = str(train_path)
    stats["val_path"] = str(val_path)
    if not args.dry_run:
        stats_path.write_text(json.dumps(stats, indent=2, ensure_ascii=False))

    print("=== L2 v5 build summary ===")
    print(f"train: {len(train_items)}  val: {len(val_items)}")
    for tname, s in stats["per_task"].items():
        print(f"  {tname:<18} train={s['train_sampled']:>4} (of {s['train_pool_size']})  val={s['val_pool_size']}")


if __name__ == "__main__":
    main()
