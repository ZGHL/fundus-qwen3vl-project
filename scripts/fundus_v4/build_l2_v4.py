#!/usr/bin/env python3
"""Build L2 v4: anatomy perception SFT (laterality + CDR + vessel metrics).

Design:
- Same image_id-level 80:20 split as build_l3_v4.py (consistency across L2/L3/L4)
- 3 sub-tasks per record (when biomarker is available):
    L2_laterality: eye_side (left/right) — binary
    L2_cdr:        cup-disc ratio (numeric + bucket)
    L2_vessel:     A/V ratio + tortuosity (with abstention when vessel_qc=false)
- 4-section English CoT mirroring L3 v4: [Observe] [Evidence] [Conclusion] [JSON]
- Abstention training built-in: vessel_qc=false → model learns to output "unknown"

Output:
  data/annotation_v4/fundus_l2_v4_train_sft.jsonl
  data/annotation_v4/fundus_l2_v4_val_sft.jsonl
  data/annotation_v4/fundus_l2_v4_stats.json
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "fundus"))
from build_fundus_sft import hbucket, read_jsonl, sft, write_jsonl  # noqa: E402

VALIDATED = Path("/home/aim_lab/LLaMA-Factory/data/fundus_validated/validated_clean.jsonl")
OUT_DIR = Path("/home/aim_lab/LLaMA-Factory/data/annotation_v4")
VAL_PCT = 20


# ---------------------------- split assignment (shared with L3) ----------------------------

def assign_splits(records: list[dict]) -> dict[str, str]:
    eval_iids = set()
    for r in records:
        if r.get("dataset") == "idrid" and r.get("split") == "test":
            eval_iids.add(r["image_id"])
        if r.get("dataset") == "ddr_seg" and r.get("split") in {"valid", "test"}:
            eval_iids.add(r["image_id"])
    iid_split: dict[str, str] = {}
    for r in records:
        iid = r["image_id"]
        if iid in eval_iids:
            iid_split[iid] = "eval"
        elif iid not in iid_split:
            iid_split[iid] = "val" if hbucket(iid) < VAL_PCT else "train"
    return iid_split


# ---------------------------- bucket logic ----------------------------

def cdr_bucket(v: float) -> str:
    if v < 0.40:
        return "normal"
    if v < 0.50:
        return "mild_elevation"
    if v < 0.65:
        return "moderate_elevation"
    return "glaucoma_suspicion"


def av_bucket(v: float) -> str:
    if v < 0.65:
        return "low"  # narrowed arteries
    if v < 0.85:
        return "normal"
    return "elevated"


def tort_bucket(v: float) -> str:
    if v < 0.20:
        return "normal"
    if v < 0.40:
        return "mild"
    if v < 0.60:
        return "moderate"
    return "severe"


def src_tag(source: str) -> str:
    return {
        "validated_retsam": "retsam",
        "strong_mask_stage1_easy": "strong_mask",
        "fgadr_lesion_only_sft_v3": "strong_mask",
    }.get(source, source)


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


def build_laterality(record: dict, split: str) -> dict | None:
    bm = record.get("biomarkers", {}).get("eye_side")
    if not isinstance(bm, dict) or not bm.get("valid"):
        return None
    side = bm.get("value")
    if side not in {"left", "right"}:
        return None
    src = src_tag(bm.get("source", "unknown"))
    nasal_side = "left" if side == "right" else "right"
    temporal_side = "right" if side == "right" else "left"
    assistant = (
        "[Observe] Determine laterality from optic disc and fovea positions. No DR grading.\n\n"
        f"[Evidence] Optic disc on the {nasal_side} side of the image, fovea on the {temporal_side} side; "
        f"eye_side={side}; source={src}.\n\n"
        f"[Conclusion] This is a {side} eye.\n\n"
        f"[JSON] {json.dumps({'task': 'L2_laterality', 'eye_side': side, 'source': bm.get('source', 'unknown')}, ensure_ascii=False, separators=(',', ':'))}"
    )
    meta = {
        "record_id": record["record_id"],
        "image_id": record["image_id"],
        "dataset": record["dataset"],
        "task": "L2_laterality",
        "split": split,
        "eye_side": side,
        "source": bm.get("source"),
    }
    return sft(LATERALITY_SYSTEM, LATERALITY_USER, assistant, record["image_path"], meta)


# ---------------------------- L2-CDR ----------------------------

CDR_SYSTEM = (
    "You are a fundus image analyst. Estimate the cup-disc ratio (CDR) only. "
    "Identify optic disc and cup, then describe the vertical cup-to-disc ratio. "
    "Do not output a DR grade or mention lesions."
)
CDR_USER = (
    "Examine the optic disc in this fundus image and estimate the cup-disc ratio. "
    "Output in the order: [Observe] -> [Evidence] -> [Conclusion] -> [JSON]."
)


def build_cdr(record: dict, split: str) -> dict | None:
    bm = record.get("biomarkers", {}).get("cdr")
    if not isinstance(bm, dict) or not bm.get("valid"):
        return None
    v = bm.get("value")
    if v is None:
        return None
    bucket = cdr_bucket(v)
    src = src_tag(bm.get("source", "unknown"))
    od_qc = record.get("biomarkers", {}).get("od_qc_flag", True)
    qc_phrase = "optic disc clearly visible" if od_qc else "optic disc partially visible"
    assistant = (
        "[Observe] Locate the optic disc and cup; estimate the vertical cup-disc ratio. No DR grading.\n\n"
        f"[Evidence] {qc_phrase}; CDR ≈ {v:.2f} ({bucket.replace('_', ' ')}); source={src}.\n\n"
        f"[Conclusion] CDR is approximately {v:.2f}, classified as {bucket.replace('_', ' ')}.\n\n"
        f"[JSON] {json.dumps({'task': 'L2_cdr', 'cdr': round(v, 4), 'cdr_bucket': bucket, 'source': bm.get('source', 'unknown')}, ensure_ascii=False, separators=(',', ':'))}"
    )
    meta = {
        "record_id": record["record_id"],
        "image_id": record["image_id"],
        "dataset": record["dataset"],
        "task": "L2_cdr",
        "split": split,
        "cdr_value": round(v, 4),
        "cdr_bucket": bucket,
        "source": bm.get("source"),
    }
    return sft(CDR_SYSTEM, CDR_USER, assistant, record["image_path"], meta)


# ---------------------------- L2-vessel ----------------------------

VESSEL_SYSTEM = (
    "You are a fundus image analyst. Assess vessel A/V ratio and tortuosity. "
    "If vessel segmentation quality is poor, abstain and output unknown. "
    "Do not output a DR grade or mention lesions."
)
VESSEL_USER = (
    "Examine the retinal vessels and report A/V ratio and tortuosity, "
    "or abstain if vessel quality is poor. "
    "Output in the order: [Observe] -> [Evidence] -> [Conclusion] -> [JSON]."
)


def build_vessel(record: dict, split: str) -> dict | None:
    bm = record.get("biomarkers", {})
    av = bm.get("av_ratio")
    tort = bm.get("tortuosity")
    vqc = bm.get("vessel_qc_flag", False)

    if not isinstance(av, dict) or not isinstance(tort, dict):
        return None

    # Two cases: valid vessel measurements OR abstention (qc_failed)
    if av.get("valid") and tort.get("valid"):
        av_v = av.get("value")
        tort_v = tort.get("value")
        if av_v is None or tort_v is None:
            return None
        av_b = av_bucket(av_v)
        tort_b = tort_bucket(tort_v)
        src = src_tag(av.get("source", "unknown"))
        assistant = (
            "[Observe] Compare artery and vein calibre; assess vessel curvature for tortuosity. No DR grading.\n\n"
            f"[Evidence] vessel_qc=ok; A/V ratio={av_v:.2f} ({av_b}); tortuosity={tort_v:.3f} ({tort_b}); source={src}.\n\n"
            f"[Conclusion] A/V ratio {av_b}; tortuosity {tort_b}.\n\n"
            f"[JSON] {json.dumps({'task': 'L2_vessel', 'av_ratio': round(av_v, 4), 'av_bucket': av_b, 'tortuosity': round(tort_v, 4), 'tortuosity_bucket': tort_b, 'source': av.get('source', 'unknown')}, ensure_ascii=False, separators=(',', ':'))}"
        )
        meta = {
            "record_id": record["record_id"],
            "image_id": record["image_id"],
            "dataset": record["dataset"],
            "task": "L2_vessel",
            "split": split,
            "vessel_state": "valid",
            "av_value": round(av_v, 4),
            "av_bucket": av_b,
            "tortuosity_value": round(tort_v, 4),
            "tortuosity_bucket": tort_b,
        }
    else:
        # Abstention training: vessel_qc poor → output unknown
        reason = av.get("cleaned_reason") or "vessel_qc_failed_or_missing"
        assistant = (
            "[Observe] Attempt to assess vessel A/V ratio and tortuosity. No DR grading.\n\n"
            f"[Evidence] vessel_qc=failed ({reason}); vessel segmentation is unreliable; cannot measure A/V or tortuosity.\n\n"
            "[Conclusion] A/V ratio and tortuosity cannot be reliably determined.\n\n"
            f"[JSON] {json.dumps({'task': 'L2_vessel', 'av_ratio': None, 'av_bucket': 'unknown', 'tortuosity': None, 'tortuosity_bucket': 'unknown', 'reason': reason}, ensure_ascii=False, separators=(',', ':'))}"
        )
        meta = {
            "record_id": record["record_id"],
            "image_id": record["image_id"],
            "dataset": record["dataset"],
            "task": "L2_vessel",
            "split": split,
            "vessel_state": "abstain",
            "reason": reason,
        }
    return sft(VESSEL_SYSTEM, VESSEL_USER, assistant, record["image_path"], meta)


# ---------------------------- per-task sampling ----------------------------

def sample_laterality(items: list[dict], seed: int, target: int) -> list[dict]:
    """Balance 1:1 by side."""
    rng = random.Random(seed)
    by_side = defaultdict(list)
    for it in items:
        by_side[it["meta"]["eye_side"]].append(it)
    for s in by_side.values():
        rng.shuffle(s)
    n_per_side = min(target // 2, min(len(by_side["left"]), len(by_side["right"])))
    return by_side["left"][:n_per_side] + by_side["right"][:n_per_side]


def sample_cdr(items: list[dict], seed: int, target: int) -> list[dict]:
    """Stratify by cdr_bucket (4 buckets)."""
    rng = random.Random(seed)
    by_bucket = defaultdict(list)
    for it in items:
        by_bucket[it["meta"]["cdr_bucket"]].append(it)
    for b in by_bucket.values():
        rng.shuffle(b)
    per_bucket = target // 4
    out = []
    for b in ("normal", "mild_elevation", "moderate_elevation", "glaucoma_suspicion"):
        out.extend(by_bucket[b][:per_bucket])
    return out


def sample_vessel(items: list[dict], seed: int, target: int) -> list[dict]:
    """Mix valid (3 AV buckets × 4 tort buckets) and abstain. Roughly 1:1 valid:abstain."""
    rng = random.Random(seed)
    valid = [it for it in items if it["meta"]["vessel_state"] == "valid"]
    abstain = [it for it in items if it["meta"]["vessel_state"] == "abstain"]
    rng.shuffle(valid)
    rng.shuffle(abstain)
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

    # Build all candidate items per split per task
    builders = [
        ("L2_laterality", build_laterality, sample_laterality, args.laterality_target),
        ("L2_cdr", build_cdr, sample_cdr, args.cdr_target),
        ("L2_vessel", build_vessel, sample_vessel, args.vessel_target),
    ]

    train_items: list[dict] = []
    val_items: list[dict] = []
    stats: dict = {"val_pct": VAL_PCT, "per_task": {}}

    for task_name, builder, sampler, target in builders:
        # Build raw item pools
        train_pool = [it for r in train_recs if (it := builder(r, "train"))]
        val_pool = [it for r in val_recs if (it := builder(r, "val"))]

        # Sample train (balanced), val stays natural
        sampled_train = sampler(train_pool, args.seed + hash(task_name) % 100000, target)
        train_items.extend(sampled_train)
        val_items.extend(val_pool)

        # Per-task breakdown
        breakdown_train = Counter()
        breakdown_val = Counter()
        for it in sampled_train:
            key = (it["meta"].get("eye_side") or it["meta"].get("cdr_bucket") or it["meta"].get("vessel_state"))
            breakdown_train[key] += 1
        for it in val_pool:
            key = (it["meta"].get("eye_side") or it["meta"].get("cdr_bucket") or it["meta"].get("vessel_state"))
            breakdown_val[key] += 1
        stats["per_task"][task_name] = {
            "train_pool_size": len(train_pool),
            "train_sampled": len(sampled_train),
            "val_pool_size": len(val_pool),
            "train_breakdown": dict(breakdown_train),
            "val_breakdown": dict(breakdown_val),
        }

    rng = random.Random(args.seed)
    rng.shuffle(train_items)
    rng.shuffle(val_items)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    train_path = args.out_dir / "fundus_l2_v4_train_sft.jsonl"
    val_path = args.out_dir / "fundus_l2_v4_val_sft.jsonl"
    stats_path = args.out_dir / "fundus_l2_v4_stats.json"

    if not args.dry_run:
        write_jsonl(train_path, train_items)
        write_jsonl(val_path, val_items)

    stats["train_total"] = len(train_items)
    stats["val_total"] = len(val_items)
    stats["train_path"] = str(train_path)
    stats["val_path"] = str(val_path)

    if not args.dry_run:
        stats_path.write_text(json.dumps(stats, indent=2, ensure_ascii=False))

    print("=== L2 v4 build summary ===")
    print(f"train: {len(train_items)} items")
    print(f"val:   {len(val_items)} items")
    for tname, s in stats["per_task"].items():
        print(f"  {tname:<18} train={s['train_sampled']:>4} (of {s['train_pool_size']:>4})  "
              f"val={s['val_pool_size']:>4}  "
              f"breakdown_train={dict(s['train_breakdown'])}")


if __name__ == "__main__":
    main()
