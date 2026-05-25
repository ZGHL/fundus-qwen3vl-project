#!/usr/bin/env python3
"""Build full English decoupled lesion-perception SFT data.

This keeps the current four-section English CoT format, but changes the data
construction objective from the older fixed 600/600 control set to a fuller
RetSAM-cleaned lesion-perception set:
  - use only records marked usable_for.L3
  - split by image_id to avoid leakage
  - use direct present/absent labels only
  - use all available positives after the existing sparse NV/IRMA augmentation
  - cap negatives so every lesion has a reasonable positive/negative ratio
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_l3_v6 as base  # noqa: E402

OUT_DIR = Path("data/annotation_v4")
VERSION = "fundus_lesion_perception_en_cot_full"
TRAIN_NAME = "fundus_lesion_perception_en_cot_full_train_sft.jsonl"
VAL_NAME = "fundus_lesion_perception_en_cot_full_val_sft.jsonl"
NV_LOCKED_EVAL_NAME = "fundus_lesion_perception_en_cot_nv_locked_eval_sft.jsonl"
STATS_NAME = "fundus_lesion_perception_en_cot_full_stats.json"

# Ratio caps are intentionally lesion-specific. Dense lesions may use up to
# 2x positives when negatives are the bottleneck, which keeps more RetSAM
# positives without letting a task become a one-sided detector. NV keeps extra
# negatives because the positive pool is extremely small even after the
# existing conservative image transforms.
POSITIVE_RATIO_CAP = {
    "MA": 2.0,
    "HE": 2.0,
    "EX": 2.0,
    "SE": 2.0,
    "IRMA": 1.0,
    "NV": 1.0,
}
NEGATIVE_RATIO_CAP = {
    "MA": 1.0,
    "HE": 1.0,
    "EX": 1.0,
    "SE": 1.0,
    "IRMA": 1.25,
    "NV": 4.0,
}

EXPLICIT_TARGETS = {
    # MA has enough high-quality positives, but clean negatives are the
    # bottleneck. Use a 2:1 exposure target and lightly repeat negatives.
    "MA": {"present": 600, "absent": 300},
    # NV positives are rare. The current pool is 25 originals plus 100
    # conservative transformed images; expose it more than once, but do not
    # inflate it back to the old 600-row control set.
    "NV": {"present": 300, "absent": 600},
}
NV_LOCKED_POSITIVES = 5
NV_LOCKED_NEGATIVES = 100


def cycle_to_n(items: list, n: int, rng: random.Random) -> list:
    if n <= 0 or not items:
        return []
    shuffled = list(items)
    rng.shuffle(shuffled)
    out = []
    while len(out) < n:
        block = list(shuffled)
        rng.shuffle(block)
        out.extend(block)
    return out[:n]


def sample_full_balanced(pool: dict[str, list], lesion_key: str, seed: int) -> list:
    rng = random.Random(seed)
    pos = list(pool.get(base.PRESENT, []))
    neg = list(pool.get(base.ABSENT, []))
    rng.shuffle(pos)

    if lesion_key in EXPLICIT_TARGETS:
        target = EXPLICIT_TARGETS[lesion_key]
        pos_items = cycle_to_n(pos, target["present"], rng)
        neg_items = cycle_to_n(base.priority_sample(neg, len(neg), rng), target["absent"], rng)
        return pos_items + neg_items

    pos_ratio = POSITIVE_RATIO_CAP[lesion_key]
    neg_ratio = NEGATIVE_RATIO_CAP[lesion_key]
    pos_n = min(len(pos), max(1, int(round(len(neg) * pos_ratio)))) if neg else len(pos)
    neg_n = min(len(neg), max(1, int(round(pos_n * neg_ratio)))) if pos_n else 0
    return pos[:pos_n] + base.priority_sample(neg, neg_n, rng)


def summarize_items(items: list[dict[str, Any]]) -> dict[str, Any]:
    counts = Counter()
    unique_images: dict[tuple[str, str], set[str]] = {}
    sources = Counter()
    for item in items:
        meta = item.get("meta", {})
        lesion = meta.get("lesion")
        state = meta.get("present_state")
        key = (lesion, state)
        counts[key] += 1
        unique_images.setdefault(key, set()).add((item.get("images") or [""])[0])
        sources[(lesion, state, meta.get("source_tag"))] += 1
    return {
        "counts": {str(k): v for k, v in sorted(counts.items(), key=lambda x: str(x[0]))},
        "unique_images": {str(k): len(v) for k, v in sorted(unique_images.items(), key=lambda x: str(x[0]))},
        "sources": {str(k): v for k, v in sorted(sources.items(), key=lambda x: str(x[0]))},
    }


def make_lesion_perception_item(
    lesion_key: str,
    state: str,
    lesion: dict[str, Any],
    record: dict[str, Any],
    split: str,
    quad_idx,
) -> dict[str, Any]:
    item = base.make_sft_item(lesion_key, state, lesion, record, split, quad_idx)
    task_name = f"lesion_perception_{lesion_key}"
    item["meta"]["task"] = task_name
    for message in item["messages"]:
        if message.get("role") != "assistant":
            continue
        message["content"] = (
            message["content"]
            .replace(f'"task":"L3_{lesion_key}"', f'"task":"{task_name}"')
            .replace(
                "No final DR grade is assigned in this L3 task.",
                "No final DR grade is assigned in this single-lesion perception task.",
            )
        )
    return item


def select_nv_locked_eval(records: list[dict[str, Any]], seed: int) -> tuple[list[tuple[str, dict[str, Any], dict[str, Any]]], set[str]]:
    rng = random.Random(seed)
    positives = []
    negatives = []
    for record in records:
        ev = base.state_for_lesion(record, "NV")
        if ev is None:
            continue
        state, lesion = ev
        if state == base.PRESENT:
            positives.append((state, lesion, record))
        elif state == base.ABSENT:
            negatives.append((state, lesion, record))

    rng.shuffle(positives)
    locked_pos = positives[:NV_LOCKED_POSITIVES]
    locked_iids = {item[2]["image_id"] for item in locked_pos}

    # Prefer direct strong-mask negatives and avoid reusing positive locked images.
    neg_candidates = [item for item in base.priority_sample(negatives, len(negatives), rng) if item[2]["image_id"] not in locked_iids]
    locked_neg = []
    for item in neg_candidates:
        iid = item[2]["image_id"]
        if iid in locked_iids:
            continue
        locked_neg.append(item)
        locked_iids.add(iid)
        if len(locked_neg) >= NV_LOCKED_NEGATIVES:
            break
    return locked_pos + locked_neg, locked_iids


def build(args: argparse.Namespace) -> dict[str, Any]:
    records = list(base.read_jsonl(args.validated))
    iid_split = base.assign_splits(records, args.val_pct)
    train_records = [r for r in records if iid_split.get(r["image_id"]) == "train"]
    val_records = [r for r in records if iid_split.get(r["image_id"]) == "val"]
    eval_records = [r for r in records if iid_split.get(r["image_id"]) == "eval"]
    locked_nv, locked_iids = select_nv_locked_eval(train_records, args.seed + 1009)
    train_records = [r for r in train_records if r["image_id"] not in locked_iids]

    quad_idx = base.load_quadrant_index()
    aug_map = base.load_sparse_aug_manifest()
    train_pools, train_raw, train_sources = base.build_pools(train_records, aug_map=aug_map)
    val_pools, val_raw, val_sources = base.build_pools(val_records)

    train_items = []
    val_items = []
    nv_locked_items = [
        make_lesion_perception_item("NV", state, lesion, record, "locked_eval", quad_idx)
        for state, lesion, record in locked_nv
    ]
    stats: dict[str, Any] = {
        "version": VERSION,
        "design": [
            "english_four_section_decoupled_lesion_perception",
            "usable_for_L3_only",
            "image_id_level_split",
            "direct_present_absent_only",
            "use_all_available_positive_pool_after_sparse_aug",
            "negative_caps_by_lesion_to_control_pos_neg_ratio",
            "not_a_dr_grade_task",
        ],
        "positive_ratio_cap": POSITIVE_RATIO_CAP,
        "negative_ratio_cap": NEGATIVE_RATIO_CAP,
        "explicit_targets": EXPLICIT_TARGETS,
        "input": str(args.validated),
        "val_pct": args.val_pct,
        "seed": args.seed,
        "records": {
            "all": len(records),
            "train_images": len(train_records),
            "val_images": len(val_records),
            "eval_images_excluded": len(eval_records),
            "nv_locked_eval_images": len(locked_iids),
            "usable_L3_all": sum(bool(r.get("usable_for", {}).get("L3")) for r in records),
        },
        "nv_locked_eval": summarize_items(nv_locked_items),
        "per_lesion": {},
    }

    for idx, lesion_key in enumerate(base.LESIONS):
        sampled_train = sample_full_balanced(train_pools[lesion_key], lesion_key, args.seed + idx)
        for state, lesion, record in sampled_train:
            train_items.append(make_lesion_perception_item(lesion_key, state, lesion, record, "train", quad_idx))

        # Validation remains natural direct present/absent, not class-balanced,
        # so it can expose the real distribution while train stays controlled.
        for state in (base.PRESENT, base.ABSENT):
            for ev_state, lesion, record in val_pools[lesion_key].get(state, []):
                val_items.append(make_lesion_perception_item(lesion_key, ev_state, lesion, record, "val", quad_idx))

        train_sample_counts = Counter(item[0] for item in sampled_train)
        stats["per_lesion"][lesion_key] = {
            "train_raw_states": dict(train_raw[lesion_key]),
            "val_raw_states": dict(val_raw[lesion_key]),
            "train_sources": dict(train_sources[lesion_key]),
            "val_sources": dict(val_sources[lesion_key]),
            "train_available_present": len(train_pools[lesion_key].get(base.PRESENT, [])),
            "train_available_absent": len(train_pools[lesion_key].get(base.ABSENT, [])),
            "train_sampled_present": train_sample_counts.get(base.PRESENT, 0),
            "train_sampled_absent": train_sample_counts.get(base.ABSENT, 0),
            "val_present": len(val_pools[lesion_key].get(base.PRESENT, [])),
            "val_absent": len(val_pools[lesion_key].get(base.ABSENT, [])),
        }

    rng = random.Random(args.seed)
    rng.shuffle(train_items)
    rng.shuffle(val_items)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    train_path = args.out_dir / TRAIN_NAME
    val_path = args.out_dir / VAL_NAME
    nv_locked_path = args.out_dir / NV_LOCKED_EVAL_NAME
    stats_path = args.out_dir / STATS_NAME
    stats["train_total"] = len(train_items)
    stats["val_total"] = len(val_items)
    stats["train_summary"] = summarize_items(train_items)
    stats["val_summary"] = summarize_items(val_items)

    if not args.dry_run:
        base.write_jsonl(train_path, train_items)
        base.write_jsonl(val_path, val_items)
        base.write_jsonl(nv_locked_path, nv_locked_items)
        stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print("=== lesion perception full build summary ===")
    print(f"records: all={len(records)} usable_L3={stats['records']['usable_L3_all']}")
    print(
        f"split images: train={len(train_records)} val={len(val_records)} "
        f"eval_excluded={len(eval_records)} nv_locked={len(locked_iids)}"
    )
    print(f"items: train={len(train_items)} val={len(val_items)} nv_locked={len(nv_locked_items)}")
    for lesion_key, s in stats["per_lesion"].items():
        print(
            f"  {lesion_key:<5} train={s['train_sampled_present']:>4}+{s['train_sampled_absent']:>4} "
            f"val={s['val_present']:>4}+{s['val_absent']:>4} "
            f"available_train={s['train_available_present']:>4}+{s['train_available_absent']:>4}"
        )
    print(f"train: {train_path}")
    print(f"val  : {val_path}")
    print(f"nv locked eval: {nv_locked_path}")
    print(f"stats: {stats_path}")
    return stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validated", type=Path, default=base.VALIDATED)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--val-pct", type=int, default=base.VAL_PCT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    build(args)


if __name__ == "__main__":
    main()
