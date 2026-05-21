#!/usr/bin/env python3
"""Mix L2+L3+L4 v5 SFT files into single mixed train/val.

Same structure as mix_v4 but reads v5 files (qualitative prose + quadrant + sequential).
"""
from __future__ import annotations
import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "fundus"))
from build_fundus_sft import read_jsonl, write_jsonl  # noqa: E402

V4_DIR = Path("/home/aim_lab/LLaMA-Factory/data/annotation_v4")

TRAIN_INPUTS = [
    ("L2", V4_DIR / "fundus_l2_v5_train_sft.jsonl"),
    ("L3", V4_DIR / "fundus_l3_v5_train_sft.jsonl"),
    ("L4", V4_DIR / "fundus_l4_v5_train_sft.jsonl"),
]
VAL_INPUTS = [
    ("L2", V4_DIR / "fundus_l2_v5_val_sft.jsonl"),
    ("L3", V4_DIR / "fundus_l3_v5_val_sft.jsonl"),
    ("L4", V4_DIR / "fundus_l4_v5_val_sft.jsonl"),
]


def collect_and_tag(inputs):
    out = []
    for layer, path in inputs:
        if not path.exists():
            print(f"  ⚠️  missing: {path}"); continue
        for item in read_jsonl(path):
            item["meta"]["layer"] = layer
            out.append(item)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    train_items = collect_and_tag(TRAIN_INPUTS)
    val_items = collect_and_tag(VAL_INPUTS)

    rng = random.Random(args.seed); rng.shuffle(train_items)
    rng2 = random.Random(args.seed + 1); rng2.shuffle(val_items)

    train_path = V4_DIR / "fundus_v5_mixed_train_sft.jsonl"
    val_path = V4_DIR / "fundus_v5_mixed_val_sft.jsonl"
    stats_path = V4_DIR / "fundus_v5_mixed_stats.json"

    if not args.dry_run:
        write_jsonl(train_path, train_items)
        write_jsonl(val_path, val_items)

    by_layer = Counter(it["meta"]["layer"] for it in train_items)
    by_task = Counter(it["meta"]["task"] for it in train_items)
    val_by_layer = Counter(it["meta"]["layer"] for it in val_items)

    stats = {
        "train_total": len(train_items),
        "val_total": len(val_items),
        "train_by_layer": dict(by_layer),
        "val_by_layer": dict(val_by_layer),
        "train_by_task": dict(by_task),
        "train_path": str(train_path),
        "val_path": str(val_path),
    }
    if not args.dry_run:
        stats_path.write_text(json.dumps(stats, indent=2, ensure_ascii=False))

    print("=== mixed v5 ===")
    print(f"train: {len(train_items)}  val: {len(val_items)}")
    print(f"  train by layer: {dict(by_layer)}")
    print(f"  train by task:  {dict(by_task)}")


if __name__ == "__main__":
    main()
