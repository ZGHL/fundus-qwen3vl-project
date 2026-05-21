#!/usr/bin/env python3
"""Mix L2 + L3 + L4 v4 SFT files into a single mixed training set.

Per-user spec (option A + I):
- Original counts kept: L2=5940 + L3=8209 + L4=6529 → ~20.7k train
- Single val file with all tasks combined → ~12.7k val
- Stable-shuffle interleave so SGD sees mixed tasks per batch
- Each item's meta.task field identifies origin (L2_*, L3_*, L4_dr_grading_v4)

Output:
  data/annotation_v4/fundus_v4_mixed_train_sft.jsonl    (~20.7k)
  data/annotation_v4/fundus_v4_mixed_val_sft.jsonl      (~12.7k)
  data/annotation_v4/fundus_v4_mixed_stats.json
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
    ("L2", V4_DIR / "fundus_l2_v4_train_sft.jsonl"),
    ("L3", V4_DIR / "fundus_l3_v4_train_sft.jsonl"),
    ("L4", V4_DIR / "fundus_l4_v4_train_sft.jsonl"),
]

VAL_INPUTS = [
    ("L2", V4_DIR / "fundus_l2_v4_val_sft.jsonl"),
    ("L3", V4_DIR / "fundus_l3_v4_val_sft.jsonl"),
    ("L4", V4_DIR / "fundus_l4_v4_val_sft.jsonl"),
]


def collect_and_tag(inputs: list[tuple[str, Path]]) -> list[dict]:
    """Read each input file; tag the meta with layer origin (L2/L3/L4) so we can
    re-split metrics later. Returns flattened list."""
    out: list[dict] = []
    for layer, path in inputs:
        if not path.exists():
            print(f"  ⚠️  missing: {path}")
            continue
        for item in read_jsonl(path):
            # Tag the meta if not already done
            item["meta"]["layer"] = layer
            out.append(item)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", type=Path, default=V4_DIR)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    train_items = collect_and_tag(TRAIN_INPUTS)
    val_items = collect_and_tag(VAL_INPUTS)

    # Stable interleave shuffle so SGD batches mix tasks
    rng = random.Random(args.seed)
    rng.shuffle(train_items)
    rng_v = random.Random(args.seed + 1)
    rng_v.shuffle(val_items)

    train_path = args.out_dir / "fundus_v4_mixed_train_sft.jsonl"
    val_path = args.out_dir / "fundus_v4_mixed_val_sft.jsonl"
    stats_path = args.out_dir / "fundus_v4_mixed_stats.json"

    if not args.dry_run:
        write_jsonl(train_path, train_items)
        write_jsonl(val_path, val_items)

    # Stats
    train_by_layer = Counter(it["meta"]["layer"] for it in train_items)
    val_by_layer = Counter(it["meta"]["layer"] for it in val_items)
    train_by_task = Counter(it["meta"]["task"] for it in train_items)
    val_by_task = Counter(it["meta"]["task"] for it in val_items)

    stats = {
        "train_total": len(train_items),
        "val_total": len(val_items),
        "train_by_layer": dict(train_by_layer),
        "val_by_layer": dict(val_by_layer),
        "train_by_task": dict(train_by_task),
        "val_by_task": dict(val_by_task),
        "train_path": str(train_path),
        "val_path": str(val_path),
    }
    if not args.dry_run:
        stats_path.write_text(json.dumps(stats, indent=2, ensure_ascii=False))

    print("=== mixed v4 dataset ===")
    print(f"train: {len(train_items)}")
    print(f"  by layer: {dict(train_by_layer)}")
    print(f"  by task : {dict(train_by_task)}")
    print(f"val:   {len(val_items)}")
    print(f"  by layer: {dict(val_by_layer)}")
    print(f"  by task : {dict(val_by_task)}")


if __name__ == "__main__":
    main()
