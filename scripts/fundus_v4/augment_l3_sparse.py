#!/usr/bin/env python3
"""Offline conservative augmentation for L3 sparse-class positives (NV + IRMA).

Why:
  L3 v4 has only 25 unique NV positives and 78 unique IRMA positives in the train
  split. Naive duplication would severely overfit at the pixel level. Conservative
  photometric + geometric augmentation expands the pixel-level pool while preserving
  the vascular geometry that lesion identification depends on.

Transforms (validated in v3's augment_nv_direct.py to preserve NV structure):
  - Brightness ±10%
  - Contrast ±10%
  - Color/saturation ±15%
  - Random crop 0.85-1.00 + resize back to original size
  - Horizontal flip 50%
NOT applied: rotation >5°, elastic transforms, vertical flip, heavy hue shift.

Outputs:
  data/cropped/_aug_v4_sparse/<lesion>/<stem>_aug{i}.png
  data/cropped/_aug_v4_sparse/manifest.jsonl  (per-image transform record)

Counts:
  NV:    25 originals × 4 = 100 augmented (→ 125 pixel-unique pool)
  IRMA:  78 originals × 8 = 624 augmented (→ 702 pixel-unique pool)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from PIL import Image, ImageEnhance

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "fundus"))
from build_fundus_sft import hbucket, read_jsonl  # noqa: E402

VALIDATED = Path("/home/aim_lab/LLaMA-Factory/data/fundus_validated/validated_clean.jsonl")
OUT_BASE = Path("/home/aim_lab/LLaMA-Factory/data/cropped/_aug_v4_sparse")
VAL_PCT = 20

# Per-lesion augmentation count
AUG_COUNT = {
    "NV": 4,    # 25 × 4 = 100 → 125 pool (matches v3 strategy with NV ceiling caveat)
    "IRMA": 8,  # 78 × 8 = 624 → 702 pool (aggressive: IRMA is recall-bottlenecked, room to grow)
}


# ---------------------------- split (matches build_l3_v4) ----------------------------

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


# ---------------------------- augmentation ----------------------------

def augment_image(src_path: Path, idx: int, seed_base: str) -> tuple[Image.Image, dict[str, Any]]:
    """Apply conservative transforms deterministically (same seed → same output)."""
    rng = random.Random(f"{seed_base}::aug{idx}")
    img = Image.open(src_path).convert("RGB")

    transforms: dict[str, Any] = {}

    b = 1.0 + rng.uniform(-0.10, 0.10)
    img = ImageEnhance.Brightness(img).enhance(b)
    transforms["brightness"] = round(b, 3)

    c = 1.0 + rng.uniform(-0.10, 0.10)
    img = ImageEnhance.Contrast(img).enhance(c)
    transforms["contrast"] = round(c, 3)

    s = 1.0 + rng.uniform(-0.15, 0.15)
    img = ImageEnhance.Color(img).enhance(s)
    transforms["color"] = round(s, 3)

    w, h = img.size
    crop_ratio = rng.uniform(0.85, 1.0)
    cw, ch = int(w * crop_ratio), int(h * crop_ratio)
    x = rng.randint(0, max(1, w - cw))
    y = rng.randint(0, max(1, h - ch))
    img = img.crop((x, y, x + cw, y + ch)).resize((w, h), Image.LANCZOS)
    transforms["crop_ratio"] = round(crop_ratio, 3)
    transforms["crop_xy"] = [x, y]

    if rng.random() < 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        transforms["hflip"] = True
    else:
        transforms["hflip"] = False

    return img, transforms


def to_data_relative(path: Path) -> str:
    """Return path relative to data/ root (no leading 'data/').

    Handles both relative ('data/cropped/...') and absolute
    ('/home/aim_lab/LLaMA-Factory/data/cropped/...') input forms.
    """
    s = str(path)
    abs_prefix = "/home/aim_lab/LLaMA-Factory/"
    if s.startswith(abs_prefix):
        s = s[len(abs_prefix):]
    if s.startswith("data/"):
        s = s[len("data/"):]
    return s


# ---------------------------- main ----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lesions", nargs="+", default=list(AUG_COUNT.keys()),
                    help="Lesions to augment (default: NV IRMA)")
    ap.add_argument("--out-base", type=Path, default=OUT_BASE)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    records = list(read_jsonl(VALIDATED))
    iid_split = assign_splits(records)
    train_recs = [r for r in records if iid_split[r["image_id"]] == "train"]

    manifest: list[dict[str, Any]] = []
    written_per_lesion: Counter = Counter()
    skipped_missing = 0

    for lesion in args.lesions:
        n_aug = AUG_COUNT.get(lesion)
        if n_aug is None:
            print(f"  ⚠️  unknown lesion {lesion}, skip")
            continue

        # Collect train positives for this lesion
        positives = []
        seen_iids = set()
        for r in train_recs:
            if not r.get("usable_for", {}).get("L3"):
                continue
            les = r.get("lesions", {}).get(lesion)
            if not isinstance(les, dict):
                continue
            if les.get("present") is not True:
                continue
            iid = r["image_id"]
            if iid in seen_iids:
                continue  # dedupe by image_id
            seen_iids.add(iid)
            positives.append(r)

        out_dir = args.out_base / lesion
        if not args.dry_run:
            out_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n=== {lesion}: {len(positives)} unique train positives ===")

        for record in positives:
            src_path = Path(record["image_path"])
            if not src_path.is_absolute():
                src_path = Path("/home/aim_lab/LLaMA-Factory") / src_path
            if not src_path.exists():
                print(f"  ⚠️  missing source: {src_path}")
                skipped_missing += 1
                continue

            stem = src_path.stem
            aug_paths: list[str] = []
            transforms_list: list[dict[str, Any]] = []

            for idx in range(1, n_aug + 1):
                out_path = out_dir / f"{stem}_aug{idx}.png"
                try:
                    aug_img, transforms = augment_image(src_path, idx, record["record_id"])
                except Exception as exc:
                    print(f"  ⚠️  augment failed {src_path} idx={idx}: {exc}")
                    continue
                if not args.dry_run:
                    aug_img.save(out_path, "PNG")
                aug_paths.append(to_data_relative(out_path))
                transforms_list.append(transforms)
                written_per_lesion[lesion] += 1

            manifest.append({
                "lesion": lesion,
                "record_id": record["record_id"],
                "image_id": record["image_id"],
                "dataset": record["dataset"],
                "original_image_path": to_data_relative(src_path) if str(src_path).startswith("/home") else record["image_path"],
                "lesion_meta": record.get("lesions", {}).get(lesion, {}),
                "augmented_paths": aug_paths,
                "transforms": transforms_list,
            })

    manifest_path = args.out_base / "manifest.jsonl"
    if not args.dry_run:
        args.out_base.mkdir(parents=True, exist_ok=True)
        with manifest_path.open("w", encoding="utf-8") as f:
            for m in manifest:
                f.write(json.dumps(m, ensure_ascii=False, separators=(",", ":")) + "\n")

    print()
    print("=== summary ===")
    for lesion in args.lesions:
        print(f"  {lesion}: original={sum(1 for m in manifest if m['lesion'] == lesion)}, "
              f"augmented_files={written_per_lesion[lesion]}, "
              f"total_pool={sum(1 for m in manifest if m['lesion'] == lesion) + written_per_lesion[lesion]}")
    print(f"  skipped (missing source): {skipped_missing}")
    print(f"  manifest: {manifest_path}")


if __name__ == "__main__":
    main()
