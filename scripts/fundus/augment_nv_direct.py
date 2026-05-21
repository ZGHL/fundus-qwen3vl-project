#!/usr/bin/env python3
"""Offline image augmentation for the 19 unique G4 NV-direct fundus images.

Why: validated_clean.jsonl has only 19 unique NV-present G4 images (all from
FGADR strong labels). Pixel-level oversampling (×11 in v2) caused overfitting.
This script generates 8 augmented copies per source image to produce
19 + 19*8 = 171 unique pixel-level variants for v3 training.

Augmentations are CONSERVATIVE to preserve NV vascular structure:
- Brightness ±10%
- Contrast ±10%
- Color/saturation ±15%
- Random crop 0.85-1.00 + resize back
- Horizontal flip 50%
NOT applied: rotation > 5°, elastic transforms, heavy hue shift, vertical flip
(all would distort vessel geometry that NV detection depends on).

Outputs:
  data/cropped/_aug_v3_nv/<stem>_aug{1..8}.png   (~152 files)
  data/cropped/_aug_v3_nv/manifest.jsonl         (per-image transform record)

The v3 build script reads manifest.jsonl to attach the original image's NV
verdict / source / lesion metadata to each augmented copy.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

from PIL import Image, ImageEnhance

VALIDATED = Path("data/fundus_validated/validated_clean.jsonl")
OUT_DIR = Path("data/cropped/_aug_v3_nv")
DEFAULT_N_AUG = 8
TARGET_GRADE = 4


def find_nv_direct_g4(records_path: Path) -> list[dict[str, Any]]:
    seen: dict[str, dict[str, Any]] = {}
    with records_path.open() as f:
        for line in f:
            r = json.loads(line)
            if r.get("grade") != TARGET_GRADE:
                continue
            nv = r.get("lesions", {}).get("NV") or {}
            if nv.get("present") is not True:
                continue
            img = r.get("image_path")
            if img and img not in seen:
                seen[img] = r
    return list(seen.values())


def augment_image(src_path: Path, idx: int, seed_base: str) -> tuple[Image.Image, dict[str, Any]]:
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
    transforms["crop_origin_xy"] = [x, y]

    if rng.random() < 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        transforms["hflip"] = True
    else:
        transforms["hflip"] = False

    return img, transforms


def to_data_relative(path: Path) -> str:
    """Return path relative to 'data/' prefix (matches dataset_info media_dir)."""
    s = str(path)
    if s.startswith("data/"):
        return s[len("data/"):]
    return s


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-aug", type=int, default=DEFAULT_N_AUG,
                        help="Augmented copies per source image (default 8)")
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--records", type=Path, default=VALIDATED)
    parser.add_argument("--dry-run", action="store_true",
                        help="Don't write files, just print plan")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    records = find_nv_direct_g4(args.records)
    print(f"Found {len(records)} unique NV-present G4 images in {args.records.name}")

    if not records:
        print("Nothing to do.")
        return 0

    manifest: list[dict[str, Any]] = []
    written = 0
    skipped_missing = 0

    for record in records:
        src_path = Path(record["image_path"])
        record_id = record["record_id"]
        if not src_path.exists():
            print(f"  ⚠️  missing source: {src_path}")
            skipped_missing += 1
            continue

        stem = src_path.stem
        aug_paths: list[str] = []
        transforms_list: list[dict[str, Any]] = []

        for idx in range(1, args.n_aug + 1):
            out_path = args.out_dir / f"{stem}_aug{idx}.png"
            try:
                aug_img, transforms = augment_image(src_path, idx, record_id)
            except Exception as exc:
                print(f"  ⚠️  augment failed for {src_path} idx={idx}: {exc}")
                continue
            if not args.dry_run:
                aug_img.save(out_path, "PNG")
            aug_paths.append(to_data_relative(out_path))
            transforms_list.append(transforms)
            written += 1

        manifest.append({
            "record_id": record_id,
            "original_image_path": to_data_relative(src_path),
            "grade": record["grade"],
            "lesions_NV": record.get("lesions", {}).get("NV", {}),
            "augmented_paths": aug_paths,
            "transforms": transforms_list,
        })

    manifest_path = args.out_dir / "manifest.jsonl"
    if not args.dry_run:
        with manifest_path.open("w", encoding="utf-8") as f:
            for m in manifest:
                f.write(json.dumps(m, ensure_ascii=False, separators=(",", ":")) + "\n")

    print(f"\n✅ Done")
    print(f"  source images : {len(records)}")
    print(f"  augmented files written : {written}")
    print(f"  missing source : {skipped_missing}")
    print(f"  manifest : {manifest_path}")
    print(f"  pixel-unique pool size = original({len(records) - skipped_missing}) + augmented({written}) = {len(records) - skipped_missing + written}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
