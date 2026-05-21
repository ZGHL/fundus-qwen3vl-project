#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

# Allow running this file directly: add repo root to sys.path.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.retsam_pseudo.common import CropMetaRow, iter_images, normalize_rel_to_data, write_jsonl  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build crop_meta.jsonl from an image directory (no cropping).")
    p.add_argument("--data-root", default="data")
    p.add_argument("--dataset", required=True, help="Output under data/cropped/<dataset>/crop_meta.jsonl")
    p.add_argument("--image-dir", required=True, help="Image directory (relative to data-root unless absolute)")
    p.add_argument("--grade-csv", default="", help="Optional grade csv (dataset-specific). If missing, grade=-1.")
    p.add_argument("--grade-csv-schema", default="auto", choices=["auto", "idrid", "fgadr_seg"])
    p.add_argument("--out-meta", default="", help="Override output meta path")
    p.add_argument("--limit", type=int, default=0)
    return p.parse_args()


def _resolve(data_root: Path, p: str) -> Path:
    pp = Path(p)
    return pp if pp.is_absolute() else (data_root / pp)


def _load_grade_map(csv_path: Path, schema: str) -> dict[str, int]:
    if not csv_path.is_file():
        return {}
    out: dict[str, int] = {}
    with csv_path.open("r", encoding="utf-8", errors="ignore") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not row:
                continue
            if schema == "idrid":
                # IDRiD grading CSV commonly uses columns: Image name, Retinopathy grade
                name = (row.get("Image name") or row.get("image") or row.get("Image") or "").strip()
                g = (row.get("Retinopathy grade") or row.get("grade") or row.get("Grade") or "").strip()
            elif schema == "fgadr_seg":
                # FGADR Seg-set label CSV: columns vary; try common ones.
                name = (row.get("img") or row.get("image") or row.get("Image") or row.get("name") or "").strip()
                g = (row.get("grade") or row.get("Grade") or row.get("DR_Grade") or row.get("dr_grade") or "").strip()
            else:
                # auto: try a few
                name = (row.get("Image name") or row.get("image") or row.get("img") or row.get("name") or "").strip()
                g = (row.get("Retinopathy grade") or row.get("grade") or row.get("Grade") or row.get("diagnosis") or "").strip()

            if not name:
                continue
            image_id = Path(name).stem
            try:
                out[image_id] = int(float(g))
            except Exception:
                continue
    return out


def main() -> int:
    args = parse_args()
    data_root = Path(args.data_root)
    if not data_root.is_absolute():
        data_root = _REPO_ROOT / data_root

    img_dir = _resolve(data_root, args.image_dir)
    grade_csv = _resolve(data_root, args.grade_csv) if args.grade_csv else None
    out_meta = _resolve(data_root, args.out_meta) if args.out_meta else (data_root / "cropped" / args.dataset / "crop_meta.jsonl")

    schema = str(args.grade_csv_schema)
    if schema == "auto":
        # cheap heuristic by path
        if grade_csv and "idrid" in grade_csv.as_posix().lower():
            schema = "idrid"
        elif grade_csv and "fgadr" in grade_csv.as_posix().lower():
            schema = "fgadr_seg"
        else:
            schema = "auto"

    grade_map = _load_grade_map(grade_csv, schema) if grade_csv else {}
    imgs = iter_images(img_dir)
    if args.limit and args.limit > 0:
        imgs = imgs[: args.limit]

    rows: list[dict[str, Any]] = []
    for p in imgs:
        image_id = p.stem
        grade = int(grade_map.get(image_id, -1))
        # No cropping in this step; cropped_path points to the same file.
        src_rel = normalize_rel_to_data(p, data_root)
        row = CropMetaRow(
            image_id=image_id,
            src_path=src_rel,
            crop_box_xyxy=[0, 0, 0, 0],
            cropped_path=src_rel,
            grade=grade,
        )
        rows.append(row.__dict__)

    write_jsonl(out_meta, rows, append=False)
    print(json.dumps({"dataset": args.dataset, "image_dir": str(img_dir), "out_meta": str(out_meta), "n": len(rows)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

