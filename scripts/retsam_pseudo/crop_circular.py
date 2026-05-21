#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

# Allow running this file directly: add repo root to sys.path.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.retsam_pseudo.common import CropMetaRow, ensure_parent, iter_images, normalize_rel_to_data, write_jsonl
from scripts.retsam_pseudo.grades import (
    load_aptos_grades,
    load_aptos_grades_from_instructions,
    load_aptos_grades_from_annotation_dir,
    load_ddr_grading_from_txts,
    load_ddr_grading_grades,
)


def _largest_contour_bbox(gray: np.ndarray, thresh: int = 15) -> tuple[int, int, int, int] | None:
    _, binary = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    (cx, cy), radius = cv2.minEnclosingCircle(largest)
    cx, cy, radius = int(cx), int(cy), int(radius)
    x1 = max(0, cx - radius)
    y1 = max(0, cy - radius)
    x2 = min(gray.shape[1], cx + radius)
    y2 = min(gray.shape[0], cy + radius)
    if x2 <= x1 or y2 <= y1:
        return None
    return (x1, y1, x2, y2)


def crop_fundus_to_square_bbox(img_bgr: np.ndarray, thresh: int = 15) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    if img_bgr is None or img_bgr.ndim != 3:
        raise ValueError("expected BGR image")
    h, w = img_bgr.shape[:2]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    bbox = _largest_contour_bbox(gray, thresh=thresh) or (0, 0, w, h)
    x1, y1, x2, y2 = bbox
    cropped = img_bgr[y1:y2, x1:x2]
    return cropped, bbox


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Circular crop (remove black borders) for RetSAM inference input.")
    p.add_argument("--data-root", default="data", help="Project data directory (default: data)")
    p.add_argument("--dataset", choices=["aptos", "ddr_grading"], required=True)
    p.add_argument("--src-dir", default="", help="Override source image directory (relative to data-root unless absolute).")
    p.add_argument("--grade-csv", default="", help="Override grade CSV path (relative to data-root unless absolute).")
    p.add_argument("--out-dir", default="", help="Override output base directory (relative to data-root unless absolute).")
    p.add_argument("--thresh", type=int, default=15)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--write-meta", default="", help="Override crop_meta.jsonl output path.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    data_root = Path(args.data_root)

    if args.dataset == "aptos":
        # In this repo, APTOS 1024 processed PNGs are present under data/processed_images/aptos/.
        # If you have raw images elsewhere, pass --src-dir explicitly.
        src_dir = Path(args.src_dir) if args.src_dir else Path("processed_images/aptos")
        # Prefer existing grading CSV in this repo.
        grade_csv = Path(args.grade_csv) if args.grade_csv else Path("DR_grading.csv")
        out_base = Path(args.out_dir) if args.out_dir else Path("cropped/aptos")

        src_dir = src_dir if src_dir.is_absolute() else (data_root / src_dir)
        grade_csv = grade_csv if grade_csv.is_absolute() else (data_root / grade_csv)
        out_base = out_base if out_base.is_absolute() else (data_root / out_base)

        ann_dir = data_root / "annotation"
        grade_map = {}
        if grade_csv.is_file():
            grade_map = load_aptos_grades(grade_csv)

        # Heuristic safety: if the provided CSV doesn't match the actual image stems,
        # fall back to deriving grades from the instruction jsonl.
        if grade_map:
            stems = {p.stem for p in iter_images(src_dir)[:200]}
            overlap = sum((1 for s in stems if s in grade_map))
            if overlap == 0:
                grade_map = {}

        if not grade_map:
            # Merge from all aptos*_instructions*.jsonl to match processed_images/aptos/ contents.
            grade_map = load_aptos_grades_from_annotation_dir(ann_dir)
    else:
        # Default to DDR official folder layout already present in this repo.
        src_dir = Path(args.src_dir) if args.src_dir else Path("DDR-dataset/DR_grading")
        grade_csv = Path(args.grade_csv) if args.grade_csv else Path("DDR-dataset/DR_grading/train.txt")
        # Prefer DDR's split txts if present; otherwise fall back to CSV schema loader.
        train_txt = data_root / "DDR-dataset" / "DR_grading" / "train.txt"
        valid_txt = data_root / "DDR-dataset" / "DR_grading" / "valid.txt"
        test_txt = data_root / "DDR-dataset" / "DR_grading" / "test.txt"
        if train_txt.is_file() and valid_txt.is_file() and test_txt.is_file():
            grade_map = load_ddr_grading_from_txts(train_txt, valid_txt, test_txt)
        else:
            grade_map = load_ddr_grading_grades(grade_csv)
        out_base = Path(args.out_dir) if args.out_dir else Path("cropped/ddr_grading")

        src_dir = src_dir if src_dir.is_absolute() else (data_root / src_dir)
        grade_csv = grade_csv if grade_csv.is_absolute() else (data_root / grade_csv)
        out_base = out_base if out_base.is_absolute() else (data_root / out_base)

    meta_path = Path(args.write_meta) if args.write_meta else (out_base / "crop_meta.jsonl")

    imgs = iter_images(src_dir)
    if args.limit and args.limit > 0:
        imgs = imgs[: args.limit]

    rows: list[dict] = []
    created = 0
    skipped = 0
    no_grade = 0
    errors: list[dict] = []

    for img_path in imgs:
        image_id = img_path.stem
        if image_id not in grade_map:
            no_grade += 1
            continue
        grade = int(grade_map[image_id])
        split = "grade0" if grade == 0 else "grade1_4"

        out_path = out_base / split / f"{image_id}.png"
        if out_path.exists() and not args.overwrite:
            skipped += 1
            # Still record meta for downstream indexing.
        try:
            bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
            if bgr is None:
                raise ValueError("cv2.imread returned None")
            cropped, bbox = crop_fundus_to_square_bbox(bgr, thresh=args.thresh)
            if args.overwrite or (not out_path.exists()):
                ensure_parent(out_path)
                ok = cv2.imwrite(str(out_path), cropped)
                if not ok:
                    raise ValueError("cv2.imwrite failed")
                if not (out_path.exists() and out_path.stat().st_size > 0):
                    raise ValueError("written file missing/empty")
                created += 1
            row = CropMetaRow(
                image_id=image_id,
                src_path=normalize_rel_to_data(img_path, data_root),
                crop_box_xyxy=[int(x) for x in bbox],
                cropped_path=normalize_rel_to_data(out_path, data_root),
                grade=grade,
            )
            rows.append(row.__dict__)
        except Exception as e:
            errors.append({"image_id": image_id, "src_path": normalize_rel_to_data(img_path, data_root), "error": str(e)})

    write_jsonl(meta_path, rows, append=False)
    if errors:
        err_path = meta_path.with_suffix(".errors.jsonl")
        write_jsonl(err_path, errors, append=False)

    print(
        json.dumps(
            {
                "dataset": args.dataset,
                "src_dir": str(src_dir),
                "grade_csv": str(grade_csv),
                "out_base": str(out_base),
                "meta_path": str(meta_path),
                "created": created,
                "skipped": skipped,
                "no_grade": no_grade,
                "errors": len(errors),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

