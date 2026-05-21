#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

# Allow running this file directly: add repo root to sys.path.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.stage1_easy.progress import default_progress_path, mark_running, update_progress


@dataclass(frozen=True)
class PreprocessMeta:
    src_rel: str
    dst_rel: str
    orig_hw: tuple[int, int]
    crop_box_xyxy: tuple[int, int, int, int]  # in original image coords
    target_size: int

    def to_dict(self) -> dict:
        return {
            "src_rel": self.src_rel,
            "dst_rel": self.dst_rel,
            "orig_hw": list(self.orig_hw),
            "crop_box_xyxy": list(self.crop_box_xyxy),
            "target_size": self.target_size,
        }


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


def preprocess_fundus_rgb(
    img_rgb: np.ndarray, output_size: int = 1024, dataset: str = "generic"
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """
    Returns:
      - preprocessed RGB uint8 image, shape (output_size, output_size, 3)
      - crop box (x1, y1, x2, y2) in original coords
    """
    if img_rgb.ndim != 3 or img_rgb.shape[2] != 3:
        raise ValueError(f"expected RGB HxWx3, got shape={img_rgb.shape}")

    orig_h, orig_w = img_rgb.shape[:2]
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    bbox = _largest_contour_bbox(gray, thresh=15)
    if bbox is None:
        bbox = (0, 0, orig_w, orig_h)
    x1, y1, x2, y2 = bbox
    cropped = img_rgb[y1:y2, x1:x2]

    img = cv2.resize(cropped, (output_size, output_size), interpolation=cv2.INTER_AREA)

    # Light contrast enhancement (keep colors): CLAHE on L channel in LAB.
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l = lab[:, :, 0]
    ds = (dataset or "").lower()
    if ds == "fgadr":
        # FGADR quality varies; reduce noise + be less aggressive.
        l = cv2.GaussianBlur(l, (3, 3), 0)
        clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
    else:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(l)
    img = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    # Circular mask outside filled with 128.
    mask = np.zeros((output_size, output_size), dtype=np.uint8)
    cv2.circle(mask, (output_size // 2, output_size // 2), output_size // 2 - 2, 255, -1)
    img = img.copy()
    for c in range(3):
        img[:, :, c] = np.where(mask == 0, 128, img[:, :, c])

    return img, bbox


def transform_coordinate_xy(coord_orig: tuple[int, int], crop_box_xyxy: tuple[int, int, int, int], target_size: int = 1024) -> tuple[int, int]:
    x, y = coord_orig
    x1, y1, x2, y2 = crop_box_xyxy
    w = max(1, x2 - x1)
    h = max(1, y2 - y1)
    x_crop = x - x1
    y_crop = y - y1
    sx = target_size / w
    sy = target_size / h
    return int(round(x_crop * sx)), int(round(y_crop * sy))


def _iter_images(root: Path) -> list[Path]:
    exts = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
    out: list[Path] = []
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.suffix.lower() in exts:
            out.append(p)
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stage1 Easy preprocessing for IDRiD + FGADR.")
    p.add_argument("--data-root", default="data", help="Project data directory (default: data)")
    p.add_argument("--target-size", type=int, default=1024)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--only", choices=["all", "idrid", "fgadr"], default="all")
    p.add_argument("--idrid-split", choices=["all", "train", "test"], default="all")
    p.add_argument("--limit", type=int, default=0, help="If >0, process at most N images (per --only scope).")
    return p.parse_args()


def _preprocess_one(
    src: Path, dst: Path, target_size: int, overwrite: bool, dataset: str
) -> tuple[PreprocessMeta | None, str | None]:
    if dst.exists() and not overwrite:
        return None, None
    img_bgr = cv2.imread(str(src), cv2.IMREAD_COLOR)
    if img_bgr is None:
        return None, f"failed_to_read={src}"
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    out_rgb, crop = preprocess_fundus_rgb(img_rgb, output_size=target_size, dataset=dataset)
    out_bgr = cv2.cvtColor(out_rgb, cv2.COLOR_RGB2BGR)
    dst.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(dst), out_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    if not ok:
        return None, f"failed_to_write={dst}"
    meta = PreprocessMeta(
        src_rel=str(src),
        dst_rel=str(dst),
        orig_hw=(img_rgb.shape[0], img_rgb.shape[1]),
        crop_box_xyxy=crop,
        target_size=target_size,
    )
    return meta, None


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root)

    out_base = data_root / "processed_images" / "stage1_easy"
    meta_path = out_base / "preprocess_meta.jsonl"
    out_base.mkdir(parents=True, exist_ok=True)

    metas: list[PreprocessMeta] = []
    errors: list[str] = []

    prog = default_progress_path()
    if args.only in ("all", "idrid"):
        mark_running(prog, "preprocess_idrid", target_size=args.target_size, overwrite=args.overwrite)
    if args.only in ("all", "fgadr"):
        mark_running(prog, "preprocess_fgadr", target_size=args.target_size, overwrite=args.overwrite)

    remaining = int(args.limit) if int(args.limit) > 0 else None

    if args.only in ("all", "idrid"):
        idrid_train = data_root / "idrid" / "images" / "train"
        idrid_test = data_root / "idrid" / "images" / "test"
        splits: list[tuple[str, Path]] = [("train", idrid_train), ("test", idrid_test)]
        if args.idrid_split != "all":
            splits = [(args.idrid_split, idrid_train if args.idrid_split == "train" else idrid_test)]

        for split_name, src_dir in splits:
            for src in _iter_images(src_dir):
                if remaining is not None and remaining <= 0:
                    break
                dst = out_base / "idrid" / split_name / (src.stem + ".jpg")
                meta, err = _preprocess_one(src, dst, args.target_size, args.overwrite, dataset="idrid")
                if err:
                    errors.append(err)
                if meta:
                    metas.append(meta)
                    if remaining is not None:
                        remaining -= 1
            if remaining is not None and remaining <= 0:
                break

    if args.only in ("all", "fgadr"):
        fgadr_src = data_root / "FGADR" / "Seg-set" / "Original_Images"
        for src in _iter_images(fgadr_src):
            if remaining is not None and remaining <= 0:
                break
            dst = out_base / "fgadr" / (src.stem + ".jpg")
            meta, err = _preprocess_one(src, dst, args.target_size, args.overwrite, dataset="fgadr")
            if err:
                errors.append(err)
            if meta:
                metas.append(meta)
                if remaining is not None:
                    remaining -= 1

    if metas:
        # When doing overwrite runs, avoid unbounded growth / duplicates in meta jsonl.
        mode = "w" if args.overwrite else "a"
        with meta_path.open(mode, encoding="utf-8") as f:
            for m in metas:
                f.write(json.dumps(m.to_dict(), ensure_ascii=False) + "\n")

    print(f"created={len(metas)}")
    print(f"errors={len(errors)}")
    print(f"meta_jsonl={meta_path}")
    if errors:
        # Print first few; keep terminal readable.
        for e in errors[:20]:
            print(e)

    try:
        if args.only in ("all", "idrid"):
            update_progress(
                prog,
                "preprocess_idrid",
                {
                    "status": "done",
                    "ended_at": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime()),
                    "created": len(metas),
                    "errors": len(errors),
                },
            )
        if args.only in ("all", "fgadr"):
            update_progress(
                prog,
                "preprocess_fgadr",
                {
                    "status": "done",
                    "ended_at": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime()),
                    "created": len(metas),
                    "errors": len(errors),
                },
            )
    except Exception as e:
        _ = e


if __name__ == "__main__":
    main()

