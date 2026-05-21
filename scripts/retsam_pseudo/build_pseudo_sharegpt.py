#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import cv2

# Allow running this file directly: add repo root to sys.path.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.retsam_pseudo.common import CropMetaRow, atomic_write_json, read_jsonl
from scripts.stage1_easy.preprocess import preprocess_fundus_rgb


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build ShareGPT multimodal JSON from pseudo CoT + 1024 CLAHE images.")
    p.add_argument("--data-root", default="data")
    p.add_argument("--dataset", choices=["aptos", "ddr_grading"], required=True)
    p.add_argument("--crop-meta", required=True, help="crop_meta.jsonl (for src images + grades).")
    p.add_argument("--cot-jsonl", required=True, help="build_pseudo_cot.py output jsonl.")
    p.add_argument("--out-json", required=True, help="Output ShareGPT JSON file.")
    p.add_argument("--processed-dir", default="", help="Override 1024 output dir (default data/processed_images/pseudo/<dataset>).")
    p.add_argument(
        "--use-existing-processed-only",
        action="store_true",
        help="Do not generate 1024 images; require they already exist in --processed-dir.",
    )
    p.add_argument("--target-size", type=int, default=1024)
    p.add_argument("--overwrite-images", action="store_true")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--stats-out", default="")
    return p.parse_args()


def _repo_rel_to_data(abs_or_rel: Path, data_root: Path) -> str:
    try:
        return abs_or_rel.relative_to(data_root).as_posix()
    except Exception:
        return abs_or_rel.as_posix()


def _load_bgr(path: Path) -> Any:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(str(path))
    return bgr


def main() -> int:
    args = parse_args()
    data_root = Path(args.data_root)
    crop_meta_path = Path(args.crop_meta)
    if not crop_meta_path.is_absolute():
        crop_meta_path = data_root / crop_meta_path

    cot_path = Path(args.cot_jsonl)
    out_json = Path(args.out_json)

    processed_dir = Path(args.processed_dir) if args.processed_dir else (data_root / "processed_images" / "pseudo" / args.dataset)
    if not processed_dir.is_absolute():
        processed_dir = data_root / processed_dir
    processed_dir.mkdir(parents=True, exist_ok=True)

    # Import prompts from Stage1 Easy to guarantee identical format.
    from scripts.stage1_easy.build_dataset import SYSTEM_PROMPT, USER_PROMPT  # noqa: WPS433

    meta_by_id: dict[str, CropMetaRow] = {}
    for obj in read_jsonl(crop_meta_path):
        r = CropMetaRow.from_obj(obj)
        meta_by_id[r.image_id] = r

    cot_rows = read_jsonl(cot_path)
    if args.limit and args.limit > 0:
        cot_rows = cot_rows[: args.limit]

    samples: list[dict[str, Any]] = []
    n_img_created = 0
    n_missing_src = 0
    n_missing_meta = 0

    for row in cot_rows:
        image_id = str(row.get("image_id", ""))
        if not image_id:
            continue
        meta = meta_by_id.get(image_id)
        if meta is None:
            n_missing_meta += 1
            continue

        src_abs = Path(meta.src_path)
        if not src_abs.is_absolute():
            src_abs = data_root / src_abs
        if not src_abs.is_file():
            n_missing_src += 1
            continue

        # Prefer existing png if present (e.g., APTOS processed_images/aptos/*.png).
        out_img = processed_dir / f"{image_id}.jpg"
        out_img_png = processed_dir / f"{image_id}.png"
        chosen = out_img_png if out_img_png.exists() else out_img

        if args.use_existing_processed_only:
            if not chosen.exists():
                n_missing_src += 1
                continue
        else:
            if (not chosen.exists()) or args.overwrite_images:
                bgr = _load_bgr(src_abs)
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                out_rgb, _crop = preprocess_fundus_rgb(rgb, output_size=int(args.target_size), dataset="generic")
                out_bgr = cv2.cvtColor(out_rgb, cv2.COLOR_RGB2BGR)
                ok = cv2.imwrite(str(out_img), out_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
                if not ok:
                    raise RuntimeError(f"failed to write {out_img}")
                chosen = out_img
                n_img_created += 1

        image_rel = _repo_rel_to_data(chosen, data_root)
        assistant_content = str(row.get("assistant_content") or "")
        if "## Analysis" not in assistant_content or "## Output" not in assistant_content:
            # Fallback: reconstruct from analysis_text + output_json.
            analysis_text = str(row.get("analysis_text") or "")
            out_json_obj = row.get("output_json") or {}
            assistant_content = "## Analysis\n" + analysis_text.strip() + "\n\n## Output\n" + json.dumps(out_json_obj, ensure_ascii=False, separators=(",", ":"))

        sample = {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"<image>\n{USER_PROMPT}"},
                {"role": "assistant", "content": assistant_content},
            ],
            "images": [image_rel],
        }
        samples.append(sample)

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(samples, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")

    stats = {
        "dataset": args.dataset,
        "n_samples": len(samples),
        "n_img_created": n_img_created,
        "n_missing_meta": n_missing_meta,
        "n_missing_src": n_missing_src,
        "processed_dir": str(processed_dir),
        "out_json": str(out_json),
    }
    stats_out = Path(args.stats_out) if args.stats_out else out_json.with_suffix(".stats.json")
    atomic_write_json(stats_out, stats, indent=2)

    print(json.dumps(stats, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

