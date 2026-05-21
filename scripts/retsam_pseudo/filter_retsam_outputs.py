#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

# Allow running this file directly: add repo root to sys.path.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.retsam_pseudo.common import CropMetaRow, KeptIndexRow, atomic_write_json, read_jsonl, write_jsonl
from scripts.retsam_pseudo.retsam_json import parse_lesion_metrics


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Filter RetSAM quantitative_analysis.json by lesion-specific heuristics.")
    p.add_argument("--data-root", default="data")
    p.add_argument("--crop-meta", required=True, help="crop_meta.jsonl (to recover grade per image_id).")
    p.add_argument("--dataset", choices=["aptos", "ddr_grading"], required=True)
    p.add_argument("--retsam-out", required=True, help="outputs/retsam_<dataset> directory.")
    p.add_argument("--out-kept-index", default="", help="kept_index.jsonl output (default: <retsam-out>/kept_index.jsonl)")
    p.add_argument("--out-stats", default="", help="filter_stats.json output (default: <retsam-out>/filter_stats.json)")
    p.add_argument("--errors-jsonl", default="", help="errors.jsonl output (default: <retsam-out>/errors.jsonl)")
    p.add_argument("--he-area-min", type=float, default=100.0)
    p.add_argument("--ex-area-min", type=float, default=100.0)
    p.add_argument("--se-area-min", type=float, default=200.0)
    p.add_argument("--grade0-he-fp-max-count", type=int, default=5)
    p.add_argument("--limit", type=int, default=0)
    return p.parse_args()


def _resolve(path_or_rel: str, data_root: Path) -> Path:
    p = Path(path_or_rel)
    return p if p.is_absolute() else (data_root / p)


def _valid(area: float, count: int, area_min: float) -> bool:
    return (count > 0) and (float(area) > float(area_min))


def main() -> int:
    args = parse_args()
    data_root = Path(args.data_root)
    crop_meta = _resolve(args.crop_meta, data_root)
    out_dir = Path(args.retsam_out)
    if not out_dir.is_absolute():
        out_dir = _REPO_ROOT / out_dir

    kept_path = Path(args.out_kept_index) if args.out_kept_index else (out_dir / "kept_index.jsonl")
    stats_path = Path(args.out_stats) if args.out_stats else (out_dir / "filter_stats.json")
    errors_path = Path(args.errors_jsonl) if args.errors_jsonl else (out_dir / "errors.jsonl")
    if not kept_path.is_absolute():
        kept_path = _REPO_ROOT / kept_path
    if not stats_path.is_absolute():
        stats_path = _REPO_ROOT / stats_path
    if not errors_path.is_absolute():
        errors_path = _REPO_ROOT / errors_path

    grade_by_id: dict[str, int] = {}
    for obj in read_jsonl(crop_meta):
        r = CropMetaRow.from_obj(obj)
        grade_by_id[r.image_id] = int(r.grade)

    qa_files = sorted(out_dir.glob("*/quantitative_analysis.json"))
    if args.limit and args.limit > 0:
        qa_files = qa_files[: args.limit]

    kept_rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    drop_reasons = Counter()
    lesion_keep = defaultdict(int)
    lesion_total = defaultdict(int)

    for qa_path in qa_files:
        image_id = qa_path.parent.name
        grade = grade_by_id.get(image_id)
        if grade is None:
            errors.append({"image_id": image_id, "qa_path": str(qa_path), "error": "missing_grade_in_crop_meta"})
            continue

        try:
            q = json.loads(qa_path.read_text(encoding="utf-8"))
        except Exception as e:
            errors.append({"image_id": image_id, "qa_path": str(qa_path), "error": f"json_load:{e}"})
            continue

        he = parse_lesion_metrics(q, "HE")
        ex = parse_lesion_metrics(q, "EX")
        se = parse_lesion_metrics(q, "SE")

        he_area = float(he.total_area) if he else 0.0
        he_cnt = int(he.count) if he else 0
        ex_area = float(ex.total_area) if ex else 0.0
        ex_cnt = int(ex.count) if ex else 0
        se_area = float(se.total_area) if se else 0.0
        se_cnt = int(se.count) if se else 0

        lesion_total["HE"] += 1
        lesion_total["EX"] += 1
        lesion_total["SE"] += 1

        he_valid = _valid(he_area, he_cnt, args.he_area_min)
        ex_valid = _valid(ex_area, ex_cnt, args.ex_area_min)
        se_valid = _valid(se_area, se_cnt, args.se_area_min)

        # Grade0 special HE FP guard (only relevant if grade0 ever goes through inference).
        if int(grade) == 0 and he_cnt > int(args.grade0_he_fp_max_count):
            if he_valid:
                drop_reasons["grade0_he_fp_drop"] += 1
            he_valid = False

        if he_valid:
            lesion_keep["HE"] += 1
        else:
            if he_cnt > 0 or he_area > 0:
                drop_reasons["he_filtered"] += 1
        if ex_valid:
            lesion_keep["EX"] += 1
        else:
            if ex_cnt > 0 or ex_area > 0:
                drop_reasons["ex_filtered"] += 1
        if se_valid:
            lesion_keep["SE"] += 1
        else:
            if se_cnt > 0 or se_area > 0:
                drop_reasons["se_filtered"] += 1

        row = KeptIndexRow(
            image_id=image_id,
            grade=int(grade),
            retsam_json_path=str(qa_path),
            he_valid=bool(he_valid),
            ex_valid=bool(ex_valid),
            se_valid=bool(se_valid),
            od_source="retsam",
        )
        kept_rows.append(row.__dict__)

    write_jsonl(kept_path, kept_rows, append=False)
    if errors:
        write_jsonl(errors_path, errors, append=True)

    stats = {
        "dataset": args.dataset,
        "n_quantitative_analysis": len(qa_files),
        "n_kept_index_rows": len(kept_rows),
        "lesion_keep_counts": dict(lesion_keep),
        "lesion_total_images": dict(lesion_total),
        "lesion_keep_rates": {k: (lesion_keep[k] / max(1, lesion_total[k])) for k in lesion_total.keys()},
        "drop_reasons": dict(drop_reasons),
        "thresholds": {
            "he_area_min": args.he_area_min,
            "ex_area_min": args.ex_area_min,
            "se_area_min": args.se_area_min,
            "grade0_he_fp_max_count": args.grade0_he_fp_max_count,
        },
        "errors_appended": len(errors),
    }
    atomic_write_json(stats_path, stats, indent=2)

    print(json.dumps({"kept_index": str(kept_path), "stats": str(stats_path), "errors_appended": len(errors)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

