#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np

# Allow running this file directly: add repo root to sys.path.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.retsam_pseudo.common import atomic_write_json, read_jsonl, write_jsonl


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build two-section pseudo CoT from validated.jsonl (consistency layer output).")
    p.add_argument("--validated-jsonl", required=True, help="Output of consistency_validate.py")
    p.add_argument("--out-jsonl", required=True, help="Per-image pseudo CoT records (jsonl).")
    p.add_argument("--out-errors", default="", help="errors.jsonl (default: alongside out-jsonl).")
    p.add_argument("--out-stats", default="", help="stats json (default: alongside out-jsonl).")
    p.add_argument("--area_bins", default="1000,5000", help="Comma bins for small/medium/large (px^2) in RetSAM mask space.")
    p.add_argument("--limit", type=int, default=0)
    return p.parse_args()


def _qty_word(n: int) -> str:
    if n <= 0:
        return "未见"
    if 1 <= n <= 3:
        return "少量"
    if 4 <= n <= 10:
        return "数处"
    if 11 <= n <= 20:
        return "较多"
    return "大量"


def _area_word(a: float, bins: tuple[float, float]) -> str:
    if a <= 0:
        return "未见"
    if a < bins[0]:
        return "小灶性"
    if a < bins[1]:
        return "中等面积"
    return "大面积"


def _burden_band(area_px2: float, count: int) -> str:
    """
    Location-band proxy that does NOT rely on coordinates/quadrants.

    Rationale:
    - Quadrants (TS/TI/NS/NI) depend on a consistent coordinate system across datasets.
    - DDR currently shows macula/OD coordinate scale anomalies; using quadrants would inject noise.
    - We instead describe a coarse *extent band* based on lesion burden (area + count),
      which is stable across datasets and aligns with stage-1 supervision goals.
    """
    if count <= 0 or area_px2 <= 0:
        return "未见"

    # Use a smooth burden score; thresholds are heuristic and can be tuned later.
    score = math.log1p(max(0.0, float(area_px2))) + 2.0 * math.log1p(max(0.0, float(count)))
    if score < 9.0:
        return "周边部"
    if score < 11.0:
        return "中周部"
    if score < 13.0:
        return "后极部"
    return "黄斑区"

def _grade0_template(image_id: str) -> tuple[str, dict[str, Any]]:
    analysis = (
        "MA：未见微动脉瘤。\n"
        "HE：未见出血。\n"
        "EX：未见硬性渗出。\n"
        "SE：未见棉绒斑。\n"
        "NV：未见新生血管。"
    )
    out = {
        "MA": {"present": False},
        "HE": {"present": False},
        "EX": {"present": False},
        "SE": {"present": False},
        "NV": {"present": False},
        "image_id": image_id,
        "data_type": "pseudo",
    }
    return analysis, out


def _ma_by_grade(grade: int) -> tuple[str, dict[str, Any]]:
    if grade <= 0:
        return "MA：未见典型红色点状微动脉瘤。", {"present": False}
    if grade == 1:
        return "MA：可见少量疑似微动脉瘤，数量不多。", {"present": True, "severity": "mild"}
    if grade == 2:
        return "MA：可见数处微动脉瘤，提示存在早期微血管损伤。", {"present": True, "severity": "moderate"}
    return "MA：可见较多微动脉瘤，数量较为明显。", {"present": True, "severity": "severe"}


def _hough_od_fallback(img_bgr: np.ndarray) -> tuple[tuple[float, float] | None, float | None]:
    """
    Very rough fallback: detect a bright circular structure (optic disc) via Hough circles.
    Returns (center_xy, radius_px) in image coords.
    """
    try:
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.medianBlur(gray, 5)
        circles = cv2.HoughCircles(
            gray,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=80,
            param1=80,
            param2=30,
            minRadius=20,
            maxRadius=140,
        )
        if circles is None:
            return None, None
        c = circles[0][0]
        x, y, r = float(c[0]), float(c[1]), float(c[2])
        return (x, y), r
    except Exception:
        return None, None


def _format_assistant(analysis: str, out_json: dict[str, Any]) -> str:
    return "## Analysis\n" + analysis.strip() + "\n\n## Output\n" + json.dumps(out_json, ensure_ascii=False, separators=(",", ":"))


def main() -> int:
    args = parse_args()
    validated_path = Path(args.validated_jsonl)
    out_jsonl = Path(args.out_jsonl)
    err_path = Path(args.out_errors) if args.out_errors else out_jsonl.with_suffix(".errors.jsonl")
    stats_path = Path(args.out_stats) if args.out_stats else out_jsonl.with_suffix(".stats.json")

    bins = tuple(float(x) for x in args.area_bins.split(",")[:2])
    rows = read_jsonl(validated_path)
    if args.limit and args.limit > 0:
        rows = rows[: args.limit]

    out_rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    n_grade0 = 0
    n_ok = 0

    for r in rows:
        try:
            image_id = str(r.get("image_id") or "")
            grade = int(r.get("grade"))
            vles = r.get("lesions") or {}
            loc = r.get("location") or {}
        except Exception as e:
            errors.append({"error": f"row_parse:{e}", "row": r})
            continue

        if grade == 0:
            n_grade0 += 1

        ma_present = bool((vles.get("MA") or {}).get("present"))
        nv_present = bool((vles.get("NV") or {}).get("present"))

        def lesion_line(code: str) -> tuple[str, dict[str, Any]]:
            blk = vles.get(code) or {}
            if not isinstance(blk, dict) or not blk.get("present"):
                return f"{code}：未见明确相关病灶。", {"present": False}
            cnt = int(blk.get("count") or 0)
            area = float(blk.get("area") or 0.0)
            qty = _qty_word(cnt)
            area_w = _area_word(area, bins)
            band = blk.get("location_band")
            if not isinstance(band, str) or not band:
                # coord anomaly fallback
                band = "后极部" if (loc.get("fallback") == "fixed_posterior_pole") else _burden_band(area, cnt)
            text = f"{code}：可见{qty}{area_w}病灶，主要分布于{band}。"
            out = {
                "present": True,
                "count_bucket": qty,
                "area_bucket": area_w,
                "location_band": band,
            }
            if "confidence" in blk:
                out["confidence"] = float(blk.get("confidence") or 0.0)
            return text, out

        ma_text, ma_json = _ma_by_grade(grade)
        if not ma_present:
            ma_text = "MA：未见典型红色点状微动脉瘤。"
            ma_json = {"present": False}

        he_text, he_json = lesion_line("HE")
        ex_text, ex_json = lesion_line("EX")
        se_text, se_json = lesion_line("SE")

        nv_text = "NV：未见新生血管证据。"
        nv_json = {"present": False}
        if nv_present:
            nv_text = "NV：提示存在新生血管风险（基于分级模板）。"
            nv_json = {"present": True, "source": "grade_template"}

        analysis = "\n".join([ma_text, he_text, ex_text, se_text, nv_text])
        outj = {
            "MA": ma_json,
            "HE": he_json,
            "EX": ex_json,
            "SE": se_json,
            "NV": nv_json,
            "image_id": image_id,
            "grade": grade,
            "data_type": "pseudo",
            "validation_flags": list(r.get("validation_flags") or []),
        }

        out_rows.append(
            {
                "image_id": image_id,
                "grade": grade,
                "analysis_text": analysis,
                "output_json": outj,
                "assistant_content": _format_assistant(analysis, outj),
                "validated_source": str(validated_path),
                "retsam_json_path": r.get("retsam_json_path"),
            }
        )
        n_ok += 1

    write_jsonl(out_jsonl, out_rows, append=False)
    if errors:
        write_jsonl(err_path, errors, append=True)

    atomic_write_json(
        stats_path,
        {"n_rows": len(out_rows), "n_ok": n_ok, "n_grade0": n_grade0, "errors_appended": len(errors)},
        indent=2,
    )
    print(json.dumps({"out_jsonl": str(out_jsonl), "stats": str(stats_path), "errors_appended": len(errors)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

