#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

# Allow running this file directly: add repo root to sys.path.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.retsam_pseudo.common import KeptIndexRow, atomic_write_json, read_jsonl, write_jsonl  # noqa: E402
from scripts.retsam_pseudo.retsam_json import parse_lesion_metrics  # noqa: E402


LESIONS = ["MA", "HE", "EX", "SE", "NV", "laser_spot"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Consistency Validation Layer: enforce grade-lesion and biomarker sanity rules.")
    p.add_argument("--kept-index", required=True)
    p.add_argument("--out-jsonl", required=True, help="validated.jsonl output")
    p.add_argument("--out-stats", default="", help="stats json output")
    p.add_argument("--out-errors", default="", help="errors jsonl output")
    p.add_argument("--se-top-pct", type=int, default=20, help="Keep top pct of SE confidence within each grade (2/3/4).")
    p.add_argument("--he-area-min", type=float, default=100.0)
    p.add_argument("--ex-area-min", type=float, default=100.0)
    p.add_argument("--se-area-min", type=float, default=200.0)
    p.add_argument("--he-grade2-area-max", type=float, default=50000.0, help="If grade=2 and HE area > this, suppress as likely FP.")
    p.add_argument("--limit", type=int, default=0)
    return p.parse_args()


def _deep_get(obj: Any, keys: list[str]) -> Any:
    cur = obj
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return None
        cur = cur[k]
    return cur


def _as_float(x: Any) -> float | None:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


def _as_xy(x: Any) -> tuple[float, float] | None:
    if isinstance(x, (list, tuple)) and len(x) >= 2:
        a = _as_float(x[0])
        b = _as_float(x[1])
        if a is None or b is None:
            return None
        return (a, b)
    if isinstance(x, dict):
        a = _as_float(x.get("x"))
        b = _as_float(x.get("y"))
        if a is None or b is None:
            return None
        return (a, b)
    return None


def _confidence(area_px2: float, count: int, *, area_min: float) -> float:
    if count <= 0 or area_px2 <= 0:
        return 0.0
    a = math.log1p(max(0.0, float(area_px2) / max(1e-6, float(area_min)))) / math.log1p(10.0)
    c = math.log1p(float(max(0, int(count)))) / math.log1p(20.0)
    return max(0.0, min(1.0, float(a))) * max(0.0, min(1.0, float(c)))


def _percentile(xs: list[float], p: float) -> float | None:
    if not xs:
        return None
    ys = sorted(float(x) for x in xs)
    k = int(round((p / 100.0) * (len(ys) - 1)))
    k = max(0, min(len(ys) - 1, k))
    return float(ys[k])


def _coord_valid(q: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
    """
    Coord sanity check:
      abs(macula_y - disc_y) > 2 * disc_radius -> anomaly
    """
    disc_center = _as_xy(_deep_get(q, ["measurements", "optic_disc_cup", "disc", "center"]))
    disc_radius = _as_float(_deep_get(q, ["measurements", "optic_disc_cup", "disc", "radius"]))
    macula_center = _as_xy(_deep_get(q, ["measurements", "macula", "center"]))
    if disc_center is None or disc_radius is None or macula_center is None:
        return False, {"reason": "missing_disc_or_macula"}
    _, disc_y = disc_center
    _, mac_y = macula_center
    if abs(float(mac_y) - float(disc_y)) > 2.0 * float(disc_radius):
        return False, {
            "reason": "macula_disc_scale_anomaly",
            "disc_center": list(disc_center),
            "disc_radius": float(disc_radius),
            "macula_center": list(macula_center),
        }
    return True, {"reason": "ok"}


def _extract_biomarkers(q: dict[str, Any]) -> dict[str, Any]:
    # A/V ratio & tortuosity exist under measurements.vessels in current schema.
    cdr = _as_float(_deep_get(q, ["measurements", "optic_disc_cup", "cup_disc_ratio", "value"]))
    if cdr is None:
        cdr = _as_float(_deep_get(q, ["measurements", "optic_disc_cup", "vertical_cd_ratio", "value"]))

    av_ratio = _as_float(_deep_get(q, ["measurements", "vessels", "av_ratio"]))
    tort_a = _as_float(_deep_get(q, ["measurements", "vessels", "tortuosity", "artery"]))
    tort_v = _as_float(_deep_get(q, ["measurements", "vessels", "tortuosity", "vein"]))
    tort = None
    if tort_a is not None and tort_v is not None:
        tort = float((tort_a + tort_v) / 2.0)
    elif tort_a is not None:
        tort = float(tort_a)
    elif tort_v is not None:
        tort = float(tort_v)

    return {
        "av_ratio": {"value": av_ratio, "valid": av_ratio is not None},
        "cdr": {"value": cdr, "valid": cdr is not None},
        "tortuosity": {"value": tort, "valid": tort is not None},
        "eye_side": {
            "value": str(_deep_get(q, ["measurements", "optic_disc_cup", "eye_side"]) or ""),
            "valid": bool(_deep_get(q, ["measurements", "optic_disc_cup", "eye_side"]) in ("left", "right")),
        },
    }


def _apply_biomarker_rules(*, grade: int, biomarkers: dict[str, Any], flags: list[str]) -> None:
    # A/V ratio rules
    av = biomarkers.get("av_ratio", {}).get("value")
    if av is not None:
        av = float(av)
        if av < 0.3 or av > 1.2:
            biomarkers["av_ratio"] = {"value": None, "valid": False, "reason": "out_of_range"}
            flags.append("av_ratio_out_of_range_null")
        else:
            if grade == 0 and av < 0.5:
                biomarkers["av_ratio"]["note"] = "grade0_low_av_suspicious"
                flags.append("av_ratio_grade0_low_suspicious")
            if grade in (3, 4) and av > 0.8:
                biomarkers["av_ratio"] = {"value": None, "valid": False, "reason": "grade34_high_av_suspicious"}
                flags.append("av_ratio_grade34_high_null")

    # CDR rules
    cdr = biomarkers.get("cdr", {}).get("value")
    if cdr is not None:
        cdr = float(cdr)
        if cdr < 0.0 or cdr > 0.9:
            biomarkers["cdr"] = {"value": None, "valid": False, "reason": "out_of_range"}
            flags.append("cdr_out_of_range_null")
        elif cdr > 0.7:
            biomarkers["cdr"]["note"] = "high_cdr_glaucoma_risk"
            flags.append("cdr_high_note")

    # Tortuosity rules
    tort = biomarkers.get("tortuosity", {}).get("value")
    if tort is not None:
        tort = float(tort)
        if tort < 1.0 or tort > 2.5:
            biomarkers["tortuosity"] = {"value": None, "valid": False, "reason": "out_of_range"}
            flags.append("tortuosity_out_of_range_null")
        else:
            if grade in (0, 1) and tort > 1.8:
                biomarkers["tortuosity"]["note"] = "grade01_high_tortuosity_suspicious"
                flags.append("tortuosity_grade01_high_suspicious")
            if grade == 4 and tort < 1.1:
                biomarkers["tortuosity"]["note"] = "grade4_low_tortuosity_suspicious"
                flags.append("tortuosity_grade4_low_suspicious")


def _burden_band(area_px2: float, count: int) -> str | None:
    if count <= 0 or area_px2 <= 0:
        return None
    score = math.log1p(max(0.0, float(area_px2))) + 2.0 * math.log1p(max(0.0, float(count)))
    if score < 9.0:
        return "周边部"
    if score < 11.0:
        return "中周部"
    if score < 13.0:
        return "后极部"
    return "黄斑区"


def main() -> int:
    args = parse_args()
    kept = [KeptIndexRow.from_obj(o) for o in read_jsonl(Path(args.kept_index))]
    if args.limit and args.limit > 0:
        kept = kept[: args.limit]

    # Precompute SE confidence thresholds per grade (2/3/4), based on RetSAM metrics.
    se_conf_by_grade: dict[int, list[float]] = {2: [], 3: [], 4: []}
    for r in kept:
        g = int(r.grade)
        if g not in (2, 3, 4):
            continue
        if not bool(r.se_valid):
            continue
        try:
            q = json.loads(Path(r.retsam_json_path).read_text(encoding="utf-8"))
        except Exception:
            continue
        se = parse_lesion_metrics(q, "SE")
        if se is None:
            continue
        se_conf_by_grade[g].append(_confidence(float(se.total_area), int(se.count), area_min=float(args.se_area_min)))

    se_thr: dict[int, float] = {}
    for g in (2, 3, 4):
        thr = _percentile(se_conf_by_grade[g], float(100 - int(args.se_top_pct)))
        if thr is None or len(se_conf_by_grade[g]) < 25:
            se_thr[g] = 0.0
        else:
            se_thr[g] = float(thr)

    out_rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    flag_counts = Counter()

    for r in kept:
        image_id = r.image_id
        grade = int(r.grade)
        flags: list[str] = []

        try:
            q = json.loads(Path(r.retsam_json_path).read_text(encoding="utf-8"))
        except Exception as e:
            errors.append({"image_id": image_id, "retsam_json_path": r.retsam_json_path, "error": f"json_load:{e}"})
            continue

        coord_ok, coord_detail = _coord_valid(q)
        if not coord_ok:
            flags.append("coord_anomaly_detected")

        biomarkers = _extract_biomarkers(q)
        _apply_biomarker_rules(grade=grade, biomarkers=biomarkers, flags=flags)

        # Base RetSAM metrics
        he = parse_lesion_metrics(q, "HE")
        ex = parse_lesion_metrics(q, "EX")
        se = parse_lesion_metrics(q, "SE")

        def lesion_block(name: str, metrics, valid_flag: bool, area_min: float) -> dict[str, Any]:
            cnt = int(metrics.count) if metrics else 0
            area = float(metrics.total_area) if metrics else 0.0
            conf = _confidence(area, cnt, area_min=area_min)
            return {
                "present": bool(valid_flag and cnt > 0 and area > 0),
                "count": cnt,
                "area": area,
                "confidence": conf,
                "source": ("retsam_valid" if valid_flag else "retsam_filtered"),
                "location_band": _burden_band(area, cnt),
            }

        lesions: dict[str, Any] = {}

        # MA: from grade template only (RetSAM doesn't provide it here)
        lesions["MA"] = {"present": bool(grade >= 1), "source": "grade_template"}
        # NV: template only (grade4 true)
        lesions["NV"] = {"present": bool(grade == 4), "source": "grade_template"}
        # laser_spot: not available in current RetSAM output -> default false
        lesions["laser_spot"] = {"present": False, "source": "missing_in_retsam"}

        lesions["HE"] = lesion_block("HE", he, bool(r.he_valid), float(args.he_area_min))
        lesions["EX"] = lesion_block("EX", ex, bool(r.ex_valid), float(args.ex_area_min))
        lesions["SE"] = lesion_block("SE", se, bool(r.se_valid), float(args.se_area_min))

        # Grade-lesion consistency overrides
        if grade == 0:
            for k in ("MA", "HE", "EX", "SE", "NV", "laser_spot"):
                lesions[k]["present"] = False
                lesions[k]["source"] = "grade_rule_override"
            flags.append("grade0_all_negative_forced")

        if grade == 1:
            # ICDR G1: MA only
            lesions["MA"] = {"present": True, "source": "grade_template"}
            for k in ("HE", "EX", "SE"):
                if lesions[k].get("present"):
                    flags.append(f"{k.lower()}_suppressed_grade1")
                lesions[k]["present"] = False
                lesions[k]["source"] = "grade_rule_override"
            lesions["NV"]["present"] = False
            lesions["NV"]["source"] = "grade_rule_override"
            flags.append("grade1_ma_only_forced")

        # SE top20% within grade 2/3/4
        if grade in (2, 3, 4):
            se_conf = float(lesions["SE"].get("confidence") or 0.0)
            if lesions["SE"].get("present") and se_conf < float(se_thr.get(grade, 0.0)):
                lesions["SE"]["present"] = False
                lesions["SE"]["source"] = "low_confidence_filtered"
                flags.append("se_low_confidence_suppressed")

        # HE area sanity for grade2
        if grade == 2 and lesions["HE"].get("present") and float(lesions["HE"].get("area") or 0.0) > float(args.he_grade2_area_max):
            lesions["HE"]["present"] = False
            lesions["HE"]["source"] = "grade2_he_area_anomaly_suppressed"
            flags.append("he_area_anomaly_suppressed")

        # Laser spot rules (if ever present in future)
        if grade in (0, 1) and lesions["laser_spot"].get("present"):
            lesions["laser_spot"]["present"] = False
            lesions["laser_spot"]["source"] = "grade_rule_override"
            flags.append("laser_spot_suppressed_grade01")

        # Coord anomaly: drop coord-dependent fields (we keep band, but degrade description)
        location = {"coord_valid": bool(coord_ok), "detail": coord_detail}
        if not coord_ok:
            location["fallback"] = "fixed_posterior_pole"
            # Force lesion location_band to None so downstream can use a fixed wording.
            for k in ("HE", "EX", "SE"):
                if isinstance(lesions.get(k), dict):
                    lesions[k]["location_band"] = None
        else:
            location["fallback"] = "burden_band"

        validated = {
            "image_id": image_id,
            "grade": grade,
            "lesions": lesions,
            "biomarkers": biomarkers,
            "location": location,
            "validation_flags": flags,
            "data_type": "pseudo",
            "retsam_json_path": r.retsam_json_path,
            "se_keep_thr": {str(k): float(v) for k, v in se_thr.items()},
        }
        out_rows.append(validated)
        for fl in flags:
            flag_counts[fl] += 1

    out_path = Path(args.out_jsonl)
    stats_path = Path(args.out_stats) if args.out_stats else out_path.with_suffix(".stats.json")
    err_path = Path(args.out_errors) if args.out_errors else out_path.with_suffix(".errors.jsonl")
    write_jsonl(out_path, out_rows, append=False)
    if errors:
        write_jsonl(err_path, errors, append=False)

    atomic_write_json(
        stats_path,
        {
            "n_in": len(kept),
            "n_out": len(out_rows),
            "n_errors": len(errors),
            "se_top_pct": int(args.se_top_pct),
            "se_thr": {str(k): float(v) for k, v in se_thr.items()},
            "flag_counts": dict(flag_counts),
        },
        indent=2,
    )
    print(json.dumps({"out_jsonl": str(out_path), "stats": str(stats_path), "errors": str(err_path), "n_out": len(out_rows)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

