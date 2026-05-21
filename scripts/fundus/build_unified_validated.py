#!/usr/bin/env python3
"""Build a unified fundus validated.jsonl from crop indices and RetSAM outputs.

The output is the conservative "trusted facts" layer used before CoT/SFT data
generation. Missing RetSAM or strong masks are represented explicitly instead
of dropping the indexed image.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


LESION_MAP = {
    "HE": "hemorrhage",
    "EX": "exudate",
    "SE": "cotton_wool_spot",
}

DEFAULT_DATASETS = [
    {
        "name": "aptos",
        "split": "all",
        "crop_meta": "data/cropped/aptos/crop_meta.jsonl",
        "retsam_dir": "outputs/retsam_aptos",
        "supervision": "grade",
    },
    {
        "name": "ddr_grading",
        "split": "all",
        "crop_meta": "data/cropped/ddr_grading/crop_meta.jsonl",
        "retsam_dir": "outputs/retsam_ddr_grading",
        "supervision": "grade",
    },
    {
        "name": "idrid",
        "split": "train",
        "crop_meta": "data/cropped/idrid/crop_meta.jsonl",
        "retsam_dir": "outputs/retsam_idrid",
        "supervision": "grade+mask",
    },
    {
        "name": "idrid",
        "split": "test",
        "crop_meta": "data/cropped/idrid_test/crop_meta.jsonl",
        "retsam_dir": None,
        "supervision": "grade+mask",
    },
    {
        "name": "fgadr_seg",
        "split": "all",
        "crop_meta": "data/cropped/fgadr_seg/crop_meta.jsonl",
        "retsam_dir": "outputs/retsam_fgadr_seg",
        "supervision": "grade+mask",
    },
    {
        "name": "ddr_seg",
        "split": "train",
        "crop_meta": "data/cropped/ddr_seg/crop_meta.jsonl",
        "retsam_dir": "outputs/retsam_ddr_seg",
        "supervision": "mask",
    },
    {
        "name": "ddr_seg",
        "split": "valid",
        "crop_meta": "data/cropped/ddr_seg_valid/crop_meta.jsonl",
        "retsam_dir": "outputs/retsam_ddr_seg",
        "supervision": "mask",
    },
    {
        "name": "ddr_seg",
        "split": "test",
        "crop_meta": "data/cropped/ddr_seg_test/crop_meta.jsonl",
        "retsam_dir": "outputs/retsam_ddr_seg",
        "supervision": "mask",
    },
]


def load_json(path: Path) -> dict[str, Any] | None:
    try:
        with path.open("r", encoding="utf-8") as f:
            obj = json.load(f)
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield line_no, json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Bad JSONL at {path}:{line_no}: {exc}") from exc


def finite_number(value: Any) -> float | None:
    if isinstance(value, (int, float)) and math.isfinite(value):
        return float(value)
    return None


def bucket_count(count: int | None) -> str:
    if count is None:
        return "unknown"
    if count <= 0:
        return "none"
    if count <= 3:
        return "few"
    if count <= 10:
        return "some"
    return "many"


def bucket_area(area: float | None) -> str:
    if area is None:
        return "unknown"
    if area <= 0:
        return "none"
    if area < 300:
        return "small"
    if area < 1500:
        return "medium"
    return "large"


def location_band_from_dd(dist_dd: float | None) -> str | None:
    if dist_dd is None:
        return None
    if dist_dd <= 1.0:
        return "黄斑区"
    if dist_dd <= 3.0:
        return "后极部"
    if dist_dd <= 6.0:
        return "中周部"
    return "周边部"


def lesion_confidence(count: int, area: float) -> float:
    if count <= 0 or area <= 0:
        return 0.0
    count_score = min(1.0, count / 8.0)
    area_score = min(1.0, area / 1500.0)
    return round(max(count_score, area_score), 6)


def get_path(obj: dict[str, Any], keys: list[str], default: Any = None) -> Any:
    cur: Any = obj
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def parse_retsam_lesions(retsam: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    lesions: dict[str, dict[str, Any]] = {}
    categories = get_path(retsam or {}, ["measurements", "lesions", "lesion_dr", "categories"], {})
    for out_name, category_name in LESION_MAP.items():
        cat = categories.get(category_name, {}) if isinstance(categories, dict) else {}
        count = int(cat.get("count") or 0)
        area = float(cat.get("area_px") or 0.0)
        conf = lesion_confidence(count, area)
        lesions[out_name] = {
            "present": count > 0 and area > 0,
            "count": count,
            "area": area,
            "confidence": conf,
            "count_bucket": bucket_count(count),
            "area_bucket": bucket_area(area),
            "location_band": None,
            "source": "validated_retsam" if count > 0 and area > 0 else "retsam_negative",
        }
    return lesions


def parse_biomarkers(retsam: dict[str, Any] | None) -> dict[str, Any]:
    vessels = get_path(retsam or {}, ["measurements", "vessels"], {}) or {}
    od = get_path(retsam or {}, ["measurements", "optic_disc_cup"], {}) or {}
    macula = get_path(retsam or {}, ["measurements", "macula"], {}) or {}
    vessel_qc = bool(vessels.get("qc_flag")) if "qc_flag" in vessels else False
    od_qc = bool(get_path(od, ["qc", "pass"], False))

    artery_t = finite_number(get_path(vessels, ["tortuosity", "artery"]))
    vein_t = finite_number(get_path(vessels, ["tortuosity", "vein"]))
    tortuosity = None
    if artery_t is not None and vein_t is not None:
        tortuosity = round((artery_t + vein_t) / 2.0, 6)

    cdr = finite_number(get_path(od, ["cup_disc_ratio", "value"]))
    av_ratio = finite_number(vessels.get("av_ratio"))

    return {
        "eye_side": {
            "value": od.get("eye_side"),
            "valid": bool(od.get("eye_side")) and od_qc,
            "source": "validated_retsam" if od.get("eye_side") else "missing",
        },
        "cdr": {
            "value": cdr,
            "valid": cdr is not None and 0.0 <= cdr <= 1.2 and od_qc,
            "source": "validated_retsam" if cdr is not None else "missing",
        },
        "av_ratio": {
            "value": av_ratio,
            "valid": av_ratio is not None and 0.2 <= av_ratio <= 1.5 and vessel_qc,
            "source": "validated_retsam" if av_ratio is not None else "missing",
        },
        "tortuosity": {
            "value": tortuosity,
            "valid": tortuosity is not None and 0.0 <= tortuosity <= 5.0 and vessel_qc,
            "source": "validated_retsam" if tortuosity is not None else "missing",
        },
        "crae": {
            "value": vessels.get("CRAE"),
            "valid": bool(vessels.get("CRAE")) and vessel_qc,
            "source": "validated_retsam" if vessels.get("CRAE") else "missing",
        },
        "crve": {
            "value": vessels.get("CRVE"),
            "valid": bool(vessels.get("CRVE")) and vessel_qc,
            "source": "validated_retsam" if vessels.get("CRVE") else "missing",
        },
        "fractal_dimension": {
            "value": vessels.get("fractal_dimension"),
            "valid": bool(vessels.get("fractal_dimension")) and vessel_qc,
            "source": "validated_retsam" if vessels.get("fractal_dimension") else "missing",
        },
        "vessel_qc_flag": vessel_qc,
        "od_qc_flag": od_qc,
        "macula_center": get_path(macula, ["center"]),
        "optic_disc": od.get("disc"),
    }


def coord_validation(biomarkers: dict[str, Any]) -> dict[str, Any]:
    disc = biomarkers.get("optic_disc") or {}
    macula = biomarkers.get("macula_center") or {}
    disc_center = disc.get("center") if isinstance(disc, dict) else None
    disc_radius = finite_number(disc.get("radius")) if isinstance(disc, dict) else None
    mx = finite_number(macula.get("x")) if isinstance(macula, dict) else None
    my = finite_number(macula.get("y")) if isinstance(macula, dict) else None
    if not disc_center or len(disc_center) != 2 or disc_radius is None or mx is None or my is None:
        return {"coord_valid": False, "detail": {"reason": "missing_od_or_macula"}, "fallback": "burden_band"}
    dx = finite_number(disc_center[0])
    dy = finite_number(disc_center[1])
    if dx is None or dy is None or disc_radius <= 0:
        return {"coord_valid": False, "detail": {"reason": "invalid_od_or_macula"}, "fallback": "burden_band"}
    distance = math.hypot(dx - mx, dy - my)
    valid = 1.0 * disc_radius <= distance <= 8.0 * disc_radius
    return {
        "coord_valid": valid,
        "detail": {
            "reason": "ok" if valid else "macula_disc_scale_anomaly",
            "disc_center": disc_center,
            "disc_radius": disc_radius,
            "macula_center": {"x": mx, "y": my},
            "distance_disc_radius_ratio": round(distance / disc_radius, 4),
        },
        "fallback": "burden_band",
    }


def apply_grade_rules(lesions: dict[str, dict[str, Any]], grade: int | None, flags: list[str]) -> None:
    lesions["MA"] = {"present": "unknown", "source": "missing_or_strong_mask_required"}
    lesions["NV"] = {"present": False, "source": "grade_rule"}
    if grade is None or grade < 0:
        return
    if grade == 0:
        for key in ["HE", "EX", "SE"]:
            if lesions[key]["present"]:
                flags.append(f"{key.lower()}_suppressed_grade0")
            lesions[key]["present"] = False
            lesions[key]["source"] = "grade_rule_override"
        lesions["MA"] = {"present": False, "source": "grade_rule"}
    elif grade == 1:
        for key in ["HE", "EX", "SE"]:
            if lesions[key]["present"]:
                flags.append(f"{key.lower()}_suppressed_grade1")
            lesions[key]["present"] = False
            lesions[key]["source"] = "grade_rule_override"
        lesions["MA"] = {"present": "template_only", "source": "grade_rule", "note": "RetSAM does not provide MA"}
        flags.append("grade1_ma_only_template")
    elif grade == 4:
        lesions["NV"] = {
            "present": "possible_by_grade_template",
            "source": "grade_rule",
            "note": "No direct NV field in current RetSAM schema",
        }


def extract_output_block(content: str) -> str | None:
    match = re.search(r"## Output\s*\n(?P<out>.*)\s*$", content, flags=re.S)
    if not match:
        return None
    return match.group("out").strip()


def parse_json_output_block(content: str) -> dict[str, Any] | None:
    block = extract_output_block(content)
    if not block:
        return None
    try:
        obj = json.loads(block)
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


def parse_kv_output_block(content: str) -> dict[str, str] | None:
    block = extract_output_block(content)
    if not block:
        return None
    out: dict[str, str] = {}
    for line in block.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        out[key.strip()] = value.strip()
    return out or None


def load_json_array(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, list):
        raise ValueError(f"Expected JSON array: {path}")
    return [x for x in obj if isinstance(x, dict)]


def load_stage1_easy_annotations(root: Path) -> dict[tuple[str, str], dict[str, Any]]:
    annotations: dict[tuple[str, str], dict[str, Any]] = {}
    paths = [
        root / "data/annotation/idrid_fgadr_stage1_easy_train.json",
        root / "data/annotation/idrid_stage1_easy_test.json",
    ]
    for path in paths:
        for row in load_json_array(path):
            images = row.get("images") or []
            messages = row.get("messages") or []
            if not images or not messages:
                continue
            output = parse_json_output_block(str(messages[-1].get("content", "")))
            if output is None:
                continue
            image_rel = str(images[0])
            stem = Path(image_rel).stem
            if "stage1_easy/idrid/" in image_rel:
                split = "test" if "stage1_easy/idrid/test/" in image_rel else "train"
                annotations[(f"idrid::{split}", stem)] = {
                    "source": "stage1_easy",
                    "annotation_path": str(path.relative_to(root)),
                    "image": image_rel,
                    "output": output,
                }
            elif "FGADR/Seg-set/Original_Images/" in image_rel:
                annotations[("fgadr_seg", stem)] = {
                    "source": "stage1_easy",
                    "annotation_path": str(path.relative_to(root)),
                    "image": image_rel,
                    "output": output,
                }
    return annotations


def load_fgadr_lesion_only_annotations(root: Path) -> dict[tuple[str, str], dict[str, Any]]:
    annotations: dict[tuple[str, str], dict[str, Any]] = {}
    path = root / "data/annotation/fgadr_lesion_only_sft_v3_lf.json"
    for row in load_json_array(path):
        images = row.get("images") or []
        messages = row.get("messages") or []
        if not images or not messages:
            continue
        image_rel = str(images[0])
        if "FGADR/Seg-set/Original_Images/" not in image_rel:
            continue
        stem = Path(image_rel).stem
        if ("fgadr_seg", stem) in annotations:
            continue
        output = parse_kv_output_block(str(messages[-1].get("content", "")))
        if output is None:
            continue
        annotations[("fgadr_seg", stem)] = {
            "source": "fgadr_lesion_only_sft_v3",
            "annotation_path": str(path.relative_to(root)),
            "image": image_rel,
            "output": output,
        }
    return annotations


def load_fgadr_grades(root: Path) -> dict[str, int]:
    grades: dict[str, int] = {}
    path = root / "data/FGADR/Seg-set/DR_Seg_Grading_Label.csv"
    if not path.exists():
        return grades
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.reader(f):
            if len(row) < 2:
                continue
            try:
                grades[Path(row[0]).stem] = int(row[1])
            except ValueError:
                continue
    return grades


def normalize_stage1_lesion(lesion: str, data: dict[str, Any]) -> dict[str, Any]:
    present = bool(data.get("present"))
    count = data.get("count_estimate", data.get("count"))
    area = data.get("total_area_px2")
    dist = finite_number(data.get("min_dist_to_fovea_dd"))
    out = {
        "present": present,
        "count": int(count) if isinstance(count, int) else None,
        "area": float(area) if isinstance(area, (int, float)) else None,
        "confidence": 1.0,
        "count_bucket": bucket_count(count if isinstance(count, int) else None),
        "area_bucket": bucket_area(float(area) if isinstance(area, (int, float)) else None),
        "location_band": location_band_from_dd(dist),
        "source": "strong_mask_stage1_easy",
    }
    for key in [
        "count_estimate",
        "diameter_range_px",
        "main_quadrant",
        "quadrant_distribution",
        "min_dist_to_fovea_dd",
        "morphology",
        "max_width_px",
        "pattern",
        "hollow_area_px2",
        "hollow_ratio",
        "csme_risk",
        "4_2_1_alert",
        "location",
    ]:
        if key in data:
            out[key] = data[key]
    return out


def apply_precomputed_annotations(
    row: dict[str, Any],
    dataset: str,
    lesions: dict[str, dict[str, Any]],
    annotations: dict[str, dict[tuple[str, str], dict[str, Any]]],
    sources: dict[str, Any],
) -> None:
    image_id = str(row["image_id"])
    stage1_key = (f"idrid::{row.get('_dataset_split', '')}", image_id) if dataset == "idrid" else (dataset, image_id)
    ann = annotations["stage1_easy"].get(stage1_key)
    if ann:
        output = ann["output"]
        for lesion in ["MA", "HE", "EX", "SE", "NV"]:
            if lesion in output and isinstance(output[lesion], dict):
                lesions[lesion] = normalize_stage1_lesion(lesion, output[lesion])
        sources["strong_annotation_path"] = ann["annotation_path"]
        sources["strong_annotation_source"] = ann["source"]
        return

    ann = annotations["fgadr_lesion_only"].get((dataset, image_id))
    if ann:
        output = ann["output"]
        present = {x.strip() for x in output.get("LESIONS", "").split(",") if x.strip() and x.strip() != "NONE"}
        for lesion in ["MA", "HE", "EX", "SE", "IRMA", "NV"]:
            lesions[lesion] = {
                "present": lesion in present,
                "source": "fgadr_lesion_only_sft_v3",
                "confidence": 1.0 if output.get("UNCERTAINTY") == "LOW" else 0.75,
                "location": output.get("LOCATION"),
                "extent": output.get("EXTENT"),
                "severity_cue": output.get("SEVERITY_CUE"),
                "uncertainty": output.get("UNCERTAINTY"),
            }
        lesions["NV"]["present"] = output.get("NEOVASCULAR_SIGN") == "PRESENT"
        sources["strong_annotation_path"] = ann["annotation_path"]
        sources["strong_annotation_source"] = ann["source"]


def build_record(root: Path, spec: dict[str, Any], row: dict[str, Any], annotations, fgadr_grades) -> dict[str, Any]:
    row = dict(row)
    row["_dataset_split"] = spec["split"]
    image_id = str(row["image_id"])
    grade = row.get("grade")
    grade = int(grade) if isinstance(grade, int) or (isinstance(grade, str) and grade.lstrip("-").isdigit()) else None
    if spec["name"] == "fgadr_seg" and (grade is None or grade < 0):
        grade = fgadr_grades.get(image_id, grade)

    retsam_path = root / spec["retsam_dir"] / image_id / "quantitative_analysis.json" if spec.get("retsam_dir") else None
    retsam = load_json(retsam_path) if retsam_path and retsam_path.exists() else None

    flags: list[str] = []
    if retsam is None:
        flags.append("retsam_missing")

    lesions = parse_retsam_lesions(retsam)
    apply_grade_rules(lesions, grade, flags)
    sources = {
        "crop_meta": spec["crop_meta"],
        "retsam_json_path": str(retsam_path.relative_to(root)) if retsam_path and retsam_path.exists() else None,
    }
    apply_precomputed_annotations(row, spec["name"], lesions, annotations, sources)

    biomarkers = parse_biomarkers(retsam)
    location = coord_validation(biomarkers)
    if not location["coord_valid"]:
        flags.append(str(location["detail"]["reason"]))
    if not biomarkers["vessel_qc_flag"]:
        flags.append("vessel_qc_low_or_missing")

    image_path = root / "data" / str(row.get("cropped_path") or row.get("src_path") or "")
    if not image_path.exists():
        fallback = root / "data" / str(row.get("src_path") or "")
        image_path = fallback if fallback.exists() else image_path
        if not image_path.exists():
            flags.append("image_path_missing")

    return {
        "record_id": f"{spec['name']}::{spec['split']}::{image_id}",
        "dataset": spec["name"],
        "split": spec["split"],
        "image_id": image_id,
        "image_path": str(image_path.relative_to(root)) if image_path.exists() else str(image_path),
        "src_path": row.get("src_path"),
        "cropped_path": row.get("cropped_path"),
        "crop_box_xyxy": row.get("crop_box_xyxy"),
        "grade": grade,
        "grade_source": "label" if grade is not None and grade >= 0 else "missing",
        "supervision": spec["supervision"],
        "lesions": lesions,
        "biomarkers": biomarkers,
        "location": location,
        "validation_flags": sorted(set(flags)),
        "sources": sources,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".", help="Repository root")
    parser.add_argument("--output", default="data/fundus_validated/validated.jsonl")
    parser.add_argument("--stats-output", default="data/fundus_validated/validated.stats.json")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    output = root / args.output
    stats_output = root / args.stats_output
    output.parent.mkdir(parents=True, exist_ok=True)

    annotations = {
        "stage1_easy": load_stage1_easy_annotations(root),
        "fgadr_lesion_only": load_fgadr_lesion_only_annotations(root),
    }
    fgadr_grades = load_fgadr_grades(root)

    stats: dict[str, Any] = {
        "datasets": defaultdict(Counter),
        "lesion_sources": defaultdict(Counter),
        "flags": Counter(),
        "n_records": 0,
    }

    with output.open("w", encoding="utf-8") as out:
        for spec in DEFAULT_DATASETS:
            crop_meta = root / spec["crop_meta"]
            if not crop_meta.exists():
                continue
            for _, row in iter_jsonl(crop_meta):
                record = build_record(root, spec, row, annotations, fgadr_grades)
                out.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
                stats["n_records"] += 1
                key = f"{spec['name']}::{spec['split']}"
                stats["datasets"][key]["n"] += 1
                if record["sources"]["retsam_json_path"]:
                    stats["datasets"][key]["retsam_present"] += 1
                if record["grade_source"] == "label":
                    stats["datasets"][key]["grade_present"] += 1
                if record["sources"].get("strong_annotation_path"):
                    stats["datasets"][key]["strong_annotation_present"] += 1
                for lesion, data in record["lesions"].items():
                    stats["lesion_sources"][lesion][data.get("source", "unknown")] += 1
                for flag in record["validation_flags"]:
                    stats["flags"][flag] += 1

    serializable_stats = {
        "n_records": stats["n_records"],
        "datasets": {k: dict(v) for k, v in stats["datasets"].items()},
        "lesion_sources": {k: dict(v) for k, v in stats["lesion_sources"].items()},
        "flags": dict(stats["flags"]),
        "output": str(output.relative_to(root)),
    }
    stats_output.write_text(json.dumps(serializable_stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(serializable_stats, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
