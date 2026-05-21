#!/usr/bin/env python3
"""Clean the unified fundus validated facts for CoT/SFT generation.

This pass keeps the raw validated layer intact and writes a conservative
training-facing copy with suppression flags and task usability metadata.
"""

from __future__ import annotations

import argparse
import copy
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


STRONG_SOURCES = {"strong_mask_stage1_easy", "fgadr_lesion_only_sft_v3"}
RETSAM_SOURCE = "validated_retsam"


def as_float(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    return None


def as_int(value: Any) -> int | None:
    if isinstance(value, int):
        return value
    return None


def add_flag(record: dict[str, Any], flag: str) -> None:
    flags = record.setdefault("cleaning_flags", [])
    if flag not in flags:
        flags.append(flag)


def suppress_lesion(data: dict[str, Any], reason: str, keep_raw: bool = True) -> None:
    if keep_raw and "raw_present" not in data:
        data["raw_present"] = data.get("present")
        data["raw_source"] = data.get("source")
    data["present"] = False
    data["source"] = "cleaning_rule"
    data["suppressed_reason"] = reason


def clean_retsam_lesion(record: dict[str, Any], lesion: str, data: dict[str, Any]) -> None:
    if data.get("source") != RETSAM_SOURCE or data.get("present") is not True:
        return

    count = as_int(data.get("count")) or 0
    area = as_float(data.get("area")) or 0.0
    conf = as_float(data.get("confidence")) or 0.0
    grade = record.get("grade")

    if lesion == "HE":
        ok = count >= 2 and area >= 50 and conf >= 0.25
        reason = "retsam_he_below_min_count_area_conf"
    elif lesion == "EX":
        ok = count >= 1 and area >= 50 and conf >= 0.25
        reason = "retsam_ex_below_min_area_conf"
    elif lesion == "SE":
        # Cotton-wool spots are the noisiest RetSAM field in this project.
        # Keep only moderate+ grade contexts with more than a tiny one-blob hit.
        ok = isinstance(grade, int) and grade >= 2 and count >= 2 and area >= 300 and conf >= 0.5
        reason = "retsam_se_low_conf_or_tiny"
    else:
        return

    if not ok:
        suppress_lesion(data, reason)
        add_flag(record, reason)


def clean_grade_consistency(record: dict[str, Any]) -> None:
    grade = record.get("grade")
    lesions = record.get("lesions", {})
    if not isinstance(grade, int) or grade < 0:
        return

    if grade == 0:
        for lesion in ["HE", "EX", "SE"]:
            data = lesions.get(lesion, {})
            if data.get("present") is True and data.get("source") not in STRONG_SOURCES:
                suppress_lesion(data, f"{lesion.lower()}_suppressed_by_grade0")
                add_flag(record, f"{lesion.lower()}_suppressed_by_grade0")
            elif data.get("present") is True and data.get("source") in STRONG_SOURCES:
                add_flag(record, f"{lesion.lower()}_strong_mask_conflicts_grade0")
        ma = lesions.get("MA")
        if ma and ma.get("source") not in STRONG_SOURCES:
            ma["present"] = False
            ma["source"] = "grade_rule"

    if grade == 1:
        for lesion in ["HE", "EX", "SE"]:
            data = lesions.get(lesion, {})
            if data.get("present") is True and data.get("source") not in STRONG_SOURCES:
                suppress_lesion(data, f"{lesion.lower()}_suppressed_by_grade1")
                add_flag(record, f"{lesion.lower()}_suppressed_by_grade1")
            elif data.get("present") is True and data.get("source") in STRONG_SOURCES:
                add_flag(record, f"{lesion.lower()}_strong_mask_conflicts_grade1")
        ma = lesions.get("MA")
        if ma and ma.get("source") not in STRONG_SOURCES:
            ma["present"] = "template_only"
            ma["source"] = "grade_rule"
            ma["note"] = "MA is inferred from Grade 1 rule template, not RetSAM"


def clean_biomarkers(record: dict[str, Any]) -> None:
    biomarkers = record.get("biomarkers", {})
    vessel_qc = bool(biomarkers.get("vessel_qc_flag"))
    od_qc = bool(biomarkers.get("od_qc_flag"))

    # coord_valid checks whether OD/macula coordinates are geometrically usable
    # for distances and quadrants. It should not override RetSAM laterality when
    # the optic-disc module itself passed QC.

    if not od_qc:
        for key in ["eye_side", "cdr"]:
            val = biomarkers.get(key)
            if isinstance(val, dict) and val.get("valid"):
                val["valid"] = False
                val["cleaned_reason"] = "od_qc_failed"
                add_flag(record, f"{key}_invalid_due_to_od_qc")

    if not vessel_qc:
        for key in ["av_ratio", "tortuosity", "crae", "crve", "fractal_dimension"]:
            val = biomarkers.get(key)
            if isinstance(val, dict):
                val["valid"] = False
                val["cleaned_reason"] = "vessel_qc_failed_or_missing"


def set_task_usability(record: dict[str, Any]) -> None:
    lesions = record.get("lesions", {})
    biomarkers = record.get("biomarkers", {})
    has_strong_l3 = any(v.get("source") in STRONG_SOURCES for v in lesions.values() if isinstance(v, dict))
    has_retsam_l3 = any(v.get("source") == RETSAM_SOURCE for k, v in lesions.items() if k in {"HE", "EX", "SE"} and v.get("present") is True)
    has_l2 = any(
        isinstance(biomarkers.get(k), dict) and biomarkers[k].get("valid")
        for k in ["eye_side", "cdr", "av_ratio", "tortuosity"]
    )
    grade = record.get("grade")
    l4_evidence = any(
        lesions.get(k, {}).get("present") is True
        for k in ["HE", "EX", "SE", "MA", "NV"]
    ) or grade in {0, 1}

    record["usable_for"] = {
        "L2": bool(has_l2),
        "L3": bool(has_strong_l3 or has_retsam_l3),
        "L4": bool(isinstance(grade, int) and grade >= 0 and l4_evidence),
        "strong_L3": bool(has_strong_l3),
        "retsam_L3": bool(has_retsam_l3),
    }


def clean_record(record: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(record)
    out["cleaning_version"] = "v1"
    out["cleaning_flags"] = []

    for lesion in ["HE", "EX", "SE"]:
        data = out.get("lesions", {}).get(lesion)
        if isinstance(data, dict):
            clean_retsam_lesion(out, lesion, data)

    clean_grade_consistency(out)
    clean_biomarkers(out)
    set_task_usability(out)
    out["cleaning_flags"] = sorted(out["cleaning_flags"])
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/fundus_validated/validated.jsonl")
    parser.add_argument("--output", default="data/fundus_validated/validated_clean.jsonl")
    parser.add_argument("--stats-output", default="data/fundus_validated/validated_clean.stats.json")
    args = parser.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)
    stats_path = Path(args.stats_output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    stats: dict[str, Any] = {
        "n_records": 0,
        "datasets": defaultdict(Counter),
        "present": defaultdict(Counter),
        "sources": defaultdict(Counter),
        "usable_for": defaultdict(Counter),
        "cleaning_flags": Counter(),
        "validation_flags": Counter(),
    }

    with in_path.open("r", encoding="utf-8") as src, out_path.open("w", encoding="utf-8") as dst:
        for line in src:
            if not line.strip():
                continue
            record = clean_record(json.loads(line))
            dst.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")

            stats["n_records"] += 1
            ds = f"{record['dataset']}::{record['split']}"
            stats["datasets"][ds]["n"] += 1
            for key, usable in record.get("usable_for", {}).items():
                if usable:
                    stats["usable_for"][key][ds] += 1
            for lesion, data in record.get("lesions", {}).items():
                if not isinstance(data, dict):
                    continue
                stats["sources"][lesion][data.get("source", "unknown")] += 1
                if data.get("present") is True:
                    stats["present"][lesion][ds] += 1
            stats["cleaning_flags"].update(record.get("cleaning_flags", []))
            stats["validation_flags"].update(record.get("validation_flags", []))

    serializable = {
        "n_records": stats["n_records"],
        "datasets": {k: dict(v) for k, v in stats["datasets"].items()},
        "present": {k: dict(v) for k, v in stats["present"].items()},
        "present_totals": {k: sum(v.values()) for k, v in stats["present"].items()},
        "sources": {k: dict(v) for k, v in stats["sources"].items()},
        "usable_for": {k: dict(v) for k, v in stats["usable_for"].items()},
        "usable_totals": {k: sum(v.values()) for k, v in stats["usable_for"].items()},
        "cleaning_flags": dict(stats["cleaning_flags"]),
        "validation_flags_top": dict(stats["validation_flags"].most_common(20)),
        "input": str(in_path),
        "output": str(out_path),
    }
    stats_path.write_text(json.dumps(serializable, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(serializable, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
