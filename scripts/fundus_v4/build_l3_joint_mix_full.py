#!/usr/bin/env python3
"""Build joint/mix lesion-perception data from the current decoupled pool."""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

LESIONS = ("HE", "EX", "MA", "SE", "IRMA", "NV")
OUT_DIR = Path("data/annotation_v4")

SOURCES = {
    "train": Path("data/annotation_v4/fundus_lesion_perception_en_cot_full_train_sft.jsonl"),
    "full_val": Path("data/annotation_v4/fundus_lesion_perception_en_cot_full_val_sft.jsonl"),
    "val_subset": Path("data/annotation_v4/fundus_lesion_perception_val_subset_eval_sft.jsonl"),
    "balanced": Path("data/annotation_v4/fundus_lesion_perception_balanced_eval_sft.jsonl"),
    "irma_locked": Path("data/annotation_v4/fundus_lesion_perception_irma_locked_eval_sft.jsonl"),
    "nv_locked": Path("data/annotation_v4/fundus_lesion_perception_en_cot_nv_locked_eval_sft.jsonl"),
}

OUTPUTS = {
    "train": "fundus_l3_joint_mix_full_train_sft.jsonl",
    "full_val": "fundus_l3_joint_mix_full_val_sft.jsonl",
    "val_subset": "fundus_l3_joint_mix_val_subset_eval_sft.jsonl",
    "balanced": "fundus_l3_joint_mix_balanced_eval_sft.jsonl",
    "irma_locked": "fundus_l3_joint_mix_irma_locked_eval_sft.jsonl",
    "nv_locked": "fundus_l3_joint_mix_nv_locked_eval_sft.jsonl",
}

LESION_FULL = {
    "HE": "retinal hemorrhage",
    "EX": "hard exudate",
    "MA": "microaneurysm",
    "SE": "soft exudate / cotton-wool spot",
    "IRMA": "intraretinal microvascular abnormality",
    "NV": "neovascularization",
}

LESION_VISUAL = {
    "HE": "dark red dot, blot, or flame-like hemorrhagic lesions",
    "EX": "bright yellow-white deposits with relatively sharp borders",
    "MA": "tiny round red dots, usually smaller than hemorrhages",
    "SE": "gray-white fluffy cotton-wool patches with soft borders",
    "IRMA": "irregular tortuous intraretinal vascular channels",
    "NV": "abnormal fine new vessels on the disc or elsewhere",
}

LESION_REFERENCE = (
    "Lesion reference definitions:\n"
    "- HE (retinal hemorrhage): dark red dot, blot, or flame-like hemorrhagic lesions; "
    "generally larger or more irregular than MA.\n"
    "- EX (hard exudate): bright yellow-white deposits with relatively sharp borders; "
    "distinguish from fluffy SE and imaging glare.\n"
    "- MA (microaneurysm): tiny round red dots, usually smaller than hemorrhages.\n"
    "- SE (soft exudate / cotton-wool spot): gray-white fluffy cotton-wool patches "
    "with soft borders; distinguish from sharper hard exudates.\n"
    "- IRMA (intraretinal microvascular abnormality): irregular tortuous intraretinal "
    "vascular channels; not preretinal new vessels.\n"
    "- NV (neovascularization): abnormal fine new vessels on the disc or elsewhere; "
    "distinguish from IRMA and ordinary retinal vessels."
)

SYSTEM_PROMPT = (
    "You are a fundus image analyst. This is a lesion-perception task. Audit the "
    "six fundus lesion categories separately, compare similar lesion patterns "
    "when needed, and do not output a final DR grade."
)


def make_user_prompt() -> str:
    requested_names = ", ".join(f"{lesion} ({LESION_FULL[lesion]})" for lesion in LESIONS)
    requested_keys = ", ".join(LESIONS)
    json_keys = ", ".join(f'"{lesion}"' for lesion in LESIONS)
    return (
        "<image>\n"
        + LESION_REFERENCE
        + "\n\n"
        f"Identify the lesion status for these six fundus lesion categories: {requested_names}.\n"
        f"For every image, audit all six lesion keys: {requested_keys}. Do not omit "
        "any of these six keys, do not add any other lesion key, and do not output "
        "a final DR grade.\n"
        "Output exactly four sections: [Global Image Review], "
        "[Lesion-by-Lesion Audit], [Cross-Lesion Distinction], and "
        "[Structured Output].\n"
        f"In [Structured Output], return JSON with task=\"joint_lesion_perception\" "
        f"and a lesions object containing exactly these keys: {json_keys}. For each "
        "key, set present to true or false and include evidence_state, strength, "
        "count, area, location, and source."
    )


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def extract_json(text: str) -> dict[str, Any]:
    start = text.find("{")
    if start < 0:
        return {}
    depth = 0
    in_string = False
    escape = False
    for idx in range(start, len(text)):
        ch = text[idx]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start : idx + 1])
                except json.JSONDecodeError:
                    return {}
    return {}


def row_label(row: dict[str, Any]) -> dict[str, Any]:
    meta = row.get("meta", {})
    payload = extract_json(row["messages"][-1]["content"])
    lesion = meta.get("lesion") or payload.get("lesion")
    present = payload.get("present")
    if present is None:
        present = meta.get("present_state") == "present"
    return {
        "lesion": lesion,
        "present": bool(present),
        "evidence_state": payload.get("evidence_state") or meta.get("present_state"),
        "strength": payload.get("strength"),
        "count": payload.get("count", meta.get("count_bucket")),
        "area": payload.get("area", meta.get("area_bucket")),
        "location": payload.get("location", meta.get("location")),
        "source": payload.get("source") or meta.get("source_tag") or meta.get("source"),
    }


def choose_label(old: dict[str, Any] | None, new: dict[str, Any]) -> dict[str, Any]:
    if old is None:
        return new
    if old.get("present") == new.get("present"):
        return old
    # A present label is more informative than an absent duplicate if conflict appears.
    return new if new.get("present") else old


def default_absent_label(lesion: str) -> dict[str, Any]:
    return {
        "lesion": lesion,
        "present": False,
        "evidence_state": "absent",
        "strength": "absent",
        "count": "none",
        "area": "none",
        "location": None,
        "source": "cleaning_rule",
    }


def complete_six_lesions(lesions: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {lesion: lesions.get(lesion) or default_absent_label(lesion) for lesion in LESIONS}


def group_rows(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    stats = {"source_rows": len(rows), "duplicates": 0, "conflicts": []}
    for row in rows:
        images = row.get("images") or []
        if not images:
            continue
        image = images[0]
        meta = row.get("meta", {})
        label = row_label(row)
        lesion = label.get("lesion")
        if lesion not in LESIONS:
            continue
        item = grouped.setdefault(
            image,
            {
                "image": image,
                "images": images,
                "meta": {
                    "image_id": meta.get("image_id"),
                    "dataset": meta.get("dataset"),
                    "grade": meta.get("grade"),
                    "source_records": [],
                },
                "lesions": {},
            },
        )
        if lesion in item["lesions"]:
            stats["duplicates"] += 1
            if item["lesions"][lesion].get("present") != label.get("present"):
                stats["conflicts"].append({"image": image, "lesion": lesion})
        item["lesions"][lesion] = choose_label(item["lesions"].get(lesion), label)
        if meta.get("record_id"):
            item["meta"]["source_records"].append(meta.get("record_id"))
    return list(grouped.values()), stats


def audit_line(lesion: str, label: dict[str, Any]) -> str:
    full = LESION_FULL[lesion]
    if label["present"]:
        count = label.get("count") or "unknown"
        area = label.get("area") or "unknown"
        location = label.get("location") or "unspecified location"
        return (
            f"- {lesion} ({full}): present. Evidence source={label.get('source')}; "
            f"visual pattern={LESION_VISUAL[lesion]}; count={count}; area={area}; location={location}."
        )
    return f"- {lesion} ({full}): absent. Evidence source={label.get('source')}; no reliable {LESION_VISUAL[lesion]} pattern is retained."


def make_assistant(lesions: dict[str, dict[str, Any]]) -> str:
    ordered = complete_six_lesions(lesions)
    present = [lesion for lesion, label in ordered.items() if label["present"]]
    absent = [lesion for lesion, label in ordered.items() if not label["present"]]
    review = (
        "The image is audited for six fundus lesion categories. "
        f"Positive targets: {', '.join(present) if present else 'none'}. "
        f"Absent targets: {', '.join(absent) if absent else 'none'}."
    )
    audit = "\n".join(audit_line(lesion, label) for lesion, label in ordered.items())
    distinction = (
        "Keep red lesions separated from yellow-white exudates; distinguish tiny MA from larger HE; "
        "separate fluffy SE from sharper EX; and do not confuse IRMA with NV or ordinary vessels."
    )
    payload = {
        "task": "joint_lesion_perception",
        "lesions": {
            lesion: {
                "present": label["present"],
                "evidence_state": label.get("evidence_state"),
                "strength": label.get("strength"),
                "count": label.get("count"),
                "area": label.get("area"),
                "location": label.get("location") if label["present"] else None,
                "source": label.get("source"),
            }
            for lesion, label in ordered.items()
        },
    }
    return (
        "[Global Image Review]\n"
        + review
        + "\n\n[Lesion-by-Lesion Audit]\n"
        + audit
        + "\n\n[Cross-Lesion Distinction]\n"
        + distinction
        + "\n\n[Structured Output]\n"
        + json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    )


def make_item(group: dict[str, Any], split: str) -> dict[str, Any]:
    lesions = complete_six_lesions(group["lesions"])
    meta = {
        **group["meta"],
        "task": "joint_lesion_perception",
        "split": split,
        "lesions": {lesion: {"present": label["present"], "source": label.get("source")} for lesion, label in lesions.items()},
        "lesion_count": len(lesions),
        "present_lesions": [lesion for lesion, label in lesions.items() if label["present"]],
        "absent_lesions": [lesion for lesion, label in lesions.items() if not label["present"]],
    }
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": make_user_prompt()},
            {"role": "assistant", "content": make_assistant(lesions)},
        ],
        "images": group["images"],
        "meta": meta,
    }


def summarize(rows: list[dict[str, Any]], source_stats: dict[str, Any]) -> dict[str, Any]:
    lesion_counts = Counter()
    per_row = Counter()
    for row in rows:
        lesions = row["meta"]["lesions"]
        per_row[len(lesions)] += 1
        for lesion, label in lesions.items():
            lesion_counts[(lesion, "present" if label["present"] else "absent")] += 1
    return {
        **source_stats,
        "rows": len(rows),
        "lesion_decisions": sum(lesion_counts.values()),
        "lesions_per_row": dict(sorted(per_row.items())),
        "per_lesion": {str(k): v for k, v in sorted(lesion_counts.items(), key=lambda x: str(x[0]))},
    }


def build_one(name: str, source: Path, out_dir: Path, seed: int) -> tuple[Path, dict[str, Any]]:
    rows = read_jsonl(source)
    groups, source_stats = group_rows(rows)
    items = [make_item(group, "train" if name == "train" else "eval") for group in groups]
    rng = random.Random(seed)
    rng.shuffle(items)
    out_path = out_dir / OUTPUTS[name]
    write_jsonl(out_path, items)
    return out_path, summarize(items, source_stats)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    stats = {"version": "fundus_l3_joint_mix_full", "sources": {}, "outputs": {}}
    for idx, (name, source) in enumerate(SOURCES.items()):
        out_path, summary = build_one(name, source, args.out_dir, args.seed + idx)
        stats["sources"][name] = str(source)
        stats["outputs"][name] = {"path": str(out_path), **summary}
        print(f"{name}: {summary['source_rows']} source rows -> {summary['rows']} joint rows, {summary['lesion_decisions']} decisions")

    stats_path = args.out_dir / "fundus_l3_joint_mix_full_stats.json"
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"stats: {stats_path}")


if __name__ == "__main__":
    main()
