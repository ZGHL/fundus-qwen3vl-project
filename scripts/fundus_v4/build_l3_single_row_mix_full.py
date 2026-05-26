#!/usr/bin/env python3
"""Build Arm C single-row mix lesion-perception data from Arm A rows.

Arm C keeps the exact single-target schema and row identity of Arm A, while
changing the prompt framing to six-lesion awareness. It does not label
non-target lesions and does not introduce evidence-limited/null labels.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
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
    "train": "fundus_l3_single_row_mix_full_train_sft.jsonl",
    "full_val": "fundus_l3_single_row_mix_full_val_sft.jsonl",
    "val_subset": "fundus_l3_single_row_mix_val_subset_eval_sft.jsonl",
    "balanced": "fundus_l3_single_row_mix_balanced_eval_sft.jsonl",
    "irma_locked": "fundus_l3_single_row_mix_irma_locked_eval_sft.jsonl",
    "nv_locked": "fundus_l3_single_row_mix_nv_locked_eval_sft.jsonl",
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

LESION_DISTINCTION = {
    "HE": "Differentiate HE from tiny MA and from IRMA/NV vascular patterns; HE is usually larger, darker, or more blot/flame-like.",
    "EX": "Differentiate EX from fluffy SE, imaging glare, and tiny reddish MA; EX is defined by sharper yellow-white deposits.",
    "MA": "Differentiate MA from larger HE and vessel crossings; MA should be tiny, round, and red.",
    "SE": "Differentiate SE from sharper hard exudates and glare; SE should look fluffy, gray-white, and soft-bordered.",
    "IRMA": "Differentiate IRMA from ordinary vessels and NV; IRMA is intraretinal, irregular, and tortuous rather than preretinal new vessels.",
    "NV": "Differentiate NV from IRMA and ordinary retinal vessels; NV should appear as abnormal fine new vessels on the disc or elsewhere.",
}

ARM_C_SYSTEM = """You are a fundus image analyst. You are familiar with all six fundus lesion categories and their distinguishing morphology:
- HE (retinal hemorrhage): dark red dot, blot, or flame-like hemorrhagic lesions.
- EX (hard exudate): bright yellow-white deposits with relatively sharp borders.
- MA (microaneurysm): tiny round red dots, usually smaller than hemorrhages.
- SE (soft exudate / cotton-wool spot): gray-white fluffy cotton-wool patches with soft borders.
- IRMA (intraretinal microvascular abnormality): irregular tortuous intraretinal vascular channels.
- NV (neovascularization): abnormal fine new vessels on the disc or elsewhere.

For each query, the user will specify exactly ONE target lesion. Inspect the image with awareness of all six lesion categories for differential diagnosis, but output only the target lesion's decision. Do not assign labels to non-target lesions. Do not output a final DR grade.

Output exactly four sections: [Global Image Review with Six-Lesion Awareness], [Target Lesion Evidence], [Cross-Lesion Distinction Notes], and [Structured Output]."""


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


def source_label(row: dict[str, Any]) -> dict[str, Any]:
    meta = row.get("meta", {})
    payload = extract_json(row["messages"][-1]["content"])
    lesion = meta.get("lesion") or payload.get("lesion")
    if lesion not in LESIONS:
        raise ValueError(f"invalid lesion: {lesion}")
    present = payload.get("present")
    if not isinstance(present, bool):
        present = meta.get("present_state") == "present"
    return {
        "lesion": lesion,
        "present": bool(present),
        "evidence_state": payload.get("evidence_state") or meta.get("present_state"),
        "strength": payload.get("strength") or ("strong" if present else "absent"),
        "count": payload.get("count", meta.get("count_bucket")),
        "area": payload.get("area", meta.get("area_bucket")),
        "location": payload.get("location", meta.get("location")),
        "source": payload.get("source") or meta.get("source_tag") or meta.get("source"),
    }


def make_user_prompt(lesion: str) -> str:
    full = LESION_FULL[lesion]
    return (
        "<image>\n\n"
        f"The target lesion for THIS query is: {lesion} ({full}).\n\n"
        f"Report your finding for {lesion} only. In [Structured Output], return JSON with "
        f'task="lesion_perception_{lesion}" and fields: lesion, present, '
        "evidence_state, strength, count, area, location, source."
    )


def make_assistant(label: dict[str, Any]) -> str:
    lesion = label["lesion"]
    full = LESION_FULL[lesion]
    visual = LESION_VISUAL[lesion]
    source = label.get("source")
    count = label.get("count")
    area = label.get("area")
    location = label.get("location")

    review = (
        "The image is reviewed with awareness of all six lesion categories for differential diagnosis. "
        f"The final decision is restricted to the target lesion {lesion} ({full}) only; "
        "non-target lesions are not labeled in this response."
    )
    if label["present"]:
        evidence = (
            f"The target lesion is {lesion}. The visual pattern is {visual}. "
            f"Evidence source: {source}. Quantitative profile: count={count or 'unknown'}, "
            f"area={area or 'unknown'}, location={location or 'unspecified location'}."
        )
    else:
        evidence = (
            f"The target lesion is {lesion}. No reliable {visual} pattern is observed or retained. "
            f"Evidence source: {source}. The finding is treated as absent for this target-lesion query."
        )
    payload = {
        "task": f"lesion_perception_{lesion}",
        "lesion": lesion,
        "present": label["present"],
        "evidence_state": label.get("evidence_state"),
        "strength": label.get("strength"),
        "count": count,
        "area": area,
        "location": location if label["present"] else None,
        "source": source,
    }
    return (
        "[Global Image Review with Six-Lesion Awareness]\n"
        + review
        + "\n\n[Target Lesion Evidence]\n"
        + evidence
        + "\n\n[Cross-Lesion Distinction Notes]\n"
        + LESION_DISTINCTION[lesion]
        + "\n\n[Structured Output]\n"
        + json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    )


def transform_row(row: dict[str, Any], split: str) -> dict[str, Any]:
    label = source_label(row)
    meta = dict(row.get("meta", {}))
    meta["task"] = f"lesion_perception_{label['lesion']}"
    meta["arm"] = "arm_c_single_row_mix"
    meta["split"] = split
    return {
        "messages": [
            {"role": "system", "content": ARM_C_SYSTEM},
            {"role": "user", "content": make_user_prompt(label["lesion"])},
            {"role": "assistant", "content": make_assistant(label)},
        ],
        "images": row.get("images", []),
        "meta": meta,
    }


def build_one(name: str, source: Path, out_dir: Path) -> tuple[Path, dict[str, Any]]:
    rows = read_jsonl(source)
    split = "train" if name == "train" else "eval"
    items = [transform_row(row, split) for row in rows]
    out_path = out_dir / OUTPUTS[name]
    write_jsonl(out_path, items)

    counts = Counter()
    sources = Counter()
    for row in items:
        meta = row["meta"]
        lesion = meta.get("lesion")
        state = meta.get("present_state") or ("present" if source_label(row)["present"] else "absent")
        counts[(lesion, state)] += 1
        sources[(lesion, state, meta.get("source_tag") or meta.get("source"))] += 1
    summary = {
        "source": str(source),
        "rows": len(items),
        "counts": {str(k): v for k, v in sorted(counts.items(), key=lambda x: str(x[0]))},
        "sources": {str(k): v for k, v in sorted(sources.items(), key=lambda x: str(x[0]))},
    }
    return out_path, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()

    stats = {
        "version": "fundus_l3_single_row_mix_full",
        "design": [
            "Arm C single-row mix derived from Arm A rows",
            "same row count and target-lesion distribution as Arm A",
            "six-lesion awareness in prompt",
            "single-target output schema identical to Arm A",
            "no non-target lesion labels",
            "no evidence_limited/null labels introduced",
        ],
        "outputs": {},
    }
    for name, source in SOURCES.items():
        out_path, summary = build_one(name, source, args.out_dir)
        stats["outputs"][name] = {"path": str(out_path), **summary}
        print(f"{name}: {summary['rows']} rows -> {out_path}")

    stats_path = args.out_dir / "fundus_l3_single_row_mix_full_stats.json"
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"stats: {stats_path}")


if __name__ == "__main__":
    main()
