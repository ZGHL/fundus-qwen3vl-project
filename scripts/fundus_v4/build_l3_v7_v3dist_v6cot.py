#!/usr/bin/env python3
"""Build L3 v7 data: v3 six-lesion balanced distribution with v6 CoT.

This experiment keeps the best previous L3 sampling recipe:
  - six single-lesion tasks
  - 1200 samples per lesion
  - 600 positive + 600 negative per lesion

It rewrites the assistant answer into the v6 four-section explicit CoT format
so we can test whether v6 prompt/answer structure helps when the data
distribution is kept stable.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


SRC = Path("data/annotation/fundus_l3_six_lesion_calib_pilot_sft.jsonl")
OUT_DIR = Path("data/annotation_v4")
LESIONS = ("MA", "HE", "EX", "SE", "IRMA", "NV")

LESION_FULL = {
    "MA": "microaneurysm",
    "HE": "retinal hemorrhage",
    "EX": "hard exudate",
    "SE": "soft exudate / cotton-wool spot",
    "IRMA": "intraretinal microvascular abnormality",
    "NV": "neovascularization",
}

LESION_VISUAL = {
    "MA": "tiny round red dots, usually smaller and more sharply punctate than hemorrhages",
    "HE": "dark red dot, blot, or flame-like hemorrhagic lesions with soft or irregular margins",
    "EX": "bright yellow-white deposits with relatively sharp borders",
    "SE": "gray-white fluffy cotton-wool patches with indistinct borders",
    "IRMA": "irregular tortuous intraretinal vascular channels near areas of ischemia",
    "NV": "abnormal fine new vessels on the disc or elsewhere, often crossing normal vessel planes",
}

DEFAULT_LOCATION = {
    "MA": "posterior retina",
    "HE": "posterior pole / midperiphery",
    "EX": "posterior pole / midperiphery",
    "SE": "posterior pole / midperiphery",
    "IRMA": "intraretinal, near major vessels",
    "NV": "at disc or elsewhere",
}

SYSTEM_PROMPT_TEMPLATE = (
    "You are a fundus image analyst. Inspect ONLY for {lesion} ({lesion_full}). "
    "This is a single-lesion perception task: do NOT output a final DR grade and "
    "do NOT combine other lesions. First describe visible morphology and location, "
    "then judge whether the target lesion is present."
)

USER_PROMPT_TEMPLATE = (
    "<image>\n"
    "Examine this fundus image for {lesion_full} ({lesion}). "
    "Output exactly four sections: [Lesion Existence and Evidence Judgment], "
    "[Basic Morphological and Location Features], [Decision Notes for This Single-Lesion Task], "
    "and [Structured Output]."
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
    tail = text.split("【JSON】", 1)[-1] if "【JSON】" in text else text
    match = re.search(r"\{.*\}", tail, flags=re.S)
    if match is None:
        raise ValueError("no JSON object found in assistant label")
    return json.loads(match.group(0))


def answer_obj(row: dict[str, Any]) -> dict[str, Any]:
    return extract_json(row["messages"][-1]["content"])


def source_tag(source: str | None) -> str:
    if not source:
        return "unknown"
    if source == "fgadr_lesion_only_sft_v3":
        return "strong_mask"
    return source


def decision_note(lesion: str, present: bool) -> str:
    evidence = (
        "This is a target-lesion positive example for learning visible morphology."
        if present
        else "This is treated as a target-lesion negative example for learning the absent boundary."
    )
    notes = {
        "MA": "Focus on tiny round red dots; do not confuse them with vessel crossings, isolated noise, or larger hemorrhages.",
        "HE": "Focus on dark red hemorrhagic lesions; do not call hard exudates or normal vessels hemorrhage.",
        "EX": "Focus on sharply bordered yellow-white lipid deposits; distinguish them from fluffy cotton-wool spots.",
        "SE": "Focus on fluffy gray-white cotton-wool patches with soft borders; distinguish them from hard exudates and glare.",
        "IRMA": "Focus on abnormal intraretinal vascular channels; distinguish IRMA from NV and from normal vessel branching.",
        "NV": "Focus on abnormal new vessels on the disc or elsewhere; distinguish NV from IRMA and ordinary vessels.",
    }
    return f"{evidence} {notes[lesion]} No final DR grade is assigned in this L3 task."


def make_assistant(lesion: str, present: bool, label_obj: dict[str, Any], source: str | None) -> str:
    full = LESION_FULL[lesion]
    source = source_tag(source)
    count = label_obj.get("count")
    area = label_obj.get("area")
    location = label_obj.get("location") or DEFAULT_LOCATION[lesion]
    if present:
        existence = f"{full} is present with direct evidence from {source}."
        morphology = (
            f"The target finding is described as {LESION_VISUAL[lesion]}. "
            f"In this sample it is annotated as count={count or 'unknown'}, "
            f"area={area or 'unknown'}, located {location}."
        )
        evidence_state = "present"
        strength = "strong"
        out_location = location
    else:
        existence = f"{full} is absent according to direct negative evidence from {source}."
        morphology = f"No reliable target-pattern evidence is identified for {lesion}."
        evidence_state = "absent"
        strength = "absent"
        out_location = None

    payload = {
        "task": f"L3_{lesion}",
        "lesion": lesion,
        "present": present,
        "evidence_state": evidence_state,
        "strength": strength,
        "count": count,
        "area": area,
        "location": out_location,
        "source": source,
    }
    return (
        "[Lesion Existence and Evidence Judgment]\n"
        + existence
        + "\n\n[Basic Morphological and Location Features]\n"
        + morphology
        + "\n\n[Decision Notes for This Single-Lesion Task]\n"
        + decision_note(lesion, present)
        + "\n\n[Structured Output]\n"
        + json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    )


def convert_row(row: dict[str, Any], split: str) -> dict[str, Any]:
    obj = answer_obj(row)
    lesion = obj["lesion"]
    present = bool(obj["present"])
    meta = dict(row.get("meta", {}))
    source = obj.get("source") or meta.get("source") or meta.get("source_file") or "unknown"
    meta.update(
        {
            "task": f"L3_{lesion}",
            "lesion": lesion,
            "present_state": "present" if present else "absent",
            "split": split,
            "source": source,
            "source_tag": source_tag(source),
            "stage_mix": "l3_v7_v3dist_v6cot",
            "v7_source_task": obj.get("task"),
        }
    )
    system = SYSTEM_PROMPT_TEMPLATE.format(lesion=lesion, lesion_full=LESION_FULL[lesion])
    user = USER_PROMPT_TEMPLATE.format(lesion=lesion, lesion_full=LESION_FULL[lesion])
    return {
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
            {"role": "assistant", "content": make_assistant(lesion, present, obj, source)},
        ],
        "images": row["images"],
        "meta": meta,
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    tasks = Counter(row["meta"]["task"] for row in rows)
    present = Counter((row["meta"]["task"], row["meta"]["present_state"]) for row in rows)
    sources = Counter((row["meta"]["task"], row["meta"].get("source_tag")) for row in rows)
    return {
        "n": len(rows),
        "tasks": dict(tasks),
        "present": {str(k): v for k, v in sorted(present.items(), key=lambda x: str(x[0]))},
        "sources": {str(k): v for k, v in sorted(sources.items(), key=lambda x: str(x[0]))},
    }


def build(args: argparse.Namespace) -> dict[str, Any]:
    rows = [convert_row(row, "train") for row in read_jsonl(args.source)]
    rng = random.Random(args.seed)
    rng.shuffle(rows)
    stats = {
        "version": "fundus_l3_v7_v3dist_v6cot",
        "design": [
            "v3_six_lesion_balanced_distribution",
            "v6_four_section_explicit_cot",
            "single_lesion_l3_only",
            "no_dr_grade_output",
        ],
        "source": str(args.source),
        "seed": args.seed,
        "train": summarize(rows),
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    train_path = args.out_dir / "fundus_l3_v7_v3dist_v6cot_train_sft.jsonl"
    stats_path = args.out_dir / "fundus_l3_v7_v3dist_v6cot_stats.json"
    if not args.dry_run:
        write_jsonl(train_path, rows)
        stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    print(f"train: {train_path}")
    print(f"stats: {stats_path}")
    return stats


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=SRC)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--seed", type=int, default=20260519)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    build(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
