#!/usr/bin/env python3
"""Build the limited Stage1.5 six-lesion, strictly decoupled SFT set."""
from __future__ import annotations

import argparse
import copy
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

LESIONS = ("MA", "HE", "EX", "SE", "IRMA", "NV")
QUOTAS = {
    "MA": {"present": 600, "absent": 600},
    "HE": {"present": 600, "absent": 600},
    "EX": {"present": 600, "absent": 600},
    "SE": {"present": 600, "absent": 600},
    "IRMA": {"present": 300, "absent": 300},
    "NV": {"present": 140, "absent": 140},
}
CONFOUNDERS = {
    "MA": ("HE",),
    "HE": ("MA",),
    "EX": ("SE",),
    "SE": ("EX",),
    "IRMA": ("NV", "HE"),
    "NV": ("IRMA", "HE"),
}
MAX_VIEWS = {"MA": 3, "IRMA": 3, "NV": 5}
TIER_RANK = {"S0": 0, "S1": 1}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, default=Path("data/annotation_v4/fundus_stage1_en_cot_train_sft.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("data/annotation_v4/fundus_stage1_5_six_lesion_train_sft.jsonl"))
    parser.add_argument("--stats", type=Path, default=Path("data/annotation_v4/fundus_stage1_5_six_lesion_stats.json"))
    parser.add_argument("--seed", type=int, default=20260609)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    rows = read_jsonl(args.train)
    states: dict[str, dict[str, str]] = defaultdict(dict)
    for row in rows:
        meta = row["meta"]
        states[str(meta["image_group"])][str(meta["lesion"])] = str(meta["present_state"])

    selected: list[dict[str, Any]] = []
    report: dict[str, Any] = {}
    for lesion in LESIONS:
        lesion_report: dict[str, Any] = {}
        for state in ("present", "absent"):
            pool = [
                row for row in rows
                if row["meta"]["lesion"] == lesion
                and row["meta"]["present_state"] == state
                and row["meta"].get("evidence_level") in TIER_RANK
            ]
            rng.shuffle(pool)
            if state == "absent":
                conf = CONFOUNDERS[lesion]
                pool.sort(key=lambda row: (
                    0 if any(states[row["meta"]["image_group"]].get(c) == "present" for c in conf) else 1,
                    TIER_RANK[row["meta"]["evidence_level"]],
                ))
            else:
                pool.sort(key=lambda row: TIER_RANK[row["meta"]["evidence_level"]])

            quota = QUOTAS[lesion][state]
            if not pool:
                raise RuntimeError(f"No rows for {lesion}/{state}")
            max_views = MAX_VIEWS.get(lesion, 2)
            chosen: list[dict[str, Any]] = []
            uses: Counter[str] = Counter()
            while len(chosen) < quota:
                progress = False
                for row in pool:
                    group = str(row["meta"]["image_group"])
                    if uses[group] >= max_views:
                        continue
                    chosen.append(row)
                    uses[group] += 1
                    progress = True
                    if len(chosen) >= quota:
                        break
                if not progress:
                    raise RuntimeError(
                        f"Insufficient controlled views for {lesion}/{state}: "
                        f"need {quota}, built {len(chosen)}, unique {len(pool)}"
                    )

            hard = 0
            for row in chosen:
                item = copy.deepcopy(row)
                meta = item["meta"]
                meta["split"] = "stage1_5_six_lesion_train"
                meta["stage1_5"] = True
                meta["source_image_group"] = meta["image_group"]
                meta["view_index"] = uses[meta["image_group"]]
                if state == "absent":
                    is_hard = any(states[meta["image_group"]].get(c) == "present" for c in CONFOUNDERS[lesion])
                    meta["hard_negative"] = is_hard
                    hard += int(is_hard)
                selected.append(item)

            lesion_report[state] = {
                "rows": len(chosen),
                "unique_images": len(uses),
                "max_views": max(uses.values()),
                "tiers": dict(Counter(row["meta"]["evidence_level"] for row in chosen)),
                "sources": dict(Counter(row["meta"]["evidence_source"] for row in chosen)),
                "hard_negatives": hard if state == "absent" else 0,
            }
        report[lesion] = lesion_report

    rng.shuffle(selected)
    write_jsonl(args.output, selected)
    stats = {
        "version": "fundus_stage1_5_six_lesion_v1",
        "seed": args.seed,
        "total_rows": len(selected),
        "quotas": QUOTAS,
        "gold_internal_val_or_locked_inputs_used": False,
        "selection": report,
    }
    args.stats.write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
