#!/usr/bin/env python3
"""Build a positive-anchored gentle Stage1 calibration set."""
from __future__ import annotations

import argparse
import copy
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

LESIONS = ("MA", "HE", "EX", "SE")
QUOTAS = {
    "MA": {"present": 450, "absent": 150},
    "HE": {"present": 500, "absent": 100},
    "EX": {"present": 500, "absent": 100},
    "SE": {"present": 450, "absent": 250},
}
CONFOUNDERS = {"MA": "HE", "HE": "MA", "EX": "SE", "SE": "EX"}
TIER_RANK = {"S0": 0, "S1": 1}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def row_key(row: dict[str, Any]) -> tuple[str, str]:
    meta = row["meta"]
    return str(meta["lesion"]), str(meta["image_group"])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, default=Path("data/annotation_v4/fundus_stage1_en_cot_train_sft.jsonl"))
    parser.add_argument("--internal-val", type=Path, default=Path("data/annotation_v4/fundus_stage1_en_cot_internal_val_sft.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("data/annotation_v4/fundus_stage1_en_cot_gentle_calibration_train_sft.jsonl"))
    parser.add_argument("--stats", type=Path, default=Path("data/annotation_v4/fundus_stage1_en_cot_gentle_calibration_stats.json"))
    parser.add_argument("--seed", type=int, default=20260609)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    rows = read_jsonl(args.train) + read_jsonl(args.internal_val)
    states: dict[str, dict[str, str]] = defaultdict(dict)
    for row in rows:
        meta = row["meta"]
        states[str(meta["image_group"])][str(meta["lesion"])] = str(meta["present_state"])

    selected: list[dict[str, Any]] = []
    stats: dict[str, Any] = {}
    used: set[tuple[str, str]] = set()
    for lesion in LESIONS:
        lesion_stats: dict[str, Any] = {}
        for state in ("present", "absent"):
            pool = [
                row for row in rows
                if str(row["meta"]["lesion"]) == lesion
                and str(row["meta"]["present_state"]) == state
                and str(row["meta"].get("evidence_level")) in TIER_RANK
                and row_key(row) not in used
            ]
            rng.shuffle(pool)
            confounder = CONFOUNDERS[lesion]
            if state == "absent":
                pool.sort(key=lambda row: (
                    0 if states[str(row["meta"]["image_group"])].get(confounder) == "present" else 1,
                    TIER_RANK[str(row["meta"]["evidence_level"])],
                ))
            else:
                pool.sort(key=lambda row: TIER_RANK[str(row["meta"]["evidence_level"])])
            quota = QUOTAS[lesion][state]
            if len(pool) < quota:
                raise RuntimeError(f"Insufficient {lesion}/{state}: need {quota}, found {len(pool)}")
            chosen = pool[:quota]
            for row in chosen:
                used.add(row_key(row))
                item = copy.deepcopy(row)
                item["meta"]["split"] = "gentle_calibration_train"
                item["meta"]["gentle_calibration"] = True
                item["meta"]["calibration_role"] = "repair" if lesion in {"MA", "SE"} else "anchor"
                item["meta"]["calibration_hard_negative"] = (
                    state == "absent"
                    and states[str(item["meta"]["image_group"])].get(confounder) == "present"
                )
                selected.append(item)
            lesion_stats[state] = {
                "selected": len(chosen),
                "tiers": dict(Counter(str(row["meta"]["evidence_level"]) for row in chosen)),
                "sources": dict(Counter(str(row["meta"].get("evidence_source")) for row in chosen)),
                "hard_negatives": sum(
                    states[str(row["meta"]["image_group"])].get(confounder) == "present" for row in chosen
                ) if state == "absent" else 0,
            }
        stats[lesion] = lesion_stats

    rng.shuffle(selected)
    write_jsonl(args.output, selected)
    result = {
        "version": "fundus_stage1_gentle_calibration_v1",
        "seed": args.seed,
        "total_rows": len(selected),
        "trusted_tiers": sorted(TIER_RANK),
        "quotas": QUOTAS,
        "gold_or_locked_inputs_used": False,
        "selection": stats,
    }
    args.stats.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
