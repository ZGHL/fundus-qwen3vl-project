#!/usr/bin/env python3
"""Build conservative Stage1 replay calibration data.

Keep the complete Stage1 training distribution, then add balanced, unseen,
high-confidence examples from internal validation. Weak S3/S4 labels are
excluded so calibration cannot dominate the original lesion-present prior.
"""
from __future__ import annotations

import argparse
import copy
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


LESIONS = ("MA", "HE", "EX", "SE")
TRUSTED_TIERS = {"S0", "S1", "S2"}
MAX_ADDITIONS = {"MA": 30, "HE": 140, "EX": 225, "SE": 190}
CONFOUNDERS = {"MA": "HE", "HE": "MA", "EX": "SE", "SE": "EX"}
TIER_RANK = {"S0": 0, "S1": 1, "S2": 2}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def key(row: dict[str, Any]) -> tuple[str, str]:
    meta = row["meta"]
    return str(meta["lesion"]), str(meta["image_group"])


def build(args: argparse.Namespace) -> dict[str, Any]:
    rng = random.Random(args.seed)
    train = read_jsonl(args.train)
    internal = read_jsonl(args.internal_val)
    train_keys = {key(row) for row in train}

    image_states: dict[str, dict[str, str]] = defaultdict(dict)
    for row in train + internal:
        meta = row["meta"]
        image_states[str(meta["image_group"])][str(meta["lesion"])] = str(meta["present_state"])

    additions: list[dict[str, Any]] = []
    stats: dict[str, Any] = {}
    for lesion in LESIONS:
        candidates = [
            row
            for row in internal
            if str(row["meta"]["lesion"]) == lesion
            and key(row) not in train_keys
            and str(row["meta"].get("evidence_level")) in TRUSTED_TIERS
        ]
        positives = [row for row in candidates if str(row["meta"]["present_state"]) == "present"]
        negatives = [row for row in candidates if str(row["meta"]["present_state"]) == "absent"]

        def negative_priority(row: dict[str, Any]) -> tuple[int, int, float]:
            meta = row["meta"]
            hard = image_states[str(meta["image_group"])].get(CONFOUNDERS[lesion]) == "present"
            return (0 if hard else 1, TIER_RANK[str(meta["evidence_level"])], rng.random())

        def positive_priority(row: dict[str, Any]) -> tuple[int, float]:
            return TIER_RANK[str(row["meta"]["evidence_level"])], rng.random()

        negatives.sort(key=negative_priority)
        positives.sort(key=positive_priority)
        count = min(MAX_ADDITIONS[lesion], len(negatives), len(positives))
        chosen_negatives = negatives[:count]
        chosen_positives = positives[:count]

        selected = chosen_positives + chosen_negatives
        for row in selected:
            item = copy.deepcopy(row)
            meta = item["meta"]
            meta["split"] = "replay_calibration_train"
            meta["replay_calibration_addition"] = True
            meta["calibration_hard_negative"] = (
                str(meta["present_state"]) == "absent"
                and image_states[str(meta["image_group"])].get(CONFOUNDERS[lesion]) == "present"
            )
            additions.append(item)

        stats[lesion] = {
            "added_present": len(chosen_positives),
            "added_absent": len(chosen_negatives),
            "hard_negatives": sum(
                image_states[str(row["meta"]["image_group"])].get(CONFOUNDERS[lesion]) == "present"
                for row in chosen_negatives
            ),
            "present_tiers": dict(Counter(str(row["meta"]["evidence_level"]) for row in chosen_positives)),
            "absent_tiers": dict(Counter(str(row["meta"]["evidence_level"]) for row in chosen_negatives)),
        }

    output = [copy.deepcopy(row) for row in train] + additions
    rng.shuffle(output)
    write_jsonl(args.output, output)
    result = {
        "version": "fundus_stage1_en_cot_replay_calibration_v2",
        "seed": args.seed,
        "base_replay_rows": len(train),
        "new_balanced_rows": len(additions),
        "total_rows": len(output),
        "trusted_tiers": sorted(TRUSTED_TIERS),
        "weak_tiers_excluded": ["S3", "S4"],
        "gold_or_locked_inputs_used": False,
        "selection": stats,
    }
    args.stats.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, default=Path("data/annotation_v4/fundus_stage1_en_cot_train_sft.jsonl"))
    parser.add_argument("--internal-val", type=Path, default=Path("data/annotation_v4/fundus_stage1_en_cot_internal_val_sft.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("data/annotation_v4/fundus_stage1_en_cot_replay_calibration_train_sft.jsonl"))
    parser.add_argument("--stats", type=Path, default=Path("data/annotation_v4/fundus_stage1_en_cot_replay_calibration_stats.json"))
    parser.add_argument("--seed", type=int, default=20260609)
    args = parser.parse_args()
    build(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
