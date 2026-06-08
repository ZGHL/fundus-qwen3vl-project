#!/usr/bin/env python3
"""Build a unique-image Stage1 English CoT calibration set.

The calibration pool reuses the existing model-visible CoT while changing the
sampling objective toward strong negatives and lesion-confounder negatives.
DDR gold dev/test and FGADR locked sets are never read by this builder.
"""
from __future__ import annotations

import argparse
import copy
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


LESIONS = ("MA", "HE", "EX", "SE", "IRMA", "NV")
TARGETS = {
    "MA": {"present": 600, "absent": 680},
    "HE": {"present": 1000, "absent": 1000},
    "EX": {"present": 1000, "absent": 1200},
    "SE": {"present": 800, "absent": 1400},
    "IRMA": {"present": 136, "absent": 272},
    "NV": {"present": 37, "absent": 74},
}
CONFOUNDERS = {
    "MA": ("HE",),
    "HE": ("MA",),
    "EX": ("SE",),
    "SE": ("EX",),
    "IRMA": ("NV",),
    "NV": ("IRMA",),
}
TIER_RANK = {"S0": 0, "S1": 1, "S2": 2, "S3": 3, "S4": 4}


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


def deduplicate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    best: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        key = row_key(row)
        current = best.get(key)
        if current is None:
            best[key] = row
            continue
        new_rank = TIER_RANK.get(str(row["meta"].get("evidence_level")), 99)
        old_rank = TIER_RANK.get(str(current["meta"].get("evidence_level")), 99)
        if new_rank < old_rank:
            best[key] = row
    return list(best.values())


def build(args: argparse.Namespace) -> dict[str, Any]:
    rng = random.Random(args.seed)
    source_rows = deduplicate(read_jsonl(args.train) + read_jsonl(args.internal_val))

    image_states: dict[str, dict[str, str]] = defaultdict(dict)
    pools: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in source_rows:
        meta = row["meta"]
        lesion = str(meta["lesion"])
        state = str(meta["present_state"])
        image_group = str(meta["image_group"])
        image_states[image_group][lesion] = state
        pools[(lesion, state)].append(row)

    selected: list[dict[str, Any]] = []
    selection_stats: dict[str, Any] = {}
    for lesion in LESIONS:
        for state in ("present", "absent"):
            candidates = list(pools[(lesion, state)])
            rng.shuffle(candidates)

            def priority(row: dict[str, Any]) -> tuple[int, int, float]:
                meta = row["meta"]
                tier = TIER_RANK.get(str(meta.get("evidence_level")), 99)
                group_states = image_states[str(meta["image_group"])]
                hard_negative = state == "absent" and any(
                    group_states.get(confounder) == "present" for confounder in CONFOUNDERS[lesion]
                )
                return tier, 0 if hard_negative else 1, rng.random()

            candidates.sort(key=priority)
            target = TARGETS[lesion][state]
            if len(candidates) < target:
                raise RuntimeError(
                    f"Insufficient unique candidates for {lesion}/{state}: "
                    f"need {target}, found {len(candidates)}"
                )
            chosen = candidates[:target]
            for row in chosen:
                item = copy.deepcopy(row)
                meta = item["meta"]
                group_states = image_states[str(meta["image_group"])]
                hard_negative = state == "absent" and any(
                    group_states.get(confounder) == "present" for confounder in CONFOUNDERS[lesion]
                )
                meta["split"] = "calibration_train"
                meta["calibration_hard_negative"] = hard_negative
                selected.append(item)

            selection_stats[f"{lesion}_{state}"] = {
                "selected": len(chosen),
                "available_unique": len(candidates),
                "tiers": dict(Counter(str(row["meta"].get("evidence_level")) for row in chosen)),
                "sources": dict(Counter(str(row["meta"].get("evidence_source")) for row in chosen)),
                "hard_negatives": sum(
                    state == "absent"
                    and any(
                        image_states[str(row["meta"]["image_group"])].get(confounder) == "present"
                        for confounder in CONFOUNDERS[lesion]
                    )
                    for row in chosen
                ),
            }

    rng.shuffle(selected)
    write_jsonl(args.output, selected)
    stats = {
        "version": "fundus_stage1_en_cot_calibration_v1",
        "seed": args.seed,
        "inputs": [str(args.train), str(args.internal_val)],
        "output": str(args.output),
        "total_rows": len(selected),
        "unique_lesion_image_pairs": len({row_key(row) for row in selected}),
        "targets": TARGETS,
        "selection": selection_stats,
        "gold_or_locked_inputs_used": False,
    }
    args.stats.write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    return stats


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train",
        type=Path,
        default=Path("data/annotation_v4/fundus_stage1_en_cot_train_sft.jsonl"),
    )
    parser.add_argument(
        "--internal-val",
        type=Path,
        default=Path("data/annotation_v4/fundus_stage1_en_cot_internal_val_sft.jsonl"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/annotation_v4/fundus_stage1_en_cot_calibration_train_sft.jsonl"),
    )
    parser.add_argument(
        "--stats",
        type=Path,
        default=Path("data/annotation_v4/fundus_stage1_en_cot_calibration_stats.json"),
    )
    parser.add_argument("--seed", type=int, default=20260608)
    args = parser.parse_args()
    build(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
