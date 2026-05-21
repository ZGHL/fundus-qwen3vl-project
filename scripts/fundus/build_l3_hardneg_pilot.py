#!/usr/bin/env python3
"""Build a hard-negative L3 lesion presence pilot set."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


LESIONS = ["MA", "HE", "EX", "SE"]
NEG_SOURCE_PRIORITY = {
    "fgadr_lesion_only_sft_v3": 0,
    "retsam_negative": 1,
    "cleaning_rule": 2,
    "grade0_rule_negative": 3,
}
POS_SOURCE_PRIORITY = {
    "strong_mask_stage1_easy": 0,
    "validated_retsam": 1,
    "fgadr_lesion_only_sft_v3": 2,
}
LESION_NEG_SOURCE_PRIORITY = {
    "SE": {
        "cleaning_rule": 0,
        "retsam_negative": 1,
        "grade_rule_override": 2,
        "fgadr_lesion_only_sft_v3": 3,
        "grade0_rule_negative": 4,
    },
    "MA": {
        "fgadr_lesion_only_sft_v3": 0,
        "grade0_rule_negative": 1,
        "cleaning_rule": 2,
        "retsam_negative": 3,
    },
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def source_rank(row: dict[str, Any], priority: dict[str, int]) -> int:
    return priority.get(row.get("meta", {}).get("source"), 99)


def take(rows: list[dict[str, Any]], n: int, rng: random.Random, priority: dict[str, int]) -> list[dict[str, Any]]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[source_rank(row, priority)].append(row)
    out: list[dict[str, Any]] = []
    for rank in sorted(grouped):
        bucket = grouped[rank]
        rng.shuffle(bucket)
        need = n - len(out)
        if need <= 0:
            break
        out.extend(bucket[:need])
    if len(out) < n:
        rest_ids = {id(r) for r in out}
        rest = [r for r in rows if id(r) not in rest_ids]
        rng.shuffle(rest)
        out.extend(rest[: n - len(out)])
    return out[:n]


def parse_lesion_plan(items: list[str] | None, default_pos: int, default_neg: int) -> dict[str, tuple[int, int]]:
    plan = {lesion: (default_pos, default_neg) for lesion in LESIONS}
    if not items:
        return plan
    for item in items:
        parts = item.split(":")
        if len(parts) != 3:
            raise ValueError(f"Invalid lesion plan item {item!r}; expected LESION:POS:NEG")
        lesion, pos_s, neg_s = parts
        if lesion not in LESIONS:
            raise ValueError(f"Invalid lesion {lesion!r}; expected one of {LESIONS}")
        plan[lesion] = (int(pos_s), int(neg_s))
    return plan


def parse_float_map(items: list[str] | None) -> dict[str, float]:
    out: dict[str, float] = {}
    for item in items or []:
        lesion, value = item.split(":", 1)
        if lesion not in LESIONS:
            raise ValueError(f"Invalid lesion {lesion!r}; expected one of {LESIONS}")
        out[lesion] = float(value)
    return out


def load_validated(path: str | None) -> dict[str, dict[str, Any]]:
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        return {}
    records: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(p):
        rid = row.get("record_id")
        if rid:
            records[rid] = row
    return records


def confidence_for(row: dict[str, Any], lesion: str, validated: dict[str, dict[str, Any]]) -> float | None:
    rid = row.get("meta", {}).get("record_id")
    d = validated.get(rid or "", {}).get("lesions", {}).get(lesion, {})
    conf = d.get("confidence")
    return float(conf) if isinstance(conf, (int, float)) else None


def passes_min_confidence(
    row: dict[str, Any],
    lesion: str,
    min_pos_confidence: dict[str, float],
    validated: dict[str, dict[str, Any]],
) -> bool:
    min_conf = min_pos_confidence.get(lesion)
    if min_conf is None:
        return True
    src = row.get("meta", {}).get("source")
    if src in {"strong_mask_stage1_easy", "fgadr_lesion_only_sft_v3"}:
        return True
    conf = confidence_for(row, lesion, validated)
    return conf is not None and conf >= min_conf


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    task = Counter()
    src = Counter()
    ds = Counter()
    present = Counter()
    for row in rows:
        meta = row.get("meta", {})
        task[meta.get("task", "unknown")] += 1
        src[(meta.get("lesion"), meta.get("present"), meta.get("source"))] += 1
        present[(meta.get("lesion"), meta.get("present"))] += 1
        ds[(meta.get("lesion"), meta.get("present"), "::".join(meta.get("record_id", "").split("::")[:2]))] += 1
    return {
        "n": len(rows),
        "tasks": dict(task),
        "present": {str(k): v for k, v in present.items()},
        "sources": {str(k): v for k, v in src.items()},
        "datasets": {str(k): v for k, v in ds.items()},
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/annotation/fundus_l3_single_lesion_sft.jsonl")
    ap.add_argument("--holdout", default="data/annotation/fundus_l3_presence_holdout80_sft.jsonl")
    ap.add_argument("--output", default="data/annotation/fundus_l3_hardneg_pilot_sft.jsonl")
    ap.add_argument("--stats", default="data/annotation/fundus_l3_hardneg_pilot_stats.json")
    ap.add_argument("--pos-per-lesion", type=int, default=200)
    ap.add_argument("--neg-per-lesion", type=int, default=400)
    ap.add_argument(
        "--lesion-plan",
        nargs="*",
        default=None,
        help="Optional per-lesion counts, e.g. EX:300:400 HE:300:450 MA:250:650 SE:250:600.",
    )
    ap.add_argument("--validated-clean", default="data/fundus_validated/validated_clean.jsonl")
    ap.add_argument(
        "--min-pos-confidence",
        nargs="*",
        default=None,
        help="Optional minimum positive confidence for RetSAM positives, e.g. SE:0.9.",
    )
    ap.add_argument(
        "--lesion-specific-neg-priority",
        action="store_true",
        help="Use lesion-specific negative source priorities for calibration.",
    )
    ap.add_argument("--seed", type=int, default=20260430)
    ap.add_argument("--stage-name", default=None)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    rows = read_jsonl(Path(args.input))
    holdout_ids = {r.get("meta", {}).get("record_id") for r in read_jsonl(Path(args.holdout))}
    holdout_ids.discard(None)
    lesion_plan = parse_lesion_plan(args.lesion_plan, args.pos_per_lesion, args.neg_per_lesion)
    validated = load_validated(args.validated_clean)
    min_pos_confidence = parse_float_map(args.min_pos_confidence)

    out: list[dict[str, Any]] = []
    for lesion in LESIONS:
        pos_n, neg_n = lesion_plan[lesion]
        positives = []
        negatives = []
        for row in rows:
            meta = row.get("meta", {})
            if meta.get("record_id") in holdout_ids:
                continue
            if meta.get("task") != f"L3_{lesion}_single":
                continue
            if meta.get("present") is True:
                if passes_min_confidence(row, lesion, min_pos_confidence, validated):
                    positives.append(row)
            elif meta.get("present") is False:
                negatives.append(row)
        out.extend(take(positives, pos_n, rng, POS_SOURCE_PRIORITY))
        neg_priority = LESION_NEG_SOURCE_PRIORITY.get(lesion, NEG_SOURCE_PRIORITY) if args.lesion_specific_neg_priority else NEG_SOURCE_PRIORITY
        out.extend(take(negatives, neg_n, rng, neg_priority))

    rng.shuffle(out)
    stage_name = args.stage_name or Path(args.output).name.removesuffix("_sft.jsonl")
    for row in out:
        row.setdefault("meta", {})["stage_mix"] = stage_name
    write_jsonl(Path(args.output), out)
    stats = summarize(out)
    stats["output"] = args.output
    stats["lesion_plan"] = {k: {"positive": v[0], "negative": v[1]} for k, v in lesion_plan.items()}
    stats["min_pos_confidence"] = min_pos_confidence
    stats["lesion_specific_neg_priority"] = args.lesion_specific_neg_priority
    stats["holdout_excluded_records"] = len(holdout_ids)
    with Path(args.stats).open("w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
