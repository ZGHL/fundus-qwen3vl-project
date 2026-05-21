#!/usr/bin/env python3
"""Build v3 of the unified lesion-driven L4 DR grading CoT.

v3 changes vs v2:
  1. G4 NV-direct uses augmented image manifest (19 originals + 152 augmented
     copies = 171 unique pixel-level samples) instead of repeating 19 images
     to ~213 — fixes the over-fitting risk identified in v2 audit.
  2. Step2b (supervised_grade4_without_direct_nv_evidence_limited) is DROPPED:
     these samples create G3↔G4 ambiguity since same input can map to either.
  3. Step3c (supervised_grade3_evidence_limited) is hard-capped at 500
     (was 1539) to reduce G2/G3 boundary noise.
  4. G1 distribution rebalanced: template_only 1531→600, ma_only 227→400,
     evidence_limited 42→100 — reduces "G1 = blank image" prior.
  5. Total selected size ~7,800 (vs v2 10,000): smaller but cleaner.

Augmented G4 images use the original record's lesion_meta (NV verdict +
source + confidence) but emit verdict_line referencing the augmented image
path. The model sees pixel-different images all carrying the same NV-present
strong supervision signal.

Holdout selection always uses original validated_clean images (no augmented).
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any

import build_stage2_lite as stage2


BASE = Path("data/annotation")
VALIDATED = Path("data/fundus_validated/validated_clean.jsonl")
GRADE4_AUG = BASE / "fundus_l4_grade4_augmented_train_sft.jsonl"
NV_AUG_MANIFEST = Path("data/cropped/_aug_v3_nv/manifest.jsonl")
LESIONS = ["MA", "HE", "EX", "SE", "IRMA", "NV"]

STRONG_SOURCES = {"validated_retsam", "strong_mask_stage1_easy", "fgadr_lesion_only_sft_v3"}
WEAK_SOURCES = {"grade_rule_override"}

# v3 budgets per selected_step (None = use all available, do not cap)
# Set to 0 to drop step entirely.
V3_STEP_BUDGETS: dict[str, int | None] = {
    "Step1": None,        # use all NV-direct (171 originals + augmented)
    "Step2a": 1100,       # G4 template NV (capped slightly from v2's 1255)
    "Step2b": 0,          # ❌ DROPPED — G4 evidence_limited without NV
    "Step3a": 200,        # G3 IRMA boundary (oversample but pool ~87 unique)
    "Step3b": 700,        # G3 heavy NPDR
    "Step3c": 500,        # G3 evidence_limited (capped from 1539)
    "Step4":  1900,       # G2 main path
    "Step4b": 100,        # G2 evidence_limited
    "Step5a": 600,        # G1 template_only (capped from 1531)
    "Step5b": 400,        # G1 ma_only (boost from 227)
    "Step5c": 100,        # G1 evidence_limited (boost from 42 for diversity)
    "Step6":  1800,       # G0 (unchanged)
    "Step6b": 0,          # G0 with unexpected lesions — drop if any exist
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def load_build_sft_module():
    path = Path("scripts/fundus/build_fundus_sft.py")
    spec = importlib.util.spec_from_file_location("build_fundus_sft", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_v2_module():
    """Reuse v2 builder functions verbatim where unchanged."""
    path = Path("scripts/fundus/build_l4_unified_lesion_cot_v2.py")
    spec = importlib.util.spec_from_file_location("build_l4_v2", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ----------------------------- augmented expansion -----------------------------

def load_aug_manifest() -> dict[str, list[str]]:
    """Returns {original_image_path_under_data: [augmented_paths_under_data]}."""
    if not NV_AUG_MANIFEST.exists():
        print(f"[warn] {NV_AUG_MANIFEST} not found — Step1 will use only originals")
        return {}
    out: dict[str, list[str]] = {}
    for row in read_jsonl(NV_AUG_MANIFEST):
        original = row["original_image_path"]  # e.g. "FGADR/Seg-set/Original_Images/0091_3.png"
        # validated_clean records use "data/<original>" — normalize key to match
        key = f"data/{original}" if not original.startswith("data/") else original
        out[key] = list(row["augmented_paths"])  # already without "data/" prefix
    return out


META_KEYS_KEEP = {
    "record_id", "task", "split", "dr_grade", "source_file",
    "evidence_limited", "decision_rule", "selected_step",
    "burden", "proliferative_evidence", "boundary_evidence",
}


def patch_task_to_v3(item: dict[str, Any]) -> dict[str, Any]:
    """Rewrite task field to v3 + normalize meta schema (drop legacy keys like
    `grade` carried over from grade4_aug which break HF datasets schema cast).
    """
    if item is None:
        return item
    msgs = item.get("messages") or []
    if msgs and msgs[-1].get("role") == "assistant":
        c = msgs[-1]["content"]
        c = c.replace('"task":"L4_unified_lesion_cot_v2"', '"task":"L4_unified_lesion_cot_v3"')
        msgs[-1]["content"] = c
    meta = item.get("meta") or {}
    if meta.get("task") == "L4_unified_lesion_cot_v2":
        meta["task"] = "L4_unified_lesion_cot_v3"
    # Drop any meta keys outside our canonical schema (e.g. legacy `grade`
    # field from grade4_aug). All kept keys must be present too — fill missing.
    cleaned: dict[str, Any] = {}
    for k in META_KEYS_KEEP:
        cleaned[k] = meta.get(k)
    item["meta"] = cleaned
    return item


def expand_with_augmented(record: dict[str, Any], aug_paths: list[str]) -> list[dict[str, Any]]:
    """Generate synthetic records: original + 1 per augmented_path. Same lesion
    metadata, different image_path, suffixed record_id."""
    out = [record]  # original
    for idx, aug_path in enumerate(aug_paths, 1):
        synth = deepcopy(record)
        synth["image_path"] = aug_path if aug_path.startswith("data/") else f"data/{aug_path}"
        synth["record_id"] = f"{record['record_id']}__aug{idx}"
        synth["__augmented_from"] = record["record_id"]
        out.append(synth)
    return out


# ----------------------------- v3-specific selection ---------------------------

def is_nv_direct_g4(record: dict[str, Any]) -> bool:
    if record.get("grade") != 4:
        return False
    nv = record.get("lesions", {}).get("NV") or {}
    return nv.get("present") is True


def build_v3_train(records: list[dict[str, Any]], v2: Any, aug_map: dict[str, list[str]]) -> list[dict[str, Any]]:
    """Walk validated_clean train records, expand NV-direct G4 with augmentation."""
    out: list[dict[str, Any]] = []
    aug_added = 0
    for r in records:
        if is_nv_direct_g4(r):
            key = r["image_path"]
            aug_paths = aug_map.get(key, [])
            expanded = expand_with_augmented(r, aug_paths)
            for synth_r in expanded:
                item = v2.build_from_validated(synth_r, "train")
                if item is not None:
                    out.append(patch_task_to_v3(item))
                    if "__augmented_from" in synth_r:
                        aug_added += 1
        else:
            item = v2.build_from_validated(r, "train")
            if item is not None:
                out.append(patch_task_to_v3(item))
    print(f"[expand] augmented G4 NV-direct samples added: {aug_added}")
    return out


def build_v3_holdout(records: list[dict[str, Any]], v2: Any) -> list[dict[str, Any]]:
    """Holdout uses ONLY original images (no augmented copies)."""
    out: list[dict[str, Any]] = []
    for r in records:
        item = v2.build_from_validated(r, "holdout")
        if item is not None:
            out.append(patch_task_to_v3(item))
    return out


def split_rows(v2: Any, aug_map: dict[str, list[str]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    mod = load_build_sft_module()
    train_records: list[dict[str, Any]] = []
    holdout_records: list[dict[str, Any]] = []
    for record in read_jsonl(VALIDATED):
        split, _, _ = mod.split_of(record, 10)
        if split == "train":
            train_records.append(record)
        else:
            holdout_records.append(record)

    train = build_v3_train(train_records, v2, aug_map)
    holdout = build_v3_holdout(holdout_records, v2)

    aug_skipped_step1 = 0
    if GRADE4_AUG.exists():
        for row in stage2.stable_shuffle(read_jsonl(GRADE4_AUG), "l4_unified_v3_grade4_aug"):
            item = v2.build_from_grade4_aug(row, "train")
            if item is None:
                continue
            # v3: skip Step1 (NV-direct G4) entries from grade4_aug. Those
            # images are 13/17 already in validated_clean and contribute zero
            # pixel-level novelty; keeping them would re-introduce the
            # over-sampling we removed via augmentation.
            if item["meta"].get("selected_step") == "Step1":
                aug_skipped_step1 += 1
                continue
            train.append(patch_task_to_v3(item))
    print(f"[grade4_aug] Step1 entries skipped (covered by aug manifest): {aug_skipped_step1}")
    return train, holdout


def filter_by_step_budget(rows: list[dict[str, Any]], budgets: dict[str, int | None]) -> list[dict[str, Any]]:
    """Apply v3 per-step budgets. None = keep all; 0 = drop; int = cap."""
    by_step: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        step = row["meta"].get("selected_step", "Unknown")
        by_step[step].append(row)

    kept: list[dict[str, Any]] = []
    summary: dict[str, dict[str, int]] = {}
    for step, items in by_step.items():
        budget = budgets.get(step, None)
        if budget == 0:
            summary[step] = {"available": len(items), "kept": 0, "action": "DROPPED"}
            continue
        # stable shuffle for deterministic subsampling
        items = stage2.stable_shuffle(items, f"l4_v3_step_{step}")
        if budget is None or budget >= len(items):
            kept.extend(items)
            summary[step] = {"available": len(items), "kept": len(items), "action": "use_all"}
        else:
            kept.extend(items[:budget])
            summary[step] = {"available": len(items), "kept": budget, "action": f"capped_to_{budget}"}
    print("[budget] step distribution after filtering:")
    for step in sorted(summary):
        s = summary[step]
        print(f"  {step:8s}  available={s['available']:5d}  kept={s['kept']:5d}  ({s['action']})")
    return kept


def balance_holdout(rows: list[dict[str, Any]], n: int) -> list[dict[str, Any]]:
    by_grade: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_grade[int(row["meta"]["dr_grade"])].append(row)
    base = n // 5
    rem = n % 5
    out = []
    for grade in range(5):
        out.extend(stage2.take(stage2.stable_shuffle(by_grade[grade], f"l4_v3_holdout_g{grade}"), base + (1 if grade < rem else 0)))
    return stage2.stable_shuffle(out, "l4_v3_holdout_final")


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grades = Counter()
    rules = Counter()
    steps = Counter()
    burdens = Counter()
    evidence_limited = Counter()
    tier_counts = Counter()
    source_files = Counter()
    aug_count = 0
    missing = 0
    for row in rows:
        meta = row.get("meta", {})
        grades[str(meta.get("dr_grade"))] += 1
        rules[meta.get("decision_rule", "unknown")] += 1
        steps[meta.get("selected_step", "unknown")] += 1
        burdens[meta.get("burden", "unknown")] += 1
        evidence_limited[str(bool(meta.get("evidence_limited")))] += 1
        source_files[meta.get("source_file", "unknown")] += 1
        rid = meta.get("record_id", "")
        if "__aug" in rid:
            aug_count += 1
        obj = stage2.extract_json(row["messages"][-1]["content"])
        for key in LESIONS:
            tier_counts[(key, obj.get(key, "unknown"))] += 1
        for image in row.get("images", []):
            if not Path("data", image).exists():
                missing += 1
    return {
        "n": len(rows),
        "n_augmented_records": aug_count,
        "grades": dict(sorted(grades.items())),
        "decision_rules": dict(rules),
        "selected_step": dict(steps),
        "burden": dict(burdens),
        "evidence_limited": dict(evidence_limited),
        "lesion_states": {str(k): v for k, v in sorted(tier_counts.items(), key=lambda x: str(x[0]))},
        "source_files": dict(source_files),
        "missing_images": missing,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--holdout-n", type=int, default=150)
    parser.add_argument("--train-output", default="data/annotation/fundus_l4_unified_lesion_cot_v3_sft.jsonl")
    parser.add_argument("--holdout-output", default="data/annotation/fundus_l4_unified_lesion_cot_v3_holdout_sft.jsonl")
    parser.add_argument("--budgets", default=None,
                        help="Optional JSON file overriding V3_STEP_BUDGETS")
    args = parser.parse_args()

    v2 = load_v2_module()
    aug_map = load_aug_manifest()
    print(f"[aug_manifest] {len(aug_map)} original images mapped to augmented copies")

    train_raw, holdout_raw = split_rows(v2, aug_map)
    print(f"[raw] train={len(train_raw)}  holdout={len(holdout_raw)}")

    budgets = dict(V3_STEP_BUDGETS)
    if args.budgets:
        budgets.update(json.load(open(args.budgets)))
    train = filter_by_step_budget(train_raw, budgets)
    holdout = balance_holdout(holdout_raw, args.holdout_n)

    write_jsonl(Path(args.train_output), train)
    write_jsonl(Path(args.holdout_output), holdout)

    stats = {
        "train_output": args.train_output,
        "holdout_output": args.holdout_output,
        "v3_budgets": {k: v for k, v in budgets.items()},
        "raw": {"train": summarize(train_raw), "holdout": summarize(holdout_raw)},
        "selected": {"train": summarize(train), "holdout": summarize(holdout)},
    }
    stats_path = BASE / "fundus_l4_unified_lesion_cot_v3_stats.json"
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
