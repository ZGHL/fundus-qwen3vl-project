#!/usr/bin/env python3
"""Build L3 v6 standardized single-lesion CoT SFT data.

Compared with v5, v6 keeps the explicit 4-section English CoT but fixes the
data construction rules:
  - use only records marked usable_for.L3
  - split by image_id, so different lesion prompts from the same image cannot
    appear in both train and validation
  - train on direct present/absent supervision only; template_only/unknown are
    counted in stats but not used as positive visual labels
  - balance dense lesions, and keep sparse IRMA/NV positives with up to 4x
    explicit negatives
  - write train/val JSONL plus a stats JSON for audit before training
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "fundus"))
from build_fundus_sft import hbucket, read_jsonl, sft, write_jsonl  # noqa: E402

VALIDATED = Path("data/fundus_validated/validated_clean.jsonl")
OUT_DIR = Path("data/annotation_v4")
SPARSE_AUG_MANIFEST = Path("data/cropped/_aug_v4_sparse/manifest.jsonl")
QUAD_INDEX = Path("data/fundus_validated/quadrant_index.jsonl")

LESIONS = ("MA", "HE", "EX", "SE", "IRMA", "NV")
SPARSE_AUG_LESIONS = {"IRMA", "NV"}
VAL_PCT = 20

PRESENT = "present"
ABSENT = "absent"
TEMPLATE_ONLY = "template_only"
UNKNOWN = "unknown"

STRONG_SOURCES = {"validated_retsam", "strong_mask_stage1_easy", "fgadr_lesion_only_sft_v3"}
SOURCE_TAG = {
    "validated_retsam": "retsam_validated",
    "strong_mask_stage1_easy": "strong_mask",
    "fgadr_lesion_only_sft_v3": "strong_mask",
    "retsam_negative": "retsam_negative",
    "cleaning_rule": "cleaning_rule",
    "grade_rule": "grade_rule",
    "grade_rule_override": "grade_rule",
}

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

Q_NAMES = {"ST": "superior-temporal", "SN": "superior-nasal", "IT": "inferior-temporal", "IN": "inferior-nasal"}

SYSTEM_PROMPT_TEMPLATE = (
    "You are a fundus image analyst. Inspect ONLY for {lesion} ({lesion_full}). "
    "This is a single-lesion perception task: do NOT output a final DR grade and "
    "do NOT combine other lesions. First describe visible morphology and location, "
    "then judge whether the target lesion is present."
)
USER_PROMPT_TEMPLATE = (
    "Examine this fundus image for {lesion_full} ({lesion}). "
    "Output exactly four sections: [Lesion Existence and Evidence Judgment], "
    "[Basic Morphological and Location Features], [Decision Notes for This Single-Lesion Task], "
    "and [Structured Output]."
)


def assign_splits(records: list[dict[str, Any]], val_pct: int) -> dict[str, str]:
    eval_iids = set()
    for r in records:
        if r.get("dataset") == "idrid" and r.get("split") == "test":
            eval_iids.add(r["image_id"])
        if r.get("dataset") == "ddr_seg" and r.get("split") in {"valid", "test"}:
            eval_iids.add(r["image_id"])
    iid_split = {}
    for r in records:
        iid = r["image_id"]
        if iid in eval_iids:
            iid_split[iid] = "eval"
        elif iid not in iid_split:
            iid_split[iid] = "val" if hbucket(iid) < val_pct else "train"
    return iid_split


def state_for_lesion(record: dict[str, Any], lesion_key: str) -> tuple[str, dict[str, Any]] | None:
    if not record.get("usable_for", {}).get("L3"):
        return None
    lesion = record.get("lesions", {}).get(lesion_key)
    if not isinstance(lesion, dict):
        return None
    present = lesion.get("present")
    if present is True:
        return PRESENT, lesion
    if present is False:
        return ABSENT, lesion
    if present in {"template_only", "possible_by_grade_template"}:
        return TEMPLATE_ONLY, lesion
    return UNKNOWN, lesion


def source_tag(source: str | None) -> str:
    return SOURCE_TAG.get(source or "unknown", source or "unknown")


def source_strength(state: str, lesion: dict[str, Any]) -> str:
    if state == PRESENT:
        return "strong" if lesion.get("source") in STRONG_SOURCES else "weak"
    if state == ABSENT:
        return "absent"
    return state


def load_quadrant_index() -> dict[tuple[str, str], dict[str, Any]]:
    out = {}
    if not QUAD_INDEX.exists():
        return out
    for row in read_jsonl(QUAD_INDEX):
        out[(row["dataset"], row["image_id"])] = row.get("quadrants", {})
    return out


def quadrants_to_location(quad: dict[str, Any] | None) -> str | None:
    if not quad or quad.get("total", 0) == 0:
        return None
    total = int(quad.get("total", 0))
    q = {k: int(quad.get(k, 0)) for k in ("ST", "SN", "IT", "IN")}
    non_zero = sum(1 for v in q.values() if v > 0)
    if non_zero == 4 and min(q.values()) >= max(1, total // 6):
        return "throughout all four macula-centered quadrants"
    if non_zero == 1:
        only_q = max(q, key=q.get)
        return f"in the {Q_NAMES[only_q]} quadrant only"
    top = [k for k, v in sorted(q.items(), key=lambda x: -x[1]) if v >= max(1, total // 4)]
    if len(top) == 1:
        return f"predominantly in the {Q_NAMES[top[0]]} quadrant"
    if len(top) == 2:
        return f"distributed in the {Q_NAMES[top[0]]} and {Q_NAMES[top[1]]} quadrants"
    if len(top) >= 3:
        return "distributed across multiple quadrants"
    return None


def lesion_location(record: dict[str, Any], lesion_key: str, lesion: dict[str, Any], quad_idx) -> str:
    if lesion_key in {"HE", "EX", "SE"}:
        quad = quad_idx.get((record.get("dataset"), record.get("image_id")), {}).get(lesion_key)
        loc = quadrants_to_location(quad)
        if loc:
            return loc
    band = lesion.get("location_band")
    if isinstance(band, str) and band:
        return {"黄斑区": "macular region", "后极部": "posterior pole", "中周部": "midperiphery", "周边部": "peripheral retina"}.get(band, band)
    return DEFAULT_LOCATION[lesion_key]


def morphology_sentence(lesion_key: str, state: str, lesion: dict[str, Any], location: str) -> str:
    visual = LESION_VISUAL[lesion_key]
    if state == PRESENT:
        cb = lesion.get("count_bucket") or "unknown"
        ab = lesion.get("area_bucket") or "unknown"
        return f"The target finding is described as {visual}. In this sample it is annotated as count={cb}, area={ab}, located {location}."
    if state == ABSENT:
        if lesion.get("raw_present") is True:
            reason = lesion.get("suppressed_reason", "low-confidence_or_tiny_signal")
            return f"No reliable target-pattern evidence is retained after cleaning; the raw {lesion_key} signal was suppressed because {reason}."
        return f"No reliable target-pattern evidence is identified for {lesion_key}."
    if state == TEMPLATE_ONLY:
        return f"The grade label may suggest {lesion_key}, but there is no direct lesion mask or validated visual evidence for this target."
    return f"The available annotation does not provide reliable visual evidence for {visual}."


def existence_sentence(lesion_key: str, state: str, lesion: dict[str, Any]) -> str:
    full = LESION_FULL[lesion_key]
    src = source_tag(lesion.get("source"))
    strength = source_strength(state, lesion)
    if state == PRESENT:
        return f"{full} is present with {strength} direct evidence from {src}."
    if state == ABSENT:
        
        if lesion.get("source") in {"grade_rule", "grade_rule_override"}:
            return f"{full} is absent according to an inferred weak negative label from {src}."
        return f"{full} is absent according to direct negative evidence from {src}."
    if state == TEMPLATE_ONLY:
        return f"{full} is not visually confirmed; the evidence is template-only from {src}."
    return f"{full} status is unknown because reliable direct annotation is unavailable."


def decision_note(lesion_key: str, state: str, lesion: dict[str, Any]) -> str:
    src = lesion.get("source")
    if state == ABSENT and src in {"grade_rule", "grade_rule_override"}:
        evidence = "This is an inferred weak negative label from the image-level grade, not a lesion-mask confirmation."
    elif state == ABSENT:
        evidence = "This is treated as a target-lesion negative example for learning the absent boundary."
    elif state == PRESENT:
        evidence = "This is a target-lesion positive example for learning visible morphology."
    else:
        evidence = "This target lesion is not used as a direct visual positive label."

    notes = {
        "MA": "Focus on tiny round red dots; do not confuse them with vessel crossings, isolated noise, or larger hemorrhages.",
        "HE": "Focus on dark red hemorrhagic lesions; do not call hard exudates or normal vessels hemorrhage.",
        "EX": "Focus on sharply bordered yellow-white lipid deposits; distinguish them from fluffy cotton-wool spots.",
        "SE": "Focus on fluffy gray-white cotton-wool patches with soft borders; distinguish them from hard exudates and glare.",
        "IRMA": "Focus on abnormal intraretinal vascular channels; distinguish IRMA from NV and from normal vessel branching.",
        "NV": "Focus on abnormal new vessels on the disc or elsewhere; distinguish NV from IRMA and ordinary vessels.",
    }
    return evidence + " " + notes[lesion_key] + " No final DR grade is assigned in this L3 task."


def structured_json(lesion_key: str, state: str, lesion: dict[str, Any], location: str) -> str:
    present_value = True if state == PRESENT else False if state == ABSENT else state
    payload = {
        "task": f"L3_{lesion_key}",
        "lesion": lesion_key,
        "present": present_value,
        "evidence_state": state,
        "strength": source_strength(state, lesion),
        "count": lesion.get("count_bucket"),
        "area": lesion.get("area_bucket"),
        "location": location if state == PRESENT else None,
        "source": source_tag(lesion.get("source")),
    }
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def make_assistant(lesion_key: str, state: str, lesion: dict[str, Any], record: dict[str, Any], quad_idx) -> str:
    loc = lesion_location(record, lesion_key, lesion, quad_idx)
    return (
        "[Lesion Existence and Evidence Judgment]\n" + existence_sentence(lesion_key, state, lesion) + "\n\n"
        "[Basic Morphological and Location Features]\n" + morphology_sentence(lesion_key, state, lesion, loc) + "\n\n"
        "[Decision Notes for This Single-Lesion Task]\n" + decision_note(lesion_key, state, lesion) + "\n\n"
        "[Structured Output]\n" + structured_json(lesion_key, state, lesion, loc)
    )


def image_path_for_sft(record: dict[str, Any]) -> str:
    return record.get("cropped_path") or record.get("image_path")


def make_sft_item(lesion_key: str, state: str, lesion: dict[str, Any], record: dict[str, Any], split: str, quad_idx) -> dict[str, Any]:
    loc = lesion_location(record, lesion_key, lesion, quad_idx) if state == PRESENT else None
    meta = {
        "record_id": record.get("record_id"),
        "image_id": record["image_id"],
        "dataset": record.get("dataset"),
        "grade": record.get("grade"),
        "split": split,
        "task": f"L3_{lesion_key}",
        "lesion": lesion_key,
        "present_state": state,
        "source": lesion.get("source", "unknown"),
        "source_tag": source_tag(lesion.get("source")),
        "count_bucket": lesion.get("count_bucket"),
        "area_bucket": lesion.get("area_bucket"),
        "location": loc,
    }
    system = SYSTEM_PROMPT_TEMPLATE.format(lesion=lesion_key, lesion_full=LESION_FULL[lesion_key])
    user = USER_PROMPT_TEMPLATE.format(lesion=lesion_key, lesion_full=LESION_FULL[lesion_key])
    return sft(system, user, make_assistant(lesion_key, state, lesion, record, quad_idx), image_path_for_sft(record), meta)


def load_sparse_aug_manifest() -> dict[str, dict[str, list[str]]]:
    out = {}
    if not SPARSE_AUG_MANIFEST.exists():
        return out
    for row in read_jsonl(SPARSE_AUG_MANIFEST):
        out.setdefault(row["lesion"], {})[row["record_id"]] = list(row.get("augmented_paths", []))
    return out


def expand_with_aug(state: str, lesion: dict[str, Any], record: dict[str, Any], aug_paths: list[str]):
    out = [(state, lesion, record)]
    for idx, aug_path in enumerate(aug_paths, 1):
        synth = dict(record)
        synth["record_id"] = f"{record['record_id']}__aug{idx}"
        synth["cropped_path"] = aug_path
        synth["image_path"] = aug_path
        out.append((state, lesion, synth))
    return out


def build_pools(records: list[dict[str, Any]], aug_map=None):
    aug_map = aug_map or {}
    pools = {k: defaultdict(list) for k in LESIONS}
    raw_counts = {k: Counter() for k in LESIONS}
    source_counts = {k: Counter() for k in LESIONS}
    for record in records:
        for lesion_key in LESIONS:
            ev = state_for_lesion(record, lesion_key)
            if ev is None:
                raw_counts[lesion_key]["missing_or_unusable"] += 1
                continue
            state, lesion = ev
            raw_counts[lesion_key][state] += 1
            source_counts[lesion_key][lesion.get("source", "unknown")] += 1
            if state not in {PRESENT, ABSENT}:
                continue
            if lesion_key in SPARSE_AUG_LESIONS and state == PRESENT:
                aug_paths = aug_map.get(lesion_key, {}).get(record.get("record_id"), [])
                pools[lesion_key][state].extend(expand_with_aug(state, lesion, record, aug_paths))
            else:
                pools[lesion_key][state].append((state, lesion, record))
    return pools, raw_counts, source_counts


def is_strong_source(item) -> bool:
    _state, lesion, _record = item
    return lesion.get("source") in {"strong_mask_stage1_easy", "fgadr_lesion_only_sft_v3"}


def priority_sample(items: list, n: int, rng: random.Random) -> list:
    strong = [x for x in items if is_strong_source(x)]
    other = [x for x in items if not is_strong_source(x)]
    rng.shuffle(strong)
    rng.shuffle(other)
    return (strong + other)[:n]


def sample_train(pool: dict[str, list], lesion_key: str, seed: int):
    rng = random.Random(seed)
    pos = list(pool.get(PRESENT, []))
    neg = list(pool.get(ABSENT, []))
    if lesion_key in {"IRMA", "NV"}:
        rng.shuffle(pos)
        # Sparse vascular lesions need more negatives; prioritize strong negatives, then use weak grade-rule negatives only if needed.
        neg_n = min(len(neg), len(pos) * 4)
        return pos + priority_sample(neg, neg_n, rng)
    n = min(len(pos), len(neg))
    # For dense lesions we keep 1:1, but prefer strong labels first and fill the remainder with RetSAM labels.
    return priority_sample(pos, n, rng) + priority_sample(neg, n, rng)


def build(args):
    records = list(read_jsonl(args.validated))
    iid_split = assign_splits(records, args.val_pct)
    train_records = [r for r in records if iid_split.get(r["image_id"]) == "train"]
    val_records = [r for r in records if iid_split.get(r["image_id"]) == "val"]
    eval_records = [r for r in records if iid_split.get(r["image_id"]) == "eval"]
    quad_idx = load_quadrant_index()
    aug_map = load_sparse_aug_manifest()
    train_pools, train_raw, train_sources = build_pools(train_records, aug_map=aug_map)
    val_pools, val_raw, val_sources = build_pools(val_records)
    train_items = []
    val_items = []
    stats = {
        "version": "fundus_l3_v6",
        "design": ["single_lesion_4_section_english_cot", "usable_for_L3_only", "image_id_level_split", "direct_present_absent_training_only", "dense_lesions_balanced_pos_neg_strong_first", "sparse_irma_nv_pos_plus_up_to_4x_neg_strong_first", "template_only_and_unknown_counted_not_trained_as_visual_positive"],
        "input": str(args.validated),
        "val_pct": args.val_pct,
        "seed": args.seed,
        "records": {"all": len(records), "train_images": len(train_records), "val_images": len(val_records), "eval_images_excluded": len(eval_records), "usable_L3_all": sum(bool(r.get("usable_for", {}).get("L3")) for r in records)},
        "per_lesion": {},
    }
    for idx, lesion_key in enumerate(LESIONS):
        sampled_train = sample_train(train_pools[lesion_key], lesion_key, args.seed + idx)
        for state, lesion, record in sampled_train:
            train_items.append(make_sft_item(lesion_key, state, lesion, record, "train", quad_idx))
        for state in (PRESENT, ABSENT):
            for ev_state, lesion, record in val_pools[lesion_key].get(state, []):
                val_items.append(make_sft_item(lesion_key, ev_state, lesion, record, "val", quad_idx))
        train_sample_counts = Counter(item[0] for item in sampled_train)
        stats["per_lesion"][lesion_key] = {
            "train_raw_states": dict(train_raw[lesion_key]),
            "val_raw_states": dict(val_raw[lesion_key]),
            "train_sources": dict(train_sources[lesion_key]),
            "val_sources": dict(val_sources[lesion_key]),
            "train_available_present": len(train_pools[lesion_key].get(PRESENT, [])),
            "train_available_absent": len(train_pools[lesion_key].get(ABSENT, [])),
            "train_sampled_present": train_sample_counts.get(PRESENT, 0),
            "train_sampled_absent": train_sample_counts.get(ABSENT, 0),
            "val_present": len(val_pools[lesion_key].get(PRESENT, [])),
            "val_absent": len(val_pools[lesion_key].get(ABSENT, [])),
        }
    rng = random.Random(args.seed)
    rng.shuffle(train_items)
    rng.shuffle(val_items)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    train_path = args.out_dir / "fundus_l3_v6_train_sft.jsonl"
    val_path = args.out_dir / "fundus_l3_v6_val_sft.jsonl"
    stats_path = args.out_dir / "fundus_l3_v6_stats.json"
    stats["train_total"] = len(train_items)
    stats["val_total"] = len(val_items)
    stats["train_task_counts"] = dict(Counter(item["meta"]["task"] for item in train_items))
    stats["val_task_counts"] = dict(Counter(item["meta"]["task"] for item in val_items))
    if not args.dry_run:
        write_jsonl(train_path, train_items)
        write_jsonl(val_path, val_items)
        stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print("=== L3 v6 build summary ===")
    print(f"records: all={len(records)} usable_L3={stats['records']['usable_L3_all']}")
    print(f"split images: train={len(train_records)} val={len(val_records)} eval_excluded={len(eval_records)}")
    print(f"items: train={len(train_items)} val={len(val_items)}")
    for lesion_key, s in stats["per_lesion"].items():
        print(f"  {lesion_key:<5} train={s['train_sampled_present']:>4}+{s['train_sampled_absent']:>4} val={s['val_present']:>4}+{s['val_absent']:>4} raw_train={s['train_raw_states']}")
    print(f"train: {train_path}")
    print(f"val  : {val_path}")
    print(f"stats: {stats_path}")
    return stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--validated", type=Path, default=VALIDATED)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--val-pct", type=int, default=VAL_PCT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    build(args)


if __name__ == "__main__":
    main()
