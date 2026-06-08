#!/usr/bin/env python3
"""Build evidence-tiered English Stage1 single-lesion CoT datasets.

This builder intentionally keeps training labels and model-visible text
separate. Evidence provenance is stored in metadata, while prompts expose only
the target lesion definition and the assistant describes only visible evidence.

Outputs:
  - evidence-tiered training set
  - internal validation set
  - DDR valid/test Main-4 gold sets, plus one file per lesion
  - weak-negative challenge set
  - FGADR IRMA/NV locked sets
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np


LESIONS = ("MA", "HE", "EX", "SE", "IRMA", "NV")
MAIN4 = ("MA", "HE", "EX", "SE")
RARE2 = ("IRMA", "NV")
VALIDATED = Path("data/fundus_validated/validated_clean.jsonl")
OUT_DIR = Path("data/annotation_v4")
DDR_ROOT = Path("data/DDR-dataset/lesion_segmentation")
FGADR_ROOT = Path("data/FGADR/Seg-set")
SPARSE_AUG = Path("data/cropped/_aug_v4_sparse/manifest.jsonl")

LESION_INFO = {
    "MA": {
        "name": "microaneurysm",
        "visual": "tiny, round, relatively well-defined red dot-like abnormalities",
        "exclude": "normal vessel crossings, imaging noise, artifacts, and larger dot or blot hemorrhages",
        "positive_confounder": "They are spatially separate from vessel crossings and are smaller and more sharply punctate than typical dot or blot hemorrhages.",
        "negative_confounder": "Any visible dark-red structures are better explained by normal vessels, vessel crossings, larger hemorrhages, or image noise.",
    },
    "HE": {
        "name": "retinal hemorrhage",
        "visual": "dark-red dot, blot, flame-shaped, or irregular hemorrhagic abnormalities",
        "exclude": "normal retinal vessels, vessel crossings, microaneurysms, pigmentation, and dark imaging artifacts",
        "positive_confounder": "The findings are not explained by normal vessels, tiny isolated microaneurysms, pigmentation, or dark image artifacts.",
        "negative_confounder": "Visible dark structures are better explained by normal vessels, vessel crossings, pigmentation, or image artifacts.",
    },
    "EX": {
        "name": "hard exudate",
        "visual": "bright yellow-white deposits with relatively sharp and well-defined borders",
        "exclude": "the optic disc, imaging reflections, glare, soft exudates, and other bright artifacts",
        "positive_confounder": "The sharp deposit-like findings are distinct from the optic disc, glare, reflections, and fluffy soft exudates.",
        "negative_confounder": "Visible bright regions are better explained by the optic disc, glare, reflections, soft exudates, or other artifacts.",
    },
    "SE": {
        "name": "soft exudate or cotton-wool spot",
        "visual": "gray-white or pale fluffy lesions with soft, indistinct borders",
        "exclude": "hard exudates, optic-disc margins, glare, overexposed regions, and other bright artifacts",
        "positive_confounder": "The fluffy, indistinct findings are distinct from sharply bordered hard exudates, the optic disc, glare, and overexposure.",
        "negative_confounder": "Visible bright regions are better explained by hard exudates, the optic disc, glare, overexposure, or other artifacts.",
    },
    "IRMA": {
        "name": "intraretinal microvascular abnormality",
        "visual": "irregular, dilated, or tortuous intraretinal vascular channels, often near ischemic retinal regions",
        "exclude": "normal vessel branching, overlapping vessels, neovascularization, and vascular imaging artifacts",
        "positive_confounder": "The abnormal intraretinal vascular channels are not explained by normal branching, overlapping vessels, or neovascularization.",
        "negative_confounder": "Visible vascular patterns are better explained by normal branching, overlapping vessels, or vascular artifacts.",
    },
    "NV": {
        "name": "neovascularization",
        "visual": "an abnormal fine new-vessel network on the optic disc or elsewhere in the retina, often crossing normal vascular planes",
        "exclude": "normal vessel branching, peripapillary vessels, intraretinal microvascular abnormalities, and vascular artifacts",
        "positive_confounder": "The fine abnormal vessel network is not explained by normal branching, peripapillary vessels, IRMA, or vascular artifacts.",
        "negative_confounder": "Visible vascular patterns are better explained by normal branching, peripapillary vessels, IRMA, or vascular artifacts.",
    },
}

# Exposure targets, not unique-image targets.
TRAIN_TARGETS = {
    "HE": {"present": 2000, "absent": 1200},
    "EX": {"present": 2000, "absent": 1200},
    "MA": {"present": 1200, "absent": 450},
    "SE": {"present": 1600, "absent": 1600},
    "IRMA": {"present": 400, "absent": 500},
    "NV": {"present": 140, "absent": 360},
}

# Desired source mix. Missing high-tier capacity is filled from the next tier.
TIER_MIX = {
    "HE": {"present": {"S0": 0.30, "S1": 0.30, "S2": 0.40}, "absent": {"S0": 0.30, "S1": 0.15, "S2": 0.35, "S3": 0.15, "S4": 0.05}},
    "EX": {"present": {"S0": 0.30, "S1": 0.30, "S2": 0.40}, "absent": {"S0": 0.35, "S1": 0.15, "S2": 0.30, "S3": 0.15, "S4": 0.05}},
    "MA": {"present": {"S0": 0.45, "S1": 0.55}, "absent": {"S0": 0.45, "S1": 0.20, "S4": 0.35}},
    "SE": {"present": {"S0": 0.20, "S1": 0.20, "S2": 0.60}, "absent": {"S0": 0.40, "S1": 0.25, "S2": 0.20, "S3": 0.10, "S4": 0.05}},
    "IRMA": {"present": {"S0": 1.0}, "absent": {"S1": 1.0}},
    "NV": {"present": {"S0": 1.0}, "absent": {"S1": 1.0}},
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def hbucket(text: str, mod: int = 100) -> int:
    return int(hashlib.sha1(text.encode("utf-8")).hexdigest()[:12], 16) % mod


def image_key(record: dict[str, Any]) -> str:
    return f"{record.get('dataset')}::{record.get('split')}::{record.get('image_id')}"


def sft_image_path(record: dict[str, Any]) -> str:
    path = str(record.get("image_path") or record.get("cropped_path") or record.get("src_path"))
    return path[5:] if path.startswith("data/") else path


def mask_label(mask_path: Path) -> dict[str, Any]:
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise RuntimeError(f"Cannot read mask: {mask_path}")
    binary = (mask > 0).astype(np.uint8)
    area = int(binary.sum())
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    components = max(0, n - 1)
    if components <= 1:
        distribution = "isolated"
    elif components <= 5:
        distribution = "multifocal"
    else:
        occupied = set()
        h, w = binary.shape
        for x, y in centroids[1:]:
            occupied.add((int(y >= h / 2), int(x >= w / 2)))
        distribution = "diffuse" if len(occupied) == 4 and components >= 12 else "scattered"
    return {
        "present": bool(area),
        "component_count": components,
        "mask_area": area,
        "mask_fraction": float(area / binary.size),
        "distribution": distribution if area else None,
    }


def ddr_mask_index(root: Path) -> dict[tuple[str, str, str], Path]:
    out: dict[tuple[str, str, str], Path] = {}
    for split in ("train", "valid", "test"):
        label_root = root / split / ("segmentation label" if split == "valid" else "label")
        for lesion in MAIN4:
            for path in (label_root / lesion).glob("*"):
                if path.is_file():
                    out[(split, path.stem, lesion)] = path
    return out


def fgadr_rare_mask_index(root: Path) -> dict[tuple[str, str], Path]:
    out: dict[tuple[str, str], Path] = {}
    for lesion, dirname in (("IRMA", "IRMA_Masks"), ("NV", "Neovascularization_Masks")):
        for path in (root / dirname).glob("*"):
            if path.is_file() and not path.name.startswith("."):
                out[(path.stem, lesion)] = path
    return out


def bucket_count(n: int | None) -> str | None:
    if not isinstance(n, int) or n <= 0:
        return None
    if n == 1:
        return "single"
    if n <= 5:
        return "few"
    return "many"


def area_thresholds(mask_facts: dict[tuple[str, str, str], dict[str, Any]]) -> dict[str, tuple[float, float]]:
    out = {}
    for lesion in MAIN4:
        vals = sorted(v["mask_fraction"] for k, v in mask_facts.items() if k[2] == lesion and v["present"])
        out[lesion] = (
            float(np.quantile(vals, 1 / 3)) if vals else 0.0,
            float(np.quantile(vals, 2 / 3)) if vals else 0.0,
        )
    return out


def bucket_area(value: float | None, thresholds: tuple[float, float] | None = None) -> str | None:
    if not isinstance(value, (int, float)) or value <= 0:
        return None
    if thresholds:
        lo, hi = thresholds
        return "small" if value <= lo else "medium" if value <= hi else "large"
    # RetSAM already supplies its own area bucket; this is only a fallback.
    return "small" if value < 300 else "medium" if value < 1500 else "large"


def evidence_from_record(
    record: dict[str, Any],
    lesion: str,
    mask_facts: dict[tuple[str, str, str], dict[str, Any]],
    rare_mask_facts: dict[tuple[str, str], dict[str, Any]],
    thresholds: dict[str, tuple[float, float]],
) -> dict[str, Any] | None:
    if record.get("dataset") == "ddr_seg" and lesion in MAIN4:
        key = (str(record.get("split")), str(record.get("image_id")), lesion)
        fact = mask_facts.get(key)
        if fact:
            return {
                "state": "present" if fact["present"] else "absent",
                "tier": "S0",
                "source": "ddr_mask",
                "label_origin": "direct_mask",
                "count_bucket": bucket_count(fact["component_count"]),
                "area_bucket": bucket_area(fact["mask_fraction"], thresholds[lesion]),
                "distribution": fact["distribution"],
            }

    if record.get("dataset") == "fgadr_seg" and lesion in RARE2:
        fact = rare_mask_facts.get((str(record.get("image_id")), lesion))
        if fact and fact["present"]:
            return {
                "state": "present",
                "tier": "S0",
                "source": "fgadr_mask",
                "label_origin": "direct_mask",
                "count_bucket": bucket_count(fact["component_count"]),
                "area_bucket": bucket_area(fact["mask_fraction"]),
                "distribution": fact["distribution"],
            }

    data = (record.get("lesions") or {}).get(lesion)
    if not isinstance(data, dict) or data.get("present") not in {True, False}:
        return None
    source = str(data.get("source") or "unknown")
    present = bool(data["present"])
    if source == "fgadr_lesion_only_sft_v3":
        tier, origin = "S1", "explicit_lesion_annotation"
    elif source == "strong_mask_stage1_easy" and lesion in MAIN4:
        tier, origin = "S0", "direct_mask"
    elif source == "validated_retsam":
        tier, origin = "S2", "validated_retsam"
    elif source == "retsam_negative":
        tier, origin = "S2", "retsam_negative"
    elif source == "cleaning_rule" and not present:
        tier, origin = "S3", "cleaning_rule_negative"
    elif source in {"grade_rule", "grade_rule_override"} and not present:
        if lesion == "MA" and record.get("grade") == 0:
            tier, origin = "S4", "grade0_weak_negative"
        elif lesion in {"HE", "EX", "SE"} and record.get("grade") == 0:
            tier, origin = "S4", "grade0_weak_negative"
        else:
            return None
    else:
        return None
    if present and tier in {"S3", "S4"}:
        return None
    if lesion in RARE2 and tier != "S1":
        return None
    if lesion == "MA" and present and tier not in {"S0", "S1"}:
        return None
    location = data.get("location")
    distribution = data.get("extent") if tier in {"S0", "S1"} else None
    return {
        "state": "present" if present else "absent",
        "tier": tier,
        "source": source,
        "label_origin": origin,
        "count_bucket": data.get("count_bucket") if present else None,
        "area_bucket": data.get("area_bucket") if present else None,
        "distribution": str(distribution).lower() if present and distribution else None,
        "location": str(location).lower() if present and location else None,
    }


def system_prompt(lesion: str) -> str:
    info = LESION_INFO[lesion]
    return (
        "You are a fundus lesion perception specialist.\n\n"
        "This is a strictly single-lesion perception task. Inspect the image only for the specified target lesion. "
        "Do not assign a diabetic retinopathy grade, diagnose a disease stage, or report non-target lesions.\n\n"
        "Target lesion:\n"
        f"- Name: {info['name']}\n"
        f"- Abbreviation: {lesion}\n"
        f"- Typical visual evidence: {info['visual']}.\n"
        f"- Important exclusions: {info['exclude']}.\n\n"
        "Base the decision only on directly visible image evidence. Do not infer lesion presence or absence from a DR grade."
    )


def user_prompt(lesion: str) -> str:
    name = LESION_INFO[lesion]["name"]
    return (
        f"<image>\n\nInspect this fundus image for {name} ({lesion}) only.\n\n"
        "Determine whether directly visible evidence of the target lesion is present. Briefly describe the relevant "
        "visual evidence, exclude plausible confounders when applicable, and return the structured result."
    )


def assistant_answer(lesion: str, evidence: dict[str, Any]) -> str:
    info = LESION_INFO[lesion]
    present = evidence["state"] == "present"
    if present:
        target = f"Visible findings are consistent with {info['visual']}."
        confounder = info["positive_confounder"]
        attrs = {
            k: evidence[k]
            for k in ("count_bucket", "area_bucket", "distribution", "location")
            if evidence.get(k)
        }
        if attrs:
            attr_text = "Reliable coarse attributes: " + ", ".join(f"{k}={v}" for k, v in attrs.items()) + "."
        else:
            attr_text = "No additional target-lesion attributes are reported."
        conclusion = f"Directly visible evidence supports the presence of {info['name']}."
    else:
        target = f"No reliable directly visible evidence consistent with {info['visual']} is identified."
        confounder = info["negative_confounder"]
        attrs = {}
        attr_text = "No target-lesion attributes are reported because no reliable target evidence is present."
        conclusion = f"No reliable visual evidence supports the presence of {info['name']}."
    payload = {
        "task": "stage1_single_lesion_perception",
        "target_lesion": {"name": info["name"], "abbreviation": lesion},
        "image_quality": "adequate",
        "evidence_state": evidence["state"],
        "present": present,
        "attributes": attrs,
    }
    return (
        "[Target Evidence]\n" + target
        + "\n\n[Confounder Assessment]\n" + confounder
        + "\n\n[Attribute Summary]\n" + attr_text
        + "\n\n[Conclusion]\n" + conclusion
        + "\n\n[Structured Output]\n" + json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    )


def make_item(record: dict[str, Any], lesion: str, evidence: dict[str, Any], split: str, image_override: str | None = None) -> dict[str, Any]:
    return {
        "messages": [
            {"role": "system", "content": system_prompt(lesion)},
            {"role": "user", "content": user_prompt(lesion)},
            {"role": "assistant", "content": assistant_answer(lesion, evidence)},
        ],
        "images": [image_override or sft_image_path(record)],
        "meta": {
            "record_id": record.get("record_id"),
            "image_group": image_key(record),
            "dataset": record.get("dataset"),
            "dataset_split": record.get("split"),
            "split": split,
            "task": "stage1_single_lesion_perception",
            "lesion": lesion,
            "present_state": evidence["state"],
            "evidence_level": evidence["tier"],
            "evidence_source": evidence["source"],
            "label_origin": evidence["label_origin"],
        },
    }


def sample_tiered(candidates: list[tuple[dict[str, Any], dict[str, Any]]], target: int, mix: dict[str, float], rng: random.Random) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    grouped: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]] = defaultdict(list)
    for item in candidates:
        grouped[item[1]["tier"]].append(item)
    for values in grouped.values():
        rng.shuffle(values)
    out: list[tuple[dict[str, Any], dict[str, Any]]] = []
    used: set[tuple[str, str]] = set()
    for tier, ratio in mix.items():
        need = round(target * ratio)
        pool = grouped.get(tier, [])
        take = pool[:need]
        out.extend(take)
        used.update((image_key(r), e["tier"]) for r, e in take)
    remaining = [x for tier in ("S0", "S1", "S2", "S3", "S4") for x in grouped.get(tier, []) if (image_key(x[0]), x[1]["tier"]) not in used]
    out.extend(remaining[: max(0, target - len(out))])
    # Exposure cycling is allowed only after all unique candidates are exhausted.
    base = list(out) if out else remaining
    while base and len(out) < target:
        block = list(base)
        rng.shuffle(block)
        out.extend(block[: target - len(out)])
    return out[:target]


def load_aug_map(path: Path) -> dict[tuple[str, str], list[str]]:
    out = {}
    if not path.exists():
        return out
    for row in read_jsonl(path):
        out[(str(row.get("lesion")), str(row.get("record_id")))] = list(row.get("augmented_paths") or [])
    return out


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    c = Counter()
    groups = defaultdict(set)
    for row in rows:
        m = row["meta"]
        c[(m["lesion"], m["present_state"], m["evidence_level"], m["evidence_source"])] += 1
        groups[(m["lesion"], m["present_state"])].add(m["image_group"])
    return {
        "n": len(rows),
        "counts": {str(k): v for k, v in sorted(c.items(), key=lambda x: str(x[0]))},
        "unique_image_groups": {str(k): len(v) for k, v in sorted(groups.items(), key=lambda x: str(x[0]))},
    }


def choose_rare_locked(records: list[dict[str, Any]], facts: dict[str, dict[str, Any]], seed: int) -> dict[str, set[str]]:
    rng = random.Random(seed)
    result: dict[str, set[str]] = {}
    for lesion, pos_n, neg_n in (("NV", 8, 80), ("IRMA", 20, 80)):
        pos, neg = [], []
        for r in records:
            e = facts.get(image_key(r), {}).get(lesion)
            if not e or e["tier"] not in {"S0", "S1"}:
                continue
            (pos if e["state"] == "present" else neg).append(image_key(r))
        rng.shuffle(pos)
        rng.shuffle(neg)
        result[lesion] = set(pos[:pos_n] + neg[:neg_n])
    return result


def build(args: argparse.Namespace) -> dict[str, Any]:
    rng = random.Random(args.seed)
    records = read_jsonl(args.validated)
    mask_paths = ddr_mask_index(args.ddr_root)
    mask_facts = {k: mask_label(v) for k, v in mask_paths.items()}
    rare_mask_paths = fgadr_rare_mask_index(args.fgadr_root)
    rare_mask_facts = {k: mask_label(v) for k, v in rare_mask_paths.items()}
    thresholds = area_thresholds(mask_facts)
    facts: dict[str, dict[str, Any]] = defaultdict(dict)
    for record in records:
        for lesion in LESIONS:
            e = evidence_from_record(record, lesion, mask_facts, rare_mask_facts, thresholds)
            if e:
                facts[image_key(record)][lesion] = e

    rare_locked = choose_rare_locked(records, facts, args.seed + 7)
    excluded_gold = {image_key(r) for r in records if r.get("dataset") == "ddr_seg" and r.get("split") in {"valid", "test"}}
    excluded_rare = set().union(*rare_locked.values())
    internal_val = {
        image_key(r)
        for r in records
        if image_key(r) not in excluded_gold | excluded_rare and hbucket(image_key(r)) < args.internal_val_pct
    }
    train_records = [r for r in records if image_key(r) not in excluded_gold | excluded_rare | internal_val]

    train_pools: dict[tuple[str, str], list[tuple[dict[str, Any], dict[str, Any]]]] = defaultdict(list)
    for record in train_records:
        for lesion, evidence in facts.get(image_key(record), {}).items():
            train_pools[(lesion, evidence["state"])].append((record, evidence))

    train_rows: list[dict[str, Any]] = []
    for lesion in LESIONS:
        for state in ("present", "absent"):
            target = TRAIN_TARGETS[lesion][state]
            selected = sample_tiered(train_pools[(lesion, state)], target, TIER_MIX[lesion][state], rng)
            train_rows.extend(make_item(r, lesion, e, "train") for r, e in selected)
    rng.shuffle(train_rows)

    internal_rows = []
    for record in records:
        if image_key(record) not in internal_val:
            continue
        for lesion, e in facts.get(image_key(record), {}).items():
            internal_rows.append(make_item(record, lesion, e, "internal_val"))
    rng.shuffle(internal_rows)

    gold_rows: dict[str, list[dict[str, Any]]] = {"dev": [], "test": []}
    per_lesion_gold: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        if record.get("dataset") != "ddr_seg" or record.get("split") not in {"valid", "test"}:
            continue
        split = "dev" if record["split"] == "valid" else "test"
        for lesion in MAIN4:
            e = facts[image_key(record)][lesion]
            item = make_item(record, lesion, e, f"gold_{split}")
            gold_rows[split].append(item)
            per_lesion_gold[(split, lesion)].append(item)

    challenge_rows = []
    for record in records:
        if image_key(record) not in internal_val:
            continue
        for lesion, e in facts.get(image_key(record), {}).items():
            if e["state"] == "absent" and e["tier"] in {"S2", "S3", "S4"}:
                challenge_rows.append(make_item(record, lesion, e, "weak_negative_challenge"))
    rng.shuffle(challenge_rows)

    rare_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        key = image_key(record)
        for lesion in RARE2:
            if key in rare_locked[lesion]:
                rare_rows[lesion].append(make_item(record, lesion, facts[key][lesion], f"{lesion.lower()}_locked"))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "train": args.out_dir / "fundus_stage1_en_cot_train_sft.jsonl",
        "internal_val": args.out_dir / "fundus_stage1_en_cot_internal_val_sft.jsonl",
        "gold_dev": args.out_dir / "fundus_stage1_en_cot_gold_dev_sft.jsonl",
        "gold_test": args.out_dir / "fundus_stage1_en_cot_gold_test_sft.jsonl",
        "weak_negative_challenge": args.out_dir / "fundus_stage1_en_cot_weak_negative_challenge_sft.jsonl",
        "irma_locked": args.out_dir / "fundus_stage1_en_cot_irma_locked_sft.jsonl",
        "nv_locked": args.out_dir / "fundus_stage1_en_cot_nv_locked_sft.jsonl",
    }
    write_jsonl(outputs["train"], train_rows)
    write_jsonl(outputs["internal_val"], internal_rows)
    write_jsonl(outputs["gold_dev"], gold_rows["dev"])
    write_jsonl(outputs["gold_test"], gold_rows["test"])
    write_jsonl(outputs["weak_negative_challenge"], challenge_rows)
    write_jsonl(outputs["irma_locked"], rare_rows["IRMA"])
    write_jsonl(outputs["nv_locked"], rare_rows["NV"])
    for (split, lesion), rows in per_lesion_gold.items():
        write_jsonl(args.out_dir / f"fundus_stage1_en_cot_gold_{split}_{lesion.lower()}_sft.jsonl", rows)

    stats = {
        "version": "fundus_stage1_en_cot_v1",
        "seed": args.seed,
        "design": [
            "strict_single_lesion_english_cot",
            "explicit_target_definition_in_system_prompt",
            "evidence_tiers_S0_to_S4",
            "DDR_valid_test_main4_gold",
            "FGADR_rare_locked_holdout",
            "image_group_disjoint_splits",
            "no_grade_in_model_visible_text",
        ],
        "area_fraction_tertiles_by_lesion": thresholds,
        "train_targets": TRAIN_TARGETS,
        "tier_mix_targets": TIER_MIX,
        "excluded_image_groups": {
            "gold": len(excluded_gold),
            "rare_locked_union": len(excluded_rare),
            "internal_val": len(internal_val),
        },
        "sets": {
            "train": summarize(train_rows),
            "internal_val": summarize(internal_rows),
            "gold_dev": summarize(gold_rows["dev"]),
            "gold_test": summarize(gold_rows["test"]),
            "weak_negative_challenge": summarize(challenge_rows),
            "irma_locked": summarize(rare_rows["IRMA"]),
            "nv_locked": summarize(rare_rows["NV"]),
        },
        "outputs": {k: str(v) for k, v in outputs.items()},
    }
    stats_path = args.out_dir / "fundus_stage1_en_cot_stats.json"
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    return stats


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validated", type=Path, default=VALIDATED)
    parser.add_argument("--ddr-root", type=Path, default=DDR_ROOT)
    parser.add_argument("--fgadr-root", type=Path, default=FGADR_ROOT)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--seed", type=int, default=20260608)
    parser.add_argument("--internal-val-pct", type=int, default=10)
    args = parser.parse_args()
    build(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
