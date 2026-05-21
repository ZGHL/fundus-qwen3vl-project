#!/usr/bin/env python3
"""Build L3 v4: full-pool, single-lesion English SFT.

Design (per v4 spec):
- 8:2 image_id-level deterministic split, leak-safe
- Single-lesion-per-prompt structure preserved from v3
- 4-section English CoT: [Observe] [Evidence] [Conclusion] [JSON]
- Visual prose preserved in [Evidence] (constraint: explicit lesion identification)
- present / count_bucket / area_bucket / source preserved (proven useful at L3)
- MA neg = v4-strict (explicit absent only; no template_only or unknown)
- IRMA/NV: take all positives + min(neg, 4*pos) — accept imbalance for sparse classes

Output:
  data/annotation_v4/fundus_l3_v4_train_sft.jsonl
  data/annotation_v4/fundus_l3_v4_val_sft.jsonl
  data/annotation_v4/fundus_l3_v4_stats.json
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

# Reuse canonical helpers from v3 pipeline
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "fundus"))
from build_fundus_sft import hbucket, read_jsonl, sft, write_jsonl  # noqa: E402

VALIDATED = Path("/home/aim_lab/LLaMA-Factory/data/fundus_validated/validated_clean.jsonl")
OUT_DIR = Path("/home/aim_lab/LLaMA-Factory/data/annotation_v4")
SPARSE_AUG_MANIFEST = Path("/home/aim_lab/LLaMA-Factory/data/cropped/_aug_v4_sparse/manifest.jsonl")
LESIONS = ("MA", "HE", "EX", "SE", "IRMA", "NV")
SPARSE_AUG_LESIONS = {"NV", "IRMA"}  # only these classes get pixel augmentation
VAL_PCT = 20

# State constants — single source of truth, prevents string-literal typos across helpers
PRESENT = "present"
ABSENT = "absent"
TEMPLATE_ONLY = "template_only"
UNKNOWN = "unknown"

# Lesion-specific visual prose (single short clause)
LESION_PROSE = {
    "MA": "small reddish dots, posterior pole / midperiphery",
    "HE": "dark red blot or dot hemorrhages",
    "EX": "bright yellow well-circumscribed deposits",
    "SE": "soft, fluffy cotton-wool patches",
    "IRMA": "tortuous intraretinal microvascular abnormalities",
    "NV": "abnormal new vessels at disc / elsewhere",
}

LESION_FULL = {
    "MA": "microaneurysm",
    "HE": "hemorrhage",
    "EX": "hard exudate",
    "SE": "soft exudate / cotton-wool spot",
    "IRMA": "intraretinal microvascular abnormality",
    "NV": "neovascularization",
}

# Source short-tag map (hoisted from per-call closure)
_SRC_TAG = {
    "strong_mask_stage1_easy": "strong_mask",
    "fgadr_lesion_only_sft_v3": "strong_mask",
    "validated_retsam": "retsam",
    "grade_rule_override": "grade_rule",
    "grade_rule": "grade_rule",
    "cleaning_rule": "cleaning_rule",
}

SYSTEM_PROMPT_TEMPLATE = (
    "You are a fundus image analyst. Inspect ONLY for {lesion} ({lesion_full}). "
    "Single-lesion task: do NOT output a DR grade and do NOT mention other lesions. "
    "Describe colour, shape, boundary and quantity first, then name the lesion."
)

USER_PROMPT_TEMPLATE = (
    "Examine this fundus image for the presence of {lesion} ({lesion_full}). "
    "Output in the order: [Observe] -> [Evidence] -> [Conclusion] -> [JSON]."
)


# ---------------------------- split assignment ----------------------------

def assign_splits(records: list[dict]) -> dict[str, str]:
    """image_id -> {'train','val','eval'}. Leak-safe: same image_id always same split."""
    eval_iids = set()
    for r in records:
        if r.get("dataset") == "idrid" and r.get("split") == "test":
            eval_iids.add(r["image_id"])
        if r.get("dataset") == "ddr_seg" and r.get("split") in {"valid", "test"}:
            eval_iids.add(r["image_id"])

    iid_split: dict[str, str] = {}
    for r in records:
        iid = r["image_id"]
        if iid in eval_iids:
            iid_split[iid] = "eval"
        elif iid not in iid_split:
            iid_split[iid] = "val" if hbucket(iid) < VAL_PCT else "train"
    return iid_split


# ---------------------------- evidence shaping ----------------------------

def evidence_for_lesion(record: dict, k: str) -> tuple[str, dict] | None:
    if not record.get("usable_for", {}).get("L3"):
        return None
    lesion = record.get("lesions", {}).get(k)
    if not isinstance(lesion, dict):
        return None
    p = lesion.get("present")
    if p is True:
        return (PRESENT, lesion)
    if p is False:
        return (ABSENT, lesion)
    if p == "template_only":
        return (TEMPLATE_ONLY, lesion)
    return (UNKNOWN, lesion)


def make_evidence_line(k: str, state: str, lesion: dict) -> str:
    src_tag = _SRC_TAG.get(lesion.get("source", "unknown"), lesion.get("source", "unknown"))
    if state == PRESENT:
        cb = lesion.get("count_bucket") or "n/a"
        ab = lesion.get("area_bucket") or "n/a"
        return (f"{k}: present=true, count={cb}, area={ab}, source={src_tag}. "
                f"Visible as {LESION_PROSE[k]}.")
    if state == ABSENT:
        # The raw_present flag tells us whether cleaning suppressed an originally-detected signal
        if lesion.get("raw_present") is True:
            return (f"{k}: present=false, source={src_tag} (cleaned: low confidence). "
                    f"No reliable {LESION_PROSE[k]}.")
        return f"{k}: present=false. No {LESION_PROSE[k]} observed."
    if state == TEMPLATE_ONLY:
        return (f"{k}: present=template_only, source=grade_rule. "
                f"Grade label suggests {k}, no direct visual confirmation.")
    return f"{k}: present=unknown, source={src_tag}. Insufficient evidence to decide."


def make_conclusion(k: str, state: str) -> str:
    if state == PRESENT:
        return f"{k} is present."
    if state == ABSENT:
        return f"{k} is absent."
    if state == TEMPLATE_ONLY:
        return f"{k} cannot be confirmed visually (label template only)."
    return f"{k} cannot be determined from the current image."


def make_json(k: str, state: str, lesion: dict) -> str:
    obj = {
        "task": f"L3_{k}",
        "lesion": k,
        "present": (
            True if state == PRESENT
            else False if state == ABSENT
            else state
        ),
        "count": lesion.get("count_bucket"),
        "area": lesion.get("area_bucket"),
        "source": lesion.get("source", "unknown"),
    }
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


def make_assistant(k: str, state: str, lesion: dict) -> str:
    return (
        f"[Observe] Inspect for {k} only. Single-lesion task, no DR grading.\n\n"
        f"[Evidence] {make_evidence_line(k, state, lesion)}\n\n"
        f"[Conclusion] {make_conclusion(k, state)}\n\n"
        f"[JSON] {make_json(k, state, lesion)}"
    )


def make_sft_item(k: str, state: str, lesion: dict, record: dict, split: str) -> dict:
    meta = {
        "record_id": record["record_id"],
        "image_id": record["image_id"],
        "dataset": record["dataset"],
        "task": f"L3_{k}",
        "lesion": k,
        "split": split,
        "present_state": state,
        "source": lesion.get("source", "unknown"),
        "count_bucket": lesion.get("count_bucket"),
        "area_bucket": lesion.get("area_bucket"),
    }
    system = SYSTEM_PROMPT_TEMPLATE.format(lesion=k, lesion_full=LESION_FULL[k])
    user = USER_PROMPT_TEMPLATE.format(lesion=k, lesion_full=LESION_FULL[k])
    assistant = make_assistant(k, state, lesion)
    return sft(system, user, assistant, record["image_path"], meta)


# ---------------------------- sparse-class augmentation manifest ----------------------------

def load_sparse_aug_manifest() -> dict[str, dict[str, list[str]]]:
    """Returns {lesion: {record_id: [augmented_data_relative_paths]}}.

    The augmented PNGs already exist on disk; we attach their paths to the
    matching original record at pool-build time, producing synthetic records
    with shared lesion_meta but distinct image_path.
    """
    out: dict[str, dict[str, list[str]]] = {}
    if not SPARSE_AUG_MANIFEST.exists():
        print(f"[aug] manifest not found at {SPARSE_AUG_MANIFEST}; sparse classes will NOT be augmented")
        return out
    for row in read_jsonl(SPARSE_AUG_MANIFEST):
        lesion = row["lesion"]
        out.setdefault(lesion, {})[row["record_id"]] = list(row["augmented_paths"])
    return out


def expand_with_aug(state: str, lesion_meta: dict, record: dict,
                    aug_paths: list[str]) -> list[tuple[str, dict, dict]]:
    """Original record + one synthetic per augmented path. All share lesion_meta."""
    out = [(state, lesion_meta, record)]
    for idx, aug_path in enumerate(aug_paths, 1):
        synth = dict(record)
        # Augmented PNG path is already relative to data/
        synth["image_path"] = aug_path if aug_path.startswith("data/") else f"data/{aug_path}"
        synth["record_id"] = f"{record['record_id']}__aug{idx}"
        out.append((state, lesion_meta, synth))
    return out


# ---------------------------- per-lesion bucketing (single pass) ----------------------------

def build_lesion_pools(records: list[dict],
                      aug_map: dict[str, dict[str, list[str]]] | None = None
                      ) -> dict[str, dict[str, list]]:
    """Single pass over records, bucket by (lesion, state).

    For NV/IRMA *positives* only, expand each record with its augmented copies
    from the manifest (preserves lesion_meta + supervision signal).
    """
    aug_map = aug_map or {}
    pools: dict[str, dict[str, list]] = {k: defaultdict(list) for k in LESIONS}
    for r in records:
        if not r.get("usable_for", {}).get("L3"):
            continue
        for k in LESIONS:
            ev = evidence_for_lesion(r, k)
            if ev is None:
                continue
            state, lesion_meta = ev
            # Augment sparse-class positives only
            if k in SPARSE_AUG_LESIONS and state == PRESENT:
                aug_paths = aug_map.get(k, {}).get(r["record_id"], [])
                if aug_paths:
                    pools[k][state].extend(
                        expand_with_aug(state, lesion_meta, r, aug_paths)
                    )
                    continue
            pools[k][state].append((state, lesion_meta, r))
    return pools


# ---------------------------- balanced sampling per lesion ----------------------------

def sample_train(pools: dict[str, list], lesion: str, seed: int) -> list:
    """v4 sampling.
    - MA/HE/EX/SE: 1:1 balanced, min(pos,neg)*2 (v4-strict, no template/unknown).
    - IRMA/NV: all positives + min(neg, 4*pos).
    """
    rng = random.Random(seed)
    pos = list(pools.get(PRESENT, []))
    neg = list(pools.get(ABSENT, []))
    rng.shuffle(pos)
    rng.shuffle(neg)
    if lesion in {"IRMA", "NV"}:
        n_pos = len(pos)
        n_neg = min(len(neg), n_pos * 4) if n_pos > 0 else 0
        return pos[:n_pos] + neg[:n_neg]
    n = min(len(pos), len(neg))
    return pos[:n] + neg[:n]


# ---------------------------- main ----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    records = list(read_jsonl(VALIDATED))
    iid_split = assign_splits(records)
    train_recs = [r for r in records if iid_split[r["image_id"]] == "train"]
    val_recs = [r for r in records if iid_split[r["image_id"]] == "val"]

    aug_map = load_sparse_aug_manifest()
    if aug_map:
        n = {k: sum(len(v) for v in m.values()) for k, m in aug_map.items()}
        print(f"[aug] loaded sparse manifest: {n}")

    # Augmentation applies only to train; val stays at natural pixel distribution
    train_pools = build_lesion_pools(train_recs, aug_map=aug_map)
    val_pools = build_lesion_pools(val_recs)

    train_items: list[dict] = []
    val_items: list[dict] = []
    stats: dict = {
        "v4_strict_ma_negatives_only": True,
        "val_pct": VAL_PCT,
        "per_lesion": {},
    }

    for idx, k in enumerate(LESIONS):
        sampled = sample_train(train_pools[k], k, args.seed + idx)
        for state, lesion, r in sampled:
            train_items.append(make_sft_item(k, state, lesion, r, "train"))

        # Val: include all available pos+neg, no balancing (full natural distribution)
        v_pos = v_neg = 0
        for state in (PRESENT, ABSENT):
            for ev_state, lesion, r in val_pools[k].get(state, []):
                val_items.append(make_sft_item(k, ev_state, lesion, r, "val"))
                if state == PRESENT:
                    v_pos += 1
                else:
                    v_neg += 1

        stats["per_lesion"][k] = {
            "train_available_pos": len(train_pools[k].get(PRESENT, [])),
            "train_available_neg": len(train_pools[k].get(ABSENT, [])),
            "train_sampled_pos": sum(1 for it in sampled if it[0] == PRESENT),
            "train_sampled_neg": sum(1 for it in sampled if it[0] == ABSENT),
            "val_pos": v_pos,
            "val_neg": v_neg,
        }

    rng = random.Random(args.seed)
    rng.shuffle(train_items)
    rng.shuffle(val_items)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    train_path = args.out_dir / "fundus_l3_v4_train_sft.jsonl"
    val_path = args.out_dir / "fundus_l3_v4_val_sft.jsonl"
    stats_path = args.out_dir / "fundus_l3_v4_stats.json"

    if not args.dry_run:
        write_jsonl(train_path, train_items)
        write_jsonl(val_path, val_items)

    stats["train_total"] = len(train_items)
    stats["val_total"] = len(val_items)
    stats["train_path"] = str(train_path)
    stats["val_path"] = str(val_path)

    if not args.dry_run:
        stats_path.write_text(json.dumps(stats, indent=2, ensure_ascii=False))

    print("=== L3 v4 build summary ===")
    print(f"train: {len(train_items)} items")
    print(f"val:   {len(val_items)} items")
    for k, s in stats["per_lesion"].items():
        print(f"  {k:<5} train={s['train_sampled_pos']:>4}+{s['train_sampled_neg']:>4}  "
              f"val={s['val_pos']:>4}+{s['val_neg']:>4}")


if __name__ == "__main__":
    main()
