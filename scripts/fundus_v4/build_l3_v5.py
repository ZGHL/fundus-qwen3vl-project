#!/usr/bin/env python3
"""Build L3 v5: per-lesion SFT with rich quadrant LOCATION (#1) from RetSAM enrichment.

Changes from L3 v4:
  - For HE/EX/SE: render quadrant-derived LOCATION prose ("predominantly inferior-temporal",
    "throughout all four quadrants", "single superior-nasal lesion") from quadrant_index.jsonl.
  - For NV present: render default "at disc or elsewhere" (no quadrant data available).
  - For MA/IRMA: render generic "intraretinal" qualifier (no quadrant data available).
  - JSON: gain `location` field (rich string when available, generic otherwise, null else).

Output:
  data/annotation_v4/fundus_l3_v5_train_sft.jsonl
  data/annotation_v4/fundus_l3_v5_val_sft.jsonl
  data/annotation_v4/fundus_l3_v5_stats.json
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "fundus"))
from build_fundus_sft import hbucket, read_jsonl, sft, write_jsonl  # noqa: E402

VALIDATED = Path("/home/aim_lab/LLaMA-Factory/data/fundus_validated/validated_clean.jsonl")
QUAD_INDEX = Path("/home/aim_lab/LLaMA-Factory/data/fundus_validated/quadrant_index.jsonl")
OUT_DIR = Path("/home/aim_lab/LLaMA-Factory/data/annotation_v4")
SPARSE_AUG_MANIFEST = Path("/home/aim_lab/LLaMA-Factory/data/cropped/_aug_v4_sparse/manifest.jsonl")
LESIONS = ("MA", "HE", "EX", "SE", "IRMA", "NV")
SPARSE_AUG_LESIONS = {"NV", "IRMA"}
VAL_PCT = 20

PRESENT, ABSENT, TEMPLATE_ONLY, UNKNOWN = "present", "absent", "template_only", "unknown"

LESION_PROSE = {
    "MA": "small reddish dots",
    "HE": "dark red blot or dot hemorrhages",
    "EX": "bright yellow well-circumscribed deposits",
    "SE": "soft, fluffy cotton-wool patches",
    "IRMA": "tortuous intraretinal microvascular abnormalities",
    "NV": "abnormal new vessels",
}
LESION_FULL = {
    "MA": "microaneurysm",
    "HE": "hemorrhage",
    "EX": "hard exudate",
    "SE": "soft exudate / cotton-wool spot",
    "IRMA": "intraretinal microvascular abnormality",
    "NV": "neovascularization",
}

Q_NAMES = {
    "ST": "superior-temporal",
    "SN": "superior-nasal",
    "IT": "inferior-temporal",
    "IN": "inferior-nasal",
}

# Default location for sparse-class lesions without quadrant data
DEFAULT_LOCATION = {
    "NV": "at disc or elsewhere",
    "IRMA": "intraretinal, near major vessels",
    "MA": "throughout posterior retina",
}

_SRC_TAG = {
    "strong_mask_stage1_easy":   "strong_mask",
    "fgadr_lesion_only_sft_v3":  "strong_mask",
    "validated_retsam":          "retsam",
    "grade_rule_override":       "grade_rule",
    "grade_rule":                "grade_rule",
    "cleaning_rule":             "cleaning_rule",
}

SYSTEM_PROMPT_TEMPLATE = (
    "You are a fundus image analyst. Inspect ONLY for {lesion} ({lesion_full}). "
    "Single-lesion task: do NOT output a DR grade and do NOT mention other lesions. "
    "Describe colour, shape, boundary, quantity AND spatial location first, then name the lesion."
)
USER_PROMPT_TEMPLATE = (
    "Examine this fundus image for the presence of {lesion} ({lesion_full}). "
    "Output in the order: [Observe] -> [Evidence] -> [Conclusion] -> [JSON]."
)


# ---------------------------- shared helpers ----------------------------

def assign_splits(records):
    eval_iids = set()
    for r in records:
        if r.get("dataset") == "idrid" and r.get("split") == "test":
            eval_iids.add(r["image_id"])
        if r.get("dataset") == "ddr_seg" and r.get("split") in {"valid", "test"}:
            eval_iids.add(r["image_id"])
    iid_split = {}
    for r in records:
        iid = r["image_id"]
        if iid in eval_iids: iid_split[iid] = "eval"
        elif iid not in iid_split:
            iid_split[iid] = "val" if hbucket(iid) < VAL_PCT else "train"
    return iid_split


def load_quadrant_index() -> dict:
    """Return {(dataset, image_id): {lesion_short: {ST,SN,IT,IN,total}}}."""
    out = {}
    if not QUAD_INDEX.exists():
        print(f"[quad] missing {QUAD_INDEX}; LOCATION will use defaults only")
        return out
    for row in read_jsonl(QUAD_INDEX):
        out[(row["dataset"], row["image_id"])] = row["quadrants"]
    return out


def quadrants_to_location(quad: dict | None) -> str | None:
    """Convert {ST,SN,IT,IN,total} → natural prose. None if no useful info."""
    if not quad or quad.get("total", 0) == 0:
        return None
    total = quad["total"]
    q = {k: quad.get(k, 0) for k in ("ST", "SN", "IT", "IN")}
    non_zero = sum(1 for v in q.values() if v > 0)

    if non_zero == 4 and min(q.values()) >= max(1, total // 6):
        return "throughout all four quadrants (macula-centered)"
    if non_zero == 1:
        only_q = max(q, key=q.get)
        return f"in the {Q_NAMES[only_q]} quadrant only"
    sorted_q = sorted(q.items(), key=lambda x: -x[1])
    top = [k for k, v in sorted_q if v >= max(1, total // 4)]
    if len(top) == 1:
        return f"predominantly in the {Q_NAMES[top[0]]} quadrant"
    if len(top) == 2:
        return f"distributed in the {Q_NAMES[top[0]]} and {Q_NAMES[top[1]]} quadrants"
    return f"distributed across {len(top)} quadrants ({', '.join(Q_NAMES[k] for k in top)})"


def quadrants_to_enum(quad: dict | None) -> str | None:
    """v5.1: short enum code for JSON. None if no useful info.

    Possible codes: 4Q | ST_only | SN_only | IT_only | IN_only |
                    ST_pred | SN_pred | IT_pred | IN_pred |
                    ST+SN | ST+IT | ST+IN | SN+IT | SN+IN | IT+IN
    """
    if not quad or quad.get("total", 0) == 0:
        return None
    total = quad["total"]
    q = {k: quad.get(k, 0) for k in ("ST", "SN", "IT", "IN")}
    non_zero = sum(1 for v in q.values() if v > 0)
    if non_zero == 4 and min(q.values()) >= max(1, total // 6):
        return "4Q"
    if non_zero == 1:
        only_q = max(q, key=q.get)
        return f"{only_q}_only"
    sorted_q = sorted(q.items(), key=lambda x: -x[1])
    top = [k for k, v in sorted_q if v >= max(1, total // 4)]
    if len(top) == 1:
        return f"{top[0]}_pred"
    if len(top) == 2:
        return f"{top[0]}+{top[1]}"
    return f"{top[0]}+{top[1]}+{top[2]}" if len(top) >= 3 else "+".join(top)


def location_to_enum(loc_str: str | None, lesion: str) -> str | None:
    """Map prose location → enum code for JSON.
    Handles both quadrant-derived prose and default fallbacks."""
    if loc_str is None:
        return None
    s = loc_str.lower()
    # Quadrant patterns
    if "all four quadrants" in s or "4q" in s:
        return "4Q"
    for q in ("ST", "SN", "IT", "IN"):
        if "quadrant only" in s and Q_NAMES[q] in s:
            return f"{q}_only"
    for q in ("ST", "SN", "IT", "IN"):
        if "predominantly" in s and Q_NAMES[q] in s:
            return f"{q}_pred"
    # Default fallbacks
    if "at disc" in s:        return "at_disc"
    if "intraretinal" in s:    return "intraretinal"
    if "posterior" in s:       return "posterior_pole"
    if "midperiphery" in s:    return "midperiphery"
    # Generic
    return "other"


def get_location_for_lesion(lesion_meta: dict, lesion_key: str,
                            record: dict, quad_idx: dict) -> str | None:
    """Best-effort: prefer quadrant_index for HE/EX/SE, else default per lesion."""
    if lesion_key in ("HE", "EX", "SE"):
        key = (record.get("dataset"), record.get("image_id"))
        quad = quad_idx.get(key, {}).get(lesion_key)
        loc = quadrants_to_location(quad)
        if loc:
            return loc
    # legacy location_band field (rare ~200 records)
    lb = lesion_meta.get("location_band")
    if lb and isinstance(lb, str):
        return {
            "黄斑区": "macular region",
            "后极部": "posterior pole",
            "中周部": "mid-periphery",
            "周边部": "peripheral retina",
        }.get(lb, lb)
    # sparse lesion default (only when present)
    return DEFAULT_LOCATION.get(lesion_key)


# ---------------------------- lesion logic ----------------------------

def evidence_for_lesion(record, k):
    if not record.get("usable_for", {}).get("L3"): return None
    lesion = record.get("lesions", {}).get(k)
    if not isinstance(lesion, dict): return None
    p = lesion.get("present")
    if p is True: return (PRESENT, lesion)
    if p is False: return (ABSENT, lesion)
    if p == "template_only": return (TEMPLATE_ONLY, lesion)
    return (UNKNOWN, lesion)


def make_evidence_line(k, state, lesion, record, quad_idx):
    src_tag = _SRC_TAG.get(lesion.get("source", "unknown"), lesion.get("source", "unknown"))
    visual = LESION_PROSE[k]

    if state == PRESENT:
        cb = lesion.get("count_bucket") or "n/a"
        ab = lesion.get("area_bucket") or "n/a"
        loc = get_location_for_lesion(lesion, k, record, quad_idx)
        if loc:
            return (f"{k}: present=true, count={cb}, area={ab}, location={loc}, "
                    f"source={src_tag}. Visible as {visual} {loc}.")
        return (f"{k}: present=true, count={cb}, area={ab}, source={src_tag}. "
                f"Visible as {visual}.")

    if state == ABSENT:
        if lesion.get("raw_present") is True:
            return (f"{k}: present=false, source={src_tag} (cleaned: low confidence). "
                    f"No reliable {visual}.")
        return f"{k}: present=false. No {visual} observed."

    if state == TEMPLATE_ONLY:
        return (f"{k}: present=template_only, source=grade_rule. "
                f"Grade label suggests {k}, no direct visual confirmation.")
    return f"{k}: present=unknown, source={src_tag}. Insufficient evidence to decide."


def make_conclusion(k, state):
    if state == PRESENT:      return f"{k} is present."
    if state == ABSENT:       return f"{k} is absent."
    if state == TEMPLATE_ONLY: return f"{k} cannot be confirmed visually (label template only)."
    return f"{k} cannot be determined from the current image."


def make_json(k, state, lesion, record, quad_idx):
    """v5.1: enum-only JSON; location uses short code (not prose)."""
    # Compute location enum from quadrants directly (HE/EX/SE) or fallback per lesion
    loc_enum = None
    if state == PRESENT:
        if k in ("HE", "EX", "SE"):
            key = (record.get("dataset"), record.get("image_id"))
            quad = quad_idx.get(key, {}).get(k)
            loc_enum = quadrants_to_enum(quad)
        if loc_enum is None:
            # Default per lesion
            loc_enum = {"NV": "at_disc", "IRMA": "intraretinal", "MA": "posterior_pole"}.get(k)
    src_short = _SRC_TAG.get(lesion.get("source", "unknown"), lesion.get("source", "unknown"))
    obj = {
        "task": f"L3_{k}",
        "present": (True if state == PRESENT else False if state == ABSENT else state),
        "count": lesion.get("count_bucket"),
        "area": lesion.get("area_bucket"),
        "location": loc_enum,
        "source": src_short,
    }
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


def make_assistant(k, state, lesion, record, quad_idx):
    """v5.2: Findings / Impression / Result bullet structure."""
    src_tag = _SRC_TAG.get(lesion.get("source", "unknown"), lesion.get("source", "unknown"))
    visual = LESION_PROSE[k]

    # Compute location enum + prose
    loc_enum = None
    loc_prose = None
    if state == PRESENT:
        if k in ("HE", "EX", "SE"):
            key = (record.get("dataset"), record.get("image_id"))
            quad = quad_idx.get(key, {}).get(k)
            loc_enum = quadrants_to_enum(quad)
            loc_prose = quadrants_to_location(quad)
        if loc_enum is None:
            loc_enum = {"NV": "at_disc", "IRMA": "intraretinal", "MA": "posterior_pole"}.get(k)
            loc_prose = {"NV": "at disc or elsewhere",
                         "IRMA": "intraretinal, near major vessels",
                         "MA": "throughout posterior retina"}.get(k)

    if state == PRESENT:
        cb = lesion.get("count_bucket") or "n/a"
        ab = lesion.get("area_bucket") or "n/a"
        findings = (
            f"- {k}: present\n"
            f"- Count: {cb}\n"
            f"- Area: {ab}\n"
            f"- Location: {loc_enum}\n"
            f"- Source: {src_tag}"
        )
        impression = (
            f"- {k} appearance: {visual}.\n"
            f"- Distribution: {loc_prose}."
        )
        result = (f"task=L3_{k} | present=true | count={cb} | area={ab} | "
                  f"location={loc_enum} | source={src_tag}")
    elif state == ABSENT:
        cleaned = " (signal cleaned: low confidence)" if lesion.get("raw_present") is True else ""
        findings = (
            f"- {k}: not observed{cleaned}\n"
            f"- Source: {src_tag}"
        )
        impression = f"- No {visual} identified."
        result = f"task=L3_{k} | present=false | source={src_tag}"
    elif state == TEMPLATE_ONLY:
        findings = (
            f"- {k}: not directly visible\n"
            f"- Note: grade-label template suggests {k} presence"
        )
        impression = f"- {k} inferred from grade template only; no direct visual confirmation."
        result = f"task=L3_{k} | present=template_only | source=grade_rule"
    else:  # UNKNOWN
        findings = (
            f"- {k}: insufficient evidence\n"
            f"- Source: {src_tag}"
        )
        impression = f"- {k} cannot be determined from current image quality / source."
        result = f"task=L3_{k} | present=unknown | source={src_tag}"

    return (
        f"[Findings]\n{findings}\n\n"
        f"[Impression]\n{impression}\n\n"
        f"[Result] {result}"
    )


def make_sft_item(k, state, lesion, record, split, quad_idx):
    loc = get_location_for_lesion(lesion, k, record, quad_idx) if state == PRESENT else None
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
        "location": loc,
    }
    system = SYSTEM_PROMPT_TEMPLATE.format(lesion=k, lesion_full=LESION_FULL[k])
    user = USER_PROMPT_TEMPLATE.format(lesion=k, lesion_full=LESION_FULL[k])
    assistant = make_assistant(k, state, lesion, record, quad_idx)
    return sft(system, user, assistant, record["image_path"], meta)


def load_sparse_aug_manifest():
    if not SPARSE_AUG_MANIFEST.exists(): return {}
    out = {}
    for row in read_jsonl(SPARSE_AUG_MANIFEST):
        out.setdefault(row["lesion"], {})[row["record_id"]] = list(row["augmented_paths"])
    return out


def expand_with_aug(state, lesion_meta, record, aug_paths):
    out = [(state, lesion_meta, record)]
    for idx, ap in enumerate(aug_paths, 1):
        synth = dict(record)
        synth["image_path"] = ap if ap.startswith("data/") else f"data/{ap}"
        synth["record_id"] = f"{record['record_id']}__aug{idx}"
        out.append((state, lesion_meta, synth))
    return out


def build_lesion_pools(records, aug_map=None):
    aug_map = aug_map or {}
    pools = {k: defaultdict(list) for k in LESIONS}
    for r in records:
        if not r.get("usable_for", {}).get("L3"): continue
        for k in LESIONS:
            ev = evidence_for_lesion(r, k)
            if ev is None: continue
            state, lm = ev
            if k in SPARSE_AUG_LESIONS and state == PRESENT:
                aug_paths = aug_map.get(k, {}).get(r["record_id"], [])
                if aug_paths:
                    pools[k][state].extend(expand_with_aug(state, lm, r, aug_paths))
                    continue
            pools[k][state].append((state, lm, r))
    return pools


def sample_train(pool, lesion, seed):
    rng = random.Random(seed)
    pos = list(pool.get(PRESENT, []))
    neg = list(pool.get(ABSENT, []))
    rng.shuffle(pos); rng.shuffle(neg)
    if lesion in {"IRMA", "NV"}:
        n_pos = len(pos)
        n_neg = min(len(neg), n_pos * 4) if n_pos > 0 else 0
        return pos[:n_pos] + neg[:n_neg]
    n = min(len(pos), len(neg))
    return pos[:n] + neg[:n]


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

    quad_idx = load_quadrant_index()
    print(f"[quad] loaded {len(quad_idx)} (dataset, image_id) entries")

    aug_map = load_sparse_aug_manifest()
    if aug_map:
        n = {k: sum(len(v) for v in m.values()) for k, m in aug_map.items()}
        print(f"[aug] loaded sparse manifest: {n}")

    train_pools = build_lesion_pools(train_recs, aug_map=aug_map)
    val_pools = build_lesion_pools(val_recs)

    train_items = []; val_items = []
    stats = {"v5_changes": ["quadrant_location", "qualitative_descriptors"],
             "val_pct": VAL_PCT, "per_lesion": {}}
    loc_train = 0; loc_train_rich = 0

    for idx, k in enumerate(LESIONS):
        sampled = sample_train(train_pools[k], k, args.seed + idx)
        for state, lesion, r in sampled:
            item = make_sft_item(k, state, lesion, r, "train", quad_idx)
            train_items.append(item)
            if item["meta"].get("location"):
                loc_train += 1
                if k in ("HE","EX","SE") and "quadrant" in (item["meta"]["location"] or "").lower():
                    loc_train_rich += 1
        v_pos = v_neg = 0
        for state in (PRESENT, ABSENT):
            for ev_state, lesion, r in val_pools[k].get(state, []):
                val_items.append(make_sft_item(k, ev_state, lesion, r, "val", quad_idx))
                if state == PRESENT: v_pos += 1
                else: v_neg += 1
        stats["per_lesion"][k] = {
            "train_available_pos": len(train_pools[k].get(PRESENT, [])),
            "train_available_neg": len(train_pools[k].get(ABSENT, [])),
            "train_sampled_pos": sum(1 for it in sampled if it[0] == PRESENT),
            "train_sampled_neg": sum(1 for it in sampled if it[0] == ABSENT),
            "val_pos": v_pos, "val_neg": v_neg,
        }

    stats["train_with_location"] = loc_train
    stats["train_with_quadrant_rich"] = loc_train_rich

    rng = random.Random(args.seed)
    rng.shuffle(train_items); rng.shuffle(val_items)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    train_path = args.out_dir / "fundus_l3_v5_train_sft.jsonl"
    val_path   = args.out_dir / "fundus_l3_v5_val_sft.jsonl"
    stats_path = args.out_dir / "fundus_l3_v5_stats.json"

    if not args.dry_run:
        write_jsonl(train_path, train_items)
        write_jsonl(val_path, val_items)

    stats["train_total"] = len(train_items)
    stats["val_total"] = len(val_items)
    if not args.dry_run:
        stats_path.write_text(json.dumps(stats, indent=2, ensure_ascii=False))

    print("=== L3 v5 build summary ===")
    print(f"train: {len(train_items)}  val: {len(val_items)}")
    print(f"train w/ any location: {loc_train}  (HE/EX/SE quadrant-rich: {loc_train_rich})")
    for k, s in stats["per_lesion"].items():
        print(f"  {k:<5} train={s['train_sampled_pos']:>4}+{s['train_sampled_neg']:>4}  "
              f"val={s['val_pos']:>4}+{s['val_neg']:>4}")


if __name__ == "__main__":
    main()
