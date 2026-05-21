#!/usr/bin/env python3
"""Build a ~112-sample quick eval subset from v4_mixed_val for Arm A vs Arm B comparison.

Per-task picks:
  L4_grade        : 25  (5 per grade G0-G4)
  L3_HE / EX / MA / SE / IRMA  : 10 each (5 pos + 5 neg)
  L3_NV           : 5   (all available pos + neg fill)
  L2_laterality   : 10  (5 L + 5 R)
  L2_cdr          : 12  (3 per bucket)
  L2_vessel       : 10  (5 valid + 5 abstain)
  -------------------------
  Total: ~112 samples

Reproducible (seed=2026). Output: data/annotation_v4/fundus_v4_quick_eval.jsonl
"""
import json
import random
from collections import defaultdict
from pathlib import Path

VAL = Path("/home/aim_lab/LLaMA-Factory/data/annotation_v4/fundus_v4_mixed_val_sft.jsonl")
OUT = Path("/home/aim_lab/LLaMA-Factory/data/annotation_v4/fundus_v4_quick_eval_sft.jsonl")
SEED = 2026

PER_GRADE_L4 = 5         # 5 × 5 = 25
PER_LESION_PN = 5        # 5 pos + 5 neg = 10 per non-sparse lesion
NV_TOTAL = 5             # full pos + neg fill
LATERALITY_HALF = 5
CDR_PER_BUCKET = 3
VESSEL_PER_STATE = 5


def main():
    rng = random.Random(SEED)
    items = [json.loads(l) for l in open(VAL)]

    by_task = defaultdict(list)
    for it in items:
        by_task[it["meta"]["task"]].append(it)
    for arr in by_task.values():
        rng.shuffle(arr)

    picked = []

    # L4: stratify by grade
    g_pool = defaultdict(list)
    for it in by_task.get("L4_dr_grading_v4", []):
        g_pool[it["meta"]["dr_grade"]].append(it)
    for g in range(5):
        picked.extend(g_pool.get(g, [])[:PER_GRADE_L4])

    # L3 per lesion: pos + neg balanced
    for lesion in ("HE", "EX", "MA", "SE", "IRMA"):
        pool = by_task.get(f"L3_{lesion}", [])
        pos = [it for it in pool if it["meta"]["present_state"] == "present"]
        neg = [it for it in pool if it["meta"]["present_state"] == "absent"]
        picked.extend(pos[:PER_LESION_PN])
        picked.extend(neg[:PER_LESION_PN])

    # L3_NV: take all positives + fill to NV_TOTAL with negatives
    nv_pool = by_task.get("L3_NV", [])
    nv_pos = [it for it in nv_pool if it["meta"]["present_state"] == "present"]
    nv_neg = [it for it in nv_pool if it["meta"]["present_state"] == "absent"]
    picked.extend(nv_pos)
    fill = max(0, NV_TOTAL - len(nv_pos))
    picked.extend(nv_neg[:fill])

    # L2_laterality
    pool = by_task.get("L2_laterality", [])
    left = [it for it in pool if it["meta"].get("eye_side") == "left"]
    right = [it for it in pool if it["meta"].get("eye_side") == "right"]
    picked.extend(left[:LATERALITY_HALF])
    picked.extend(right[:LATERALITY_HALF])

    # L2_cdr stratified by bucket
    pool = by_task.get("L2_cdr", [])
    by_bucket = defaultdict(list)
    for it in pool:
        by_bucket[it["meta"].get("cdr_bucket")].append(it)
    for b in ("normal", "mild_elevation", "moderate_elevation", "glaucoma_suspicion"):
        picked.extend(by_bucket.get(b, [])[:CDR_PER_BUCKET])

    # L2_vessel mixed valid + abstain
    pool = by_task.get("L2_vessel", [])
    valid = [it for it in pool if it["meta"].get("vessel_state") == "valid"]
    abstain = [it for it in pool if it["meta"].get("vessel_state") == "abstain"]
    picked.extend(valid[:VESSEL_PER_STATE])
    picked.extend(abstain[:VESSEL_PER_STATE])

    rng.shuffle(picked)

    with OUT.open("w") as f:
        for it in picked:
            f.write(json.dumps(it, ensure_ascii=False, separators=(",", ":")) + "\n")

    from collections import Counter
    tasks = Counter(it["meta"]["task"] for it in picked)
    print(f"wrote {len(picked)} samples → {OUT}")
    for t, n in sorted(tasks.items()):
        print(f"  {t:<25} {n:>3}")


if __name__ == "__main__":
    main()
