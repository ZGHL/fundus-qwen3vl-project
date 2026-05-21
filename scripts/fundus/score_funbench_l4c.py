#!/usr/bin/env python3
"""Score v3 predictions against FunBench L4c (Scheme B).

Pipeline:
  1. Read generated_predictions.jsonl from v3 inference
  2. For each prediction, extract dr_grade from JSON (0-4)
  3. Read meta json which has FunBench's options (shuffled per entry) + gt_letter
  4. Map predicted grade → ICDR text → option letter for that entry
  5. Compare predicted_letter vs gt_letter
  6. Report:
     - exact letter accuracy (= grade accuracy under fixed mapping)
     - per-grade F1 (FunBench's main metric)
     - harmonic mean F1 across grades (FunBench L4 score)
     - QWK
     - confusion matrix
     - per-dataset breakdown
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.metrics import cohen_kappa_score, f1_score

ICDR_TEXT_TO_GRADE = {
    "No any diabetic retinopathy": 0,
    "Mild nonproliferative diabetic retinopathy": 1,
    "Moderate nonproliferative diabetic retinopathy": 2,
    "Severe nonproliferative diabetic retinopathy": 3,
    "Proliferative diabetic retinopathy": 4,
}
GRADE_TO_ICDR_TEXT = {v: k for k, v in ICDR_TEXT_TO_GRADE.items()}


def extract_grade(text: str) -> int | None:
    m = re.search(r'"dr_grade"\s*:\s*([0-4])', text)
    if m:
        return int(m.group(1))
    if "【结论】" in text:
        tail = text.split("【结论】", 1)[1]
        m = re.search(r"DR\s*Grade\s*([0-4])", tail, re.IGNORECASE)
        if m:
            return int(m.group(1))
    m = re.search(r"DR\s*Grade\s*([0-4])", text, re.IGNORECASE)
    return int(m.group(1)) if m else None


def grade_to_letter(grade: int, options: list[str]) -> str | None:
    """Find which option letter (A,B,C,...) corresponds to this grade."""
    target_text = GRADE_TO_ICDR_TEXT[grade]
    for i, opt in enumerate(options):
        if opt == target_text:
            return chr(ord("A") + i)
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", type=Path,
                    default=Path("saves/qwen3-vl-8b-fundus/lora/l4_unified_lesion_cot_v3_predict_funbench_l4c/generated_predictions.jsonl"))
    ap.add_argument("--meta", type=Path,
                    default=Path("data/annotation/fundus_funbench_l4c_eval_meta.json"))
    ap.add_argument("--out", type=Path, default=Path("/tmp/funbench_l4c_score.json"))
    args = ap.parse_args()

    preds = [json.loads(l) for l in args.predictions.read_text().splitlines() if l.strip()]
    meta = json.load(args.meta.open())
    if len(preds) != len(meta):
        print(f"⚠️  count mismatch: predictions={len(preds)}, meta={len(meta)}")
        # try to align by truncation
        n = min(len(preds), len(meta))
        preds = preds[:n]; meta = meta[:n]

    n = len(preds)
    print(f"scoring n={n}")

    parse_fail = 0
    letter_correct = 0
    grade_pairs: list[tuple[int, int]] = []  # (gt_grade, pred_grade)
    confusion = Counter()
    by_dataset_correct: dict[str, int] = Counter()
    by_dataset_total: dict[str, int] = Counter()
    failure_examples = []

    for pred_entry, meta_entry in zip(preds, meta):
        pred_text = pred_entry.get("predict", "")
        pred_grade = extract_grade(pred_text)
        if pred_grade is None:
            parse_fail += 1
            if len(failure_examples) < 3:
                failure_examples.append(pred_text[:300])
            continue

        gt_grade = meta_entry["gt_grade"]
        gt_letter = meta_entry["gt_letter"]
        options = meta_entry["options"]
        dataset = meta_entry["dataset"]

        pred_letter = grade_to_letter(pred_grade, options)
        is_correct = (pred_letter == gt_letter)
        letter_correct += int(is_correct)
        by_dataset_total[dataset] += 1
        by_dataset_correct[dataset] += int(is_correct)

        grade_pairs.append((gt_grade, pred_grade))
        confusion[(gt_grade, pred_grade)] += 1

    out: dict = {
        "n": n,
        "parse_fail": parse_fail,
        "letter_accuracy": letter_correct / n if n else 0,
    }

    if grade_pairs:
        gts = np.array([p[0] for p in grade_pairs])
        prs = np.array([p[1] for p in grade_pairs])
        out["grade_accuracy"] = float((gts == prs).mean())
        out["qwk"] = float(cohen_kappa_score(gts, prs, weights="quadratic"))
        out["macro_f1"] = float(f1_score(gts, prs, average="macro", zero_division=0))
        # Per-grade F1
        per_class_f1 = f1_score(gts, prs, average=None, labels=[0,1,2,3,4], zero_division=0)
        out["per_grade_f1"] = {f"G{g}": float(per_class_f1[g]) for g in range(5)}
        # FunBench's "harmonic mean F1" — across classes (paper definition)
        non_zero = [v for v in per_class_f1 if v > 0]
        if non_zero:
            harmonic = len(non_zero) / sum(1.0/v for v in non_zero)
            out["harmonic_f1_FunBench_style"] = float(harmonic)
        else:
            out["harmonic_f1_FunBench_style"] = 0.0
        out["per_grade_recall"] = {
            f"G{g}": float((prs[gts==g] == g).mean()) if (gts==g).sum() > 0 else None
            for g in range(5)
        }
        out["per_grade_support"] = {f"G{g}": int((gts==g).sum()) for g in range(5)}
        out["confusion"] = {f"L{l}_P{p}": int(c) for (l,p), c in sorted(confusion.items())}
        # Pred distribution
        out["pred_grade_dist"] = {f"P{g}": int((prs==g).sum()) for g in range(5)}

    # By dataset
    out["by_dataset"] = {
        ds: {"correct": by_dataset_correct[ds], "total": by_dataset_total[ds],
             "accuracy": by_dataset_correct[ds]/by_dataset_total[ds] if by_dataset_total[ds] else 0}
        for ds in by_dataset_total
    }

    if failure_examples:
        out["parse_failure_examples"] = failure_examples

    args.out.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
