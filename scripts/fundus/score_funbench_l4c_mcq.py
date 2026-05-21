#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.metrics import cohen_kappa_score, f1_score


TEXT_TO_GRADE = {
    "No any diabetic retinopathy": 0,
    "Mild nonproliferative diabetic retinopathy": 1,
    "Moderate nonproliferative diabetic retinopathy": 2,
    "Severe nonproliferative diabetic retinopathy": 3,
    "Proliferative diabetic retinopathy": 4,
}


def extract_letter(text: str) -> str | None:
    if not text:
        return None
    s = text.strip()
    m = re.search(r"\b([A-E])\b", s, re.I)
    if m:
        return m.group(1).upper()
    m = re.search(r"(?:answer|option|答案|选项)\s*[:：]?\s*([A-E])", s, re.I)
    if m:
        return m.group(1).upper()
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", type=Path, required=True)
    ap.add_argument("--meta", type=Path, default=Path("data/annotation/fundus_funbench_l4c_eval_meta.json"))
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    preds = [json.loads(l) for l in args.predictions.read_text(encoding="utf-8").splitlines() if l.strip()]
    meta = json.load(args.meta.open(encoding="utf-8"))
    n = min(len(preds), len(meta))
    preds, meta = preds[:n], meta[:n]

    y_true: list[int] = []
    y_pred: list[int] = []
    letter_correct = 0
    parse_fail = 0
    pred_letters = Counter()
    by_dataset_total = Counter()
    by_dataset_correct = Counter()
    failures = []

    for p, m in zip(preds, meta):
        gt_letter = m["gt_letter"]
        gt_grade = int(m["gt_grade"])
        pr_letter = extract_letter(p.get("predict", ""))
        if pr_letter is None:
            parse_fail += 1
            if len(failures) < 5:
                failures.append(p.get("predict", "")[:300])
            continue
        pred_letters[pr_letter] += 1
        options = m["options"]
        pr_idx = ord(pr_letter) - 65
        if not 0 <= pr_idx < len(options):
            parse_fail += 1
            continue
        pr_grade = TEXT_TO_GRADE.get(options[pr_idx])
        if pr_grade is None:
            parse_fail += 1
            continue
        ok = pr_letter == gt_letter
        letter_correct += int(ok)
        by_dataset_total[m["dataset"]] += 1
        by_dataset_correct[m["dataset"]] += int(ok)
        y_true.append(gt_grade)
        y_pred.append(pr_grade)

    out = {
        "n_rows": len(preds),
        "n_scored": len(y_true),
        "parse_fail": parse_fail,
        "parse_rate": len(y_true) / len(preds) if preds else 0.0,
        "letter_accuracy": letter_correct / len(preds) if preds else 0.0,
        "pred_letters": dict(sorted(pred_letters.items())),
        "by_dataset": {
            k: {
                "correct": by_dataset_correct[k],
                "total": by_dataset_total[k],
                "accuracy": by_dataset_correct[k] / by_dataset_total[k] if by_dataset_total[k] else 0.0,
            }
            for k in sorted(by_dataset_total)
        },
    }
    if y_true:
        yt = np.array(y_true)
        yp = np.array(y_pred)
        per_f1 = f1_score(yt, yp, labels=[0, 1, 2, 3, 4], average=None, zero_division=0)
        nonzero = [float(x) for x in per_f1 if x > 0]
        out.update(
            {
                "grade_accuracy": float((yt == yp).mean()),
                "macro_f1": float(f1_score(yt, yp, labels=[0, 1, 2, 3, 4], average="macro", zero_division=0)),
                "qwk": float(cohen_kappa_score(yt, yp, weights="quadratic")),
                "harmonic_f1_nonzero": float(len(nonzero) / sum(1 / x for x in nonzero)) if nonzero else 0.0,
                "per_grade_f1": {str(i): float(v) for i, v in enumerate(per_f1)},
                "per_grade_recall": {
                    str(i): float((yp[yt == i] == i).mean()) if (yt == i).sum() else None for i in range(5)
                },
                "label_counts": {str(i): int((yt == i).sum()) for i in range(5)},
                "pred_counts": {str(i): int((yp == i).sum()) for i in range(5)},
                "confusion": [[int(((yt == i) & (yp == j)).sum()) for j in range(5)] for i in range(5)],
            }
        )
    if failures:
        out["parse_failure_examples"] = failures
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
