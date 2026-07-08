#!/usr/bin/env python3
"""Compute DR grade metrics from LLaMA-Factory generated_predictions.jsonl (predict/label fields).

Matches logic in src/llamafactory/train/sft/metric.py (ComputeSimilarity._compute_grade_metrics).
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Optional

import numpy as np


def _extract_grade(text: str) -> Optional[int]:
    # 1) JSON-style "dr_grade": N (most robust for stage2_grade_en_v2 output)
    j = re.search(r'"?dr_grade"?\s*[:：]\s*([0-4])', text)
    if j is not None:
        return int(j.group(1))
    # 2) "GRADE: N" / "DR Grade N"
    explicit = re.search(r"GRADE\s*[:：]?\s*([0-4])\b", text, flags=re.IGNORECASE)
    if explicit is not None:
        return int(explicit.group(1))
    # 3) last-resort single digit
    fallback = re.search(r"\b([0-4])\b", text)
    if fallback is not None:
        return int(fallback.group(1))
    return None


def compute_metrics(grade_preds: list[int], grade_labels: list[int], total_examples: int) -> dict[str, float]:
    metrics: dict[str, float] = {}
    parsed = len(grade_labels)
    metrics["grade_parse_rate"] = float(parsed / total_examples) if total_examples > 0 else 0.0
    if parsed == 0:
        return metrics

    num_classes = 5
    confusion = np.zeros((num_classes, num_classes), dtype=np.float64)
    for label, pred in zip(grade_labels, grade_preds):
        confusion[label, pred] += 1.0

    total = float(confusion.sum())
    diagonal = np.diag(confusion)
    metrics["grade_acc"] = float(diagonal.sum() / total) if total > 0 else 0.0

    recalls = []
    f1_scores = []
    for grade in range(num_classes):
        tp = float(confusion[grade, grade])
        fn = float(confusion[grade, :].sum() - tp)
        fp = float(confusion[:, grade].sum() - tp)
        recall = tp / (tp + fn) if tp + fn > 0 else 0.0
        precision = tp / (tp + fp) if tp + fp > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
        metrics[f"recall_grade_{grade}"] = recall
        recalls.append(recall)
        f1_scores.append(f1)

    metrics["macro_recall"] = float(np.mean(recalls))
    metrics["macro_f1"] = float(np.mean(f1_scores))

    weights = np.zeros((num_classes, num_classes), dtype=np.float64)
    denom = float((num_classes - 1) ** 2)
    for i in range(num_classes):
        for j in range(num_classes):
            weights[i, j] = ((i - j) ** 2) / denom

    hist_true = confusion.sum(axis=1)
    hist_pred = confusion.sum(axis=0)
    observed = confusion / total if total > 0 else np.zeros_like(confusion)
    expected = np.outer(hist_true, hist_pred) / (total**2) if total > 0 else np.zeros_like(confusion)
    observed_weighted = float(np.sum(weights * observed))
    expected_weighted = float(np.sum(weights * expected))
    metrics["qwk"] = 1.0 - observed_weighted / expected_weighted if expected_weighted > 0 else 0.0

    # within-1 (off-by-one) accuracy — ordinal tolerance
    within1 = 0.0
    for i in range(num_classes):
        for j in range(num_classes):
            if abs(i - j) <= 1:
                within1 += confusion[i, j]
    metrics["grade_within1_acc"] = float(within1 / total) if total > 0 else 0.0

    # referable DR (grade >= 2) binary metrics — clinical triage primary metric
    ref = 2  # referable threshold
    tp = float(confusion[ref:, ref:].sum())
    fn = float(confusion[ref:, :ref].sum())
    fp = float(confusion[:ref, ref:].sum())
    tn = float(confusion[:ref, :ref].sum())
    sens = tp / (tp + fn) if tp + fn > 0 else 0.0
    spec = tn / (tn + fp) if tn + fp > 0 else 0.0
    ppv = tp / (tp + fp) if tp + fp > 0 else 0.0
    npv = tn / (tn + fn) if tn + fn > 0 else 0.0
    ref_f1 = 2 * ppv * sens / (ppv + sens) if ppv + sens > 0 else 0.0
    metrics["referable_sensitivity"] = sens
    metrics["referable_specificity"] = spec
    metrics["referable_ppv"] = ppv
    metrics["referable_npv"] = npv
    metrics["referable_f1"] = ref_f1
    metrics["referable_balanced_acc"] = float((sens + spec) / 2)
    metrics["referable_tp"] = tp
    metrics["referable_fn"] = fn
    metrics["referable_fp"] = fp
    metrics["referable_tn"] = tn

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Grade metrics from generated_predictions.jsonl")
    parser.add_argument("jsonl_path", type=Path, help="Path to generated_predictions.jsonl")
    parser.add_argument("--json-out", type=Path, default=None, help="Optional path to write metrics JSON")
    args = parser.parse_args()

    lines = args.jsonl_path.read_text(encoding="utf-8").strip().splitlines()
    total_examples = len(lines)
    grade_preds: list[int] = []
    grade_labels: list[int] = []
    parse_fail_pred = 0
    parse_fail_label = 0

    for line in lines:
        row = json.loads(line)
        pred = row.get("predict", "")
        label = row.get("label", "")
        pg = _extract_grade(pred)
        lg = _extract_grade(label)
        if pg is None:
            parse_fail_pred += 1
        if lg is None:
            parse_fail_label += 1
        if pg is not None and lg is not None:
            grade_preds.append(pg)
            grade_labels.append(lg)

    metrics = compute_metrics(grade_preds, grade_labels, total_examples)
    metrics["n_examples"] = float(total_examples)
    metrics["n_pairs_used"] = float(len(grade_labels))
    metrics["parse_fail_pred"] = float(parse_fail_pred)
    metrics["parse_fail_label"] = float(parse_fail_label)

    out = json.dumps(metrics, indent=2, ensure_ascii=False)
    print(out)
    if args.json_out:
        args.json_out.write_text(out + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
