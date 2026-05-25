#!/usr/bin/env python3
"""Score decoupled lesion-perception predictions.

Input is LLaMA-Factory `generated_predictions.jsonl`, where each row contains
`predict` and `label`. Metrics use only the JSON object in the structured
output section.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

LESIONS = ("HE", "EX", "MA", "SE", "IRMA", "NV")
GRADE_PAT = re.compile(r"\b(DR\s*grade|dr_grade|final\s+grade|grade\s*[0-4])\b", re.I)


def extract_json(text: str) -> dict[str, Any] | None:
    if not text:
        return None
    tail = text.split("[Structured Output]", 1)[-1]
    match = re.search(r"\{.*?\}", tail, flags=re.S)
    if match is None:
        match = re.search(r"\{.*?\}", text, flags=re.S)
    if match is None:
        return None
    try:
        obj = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


def bool_present(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        v = value.strip().lower()
        if v in {"true", "present", "positive", "yes", "1"}:
            return True
        if v in {"false", "absent", "negative", "no", "0"}:
            return False
    return None


def div(num: int | float, den: int | float) -> float | None:
    return float(num / den) if den else None


def f1(precision: float | None, recall: float | None) -> float | None:
    if precision is None or recall is None or precision + recall == 0:
        return None
    return 2 * precision * recall / (precision + recall)


def rate(num: int, den: int) -> float:
    return float(num / den) if den else 0.0


def metric_block(c: Counter) -> dict[str, Any]:
    tp, fp, fn, tn = c["tp"], c["fp"], c["fn"], c["tn"]
    precision = div(tp, tp + fp)
    recall = div(tp, tp + fn)
    specificity = div(tn, tn + fp)
    accuracy = div(tp + tn, tp + fp + fn + tn)
    balanced_accuracy = None
    if recall is not None and specificity is not None:
        balanced_accuracy = (recall + specificity) / 2
    return {
        "n": tp + fp + fn + tn,
        "positive": tp + fn,
        "negative": tn + fp,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "sensitivity": recall,
        "specificity": specificity,
        "f1": f1(precision, recall),
        "balanced_accuracy": balanced_accuracy,
    }


def mean_defined(values: list[float | None]) -> float | None:
    vals = [v for v in values if v is not None]
    return sum(vals) / len(vals) if vals else None


def score(rows: list[dict[str, Any]]) -> dict[str, Any]:
    totals = Counter()
    by_lesion: dict[str, Counter] = defaultdict(Counter)
    parse_errors = []

    for idx, row in enumerate(rows):
        pred_text = row.get("predict", "")
        label_text = row.get("label", "")
        pred = extract_json(pred_text)
        label = extract_json(label_text)

        totals["n"] += 1
        totals["pred_json_ok"] += int(pred is not None)
        totals["label_json_ok"] += int(label is not None)
        totals["no_grade_output"] += int(not GRADE_PAT.search(pred_text))
        if pred is None or label is None:
            parse_errors.append({"row": idx, "reason": "json_parse_failed"})
            continue

        label_lesion = str(label.get("lesion") or "").upper()
        pred_lesion = str(pred.get("lesion") or "").upper()
        label_present = bool_present(label.get("present"))
        pred_present = bool_present(pred.get("present"))

        if label_lesion in LESIONS:
            by_lesion[label_lesion]["n_label"] += 1
        totals["target_lesion_match"] += int(pred_lesion == label_lesion)
        if label_lesion not in LESIONS or label_present is None:
            parse_errors.append({"row": idx, "reason": "invalid_label_json"})
            continue
        if pred_present is None:
            by_lesion[label_lesion]["fn" if label_present else "tn"] += 1
            parse_errors.append({"row": idx, "reason": "invalid_pred_present"})
            continue

        if pred_lesion != label_lesion:
            if label_present:
                by_lesion[label_lesion]["fn"] += 1
            else:
                by_lesion[label_lesion]["tn"] += 1
            continue

        if pred_present and label_present:
            by_lesion[label_lesion]["tp"] += 1
        elif pred_present and not label_present:
            by_lesion[label_lesion]["fp"] += 1
        elif not pred_present and label_present:
            by_lesion[label_lesion]["fn"] += 1
        else:
            by_lesion[label_lesion]["tn"] += 1

    lesion_metrics = {lesion: metric_block(by_lesion[lesion]) for lesion in LESIONS if by_lesion[lesion]["n_label"]}
    micro_counts = Counter()
    for c in by_lesion.values():
        for key in ("tp", "fp", "fn", "tn"):
            micro_counts[key] += c[key]
    macro = {
        "accuracy": mean_defined([m["accuracy"] for m in lesion_metrics.values()]),
        "precision": mean_defined([m["precision"] for m in lesion_metrics.values()]),
        "recall": mean_defined([m["recall"] for m in lesion_metrics.values()]),
        "specificity": mean_defined([m["specificity"] for m in lesion_metrics.values()]),
        "f1": mean_defined([m["f1"] for m in lesion_metrics.values()]),
        "balanced_accuracy": mean_defined([m["balanced_accuracy"] for m in lesion_metrics.values()]),
    }
    rare = {
        "f1": mean_defined([lesion_metrics.get(k, {}).get("f1") for k in ("IRMA", "NV")]),
        "recall": mean_defined([lesion_metrics.get(k, {}).get("recall") for k in ("IRMA", "NV")]),
        "balanced_accuracy": mean_defined([lesion_metrics.get(k, {}).get("balanced_accuracy") for k in ("IRMA", "NV")]),
    }
    return {
        "n": totals["n"],
        "json_parse_success": rate(totals["pred_json_ok"], totals["n"]),
        "label_json_parse_success": rate(totals["label_json_ok"], totals["n"]),
        "target_lesion_consistency": rate(totals["target_lesion_match"], totals["label_json_ok"]),
        "no_grade_output_rate": rate(totals["no_grade_output"], totals["n"]),
        "micro": metric_block(micro_counts),
        "macro": macro,
        "rare_lesion_macro": rare,
        "by_lesion": lesion_metrics,
        "parse_errors_sample": parse_errors[:20],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("predictions", type=Path)
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()

    rows = [json.loads(line) for line in args.predictions.read_text(encoding="utf-8").splitlines() if line.strip()]
    out = score(rows)
    text = json.dumps(out, ensure_ascii=False, indent=2)
    print(text)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
