#!/usr/bin/env python3
"""Score L3 joint/mix lesion-perception predictions."""
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
    start = tail.find("{")
    if start < 0:
        start = text.find("{")
        tail = text
    if start < 0:
        return None
    depth = 0
    in_string = False
    escape = False
    for idx in range(start, len(tail)):
        ch = tail[idx]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    obj = json.loads(tail[start : idx + 1])
                except json.JSONDecodeError:
                    return None
                return obj if isinstance(obj, dict) else None
    return None


def bool_present(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        value = value.strip().lower()
        if value in {"true", "present", "positive", "yes", "1"}:
            return True
        if value in {"false", "absent", "negative", "no", "0"}:
            return False
    return None


def lesion_map(obj: dict[str, Any] | None) -> dict[str, bool | None]:
    if not obj:
        return {}
    lesions_obj = obj.get("lesions")
    if isinstance(lesions_obj, dict):
        out = {}
        for lesion, item in lesions_obj.items():
            key = str(lesion).upper()
            if key not in LESIONS:
                continue
            if isinstance(item, dict):
                out[key] = bool_present(item.get("present"))
            else:
                out[key] = bool_present(item)
        return out
    lesion = str(obj.get("lesion") or "").upper()
    if lesion in LESIONS:
        return {lesion: bool_present(obj.get("present"))}
    return {}


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
    balanced_accuracy = (recall + specificity) / 2 if recall is not None and specificity is not None else None
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
    errors = []
    for idx, row in enumerate(rows):
        pred_text = row.get("predict", "")
        label_text = row.get("label", "")
        pred_obj = extract_json(pred_text)
        label_obj = extract_json(label_text)
        pred = lesion_map(pred_obj)
        label = lesion_map(label_obj)
        totals["n"] += 1
        totals["pred_json_ok"] += int(pred_obj is not None)
        totals["label_json_ok"] += int(label_obj is not None)
        totals["no_grade_output"] += int(not GRADE_PAT.search(pred_text))
        if not label:
            errors.append({"row": idx, "reason": "missing_label_lesions"})
            continue
        for lesion, label_present in label.items():
            if label_present is None:
                errors.append({"row": idx, "lesion": lesion, "reason": "invalid_label_present"})
                continue
            pred_present = pred.get(lesion)
            totals["target_lesion_match"] += int(lesion in pred)
            by_lesion[lesion]["n_label"] += 1
            if pred_present is None:
                if label_present:
                    by_lesion[lesion]["fn"] += 1
                else:
                    by_lesion[lesion]["tn"] += 1
                errors.append({"row": idx, "lesion": lesion, "reason": "missing_or_invalid_prediction"})
                continue
            if pred_present and label_present:
                by_lesion[lesion]["tp"] += 1
            elif pred_present and not label_present:
                by_lesion[lesion]["fp"] += 1
            elif not pred_present and label_present:
                by_lesion[lesion]["fn"] += 1
            else:
                by_lesion[lesion]["tn"] += 1
    lesion_metrics = {lesion: metric_block(by_lesion[lesion]) for lesion in LESIONS if by_lesion[lesion]["n_label"]}
    micro = Counter()
    for counts in by_lesion.values():
        for key in ("tp", "fp", "fn", "tn"):
            micro[key] += counts[key]
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
    total_labels = sum(c["n_label"] for c in by_lesion.values())
    return {
        "n": totals["n"],
        "label_decisions": total_labels,
        "json_parse_success": rate(totals["pred_json_ok"], totals["n"]),
        "label_json_parse_success": rate(totals["label_json_ok"], totals["n"]),
        "target_lesion_consistency": rate(totals["target_lesion_match"], total_labels),
        "no_grade_output_rate": rate(totals["no_grade_output"], totals["n"]),
        "micro": metric_block(micro),
        "macro": macro,
        "rare_lesion_macro": rare,
        "by_lesion": lesion_metrics,
        "parse_errors_sample": errors[:20],
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
