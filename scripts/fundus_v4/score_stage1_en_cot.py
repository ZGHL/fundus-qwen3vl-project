#!/usr/bin/env python3
"""Score Stage1 English single-lesion CoT generated predictions."""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


LESIONS = ("MA", "HE", "EX", "SE", "IRMA", "NV")


def extract(text: str) -> dict[str, Any] | None:
    marker = "[Structured Output]"
    tail = text.split(marker, 1)[-1]
    match = re.search(r"\{.*\}", tail, flags=re.S)
    if not match:
        return None
    try:
        obj = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


def target(obj: dict[str, Any] | None) -> str | None:
    if not obj:
        return None
    value = obj.get("target_lesion")
    if isinstance(value, dict):
        value = value.get("abbreviation")
    value = str(value or obj.get("lesion") or "").upper()
    return value if value in LESIONS else None


def present(obj: dict[str, Any] | None) -> bool | None:
    if not obj:
        return None
    value = obj.get("present")
    return value if isinstance(value, bool) else None


def metrics(c: Counter) -> dict[str, Any]:
    tp, fp, fn, tn = (c[k] for k in ("tp", "fp", "fn", "tn"))
    div = lambda a, b: a / b if b else None
    precision = div(tp, tp + fp)
    recall = div(tp, tp + fn)
    specificity = div(tn, tn + fp)
    f1 = div(2 * tp, 2 * tp + fp + fn)
    bal = (recall + specificity) / 2 if recall is not None and specificity is not None else None
    return {
        "n": tp + fp + fn + tn, "positive": tp + fn, "negative": tn + fp,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": precision, "recall": recall, "specificity": specificity,
        "f1": f1, "balanced_accuracy": bal,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("predictions", type=Path)
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()
    rows = [json.loads(x) for x in args.predictions.read_text(encoding="utf-8").splitlines() if x.strip()]
    counts: dict[str, Counter] = defaultdict(Counter)
    totals = Counter()
    for row in rows:
        pred_text, label_text = str(row.get("predict", "")), str(row.get("label", ""))
        pred, label = extract(pred_text), extract(label_text)
        lesion, gold, guess = target(label), present(label), present(pred)
        totals["n"] += 1
        totals["json_ok"] += pred is not None
        totals["target_ok"] += pred is not None and target(pred) == lesion
        totals["no_grade_output"] += not bool(re.search(r"\b(?:DR\s*)?grade\s*[0-4]\b", pred_text, re.I))
        if lesion is None or gold is None:
            totals["invalid_label"] += 1
            continue
        # Parse failure, missing present, and wrong target are all scored as errors.
        if guess is None or target(pred) != lesion:
            counts[lesion]["fn" if gold else "fp"] += 1
        elif guess and gold:
            counts[lesion]["tp"] += 1
        elif guess and not gold:
            counts[lesion]["fp"] += 1
        elif not guess and gold:
            counts[lesion]["fn"] += 1
        else:
            counts[lesion]["tn"] += 1
    by = {k: metrics(counts[k]) for k in LESIONS if sum(counts[k].values())}
    main4 = [by[k] for k in ("MA", "HE", "EX", "SE") if k in by]
    mean = lambda key: sum(x[key] for x in main4 if x[key] is not None) / sum(x[key] is not None for x in main4) if any(x[key] is not None for x in main4) else None
    out = {
        "n": totals["n"],
        "json_parse_success": totals["json_ok"] / totals["n"] if totals["n"] else None,
        "target_consistency": totals["target_ok"] / totals["n"] if totals["n"] else None,
        "no_grade_output_rate": totals["no_grade_output"] / totals["n"] if totals["n"] else None,
        "main4_macro": {"f1": mean("f1"), "recall": mean("recall"), "specificity": mean("specificity"), "balanced_accuracy": mean("balanced_accuracy")},
        "by_lesion": by,
    }
    text = json.dumps(out, ensure_ascii=False, indent=2)
    print(text)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
