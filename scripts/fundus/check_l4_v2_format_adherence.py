#!/usr/bin/env python3
"""Check how well predictions follow the v2 CoT structure.

Reports:
- per-section presence rate (【逐项核查】/【证据强度归类】/【病灶负担】/【分级路径】/【结论】/【JSON】)
- JSON block parseable rate
- DR grade extractable rate (via "DR Grade X" / "dr_grade":X / fallback)
- per-grade confusion vs label
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path


SECTIONS = ["【逐项核查】", "【证据强度归类】", "【病灶负担】", "【分级路径】", "【结论】", "【JSON】"]


def extract_grade(text: str) -> int | None:
    m = re.search(r"DR\s*Grade\s*([0-4])", text, re.IGNORECASE)
    if m:
        return int(m.group(1))
    m = re.search(r'"dr_grade"\s*:\s*([0-4])', text)
    if m:
        return int(m.group(1))
    m = re.search(r"\bGrade\s*([0-4])\b", text, re.IGNORECASE)
    if m:
        return int(m.group(1))
    return None


def extract_json_block(text: str) -> dict | None:
    # Try fenced ```json ... ``` first
    m = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            pass
    # Try after 【JSON】 marker
    m = re.search(r"【JSON】\s*(\{.*?\})\s*(?:$|\n\n)", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            pass
    # Fallback: any {...} that contains dr_grade
    for m in re.finditer(r"\{[^{}]*\"dr_grade\"[^{}]*\}", text):
        try:
            return json.loads(m.group(0))
        except Exception:
            continue
    return None


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("jsonl_path", type=Path)
    p.add_argument("--json-out", type=Path, default=None)
    args = p.parse_args()

    rows = [json.loads(l) for l in args.jsonl_path.read_text().splitlines() if l.strip()]
    n = len(rows)

    section_hits = Counter()
    json_parsed = 0
    grade_pred_extracted = 0
    grade_label_extracted = 0
    confusion = Counter()
    pred_distribution = Counter()
    label_distribution = Counter()
    parse_examples_first_failures = []

    for r in rows:
        pred = r.get("predict", "") or ""
        label = r.get("label", "") or ""
        for s in SECTIONS:
            if s in pred:
                section_hits[s] += 1
        if extract_json_block(pred) is not None:
            json_parsed += 1
        gp = extract_grade(pred)
        gl = extract_grade(label)
        if gp is not None:
            grade_pred_extracted += 1
            pred_distribution[gp] += 1
        else:
            if len(parse_examples_first_failures) < 3:
                parse_examples_first_failures.append(pred[:300])
        if gl is not None:
            grade_label_extracted += 1
            label_distribution[gl] += 1
        if gp is not None and gl is not None:
            confusion[(gl, gp)] += 1

    out = {
        "n": n,
        "section_presence_rate": {s: section_hits[s] / n for s in SECTIONS},
        "json_parse_rate": json_parsed / n,
        "grade_pred_parse_rate": grade_pred_extracted / n,
        "grade_label_parse_rate": grade_label_extracted / n,
        "pred_grade_distribution": dict(sorted(pred_distribution.items())),
        "label_grade_distribution": dict(sorted(label_distribution.items())),
        "confusion": {f"label={l}_pred={p}": c for (l, p), c in sorted(confusion.items())},
        "first_parse_failure_examples": parse_examples_first_failures,
    }
    text = json.dumps(out, indent=2, ensure_ascii=False)
    print(text)
    if args.json_out:
        args.json_out.write_text(text + "\n")


if __name__ == "__main__":
    main()
