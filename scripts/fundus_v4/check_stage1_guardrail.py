#!/usr/bin/env python3
"""Check a Stage1 candidate against baseline gold-dev F1 guardrails."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


LESIONS = ("MA", "HE", "EX", "SE")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--macro-f1-drop", type=float, default=0.02)
    parser.add_argument("--lesion-f1-drop", type=float, default=0.05)
    args = parser.parse_args()
    baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
    candidate = json.loads(args.candidate.read_text(encoding="utf-8"))
    checks = {}
    checks["macro_f1"] = (
        candidate["main4_macro"]["f1"]
        >= baseline["main4_macro"]["f1"] - args.macro_f1_drop
    )
    for lesion in LESIONS:
        checks[f"{lesion}_f1"] = (
            candidate["by_lesion"][lesion]["f1"]
            >= baseline["by_lesion"][lesion]["f1"] - args.lesion_f1_drop
        )
    result = {
        "passed": all(checks.values()),
        "checks": checks,
        "baseline_macro": baseline["main4_macro"],
        "candidate_macro": candidate["main4_macro"],
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
