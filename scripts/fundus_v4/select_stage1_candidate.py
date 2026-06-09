#!/usr/bin/env python3
"""Select a replay-calibration checkpoint using gold-dev F1 guardrails."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


LESIONS = ("MA", "HE", "EX", "SE")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--metrics-root", type=Path, required=True)
    parser.add_argument("--adapter-root", type=Path, required=True)
    parser.add_argument("--json-out", type=Path, required=True)
    parser.add_argument("--macro-f1-drop", type=float, default=0.02)
    parser.add_argument("--lesion-f1-drop", type=float, default=0.05)
    args = parser.parse_args()

    baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
    candidates = []
    for path in sorted(args.metrics_root.glob("*/stage1_metrics.json")):
        name = path.parent.name
        metrics = json.loads(path.read_text(encoding="utf-8"))
        checks = {
            "macro_f1": metrics["main4_macro"]["f1"]
            >= baseline["main4_macro"]["f1"] - args.macro_f1_drop
        }
        for lesion in LESIONS:
            checks[f"{lesion}_f1"] = (
                metrics["by_lesion"][lesion]["f1"]
                >= baseline["by_lesion"][lesion]["f1"] - args.lesion_f1_drop
            )
        adapter = args.adapter_root if name == "final" else args.adapter_root / name
        candidates.append(
            {
                "name": name,
                "adapter": str(adapter),
                "passed": all(checks.values()),
                "checks": checks,
                "metrics": metrics,
            }
        )

    passed = [item for item in candidates if item["passed"]]
    if not passed:
        result = {"selected": None, "candidates": candidates}
        args.json_out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 1

    selected = max(
        passed,
        key=lambda item: (
            item["metrics"]["main4_macro"]["balanced_accuracy"],
            item["metrics"]["main4_macro"]["f1"],
            item["metrics"]["main4_macro"]["specificity"],
        ),
    )
    result = {"selected": selected, "candidates": candidates}
    args.json_out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
