#!/usr/bin/env python3
"""Select a targeted calibration checkpoint with strict preservation guardrails."""
from __future__ import annotations
import argparse, json
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", type=Path, required=True)
    ap.add_argument("--metrics-root", type=Path, required=True)
    ap.add_argument("--adapter-root", type=Path, required=True)
    ap.add_argument("--json-out", type=Path, required=True)
    args = ap.parse_args()
    baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
    candidates = []
    for path in sorted(args.metrics_root.glob("*/stage1_metrics.json")):
        name = path.parent.name
        metrics = json.loads(path.read_text(encoding="utf-8"))
        b = baseline["by_lesion"]; m = metrics["by_lesion"]
        checks = {
            "macro_f1": metrics["main4_macro"]["f1"] >= baseline["main4_macro"]["f1"] - 0.01,
            "HE_preserved": m["HE"]["f1"] >= b["HE"]["f1"] - 0.02,
            "EX_preserved": m["EX"]["f1"] >= b["EX"]["f1"] - 0.02,
            "MA_non_decreasing": m["MA"]["f1"] >= b["MA"]["f1"],
            "SE_non_decreasing": m["SE"]["f1"] >= b["SE"]["f1"],
            "format": metrics["json_parse_success"] >= 0.99,
        }
        adapter = args.adapter_root if name == "final" else args.adapter_root / name
        candidates.append({"name": name, "adapter": str(adapter), "passed": all(checks.values()), "checks": checks, "metrics": metrics})
    passed = [item for item in candidates if item["passed"]]
    selected = max(passed, key=lambda item: (
        min(item["metrics"]["by_lesion"]["MA"]["f1"] - baseline["by_lesion"]["MA"]["f1"],
            item["metrics"]["by_lesion"]["SE"]["f1"] - baseline["by_lesion"]["SE"]["f1"]),
        item["metrics"]["main4_macro"]["f1"],
        item["metrics"]["main4_macro"]["balanced_accuracy"],
    )) if passed else None
    result = {"selected": selected, "candidates": candidates}
    args.json_out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if selected else 2


if __name__ == "__main__":
    raise SystemExit(main())
