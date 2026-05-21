#!/usr/bin/env python3
"""Score fundus CoT JSON predictions from LLaMA-Factory generated_predictions.jsonl."""
from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def extract_json(text: str) -> dict[str, Any] | None:
    if not text:
        return None
    tail = text.split("【JSON】", 1)[-1] if "【JSON】" in text else text
    match = re.search(r"\{.*\}", tail, flags=re.S)
    if match is None:
        return None
    try:
        obj = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


def same_set(a: Any, b: Any) -> bool:
    return sorted(a or []) == sorted(b or [])


def as_float(x: Any) -> float | None:
    if isinstance(x, bool):
        return None
    if isinstance(x, (int, float)) and not isinstance(x, bool):
        return float(x)
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def pct(num: int | float, den: int | float) -> float:
    return float(num / den) if den else 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("predictions", type=Path)
    ap.add_argument("--json-out", type=Path, default=None)
    args = ap.parse_args()

    rows = [json.loads(line) for line in args.predictions.read_text(encoding="utf-8").splitlines() if line.strip()]
    counters = Counter()
    by_task: dict[str, Counter] = defaultdict(Counter)
    by_lesion: dict[str, Counter] = defaultdict(Counter)
    cdr_abs_errors: list[float] = []
    count_abs_errors: list[float] = []
    area_rel_errors: list[float] = []

    for row in rows:
        pred_text = row.get("predict", "")
        label_text = row.get("label", "")
        pred = extract_json(pred_text)
        label = extract_json(label_text)

        counters["total"] += 1
        counters["text_exact"] += int(pred_text.strip() == label_text.strip())
        counters["pred_json_ok"] += int(pred is not None)
        counters["label_json_ok"] += int(label is not None)
        if pred is None or label is None:
            continue

        task = str(label.get("task", "unknown"))
        by_task[task]["n"] += 1
        counters["task_match"] += int(pred.get("task") == label.get("task"))
        by_task[task]["task_match"] += int(pred.get("task") == label.get("task"))
        by_task[task]["json_exact"] += int(pred == label)

        if task == "L2_laterality":
            by_task[task]["eye_side_acc"] += int(pred.get("eye_side") == label.get("eye_side"))
        elif task == "L2_cdr":
            by_task[task]["cdr_bucket_acc"] += int(pred.get("cdr_bucket") == label.get("cdr_bucket"))
            pv, lv = as_float(pred.get("cdr")), as_float(label.get("cdr"))
            if pv is not None and lv is not None and math.isfinite(pv) and math.isfinite(lv):
                cdr_abs_errors.append(abs(pv - lv))
        elif task == "L2_vessel_metrics":
            by_task[task]["av_ratio_bucket_acc"] += int(pred.get("av_ratio_bucket") == label.get("av_ratio_bucket"))
            by_task[task]["tortuosity_bucket_acc"] += int(
                pred.get("tortuosity_bucket") == label.get("tortuosity_bucket")
            )
            by_task[task]["reason_acc"] += int(pred.get("reason") == label.get("reason"))
        elif task.startswith("L3_") and task.endswith("_single"):
            by_task[task]["lesion_acc"] += int(pred.get("lesion") == label.get("lesion"))
            by_task[task]["present_acc"] += int(pred.get("present") == label.get("present"))
            lesion = str(label.get("lesion") or task.split("_")[1])
            pred_present = pred.get("present") is True
            label_present = label.get("present") is True
            if pred.get("lesion") == label.get("lesion") or pred.get("task") == label.get("task"):
                if pred_present and label_present:
                    by_lesion[lesion]["tp"] += 1
                elif pred_present and not label_present:
                    by_lesion[lesion]["fp"] += 1
                elif not pred_present and label_present:
                    by_lesion[lesion]["fn"] += 1
                else:
                    by_lesion[lesion]["tn"] += 1
            else:
                if label_present:
                    by_lesion[lesion]["fn"] += 1
                else:
                    by_lesion[lesion]["tn"] += 1
            pc, lc = as_float(pred.get("count")), as_float(label.get("count"))
            if pc is not None and lc is not None:
                count_abs_errors.append(abs(pc - lc))
            pa, la = as_float(pred.get("area")), as_float(label.get("area"))
            if pa is not None and la is not None and la > 0:
                area_rel_errors.append(abs(pa - la) / la)
        elif task == "L3_lesion_only":
            by_task[task]["lesion_set_acc"] += int(same_set(pred.get("lesions"), label.get("lesions")))
        elif task.startswith("L4_"):
            by_task[task]["grade_acc"] += int(pred.get("dr_grade") == label.get("dr_grade"))
            by_task[task]["evidence_set_acc"] += int(same_set(pred.get("evidence"), label.get("evidence")))

    out: dict[str, Any] = {
        "n": counters["total"],
        "text_exact_rate": pct(counters["text_exact"], counters["total"]),
        "pred_json_parse_rate": pct(counters["pred_json_ok"], counters["total"]),
        "label_json_parse_rate": pct(counters["label_json_ok"], counters["total"]),
        "task_match_rate": pct(counters["task_match"], counters["label_json_ok"]),
        "cdr_mae": sum(cdr_abs_errors) / len(cdr_abs_errors) if cdr_abs_errors else None,
        "lesion_count_mae_numeric_pairs": sum(count_abs_errors) / len(count_abs_errors) if count_abs_errors else None,
        "lesion_area_mape_numeric_pairs": sum(area_rel_errors) / len(area_rel_errors) if area_rel_errors else None,
        "lesion_presence": {},
        "by_task": {},
    }
    micro = Counter()
    for lesion, c in sorted(by_lesion.items()):
        tp, fp, fn, tn = c["tp"], c["fp"], c["fn"], c["tn"]
        for key in ["tp", "fp", "fn", "tn"]:
            micro[key] += c[key]
        precision = pct(tp, tp + fp)
        recall = pct(tp, tp + fn)
        specificity = pct(tn, tn + fp)
        f1 = pct(2 * precision * recall, precision + recall)
        out["lesion_presence"][lesion] = {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "precision": precision,
            "recall": recall,
            "specificity": specificity,
            "false_positive_rate": pct(fp, fp + tn),
            "f1": f1,
        }
    if micro:
        tp, fp, fn, tn = micro["tp"], micro["fp"], micro["fn"], micro["tn"]
        precision = pct(tp, tp + fp)
        recall = pct(tp, tp + fn)
        out["lesion_presence_micro"] = {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "precision": precision,
            "recall": recall,
            "false_positive_rate": pct(fp, fp + tn),
            "f1": pct(2 * precision * recall, precision + recall),
        }
    for task, c in sorted(by_task.items()):
        n = c["n"]
        task_out = {"n": n, "task_match_rate": pct(c["task_match"], n), "json_exact_rate": pct(c["json_exact"], n)}
        for key, val in sorted(c.items()):
            if key in {"n", "task_match", "json_exact"}:
                continue
            task_out[key + "_rate"] = pct(val, n)
        out["by_task"][task] = task_out

    text = json.dumps(out, ensure_ascii=False, indent=2)
    print(text)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
