#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import cv2
import numpy as np

# Allow running this file directly: add repo root to sys.path.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.stage1_easy.progress import default_progress_path, mark_done, mark_running


def parse_output_json(model_output: str) -> dict | None:
    m = re.search(r"## Output\s*\n([\s\S]+)$", model_output)
    if not m:
        return None
    s = m.group(1).strip()
    s = re.sub(r"^```json\s*", "", s)
    s = re.sub(r"\s*```$", "", s)
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        return None


def get_confidence_score(lesion_info: dict | None) -> float:
    if not lesion_info:
        return 0.0
    return 1.0 if lesion_info.get("present", False) else 0.0


def _auc(y_true: np.ndarray, y_score: np.ndarray) -> float | None:
    # lightweight AUC to avoid sklearn dependency
    if y_true.ndim != 1:
        y_true = y_true.reshape(-1)
    if y_score.ndim != 1:
        y_score = y_score.reshape(-1)
    if len(np.unique(y_true)) < 2:
        return None
    order = np.argsort(y_score)
    y_true_sorted = y_true[order]
    n_pos = float(np.sum(y_true_sorted == 1))
    n_neg = float(np.sum(y_true_sorted == 0))
    if n_pos == 0 or n_neg == 0:
        return None
    ranks = np.arange(1, len(y_true_sorted) + 1, dtype=np.float64)
    rank_sum_pos = float(np.sum(ranks[y_true_sorted == 1]))
    auc = (rank_sum_pos - n_pos * (n_pos + 1.0) / 2.0) / (n_pos * n_neg)
    return float(auc)


def _f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    tp = float(np.sum((y_true == 1) & (y_pred == 1)))
    fp = float(np.sum((y_true == 0) & (y_pred == 1)))
    fn = float(np.sum((y_true == 1) & (y_pred == 0)))
    if tp == 0 and (fp > 0 or fn > 0):
        return 0.0
    denom = (2.0 * tp + fp + fn)
    return float((2.0 * tp) / denom) if denom > 0 else 0.0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stage1 Easy metrics on IDRiD test masks from vLLM predictions.")
    p.add_argument("--pred-jsonl", required=True, help="JSONL produced by scripts/vllm_infer.py")
    p.add_argument("--test-json", required=True, help="ShareGPT test json (idrid_stage1_easy_test.json)")
    p.add_argument("--mask-dir", default="data/idrid/segmentation/test", help="IDRiD test segmentation dir")
    p.add_argument("--out", default="reports/stage1_easy_idrid_metrics.json")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    prog = default_progress_path()
    mark_running(prog, "eval_metrics", pred_jsonl=str(args.pred_jsonl), test_json=str(args.test_json), mask_dir=str(args.mask_dir))
    pred_path = Path(args.pred_jsonl)
    test_path = Path(args.test_json)
    mask_dir = Path(args.mask_dir)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    test_samples = json.loads(test_path.read_text(encoding="utf-8"))
    # vllm_infer writes one line per dataset row in order
    preds = [json.loads(x) for x in pred_path.read_text(encoding="utf-8").splitlines() if x.strip()]
    if len(preds) != len(test_samples):
        raise RuntimeError(f"pred_count({len(preds)}) != test_count({len(test_samples)})")

    lesion_types = ["MA", "HE", "EX", "SE"]
    y_true: dict[str, list[int]] = {k: [] for k in lesion_types}
    y_score: dict[str, list[float]] = {k: [] for k in lesion_types}
    parse_fail = 0
    format_ok = 0
    negative_fp = 0
    negative_total = 0

    for sample, pred in zip(test_samples, preds):
        img_rel = sample["images"][0]
        image_id = Path(img_rel).stem  # IDRiD_055 etc
        text = str(pred.get("predict", ""))

        if ("## Analysis" in text) and ("## Output" in text):
            format_ok += 1

        parsed = parse_output_json(text)
        if parsed is None:
            parse_fail += 1
            for lt in lesion_types:
                y_score[lt].append(0.0)
        else:
            for lt in lesion_types:
                info = parsed.get(lt, {"present": False})
                y_score[lt].append(get_confidence_score(info))

        # GT presence from masks
        for lt in lesion_types:
            # masks use IDRiD_55? Actually segmentation uses 2-digit in test too: IDRiD_55_MA.tif
            n = int(image_id.split("_")[1])
            two = f"{n:02d}"
            mpath = mask_dir / lt / f"IDRiD_{two}_{lt}.tif"
            if mpath.exists():
                m = cv2.imread(str(mpath), cv2.IMREAD_GRAYSCALE)
                has = int(m is not None and np.any(m > 0))
            else:
                has = 0
            y_true[lt].append(has)

        all_negative = all(y_true[lt][-1] == 0 for lt in lesion_types)
        if all_negative:
            negative_total += 1
            if parsed is not None and any(bool(parsed.get(lt, {}).get("present", False)) for lt in lesion_types):
                negative_fp += 1

    results: dict[str, object] = {
        "n_samples": len(test_samples),
        "cot_format_compliance": round(format_ok / len(test_samples), 4),
        "parse_fail_rate": round(parse_fail / len(test_samples), 4),
        "negative_false_positive_rate": round(negative_fp / negative_total, 4) if negative_total else None,
    }

    aucs = []
    f1s = []
    for lt in lesion_types:
        yt = np.array(y_true[lt], dtype=np.int32)
        ys = np.array(y_score[lt], dtype=np.float64)
        auc = _auc(yt, ys)
        results[f"{lt}_AUC"] = None if auc is None else round(float(auc), 4)
        results[f"{lt}_N_positive"] = int(np.sum(yt))
        if auc is not None:
            aucs.append(float(auc))

        yp = (ys >= 0.5).astype(np.int32)
        f1 = _f1(yt, yp)
        results[f"{lt}_F1"] = round(float(f1), 4)
        f1s.append(float(f1))

    results["mAUC"] = round(float(np.mean(aucs)), 4) if aucs else None
    results["macro_F1"] = round(float(np.mean(f1s)), 4) if f1s else None

    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(results, ensure_ascii=False, indent=2))
    print(f"saved={out_path}")
    mark_done(prog, "eval_metrics", out=str(out_path), **results)


if __name__ == "__main__":
    main()

