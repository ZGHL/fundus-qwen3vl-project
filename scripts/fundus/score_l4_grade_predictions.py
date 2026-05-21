#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, re
from pathlib import Path
import numpy as np
from sklearn.metrics import cohen_kappa_score, f1_score

def extract_grade(text: str):
    if not text:
        return None
    patterns = [r"\"dr_grade\"\s*:\s*([0-4])", r"\"grade\"\s*:\s*([0-4])", r"DR\s*Grade\s*[:：]?\s*([0-4])", r"dr_grade\s*[:：]?\s*([0-4])", r"Grade\s*[:：]?\s*([0-4])"]
    for marker in ["【结论】", "【JSON】"]:
        if marker in text:
            tail = text.split(marker, 1)[1]
            for pat in patterns:
                m = re.search(pat, tail, re.IGNORECASE)
                if m:
                    return int(m.group(1))
    for pat in patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            return int(m.group(1))
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    rows = [json.loads(l) for l in args.predictions.read_text().splitlines() if l.strip()]
    y_true, y_pred, failures = [], [], []
    parse_fail_label = parse_fail_pred = 0
    for i, row in enumerate(rows):
        gt = extract_grade(row.get("label", ""))
        pr = extract_grade(row.get("predict", ""))
        if gt is None:
            parse_fail_label += 1; continue
        if pr is None:
            parse_fail_pred += 1
            if len(failures) < 5: failures.append({"idx": i, "predict": row.get("predict", "")[:500]})
            continue
        y_true.append(gt); y_pred.append(pr)
    out = {"n_rows": len(rows), "n_scored": len(y_true), "parse_fail_label": parse_fail_label, "parse_fail_pred": parse_fail_pred, "parse_rate": len(y_true)/len(rows) if rows else 0.0, "label_counts": {str(g): 0 for g in range(5)}, "pred_counts": {str(g): 0 for g in range(5)}}
    for g in y_true: out["label_counts"][str(g)] += 1
    for g in y_pred: out["pred_counts"][str(g)] += 1
    if y_true:
        yt = np.array(y_true); yp = np.array(y_pred)
        out["accuracy"] = float((yt == yp).mean())
        out["macro_f1"] = float(f1_score(yt, yp, labels=[0,1,2,3,4], average="macro", zero_division=0))
        out["qwk"] = float(cohen_kappa_score(yt, yp, weights="quadratic"))
        out["per_grade_f1"] = {str(g): float(v) for g, v in enumerate(f1_score(yt, yp, labels=[0,1,2,3,4], average=None, zero_division=0))}
        out["per_grade_recall"] = {str(g): (float((yp[yt == g] == g).mean()) if (yt == g).sum() else None) for g in range(5)}
        out["per_grade_precision"] = {str(g): (float((yt[yp == g] == g).mean()) if (yp == g).sum() else None) for g in range(5)}
        out["confusion"] = [[int(((yt == i) & (yp == j)).sum()) for j in range(5)] for i in range(5)]
    if failures: out["parse_failure_examples"] = failures
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, ensure_ascii=False, indent=2))
    print(json.dumps(out, ensure_ascii=False, indent=2))
if __name__ == "__main__": main()
