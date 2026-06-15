#!/usr/bin/env python3
"""Stage-1.5 perception ladder: compare several models on the SAME present/absent test.

This is the NEW-mainline foundation evidence: lesion-perception capability of
base (zero-shot) -> Adapter1 -> Stage-1.5 v3 -> v4, all on one identical test set, so
per-lesion Recall / Specificity / F1 (+ macro + balanced accuracy) are directly comparable.

Usage:
  perception_ladder.py <test_jsonl> <label1>:<pred1.jsonl> <label2>:<pred2.jsonl> ... [out_md]

Predictions align to test rows by order (vllm_infer preserves order). Any trailing argument
that is not "<label>:<path>" is treated as the output markdown path. Reuses score_proof's
scorer so numbers match the proof/v3 pipeline exactly.

Suggested ladder:
  perception_ladder.py data/stage1_5_v4_test_sft.jsonl \
      base:preds/base_zeroshot.jsonl adapter1:preds/adapter1.jsonl \
      v3:preds/v3.jsonl v4:preds/v4.jsonl  reports/STAGE1_5_PERCEPTION_LADDER.md
"""
import sys
from score_proof import score, agg, load_pred
import json

LES = ["MA", "HE", "EX", "SE"]


def balacc(model_agg):
    """macro balanced accuracy = mean over lesions of (recall+spec)/2."""
    vals = [(model_agg[l]["recall"] + model_agg[l]["spec"]) / 2 for l in LES if l in model_agg]
    return round(sum(vals) / len(vals), 3) if vals else 0.0


def main():
    test = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
    models, out_md = [], None
    for a in sys.argv[2:]:
        if ":" in a and not a.endswith(".md"):
            label, path = a.split(":", 1)
            models.append((label, agg(score(test, load_pred(path)))))
        else:
            out_md = a
    if not models:
        sys.exit("no <label>:<pred.jsonl> models given")

    labels = [m[0] for m in models]
    L = ["# Stage-1.5 perception ladder (same present/absent test)", "",
         f"test = {sys.argv[1].split('/')[-1]}  (n={len(test)} single-lesion rows)　models: {', '.join(labels)}",
         "",
         "Lesion-perception capability head-to-head on one identical, image-disjoint test.",
         "Recall = sensitivity to a present lesion; Spec = not over-reporting on absent; "
         "BalAcc = (Recall+Spec)/2. This is the NEW-mainline foundation (the faithful grader is "
         "bounded by this audit quality).", ""]

    # per-lesion blocks
    for metric in ["recall", "spec", "f1"]:
        L.append(f"### {metric.upper()} by lesion")
        L.append("| lesion | " + " | ".join(labels) + " |")
        L.append("|---" * (len(labels) + 1) + "|")
        for les in LES:
            cells = [f"{m[1].get(les, {}).get(metric, float('nan')):.3f}" for m in models]
            L.append(f"| {les} | " + " | ".join(cells) + " |")
        L.append("")

    # summary
    L.append("### Summary (macro)")
    L.append("| metric | " + " | ".join(labels) + " |")
    L.append("|---" * (len(labels) + 1) + "|")
    for metric in ["recall", "spec", "f1"]:
        cells = [f"{m[1]['MACRO'][metric]:.3f}" for m in models]
        L.append(f"| macro {metric} | " + " | ".join(cells) + " |")
    L.append("| **macro BalAcc** | " + " | ".join(f"**{balacc(m[1])}**" for m in models) + " |")
    # parse failures
    pf = []
    for label, a in models:
        tot = sum(a.get(les, {}).get("parse_fail", 0) for les in LES)
        pf.append(str(tot))
    L.append("| parse_fail (total) | " + " | ".join(pf) + " |")
    L += ["", "Reading: a trained perception model should lift Recall AND Spec over base "
          "zero-shot (which typically over-reports -> low Spec, or fails to emit the schema "
          "-> parse_fail). v3 already fixed Spec (0.20->0.59 vs Adapter1); v4 should additionally "
          "lift Recall (esp. MA/HE/EX) without losing that Spec."]
    txt = "\n".join(L)
    print(txt)
    if out_md:
        open(out_md, "w").write(txt + "\n")


if __name__ == "__main__":
    main()
