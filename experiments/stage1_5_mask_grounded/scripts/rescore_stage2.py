#!/usr/bin/env python3
"""Re-score every Stage-2 checkpoint prediction in BOTH modes and emit a comparison table.

Plan A: quantify the decoupled, faithful-by-construction design (tier = fitted_map[audit])
against the model's free dr_tier generation — using the SAME predictions, no retraining.

Usage:
  rescore_stage2.py <test_jsonl> <preds_dir_or_glob> <dist_json> [out_md]

<preds_dir_or_glob> may be a directory (scans *.jsonl) or a glob like 'preds/*pred*.jsonl'.
Each prediction file is one checkpoint's vLLM output (rows with a "predict" field), aligned
by row order to <test_jsonl>. The checkpoint label is taken from the file name.
"""
import json, sys, os, glob, re
from score_stage2 import evaluate, load_map


def ckpt_label(path):
    b = os.path.basename(path)
    m = re.search(r"(checkpoint[-_]?\d+|final|ckpt[-_]?\d+|\d+)", b)
    return m.group(1) if m else b


def load_preds(path):
    return [json.loads(l).get("predict", "") for l in open(path) if l.strip()]


def main():
    test_path, where, dist = sys.argv[1], sys.argv[2], sys.argv[3]
    out_md = sys.argv[4] if len(sys.argv) > 4 else None
    test = [json.loads(l) for l in open(test_path) if l.strip()]
    fmap = load_map(dist)

    if os.path.isdir(where):
        files = sorted(glob.glob(os.path.join(where, "*.jsonl")))
    else:
        files = sorted(glob.glob(where))
    if not files:
        sys.exit(f"no prediction files matched: {where}")

    rows = []
    for f in files:
        preds = load_preds(f)
        lbl = ckpt_label(f)
        free, _, _, _ = evaluate(test, preds, False, fmap)
        aud, _, src, _ = evaluate(test, preds, True, fmap)
        rows.append((lbl, free, aud, src))

    def fmt(label, m):
        return (f"| {label} | {m['valid_rate']:.3f} | {m['qwk']:.3f} | {m['macro_f1']:.3f} | "
                f"{m['mae']:.3f} | {m['sens']:.3f} | {m['spec']:.3f} | {m['sev_recall']:.3f} | "
                f"{m['faith']:.3f} | {m['fab']} | {m['abstain_rate']:.3f} |")

    H = ("| ckpt / mode | valid | QWK | MacroF1 | MAE | RefSens | RefSpec | SevRecall | "
         "Faithful | Fab | Abstain |")
    SEP = "|" + "---|" * 11
    L = ["# Stage-2 re-score: FREE vs FROM-AUDIT (decoupled, faithful by construction)", "",
         f"test = {os.path.basename(test_path)}  (n={len(test)})   checkpoints = {len(files)}", "",
         "FREE = model's generated dr_tier (reproduces the sweep report).",
         "FROM-AUDIT = tier computed by fitted_map[ model's lesion audit ]; truncated JSON is",
         "recovered from the [Lesion Audit] block, so valid-tier coverage rises and faithfulness",
         "is 1.000 by construction.", "", H, SEP]
    for lbl, free, aud, src in rows:
        L.append(fmt(f"{lbl} · free", free))
        rescued = src.get("audit", 0)
        L.append(fmt(f"{lbl} · audit", aud) + f"  <!-- rescued-from-audit={rescued} -->")
    L += ["", "## Per-checkpoint audit-source recovery (FROM-AUDIT)",
          "| ckpt | json | rescued-from-audit | unrecoverable |", "|---|---|---|---|"]
    for lbl, _, _, src in rows:
        L.append(f"| {lbl} | {src.get('json',0)} | {src.get('audit',0)} | {src.get('none',0)} |")
    L += ["", "(reference: faithful ceiling = 0.688 4-tier accuracy on GT presence; "
          "gold-audit upper bound for from-audit referable = sens 0.982 / spec 0.886)"]
    txt = "\n".join(L)
    print(txt)
    if out_md:
        open(out_md, "w").write(txt + "\n")


if __name__ == "__main__":
    main()
