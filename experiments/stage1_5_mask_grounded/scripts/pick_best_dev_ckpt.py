#!/usr/bin/env python3
"""Pick the best Stage-1.5 v5 checkpoint on the DEV set — mechanically, no eyeballing.

Reuses score_proof's scorer so DEV numbers match the proof/ladder pipeline exactly. Reads a
directory of per-checkpoint DEV prediction jsonls named `dev_ckpt-<step>.jsonl` (as written by
run_stage1_5_v5_dev_sweep.sh via vllm_infer --save_name), scores each against the DEV set, and:
  - writes dev_curve.csv (step, macro_balacc, macro_recall, macro_spec, per-lesion recall/spec,
    parse_fail) -> the "is it still learning / has it plateaued" curve;
  - prints the peak checkpoint by macro balanced accuracy (the selection rule).

The chosen checkpoint is then evaluated ONCE on the TEST set (never used here) for the final number.

Usage:
  pick_best_dev_ckpt.py <dev_sft.jsonl> <preds_dir> [out_csv]
    preds_dir: directory containing dev_ckpt-<step>.jsonl files
    out_csv:   default <preds_dir>/dev_curve.csv
"""
import json, os, re, sys, glob
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from score_proof import score, agg, load_pred

LES = ["MA", "HE", "EX", "SE"]


def balacc(a):
    vals = [(a[l]["recall"] + a[l]["spec"]) / 2 for l in LES if l in a]
    return round(sum(vals) / len(vals), 4) if vals else 0.0


def main():
    dev = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
    preds_dir = sys.argv[2]
    out_csv = sys.argv[3] if len(sys.argv) > 3 else os.path.join(preds_dir, "dev_curve.csv")

    rows = []
    for f in glob.glob(os.path.join(preds_dir, "dev_ckpt-*.jsonl")):
        m = re.search(r"dev_ckpt-(\d+)\.jsonl$", f)
        if not m:
            continue
        step = int(m.group(1))
        a = agg(score(dev, load_pred(f)))
        rows.append({
            "step": step, "macro_balacc": balacc(a),
            "macro_recall": a["MACRO"]["recall"], "macro_spec": a["MACRO"]["spec"],
            "per_lesion": {l: (a.get(l, {}).get("recall"), a.get(l, {}).get("spec")) for l in LES},
            "parse_fail": sum(a.get(l, {}).get("parse_fail", 0) for l in LES),
        })
    if not rows:
        sys.exit(f"no dev_ckpt-*.jsonl predictions found in {preds_dir}")
    rows.sort(key=lambda r: r["step"])

    header = ["step", "macro_balacc", "macro_recall", "macro_spec"] + \
             [f"{l}_recall" for l in LES] + [f"{l}_spec" for l in LES] + ["parse_fail"]
    with open(out_csv, "w") as fo:
        fo.write(",".join(header) + "\n")
        for r in rows:
            cells = [r["step"], r["macro_balacc"], r["macro_recall"], r["macro_spec"]]
            cells += [r["per_lesion"][l][0] for l in LES] + [r["per_lesion"][l][1] for l in LES]
            cells += [r["parse_fail"]]
            fo.write(",".join(str(c) for c in cells) + "\n")

    best = max(rows, key=lambda r: r["macro_balacc"])
    print(f"# Stage-1.5 v5 DEV checkpoint curve  (n_dev={len(dev)})  -> {out_csv}")
    print(f"{'step':>7} {'BalAcc':>8} {'recall':>8} {'spec':>8}  per-lesion spec (MA/HE/EX/SE)  parse_fail")
    def _f(v): return f"{v:.2f}" if isinstance(v, (int, float)) else "  NA"
    for r in rows:
        mark = "  <-- BEST" if r["step"] == best["step"] else ""
        sp = "/".join(_f(r['per_lesion'][l][1]) for l in LES)
        print(f"{r['step']:>7} {r['macro_balacc']:>8.4f} {r['macro_recall']:>8.3f} "
              f"{r['macro_spec']:>8.3f}  {sp:>22}  {r['parse_fail']:>5}{mark}")
    print(f"\nSELECT checkpoint-{best['step']}  (macro BalAcc {best['macro_balacc']:.4f}). "
          f"Evaluate THIS checkpoint once on the TEST set for the final reported number.")


if __name__ == "__main__":
    main()
