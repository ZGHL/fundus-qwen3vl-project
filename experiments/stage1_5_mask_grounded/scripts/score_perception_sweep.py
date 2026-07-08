#!/usr/bin/env python3
"""Score a set of single-lesion present/absent prediction files (schema JSON) vs a fixed test.

Usage: score_perception_sweep.py <test_sft.jsonl> <out.md> <label1>:<pred1.jsonl> ...
Emits per-lesion Recall/Spec/F1 + macro avg F1, one row per model. Used by the auto pipeline.
"""
import json, re, sys, os, statistics as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from score_proof import jget

LES = ["MA", "HE", "EX", "SE"]

def sch(t):
    j = jget(t)
    if j is not None and "present" in j:
        return bool(j["present"])
    m = re.search(r'"present"\s*:\s*(true|false)', t)
    return {"true": True, "false": False}[m.group(1)] if m else None

def score(test, pf):
    P = [json.loads(l).get("predict", "") for l in open(pf)]
    per = {l: {"tp": 0, "fp": 0, "fn": 0, "tn": 0, "pf": 0} for l in LES}
    for r, p in zip(test, P):
        les = r["meta"]["lesion"]; gt = r["meta"]["present_state"] == "present"; d = per[les]; pr = sch(p)
        if pr is None: d["pf"] += 1; continue
        d["tp" if (gt and pr) else "fn" if gt else "fp" if pr else "tn"] += 1
    return per

def mt(d):
    tp, fp, fn, tn = d["tp"], d["fp"], d["fn"], d["tn"]
    rec = tp / (tp + fn) if tp + fn else 0; spec = tn / (tn + fp) if tn + fp else 0
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0
    return rec, spec, f1

def main():
    test = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
    out_md = sys.argv[2]
    models = []
    for a in sys.argv[3:]:
        if ":" in a:
            label, pf = a.split(":", 1)
            if os.path.exists(pf):
                models.append((label, pf))
    L = ["# 病灶感知 checkpoint 扫描(不变的测试集)", "",
         "| 模型 | SE_R | SE_Sp | SE_F1 | SE_FP | MA_F1 | HE_F1 | EX_F1 | avg_F1 |",
         "|---|--:|--:|--:|--:|--:|--:|--:|--:|"]
    for label, pf in models:
        per = score(test, pf)
        se = mt(per["SE"]); f1 = {l: mt(per[l])[2] for l in LES}
        L.append(f"| {label} | {se[0]:.3f} | {se[1]:.3f} | {se[2]:.3f} | {per['SE']['fp']} | "
                 f"{f1['MA']:.3f} | {f1['HE']:.3f} | {f1['EX']:.3f} | {st.mean(f1.values()):.3f} |")
    txt = "\n".join(L)
    open(out_md, "w").write(txt + "\n")
    print(txt)

if __name__ == "__main__":
    main()
