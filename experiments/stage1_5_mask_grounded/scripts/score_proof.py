#!/usr/bin/env python3
"""Score Stage-1.5 proof: present/absent + count/area bucket accuracy.
Usage: score_proof.py <test_jsonl> <baseline_pred> <trained_pred> <out_md>
Predictions aligned to test rows by order (vllm_infer preserves order)."""
import json, re, sys
from collections import defaultdict

def jget(text):
    objs, depth, start = [], 0, None
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0: start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start is not None:
                objs.append(text[start:i+1]); start = None
    for m in reversed(objs):
        try: return json.loads(m)
        except Exception: pass
    return None

def parse(pred):
    j = jget(pred)
    if not j:
        m = re.search(r'"present"\s*:\s*(true|false)', pred)
        return ({"true": True, "false": False}[m.group(1)] if m else None), None, None
    pres = j.get("present")
    at = j.get("attributes") or {}
    return pres, at.get("count_bucket"), at.get("area_bucket")

def load_pred(p):
    return [json.loads(l).get("predict", "") for l in open(p) if l.strip()]

def score(test, preds):
    per = defaultdict(lambda: {"tp":0,"fp":0,"fn":0,"tn":0,"cc_n":0,"cc_ok":0,"ac_n":0,"ac_ok":0,"parse_fail":0})
    for r, pr in zip(test, preds):
        les = r["meta"]["lesion"]; gt_pres = r["meta"]["present_state"] == "present"
        p_pres, p_cb, p_ab = parse(pr)
        d = per[les]
        if p_pres is None: d["parse_fail"] += 1; continue
        if gt_pres and p_pres: d["tp"] += 1
        elif gt_pres and not p_pres: d["fn"] += 1
        elif (not gt_pres) and p_pres: d["fp"] += 1
        else: d["tn"] += 1
        if gt_pres and p_pres:  # bucket accuracy only when both present
            if r["meta"].get("count_bucket"):
                d["cc_n"] += 1; d["cc_ok"] += int(p_cb == r["meta"]["count_bucket"])
            if r["meta"].get("area_bucket"):
                d["ac_n"] += 1; d["ac_ok"] += int(p_ab == r["meta"]["area_bucket"])
    return per

def agg(per):
    out = {}
    f1s=recs=specs=[]; f1l=[]; recl=[]; specl=[]; ccok=ccn=acok=acn=0
    for les, d in per.items():
        tp,fp,fn,tn = d["tp"],d["fp"],d["fn"],d["tn"]
        rec = tp/(tp+fn) if tp+fn else 0.0
        prec = tp/(tp+fp) if tp+fp else 0.0
        f1 = 2*prec*rec/(prec+rec) if prec+rec else 0.0
        spec = tn/(tn+fp) if tn+fp else 0.0
        cc = d["cc_ok"]/d["cc_n"] if d["cc_n"] else float("nan")
        ac = d["ac_ok"]/d["ac_n"] if d["ac_n"] else float("nan")
        out[les] = {"f1":round(f1,3),"recall":round(rec,3),"spec":round(spec,3),
                    "count_acc":round(cc,3) if d["cc_n"] else None,"area_acc":round(ac,3) if d["ac_n"] else None,
                    "parse_fail":d["parse_fail"]}
        f1l.append(f1); recl.append(rec); specl.append(spec)
        ccok+=d["cc_ok"]; ccn+=d["cc_n"]; acok+=d["ac_ok"]; acn+=d["ac_n"]
    out["MACRO"] = {"f1":round(sum(f1l)/len(f1l),3),"recall":round(sum(recl)/len(recl),3),
                    "spec":round(sum(specl)/len(specl),3),
                    "count_acc":round(ccok/ccn,3) if ccn else None,"area_acc":round(acok/acn,3) if acn else None}
    return out

def main():
    test = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
    base = agg(score(test, load_pred(sys.argv[2])))
    trn = agg(score(test, load_pred(sys.argv[3])))
    lines = ["# Stage-1.5 Proof Results (FGADR MAIN4 held-out, image-disjoint)\n",
             f"Test set: {len(test)} single-lesion samples.\n",
             "## Baseline = Adapter 1 (no Stage-1.5) vs Trained = Adapter1 + FGADR count/area+negatives\n",
             "| Lesion | metric | Adapter1 | Stage1.5 | Δ |", "|---|---|---:|---:|---:|"]
    for les in ["MA","HE","EX","SE","MACRO"]:
        b, t = base.get(les,{}), trn.get(les,{})
        for k in ["f1","recall","spec","count_acc","area_acc"]:
            bv, tv = b.get(k), t.get(k)
            dv = round(tv-bv,3) if isinstance(bv,(int,float)) and isinstance(tv,(int,float)) else ""
            lines.append(f"| {les} | {k} | {bv} | {tv} | {dv} |")
    lines.append("\n**Read:** count_acc/area_acc = bucket accuracy on samples both-present (Adapter1 mostly can't emit buckets → near 0/parse-fail = the new capability). spec ↑ = strong negatives fixing over-report.")
    open(sys.argv[4],"w").write("\n".join(lines)+"\n")
    print("\n".join(lines))

if __name__ == "__main__":
    main()
