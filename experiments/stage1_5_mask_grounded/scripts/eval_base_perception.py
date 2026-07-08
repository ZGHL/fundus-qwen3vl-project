#!/usr/bin/env python3
"""Fair base-model lesion-perception eval (present/absent), reproducible.

The trained models emit a strict single-lesion JSON audit schema; the untrained base model
cannot follow that schema (~74% unparseable) — that measures FORMAT compliance, not perception.
To compare PERCEPTION fairly, re-prompt the base model with a plain question it can answer
(give the lesion name + visual description, ask only "Present or Absent"), then parse leniently.

Modes:
  build : eval_base_perception.py build <test_sft.jsonl> <out_sft.jsonl> <lesion_info.json>
          -> writes a base-friendly dataset (same images/labels, simple Present/Absent prompt).
          Register it in LLaMA-Factory dataset_info.json, run vllm_infer (no adapter), then:
  score : eval_base_perception.py score <test_sft.jsonl> <pred.jsonl>
          -> lenient Present/Absent parse -> per-lesion Recall/Spec/F1 + macro + parse_fail.
"""
import json, re, sys
from collections import defaultdict

LES = ["MA", "HE", "EX", "SE"]


def build(test_path, out_path, lesion_info_path):
    LI = json.load(open(lesion_info_path))
    rows = [json.loads(l) for l in open(test_path) if l.strip()]
    with open(out_path, "w") as fo:
        for r in rows:
            les = r["meta"]["lesion"]; info = LI[les]
            sysp = ("You are an experienced ophthalmologist reading a fundus photograph. "
                    "Judge ONLY whether the specified lesion is visibly present, based strictly on what you can see.")
            usr = (f"<image>\nTarget lesion: {info['name']} ({les}) — typical appearance: {info['visual']}.\n\n"
                   f"Is {info['name']} visibly present in THIS image?\n"
                   "Answer on the first line with exactly one word: Present or Absent. Then give a one-sentence reason.")
            fo.write(json.dumps({"messages": [{"role": "system", "content": sysp},
                                              {"role": "user", "content": usr},
                                              {"role": "assistant", "content": "Absent."}],
                                 "images": r["images"], "meta": r["meta"]}, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} base-friendly rows -> {out_path}")


def parse_pa(t):
    s = t.strip().lower()
    m = re.search(r"\b(present|absent|yes|no)\b", s[:60]) or re.search(r"\b(present|absent|yes|no)\b", s)
    if not m:
        return None
    return m.group(1) in ("present", "yes")


def score(test_path, pred_path):
    test = [json.loads(l) for l in open(test_path) if l.strip()]
    preds = [json.loads(l).get("predict", "") for l in open(pred_path) if l.strip()]
    per = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0, "tn": 0, "pf": 0})
    for r, p in zip(test, preds):
        les = r["meta"]["lesion"]; gt = r["meta"]["present_state"] == "present"
        pr = parse_pa(p); d = per[les]
        if pr is None: d["pf"] += 1; continue
        if gt and pr: d["tp"] += 1
        elif gt and not pr: d["fn"] += 1
        elif (not gt) and pr: d["fp"] += 1
        else: d["tn"] += 1
    recs = specs = f1s = 0.0; pf = 0; out = []
    macro = {"recall": [], "spec": [], "f1": []}
    for les in LES:
        d = per[les]; tp, fp, fn, tn = d["tp"], d["fp"], d["fn"], d["tn"]
        rec = tp / (tp + fn) if tp + fn else 0; spec = tn / (tn + fp) if tn + fp else 0
        prec = tp / (tp + fp) if tp + fp else 0; f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0
        macro["recall"].append(rec); macro["spec"].append(spec); macro["f1"].append(f1); pf += d["pf"]
        out.append(f"{les}: Recall {rec:.3f}  Spec {spec:.3f}  F1 {f1:.3f}  parse_fail {d['pf']}  (tp{tp} fp{fp} fn{fn} tn{tn})")
    mr = sum(macro["recall"]) / 4; ms = sum(macro["spec"]) / 4; mf = sum(macro["f1"]) / 4
    bal = sum((r + s) / 2 for r, s in zip(macro["recall"], macro["spec"])) / 4
    print("\n".join(out))
    print(f"MACRO: Recall {mr:.3f}  Spec {ms:.3f}  F1 {mf:.3f}  BalAcc {bal:.3f}  parse_fail {pf}/{len(test)}")


if __name__ == "__main__":
    if sys.argv[1] == "build":
        build(sys.argv[2], sys.argv[3], sys.argv[4])
    elif sys.argv[1] == "score":
        score(sys.argv[2], sys.argv[3])
    else:
        sys.exit("mode must be build|score")
