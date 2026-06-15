#!/usr/bin/env python3
"""Score Stage-2 faithful triage.

Usage:
  score_stage2.py <test_jsonl> <pred_jsonl> [out_md] [--from-audit] [--dist <distribution.json>]

Two scoring modes
-----------------
default  : predicted tier = the model's FREELY GENERATED `dr_tier` (reproduces the
           checkpoint-sweep report; lets the model drift from its own audit).
--from-audit : predicted tier = fitted_map[ pattern(model's lesions_present) ].
           The model only supplies the verifiable lesion AUDIT; the tier is COMPUTED
           by the transparent fitted presence->tier map. Faithful by construction
           (faithfulness == 1.0). Because the [Lesion Audit] block is emitted BEFORE
           the final JSON, truncated outputs that lost their JSON are still recovered
           from the audit lines -> this also rescues most "invalid tier" predictions.

Alignment is by row order. test meta carries clinical_grade / clinical_tier /
dr_tier (GT-presence faithful target, the true label for QWK/F1/confusion) / pattern.
"""
import json, re, sys, os
from collections import Counter

# ---- project faithful-tier ordering (for QWK / MAE / confusion) ----
ORDER = ["No-DR", "Mild", "Moderate", "Mod-or-Severe-indeterminate", "Severe"]
IDX = {t: i for i, t in enumerate(ORDER)}
REFER = {"Moderate", "Mod-or-Severe-indeterminate", "Severe"}
LES4 = ["MA", "HE", "EX", "SE"]

# ---- fitted presence->tier map (embedded fallback; overridden by --dist) ----
FITTED_MAP = {
    "none": "No-DR", "MA": "Mild",
    "SE": "Moderate", "HE": "Moderate", "HESE": "Moderate",
    "MASE": "Moderate", "MAEX": "Moderate", "MAEXSE": "Moderate", "MAHE": "Moderate",
    "EX": "Mod-or-Severe-indeterminate", "EXSE": "Mod-or-Severe-indeterminate",
    "MAHESE": "Mod-or-Severe-indeterminate", "MAHEEX": "Mod-or-Severe-indeterminate",
    "HEEX": "Severe", "HEEXSE": "Severe", "MAHEEXSE": "Severe",
}

# lesion-name aliases -> canonical code
ALIASES = [
    ("MA", ("ma", "microaneurysm", "micro-aneurysm")),
    ("HE", ("he", "hemorrhage", "haemorrhage", "hemorrhages", "haemorrhages")),
    ("EX", ("ex", "hard exudate", "hardexudate", "hard-exudate")),
    ("SE", ("se", "soft exudate", "softexudate", "soft-exudate", "cotton wool", "cotton-wool")),
]


def load_map(dist_path):
    if dist_path and os.path.exists(dist_path):
        try:
            d = json.load(open(dist_path))
            for k in ("fitted_map(pattern->tier)", "fitted_map", "fit_map"):
                if isinstance(d.get(k), dict):
                    return d[k]
            for v in d.values():
                if isinstance(v, dict) and v.get("none") == "No-DR":
                    return v
        except Exception:
            pass
    return FITTED_MAP


def last_json(t):
    objs, depth, st = [], 0, None
    for i, c in enumerate(t):
        if c == "{":
            if depth == 0:
                st = i
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0 and st is not None:
                objs.append(t[st:i + 1]); st = None
    for m in reversed(objs):
        try:
            return json.loads(m)
        except Exception:
            pass
    return None


def canon_lesion(tok):
    s = str(tok).strip().lower()
    for code, names in ALIASES:
        if s == code.lower() or s in names or any(n in s for n in names):
            return code
    return None


def audit_from_text(t):
    """Recover MA/HE/EX/SE audit from the [Lesion Audit] lines (rescues truncated JSON).
    Returns (present_set, complete): a lesion is present only on an explicit
    '<code>: present' line; `complete` is True only when ALL FOUR lesions have an explicit
    present|absent verdict. Missing a lesion line is NOT treated as absent -> the audit is
    incomplete and must not be scored (avoids a falsely-valid 'all absent -> No-DR')."""
    present, resolved = set(), set()
    for code in LES4:
        # e.g.  "- MA: present — ..."   /  "MA : present"
        if re.search(rf"(?mi)^\s*[-*]?\s*{code}\s*[:：]\s*present\b", t):
            present.add(code); resolved.add(code)
        elif re.search(rf"(?mi)^\s*[-*]?\s*{code}\s*[:：]\s*absent\b", t):
            resolved.add(code)
    return present, len(resolved) == len(LES4)


def nv_irma_fabricated(j, text):
    """True if NV or IRMA is claimed PRESENT (must never happen — they are always abstained)."""
    if j:
        pres = j.get("lesions_present") or []
        if any(str(x).upper() in ("NV", "IRMA") for x in pres):
            return True
    # explicit present line in audit text
    if re.search(r"(?mi)^\s*[-*]?\s*(NV|IRMA)\s*[:：]\s*present\b", text):
        return True
    return False


def pattern_key(present):
    k = "".join(c for c in LES4 if c in present)
    return k if k else "none"


def extract_present(j, text):
    """Return (present_set, source) where source in {json, audit, none}."""
    if j is not None and isinstance(j.get("lesions_present"), list):
        # a parsed JSON list is complete by schema (absent lesions are simply omitted)
        pres = {c for c in (canon_lesion(x) for x in j["lesions_present"]) if c}
        return pres, "json"
    pres, complete = audit_from_text(text)
    if complete:   # require all 4 lesions explicitly resolved; else unrecoverable
        return pres, "audit"
    return set(), "none"


def norm_free_tier(j, text):
    """Normalize the model's freely generated dr_tier to a valid faithful tier, else None."""
    raw = (j or {}).get("dr_tier")
    s = str(raw).strip().lower() if raw is not None else ""
    if not s:
        return None
    if s in ("no dr", "no-dr", "no_dr", "nodr", "0", "dr0"):
        return "No-DR"
    if s in ("mild", "mild npdr", "1", "dr1"):
        return "Mild"
    if s in ("moderate", "moderate npdr", "dr2", "2"):
        return "Moderate"
    if s in ("severe", "severe npdr", "3", "dr3", "4", "dr4", "pdr", "proliferative"):
        return "Severe"
    if "indetermin" in s or "mod-or-severe" in s or "ungradable" in s:
        return "Mod-or-Severe-indeterminate"
    return None


def qwk(true_idx, pred_idx, k=5):
    """Quadratic weighted kappa over a list of (i,j) index pairs."""
    n = len(true_idx)
    if n == 0:
        return 0.0
    O = [[0] * k for _ in range(k)]
    for i, j in zip(true_idx, pred_idx):
        O[i][j] += 1
    rt = [sum(O[i]) for i in range(k)]
    ct = [sum(O[i][j] for i in range(k)) for j in range(k)]
    num = den = 0.0
    for i in range(k):
        for j in range(k):
            w = ((i - j) / (k - 1)) ** 2
            E = rt[i] * ct[j] / n
            num += w * O[i][j]
            den += w * E
    return 1.0 - num / den if den else 0.0


def evaluate(test, preds, from_audit, fmap):
    """Core scorer. Returns (metrics dict, confusion Counter, src Counter, per-tier F1 list)."""
    n = min(len(test), len(preds))

    valid = 0
    fab = 0
    abstain = 0
    src_ct = Counter()
    ti, pj = [], []                       # index pairs for QWK (valid only)
    ae_sum = 0.0                          # MAE over all n (invalid -> max penalty)
    perclass = {t: [0, 0, 0] for t in ORDER}   # tier -> [TP, FP, FN]
    ref_tp = ref_fp = ref_fn = ref_tn = 0
    sev_total = sev_ref = 0
    faith_ok = faith_n = 0
    cm = Counter()                        # (true dr_tier, pred or 'Invalid')

    for r, raw in zip(test[:n], preds[:n]):
        meta = r["meta"]
        true_tier = meta["dr_tier"]               # GT-presence faithful target (true label)
        cg = meta["clinical_grade"]
        clin_ref = cg >= 2
        j = last_json(raw)

        if nv_irma_fabricated(j, raw):
            fab += 1

        if from_audit:
            present, src = extract_present(j, raw)
            src_ct[src] += 1
            pred = fmap.get(pattern_key(present)) if src != "none" else None
            # faithful by construction
            if pred is not None:
                faith_n += 1; faith_ok += 1
        else:
            pred = norm_free_tier(j, raw)
            if pred is not None:
                faith_n += 1
                present, _ = extract_present(j, raw)
                if pred == fmap.get(pattern_key(present)):
                    faith_ok += 1

        # ---- aggregate ----
        if pred is not None:
            valid += 1
            ti.append(IDX[true_tier]); pj.append(IDX[pred])
            ae_sum += abs(IDX[true_tier] - IDX[pred])
            if pred == "Mod-or-Severe-indeterminate":
                abstain += 1
            # per-class TP/FP/FN
            if pred == true_tier:
                perclass[true_tier][0] += 1
            else:
                perclass[pred][1] += 1
                perclass[true_tier][2] += 1
            cm[(true_tier, pred)] += 1
        else:
            ae_sum += (len(ORDER) - 1)        # max ordinal penalty
            perclass[true_tier][2] += 1       # missed true class
            cm[(true_tier, "Invalid")] += 1

        pred_ref = pred in REFER if pred else False   # invalid -> non-referable (safety)
        if clin_ref and pred_ref:
            ref_tp += 1
        elif clin_ref and not pred_ref:
            ref_fn += 1
        elif (not clin_ref) and pred_ref:
            ref_fp += 1
        else:
            ref_tn += 1
        if cg in (3, 4):
            sev_total += 1; sev_ref += int(pred_ref)

    # ---- metrics ----
    valid_rate = valid / n if n else 0
    kappa = qwk([t for t in ti], [p for p in pj], k=len(ORDER))
    mae = ae_sum / n if n else 0
    f1s = []
    for t in ORDER:
        tp, fp, fn = perclass[t]
        p = tp / (tp + fp) if tp + fp else 0.0
        rc = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * p * rc / (p + rc) if p + rc else 0.0)
    macro_f1 = sum(f1s) / len(f1s)
    sens = ref_tp / (ref_tp + ref_fn) if ref_tp + ref_fn else 0
    spec = ref_tn / (ref_tn + ref_fp) if ref_tn + ref_fp else 0
    ppv = ref_tp / (ref_tp + ref_fp) if ref_tp + ref_fp else 0
    sev_recall = sev_ref / sev_total if sev_total else 0
    faith = faith_ok / faith_n if faith_n else 0
    fab_rate = fab / valid if valid else 0

    m = dict(n=n, valid=valid, valid_rate=valid_rate, qwk=kappa, macro_f1=macro_f1, mae=mae,
             sens=sens, spec=spec, ppv=ppv, sev_recall=sev_recall,
             faith=faith, faith_n=faith_n, faith_ok=faith_ok, fab=fab, fab_rate=fab_rate,
             abstain=abstain, abstain_rate=abstain / n if n else 0,
             ref=(ref_tp, ref_fp, ref_fn, ref_tn), sev=(sev_ref, sev_total))
    return m, cm, src_ct, f1s


def main():
    pos = [a for a in sys.argv[1:] if not a.startswith("--")]
    from_audit = "--from-audit" in sys.argv
    dist = None
    if "--dist" in sys.argv:
        dist = sys.argv[sys.argv.index("--dist") + 1]
    test_path, pred_path = pos[0], pos[1]
    out_md = pos[2] if len(pos) > 2 else None
    fmap = load_map(dist)
    test = [json.loads(l) for l in open(test_path) if l.strip()]
    preds = [json.loads(l).get("predict", "") for l in open(pred_path) if l.strip()]

    m, cm, src_ct, f1s = evaluate(test, preds, from_audit, fmap)
    n = m["n"]
    valid_rate, kappa, macro_f1, mae = m["valid_rate"], m["qwk"], m["macro_f1"], m["mae"]
    sens, spec, ppv, sev_recall = m["sens"], m["spec"], m["ppv"], m["sev_recall"]
    faith, faith_ok, faith_n = m["faith"], m["faith_ok"], m["faith_n"]
    fab, fab_rate, abstain, valid = m["fab"], m["fab_rate"], m["abstain"], m["valid"]
    ref_tp, ref_fp, ref_fn, ref_tn = m["ref"]
    sev_ref, sev_total = m["sev"]

    mode = "FROM-AUDIT (tier = fitted_map[audit], faithful by construction)" if from_audit \
        else "FREE (tier = model's generated dr_tier)"
    L = [f"# Stage-2 results  (n={n})   mode = {mode}", ""]
    if from_audit:
        L += [f"audit source: {dict(src_ct)}  (json=parsed JSON list, audit=recovered from "
              f"[Lesion Audit] lines after truncation, none=unrecoverable)", ""]
    L += [
        "## Headline",
        f"valid tier rate = {valid_rate:.4f}   QWK(valid) = {kappa:.4f}   Macro-F1 = {macro_f1:.4f}   MAE = {mae:.4f}",
        f"per-tier F1: " + "  ".join(f"{t}={f:.3f}" for t, f in zip(ORDER, f1s)),
        "",
        "## Referable (pred >=Moderate  vs  clinical grade >=2)",
        f"sensitivity = {sens:.4f}   specificity = {spec:.4f}   PPV = {ppv:.4f}   "
        f"(TP{ref_tp} FP{ref_fp} FN{ref_fn} TN{ref_tn})",
        f"severe-safety recall (true g3/g4 -> referable) = {sev_ref}/{sev_total} = {sev_recall:.4f}",
        "",
        "## Faithfulness / safety",
        f"tier consistent with own audit = {faith_ok}/{faith_n} = {faith:.4f}"
        + ("  (==1.000 by construction)" if from_audit else ""),
        f"NV/IRMA fabrication = {fab} / {valid} valid = {fab_rate:.4f}  (want 0)",
        f"abstention (Mod-or-Severe-indeterminate) = {abstain}/{n} = {abstain / n if n else 0:.4f}",
        "",
        "## Confusion: true faithful tier (dr_tier) -> predicted",
        "| true \\ pred | " + " | ".join(ORDER) + " | Invalid |",
        "|" + "---|" * (len(ORDER) + 2),
    ]
    for tt in ORDER:
        row = [str(cm.get((tt, pt), 0)) for pt in ORDER] + [str(cm.get((tt, "Invalid"), 0))]
        L.append(f"| {tt} | " + " | ".join(row) + " |")
    L += ["", "(reference: faithful ceiling = 0.688 4-tier accuracy on GT presence)"]
    txt = "\n".join(L)
    print(txt)
    if out_md:
        open(out_md, "w").write(txt + "\n")


if __name__ == "__main__":
    main()
