#!/usr/bin/env python3
"""Figure 3 (lesion positive rate per DR grade) + faithful ceiling, from FGADR GT masks.

Reproduces the design-evidence figure/numbers in STAGE2_REDESIGN.md §1.5:
  - per-grade positive rate of MA/HE/EX/SE (motivation: G2-G4 profiles overlap),
  - pattern -> grade distribution (e.g. MAHEEX ~50/50 G2/G3),
  - best presence->grade map 4-class accuracy = the faithful ceiling (0.688 on FGADR).

Run from the LLaMA-Factory root (needs data/fundus_validated/validated_clean.jsonl and
data/FGADR/Seg-set/*_Masks/*.png). Writes the PNG to the path given as argv[1].
"""
import os, sys, json
from collections import Counter, defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

OUT = sys.argv[1] if len(sys.argv) > 1 else "figures/fig3_lesion_posrate_by_grade.png"
VAL = "data/fundus_validated/validated_clean.jsonl"
FG = {"MA": "Microaneurysms_Masks", "HE": "Hemohedge_Masks", "EX": "HardExudate_Masks", "SE": "SoftExudate_Masks"}
LES = ["MA", "HE", "EX", "SE"]


def stem(p): return os.path.splitext(os.path.basename(p))[0] if p else ""
def grade(r):
    x = r.get("grade")
    try: return int(x) if x is not None and 0 <= int(x) <= 4 else None
    except Exception: return None
def mask_present(path):                       # FGADR masks are 0/255 -> >=128
    try: im = np.asarray(Image.open(path).convert("L"))
    except Exception: return None
    return int((im >= 128).sum()) >= 5


def main():
    recs = [json.loads(l) for l in open(VAL) if l.strip()]
    rows = []
    for r in [x for x in recs if x["dataset"] == "fgadr_seg"]:
        st = stem(r.get("cropped_path") or r.get("image_path")); pres = {}; ok = True
        for les in LES:
            p = mask_present(f"data/FGADR/Seg-set/{FG[les]}/{st}.png")
            if p is None: ok = False; break
            pres[les] = p
        if ok: rows.append((grade(r), pres))
    print(f"FGADR with GT mask+grade: {len(rows)}")

    grades = [0, 1, 2, 3, 4]; ns = []; rate = np.zeros((len(LES), len(grades)))
    for j, g in enumerate(grades):
        sub = [p for gg, p in rows if gg == g]; ns.append(len(sub))
        for i, l in enumerate(LES): rate[i, j] = sum(p[l] for p in sub) / len(sub)

    plt.figure(figsize=(7.2, 4.2)); x = np.arange(len(grades)); w = 0.2
    for i, (l, c) in enumerate(zip(LES, ["#4C72B0", "#DD8452", "#55A868", "#C44E52"])):
        plt.bar(x + (i - 1.5) * w, rate[i], w, label=l, color=c)
    plt.xticks(x, [f"G{g}\n(n={ns[j]})" for j, g in enumerate(grades)])
    plt.ylabel("Lesion positive rate"); plt.ylim(0, 1.05)
    plt.title(f"Lesion positive rate per DR grade (FGADR, GT masks, n={len(rows)})")
    plt.legend(title="Lesion", ncol=4, loc="upper left", frameon=False)
    plt.axvspan(1.5, 4.5, color="grey", alpha=0.07)
    plt.text(3, 1.0, "G2-G4: MA/HE/EX/SE profiles overlap\n-> not separable by presence alone",
             ha="center", va="top", fontsize=8, color="#555")
    plt.tight_layout(); os.makedirs(os.path.dirname(OUT) or ".", exist_ok=True)
    plt.savefig(OUT, dpi=160); print("saved", OUT)

    # faithful ceiling: best presence->grade map, 4-class (No-DR/Mild/Mod/Sev=3&4)
    def key(p): return "".join(l for l in LES if p[l]) or "none"
    def cls(g): return min(g, 3)
    pat = defaultdict(Counter)
    for g, p in rows:
        if g is not None: pat[key(p)][cls(g)] += 1
    tot = sum(sum(c.values()) for c in pat.values())
    correct = sum(c.most_common(1)[0][1] for c in pat.values())
    print(f"faithful ceiling (best presence->grade, 4-class) = {correct}/{tot} = {correct/tot:.3f}")
    for k in ["MAHEEX", "HEEX", "MAHEEXSE", "MA", "none"]:
        g = defaultdict(int)
        for gg, p in rows:
            if key(p) == k and gg is not None: g[gg] += 1
        print(f"  {k:<9} G0-4 = {[g.get(i,0) for i in range(5)]}")


if __name__ == "__main__":
    main()
