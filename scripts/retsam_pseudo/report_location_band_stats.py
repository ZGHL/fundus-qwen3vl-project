from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Allow running this file directly: add repo root to sys.path.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.retsam_pseudo.common import atomic_write_json, now_iso, read_jsonl  # noqa: E402


BANDS = ["黄斑区", "后极部", "中周部", "周边部", "未见", "未知"]
LESIONS = ["HE", "EX", "SE"]


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _savefig(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def _get_band(out_json: dict[str, Any], lesion: str) -> str:
    blk = out_json.get(lesion) or {}
    if not isinstance(blk, dict):
        return "未知"
    if not blk.get("present"):
        return "未见"
    b = blk.get("location_band")
    if isinstance(b, str) and b:
        return b
    return "未知"


def _bar_dist(counter: Counter[str], title: str, out_png: Path) -> dict[str, Any]:
    total = sum(counter.values())
    xs = [b for b in BANDS if b in counter]
    ys = [counter[b] for b in xs]
    plt.figure(figsize=(7.8, 3.6))
    plt.title(title)
    plt.bar(xs, ys)
    plt.ylabel("images")
    for i, b in enumerate(xs):
        pct = (ys[i] / total * 100.0) if total else 0.0
        plt.text(i, ys[i] + max(1, total * 0.005), f"{pct:.1f}%", ha="center", fontsize=10)
    _savefig(out_png)
    return {"total": total, "counts": dict(counter), "rates": {k: (counter[k] / total if total else 0.0) for k in counter}}


def summarize(pseudo_cot_jsonl: Path) -> dict[str, Any]:
    rows = read_jsonl(pseudo_cot_jsonl)
    # overall and by-grade distributions
    overall: dict[str, Counter[str]] = {k: Counter() for k in LESIONS}
    by_grade: dict[int, dict[str, Counter[str]]] = defaultdict(lambda: {k: Counter() for k in LESIONS})

    n = 0
    for r in rows:
        outj = r.get("output_json") or {}
        if not isinstance(outj, dict):
            continue
        grade = int(r.get("grade", -1))
        n += 1
        for lesion in LESIONS:
            band = _get_band(outj, lesion)
            overall[lesion][band] += 1
            by_grade[grade][lesion][band] += 1

    return {"n_rows": n, "overall": overall, "by_grade": by_grade}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--aptos-jsonl", required=True)
    ap.add_argument("--ddr-jsonl", required=True)
    ap.add_argument("--out-dir", default="reports/location_band_stats")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    fig_dir = out_dir / "figures"
    _ensure_dir(fig_dir)

    aptos_path = Path(args.aptos_jsonl)
    ddr_path = Path(args.ddr_jsonl)

    apt = summarize(aptos_path)
    ddr = summarize(ddr_path)

    # Make plots
    plots: dict[str, Any] = {"aptos": {"overall": {}}, "ddr": {"overall": {}}}
    for lesion in LESIONS:
        plots["aptos"]["overall"][lesion] = _bar_dist(
            apt["overall"][lesion],
            f"APTOS: {lesion} location_band distribution",
            fig_dir / f"aptos_{lesion.lower()}_band.png",
        )
        plots["ddr"]["overall"][lesion] = _bar_dist(
            ddr["overall"][lesion],
            f"DDR: {lesion} location_band distribution",
            fig_dir / f"ddr_{lesion.lower()}_band.png",
        )

    # by-grade plots (stacked bars)
    def stacked(dataset_name: str, by_grade: dict[int, dict[str, Counter[str]]]) -> None:
        grades = sorted([g for g in by_grade.keys() if g >= 0])
        for lesion in LESIONS:
            plt.figure(figsize=(8.4, 3.8))
            plt.title(f"{dataset_name}: {lesion} location_band by grade")
            bottoms = [0] * len(grades)
            for band in ["黄斑区", "后极部", "中周部", "周边部", "未见", "未知"]:
                ys = [by_grade[g][lesion].get(band, 0) for g in grades]
                plt.bar([str(g) for g in grades], ys, bottom=bottoms, label=band)
                bottoms = [bottoms[i] + ys[i] for i in range(len(ys))]
            plt.xlabel("grade")
            plt.ylabel("images")
            plt.legend(ncol=3, fontsize=9)
            _savefig(fig_dir / f"{dataset_name.lower()}_{lesion.lower()}_band_by_grade.png")

    stacked("APTOS", apt["by_grade"])
    stacked("DDR", ddr["by_grade"])

    # Write summary JSON + HTML
    summary = {
        "generated_at": now_iso(),
        "inputs": {"aptos": str(aptos_path), "ddr": str(ddr_path)},
        "aptos_n": apt["n_rows"],
        "ddr_n": ddr["n_rows"],
        "aptos_overall_counts": {k: dict(apt["overall"][k]) for k in LESIONS},
        "ddr_overall_counts": {k: dict(ddr["overall"][k]) for k in LESIONS},
    }
    atomic_write_json(out_dir / "summary.json", summary, indent=2)

    def esc(s: str) -> str:
        return (
            s.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
            .replace("'", "&#39;")
        )

    html = [
        "<!doctype html>",
        "<html><head><meta charset='utf-8'/>",
        "<title>location_band stats</title>",
        "<style>body{font-family:system-ui,-apple-system,Segoe UI,Roboto,Arial;margin:24px;}",
        "code,pre{font-family:ui-monospace,SFMono-Regular,Menlo,Monaco,Consolas,monospace;}",
        "img{max-width:100%;border:1px solid #eee;} .grid{display:grid;grid-template-columns:1fr;gap:16px;}",
        ".card{padding:14px;border:1px solid #e6e6e6;border-radius:10px;}</style>",
        "</head><body>",
        "<h2>location_band distribution (APTOS vs DDR)</h2>",
        "<div class='card'><pre>",
        esc(json.dumps(summary, ensure_ascii=False, indent=2)),
        "</pre></div>",
        "<div class='grid'>",
    ]
    for lesion in LESIONS:
        html.append("<div class='card'>")
        html.append(f"<div style='font-weight:700;margin-bottom:8px;'>APTOS {lesion}</div>")
        html.append(f"<img src='figures/aptos_{lesion.lower()}_band.png'/>")
        html.append("</div>")
        html.append("<div class='card'>")
        html.append(f"<div style='font-weight:700;margin-bottom:8px;'>DDR {lesion}</div>")
        html.append(f"<img src='figures/ddr_{lesion.lower()}_band.png'/>")
        html.append("</div>")
        html.append("<div class='card'>")
        html.append(f"<div style='font-weight:700;margin-bottom:8px;'>APTOS {lesion} by grade</div>")
        html.append(f"<img src='figures/aptos_{lesion.lower()}_band_by_grade.png'/>")
        html.append("</div>")
        html.append("<div class='card'>")
        html.append(f"<div style='font-weight:700;margin-bottom:8px;'>DDR {lesion} by grade</div>")
        html.append(f"<img src='figures/ddr_{lesion.lower()}_band_by_grade.png'/>")
        html.append("</div>")
    html.append("</div></body></html>")
    (out_dir / "report.html").write_text("\n".join(html), encoding="utf-8")

    print(f"[OK] wrote {out_dir / 'report.html'}")


if __name__ == "__main__":
    main()

