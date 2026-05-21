from __future__ import annotations

import argparse
import json
import math
import os
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .common import CropMetaRow, atomic_write_json, now_iso, read_jsonl
from .retsam_json import parse_fovea_xy, parse_lesion_metrics, parse_od_info


LESIONS = ["HE", "EX", "SE"]
QUADS = ["TS", "NS", "TI", "NI"]


@dataclass
class ImageRow:
    image_id: str
    grade: int
    qa_path: str
    he_count: int
    he_area: float
    ex_count: int
    ex_area: float
    se_count: int
    se_area: float
    od_radius_px: float | None
    has_fovea: bool


def _safe_load_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _dist(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _savefig(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def _bar_present_rate(
    rows: list[ImageRow], out_dir: Path, *, title: str, by_grade: bool
) -> dict[str, Any]:
    def present(r: ImageRow, lesion: str) -> int:
        if lesion == "HE":
            return 1 if (r.he_count > 0 or r.he_area > 0) else 0
        if lesion == "EX":
            return 1 if (r.ex_count > 0 or r.ex_area > 0) else 0
        if lesion == "SE":
            return 1 if (r.se_count > 0 or r.se_area > 0) else 0
        raise ValueError(lesion)

    stats: dict[str, Any] = {}
    if not by_grade:
        denom = max(1, len(rows))
        rates = {k: sum(present(r, k) for r in rows) / denom for k in LESIONS}
        plt.figure(figsize=(6.2, 3.6))
        plt.title(title)
        plt.ylim(0, 1.0)
        plt.bar(list(rates.keys()), list(rates.values()))
        for i, k in enumerate(LESIONS):
            plt.text(i, rates[k] + 0.02, f"{rates[k]*100:.1f}%", ha="center", fontsize=10)
        _savefig(out_dir / "present_rate_overall.png")
        stats["overall"] = rates
        return stats

    grades = sorted({r.grade for r in rows})
    denom_by = {g: max(1, sum(1 for r in rows if r.grade == g)) for g in grades}
    rates_by_grade: dict[int, dict[str, float]] = {}
    for g in grades:
        subset = [r for r in rows if r.grade == g]
        denom = denom_by[g]
        rates_by_grade[g] = {k: sum(present(r, k) for r in subset) / denom for k in LESIONS}

    plt.figure(figsize=(7.4, 4.0))
    plt.title(title)
    x = list(range(len(grades)))
    w = 0.25
    for j, lesion in enumerate(LESIONS):
        ys = [rates_by_grade[g][lesion] for g in grades]
        plt.bar([i + (j - 1) * w for i in x], ys, width=w, label=lesion)
    plt.xticks(x, [str(g) for g in grades])
    plt.ylim(0, 1.0)
    plt.legend()
    _savefig(out_dir / "present_rate_by_grade.png")
    stats["by_grade"] = {str(g): rates_by_grade[g] for g in grades}
    stats["grade_denoms"] = {str(g): denom_by[g] for g in grades}
    return stats


def _hist(
    values: list[float],
    out_path: Path,
    *,
    title: str,
    xlabel: str,
    bins: int = 30,
    log1p: bool = False,
) -> dict[str, Any]:
    xs = [float(v) for v in values if v is not None and not math.isnan(float(v))]
    if log1p:
        xs_plot = [math.log1p(max(0.0, x)) for x in xs]
        xlabel_plot = f"log1p({xlabel})"
    else:
        xs_plot = xs
        xlabel_plot = xlabel
    plt.figure(figsize=(7.0, 3.8))
    plt.title(title)
    if xs_plot:
        plt.hist(xs_plot, bins=bins)
    plt.xlabel(xlabel_plot)
    plt.ylabel("count")
    _savefig(out_path)
    return {
        "n": len(xs),
        "min": min(xs) if xs else None,
        "p50": _percentile(xs, 50) if xs else None,
        "p90": _percentile(xs, 90) if xs else None,
        "p99": _percentile(xs, 99) if xs else None,
        "max": max(xs) if xs else None,
    }


def _percentile(xs: list[float], p: float) -> float | None:
    if not xs:
        return None
    ys = sorted(xs)
    k = int(round((p / 100.0) * (len(ys) - 1)))
    k = max(0, min(len(ys) - 1, k))
    return float(ys[k])


def _quad_stacked(
    quad_counts: dict[str, dict[str, int]],
    out_path: Path,
    *,
    title: str,
) -> None:
    plt.figure(figsize=(7.0, 3.8))
    plt.title(title)
    bottoms = [0] * len(QUADS)
    for lesion in LESIONS:
        ys = [int(quad_counts.get(lesion, {}).get(q, 0)) for q in QUADS]
        plt.bar(QUADS, ys, bottom=bottoms, label=lesion)
        bottoms = [bottoms[i] + ys[i] for i in range(len(QUADS))]
    plt.ylabel("count (sum of per-image quadrant counts)")
    plt.legend()
    _savefig(out_path)


def build_report(
    *,
    dataset: str,
    crop_meta_path: Path,
    outputs_root: Path,
    out_dir: Path,
    max_examples: int = 50,
) -> None:
    _ensure_dir(out_dir)

    crop_rows = [CropMetaRow.from_obj(o) for o in read_jsonl(crop_meta_path)]
    id2grade: dict[str, int] = {r.image_id: r.grade for r in crop_rows}

    qa_paths = sorted(outputs_root.glob("*/quantitative_analysis.json"))
    rows: list[ImageRow] = []
    bad_json = 0
    bad_json_paths: list[str] = []
    missing_grade = 0

    quad_counts: dict[str, dict[str, int]] = {k: {q: 0 for q in QUADS} for k in LESIONS}
    missing_fovea = 0
    missing_od = 0
    od_radii: list[float] = []

    lesion_counts: dict[str, list[int]] = {k: [] for k in LESIONS}
    lesion_areas: dict[str, list[float]] = {k: [] for k in LESIONS}

    # distance bands in "OD diameter units" (d/OD_diam)
    dd_bands = [0.0, 1.0, 2.0, 3.0, 4.0, 999.0]
    dd_band_labels = ["<1DD", "1-2DD", "2-3DD", "3-4DD", ">=4DD"]
    dd_band_counts = {k: Counter() for k in LESIONS}

    for p in qa_paths:
        image_id = p.parent.name
        grade = id2grade.get(image_id)
        if grade is None:
            missing_grade += 1
            continue

        q = _safe_load_json(p)
        if q is None or not isinstance(q, dict):
            bad_json += 1
            bad_json_paths.append(p.as_posix())
            continue

        m = {k: parse_lesion_metrics(q, k) for k in LESIONS}
        he = m["HE"]
        ex = m["EX"]
        se = m["SE"]

        for lesion, met in m.items():
            if met is None:
                lesion_counts[lesion].append(0)
                lesion_areas[lesion].append(0.0)
                continue
            lesion_counts[lesion].append(int(met.count))
            lesion_areas[lesion].append(float(met.total_area))
            for qk in QUADS:
                quad_counts[lesion][qk] += int(met.quadrant_distribution.get(qk, 0))

        fovea = parse_fovea_xy(q)
        if fovea is None:
            missing_fovea += 1

        od = parse_od_info(q)
        if od.od_radius_px is None:
            missing_od += 1
        else:
            od_r = float(od.od_radius_px)
            od_radii.append(od_r)

        # fovea distance bands per lesion centroid if possible
        if fovea is not None and od.od_radius_px is not None and od.od_radius_px > 1e-6:
            od_d = float(od.od_radius_px) * 2.0
            for lesion, met in m.items():
                if met is None or not met.centroids:
                    continue
                for c in met.centroids:
                    d = _dist(fovea, c) / od_d
                    # assign to band
                    for i in range(len(dd_bands) - 1):
                        if dd_bands[i] <= d < dd_bands[i + 1]:
                            dd_band_counts[lesion][dd_band_labels[i]] += 1
                            break

        rows.append(
            ImageRow(
                image_id=image_id,
                grade=int(grade),
                qa_path=p.as_posix(),
                he_count=int(he.count) if he else 0,
                he_area=float(he.total_area) if he else 0.0,
                ex_count=int(ex.count) if ex else 0,
                ex_area=float(ex.total_area) if ex else 0.0,
                se_count=int(se.count) if se else 0,
                se_area=float(se.total_area) if se else 0.0,
                od_radius_px=float(od.od_radius_px) if od.od_radius_px is not None else None,
                has_fovea=bool(fovea is not None),
            )
        )

    # Figures
    fig_dir = out_dir / "figures"
    _ensure_dir(fig_dir)

    present_stats = {}
    present_stats.update(_bar_present_rate(rows, fig_dir, title=f"{dataset}: lesion present rate (overall)", by_grade=False))
    present_stats.update(_bar_present_rate(rows, fig_dir, title=f"{dataset}: lesion present rate (by grade)", by_grade=True))

    # count / area hists
    hist_stats: dict[str, Any] = {"count": {}, "area": {}, "od_radius_px": None}
    for lesion in LESIONS:
        hist_stats["count"][lesion] = _hist(
            [float(x) for x in lesion_counts[lesion]],
            fig_dir / f"{lesion.lower()}_count_hist.png",
            title=f"{dataset}: {lesion} count histogram",
            xlabel=f"{lesion}_count",
            bins=40,
            log1p=True,
        )
        hist_stats["area"][lesion] = _hist(
            [float(x) for x in lesion_areas[lesion]],
            fig_dir / f"{lesion.lower()}_area_hist.png",
            title=f"{dataset}: {lesion} total area histogram (px^2)",
            xlabel=f"{lesion}_area_px2",
            bins=40,
            log1p=True,
        )

    hist_stats["od_radius_px"] = _hist(
        od_radii,
        fig_dir / "od_radius_hist.png",
        title=f"{dataset}: optic disc radius (px) histogram",
        xlabel="od_radius_px",
        bins=50,
        log1p=False,
    )

    _quad_stacked(quad_counts, fig_dir / "quadrant_stacked.png", title=f"{dataset}: quadrant distribution (sum)")

    # distance band plot (stacked bars per lesion)
    plt.figure(figsize=(7.4, 3.8))
    plt.title(f"{dataset}: lesion centroid fovea-distance bands (in OD diameters)")
    x = list(range(len(dd_band_labels)))
    w = 0.25
    for j, lesion in enumerate(LESIONS):
        ys = [dd_band_counts[lesion].get(lbl, 0) for lbl in dd_band_labels]
        plt.bar([i + (j - 1) * w for i in x], ys, width=w, label=lesion)
    plt.xticks(x, dd_band_labels)
    plt.ylabel("centroid count")
    plt.legend()
    _savefig(fig_dir / "fovea_distance_bands.png")

    # Top examples by area per lesion
    top_examples: dict[str, list[dict[str, Any]]] = {}
    for lesion in LESIONS:
        key = f"{lesion.lower()}_area"
        scored = sorted(rows, key=lambda r: getattr(r, key), reverse=True)
        top = scored[: max_examples]
        top_examples[lesion] = [
            {"image_id": r.image_id, "grade": r.grade, "area_px2": float(getattr(r, key)), "qa_path": r.qa_path}
            for r in top
        ]

    # Write JSON summary
    summary = {
        "generated_at": now_iso(),
        "dataset": dataset,
        "crop_meta_path": crop_meta_path.as_posix(),
        "outputs_root": outputs_root.as_posix(),
        "n_images": len(rows),
        "n_qa_files": len(qa_paths),
        "bad_json": bad_json,
        "bad_json_paths": bad_json_paths,
        "missing_grade": missing_grade,
        "missing_fovea": missing_fovea,
        "missing_od": missing_od,
        "present_rates": present_stats,
        "hist_stats": hist_stats,
        "quadrant_counts": quad_counts,
        "fovea_distance_bands": {k: dict(v) for k, v in dd_band_counts.items()},
        "top_examples_by_area": top_examples,
    }
    atomic_write_json(out_dir / "stats.json", summary, indent=2)

    # Write a simple HTML report (no external deps)
    figs = [
        ("present_rate_overall.png", "Lesion present rate (overall)"),
        ("present_rate_by_grade.png", "Lesion present rate (by grade)"),
        ("quadrant_stacked.png", "Quadrant distribution (sum)"),
        ("fovea_distance_bands.png", "Fovea distance bands (centroids)"),
        ("od_radius_hist.png", "Optic disc radius histogram"),
        ("he_count_hist.png", "HE count histogram"),
        ("ex_count_hist.png", "EX count histogram"),
        ("se_count_hist.png", "SE count histogram"),
        ("he_area_hist.png", "HE total area histogram"),
        ("ex_area_hist.png", "EX total area histogram"),
        ("se_area_hist.png", "SE total area histogram"),
    ]

    def esc(s: str) -> str:
        return (
            s.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
            .replace("'", "&#39;")
        )

    html_parts = [
        "<!doctype html>",
        "<html><head><meta charset='utf-8'/>",
        f"<title>RetSAM report: {esc(dataset)}</title>",
        "<style>body{font-family:system-ui, -apple-system, Segoe UI, Roboto, Arial; margin: 24px;} ",
        "code,pre{font-family:ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;} ",
        "img{max-width: 100%; border: 1px solid #eee;} ",
        ".grid{display:grid; grid-template-columns: 1fr; gap: 16px;} ",
        ".card{padding:14px; border:1px solid #e6e6e6; border-radius:10px;} ",
        "table{border-collapse:collapse;} td,th{border:1px solid #ddd; padding:6px 8px;} ",
        "</style></head><body>",
        f"<h2>RetSAM statistics report: {esc(dataset)}</h2>",
        "<div class='card'><pre>",
        esc(json.dumps({k: summary[k] for k in ['generated_at','dataset','n_images','bad_json','missing_grade','missing_fovea','missing_od']}, indent=2, ensure_ascii=False)),
        "</pre>",
        f"<div>Full JSON: <code>{esc((out_dir / 'stats.json').name)}</code></div>",
        "</div>",
        "<div class='grid'>",
    ]
    for fn, cap in figs:
        html_parts.append("<div class='card'>")
        html_parts.append(f"<div style='font-weight:700; margin-bottom:8px;'>{esc(cap)}</div>")
        html_parts.append(f"<img src='figures/{esc(fn)}'/>")
        html_parts.append("</div>")
    html_parts.append("</div>")

    # top examples
    html_parts.append("<h3>Top examples by lesion area</h3>")
    for lesion in LESIONS:
        html_parts.append(f"<h4>{esc(lesion)}</h4>")
        html_parts.append("<div class='card'><table>")
        html_parts.append("<tr><th>rank</th><th>image_id</th><th>grade</th><th>area_px2</th><th>qa_path</th></tr>")
        for i, it in enumerate(top_examples[lesion][: min(max_examples, 20)]):
            html_parts.append(
                "<tr>"
                f"<td>{i+1}</td>"
                f"<td><code>{esc(it['image_id'])}</code></td>"
                f"<td>{int(it['grade'])}</td>"
                f"<td>{float(it['area_px2']):.1f}</td>"
                f"<td><code>{esc(it['qa_path'])}</code></td>"
                "</tr>"
            )
        html_parts.append("</table></div>")

    html_parts.append("</body></html>")
    (out_dir / "report.html").write_text("\n".join(html_parts), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="aptos", choices=["aptos", "ddr_grading"])
    ap.add_argument("--crop-meta", default="", help="Override crop_meta.jsonl path")
    ap.add_argument("--outputs-root", default="", help="Override outputs dir (retsam outputs)")
    ap.add_argument("--out-dir", default="", help="Override report output dir")
    ap.add_argument("--max-examples", type=int, default=50)
    args = ap.parse_args()

    repo = Path(__file__).resolve().parents[2]
    dataset = str(args.dataset)

    if args.crop_meta:
        crop_meta = Path(args.crop_meta)
    else:
        crop_meta = repo / "data" / "cropped" / dataset / "crop_meta.jsonl"

    if args.outputs_root:
        outputs_root = Path(args.outputs_root)
    else:
        outputs_root = repo / "outputs" / f"retsam_{dataset}"

    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        out_dir = repo / "reports" / f"retsam_{dataset}_stats"

    build_report(
        dataset=dataset,
        crop_meta_path=crop_meta,
        outputs_root=outputs_root,
        out_dir=out_dir,
        max_examples=int(args.max_examples),
    )

    print(f"[OK] wrote report to {out_dir}")
    print(f"[OK] open {out_dir / 'report.html'}")


if __name__ == "__main__":
    main()

