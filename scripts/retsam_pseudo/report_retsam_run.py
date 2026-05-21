from __future__ import annotations

import argparse
import json
import math
import os
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .common import CropMetaRow, KeptIndexRow, atomic_write_json, now_iso, read_jsonl
from .retsam_json import parse_lesion_metrics


LESIONS = ["HE", "EX", "SE"]
QUADS = ["TS", "NS", "TI", "NI"]


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _savefig(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def _safe_load_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _percentile(xs: list[float], p: float) -> float | None:
    if not xs:
        return None
    ys = sorted(xs)
    k = int(round((p / 100.0) * (len(ys) - 1)))
    k = max(0, min(len(ys) - 1, k))
    return float(ys[k])


def _summ(xs: list[float]) -> dict[str, Any]:
    xs2 = [float(x) for x in xs if x is not None and not math.isnan(float(x))]
    return {
        "n": len(xs2),
        "min": min(xs2) if xs2 else None,
        "p50": _percentile(xs2, 50) if xs2 else None,
        "p90": _percentile(xs2, 90) if xs2 else None,
        "p99": _percentile(xs2, 99) if xs2 else None,
        "max": max(xs2) if xs2 else None,
        "mean": (sum(xs2) / len(xs2)) if xs2 else None,
    }


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def _confidence(area: float, count: int, *, area_min: float) -> float:
    """
    Heuristic confidence in [0,1], based on how far area & count are above thresholds.
    - No explicit RetSAM probability is available in quantitative_analysis.json (for this schema),
      so we build an interpretable score consistent with filtering heuristics.
    """
    if count <= 0 or area <= 0:
        return 0.0
    # Area term: >= area_min gives >0; saturates around 10x.
    a = math.log1p(max(0.0, area / max(1e-6, area_min))) / math.log1p(10.0)
    # Count term: saturates around 20.
    c = math.log1p(float(max(0, count))) / math.log1p(20.0)
    return _clamp01(a) * _clamp01(c)


@dataclass(frozen=True)
class PerImage:
    image_id: str
    grade: int
    qa_path: str
    counts: dict[str, int]
    areas: dict[str, float]
    quads: dict[str, dict[str, int]]
    valid: dict[str, bool]
    conf: dict[str, float]


def build_run_report(
    *,
    dataset: str,
    crop_meta_path: Path,
    outputs_root: Path,
    kept_index_path: Path,
    filter_stats_path: Path,
    filter_errors_path: Path,
    crop_log_path: Path | None,
    run_log_path: Path | None,
    out_dir: Path,
) -> None:
    _ensure_dir(out_dir)
    fig_dir = out_dir / "figures"
    _ensure_dir(fig_dir)

    crop_rows = [CropMetaRow.from_obj(o) for o in read_jsonl(crop_meta_path)]
    grade_by_id = {r.image_id: int(r.grade) for r in crop_rows}

    kept_rows = [KeptIndexRow.from_obj(o) for o in read_jsonl(kept_index_path)]
    kept_by_id = {r.image_id: r for r in kept_rows}

    filter_stats = json.loads(filter_stats_path.read_text(encoding="utf-8"))
    thresholds = filter_stats.get("thresholds") or {}
    area_min = {
        "HE": float(thresholds.get("he_area_min", 100.0)),
        "EX": float(thresholds.get("ex_area_min", 100.0)),
        "SE": float(thresholds.get("se_area_min", 200.0)),
    }

    qa_paths = sorted(outputs_root.glob("*/quantitative_analysis.json"))

    # Aggregations
    bad_json: list[str] = []
    per: list[PerImage] = []

    present_raw = {k: 0 for k in LESIONS}
    present_valid = {k: 0 for k in LESIONS}
    quad_raw = {k: {q: 0 for q in QUADS} for k in LESIONS}
    quad_valid = {k: {q: 0 for q in QUADS} for k in LESIONS}

    conf_all: dict[str, list[float]] = {k: [] for k in LESIONS}
    conf_valid: dict[str, list[float]] = {k: [] for k in LESIONS}
    conf_by_grade: dict[int, dict[str, list[float]]] = defaultdict(lambda: {k: [] for k in LESIONS})

    for p in qa_paths:
        image_id = p.parent.name
        grade = grade_by_id.get(image_id)
        if grade is None:
            continue
        q = _safe_load_json(p)
        if q is None:
            bad_json.append(p.as_posix())
            continue

        m = {k: parse_lesion_metrics(q, k) for k in LESIONS}
        counts = {k: int(m[k].count) if m[k] else 0 for k in LESIONS}
        areas = {k: float(m[k].total_area) if m[k] else 0.0 for k in LESIONS}
        quads = {k: (m[k].quadrant_distribution if m[k] else {q: 0 for q in QUADS}) for k in LESIONS}

        valid_row = kept_by_id.get(image_id)
        valid = {
            "HE": bool(valid_row.he_valid) if valid_row else False,
            "EX": bool(valid_row.ex_valid) if valid_row else False,
            "SE": bool(valid_row.se_valid) if valid_row else False,
        }
        conf = {k: _confidence(areas[k], counts[k], area_min=area_min[k]) for k in LESIONS}

        for lesion in LESIONS:
            if counts[lesion] > 0 or areas[lesion] > 0:
                present_raw[lesion] += 1
                for qk in QUADS:
                    quad_raw[lesion][qk] += int(quads[lesion].get(qk, 0))
            if valid[lesion]:
                present_valid[lesion] += 1
                for qk in QUADS:
                    quad_valid[lesion][qk] += int(quads[lesion].get(qk, 0))

            conf_all[lesion].append(float(conf[lesion]))
            conf_by_grade[int(grade)][lesion].append(float(conf[lesion]))
            if valid[lesion]:
                conf_valid[lesion].append(float(conf[lesion]))

        per.append(
            PerImage(
                image_id=image_id,
                grade=int(grade),
                qa_path=p.as_posix(),
                counts=counts,
                areas=areas,
                quads={k: {q: int(quads[k].get(q, 0)) for q in QUADS} for k in LESIONS},
                valid=valid,
                conf=conf,
            )
        )

    n = max(1, len(per))
    present_rate_raw = {k: present_raw[k] / n for k in LESIONS}
    present_rate_valid = {k: present_valid[k] / n for k in LESIONS}

    # Plots: confidence hist
    for lesion in LESIONS:
        plt.figure(figsize=(7.0, 3.6))
        plt.title(f"{dataset}: {lesion} heuristic confidence (all images)")
        plt.hist(conf_all[lesion], bins=40, range=(0.0, 1.0))
        plt.xlabel("confidence (0-1)")
        plt.ylabel("images")
        _savefig(fig_dir / f"{lesion.lower()}_confidence_hist_all.png")

        plt.figure(figsize=(7.0, 3.6))
        plt.title(f"{dataset}: {lesion} heuristic confidence (kept/valid only)")
        plt.hist(conf_valid[lesion], bins=40, range=(0.0, 1.0))
        plt.xlabel("confidence (0-1)")
        plt.ylabel("images")
        _savefig(fig_dir / f"{lesion.lower()}_confidence_hist_valid.png")

    # Confidence by grade (mean)
    grades = sorted(conf_by_grade.keys())
    plt.figure(figsize=(7.6, 3.8))
    plt.title(f"{dataset}: mean heuristic confidence by grade")
    x = list(range(len(grades)))
    w = 0.25
    for j, lesion in enumerate(LESIONS):
        ys = []
        for g in grades:
            xs = conf_by_grade[g][lesion]
            ys.append((sum(xs) / max(1, len(xs))) if xs else 0.0)
        plt.bar([i + (j - 1) * w for i in x], ys, width=w, label=lesion)
    plt.xticks(x, [str(g) for g in grades])
    plt.ylim(0, 1.0)
    plt.ylabel("mean confidence")
    plt.legend()
    _savefig(fig_dir / "confidence_mean_by_grade.png")

    # Quadrant stacked raw vs valid
    def stacked(quad: dict[str, dict[str, int]], title: str, out: Path) -> None:
        plt.figure(figsize=(7.0, 3.8))
        plt.title(title)
        bottoms = [0] * len(QUADS)
        for lesion in LESIONS:
            ys = [int(quad[lesion][q]) for q in QUADS]
            plt.bar(QUADS, ys, bottom=bottoms, label=lesion)
            bottoms = [bottoms[i] + ys[i] for i in range(len(QUADS))]
        plt.ylabel("count (sum of per-image quadrant counts)")
        plt.legend()
        _savefig(out)

    stacked(quad_raw, f"{dataset}: quadrant distribution (raw)", fig_dir / "quadrant_stacked_raw.png")
    stacked(quad_valid, f"{dataset}: quadrant distribution (valid/kept)", fig_dir / "quadrant_stacked_valid.png")

    # Presence rate bar: raw vs valid
    plt.figure(figsize=(7.0, 3.6))
    plt.title(f"{dataset}: lesion present rate (raw vs valid)")
    x = list(range(len(LESIONS)))
    plt.bar([i - 0.18 for i in x], [present_rate_raw[k] for k in LESIONS], width=0.35, label="raw")
    plt.bar([i + 0.18 for i in x], [present_rate_valid[k] for k in LESIONS], width=0.35, label="valid/kept")
    plt.xticks(x, LESIONS)
    plt.ylim(0, 1.0)
    plt.legend()
    _savefig(fig_dir / "present_rate_raw_vs_valid.png")

    # Top/bottom examples by confidence among valid
    top_examples: dict[str, list[dict[str, Any]]] = {}
    bottom_examples: dict[str, list[dict[str, Any]]] = {}
    for lesion in LESIONS:
        valid_imgs = [r for r in per if r.valid.get(lesion)]
        valid_imgs.sort(key=lambda r: float(r.conf.get(lesion, 0.0)), reverse=True)
        top_examples[lesion] = [
            {
                "image_id": r.image_id,
                "grade": r.grade,
                "confidence": float(r.conf[lesion]),
                "count": int(r.counts[lesion]),
                "area_px2": float(r.areas[lesion]),
                "qa_path": r.qa_path,
            }
            for r in valid_imgs[:20]
        ]
        bottom_examples[lesion] = [
            {
                "image_id": r.image_id,
                "grade": r.grade,
                "confidence": float(r.conf[lesion]),
                "count": int(r.counts[lesion]),
                "area_px2": float(r.areas[lesion]),
                "qa_path": r.qa_path,
            }
            for r in valid_imgs[-20:]
        ]

    # Read optional logs
    crop_log = None
    if crop_log_path and crop_log_path.is_file():
        crop_log = crop_log_path.read_text(encoding="utf-8", errors="ignore").strip().splitlines()[-1:]
        crop_log = crop_log[0] if crop_log else None
    run_log_tail = None
    if run_log_path and run_log_path.is_file():
        lines = run_log_path.read_text(encoding="utf-8", errors="ignore").strip().splitlines()
        run_log_tail = lines[-1] if lines else None

    filter_errors = read_jsonl(filter_errors_path) if filter_errors_path.is_file() else []

    summary = {
        "generated_at": now_iso(),
        "dataset": dataset,
        "paths": {
            "crop_meta": crop_meta_path.as_posix(),
            "outputs_root": outputs_root.as_posix(),
            "kept_index": kept_index_path.as_posix(),
            "filter_stats": filter_stats_path.as_posix(),
            "filter_errors": filter_errors_path.as_posix(),
            "crop_log": crop_log_path.as_posix() if crop_log_path else None,
            "run_log": run_log_path.as_posix() if run_log_path else None,
        },
        "counts": {
            "n_crop_meta_rows": len(crop_rows),
            "n_qa_files": len(qa_paths),
            "n_images_ok": len(per),
            "bad_json": len(bad_json),
        },
        "bad_json_paths": bad_json,
        "filter_thresholds": thresholds,
        "filter_keep_rates": filter_stats.get("lesion_keep_rates"),
        "present_rate_raw": present_rate_raw,
        "present_rate_valid": present_rate_valid,
        "quadrant_counts_raw": quad_raw,
        "quadrant_counts_valid": quad_valid,
        "confidence_summary_all": {k: _summ(conf_all[k]) for k in LESIONS},
        "confidence_summary_valid": {k: _summ(conf_valid[k]) for k in LESIONS},
        "top_examples_by_confidence_valid": top_examples,
        "bottom_examples_by_confidence_valid": bottom_examples,
        "filter_errors": filter_errors,
        "crop_log_last_line": crop_log,
        "run_log_last_line": run_log_tail,
    }
    atomic_write_json(out_dir / "run_report.json", summary, indent=2)

    # HTML (no external deps)
    figs = [
        ("present_rate_raw_vs_valid.png", "Lesion present rate (raw vs valid/kept)"),
        ("quadrant_stacked_raw.png", "Quadrant distribution (raw, sum)"),
        ("quadrant_stacked_valid.png", "Quadrant distribution (valid/kept, sum)"),
        ("confidence_mean_by_grade.png", "Mean heuristic confidence by grade"),
        ("he_confidence_hist_all.png", "HE confidence histogram (all)"),
        ("he_confidence_hist_valid.png", "HE confidence histogram (valid only)"),
        ("ex_confidence_hist_all.png", "EX confidence histogram (all)"),
        ("ex_confidence_hist_valid.png", "EX confidence histogram (valid only)"),
        ("se_confidence_hist_all.png", "SE confidence histogram (all)"),
        ("se_confidence_hist_valid.png", "SE confidence histogram (valid only)"),
    ]

    def esc(s: str) -> str:
        return (
            s.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
            .replace("'", "&#39;")
        )

    head = {
        "generated_at": summary["generated_at"],
        "dataset": dataset,
        "n_qa_files": summary["counts"]["n_qa_files"],
        "n_images_ok": summary["counts"]["n_images_ok"],
        "bad_json": summary["counts"]["bad_json"],
        "present_rate_raw": present_rate_raw,
        "present_rate_valid": present_rate_valid,
        "filter_keep_rates": summary["filter_keep_rates"],
        "filter_thresholds": thresholds,
    }

    html = [
        "<!doctype html>",
        "<html><head><meta charset='utf-8'/>",
        f"<title>RetSAM run report: {esc(dataset)}</title>",
        "<style>body{font-family:system-ui, -apple-system, Segoe UI, Roboto, Arial; margin: 24px;} ",
        "code,pre{font-family:ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;} ",
        "img{max-width: 100%; border: 1px solid #eee;} ",
        ".grid{display:grid; grid-template-columns: 1fr; gap: 16px;} ",
        ".card{padding:14px; border:1px solid #e6e6e6; border-radius:10px;} ",
        "table{border-collapse:collapse; width:100%;} td,th{border:1px solid #ddd; padding:6px 8px; font-size:12px;} ",
        "</style></head><body>",
        f"<h2>RetSAM run report: {esc(dataset)}</h2>",
        "<div class='card'><pre>",
        esc(json.dumps(head, indent=2, ensure_ascii=False)),
        "</pre>",
        f"<div>JSON: <code>{esc('run_report.json')}</code></div>",
        "</div>",
        "<div class='grid'>",
    ]
    for fn, cap in figs:
        html.append("<div class='card'>")
        html.append(f"<div style='font-weight:700; margin-bottom:8px;'>{esc(cap)}</div>")
        html.append(f"<img src='figures/{esc(fn)}'/>")
        html.append("</div>")
    html.append("</div>")

    html.append("<h3>Bad JSON / errors</h3>")
    html.append("<div class='card'><pre>")
    html.append(esc(json.dumps({"bad_json_paths": bad_json, "filter_errors": filter_errors}, indent=2, ensure_ascii=False)))
    html.append("</pre></div>")

    def table_examples(title: str, ex: dict[str, list[dict[str, Any]]]) -> None:
        html.append(f"<h3>{esc(title)}</h3>")
        for lesion in LESIONS:
            html.append(f"<h4>{esc(lesion)}</h4>")
            html.append("<div class='card'><table>")
            html.append("<tr><th>rank</th><th>image_id</th><th>grade</th><th>confidence</th><th>count</th><th>area_px2</th><th>qa_path</th></tr>")
            for i, it in enumerate(ex[lesion]):
                html.append(
                    "<tr>"
                    f"<td>{i+1}</td>"
                    f"<td><code>{esc(str(it['image_id']))}</code></td>"
                    f"<td>{int(it['grade'])}</td>"
                    f"<td>{float(it['confidence']):.3f}</td>"
                    f"<td>{int(it['count'])}</td>"
                    f"<td>{float(it['area_px2']):.1f}</td>"
                    f"<td><code>{esc(str(it['qa_path']))}</code></td>"
                    "</tr>"
                )
            html.append("</table></div>")

    table_examples("Top confidence examples (valid only)", top_examples)
    table_examples("Bottom confidence examples (valid only)", bottom_examples)

    html.append("</body></html>")
    (out_dir / "report.html").write_text("\n".join(html), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="aptos", choices=["aptos", "ddr_grading"])
    ap.add_argument("--crop-meta", default="", help="Override crop_meta.jsonl path")
    ap.add_argument("--outputs-root", default="", help="Override outputs dir (retsam outputs)")
    ap.add_argument("--kept-index", default="", help="kept_index.jsonl path")
    ap.add_argument("--filter-stats", default="", help="filter_stats.json path")
    ap.add_argument("--filter-errors", default="", help="filter_errors.jsonl path")
    ap.add_argument("--crop-log", default="", help="crop log path (optional)")
    ap.add_argument("--run-log", default="", help="run log path (optional)")
    ap.add_argument("--out-dir", default="", help="Override report output dir")
    args = ap.parse_args()

    repo = Path(__file__).resolve().parents[2]
    dataset = str(args.dataset)

    crop_meta = Path(args.crop_meta) if args.crop_meta else (repo / "data" / "cropped" / dataset / "crop_meta.jsonl")
    outputs_root = Path(args.outputs_root) if args.outputs_root else (repo / "outputs" / f"retsam_{dataset}")

    # Default paths (prefer reports/.. if present)
    default_report_dir = repo / "reports" / f"retsam_{dataset}_run_report"
    out_dir = Path(args.out_dir) if args.out_dir else default_report_dir

    kept_index = Path(args.kept_index) if args.kept_index else (out_dir / "kept_index.jsonl")
    filter_stats = Path(args.filter_stats) if args.filter_stats else (out_dir / "filter_stats.json")
    filter_errors = Path(args.filter_errors) if args.filter_errors else (out_dir / "filter_errors.jsonl")

    crop_log = Path(args.crop_log) if args.crop_log else (outputs_root / "crop_full.log")
    run_log = Path(args.run_log) if args.run_log else (outputs_root / "full_run.log")
    if not crop_log.is_file():
        crop_log = repo / "outputs" / f"retsam_{dataset}" / "crop_full.log"
    if not run_log.is_file():
        run_log = repo / "outputs" / f"retsam_{dataset}" / "full_run.log"

    build_run_report(
        dataset=dataset,
        crop_meta_path=crop_meta,
        outputs_root=outputs_root,
        kept_index_path=kept_index,
        filter_stats_path=filter_stats,
        filter_errors_path=filter_errors,
        crop_log_path=crop_log if crop_log.is_file() else None,
        run_log_path=run_log if run_log.is_file() else None,
        out_dir=out_dir,
    )

    print(f"[OK] wrote run report to {out_dir}")
    print(f"[OK] open {out_dir / 'report.html'}")


if __name__ == "__main__":
    main()

