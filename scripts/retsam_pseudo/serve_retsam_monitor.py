#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import time
import sys
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

# Allow running this file directly: add repo root to sys.path.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.retsam_pseudo.common import CropMetaRow, read_jsonl
from scripts.retsam_pseudo.retsam_json import parse_lesion_metrics


def _default_bind_host() -> str:
    return "0.0.0.0" if Path("/.dockerenv").is_file() else "127.0.0.1"


def _iso(ts: float) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(ts))


def _disk_usage(path: Path) -> dict[str, Any]:
    try:
        st = os.statvfs(str(path))
        total = st.f_frsize * st.f_blocks
        free = st.f_frsize * st.f_bavail
        used = total - (st.f_frsize * st.f_bfree)
        gb = 1024**3
        return {"total_gb": round(total / gb, 2), "used_gb": round(used / gb, 2), "free_gb": round(free / gb, 2)}
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}


def _mem_usage() -> dict[str, Any]:
    try:
        meminfo = Path("/proc/meminfo").read_text(encoding="utf-8", errors="ignore").splitlines()
        kv: dict[str, int] = {}
        for line in meminfo:
            if ":" not in line:
                continue
            k, rest = line.split(":", 1)
            val = rest.strip().split()[0]
            try:
                kv[k] = int(val)  # kB
            except Exception:
                pass
        total_kb = kv.get("MemTotal", 0)
        avail_kb = kv.get("MemAvailable", 0)
        used_kb = max(0, total_kb - avail_kb)
        gb = 1024**2
        return {"total_gb": round(total_kb / gb, 2), "used_gb": round(used_kb / gb, 2), "avail_gb": round(avail_kb / gb, 2)}
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}


def _gpu_status() -> dict[str, Any]:
    # Keep it dependency-free (shell to nvidia-smi). If unavailable, return available=false.
    import subprocess

    q = "index,name,memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu"
    cmd = ["bash", "-lc", f"nvidia-smi --query-gpu={q} --format=csv,noheader,nounits 2>/dev/null || true"]
    try:
        p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=3)
        out = (p.stdout or "").strip()
    except Exception:
        out = ""
    line = out.splitlines()[0].strip() if out else ""
    if not line:
        return {"available": False}
    parts = [x.strip() for x in line.split(",")]

    def _num(x: str) -> float | None:
        try:
            x = x.strip()
            if not x or x.upper() == "N/A":
                return None
            return float(x)
        except Exception:
            return None

    return {
        "available": True,
        "index": parts[0] if len(parts) > 0 else "0",
        "name": parts[1] if len(parts) > 1 else "",
        "mem_total_mb": _num(parts[2]) if len(parts) > 2 else None,
        "mem_used_mb": _num(parts[3]) if len(parts) > 3 else None,
        "mem_free_mb": _num(parts[4]) if len(parts) > 4 else None,
        "util_gpu_pct": _num(parts[5]) if len(parts) > 5 else None,
        "temp_c": _num(parts[6]) if len(parts) > 6 else None,
    }


@dataclass
class MonitorConfig:
    root: Path
    data_root: Path
    outputs_root: Path
    html: str
    errors_tail_lines: int = 40
    scan_cooldown_sec: float = 2.0


class Cache:
    def __init__(self) -> None:
        self.last_scan_at: float = 0.0
        self.progress: dict[str, Any] = {}
        self.stats: dict[str, Any] = {}


def _read_errors_tail(errors_jsonl: Path, tail_lines: int) -> tuple[int, str]:
    if not errors_jsonl.is_file():
        return 0, ""
    lines = errors_jsonl.read_text(encoding="utf-8", errors="replace").splitlines()
    return len(lines), "\n".join(lines[-tail_lines:]) + ("\n" if lines else "")


def _load_total_from_crop_meta(crop_meta: Path) -> int:
    try:
        rows = read_jsonl(crop_meta)
        return len(rows)
    except Exception:
        return 0


def _compute_progress_and_stats(cfg: MonitorConfig, dataset: str) -> tuple[dict[str, Any], dict[str, Any]]:
    out_dir = cfg.outputs_root / f"retsam_{dataset}"
    crop_meta = cfg.data_root / "cropped" / dataset / "crop_meta.jsonl"
    if dataset == "ddr_grading":
        crop_meta = cfg.data_root / "cropped" / "ddr_grading" / "crop_meta.jsonl"

    total = _load_total_from_crop_meta(crop_meta) if crop_meta.is_file() else 0
    qa_paths = list(out_dir.glob("*/quantitative_analysis.json")) if out_dir.is_dir() else []
    done = len(qa_paths)

    # Estimate speed based on the newest N file mtimes.
    rate_ips = None
    avg_sec = None
    eta_sec = None
    if done >= 2:
        mt = sorted((p.stat().st_mtime for p in qa_paths), reverse=True)
        window = mt[: min(50, len(mt))]
        t_new, t_old = max(window), min(window)
        dt = max(1e-6, float(t_new - t_old))
        rate_ips = (len(window) - 1) / dt
        avg_sec = (1.0 / rate_ips) if rate_ips and rate_ips > 0 else None
        if total and rate_ips and rate_ips > 0:
            eta_sec = max(0.0, float(total - done)) / rate_ips

    errors_jsonl = out_dir / "errors.jsonl"
    errors_lines, errors_tail = _read_errors_tail(errors_jsonl, cfg.errors_tail_lines)

    # Lightweight lesion stats: present rates + quadrant sums.
    quad_sums = {"HE": {"TS": 0, "TI": 0, "NS": 0, "NI": 0}, "EX": {"TS": 0, "TI": 0, "NS": 0, "NI": 0}, "SE": {"TS": 0, "TI": 0, "NS": 0, "NI": 0}}
    present = {"HE": 0, "EX": 0, "SE": 0}
    # RetSAM-only biomarkers (coordinate-free)
    eye_side = {"left": 0, "right": 0, "unknown": 0}
    av_vals: list[float] = []
    cdr_vals: list[float] = []
    tort_vals: list[float] = []
    qc_flag_true = 0
    coord_anomaly = 0
    scanned = 0
    # Sample up to 300 recent files for realtime responsiveness.
    qa_paths_sorted = sorted(qa_paths, key=lambda p: p.stat().st_mtime, reverse=True)[:300]
    for p in qa_paths_sorted:
        try:
            q = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        meas = q.get("measurements", {}) if isinstance(q, dict) else {}

        # eye_side
        es = None
        try:
            es = (meas.get("optic_disc_cup", {}) or {}).get("eye_side")
        except Exception:
            es = None
        es = str(es).lower() if es is not None else ""
        if es in ("left", "right"):
            eye_side[es] += 1
        else:
            eye_side["unknown"] += 1

        # vessels biomarkers
        vessels = meas.get("vessels", {}) if isinstance(meas, dict) else {}
        if isinstance(vessels, dict):
            av = vessels.get("av_ratio")
            if isinstance(av, (int, float)):
                av_vals.append(float(av))
            tort = vessels.get("tortuosity")
            if isinstance(tort, dict):
                a = tort.get("artery")
                v = tort.get("vein")
                if isinstance(a, (int, float)) and isinstance(v, (int, float)):
                    tort_vals.append(float(a + v) / 2.0)
            qc = vessels.get("qc_flag")
            if qc is True:
                qc_flag_true += 1

        # CDR
        odc = meas.get("optic_disc_cup", {}) if isinstance(meas, dict) else {}
        if isinstance(odc, dict):
            cdr = odc.get("cup_disc_ratio")
            if isinstance(cdr, dict):
                cdr_v = cdr.get("value")
                if isinstance(cdr_v, (int, float)):
                    cdr_vals.append(float(cdr_v))

        # coord anomaly quick check
        try:
            disc = (odc.get("disc") or {}) if isinstance(odc, dict) else {}
            disc_center = disc.get("center")
            disc_radius = disc.get("radius")
            mac = (meas.get("macula") or {}) if isinstance(meas, dict) else {}
            mac_center = mac.get("center")
            if isinstance(disc_center, list) and len(disc_center) >= 2 and isinstance(disc_radius, (int, float)):
                disc_y = float(disc_center[1])
                mac_y = None
                if isinstance(mac_center, dict) and isinstance(mac_center.get("y"), (int, float)):
                    mac_y = float(mac_center["y"])
                elif isinstance(mac_center, list) and len(mac_center) >= 2 and isinstance(mac_center[1], (int, float)):
                    mac_y = float(mac_center[1])
                if mac_y is not None and abs(mac_y - disc_y) > 2.0 * float(disc_radius):
                    coord_anomaly += 1
        except Exception:
            pass
        for lesion in ("HE", "EX", "SE"):
            m = parse_lesion_metrics(q, lesion)
            if m is None:
                continue
            if int(m.count) > 0:
                present[lesion] += 1
            for k in ("TS", "TI", "NS", "NI"):
                quad_sums[lesion][k] += int(m.quadrant_distribution.get(k, 0))
        scanned += 1

    present_rates = {k: (present[k] / scanned if scanned else 0.0) for k in present.keys()}
    coord_anomaly_rate = (coord_anomaly / scanned) if scanned else 0.0

    def _summary(xs: list[float]) -> dict[str, Any]:
        if not xs:
            return {"n": 0}
        ys = sorted(xs)

        def pct(p: float) -> float:
            k = int(round(p * (len(ys) - 1)))
            return float(ys[max(0, min(len(ys) - 1, k))])

        return {"n": len(ys), "p10": pct(0.1), "p50": pct(0.5), "p90": pct(0.9)}

    progress = {
        "dataset": dataset,
        "outputs_dir": str(out_dir),
        "crop_meta": str(crop_meta),
        "total": total,
        "done": done,
        "skipped": None,
        "failed": None,
        "errors_lines": errors_lines,
        "errors_tail": errors_tail,
        "rate_ips": round(rate_ips, 4) if rate_ips else None,
        "avg_sec_per_img": round(avg_sec, 2) if avg_sec else None,
        "eta_sec": int(eta_sec) if eta_sec is not None else None,
        "updated_at": _iso(time.time()),
    }
    stats = {
        "dataset": dataset,
        "scanned_recent": scanned,
        "present_rates": present_rates,
        "quadrant_sums": quad_sums,
        "eye_side": eye_side,
        "av_ratio": _summary(av_vals),
        "cdr": _summary(cdr_vals),
        "tortuosity": _summary(tort_vals),
        "vessel_qc_flag_true_rate": (qc_flag_true / scanned) if scanned else 0.0,
        "coord_anomaly_rate": coord_anomaly_rate,
        "updated_at": progress["updated_at"],
    }
    return progress, stats


def main() -> None:
    ap = argparse.ArgumentParser(description="RetSAM real-time monitor (progress + stats)")
    ap.add_argument("--host", default=_default_bind_host())
    ap.add_argument("--port", type=int, default=8788)
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--outputs-root", default="outputs")
    ap.add_argument("--errors-tail-lines", type=int, default=40)
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[2]
    data_root = Path(args.data_root)
    if not data_root.is_absolute():
        data_root = root / data_root
    outputs_root = Path(args.outputs_root)
    if not outputs_root.is_absolute():
        outputs_root = root / outputs_root

    html_path = Path(__file__).resolve().parent / "monitor_retsam.html"
    html = html_path.read_text(encoding="utf-8")
    cfg = MonitorConfig(root=root, data_root=data_root, outputs_root=outputs_root, html=html, errors_tail_lines=int(args.errors_tail_lines))
    cache_by_ds: dict[str, Cache] = {"aptos": Cache(), "ddr_grading": Cache()}

    def system_payload() -> dict[str, Any]:
        return {"server_time": _iso(time.time()), "mem": _mem_usage(), "disk": _disk_usage(root), "gpu": _gpu_status()}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            return

        def _set_no_cache(self):
            self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
            self.send_header("Pragma", "no-cache")
            self.send_header("Expires", "0")

        def _json(self, obj: Any):
            raw = json.dumps(obj, ensure_ascii=False, indent=2).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(raw)))
            self._set_no_cache()
            self.end_headers()
            self.wfile.write(raw)

        def do_GET(self):
            path = self.path.split("?", 1)[0]
            if path in ("/", "/index.html"):
                body = cfg.html.encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self._set_no_cache()
                self.end_headers()
                self.wfile.write(body)
                return

            if path == "/api/system":
                self._json(system_payload())
                return

            # naive query parsing
            qs = self.path.split("?", 1)[1] if "?" in self.path else ""
            params: dict[str, str] = {}
            for part in qs.split("&"):
                if "=" in part:
                    k, v = part.split("=", 1)
                    params[k] = v
            ds = params.get("dataset", "aptos").strip() or "aptos"
            if ds not in cache_by_ds:
                cache_by_ds[ds] = Cache()
            c = cache_by_ds[ds]
            now = time.time()
            if now - c.last_scan_at >= cfg.scan_cooldown_sec:
                prog, stats = _compute_progress_and_stats(cfg, ds)
                c.progress = prog
                c.stats = stats
                c.last_scan_at = now

            if path == "/api/progress":
                self._json(c.progress)
                return
            if path == "/api/stats":
                self._json(c.stats)
                return

            self.send_response(404)
            self._set_no_cache()
            self.end_headers()
            self.wfile.write(b"not found\n")

    srv = ThreadingHTTPServer((args.host, int(args.port)), Handler)
    print(f"retsam_monitor_listen=http://{args.host}:{int(args.port)}", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()

