#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import time
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Optional

# Allow running this file directly: add repo root to sys.path.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.stage1_easy.progress import default_progress_path, read_progress


def _default_bind_host() -> str:
    return "0.0.0.0" if Path("/.dockerenv").is_file() else "127.0.0.1"


def _iso(ts: float) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(ts))


def _run(cmd: list[str], timeout: int = 3) -> tuple[int, str]:
    try:
        p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=timeout)
        return p.returncode, p.stdout.strip()
    except Exception as e:
        return 1, f"{type(e).__name__}: {e}"


def _pidfile_path(port: int) -> Path:
    return Path("/tmp") / f"stage1_easy_monitor.{int(port)}.pid"


def _read_pidfile(path: Path) -> int | None:
    try:
        raw = path.read_text(encoding="utf-8").strip()
        return int(raw)
    except Exception:
        return None


def _write_pidfile(path: Path, pid: int) -> None:
    path.write_text(str(int(pid)), encoding="utf-8")


def _cmdline(pid: int) -> str:
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as f:
            return f.read().replace(b"\0", b" ").decode("utf-8", "replace").strip()
    except Exception:
        return ""


def _stop_previous_monitor(port: int, current_pid: int) -> None:
    """
    Ensure only one monitor per port.

    IMPORTANT: do NOT `pkill -f serve_stage1_easy_monitor.py...` during startup — it can match the
    currently-starting process and kill itself. Instead, stop the PID recorded in the pidfile.
    """
    pf = _pidfile_path(port)
    old = _read_pidfile(pf)
    if old is None or old == current_pid:
        return
    cmd = _cmdline(old)
    if not cmd or "serve_stage1_easy_monitor.py" not in cmd:
        return
    try:
        os.kill(old, signal.SIGTERM)
    except ProcessLookupError:
        return
    except PermissionError:
        return

    # Wait briefly for release.
    deadline = time.time() + 3.0
    while time.time() < deadline:
        try:
            os.kill(old, 0)
        except ProcessLookupError:
            return
        except PermissionError:
            return
        time.sleep(0.05)


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
    q = "index,name,memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu"
    cmd = ["bash", "-lc", f"nvidia-smi --query-gpu={q} --format=csv,noheader,nounits 2>/dev/null || true"]
    _, out = _run(cmd, timeout=3)
    line = out.splitlines()[0].strip() if out.strip() else ""
    if not line:
        return {"available": False}
    parts = [p.strip() for p in line.split(",")]

    def _num(x: str) -> Optional[float]:
        x = x.strip()
        if not x or x.upper() == "N/A" or x.startswith("["):
            return None
        try:
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
        "raw": line,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Stage1 Easy monitor server")
    ap.add_argument("--host", default=_default_bind_host())
    ap.add_argument("--port", type=int, default=8777)
    ap.add_argument(
        "--replace",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If true, best-effort kill previous serve_stage1_easy_monitor.py on the same --port before binding.",
    )
    ap.add_argument("--state", default=str(default_progress_path()))
    ap.add_argument("--tail-lines", type=int, default=120)
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[2]
    state_path = Path(args.state)
    if not state_path.is_absolute():
        state_path = root / state_path
    html_path = Path(__file__).resolve().parent / "monitor_index.html"
    html = html_path.read_text(encoding="utf-8")

    current_pid = os.getpid()
    if args.replace:
        _stop_previous_monitor(int(args.port), current_pid=current_pid)

    def build_system_payload() -> dict[str, Any]:
        return {"server_time": _iso(time.time()), "mem": _mem_usage(), "disk": _disk_usage(root), "gpu": _gpu_status()}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            return

        def _set_no_cache(self):
            self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
            self.send_header("Pragma", "no-cache")
            self.send_header("Expires", "0")

        def do_GET(self):
            path = self.path.split("?", 1)[0]
            if path in ("/", "/index.html"):
                body = html.encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self._set_no_cache()
                self.end_headers()
                self.wfile.write(body)
                return
            if path == "/api/state":
                st = read_progress(state_path)
                raw = json.dumps(st, ensure_ascii=False, indent=2)
                self.send_response(200)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self._set_no_cache()
                self.end_headers()
                self.wfile.write(raw.encode("utf-8"))
                return
            if path == "/api/system":
                raw = json.dumps(build_system_payload(), ensure_ascii=False, indent=2)
                self.send_response(200)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self._set_no_cache()
                self.end_headers()
                self.wfile.write(raw.encode("utf-8"))
                return
            if path == "/api/log":
                # /api/log?step=build_dataset
                qs = self.path.split("?", 1)[1] if "?" in self.path else ""
                step = ""
                for part in qs.split("&"):
                    if part.startswith("step="):
                        step = part.split("=", 1)[1]
                st = read_progress(state_path)
                log_path = None
                try:
                    log_path = st.get("steps", {}).get(step, {}).get("log_path")
                except Exception:
                    log_path = None
                if not log_path:
                    body = b"(no log_path for step)\n"
                    self.send_response(200)
                    self.send_header("Content-Type", "text/plain; charset=utf-8")
                    self._set_no_cache()
                    self.end_headers()
                    self.wfile.write(body)
                    return
                lp = Path(log_path)
                if not lp.is_absolute():
                    lp = root / lp
                if not lp.is_file():
                    body = f"(log missing) {lp}\n".encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "text/plain; charset=utf-8")
                    self._set_no_cache()
                    self.end_headers()
                    self.wfile.write(body)
                    return
                lines = lp.read_text(encoding="utf-8", errors="replace").splitlines()
                tail = "\n".join(lines[-args.tail_lines :]) + "\n"
                body = tail.encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/plain; charset=utf-8")
                self._set_no_cache()
                self.end_headers()
                self.wfile.write(body)
                return
            self.send_error(404)

    httpd = ThreadingHTTPServer((args.host, args.port), Handler)
    _write_pidfile(_pidfile_path(int(args.port)), os.getpid())
    print(f"Stage1 Easy monitor listening on http://{args.host}:{args.port}/")
    print(f"  state: {state_path}")
    print(f"  pidfile: {_pidfile_path(int(args.port))}")
    if args.host == "0.0.0.0":
        print("  (0.0.0.0) If in Docker: map port -p PORT:PORT")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()

