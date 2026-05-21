#!/usr/bin/env python3
"""Live HTML dashboard for v4 Arm A vs Arm B experiment.

Runs on the host (not in container). Reads
/home/aim_lab/LLaMA-Factory/_v4_exp_status.json (written by launcher.py)
and serves an auto-refreshing dashboard at http://0.0.0.0:6008/.

Usage:
  python3 scripts/fundus_v4/dashboard.py
Then open http://100.97.34.21:6008/ in any device on the Tailscale network.
"""
from __future__ import annotations

import http.server
import json
import socketserver
import time
from pathlib import Path

STATUS = Path("/home/aim_lab/LLaMA-Factory/_v4_exp_status.json")
PORT = 6008


HTML = """<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<title>v4 Mixed vs Staged — Live</title>
<meta http-equiv="refresh" content="5">
<style>
  body { font-family: -apple-system, "SF Pro", Helvetica, sans-serif; max-width: 1100px;
         margin: 1.5em auto; padding: 0 1em; color: #222; }
  h1 { font-size: 1.4em; margin: 0; }
  .subtitle { color: #666; font-size: 0.9em; }
  .header-bar { display: flex; justify-content: space-between; align-items: baseline;
                margin-bottom: 1em; border-bottom: 1px solid #ddd; padding-bottom: 0.5em; }
  .gpu-bar { background: #f0f0f0; border-radius: 5px; padding: 0.4em 0.8em;
             font-family: monospace; font-size: 0.9em; }
  .gpu-bar .gpu-util { display: inline-block; min-width: 3em; font-weight: bold; }
  .gpu-bar.idle { background: #f0f0f0; color: #888; }
  .gpu-bar.active { background: #e3f2fd; color: #0d47a1; }
  table { width: 100%; border-collapse: collapse; margin-top: 1em; }
  th, td { padding: 0.5em 0.7em; text-align: left; border-bottom: 1px solid #eee;
           vertical-align: top; }
  th { background: #fafafa; font-weight: 600; font-size: 0.85em; color: #555; }
  td { font-size: 0.9em; }
  .badge { display: inline-block; padding: 0.15em 0.55em; border-radius: 3px;
           font-size: 0.78em; font-weight: 600; }
  .badge-pending { background: #eee; color: #888; }
  .badge-running { background: #1976d2; color: white; }
  .badge-done    { background: #2e7d32; color: white; }
  .badge-failed  { background: #c62828; color: white; }
  .progress { background: #eee; border-radius: 3px; height: 14px; overflow: hidden;
              position: relative; min-width: 100px; }
  .progress-bar { background: linear-gradient(90deg, #4caf50, #2e7d32);
                  height: 100%; transition: width 0.5s; }
  .progress-bar.running { background: linear-gradient(90deg, #2196f3, #1976d2); }
  .progress-bar.failed  { background: #c62828; }
  .progress-text { position: absolute; top: 0; left: 0; right: 0; bottom: 0;
                   display: flex; align-items: center; justify-content: center;
                   font-size: 0.75em; font-weight: 600; color: #222; mix-blend-mode: difference;
                   color: white; }
  .sparkline { font-family: monospace; font-size: 0.85em; color: #666; }
  .loss-num { color: #1976d2; font-weight: 600; font-family: monospace; }
  .footer { color: #888; font-size: 0.8em; margin-top: 2em; text-align: center; }
  .log-tail { font-family: monospace; font-size: 0.78em; color: #555;
              background: #fafafa; padding: 0.3em 0.5em; border-radius: 3px;
              white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
              max-width: 380px; }
  .err { color: #c62828; }
</style>
</head>
<body>

<div class="header-bar">
  <div>
    <h1>v4 Mixed vs Staged</h1>
    <div class="subtitle">__EXP_TITLE__ · started __STARTED_AT__ · last updated __LAST_UPDATED__</div>
  </div>
  <div class="gpu-bar __GPU_CLASS__">
    GPU <span class="gpu-util">__GPU_UTIL_STR__</span> mem <strong>__GPU_MEM_STR__</strong>
  </div>
</div>

<table>
<thead>
<tr>
  <th>Stage</th>
  <th>Status</th>
  <th>Progress</th>
  <th>Elapsed / ETA</th>
  <th>Last loss</th>
  <th>Log tail</th>
</tr>
</thead>
<tbody>
__ROWS__
</tbody>
</table>

<div class="footer">Auto-refresh every 5s · port __PORT__ · status file: __STATUS_PATH__</div>
</body>
</html>
"""


def fmt_time(seconds: int) -> str:
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    if h: return f"{h}h{m:02d}m"
    if m: return f"{m}m{s:02d}s"
    return f"{s}s"


def sparkline(values: list[float]) -> str:
    if not values:
        return ""
    chars = " ▁▂▃▄▅▆▇█"
    lo, hi = min(values), max(values)
    rng = max(hi - lo, 1e-9)
    out = []
    for v in values[-20:]:
        idx = int((v - lo) / rng * (len(chars) - 1))
        out.append(chars[max(0, min(len(chars)-1, idx))])
    return "".join(out)


def render() -> str:
    if not STATUS.exists():
        body = "<p>Waiting for launcher to start (no status file yet)...</p>"
        return f"<html><head><meta http-equiv='refresh' content='3'></head><body>{body}</body></html>"

    try:
        data = json.loads(STATUS.read_text())
    except Exception as e:
        return f"<html><body><p class='err'>JSON parse error: {e}</p></body></html>"

    rows = []
    cur_idx = data.get("current_stage_idx", -1)
    for i, s in enumerate(data.get("stages", [])):
        status = s.get("status", "pending")
        cur = s.get("current_step", 0)
        total = s.get("expected_steps", 1)
        pct = min(100, int(cur / max(total, 1) * 100)) if status != "pending" else 0
        bar_class = "running" if status == "running" else ("failed" if status == "failed" else "")

        elapsed = s.get("elapsed_seconds", 0)
        expected_min = s.get("expected_minutes", 0)
        if status == "running" and cur > 0 and total > 0:
            est_remaining = int(elapsed * (total - cur) / cur)
        elif status == "pending":
            est_remaining = expected_min * 60
        elif status == "done":
            est_remaining = 0
        else:
            est_remaining = 0
        elapsed_str = fmt_time(elapsed) if elapsed else "-"
        eta_str = fmt_time(est_remaining) if est_remaining else "-"

        losses = s.get("loss_history", [])
        last_loss = ""
        spark = ""
        if losses:
            last_loss = f"<span class='loss-num'>{losses[-1]['loss']:.3f}</span>"
            spark = f"<span class='sparkline'>{sparkline([x['loss'] for x in losses])}</span>"

        log_tail = s.get("last_log_line", "")
        if s.get("fail_reason"):
            log_tail = f"<span class='err'>{s['fail_reason'][:200]}</span>"

        rows.append(f"""
        <tr>
          <td><strong>{s.get('label', s.get('name', '?'))}</strong></td>
          <td><span class="badge badge-{status}">{status.upper()}</span></td>
          <td>
            <div class="progress">
              <div class="progress-bar {bar_class}" style="width:{pct}%"></div>
              <div class="progress-text">{cur}/{total} · {pct}%</div>
            </div>
          </td>
          <td>{elapsed_str}<br><small>ETA {eta_str}</small></td>
          <td>{last_loss}<br>{spark}</td>
          <td><div class="log-tail">{log_tail}</div></td>
        </tr>""")

    gpu = data.get("gpu", {})
    gpu_util = gpu.get("util_pct")
    gpu_mem = gpu.get("memory_mb")
    gpu_active = gpu_util and gpu_util > 5
    subs = {
        "__EXP_TITLE__": data.get("experiment", ""),
        "__STARTED_AT__": data.get("started_at", "-"),
        "__LAST_UPDATED__": gpu.get("updated_at", "-"),
        "__GPU_CLASS__": "active" if gpu_active else "idle",
        "__GPU_UTIL_STR__": f"{gpu_util}%" if gpu_util is not None else "?",
        "__GPU_MEM_STR__": f"{gpu_mem/1024:.1f}GB" if gpu_mem else "?",
        "__ROWS__": "".join(rows),
        "__PORT__": str(PORT),
        "__STATUS_PATH__": str(STATUS),
    }
    out = HTML
    for k, v in subs.items():
        out = out.replace(k, v)
    return out


class Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/status.json":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(STATUS.read_bytes() if STATUS.exists() else b"{}")
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(render().encode("utf-8"))

    def log_message(self, *_a, **_kw):
        pass  # silence


def main():
    with socketserver.ThreadingTCPServer(("0.0.0.0", PORT), Handler) as httpd:
        print(f"Dashboard serving at http://0.0.0.0:{PORT}/")
        print(f"On Tailscale: http://100.97.34.21:{PORT}/")
        httpd.serve_forever()


if __name__ == "__main__":
    main()
