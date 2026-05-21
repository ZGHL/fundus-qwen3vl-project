#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import subprocess
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any


HTML = """<!doctype html>
<html lang="zh">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>Fundus SFT Monitor</title>
  <style>
    body{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI","Microsoft YaHei",sans-serif;margin:0;background:#f6f8fb;color:#182230}
    header{background:#17365d;color:white;padding:18px 24px}
    h1{margin:0;font-size:20px}.sub{opacity:.85;margin-top:4px;font-size:13px}
    main{padding:18px 24px;max-width:1240px;margin:auto}
    .grid{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px}
    .card{background:white;border:1px solid #d9e2ec;border-radius:8px;padding:14px;box-shadow:0 1px 2px rgba(16,24,40,.04)}
    .label{font-size:12px;color:#667085}.value{font-size:22px;font-weight:700;margin-top:4px}.hint{font-size:12px;color:#667085;margin-top:4px}.ok{color:#067647}.warn{color:#b54708}.bad{color:#b42318}
    table{width:100%;border-collapse:collapse;background:white;border:1px solid #d9e2ec;border-radius:8px;overflow:hidden}
    th,td{border-bottom:1px solid #e5e7eb;padding:8px 10px;text-align:left;font-size:13px}th{background:#eef5fb;color:#17365d}
    pre{white-space:pre-wrap;background:#111827;color:#e5e7eb;border-radius:8px;padding:12px;max-height:360px;overflow:auto;font-size:12px}
    .bar{height:12px;background:#e5e7eb;border-radius:999px;overflow:hidden}.fill{height:12px;background:#2f80ed;transition:width .4s ease}.progressline{display:flex;justify-content:space-between;font-size:13px;color:#475467;margin-bottom:8px}
    .section{margin-top:18px}.muted{color:#667085}.mono{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace}
  </style>
</head>
<body>
<header><h1>Qwen3-VL Fundus SFT Monitor</h1><div class="sub" id="subtitle">loading...</div></header>
<main>
  <section class="grid" id="cards"></section>
  <div class="section"><h3>Progress</h3><div class="card"><div class="progressline"><span id="progressText">waiting</span><span id="etaText"></span></div><div class="bar"><div class="fill" id="progress"></div></div></div></div>
  <div class="section"><h3>Recent Trainer Logs</h3><table><thead><tr><th>step</th><th>epoch</th><th>loss</th><th>lr</th><th>grad norm</th><th>elapsed</th><th>remaining</th></tr></thead><tbody id="rows"></tbody></table></div>
  <div class="section"><h3>Launch Log Tail</h3><pre id="tail"></pre></div>
</main>
<script>
function fmt(v, digits=4){
  if(v===null || v===undefined || v==='') return '<span class="muted">waiting</span>';
  if(typeof v==='number') return Number.isFinite(v) ? v.toPrecision(digits) : '<span class="muted">waiting</span>';
  return String(v);
}
async function refresh(){
  const r=await fetch('/data'); const d=await r.json();
  document.getElementById('subtitle').textContent=`${d.name} | ${new Date(d.now*1000).toLocaleString()}`;
  const pct=d.max_steps?Math.min(100,100*d.global_step/d.max_steps):(d.percentage||0);
  document.getElementById('progress').style.width=pct+'%';
  document.getElementById('progressText').textContent=d.max_steps?`${d.global_step} / ${d.max_steps} steps (${pct.toFixed(1)}%)`:(d.status==='starting'?'waiting for first log':'step unavailable');
  document.getElementById('etaText').textContent=d.remaining_time?`ETA ${d.remaining_time}`:'';
  const gpuText=d.gpu.available?`${d.gpu.util_gpu_pct}% · ${d.gpu.mem_used_mb}/${d.gpu.mem_total_mb} MB`:'not detected';
  const memText=d.mem.total_gb?`${d.mem.used_gb}/${d.mem.total_gb} GB`:'not detected';
  const cards=[
    ['Status',d.status,d.status==='running'?'ok':(d.status==='completed'?'ok':'warn'),d.latest_loss!==null&&d.latest_loss!==undefined?`loss ${fmt(d.latest_loss)}`:'waiting for metrics'],
    ['Step',d.max_steps?`${d.global_step}/${d.max_steps}`:fmt(d.global_step),'',d.elapsed_time||''],
    ['GPU',gpuText,'',d.gpu.name||''],
    ['Memory',memText,'',d.mem.avail_gb?`${d.mem.avail_gb} GB available`:'' ]
  ];
  document.getElementById('cards').innerHTML=cards.map(c=>`<div class="card"><div class="label">${c[0]}</div><div class="value ${c[2]}">${c[1]}</div><div class="hint">${c[3]||''}</div></div>`).join('');
  document.getElementById('rows').innerHTML=(d.logs||[]).slice(-20).reverse().map(x=>`<tr><td>${fmt(x.step)}</td><td>${fmt(x.epoch)}</td><td>${fmt(x.loss)}</td><td class="mono">${fmt(x.learning_rate,3)}</td><td>${fmt(x.grad_norm)}</td><td>${fmt(x.elapsed_time)}</td><td>${fmt(x.remaining_time)}</td></tr>`).join('');
  document.getElementById('tail').textContent=d.tail||'';
}
refresh(); setInterval(refresh,2000);
</script>
</body></html>"""


def run(cmd: str, timeout: int = 3) -> str:
    try:
        return subprocess.run(cmd, shell=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=timeout).stdout.strip()
    except Exception as e:
        return f"{type(e).__name__}: {e}"


def gpu() -> dict[str, Any]:
    out = run("nvidia-smi --query-gpu=name,memory.total,memory.used,utilization.gpu,temperature.gpu --format=csv,noheader,nounits 2>/dev/null || true")
    if not out:
        return {"available": False}
    p = [x.strip() for x in out.splitlines()[0].split(",")]
    return {"available": True, "name": p[0], "mem_total_mb": p[1], "mem_used_mb": p[2], "util_gpu_pct": p[3], "temp_c": p[4]}


def mem() -> dict[str, Any]:
    vals = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            try:
                vals[k] = int(v.strip().split()[0])
            except Exception:
                pass
    total = vals.get("MemTotal", 0) / 1024 / 1024
    avail = vals.get("MemAvailable", 0) / 1024 / 1024
    return {"total_gb": round(total, 2), "used_gb": round(total - avail, 2), "avail_gb": round(avail, 2)}


def parse_trainer_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    logs = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except Exception:
            continue
        step = item.get("current_steps", item.get("step"))
        if step is None:
            continue
        logs.append(
            {
                "step": step,
                "epoch": item.get("epoch"),
                "loss": item.get("loss"),
                "learning_rate": item.get("lr", item.get("learning_rate")),
                "grad_norm": item.get("grad_norm"),
                "elapsed_time": item.get("elapsed_time"),
                "remaining_time": item.get("remaining_time"),
                "percentage": item.get("percentage"),
                "total_steps": item.get("total_steps"),
            }
        )
    return logs


def parse_trainer_state(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    logs = []
    for item in state.get("log_history", []):
        step = item.get("step")
        if step is None:
            continue
        logs.append(
            {
                "step": step,
                "epoch": item.get("epoch"),
                "loss": item.get("loss"),
                "learning_rate": item.get("learning_rate"),
                "grad_norm": item.get("grad_norm"),
                "elapsed_time": None,
                "remaining_time": None,
                "percentage": None,
                "total_steps": state.get("max_steps"),
            }
        )
    return logs


def parse_launch_progress(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        text = path.read_text(encoding="utf-8", errors="replace")[-50000:]
    except Exception:
        return {}
    total = None
    total_match = re.findall(r"Total optimization steps\s*=\s*([0-9]+)", text)
    if total_match:
        total = int(total_match[-1])
    progress = re.findall(r"([0-9]+)%\|.*?\|\s*([0-9]+)/([0-9]+)\s*\[([^\]]+)\]", text)
    if not progress:
        return {"max_steps": total or 0}
    pct, step, total_from_bar, timing = progress[-1]
    step_i = int(step)
    total_i = int(total_from_bar)
    return {
        "global_step": step_i,
        "max_steps": total or total_i,
        "percentage": float(pct),
        "elapsed_time": timing.split("<", 1)[0].strip() if timing else None,
    }


def parse_state(out_dir: Path, launch_log: Path) -> dict[str, Any]:
    logs = parse_trainer_jsonl(out_dir / "trainer_log.jsonl")
    if not logs:
        logs = parse_trainer_state(out_dir / "trainer_state.json")
    latest = logs[-1] if logs else {}
    launch_progress = parse_launch_progress(launch_log)
    step = int(latest.get("step") or launch_progress.get("global_step") or 0)
    max_steps = int(latest.get("total_steps") or launch_progress.get("max_steps") or 0)
    if not max_steps and (out_dir / "trainer_state.json").exists():
        try:
            max_steps = int(json.loads((out_dir / "trainer_state.json").read_text(encoding="utf-8")).get("max_steps") or 0)
        except Exception:
            pass
    status = "starting"
    if step:
        status = "running"
    if (out_dir / "all_results.json").exists() or (max_steps and step >= max_steps):
        status = "completed"
    return {
        "logs": logs,
        "global_step": step,
        "max_steps": max_steps,
        "status": status,
        "latest_loss": latest.get("loss"),
        "elapsed_time": latest.get("elapsed_time") or launch_progress.get("elapsed_time"),
        "remaining_time": latest.get("remaining_time"),
        "percentage": latest.get("percentage") or launch_progress.get("percentage"),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8788)
    ap.add_argument("--output-dir", default="saves/qwen3-vl-8b-fundus/lora/stage1_smoke")
    ap.add_argument("--launch-log", default="logs/train_fundus_stage1_smoke.log")
    args = ap.parse_args()
    root = Path(__file__).resolve().parents[2]
    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = root / out_dir
    launch_log = Path(args.launch_log)
    if not launch_log.is_absolute():
        launch_log = root / launch_log

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):  # noqa: ANN001
            return

        def do_GET(self):  # noqa: N802
            if self.path == "/data":
                state = parse_state(out_dir, launch_log)
                tail = ""
                if launch_log.exists():
                    tail = "\n".join(launch_log.read_text(encoding="utf-8", errors="replace").splitlines()[-80:])
                payload = {
                    "now": time.time(),
                    "name": out_dir.name,
                    "output_dir": str(out_dir),
                    **state,
                    "gpu": gpu(),
                    "mem": mem(),
                    "tail": tail,
                }
                body = json.dumps(payload, ensure_ascii=False).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            else:
                body = HTML.encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"monitor: http://{args.host}:{args.port}/ output_dir={out_dir}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
