#!/usr/bin/env python3
"""Read-only public dashboard for the Stage1 English Arm B training run."""
from __future__ import annotations

import argparse
import ast
import json
import os
import re
import shutil
import socket
import subprocess
import time
from collections import Counter
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import yaml


HTML = r"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Fundus Stage1 · Arm B Monitor</title>
<style>
:root{--bg:#07111f;--panel:#0d1b2d;--panel2:#11243a;--line:#203a55;--text:#eaf2fb;--muted:#91a8c0;--cyan:#37d5d8;--blue:#5596ff;--green:#4dd8a5;--amber:#ffbd66;--red:#ff6b7a}
*{box-sizing:border-box}body{margin:0;background:radial-gradient(circle at 15% -10%,#173c62 0,transparent 35%),var(--bg);color:var(--text);font-family:Inter,ui-sans-serif,system-ui,-apple-system,"Segoe UI","Microsoft YaHei",sans-serif}
header{padding:22px 28px;border-bottom:1px solid var(--line);background:rgba(7,17,31,.82);backdrop-filter:blur(12px);position:sticky;top:0;z-index:3}.head{max-width:1540px;margin:auto;display:flex;justify-content:space-between;gap:18px}.title{font-size:23px;font-weight:780}.sub{font-size:13px;color:var(--muted);margin-top:6px}.live{display:flex;align-items:center;gap:8px;color:var(--green);font-size:13px}.dot{width:9px;height:9px;border-radius:50%;background:var(--green);box-shadow:0 0 14px var(--green);animation:pulse 1.4s infinite}@keyframes pulse{50%{opacity:.35}}
main{max-width:1540px;margin:auto;padding:20px 28px 36px}.grid{display:grid;gap:13px}.kpis{grid-template-columns:repeat(6,minmax(0,1fr))}.two{grid-template-columns:1.15fr .85fr}.three{grid-template-columns:repeat(3,minmax(0,1fr))}
.card,.panel{background:linear-gradient(145deg,rgba(17,36,58,.96),rgba(11,27,45,.96));border:1px solid var(--line);border-radius:13px;box-shadow:0 14px 38px rgba(0,0,0,.18)}.card{padding:14px 15px;min-height:100px}.label{font-size:11px;color:var(--muted);text-transform:uppercase;letter-spacing:.07em}.value{font-size:23px;font-weight:780;margin-top:8px}.hint{font-size:12px;color:var(--muted);margin-top:6px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}.panel{padding:16px;margin-top:13px}.panel h2{font-size:15px;margin:0;font-weight:740}.ph{display:flex;justify-content:space-between;align-items:center;margin-bottom:13px;gap:10px}
.badge{border:1px solid var(--line);background:#0a192a;border-radius:999px;padding:4px 9px;color:var(--muted);font-size:11px}.ok{color:var(--green)}.warn{color:var(--amber)}.bad{color:var(--red)}.blue{color:var(--blue)}.cyan{color:var(--cyan)}
.bar{height:12px;background:#071523;border:1px solid #183550;border-radius:999px;overflow:hidden}.fill{height:100%;width:0;background:linear-gradient(90deg,var(--blue),var(--cyan),var(--green));transition:width .5s}.split{display:flex;justify-content:space-between;gap:12px;margin-bottom:9px;font-size:12px;color:var(--muted)}
canvas{width:100%;height:225px;background:#091827;border:1px solid var(--line);border-radius:10px}table{width:100%;border-collapse:collapse}th,td{font-size:12px;text-align:left;padding:8px 9px;border-bottom:1px solid #19334d}th{color:#b9cee2;background:#0a1929}tr:last-child td{border:0}.num{text-align:right;font-variant-numeric:tabular-nums}.mono{font-family:ui-monospace,SFMono-Regular,Menlo,monospace}.scroll{overflow:auto;max-height:390px}pre{margin:0;background:#06111e;border:1px solid var(--line);border-radius:10px;padding:13px;max-height:420px;overflow:auto;color:#bcd0e3;font-size:11px;line-height:1.45;white-space:pre-wrap}.kv{display:grid;grid-template-columns:1fr auto;gap:8px 14px;font-size:12px}.kv div:nth-child(odd){color:var(--muted)}.empty{color:var(--muted);padding:30px;text-align:center}
@media(max-width:1200px){.kpis{grid-template-columns:repeat(3,1fr)}.two,.three{grid-template-columns:1fr}}@media(max-width:700px){main{padding:13px}header{padding:16px}.kpis{grid-template-columns:1fr 1fr}.head{display:block}.live{margin-top:10px}.value{font-size:19px}}
</style></head><body>
<header><div class="head"><div><div class="title">Fundus Qwen3-VL · Stage1 Arm B</div><div class="sub">English single-lesion CoT · Language LoRA + Vision LoRA · Projector frozen</div></div><div><div class="live"><span class="dot"></span><span id="status">CONNECTING</span></div><div class="sub mono" id="clock"></div></div></div></header>
<main>
<section class="grid kpis" id="kpis"></section>
<section class="grid two">
 <div class="panel"><div class="ph"><h2>Training Progress & Loss</h2><span class="badge" id="etaBadge">waiting</span></div><div class="split"><span id="progressText"></span><span id="rateText"></span></div><div class="bar"><div class="fill" id="fill"></div></div><div style="height:12px"></div><canvas id="chart" width="1000" height="300"></canvas></div>
 <div class="panel"><div class="ph"><h2>Run Configuration</h2><span class="badge cyan">Arm B</span></div><div class="kv" id="config"></div></div>
</section>
<section class="grid three">
 <div class="panel"><div class="ph"><h2>Training Distribution</h2><span class="badge">12,650 rows</span></div><div id="dist"></div></div>
 <div class="panel"><div class="ph"><h2>Evidence Tiers</h2><span class="badge">S0 → S4</span></div><div id="tiers"></div></div>
 <div class="panel"><div class="ph"><h2>Evaluation Sets</h2><span class="badge">read-only</span></div><div id="evals"></div></div>
</section>
<section class="grid two">
 <div class="panel"><div class="ph"><h2>Recent Trainer Metrics</h2><span class="badge" id="metricCount"></span></div><div class="scroll" id="metrics"></div></div>
 <div class="panel"><div class="ph"><h2>Process & Artifact Health</h2><span class="badge" id="healthBadge"></span></div><div id="health"></div></div>
</section>
<section class="panel"><div class="ph"><h2>Training Log Tail</h2><span class="badge mono" id="logPath"></span></div><pre id="tail"></pre></section>
</main>
<script>
const f=(v,d=2)=>v===null||v===undefined||v===''?'—':typeof v==='number'?v.toFixed(d).replace(/\.?0+$/,''):v;
const dur=s=>{if(!s&&s!==0)return'—';s=Math.max(0,Math.round(s));let h=Math.floor(s/3600),m=Math.floor(s%3600/60),x=s%60;return h?`${h}h ${m}m`:m?`${m}m ${x}s`:`${x}s`};
const card=(l,v,h,c='')=>`<div class="card"><div class="label">${l}</div><div class="value ${c}">${v}</div><div class="hint">${h||''}</div></div>`;
const table=(hs,rs)=>`<table><thead><tr>${hs.map(x=>`<th>${x}</th>`).join('')}</tr></thead><tbody>${rs.join('')}</tbody></table>`;
function chart(logs){let c=document.getElementById('chart'),x=c.getContext('2d');x.clearRect(0,0,c.width,c.height);x.fillStyle='#091827';x.fillRect(0,0,c.width,c.height);let p=(logs||[]).filter(z=>typeof z.loss==='number').slice(-180);x.strokeStyle='#183a56';for(let i=1;i<5;i++){x.beginPath();x.moveTo(0,i*c.height/5);x.lineTo(c.width,i*c.height/5);x.stroke()}if(p.length<2){x.fillStyle='#91a8c0';x.font='15px system-ui';x.fillText('Preprocessing / waiting for first trainer metrics',28,45);return}let lo=Math.min(...p.map(z=>z.loss)),hi=Math.max(...p.map(z=>z.loss)),a=p[0].step,b=p[p.length-1].step;let px=z=>42+(z-a)/(b-a||1)*(c.width-70),py=z=>24+(hi-z)/(hi-lo||1)*(c.height-62);let g=x.createLinearGradient(0,0,c.width,0);g.addColorStop(0,'#5596ff');g.addColorStop(.5,'#37d5d8');g.addColorStop(1,'#4dd8a5');x.strokeStyle=g;x.lineWidth=3;x.beginPath();p.forEach((z,i)=>i?x.lineTo(px(z.step),py(z.loss)):x.moveTo(px(z.step),py(z.loss)));x.stroke();x.fillStyle='#eaf2fb';x.font='13px system-ui';x.fillText(`loss ${f(p.at(-1).loss,4)} · step ${p.at(-1).step}`,44,20)}
async function refresh(){let d=await(await fetch('/api/state',{cache:'no-store'})).json();document.getElementById('clock').textContent=new Date(d.now*1000).toLocaleString();document.getElementById('status').textContent=(d.train.status||'starting').toUpperCase();let t=d.train,g=d.gpu,m=d.memory,p=t.max_steps?Math.min(1,t.step/t.max_steps):0;document.getElementById('fill').style.width=(p*100).toFixed(1)+'%';document.getElementById('progressText').textContent=t.max_steps?`${t.step} / ${t.max_steps} optimizer steps · ${(p*100).toFixed(1)}%`:'Preprocessing dataset / loading model';document.getElementById('rateText').textContent=t.seconds_per_step?`${f(t.seconds_per_step,1)} s/step · ${f(t.samples_per_second,2)} samples/s`:'';document.getElementById('etaBadge').textContent=t.eta_seconds?`ETA ${dur(t.eta_seconds)}`:'ETA waiting';document.getElementById('kpis').innerHTML=[card('Status',t.status||'starting',t.phase||'',t.status==='running'?'ok':'warn'),card('Latest Loss',f(t.latest_loss,4),t.latest_lr?`LR ${t.latest_lr.toExponential(2)}`:'waiting'),card('Progress',t.max_steps?`${(p*100).toFixed(1)}%`:'—',t.max_steps?`${t.step}/${t.max_steps} steps`:'waiting'),card('Elapsed',dur(t.elapsed_seconds),t.eta_seconds?`ETA ${dur(t.eta_seconds)}`:'estimating'),card('GPU',g.available?`${g.util}%`:'—',g.available?`${g.mem_used}/${g.mem_total} MiB · ${g.temp}°C · ${g.power}W`:'unavailable','cyan'),card('Host Memory',`${f(m.used_gb,1)}/${f(m.total_gb,1)} GB`,`${f(m.available_gb,1)} GB available`) ].join('');chart(t.logs);
let cfg=d.config;document.getElementById('config').innerHTML=Object.entries(cfg).map(([k,v])=>`<div>${k}</div><div class="mono">${v}</div>`).join('');
document.getElementById('dist').innerHTML=table(['Lesion','Positive','Negative'],d.dataset.distribution.map(x=>`<tr><td>${x.lesion}</td><td class="num ok">${x.positive}</td><td class="num">${x.negative}</td></tr>`));
document.getElementById('tiers').innerHTML=table(['Tier','Rows','Meaning'],d.dataset.tiers.map(x=>`<tr><td class="${x.tier==='S0'?'ok':''}">${x.tier}</td><td class="num">${x.rows}</td><td>${x.meaning}</td></tr>`));
document.getElementById('evals').innerHTML=table(['Set','Rows','Purpose'],d.dataset.evals.map(x=>`<tr><td>${x.name}</td><td class="num">${x.rows}</td><td>${x.purpose}</td></tr>`));
document.getElementById('metricCount').textContent=`${t.logs.length} metric points`;document.getElementById('metrics').innerHTML=t.logs.length?table(['Step','Epoch','Loss','LR','Grad norm'],t.logs.slice(-25).reverse().map(x=>`<tr><td>${x.step}</td><td>${f(x.epoch,3)}</td><td class="num">${f(x.loss,4)}</td><td class="mono">${x.learning_rate?x.learning_rate.toExponential(2):'—'}</td><td class="num">${f(x.grad_norm,3)}</td></tr>`)):'<div class="empty">Waiting for first optimizer metrics</div>';
document.getElementById('healthBadge').textContent=d.health.train_processes+' train processes';document.getElementById('health').innerHTML=table(['Item','Value'],Object.entries(d.health).map(([k,v])=>`<tr><td>${k}</td><td class="mono">${v}</td></tr>`));document.getElementById('logPath').textContent=d.log_path;document.getElementById('tail').textContent=d.tail}refresh();setInterval(refresh,2500);
</script></body></html>"""


def run(args: list[str], timeout: int = 3) -> str:
    try:
        return subprocess.run(args, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=timeout).stdout.strip()
    except Exception:
        return ""


def read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def tail(path: Path, n: int = 100) -> str:
    if not path.exists():
        return ""
    return "\n".join(path.read_text(encoding="utf-8", errors="replace").splitlines()[-n:])


def gpu() -> dict[str, Any]:
    text = run(["nvidia-smi", "--query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw", "--format=csv,noheader,nounits"])
    if not text:
        return {"available": False}
    p = [x.strip() for x in text.splitlines()[0].split(",")]
    return {"available": True, "name": p[0], "util": int(float(p[1])), "mem_used": int(float(p[2])), "mem_total": int(float(p[3])), "temp": int(float(p[4])), "power": round(float(p[5]), 1)}


def memory() -> dict[str, Any]:
    vals = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        key, value = line.split(":", 1)
        vals[key] = int(value.strip().split()[0])
    total, avail = vals["MemTotal"] / 1048576, vals["MemAvailable"] / 1048576
    return {"total_gb": total, "used_gb": total - avail, "available_gb": avail}


def _float(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except Exception:
        return None


def _parse_duration_seconds(value: str) -> float | None:
    # tqdm elapsed can be H:MM:SS or MM:SS.
    try:
        left = value.split("<", 1)[0].split(",", 1)[0].strip()
        parts = [int(x) for x in left.split(":")]
        if len(parts) == 3:
            return parts[0] * 3600 + parts[1] * 60 + parts[2]
        if len(parts) == 2:
            return parts[0] * 60 + parts[1]
    except Exception:
        return None
    return None


def parse_log(path: Path, started: float) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""
    max_match = re.findall(r"Total optimization steps\s*=\s*([0-9,]+)", text)
    max_steps = int(max_match[-1].replace(",", "")) if max_match else 0
    logs = []
    for line in text.splitlines():
        if line.startswith("{") and "'loss'" in line:
            try:
                item = ast.literal_eval(line)
            except Exception:
                continue
            epoch = _float(item.get("epoch"))
            logs.append({
                "step": int(item.get("step") or round((epoch or 0) * max_steps) or len(logs) * 10 + 10),
                "epoch": epoch,
                "loss": _float(item.get("loss")),
                "learning_rate": _float(item.get("learning_rate")),
                "grad_norm": _float(item.get("grad_norm")),
            })
    progress = re.findall(r"(\d+)%\|.*?\|\s*(\d+)/(\d+)\s*\[([^\]]+)\]", text)
    trainer_progress = [p for p in progress if max_steps and int(p[2]) == max_steps]
    step = int(trainer_progress[-1][1]) if trainer_progress else (logs[-1]["step"] if logs else 0)
    elapsed = _parse_duration_seconds(trainer_progress[-1][3]) if trainer_progress else None
    if elapsed is None:
        elapsed = time.time() - started
    seconds_per_step = elapsed / step if step else None
    eta = seconds_per_step * (max_steps - step) if seconds_per_step and max_steps else None
    processes = run(["pgrep", "-fc", "llamafactory-cli train.*stage1_en_cot.yaml"])
    running = int(processes or 0) > 0
    completed = (path.parent.parent / "saves/qwen3-vl-8b-fundus/lora/stage1_en_cot/all_results.json").exists()
    phase = "optimizer steps" if step else ("tokenizing / loading model" if running else "not running")
    return {"status": "completed" if completed else "running" if running else "stopped", "phase": phase, "step": step, "max_steps": max_steps, "elapsed_seconds": elapsed, "eta_seconds": eta, "seconds_per_step": seconds_per_step, "samples_per_second": 16 / seconds_per_step if seconds_per_step else None, "latest_loss": logs[-1].get("loss") if logs else None, "latest_lr": logs[-1].get("learning_rate") if logs else None, "logs": logs[-250:]}


def dataset_state(stats_path: Path) -> dict[str, Any]:
    s = read_json(stats_path)
    train = s.get("sets", {}).get("train", {})
    dist = []
    tiers = Counter()
    for raw, count in train.get("counts", {}).items():
        try:
            lesion, state, tier, source = ast.literal_eval(raw)
        except Exception:
            continue
        tiers[tier] += count
    groups = train.get("unique_image_groups", {})
    for lesion in ("MA", "HE", "EX", "SE", "IRMA", "NV"):
        dist.append({"lesion": lesion, "positive": sum(v for k, v in train.get("counts", {}).items() if k.startswith(f"('{lesion}', 'present'")), "negative": sum(v for k, v in train.get("counts", {}).items() if k.startswith(f"('{lesion}', 'absent'"))})
    meanings = {"S0": "direct pixel mask", "S1": "explicit lesion label", "S2": "validated RetSAM", "S3": "cleaning-rule negative", "S4": "grade-rule weak negative"}
    sets = s.get("sets", {})
    evals = [{"name": "Gold dev", "rows": sets.get("gold_dev", {}).get("n", 0), "purpose": "DDR S0 checkpoint validation"}, {"name": "Gold test", "rows": sets.get("gold_test", {}).get("n", 0), "purpose": "locked Main-4 test"}, {"name": "Weak challenge", "rows": sets.get("weak_negative_challenge", {}).get("n", 0), "purpose": "false-positive stress test"}, {"name": "IRMA locked", "rows": sets.get("irma_locked", {}).get("n", 0), "purpose": "rare lesion recall"}, {"name": "NV locked", "rows": sets.get("nv_locked", {}).get("n", 0), "purpose": "rare lesion recall"}]
    return {"distribution": dist, "tiers": [{"tier": k, "rows": tiers[k], "meaning": meanings[k]} for k in ("S0", "S1", "S2", "S3", "S4")], "evals": evals}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8788)
    ap.add_argument("--root", type=Path, default=Path("/workspace/LLaMA-Factory"))
    args = ap.parse_args()
    root = args.root
    log = root / "logs/stage1_en_cot_arm_b_train.log"
    stats = root / "data/annotation_v4/fundus_stage1_en_cot_stats.json"
    config_path = Path("/workspace/fundus-qwen3vl-project/configs/train/stage1_en_cot.yaml")
    cfg = yaml.safe_load(config_path.read_text())
    started = log.stat().st_mtime if log.exists() else time.time()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args): return
        def do_GET(self):  # noqa: N802
            if self.path.startswith("/api/state"):
                disk = shutil.disk_usage(root)
                payload = {"now": time.time(), "train": parse_log(log, started), "gpu": gpu(), "memory": memory(), "dataset": dataset_state(stats), "config": {"Base model": cfg["model_name_or_path"], "LoRA": f"r={cfg['lora_rank']}, alpha={cfg['lora_alpha']}, dropout={cfg['lora_dropout']}", "Vision tower": "LoRA enabled", "Projector": "frozen", "Learning rate": cfg["learning_rate"], "Effective batch": cfg["per_device_train_batch_size"] * cfg["gradient_accumulation_steps"], "Epochs": cfg["num_train_epochs"], "Image max pixels": cfg["image_max_pixels"], "Precision": "bf16", "Eval set": cfg["eval_dataset"]}, "health": {"train_processes": int(run(["pgrep", "-fc", "llamafactory-cli train.*stage1_en_cot.yaml"]) or 0), "log_size_mb": round(log.stat().st_size / 1048576, 2) if log.exists() else 0, "output_exists": str((root / cfg["output_dir"]).exists()), "disk_free_gb": round(disk.free / 1073741824, 1), "hostname": socket.gethostname(), "public_bind": f"{args.host}:{args.port}"}, "log_path": str(log), "tail": tail(log)}
                body = json.dumps(payload, ensure_ascii=False).encode()
                self.send_response(200); self.send_header("Content-Type", "application/json; charset=utf-8"); self.send_header("Cache-Control", "no-store"); self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)
            else:
                body = HTML.encode()
                self.send_response(200); self.send_header("Content-Type", "text/html; charset=utf-8"); self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"Stage1 Arm B monitor listening on http://{args.host}:{args.port}/", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
