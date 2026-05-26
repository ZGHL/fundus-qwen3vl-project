#!/usr/bin/env python3
"""Public dashboard for Fundus Qwen3-VL SFT and evaluation runs."""
from __future__ import annotations

import argparse
import ast
import json
import re
import shutil
import subprocess
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

try:
    import yaml
except Exception:  # pragma: no cover
    yaml = None

LLAMA_ROOT = Path("/workspace/LLaMA-Factory")
SAVES_ROOT = LLAMA_ROOT / "saves/qwen3-vl-8b-fundus/lora"
DECOUPLED_DATASET_STATS = LLAMA_ROOT / "data/annotation_v4/fundus_lesion_perception_en_cot_full_stats.json"
JOINT_MIX_DATASET_STATS = LLAMA_ROOT / "data/annotation_v4/fundus_l3_joint_mix_full_stats.json"
SINGLE_ROW_MIX_DATASET_STATS = LLAMA_ROOT / "data/annotation_v4/fundus_l3_single_row_mix_full_stats.json"

HTML = r"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Fundus Qwen3-VL Experiment Monitor</title>
  <style>
    :root{--bg:#f4f6f8;--panel:#fff;--ink:#17202a;--muted:#667085;--line:#d8dee8;--blue:#2457a6;--cyan:#247c8c;--green:#147a52;--amber:#a15c10;--red:#b42318;--shadow:0 1px 2px rgba(16,24,40,.06),0 8px 24px rgba(16,24,40,.05)}
    *{box-sizing:border-box} body{margin:0;background:var(--bg);color:var(--ink);font-family:Inter,ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI","Microsoft YaHei",sans-serif}
    header{background:#0f2742;color:white;padding:18px 28px;border-bottom:1px solid rgba(255,255,255,.14)} .top{display:flex;align-items:flex-start;justify-content:space-between;gap:18px;max-width:1480px;margin:0 auto}
    h1{font-size:20px;line-height:1.2;margin:0;font-weight:720;letter-spacing:0}.subtitle{margin-top:6px;color:#c7d7ea;font-size:13px}.stamp{font-size:12px;color:#d8e4f2;text-align:right;line-height:1.6}
    main{max-width:1480px;margin:0 auto;padding:18px 28px 32px}.grid{display:grid;gap:12px}.kpis{grid-template-columns:repeat(6,minmax(0,1fr))}.two{grid-template-columns:1.15fr .85fr}.three{grid-template-columns:1fr 1fr 1fr}
    .panel,.metric{background:var(--panel);border:1px solid var(--line);border-radius:8px;box-shadow:var(--shadow)}.metric{padding:13px 14px;min-height:92px}.label{font-size:12px;color:var(--muted);white-space:nowrap;overflow:hidden;text-overflow:ellipsis}.value{font-size:22px;line-height:1.15;font-weight:760;margin-top:7px}.hint{font-size:12px;color:var(--muted);margin-top:6px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
    .panel{padding:14px;margin-top:14px}.panel h2{font-size:15px;margin:0 0 12px;font-weight:720}.panelHead{display:flex;align-items:center;justify-content:space-between;gap:12px;margin-bottom:10px}.panelHead h2{margin:0}
    .badge{display:inline-flex;align-items:center;border:1px solid var(--line);border-radius:999px;padding:3px 8px;font-size:12px;color:#344054;background:#f8fafc}.ok{color:var(--green)}.warn{color:var(--amber)}.bad{color:var(--red)}.blue{color:var(--blue)}.mono{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace}
    table{width:100%;border-collapse:collapse}th,td{text-align:left;border-bottom:1px solid #e7ebf0;padding:8px 9px;font-size:12px;vertical-align:top}th{color:#344054;background:#f8fafc;font-weight:680}tr:last-child td{border-bottom:0}.num{text-align:right;font-variant-numeric:tabular-nums}
    .bar{height:8px;background:#e8edf4;border-radius:999px;overflow:hidden}.fill{height:100%;background:linear-gradient(90deg,var(--blue),var(--cyan));width:0}.stack{display:flex;flex-direction:column;gap:8px}.split{display:flex;align-items:center;justify-content:space-between;gap:12px}.small{font-size:12px;color:var(--muted)}pre{margin:0;background:#101828;color:#e4e7ec;border-radius:8px;padding:12px;max-height:360px;overflow:auto;font-size:12px;line-height:1.45}
    canvas{width:100%;height:190px;border:1px solid #e7ebf0;border-radius:8px;background:#fbfcfe}.empty{padding:28px;color:var(--muted);text-align:center;border:1px dashed var(--line);border-radius:8px;background:#fbfcfe}.truncate{max-width:240px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
    @media (max-width:1180px){.kpis{grid-template-columns:repeat(3,minmax(0,1fr))}.two,.three{grid-template-columns:1fr}.top{display:block}.stamp{text-align:left;margin-top:8px}} @media (max-width:720px){main{padding:14px}header{padding:16px}.kpis{grid-template-columns:1fr 1fr}.value{font-size:18px}}
  </style>
</head>
<body>
<header><div class="top"><div><h1>Fundus Qwen3-VL Experiment Monitor</h1><div class="subtitle">SFT, evaluation, dataset, GPU and output health dashboard</div></div><div class="stamp"><div id="clock">loading</div><div class="mono" id="host"></div></div></div></header>
<main>
  <section class="grid kpis" id="kpis"></section>
  <section class="grid two">
    <div class="panel"><div class="panelHead"><h2>SFT Progress</h2><span class="badge" id="trainStatus">waiting</span></div><div class="stack"><div class="split"><span id="progressText" class="small">No trainer state yet</span><span id="eta" class="small"></span></div><div class="bar"><div class="fill" id="progressFill"></div></div><canvas id="lossChart" width="900" height="260"></canvas></div></div>
    <div class="panel"><div class="panelHead"><h2>Dataset Distribution</h2><span class="badge">six-lesion mix</span></div><div id="datasetTable"></div></div>
  </section>
  <section class="grid two"><div class="panel"><div class="panelHead"><h2>Training Runs</h2><span class="badge" id="runCount"></span></div><div id="runs"></div></div><div class="panel"><div class="panelHead"><h2>Evaluation Runs</h2><span class="badge" id="evalCount"></span></div><div id="evals"></div></div></section>
  <section class="grid three"><div class="panel"><h2>Latest Metrics</h2><div id="metrics"></div></div><div class="panel"><h2>Configured Commands</h2><div id="commands"></div></div><div class="panel"><h2>Artifacts</h2><div id="artifacts"></div></div></section>
  <section class="panel"><div class="panelHead"><h2>Log Tail</h2><span class="badge mono" id="tailPath"></span></div><pre id="tail"></pre></section>
</main>
<script>
const fmt=(v,d=3)=>v===null||v===undefined||v===''?'—':(typeof v==='number'?(Number.isFinite(v)?v.toFixed(d).replace(/\.?0+$/,''):'—'):String(v));const pct=v=>v===null||v===undefined?'—':`${(100*v).toFixed(1)}%`;function clsStatus(s){return s==='completed'?'ok':s==='running'?'blue':s==='failed'?'bad':'warn'}function metric(label,value,hint,klass=''){return `<div class="metric"><div class="label">${label}</div><div class="value ${klass}">${value}</div><div class="hint">${hint||''}</div></div>`}function table(headers,rows){if(!rows||!rows.length)return'<div class="empty">No data yet</div>';return `<table><thead><tr>${headers.map(h=>`<th>${h}</th>`).join('')}</tr></thead><tbody>${rows.join('')}</tbody></table>`}
function drawLoss(logs){const c=document.getElementById('lossChart'),ctx=c.getContext('2d');ctx.clearRect(0,0,c.width,c.height);const pts=(logs||[]).filter(x=>typeof x.loss==='number'&&typeof x.step==='number').slice(-160);ctx.fillStyle='#fbfcfe';ctx.fillRect(0,0,c.width,c.height);ctx.strokeStyle='#d8dee8';ctx.lineWidth=1;for(let i=1;i<5;i++){let y=i*c.height/5;ctx.beginPath();ctx.moveTo(0,y);ctx.lineTo(c.width,y);ctx.stroke()}if(pts.length<2){ctx.fillStyle='#667085';ctx.font='16px system-ui';ctx.fillText('Waiting for trainer loss logs',24,48);return}const xs=pts.map(p=>p.step),ys=pts.map(p=>p.loss),minX=Math.min(...xs),maxX=Math.max(...xs),minY=Math.min(...ys),maxY=Math.max(...ys);const px=x=>40+(x-minX)/(maxX-minX||1)*(c.width-64),py=y=>24+(maxY-y)/(maxY-minY||1)*(c.height-58);ctx.strokeStyle='#2457a6';ctx.lineWidth=3;ctx.beginPath();pts.forEach((p,i)=>{const x=px(p.step),y=py(p.loss);if(i)ctx.lineTo(x,y);else ctx.moveTo(x,y)});ctx.stroke();ctx.fillStyle='#17202a';ctx.font='13px system-ui';ctx.fillText(`loss ${fmt(pts[pts.length-1].loss,4)} @ step ${pts[pts.length-1].step}`,42,22)}
async function refresh(){const d=await(await fetch('/api/state')).json();document.getElementById('clock').textContent=new Date(d.now*1000).toLocaleString();document.getElementById('host').textContent=`${d.host}:${d.port}`;const tr=d.active_train||{},gpu=d.gpu||{},disk=d.disk||{};const p=tr.max_steps?Math.min(1,(tr.global_step||0)/tr.max_steps):0;document.getElementById('progressFill').style.width=(100*p).toFixed(1)+'%';document.getElementById('progressText').textContent=tr.max_steps?`${tr.global_step||0} / ${tr.max_steps} steps (${(100*p).toFixed(1)}%)`:'No trainer state yet';document.getElementById('eta').textContent=tr.remaining_time?`ETA ${tr.remaining_time}`:'';document.getElementById('trainStatus').textContent=tr.status||'waiting';document.getElementById('trainStatus').className=`badge ${clsStatus(tr.status)}`;document.getElementById('kpis').innerHTML=[metric('Training status',tr.status||'waiting',tr.name||'',clsStatus(tr.status)),metric('Latest loss',fmt(tr.latest_loss,4),tr.latest_lr?`lr ${fmt(tr.latest_lr,3)}`:'waiting'),metric('GPU',gpu.available?`${gpu.util_gpu_pct}%`:'unavailable',gpu.available?`${gpu.mem_used_mb}/${gpu.mem_total_mb} MB · ${gpu.temp_c} C`:'nvidia-smi unavailable'),metric('Disk free',disk.free_gb?`${fmt(disk.free_gb,1)} GB`:'—',disk.total_gb?`${fmt(disk.used_gb,1)} / ${fmt(disk.total_gb,1)} GB used`:''),metric('Train images',fmt(d.dataset?.train_total,0),`Decisions ${fmt(d.dataset?.train_decisions,0)}`),metric('Eval images',fmt(d.dataset?.val_total,0),`Val decisions ${fmt(d.dataset?.val_decisions,0)} · NV locked ${fmt(d.dataset?.nv_locked_total,0)}`)].join('');drawLoss(tr.logs||[]);const dist=d.dataset?.distribution||[];document.getElementById('datasetTable').innerHTML=table(['Lesion','Train + / -','Val + / -','NV locked + / -'],dist.map(x=>`<tr><td>${x.lesion}</td><td class="num">${x.train_pos} / ${x.train_neg}</td><td class="num">${x.val_pos} / ${x.val_neg}</td><td class="num">${x.locked_pos||0} / ${x.locked_neg||0}</td></tr>`));document.getElementById('runCount').textContent=`${(d.train_runs||[]).length} configs`;document.getElementById('runs').innerHTML=table(['Name','Status','Step','Loss','Output'],(d.train_runs||[]).map(r=>`<tr><td class="truncate">${r.config}</td><td class="${clsStatus(r.status)}">${r.status}</td><td class="num">${r.max_steps?`${r.global_step}/${r.max_steps}`:fmt(r.global_step,0)}</td><td class="num">${fmt(r.latest_loss,4)}</td><td class="mono truncate">${r.output_dir}</td></tr>`));document.getElementById('evalCount').textContent=`${(d.eval_runs||[]).length} configs`;document.getElementById('evals').innerHTML=table(['Name','Dataset','Status','Rows','Output'],(d.eval_runs||[]).map(r=>`<tr><td class="truncate">${r.config}</td><td class="truncate">${r.dataset||''}</td><td class="${clsStatus(r.status)}">${r.status}</td><td class="num">${fmt(r.pred_rows,0)}</td><td class="mono truncate">${r.output_dir}</td></tr>`));const latest=d.latest_score;document.getElementById('metrics').innerHTML=latest?table(['Metric','Value'],[`<tr><td>JSON parse</td><td class="num">${pct(latest.json_parse_success)}</td></tr>`,`<tr><td>Target consistency</td><td class="num">${pct(latest.target_lesion_consistency)}</td></tr>`,`<tr><td>Macro F1</td><td class="num">${pct(latest.macro?.f1)}</td></tr>`,`<tr><td>Rare F1</td><td class="num">${pct(latest.rare_lesion_macro?.f1)}</td></tr>`,`<tr><td>No-grade output</td><td class="num">${pct(latest.no_grade_output_rate)}</td></tr>`]):'<div class="empty">Current six-lesion mix run is training or waiting for fresh evaluation scores</div>';document.getElementById('commands').innerHTML=table(['Stage','Command'],(d.commands||[]).map(x=>`<tr><td>${x.stage}</td><td class="mono">${x.command}</td></tr>`));document.getElementById('artifacts').innerHTML=table(['Artifact','Path'],(d.artifacts||[]).map(x=>`<tr><td>${x.name}</td><td class="mono truncate">${x.path}</td></tr>`));document.getElementById('tailPath').textContent=d.tail_path||'';document.getElementById('tail').textContent=d.tail||''}refresh();setInterval(refresh,3000);
</script>
</body></html>"""


def run(cmd: list[str], timeout: int = 3) -> str:
    try:
        return subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=timeout).stdout.strip()
    except Exception as exc:
        return f"{type(exc).__name__}: {exc}"


def read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def read_yaml(path: Path) -> dict[str, Any]:
    if yaml is None or not path.exists():
        return {}
    try:
        return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}


def tail(path: Path, lines: int = 120) -> str:
    if not path.exists():
        return ""
    try:
        return "\n".join(path.read_text(encoding="utf-8", errors="replace").splitlines()[-lines:])
    except Exception as exc:
        return f"{type(exc).__name__}: {exc}"


def gpu_state() -> dict[str, Any]:
    out = run(["nvidia-smi", "--query-gpu=name,memory.total,memory.used,utilization.gpu,temperature.gpu", "--format=csv,noheader,nounits"])
    if not out or "NVIDIA-SMI has failed" in out:
        return {"available": False}
    parts = [p.strip() for p in out.splitlines()[0].split(",")]
    if len(parts) < 5:
        return {"available": False, "raw": out}
    return {
        "available": True,
        "name": parts[0],
        "mem_total_mb": int(float(parts[1])),
        "mem_used_mb": int(float(parts[2])),
        "util_gpu_pct": int(float(parts[3])),
        "temp_c": int(float(parts[4])),
    }


def disk_state(path: Path) -> dict[str, Any]:
    usage = shutil.disk_usage(path)
    gb = 1024**3
    return {"total_gb": usage.total / gb, "used_gb": usage.used / gb, "free_gb": usage.free / gb}


def trainer_logs(out_dir: Path) -> list[dict[str, Any]]:
    logs = []
    state = read_json(out_dir / "trainer_state.json")
    for item in state.get("log_history", []):
        if "step" in item:
            logs.append(
                {
                    "step": item.get("step"),
                    "epoch": item.get("epoch"),
                    "loss": item.get("loss"),
                    "learning_rate": item.get("learning_rate"),
                    "grad_norm": item.get("grad_norm"),
                }
            )
    return logs


def log_file_training_state(log_dir: Path) -> dict[str, Any]:
    if not log_dir.exists():
        return {"logs": [], "max_steps": 0}
    files = sorted(log_dir.glob("*.log"), key=lambda p: p.stat().st_mtime)
    logs: list[dict[str, Any]] = []
    max_steps = 0
    for file in files:
        try:
            lines = file.read_text(encoding="utf-8", errors="replace").splitlines()
        except Exception:
            continue
        for line in lines:
            match = re.search(r"Total optimization steps =\s*(\d+)", line)
            if match:
                max_steps = int(match.group(1))
            if line.startswith("{") and "'loss'" in line:
                try:
                    item = ast.literal_eval(line)
                except Exception:
                    continue
                step = len(logs) * 10 + 10
                logs.append(
                    {
                        "step": step,
                        "epoch": float(item.get("epoch", 0) or 0),
                        "loss": float(item.get("loss", 0) or 0),
                        "learning_rate": float(item.get("learning_rate", 0) or 0),
                        "grad_norm": float(item.get("grad_norm", 0) or 0),
                    }
                )
    return {"logs": logs, "max_steps": max_steps}


def _process_running(pattern: str) -> bool:
    out = run(["pgrep", "-af", pattern], timeout=1)
    return bool(out and "TimeoutExpired" not in out)


def parse_run(config_path: Path, kind: str) -> dict[str, Any]:
    cfg = read_yaml(config_path)
    out = cfg.get("output_dir") or ""
    out_dir = LLAMA_ROOT / out if out and not Path(out).is_absolute() else Path(out)
    log_dir = LLAMA_ROOT / "logs" / config_path.stem
    logs = trainer_logs(out_dir)
    log_state = log_file_training_state(log_dir)
    newest_log_mtime = max((p.stat().st_mtime for p in log_dir.glob("*.log")), default=0) if log_dir.exists() else 0
    result_mtime = max(
        [p.stat().st_mtime for p in (out_dir / "all_results.json", out_dir / "trainer_state.json") if p.exists()],
        default=0,
    )
    if not logs or newest_log_mtime > result_mtime:
        logs = log_state.get("logs", [])
    state = read_json(out_dir / "trainer_state.json")
    all_results = read_json(out_dir / "all_results.json")
    latest = logs[-1] if logs else {}
    pred = out_dir / "generated_predictions.jsonl"
    log_files = sorted(log_dir.glob("*.log")) if log_dir.exists() else []
    out_mtime = out_dir.stat().st_mtime if out_dir.exists() else 0
    run_mtime = max(out_mtime, newest_log_mtime, result_mtime)
    process_running = _process_running(rf"llamafactory-cli train .*{re.escape(config_path.name)}")
    status = "not_started"
    if process_running:
        status = "running"
    elif pred.exists() or all_results:
        status = "completed"
    elif logs and not all_results:
        status = "running"
    elif out_dir.exists():
        status = "created"
    if status == "running":
        global_step = int(latest.get("step") or 0)
        max_steps = int(log_state.get("max_steps") or 0)
    else:
        global_step = int(state.get("global_step") or latest.get("step") or 0)
        max_steps = int(state.get("max_steps") or log_state.get("max_steps") or 0)
    return {
        "kind": kind,
        "config": config_path.name,
        "dataset": cfg.get("dataset") or cfg.get("eval_dataset"),
        "output_dir": str(out_dir.relative_to(LLAMA_ROOT)) if str(out_dir).startswith(str(LLAMA_ROOT)) else str(out_dir),
        "status": status,
        "global_step": global_step,
        "max_steps": max_steps,
        "latest_loss": latest.get("loss"),
        "latest_lr": latest.get("learning_rate"),
        "logs": logs[-200:],
        "pred_rows": sum(1 for _ in pred.open(encoding="utf-8")) if pred.exists() else 0,
        "mtime": run_mtime,
    }


def _pair_count(mapping: dict[str, Any], lesion: str, state: str) -> int:
    return int(mapping.get(str((lesion, state)), 0) or mapping.get(f"('{lesion}', '{state}')", 0) or 0)


def _joint_dataset_state(stats: dict[str, Any]) -> dict[str, Any]:
    outputs = stats.get("outputs", {})
    train = outputs.get("train", {})
    val = outputs.get("val_subset", {})
    locked = outputs.get("nv_locked", {}).get("per_lesion", {})
    rows = []
    for lesion in ("HE", "EX", "MA", "SE", "IRMA", "NV"):
        train_per = train.get("per_lesion", {})
        val_per = val.get("per_lesion", {})
        rows.append(
            {
                "lesion": lesion,
                "train_pos": _pair_count(train_per, lesion, "present"),
                "train_neg": _pair_count(train_per, lesion, "absent"),
                "val_pos": _pair_count(val_per, lesion, "present"),
                "val_neg": _pair_count(val_per, lesion, "absent"),
                "locked_pos": _pair_count(locked, lesion, "present"),
                "locked_neg": _pair_count(locked, lesion, "absent"),
            }
        )
    return {
        "name": "six_lesion_joint_mix",
        "train_total": train.get("rows", 0),
        "val_total": val.get("rows", 0),
        "nv_locked_total": outputs.get("nv_locked", {}).get("rows", 0),
        "train_decisions": train.get("lesion_decisions", 0),
        "val_decisions": val.get("lesion_decisions", 0),
        "nv_locked_decisions": outputs.get("nv_locked", {}).get("lesion_decisions", 0),
        "distribution": rows,
    }


def _decoupled_dataset_state(stats: dict[str, Any]) -> dict[str, Any]:
    per = stats.get("per_lesion", {})
    locked = stats.get("nv_locked_eval", {}).get("counts", {})
    rows = []
    for lesion in ("HE", "EX", "MA", "SE", "IRMA", "NV"):
        item = per.get(lesion, {})
        rows.append(
            {
                "lesion": lesion,
                "train_pos": item.get("train_sampled_present", 0),
                "train_neg": item.get("train_sampled_absent", 0),
                "val_pos": item.get("val_present", 0),
                "val_neg": item.get("val_absent", 0),
                "locked_pos": locked.get(str((lesion, "present")), 0),
                "locked_neg": locked.get(str((lesion, "absent")), 0),
            }
        )
    return {
        "name": "lesion_perception_en_cot_full",
        "train_total": stats.get("train_total", 0),
        "val_total": stats.get("val_total", 0),
        "nv_locked_total": sum(locked.values()) if locked else 0,
        "distribution": rows,
    }


def _single_row_mix_dataset_state(stats: dict[str, Any]) -> dict[str, Any]:
    train_counts = stats.get("outputs", {}).get("train", {}).get("counts", {})
    val_counts = stats.get("outputs", {}).get("val_subset", {}).get("counts", {})
    locked_counts = stats.get("outputs", {}).get("nv_locked", {}).get("counts", {})
    rows = []
    for lesion in ("HE", "EX", "MA", "SE", "IRMA", "NV"):
        rows.append(
            {
                "lesion": lesion,
                "train_pos": _pair_count(train_counts, lesion, "present"),
                "train_neg": _pair_count(train_counts, lesion, "absent"),
                "val_pos": _pair_count(val_counts, lesion, "present"),
                "val_neg": _pair_count(val_counts, lesion, "absent"),
                "locked_pos": _pair_count(locked_counts, lesion, "present"),
                "locked_neg": _pair_count(locked_counts, lesion, "absent"),
            }
        )
    outputs = stats.get("outputs", {})
    return {
        "name": "arm_c_single_row_mix",
        "train_total": outputs.get("train", {}).get("rows", 0),
        "val_total": outputs.get("val_subset", {}).get("rows", 0),
        "nv_locked_total": outputs.get("nv_locked", {}).get("rows", 0),
        "train_decisions": outputs.get("train", {}).get("rows", 0),
        "val_decisions": outputs.get("val_subset", {}).get("rows", 0),
        "nv_locked_decisions": outputs.get("nv_locked", {}).get("rows", 0),
        "distribution": rows,
    }


def dataset_state() -> dict[str, Any]:
    single = read_json(SINGLE_ROW_MIX_DATASET_STATS)
    if single:
        return _single_row_mix_dataset_state(single)
    joint = read_json(JOINT_MIX_DATASET_STATS)
    if joint:
        return _joint_dataset_state(joint)
    return _decoupled_dataset_state(read_json(DECOUPLED_DATASET_STATS))


def latest_score() -> dict[str, Any] | None:
    if not SAVES_ROOT.exists():
        return None
    if (
        _process_running("run_l3_joint_mix_pipeline.sh")
        or _process_running("run_l3_single_row_mix_pipeline.sh")
        or _process_running("llamafactory-cli train examples/train_lora/l3_joint_mix")
        or _process_running("llamafactory-cli train examples/train_lora/l3_single_row_mix")
    ):
        return None
    score_files = list(SAVES_ROOT.glob("**/*score*.json"))
    single = sorted([p for p in score_files if "l3_single_row_mix" in str(p)], key=lambda p: p.stat().st_mtime, reverse=True)
    joint = sorted([p for p in score_files if "l3_joint_mix" in str(p)], key=lambda p: p.stat().st_mtime, reverse=True)
    if not single and SINGLE_ROW_MIX_DATASET_STATS.exists():
        return None
    if not single and not joint and JOINT_MIX_DATASET_STATS.exists():
        return None
    legacy = sorted([p for p in score_files if "lesion_perception" in str(p)], key=lambda p: p.stat().st_mtime, reverse=True)
    candidates = single or joint or legacy
    if not candidates:
        return None
    obj = read_json(candidates[0])
    obj["_path"] = str(candidates[0])
    obj["_experiment"] = "l3_single_row_mix" if single else ("l3_joint_mix" if joint else "lesion_perception_en_cot_full")
    return obj


def collect_state(host: str, port: int) -> dict[str, Any]:
    all_cfgs = sorted((LLAMA_ROOT / "examples/train_lora").glob("*.yaml")) + sorted((LLAMA_ROOT / "examples/eval").glob("*.yaml"))
    relevant_cfgs = [p for p in all_cfgs if "lesion_perception" in p.name or "l3_joint_mix" in p.name or "l3_single_row_mix" in p.name]
    train_cfgs = [p for p in relevant_cfgs if read_yaml(p).get("do_train") is True]
    eval_cfgs = [p for p in relevant_cfgs if read_yaml(p).get("do_predict") is True]
    train_runs = [parse_run(p, "train") for p in train_cfgs]
    eval_runs = [parse_run(p, "eval") for p in eval_cfgs]
    active_train = max(train_runs, key=lambda r: (r.get("status") == "running", r.get("mtime", 0)), default={})
    pred_files = list(SAVES_ROOT.glob("**/generated_predictions.jsonl")) if SAVES_ROOT.exists() else []
    save_logs = list(SAVES_ROOT.glob("**/trainer_state.json")) + pred_files if SAVES_ROOT.exists() else []
    pipeline_logs = list((LLAMA_ROOT / "logs").glob("**/*.log")) if (LLAMA_ROOT / "logs").exists() else []
    relevant_logs = [p for p in pipeline_logs if "lesion_perception" in str(p) or "l3_joint_mix" in str(p) or "l3_single_row_mix" in str(p)]
    log_candidates = sorted(save_logs + relevant_logs, key=lambda p: p.stat().st_mtime, reverse=True)
    tail_path = log_candidates[0] if log_candidates else Path()
    return {
        "now": time.time(),
        "host": host,
        "port": port,
        "gpu": gpu_state(),
        "disk": disk_state(Path("/workspace")),
        "dataset": dataset_state(),
        "train_runs": train_runs,
        "eval_runs": eval_runs,
        "active_train": active_train,
        "prediction_files": len(pred_files),
        "latest_score": latest_score(),
        "commands": [
            {"stage": "Joint mix SFT", "command": "llamafactory-cli train examples/train_lora/l3_joint_mix_full.yaml"},
            {"stage": "Joint mix eval", "command": "bash /workspace/fundus-qwen3vl-project/scripts/run_l3_joint_mix_pipeline.sh"},
            {"stage": "Joint mix score", "command": "python /workspace/fundus-qwen3vl-project/scripts/fundus/score_l3_joint_mix_predictions.py <generated_predictions.jsonl> --json-out <score.json>"},
            {"stage": "Arm C SFT", "command": "llamafactory-cli train examples/train_lora/l3_single_row_mix_full.yaml"},
            {"stage": "Arm C eval", "command": "bash /workspace/fundus-qwen3vl-project/scripts/run_l3_single_row_mix_pipeline.sh"},
            {"stage": "Decoupled SFT", "command": "llamafactory-cli train examples/train_lora/lesion_perception_en_cot_full.yaml"},
        ],
        "artifacts": [
            {"name": "Joint train config", "path": "examples/train_lora/l3_joint_mix_full.yaml"},
            {"name": "Joint val subset config", "path": "examples/train_lora/l3_joint_mix_predict_val_subset.yaml"},
            {"name": "Joint dataset stats", "path": str(JOINT_MIX_DATASET_STATS)},
            {"name": "Joint scorer", "path": "scripts/fundus/score_l3_joint_mix_predictions.py"},
            {"name": "Arm C train config", "path": "examples/train_lora/l3_single_row_mix_full.yaml"},
            {"name": "Arm C dataset stats", "path": str(SINGLE_ROW_MIX_DATASET_STATS)},
            {"name": "Decoupled train config", "path": "examples/train_lora/lesion_perception_en_cot_full.yaml"},
        ],
        "tail_path": str(tail_path),
        "tail": tail(tail_path, 100) if tail_path else "",
    }


class Handler(BaseHTTPRequestHandler):
    host = "0.0.0.0"
    port = 8790

    def log_message(self, *_args: Any) -> None:
        return

    def send_body(self, status: int, content_type: str, body: bytes) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        if self.path.startswith("/api/state"):
            body = json.dumps(collect_state(self.host, self.port), ensure_ascii=False).encode("utf-8")
            self.send_body(200, "application/json; charset=utf-8", body)
            return
        self.send_body(200, "text/html; charset=utf-8", HTML.encode("utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8790)
    args = parser.parse_args()
    Handler.host = args.host
    Handler.port = args.port
    httpd = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"monitor listening on http://{args.host}:{args.port}/", flush=True)
    httpd.serve_forever()


if __name__ == "__main__":
    main()
