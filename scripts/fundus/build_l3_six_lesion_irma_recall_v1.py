#!/usr/bin/env python3
"""Build an IRMA-recall repair mix on top of the six-lesion L3 adapter.

The previous six-lesion pilot preserved MA/HE/EX/SE and improved NV, but IRMA
became too conservative.  This mix increases IRMA-positive exposure, keeps hard
IRMA negatives with NV present, and replays NV plus the four calibrated lesions.
It remains a pure L3 lesion-presence stage and does not add DR grading samples.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import build_stage2_lite as stage2


BASE = Path("data/annotation")
L3_FULL = BASE / "fundus_l3_targeted_calib_v3_full_sft.jsonl"
NV_TRAIN = BASE / "fundus_l3_nv_single_train_sft.jsonl"
IRMA_TRAIN = BASE / "fundus_l3_irma_single_train_sft.jsonl"

FOUR_TASKS = ["L3_MA_single", "L3_HE_single", "L3_EX_single", "L3_SE_single"]
TASKS = FOUR_TASKS + ["L3_NV_single", "L3_IRMA_single"]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def clone(row: dict[str, Any]) -> dict[str, Any]:
    return json.loads(json.dumps(row, ensure_ascii=False))


def answer_obj(row: dict[str, Any]) -> dict[str, Any]:
    return json.loads(row["messages"][-1]["content"].split("【JSON】", 1)[-1].strip())


def take_cycle(rows: list[dict[str, Any]], n: int) -> list[dict[str, Any]]:
    if n <= 0 or not rows:
        return []
    out: list[dict[str, Any]] = []
    while len(out) < n:
        out.extend(rows[: min(len(rows), n - len(out))])
    return out


def normalize(row: dict[str, Any], stage: str, component: str) -> dict[str, Any]:
    item = clone(row)
    obj = answer_obj(item)
    meta = dict(item.get("meta", {}))
    meta["task"] = obj.get("task", meta.get("task"))
    meta["lesion"] = obj.get("lesion", meta.get("lesion"))
    meta["present"] = bool(obj.get("present"))
    meta["stage_mix"] = stage
    meta["mix_component"] = component
    item["meta"] = meta
    return item


def rewrite_irma_answer(row: dict[str, Any], stage: str, component: str) -> dict[str, Any]:
    item = normalize(row, stage, component)
    obj = answer_obj(item)
    present = bool(obj.get("present"))
    source = obj.get("source", "fgadr_lesion_only_sft_v3")
    old = item["messages"][-1]["content"]
    nv_present = "NV=PRESENT" in old
    nv_state = "present" if nv_present else "absent_or_unknown"
    item["messages"][0]["content"] = (
        "你是眼底病灶识别助手。本题只判断 IRMA 是否存在；不得输出 DR grade，"
        "也不得把 NV、出血或普通血管迂曲当作 IRMA。"
    )
    item["messages"][1]["content"] = (
        "<image>\n请只判断图中是否可见视网膜内微血管异常（IRMA）：重点寻找视网膜内不规则、"
        "扩张、迂曲的短段血管或旁路样血管；若是盘面/视网膜表面的细小新生血管网，应归为 NV 而不是 IRMA。"
    )
    if present:
        conclusion = (
            "支持 IRMA 阳性；依据是视网膜内异常血管形态，而不是 DR grade 或 NV 标签。"
        )
        evidence = f"IRMA present=true; NV_context={nv_state}; source={source}"
    else:
        conclusion = (
            "未见可靠 IRMA 阳性证据；即使存在 NV 或其他病灶，也不能替代 IRMA。"
        )
        evidence = f"IRMA present=false; NV_context={nv_state}; source={source}"
    new_obj = {
        "task": "L3_IRMA_single",
        "lesion": "IRMA",
        "present": present,
        "count": obj.get("count", "unknown"),
        "area": obj.get("area", "unknown"),
        "nv_context": nv_state,
        "source": source,
    }
    item["messages"][-1]["content"] = (
        "【观察】先只观察 IRMA：寻找视网膜内不规则扩张、迂曲、旁路样的异常血管段；"
        "同时排除盘面或视网膜表面的 NV、普通血管走行、出血和渗出。\n\n"
        f"【证据】{evidence}\n\n"
        f"【结论】{conclusion}本题不输出 DR 分级，也不合并其他病灶结论。\n\n"
        "【JSON】\n"
        + json.dumps(new_obj, ensure_ascii=False, separators=(",", ":"))
    )
    return item


def collect_sources() -> dict[str, list[dict[str, Any]]]:
    sources = read_jsonl(L3_FULL) + read_jsonl(NV_TRAIN) + read_jsonl(IRMA_TRAIN)
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in sources:
        try:
            obj = answer_obj(row)
        except Exception:
            continue
        task = obj.get("task")
        if task in TASKS:
            buckets[task].append(row)
    return buckets


def select_balanced(rows: list[dict[str, Any]], task: str, n: int, stage: str) -> list[dict[str, Any]]:
    buckets = {True: [], False: []}
    for row in rows:
        obj = answer_obj(row)
        if obj.get("task") == task:
            buckets[bool(obj.get("present"))].append(row)
    pos = take_cycle(stage2.stable_shuffle(buckets[True], f"{stage}_{task}_pos"), n // 2 + n % 2)
    neg = take_cycle(stage2.stable_shuffle(buckets[False], f"{stage}_{task}_neg"), n // 2)
    return [normalize(row, stage, task) for row in (pos + neg)[:n]]


def select_irma_focus(rows: list[dict[str, Any]], pos_n: int, neg_n: int, stage: str) -> list[dict[str, Any]]:
    pos: list[dict[str, Any]] = []
    neg_hard: list[dict[str, Any]] = []
    neg_easy: list[dict[str, Any]] = []
    for row in rows:
        obj = answer_obj(row)
        if obj.get("task") != "L3_IRMA_single":
            continue
        if bool(obj.get("present")):
            pos.append(row)
        elif "NV=PRESENT" in row["messages"][-1]["content"]:
            neg_hard.append(row)
        else:
            neg_easy.append(row)

    pos_sel = take_cycle(stage2.stable_shuffle(pos, f"{stage}_irma_pos"), pos_n)
    hard_n = min(max(neg_n // 3, 1), neg_n)
    hard = take_cycle(stage2.stable_shuffle(neg_hard, f"{stage}_irma_neg_hard"), hard_n)
    easy = take_cycle(stage2.stable_shuffle(neg_easy, f"{stage}_irma_neg_easy"), neg_n - hard_n)
    selected = pos_sel + hard + easy
    return [rewrite_irma_answer(row, stage, "L3_IRMA_single_focus") for row in selected]


def summarize(rows: list[dict[str, Any]], source_buckets: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    tasks = Counter()
    present = Counter()
    components = Counter()
    missing = 0
    source_counts = {}
    for task, src_rows in source_buckets.items():
        c = Counter(bool(answer_obj(row).get("present")) for row in src_rows)
        source_counts[task] = {"total": len(src_rows), "positive": c[True], "negative": c[False]}
    for row in rows:
        meta = row.get("meta", {})
        task = meta.get("task", "unknown")
        tasks[task] += 1
        present[(task, bool(meta.get("present")))] += 1
        components[meta.get("mix_component", "unknown")] += 1
        for image in row.get("images", []):
            if not Path("data", image).exists():
                missing += 1
    return {
        "n": len(rows),
        "tasks": dict(tasks),
        "present": {str(k): v for k, v in sorted(present.items(), key=lambda x: str(x[0]))},
        "components": dict(components),
        "source_counts": source_counts,
        "missing_images": missing,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--four-per-task", type=int, default=600)
    parser.add_argument("--nv-total", type=int, default=800)
    parser.add_argument("--irma-pos", type=int, default=2400)
    parser.add_argument("--irma-neg", type=int, default=800)
    parser.add_argument("--output", default="data/annotation/fundus_l3_six_lesion_irma_recall_v1_sft.jsonl")
    args = parser.parse_args()

    stage = "l3_six_lesion_irma_recall_v1"
    sources = collect_sources()
    rows: list[dict[str, Any]] = []
    for task in FOUR_TASKS:
        rows.extend(select_balanced(sources.get(task, []), task, args.four_per_task, stage))
    rows.extend(select_balanced(sources.get("L3_NV_single", []), "L3_NV_single", args.nv_total, stage))
    rows.extend(select_irma_focus(sources.get("L3_IRMA_single", []), args.irma_pos, args.irma_neg, stage))
    rows = stage2.stable_shuffle(rows, f"{stage}_final")

    out_path = Path(args.output)
    write_jsonl(out_path, rows)
    stats = {
        "output": str(out_path),
        "four_per_task": args.four_per_task,
        "nv_total": args.nv_total,
        "irma_pos": args.irma_pos,
        "irma_neg": args.irma_neg,
        "requested_total": args.four_per_task * len(FOUR_TASKS) + args.nv_total + args.irma_pos + args.irma_neg,
        "summary": summarize(rows, sources),
    }
    stats_path = BASE / "fundus_l3_six_lesion_irma_recall_v1_stats.json"
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
