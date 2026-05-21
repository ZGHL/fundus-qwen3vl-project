#!/usr/bin/env python3
"""Build a balanced lesion-presence eval set with positive and negative examples."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any


LESION_CN = {"MA": "微动脉瘤", "HE": "出血", "EX": "硬性渗出", "SE": "软性渗出"}
LESION_CUE = {
    "MA": "微小红色圆点样病灶",
    "HE": "暗红点片状或不规则斑块状病灶",
    "EX": "亮黄色、边界较清楚的沉积样病灶",
    "SE": "灰白色、棉絮样、边界较模糊的病灶",
}
STRONG_SOURCES = {"strong_mask_stage1_easy", "fgadr_lesion_only_sft_v3"}
POS_SOURCES = STRONG_SOURCES | {"validated_retsam"}
NEG_SOURCES = {"retsam_negative", "grade_rule_override", "fgadr_lesion_only_sft_v3", "grade_rule", "cleaning_rule"}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def rid(r: dict[str, Any]) -> str:
    return r.get("record_id") or f"{r.get('dataset')}::{r.get('split')}::{Path(r.get('image_path', 'unknown')).stem}"


def image_path(r: dict[str, Any]) -> str:
    path = r["image_path"]
    return path[len("data/") :] if path.startswith("data/") else path


def hbucket(text: str) -> int:
    return int(hashlib.sha1(text.encode("utf-8")).hexdigest()[:8], 16) % 100


def answer(obs: str, ev: str, concl: str, payload: dict[str, Any]) -> str:
    return (
        f"【观察】{obs}\n\n【证据】{ev}\n\n【结论】{concl}\n\n【JSON】\n"
        + json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    )


def make_item(r: dict[str, Any], lesion: str, present: bool, source: str, eval_name: str) -> dict[str, Any]:
    count = "unknown"
    area = "unknown"
    d = r.get("lesions", {}).get(lesion) or {}
    if present:
        count = d.get("count", "unknown")
        area = d.get("area", "unknown")
        obs = f"围绕{LESION_CN[lesion]}的典型外观进行观察：{LESION_CUE[lesion]}。本题只训练该单一病灶概念。"
        ev = f"{lesion} present=true; count={count}; area={area}"
        concl = f"支持{LESION_CN[lesion]}阳性；本题不输出 DR 分级，也不合并其他病灶结论。"
    else:
        count = "unknown"
        area = "unknown"
        obs = f"围绕{LESION_CN[lesion]}的典型外观进行观察：{LESION_CUE[lesion]}。本题只判断该单一病灶是否存在。"
        ev = f"{lesion} present=false; count=unknown; area=unknown"
        concl = f"未见可靠{LESION_CN[lesion]}阳性证据；本题不输出 DR 分级，也不合并其他病灶结论。"

    return {
        "messages": [
            {
                "role": "system",
                "content": f"你是眼底病灶识别助手。本题只判断{LESION_CN[lesion]}是否存在；不得输出 DR grade，也不得评价其他病灶。",
            },
            {"role": "user", "content": f"<image>\n请只判断图中是否可见{LESION_CUE[lesion]}。"},
            {
                "role": "assistant",
                "content": answer(
                    obs,
                    ev,
                    concl,
                    {
                        "task": f"L3_{lesion}_single",
                        "lesion": lesion,
                        "present": present,
                        "count": count,
                        "area": area,
                    },
                ),
            },
        ],
        "images": [image_path(r)],
        "meta": {
            "record_id": rid(r),
            "task": f"L3_{lesion}_single",
            "lesion": lesion,
            "present": present,
            "source": source,
            "eval_set": eval_name,
        },
    }


def eligible(r: dict[str, Any], lesion: str) -> tuple[bool | None, str | None]:
    d = r.get("lesions", {}).get(lesion)
    if not isinstance(d, dict):
        return None, None
    present = d.get("present")
    source = d.get("source")
    if present is True:
        if lesion == "MA" and source not in STRONG_SOURCES:
            return None, None
        if lesion != "MA" and source not in POS_SOURCES:
            return None, None
        return True, source
    if present is False:
        if source in NEG_SOURCES:
            return False, source
    return None, None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/fundus_validated/validated_clean.jsonl")
    ap.add_argument("--smoke", default="data/annotation/fundus_stage1_smoke_sft.jsonl")
    ap.add_argument("--output", default="data/annotation/fundus_l3_presence_holdout80_sft.jsonl")
    ap.add_argument("--per-class", type=int, default=10, help="Positive and negative samples per lesion.")
    ap.add_argument("--seed", type=int, default=20260430)
    ap.add_argument("--eval-name", default="fundus_l3_presence_holdout80")
    args = ap.parse_args()

    records = read_jsonl(Path(args.input))
    smoke_records: set[str] = set()
    smoke_path = Path(args.smoke)
    if smoke_path.exists():
        for row in read_jsonl(smoke_path):
            record_id = row.get("meta", {}).get("record_id")
            if record_id:
                smoke_records.add(record_id)

    rng = random.Random(args.seed)
    rows: list[dict[str, Any]] = []
    stats: dict[str, Any] = {"eval_name": args.eval_name, "per_class": args.per_class, "lesions": {}}
    for lesion in ["MA", "HE", "EX", "SE"]:
        pos: list[tuple[dict[str, Any], str]] = []
        neg: list[tuple[dict[str, Any], str]] = []
        for r in records:
            if rid(r) in smoke_records:
                continue
            state, source = eligible(r, lesion)
            if state is True and source:
                pos.append((r, source))
            elif state is False and source:
                neg.append((r, source))
        rng.shuffle(pos)
        rng.shuffle(neg)
        pos_take = pos[: args.per_class]
        neg_take = neg[: args.per_class]
        stats["lesions"][lesion] = {
            "available_positive": len(pos),
            "available_negative": len(neg),
            "selected_positive": len(pos_take),
            "selected_negative": len(neg_take),
        }
        for r, source in pos_take:
            rows.append(make_item(r, lesion, True, source, args.eval_name))
        for r, source in neg_take:
            rows.append(make_item(r, lesion, False, source, args.eval_name))

    rng.shuffle(rows)
    write_jsonl(Path(args.output), rows)
    stats["n"] = len(rows)
    stats["output"] = args.output
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
