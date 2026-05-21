#!/usr/bin/env python3
"""Build v2 G3/G4 reranker data with explicit NV-only and IRMA-boundary strata."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


BASE = Path("data/annotation")
TRAIN_SRC = BASE / "fundus_l4_grade4_augmented_train_sft.jsonl"
HOLDOUT_SRC = BASE / "fundus_l4_grade4_augmented_holdout_sft.jsonl"
STAGE2_HOLDOUT = BASE / "fundus_stage2_lite_l4_holdout120_sft.jsonl"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def stable_hash(text: str) -> str:
    import hashlib

    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def stable_shuffle(rows: list[dict[str, Any]], salt: str) -> list[dict[str, Any]]:
    return sorted(rows, key=lambda row: stable_hash(f"{salt}::{row.get('meta', {}).get('record_id', '')}::{row.get('images', [''])[0]}"))


def take_cycle(rows: list[dict[str, Any]], n: int) -> list[dict[str, Any]]:
    if n <= 0 or not rows:
        return []
    out: list[dict[str, Any]] = []
    while len(out) < n:
        out.extend(rows[: min(len(rows), n - len(out))])
    return out


def extract_obj(row: dict[str, Any]) -> dict[str, Any]:
    return json.loads(row["messages"][-1]["content"].split("【JSON】", 1)[-1].strip())


def state(value: Any) -> str:
    text = str(value).strip().lower()
    if text in {"present", "true", "1", "yes", "positive"}:
        return "present"
    if text in {"absent", "false", "0", "no", "negative"}:
        return "absent"
    return "unknown"


def bucket(rows: list[dict[str, Any]], grade: int, nv: str, irma: str) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        obj = extract_obj(row)
        if int(obj.get("dr_grade", -1)) == grade and state(obj.get("NV")) == nv and state(obj.get("IRMA")) == irma:
            out.append(row)
    return out


def convert(row: dict[str, Any], component: str, split: str) -> dict[str, Any]:
    src = json.loads(json.dumps(row, ensure_ascii=False))
    obj = extract_obj(src)
    grade = int(obj["dr_grade"])
    nv = state(obj.get("NV"))
    irma = state(obj.get("IRMA"))
    evidence = list(obj.get("evidence") or [])
    if nv == "present" and "NV" not in evidence:
        evidence = ["NV"] + evidence
    if irma == "present" and "IRMA" not in evidence:
        evidence = ["IRMA"] + evidence

    if grade == 4:
        rule = "NV_PRESENT_IMPLIES_G4"
        rationale = "NV present 是 PDR/Grade 4 的直接证据；即使 IRMA absent，也不能把明确 NV 降成 Grade 3。"
    else:
        rule = "NO_NV_NOT_G4"
        if irma == "present":
            rationale = "IRMA present 是严重 NPDR/G3 边界证据；NV absent 时，IRMA 不能单独等同于 PDR/G4。"
        else:
            rationale = "HE/EX/SE 负担较重但 NV absent 时，缺少直接 PDR 证据，应保留为 Grade 3。"

    content = (
        f"【观察】只做 G3/G4 判别，先检查 NV，再检查 IRMA。\n\n"
        f"【证据】NV={nv}; IRMA={irma}; visible_lesions={','.join(evidence) if evidence else 'none'}\n\n"
        f"【判别规则】{rationale}\n\n"
        f"【结论】DR Grade {grade}。\n\n"
        "【JSON】\n"
        + json.dumps(
            {
                "task": "L4_g34_nv_rule_reranker_v2",
                "dr_grade": grade,
                "NV": nv,
                "IRMA": irma,
                "rule": rule,
                "evidence": evidence,
                "component": component,
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )
    )
    return {
        "messages": [
            {
                "role": "system",
                "content": "你是眼底 G3/G4 判别助手。本题只输出 DR Grade 3 或 4。规则：NV present 直接支持 G4；IRMA present 但 NV absent 是 G3 边界，不能单独当作 G4。",
            },
            {"role": "user", "content": "<image>\n请只在 DR Grade 3 和 Grade 4 之间选择，并用 NV/IRMA 规则说明。"},
            {"role": "assistant", "content": content},
        ],
        "images": src.get("images", []),
        "meta": {
            **src.get("meta", {}),
            "task": "L4_g34_nv_rule_reranker_v2",
            "split": split,
            "dr_grade": grade,
            "NV": nv,
            "IRMA": irma,
            "mix_component": component,
            "source_task": src.get("meta", {}).get("task"),
        },
    }


def convert_stage2_holdout(row: dict[str, Any]) -> dict[str, Any] | None:
    src = json.loads(json.dumps(row, ensure_ascii=False))
    obj = extract_obj(src)
    grade = obj.get("dr_grade")
    if grade not in {3, 4}:
        return None
    evidence = list(obj.get("evidence") or [])
    content = (
        "【观察】原始全局 holdout 多数没有可靠 NV/IRMA 标注，本样本只评估 G3/G4 迁移。\n\n"
        f"【证据】NV=unknown; IRMA=unknown; visible_lesions={','.join(evidence) if evidence else 'unknown'}\n\n"
        "【判别规则】unknown 的 NV/IRMA 不能被编造成阳性；此处按原始监督标签输出。\n\n"
        f"【结论】DR Grade {grade}。\n\n"
        "【JSON】\n"
        + json.dumps(
            {
                "task": "L4_g34_nv_rule_reranker_v2",
                "dr_grade": grade,
                "NV": "unknown",
                "IRMA": "unknown",
                "rule": "LABEL_SUPERVISED_UNKNOWN_NV_IRMA",
                "evidence": evidence,
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )
    )
    src["messages"][0]["content"] = "你是眼底 G3/G4 判别助手。本题只输出 DR Grade 3 或 4。若 NV/IRMA unknown，不得编造成阳性。"
    src["messages"][1]["content"] = "<image>\n请只在 DR Grade 3 和 Grade 4 之间选择。"
    src["messages"][-1]["content"] = content
    meta = dict(src.get("meta", {}))
    meta["task"] = "L4_g34_nv_rule_reranker_v2"
    meta["split"] = "holdout"
    meta["dr_grade"] = grade
    meta["NV"] = "unknown"
    meta["IRMA"] = "unknown"
    meta["mix_component"] = "stage2_l4_g34_holdout"
    src["meta"] = meta
    return src


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grades = Counter()
    components = Counter()
    states = Counter()
    missing = 0
    for row in rows:
        obj = extract_obj(row)
        grades[str(obj.get("dr_grade"))] += 1
        components[row.get("meta", {}).get("mix_component", "unknown")] += 1
        states[(obj.get("dr_grade"), obj.get("NV"), obj.get("IRMA"))] += 1
        for image in row.get("images", []):
            if not Path("data", image).exists():
                missing += 1
    return {
        "n": len(rows),
        "grades": dict(grades),
        "components": dict(components),
        "grade_nv_irma": {str(k): v for k, v in sorted(states.items(), key=lambda x: str(x[0]))},
        "missing_images": missing,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--total", type=int, default=1600)
    args = parser.parse_args()

    train = stable_shuffle(read_jsonl(TRAIN_SRC), "g34_v2_train_src")
    holdout = stable_shuffle(read_jsonl(HOLDOUT_SRC), "g34_v2_holdout_src")

    train_buckets = {
        "nv_only_g4": stable_shuffle(bucket(train, 4, "present", "absent"), "nv_only_g4"),
        "nv_irma_g4": stable_shuffle(bucket(train, 4, "present", "present"), "nv_irma_g4"),
        "irma_only_g3": stable_shuffle(bucket(train, 3, "absent", "present"), "irma_only_g3"),
        "no_nv_no_irma_g3": stable_shuffle(bucket(train, 3, "absent", "absent"), "no_nv_no_irma_g3"),
    }
    budgets = {
        "nv_only_g4": round(args.total * 0.35),
        "nv_irma_g4": round(args.total * 0.15),
        "irma_only_g3": round(args.total * 0.30),
    }
    budgets["no_nv_no_irma_g3"] = args.total - sum(budgets.values())

    train_rows: list[dict[str, Any]] = []
    for name, n in budgets.items():
        train_rows.extend(convert(row, name, "train") for row in take_cycle(train_buckets[name], n))
    train_rows = stable_shuffle(train_rows, "g34_v2_train_final")

    holdout_rows: list[dict[str, Any]] = []
    for name in ["nv_only_g4", "nv_irma_g4", "irma_only_g3", "no_nv_no_irma_g3"]:
        grade, nv, irma = {
            "nv_only_g4": (4, "present", "absent"),
            "nv_irma_g4": (4, "present", "present"),
            "irma_only_g3": (3, "absent", "present"),
            "no_nv_no_irma_g3": (3, "absent", "absent"),
        }[name]
        holdout_rows.extend(convert(row, name, "holdout") for row in bucket(holdout, grade, nv, irma))
    holdout_rows = stable_shuffle(holdout_rows, "g34_v2_evidence_holdout_final")

    stage2_rows = [row for row in (convert_stage2_holdout(row) for row in read_jsonl(STAGE2_HOLDOUT)) if row is not None]
    stage2_rows = stable_shuffle(stage2_rows, "g34_v2_stage2_holdout_final")

    outputs = {
        BASE / "fundus_g34_nv_rule_reranker_v2_train_sft.jsonl": train_rows,
        BASE / "fundus_g34_nv_rule_reranker_v2_evidence_holdout_sft.jsonl": holdout_rows,
        BASE / "fundus_g34_nv_rule_reranker_v2_stage2_holdout_sft.jsonl": stage2_rows,
    }
    for path, rows in outputs.items():
        write_jsonl(path, rows)
    stats = {str(path): summarize(rows) for path, rows in outputs.items()}
    stats["source_counts"] = {name: len(rows) for name, rows in train_buckets.items()}
    stats["budgets"] = budgets
    stats_path = BASE / "fundus_g34_nv_rule_reranker_v2_stats.json"
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
