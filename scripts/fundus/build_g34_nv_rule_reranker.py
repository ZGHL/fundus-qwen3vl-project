#!/usr/bin/env python3
"""Build a clean G3/G4 evidence-bound reranker dataset.

The goal is not general 0-4 DR grading.  This small dataset isolates one rule:
visible NV is direct proliferative DR evidence and should push the decision to
Grade 4, while IRMA or heavy HE/EX/SE without NV stays at the G3/G4 boundary and
should not by itself become PDR.
"""

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
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def stable_hash(text: str) -> str:
    import hashlib

    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def stable_shuffle(rows: list[dict[str, Any]], salt: str) -> list[dict[str, Any]]:
    return sorted(
        rows,
        key=lambda row: stable_hash(f"{salt}::{row.get('meta', {}).get('record_id', '')}::{row.get('images', [''])[0]}"),
    )


def take_cycle(rows: list[dict[str, Any]], n: int) -> list[dict[str, Any]]:
    if n <= 0 or not rows:
        return []
    out: list[dict[str, Any]] = []
    while len(out) < n:
        out.extend(rows[: min(len(rows), n - len(out))])
    return out


def extract_obj(row: dict[str, Any]) -> dict[str, Any]:
    text = row["messages"][-1]["content"]
    return json.loads(text.split("【JSON】", 1)[-1].strip())


def state(value: Any) -> str:
    text = str(value).strip().lower()
    if text in {"present", "true", "1", "yes", "positive"}:
        return "present"
    if text in {"absent", "false", "0", "no", "negative"}:
        return "absent"
    return "unknown"


def answer(observation: str, evidence: str, decision_rule: str, conclusion: str, obj: dict[str, Any]) -> str:
    return (
        f"【观察】{observation}\n\n"
        f"【证据】{evidence}\n\n"
        f"【判别规则】{decision_rule}\n\n"
        f"【结论】{conclusion}\n\n"
        "【JSON】\n"
        + json.dumps(obj, ensure_ascii=False, separators=(",", ":"))
    )


def convert(row: dict[str, Any], component: str, split: str) -> dict[str, Any]:
    src = json.loads(json.dumps(row, ensure_ascii=False))
    src_obj = extract_obj(src)
    grade = int(src_obj["dr_grade"])
    nv = state(src_obj.get("NV"))
    irma = state(src_obj.get("IRMA"))
    evidence = list(src_obj.get("evidence") or [])

    if nv == "present" and "NV" not in evidence:
        evidence = ["NV"] + evidence
    if irma == "present" and "IRMA" not in evidence:
        evidence = ["IRMA"] + evidence

    if grade == 4:
        decision_rule = "NV present 是直接增殖性 DR 证据；只要可见 NV，应优先判为 DR Grade 4，而不是仅按 HE/EX/SE 负担降为 Grade 3。"
        conclusion = "支持 DR Grade 4；决定性证据是 NV，而不是单纯出血或渗出负担。"
    else:
        if irma == "present":
            decision_rule = "IRMA present 提示严重缺血/重度 NPDR 边界；但 NV absent 时，IRMA 不能单独等同于 PDR。"
        else:
            decision_rule = "HE/EX/SE 或 MA 即使较重，若 NV absent 且无直接增殖性证据，本题不应直接升为 Grade 4。"
        conclusion = "支持 DR Grade 3；当前缺少 NV 这种直接增殖性证据。"

    obj = {
        "task": "L4_g34_nv_rule_reranker",
        "dr_grade": grade,
        "referable_dr": True,
        "NV": nv,
        "IRMA": irma,
        "evidence": evidence,
        "decision": "PDR_G4" if grade == 4 else "severe_NPDR_G3",
        "component": component,
    }
    return {
        "messages": [
            {
                "role": "system",
                "content": (
                    "你是眼底 G3/G4 判别助手。本题只在 DR Grade 3 和 Grade 4 之间选择。"
                    "必须先核查 NV 和 IRMA：NV present 是 PDR/G4 的直接证据；"
                    "IRMA present 但 NV absent 时属于严重 NPDR/G3 边界，不能单独当作 NV。"
                ),
            },
            {
                "role": "user",
                "content": "<image>\n请只判断该眼底更支持 DR Grade 3 还是 Grade 4，并明确说明 NV/IRMA 对判别的作用。",
            },
            {
                "role": "assistant",
                "content": answer(
                    "先寻找跨越视网膜表面的新生血管/纤维血管增生；再看是否只有 IRMA、出血、渗出或软性渗出。",
                    f"visible_lesions={','.join(evidence) if evidence else 'none'}; NV={nv}; IRMA={irma}; target_grade={grade}",
                    decision_rule,
                    conclusion,
                    obj,
                ),
            },
        ],
        "images": src.get("images", []),
        "meta": {
            **src.get("meta", {}),
            "task": "L4_g34_nv_rule_reranker",
            "split": split,
            "dr_grade": grade,
            "mix_component": component,
            "NV": nv,
            "IRMA": irma,
            "source_task": src.get("meta", {}).get("task"),
        },
    }


def clean_grade4_nv(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        obj = extract_obj(row)
        if int(obj.get("dr_grade", -1)) == 4 and state(obj.get("NV")) == "present":
            out.append(row)
    return out


def clean_grade3_no_nv(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        obj = extract_obj(row)
        if int(obj.get("dr_grade", -1)) == 3 and state(obj.get("NV")) == "absent":
            out.append(row)
    return out


def stage2_g34_holdout(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        obj = extract_obj(row)
        grade = obj.get("dr_grade")
        if grade in {3, 4}:
            item = json.loads(json.dumps(row, ensure_ascii=False))
            src_obj = extract_obj(item)
            src_obj["task"] = "L4_g34_nv_rule_reranker"
            src_obj["NV"] = src_obj.get("NV", "unknown")
            src_obj["IRMA"] = src_obj.get("IRMA", "unknown")
            item["messages"][0]["content"] = (
                "你是眼底 G3/G4 判别助手。本题只在 DR Grade 3 和 Grade 4 之间选择。"
                "若没有可靠 NV/IRMA 证据，必须说明这是标签监督分级，不能编造增殖性证据。"
            )
            item["messages"][1]["content"] = "<image>\n请只判断该眼底更支持 DR Grade 3 还是 Grade 4。"
            item["messages"][-1]["content"] = answer(
                "该样本来自原始全局 L4 holdout，NV/IRMA 多数为 unknown，因此只用于观察 reranker 在真实标签分布上的迁移。",
                f"visible_lesions={','.join(src_obj.get('evidence') or []) or 'unknown'}; NV={src_obj['NV']}; IRMA={src_obj['IRMA']}; target_grade={grade}",
                "unknown 的 NV/IRMA 不能被编造成阳性；此样本按原始监督标签输出 G3/G4。",
                f"支持 DR Grade {grade}。",
                src_obj,
            )
            meta = dict(item.get("meta", {}))
            meta["task"] = "L4_g34_nv_rule_reranker"
            meta["split"] = "holdout"
            meta["dr_grade"] = grade
            meta["mix_component"] = "stage2_l4_g34_holdout"
            item["meta"] = meta
            out.append(item)
    return out


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    c = Counter()
    states = Counter()
    comp = Counter()
    missing = 0
    for row in rows:
        obj = extract_obj(row)
        c[str(obj.get("dr_grade"))] += 1
        states[(obj.get("dr_grade"), obj.get("NV"), obj.get("IRMA"))] += 1
        comp[row.get("meta", {}).get("mix_component", "unknown")] += 1
        for image in row.get("images", []):
            if not Path("data", image).exists():
                missing += 1
    return {
        "n": len(rows),
        "grades": dict(c),
        "components": dict(comp),
        "grade_nv_irma": {str(k): v for k, v in sorted(states.items(), key=lambda x: str(x[0]))},
        "missing_images": missing,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-total", type=int, default=1200)
    args = parser.parse_args()

    train_src = stable_shuffle(read_jsonl(TRAIN_SRC), "g34_train_src")
    holdout_src = stable_shuffle(read_jsonl(HOLDOUT_SRC), "g34_holdout_src")

    g4_train = stable_shuffle(clean_grade4_nv(train_src), "g34_g4_nv_train")
    g3_train = stable_shuffle(clean_grade3_no_nv(train_src), "g34_g3_no_nv_train")
    half = args.train_total // 2
    train_rows = [convert(row, "nv_present_g4", "train") for row in take_cycle(g4_train, half)]
    train_rows += [convert(row, "nv_absent_g3_boundary", "train") for row in take_cycle(g3_train, args.train_total - len(train_rows))]
    train_rows = stable_shuffle(train_rows, "g34_train_final")

    holdout_rows = [convert(row, "nv_present_g4", "holdout") for row in clean_grade4_nv(holdout_src)]
    holdout_rows += [convert(row, "nv_absent_g3_boundary", "holdout") for row in clean_grade3_no_nv(holdout_src)]
    holdout_rows = stable_shuffle(holdout_rows, "g34_evidence_holdout_final")

    stage2_holdout = stable_shuffle(stage2_g34_holdout(read_jsonl(STAGE2_HOLDOUT)), "g34_stage2_holdout_final")

    outputs = {
        BASE / "fundus_g34_nv_rule_reranker_train_sft.jsonl": train_rows,
        BASE / "fundus_g34_nv_rule_reranker_evidence_holdout_sft.jsonl": holdout_rows,
        BASE / "fundus_g34_nv_rule_reranker_stage2_holdout_sft.jsonl": stage2_holdout,
    }
    for path, rows in outputs.items():
        write_jsonl(path, rows)

    stats = {
        str(path): summarize(rows)
        for path, rows in outputs.items()
    }
    stats["source_clean_counts"] = {
        "train_g4_nv_present": len(g4_train),
        "train_g3_nv_absent": len(g3_train),
        "holdout_g4_nv_present": len(clean_grade4_nv(holdout_src)),
        "holdout_g3_nv_absent": len(clean_grade3_no_nv(holdout_src)),
    }
    stats_path = BASE / "fundus_g34_nv_rule_reranker_stats.json"
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
