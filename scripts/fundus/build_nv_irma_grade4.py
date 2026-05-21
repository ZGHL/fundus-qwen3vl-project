#!/usr/bin/env python3
"""Build NV / IRMA / Grade4-focused fundus SFT datasets.

This script reuses the raw FGADR lesion-only structured annotations and the
validated_clean grade mapping to build:

* L3_NV_single
* L3_IRMA_single
* L4_grade4_augmented

The goal is to expose NV explicitly, keep IRMA as a proliferative-adjacent cue,
and rebalance grade-4 supervision so it is not trained only from HE/EX/SE.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


BASE = Path("data/annotation")
RAW_FGADR = BASE / "fgadr_lesion_only_sft_v3_lf.json"
VALIDATED = Path("data/fundus_validated/validated_clean.jsonl")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_raw_fgadr(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        data = data.get("data") or data.get("instances") or data.get("examples") or []
    if not isinstance(data, list):
        raise TypeError(f"Unexpected FGADR raw structure in {path}")
    return data


def stable_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def split_key(image_path: str, holdout_pct: int) -> str:
    bucket = int(stable_hash(image_path)[:8], 16) % 100
    return "holdout" if bucket < holdout_pct else "train"


def extract_assistant_text(row: dict[str, Any]) -> str:
    msgs = row.get("messages") or []
    for msg in msgs:
        if isinstance(msg, dict) and msg.get("role") == "assistant":
            return msg.get("content", "")
    return ""


def parse_output(assistant_text: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    m = re.search(r"LESIONS:\s*([^\n]+)", assistant_text)
    if m:
        vals = [x.strip() for x in m.group(1).split(",") if x.strip() and x.strip() != "NONE"]
        out["lesions"] = vals
    m = re.search(r"NEOVASCULAR_SIGN:\s*(\w+)", assistant_text)
    if m:
        out["neovascular_sign"] = m.group(1)
    m = re.search(r"LOCATION:\s*(\w+)", assistant_text)
    if m:
        out["location"] = m.group(1)
    m = re.search(r"EXTENT:\s*(\w+)", assistant_text)
    if m:
        out["extent"] = m.group(1)
    m = re.search(r"SEVERITY_CUE:\s*(\w+)", assistant_text)
    if m:
        out["severity_cue"] = m.group(1)
    m = re.search(r"UNCERTAINTY:\s*(\w+)", assistant_text)
    if m:
        out["uncertainty"] = m.group(1)
    return out


def image_path(row: dict[str, Any]) -> str:
    imgs = row.get("images") or []
    if not imgs:
        return ""
    return str(imgs[0])


def load_grade_map() -> dict[str, int]:
    out: dict[str, int] = {}
    for row in read_jsonl(VALIDATED):
        p = str(row.get("image_path") or "").replace("data/", "")
        grade = row.get("grade")
        if isinstance(grade, int):
            out[p] = grade
    return out


def sft(system: str, user: str, assistant: str, image: str, meta: dict[str, Any]) -> dict[str, Any]:
    return {
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": "<image>\n" + user},
            {"role": "assistant", "content": assistant},
        ],
        "images": [image],
        "meta": meta,
    }


def answer(observation: str, evidence: str, conclusion: str, payload: dict[str, Any]) -> str:
    return (
        f"【观察】{observation}\n\n"
        f"【证据】{evidence}\n\n"
        f"【结论】{conclusion}\n\n"
        "【JSON】\n"
        + json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    )


def rank_nv_negative(grade: int | None) -> tuple[int, int]:
    grade_rank = 9 if grade is None else 4 - int(grade)
    # Prefer harder negatives first: grade 4 without NV, then 3, then 2, then 1/0.
    return (grade_rank, 0 if grade is not None else 1)


def rank_irma_negative(grade: int | None) -> tuple[int, int]:
    grade_rank = 9 if grade is None else 4 - int(grade)
    return (grade_rank, 0 if grade is not None else 1)


def build_single_task(
    rows: list[dict[str, Any]],
    grade_map: dict[str, int],
    lesion: str,
    positive_key: str,
    negative_key: str,
    system_prompt: str,
    user_prompt: str,
    pos_cue: str,
    neg_cue: str,
    task_name: str,
    holdout_pct: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    buckets = {"train": {"pos": [], "neg": []}, "holdout": {"pos": [], "neg": []}}
    for row in rows:
        path = image_path(row)
        if not path:
            continue
        split = split_key(path, holdout_pct)
        out = parse_output(extract_assistant_text(row))
        lesions = set(out.get("lesions") or [])
        nv_sign = out.get("neovascular_sign")
        pos = False
        if positive_key == "NV":
            pos = lesion in lesions and nv_sign == "PRESENT"
        elif positive_key == "IRMA":
            pos = lesion in lesions
        if pos:
            buckets[split]["pos"].append((path, row, out, grade_map.get(path)))
        else:
            buckets[split]["neg"].append((path, row, out, grade_map.get(path)))

    train_rows: list[dict[str, Any]] = []
    holdout_rows: list[dict[str, Any]] = []
    stats: dict[str, Any] = {"split": {}}
    for split in ["train", "holdout"]:
        pos_list = buckets[split]["pos"]
        neg_list = buckets[split]["neg"]
        # Hard negatives first, then deterministic sampling.
        if positive_key == "NV":
            neg_list = sorted(neg_list, key=lambda x: rank_nv_negative(x[3]))
        else:
            neg_list = sorted(neg_list, key=lambda x: rank_irma_negative(x[3]))
        n = min(len(pos_list), len(neg_list))
        pos_list = sorted(pos_list, key=lambda x: stable_hash(f"{task_name}::{x[0]}::pos"))
        neg_list = sorted(neg_list, key=lambda x: stable_hash(f"{task_name}::{x[0]}::neg"))
        selected = pos_list[:n] + neg_list[:n]

        out_rows = []
        for path, row, parsed, _grade in selected:
            present = path in {p for p, _, _, _ in pos_list[:n]}
            if positive_key == "NV":
                present = present and parsed.get("neovascular_sign") == "PRESENT"
            elif positive_key == "IRMA":
                present = lesion in set(parsed.get("lesions") or [])
            obs = f"围绕{pos_cue if present else neg_cue}进行核查。本题只判断单一病灶是否存在，不输出 DR 分级。"
            if positive_key == "NV":
                evidence = f"NV present={str(present).lower()}; IRMA={'PRESENT' if 'IRMA' in set(parsed.get('lesions') or []) else 'ABSENT'}; source=fgadr_lesion_only_sft_v3"
            else:
                evidence = f"IRMA present={str(present).lower()}; NV={'PRESENT' if 'NV' in set(parsed.get('lesions') or []) else 'ABSENT'}; source=fgadr_lesion_only_sft_v3"
            concl = f"支持{pos_cue if present else neg_cue}阳性；本题不输出 DR 分级，也不合并其他病灶结论。" if present else f"未见可靠{neg_cue}阳性证据；本题不输出 DR 分级，也不合并其他病灶结论。"
            payload = {
                "task": task_name,
                "lesion": lesion,
                "present": present,
                "count": "unknown",
                "area": "unknown",
                "source": "fgadr_lesion_only_sft_v3",
            }
            out_rows.append(
                sft(
                    system_prompt,
                    user_prompt,
                    answer(obs, evidence, concl, payload),
                    path,
                    {
                        "record_id": row.get("meta", {}).get("record_id") or path,
                        "task": task_name,
                        "split": split,
                        "source_file": "fgadr_lesion_only_sft_v3_lf.json",
                        "grade": _grade,
                    },
                )
            )

        if split == "train":
            train_rows = out_rows
        else:
            holdout_rows = out_rows
        stats["split"][split] = {
            "pos": len(pos_list[:n]),
            "neg": len(neg_list[:n]),
            "total": len(out_rows),
        }
    return train_rows, holdout_rows, stats


def build_grade4_task(
    rows: list[dict[str, Any]],
    grade_map: dict[str, int],
    holdout_pct: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    buckets = {"train": {"pos": [], "neg": []}, "holdout": {"pos": [], "neg": []}}
    for row in rows:
        path = image_path(row)
        if not path:
            continue
        grade = grade_map.get(path)
        if grade is None:
            continue
        split = split_key(path, holdout_pct)
        parsed = parse_output(extract_assistant_text(row))
        lesions = set(parsed.get("lesions") or [])
        buckets[split]["pos" if grade == 4 else "neg"].append((path, row, parsed, grade, lesions))

    train_rows: list[dict[str, Any]] = []
    holdout_rows: list[dict[str, Any]] = []
    stats: dict[str, Any] = {"split": {}}
    for split in ["train", "holdout"]:
        pos_list = buckets[split]["pos"]
        neg_list = buckets[split]["neg"]
        pos_list = sorted(
            pos_list,
            key=lambda x: (
                0 if x[2].get("neovascular_sign") == "PRESENT" or "NV" in x[4] else 1,
                0 if "IRMA" in x[4] else 1,
                stable_hash(f"grade4::{x[0]}::pos"),
            ),
        )
        neg_list = sorted(
            neg_list,
            key=lambda x: (
                rank_nv_negative(x[3]),
                0 if x[2].get("neovascular_sign") == "PRESENT" or "NV" in x[4] else 1,
                stable_hash(f"grade4::{x[0]}::neg"),
            ),
        )
        n = min(len(pos_list), len(neg_list))
        selected = pos_list[:n] + neg_list[:n]
        out_rows = []
        for path, row, parsed, grade, lesions in selected:
            visible = sorted(lesions, key=lambda k: (0 if k == "NV" else 1 if k == "IRMA" else 2, k))
            if not visible:
                visible = ["HE", "EX", "SE"] if grade == 4 else []
            evidence = ",".join(visible) if visible else "none"
            support_grade4 = grade == 4
            obs = "先核查是否存在新生血管和 IRMA，再结合出血/渗出负担判断是否支持 DR Grade 4。"
            if grade == 4:
                concl = f"监督分级为 DR Grade 4，主要依据为 {evidence if evidence != 'none' else 'HE,EX,SE'}。"
            else:
                concl = f"监督分级为 DR Grade {grade}，虽可见 {evidence if evidence != 'none' else '无明确增殖证据'}，但不足以支持 Grade 4。"
            payload = {
                "task": "L4_grade4_augmented",
                "dr_grade": grade,
                "evidence": visible,
                "support_grade4": support_grade4,
                "NV": "present" if "NV" in visible else "absent",
                "IRMA": "present" if "IRMA" in visible else "absent",
                "source": "validated_clean+fgadr_lesion_only_sft_v3",
            }
            out_rows.append(
                sft(
                    "你是眼底分级助手。先核查新生血管与 IRMA，再判断 DR 分级；不能把 HE/EX/SE 直接等同于 PDR。",
                    "请先判断是否存在增殖性证据，再给出 DR 分级。",
                    answer(
                        obs,
                        f"visible_lesions={evidence}; dr_grade={grade}; NV={'PRESENT' if 'NV' in visible else 'ABSENT'}; IRMA={'PRESENT' if 'IRMA' in visible else 'ABSENT'}",
                        concl,
                        payload,
                    ),
                    path,
                    {
                        "record_id": row.get("meta", {}).get("record_id") or path,
                        "task": "L4_grade4_augmented",
                        "split": split,
                        "source_file": "fgadr_lesion_only_sft_v3_lf.json",
                        "grade": grade,
                    },
                )
            )
        if split == "train":
            train_rows = out_rows
        else:
            holdout_rows = out_rows
        stats["split"][split] = {
            "pos": len(pos_list[:n]),
            "neg": len(neg_list[:n]),
            "total": len(out_rows),
            "grade4_with_nv": sum(1 for _, _, parsed, _, lesions in pos_list[:n] if parsed.get("neovascular_sign") == "PRESENT" or "NV" in lesions),
            "grade4_with_irma": sum(1 for _, _, _, _, lesions in pos_list[:n] if "IRMA" in lesions),
        }
    return train_rows, holdout_rows, stats


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    tasks = Counter()
    datasets = Counter()
    grades = Counter()
    lesions = Counter()
    for row in rows:
        meta = row.get("meta", {})
        task = meta.get("task", "unknown")
        tasks[task] += 1
        datasets["::".join(str(meta.get("record_id", "unknown")).split("::")[:2])] += 1
        assistant = extract_assistant_text(row)
        obj_match = re.search(r"\{.*\}", assistant.split("【JSON】")[-1], flags=re.S)
        if obj_match:
            try:
                obj = json.loads(obj_match.group(0))
                if "dr_grade" in obj:
                    grades[obj["dr_grade"]] += 1
                if "lesion" in obj:
                    lesions[obj["lesion"]] += 1
            except json.JSONDecodeError:
                pass
    return {"n": len(rows), "tasks": dict(tasks), "grades": dict(grades), "lesions": dict(lesions), "datasets": dict(datasets)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--holdout-pct", type=int, default=20)
    args = parser.parse_args()

    raw = load_raw_fgadr(RAW_FGADR)
    grade_map = load_grade_map()

    nv_train, nv_holdout, nv_stats = build_single_task(
        raw,
        grade_map,
        lesion="NV",
        positive_key="NV",
        negative_key="NV",
        system_prompt="你是眼底病灶识别助手。本题只判断新生血管是否存在；不得输出 DR grade，也不得把 HE/EX/SE 直接当作新生血管。",
        user_prompt="请只判断图中是否可见跨越视网膜表面的新生血管或纤维血管增生样病灶。",
        pos_cue="新生血管",
        neg_cue="新生血管",
        task_name="L3_NV_single",
        holdout_pct=args.holdout_pct,
    )
    irma_train, irma_holdout, irma_stats = build_single_task(
        raw,
        grade_map,
        lesion="IRMA",
        positive_key="IRMA",
        negative_key="IRMA",
        system_prompt="你是眼底病灶识别助手。本题只判断 IRMA 是否存在；不得输出 DR grade，也不得把 NV 直接当作 IRMA。",
        user_prompt="请只判断图中是否可见视网膜内微血管异常（IRMA）。",
        pos_cue="IRMA",
        neg_cue="IRMA",
        task_name="L3_IRMA_single",
        holdout_pct=args.holdout_pct,
    )
    pdr_train, pdr_holdout, pdr_stats = build_grade4_task(raw, grade_map, args.holdout_pct)

    outputs = {
        "fundus_l3_nv_single_train_sft.jsonl": nv_train,
        "fundus_l3_nv_single_holdout_sft.jsonl": nv_holdout,
        "fundus_l3_irma_single_train_sft.jsonl": irma_train,
        "fundus_l3_irma_single_holdout_sft.jsonl": irma_holdout,
        "fundus_l4_grade4_augmented_train_sft.jsonl": pdr_train,
        "fundus_l4_grade4_augmented_holdout_sft.jsonl": pdr_holdout,
    }

    stats: dict[str, Any] = {
        "nv_single": nv_stats,
        "irma_single": irma_stats,
        "grade4_augmented": pdr_stats,
    }

    out_stats: dict[str, Any] = {}
    for filename, rows in outputs.items():
        (BASE / filename).write_text("\n".join(json.dumps(r, ensure_ascii=False, separators=(",", ":")) for r in rows) + "\n", encoding="utf-8")
        out_stats[filename] = summarize(rows)
    stats["files"] = out_stats

    (BASE / "fundus_nv_irma_grade4_stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
