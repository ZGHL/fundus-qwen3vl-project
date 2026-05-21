#!/usr/bin/env python3
"""Build FunBench L4c (DR grading) eval set in v3 prompt format (Scheme B).

Maps FunBench's image paths → local paths (DDR + IDRiD available locally;
DeepDRiD + Retinal-Lesions not present in this project).

Output is a v3-format SFT JSONL. The model is prompted with v3's 6-section
CoT prompt (NOT FunBench's prompt — that's the point of Scheme B). The label
is a minimal v3 CoT containing only the GT grade, since lesion labels are
unavailable in FunBench.

The evaluation scorer (separate script) extracts v3's predicted dr_grade
from the generated output, maps grade → ICDR text, finds matching option
letter in the entry's options list (which is shuffled per-entry), and
computes FunBench-style accuracy / F1.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

FUNBENCH_L4C = Path("data/funbench/L4-disease_diagnosis/L4c-dr_grading.json")
OUT_PATH = Path("data/annotation/fundus_funbench_l4c_eval_sft.jsonl")
META_PATH = Path("data/annotation/fundus_funbench_l4c_eval_meta.json")

DDR_LOCAL_GLOBS = [
    "data/cropped/ddr_grading/grade0",
    "data/cropped/ddr_grading/grade1_4",
]
IDRID_LOCAL_GLOB = "data/idrid/images"

ICDR_TEXT = {
    0: "No any diabetic retinopathy",
    1: "Mild nonproliferative diabetic retinopathy",
    2: "Moderate nonproliferative diabetic retinopathy",
    3: "Severe nonproliferative diabetic retinopathy",
    4: "Proliferative diabetic retinopathy",
}
TEXT_TO_GRADE = {v: k for k, v in ICDR_TEXT.items()}

SYSTEM_PROMPT = (
    "你是眼底 DR 分级助手。必须按 MA/HE/EX/SE/IRMA/NV 顺序逐项核查每个病灶（present_strong/"
    "present_weak/absent/unknown/template），再聚合为 NPDR_burden 和 proliferative/boundary"
    " 证据，最后按 Step1..Step6 有序规则输出 DR Grade 0-4 和固定 JSON。NV 是 Grade4/PDR 的"
    "直接证据；IRMA 仅作为 Grade3 边界；不能把 HE/EX/SE 编造成 PDR 证据。"
)
USER_PROMPT = (
    "请按【逐项核查】→【证据强度归类】→【病灶负担】→【分级路径】→【结论】→【JSON】的顺序输出。"
)


def index_local_ddr() -> dict[str, str]:
    """Map DDR image_id (e.g. '20170413102628830') → local path under data/."""
    out: dict[str, str] = {}
    for d in DDR_LOCAL_GLOBS:
        for f in glob.glob(f"{d}/*.png"):
            stem = os.path.basename(f).split(".")[0]
            rel = os.path.relpath(f, "data")
            out[stem] = rel
    return out


def index_local_idrid() -> dict[str, str]:
    out: dict[str, str] = {}
    for f in glob.glob(f"{IDRID_LOCAL_GLOB}/test/*.jpg") + glob.glob(f"{IDRID_LOCAL_GLOB}/train/*.jpg"):
        stem = os.path.basename(f).split(".")[0]
        rel = os.path.relpath(f, "data")
        out[stem] = rel
    return out


def fb_path_to_local(fb_path: str, ddr_idx: dict, idrid_idx: dict) -> str | None:
    fn = os.path.basename(fb_path).split(".")[0]
    if fb_path.startswith("DDR/"):
        return ddr_idx.get(fn)
    if fb_path.startswith("IDRiD/"):
        return idrid_idx.get(fn)
    return None  # DeepDRiD / Retinal-Lesions not local


def make_v3_label(grade: int) -> str:
    """Minimal v3-shape label carrying only the gt grade (lesion fields=unknown)."""
    payload = {
        "task": "L4_unified_lesion_cot_v3",
        "dr_grade": grade,
        "referable_dr": grade >= 2,
        "MA": "unknown", "HE": "unknown", "EX": "unknown", "SE": "unknown",
        "IRMA": "unknown", "NV": "unknown",
        "evidence_strong": [], "evidence_weak": [],
        "burden": "unknown", "proliferative_evidence": "unknown",
        "boundary_evidence": "unknown",
        "selected_step": "external_funbench_l4c",
        "decision_rule": "external_funbench_l4c_grade_only_label",
        "evidence_limited": True,
    }
    return (
        "【逐项核查】按 MA/HE/EX/SE/IRMA/NV 顺序逐项核查：\n"
        "MA: unknown\nHE: unknown\nEX: unknown\nSE: unknown\nIRMA: unknown\nNV: unknown\n\n"
        "【证据强度归类】unknown_or_template=['MA','HE','EX','SE','IRMA','NV']\n\n"
        "【病灶负担】NPDR_burden=unknown\n\n"
        "【分级路径】FunBench L4c external eval\n\n"
        f"【结论】DR Grade {grade}, evidence_tier=external_label_only。\n\n"
        "【JSON】\n" + json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    )


def build(per_grade_cap: int, seed: int) -> None:
    rng = random.Random(seed)
    fb = json.load(FUNBENCH_L4C.open())["data"]
    ddr_idx = index_local_ddr()
    idrid_idx = index_local_idrid()
    print(f"local DDR pool: {len(ddr_idx)}, local IDRiD pool: {len(idrid_idx)}")

    rows_by_grade: dict[int, list[dict[str, Any]]] = defaultdict(list)
    skipped_no_image = 0
    skipped_unknown_gt = 0

    for fb_path, entry in fb.items():
        local_rel = fb_path_to_local(fb_path, ddr_idx, idrid_idx)
        if local_rel is None:
            skipped_no_image += 1
            continue
        gt_text = entry["raw_data"]["gt"]
        if gt_text not in TEXT_TO_GRADE:
            skipped_unknown_gt += 1
            continue
        gt_grade = TEXT_TO_GRADE[gt_text]
        rows_by_grade[gt_grade].append({
            "fb_path": fb_path,
            "local_rel": local_rel,
            "gt_grade": gt_grade,
            "gt_letter": entry["gt"],
            "options": entry["raw_data"]["options"],  # shuffled per entry
            "question": entry["raw_data"]["question"],
        })

    print(f"available by grade: {dict((g, len(v)) for g, v in sorted(rows_by_grade.items()))}")
    print(f"skipped: no_image={skipped_no_image}, unknown_gt={skipped_unknown_gt}")

    selected = []
    for g in range(5):
        pool = rows_by_grade.get(g, [])
        rng.shuffle(pool)
        take = min(per_grade_cap, len(pool))
        selected.extend(pool[:take])
        print(f"  G{g}: take {take}/{len(pool)}")
    rng.shuffle(selected)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    META_PATH.parent.mkdir(parents=True, exist_ok=True)

    sft_rows = []
    meta_rows = []
    for i, entry in enumerate(selected):
        sft_rows.append({
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": "<image>\n" + USER_PROMPT},
                {"role": "assistant", "content": make_v3_label(entry["gt_grade"])},
            ],
            "images": [entry["local_rel"]],
            "meta": {
                "record_id": f"funbench_l4c::{i:05d}",
                "task": "FunBench_L4c_DRgrading",
                "split": "external_funbench",
                "dr_grade": entry["gt_grade"],
                "source_file": "FunBench_L4c-dr_grading.json",
                "evidence_limited": True,
                "decision_rule": "external_funbench_l4c",
                "selected_step": "funbench_l4c",
                "burden": "unknown",
                "proliferative_evidence": "unknown",
                "boundary_evidence": "unknown",
            },
        })
        meta_rows.append({
            "idx": i,
            "fb_path": entry["fb_path"],
            "local_rel": entry["local_rel"],
            "gt_grade": entry["gt_grade"],
            "gt_letter": entry["gt_letter"],
            "options": entry["options"],
            "question": entry["question"],
            "dataset": entry["fb_path"].split("/")[0],
        })

    with OUT_PATH.open("w", encoding="utf-8") as f:
        for r in sft_rows:
            f.write(json.dumps(r, ensure_ascii=False, separators=(",", ":")) + "\n")
    with META_PATH.open("w", encoding="utf-8") as f:
        json.dump(meta_rows, f, ensure_ascii=False, indent=2)

    print(f"\nwrote {len(sft_rows)} samples to {OUT_PATH}")
    print(f"wrote meta to {META_PATH}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-grade-cap", type=int, default=60,
                    help="Max samples per grade (default 60 → 300 total stratified)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    build(args.per_grade_cap, args.seed)


if __name__ == "__main__":
    main()
