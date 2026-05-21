#!/usr/bin/env python3
"""Build MESSIDOR-2 external holdout for L4 v2 evaluation.

Creates a v2-format SFT JSONL where:
- prompt = same system+user as fundus_l4_unified_lesion_cot_v2
- image = data/messidor-2/messidor-2/preprocess/<id>.png
- label = minimal v2 CoT with grade-only ground truth (lesions=unknown, JSON
  has correct dr_grade) — only the grade is used for evaluation; lesion fields
  can't be evaluated since MESSIDOR-2 doesn't have lesion labels.

Stratified 30 per grade by default (matches v2 internal holdout150 layout for
direct comparison). Deterministic seed.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path

CSV_PATH = Path("data/messidor_data.csv")
IMG_DIR_REL = "messidor-2/messidor-2/preprocess"
IMG_DIR_ABS = Path("data") / IMG_DIR_REL
OUT_DEFAULT = Path("data/annotation/fundus_messidor2_external_holdout_sft.jsonl")

SYSTEM_PROMPT = (
    "你是眼底 DR 分级助手。必须按 MA/HE/EX/SE/IRMA/NV 顺序逐项核查每个病灶（present_strong/"
    "present_weak/absent/unknown/template），再聚合为 NPDR_burden 和 proliferative/boundary"
    " 证据，最后按 Step1..Step6 有序规则输出 DR Grade 0-4 和固定 JSON。NV 是 Grade4/PDR 的"
    "直接证据；IRMA 仅作为 Grade3 边界；不能把 HE/EX/SE 编造成 PDR 证据。"
)
USER_PROMPT = (
    "请按【逐项核查】→【证据强度归类】→【病灶负担】→【分级路径】→【结论】→【JSON】的顺序输出。"
)


def make_minimal_label(grade: int) -> str:
    """A v2-shape label that contains the correct grade so metrics can extract it.

    All lesion fields set unknown because MESSIDOR-2 has no lesion ground truth.
    Decision rule and step are placeholders; only dr_grade is authoritative.
    """
    payload = {
        "task": "L4_unified_lesion_cot_v2",
        "dr_grade": grade,
        "referable_dr": grade >= 2,
        "MA": "unknown",
        "HE": "unknown",
        "EX": "unknown",
        "SE": "unknown",
        "IRMA": "unknown",
        "NV": "unknown",
        "evidence_strong": [],
        "evidence_weak": [],
        "burden": "unknown",
        "proliferative_evidence": "unknown",
        "boundary_evidence": "unknown",
        "selected_step": "external_messidor2",
        "decision_rule": "external_messidor2_grade_only_label",
        "evidence_limited": True,
    }
    return (
        "【逐项核查】按 MA/HE/EX/SE/IRMA/NV 顺序逐项核查：\n"
        "MA: unknown（外部数据集无病灶标签）\n"
        "HE: unknown（外部数据集无病灶标签）\n"
        "EX: unknown（外部数据集无病灶标签）\n"
        "SE: unknown（外部数据集无病灶标签）\n"
        "IRMA: unknown（外部数据集无病灶标签）\n"
        "NV: unknown（外部数据集无病灶标签）\n\n"
        "【证据强度归类】strong_present=none; weak_present=none; "
        "unknown_or_template=['MA','HE','EX','SE','IRMA','NV']; absent=none\n\n"
        "【病灶负担】NPDR_burden=unknown; proliferative_evidence=unknown; boundary_evidence=unknown\n\n"
        "【分级路径】外部 MESSIDOR-2 评估，仅有 grade label。\n\n"
        f"【结论】DR Grade {grade}，referable_dr={'true' if grade>=2 else 'false'}，"
        f"evidence_tier=external_label_only。\n\n"
        "【JSON】\n"
        + json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    )


def build(per_grade: int, seed: int, out_path: Path) -> None:
    rng = random.Random(seed)
    rows = list(csv.DictReader(CSV_PATH.open()))
    by_grade: dict[int, list[dict]] = defaultdict(list)
    for r in rows:
        if r.get("adjudicated_gradable") != "1":
            continue
        try:
            g = int(r["diagnosis"])
        except (TypeError, ValueError):
            continue
        if 0 <= g <= 4:
            by_grade[g].append(r)

    print("MESSIDOR-2 gradable distribution:")
    for g in range(5):
        print(f"  G{g}: {len(by_grade[g])} available")

    selected: list[dict] = []
    for g in range(5):
        pool = by_grade[g][:]
        rng.shuffle(pool)
        take = min(per_grade, len(pool))
        if take < per_grade:
            print(f"  ⚠️  G{g} only has {take} samples (requested {per_grade})")
        selected.extend([(g, r) for r in pool[:take]])

    rng.shuffle(selected)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    missing = 0
    with out_path.open("w", encoding="utf-8") as f:
        for grade, row in selected:
            img_rel = f"{IMG_DIR_REL}/{row['id_code']}"
            if not (IMG_DIR_ABS / row["id_code"]).exists():
                missing += 1
                continue
            sample = {
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": "<image>\n" + USER_PROMPT},
                    {"role": "assistant", "content": make_minimal_label(grade)},
                ],
                "images": [img_rel],
                "meta": {
                    "record_id": f"messidor2::{row['id_code']}",
                    "task": "L4_unified_lesion_cot_v2_messidor2_external",
                    "split": "external_messidor2",
                    "dr_grade": grade,
                    "source_file": "messidor_data.csv",
                    "evidence_limited": True,
                    "decision_rule": "external_messidor2_grade_only_label",
                    "adjudicated_dme": int(row.get("adjudicated_dme", 0) or 0),
                },
            }
            f.write(json.dumps(sample, ensure_ascii=False, separators=(",", ":")) + "\n")
            written += 1

    print(f"\n[done] wrote {written} samples to {out_path}")
    if missing:
        print(f"[warn] {missing} rows skipped due to missing image file")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--per-grade", type=int, default=30)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=Path, default=OUT_DEFAULT)
    args = p.parse_args()
    build(args.per_grade, args.seed, args.out)


if __name__ == "__main__":
    main()
