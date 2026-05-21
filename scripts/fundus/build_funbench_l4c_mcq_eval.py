#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

META = Path("data/annotation/fundus_funbench_l4c_eval_meta.json")
OUT = Path("data/annotation/fundus_funbench_l4c_mcq_eval_sft.jsonl")


SYSTEM = (
    "You are a retinal disease grading assistant. Answer the multiple-choice "
    "question using only one option letter: A, B, C, D, or E."
)


def main() -> None:
    meta = json.load(META.open(encoding="utf-8"))
    with OUT.open("w", encoding="utf-8") as f:
        for row in meta:
            options = row["options"]
            option_lines = "\n".join(f"{chr(65+i)}. {text}" for i, text in enumerate(options))
            user = (
                "<image>\n"
                f"Question: {row['question']}\n\n"
                f"Options:\n{option_lines}\n\n"
                "Answer with the option letter only."
            )
            item = {
                "messages": [
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": user},
                    {"role": "assistant", "content": row["gt_letter"]},
                ],
                "images": [row["local_rel"]],
                "meta": {
                    "record_id": f"funbench_l4c_mcq::{row['idx']:05d}",
                    "gt_letter": row["gt_letter"],
                    "gt_grade": row["gt_grade"],
                    "dataset": row["dataset"],
                },
            }
            f.write(json.dumps(item, ensure_ascii=False, separators=(",", ":")) + "\n")
    print(f"wrote {len(meta)} rows to {OUT}")


if __name__ == "__main__":
    main()
