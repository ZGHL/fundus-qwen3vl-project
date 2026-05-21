#!/usr/bin/env python3
"""Extract per-lesion quadrant_distribution from RetSAM outputs and produce
a supplementary index keyed by image_id.

For each validated_clean record with a RetSAM source, we look up the original
RetSAM JSON and extract `lesions.lesion_dr.categories.<cat>.quadrant_distribution`
for hemorrhage (HE), exudate (EX), cotton_wool_spot (SE).

Output: data/fundus_validated/quadrant_index.jsonl

Each line:
{
  "image_id": "...",
  "dataset": "aptos|ddr_grading|idrid|fgadr|ddr_seg",
  "quadrants": {
    "HE": {"ST": N, "SN": N, "IT": N, "IN": N, "total": N},
    "EX": {...},
    "SE": {...},
  }
}

Coverage target: ~9,875 records (APTOS + DDR + IDRiD + FGADR + DDR-seg via RetSAM).
"""
from __future__ import annotations

import glob
import json
import os
from collections import Counter
from pathlib import Path

OUTPUTS = Path("/home/aim_lab/LLaMA-Factory/outputs")
OUT = Path("/home/aim_lab/LLaMA-Factory/data/fundus_validated/quadrant_index.jsonl")

RETSAM_DIRS = [
    ("aptos",       OUTPUTS / "retsam_aptos"),
    ("ddr_grading", OUTPUTS / "retsam_ddr_grading"),
    ("idrid",       OUTPUTS / "retsam_idrid"),
    ("fgadr_seg",   OUTPUTS / "retsam_fgadr_seg"),
    ("ddr_seg",     OUTPUTS / "retsam_ddr_seg"),
]

# RetSAM cat name → our lesion key
CAT_MAP = {
    "hemorrhage": "HE",
    "exudate": "EX",
    "cotton_wool_spot": "SE",
}

# RetSAM quadrant key → short
Q_MAP = {
    "superior_temporal": "ST",
    "superior_nasal":    "SN",
    "inferior_temporal": "IT",
    "inferior_nasal":    "IN",
}


def extract_record(d: dict) -> dict | None:
    """Extract quadrant info for a single RetSAM JSON."""
    cats = (d.get("measurements", {})
              .get("lesions", {})
              .get("lesion_dr", {})
              .get("categories", {}))
    if not cats:
        return None
    out = {}
    for cat_name, cat in cats.items():
        if cat_name not in CAT_MAP:
            continue
        q = cat.get("quadrant_distribution")
        if not q:
            continue
        short = CAT_MAP[cat_name]
        out[short] = {Q_MAP[k]: v for k, v in q.items() if k in Q_MAP}
        out[short]["total"] = sum(out[short].values())
    return out if out else None


def derive_image_id(file_path: Path, dataset: str) -> str:
    """The RetSAM file path is /<dir>/<image_id_or_stem>/quantitative_analysis.json
    image_id matches our validated_clean.jsonl image_id."""
    parent = file_path.parent.name
    # Strip extension if present
    if parent.endswith(".png") or parent.endswith(".jpg"):
        parent = parent.rsplit(".", 1)[0]
    return parent


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    seen_ids = set()
    written = 0
    coverage_per_dataset = Counter()
    coverage_per_lesion = Counter()

    with OUT.open("w") as fout:
        for dataset, base_dir in RETSAM_DIRS:
            if not base_dir.exists():
                print(f"  ⚠️  missing {base_dir}")
                continue
            files = sorted(glob.glob(f"{base_dir}/*/quantitative_analysis.json"))
            print(f"\n=== {dataset}: {len(files)} files ===")
            for f in files:
                fp = Path(f)
                try:
                    d = json.load(open(f))
                except Exception:
                    continue
                quads = extract_record(d)
                if not quads:
                    continue
                iid = derive_image_id(fp, dataset)
                key = (dataset, iid)
                if key in seen_ids:
                    continue
                seen_ids.add(key)
                row = {
                    "image_id": iid,
                    "dataset": dataset,
                    "quadrants": quads,
                }
                fout.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
                written += 1
                coverage_per_dataset[dataset] += 1
                for k in quads:
                    coverage_per_lesion[k] += 1

    print(f"\n=== summary ===")
    print(f"written: {written}")
    print(f"by dataset: {dict(coverage_per_dataset)}")
    print(f"by lesion: {dict(coverage_per_lesion)}")
    print(f"output: {OUT}")


if __name__ == "__main__":
    main()
