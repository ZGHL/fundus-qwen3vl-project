from __future__ import annotations

import csv
import json
from pathlib import Path


def load_aptos_grades(csv_path: Path) -> dict[str, int]:
    """
    Expected Kaggle APTOS 2019 schema: id_code,diagnosis
    Returns map: image_id (stem) -> grade int [0..4]
    """
    out: dict[str, int] = {}
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not row:
                continue
            image_id = (row.get("id_code") or "").strip()
            diag = (row.get("diagnosis") or "").strip()
            if not image_id:
                continue
            try:
                out[image_id] = int(diag)
            except Exception:
                continue
    return out


def load_aptos_grades_from_instructions(jsonl_path: Path) -> dict[str, int]:
    """
    Fallback when Kaggle train.csv is unavailable.

    Parse LLaMA-Factory-style aptos grade instruction jsonl, where:
      - images: ["processed_images/aptos/<id>.png"]
      - output: "GRADE: <0-4>"

    Returns map: image_id (stem) -> grade
    """
    out: dict[str, int] = {}
    for line in jsonl_path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s:
            continue
        try:
            obj = json.loads(s)
        except Exception:
            continue
        imgs = obj.get("images") or []
        if not imgs:
            continue
        image_id = Path(str(imgs[0])).stem
        out_text = str(obj.get("output") or "")
        # Accept "GRADE: 2" etc.
        if "GRADE" not in out_text.upper():
            continue
        try:
            g = int(out_text.split(":")[-1].strip())
        except Exception:
            continue
        out[image_id] = g
    return out


def load_aptos_grades_from_annotation_dir(annotation_dir: Path) -> dict[str, int]:
    """
    Merge grades from all APTOS instruction jsonl files under data/annotation/.
    This is useful when `processed_images/aptos/` contains multiple splits.
    """
    out: dict[str, int] = {}
    if not annotation_dir.is_dir():
        return out
    for p in sorted(annotation_dir.glob("aptos*_instructions*.jsonl")):
        try:
            out.update(load_aptos_grades_from_instructions(p))
        except Exception:
            continue
    return out


def load_space_separated_list(path: Path) -> dict[str, int]:
    """
    Parse lines like: "<filename> <grade>" (e.g. DDR Grading train.txt/valid.txt/test.txt).
    Returns map: image_id (stem) -> grade int.
    """
    out: dict[str, int] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s:
            continue
        parts = s.split()
        if len(parts) < 2:
            continue
        name, grade = parts[0], parts[1]
        image_id = Path(name).stem
        try:
            out[image_id] = int(float(grade))
        except Exception:
            continue
    return out


def load_ddr_grading_from_txts(train_txt: Path, valid_txt: Path, test_txt: Path) -> dict[str, int]:
    out: dict[str, int] = {}
    for p in (train_txt, valid_txt, test_txt):
        if p.is_file():
            out.update(load_space_separated_list(p))
    return out


def load_ddr_grading_grades(csv_path: Path) -> dict[str, int]:
    """
    Default DDR grading label schema assumption: image,grade
    Returns map: image_id (stem, without extension) -> grade int [0..4]
    """
    out: dict[str, int] = {}
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not row:
                continue
            name = (row.get("image") or row.get("img") or row.get("image_id") or "").strip()
            grade = (row.get("grade") or row.get("label") or "").strip()
            if not name:
                continue
            image_id = Path(name).stem
            try:
                out[image_id] = int(grade)
            except Exception:
                continue
    return out

