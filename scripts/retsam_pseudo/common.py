from __future__ import annotations

import json
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(time.time()))


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def atomic_write_json(path: Path, obj: Any, *, indent: int = 2) -> None:
    ensure_parent(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=indent), encoding="utf-8")
    os.replace(tmp, path)


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]], *, append: bool = False) -> None:
    ensure_parent(path)
    mode = "a" if append else "w"
    with path.open(mode, encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    out: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        out.append(json.loads(line))
    return out


def iter_images(root: Path) -> list[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    out: list[Path] = []
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.suffix.lower() in exts:
            out.append(p)
    return out


def normalize_rel_to_data(path: Path, data_root: Path) -> str:
    """Return a stable relative path string (posix)."""
    try:
        rel = path.relative_to(data_root)
    except Exception:
        rel = path
    return rel.as_posix()


def sample_n(items: list[Any], n: int, seed: int = 42) -> list[Any]:
    if n <= 0:
        return []
    if len(items) <= n:
        return list(items)
    rnd = random.Random(seed)
    idxs = list(range(len(items)))
    rnd.shuffle(idxs)
    return [items[i] for i in idxs[:n]]


@dataclass(frozen=True)
class CropMetaRow:
    image_id: str
    src_path: str
    crop_box_xyxy: list[int]
    cropped_path: str
    grade: int

    @staticmethod
    def required_keys() -> set[str]:
        return {"image_id", "src_path", "crop_box_xyxy", "cropped_path", "grade"}

    @staticmethod
    def from_obj(obj: dict[str, Any]) -> "CropMetaRow":
        missing = CropMetaRow.required_keys() - set(obj.keys())
        if missing:
            raise KeyError(f"crop_meta missing keys: {sorted(missing)}")
        crop = obj["crop_box_xyxy"]
        if not (isinstance(crop, list) and len(crop) == 4):
            raise ValueError("crop_box_xyxy must be list[int] of len=4")
        return CropMetaRow(
            image_id=str(obj["image_id"]),
            src_path=str(obj["src_path"]),
            crop_box_xyxy=[int(x) for x in crop],
            cropped_path=str(obj["cropped_path"]),
            grade=int(obj["grade"]),
        )


@dataclass(frozen=True)
class KeptIndexRow:
    image_id: str
    grade: int
    retsam_json_path: str
    he_valid: bool
    ex_valid: bool
    se_valid: bool
    od_source: str  # "retsam" | "hough_fallback"

    @staticmethod
    def required_keys() -> set[str]:
        return {"image_id", "grade", "retsam_json_path", "he_valid", "ex_valid", "se_valid", "od_source"}

    @staticmethod
    def from_obj(obj: dict[str, Any]) -> "KeptIndexRow":
        missing = KeptIndexRow.required_keys() - set(obj.keys())
        if missing:
            raise KeyError(f"kept_index missing keys: {sorted(missing)}")
        return KeptIndexRow(
            image_id=str(obj["image_id"]),
            grade=int(obj["grade"]),
            retsam_json_path=str(obj["retsam_json_path"]),
            he_valid=bool(obj["he_valid"]),
            ex_valid=bool(obj["ex_valid"]),
            se_valid=bool(obj["se_valid"]),
            od_source=str(obj["od_source"]),
        )

