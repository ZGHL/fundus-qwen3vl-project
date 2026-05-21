from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path

import sys

# Allow running this file directly: add repo root to sys.path.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.retsam_pseudo.common import CropMetaRow, ensure_parent, write_jsonl  # noqa: E402
from scripts.retsam_pseudo.grades import load_ddr_grading_from_txts  # noqa: E402


@dataclass(frozen=True)
class Candidate:
    image_id: str
    cropped_path: str  # relative to data-root (posix)
    grade: int


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build DDR crop_meta.jsonl by sampling N per grade from cropped PNGs.")
    p.add_argument("--data-root", default="data")
    p.add_argument("--cropped-base", default="cropped/ddr_grading/grade1_4", help="Relative to data-root unless absolute")
    p.add_argument("--ddr-split-dir", default="DDR-dataset/DR_grading", help="Where train.txt/valid.txt/test.txt live (relative to data-root)")
    p.add_argument("--per-grade", type=int, default=200, help="Sample size per grade (1-4). Ignored if --all is set.")
    p.add_argument("--all", action="store_true", help="Use all available grade 1-4 images (no sampling).")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-meta", default="cropped/ddr_grading/crop_meta.jsonl", help="Relative to data-root unless absolute")
    p.add_argument("--backup-old", action="store_true", help="If out-meta exists, back it up with .bak suffix")
    return p.parse_args()


def _resolve(data_root: Path, p: str) -> Path:
    pp = Path(p)
    return pp if pp.is_absolute() else (data_root / pp)


def main() -> int:
    args = parse_args()
    data_root = Path(args.data_root)
    cropped_base = _resolve(data_root, args.cropped_base)
    split_dir = _resolve(data_root, args.ddr_split_dir)
    out_meta = _resolve(data_root, args.out_meta)

    train_txt = split_dir / "train.txt"
    valid_txt = split_dir / "valid.txt"
    test_txt = split_dir / "test.txt"
    grade_map = load_ddr_grading_from_txts(train_txt, valid_txt, test_txt)
    if not grade_map:
        raise SystemExit(f"no grades loaded from: {train_txt}, {valid_txt}, {test_txt}")

    # collect candidates from existing cropped PNGs (no need to read original images)
    cands_by_grade: dict[int, list[Candidate]] = {1: [], 2: [], 3: [], 4: []}
    for p in sorted(cropped_base.glob("*.png")):
        image_id = p.stem
        g = grade_map.get(image_id)
        if g not in (1, 2, 3, 4):
            continue
        rel = p.relative_to(data_root).as_posix()
        cands_by_grade[int(g)].append(Candidate(image_id=image_id, cropped_path=rel, grade=int(g)))

    rnd = random.Random(int(args.seed))
    picked: list[Candidate] = []
    shortages: dict[int, int] = {}
    if bool(args.all):
        for g in (1, 2, 3, 4):
            items = list(cands_by_grade[g])
            picked.extend(items)
    else:
        need = int(args.per_grade)
        for g in (1, 2, 3, 4):
            items = list(cands_by_grade[g])
            rnd.shuffle(items)
            if len(items) < need:
                shortages[g] = len(items)
                picked.extend(items)
            else:
                picked.extend(items[:need])

    if args.backup_old and out_meta.exists():
        bak = out_meta.with_suffix(out_meta.suffix + ".bak")
        ensure_parent(bak)
        bak.write_text(out_meta.read_text(encoding="utf-8"), encoding="utf-8")

    # CropMetaRow requires bbox/src_path; these are not used by inference, so we set placeholders.
    rows: list[dict] = []
    for c in picked:
        rows.append(
            CropMetaRow(
                image_id=c.image_id,
                src_path="DDR-dataset/DR_grading/[unknown]",  # placeholder
                crop_box_xyxy=[0, 0, 0, 0],  # placeholder
                cropped_path=c.cropped_path,
                grade=int(c.grade),
            ).__dict__
        )

    write_jsonl(out_meta, rows, append=False)
    out = {
        "out_meta": out_meta.as_posix(),
        "per_grade_requested": (None if bool(args.all) else int(args.per_grade)),
        "all": bool(args.all),
        "picked_total": len(picked),
        "picked_by_grade": {str(g): sum(1 for c in picked if c.grade == g) for g in (1, 2, 3, 4)},
        "shortages": {str(k): v for k, v in shortages.items()},
        "cropped_base": cropped_base.as_posix(),
        "split_dir": split_dir.as_posix(),
    }
    print(json.dumps(out, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

