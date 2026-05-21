#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

# Allow running this file directly: add repo root to sys.path.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.retsam_pseudo.common import CropMetaRow, normalize_rel_to_data, read_jsonl, write_jsonl


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Wrapper runner for RetSAM inference with robust resume-by-output scan.")
    p.add_argument("--data-root", default="data")
    p.add_argument("--crop-meta", required=True, help="Path to crop_meta.jsonl (relative to data-root unless absolute).")
    p.add_argument("--dataset", required=True, help="Dataset name used for outputs/retsam_<dataset>.")
    p.add_argument("--retsam-root", required=True, help="Path to cloned RetSAM repo.")
    p.add_argument("--checkpoint", required=True, help="Path to RetSAM checkpoint file.")
    p.add_argument("--output-dir", default="", help="Override outputs base dir (default outputs/retsam_<dataset>).")
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    # keep flags explicit (no confusing store_true default=True)
    p.add_argument("--analysis-only", action="store_true", help="Pass --analysis_only to RetSAM.")
    p.add_argument("--enable-analysis", action="store_true", help="Pass --enable_analysis to RetSAM.")
    p.add_argument("--classify-diseases", action="store_true", help="Pass --classify_diseases to RetSAM.")
    p.add_argument("--disease-types", default="dr")
    p.add_argument("--output-channels", default="(2,3,2,4,5)")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument(
        "--include-grades",
        default="1,2,3,4",
        help="Comma-separated grades to include. Use 'all' to include every row (e.g. for strong-labeled datasets).",
    )
    p.add_argument("--errors-jsonl", default="", help="Where to write errors (default: <output_dir>/errors.jsonl)")
    p.add_argument("--tmp-parent", default="", help="Optional parent dir for temporary per-image input folders.")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def _resolve(path_or_rel: str, data_root: Path) -> Path:
    p = Path(path_or_rel)
    return p if p.is_absolute() else (data_root / p)


def _expected_done_marker(out_base: Path, image_id: str) -> Path:
    return out_base / image_id / "quantitative_analysis.json"


def _is_done_marker_valid(path: Path) -> bool:
    """
    A valid done marker must exist and be non-empty.
    (Empty JSON files can occur if inference was interrupted.)
    """
    try:
        return path.is_file() and path.stat().st_size > 0
    except Exception:
        return False


def _run_one(
    *,
    retsam_root: Path,
    checkpoint: Path,
    input_dir: Path,
    out_base: Path,
    device: str,
    output_channels: str,
    enable_analysis: bool,
    classify_diseases: bool,
    disease_types: str,
    analysis_only: bool,
    dry_run: bool,
) -> tuple[int, list[str], str]:
    """
    Return (exit_code, cmd, stdout_tail).

    We invoke RetSAM's python entry (`inference.py`) directly.
    (The repo's `scripts/inference.sh` is an example script with hard-coded paths.)
    """
    inference_py = retsam_root / "inference.py"
    if not inference_py.is_file():
        raise FileNotFoundError(f"RetSAM inference entry not found: {inference_py}")

    cmd = [
        "python3",
        str(inference_py),
        "--input_dir",
        str(input_dir),
        "--output_dir",
        str(out_base),
        "--model_path",
        str(checkpoint),
        "--multitask",
        "--output_channels",
        str(output_channels),
        "--has_coordinate_head",
        "--num_coordinates",
        "2",
        "--device",
        str(device),
    ]
    if enable_analysis:
        cmd.append("--enable_analysis")
    if classify_diseases:
        cmd.extend(["--classify_diseases", "--disease_types", str(disease_types)])
    if analysis_only:
        cmd.append("--analysis_only")

    if dry_run:
        return 0, cmd, ""

    p = subprocess.Popen(cmd, cwd=str(retsam_root), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    out, _ = p.communicate()
    tail = (out or "")[-4000:]
    return int(p.returncode or 0), cmd, tail


def main() -> int:
    args = parse_args()
    data_root = Path(args.data_root)
    crop_meta = _resolve(args.crop_meta, data_root)
    retsam_root = Path(args.retsam_root)
    checkpoint = Path(args.checkpoint)

    out_base = Path(args.output_dir) if args.output_dir else (Path("outputs") / f"retsam_{args.dataset}")
    out_base = out_base if out_base.is_absolute() else (_REPO_ROOT / out_base)
    out_base.mkdir(parents=True, exist_ok=True)

    errors_path = Path(args.errors_jsonl) if args.errors_jsonl else (out_base / "errors.jsonl")
    if not errors_path.is_absolute():
        errors_path = _REPO_ROOT / errors_path

    rows = [CropMetaRow.from_obj(x) for x in read_jsonl(crop_meta)]
    if str(args.include_grades).strip().lower() != "all":
        wanted = set()
        for part in str(args.include_grades).split(","):
            part = part.strip()
            if not part:
                continue
            try:
                wanted.add(int(part))
            except Exception:
                continue
        if wanted:
            rows = [r for r in rows if int(r.grade) in wanted]
    if args.limit and args.limit > 0:
        rows = rows[: args.limit]

    tmp_parent = Path(args.tmp_parent) if args.tmp_parent else None
    if tmp_parent is not None and not tmp_parent.is_absolute():
        tmp_parent = _REPO_ROOT / tmp_parent
    if tmp_parent is not None:
        tmp_parent.mkdir(parents=True, exist_ok=True)

    errs: list[dict] = []
    processed = 0
    skipped = 0
    failed = 0

    for r in rows:
        done_marker = _expected_done_marker(out_base, r.image_id)
        if _is_done_marker_valid(done_marker):
            skipped += 1
            continue

        cropped_abs = _resolve(r.cropped_path, data_root)
        if not cropped_abs.is_file():
            failed += 1
            errs.append({"image_id": r.image_id, "cropped_path": r.cropped_path, "error": "cropped_missing"})
            continue

        # Per-image input dir to guarantee resume & isolation.
        if tmp_parent is not None:
            one_tmp = tmp_parent / f"{args.dataset}_{r.image_id}"
            if one_tmp.exists():
                shutil.rmtree(one_tmp)
            one_tmp.mkdir(parents=True, exist_ok=True)
        else:
            one_tmp = Path(tempfile.mkdtemp(prefix=f"retsam_{args.dataset}_{r.image_id}_"))

        try:
            dst = one_tmp / f"{r.image_id}{cropped_abs.suffix.lower()}"
            shutil.copy2(cropped_abs, dst)

            rc, cmd, out_tail = _run_one(
                retsam_root=retsam_root,
                checkpoint=checkpoint,
                input_dir=one_tmp,
                out_base=out_base,
                device=args.device,
                output_channels=args.output_channels,
                enable_analysis=bool(args.enable_analysis),
                classify_diseases=bool(args.classify_diseases),
                disease_types=args.disease_types,
                analysis_only=bool(args.analysis_only),
                dry_run=bool(args.dry_run),
            )
            if rc != 0:
                failed += 1
                errs.append(
                    {
                        "image_id": r.image_id,
                        "cropped_path": r.cropped_path,
                        "exit_code": rc,
                        "cmd": cmd,
                        "stdout_tail": out_tail,
                    }
                )
            else:
                # Best-effort check: output marker should exist; if not, treat as failure.
                if not args.dry_run and not done_marker.is_file():
                    failed += 1
                    errs.append(
                        {
                            "image_id": r.image_id,
                            "cropped_path": r.cropped_path,
                            "error": "missing_quantitative_analysis_after_success",
                            "cmd": cmd,
                        }
                    )
                else:
                    processed += 1
        except Exception as e:
            failed += 1
            errs.append({"image_id": r.image_id, "cropped_path": r.cropped_path, "error": str(e)})
        finally:
            if tmp_parent is None:
                shutil.rmtree(one_tmp, ignore_errors=True)

        if (processed + skipped + failed) % 25 == 0:
            print(json.dumps({"processed": processed, "skipped": skipped, "failed": failed}, ensure_ascii=False), flush=True)

    if errs:
        write_jsonl(errors_path, errs, append=True)

    print(
        json.dumps(
            {
                "dataset": args.dataset,
                "out_base": str(out_base),
                "crop_meta": str(crop_meta),
                "processed": processed,
                "skipped": skipped,
                "failed": failed,
                "errors_jsonl": normalize_rel_to_data(errors_path, _REPO_ROOT),
                "dry_run": bool(args.dry_run),
            },
            ensure_ascii=False,
        )
    )
    return 0 if failed == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())

