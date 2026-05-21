#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

import cv2
import numpy as np

# Allow running this file directly: add repo root to sys.path.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.stage1_easy.preprocess import transform_coordinate_xy
from scripts.stage1_easy.rule_engine import estimate_od_fovea, format_assistant_output, generate_cot
from scripts.stage1_easy.progress import default_progress_path, mark_done, mark_running, update_progress


SYSTEM_PROMPT = """你是一名经验丰富的视网膜专科医生，正在分析一张来自2型糖尿病患者的眼底彩色照片（45°视角，以黄斑为中心）。请按照临床阅片顺序（MA→HE→EX→SE→NV）依次分析图中的视网膜病灶。

各类病灶形态参考（通用知识，非针对本图标注）：
- 微动脉瘤（MA）：红色至暗红色规则圆点，边界清晰锐利，直径通常小于125μm；区别于出血的不规则边缘
- 出血（HE）：暗红色至黑红色斑块，边缘不规则，大小不一；面积明显大于微动脉瘤
- 硬性渗出（EX）：黄白色蜡样沉积物，边界清晰锐利；不遮挡其下血管，区别于棉绒斑的模糊边界
- 软性渗出/棉绒斑（SE）：灰白色棉絮状斑块，边界模糊；可遮挡局部血管走行，区别于硬性渗出的清晰边界
- 新生血管（NV）：视网膜表面迂曲紊乱的细小血管网；需区分NVD（视盘旁1DD内）和NVE（其他位置）"""

USER_PROMPT = "请分析这张眼底图像中所有可见的DR相关病灶，按规定格式分别输出每类病灶的描述与判断结果。"


def _load_preprocess_meta(meta_jsonl: Path) -> dict[str, dict]:
    """
    Returns dict keyed by destination image path (string as stored), value=meta dict.
    """
    out: dict[str, dict] = {}
    if not meta_jsonl.exists():
        return out
    for line in meta_jsonl.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        out[str(obj["dst_rel"])] = obj
    return out


def _read_csv_xy(csv_path: Path) -> dict[str, tuple[int, int]]:
    out: dict[str, tuple[int, int]] = {}
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = row["Image No"].strip()
            x = int(float(row["X- Coordinate"]))
            y = int(float(row["Y - Coordinate"]))
            out[key] = (x, y)
    return out


def _idrid_mask_path(seg_root: Path, lesion: str, image_id_3: str) -> Path | None:
    """
    image_id_3: e.g. IDRiD_001
    masks use 2-digit: IDRiD_01_MA.tif
    """
    n = int(image_id_3.split("_")[1])
    two = f"{n:02d}"
    p = seg_root / lesion / f"IDRiD_{two}_{lesion}.tif"
    return p if p.exists() else None


def _fgadr_mask_path(seg_root: Path, lesion_dir: str, image_name: str) -> Path | None:
    p = seg_root / lesion_dir / image_name
    return p if p.exists() else None


def build_sharegpt_sample(image_rel: str, analysis_text: str, output_json: dict) -> dict:
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"<image>\n{USER_PROMPT}"},
            {"role": "assistant", "content": format_assistant_output(analysis_text, output_json)},
        ],
        "images": [image_rel],
    }


def _load_fgadr_grades(csv_path: Path) -> dict[str, int]:
    out: dict[str, int] = {}
    with csv_path.open("r", encoding="utf-8") as f:
        for line in f.read().splitlines():
            if not line.strip():
                continue
            if "," not in line:
                continue
            name, grade = line.split(",", 1)
            name = name.strip()
            grade = grade.strip()
            try:
                out[name] = int(grade)
            except ValueError:
                continue
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build Stage1 Easy ShareGPT datasets for LLaMA-Factory.")
    p.add_argument("--data-root", default="data")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--fgadr-per-grade", type=int, default=300)
    p.add_argument("--idrid-train-max-id", type=int, default=54, help="Use IDRiD_001..IDRiD_{N:03d} for training")
    p.add_argument("--idrid-test-min-id", type=int, default=55, help="Use IDRiD_{min:03d}..IDRiD_{max:03d} for testing")
    p.add_argument("--idrid-test-max-id", type=int, default=81)
    p.add_argument("--out-train", default="data/annotation/idrid_fgadr_stage1_easy_train.json")
    p.add_argument("--out-idrid-test", default="data/annotation/idrid_stage1_easy_test.json")
    p.add_argument("--validate", action="store_true", help="Print a few samples and run minimal checks.")
    p.add_argument("--validate-n", type=int, default=5)
    return p.parse_args()


def _load_rgb(path: Path) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(str(path))
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    data_root = Path(args.data_root)
    out_train = Path(args.out_train)
    out_test = Path(args.out_idrid_test)
    out_train.parent.mkdir(parents=True, exist_ok=True)
    out_test.parent.mkdir(parents=True, exist_ok=True)

    processed_root = data_root / "processed_images" / "stage1_easy"
    meta_map = _load_preprocess_meta(processed_root / "preprocess_meta.jsonl")
    print("stage1_easy_build=start", flush=True)
    prog = default_progress_path()
    mark_running(
        prog,
        "build_dataset",
        fgadr_per_grade=args.fgadr_per_grade,
        idrid_train_max_id=args.idrid_train_max_id,
        idrid_test_min_id=args.idrid_test_min_id,
        idrid_test_max_id=args.idrid_test_max_id,
        out_train=str(out_train),
        out_test=str(out_test),
    )

    # ---------------- IDRiD ----------------
    idrid_loc = data_root / "idrid" / "localization"
    fovea_train = _read_csv_xy(idrid_loc / "train_fovea.csv")
    od_train = _read_csv_xy(idrid_loc / "train_od.csv")
    fovea_test = _read_csv_xy(idrid_loc / "test_fovea.csv")
    od_test = _read_csv_xy(idrid_loc / "test_od.csv")

    idrid_seg_train = data_root / "idrid" / "segmentation" / "train"
    idrid_seg_test = data_root / "idrid" / "segmentation" / "test"

    train_samples: list[dict] = []
    test_samples: list[dict] = []

    for split in ("train", "test"):
        img_dir = data_root / "idrid" / "images" / split
        proc_dir = processed_root / "idrid" / split
        for img_path in sorted(img_dir.glob("IDRiD_*.jpg")):
            image_id = img_path.stem  # IDRiD_001
            try:
                n_id = int(image_id.split("_")[1])
            except Exception:
                continue
            if split == "train" and n_id > args.idrid_train_max_id:
                continue
            if split == "test" and not (args.idrid_test_min_id <= n_id <= args.idrid_test_max_id):
                continue
            proc_path = proc_dir / f"{image_id}.jpg"
            if not proc_path.exists():
                continue

            meta = meta_map.get(str(proc_path))
            if meta is None:
                continue
            crop = tuple(int(x) for x in meta["crop_box_xyxy"])
            target_size = int(meta.get("target_size", 1024))

            if split == "train":
                fov0 = fovea_train.get(image_id)
                od0 = od_train.get(image_id)
                seg_root = idrid_seg_train
            else:
                fov0 = fovea_test.get(image_id)
                od0 = od_test.get(image_id)
                seg_root = idrid_seg_test

            if fov0 is None or od0 is None:
                continue

            fovea_center = transform_coordinate_xy(fov0, crop, target_size=target_size)
            od_center = transform_coordinate_xy(od0, crop, target_size=target_size)
            od_radius = 70  # not used for IDRiD Easy

            mask_paths = {
                "MA": _idrid_mask_path(seg_root, "MA", image_id),
                "HE": _idrid_mask_path(seg_root, "HE", image_id),
                "EX": _idrid_mask_path(seg_root, "EX", image_id),
                "SE": _idrid_mask_path(seg_root, "SE", image_id),
            }
            analysis, out_json = generate_cot(
                image_id,
                mask_paths,
                fovea_center,
                od_center,
                od_radius,
                has_nv=False,
                crop_box_xyxy=crop,
            )
            image_rel = str(proc_path.relative_to(data_root))
            sample = build_sharegpt_sample(image_rel, analysis, out_json)
            (train_samples if split == "train" else test_samples).append(sample)

    # ---------------- FGADR ----------------
    fgadr_root = data_root / "FGADR" / "Seg-set"
    grades = _load_fgadr_grades(fgadr_root / "DR_Seg_Grading_Label.csv")
    g2 = [k for k, v in grades.items() if v == 2]
    g3 = [k for k, v in grades.items() if v == 3]
    random.shuffle(g2)
    random.shuffle(g3)
    g2 = g2[: args.fgadr_per_grade]
    g3 = g3[: args.fgadr_per_grade]
    fgadr_selected = g2 + g3
    print(f"fgadr_selected={len(fgadr_selected)} (grade2={len(g2)} grade3={len(g3)})", flush=True)
    update_progress(prog, "build_dataset", {"fgadr_selected": len(fgadr_selected)})

    fgadr_proc_dir = processed_root / "fgadr"
    for idx, name in enumerate(fgadr_selected, start=1):
        # name is like 0000_1.png
        stem = Path(name).stem
        proc_path = fgadr_proc_dir / f"{stem}.jpg"
        if not proc_path.exists():
            continue

        img_rgb = _load_rgb(proc_path)
        meta = meta_map.get(str(proc_path))
        crop = None
        if meta is not None:
            try:
                crop = tuple(int(x) for x in meta["crop_box_xyxy"])
            except Exception:
                crop = None
        od_center, od_r, fovea_center = estimate_od_fovea(img_rgb)

        mask_paths = {
            "MA": _fgadr_mask_path(fgadr_root, "Microaneurysms_Masks", name),
            "HE": _fgadr_mask_path(fgadr_root, "Hemohedge_Masks", name),
            "EX": _fgadr_mask_path(fgadr_root, "HardExudate_Masks", name),
            "SE": _fgadr_mask_path(fgadr_root, "SoftExudate_Masks", name),
            "NV": _fgadr_mask_path(fgadr_root, "Neovascularization_Masks", name),
        }
        image_id = stem
        analysis, out_json = generate_cot(
            image_id,
            mask_paths,
            fovea_center,
            od_center,
            od_r,
            has_nv=True,
            crop_box_xyxy=crop,
        )
        image_rel = str(proc_path.relative_to(data_root))
        train_samples.append(build_sharegpt_sample(image_rel, analysis, out_json))
        if idx % 10 == 0:
            print(f"fgadr_progress={idx}/{len(fgadr_selected)} train_samples={len(train_samples)}", flush=True)

    out_train.write_text(json.dumps(train_samples, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    out_test.write_text(json.dumps(test_samples, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")

    print(f"train_samples={len(train_samples)} saved={out_train}")
    print(f"idrid_test_samples={len(test_samples)} saved={out_test}")
    print(f"processed_root={processed_root}")
    mark_done(
        prog,
        "build_dataset",
        train_samples=len(train_samples),
        idrid_test_samples=len(test_samples),
        out=str(out_train),
    )

    if args.validate:
        n = min(args.validate_n, len(train_samples))
        print(f"\n--- first {n} train samples ---")
        for i in range(n):
            s = train_samples[i]
            assert "messages" in s and "images" in s
            assert len(s["messages"]) == 3
            assert "<image>" in s["messages"][1]["content"]
            content = s["messages"][2]["content"]
            assert "## Analysis" in content and "## Output" in content
            # JSON parse check
            out_json_str = content.split("## Output", 1)[1].strip()
            json.loads(out_json_str)
            print(json.dumps({"images": s["images"][0]}, ensure_ascii=False))
        print("validate_ok=true")


if __name__ == "__main__":
    main()

