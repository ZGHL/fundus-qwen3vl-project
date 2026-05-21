#!/usr/bin/env python3
"""Build split-aware L2/L3/L4 fundus SFT datasets from validated_clean.jsonl."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


STRONG_SOURCES = {"strong_mask_stage1_easy", "fgadr_lesion_only_sft_v3"}
NEGATIVE_SOURCES = {"strong_mask_stage1_easy", "fgadr_lesion_only_sft_v3", "retsam_negative", "cleaning_rule"}
LESION_CN = {"MA": "微动脉瘤", "HE": "出血", "EX": "硬性渗出", "SE": "软性渗出"}
LESION_CUE = {
    "MA": "微小红色圆点样病灶",
    "HE": "暗红点片状或不规则斑块状病灶",
    "EX": "亮黄色、边界较清楚的沉积样病灶",
    "SE": "灰白色、棉絮样、边界较模糊的病灶",
}


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def hbucket(text: str) -> int:
    return int(hashlib.sha1(text.encode("utf-8")).hexdigest()[:8], 16) % 100


def image_key(r: dict[str, Any]) -> str:
    return r.get("image_path") or f"{r.get('dataset')}::{r.get('split')}::{r.get('source_image_id')}"


def rid(r: dict[str, Any]) -> str:
    return f"{r.get('dataset')}::{r.get('split')}::{Path(r.get('image_path', 'unknown')).stem}"


def split_of(r: dict[str, Any], val_pct: int) -> tuple[str, str, int]:
    bucket = hbucket(image_key(r))
    if r.get("dataset") == "idrid" and r.get("split") == "test":
        return "eval", "official_idrid_test", bucket
    if r.get("dataset") == "ddr_seg" and r.get("split") in {"valid", "test"}:
        return "eval", f"official_ddr_seg_{r.get('split')}", bucket
    if bucket < val_pct:
        return "val", f"hash_bucket_lt_{val_pct}", bucket
    return "train", "hash_train", bucket


def sft(system: str, user: str, assistant: str, image: str, meta: dict[str, Any]) -> dict[str, Any]:
    if image.startswith("data/"):
        image = image[len("data/") :]
    return {
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": "<image>\n" + user},
            {"role": "assistant", "content": assistant},
        ],
        "images": [image],
        "meta": meta,
    }


def answer(obs: str, ev: str, concl: str, payload: dict[str, Any]) -> str:
    return (
        f"【观察】{obs}\n\n【证据】{ev}\n\n【结论】{concl}\n\n【JSON】\n"
        + json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    )


def valid_bio(r: dict[str, Any], key: str) -> dict[str, Any] | None:
    v = r.get("biomarkers", {}).get(key)
    return v if isinstance(v, dict) and v.get("valid") is True else None


def lesion(r: dict[str, Any], key: str) -> dict[str, Any] | None:
    v = r.get("lesions", {}).get(key)
    return v if isinstance(v, dict) else None


def is_true_lesion(r: dict[str, Any], key: str) -> dict[str, Any] | None:
    v = lesion(r, key)
    if not v or v.get("present") is not True:
        return None
    src = v.get("source")
    if key == "MA":
        return v if src in STRONG_SOURCES else None
    return v if src in STRONG_SOURCES or src == "validated_retsam" else None


def is_false_lesion(r: dict[str, Any], key: str) -> dict[str, Any] | None:
    v = lesion(r, key)
    if not v or v.get("present") is not False:
        return None
    src = v.get("source")
    if src in NEGATIVE_SOURCES:
        return v
    if r.get("grade") == 0 and src in {"grade_rule", "grade_rule_override"}:
        item = dict(v)
        item["source"] = "grade0_rule_negative"
        return item
    return None


def l2_laterality(r: dict[str, Any], sp: str):
    eye = valid_bio(r, "eye_side")
    if not eye:
        return None
    val = eye.get("value")
    return sft(
        "你是眼底图像分析助手。本题只判断 laterality；先说明视盘与黄斑/中央凹的相对关系，再给出左眼或右眼。",
        "请判断这张眼底图来自左眼还是右眼。",
        answer(
            "laterality 主要依据视盘与黄斑/中央凹的相对关系。本题不使用病灶或 DR 分级信息。",
            f"eye_side={val}; eye_side.valid=true; source={eye.get('source')}",
            f"该图判断为{'右眼' if val == 'right' else '左眼' if val == 'left' else val}。",
            {"task": "L2_laterality", "eye_side": val, "source": eye.get("source")},
        ),
        r["image_path"],
        {"record_id": rid(r), "task": "L2_laterality", "split": sp},
    )


def l2_cdr(r: dict[str, Any], sp: str):
    cdr = valid_bio(r, "cdr")
    if not cdr:
        return None
    val = float(cdr.get("value"))
    bucket = "normal_or_mild" if val < 0.4 else "borderline" if val < 0.6 else "high"
    return sft(
        "你是眼底图像分析助手。本题只回答 CDR；先说明视盘和视杯如何识别，再根据垂直径比例分档。",
        "请观察视盘和视杯，估计杯盘比是否偏高。",
        answer(
            "视盘通常是较亮的橙黄色圆/椭圆结构；视杯位于视盘中央，颜色更浅。CDR 主要比较视杯垂直径与视盘垂直径的比例。",
            f"cdr={val:.4f}; cdr.valid=true; source={cdr.get('source')}",
            f"CDR 约 {val:.2f}，分档为 {bucket}。",
            {"task": "L2_cdr", "cdr": round(val, 4), "cdr_bucket": bucket, "source": cdr.get("source")},
        ),
        r["image_path"],
        {"record_id": rid(r), "task": "L2_cdr", "split": sp},
    )


def l2_vessel_abstain(r: dict[str, Any], sp: str):
    b = r.get("biomarkers", {})
    if b.get("vessel_qc_flag") is not False:
        return None
    av, tort = b.get("av_ratio", {}), b.get("tortuosity", {})
    if not isinstance(av, dict) or not isinstance(tort, dict):
        return None
    if av.get("valid") is not False and tort.get("valid") is not False:
        return None
    return sft(
        "你是眼底图像分析助手。本题只回答血管 A/V ratio 和迂曲度；血管 QC 不可靠时必须输出 unknown。",
        "请判断动静脉比例和血管迂曲度是否异常。",
        answer(
            "A/V ratio 需要可靠地区分动脉和静脉粗细，迂曲度需要可靠血管分割。当前血管质量控制未通过。",
            "vessel_qc_flag=false; av_ratio.valid=false; tortuosity.valid=false",
            "不能可靠判断 A/V ratio 或血管迂曲度，输出 unknown。",
            {"task": "L2_vessel_metrics", "av_ratio_bucket": "unknown", "tortuosity_bucket": "unknown", "reason": "vessel_qc_failed_or_missing"},
        ),
        r["image_path"],
        {"record_id": rid(r), "task": "L2_vessel_abstain", "split": sp},
    )


def l3_single(r: dict[str, Any], sp: str, key: str):
    d = is_true_lesion(r, key)
    present = True
    if not d:
        d = is_false_lesion(r, key)
        present = False
    if not d:
        return None
    if present:
        obs = f"围绕{LESION_CN[key]}的典型外观进行观察：{LESION_CUE[key]}。本题只训练该单一病灶概念。"
        ev = f"{key} present=true; count={d.get('count','unknown')}; area={d.get('area','unknown')}"
        concl = f"支持{LESION_CN[key]}阳性；本题不输出 DR 分级，也不合并其他病灶结论。"
        payload = {
            "task": f"L3_{key}_single",
            "lesion": key,
            "present": True,
            "count": d.get("count", "unknown"),
            "area": d.get("area", "unknown"),
        }
    else:
        obs = f"围绕{LESION_CN[key]}的典型外观进行观察：{LESION_CUE[key]}。本题只判断该单一病灶是否存在。"
        ev = f"{key} present=false; count=unknown; area=unknown"
        concl = f"未见可靠{LESION_CN[key]}阳性证据；本题不输出 DR 分级，也不合并其他病灶结论。"
        payload = {
            "task": f"L3_{key}_single",
            "lesion": key,
            "present": False,
            "count": "unknown",
            "area": "unknown",
        }
    return sft(
        f"你是眼底病灶识别助手。本题只判断{LESION_CN[key]}是否存在；不得输出 DR grade，也不得评价其他病灶。",
        f"请只判断图中是否可见{LESION_CUE[key]}。",
        answer(
            obs,
            ev,
            concl,
            payload,
        ),
        r["image_path"],
        {
            "record_id": rid(r),
            "task": f"L3_{key}_single",
            "split": sp,
            "lesion": key,
            "present": present,
            "source": d.get("source"),
        },
    )


def l3_se_abstain(r: dict[str, Any], sp: str):
    d = lesion(r, "SE")
    if not d or d.get("source") != "cleaning_rule" or d.get("raw_present") is not True:
        return None
    reason = d.get("suppressed_reason", "cleaning_rule")
    return sft(
        "你是眼底病灶识别助手。被 cleaning_rule 降级的病灶不能作为阳性证据。",
        "请判断是否可见灰白棉絮样、边界模糊的软性渗出样病灶。",
        answer(
            "灰白棉絮样、边界模糊的软性渗出证据不足。",
            f"SE raw_present=true; cleaned SE present=false; source=cleaning_rule; suppressed_reason={reason}",
            "不支持可靠 SE 阳性。",
            {"task": "L3_SE_abstain", "SE": False, "reason": reason, "source": "validated_clean"},
        ),
        r["image_path"],
        {"record_id": rid(r), "task": "L3_SE_abstain", "split": sp},
    )


def visible_lesions(r: dict[str, Any]) -> list[str]:
    out = []
    for k in ["MA", "HE", "EX", "SE"]:
        if is_true_lesion(r, k):
            out.append(k)
    for k in ["IRMA", "NV"]:
        d = lesion(r, k)
        if d and d.get("present") is True and d.get("source") in STRONG_SOURCES:
            out.append(k)
    return out


def l3_lesion_only(r: dict[str, Any], sp: str):
    lesions = visible_lesions(r)
    if not lesions:
        return None
    return sft(
        "你是眼底病灶识别助手。本题只从闭集 MA/HE/EX/SE/IRMA/NV 中选择可见病灶，不输出 DR grade。",
        "请从 MA、HE、EX、SE、IRMA、NV 中选择本图可见的糖尿病视网膜病变相关病灶。",
        answer(
            "按闭集病灶逐项核查可见证据；本题只做病灶汇总，不进行 DR 分级。",
            "visible_lesions=" + ",".join(lesions),
            "可见病灶为：" + "、".join(lesions) + "。",
            {"task": "L3_lesion_only", "lesions": lesions, "source": "validated_clean"},
        ),
        r["image_path"],
        {"record_id": rid(r), "task": "L3_lesion_only", "split": sp},
    )


def l3_burden(r: dict[str, Any], sp: str, key: str):
    d = is_true_lesion(r, key)
    if not d or (d.get("count_bucket") in {None, "unknown"} and d.get("area_bucket") in {None, "unknown"}):
        return None
    return sft(
        f"你是眼底病灶负担评估助手。本题只评估{LESION_CN[key]}的数量和面积负担，不输出 DR grade。",
        f"请评估图中{LESION_CN[key]}的数量负担和面积负担。",
        answer(
            f"只围绕{LESION_CN[key]}统计负担，不合并其他病灶。",
            f"{key} count={d.get('count')}; count_bucket={d.get('count_bucket')}; area={d.get('area')}; area_bucket={d.get('area_bucket')}",
            f"{LESION_CN[key]}数量分档为 {d.get('count_bucket')}，面积分档为 {d.get('area_bucket')}。",
            {"task": "L3_lesion_burden", "lesion": key, "count_bucket": d.get("count_bucket"), "area_bucket": d.get("area_bucket"), "source": d.get("source")},
        ),
        r["image_path"],
        {"record_id": rid(r), "task": f"L3_{key}_burden", "split": sp},
    )


def l4_evidence(r: dict[str, Any], sp: str):
    grade = r.get("grade")
    if not isinstance(grade, int) or grade < 0 or not r.get("usable_for", {}).get("L4"):
        return None
    image = r["image_path"]
    ev = visible_lesions(r)
    ma = lesion(r, "MA") or {}
    if grade == 0:
        task = "L4_grade0_no_reliable_dr"
        system = "你是眼底分级助手。Grade 0 样本不得编造 HE/EX/SE/MA 病灶。"
        user = "请核查是否有可靠糖尿病视网膜病变病灶证据，并给出分级结论。"
        ans = answer("清洗事实层未保留可靠 DR 病灶阳性证据。", "dr_grade=0; reliable_DR_lesions=false", "支持 DR Grade 0，即未见可靠 DR 病灶证据。", {"task": task, "dr_grade": 0, "evidence": [], "source": "label+validated_clean"})
    elif grade == 1 and ma.get("present") == "template_only":
        task = "L4_grade1_template"
        system = "你是眼底分级助手。RetSAM 不提供 MA；template_only 不能写成图像直接检出，也不能写成 RetSAM 检出。"
        user = "该图标注为轻度 DR，应如何给出证据解释？"
        ans = answer("清洗后没有可作为 L3 强监督的可见 MA 字段；HE/EX/SE 不作为阳性证据。", "dr_grade=1; MA present=template_only; MA source=grade_rule", "可按 Grade 1 的 MA-only 规则模板解释为轻度 DR，但不能表述为 RetSAM 检出 MA。", {"task": task, "dr_grade": 1, "MA": "template_only", "ma_source": "grade_rule", "forbid": "RetSAM_detected_MA"})
    else:
        if not ev:
            return None
        task = "L4_evidence_bound_grading"
        system = "你是眼底分级助手。分级必须引用病灶证据；MA unknown 时不得编造 MA。"
        user = "请先核查可见病灶证据，再判断该 DR 分级是否有依据。"
        ma_state = ma.get("present", "unknown")
        ans = answer("先核查 L3 病灶证据，再给出 DR 分级；不使用不存在或 unknown 的 MA 作为可见证据。", f"visible_lesions={','.join(ev)}; dr_grade={grade}; MA={ma_state}", f"监督分级为 DR Grade {grade}，主要解释证据为 {','.join(ev)}。", {"task": task, "dr_grade": grade, "evidence": ev, "MA": ma_state, "source": "validated_clean"})
    return sft(system, user, ans, image, {"record_id": rid(r), "task": task, "split": sp})


def l4_conflict(r: dict[str, Any], sp: str):
    flags = [f for f in r.get("cleaning_flags", []) if "strong_mask_conflicts_grade" in f]
    if not flags:
        return None
    return sft(
        "你是眼底分级质控助手。若强标注病灶与 grade 规则冲突，应提示需要复核，不要强行合理化。",
        "请判断该图的病灶证据与 DR 分级标签是否一致。",
        answer("强标注病灶与当前 grade 规则存在冲突，需要作为质控样本复核。", "cleaning_flags=" + ",".join(flags) + f"; dr_grade={r.get('grade')}", "该样本不适合作为普通证据绑定分级样本，应标记 needs_review=true。", {"task": "L4_conflict_review", "dr_grade": r.get("grade"), "needs_review": True, "flags": flags}),
        r["image_path"],
        {"record_id": rid(r), "task": "L4_conflict_review", "split": sp},
    )


def summarize(files: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    stats: dict[str, Any] = {"files": {}, "tasks": Counter(), "datasets": Counter()}
    for name, rows in files.items():
        ft, fd = Counter(), Counter()
        for row in rows:
            meta = row.get("meta", {})
            task = meta.get("task", "unknown")
            ds = "::".join(meta.get("record_id", "unknown").split("::")[:2])
            ft[task] += 1
            fd[ds] += 1
            stats["tasks"][task] += 1
            stats["datasets"][ds] += 1
        stats["files"][name] = {"n": len(rows), "tasks": dict(ft), "datasets": dict(fd)}
    stats["tasks"] = dict(stats["tasks"])
    stats["datasets"] = dict(stats["datasets"])
    return stats


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/fundus_validated/validated_clean.jsonl")
    ap.add_argument("--out-dir", default="data/annotation")
    ap.add_argument("--val-pct", type=int, default=10)
    ap.add_argument("--smoke-per-file", type=int, default=20)
    args = ap.parse_args()

    records = list(read_jsonl(Path(args.input)))
    out_dir = Path(args.out_dir)
    splits, split_stats = [], {"n_records": len(records), "val_pct": args.val_pct, "splits": defaultdict(Counter), "reasons": Counter()}
    split_map = {}
    for r in records:
        sp, reason, bucket = split_of(r, args.val_pct)
        split_map[image_key(r)] = sp
        ds = f"{r.get('dataset')}::{r.get('split')}"
        split_stats["splits"][sp][ds] += 1
        split_stats["reasons"][reason] += 1
        splits.append({"record_id": rid(r), "image_path": r.get("image_path"), "dataset": r.get("dataset"), "source_split": r.get("split"), "sft_split": sp, "reason": reason, "hash_bucket": bucket})

    files = {
        "fundus_l2_laterality_sft.jsonl": [],
        "fundus_l2_cdr_sft.jsonl": [],
        "fundus_l2_vessel_abstain_sft.jsonl": [],
        "fundus_l3_single_lesion_sft.jsonl": [],
        "fundus_l3_lesion_only_sft.jsonl": [],
        "fundus_l3_burden_sft.jsonl": [],
        "fundus_l4_evidence_grading_sft.jsonl": [],
        "fundus_l4_conflict_review_sft.jsonl": [],
    }
    for r in records:
        sp = split_map[image_key(r)]
        if sp != "train":
            continue
        for name, item in [
            ("fundus_l2_laterality_sft.jsonl", l2_laterality(r, sp)),
            ("fundus_l2_cdr_sft.jsonl", l2_cdr(r, sp)),
            ("fundus_l2_vessel_abstain_sft.jsonl", l2_vessel_abstain(r, sp)),
            ("fundus_l3_lesion_only_sft.jsonl", l3_lesion_only(r, sp)),
            ("fundus_l4_evidence_grading_sft.jsonl", l4_evidence(r, sp)),
            ("fundus_l4_conflict_review_sft.jsonl", l4_conflict(r, sp)),
        ]:
            if item:
                files[name].append(item)
        for key in ["MA", "HE", "EX", "SE"]:
            item = l3_single(r, sp, key)
            if item:
                files["fundus_l3_single_lesion_sft.jsonl"].append(item)
            item = l3_burden(r, sp, key)
            if item:
                files["fundus_l3_burden_sft.jsonl"].append(item)
        item = l3_se_abstain(r, sp)
        if item:
            files["fundus_l3_single_lesion_sft.jsonl"].append(item)

    write_jsonl(out_dir / "fundus_sft_split.jsonl", splits)
    for name, rows in files.items():
        write_jsonl(out_dir / name, rows)
    smoke = []
    smoke_counts = {}
    for name, rows in files.items():
        sample = rows[: args.smoke_per_file]
        smoke.extend(sample)
        smoke_counts[name] = len(sample)
    write_jsonl(out_dir / "fundus_sft_smoke.jsonl", smoke)

    stats = summarize(files)
    stats["split"] = {"n_records": len(records), "val_pct": args.val_pct, "splits": {k: dict(v) for k, v in split_stats["splits"].items()}, "reasons": dict(split_stats["reasons"])}
    stats["smoke"] = {"n": len(smoke), "per_file": smoke_counts}
    stats["outputs"] = {"split_file": str(out_dir / "fundus_sft_split.jsonl"), "smoke_file": str(out_dir / "fundus_sft_smoke.jsonl"), "sft_files": {name: str(out_dir / name) for name in files}}
    with (out_dir / "fundus_sft_stats.json").open("w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
