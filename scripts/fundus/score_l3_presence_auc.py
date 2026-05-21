#!/usr/bin/env python3
"""Compute forced-choice ROC-AUC for L3 lesion presence tasks."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from peft import PeftModel
from transformers import AutoModelForImageTextToText, AutoProcessor


LESION_CN = {"MA": "微动脉瘤", "HE": "出血", "EX": "硬性渗出", "SE": "软性渗出"}
LESION_CUE = {
    "MA": "微小红色圆点样病灶",
    "HE": "片状或点状红色出血灶",
    "EX": "边界较清楚的黄白色硬性渗出",
    "SE": "边界较模糊的棉絮样白色软性渗出",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def auc_rank(labels: list[int], scores: list[float]) -> float | None:
    pos = sum(labels)
    neg = len(labels) - pos
    if pos == 0 or neg == 0:
        return None
    order = sorted(range(len(scores)), key=lambda i: scores[i])
    ranks = [0.0] * len(scores)
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and scores[order[j]] == scores[order[i]]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        for k in range(i, j):
            ranks[order[k]] = avg_rank
        i = j
    rank_sum_pos = sum(ranks[i] for i, y in enumerate(labels) if y == 1)
    return (rank_sum_pos - pos * (pos + 1) / 2.0) / (pos * neg)


def canonical_answer(lesion: str, present: bool) -> str:
    cn = LESION_CN[lesion]
    cue = LESION_CUE[lesion]
    if present:
        obs = f"围绕{cn}的典型外观进行观察：{cue}。本题只训练该单一病灶概念。"
        ev = f"{lesion} present=true; count=unknown; area=unknown"
        concl = f"支持{cn}阳性；本题不输出 DR 分级，也不合并其他病灶结论。"
    else:
        obs = f"围绕{cn}的典型外观进行观察：{cue}。本题只判断该单一病灶是否存在。"
        ev = f"{lesion} present=false; count=unknown; area=unknown"
        concl = f"未见可靠{cn}阳性证据；本题不输出 DR 分级，也不合并其他病灶结论。"
    payload = {
        "task": f"L3_{lesion}_single",
        "lesion": lesion,
        "present": present,
        "count": "unknown",
        "area": "unknown",
    }
    return f"【观察】{obs}\n\n【证据】{ev}\n\n【结论】{concl}\n\n【JSON】\n{json.dumps(payload, ensure_ascii=False, separators=(',', ':'))}"


def normalize_messages(row: dict[str, Any], assistant: str | None) -> list[dict[str, Any]]:
    messages = []
    for m in row["messages"]:
        if m["role"] == "assistant":
            if assistant is not None:
                messages.append({"role": "assistant", "content": assistant})
            break
        content = m["content"]
        if m["role"] == "user" and isinstance(content, str) and content.startswith("<image>\n"):
            content = content.split("\n", 1)[1]
            messages.append({"role": "user", "content": [{"type": "image"}, {"type": "text", "text": content}]})
        else:
            messages.append({"role": m["role"], "content": content})
    return messages


def move_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {k: v.to(device) if hasattr(v, "to") else v for k, v in batch.items()}


def candidate_logprob(model, processor, row: dict[str, Any], image: Image.Image, answer: str) -> float:
    prompt_messages = normalize_messages(row, None)
    full_messages = normalize_messages(row, answer)
    prompt_text = processor.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)
    full_text = processor.apply_chat_template(full_messages, tokenize=False, add_generation_prompt=False)

    prompt_inputs = processor(text=[prompt_text], images=[image], return_tensors="pt", padding=True)
    full_inputs = processor(text=[full_text], images=[image], return_tensors="pt", padding=True)
    prompt_len = int(prompt_inputs["attention_mask"][0].sum().item())

    labels = full_inputs["input_ids"].clone()
    labels[:, :prompt_len] = -100
    labels[full_inputs["attention_mask"] == 0] = -100
    full_inputs = move_to_device(full_inputs, model.device)
    labels = labels.to(model.device)

    with torch.inference_mode():
        logits = model(**full_inputs).logits
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        valid = shift_labels.ne(-100)
        log_probs = torch.nn.functional.log_softmax(shift_logits, dim=-1)
        token_lp = log_probs.gather(-1, shift_labels.clamp_min(0).unsqueeze(-1)).squeeze(-1)
        return float(token_lp[valid].sum().item() / max(int(valid.sum().item()), 1))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="./models/Qwen3-VL-8B-Instruct")
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--data", default="data/annotation/fundus_l3_presence_holdout80_sft.jsonl")
    ap.add_argument("--media-dir", default="data")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    model = AutoModelForImageTextToText.from_pretrained(
        args.model,
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()

    rows = read_jsonl(Path(args.data))
    media_dir = Path(args.media_dir)
    scored = []
    by_lesion: dict[str, dict[str, list[float] | list[int]]] = {}
    for idx, row in enumerate(rows):
        meta = row.get("meta", {})
        lesion = meta["lesion"]
        y = 1 if meta["present"] is True else 0
        image_path = Path(row["images"][0])
        if not image_path.is_absolute():
            image_path = media_dir / image_path
        image = Image.open(image_path).convert("RGB")
        lp_true = candidate_logprob(model, processor, row, image, canonical_answer(lesion, True))
        lp_false = candidate_logprob(model, processor, row, image, canonical_answer(lesion, False))
        score = lp_true - lp_false
        rec = {
            "idx": idx,
            "record_id": meta.get("record_id"),
            "lesion": lesion,
            "label": y,
            "score": score,
            "lp_true": lp_true,
            "lp_false": lp_false,
            "hard_pred": int(score >= 0),
        }
        scored.append(rec)
        bucket = by_lesion.setdefault(lesion, {"labels": [], "scores": []})
        bucket["labels"].append(y)
        bucket["scores"].append(score)
        print(json.dumps(rec, ensure_ascii=False), flush=True)

    labels_all = [r["label"] for r in scored]
    scores_all = [r["score"] for r in scored]
    metrics: dict[str, Any] = {
        "n": len(scored),
        "adapter": args.adapter,
        "data": args.data,
        "micro_auc": auc_rank(labels_all, scores_all),
        "by_lesion": {},
    }
    for lesion, data in sorted(by_lesion.items()):
        labels = data["labels"]
        scores = data["scores"]
        auc = auc_rank(labels, scores)
        tp = fp = tn = fn = 0
        for y, s in zip(labels, scores):
            pred = s >= 0
            if y == 1 and pred:
                tp += 1
            elif y == 1:
                fn += 1
            elif pred:
                fp += 1
            else:
                tn += 1
        metrics["by_lesion"][lesion] = {
            "n": len(labels),
            "auc": auc,
            "score_mean_pos": sum(s for y, s in zip(labels, scores) if y == 1) / max(sum(labels), 1),
            "score_mean_neg": sum(s for y, s in zip(labels, scores) if y == 0) / max(len(labels) - sum(labels), 1),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
        }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with Path(args.out).open("w", encoding="utf-8") as f:
        json.dump({"metrics": metrics, "scores": scored}, f, ensure_ascii=False, indent=2)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
