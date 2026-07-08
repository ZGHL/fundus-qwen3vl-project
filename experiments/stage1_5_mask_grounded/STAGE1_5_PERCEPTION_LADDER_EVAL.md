# Stage-1.5 perception ladder — eval checklist (VM)

Goal: complete the NEW-mainline foundation comparison — **base (zero-shot) → Adapter1 →
Stage-1.5 v3 → v4** on ONE identical present/absent test, so per-lesion Recall / Specificity /
F1 (+ macro + BalAcc) are directly comparable. This is the capability axis the paper rests on
(the faithful grader is bounded by audit quality), NOT DR-grade QWK.

The v4 test (`stage1_5_v4_test`) uses the SAME image selection as v3, so one test set covers the
whole ladder. We already have v3 and Adapter1 predictions from the v3 head-to-head; the two NEW
runs are **base zero-shot** and **v4**.

## What each model needs
All run on `data/stage1_5_v4_test_sft.jsonl` (n=1108, Adapter1-unseen, image-disjoint), via vLLM
on the merged model, predictions row-aligned (vllm_infer preserves order). Same single-lesion
present/absent prompt for every model (it is already baked into the test rows).

| model | how |
|---|---|
| **base (zero-shot)** | the raw `Qwen3-VL-8B-Instruct`, NO adapter. Expected to over-report (low Spec) and/or fail to emit the JSON schema (parse_fail) — that gap is the point. |
| Adapter1 | merge `saves/.../lora/stage1_en_cot` (the same merged+tokenizer-fixed path used in the v3 head-to-head). |
| v3 | merge `saves/.../lora/stage1_5_v3/checkpoint-400`. |
| v4 | merge the chosen v4 checkpoint. |

> Eval hygiene (from the v3 run): merge LoRA via `llamafactory-cli export`, fix
> `tokenizer_config` `extra_special_tokens` (list -> {}), then vLLM the merged model — avoids the
> dynamic-visual-LoRA assertion and the all-`!` output bug. Base uses the plain base model.

## Steps
```bash
cd /workspace/stage1_5_experiment && git pull
TEST=data/stage1_5_v4_test_sft.jsonl   # == v3 test selection

# produce predictions (row order preserved) for each model:
python scripts/vllm_infer.py --model <base_or_merged_path> [--adapter ...] ... > preds/<label>.jsonl
#   labels: base_zeroshot, adapter1, v3, v4   (each a jsonl with a "predict" field per row)

# aggregate the ladder:
python scripts/perception_ladder.py "$TEST" \
    base:preds/base_zeroshot.jsonl adapter1:preds/adapter1.jsonl \
    v3:preds/v3.jsonl v4:preds/v4.jsonl \
    reports/STAGE1_5_PERCEPTION_LADDER.md
```

## What we expect (and what to report)
- **base zero-shot**: weak — either over-reports (Spec low) or high parse_fail (can't emit schema).
- **Adapter1 → v3**: Spec jumps (v3 head-to-head was macro Spec 0.198 → 0.594, BalAcc 0.711),
  Recall trades down slightly. Confirms v3 fixed over-reporting.
- **v3 → v4**: Recall should rise (esp. MA / HE / EX) while Spec holds — the v4 thesis.
- Report the full per-lesion table + macro + BalAcc + parse_fail. The base→v4 monotone story
  (capability built from nothing) is the figure for the paper's foundation section.

## Note
This ladder measures perception (present/absent), the new mainline's core. DR-grade QWK is
deliberately NOT the headline here; the old joint-CoT grade results (QWK 0.735 / messidor2 0.695,
plus the nv_irma 0.841 shortcut trap) are kept as the "accurate-but-unfaithful" foil elsewhere.
```
