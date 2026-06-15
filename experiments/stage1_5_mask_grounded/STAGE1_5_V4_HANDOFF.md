# Stage-1.5 v4 — recall-rebalance handoff (VM)

## Why (one line)
Under the decoupled from-audit grader, the only remaining real weaknesses are **referable
sensitivity 0.71 / severe recall 0.84** (audit under-detects HE/EX on ~49/168 referable cases)
and **Mild F1 = 0** (no aptos-domain MA in v3). Both are *recall* problems from v3's
specificity-first balance. v4 rebalances the DATA only — same format, same warm-start, same
LoRA recipe, **same test set** — so the effect is measurable head-to-head against v3.

## What changed vs v3 (data only)
- caps: `PRES_CAP 1000→1300`, `ABS_CAP 1300→1100` (present-leaning; absent stays hard-neg-first,
  trims bulk grade-0 easy negatives that suppressed recall).
- new source `g1_ma_derived`: aptos/ddr_grading **g1 → MA-present** (grade-derived, clinically
  mild = MA-only), capped `MA_DERIVED_CAP=600` → teaches MA on the aptos domain → fixes Mild.
- test = **identical to v3** (same `N_TEST_MASK_IMG=150`, `N_TEST_G0_IMG=120`, same hashing).

## Steps
```bash
cd /workspace/stage1_5_experiment && git pull          # gets build_stage1_5_v4.py + config

# 1) build (needs FGADR/DDR-seg masks + validated_clean.jsonl, same as v3)
python scripts/build_stage1_5_v4.py
#   -> data/annotation/stage1_5_v4_{train,test}_sft.jsonl + data/stage1_5_v4_distribution.json
#   SANITY-CHECK the printed distribution before training:
#     - MA/present should now include ~600 'g1_ma_derived' (by_source)
#     - present:absent per lesion ~ 1300:1100 (was 1000:1300)
#     - test must be byte-identical to stage1_5_v3_test (same n=1108, all_unseen=true)

# 2) register both datasets in LLaMA-Factory data/annotation/dataset_info.json (sharegpt):
#   "stage1_5_v4_train": {"file_name":"stage1_5_v4_train_sft.jsonl","formatting":"sharegpt",
#       "columns":{"messages":"messages","images":"images"},
#       "tags":{"role_tag":"role","content_tag":"content","user_tag":"user","assistant_tag":"assistant","system_tag":"system"}}
#   "stage1_5_v4_test":  same with the _test file.

# 3) train (warm-start Adapter1, identical recipe to v3)
llamafactory-cli train configs/stage1_5_v4_warmstart.yaml
#   -> saves/qwen3-vl-8b-fundus/lora/stage1_5_v4/checkpoint-*

# 4) EVAL on the v3-identical test, head-to-head with v3 (and Adapter1) via vLLM:
#   for each candidate ckpt: merge -> vllm_infer on stage1_5_v4_test -> score_proof.py
python scripts/vllm_infer.py --adapter_name_or_path saves/.../stage1_5_v4/checkpoint-XXX \
    --max_lora_rank 32 --enforce_eager true   # produce predictions on stage1_5_v4_test
python scripts/score_proof.py data/stage1_5_v4_test_sft.jsonl <pred.jsonl>   # present/absent F1/Recall/Spec
```

## Success criteria (accept v4 only if BOTH hold)
- **Recall up**: per-lesion recall (esp. MA, HE, EX) clearly above v3 on the identical test.
- **Specificity holds**: macro/per-lesion specificity not materially below v3 (some drop is
  expected and acceptable; a collapse is not). If spec collapses, lower `PRES_CAP`/raise
  `ABS_CAP` and rebuild — the caps are the knob.

## Then: re-run the grader (Stage-2 unchanged, just better audits)
v4 is a drop-in better audit model. Re-warm-start Stage-2 from the best v4 ckpt
(`configs/stage2_grade_warmstart.yaml`, change `adapter_name_or_path` to the v4 ckpt), keep the
SAME stage2_grade data + map, then re-run from-audit eval:
```bash
python scripts/score_stage2.py data/stage2_grade_test_sft.jsonl <v4grader_pred.jsonl> \
    --from-audit --dist data/stage2_grade_distribution.json
```
Expect: **referable sensitivity ↑, severe recall ↑, Mild F1 > 0**; faithfulness stays 1.000
(map unchanged), abstention roughly stable. The Stage-2 decision map is NOT touched — this run
only improves the audit that feeds it.

## Do NOT
- install liger-kernel (breaks triton/vLLM on this stack).
- change the single-lesion CoT format or the Stage-2 map (keeps v4 comparable + faithful).
