# Stage-1.5 v4 — faithful recall-rebalance handoff (VM)

## Why (one line)
The decoupled from-audit grader's only clinically-valuable weakness is recall (referable
sensitivity 0.71 / severe recall 0.84 — the audit under-detects HE/EX). v3 was specificity-first
(present:absent = 1000:1300), so recall was conservative. v4 rebalances toward recall using ONLY
mask-grounded positives — no grade-derived weak labels — and fixes a Stage-2 eval leak.

## What changed vs v3 (data only — config is identical, see below)
- `PRES_CAP 1000 -> 2000`: use ALL real mask positives we have (MA ~2075, HE ~2137, EX ~1846,
  SE ~906). v3 capped present at 1000 and left half the real MA unused.
- `ABS_CAP` unchanged at 1300 (same proven negative budget -> specificity preserved). Net per-
  lesion ratio flips from absent-leaning 0.77:1 to present-leaning ~1.4:1 -> recall up, spec held.
- **Removed `g1_ma_derived`.** The earlier draft inferred MA-present from grade-1 labels, which
  (a) violates the single-lesion rule "do not infer presence from a DR grade", (b) risks teaching
  MA hallucination -> hurts MA specificity, and (c) actually cut grounded MA to 700 (< v3's 1000).
- **Leakage fix (critical):** the Stage-2 grading test (297 images) overlapped the v3/v4
  Stage-1.5 training pool by 172. v4 train now EXCLUDES every Stage-2-test stem
  (`data/stage2_test_heldout_stems.txt`). A built-in assertion fails the build if any leak
  remains. So warm-starting Stage-2 from v4 and evaluating on the Stage-2 test is leak-free.
  (The v3 referable/Mild numbers were leak-affected — do NOT cite them as generalization.)
- Test selection = identical to v3 (same `N_TEST_MASK_IMG=150`, `N_TEST_G0_IMG=120`, hashing).

Mild / aptos: there is no faithful MA label for aptos (no masks; RetSAM weak labels have no MA),
so v4 does not touch aptos. Mild is represented faithfully by real MA-only mask images and MA
recall is measured in-domain on the test. Whether mask-domain MA transfers to aptos is an honest,
separate limitation — not faked with grade-derived labels.

## Training config — MUST be identical to v3 (only data differs)
`configs/stage1_5_v4_warmstart.yaml` mirrors the v3 RUN as reported (LR 3e-6, batch 2x8,
gradient_checkpointing on, sdpa). **If your actual v3 run config differs in any field, copy v3's
config verbatim and change ONLY `dataset` -> stage1_5_v4_train and `output_dir` -> .../stage1_5_v4.**
Apples-to-apples requires the optimizer/batch/attn/cache to match; the only variable is the data.

## Steps
```bash
cd /workspace/stage1_5_experiment && git pull     # build_stage1_5_v4.py + config + heldout stems

# 1) build (real FGADR/DDR-seg masks + validated_clean; reads data/stage2_test_heldout_stems.txt)
python scripts/build_stage1_5_v4.py
#   SANITY-CHECK the printed distribution before training:
#     - MA/present clearly > v3's 1000 (expect ~1600-1800 real mask MA); NO g1_ma_derived source
#     - by_source = fgadr_mask / ddr_mask / grade0_neg / strong_mask only (no derived)
#     - stage2_test_stems_excluded_from_train = 297 ; train_skipped_as_heldout shows the drops
#     - build ASSERTS no Stage-2 test stem leaked into train (fails loudly if so)

# 2) register stage1_5_v4_train / stage1_5_v4_test in data/annotation/dataset_info.json (sharegpt,
#    columns messages,images) — same as v3 registration.

# 3) train (warm-start Adapter1, recipe identical to v3)
llamafactory-cli train configs/stage1_5_v4_warmstart.yaml

# 4) EVAL head-to-head with v3 (and Adapter1) on the v3-identical test, via vLLM + merged model:
python scripts/vllm_infer.py --adapter_name_or_path saves/.../stage1_5_v4/checkpoint-XXX \
    --max_lora_rank 32 --enforce_eager true        # predictions on stage1_5_v4_test
python scripts/score_proof.py data/stage1_5_v4_test_sft.jsonl <pred.jsonl>   # present/absent F1/Recall/Spec
```

## Success criteria (accept v4 only if BOTH hold)
- **Recall up**: per-lesion recall (esp. MA, HE, EX) clearly above v3 on the identical test.
- **Specificity holds**: macro/per-lesion spec not materially below v3 (~0.59). If spec drops too
  far, raise `ABS_CAP` / lower `PRES_CAP` and rebuild — the caps are the knob.

## Then: leak-free grader eval (Stage-2 unchanged, just better audits)
Re-warm-start Stage-2 from the best v4 ckpt (`configs/stage2_grade_warmstart.yaml`, set
`adapter_name_or_path` to the v4 ckpt), keep the SAME stage2_grade data + map, then:
```bash
python scripts/score_stage2.py data/stage2_grade_test_sft.jsonl <v4grader_pred.jsonl> \
    --from-audit --dist data/stage2_grade_distribution.json
```
Because v4 train excluded all 297 Stage-2-test stems, this is a leak-free generalization number.
Expect: referable sensitivity ↑, severe recall ↑; faithfulness stays 1.000 (map untouched).
Mild depends on mask-domain MA transferring to the aptos-derived Mild cases — report honestly.

## Do NOT
- install liger-kernel (breaks triton/vLLM on this stack).
- add grade-derived labels or change the single-lesion CoT format / Stage-2 map.
- change any training hyperparameter relative to v3.
