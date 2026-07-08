# Stage-1.5 v5 — per-lesion differentiated rebalance (VM handoff)

## Why (one line)
v4 raised positives for ALL four lesions uniformly. That is right only for HE (recall-bound),
pointless for EX (already balanced), and the WRONG direction for MA and SE — both OVER-REPORT
(spec 0.48 / 0.40, recall already 0.95), so more positives makes them report even more. v5 treats
each lesion on its own bottleneck: **push HE recall, push MA/SE specificity, leave EX alone.**

## v3 ckpt-400 per-lesion profile this is built from
| lesion | recall | spec | F1 | bottleneck | v5 move |
|---|---:|---:|---:|---|---|
| HE | 0.706 | 0.810 | 0.706 | recall (caps grader referable sens 0.71) | present 1000→1800 |
| EX | 0.822 | 0.816 | 0.747 | none (balanced) | unchanged (1000/1300) |
| MA | 0.955 | 0.481 | 0.705 | specificity / over-report | absent 1300→2000 |
| SE | 0.950 | 0.397 | 0.344 | specificity / over-report (prec ~0.21) | absent 1300→2000 + confounder-first |

## What changed vs v4 (DATA ONLY — config still mirrors v3)
- `PRES_CAP` / `ABS_CAP` are now **per-lesion dicts**:
  - HE: present **1800** / absent 1300  (recall push; HE spec 0.810 has headroom)
  - EX: present 1000 / absent 1300      (unchanged = v3)
  - MA: present 1000 / absent **2000**  (spec push; recall is maxed, keep present moderate)
  - SE: present 1000 / absent **2000**  (spec push; SE has only ~906 real positives, so present
        resolves to ~860 either way — we do NOT add SE positives, we add SE NEGATIVES)
- **SE/MA absent draw HARD negatives first, and confounder-rich images first within that.** SE's
  hard-neg pool (~1733, empty-SE-mask DR images that carry EX/HE/glare) nearly fills ABS_CAP=2000
  on its own → the model gets exactly the bright-region confounders its `[Confounder Assessment]`
  step must learn to reject. (`CONFOUNDERS = {SE:{EX,HE}, MA:{HE}, ...}`.)
- **No RetSAM weak SE.** RetSAM SE is low-confidence and was suppressed in cleaning; adding it
  would be both noisy and the wrong direction (more SE positives). v5 stays real-mask-only.

## 3-way split (NEW in v5): train / dev / test, image-disjoint
The build now emits **three** image-disjoint sets so we never select the checkpoint on the set we
report:
- `stage1_5_v5_train` — training.
- `stage1_5_v5_dev` — **checkpoint selection ONLY** (150 mask + 120 grade-0 images, Adapter1-unseen).
- `stage1_5_v5_test` — **final number, never touched during selection**; first-slice selection ==
  v3/v4 test → the base→Adapter1→v3→v4→v5 ladder stays directly comparable.

DEV is carved from the Adapter1-unseen pool that would otherwise feed train (build asserts the pool
is large enough and that DEV∩TEST=∅), so train loses ~150 images — negligible for HE/MA/EX, and SE
present drops ~860→~820 (SE is data-capped anyway; an honest limitation). Selecting on macro-BalAcc
(dominated by the well-powered lesions) keeps DEV's small SE count from biasing the choice.

## Kept identical to v3/v4 (so the ladder stays comparable, eval stays leak-free)
- Single-lesion present/absent CoT (no count/area), Adapter1 warm-start, recipe = v3 (LR 3e-6,
  batch 2×8, gc on, sdpa).
- TEST selection identical → `stage1_5_v5_test` content == v3/v4 test. Per-lesion recall/spec
  directly comparable across base→Adapter1→v3→v4→v5.
- Stage-2 test stems excluded from TRAIN (`data/stage2_test_heldout_stems.txt`) + build asserts no
  leak → warm-starting Stage-2 from v5 and evaluating on the Stage-2 test is leak-free.

## Preprocessing: pay the ~20-min tokenize/image pass ONCE
The old configs had `overwrite_cache: true`, which forced a full re-preprocess on EVERY run. v5
config fixes this:
- `tokenized_path: data/tokenized/stage1_5_v5_train` — first run writes the preprocessed dataset
  there; every later run (resume/retune/restart) loads it in seconds. Put it on PERSISTENT disk.
- `overwrite_cache: false` — reuse the cache (auto-invalidates if data/tokenizer/cutoff/pixels change).
- `preprocessing_num_workers: 16` — set to the new VM's `nproc` (higher = faster first pass).
- Apply patches first (`scripts/setup/apply_llamafactory_patches.sh`) incl. the truncated-Pillow
  patch, or preprocessing can stall/fail on bad images.
- Data-scaling arms (25/50/100%) are different datasets → give each its OWN `tokenized_path`.

## Steps (VM)
```bash
cd /workspace/stage1_5_experiment && git pull     # build_v5 + config + dev sweep scripts

# 1) build TRAIN/DEV/TEST (real FGADR/DDR-seg masks + grade-0 negatives; reads heldout stems)
python scripts/build_stage1_5_v5.py
#   SANITY-CHECK the printed distribution before training:
#     - HE/present clearly > v3's 1000 (expect ~1700-1800); MA/absent ~2000; SE/absent ~2000
#     - SE/present ~860 (NOT inflated), EX unchanged ~1000/1300
#     - by_source has NO grade-derived / RetSAM source; only fgadr_mask/ddr_mask/grade0_neg/strong_mask
#     - stage2_test_stems_excluded_from_train = 297; ASSERTS no leak and DEV∩TEST=∅
#     - three files written: stage1_5_v5_{train,dev,test}_sft.jsonl

# 2) register stage1_5_v5_train / stage1_5_v5_dev / stage1_5_v5_test in data/annotation/dataset_info.json
#    (sharegpt, columns messages,images) — same as v3/v4 registration. DEV+TEST needed by vllm_infer --dataset.

# 3) train (warm-start Adapter1, recipe identical to v3/v4; first run pays preprocess ONCE -> tokenized_path)
llamafactory-cli train configs/stage1_5_v5_warmstart.yaml

# 4) SELECT the checkpoint on DEV (never on TEST). Single GPU -> run after training; second GPU -> POLL=1.
EXP=/workspace/stage1_5_experiment LF=/workspace/LLaMA-Factory bash scripts/run_stage1_5_v5_dev_sweep.sh
#   -> writes eval/v5_dev/dev_curve.csv (step vs macro-BalAcc + per-lesion recall/spec) and prints
#      the peak checkpoint. The curve flattening = "learned enough"; its peak = the checkpoint to keep.

# 5) FINAL number: run the SELECTED checkpoint ONCE on TEST, then the ladder (TEST never used in step 4):
cd /workspace/LLaMA-Factory && python scripts/vllm_infer.py \
    --model_name_or_path models/Qwen3-VL-8B-Instruct \
    --adapter_name_or_path saves/qwen3-vl-8b-fundus/lora/stage1_5_v5/checkpoint-<BEST> \
    --dataset stage1_5_v5_test --dataset_dir data/annotation --media_dir data \
    --template qwen3_vl_nothink --cutoff_len 2304 --max_new_tokens 512 \
    --image_max_pixels 589824 --image_min_pixels 65536 --batch_size 16 --enforce_eager true \
    --max_lora_rank 32 --gpu_memory_utilization 0.80 --save_name $EXP/eval/v5_test.jsonl
python $EXP/scripts/perception_ladder.py data/annotation/stage1_5_v5_test_sft.jsonl \
    base:preds/base.jsonl adapter1:preds/adapter1.jsonl v3:preds/v3.jsonl v4:preds/v4.jsonl \
    v5:$EXP/eval/v5_test.jsonl  $EXP/eval/STAGE1_PERCEPTION_LADDER_v5.md
```

## Success criteria (accept v5 only if ALL hold; per-lesion, on the identical test)
- **HE recall ↑** clearly above v3's 0.706 (without HE spec collapsing below ~0.70).
- **MA spec ↑** clearly above v3's 0.481 (MA recall may dip from 0.955 — that is the intended trade).
- **SE spec ↑** clearly above v3's 0.397 / precision above ~0.21 (SE recall may dip from 0.950).
- **EX unchanged** (sanity: same data → same numbers within noise).
If MA/SE spec still won't move, raise their `ABS_CAP` further (the cap is the knob). If HE recall
won't move, raise HE `PRES_CAP` (bounded by ~2137 available).

## Then: leak-free grader eval (Stage-2 unchanged, just better audits)
Re-warm-start Stage-2 from the best v5 ckpt (`configs/stage2_grade_warmstart.yaml`, set
`adapter_name_or_path` to the v5 ckpt), keep the SAME stage2_grade data + map, then:
```bash
python scripts/score_stage2.py data/stage2_grade_test_sft.jsonl <v5grader_pred.jsonl> \
    --from-audit --dist data/stage2_grade_distribution.json
```
Expect (vs v4): referable sensitivity holds/↑ from the HE recall gain, AND fewer MA/SE false
positives → cleaner tier assignment; faithfulness stays 1.000 (map untouched).

## Do NOT
- install liger-kernel (breaks triton/vLLM on this stack).
- add grade-derived or RetSAM-SE weak labels; do not change the single-lesion CoT / Stage-2 map.
- change any training hyperparameter relative to v3/v4.
