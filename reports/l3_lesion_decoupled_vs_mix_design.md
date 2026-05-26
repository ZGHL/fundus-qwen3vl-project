# L3 Lesion Perception: Decoupled vs Mix Design

Date: 2026-05-25

## Question

Does L3 lesion perception improve when the model learns one lesion at a time
instead of learning a mixed lesion-perception objective?

This experiment intentionally excludes L2 biomarker tasks and L4 DR grading.
The comparison is only about L3 lesion perception.

## Definitions

### Decoupled L3

One training row asks about exactly one target lesion:

- HE
- EX
- MA
- SE
- IRMA
- NV

The model outputs only the target lesion's `present` decision. This is the
current single-lesion setup.

### Mix L3

The model sees L3 lesion-perception supervision in a mixed objective. There are
two useful variants:

1. `single-row mixed`: still one lesion per row, but lesions are mixed together
   in one training dataset without the expanded decoupled balancing used by the
   current full run.
2. `joint-audit mixed`: one row asks the model to audit all six lesions in the
   same image and output six `present` decisions.

For the core scientific question, `joint-audit mixed` is the cleaner opposite of
decoupling. The existing `fundus_v5_mixed_train` L3 subset is useful as a
historical single-row mixed baseline.

## Current Data Facts

### Current Decoupled Full

Dataset: `fundus_lesion_perception_en_cot_full_train`

Rows: 12,408

| Lesion | Positive | Negative |
|---|---:|---:|
| HE | 1,862 | 931 |
| EX | 2,652 | 1,326 |
| MA | 600 | 300 |
| SE | 1,240 | 1,240 |
| IRMA | 603 | 754 |
| NV | 300 | 600 |

### Existing L3 Mixed Subset

Source: L3 rows inside `fundus_v5_mixed_train`

Rows: 9,992

| Lesion | Positive | Negative |
|---|---:|---:|
| HE | 954 | 954 |
| EX | 1,355 | 1,355 |
| MA | 264 | 264 |
| SE | 1,274 | 1,274 |
| IRMA | 702 | 971 |
| NV | 125 | 500 |

This historical subset is useful as a reference, but it should not be the main
fair baseline. The current decoupled run uses a newer sample-construction idea,
especially for MA/NV/IRMA. A fair practical comparison should let the mix arm
use the same reconstructed lesion evidence pool, then change only how lesion
supervision is presented to the model.

## Experimental Arms

### Arm A: Decoupled Full

Use the existing trained model:

`saves/qwen3-vl-8b-fundus/lora/lesion_perception_en_cot_full`

Purpose:

Measure the current best decoupled L3 setup.

### Arm B: Decoupled Full Rerun

Use the same dataset recipe as Arm A, optionally rerun with a new output
directory for strict reproducibility.

Rows: 12,408

Purpose:

This is the decoupled side of the practical head-to-head comparison.

### Arm C: L3 Single-Row Mix, Same Samples

Build a new L3-only mixed dataset from the same reconstructed sample pool as the
current decoupled full run. Keep the same lesion-level counts:

| Lesion | Positive | Negative |
|---|---:|---:|
| HE | 1,862 | 931 |
| EX | 2,652 | 1,326 |
| MA | 600 | 300 |
| SE | 1,240 | 1,240 |
| IRMA | 603 | 754 |
| NV | 300 | 600 |

Rows: 12,408

Purpose:

Compare against a lesion-only mix recipe while keeping the same reconstructed
NV/IRMA/MA sample budget. This tests whether the decoupled prompt/objective is
better than mixed lesion exposure under the same evidence pool.

Implementation detail:

Use a mixed lesion-perception system prompt and row metadata, but still output
one target lesion per row so the existing scorer and eval sets remain valid.
This is not a joint-audit task; it is a same-sample L3 mixed baseline.

### Arm D: L3 Joint-Audit Mix, Same Evidence Pool

Build a new L3-only dataset where each image-level row asks the model to audit
all six lesions at once:

```json
{
  "task": "lesion_perception_joint",
  "lesions": {
    "HE": {"present": true},
    "EX": {"present": false},
    "MA": {"present": true},
    "SE": {"present": false},
    "IRMA": {"present": false},
    "NV": {"present": false}
  }
}
```

Purpose:

This is the clean conceptual test: one-lesion-at-a-time vs all-lesions-at-once.

Because one joint row contains up to six lesion labels, report both:

- row count
- lesion-decision count

## Main Fair Comparison

Primary comparison:

Arm A/B Decoupled Full vs Arm C L3 Single-Row Mix, Same Samples

Reason:

Both use L3-only data, the same reconstructed MA/NV/IRMA sample strategy, the
same row count, and the same evaluation sets. The remaining intended difference
is task framing: explicitly decoupled single-lesion learning vs mixed L3 lesion
perception exposure.

Secondary conceptual comparison:

Arm A/B Decoupled Full vs Arm D L3 Joint-Audit Mix

Reason:

This tests task decomposition more directly, but requires a new joint-audit
builder and scorer.

Reference-only comparison:

Arm A Decoupled Full vs historical L3 subset from `fundus_v5_mixed_train`

Reason:

This shows whether the newer sample reconstruction helped, but it should not be
used as the main decoupled-vs-mix conclusion because the sample budgets differ.

## Evaluation Sets

Use the exact same eval sets for every arm:

| Eval set | Role |
|---|---|
| `fundus_lesion_perception_val_subset_eval` | Main medium validation subset |
| `fundus_lesion_perception_balanced_eval` | Balanced common-lesion stress test |
| `fundus_lesion_perception_irma_locked_eval` | IRMA rare-lesion locked eval |
| `fundus_lesion_perception_en_cot_nv_locked_eval` | NV rare-lesion locked eval |

Do not use training-overlapping historical NV/IRMA holdouts for the main table.

## Metrics

Primary:

- Macro F1 over lesion presence.

Per lesion:

- Precision
- Recall / sensitivity
- Specificity
- F1
- Balanced accuracy

Rare lesion:

- IRMA recall
- NV recall
- Rare macro recall
- Rare macro F1
- Rare macro balanced accuracy

Format:

- JSON parse success
- Target lesion consistency
- No-grade-output rate

AUC is excluded because the current schema does not produce calibrated
continuous probability scores.

## Fairness Controls

Keep fixed:

- Base model: `Qwen3-VL-8B-Instruct`
- LoRA rank: 16
- LoRA alpha: 32
- LoRA dropout: 0.05
- LoRA target: all
- Effective batch size: 16
- Learning rate: 6e-6
- Epochs: 1.0
- Scheduler: cosine
- Warmup ratio: 0.03
- Image min/max pixels: 65536 / 262144
- Cutoff length: 2304
- Greedy generation for evaluation

Keep evaluation prompts identical. Arm C should be evaluated through the same
single-lesion eval prompts as Arm A/B, so differences in the metric are not
caused by the evaluator asking a different question.

## Current Baseline Result

Arm A has already been trained.

| Eval set | n | Macro F1 | Rare F1 |
|---|---:|---:|---:|
| Balanced eval | 200 | 0.6243 | n/a |
| Validation subset | 1,230 | 0.5699 | 0.0800 |
| IRMA locked eval | 100 | n/a | n/a |
| NV locked eval | 105 | n/a | n/a |

Interpretation:

The current decoupled full model learns HE, EX, and MA reasonably well, but
IRMA/NV recall is still poor. A mix baseline is only better if it improves rare
lesion recall without destroying common-lesion F1 and specificity.

## Decision Criteria

Claim decoupling helps if Arm A/B beats Arm C on:

- `fundus_lesion_perception_val_subset_eval` macro F1.
- `fundus_lesion_perception_balanced_eval` macro F1.
- Common-lesion average F1 over HE/EX/MA/SE.
- No worse than 2 percentage points drop in specificity.

Claim decoupling helps rare lesions only if it also beats Arm C on:

- IRMA recall.
- NV recall.
- Rare macro balanced accuracy.

If Arm B improves common lesions but not IRMA/NV, the conclusion should be:

"L3 lesion decoupling improves common lesion perception, but rare vascular
lesion recall still needs a targeted training strategy."

## Recommended Implementation Order

1. Build Arm C: `fundus_l3_single_row_mix_full_train` from the same sample pool
   and lesion counts as `fundus_lesion_perception_en_cot_full_train`.
2. Use the same LoRA hyperparameters as Arm A.
3. Evaluate Arm C on the same four fixed eval sets.
4. Score Arm C with the same single-lesion scorer.
5. Compare Arm A vs Arm C in a new report.
6. Build Arm D only after Arm A vs Arm C is clear, because joint-audit scoring
   requires an expanded parser.
