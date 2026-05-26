# L3 Joint Mix Baseline on Current Full Pool

Date: 2026-05-25

## Goal

Compare the current decoupled L3 lesion-perception model against a practical
L3 joint/mix lesion-perception baseline using the same effective image pool and
the same evaluation policy.

This design does not include L2 biomarkers or L4 DR grading.

## Baseline Definitions

### Arm A: Current Decoupled Full

Already trained:

`saves/qwen3-vl-8b-fundus/lora/lesion_perception_en_cot_full`

Training format:

One row asks for one target lesion only.

Training rows:

12,408 single-lesion rows.

### Arm B: Joint Mix Full Pool

New baseline to build:

`fundus_l3_joint_mix_full_train`

Training format:

One row corresponds to one effective image. The prompt asks the model to audit
all lesions that have usable supervision for that image and output one joint
structured JSON block.

Example output:

```json
{
  "task": "l3_joint_lesion_perception",
  "lesions": {
    "HE": {"present": true, "evidence_state": "present", "source": "retsam_validated"},
    "EX": {"present": false, "evidence_state": "absent", "source": "retsam_negative"},
    "MA": {"present": true, "evidence_state": "present", "source": "strong_mask"}
  }
}
```

Only lesions with direct present/absent supervision should appear. Unknown or
template-only labels should not be forced into the JSON as visual labels.

## Current Full Pool Statistics

Current decoupled train file:

`data/annotation_v4/fundus_lesion_perception_en_cot_full_train_sft.jsonl`

| Quantity | Count |
|---|---:|
| Decoupled rows | 12,408 |
| Effective unique training images | 5,458 |
| Image-level lesion decisions after de-duplication | 12,145 |

Image-level train labels after de-duplication:

| Lesion | Positive | Negative |
|---|---:|---:|
| HE | 1,862 | 931 |
| EX | 2,652 | 1,326 |
| MA | 600 | 237 |
| SE | 1,240 | 1,240 |
| IRMA | 603 | 754 |
| NV | 100 | 600 |

The row count becomes smaller because repeated rows used for exposure balancing
collapse to one effective image-level joint sample. This is acceptable for this
baseline because the question is whether a joint mix prompt can use the same
effective images better, not whether it sees the same number of repeated rows.

## Why This Is Fair Enough

The comparison controls the important parts:

- Same image pool for training.
- Same underlying lesion evidence.
- Same train/validation/locked image split.
- Same model and LoRA hyperparameters.
- Same evaluation images.
- Same per-lesion metrics.

The comparison does not control row count, by design. The mix baseline uses fewer
rows because it compresses multiple lesion decisions into one image-level prompt.
Report both row count and lesion-decision count.

## Training Plan

Use the same LoRA configuration as Arm A:

- Base model: `models/Qwen3-VL-8B-Instruct`
- LoRA rank: 16
- LoRA alpha: 32
- LoRA dropout: 0.05
- Effective batch size: 16
- Learning rate: 6e-6
- Epochs: 1.0
- bf16
- image pixels: 65536 to 262144
- cutoff length: 2304

Expected cost should be lower than Arm A because the joint mix train set is
about 5,458 rows instead of 12,408 rows.

## Evaluation Strategy

Build joint versions of the same evaluation sets:

| Decoupled eval set | Joint mix eval set |
|---|---|
| `fundus_lesion_perception_val_subset_eval` | `fundus_l3_joint_mix_val_subset_eval` |
| `fundus_lesion_perception_balanced_eval` | `fundus_l3_joint_mix_balanced_eval` |
| `fundus_lesion_perception_irma_locked_eval` | `fundus_l3_joint_mix_irma_locked_eval` |
| `fundus_lesion_perception_en_cot_nv_locked_eval` | `fundus_l3_joint_mix_nv_locked_eval` |

Each joint eval row should correspond to one image and contain all lesion labels
available for that image in the source eval set.

Scoring:

Expand each joint JSON prediction into lesion-level decisions and compute the
same metrics already used for the decoupled model:

- Precision
- Recall / sensitivity
- Specificity
- F1
- Balanced accuracy
- Macro F1
- Rare macro F1
- Rare macro balanced accuracy
- JSON parse success
- No-grade-output rate

## Reporting

Main table:

| Model | Train rows | Lesion decisions | Eval set | Macro F1 | Rare F1 | JSON parse |
|---|---:|---:|---|---:|---:|---:|
| Decoupled full | 12,408 | 12,408 row-level | balanced | existing | existing | existing |
| Joint mix full pool | ~5,458 | ~12,145 image-level | balanced | new | new | new |

Per-lesion table:

Report HE, EX, MA, SE, IRMA, and NV separately. Do not hide NV/IRMA under macro
averages.

## Interpretation Rule

Claim joint mix is better only if it improves at least one of:

- IRMA recall
- NV recall
- rare macro balanced accuracy

without a large drop in:

- HE/EX/MA/SE average F1
- specificity
- JSON parse success

Claim decoupled is better only if it remains stronger on common-lesion macro F1
and joint mix does not improve rare lesion recall.

If both fail on IRMA/NV, the conclusion should be:

"The current data construction and prompt style are insufficient for rare
vascular lesion recall; the next step should target rare-lesion supervision
rather than decoupled-vs-joint framing."

## Implementation Tasks

1. Build `fundus_l3_joint_mix_full_train` by grouping the current decoupled full
   train JSONL by image path.
2. Build joint eval JSONL files by grouping each existing decoupled eval set by
   image path.
3. Register the new datasets in `dataset_info.json`.
4. Add LLaMA-Factory train/eval YAML files for the joint mix arm.
5. Extend the scorer to parse joint JSON and expand it into lesion-level
   decisions.
6. Train only the joint mix baseline and compare against existing Arm A outputs.
