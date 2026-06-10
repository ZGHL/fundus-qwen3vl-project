# Stage 1.5 Six-Lesion Limited Experiment

Date: 2026-06-09

## Decision

Run a limited six-lesion Stage 1.5 experiment before Stage 2.

The experiment has two unequal objectives:

1. Primary: improve specificity and balanced accuracy for MA/HE/EX/SE without
   losing their F1.
2. Exploratory: determine whether IRMA and NV can acquire useful visual signal
   without degrading the four-lesion baseline.

IRMA may become usable with the current data. NV cannot yet be considered a
reliable production capability because its independent positive count is too
small.

## Independent rare-lesion data audit

Current disjoint Stage 1 splits:

| Lesion | Train positive images | Internal-val positive images | Locked-test positive images | Total independent positives |
|---|---:|---:|---:|---:|
| IRMA | 120 | 16 | 20 | 156 |
| NV | 34 | 3 | 8 | 45 |

Negative pools:

| Lesion | Train negatives | Internal-val negatives | Locked-test negatives |
|---|---:|---:|---:|
| IRMA | 500 | 88 | 80 |
| NV | 360 | 98 | 80 |

All rare-lesion positives use direct FGADR mask evidence. Split overlap is zero.

The existing 400 IRMA-positive training rows and 140 NV-positive training rows
are generated from 120 and 34 independent positive images respectively.
Augmentation increases visual variation but does not increase independent
clinical evidence.

## Feasibility judgment

### IRMA

IRMA is suitable for a limited learning experiment:

- 120 independent positive training images are small but usable for
  transfer-learning.
- There are enough negative images for hard-negative selection.
- The 20-positive locked test can detect a large improvement, although the
  confidence interval remains wide.

IRMA should remain an optional Stage 2 input until it passes the locked
guardrails.

### NV

NV is a few-shot exploratory target:

- 34 independent positive training images are below the level required for a
  stable general-purpose detector.
- Internal validation has only 3 positives and cannot reliably select
  checkpoints.
- Locked evaluation has only 8 positives; one image changes recall by 12.5
  percentage points.

The experiment can show whether NV signal is learnable, but a positive result
must not be interpreted as a stable NV capability.

## Training data distribution

Use strict single-lesion decoupled prompts. Do not use a joint six-lesion
output because previous experiments showed severe recall and formatting
failures.

Recommended mixed training set:

| Lesion | Positive rows | Negative rows | Total | Independent positive images |
|---|---:|---:|---:|---:|
| MA | 600 | 600 | 1,200 | prioritize unique S0/S1 |
| HE | 600 | 600 | 1,200 | prioritize unique S0/S1 |
| EX | 600 | 600 | 1,200 | prioritize unique S0/S1 |
| SE | 600 | 600 | 1,200 | prioritize unique S0/S1 |
| IRMA | 300 | 300 | 600 | 120; at most 3 views per positive |
| NV | 140 | 140 | 280 | 34; use existing conservative views |
| **Total** | **2,840** | **2,840** | **5,680** | |

Rare lesions represent 15.5% of task rows. This is enough to provide learning
signal without allowing 34 NV-positive images to dominate the visual adapter.

## Negative selection

Random negatives are insufficient. At least 70% of rare-lesion negatives must
be confusion-aware hard negatives.

IRMA hard negatives:

- Normal tortuous retinal vessels.
- Vessel crossings and branching points.
- HE adjacent to vessels.
- NV-positive images when IRMA is explicitly absent.
- Severe DR images without IRMA mask.

NV hard negatives:

- IRMA-positive images without NV.
- Normal disc and peripapillary vessels.
- Prominent ordinary vascular branching.
- Hemorrhage near vessels.
- Grade 3/severe images without direct NV mask.

Do not use Grade 4 as an NV-positive label. Do not use Grade 3 as an
IRMA-positive label.

## Positive augmentation

Only conservative image transformations are allowed:

- Brightness and contrast within 10%.
- Color/saturation within 15%.
- Crop ratio 0.90-1.00 when the lesion remains visible.
- Horizontal flip.

Avoid heavy crop, rotation, elastic transforms, and transformations that can
change vascular morphology.

Maximum task views:

| Lesion | Maximum views per independent positive |
|---|---:|
| IRMA | 3 |
| NV | approximately 4 |

Every augmented positive must retain linkage to its original image group so it
cannot cross train/dev/test splits.

## Output contract

Keep the proven Stage 1 strict single-lesion task:

```text
Target lesion: NV
Visible target-pattern evidence: present | absent
Confounder check: ...
Decision: present | absent
Structured output: {"lesion":"NV","present":true|false}
```

For IRMA/NV, the confounder sentence is required:

- IRMA: distinguish from NV, ordinary vessel branching, and vascular
  artifacts.
- NV: distinguish from IRMA, ordinary peripapillary vessels, and vessel
  crossings.

Do not ask the model to output all six lesions in one response.

## Model and optimization

Initialize from:

```text
/workspace/LLaMA-Factory/saves/qwen3-vl-8b-fundus/lora/stage1_en_cot_gentle_calibrated/checkpoint-20
```

Recommended limited adaptation:

- Continue language LoRA.
- Enable vision-tower LoRA.
- Enable the multimodal projector.
- Use a learning rate between `3e-7` and `8e-7`.
- Effective batch size 16.
- Run approximately 120-240 optimizer steps.
- Save and evaluate every 20 steps.
- Stop early when common-lesion guardrails fail.

The low learning rate and short run are necessary because the rare positive
images are heavily repeated.

## Evaluation sets

### Checkpoint selection

Use:

- A newly balanced four-lesion calibration Dev set.
- Rare-lesion internal validation as a diagnostic only.
- Main-four hard-negative challenge set.

Do not select checkpoints using locked IRMA/NV tests.

NV internal validation contains only three positives, so it cannot be a hard
selection gate.

### Final evaluation

After selecting one checkpoint:

- Evaluate Stage 1 four-lesion Gold Test once.
- Evaluate IRMA locked test once.
- Evaluate NV locked test once.

Report the raw confusion matrix and positive count, not only F1.

## Candidate acceptance rules

Primary four-lesion requirements:

| Metric | Requirement |
|---|---:|
| Four-lesion Macro F1 | at least current checkpoint-20 minus 0.01 |
| Four-lesion Macro specificity | at least 0.50 |
| Four-lesion balanced accuracy | improve over checkpoint-20 |
| HE F1 | at least 0.87 |
| EX F1 | at least 0.84 |
| MA specificity | at least 0.35 |
| SE specificity | at least 0.45 |

Rare-lesion exploratory targets:

| Metric | Target |
|---|---:|
| IRMA locked recall | at least 0.60 |
| IRMA locked specificity | at least 0.60 |
| IRMA locked F1 | improve over 0.326 |
| NV locked recall | at least 0.50, meaning at least 4/8 positives |
| NV locked specificity | at least 0.60 |
| NV locked F1 | improve over 0.208 |

The rare targets do not override the primary four-lesion guardrails. A
checkpoint that improves NV/IRMA but damages MA/HE/EX/SE is rejected.

## Interpretation rules

If IRMA passes its targets, it may be included in Stage 2 as weak or partial
evidence, with continued abstention support.

Even if NV passes the exploratory target, it remains `unknown` by default in
Stage 2 until additional independent NV-positive data validates the result.

If neither rare lesion improves, retain the four-lesion Stage 2 design and
treat this experiment as evidence that data quantity, rather than optimization,
is the limiting factor.
