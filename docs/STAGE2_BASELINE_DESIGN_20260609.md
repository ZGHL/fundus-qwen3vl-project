# Stage 2 Baseline Design

Date: 2026-06-09

The final detailed CoT contract is defined in
`docs/STAGE2_STRUCTURED_COT_SPEC_20260609.md`. It supersedes the earlier
free-text reasoning examples in this document.

## Objective

Train the first Stage 2 DR-grading baseline from the selected Stage 1 balanced
adapter:

```text
/workspace/LLaMA-Factory/saves/qwen3-vl-8b-fundus/lora/stage1_en_cot_gentle_calibrated/checkpoint-20
```

The Stage 2 baseline must:

1. Predict DR Grade 0-4.
2. Bind the grade to visible lesion evidence.
3. Preserve Stage 1 MA/HE/EX/SE capability.
4. Never invent IRMA or NV evidence from a grade label.
5. Preserve a six-lesion output schema while returning `unknown` for
   unsupported IRMA/NV findings.

## Why the previous L4 dataset should not be reused unchanged

The existing L4 v5 dataset is useful lineage, but it is not the recommended new
baseline:

- It applies quadrant-based ETDRS 4-2-1 reasoning even though the current
  coordinate and quadrant layer has known reliability problems.
- It contains `possible_by_grade_template` NV states, which can teach the model
  to reverse-infer NV from a Grade 4 label.
- About 32% of its selected training samples are evidence-limited.
- It starts from an older six-lesion adapter rather than the selected Stage 1
  balanced checkpoint.

The new baseline therefore removes quadrant-dependent reasoning and separates
visible findings from label-supervised grading.

## Baseline training structure

Use one Stage 2 training run containing two task families:

| Task family | Purpose | Share |
|---|---|---:|
| L4 evidence-bound DR grading | Learn Grade 0-4 and ordered decision logic | 80% |
| Stage 1 four-lesion replay | Prevent MA/HE/EX/SE catastrophic forgetting | 20% |

This is still a single-stage Stage 2 baseline. Replay rows are mixed into the
same SFT run rather than trained as a separate calibration stage.

## 1. Proposed data distribution

### Stage 2 grading rows

Target: 5,600 image-level grading rows.

| Grade | Target rows | Share | Sampling intent |
|---|---:|---:|---|
| Grade 0 | 1,100 | 19.6% | Reliable no-DR negatives; reduce lesion hallucination |
| Grade 1 | 900 | 16.1% | Mild/MA-only boundary; include strong MA when available |
| Grade 2 | 1,400 | 25.0% | Preserve the broad and clinically common moderate class |
| Grade 3 | 1,100 | 19.6% | Severe NPDR boundary; prioritize heavy trusted-lesion burden |
| Grade 4 | 1,100 | 19.6% | PDR label learning without fabricating NV |
| **Total** | **5,600** | **100%** | |

This distribution is moderately balanced rather than perfectly uniform. Grade
2 retains a larger share because it is heterogeneous and is the main confusion
hub between mild and severe disease.

### Evidence quality rules

Sampling priority for every grade:

1. Strong masks and trusted direct lesion evidence.
2. Clean validated evidence.
3. Label-supervised evidence-limited examples.

Target caps:

| Category | Target |
|---|---:|
| Direct/validated evidence rows | at least 65% |
| Evidence-limited rows | at most 35% overall |
| Duplicate image IDs across train/dev/test | 0 |
| Grade-template-derived positive IRMA/NV | 0 |
| Quadrant-dependent 4-2-1 reasoning rows | 0 |

Grade 4 will naturally contain many evidence-limited examples because trusted
NV capability is not ready. These rows must say `NV=unknown`; they may still
supervise `grade=4`, but must set `evidence_limited=true`.

The 35% ceiling is intentional. A lower ceiling is not realistic before NV and
the Grade 3 boundary evidence are improved: all unsupported Grade 4 rows and a
substantial part of Grade 3 must remain explicitly evidence-limited.

### Stage 1 replay rows

Target: 1,400 lesion-level rows mixed into Stage 2 training.

| Lesion | Present | Absent | Total |
|---|---:|---:|---:|
| MA | 175 | 175 | 350 |
| HE | 175 | 175 | 350 |
| EX | 175 | 175 | 350 |
| SE | 175 | 175 | 350 |
| **Total** | **700** | **700** | **1,400** |

Replay rows should be sampled from trusted S0/S1 data, include hard negatives,
and exclude Stage 1 Gold Dev/Gold Test. IRMA/NV replay is excluded from this
baseline.

### Final mixed training set

| Component | Rows | Share |
|---|---:|---:|
| Stage 2 grading | 5,600 | 80% |
| Stage 1 replay | 1,400 | 20% |
| **Total** | **7,000** | **100%** |

### Split policy

- Split by canonical image ID before generating prompts.
- Keep all variants and tasks from one image in exactly one split.
- Use a Stage 2 Dev set for checkpoint selection.
- Keep a locked internal Stage 2 Test set for one final evaluation.
- Messidor-2 and FunBench remain external tests and must not select
  checkpoints.
- Stage 1 Gold Dev is a checkpoint guardrail.
- Stage 1 Gold Test is evaluated once after final Stage 2 selection.

Recommended Stage 2 evaluation sizes:

| Set | Suggested size | Distribution |
|---|---:|---|
| Stage 2 Dev | 500 | 100 per grade |
| Stage 2 locked Test | 1,000 | 200 per grade |
| External tests | Existing full sets | Report native distributions |

## 2. CoT design

### Core principle

The model should first report visible facts, then make an ordered grade
decision. The grade must never be used to manufacture a lesion finding.

Do not use long free-form CoT. Use a short, auditable rationale with a fixed
schema:

1. `[Findings]`: six lesion fields.
2. `[Reasoning]`: brief ordered clinical decision.
3. `[Result]`: strict JSON.

### Six-field lesion policy

| Lesion | Stage 2 baseline behavior |
|---|---|
| MA | Trusted Stage 1 evidence; use `present`, `absent`, or `unknown` |
| HE | Trusted Stage 1 evidence; use `present`, `absent`, or `unknown` |
| EX | Trusted Stage 1 evidence; use `present`, `absent`, or `unknown` |
| SE | Trusted Stage 1 evidence; use `present`, `absent`, or `unknown` |
| IRMA | Default `unknown`; never infer from Grade 3 |
| NV | Default `unknown`; never infer from Grade 4 |

`absent` means reliable negative evidence. Missing or unsupported evidence must
be `unknown`, not `absent`.

### Ordered reasoning path

The compact reasoning path is:

1. State whether direct proliferative evidence is available. For the baseline,
   NV is usually `unknown`.
2. Assess trusted NPDR lesion burden from MA/HE/EX/SE.
3. Identify the likely severity boundary.
4. Output the supervised grade.
5. Mark `evidence_limited=true` when visible trusted lesions do not fully
   explain the grade.

Quadrant counts and the ETDRS 4-2-1 rule are excluded until location evidence
is independently validated.

### Recommended output example

```text
[Findings]
MA=present
HE=present
EX=present
SE=absent
IRMA=unknown
NV=unknown

[Reasoning]
No reliable direct proliferative evidence is available. Multiple trusted NPDR
lesions are visible with a heavy burden, supporting advanced NPDR. The
supervised grade is Grade 3 and the rationale is evidence-supported.

[Result]
{"grade":3,"lesions":{"MA":"present","HE":"present","EX":"present","SE":"absent","IRMA":"unknown","NV":"unknown"},"burden":"heavy","evidence_limited":false}
```

Grade 4 evidence-limited example:

```text
[Findings]
MA=present
HE=present
EX=present
SE=present
IRMA=unknown
NV=unknown

[Reasoning]
Trusted NPDR lesions are visible, but direct NV evidence is unavailable. The
supervised label is Grade 4; therefore this conclusion is evidence-limited and
must not be interpreted as a confirmed NV finding.

[Result]
{"grade":4,"lesions":{"MA":"present","HE":"present","EX":"present","SE":"present","IRMA":"unknown","NV":"unknown"},"burden":"heavy","evidence_limited":true}
```

### Training configuration recommendation

For the first baseline:

- Initialize from Stage 1 balanced checkpoint-20.
- Keep vision tower and multimodal projector frozen.
- Continue language-model LoRA only.
- Use one epoch with checkpoint evaluation during training.
- Start with learning rate `1e-6` to `2e-6`.
- Effective batch size: 16.
- Use the existing accelerated preprocessing/cache path.

This is the lowest-risk baseline for preserving Stage 1. Projector or vision
unfreezing should be a later ablation after this baseline is established.

## 3. Final evaluation metrics

### Primary Stage 2 grading metrics

| Metric | Role |
|---|---|
| Quadratic Weighted Kappa (QWK) | Primary ordinal grading metric |
| Macro F1 across Grade 0-4 | Prevent majority-grade dominance |
| Per-grade F1, recall, precision | Detect collapse of Grade 1, 3, or 4 |
| Accuracy | Standard exact-grade result |
| Mean Absolute Error (MAE) | Penalize distance between predicted and true grade |
| Within-one-grade accuracy | Measure clinically near-miss predictions |
| Confusion matrix | Show systematic under/over-grading |

### Clinical binary metrics

Report sensitivity, specificity, F1, and AUROC when probabilities are
available:

| Binary task | Definition |
|---|---|
| Any DR | Grade 0 vs Grade 1-4 |
| Referable DR | Grade 0-1 vs Grade 2-4 |
| Severe-or-worse | Grade 0-2 vs Grade 3-4 |
| PDR | Grade 0-3 vs Grade 4 |

### Evidence and output-quality metrics

| Metric | Purpose |
|---|---|
| JSON parse success | Verify deployable output |
| Grade-field consistency | Text grade and JSON grade agree |
| Evidence-limited classification accuracy | Verify the model admits unsupported rationales |
| IRMA hallucination rate | Must not invent IRMA from Grade 3 |
| NV hallucination rate | Must not invent NV from Grade 4 |
| Lesion-grade contradiction rate | Detect impossible or unsupported rationales |

### Stage 1 preservation metrics

Evaluate the Stage 2 candidate on Stage 1 Gold Dev during checkpoint selection:

| Guardrail | Recommended requirement |
|---|---:|
| Four-lesion Macro F1 | no more than 0.02 below checkpoint-20 |
| HE F1 | no more than 0.02 below checkpoint-20 |
| EX F1 | no more than 0.02 below checkpoint-20 |
| MA F1 | no more than 0.03 below checkpoint-20 |
| SE F1 | no more than 0.03 below checkpoint-20 |
| JSON parse rate | at least 0.99 |
| IRMA/NV hallucination rate | below 0.01 |

After final checkpoint selection, evaluate Stage 1 Gold Test exactly once.

## Checkpoint selection rule

Select a Stage 2 checkpoint in this order:

1. Reject candidates that fail any Stage 1 preservation or output-quality
   guardrail.
2. Among remaining candidates, select highest Stage 2 Dev QWK.
3. Break close ties using Macro F1.
4. Break remaining ties using Grade 3/4 recall and lower MAE.
5. Run locked Stage 2 Test, external tests, and Stage 1 Gold Test only after
   selection.

## Baseline success criteria

The first Stage 2 baseline is successful when:

- Stage 2 Dev/Test QWK and Macro F1 clearly exceed the base model.
- Stage 1 MA/HE/EX/SE guardrails pass.
- Grade 3 and Grade 4 do not collapse to Grade 2.
- NV/IRMA hallucination remains near zero.
- JSON parse rate is at least 99%.

This baseline establishes the reference point for later experiments involving
projector unfreezing, vision LoRA continuation, improved IRMA/NV supervision,
or validated quadrant/4-2-1 reasoning.
## Baseline Scale and Concrete CoT Revision

This section supersedes the earlier 7,000-row conservative first-run budget.
For the strongest first baseline, use approximately 9,000 task rows:

| Component | Rows | Notes |
|---|---:|---|
| Unique-image Stage 2 grading | 7,200 | Do not repeat grading images merely to increase row count |
| Stage 1 four-lesion replay | 1,800 | 450 rows per lesion; 225 present + 225 absent |
| **Total** | **9,000** | Single mixed SFT run |

The 7,200 grading-row target is derived from the actual local pool:

- The trusted facts layer has 9,493 L4-usable records before splitting.
- Existing L4 v5 used 6,629 unique training images.
- Existing selected L4 v3 used 6,511 rows but only 6,391 unique images.
- The old 16,000-row Stage 2 dataset increased task rows through replay and repeated images; it did not contain 16,000 independent grading images.
- Grade 1, Grade 3, and Grade 4 are the practical bottlenecks. Additional duplicated prompts do not replace new visual evidence.

Recommended unique-image grading distribution:

| Grade | Rows | Design reason |
|---|---:|---|
| Grade 0 | 1,500 | Control false positives and lesion hallucination |
| Grade 1 | 950 | Use nearly all clean mild cases; prioritize direct MA evidence |
| Grade 2 | 2,000 | Keep the heterogeneous central class adequately represented |
| Grade 3 | 1,300 | Use available severe cases; prioritize heavy trusted-lesion burden |
| Grade 4 | 1,450 | Learn image-level PDR grade without inventing NV |
| **Total** | **7,200** | |

### Concrete CoT generation logic

Stage 2 grading rows and Stage 1 replay rows use different output contracts.
Grading rows use the complete findings-to-grade format. Replay rows retain the
original Stage 1 single-lesion format so that replay preserves the exact Stage
1 capability rather than merely mentioning lesions inside grading answers.

Each grading row uses one of five fixed clinical branches:

| Branch | Visible-fact requirement | Reasoning target | Evidence alignment |
|---|---|---|---|
| G0 no DR | No trusted MA/HE/EX/SE positives | No reliable DR lesion evidence | `supported` |
| G1 mild | Direct MA only, or MA-only label template | Isolated MA-level finding supports mild NPDR | `supported` with direct MA; otherwise `limited` |
| G2 moderate | Trusted HE/EX/SE or multiple NPDR lesions, without a heavy severe pattern | More than isolated MA, but no reliable severe/PDR boundary | usually `supported` |
| G3 severe | Heavy trusted MA/HE/EX/SE burden; IRMA remains unknown | Advanced NPDR appearance without claiming IRMA | `supported` or `limited` according to burden |
| G4 PDR | Image-level Grade 4 label; NV remains unknown | Grade 4 classification while admitting direct NV cannot be confirmed | `limited` in this baseline |

Allowed lesion states are only `present`, `absent`, and `unknown`. An optional
`light`, `moderate`, or `heavy` qualifier is included only when supported by the
validated facts layer. Exact counts, coordinates, quadrants, and probabilities
are excluded from the baseline.

The model should not say "the supervised label is Grade 4" at inference. It
should instead expose whether the visible findings support the final grade via
`evidence_alignment=supported|limited`.

G2 example:

```text
[Findings]
MA=present | HE=present(moderate) | EX=present(light) | SE=absent | IRMA=unknown | NV=unknown

[Reasoning]
The image shows more than isolated MA, with trusted hemorrhage and exudate
evidence. The visible burden supports moderate NPDR, while no reliable severe
or proliferative evidence is available.

[Result]
{"grade":2,"burden":"moderate","evidence_alignment":"supported","lesions":{"MA":"present","HE":"present","EX":"present","SE":"absent","IRMA":"unknown","NV":"unknown"}}
```

G3 example:

```text
[Findings]
MA=present | HE=present(heavy) | EX=present(heavy) | SE=present | IRMA=unknown | NV=unknown

[Reasoning]
Multiple trusted NPDR lesions are present with a heavy overall burden. This
supports an advanced non-proliferative grade. IRMA and NV cannot be confirmed
and are not used as invented evidence.

[Result]
{"grade":3,"burden":"heavy","evidence_alignment":"supported","lesions":{"MA":"present","HE":"present","EX":"present","SE":"present","IRMA":"unknown","NV":"unknown"}}
```

G4 evidence-limited example:

```text
[Findings]
MA=present | HE=present(heavy) | EX=present | SE=present | IRMA=unknown | NV=unknown

[Reasoning]
The overall image is classified as Grade 4, but direct proliferative evidence
such as NV cannot be reliably confirmed. The grade-to-evidence alignment is
therefore limited; NV remains unknown.

[Result]
{"grade":4,"burden":"heavy","evidence_alignment":"limited","lesions":{"MA":"present","HE":"present","EX":"present","SE":"present","IRMA":"unknown","NV":"unknown"}}
```

Use 3-5 controlled wording variants per branch to avoid learning a single
sentence template. Keep the clinical logic and JSON schema unchanged across
variants.
