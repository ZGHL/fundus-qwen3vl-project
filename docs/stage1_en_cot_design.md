# Stage1 English Single-Lesion CoT Design

This document records the current Stage1 training design. The main line is no longer L2-to-L4. It is a two-stage pipeline:

1. Stage1: single-lesion decoupled perception.
2. Stage2: disease diagnosis using Stage1 lesion evidence.

Stage1 trains the model to inspect one target lesion at a time and produce a compact English reasoning trace plus structured JSON. It must not assign diabetic retinopathy grade, diagnose disease stage, or report non-target lesions.

## Target Lesions

Stage1 covers six lesion targets:

| Abbreviation | Full name |
|---|---|
| MA | microaneurysm |
| HE | retinal hemorrhage |
| EX | hard exudate |
| SE | soft exudate or cotton-wool spot |
| IRMA | intraretinal microvascular abnormality |
| NV | neovascularization |

The prompt explicitly includes the full lesion name, visual definition, and important exclusions. The model-visible prompt should not rely on abbreviations alone.

## Prompt Placement

Lesion definitions are placed in the `system` message because they define the task contract for that sample:

- role: single-lesion fundus perception specialist
- target lesion full name and abbreviation
- typical visual evidence
- important exclusions/confounders
- prohibition on DR grade or stage diagnosis

The `user` message repeats the target lesion and asks for direct visual evidence only. The assistant CoT then applies the definition to the image.

## Model-Visible CoT Structure

The assistant answer uses a stable English format:

```text
[Target Evidence]
...

[Confounder Assessment]
...

[Attribute Summary]
...

[Conclusion]
...

[Structured Output]
{"task":"stage1_single_lesion_perception", ...}
```

The structured output is the scoring target. The free-text reasoning is deliberately short and grounded in visible evidence.

## Hidden Metadata Rule

Evidence provenance is stored only in `meta`. It is not exposed to the model in prompts or assistant text.

The following fields must remain hidden from model-visible content:

- evidence tier (`S0` to `S4`)
- source names such as RetSAM, mask, grade rule, cleaning rule
- dataset name or split
- label origin
- DR grade

## Evidence Tiers

Training prioritizes stronger evidence and fills remaining capacity from weaker evidence:

| Tier | Meaning | Use |
|---|---|---|
| S0 | direct pixel mask | preferred positive/negative evidence when available |
| S1 | explicit lesion label | strong image-level lesion annotation |
| S2 | validated RetSAM or RetSAM negative | pseudo evidence after validation/cleaning |
| S3 | cleaning-rule negative | weak negative only |
| S4 | grade-rule weak negative | weakest negative only, mainly grade 0 absence assumptions |

Positive labels do not come from S3 or S4. MA positives are limited to S0/S1. Rare lesion positives for IRMA/NV use FGADR mask evidence when available.

## Attribute Policy

Only coarse attributes are included, and only when available:

- `count_bucket`: `single`, `few`, `many`
- `area_bucket`: `small`, `medium`, `large`
- `distribution`: `isolated`, `multifocal`, `scattered`, `diffuse`, or source-derived distribution
- `location`: source-derived coarse location when available

Exact counts and exact mask areas are not model-visible. Missing attributes are omitted rather than emitted as `unknown`. Omission keeps the CoT focused on reliable evidence and avoids teaching the model to generate filler fields.

## Negative Sample Rule

Negative samples should follow the strongest available evidence first:

1. S0 direct mask absence.
2. S1 explicit absence or lesion-only dataset negatives.
3. S2 validated RetSAM negatives.
4. S3 cleaning-rule negatives.
5. S4 grade-0 weak negatives.

Grade-0 weak negatives can be used, but they are intentionally low priority. The prompt still says decisions must be based on visible image evidence and must not infer from DR grade.

## Current Train Distribution

Current generated train rows: 12,650.

| Lesion | Positive | Negative |
|---|---:|---:|
| MA | 1,200 | 450 |
| HE | 2,000 | 1,200 |
| EX | 2,000 | 1,200 |
| SE | 1,600 | 1,600 |
| IRMA | 400 | 500 |
| NV | 140 | 360 |

Current train evidence tier counts:

| Tier | Rows |
|---|---:|
| S0 | 2,193 |
| S1 | 4,982 |
| S2 | 4,597 |
| S3 | 520 |
| S4 | 358 |

## Evaluation Design

Stage1 is evaluated as single-lesion binary perception from generated structured JSON.

Primary evaluation:

- DDR Main-4 gold dev/test for MA, HE, EX, SE using S0 mask-derived labels.
- Metrics by lesion: precision, recall, specificity, F1, balanced accuracy.
- Main-4 macro metrics across MA/HE/EX/SE.

Auxiliary evaluation:

- weak-negative challenge set for false-positive stress testing.
- IRMA locked set for rare lesion recall/specificity.
- NV locked set for rare lesion recall/specificity.

Scoring also checks structured-output quality:

- JSON parse success
- target-lesion consistency
- no grade/stage output rate

## Files

Dataset builder:

```text
scripts/fundus_v4/build_stage1_en_cot.py
```

Scorer:

```text
scripts/fundus_v4/score_stage1_en_cot.py
```

Main training config:

```text
configs/train/stage1_en_cot.yaml
```

Ablation configs:

```text
configs/train/stage1_en_cot_language_only_ablation.yaml
configs/train/stage1_en_cot_projector_ablation.yaml
```

Gold test prediction config:

```text
configs/eval/stage1_en_cot_gold_test.yaml
```
