# Stage 2 Structured CoT Specification

Date: 2026-06-09

## Decision

The Stage 2 baseline should not use long free-form chain-of-thought.

Use a compact structured reasoning target with three blocks:

1. Trusted lesion evidence.
2. Four ordinal grading gates.
3. Final grade and evidence support status.

This design is intended to improve grading QWK and Macro F1 while limiting
hallucinated lesion explanations.

## Why this design

Medical visual CoT can hurt accuracy when the first visual observation is
wrong, because later text expands the original perception error. Structured
medical CoT work instead emphasizes clinically ordered intermediate steps and
visual grounding. DR grading is also an ordinal problem: confusing Grade 2
with Grade 3 is less severe than confusing Grade 0 with Grade 4, which is why
QWK is a primary metric.

The training target should therefore supervise useful intermediate decisions,
not prose length.

## Final output contract

```text
[Evidence]
MA=present|absent|unknown
HE=present|absent|unknown
EX=present|absent|unknown
SE=present|absent|unknown
IRMA=unknown
NV=unknown
burden=none|light|moderate|heavy

[Ordinal Gates]
any_dr=yes|no
referable_dr=yes|no
severe_or_worse=yes|no
pdr=yes|no

[Result]
{"grade":0-4,"evidence_support":"direct|partial|unverified","lesions":{...},"burden":"...","gates":{...}}
```

The gates map deterministically to grades:

| Gate pattern | Grade |
|---|---:|
| `any_dr=no` | 0 |
| `any_dr=yes`, `referable_dr=no` | 1 |
| `referable_dr=yes`, `severe_or_worse=no` | 2 |
| `severe_or_worse=yes`, `pdr=no` | 3 |
| `pdr=yes` | 4 |

All generated gates must be monotonic:

```text
pdr=yes -> severe_or_worse=yes -> referable_dr=yes -> any_dr=yes
```

This turns the textual target into an ordinal decomposition aligned with QWK,
while retaining a single generative output.

## Evidence support values

| Value | Meaning |
|---|---|
| `direct` | Trusted visible evidence supports the final grade |
| `partial` | Trusted evidence supports disease but not the exact grade boundary |
| `unverified` | Image-level grade is supervised, but the current trusted lesion fields cannot verify the boundary |

Do not use `evidence_limited=true` as the only explanation. The three-level
support field is more useful for training, deployment, and stratified
evaluation.

## Grade-specific generation rules

### Grade 0

- Trusted MA/HE/EX/SE are absent.
- All ordinal gates are `no`.
- Evidence support is `direct`.

```text
[Evidence]
MA=absent | HE=absent | EX=absent | SE=absent | IRMA=unknown | NV=unknown | burden=none

[Ordinal Gates]
any_dr=no | referable_dr=no | severe_or_worse=no | pdr=no

[Result]
{"grade":0,"evidence_support":"direct",...}
```

### Grade 1

Direct version:

- Trusted isolated MA is present.
- Other trusted lesions are absent.
- Only `any_dr=yes`.
- Evidence support is `direct`.

Template-only version:

- MA cannot be visually verified.
- Only `any_dr=yes`.
- Evidence support is `unverified`.
- Do not write `MA=present`; use `MA=unknown`.

### Grade 2

- More than isolated MA is visible, usually HE, EX, or SE.
- `any_dr=yes`, `referable_dr=yes`.
- `severe_or_worse=no`, `pdr=no`.
- Evidence support is usually `direct`.

### Grade 3

Direct/partial version:

- Trusted MA/HE/EX/SE burden is heavy enough to support advanced NPDR.
- `severe_or_worse=yes`, `pdr=no`.
- IRMA remains `unknown`.
- Evidence support is `direct` when the trusted burden is clearly heavy;
  otherwise `partial`.

Do not infer IRMA from Grade 3.

### Grade 4

The baseline does not have trustworthy NV supervision. Therefore:

- `pdr=yes` is supervised from the official image-level Grade 4 label.
- NV remains `unknown`.
- Evidence support is `unverified` unless future trusted NV/PDR evidence exists.
- Do not generate a natural-language claim that a specific proliferative
  lesion was observed.

```text
[Evidence]
MA=present | HE=present | EX=present | SE=present | IRMA=unknown | NV=unknown | burden=heavy

[Ordinal Gates]
any_dr=yes | referable_dr=yes | severe_or_worse=yes | pdr=yes

[Result]
{"grade":4,"evidence_support":"unverified",...}
```

This is intentionally a structured target rather than a narrative rationale.
It allows the model to learn the Grade 4 image-level boundary without teaching
it a fabricated NV explanation.

## CoT text policy

Do not include a long `[Reasoning]` paragraph in the baseline target.

Reasons:

- Free-form text adds many easy language tokens but little grading signal.
- It can teach label-conditioned explanations rather than visual reasoning.
- It amplifies incorrect lesion observations.
- It increases training and inference cost.
- It is difficult to score automatically.

If a human-readable explanation is required, generate one deterministic
sentence from the structured fields after inference. For example:

```text
Grade 3 is predicted because the trusted NPDR lesion burden is heavy; direct
proliferative evidence is not verified.
```

This sentence is a renderer output, not a learned CoT target.

## Training task mixture

Recommended strongest baseline:

| Task | Rows | Share |
|---|---:|---:|
| Structured ordinal grading | 7,200 | 80% |
| Original Stage 1 single-lesion replay | 1,800 | 20% |
| **Total** | **9,000** | **100%** |

The Stage 1 replay rows retain their original prompt and output contract.

Within the 7,200 grading rows:

| Subset | Target share |
|---|---:|
| Direct evidence support | 55-65% |
| Partial evidence support | 10-15% |
| Unverified image-level supervision | 20-30% |

Do not force rows into these percentages. The final distribution must be
reported after applying the data-audit rules.

## Data admission rules

A grading row is admitted only when:

1. The image is readable and exists locally.
2. Grade is an official image-level G0-G4 label.
3. Canonical image ID does not cross train/dev/test.
4. The image does not overlap Stage 1 Gold Dev/Test.
5. MA/HE/EX/SE fields come only from the trusted facts layer.
6. IRMA/NV are `unknown` without trusted direct evidence.
7. Grade does not generate a lesion-positive label.
8. Structured evidence matches the validated source fields.
9. No quadrant or 4-2-1 claim is generated.
10. Strong grade/evidence conflicts are routed to a separate review set.

Additional CoT-specific admission checks:

11. Gate sequence is monotonic.
12. Gate sequence maps exactly to the official grade.
13. `direct` support is used only when trusted evidence supports the boundary.
14. G4 without trusted NV/PDR evidence is always `unverified`.
15. G3 never creates IRMA-positive supervision.

## Loss and optimization implication

With standard SFT, later answer tokens can dominate because the JSON schema is
easy to memorize. Keep the target short and place the evidence and gates before
the final grade so the model must generate intermediate supervision first.

For the baseline, use standard SFT for reproducibility. A later ablation can
add token weighting or auxiliary classification losses, but those should not
be mixed into the first reference result.

## Evaluation directly enabled by this format

In addition to QWK, Macro F1, per-grade F1, accuracy, and MAE, score:

- Gate accuracy for each clinical threshold.
- Gate monotonicity violation rate.
- Grade/gate consistency rate.
- MA/HE/EX/SE evidence F1.
- IRMA/NV hallucination rate.
- Evidence-support accuracy.
- Grade metrics stratified by `direct`, `partial`, and `unverified`.

## Required baseline ablation

To verify that structured reasoning actually helps rather than merely adding
tokens, run:

| Arm | Output target |
|---|---|
| A: Direct grade | Strict JSON with grade only |
| B: Structured ordinal CoT | Evidence + ordinal gates + strict JSON |

Use the same images, split, optimizer steps, and Stage 1 replay rows. Select
both arms with the same Stage 1 preservation guardrails.

The structured CoT is accepted as the Stage 2 mainline only if it improves Dev
QWK or Macro F1 without worsening locked-test performance or Stage 1
preservation.
