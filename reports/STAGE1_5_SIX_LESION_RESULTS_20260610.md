# Stage 1.5 Six-Lesion Limited Experiment Results

Date: 2026-06-10

## Decision

Preserve this experiment as a valid specificity-oriented research branch, but
do not replace the current Stage 1 baseline with it.

The run successfully increased specificity and balanced accuracy on Gold Dev.
It also reduced recall and F1 for several common lesions, so no checkpoint
passed the common-lesion preservation guardrails. The pipeline therefore did
not evaluate Gold Test, IRMA Locked Test, or NV Locked Test.

The current Stage 2 initialization remains:

```text
/workspace/LLaMA-Factory/saves/qwen3-vl-8b-fundus/lora/stage1_en_cot_gentle_calibrated/checkpoint-20
```

## Preserved Stage 1.5 checkpoints

| Checkpoint | Research role | Gold Dev Macro F1 | Recall | Specificity | Balanced accuracy |
|---|---|---:|---:|---:|---:|
| checkpoint-40 | Highest Stage 1.5 Macro F1 | 0.6991 | 0.6899 | 0.5115 | 0.6007 |
| checkpoint-80 | Highest Stage 1.5 specificity | 0.6685 | 0.6378 | 0.5918 | 0.6148 |
| checkpoint-160 | Highest Stage 1.5 balanced accuracy | 0.6979 | 0.6914 | 0.5383 | 0.6149 |

The full resumable checkpoints include adapter weights, optimizer state,
scheduler state, RNG state, and trainer state.

## Valid Dev-to-Dev comparison

All rows below were evaluated on the same 596-example Gold Dev set.

| Model | Macro F1 | Recall | Specificity | Balanced accuracy |
|---|---:|---:|---:|---:|
| Current Stage 1 baseline checkpoint-20 | **0.7314** | **0.7615** | 0.3950 | 0.5782 |
| Stage 1.5 checkpoint-40 | 0.6991 | 0.6899 | 0.5115 | 0.6007 |
| Stage 1.5 checkpoint-80 | 0.6685 | 0.6378 | **0.5918** | 0.6148 |
| Stage 1.5 checkpoint-160 | 0.6979 | 0.6914 | 0.5383 | **0.6149** |

Checkpoint-160 compared with the current baseline:

| Lesion | Baseline F1 | Checkpoint-160 F1 | Baseline specificity | Checkpoint-160 specificity |
|---|---:|---:|---:|---:|
| MA | **0.6606** | 0.6146 | 0.0588 | **0.4118** |
| HE | **0.8131** | 0.7653 | 0.6111 | **0.7778** |
| EX | **0.6776** | 0.6742 | 0.3544 | **0.3924** |
| SE | **0.7742** | 0.7374 | 0.5556 | **0.5714** |

This confirms that the hard-negative six-lesion adaptation changed the
decision policy in the intended direction. It reduced false positives, most
strongly for MA and HE, but over-corrected sensitivity.

## Evaluation set map

### Internal validation

- 3,773 examples from mixed trusted and weaker evidence tiers.
- Used for routine training visibility and diagnostics.
- Not used as the final Stage 1 comparison.

### Gold Dev

- 596 lesion-level examples from the trusted `S0` DDR mask subset.
- Uses DDR held-out images that are disjoint from training.
- Used for checkpoint selection and development decisions.
- It is an internal held-out development benchmark, not an external-domain
  benchmark and not a final test result.
- Its class balance is uneven, especially MA with only 17 negatives, so MA
  specificity has high variance.

| Lesion | Present | Absent | Total |
|---|---:|---:|---:|
| MA | 132 | 17 | 149 |
| HE | 113 | 36 | 149 |
| EX | 70 | 79 | 149 |
| SE | 86 | 63 | 149 |

### Gold Test

- 900 lesion-level examples from a separate trusted `S0` DDR mask subset.
- Uses held-out DDR images disjoint from training and Gold Dev.
- Locked until one checkpoint is selected on Gold Dev.
- This is the final internal Stage 1 four-lesion benchmark.

| Lesion | Present | Absent | Total |
|---|---:|---:|---:|
| MA | 124 | 101 | 225 |
| HE | 194 | 31 | 225 |
| EX | 171 | 54 | 225 |
| SE | 42 | 183 | 225 |

### Weak-negative challenge

- 1,930 negative examples from weaker evidence tiers.
- Used to diagnose false-positive behavior and robustness.
- Not a replacement for Gold Test because labels are less direct.

### IRMA Locked Test

- 100 independent FGADR examples: 20 positive and 80 negative.
- Direct mask evidence for positives.
- Used only after common-lesion checkpoint selection.

### NV Locked Test

- 88 independent FGADR examples: 8 positive and 80 negative.
- Direct mask evidence for positives.
- The positive count is too small for a stable production conclusion.

## Why Gold Dev F1 appears higher

The apparent increase is caused primarily by evaluation-set composition, not a
sudden model improvement.

Gold Dev and Gold Test have different lesion prevalence. For example:

- Gold Dev MA contains 132 positives and only 17 negatives.
- Gold Test MA contains 124 positives and 101 negatives.
- Gold Dev SE contains 86 positives and 63 negatives.
- Gold Test SE contains 42 positives and 183 negatives.

A positive-biased model can therefore obtain a higher F1 on Gold Dev while
performing poorly on specificity. Results must only be compared when the model,
scoring rule, and evaluation set are all the same.

The trusted final result for the current Stage 1 baseline remains its Gold Test
Macro F1 of `0.6595`. The Stage 1.5 values in this report are Gold Dev
development metrics and must not be presented as final Gold Test metrics.

## External evaluation status

Gold Dev and Gold Test are internally controlled held-out splits derived from
DDR. They are not external-domain evaluations.

Older project reports contain grading-oriented evaluations such as Messidor-2,
but those do not directly validate the Stage 1 lesion-detection contract.
A true external Stage 1 lesion benchmark still requires an independent dataset
with compatible lesion-level ground truth and no overlap with training sources.

## Reproducibility

Training data:

- 5,680 rows.
- Balanced positive/negative rows per lesion.
- 2,060 hard-negative rows.
- IRMA: 300 positive and 300 negative rows.
- NV: 140 positive and 140 negative rows.
- Gold Dev, Gold Test, internal validation, and locked-test inputs were not
  used in Stage 1.5 training.

Relevant files:

- `configs/train/stage1_5_six_lesion_limited.yaml`
- `scripts/fundus_v4/build_stage1_5_six_lesion.py`
- `scripts/run_stage1_5_six_lesion.sh`
- `docs/STAGE1_5_SIX_LESION_LIMITED_EXPERIMENT_20260609.md`
- `data/annotation_v4/fundus_stage1_5_six_lesion_stats.json`

