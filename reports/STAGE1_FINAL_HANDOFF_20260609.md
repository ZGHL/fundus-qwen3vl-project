# Stage 1 Final Handoff

Date: 2026-06-09

## Executive decision

Use the balanced calibration candidate below as the Stage 1 adapter for Stage 2:

```text
/workspace/LLaMA-Factory/saves/qwen3-vl-8b-fundus/lora/stage1_en_cot_gentle_calibrated/checkpoint-20
```

Its `adapter_model.safetensors` SHA256 is:

```text
9d35140057356c4fd07bca095b969ac4e80d86faf65134d38c1c37a8130b4976
```

Keep the original Adapter 1 as the conservative fallback:

```text
/workspace/LLaMA-Factory/saves/qwen3-vl-8b-fundus/lora/stage1_en_cot
```

Its `adapter_model.safetensors` SHA256 is:

```text
086bca5057c4e021cbf3f18ec9bb99b1df5a70c7cc3e8f98c1a90c77ec10dea3
```

The balanced candidate is recommended because it preserves the strong HE/EX
capability of Adapter 1 while improving SE, specificity, and balanced accuracy.
MA falls slightly on the held-out Gold Test, so Stage 2 must retain lesion-level
guardrails.

## Final Gold Test comparison

The Base Model result uses relaxed semantic scoring because its strict score is
zero due to output-schema mismatch, not because every lesion prediction is
wrong.

| Model | Macro F1 | Recall | Specificity | Balanced Acc. | MA F1 | HE F1 | EX F1 | SE F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen3-VL 8B Base Model | 0.3237 | 0.2532 | 0.8011 | 0.5272 | 0.0000 | 0.5128 | 0.6360 | 0.1458 |
| Adapter 1 | 0.6524 | 0.8758 | 0.1793 | 0.5276 | 0.5249 | 0.9008 | 0.8680 | 0.3158 |
| Balanced checkpoint-20 | **0.6595** | 0.8164 | **0.3846** | **0.6005** | 0.5161 | 0.8973 | 0.8652 | **0.3594** |

Balanced checkpoint-20 versus Adapter 1:

| Metric | Delta |
|---|---:|
| Macro F1 | +0.0071 |
| Specificity | +0.2053 |
| Balanced accuracy | +0.0729 |
| MA F1 | -0.0088 |
| HE F1 | -0.0035 |
| EX F1 | -0.0029 |
| SE F1 | +0.0437 |

Formatting reliability:

| Model | JSON parse rate | Target consistency |
|---|---:|---:|
| Adapter 1 | 1.0000 | 1.0000 |
| Balanced checkpoint-20 | 0.9989 | 0.9922 |

## Stage 1 experiment design

### Adapter 1: broad lesion capability

Config: `configs/train/stage1_en_cot.yaml`

- Base model: Qwen3-VL 8B.
- LoRA: rank 16, alpha 32, dropout 0.05, all linear targets.
- Vision tower LoRA enabled; projector frozen.
- Image budget: approximately 768-equivalent pixels.
- Effective batch: 16 (`per_device_train_batch_size=1`,
  `gradient_accumulation_steps=16`).
- Learning rate: `6e-6`.
- One epoch, 791 optimizer steps, 12,650 training examples.

This run established strong HE/EX recall and meaningful MA/SE capability, but
was too positive-biased and therefore had low specificity.

### Balanced calibration: short conservative correction

Config: `configs/train/stage1_en_cot_gentle_calibration.yaml`

- Starts from Adapter 1.
- Vision tower and projector frozen.
- Continues language-model LoRA only.
- Effective batch: 16 (`per_device_train_batch_size=2`,
  `gradient_accumulation_steps=8`).
- Learning rate: `2e-7`.
- Maximum 40 steps, checkpoint every 10 steps.
- Selected checkpoint-20, corresponding to global step 20 and epoch 0.128.

The candidate was selected only on Gold Dev. Gold Test was evaluated once
after candidate selection and was not used for tuning.

Checkpoint-20 passed the selection guardrails:

- Macro F1 improves.
- HE and EX remain within preservation limits.
- MA and SE do not decrease on Gold Dev.
- Output-format quality remains acceptable.

## Data distribution

### Original Stage 1 training set

Total: 12,650 lesion-level samples.

| Lesion | Present | Absent | Total | Positive rate |
|---|---:|---:|---:|---:|
| HE | 2,000 | 1,200 | 3,200 | 62.5% |
| EX | 2,000 | 1,200 | 3,200 | 62.5% |
| MA | 1,200 | 450 | 1,650 | 72.7% |
| SE | 1,600 | 1,600 | 3,200 | 50.0% |
| IRMA | 400 | 500 | 900 | 44.4% |
| NV | 140 | 360 | 500 | 28.0% |
| **Total** | **7,340** | **5,310** | **12,650** | **58.0%** |

The original set intentionally provided broad positive coverage. Its main
weakness was insufficient hard-negative pressure for MA and SE, which explains
the high recall and low specificity of Adapter 1.

### Internal validation

Total: 3,773 samples. It was used for routine training visibility, not for the
final locked comparison.

### Gold Dev

Total: 596 examples, all from the trusted S0 DDR mask subset.

| Lesion | Present | Absent | Total |
|---|---:|---:|---:|
| MA | 132 | 17 | 149 |
| HE | 113 | 36 | 149 |
| EX | 70 | 79 | 149 |
| SE | 86 | 63 | 149 |

Gold Dev was used for checkpoint selection. Its class balance differs from
Gold Test, especially for MA and SE, so preservation rules were required rather
than optimizing Macro F1 alone.

### Gold Test

Total: 900 examples, all from the trusted S0 DDR mask subset.

| Lesion | Present | Absent | Total |
|---|---:|---:|---:|
| MA | 124 | 101 | 225 |
| HE | 194 | 31 | 225 |
| EX | 171 | 54 | 225 |
| SE | 42 | 183 | 225 |

Gold Test remained locked until the balanced candidate was selected.

### Gentle calibration set

Total: 2,500 examples. It contains only trusted S0/S1 data and excludes Gold
Dev, Gold Test, and locked IRMA/NV evaluation examples.

| Lesion | Present | Absent | Total | Hard negatives |
|---|---:|---:|---:|---:|
| MA | 450 | 150 | 600 | 150 |
| HE | 500 | 100 | 600 | 100 |
| EX | 500 | 100 | 600 | 95 |
| SE | 450 | 250 | 700 | 250 |

Tier composition:

| Lesion | Positive S0 | Positive S1 | Negative S0 | Negative S1 |
|---|---:|---:|---:|---:|
| MA | 423 | 27 | 64 | 86 |
| HE | 374 | 126 | 85 | 15 |
| EX | 326 | 174 | 34 | 66 |
| SE | 156 | 294 | 204 | 46 |

This set adds hard negatives while retaining enough positive anchors to avoid
the lesion collapse observed in more aggressive calibration attempts.

## What did not work

| Experiment | Result | Decision |
|---|---|---|
| Original aggressive calibration | Gold Test Macro F1 0.5786; specificity 0.5413; MA F1 collapsed to 0.3378 | Rejected |
| V2 single-stage projector run from base | Macro F1 0.5555; MA 0.6263; HE 0.4945; EX 0.7619; SE 0.3392 | Rejected because HE/EX regressed heavily |
| Targeted hard-negative calibration checkpoints 40/80 | Became over-conservative and failed Gold Dev preservation guards | Rejected |
| Gentle positive-anchored early checkpoint search | Preserved HE/EX and improved SE/specificity | Selected checkpoint-20 |

The experiments show that lesion balance is governed by sampling, label
reliability, decision threshold behavior, and optimization duration. It is not
random, although small-sample evaluation variance can make individual lesions
look unstable.

## IRMA and NV status

IRMA and NV are not yet reliable enough to be trusted Stage 2 supervision
targets.

| Lesion | Positive | Negative | F1 | Recall | Specificity |
|---|---:|---:|---:|---:|---:|
| IRMA locked | 20 | 80 | 0.3256 | 0.7000 | 0.3500 |
| NV locked | 8 | 80 | 0.2078 | 1.0000 | 0.2375 |

Stage 2 should retain a six-field output schema for forward compatibility, but
only MA, HE, EX, and SE should be treated as trusted Stage 1 evidence. IRMA and
NV should be represented as unknown/abstain unless a later focused experiment
passes locked evaluation guardrails.

## Stage 2 handoff rules

1. Initialize Stage 2 from balanced checkpoint-20.
2. Keep Adapter 1 available as the rollback baseline.
3. Track MA, HE, EX, and SE separately; do not select by Macro F1 alone.
4. Require HE/EX preservation and explicit MA/SE floors.
5. Keep a locked Stage 2 test set and evaluate it only after candidate
   selection.
6. Preserve the six-field schema, but mask or abstain on IRMA/NV supervision.
7. Compare every Stage 2 candidate against both Adapter 1 and balanced
   checkpoint-20.

## Reproducibility files

Primary metrics:

- `reports/metrics/stage1_overnight_base_relaxed.json`
- `reports/metrics/stage1_overnight_base_strict.json`
- `reports/metrics/stage1_overnight_adapter1.json`
- `reports/metrics/stage1_overnight_balanced.json`
- `reports/metrics/stage1_overnight_selection.json`
- `reports/metrics/stage1_stage2_recommended_adapter.json`

Supporting reports:

- `reports/stage1_adapter1_vs_base_model_20260609.md`
- `reports/stage1_overnight_results_20260609.md`

Dataset statistics:

- `/workspace/LLaMA-Factory/data/annotation_v4/fundus_stage1_en_cot_stats.json`
- `/workspace/LLaMA-Factory/data/annotation_v4/fundus_stage1_en_cot_gentle_calibration_stats.json`

## Artifact backup

The Stage 2 handoff archive contains:

- Full resumable balanced `checkpoint-20`, including optimizer/trainer state.
- Lightweight final Adapter 1 fallback files.
- Final reports, metrics, configs, data-distribution statistics, and restore
  manifest.

R2 object:

```text
s3://fundusv1/models/stage1/stage1_stage2_handoff_20260609.tar.zst
```

Checksum object:

```text
s3://fundusv1/models/stage1/stage1_stage2_handoff_20260609.tar.zst.sha256
```

Archive SHA256 and size are recorded in
`manifests/models/stage1_stage2_handoff_20260609.yaml`.

The previous complete Adapter 1 training snapshot remains available in R2:

```text
s3://fundusv1/models/stage1/stage1_en_cot_20260608_full.tar.gz
```

## Restore outline

```bash
aws s3 cp \
  s3://fundusv1/models/stage1/stage1_stage2_handoff_20260609.tar.zst \
  /workspace/artifacts/stage1_stage2_handoff_20260609.tar.zst \
  --endpoint-url "$R2_ENDPOINT"

sha256sum -c stage1_stage2_handoff_20260609.tar.zst.sha256
tar --zstd -xf stage1_stage2_handoff_20260609.tar.zst
```

Credentials and endpoints are intentionally excluded from this repository.
