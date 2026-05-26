# Decoupled vs Six-Lesion Mix Lesion-Perception Experiment

Date: 2026-05-26

This report summarizes the current comparison between the decoupled lesion-perception SFT and the six-lesion mix lesion-perception SFT for Qwen3-VL-8B-Instruct + LLaMA-Factory LoRA training.

## Objective

The main question is whether decoupled lesion perception, where the model learns one lesion target at a time, is more stable and effective than a traditional mixed lesion-perception setup, where the model outputs all six lesion states in one response.

Only the lesion-perception task is compared here. L2 grading and L4 grading are not included.

## Compared Methods

### Decoupled Lesion Perception

Each training example asks the model to inspect exactly one lesion type. The target lesion is explicit in the system/user prompt.

Example prompt:

```text
You are a fundus image analyst. Inspect ONLY for EX (hard exudate). This is a single-lesion perception task: do NOT output a final DR grade and do NOT combine other lesions. First describe visible morphology and location, then judge whether the target lesion is present.

<image>
Examine this fundus image for hard exudate (EX). Output exactly four sections: [Lesion Existence and Evidence Judgment], [Basic Morphological and Location Features], [Decision Notes for This Single-Lesion Task], and [Structured Output].
```

Example CoT/label:

```text
[Lesion Existence and Evidence Judgment]
hard exudate is present with strong direct evidence from retsam_validated.

[Basic Morphological and Location Features]
The target finding is described as bright yellow-white deposits with relatively sharp borders. In this sample it is annotated as count=some, area=medium, located in the superior-temporal quadrant only.

[Decision Notes for This Single-Lesion Task]
This is a target-lesion positive example for learning visible morphology. Focus on sharply bordered yellow-white lipid deposits; distinguish them from fluffy cotton-wool spots. No final DR grade is assigned in this single-lesion perception task.

[Structured Output]
{"task":"lesion_perception_EX","lesion":"EX","present":true,"evidence_state":"present","strength":"strong","count":"some","area":"medium","location":"in the superior-temporal quadrant only","source":"retsam_validated"}
```

### Six-Lesion Mix Lesion Perception

Each training example asks the model to inspect all six lesion categories and output all six lesion keys in one structured response:

- HE: retinal hemorrhage
- EX: hard exudate
- MA: microaneurysm
- SE: soft exudate / cotton-wool spot
- IRMA: intraretinal microvascular abnormality
- NV: neovascularization

Example prompt:

```text
You are a fundus image analyst. This is a lesion-perception task. Audit the six fundus lesion categories separately, compare similar lesion patterns when needed, and do not output a final DR grade.

<image>
Lesion reference definitions:
- HE (retinal hemorrhage): dark red dot, blot, or flame-like hemorrhagic lesions; generally larger or more irregular than MA.
- EX (hard exudate): bright yellow-white deposits with relatively sharp borders; distinguish from fluffy SE and imaging glare.
- MA (microaneurysm): tiny round red dots, usually smaller than hemorrhages.
- SE (soft exudate / cotton-wool spot): gray-white fluffy cotton-wool patches with soft borders; distinguish from sharper hard exudates.
- IRMA (intraretinal microvascular abnormality): irregular tortuous intraretinal vascular channels; not preretinal new vessels.
- NV (neovascularization): abnormal fine new vessels on the disc or elsewhere; distinguish from IRMA and ordinary retinal vessels.

Identify the lesion status for these six fundus lesion categories: HE (retinal hemorrhage), EX (hard exudate), MA (microaneurysm), SE (soft exudate / cotton-wool spot), IRMA (intraretinal microvascular abnormality), NV (neovascularization).
For every image, audit all six lesion keys: HE, EX, MA, SE, IRMA, NV. Do not omit any of these six keys, do not add any other lesion key, and do not output a final DR grade.
Output exactly four sections: [Global Image Review], [Lesion-by-Lesion Audit], [Cross-Lesion Distinction], and [Structured Output].
In [Structured Output], return JSON with task="joint_lesion_perception" and a lesions object containing exactly these keys: "HE", "EX", "MA", "SE", "IRMA", "NV". For each key, set present to true or false and include evidence_state, strength, count, area, location, and source.
```

Example CoT/label:

```text
[Global Image Review]
The image is audited for six fundus lesion categories. Positive targets: EX, MA. Absent targets: HE, SE, IRMA, NV.

[Lesion-by-Lesion Audit]
- HE (retinal hemorrhage): absent. Evidence source=cleaning_rule; no reliable dark red dot, blot, or flame-like hemorrhagic lesions pattern is retained.
- EX (hard exudate): present. Evidence source=strong_mask; visual pattern=bright yellow-white deposits with relatively sharp borders; count=unknown; area=unknown; location=in the superior-nasal quadrant only.
- MA (microaneurysm): present. Evidence source=strong_mask; visual pattern=tiny round red dots, usually smaller than hemorrhages; count=unknown; area=unknown; location=posterior retina.
- SE (soft exudate / cotton-wool spot): absent. Evidence source=cleaning_rule; no reliable gray-white fluffy cotton-wool patches with soft borders pattern is retained.
- IRMA (intraretinal microvascular abnormality): absent. Evidence source=strong_mask; no reliable irregular tortuous intraretinal vascular channels pattern is retained.
- NV (neovascularization): absent. Evidence source=cleaning_rule; no reliable abnormal fine new vessels on the disc or elsewhere pattern is retained.

[Cross-Lesion Distinction]
Keep red lesions separated from yellow-white exudates; distinguish tiny MA from larger HE; separate fluffy SE from sharper EX; and do not confuse IRMA with NV or ordinary vessels.

[Structured Output]
{"task":"joint_lesion_perception","lesions":{"HE":{"present":false,"evidence_state":"absent","strength":"absent","count":"none","area":"none","location":null,"source":"cleaning_rule"},"EX":{"present":true,"evidence_state":"present","strength":"strong","count":null,"area":null,"location":"in the superior-nasal quadrant only","source":"strong_mask"},"MA":{"present":true,"evidence_state":"present","strength":"strong","count":null,"area":null,"location":"posterior retina","source":"strong_mask"},"SE":{"present":false,"evidence_state":"absent","strength":"absent","count":"none","area":"none","location":null,"source":"cleaning_rule"},"IRMA":{"present":false,"evidence_state":"absent","strength":"absent","count":null,"area":null,"location":null,"source":"strong_mask"},"NV":{"present":false,"evidence_state":"absent","strength":"absent","count":"none","area":"none","location":null,"source":"cleaning_rule"}}}
```

Important labeling note: the six-lesion mix dataset expands each image to six lesion decisions. Lesions not retained as positive in the cleaned pool are filled as absent with `source="cleaning_rule"`. This makes the mix task match the intuitive "identify all six lesions" format, but it may introduce pseudo-negative noise for lesions without full image-level annotation.

## Data

### Decoupled Train Data

Dataset: `fundus_lesion_perception_en_cot_full_train`

Rows: 12,408 single-lesion examples.

Approximate train distribution:

| Lesion | Positive | Negative |
|---|---:|---:|
| HE | 1,862 | 931 |
| EX | 2,652 | 1,326 |
| MA | 600 | 300 |
| SE | 1,240 | 1,240 |
| IRMA | 603 | 754 |
| NV | 300 | 600 |

### Six-Lesion Mix Train Data

Dataset: `fundus_l3_joint_mix_full_train`

Rows: 5,458 image-level examples.

Each row contains six lesion decisions, for a total of 32,748 lesion decisions.

Train distribution:

| Lesion | Positive | Negative |
|---|---:|---:|
| HE | 1,862 | 3,596 |
| EX | 2,652 | 2,806 |
| MA | 600 | 4,858 |
| SE | 1,240 | 4,218 |
| IRMA | 603 | 4,855 |
| NV | 100 | 5,358 |

The mix distribution is much more negative-heavy because every image is forced to emit all six lesion states.

## Training Configuration

Both experiments used the same base model and LoRA/SFT setup.

| Parameter | Decoupled | Six-Lesion Mix |
|---|---|---|
| Base model | `./models/Qwen3-VL-8B-Instruct` | same |
| Framework | LLaMA-Factory | same |
| Stage | SFT | same |
| Finetuning | LoRA | same |
| LoRA rank | 16 | same |
| LoRA alpha | 32 | same |
| LoRA dropout | 0.05 | same |
| LoRA target | all | same |
| Template | `qwen3_vl_nothink` | same |
| cutoff_len | 2304 | same |
| image_max_pixels | 262144 | same |
| image_min_pixels | 65536 | same |
| batch size | 1 | same |
| grad accumulation | 16 | same |
| learning rate | 6e-6 | same |
| epochs | 1.0 | same |
| scheduler | cosine | same |
| warmup | 0.03 | same |
| bf16 | true | same |
| gradient checkpointing | true | same |
| optimizer | adamw_torch | same |

Output directories:

- Decoupled: `saves/qwen3-vl-8b-fundus/lora/lesion_perception_en_cot_full`
- Six-lesion mix: `saves/qwen3-vl-8b-fundus/lora/l3_joint_mix_full`

## Training Results

| Method | Train Rows / Images | Lesion Decisions | Steps | Train Loss | Runtime | Samples/s |
|---|---:|---:|---:|---:|---:|---:|
| Decoupled | 12,408 rows | 12,408 | 776 | 0.3439 | 8,141.7 s | 1.524 |
| Six-lesion Mix | 5,458 images | 32,748 | 342 | 0.2364 | 5,876.4 s | 0.929 |

The six-lesion mix loss is lower, but this did not translate into useful positive-lesion recall. The low loss is consistent with the strong negative skew created by forcing all six keys for every image.

## Evaluation Setup

The same evaluation image pools were used where applicable, but the scoring granularity differs:

- Decoupled prediction rows represent one lesion decision per row.
- Mix prediction rows represent one image-level response with six lesion decisions.

Evaluation splits:

- `val_subset`
- `balanced`
- `irma_locked`
- `nv_locked`

Scoring focuses on:

- JSON parse success.
- Target lesion consistency.
- Macro recall.
- Macro F1.
- Balanced accuracy.
- Rare-lesion recall/F1 for IRMA/NV.

`no_grade_output_rate` means the response did not output a DR grade, so higher is better for this lesion-perception task.

## Main Evaluation Results

| Eval Set | Method | Rows | Decisions | JSON Parse | Target Consistency | Macro Recall | Macro F1 | Balanced Acc |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| val_subset | Decoupled | 1,230 | 1,230 | 98.9% | 98.5% | 58.4% | 0.5699 | 0.5646 |
| val_subset | Six-lesion Mix | 748 | 4,488 | 14.4% | 14.4% | 0.0% | N/A | 0.4998 |
| balanced | Decoupled | 200 | 200 | 99.5% | 99.0% | 57.4% | 0.6243 | 0.5718 |
| balanced | Six-lesion Mix | 177 | 1,062 | 13.6% | 13.6% | 0.0% | N/A | 0.5000 |
| IRMA locked | Decoupled | 100 | 100 | 99.0% | 96.0% | 0.0% | N/A | 0.5000 |
| IRMA locked | Six-lesion Mix | 100 | 600 | 19.0% | 19.0% | 0.0% | N/A | 0.5000 |
| NV locked | Decoupled | 105 | 105 | 100.0% | 100.0% | 0.0% | N/A | 0.4950 |
| NV locked | Six-lesion Mix | 105 | 630 | 20.0% | 20.0% | 0.0% | N/A | 0.5000 |

## Mix Per-Lesion Behavior

The six-lesion mix model collapsed to nearly all-negative outputs.

On `val_subset`:

| Lesion | Positive Labels | Negative Labels | TP | FP | FN | TN | Recall | Specificity |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| HE | 199 | 549 | 0 | 1 | 199 | 548 | 0.0% | 99.8% |
| EX | 195 | 553 | 0 | 0 | 195 | 553 | 0.0% | 100.0% |
| MA | 186 | 562 | 0 | 0 | 186 | 562 | 0.0% | 100.0% |
| SE | 61 | 687 | 0 | 0 | 61 | 687 | 0.0% | 100.0% |
| IRMA | 20 | 728 | 0 | 0 | 20 | 728 | 0.0% | 100.0% |
| NV | 0 | 748 | 0 | 0 | 0 | 748 | N/A | 100.0% |

On `balanced`, HE/EX/MA/SE/IRMA all had recall 0.0%, while NV had no positive labels in that split after six-key expansion.

## Interpretation

The decoupled model is clearly more stable in the current setup:

1. It follows the requested output format reliably.
2. It preserves target-lesion consistency.
3. It maintains non-zero positive recall on common lesions.
4. It achieves meaningful macro F1 on `val_subset` and `balanced`.

The six-lesion mix model failed in two ways:

1. Structural failure: JSON parse success stayed around 13-20%.
2. Detection failure: positive recall collapsed to 0.0% across evaluated positive lesions.

The low mix training loss is not evidence of better learning. It is likely driven by the many negative labels introduced by the six-key output format. Because each image emits all six lesion states, the label distribution becomes strongly negative-heavy, especially for MA, IRMA, and NV.

## Fairness Notes and Limitations

This is a stricter and more natural mix baseline than the earlier controlled mix draft because every image asks for all six lesions. However, the current cleaned pool does not provide complete gold-standard image-level labels for every lesion on every image. Therefore, missing non-positive lesions were filled as absent via `cleaning_rule`.

This design matches the intuitive clinical task:

> identify which of the six lesion categories are present in the image.

But it may also penalize the mix approach through pseudo-negative noise. Even with that caveat, the practical result is useful: under the available cleaned data and current Qwen3-VL LoRA setup, the direct six-lesion mix formulation is much less stable than decoupled lesion perception.

## Current Conclusion

For the current RetSAM-cleaned fundus dataset and Qwen3-VL-8B LoRA/SFT setup, decoupled lesion-perception training is substantially stronger than six-lesion mix training.

The strongest evidence is not only the metric gap, but the failure mode:

- Decoupled: high format compliance and meaningful common-lesion recall.
- Mix: low JSON compliance and all-negative collapse despite low training loss.

This supports using decoupled lesion perception as the main lesion-learning strategy before integrating lesion evidence into downstream grading stages.

## Artifact Paths

Data/config/code:

- Decoupled train config: `/workspace/LLaMA-Factory/examples/train_lora/lesion_perception_en_cot_full.yaml`
- Mix train config: `/workspace/LLaMA-Factory/examples/train_lora/l3_joint_mix_full.yaml`
- Mix data builder: `/workspace/fundus-qwen3vl-project/scripts/fundus_v4/build_l3_joint_mix_full.py`
- Mix pipeline: `/workspace/fundus-qwen3vl-project/scripts/run_l3_joint_mix_pipeline.sh`
- Mix scorer: `/workspace/fundus-qwen3vl-project/scripts/fundus/score_l3_joint_mix_predictions.py`

Model outputs:

- Decoupled adapter: `/workspace/LLaMA-Factory/saves/qwen3-vl-8b-fundus/lora/lesion_perception_en_cot_full`
- Mix adapter: `/workspace/LLaMA-Factory/saves/qwen3-vl-8b-fundus/lora/l3_joint_mix_full`

Score files:

- Decoupled balanced: `/workspace/LLaMA-Factory/saves/qwen3-vl-8b-fundus/lora/lesion_perception_en_cot_predict_balanced/lesion_perception_score.json`
- Decoupled val subset: `/workspace/LLaMA-Factory/saves/qwen3-vl-8b-fundus/lora/lesion_perception_en_cot_predict_val_subset/lesion_perception_score.json`
- Decoupled IRMA locked: `/workspace/LLaMA-Factory/saves/qwen3-vl-8b-fundus/lora/lesion_perception_en_cot_predict_irma_locked/lesion_perception_score.json`
- Decoupled NV locked: `/workspace/LLaMA-Factory/saves/qwen3-vl-8b-fundus/lora/lesion_perception_en_cot_predict_nv_locked/lesion_perception_score.json`
- Mix balanced: `/workspace/LLaMA-Factory/saves/qwen3-vl-8b-fundus/lora/l3_joint_mix_full_predict_balanced/l3_joint_mix_score.json`
- Mix val subset: `/workspace/LLaMA-Factory/saves/qwen3-vl-8b-fundus/lora/l3_joint_mix_full_predict_val_subset/l3_joint_mix_score.json`
- Mix IRMA locked: `/workspace/LLaMA-Factory/saves/qwen3-vl-8b-fundus/lora/l3_joint_mix_full_predict_irma_locked/l3_joint_mix_score.json`
- Mix NV locked: `/workspace/LLaMA-Factory/saves/qwen3-vl-8b-fundus/lora/l3_joint_mix_full_predict_nv_locked/l3_joint_mix_score.json`
