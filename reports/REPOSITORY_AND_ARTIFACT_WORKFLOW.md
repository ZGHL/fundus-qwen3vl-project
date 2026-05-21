# GitHub / Hugging Face / R2 Submission and Merge Workflow

## Purpose

This document records how the fundus Qwen3-VL project is split across GitHub, Hugging Face, and Cloudflare R2, what has already been submitted, and how later experiments should be merged back into the main project line.

The project should keep a clear separation between lightweight reproducible code and large research artifacts:

- GitHub stores code, configuration, patches, manifests, experiment registry, and metric summaries.
- Hugging Face Dataset stores generated annotations, CoT/SFT JSONL files, evaluation JSONL files, and statistics.
- Cloudflare R2 stores raw and processed fundus image packages.
- Model checkpoints and LoRA adapters should not be committed to GitHub. They should be stored in Hugging Face Model repositories or object storage when they need to be preserved.

## Current GitHub Submission

Repository:

```text
https://github.com/ZGHL/fundus-qwen3vl-project.git
```

Current pushed branch:

```text
main
```

Initial commit:

```text
42fd54a Initialize fundus Qwen3-VL project
```

The GitHub repository currently contains the lightweight project skeleton needed to reproduce and continue the work on a new GPU server:

```text
configs/      training and evaluation YAML configs
scripts/      fundus data builders, evaluators, RetSAM utilities, and monitor scripts
patches/      local LLaMA-Factory modifications needed for this project
manifests/    dataset, artifact, model, and experiment registries
reports/      project inventory, upload records, dataset summary, and metric summaries
docs/         cloud setup and artifact transfer documentation
```

The repository intentionally does not contain:

```text
data/
models/
saves/
outputs/
logs/
large JSONL/CSV artifacts
raw fundus images
base model weights
LoRA or merged model checkpoints
RetSAM raw output directories
```

This makes the GitHub repository suitable for version control, cloud transfer, review, and collaboration without pushing private or very large assets.

## Current Hugging Face Submission

Dataset repository:

```text
https://huggingface.co/datasets/Guohou/fundusAnnotationsV1
```

Uploaded package:

```text
fundus_generated_annotations_20260521.tar.gz
```

Uploaded companion files:

```text
ANNOTATION_PACKAGE_CONTENTS.txt
fundus_generated_annotations.filelist.txt
fundus_generated_annotations_20260521.tar.gz.sha256
README.md
LICENSE
```

This package is the generated annotation package. It contains cleaned RetSAM/strong-label evidence, generated CoT/SFT JSONL files, evaluation JSONL files, and statistics. It does not contain raw fundus images.

The package is intended to restore the annotation side of the project on a new server without rebuilding all intermediate files from scratch.

## Current R2 Submission

Cloudflare R2 bucket:

```text
fundusv1
```

Uploaded keys:

```text
images/IMAGE_PACKAGE_CONTENTS.txt
images/fundus_image_dataset.filelist.txt
images/fundus_image_dataset_20260521.tar.sha256
images/fundus_image_dataset_20260521.tar
```

Large image package size:

```text
67500830720 bytes
```

This R2 package stores the image side of the project, including the fundus image roots and processed images needed by the annotation JSONL files. It is intentionally kept outside GitHub.

## Why The Three-Way Split Is Necessary

GitHub should remain small and human-reviewable. It is the source of truth for experiment definitions, code, patches, evaluation scripts, and metric summaries.

Hugging Face Dataset is better suited for generated text artifacts such as CoT JSONL, SFT data, holdout files, and statistics. These files are versioned research artifacts but can become too large or too frequently regenerated for normal GitHub commits.

R2 is better suited for large image packages. The current image tar is about 63 GiB, which is inappropriate for GitHub and inconvenient for ordinary Git LFS management.

This split also makes new GPU server setup simpler: clone GitHub for code, download Hugging Face for annotations, download or mount R2 for images, then run the configured training jobs.

## Restore Workflow On A New GPU Server

A new GPU server should start from a fresh LLaMA-Factory checkout and this project repository:

```bash
cd /workspace
git clone https://github.com/hiyouga/LLaMA-Factory.git
git clone https://github.com/ZGHL/fundus-qwen3vl-project.git
cd /workspace/fundus-qwen3vl-project
bash scripts/setup/apply_llamafactory_patches.sh /workspace/LLaMA-Factory
bash scripts/setup/sync_project_files.sh /workspace/LLaMA-Factory
```

Then restore generated annotations from Hugging Face:

```bash
cd /workspace
mkdir -p artifacts/fundus_generated_annotations
hf download Guohou/fundusAnnotationsV1 --repo-type dataset --local-dir artifacts/fundus_generated_annotations
cd /workspace/LLaMA-Factory
tar -xzf /workspace/artifacts/fundus_generated_annotations/fundus_generated_annotations_20260521.tar.gz
```

Then restore images from R2 or mount an equivalent image directory. After downloading the R2 tar package:

```bash
cd /workspace/artifacts/fundus_images
sha256sum -c fundus_image_dataset_20260521.tar.sha256
cd /workspace/LLaMA-Factory
tar -xf /workspace/artifacts/fundus_images/fundus_image_dataset_20260521.tar
```

After restoration, the LLaMA-Factory tree should contain the expected `data/`, `scripts/`, `examples/train_lora/`, and patched source files needed by the project configs.

## How New CoT Files Should Be Handled

New CoT files should not be committed directly to GitHub when they are large JSONL artifacts. The preferred workflow is:

1. Generate the new CoT/SFT JSONL files inside the LLaMA-Factory workspace.
2. Validate them with statistics and a small sample inspection.
3. Upload the generated JSONL package or tar archive to Hugging Face Dataset.
4. Add or update a manifest entry in GitHub describing the artifact name, source script, sample count, class balance, lesion distribution, and expected image roots.
5. Add or update the training YAML in GitHub so a GPU server can run the experiment by pulling code and downloading the referenced artifact.

For small representative examples, metric summaries, or schema documentation, GitHub is appropriate. For full generated datasets, Hugging Face is the correct target.

## How Training Results Should Be Merged Back

After a GPU VM finishes an experiment, only lightweight results should be pushed back to GitHub:

```text
configs/train/*.yaml       if a new or corrected training config was used
configs/eval/*.yaml        if a new or corrected evaluation config was used
manifests/experiments/*.yaml or experiment_registry.yaml
manifests/datasets/*.json  dataset statistics and balance summaries
reports/metrics/*.json     final evaluation metrics
reports/*.md               experiment summary and interpretation
scripts/*.py               only if evaluation or data generation logic changed
```

The following should not be pushed to GitHub:

```text
saves/                     LoRA adapters and checkpoints
models/                    base models or merged models
outputs/                   raw RetSAM or inference dumps
logs/                      long training logs
large generated_predictions.jsonl files
large SFT JSONL files
```

If a checkpoint is important, upload it to a Hugging Face Model repository or R2 and record the location in `manifests/models/model_manifest.yaml` and `manifests/experiments/experiment_registry.yaml`.

If generated predictions are needed for audit, compress and upload them to Hugging Face Dataset or R2, then record the link and checksum in GitHub.

## Merge Policy For Future Experiments

The main project line should use semantic experiment names rather than temporary version numbers such as v7 or v8.

Current canonical names:

```text
l3_zh_cot_baseline_step1_targeted_calib
l3_zh_cot_baseline_step2_six_lesion
l4_zh_cot_pipeline_baseline
l3_en_cot_incomplete_step2_only_do_not_use_as_main
```

The next valid English L3 experiment should be named by role, for example:

```text
l3_en_cot_baseline_step1_targeted_calib
l3_en_cot_baseline_step2_six_lesion
```

A result should be considered mergeable into the mainline only when it includes:

```text
training config
evaluation config
generated data manifest
sample count and lesion distribution
final metric summary
short interpretation report
checkpoint or adapter storage location, if preserved
```

A failed or incomplete experiment can still be recorded, but it should be marked clearly as incomplete or not used as the main baseline.

## Recommended Branch Workflow

For routine updates, use short feature branches:

```bash
git checkout -b exp/l3-en-cot-baseline
```

After generating configs, manifests, and metric summaries:

```bash
git status -sb
git add configs manifests reports scripts
git commit -m "Add English L3 CoT baseline experiment records"
git push -u origin exp/l3-en-cot-baseline
```

After review, merge the branch into `main` on GitHub. If the work is a local-only continuation and there is no collaboration conflict, it is also acceptable to commit directly to `main`, but experiment branches make the history cleaner.

## Immediate Next Mainline Action

The next mainline experiment should not reuse the incomplete English CoT run as the official baseline. It should rebuild the full L3 path in English CoT format:

1. Generate English CoT equivalent of the targeted calibration step.
2. Train from the same starting checkpoint and data distribution as the Chinese L3 baseline step 1.
3. Generate English CoT equivalent of the six-lesion calibration step.
4. Continue training from the English step 1 checkpoint.
5. Evaluate on the same L3 holdout sets used by the Chinese baseline.
6. Record metrics, data distribution, and interpretation in GitHub.
7. Store large generated CoT files on Hugging Face and any preserved adapters outside GitHub.

This keeps the comparison fair: the only intended difference between the Chinese and English L3 baselines should be the CoT language and format, not the training path or data distribution.
