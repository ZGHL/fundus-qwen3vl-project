# Upload Configuration: Hugging Face + Cloudflare R2

This project uses two external artifact stores:

1. Hugging Face private Dataset repo for generated annotations, CoT/SFT JSONL files, evaluation JSONL files, and stats.
2. Cloudflare R2 bucket for raw/processed fundus images, large prediction dumps, and optional adapters.

## Current Frozen Snapshots

The current 2026-05-21 uploads are frozen restore snapshots:

```text
HF Dataset: Guohou/fundusAnnotationsV1/fundus_generated_annotations_20260521.tar.gz
R2: fundusv1/images/fundus_image_dataset_20260521.tar
```

Keep them for exact restoration. Do not use this tar-based pattern as the default for new iterative updates.

## Hugging Face Dataset: Future Native Layout

Repository:

```text
Guohou/fundusAnnotationsV1
```

Recommended future layout:

```text
data/fundus_validated/
data/annotation/
data/annotation_v4/
eval/
stats/
reports/
```

Upload new generated CoT/SFT/evaluation files directly as JSONL, JSONL.GZ, or normal directories:

```bash
hf upload Guohou/fundusAnnotationsV1 data/annotation_v4/ data/annotation_v4/   --repo-type dataset   --commit-message "Add English L3 CoT annotation files"

hf upload Guohou/fundusAnnotationsV1 data/fundus_validated/ data/fundus_validated/   --repo-type dataset   --commit-message "Update validated fundus evidence"
```

Download on a new server:

```bash
mkdir -p /workspace/artifacts/fundus_annotations
hf download Guohou/fundusAnnotationsV1   --repo-type dataset   --local-dir /workspace/artifacts/fundus_annotations
```

If restoring the frozen snapshot:

```bash
cd /workspace/artifacts/fundus_annotations
sha256sum -c fundus_generated_annotations_20260521.tar.gz.sha256
cd /workspace/LLaMA-Factory
tar -xzf /workspace/artifacts/fundus_annotations/fundus_generated_annotations_20260521.tar.gz
```

## R2: Future Directory Layout

Bucket:

```text
fundusv1
```

Recommended layout:

```text
images/FGADR/
images/DDR-dataset/
images/idrid/
images/messidor-2/
images/cropped/
images/aptos_processed/
labels/DR_grading.csv
labels/messidor_data.csv
labels/idrid_old/idrid_labels.csv
adapters/<experiment_name>/
predictions/<experiment_name>/
```

### R2 Credentials

Create an R2 API token with read/write access to the bucket. Configure credentials locally or on the VM. Do not commit credentials.

```bash
export R2_ENDPOINT=https://4ff11044d39e473b1c3f56367f45fe71.r2.cloudflarestorage.com
export R2_BUCKET=fundusv1
export AWS_DEFAULT_REGION=auto
export AWS_ACCESS_KEY_ID=<your_r2_access_key_id>
export AWS_SECRET_ACCESS_KEY=<your_r2_secret_access_key>
```

For AWS CLI, the endpoint is the account endpoint without the bucket suffix.

### Upload Images With rclone

```bash
rclone sync /workspace/LLaMA-Factory/data/FGADR/ r2:fundusv1/images/FGADR/ -P --transfers 32
rclone sync /workspace/LLaMA-Factory/data/DDR-dataset/ r2:fundusv1/images/DDR-dataset/ -P --transfers 32
rclone sync /workspace/LLaMA-Factory/data/idrid/ r2:fundusv1/images/idrid/ -P --transfers 32
rclone sync /workspace/LLaMA-Factory/data/messidor-2/ r2:fundusv1/images/messidor-2/ -P --transfers 32
rclone sync /workspace/LLaMA-Factory/data/cropped/ r2:fundusv1/images/cropped/ -P --transfers 32
rclone sync /workspace/LLaMA-Factory/data/processed_images/ r2:fundusv1/images/aptos_processed/ -P --transfers 32
```

### Partial Restore On A VM

```bash
mkdir -p /workspace/LLaMA-Factory/data
rclone copy r2:fundusv1/images/idrid/ /workspace/LLaMA-Factory/data/idrid/ -P
rclone copy r2:fundusv1/images/FGADR/ /workspace/LLaMA-Factory/data/FGADR/ -P
```

Use the old tar only for full cold restore:

```bash
aws --endpoint-url "$R2_ENDPOINT" s3 cp   s3://$R2_BUCKET/images/fundus_image_dataset_20260521.tar   /workspace/artifacts/fundus_images/fundus_image_dataset_20260521.tar
```

## Upload Training Results Back From A VM

Push only small reproducibility records to GitHub:

```bash
cd /workspace/fundus-qwen3vl-project
git pull
git add configs manifests reports scripts
git commit -m "Record <experiment_name> results"
git push
```

Upload adapters to R2:

```bash
EXP=<experiment_name>
rclone copy /workspace/LLaMA-Factory/saves/qwen3-vl-8b-fundus/lora/$EXP/   r2:fundusv1/adapters/$EXP/ -P --transfers 16
```

Upload large prediction dumps to R2 or Hugging Face Dataset:

```bash
EXP=<experiment_name>
rclone copy /workspace/LLaMA-Factory/saves/qwen3-vl-8b-fundus/lora/$EXP/generated_predictions.jsonl   r2:fundusv1/predictions/$EXP/ -P
```

Then record URI, checksum, train config, eval config, and metric path in the experiment manifest.

## License Recommendation

For the Hugging Face generated-annotation dataset, use:

```text
license: other
visibility: private
```

Reason: this package contains generated annotations/CoT derived from multiple fundus datasets whose original licenses and redistribution rules may differ. Do not mark it as CC-BY, MIT, or Apache unless every upstream dataset explicitly permits that redistribution.

A safe dataset-card note is:

```markdown
License: Other / Research-only. This repository contains derived annotation files for internal research use. It does not include raw fundus images. Users must obtain the original image datasets under their respective licenses before use.
```
