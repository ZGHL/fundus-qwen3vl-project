# Upload Configuration: Hugging Face + Cloudflare R2

This project uses two external artifact stores:

1. Hugging Face private Dataset repo for compact generated annotations/CoT JSONL.
2. Cloudflare R2 bucket for the full image dataset archive.

Generated local artifacts:

```text
/workspace/artifacts/fundus_transfer_20260521/fundus_generated_annotations_20260521.tar.gz
/workspace/artifacts/fundus_transfer_20260521/fundus_generated_annotations_20260521.tar.gz.sha256
/workspace/artifacts/fundus_transfer_20260521/fundus_generated_annotations.filelist.txt

/workspace/artifacts/fundus_transfer_20260521/fundus_image_dataset_20260521.tar
/workspace/artifacts/fundus_transfer_20260521/fundus_image_dataset_20260521.tar.sha256
/workspace/artifacts/fundus_transfer_20260521/fundus_image_dataset.filelist.txt
```

## Hugging Face Dataset Repo

Recommended repo:

```text
<HF_USERNAME_OR_ORG>/fundus-qwen3vl-generated-annotations
```

Recommended visibility: private.

Recommended files:

```text
fundus_generated_annotations_20260521.tar.gz
fundus_generated_annotations_20260521.tar.gz.sha256
fundus_generated_annotations.filelist.txt
README.md
```

Suggested dataset card text:

```markdown
# Fundus Qwen3-VL Generated Annotations

Compact generated annotation package for Qwen3-VL fundus fine-tuning.

This repo contains generated data assets only:
- cleaned RetSAM/strong-label evidence
- ShareGPT/SFT JSONL files
- CoT training samples
- L3/L4 evaluation JSONL files
- RetSAM cleaning/statistics files

It does not contain raw fundus images or model weights. The image archive is stored separately in Cloudflare R2.
```

Upload with `huggingface_hub`:

```bash
pip install -U huggingface_hub
huggingface-cli login

python - <<PY
from huggingface_hub import HfApi, create_repo
from pathlib import Path

repo_id = "<HF_USERNAME_OR_ORG>/fundus-qwen3vl-generated-annotations"
artifact_dir = Path("/workspace/artifacts/fundus_transfer_20260521")

create_repo(repo_id, repo_type="dataset", private=True, exist_ok=True)
api = HfApi()
for name in [
    "fundus_generated_annotations_20260521.tar.gz",
    "fundus_generated_annotations_20260521.tar.gz.sha256",
    "fundus_generated_annotations.filelist.txt",
    "ANNOTATION_PACKAGE_CONTENTS.txt",
]:
    api.upload_file(
        path_or_fileobj=artifact_dir / name,
        path_in_repo=name,
        repo_id=repo_id,
        repo_type="dataset",
    )
PY
```

Download on a new server:

```bash
pip install -U huggingface_hub
huggingface-cli login

huggingface-cli download \
  <HF_USERNAME_OR_ORG>/fundus-qwen3vl-generated-annotations \
  --repo-type dataset \
  --local-dir /workspace/artifacts/fundus_generated_annotations

cd /workspace/artifacts/fundus_generated_annotations
sha256sum -c fundus_generated_annotations_20260521.tar.gz.sha256
cd /workspace/LLaMA-Factory
tar -xzf /workspace/artifacts/fundus_generated_annotations/fundus_generated_annotations_20260521.tar.gz
```

## Cloudflare R2 Bucket

Recommended bucket:

```text
fundusv1
```

Recommended prefix layout:

```text
r2://fundusv1/images/fundus_image_dataset_20260521.tar
r2://fundusv1/images/fundus_image_dataset_20260521.tar.sha256
r2://fundusv1/images/fundus_image_dataset.filelist.txt
r2://fundusv1/images/IMAGE_PACKAGE_CONTENTS.txt
```

### R2 Credentials

Create an R2 API token with read/write access to the bucket. Record:

```text
R2_ACCOUNT_ID=<cloudflare_account_id>
R2_ACCESS_KEY_ID=<r2_access_key_id>
R2_SECRET_ACCESS_KEY=<r2_secret_access_key>
R2_BUCKET=fundusv1
R2_ENDPOINT=https://4ff11044d39e473b1c3f56367f45fe71.r2.cloudflarestorage.com
```

Do not commit these values.

### Option A: Upload With AWS CLI

```bash
pip install -U awscli

export AWS_ACCESS_KEY_ID=<r2_access_key_id>
export AWS_SECRET_ACCESS_KEY=<r2_secret_access_key>
export AWS_DEFAULT_REGION=auto
export R2_ENDPOINT=https://4ff11044d39e473b1c3f56367f45fe71.r2.cloudflarestorage.com
export R2_BUCKET=fundusv1

aws --endpoint-url "$R2_ENDPOINT" s3 cp \
  /workspace/artifacts/fundus_transfer_20260521/fundus_image_dataset_20260521.tar \
  s3://$R2_BUCKET/images/fundus_image_dataset_20260521.tar

aws --endpoint-url "$R2_ENDPOINT" s3 cp \
  /workspace/artifacts/fundus_transfer_20260521/fundus_image_dataset_20260521.tar.sha256 \
  s3://$R2_BUCKET/images/fundus_image_dataset_20260521.tar.sha256

aws --endpoint-url "$R2_ENDPOINT" s3 cp \
  /workspace/artifacts/fundus_transfer_20260521/fundus_image_dataset.filelist.txt \
  s3://$R2_BUCKET/images/fundus_image_dataset.filelist.txt

aws --endpoint-url "$R2_ENDPOINT" s3 cp \
  /workspace/artifacts/fundus_transfer_20260521/IMAGE_PACKAGE_CONTENTS.txt \
  s3://$R2_BUCKET/images/IMAGE_PACKAGE_CONTENTS.txt
```

Download on a new server:

```bash
mkdir -p /workspace/artifacts/fundus_images

aws --endpoint-url "$R2_ENDPOINT" s3 cp \
  s3://$R2_BUCKET/images/fundus_image_dataset_20260521.tar \
  /workspace/artifacts/fundus_images/fundus_image_dataset_20260521.tar

aws --endpoint-url "$R2_ENDPOINT" s3 cp \
  s3://$R2_BUCKET/images/fundus_image_dataset_20260521.tar.sha256 \
  /workspace/artifacts/fundus_images/fundus_image_dataset_20260521.tar.sha256

cd /workspace/artifacts/fundus_images
sha256sum -c fundus_image_dataset_20260521.tar.sha256
cd /workspace/LLaMA-Factory
tar -xf /workspace/artifacts/fundus_images/fundus_image_dataset_20260521.tar
```

### Option B: Upload With rclone

Create `~/.config/rclone/rclone.conf`:

```ini
[r2]
type = s3
provider = Cloudflare
access_key_id = <r2_access_key_id>
secret_access_key = <r2_secret_access_key>
endpoint = https://4ff11044d39e473b1c3f56367f45fe71.r2.cloudflarestorage.com
acl = private
```

Upload:

```bash
rclone copy /workspace/artifacts/fundus_transfer_20260521/fundus_image_dataset_20260521.tar r2:fundusv1/images/ --progress
rclone copy /workspace/artifacts/fundus_transfer_20260521/fundus_image_dataset_20260521.tar.sha256 r2:fundusv1/images/ --progress
rclone copy /workspace/artifacts/fundus_transfer_20260521/fundus_image_dataset.filelist.txt r2:fundusv1/images/ --progress
rclone copy /workspace/artifacts/fundus_transfer_20260521/IMAGE_PACKAGE_CONTENTS.txt r2:fundusv1/images/ --progress
```

Download:

```bash
rclone copy r2:fundusv1/images/fundus_image_dataset_20260521.tar /workspace/artifacts/fundus_images/ --progress
rclone copy r2:fundusv1/images/fundus_image_dataset_20260521.tar.sha256 /workspace/artifacts/fundus_images/ --progress
```

## Path Contract

Extract both packages under the LLaMA-Factory root:

```text
/workspace/LLaMA-Factory/data/...
/workspace/LLaMA-Factory/reports/...
```

The JSONL files use relative image paths. Keeping this layout avoids rewriting `images` fields.

## License Recommendation

For the Hugging Face generated-annotation dataset, use:

```text
license: other
visibility: private
```

Reason: this package contains generated annotations/CoT derived from multiple fundus datasets whose original licenses and redistribution rules may differ. Do not mark it as CC-BY, MIT, or Apache unless every upstream dataset explicitly permits that redistribution. A safe dataset-card note is:

```markdown
License: Other / Research-only. This repository contains derived annotation files for internal research use. It does not include raw fundus images. Users must obtain the original image datasets under their respective licenses before use.
```

If you later publish only code/configs, that GitHub repository can use Apache-2.0 or MIT. For this annotation dataset, keep `other` unless you have checked all source licenses.

## Your R2 Settings

Use these values for this project:

```bash
export R2_ENDPOINT=https://4ff11044d39e473b1c3f56367f45fe71.r2.cloudflarestorage.com
export R2_BUCKET=fundusv1
export AWS_DEFAULT_REGION=auto
export AWS_ACCESS_KEY_ID=<your_r2_access_key_id>
export AWS_SECRET_ACCESS_KEY=<your_r2_secret_access_key>
```

Important: the AWS CLI endpoint is the account endpoint without the bucket path. Your browser-style/API URL may appear as:

```text
https://4ff11044d39e473b1c3f56367f45fe71.r2.cloudflarestorage.com/fundusv1
```

But for AWS CLI use:

```text
--endpoint-url https://4ff11044d39e473b1c3f56367f45fe71.r2.cloudflarestorage.com
s3://fundusv1/...
```
