# Upload Record 2026-05-21

## Hugging Face Dataset

Repo:

```text
https://huggingface.co/datasets/Guohou/fundusAnnotationsV1
```

Uploaded files:

```text
ANNOTATION_PACKAGE_CONTENTS.txt
LICENSE
README.md
fundus_generated_annotations.filelist.txt
fundus_generated_annotations_20260521.tar.gz
fundus_generated_annotations_20260521.tar.gz.sha256
```

Package role: generated annotation and CoT/SFT JSONL package. It does not contain raw fundus images.

## Cloudflare R2

Endpoint:

```text
https://4ff11044d39e473b1c3f56367f45fe71.r2.cloudflarestorage.com
```

Bucket:

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

Large object size:

```text
67500830720 bytes
```

## Restore

```bash
cd /workspace/LLaMA-Factory

# annotations from HF download directory
tar -xzf /workspace/artifacts/fundus_generated_annotations/fundus_generated_annotations_20260521.tar.gz

# images from R2 download directory
cd /workspace/artifacts/fundus_images
sha256sum -c fundus_image_dataset_20260521.tar.sha256
cd /workspace/LLaMA-Factory
tar -xf /workspace/artifacts/fundus_images/fundus_image_dataset_20260521.tar
```
