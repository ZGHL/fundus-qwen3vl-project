# FundusChain Stage-1.5 v3 Experiment Handoff

Date: 2026-06-13

## Outcome

Stage-1.5 v3 is a present/absent-only specificity-correction warm start from
Adapter1. Training was intentionally stopped at optimizer step 470 after the
objective had converged. The selected handoff model is **checkpoint-400**.

Checkpoint-400 is selected by Macro balanced accuracy with parse coverage and
recall guardrails. See `eval/CHECKPOINT_SELECTION_v3.md`.

Key checkpoint-400 results on the clean image-disjoint v3 test:

- Macro: F1 0.616, recall 0.844, specificity 0.606, balanced accuracy 0.725
- Micro: F1 0.619, recall 0.827, specificity 0.594, balanced accuracy 0.711
- Adapter1 baseline Macro: F1 0.528, recall 0.965, specificity 0.213,
  balanced accuracy 0.589

## Canonical Sources

- Git repository: `https://github.com/ZGHL/fundus-qwen3vl-project.git`
- Git branch: `stage1_5_mask_grounded`
- Experiment record commit before this handoff update:
  `974e069 Stage-1.5 v3: record stable run and clean head-to-head results`
- LLaMA-Factory root used on this machine: `/root/fundus-work/LLaMA-Factory`
- Base model relative path: `models/Qwen3-VL-8B-Instruct`
- Selected adapter relative path:
  `saves/qwen3-vl-8b-fundus/lora/stage1_5_v3/checkpoint-400`
- Selected merged model relative path: `merged/v3`

## R2 Inventory

Bucket: `s3://fundusv1`

- Dataset:
  `datasets/stage1_5_v3_20260613.tar.zst`
- Dataset SHA sidecar:
  `datasets/stage1_5_v3_20260613.tar.zst.sha256`
- Dataset SHA256:
  `52644d761ca2be4c86f92c532bb9b66778de0bd83a3fcdc9366e4355d9e0ca69`
- Full checkpoint-400:
  `models/stage1_5_v3_20260613/checkpoint-400/`
- Selected merged model:
  `models/stage1_5_v3_20260613/merged-v3/`
- Evaluation predictions and reports:
  `predictions/stage1_5_v3_20260613/`

The other complete checkpoints are 80, 160, 240, and 320. Their adapters,
trainer state, optimizer state, predictions, and reports are uploaded with this
handoff. Candidate merged models are not canonical artifacts because they can be
recreated exactly from the base model and checkpoint adapter.

## Restore On A New Machine

Credentials must be supplied only through environment variables. Do not write
them into scripts or config files.

```bash
git clone --branch stage1_5_mask_grounded \
  https://github.com/ZGHL/fundus-qwen3vl-project.git /workspace/fundus-qwen3vl-project

export LF=/workspace/LLaMA-Factory
cd /workspace/fundus-qwen3vl-project
bash scripts/setup/apply_llamafactory_patches.sh "$LF"

aws s3 cp s3://fundusv1/datasets/stage1_5_v3_20260613.tar.zst /tmp/v3.tar.zst \
  --endpoint-url "$R2_ENDPOINT"
echo '52644d761ca2be4c86f92c532bb9b66778de0bd83a3fcdc9366e4355d9e0ca69  /tmp/v3.tar.zst' \
  | sha256sum -c -
tar --zstd -xf /tmp/v3.tar.zst -C "$LF/data/annotation"

aws s3 sync s3://fundusv1/models/stage1_5_v3_20260613/checkpoint-400/ \
  "$LF/saves/qwen3-vl-8b-fundus/lora/stage1_5_v3/checkpoint-400/" \
  --endpoint-url "$R2_ENDPOINT"
aws s3 sync s3://fundusv1/models/stage1_5_v3_20260613/merged-v3/ \
  "$LF/merged/v3/" --endpoint-url "$R2_ENDPOINT"
aws s3 sync s3://fundusv1/predictions/stage1_5_v3_20260613/ \
  /workspace/fundus-qwen3vl-project/experiments/stage1_5_mask_grounded/eval/ \
  --endpoint-url "$R2_ENDPOINT"
```

Register `stage1_5_v3_train` and `stage1_5_v3_test` in
`$LF/data/annotation/dataset_info.json` as ShareGPT datasets with `messages` and
`images` columns. Copy `configs/stage1_5_v3_warmstart.yaml` to
`$LF/examples/train_lora/` if training must be reproduced.

Before training or evaluation, verify every referenced image under `$LF/data/`.
This run required APTOS grade-0 images from the full image snapshot because they
were not available under the incremental R2 image prefix.

## Resume Or Reproduce Training

The selected adapter is a complete Trainer checkpoint, including optimizer and
trainer state. To continue from step 400:

```bash
cd "$LF"
export DISABLE_VERSION_CHECK=1
llamafactory-cli train examples/train_lora/stage1_5_v3_warmstart.yaml \
  --resume_from_checkpoint saves/qwen3-vl-8b-fundus/lora/stage1_5_v3/checkpoint-400
```

For a clean reproduction from Adapter1, use the committed v3 config without
`--resume_from_checkpoint`. The stable run used LR `3e-6`, SDPA, per-device batch
size 2, gradient accumulation 8, and gradient checkpointing.

## Reproduce Evaluation

Use the isolated vLLM environment and merge adapters before evaluation. Dynamic
LoRA on the visual model hits an assertion and is not the validated path.

```bash
cd "$LF"
export DISABLE_VERSION_CHECK=1
llamafactory-cli export \
  --model_name_or_path models/Qwen3-VL-8B-Instruct \
  --adapter_name_or_path saves/qwen3-vl-8b-fundus/lora/stage1_5_v3/checkpoint-400 \
  --template qwen3_vl_nothink --finetuning_type lora \
  --export_dir merged/v3 --export_size 5
```

After export, remove `extra_special_tokens` from `tokenizer_config.json` if its
value is a list. Transformers expects a mapping; leaving the list caused the
earlier Adapter1 evaluation to degenerate into `!` output.

Validated inference parameters:

```text
dataset=stage1_5_v3_test
template=qwen3_vl_nothink
cutoff_len=2304
max_new_tokens=256
image_min_pixels=65536
image_max_pixels=262144
batch_size=24
enforce_eager=true
gpu_memory_utilization=0.85
temperature=0
top_p=1
top_k=-1
seed=20260613
```

Score with:

```bash
python experiments/stage1_5_mask_grounded/scripts/score_proof.py \
  "$LF/data/annotation/stage1_5_v3_test_sft.jsonl" \
  experiments/stage1_5_mask_grounded/eval/base_adapter1_v3test.jsonl \
  experiments/stage1_5_mask_grounded/eval/v3_v3test.jsonl \
  experiments/stage1_5_mask_grounded/eval/PROOF_RESULTS_v3.md
```

## Known Issues And Guardrails

- Do not install `liger-kernel`; it can upgrade Triton and break the validated
  vLLM environment.
- This machine had no compatible flash-attn, so the stable config uses SDPA.
- The validated evaluation stack is vLLM 0.11, Transformers 4.57.1, Triton 3.4.
- Merge LoRA before vLLM evaluation; dynamic visual LoRA is not supported here.
- Fix list-valued `extra_special_tokens` after merge.
- Stop on any non-finite adapter tensor, missing image, bad dataset SHA, or
  dependency incompatibility.
- v2 encountered non-finite checkpoint behavior. All five v3 checkpoints passed
  the finite-tensor check.
- The v3 clean test has now been used for checkpoint selection. Reserve a new
  image-disjoint holdout for final Stage-2 reporting.

## Stage-2 Recommendation

Start Stage-2 from checkpoint-400, not from the merged model. Keep v3
present/absent examples in a replay mixture to preserve specificity, and add a
new untouched holdout before tuning Stage-2. MA specificity and SE precision are
the main failure modes to monitor.
