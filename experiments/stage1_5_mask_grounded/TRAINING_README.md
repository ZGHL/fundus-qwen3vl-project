# Stage-1.5 全量训练说明（VM）

目标：在 Adapter 1 基础上 warm-start 续训，把单病灶感知升级为 **present + count/area 桶**（mask 接地），
并用全量负样本把**特异度**拉上去（GB10 小规模证明里特异度未提升，归因负样本太少/训练太短——见 `results/PROOF_RESULTS.md`）。

## 0. 前置
- 起点 ckpt：Adapter 1 = `saves/qwen3-vl-8b-fundus/lora/stage1_en_cot`（SHA256 `086bca5057c4e021cbf3f18ec9bb99b1df5a70c7cc3e8f98c1a90c77ec10dea3`；在 R2 handoff tar 内）。
- 基模：`models/Qwen3-VL-8B-Instruct`。
- **勿装 liger-kernel**（不支持 Qwen3-VL，且会升级 triton 破坏 vLLM；GB10 上踩过，见下）。

## 1. 数据
- **从 R2 拉**（数据不进 git）：`s3://fundusv1/datasets/stage1_5_mask_grounded_20260613.tar.zst`
  （SHA256 `66064d872926d00592747a87c716ee0f23dee7768c7400665b17113315932e6a`，解包得
  `stage1_5_full_{train,test}_sft.jsonl` + `fgadr_main4_proof_{train,test}_sft.jsonl` + 分布 json）。
- 放到 `data/annotation/` 并注册（键 `stage1_5_full_train` / `stage1_5_full_test`，sharegpt 格式，columns=messages,images）。
- 或在 VM 上重建：`python scripts/build_stage1_5_full.py`（需 FGADR/DDR-seg masks + `validated_clean.jsonl`，路径见脚本头）。
- 图像：FGADR `Original_Images`、DDR-seg `lesion_segmentation/*/image`、IDRiD（复用）。

## 2. 训练配置
- 模板：`configs/stage1_5_full_TEMPLATE.yaml`（把 `dataset:` 改为 `stage1_5_full_train`，`eval_dataset: fundus_stage1_en_cot_gold_dev`）。
- 关键项：`adapter_name_or_path=…/stage1_en_cot` + `create_new_adapter=false`（warm-start 续 Adapter1）；
  LoRA rank16/alpha32/dropout0.05/target=all；`freeze_vision_tower=false`、`freeze_multi_modal_projector=true`；
  template `qwen3_vl_nothink`、cutoff_len 2304、image 589824、bf16。
- VM 提速：装 **flash-attn**（`flash_attn: fa2`）、加大 batch、`gradient_checkpointing=false`。
- 超参建议：LR 6e-6、cosine、**2 epoch**、dense save、dev 选模。
- **针对证明暴露的问题的两项调整（重要）**：
  1. **负样本已全量纳入**（train 含 SE absent 1515 等）；若 present/absent 仍回退，可对 present 样本轻微上采样或对两类任务分别加权。
  2. count/area 桶若准确率不足，可加大数据/epoch；area 桶定义保持每(数据集,病灶)分位。

## 3. 评测（务必走 vLLM；transformers.generate 在 GB10 会 nvrtc 崩，VM 视情况）
- **present/absent（主对标）**：真实 Gold-Dev(596) / Gold-Test(900)，算 Macro-F1 / Recall / **Specificity** / per-lesion F1，
  三方对比 base / Adapter1(Macro-F1 0.66, spec 0.385) / ckpt-20(spec 0.385→balacc 0.601) / 本模型。**特异度为主验收**。
- **count/area 桶**：在 `stage1_5_full_test`（图像互斥）上，用 `scripts/score_proof.py` 算每病灶 count_bucket/area_bucket 准确率 + 解析率。
- 推理用 `scripts/vllm_infer.py --adapter_name_or_path <out> --max_lora_rank 32 --enforce_eager true`。

## 4. 成功标准
- present/absent **不低于 Adapter1**，且 **特异度显著提升**（目标 spec 明显 > 0.385，HE/EX/MA/SE 不塌）。
- count/area 桶准确率 **显著高于 3 类随机(0.33)**（证明里 macro count 0.54、HE 0.76，VM 全量应更高），相邻桶可容忍。
- 通过后即可把该 Stage-1.5 感知接入 Stage-2 分级 CoT，让 NPDR burden 真接地。

## 5. 环境坑（GB10 实记，VM 注意）
- `pip install liger-kernel` 会把 NVIDIA 内部 `pytorch_triton 3.1.0` 覆盖成 PyPI `triton 3.7.0` → vLLM 0.14.1 崩（`AttrsDescriptor`/`target_info`）。修复=从镜像还原 triton 3.1.0。**VM 上同样别装 liger。**
