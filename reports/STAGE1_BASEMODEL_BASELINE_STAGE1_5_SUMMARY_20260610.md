# Stage 1 基模、Baseline 与 Stage 1.5 实验汇总

日期：2026-06-10

## 1. 实验范围与当前结论

本报告统一汇总以下模型与实验：

1. **Qwen3-VL 8B 基模**：`Qwen/Qwen3-VL-8B-Instruct`，不加载任何眼底适配器。
2. **Stage 1 Adapter 1**：从基模训练的首个英文单病灶 CoT LoRA。
3. **当前 Stage 1 Baseline**：从 Adapter 1 进行温和平衡校准后选出的
   `checkpoint-20`。
4. **Stage 1.5**：从当前 Stage 1 Baseline 继续进行六病灶、强
   hard-negative、视觉塔与 Projector 适配的有限实验。

当前正式结论：

- 当前 Stage 2 初始化模型仍为 Stage 1 Baseline：

  ```text
  /workspace/LLaMA-Factory/saves/qwen3-vl-8b-fundus/lora/stage1_en_cot_gentle_calibrated/checkpoint-20
  ```

- Stage 1 Baseline 是目前唯一同时通过 Gold Dev 选择，并完成 Gold Test
  最终评估的推荐模型。
- Stage 1.5 有效提升了特异性和 Balanced Accuracy，但损失了部分基础病灶
  Recall/F1，因此作为特异性研究分支保存，不替代当前 baseline。
- Gold Dev 和 Gold Test 均为 DDR 内部高质量、图像组隔离的留出集，不是外部
  数据集评估。

## 2. 数据与样本分配

### 2.1 原始 Stage 1 Adapter 1 训练集

原始 Stage 1 使用严格单病灶英文 CoT，每条训练样本只询问一个目标病灶。

总计：`12,650` 条 lesion-level 样本。

| 病灶 | Present | Absent | 总计 | 阳性率 |
|---|---:|---:|---:|---:|
| MA | 1,200 | 450 | 1,650 | 72.7% |
| HE | 2,000 | 1,200 | 3,200 | 62.5% |
| EX | 2,000 | 1,200 | 3,200 | 62.5% |
| SE | 1,600 | 1,600 | 3,200 | 50.0% |
| IRMA | 400 | 500 | 900 | 44.4% |
| NV | 140 | 360 | 500 | 28.0% |
| **总计** | **7,340** | **5,310** | **12,650** | **58.0%** |

设计目标是优先建立广泛病灶感知能力。主要问题是 MA、SE 等病灶的高质量
hard negative 压力不足，使 Adapter 1 具有高召回、低特异性的正样本偏置。

IRMA 和 NV 阳性训练行中包含重复增强视图：

- IRMA：400 条阳性训练行来自 120 张独立阳性图像。
- NV：140 条阳性训练行来自 34 张独立阳性图像。

因此，样本行数不能被解释为同等数量的独立临床证据。

### 2.2 Stage 1 Baseline 温和平衡校准集

当前 Stage 1 Baseline 不是从基模重新训练，而是从 Adapter 1 继续进行短程校准。

总计：`2,500` 条，只使用可信 `S0/S1` 数据，并排除 Gold Dev、Gold Test、
IRMA Locked Test 和 NV Locked Test。

| 病灶 | Present | Absent | 总计 | Hard negatives |
|---|---:|---:|---:|---:|
| MA | 450 | 150 | 600 | 150 |
| HE | 500 | 100 | 600 | 100 |
| EX | 500 | 100 | 600 | 95 |
| SE | 450 | 250 | 700 | 250 |
| **总计** | **1,900** | **600** | **2,500** | **595** |

该分配保留较多阳性锚点，同时加入高质量 hard negatives，以降低假阳性但避免
病灶能力坍塌。

### 2.3 Stage 1.5 六病灶训练集

Stage 1.5 从当前 Stage 1 Baseline 继续训练，目标是进一步提高四个基础病灶的
特异性，并探索 IRMA/NV 信号。

总计：`5,680` 条；正负样本完全平衡。

| 病灶 | Present | Absent | 总计 | Hard-negative 行数 |
|---|---:|---:|---:|---:|
| MA | 600 | 600 | 1,200 | 428 |
| HE | 600 | 600 | 1,200 | 430 |
| EX | 600 | 600 | 1,200 | 176 |
| SE | 600 | 600 | 1,200 | 586 |
| IRMA | 300 | 300 | 600 | 300 |
| NV | 140 | 140 | 280 | 140 |
| **总计** | **2,840** | **2,840** | **5,680** | **2,060** |

独立阳性图像情况：

| 病灶 | 阳性训练行 | 独立阳性图像 | 最大视图数 |
|---|---:|---:|---:|
| MA | 600 | 567 | 2 |
| HE | 600 | 600 | 1 |
| EX | 600 | 600 | 1 |
| SE | 600 | 470 | 2 |
| IRMA | 300 | 119 | 3 |
| NV | 140 | 34 | 5 |

Stage 1.5 未使用 Gold Dev、Gold Test、Internal Validation 或 Locked Test
输入进行训练。

## 3. 评估集分配

所有数据按照原始图像组隔离，增强视图不能跨训练、开发和测试集合。

| 评估集 | 数量 | 来源与质量 | 用途 |
|---|---:|---|---|
| Internal Validation | 3,773 | 多来源、多证据层级 | 日常训练可见性与诊断 |
| Gold Dev | 596 | DDR 可信 `S0` mask | checkpoint 选择与开发决策 |
| Gold Test | 900 | 独立 DDR 可信 `S0` mask | 最终内部四病灶评估 |
| Weak-negative Challenge | 1,930 | 较弱标签负样本 | 假阳性与鲁棒性诊断 |
| IRMA Locked Test | 100 | FGADR，20 阳性/80 阴性 | 最终 IRMA 探索评估 |
| NV Locked Test | 88 | FGADR，8 阳性/80 阴性 | 最终 NV 探索评估 |

### 3.1 Gold Dev 分配

Gold Dev 用于 checkpoint 选择，不应作为最终成绩。其类别分布偏斜，尤其 MA
只有 17 个阴性，因此 MA 特异性方差较大。

| 病灶 | Present | Absent | 总计 |
|---|---:|---:|---:|
| MA | 132 | 17 | 149 |
| HE | 113 | 36 | 149 |
| EX | 70 | 79 | 149 |
| SE | 86 | 63 | 149 |
| **总计** | **401** | **195** | **596** |

### 3.2 Gold Test 分配

Gold Test 是当前主要四病灶最终内部测试集，仅在候选通过 Gold Dev 选择后运行。

| 病灶 | Present | Absent | 总计 |
|---|---:|---:|---:|
| MA | 124 | 101 | 225 |
| HE | 194 | 31 | 225 |
| EX | 171 | 54 | 225 |
| SE | 42 | 183 | 225 |
| **总计** | **531** | **369** | **900** |

### 3.3 各模型实际完成的评估

| 模型 | Gold Dev | Gold Test | IRMA Locked | NV Locked | Weak-negative |
|---|---|---|---|---|---|
| Qwen3-VL 8B 基模 | 严格评分 | 严格与宽松语义评分 | 未运行 | 未运行 | 未运行 |
| Stage 1 Adapter 1 | 已运行 | 已运行 | 已运行 | 已运行 | 无正式结果 |
| 当前 Stage 1 Baseline | 已运行并用于选择 | 已运行一次 | 未单独运行 | 未单独运行 | 无正式结果 |
| Stage 1.5 | 评估 5 个 checkpoint | 未运行 | 未运行 | 未运行 | 未运行 |

Stage 1.5 未运行 Gold Test/Locked Test，是因为所有候选均未通过四个基础病灶
保护线。该停止策略避免使用最终测试集反复调参。

## 4. 训练与推理配置

### 4.1 共同设置

| 项目 | 设置 |
|---|---|
| 基模 | Qwen3-VL-8B-Instruct |
| 任务 | 严格单病灶英文 CoT 二分类 |
| 模板 | `qwen3_vl_nothink` |
| Cutoff length | 2,304 |
| 图像像素范围 | 65,536 至 589,824 |
| 精度 | BF16 |
| Attention | PyTorch SDPA |
| LoRA | Rank 16，Alpha 32，Dropout 0.05，target `all` |
| Optimizer | `adamw_torch` |
| Scheduler | cosine |
| Gradient checkpointing | 开启 |
| 有效 batch size | 16 |

### 4.2 各阶段差异

| 配置 | Adapter 1 | 当前 Stage 1 Baseline 校准 | Stage 1.5 |
|---|---:|---:|---:|
| 初始化 | Qwen3-VL 8B 基模 | Adapter 1 | Stage 1 Baseline checkpoint-20 |
| 训练样本 | 12,650 | 2,500 | 5,680 |
| 最大训练长度 | 1 epoch / 791 steps | 40 steps | 180 steps |
| 选中 checkpoint | 最终 Adapter 1 | checkpoint-20 | 无正式通过候选 |
| Learning rate | `6e-6` | `2e-7` | `5e-7` |
| Micro batch | 1 | 2 | 2 |
| Gradient accumulation | 16 | 8 | 8 |
| Vision tower | LoRA 开启 | 冻结 | LoRA 开启 |
| Multimodal projector | 冻结 | 冻结 | 开启并训练 `visual.merger` |
| Language model LoRA | 开启 | 继续训练 | 继续训练 |
| 保存间隔 | 200 steps | 10 steps | 20 steps |

相关配置：

- `configs/train/stage1_en_cot.yaml`
- `configs/train/stage1_en_cot_gentle_calibration.yaml`
- `configs/train/stage1_5_six_lesion_limited.yaml`

## 5. 最终结果

### 5.1 Gold Test 最终四病灶结果

Gold Test 是当前可用于最终横向比较的主要评估集。

基模严格评分为 `0`，原因是其未遵循所要求的目标病灶结构化输出协议。为区分
格式能力与语义病灶识别能力，基模同时报告宽松语义评分。

| 模型 | 评分方式 | Macro F1 | Recall | Specificity | Balanced Acc. |
|---|---|---:|---:|---:|---:|
| Qwen3-VL 8B 基模 | 严格结构化评分 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Qwen3-VL 8B 基模 | 宽松语义评分 | 0.3237 | 0.2532 | **0.8011** | 0.5272 |
| Stage 1 Adapter 1 | 严格结构化评分 | 0.6524 | **0.8758** | 0.1793 | 0.5276 |
| **当前 Stage 1 Baseline** | 严格结构化评分 | **0.6595** | 0.8164 | 0.3846 | **0.6005** |

各病灶 F1：

| 模型 | MA | HE | EX | SE |
|---|---:|---:|---:|---:|
| Qwen3-VL 8B 基模，宽松语义评分 | 0.0000 | 0.5128 | 0.6360 | 0.1458 |
| Stage 1 Adapter 1 | **0.5249** | **0.9008** | **0.8680** | 0.3158 |
| 当前 Stage 1 Baseline | 0.5161 | 0.8973 | 0.8652 | **0.3594** |

当前 Stage 1 Baseline 的完整病灶结果：

| 病灶 | F1 | Recall | Specificity | Balanced Acc. |
|---|---:|---:|---:|---:|
| MA | 0.5161 | 0.5806 | 0.1782 | 0.3794 |
| HE | 0.8973 | 0.8557 | 0.6774 | 0.7665 |
| EX | 0.8652 | 0.9006 | 0.4259 | 0.6633 |
| SE | 0.3594 | 0.9286 | 0.2568 | 0.5927 |

当前 Stage 1 Baseline 相比 Adapter 1：

| 指标 | 变化 |
|---|---:|
| Macro F1 | +0.0071 |
| Specificity | +0.2053 |
| Balanced Accuracy | +0.0729 |
| MA F1 | -0.0088 |
| HE F1 | -0.0035 |
| EX F1 | -0.0029 |
| SE F1 | +0.0437 |

### 5.2 Gold Dev 同集比较

以下结果均来自同一个 596 条 Gold Dev，可用于开发阶段横向比较。

| 模型 | Macro F1 | Recall | Specificity | Balanced Acc. |
|---|---:|---:|---:|---:|
| Qwen3-VL 8B 基模，严格评分 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Stage 1 Adapter 1 | 0.7081 | **0.8240** | 0.1595 | 0.4917 |
| **当前 Stage 1 Baseline** | **0.7314** | 0.7615 | 0.3950 | 0.5782 |
| Stage 1.5 checkpoint-40 | 0.6991 | 0.6899 | 0.5115 | 0.6007 |
| Stage 1.5 checkpoint-80 | 0.6685 | 0.6378 | **0.5918** | 0.6148 |
| Stage 1.5 checkpoint-120 | 0.6713 | 0.6501 | 0.5550 | 0.6025 |
| Stage 1.5 checkpoint-160 | 0.6979 | 0.6914 | 0.5383 | **0.6149** |
| Stage 1.5 checkpoint-180 | 0.6792 | 0.6685 | 0.5138 | 0.5912 |

Stage 1.5 三个保留 checkpoint：

- `checkpoint-40`：Stage 1.5 最高 Gold Dev Macro F1。
- `checkpoint-80`：Stage 1.5 最高 Gold Dev 特异性。
- `checkpoint-160`：Stage 1.5 最高 Gold Dev Balanced Accuracy。

### 5.3 当前 Baseline 与 Stage 1.5 checkpoint-160

同一 Gold Dev 上比较：

| 病灶 | Baseline F1 | Stage 1.5 F1 | Baseline Specificity | Stage 1.5 Specificity |
|---|---:|---:|---:|---:|
| MA | **0.6606** | 0.6146 | 0.0588 | **0.4118** |
| HE | **0.8131** | 0.7653 | 0.6111 | **0.7778** |
| EX | **0.6776** | 0.6742 | 0.3544 | **0.3924** |
| SE | **0.7742** | 0.7374 | 0.5556 | **0.5714** |

Stage 1.5 显著减少了假阳性，特别是 MA 和 HE，但同时降低了 MA、HE、SE 的
Recall/F1。因此它证明 hard-negative 与视觉适配方向有效，但当前校正强度过高。

### 5.4 IRMA 与 NV Locked Test

以下 Locked Test 结果来自原始 Stage 1 Adapter 1。当前 Stage 1 Baseline 与
Stage 1.5 均没有单独运行这两个 Locked Test。

| 病灶 | 样本组成 | F1 | Recall | Specificity | Balanced Acc. |
|---|---|---:|---:|---:|---:|
| IRMA | 20 阳性 / 80 阴性 | 0.3256 | 0.7000 | 0.3500 | 0.5250 |
| NV | 8 阳性 / 80 阴性 | 0.2078 | 1.0000 | 0.2375 | 0.6188 |

IRMA/NV 当前只能作为探索性能力。尤其 NV 只有 8 个 Locked Test 阳性样本，
单张图像会使 Recall 变化 12.5 个百分点。

## 6. 结果解释与使用规则

### 6.1 为什么 Gold Dev F1 高于 Gold Test

Gold Dev 与 Gold Test 的病灶分布不同。例如：

- Gold Dev MA：132 阳性、17 阴性。
- Gold Test MA：124 阳性、101 阴性。
- Gold Dev SE：86 阳性、63 阴性。
- Gold Test SE：42 阳性、183 阴性。

偏向预测阳性的模型更容易在 Gold Dev 获得较高 F1，但可能隐藏低特异性。
因此：

- Gold Dev 仅用于候选选择。
- Gold Test 用于最终内部结果。
- 不允许将 Stage 1.5 的 Gold Dev 指标直接宣称为超过当前 baseline 的
  Gold Test 指标。

### 6.2 推荐用途

- **Stage 2 主 baseline**：当前 Stage 1 Baseline `checkpoint-20`。
- **高召回回滚模型**：Stage 1 Adapter 1。
- **特异性消融候选**：Stage 1.5 `checkpoint-80` 和 `checkpoint-160`。
- **IRMA/NV**：保留六字段输出结构，但在获得更稳定的 Locked Test 结果前，
  不作为可信 Stage 2 监督证据。

## 7. 产物与复现位置

主要报告与指标：

- `reports/STAGE1_FINAL_HANDOFF_20260609.md`
- `reports/STAGE1_5_SIX_LESION_RESULTS_20260610.md`
- `reports/metrics/stage1_overnight_base_relaxed.json`
- `reports/metrics/stage1_overnight_base_strict.json`
- `reports/metrics/stage1_overnight_adapter1.json`
- `reports/metrics/stage1_overnight_balanced.json`
- `reports/metrics/stage1_overnight_selection.json`
- `reports/metrics/stage1_5/`

R2 归档：

```text
s3://fundusv1/models/stage1/stage1_stage2_handoff_20260609.tar.zst
s3://fundusv1/models/stage1_5/stage1_5_six_lesion_specificity_20260610.tar.zst
```

归档清单：

- `manifests/models/stage1_stage2_handoff_20260609.yaml`
- `manifests/models/stage1_5_six_lesion_specificity_20260610.yaml`

