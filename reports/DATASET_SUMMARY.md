## 数据汇总（截至 2026-04-29）

本文件总结当前仓库里已准备/已计算的眼底数据（APTOS、DDR Grading、IDRiD、FGADR Seg-set、DDR lesion_segmentation），以及每个数据集包含的信息类型：**病灶标注（强标注/伪标签）、分级标注、RetSAM 独有量化指标（左右眼、杯盘比、A/V ratio、血管迂曲度等）**。

> 统计口径
>
> - **图像数量**优先以各数据集对应的 `data/cropped/<dataset>/crop_meta.jsonl` 为准（这是 pipeline 的输入索引）。
> - **RetSAM 输出数量**以 `outputs/retsam_<dataset>/*/quantitative_analysis.json` 为准（可解析且非空计为有效）。

### 1) 数据集清单与包含信息（强标注 vs 伪标签）


| 数据集 | 分级标注 | 病灶标注（Mask/量化） | eye_side | CDR | A/V ratio | CRAE/CRVE | tortuosity | fractal_dimension | vessel_qc_flag | OD/黄斑坐标 | 备注 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **APTOS** | 强标注 | 伪标签（RetSAM HE/EX/SE 量化） | 伪标签（RetSAM） | 伪标签（RetSAM） | 伪标签（RetSAM） | 伪标签（RetSAM） | 伪标签（RetSAM） | 伪标签（RetSAM） | 伪标签（RetSAM） | 伪标签（RetSAM；DDR 上坐标异常会在校验层降级） | RetSAM 仅对 Grade 1-4 计算（见下表）。 |
| **DDR Grading** | 强标注 | 伪标签（RetSAM HE/EX/SE 量化） | 伪标签（RetSAM） | 伪标签（RetSAM） | 伪标签（RetSAM） | 伪标签（RetSAM） | 伪标签（RetSAM） | 伪标签（RetSAM） | 伪标签（RetSAM） | 伪标签（RetSAM；已知 DDR macula 坐标尺度异常） | RetSAM 覆盖 Grade 1-4。 |
| **IDRiD（train+test）** | 强标注 | 强标注（像素级 mask） + 伪标签（RetSAM 量化补充） | 伪标签（RetSAM） | 伪标签（RetSAM） | 伪标签（RetSAM） | 伪标签（RetSAM） | 伪标签（RetSAM） | 伪标签（RetSAM） | 伪标签（RetSAM） | 强标注（官方定位） + 伪标签（RetSAM） | 强标注 mask 质量高；RetSAM 仅补独有量化指标。 |
| **FGADR Seg-set** | 强标注（分级在官方 CSV；此处不展开） | 强标注（像素级 mask） + 伪标签（RetSAM 量化补充） | 伪标签（RetSAM） | 伪标签（RetSAM） | 伪标签（RetSAM） | 伪标签（RetSAM） | 伪标签（RetSAM） | 伪标签（RetSAM） | 伪标签（RetSAM） | 伪标签（RetSAM） | 强标注 mask 质量高；RetSAM 仅补独有量化指标。 |
| **DDR lesion_segmentation（train+valid+test）** | 无 | 强标注（像素级 mask） + 伪标签（RetSAM 量化补充） | 伪标签（RetSAM） | 伪标签（RetSAM） | 伪标签（RetSAM） | 伪标签（RetSAM） | 伪标签（RetSAM） | 伪标签（RetSAM） | 伪标签（RetSAM） | 伪标签（RetSAM） | 分割集不带 grade；RetSAM 仅补独有量化指标。 |


### 2) RetSAM 已计算覆盖率（analysis-only 输出）


| 数据集 | 数据集总数 | RetSAM 伪标签个数 | 备注 |
|---|---:|---:|---|
| APTOS | 3662 | 1857 | RetSAM 仅对 Grade 1-4 计算；Grade 0 不跑（使用分级模板）。 |
| DDR Grading | 5006 | 5006 | 仅 Grade 1-4；已丢弃 Grade 5。 |
| IDRiD（train+test） | 516 | 413 | 当前仅 train 已跑 RetSAM；test 尚未跑（可按需补齐）。 |
| FGADR Seg-set | 1842 | 1842 | 按“已全部完成”口径填写。 |
| DDR lesion_segmentation（train+valid+test） | 757 | 383 | 当前仅 train 子集已跑 RetSAM；valid/test 尚未跑（可按需补齐）。 |


### 3) 数据量汇总（按“图像索引行数”）

- **索引总图像数（全部）**：APTOS 3662 + DDR Grading 5006 + IDRiD 516 + FGADR Seg-set 1842 + DDR lesion_segmentation 757 = **11783**
- **已完成 RetSAM analysis-only 的图像数（有效 JSON）**：APTOS 1857 + DDR Grading 5006 + IDRiD 413 + FGADR Seg 1842 + DDR seg(train) 383 = **9501**

### 4) 重要说明：RetSAM 无法生成 MA 病灶

当前 RetSAM 的 `quantitative_analysis.json`（你这套输出 schema）**不包含 MA（微动脉瘤）**的显式病灶检测/量化字段。  
因此我们在后续 CoT/监督数据中对 **MA** 的描述必须来自：

- **分级标签的规则模板**（例如 ICDR Grade 1 = MA only），或
- 其它独立的 MA 检测器/强标注（如 FGADR/IDRiD 的 MA mask），而不是 RetSAM。

## 第二部分：数据清洗与一致性校验逻辑

本部分定义 CoT 生成前的“可信事实层”。所有 RetSAM 伪标签、强标注 mask、分级标签和 RetSAM 独有量化指标，必须先经过统一清洗与一致性校验，再进入后续问题生成与训练。目标是避免模型学习到 grade-lesion 矛盾、低置信伪标签、坐标异常和 RetSAM 不支持字段造成的幻觉。

### 1) 输入与输出

| 输入来源 | 主要内容 | 用途 | 清洗后来源标记 |
|---|---|---|---|
| APTOS / DDR Grading | 图像 + DR grade | L4 分级监督；辅助约束 L3 病灶合理性 | `label` / `grade_rule` |
| IDRiD / FGADR / DDR lesion_segmentation | 像素级病灶 mask | L3 病灶 presence、count、area、bucket 的强监督 | `strong_mask` |
| RetSAM `quantitative_analysis.json` | HE/EX/SE 量化、eye_side、CDR、A/V、tortuosity 等 | 弱监督病灶与 L2 解剖/血管量化 | `validated_retsam` |
| crop/meta 索引 | 图像路径、数据集、grade、裁剪信息 | 样本对齐与可追溯 | `meta` |

清洗后统一输出 `validated.jsonl`，每张图至少包含：

- `grade`：若数据集有分级标签，则记录原始 grade。
- `lesions`：每类病灶的 `present/count/area/confidence/source/location_band`。
- `biomarkers`：`eye_side/cdr/av_ratio/tortuosity/vessel_qc_flag` 等。
- `validation_flags`：记录低置信、坐标异常、QC 失败、规则覆盖等信息。

### 2) 病灶清洗规则

| 病灶/字段 | 清洗逻辑 | 目的 |
|---|---|---|
| HE（出血） | RetSAM 输出需经过最小面积、最小数量和置信度过滤；若有强 mask，则强 mask 优先 | 减少伪阳性出血，保留可靠病灶负担 |
| EX（硬性渗出） | 同样经过面积/数量/置信度过滤；强 mask 优先 | 避免把亮斑、反光或噪声当作 EX |
| SE（软性渗出） | 因 RetSAM 置信度偏低，只保留高置信部分（如 Grade 2-4 中 top 百分位）；低置信时降级为 absent/unknown | 降低 SE 伪阳性对 CoT 的污染 |
| MA（微动脉瘤） | RetSAM 不提供 MA 字段；只能来自强标注 MA mask 或 grade 规则模板 | 防止模型学习“RetSAM 检出 MA”的错误说法 |
| NV（新生血管） | 若作为高级分级规则项出现，只能作为诊断规则/外部标注字段，不从当前 RetSAM HE/EX/SE 输出中生成 | 防止把 RetSAM 病灶字段误解释为 NV |
| location_band | 由病灶负担或校验层生成，只表达黄斑区/后极部/中周部/周边部/无 | 替代不稳定象限，不使用 TS/TI/NS/NI |

### 3) 病灶-分级一致性规则

| DR grade | 规则约束 | CoT 生成要求 |
|---|---|---|
| Grade 0 | 不应生成 HE/EX/SE 阳性描述；若 RetSAM 有低置信阳性，按一致性规则降级 | 输出“未见可靠 DR 病灶证据”或正常模板 |
| Grade 1 | 视为 MA-only 规则模板；HE/EX/SE 不应作为阳性证据 | 可写“符合轻度 DR/MA-only 模板”，但必须注明 MA 不来自 RetSAM |
| Grade 2 | 可允许有限 HE/EX 证据；SE 需高置信才保留 | CoT 以轻中度病灶负担描述为主，避免夸大 |
| Grade 3 | 需要较明显 HE/EX/SE 或较高病灶负担支撑 | 分级解释必须引用 L3 结构化证据 |
| Grade 4 | 高分级需要强病灶/增殖性线索支撑；若使用 NV，必须来自规则/外部标注而非当前 RetSAM 病灶字段 | 禁止在无证据时直接输出高分级 |
| Grade 5 | 当前 DDR Grading 中已丢弃，不进入训练 | 不生成训练样本 |

### 4) 解剖/血管量化校验规则

| 指标 | 校验逻辑 | 训练时处理 |
|---|---|---|
| `eye_side` | 依赖 OD-黄斑相对位置和血管质量；若质量差则标记 unknown | L2 laterality 中允许 abstain |
| `cdr` | 检查是否在合理数值范围；异常或缺失时置为 `null/unknown` | CDR 分档不强行输出 |
| `av_ratio` | 检查血管分割质量和数值合理性 | 低 QC 时输出 unknown |
| `tortuosity` | 检查血管提取质量和异常离群值 | 低 QC 时输出 unknown |
| `coord_valid` | 若 OD/黄斑坐标尺度异常或相对位置不可信，则置为 false | 不使用精确坐标和象限，仅保留 `location_band` |
| `vessel_qc_flag` | 血管/结构质量控制总标志 | 作为 system/JSON 证据，不写进普通用户问题 |

### 5) 进入 CoT 生成的原则

- **强标注优先**：mask 强标注 > validated RetSAM 伪标签 > grade 规则模板。
- **先校验后生成**：CoT 只能引用 `validated.jsonl` 中的字段，不直接引用原始 RetSAM JSON。
- **保守输出**：低置信、低 QC、字段缺失或规则冲突时输出 `unknown/不确定`，不强行给结论。
- **规则留痕**：被一致性规则覆盖的字段必须保留 `source` 或 `validation_flags`，便于后续误差分析和偏好对构造。
- **避免坐标噪声**：不生成象限描述，不触发依赖象限计数的 4-2-1 规则；统一使用 `location_band`。

## 第三部分：基于 FunBench 层级拆分的 L2-L4 问题与 CoT 设计（重构版）

本部分按 FunBench 的层级思想重新拆分：**每条样本只训练一个清晰能力点**。后续 SFT/MPO 数据生成只读取 `data/fundus_validated/validated_clean.jsonl`，不直接读取原始 RetSAM JSON 或未清洗 mask。

FunBench 将眼底读片拆成 L1-L4：L1 modality perception，L2 anatomy perception，L3 lesion analysis，L4 disease diagnosis。我们当前任务只覆盖 L2-L4。该拆分的关键不是写更多问题，而是避免把多个能力混在同一个问答里。比如 laterality、CDR、血管、病灶、分级应分别训练；混合问题只适合后期少量综合样本。

当前清洗后可用量：

| Level | 可用样本数 | 主要来源 | 训练目标 |
|---|---:|---|---|
| L2 | 9501 | RetSAM anatomy/OD/laterality/CDR | 解剖结构识别与基础量化 |
| L3 | 6829 | 强标注结构化 1370 + RetSAM 清洗后弱监督 5459 | 病灶属性、存在性、负担 |
| L4 | 9493 | grade label + L3 证据/Grade 0-1 模板 | 证据绑定分级与诊断核查 |

### 1) 总原则：拆子任务，不混能力

| 层级 | 不推荐 | 推荐 |
|---|---|---|
| L2 | “判断左右眼、CDR、A/V、病灶和分级” | 每题只问 laterality / CDR / vessel metric 之一 |
| L3 | “这是什么病、几级？” | 每题只问某类病灶属性、存在性、数量/面积或闭集 lesion-only |
| L4 | “这张图几级？” | 先核查 L3 证据，再问 grade 是否被支持 |

训练样本中的三处文本职责固定：

- **system**：放任务边界、输出格式、拒答规则、病灶鉴别原则。
- **user**：放观察任务和观察维度，不泄漏当前图真实标签。
- **assistant**：放显式视觉描述、结构化证据、结论和最小 JSON。

显式描述应主要放在 assistant 中。user 可以提示“看颜色/形态/边界/数量”，但不应把真实答案写出来。这样模型需要从图像中寻找对应结构，而不是从问题词猜标签。

### 2) L2：Anatomy Perception 子任务拆分

FunBench 的 L2 关注 major anatomical structures、optic disc/fovea 和 laterality。结合我们的字段，L2 拆成 3 个独立子任务。

| 子任务 | 生成条件 | user 问法 | assistant 学习点 | 输出字段 |
|---|---|---|---|---|
| L2-1 laterality | `eye_side.valid=true` | “请判断左眼还是右眼。” | 显式描述视盘/黄斑相对关系 | `eye_side` |
| L2-2 CDR | `cdr.valid=true` | “请观察视盘和视杯，估计杯盘比。” | 显式说明视盘、视杯、垂直径比例 | `cdr`, `cdr_bucket` |
| L2-3 vessel metrics | `av_ratio/tortuosity.valid` 或 invalid | “能否可靠判断 A/V 或迂曲度？” | 显式说明动静脉/走行；低 QC 拒答 | `av_ratio_bucket`, `tortuosity_bucket` |

#### L2-1 真实样本：laterality

来源：`aptos::all::000c1434d8d7`，图像 `data/cropped/aptos/grade1_4/000c1434d8d7.png`。

```json
{
  "messages": [
    {"role": "system", "content": "你是眼底图像分析助手。只判断 laterality；先描述视盘与黄斑/中央凹的相对关系，再给出左右眼。"},
    {"role": "user", "content": "<image> 请判断这张眼底图来自左眼还是右眼。"},
    {"role": "assistant", "content": "【观察】laterality 主要依据视盘与黄斑/中央凹的相对关系：视盘位于鼻侧，黄斑位于颞侧。本题不使用病灶或分级信息。\n\n【证据】eye_side=right；eye_side.valid=true；source=validated_retsam。\n\n【结论】该图判断为右眼。\n\n【JSON】\n{\"task\":\"L2_laterality\",\"eye_side\":\"right\",\"source\":\"validated_retsam\"}"}
  ],
  "images": ["data/cropped/aptos/grade1_4/000c1434d8d7.png"]
}
```

#### L2-2 真实样本：CDR

同一真实样本，字段 `cdr=0.3495, valid=true`。

```json
{
  "messages": [
    {"role": "system", "content": "你是眼底图像分析助手。只回答 CDR；先说明视盘和视杯如何识别，再根据垂直径比例分档。"},
    {"role": "user", "content": "<image> 请观察视盘和视杯，估计杯盘比是否偏高。"},
    {"role": "assistant", "content": "【观察】视盘通常是较亮的橙黄色圆/椭圆结构；视杯位于视盘中央，颜色更浅。CDR 主要比较视杯垂直径与视盘垂直径的比例。\n\n【证据】cdr=0.3495；cdr.valid=true；od_qc_flag=true；source=validated_retsam。\n\n【结论】CDR 约 0.35，属于正常或轻度范围，不提示明显杯盘比增大。\n\n【JSON】\n{\"task\":\"L2_cdr\",\"cdr\":0.3495,\"cdr_bucket\":\"normal_or_mild\",\"source\":\"validated_retsam\"}"}
  ],
  "images": ["data/cropped/aptos/grade1_4/000c1434d8d7.png"]
}
```

#### L2-3 真实样本：血管指标拒答

同一真实样本，`vessel_qc_flag=false`，`av_ratio/tortuosity.valid=false`。

```json
{
  "messages": [
    {"role": "system", "content": "你是眼底图像分析助手。只回答血管 A/V 和迂曲度；血管 QC 不可靠时必须输出 unknown。"},
    {"role": "user", "content": "<image> 请判断动静脉比例和血管迂曲度是否异常。"},
    {"role": "assistant", "content": "【观察】A/V ratio 需要比较动脉与静脉粗细，迂曲度需要观察血管走行是否明显弯曲；这类判断依赖可靠血管分割。\n\n【证据】vessel_qc_flag=false；av_ratio.valid=false；tortuosity.valid=false。\n\n【结论】不能可靠判断 A/V ratio 或血管迂曲度，输出 unknown。\n\n【JSON】\n{\"task\":\"L2_vessel_metrics\",\"av_ratio_bucket\":\"unknown\",\"tortuosity_bucket\":\"unknown\",\"reason\":\"vessel_qc_failed_or_missing\"}"}
  ],
  "images": ["data/cropped/aptos/grade1_4/000c1434d8d7.png"]
}
```

### 3) L3：Lesion Analysis 子任务拆分

FunBench 的 L3 是 lesion analysis，包含病灶识别、位置、大小、数量等。结合我们的字段，L3 不应混入 DR grade。

| 子任务 | 生成条件 | user 问法 | 输出字段 |
|---|---|---|---|
| L3-1 MA 显式识别 | `MA.present=true` 且强标注 | “是否可见微小红色圆点样病灶？” | `MA.present`, `count`, `source` |
| L3-2 HE 显式识别 | HE 阳性 | “是否可见暗红点片状/不规则斑块？” | `HE.present`, `count_bucket` |
| L3-3 EX 显式识别 | EX 阳性 | “是否可见亮黄色边界清楚沉积？” | `EX.present`, `area_bucket` |
| L3-4 SE 显式识别/拒答 | SE 阳性或被清洗 | “是否可见灰白棉絮状边界模糊病灶？” | `SE.present/false/unknown` |
| L3-5 lesion-only 闭集 | strong_L3 或 clean RetSAM L3 | “从 MA/HE/EX/SE/IRMA/NV 选择可见项。” | `lesions[]` |
| L3-6 病灶负担 | count/area 可用 | “数量或面积负担如何？” | `count_bucket`, `area_bucket` |

L3 推荐比例：显式属性题 60%-70%，闭集 lesion-only 20%-30%，unknown/abstain 约 10%。

#### L3-1/2/3 真实样本：强标注显式病灶

来源：`idrid::train::IDRiD_001`，图像 `data/idrid/images/train/IDRiD_001.jpg`。

真实字段：MA count=18，HE count=15，EX area=12027.0，source=`strong_mask_stage1_easy`。

```json
{
  "messages": [
    {"role": "system", "content": "你是眼底病灶识别助手。只做病灶分析，不输出 DR grade。先描述颜色、形态、边界和数量，再映射到病灶名。"},
    {"role": "user", "content": "<image> 请观察是否存在微小红色圆点、暗红不规则斑块、亮黄色边界清楚沉积或灰白棉絮样病灶。"},
    {"role": "assistant", "content": "【观察】可见多处微小红色规则圆点，另有多处暗红不规则斑块状病灶；同时可见较大面积、亮黄色、边界较清楚的多灶散在沉积。未见明确灰白棉絮样病灶或新生血管。\n\n【证据】MA present=true, count=18, source=strong_mask_stage1_easy；HE present=true, count=15, morphology=不规则斑块状；EX present=true, area=12027.0, pattern=多灶散在分布；SE/NV present=false。\n\n【结论】可见 MA、HE、EX；未见可靠 SE 或 NV。\n\n【JSON】\n{\"task\":\"L3_explicit_lesion_detection\",\"lesions\":[\"MA\",\"HE\",\"EX\"],\"absent\":[\"SE\",\"NV\"],\"source\":\"strong_mask_stage1_easy\"}"}
  ],
  "images": ["data/idrid/images/train/IDRiD_001.jpg"]
}
```

#### L3-4 真实样本：SE 清洗降级

来源：`aptos::all::000c1434d8d7`。RetSAM 原始 SE 有 1 个小灶，但清洗后因 `retsam_se_low_conf_or_tiny` 降级为阴性。

```json
{
  "messages": [
    {"role": "system", "content": "你是眼底病灶识别助手。被 cleaning_rule 降级的病灶不能作为阳性证据。"},
    {"role": "user", "content": "<image> 请判断是否可见灰白棉絮样、边界模糊的软性渗出样病灶。"},
    {"role": "assistant", "content": "【观察】灰白棉絮样、边界模糊的病灶证据不足。\n\n【证据】SE raw_present=true，但清洗后 SE present=false；source=cleaning_rule；suppressed_reason=retsam_se_low_conf_or_tiny。\n\n【结论】不支持可靠 SE 阳性。\n\n【JSON】\n{\"task\":\"L3_SE_abstain\",\"SE\":false,\"reason\":\"retsam_se_low_conf_or_tiny\",\"source\":\"validated_clean\"}"}
  ],
  "images": ["data/cropped/aptos/grade1_4/000c1434d8d7.png"]
}
```

### 4) L4：Disease Diagnosis 子任务拆分

FunBench 的 L4 是 disease diagnosis，需要整合 L3 病灶与解剖变化。L4 可以使用 grade，但必须引用证据；不能让模型只学 label prior。

| 子任务 | 生成条件 | user 问法 | 输出字段 |
|---|---|---|---|
| L4-1 Grade 0 正常/无可靠 DR 证据 | grade=0 | “是否有可靠 DR 病灶证据？” | `dr_grade=0`, `evidence_absent` |
| L4-2 Grade 1 模板 | grade=1 | “轻度 DR 如何解释？” | `MA=template_only` 或 strong MA |
| L4-3 Grade 2-4 证据绑定 | grade>=2 且 L3 证据可用 | “哪些病灶证据支持该分级？” | `dr_grade`, `evidence[]` |
| L4-4 证据冲突/拒答 | strong mask 与 grade 冲突 | “证据与标签是否一致？” | `needs_review` |

#### L4-2 真实样本：Grade 1 模板，不伪造 MA

来源：`aptos::all::0024cdab0c1e`。

```json
{
  "messages": [
    {"role": "system", "content": "你是眼底分级助手。RetSAM 不提供 MA；template_only 不能写成图像直接检出。"},
    {"role": "user", "content": "<image> 该图标注为轻度 DR，应如何给出证据解释？"},
    {"role": "assistant", "content": "【观察】清洗后未保留可靠 HE、EX 或 SE 阳性证据；当前事实层没有强标注 MA。\n\n【证据】dr_grade=1；MA present=template_only, source=grade_rule；HE/EX/SE present=false, source=grade_rule_override。\n\n【结论】该样本可按 Grade 1 的 MA-only 规则模板解释为轻度 DR，但不能表述为 RetSAM 检出 MA，也不能把 MA 当作 L3 可见病灶监督。\n\n【JSON】\n{\"task\":\"L4_grade1_template\",\"dr_grade\":1,\"MA\":\"template_only\",\"ma_source\":\"grade_rule\",\"forbid\":\"RetSAM_detected_MA\"}"}
  ],
  "images": ["data/cropped/aptos/grade1_4/0024cdab0c1e.png"]
}
```

#### L4-3 真实样本：Grade 3 证据绑定

来源：`aptos::all::0104b032c141`。

```json
{
  "messages": [
    {"role": "system", "content": "你是眼底分级助手。分级必须引用病灶证据；MA unknown 时不得编造 MA。"},
    {"role": "user", "content": "<image> 请先核查可见病灶证据，再判断该 DR 分级是否有依据。"},
    {"role": "assistant", "content": "【观察】可见少量暗红出血样病灶，同时可见大量亮黄色硬性渗出样病灶；未见可靠 SE 证据。\n\n【证据】HE present=true, count=3, area=157.0, source=validated_retsam；EX present=true, count=81, area=30470.0, source=validated_retsam；SE present=false；MA=unknown；dr_grade=3, grade_source=label。\n\n【结论】该样本监督分级为 DR Grade 3，主要解释证据是大量 EX 伴少量 HE。由于 MA unknown，不在解释中编造 MA。\n\n【JSON】\n{\"task\":\"L4_evidence_bound_grading\",\"dr_grade\":3,\"evidence\":[\"EX\",\"HE\"],\"MA\":\"unknown\",\"source\":\"validated_clean\"}"}
  ],
  "images": ["data/cropped/aptos/grade1_4/0104b032c141.png"]
}
```

### 5) 混合策略

训练阶段应先分任务，再混合：

| 阶段 | 数据 | 目的 |
|---|---|---|
| Stage A | L2-1/L2-2/L2-3 分开训练 | 建立基础解剖能力 |
| Stage B | L3-1 到 L3-6 分开训练 | 建立病灶属性与病灶概念映射 |
| Stage C | L4-1 到 L4-4 | 建立证据绑定诊断 |
| Stage D | 少量综合问答 | 检查读片链，但不作为主训练形式 |

第一版 SFT 建议比例：L2:L3:L4 = 2:5:3。L3 中显式属性题占 60%-70%，闭集 lesion-only 占 20%-30%，unknown/abstain 占约 10%。

## 第四部分：训练过程设计（固定 split → 子任务文件 → 分阶段混合 SFT）

### 1) 核心原则

本项目不建议“每个小问题单独微调一次”。L2-1、L2-2、L3-1、L4-1 等小问题应先拆成独立数据文件，便于统计、抽检和消融；真正训练时采用**分阶段混合 continued SFT**。这样既能利用 FunBench 的分层思想，又避免为每个能力点维护一套 LoRA，减少灾难性遗忘和小类过拟合。

比例只在**单个训练阶段内部**归一化。例如 Stage 1 的 L2 30% + L3 70% = 100%，Stage 2 的 L2 20% + L3 50% + L4 30% = 100%。不同阶段会从前一阶段 checkpoint 继续训练，训练集内部样本可以重复出现；但验证集和测试集必须按图像级隔离，不能泄露。

当前硬件为 GB10 统一内存 128G，Qwen3-VL-8B 建议使用 LoRA 或 QLoRA。优先保证 batch 组成、图像分辨率和梯度累积稳定；不建议一开始全量微调。

### 2) Stage 0：固定事实层与数据隔离

当前已生成：

- `data/fundus_validated/validated.jsonl`：原始统一事实层，用于审计。
- `data/fundus_validated/validated_clean.jsonl`：训练用清洗事实层。
- `data/fundus_validated/validated_clean.stats.json`：清洗统计。

后续样本生成只使用 `validated_clean.jsonl`。生成 SFT 前必须先固定 split：

| 规则 | 处理方式 |
|---|---|
| 图像级隔离 | 先按 `image_id`/原始文件路径/图像 hash 分组，再划分 train/val/test。 |
| 官方测试集隔离 | `idrid::test` 只用于验证或测试，不进入训练。 |
| 多问题同图 | 同一张图可生成多个 L2/L3/L4 训练问题，但这些问题必须全部留在同一个 split。 |
| 跨数据集重复图 | 对 APTOS、DDR、FGADR、IDRiD 生成 perceptual hash 或 SHA256，重复/近重复图按同一组处理。 |
| 规则调参 | 清洗阈值和模板规则只根据 train 统计和少量人工抽检确定，不能反复用 test 表现调规则。 |

### 3) Stage 0.5：按子任务生成文件

建议先分别生成以下 ShareGPT/多模态 SFT 文件：

- `fundus_l2_laterality_sft.jsonl`
- `fundus_l2_cdr_sft.jsonl`
- `fundus_l2_vessel_abstain_sft.jsonl`
- `fundus_l3_ma_he_ex_se_sft.jsonl`
- `fundus_l3_lesion_only_sft.jsonl`
- `fundus_l3_burden_sft.jsonl`
- `fundus_l4_evidence_grading_sft.jsonl`
- `fundus_l4_conflict_review_sft.jsonl`

每个文件只包含单一或相近子任务。这样做不是为了分别训练 8 个模型，而是为了后续按比例采样、质量抽检和 ablation。

### 4) Stage 1：视觉感知 SFT（L2 + L3）

第一阶段只训练解剖和病灶感知，不加入 DR grade。目标是先让模型学习“看见什么”和“如何描述证据”，避免一开始就学习 grade shortcut。

| 组成 | 阶段内比例 | 内容 |
|---|---:|---|
| L2 Anatomy | 30% | 左右眼、视盘视杯/CDR、血管指标可用性与拒答。 |
| L3 Lesion | 70% | MA/HE/EX/SE 显式病灶、lesion-only 闭集判断、病灶负担和 unknown/abstain。 |

L2 内部建议：laterality 40%，CDR 40%，vessel/QC abstain 20%。L3 内部建议：显式属性题 60%，lesion-only 闭集题 25%，病灶负担题 10%，unknown/abstain 5%。

### 5) Stage 2：证据绑定诊断 SFT（L2 + L3 + L4）

第二阶段从 Stage 1 checkpoint 继续训练，加入 L4 分级，但仍保留 L2/L3。目标是让模型把诊断建立在可见证据上，而不是只根据数据集先验输出 grade。

| 组成 | 阶段内比例 | 内容 |
|---|---:|---|
| L2 Anatomy | 20% | 保持 laterality、CDR、血管 QC 能力，防止遗忘。 |
| L3 Lesion | 50% | 保持病灶显式描述和强/弱监督来源区分。 |
| L4 Disease | 30% | DR Grade 0-4、referable DR、证据绑定分级、冲突核查。 |

L4 内部建议：Grade 0 约 15%，Grade 1/MA-only 约 20%，Grade 2-4 evidence-bound grading 约 55%，conflict/review 约 10%。Grade 1 必须明确区分 `template_only` MA 与强标注 MA；RetSAM 不提供 MA 时不得写成 RetSAM 检出。

### 6) Stage 3：平衡混合收敛

第三阶段使用更小学习率继续训练 0.5-1 个 epoch，作为最终混合收敛阶段。

| 组成 | 阶段内比例 | 目的 |
|---|---:|---|
| L2 Anatomy | 20% | 维持基础解剖和拒答稳定性。 |
| L3 Lesion | 55% | 维持病灶定位、属性、数量和可见性描述。 |
| L4 Disease | 25% | 保持诊断输出，但不让 grade 压过视觉证据。 |

Stage 3 不宜继续大量 upsample 高分级或少数类，否则模型容易把“罕见阳性”泛化成过度诊断。NV 当前只有 27 条，更适合作为“可见时识别/不可见时不编造”的小权重监督，不适合作为大比例目标。

### 7) Stage 4：DPO/MPO 可选

如果 Stage 2/3 后出现典型幻觉，可再做轻量 DPO/MPO。负样本只从真实错误点构造：

- 把 `template_only` MA 写成 RetSAM 检出。
- 把 `cleaning_rule` SE 写成阳性。
- `vessel_qc_flag=false` 时强判 A/V ratio 或 tortuosity。
- `coord_valid=false` 时输出精确距离/象限或 4-2-1 规则。
- L4 不引用 L3 证据直接给 grade。
- Grade 0/1 样本中编造 HE/EX/SE。

偏好对齐的规模应小于 SFT，重点修正规则性错误，不用于重新学习全部医学知识。

## 第五部分：评估设计（关键传统指标 + FunBench 分层消融 + 相似模型对比）

评估必须和训练 split 隔离。所有自动评估样本先按图像级 group split 固定，`idrid::test` 保持独立。评估报告不只看最终 DR grade，还要报告 L2、L3、L4 分层得分，以及“诊断是否引用了正确病灶证据”。这与 FunBench 的思路一致：将 fundus reading 拆成 anatomy perception、lesion analysis、disease diagnosis，并通过分层评估定位模型短板。

参考评估来源：

- [FunBench: Benchmarking Fundus Reading Skills of MLLMs, MICCAI 2025](https://papers.miccai.org/miccai-2025/0361-Paper2156.html)。采用 L1-L4 分层任务，并提出视觉编码器、语言侧知识提示、端到端整体三类评估模式。
- [LMOD: A Large Multimodal Ophthalmology Dataset and Benchmark for Large Vision-Language Models, NAACL Findings 2025](https://aclanthology.org/2025.findings-naacl.135/)。强调多模态、多任务、解剖理解和疾病诊断的联合评估。
- [OphthaMMBench, 2025](https://www.aimodels.fyi/papers/arxiv/novel-ophthalmic-benchmark-evaluating-multimodal-large-language)。覆盖眼底照片和 OCT，按 description、diagnosis、triaging、reasoning、treatment 等问题类型评价。
- [Can Multimodal Large Language Models Diagnose Diabetic Retinopathy from Fundus Photos? 2025](https://pubmed.ncbi.nlm.nih.gov/41030829/)。DR MLLM 定量评估通常报告 DR 分级准确率、referable DR、敏感性/特异性，并与 GPT-4o、Gemini、Claude、Qwen-VL 系列等通用 MLLM 比较。

### 1) 测试集设置

| 测试集 | 用途 | 注意事项 |
|---|---|---|
| IDRiD test 27 条强标注 | 小规模强监督 sanity check | 不进入训练；用于 MA/HE/EX/SE 与证据一致性评估。 |
| DDR lesion valid/test | 病灶泛化评估 | 不带 grade 时只评 L3，不评 L4 DR grade。 |
| APTOS/DDR grading held-out | 分级泛化评估 | 按图像 hash 隔离；只使用 grade 和清洗后的 RetSAM/模板证据。 |
| FGADR held-out | 强标注病灶与分级联合评估 | 注意 upsample 前后去重，评估只用 unique image。 |
| 人工抽检子集 | 开放式 CoT 质量评估 | 每个 level 抽样，检查是否编造病灶、是否错误引用来源。 |

### 2) 传统关键指标

传统指标只保留最重要的，避免把不可稳定测量的开放式文本指标堆得过多。

| 评估对象 | 任务 | 推荐指标 | 主要测试来源 |
|---|---|---|---|
| L2 laterality | 左/右眼 | Accuracy、Balanced Accuracy | RetSAM eye_side 有效样本 + 人工抽检 |
| L2 CDR | CDR 数值或分档 | MAE、Bucket Accuracy、Macro-F1 | RetSAM CDR 有效样本 + 人工抽检 |
| L3 lesion presence | MA/HE/EX/SE 是否存在 | Macro-F1、Sensitivity、Specificity、AUPRC | IDRiD/FGADR/DDR lesion 强标注优先 |
| L3 lesion burden | none/few/some/many | Macro-F1、Weighted-F1 | 强标注和清洗 RetSAM 样本 |
| L4 DR grade | Grade 0-4 | Accuracy、Macro-F1、Quadratic Weighted Kappa | APTOS/DDR grading held-out、IDRiD/FGADR |
| Referable DR | Grade >= 2 | AUROC、Sensitivity、Specificity、F1 | APTOS/DDR grading held-out |
| 证据一致性 | grade 与病灶证据是否矛盾 | Evidence Consistency Rate、Contradiction Rate | 全部 L4 评估样本 |
| 拒答稳定性 | unknown/invalid 时是否拒答 | Abstention Accuracy、Rule Violation Rate | vessel_qc false、MA unknown、cleaning_rule SE |

开放式 CoT 不用 BLEU/ROUGE 作为主指标。眼底任务更关心事实正确性和证据一致性，文本相似度只可作为辅助排版检查。

### 3) FunBench 风格分层评估

| Level | 子任务 | 本项目评估问题 | 推荐指标 |
|---|---|---|---|
| L2 Anatomy | laterality | 是否能依据 OD-黄斑相对位置判断左右眼 | Accuracy、Balanced Accuracy |
| L2 Anatomy | optic disc/cup | 是否能描述视盘、视杯并估计 CDR bucket | Bucket Accuracy、MAE |
| L2 Anatomy | vessel/QC | 血管质量不足时是否拒绝给 A/V 或 tortuosity | Abstention Accuracy、Violation Rate |
| L3 Lesion | explicit lesion | 是否能识别 MA/HE/EX/SE 的显式形态描述 | Macro-F1、Per-lesion F1 |
| L3 Lesion | lesion-only | 是否能在闭集内选择可见病灶，不输出 DR grade | Accuracy、Macro-F1、Format Validity |
| L3 Lesion | burden | 数量、面积、位置带分档是否正确 | Macro-F1、Weighted-F1 |
| L3 Lesion | unknown/abstain | RetSAM 不支持 MA 或低置信 SE 时是否不编造 | Rule Violation Rate |
| L4 Disease | evidence-bound grade | 是否基于 L3 证据给出 DR grade | Accuracy、Macro-F1、QWK |
| L4 Disease | referable DR | 是否识别 Grade >= 2 | AUROC、Sensitivity、Specificity |
| L4 Disease | conflict review | 标签与病灶证据冲突时是否提示需复核 | Conflict Detection Accuracy |

建议报告 `L2_mean`、`L3_mean`、`L4_mean` 和 `Overall_mean`，但论文/报告主表不要只给 Overall。若 L4 提升而 L3 下降，说明模型可能在学分级捷径；若 L3 提升而 L4 不提升，说明证据到诊断的映射仍不足。

### 4) 消融实验

| 实验 | 对照 | 目的 |
|---|---|---|
| Full pipeline | 使用 Stage 1+2+3 全流程 | 主结果。 |
| No Stage 1 | 直接 L2/L3/L4 混合 SFT | 检查先学视觉证据是否必要。 |
| No L2 replay | Stage 2/3 不混入 L2 | 检查 laterality/CDR 是否遗忘。 |
| No L3 replay | Stage 2/3 不混入 L3 | 检查 L4 是否退化成 grade classifier。 |
| No explicit lesion CoT | L3 只保留 lesion-only 标签 | 检查显式病灶描述是否提升视觉 grounding。 |
| No strong-mask SFT | 移除 IDRiD/FGADR 强标注派生样本 | 检查强标注对 MA/HE/EX/SE 的贡献。 |
| RetSAM raw vs clean | 使用未清洗 RetSAM 与清洗后 RetSAM | 验证清洗规则是否降低幻觉和冲突。 |
| Grade-only L4 | L4 答案只给 grade，不给证据 | 检查证据绑定对一致性的影响。 |
| With/without DPO/MPO | Stage 3 checkpoint vs Stage 4 | 检查偏好对齐是否降低规则性错误。 |

消融主看三类变化：L3 病灶 F1、L4 QWK、Evidence Consistency Rate。若某个设置只提高 DR accuracy 但显著降低证据一致性，不应作为最终方案。

### 5) 相似模型与基线对比

模型对比分三组，避免把“通用闭源模型能力”和“本项目可部署能力”混在一个结论里。

| 组别 | 模型/方法 | 评估方式 |
|---|---|---|
| 传统监督基线 | EfficientNet/ConvNeXt/ViT DR grade classifier | 只评 DR grade、referable DR，不评开放式 CoT。 |
| 通用 MLLM 零样本/少样本 | GPT-4o、Gemini、Claude、Qwen2.5-VL/Qwen3-VL 原始模型 | 使用同一套 L2/L3/L4 prompt，评结构化答案和人工抽检。 |
| 领域 MLLM/医学 MLLM | LLaVA-Med、Med-Flamingo、BiomedGPT/眼科相关开源模型（可获得时） | 同一测试集、同一输出格式；无法输出结构化 JSON 时做人工或规则解析。 |
| 本项目模型 | Qwen3-VL-8B + LoRA/QLoRA | 报告 Stage 1、Stage 2、Stage 3、可选 Stage 4 的逐阶段结果。 |

主要表格建议：

| Model | L2 mean | L3 Macro-F1 | L4 QWK | Referable Sens/Spec | Evidence Consistency | Rule Violation |
|---|---:|---:|---:|---:|---:|---:|
| Qwen3-VL-8B zero-shot | - | - | - | - | - | - |
| Qwen3-VL-8B Stage 1 | - | - | 不评 | 不评 | - | - |
| Qwen3-VL-8B Stage 2 | - | - | - | - | - | - |
| Qwen3-VL-8B Stage 3 | - | - | - | - | - | - |
| Qwen3-VL-8B Stage 4 optional | - | - | - | - | - | - |

### 6) 人工评估要点

人工评估只抽少量但必须覆盖高风险类型：

- Grade 0/1：是否编造 HE/EX/SE，是否把 `template_only` MA 写成可见强证据。
- Grade 2-4：是否引用真实可见病灶，而不是只复述标签。
- MA unknown：是否明确“不评价 MA”。
- SE cleaning_rule：是否把低置信/小面积 SE 当成阳性。
- vessel_qc false：是否拒绝输出 A/V ratio、tortuosity 等不可靠指标。
- IDRiD/FGADR 强标注样本：是否保留 MA/HE/EX/SE 的来源和显式形态描述。
