# 忠实 DR 分诊 — 多厂家基线对比(本轮)与下一轮(Lingshu-I-8B 微调)方案

更新:2026-06-23　|　记录仓:`ZGHL/fundus-qwen3vl-project @ stage1_5_mask_grounded`

本轮目标:在**统一口径(from-audit:审计→透明映射→tier,忠实度恒 1.0)**下,把我们的最优感知模型 **v3se-270** 与 **5 个同体量、多厂家基模(2 通用 + 3 医学)** 在三个层面对比:Stage-1 病灶感知、Stage-2 内部分诊、Stage-2 外部(Messidor-2)分诊。

---

## 1. 实验详情

### 1.(a) 真实数据

| 用途 | 数据集来源 | 规模 |
|---|---|---|
| **Stage-1 感知测试** | FGADR 472 / DDR 168 / IDRiD 24 / **aptos 444**(grade-0 负样本) | 1108 单病灶样本(Adapter1 未见、图像互斥) |
| **Stage-2 分级训练** | grounded(FGADR/IDRiD 真 mask)2077 + grade-derived(aptos/ddr g0/g1)2004 | 4081 |
| **Stage-2 内部测试** | FGADR 152 / IDRiD 39 / DDR 33 / aptos 76(grounded 191 + grade-derived 109) | 300 |
| **Stage-2 外部测试** | **Messidor-2**(均衡 30/grade,0–4) | 150 |

**病灶**:MA(微动脉瘤)/HE(出血)/EX(硬性渗出)/SE(软性渗出);IRMA/NV 数据稀缺(159/49)→ 永久弃权。
**tier**:由数据拟合的 presence→tier 透明映射给出;G2/G3 视觉不可分边界 → 校准弃权。

**⚠️ 数据泄漏分析(关键,决定用哪个测试集下结论):**
- **内部测试集对多方都不干净**:
  - 我们的模型:Stage-2 内部 300 中 **172/297** 图在我们感知训练里见过。
  - Lingshu-I-8B / QoQ-Med(均训练含 **APTOS 2019**):在内部集的 **aptos 部分**泄漏(Stage-1 ~40%、Stage-2 ~25%)。
- **Messidor-2 对所有模型都干净**:Lingshu 命名的眼底数据为 BRESET/PAPILA/EyePACS/APTOS,**不含 Messidor-2**;我们也未训练 Messidor-2。
- **结论:跨模型对比一律以 Messidor-2 外部为准;内部集仅作我们自己的选模/消融并标注泄漏。**

### 1.(b) 对比模型背景(它们是怎么造出来的)

**医学专用 VLM(3 个):**

- **Lingshu-I-8B**(阿里达摩,arXiv 2506.07044,2025-06)。底座 **InternVL3**(InternViT 视觉塔 + Qwen2.5-7B 语言模型)。在 **~505 万医学样本**(375 万真实开源 + 130 万合成)上做多模态医学训练,覆盖 12+ 模态(含 **眼底**:BRESET/PAPILA/EyePACS/**APTOS**)。<10B 医学 SOTA(医学多模态均分 61.8)。注:论文主力 **Lingshu-7B 是 Qwen2.5-VL 版**;本地这个 "-I-8B" 是**基于 InternVL3 的变体**(同一训练流程)。→ 与通用 InternVL3.5-8B 天然形成"通用 vs 医学微调"对照。**训练含 APTOS DR 数据**(故在 DR 上有备而来,非天真零样本)。

- **QoQ-Med-VL-7B**(arXiv 2506.00711,2025-06)。底座 **Qwen2.5-VL**。**首个开源"通才"临床基础模型**,跨医学图像 + 时序信号 + 文本联合推理(1D/2D/3D)。在 **2.61M 指令对、9 个临床域**(基于 CLIMB 数据集)上训练;训练方法是新颖的 **DRPO(Domain-aware Relative Policy Optimization,GRPO 的域感知强化学习变体)**,按域稀有度/模态难度缩放奖励。→ 是**跨 9 域的通才**(非眼底病灶专精),解释了它在眼底病灶上偏弱(recall 0.23)。

- **MedGemma-4B-IT**(Google,arXiv 2507.05201,2025-07)。底座 **Gemma 3 4B**,视觉编码器 **MedSigLIP**(医学版 SigLIP,固定 896×896)。在医学图文上继续训练,预训练含 **胸片/病理/皮肤镜/眼底**,任务以医学 VQA 为主。→ **通用医学 + 仅 4B**(其余为 7-8B),非眼底病灶专精,在强制病灶判定下极度保守(见 §1.c 核查)。

**通用 VLM(2 个,作"非医学底座"对照):**

- **Qwen3-VL-8B-Instruct**(阿里 Qwen 团队,2025)。通用多模态大模型,Qwen3-VL 架构(原生动态分辨率 ViT + Qwen3 LLM + M-RoPE)。
- **InternVL3.5-8B**(上海 AI Lab / OpenGVLab,2025)。InternVL 系列(InternViT + LLM,pixel-shuffle + 动态切片)。**与 Lingshu-I-8B 同族**(Lingshu-I = 其医学训练版),构成干净的"通用 InternVL vs 医学 InternVL"对照。

**一句话**:覆盖 2 个通用 + 3 个医学,多厂家(阿里/达摩/上海AILab/Google),且训练数据量从 2.61M~5.05M 不等——与我们 **~1.3 万**任务样本形成数据效率对照。

### 1.(c) 最终结果、结论与分析

#### 第一阶段:病灶感知(present/absent,1108 图像互斥测试集)

**(i) 我们的能力构建阶梯**(从基模到最终感知模型,宏平均):

| 模型 | Recall | Spec | F1 | 说明 |
|---|--:|--:|--:|---|
| 基模(Qwen3-VL-8B 零样本) | 0.335 | 0.941 | 0.411 | 保守、漏检多 |
| Adapter1(早期感知) | 0.965 | 0.213 | 0.528 | 严重过报(高召回/低特异度) |
| v3 ckpt-400(特异度修正) | 0.844 | 0.606 | 0.616 | 过报基本修复 |
| **v3se-270(SE 专项再平衡,最终)** | 0.69 | **0.86** | **0.68** | 最均衡 |

> SE 专项修复:把 SE 误报从 143→11(特异度 0.40→0.95),SE F1 0.34→0.43,且 MA/HE/EX 不降反升(保本)。

**(ii) 最终感知模型(v3se-270)逐病灶**:MA F1 0.736(Rec 0.91/Spec 0.41)、HE 0.797(0.71/0.81)、EX 0.772(0.81/0.82)、SE 0.431(0.35/0.95)。HE/EX 接地稳;MA 高召回、SE 已不过报。

**(iii) 第一阶段 vs 5 基模(宏平均 F1 / Recall / Spec)**:

| 模型 | Recall | Spec | F1 |
|---|--:|--:|--:|
| Qwen3-VL-8B 通用 | 0.33 | 0.94 | 0.41 |
| InternVL3.5-8B 通用 | 0.48 | 0.93 | 0.57 |
| Lingshu-I-8B 医学 | 0.86 | 0.72 | 0.67 |
| QoQ-Med-VL-7B 医学 | 0.23 | 0.92 | 0.29 |
| MedGemma-4B 医学 | 0.00 | 1.00 | 0.00 |
| **★ 我们 v3se-270** | 0.69 | 0.86 | **0.68** |

**Stage-1 结论**:我们的感知 **F1 0.68 最高**且**最均衡**;Lingshu-I 0.67 紧随但过报(Spec 0.72);通用基模偏保守;QoQ-Med/MedGemma 作为"通才/小模型医学 VLM"在细粒度眼底病灶上明显偏弱(见 §1.c MedGemma 核查)。

#### 第二阶段:忠实分诊

**5 基模 + 我们(三阶段,同测试集,from-audit,忠实度恒 1.0)**

| 模型 | S1 感知 F1(Rec/Spec) | S2 内部 QWK(Sens/Spec) | **S2 外部 Messidor QWK(Sens/Spec)** | 重病召回(外) |
|---|---|---|---|--:|
| Qwen3-VL-8B 通用 | 0.41 (0.33/0.94) | 0.54 (0.70/0.90) | 0.57 (0.64/0.88) | 0.80 |
| InternVL3.5-8B 通用 | 0.57 (0.48/0.93) | 0.52 (0.68/0.84) | 0.48 (0.54/0.93) | 0.73 |
| Lingshu-I-8B 医学 | 0.67 (0.86/0.72) | 0.61 (0.92/0.56) | 0.63 (0.84/0.65) | 0.97 |
| QoQ-Med-VL-7B 医学 | 0.29 (0.23/0.92) | 0.21 (0.48/0.76) | 0.33 (0.42/0.90) | 0.55 |
| MedGemma-4B 医学 | 0.00 (0.00/1.00) | 0.01 (0.01/1.00) | 0.00 (0.00/1.00) | 0.00 |
| **★ 我们 v3se-270** | **0.68 (0.69/0.86)** | 0.61 (0.87/0.66) | **0.64 (0.76/0.77)** | 0.90 |
| *l4_v3(旧·不忠实,参照)* | — | — | *0.695 (0.87/0.92)* | *0.98* |

**结论:**
1. **干净外部(Messidor-2):我们 QWK 0.64 第一**,Lingshu 0.63 第二;且**我们最均衡**(Sens/Spec 0.76/0.77 vs Lingshu 0.84/0.65,后者过度转诊)。感知 F1 我们 0.68 亦第一。
2. **5 个基模里仅 Lingshu 真正能打**;两个通用中等偏下;**两个新医学模型反而最弱**(QoQ-Med 过保守 recall 0.23;MedGemma-4B 几乎全判 absent)。其低分经原始输出核验为**真实**(非解析问题)。
3. **数据效率**:Lingshu 用 **~505 万**医学样本(含百万级 DR 图);我们整套仅 **~1.3 万** mask 接地样本 + 透明映射,**少 2–3 个数量级**,仍在干净外部上持平略胜。
4. **忠实性**:所有模型走同一透明映射 → 忠实度恒 1.0;但 Lingshu/基模本身无忠实约束、无校准弃权、不给可核对证据链——**这是我们框架独有的**。
5. **忠实代价已量化**:不忠实的旧 l4_v3 外部 QWK 0.695 > 忠实天花板 0.688 → 证明其高分依赖不可忠实感知的捷径(NV/IRMA)。我们忠实地做到 0.64,差距即"忠实的定价"。

**一句话**:仅用百分之一的数据、且额外提供忠实证据链与校准弃权,我们在干净外部上**整体最优、最均衡**,唯一接近者是喂了 500 万数据的 Lingshu(我们还略胜且更稳)。

**产物**:`eval/FIVE_BASELINES_VS_OURS.md`、`eval/PERCEPTION_VS_BASELINES.md`、`eval/SE_FIX_V3SE2_RESULTS.md`;预测在 `preds/`,模型/数据在 R2。

---

## 2. 下一轮实验设计

### 2.(a) 基于 Lingshu-I-8B 的微调实验方案

**动机**:Lingshu-I-8B 是 <10B 医学 SOTA 底座但**无忠实性/无弃权**。把它当审计底座、套我们的忠实框架做 Stage-2 SFT,验证 **"框架不挑底座、能把 SOTA 医学模型升级成忠实可解释系统"**——这是很强的一个论点。

**方法**(只做 Stage-2 忠实分级 SFT,不重训感知——Lingshu 感知已强):
- 起点:`models/Lingshu-I-8B`(InternVL3,InternViT + Qwen2.5-7B),**新 LoRA**,`template: intern_vl`。
- 训练数据:`stage2_grade_train`(4081,完整忠实 CoT:审计→决策→分级 JSON)。
- 配置:LoRA rank16/alpha32、cutoff 2304、image 262144、batch 2×8、LR 5e-6、cosine。
- **兼容性 + 测速已 smoke 验证**(2026-06-23):LLaMA-Factory + intern_vl 模板可训,**~72 s/step** → **1 epoch ≈ 5.1h,2 epochs ≈ 10.2h**(GB10)。
- 推理评估放**新推理容器** `gb10_vllm_infer`(InternVL 推理正常)。

**评估口径(防复合污染)**:
- **只在 Messidor-2 外部评**(对 Lingshu 训练 + 我们训练数据**都干净**)。
- 内部集不用于跨模型结论(aptos 对 Lingshu 与我们双重泄漏)。
- 训练数据与 Lingshu 训练有 **aptos 部分重叠**——对"训练"无害(重看训练数据不是问题),且我们加入的是 mask 接地 + 忠实 CoT 这类 Lingshu **没有的新监督**;只要评估用 Messidor-2 即干净。

**预期(务实)**:
- ✅ 交付:一个 **"SOTA 医学底座 + 原生忠实单模型"**,单次前向出可核对报告 + 校准弃权。
- ⚠️ **不指望刷分**:Lingshu 零样本+映射已 0.63、逼近忠实天花板;且我们在自有底座上 SFT 曾在外部**过拟合掉点**(0.64→0.54)——Lingshu 底座 SFT 同样可能持平或略降。价值在"框架可移植 + 可部署忠实",非精度 SOTA。

**成功判据**:Messidor-2 上,忠实度=1.0(构造),referable/QWK 不显著低于其零样本+映射(0.63),并产出完整忠实 CoT 报告。

**可选扩展**:补 Qwen2.5-VL 版 **Lingshu-7B**,覆盖 Lingshu 两个变体(InternVL3 / Qwen2.5-VL),论文更完备(QoQ-Med 已覆盖 qwen2.5-vl 医学一类)。

---

*脚本:`run_vlm_perception.py`(多厂家零样本推理)、`score_stage2_from_audit.py`(审计→映射→tier 评分)、`score_perception_sweep.py`、`run_new_baselines_robust.sh`(GPU 自愈批跑)。所有基线预测已归档。*

---

# 补充材料(导师汇报版)

## A. 明确的训练目标

**总目标**:把通用/医学视觉语言模型,改造成一个**忠实(faithful)、可解释、会校准弃权**的 DR 分诊系统——每个分级都由**可验证的病灶证据**支撑,且在视觉证据不可判定处**显式弃权**而非猜测。分两阶段:

- **第一阶段(病灶感知)训练目标**:对单张眼底图,逐个判断 4 种可靠病灶 **MA/HE/EX/SE 的 present/absent**(像素 mask 接地);**强制只依据可见证据、不从 DR 等级反推**;对数据稀缺、视觉不可靠的 **IRMA/NV 永久弃权**。产物 = 可靠的"病灶审计"。
- **第二阶段(可解释分诊)训练目标**:在第一阶段之上,学会输出一条**忠实思维链**:`[病灶审计] → [决策路径] → [结论] → [JSON]`;**分级 = 病灶组合的透明函数**;遇到视觉上无法区分"中度 vs 重度"的模式时,输出**校准弃权档**。

**系统提示(真实,内部 300 测试集所用)**:
> *You are a diabetic retinopathy triage assistant. Audit MA/HE/EX/SE from directly visible evidence; IRMA and NV are not visually reliable and must be abstained (never claim to see them). Using only verifiable present/absent evidence, assign a DR tier; when visible evidence cannot separate moderate from severe NPDR, say so (severity indeterminate) rather than guessing. Do not use lesion counts.*

## B. 五档分诊的具体定义

| 档位 | 定义(基于可验证病灶组合) | referable | 临床动作 |
|---|---|---|---|
| **No-DR** | 无任何可靠病灶(MA/HE/EX/SE 全阴) | 否 | 常规随访 |
| **Mild** | 仅微动脉瘤(MA-only) | 否 | 监测 |
| **Moderate** | 中度病灶负担(如 HE / EX / 多病灶组合,但未达重度模式) | **是** | 转诊 |
| **Severe** | 重度负担模式(HE+EX 共存等) | **是** | 转诊,优先 |
| **Mod-or-Severe-indeterminate(校准弃权)** | 证据确属可转诊,但**可见证据无法区分中度 vs 重度/PDR**(需逐象限出血计数/IRMA/NV/静脉串珠,均不可靠) | **是** | 转诊 + 标记需人工细分 |

**病灶组合 → 档位的透明映射(数据拟合,16 模式,非手写)**:
```
none->No-DR  MA->Mild  HE/HESE/MASE/MAEX/MAEXSE/MAHE/SE->Moderate
HEEX/HEEXSE/MAHEEXSE->Severe   EX/EXSE/MAHESE/MAHEEX->Mod-or-Severe-indeterminate
```
(IRMA/NV 永不进入审计;无 count/area。)

## C. 真实 Stage-2 输入 / 输出样例(取自本地真实数据)

**统一输入(User)**:`<image>\nAudit MA/HE/EX/SE (abstain on IRMA/NV), then assign the DR tier. Output: [Lesion Audit] -> [Decision Path] -> [Conclusion] -> [JSON].`

**样例 1 — No-DR**(真实训练样本,`cropped/aptos/grade0/18ce0cdc473d.png`,clinical_grade=0):
```
[Lesion Audit]
- MA: absent. / - HE: absent. / - EX: absent. / - SE: absent.
- IRMA: not visually assessable — abstained. / - NV: not visually assessable — abstained.
[Decision Path] Step1 no reliable DR lesion (MA/HE/EX/SE) visible -> No-DR.
[Conclusion] DR tier = No-DR; referable_dr = no; evidence is directly visible.
[JSON] {"dr_tier":"No-DR","referable_dr":false,"lesions_present":[],"abstained":["IRMA","NV"],"severity_indeterminate":false,"evidence_limited":false}
```

**样例 2 — Mild**(`cropped/aptos/grade1_4/29b52f64d2db.png`,clinical_grade=1):
```
[Lesion Audit] - MA: present — tiny round red dot-like abnormalities. - HE/EX/SE: absent. - IRMA/NV: abstained.
[Decision Path] Step1 reliable DR lesion present: MA. Step2 only MA, no HE/EX -> Mild.
[Conclusion] DR tier = Mild; referable_dr = no.
[JSON] {"dr_tier":"Mild","referable_dr":false,"lesions_present":["MA"],"abstained":["IRMA","NV"],...}
```

**样例 3 — Moderate**(`FGADR/Seg-set/Original_Images/0099_1.png`,clinical_grade=2):
```
[Lesion Audit] - MA: present. - HE: present. - EX/SE: absent. - IRMA/NV: abstained.
[Decision Path] Step1 present: MA, HE. Step2 HE/EX present -> referable. Step3 pattern consistent with moderate NPDR -> Moderate.
[JSON] {"dr_tier":"Moderate","referable_dr":true,"lesions_present":["MA","HE"],...}
```

**样例 4 — Mod-or-Severe-indeterminate(弃权,核心)**(`FGADR/Seg-set/Original_Images/1490_1.png`,clinical_grade=2):
```
[Lesion Audit] - MA: present. - HE: present. - EX: present. - SE: absent. - IRMA/NV: abstained.
[Decision Path] Step1 present: MA, HE, EX. Step2 HE/EX present -> referable.
  Step3 this lesion pattern does not separate moderate from severe NPDR on visible evidence
  (distinguishing them needs per-quadrant hemorrhage counts / IRMA / NV, not visually assessable)
  -> referable, severity indeterminate.
[Conclusion] DR tier = Mod-or-Severe-indeterminate; referable_dr = yes; severity indeterminate; evidence_limited.
[JSON] {"dr_tier":"Mod-or-Severe-indeterminate","referable_dr":true,"lesions_present":["MA","HE","EX"],
        "abstained":["IRMA","NV"],"severity_indeterminate":true,"evidence_limited":true}
```

**样例 5 — 真实模型推理输出**(我们的 Stage-2 SFT 模型在 **Messidor-2 外部图**上的实际生成,非训练标签):
```
[Lesion Audit] - MA: present — tiny round red dots. - HE: present — dark-red blot/flame hemorrhages.
- EX: present — bright yellow-white sharp-bordered deposits. - SE: absent. - IRMA/NV: abstained.
[Decision Path] Step1 present: MA, HE, EX. Step2 HE/EX -> referable.
  Step3 cannot separate moderate vs severe on visible evidence -> referable, severity indeterminate.
[Conclusion] referable_dr = yes; severity indeterminate; evidence_limited.
[JSON] {"dr_tier":"Mod-or-Severe-indeterminate","referable_dr":true,"lesions_present":["MA","HE","EX"],...}
```
→ 模型**自己**产出可核对的证据链 + 在中/重边界**主动弃权**,这就是"忠实可解释"的最终形态。

## D. 贡献分析与论文故事

### D.1 我们目前的贡献(基于现有结果)
1. **方法**:`from-audit` 忠实解耦框架——模型只产可验证审计,分级由**透明映射**算出 → **忠实 by construction**;并在视觉不可分边界**校准弃权**。SFT 后我们的模型可**原生**产出该忠实链(自由生成忠实度 0.43→~1.0)。
2. **评估概念**:提出 **"忠实天花板 0.688"**(仅凭可见病灶证据、4 类分级的准确率上限);并证明 SOTA 级的不忠实模型(旧 l4_v3 外部 QWK 0.695)**越过该天花板** → 实证其高分依赖**不可忠实感知的捷径**(NV/IRMA;G4 召回仅 0.045)。
3. **第一阶段地基**:把病灶感知从基模的 F1 0.41 提到 0.68、**最均衡**(含 SE 专项修复:误报 143→11);跨 5 个同体量多厂家模型对比中**感知 F1 最高**。
4. **数据效率**:全套仅 ~1.3 万任务样本 + 透明规则,在干净外部上与喂了 ~505 万医学样本的 Lingshu 持平略胜。
5. **诚实的负面发现**(本身是贡献):计数/面积不可泛化、决策规则非杠杆、端到端微调分级头在外部反而过拟合掉点、SE 存在特征天花板。

### D.2 论文故事应该怎么讲(定位)
- **不是"打榜/刷 SOTA"论文**(忠实分级的精度由天花板 0.688 封顶,刷过它=不忠实)。
- **是"可信/可解释医疗 AI + 评估方法学"论文**,核心论点:
  > **"DR 分级的高准确率,有多少是真正基于可见病灶证据,有多少是靠不可验证的捷径?我们用一个忠实 by construction、会校准弃权的框架,把'可忠实达到的精度'界定出来(忠实天花板 0.688),并证明 SOTA 模型的超额精度来自不可信证据;我们的框架以百分之一的数据匹配该上限,同时提供可核对的证据链与安全弃权。"**
- **公平对比设计(回应循环质疑)**:
  - *消融 1(底座)*:固定我们的框架、换底座(from-audit 表)→ 比"谁的审计更好"。
  - *消融 2(框架)*:同一底座下 **黑箱原生分级(框架关) vs 我们框架(框架开)** → 比框架贡献;并从黑箱的自由推理中检测**原生不忠实**(是否援引 NV/IRMA)。
  - 公共指标用 **referable 灵敏度/特异度 + 重病召回**(二元、跨框架对称),避免用我们的五档去套黑箱。
- **目标会议**:AJCAI 2026(CORE B),医学 AI 受欢迎、不强求 SOTA → **故事成立、命中率中上**。需补强:外部验证扩到全量 Messidor-2(~1744)、related work 精确定位"忠实/selective-prediction"。

### D.3 诚实的局限(论文须写明)
- 忠实精度低于不忠实 SOTA(代价);感知较强医学底座(Lingshu)仅小幅领先;SE/MA 受特征与数据上限;内部测试集对多模型有不同泄漏(故以外部 Messidor-2 为准)。
