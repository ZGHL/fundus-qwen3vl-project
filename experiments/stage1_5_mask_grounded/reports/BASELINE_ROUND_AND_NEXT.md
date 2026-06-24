# 忠实可解释 DR 分诊 — 实验进展汇报

更新:2026-06-24　|　记录仓:`ZGHL/fundus-qwen3vl-project @ stage1_5_mask_grounded`

---

## 1. 研究目标与系统概述

把通用/医学视觉语言模型,改造成一个**忠实(faithful)、可解释、会校准弃权**的糖尿病视网膜病变(DR)分诊系统:**每个分级都由可验证的病灶证据支撑,且在视觉证据不可判定处显式弃权而非猜测**。分两阶段:

- **第一阶段 — 病灶感知**:对单图逐个判断 4 种可靠病灶 **MA/HE/EX/SE 的有/无**(像素 mask 接地),只依据可见证据;对数据稀缺、视觉不可靠的 **IRMA/NV 永久弃权**。产物 = 可靠的"病灶审计"。
- **第二阶段 — 可解释分诊**:在审计之上输出忠实思维链 `[病灶审计] → [决策路径] → [结论] → [JSON]`;**分级 = 病灶组合的透明函数**;视觉无法区分中/重度时输出**校准弃权档**。

**系统提示(真实)**:*"You are a diabetic retinopathy triage assistant. Audit MA/HE/EX/SE from directly visible evidence; IRMA and NV are not visually reliable and must be abstained (never claim to see them). Using only verifiable present/absent evidence, assign a DR tier; when visible evidence cannot separate moderate from severe NPDR, say so (severity indeterminate) rather than guessing. Do not use lesion counts."*

---

## 2. 实验基础信息

### 2.1 数据

| 用途 | 来源 | 规模 |
|---|---|---|
| Stage-1 感知测试 | FGADR 472 / DDR 168 / IDRiD 24 / aptos 444(g0 负样本) | 1108 单病灶样本(图像互斥) |
| Stage-2 分级训练 | grounded(FGADR/IDRiD 真 mask)2077 + grade-derived(aptos/ddr)2004 | 4081 |
| Stage-2 内部测试 | FGADR 152 / IDRiD 39 / DDR 33 / aptos 76 | 300 |
| **Stage-2 外部测试** | **Messidor-2**(均衡 30/grade) | 150 |

> 病灶:MA(微动脉瘤)/HE(出血)/EX(硬性渗出)/SE(软性渗出);IRMA/NV 数据稀缺(159/49)→ 永久弃权。
> **数据洁净度**:内部测试集对多方都不干净(我们 172/297 见过;Lingshu/QoQ 训练含 APTOS → aptos 部分泄漏)。**Messidor-2 对所有模型都干净 → 跨模型结论一律以 Messidor-2 为准。**

### 2.2 对比模型(同体量,2 通用 + 3 医学,多厂家)

- **Qwen3-VL-8B-Instruct**(阿里,通用底座,**也是我们框架的底座**)。
- **InternVL3.5-8B**(上海 AI Lab,通用;与 Lingshu 同族,构成"通用 vs 医学"对照)。
- **Lingshu-I-8B**(达摩,医学;InternVL3 底座,~505 万医学样本训练,含眼底 BRESET/PAPILA/EyePACS/**APTOS**)。
- **QoQ-Med-VL-7B**(医学通才,Qwen2.5-VL 底座,2.61M 跨 9 临床域,DRPO 强化学习)。
- **MedGemma-4B-IT**(Google,医学,Gemma3-4B + MedSigLIP)。

---

## 3. 五档分诊与透明映射(框架核心)

### 3.1 五档定义及与 ICDR 的关系

我们的分诊档是 **ICDR 国际临床分级在"仅可验证证据"上的重参数化**:凡是 ICDR 中**必须依赖不可验证证据**(逐象限出血计数、IRMA、NV)才能区分的等级,统一折叠进**弃权档**。

| 我们的档位 | 定义(基于可验证病灶) | 对应 ICDR | referable | 临床动作 |
|---|---|:--:|:--:|---|
| **No-DR** | MA/HE/EX/SE 全阴 | 0 | 否 | 常规随访 |
| **Mild** | 仅 MA | 1 | 否 | 监测 |
| **Moderate** | 中度病灶负担(HE / 多病灶组合) | 2 | **是** | 转诊 |
| **Severe** | 重度模式(HE+EX 共存等) | 3(及 4/PDR*) | **是** | 转诊,优先 |
| **Mod-or-Severe-indeterminate** | 证据确属可转诊,但**可见证据无法区分中度 vs 重度/PDR** | 2/3 边界 | **是** | 转诊 + 标记需人工细分 |

> *PDR(ICDR 4)的金标准是 NV,而 NV 我们弃权,故 PDR 不单独成档,其可见出血/渗出证据归入 Severe/可转诊。

### 3.2 病灶组合 → 档位 的透明映射(数据拟合,16 模式,非手写规则)

| 病灶组合 | 档位 | 病灶组合 | 档位 |
|---|---|---|---|
| none | No-DR | MA | Mild |
| HE / HESE / SE | Moderate | MASE / MAEX / MAEXSE / MAHE | Moderate |
| HEEX / HEEXSE / MAHEEXSE | Severe | EX / EXSE / MAHESE / MAHEEX | **弃权(indeterminate)** |

**弃权触发规则**:(1) IRMA/NV **永远**弃权(从不进入审计);(2) 病灶组合落在 EX / EXSE / MAHESE / MAHEEX 时,中/重度在可见证据上不可分 → 触发**校准弃权**(仍判定为可转诊)。
**与 ICDR 的关系**:档位与 ICDR 一一对应,差异仅在于——ICDR 用计数/IRMA/NV 强行区分中/重度,我们**拒绝用不可验证证据做此区分**,改为透明弃权。

---

## 4. 实验结果(以干净外部 Messidor-2 为准)

### 4.1 第一阶段:病灶感知(1108 图像互斥测试集,宏平均)

**能力构建阶梯**:base Qwen3-VL-8B(R0.34/S0.94/**F0.41**)→ Adapter1 过报(0.97/0.21/0.53)→ v3-400 修特异度(0.84/0.61/0.62)→ **v3se-270 最终(0.69/0.86/F0.68)**。
> SE 专项修复:误报 143→11(Spec 0.40→0.95),且 MA/HE/EX 不降反升。

**vs 5 基模(宏 F1)**:我们 **0.68 最高**;Lingshu 0.67(过报,Spec 0.72);InternVL 0.57;Qwen 0.41;QoQ 0.29;MedGemma 0.00(4B,极保守,经核验为真实非解析问题)。

### 4.2 第二阶段:忠实分诊(from-audit,**固定我们框架、只换底座**)

> **口径说明**:下表对所有模型施加**同一透明映射**(审计→映射→tier),故忠实度对所有模型恒为 1.0——这是**协议的构造性属性,非模型自身能力**。它消融的是**"谁的感知/审计更好"**(底座消融),**不是**"有无框架"。各模型原生黑箱分级的 apple-to-apple 精度对比正在补充中。

| 模型 | S1 感知 F1 | S2 外部 QWK | RefSens/Spec | 重病召回 |
|---|--:|--:|--:|--:|
| Qwen3-VL-8B 通用 | 0.41 | 0.57 | 0.64/0.88 | 0.80 |
| InternVL3.5-8B 通用 | 0.57 | 0.48 | 0.54/0.93 | 0.73 |
| Lingshu-I-8B 医学 | 0.67 | 0.63 | 0.84/0.65 | 0.97 |
| QoQ-Med-VL-7B 医学 | 0.29 | 0.33 | 0.42/0.90 | 0.55 |
| MedGemma-4B 医学 | 0.00 | 0.00 | 0.00/1.00 | 0.00 |
| **★ 我们 v3se-270** | **0.68** | **0.64** | 0.76/0.77 | 0.90 |
| *l4_v3(旧·不忠实,参照)* | — | *0.695* | *0.87/0.92* | *0.98* |

**结论**:干净外部上我们 QWK 0.64 居首且**最均衡**(0.76/0.77),唯一接近者是喂了 505 万医学样本的 Lingshu(0.63,但过度转诊);我们整套仅 ~1.3 万样本 + 透明映射,**少 2–3 个数量级**。

### 4.3 弃权率与被弃权样本分布(Messidor-2,我们独有的能力)

| 模型 | 弃权率 | 被弃权样本的真实 ICDR 分布 |
|---|--:|---|
| **我们 v3se-270** | **31/150 = 20.7%** | G0:5　G1:4　**G2:9　G3:5　G4:8** |
| Lingshu(同走映射) | 10/150 = 6.7% | G0:1　G1:4　G2:1　G3:0　G4:4 |

**解读**:我们弃权的 31 例中 **22 例(71%)真实为 G2–G4**——正是中/重度边界**临床上本就难分**的样本;弃权机制把它们安全导向"转诊 + 人工细分",而非瞎猜一个等级。少数误弃权(G0/G1 共 9 例)来自 EX/HE 假阳性,是后续可收紧的方向。

### 4.4 Lingshu 过度转诊的临床代价(量化)

在 150 张 Messidor-2 中,**不需转诊的病人共 60 名**:

| 模型 | 误转人数(不需转诊却被转) | 漏转人数(需转诊却漏) | 净含义 |
|---|--:|--:|---|
| **我们 v3se-270** | **14 / 60** | 22 / 90 | 更均衡 |
| Lingshu-I-8B | **21 / 60** | 14 / 90 | 多误转 7 人、少漏 8 人 |

**临床解读**:Lingshu 的 Spec=0.65 具体意味着——它把 **21 名健康/轻症病人错误送去转诊**(我们 14 名),即**多产生 7 例不必要转诊**;代价换来多接住 8 名真需转诊者。我们的系统在"少打扰健康人"与"接住病人"之间更平衡。

---

## 5. 模型输出的思维链(真实生成,附我们的解释)

以下均为我们模型在 **Messidor-2 外部图**上的**真实输出**(非训练标签)。展示的是模型自己生成的思维链。

### 样例 A — 无病灶
```
[Lesion Audit]
- MA: absent. / - HE: absent. / - EX: absent. / - SE: absent.
- IRMA: not visually assessable — abstained.
- NV:   not visually assessable — abstained.
[Decision Path]  Step1 no reliable DR lesion (MA/HE/EX/SE) visible -> No-DR.
[Conclusion]     DR tier = No-DR; referable_dr = no; evidence is directly visible.
[JSON] {"dr_tier":"No-DR","referable_dr":false,"lesions_present":[],"abstained":["IRMA","NV"],"severity_indeterminate":false}
```
**我们的解释**:模型逐项核查 4 个可靠病灶并全部判阴 → No-DR;**即便此处无病灶,仍对 IRMA/NV 显式标注"不可视觉判定—弃权"**,从不臆造。结论中"evidence is directly visible"表明该判断完全可核对。

### 样例 B — 校准弃权(框架核心能力)
```
[Lesion Audit]
- MA: present — tiny, round, well-defined red dot-like abnormalities.
- HE: present — dark-red dot/blot/flame-shaped hemorrhagic abnormalities.
- EX: present — bright yellow-white deposits with sharp borders.
- SE: absent.
- IRMA / NV: not visually assessable — abstained.
[Decision Path]
  Step1 reliable DR lesion present: MA, HE, EX.
  Step2 HE and/or EX present -> referable.
  Step3 this lesion pattern does not separate moderate from severe NPDR on visible
        evidence (distinguishing them needs per-quadrant hemorrhage counts / IRMA / NV,
        which are not visually assessable) -> referable, severity indeterminate.
[Conclusion] DR tier = Mod-or-Severe-indeterminate; referable_dr = yes;
             severity indeterminate; evidence_limited.
[JSON] {"dr_tier":"Mod-or-Severe-indeterminate","referable_dr":true,"lesions_present":["MA","HE","EX"],"abstained":["IRMA","NV"],"severity_indeterminate":true}
```
**我们的解释**:这是最能体现"忠实"的一例。模型(1)**先列出可核对的病灶证据**(MA/HE/EX 在、SE 不在);(2)据证据**确定地给出"需转诊"**(HE/EX → referable);(3)在要进一步区分中度 vs 重度时,**明确指出可见证据不足、所需的逐象限计数/IRMA/NV 不可靠,于是主动弃权**(severity indeterminate),而不是猜一个等级。每一步的结论都能溯回到它陈述的证据——**会转诊、但不假装能看出它看不出的东西**,正是黑箱模型不具备的行为。

---

## 6. 贡献与论文故事

### 6.1 贡献
1. **方法**:`from-audit` 忠实解耦框架——模型只产可验证审计,分级由**透明映射**算出,忠实 by construction;并在视觉不可分边界**校准弃权**。SFT 后我们的模型可**原生**产出该忠实链。
2. **评估概念**:提出 **"忠实天花板 0.688"**(仅凭可见病灶证据可达到的 4 档准确率上限),并证明不忠实的旧 l4_v3 外部 0.695 **越过该天花板** → 实证其高分依赖不可验证的捷径(NV/IRMA;G4 召回仅 0.045)。
3. **第一阶段地基**:感知 F1 0.41→0.68 且最均衡,跨 5 厂家最高。
4. **数据效率**:~1.3 万样本 + 透明规则,在干净外部上与 505 万样本的 Lingshu 持平略胜。

### 6.2 论文定位(故事怎么讲)
- **不是打榜论文**(忠实分级精度被天花板 0.688 封顶,刷过它=不忠实)。
- **是"可信/可解释医疗 AI + 评估方法学"论文**,核心论点:
  > "DR 分级的高准确率,有多少真正基于可见病灶证据、有多少靠不可验证捷径?我们用一个忠实 by construction、会校准弃权的框架,界定出'可忠实达到的精度'(天花板 0.688),证明 SOTA 的超额精度来自不可信证据;我们以百分之一的数据匹配该上限,并提供可核对证据链与安全弃权。"
- 目标:**AJCAI 2026(CORE B)**,医学 AI 受欢迎、不强求 SOTA → 故事成立、命中率中上。

### 6.3 诚实局限
忠实精度低于不忠实 SOTA(这是忠实的定价);强医学底座(Lingshu)仅小幅领先;SE/MA 受特征与数据上限;少数误弃权来自 EX/HE 假阳性;外部仅 Messidor-2 150 张,拟扩至全量 ~1744 以增稳。
