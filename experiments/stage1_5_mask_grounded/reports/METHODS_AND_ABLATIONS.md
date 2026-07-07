# 方法、统计口径与消融计划(逐实验说明)

更新:2026-07-07。本文配合 `MAINLINE_REPORT_FOR_ADVISOR.md`,把每个结果**怎么设计、用什么数据/模型/参数、怎么统计出来的、特殊值(负/0/横杠)是什么意思**讲清楚。

---

## 第一部分:每个实验的设计与统计口径

### 通用推理参数(所有 VLM 实验)
- 引擎:vLLM,`enforce_eager=true`,贪心解码(temperature=0, top_p=1, top_k=−1),固定 seed。
- 图像:`image_max_pixels=262144`(≈512²)预缩放上限,`image_min_pixels=65536`。
- 我们模型:底座 **Qwen3-VL-8B-Instruct** + **LoRA(rank=16, alpha=32, dropout=0.05, target=全部注意力+MLP+视觉 merger)**;通过 adapter 直挂推理,不 merge。
- 基模均为**零样本**(它们各自的原生对话模板),不加我们的框架。

---

### 阶段一:病灶感知

**(a) 设计与 Prompt**:逐病灶单独提问"这个病灶是否可见",强制只依据可见证据。
- System:`You are an experienced ophthalmologist reading a fundus photograph. Judge ONLY whether the specified lesion is visibly present, based strictly on what you can see.`
- User(以 MA 为例):`Target lesion: microaneurysm (MA) — typical appearance: tiny, round ... Is microaneurysm visibly present? Answer on the first line with exactly one word: Present or Absent. Then give a one-sentence reason.`(HE/EX/SE 同模板换病灶)
- 我们微调模型输出结构化 CoT,末尾带 JSON。

**(b) 数据**(4 病灶 MA/HE/EX/SE;IRMA/NV 因样本稀缺不训、永久弃权):
- 训练:**9060** 单病灶样本(FGADR 4953 + DDR 2023 + APTOS/cropped 1767 + IDRiD 317),present 占 43%。
- 测试:**1108** 单病灶样本(FGADR 472 + DDR 168 + APTOS 444 + IDRiD 24),每病灶 277,**图像与训练互斥**,present 占 **32%**。

**(c) 模型与参数**:我们模型(Qwen3-VL-8B+LoRA,上述);对比 5 基模零样本(Qwen3-VL-8B / InternVL3.5-8B / Lingshu-I-8B / QoQ-Med-7B / MedGemma-4B)。

**统计方法**:
- **解析**:微调模型走 schema——从末尾 JSON 取 `"present": true/false`(原始 JSON:`{"task":"stage1_single_lesion_perception","target_lesion":{...},"evidence_state":"present/absent","present":true/false,...}`);零样本模型走 lenient——从文本首句取 Present/Absent。
- **逐病灶**:对每种病灶,统计 TP/FP/FN/TN → Recall=TP/(TP+FN)、Spec=TN/(TN+FP)、Precision=TP/(TP+FP)、**F1=2·P·R/(P+R)**。
- **汇总**:对 4 病灶取**宏平均**(macro F1/Recall/Spec);**BalAcc=(宏Recall+宏Spec)/2**。
- "判 present 率" = 模型判 present 的样本数 / 总样本,用于看校准(应 ≈ GT 的 32%)。

---

### 阶段二:基于感知的忠实诊断

**(a) 设计与 Prompt**:
- **我们(from-audit)**:先做 4 病灶 present/absent 审计,再由**透明映射**把病灶组合 → DR 档;中/重度视觉不可分时输出**弃权档**。
  - System:`You are a diabetic retinopathy triage assistant. Audit MA/HE/EX/SE from directly visible evidence; IRMA and NV are not visually reliable and must be abstained ... when visible evidence cannot separate moderate from severe NPDR, say so (severity indeterminate) rather than guessing. Do not use lesion counts.`
- **基模(黑箱原生,apple-to-apple)**:不加任何我们的框架/病灶词。
  - System:`You are an ophthalmologist grading diabetic retinopathy.` User:`Examine this fundus image and decide its DR severity. Briefly explain your reasoning, then on a final line write exactly 'Final grade: X' (0=no DR,1=mild,2=moderate,3=severe NPDR,4=proliferative).`

**(b) 数据**:
- 训练(忠实 CoT):**4081**(FGADR/IDRiD 真 mask 接地 2077 + APTOS/DDR 由分级派生 2004),clinical_grade 0–4 均衡。
- 内部测试:300。**外部测试:Messidor-2**,均衡子集 150(30/级)与**全量 1744**(自然患病率:G0 1017/G1 270/G2 347/G3 75/G4 35)。

**(c) 模型与参数**:同阶段一;黑箱基模零样本,`max_new_tokens=400`(容许自由推理)。

**统计方法**:
- **我们的档 → 是否转诊/等级**:装配 4 病灶审计 → 病灶组合(pattern)→ **数据拟合的透明映射** → tier(No-DR/Mild/Moderate/弃权/Severe)。
- **黑箱**:正则提取 `Final grade: X`(0–4)。
- **referable(是否转诊)**:预测端——我们 tier∈{Moderate,弃权,Severe} 记为需转诊;黑箱 grade≥2 记为需转诊。真值——clinical_grade≥2。→ **灵敏度=TP/(TP+FN)、特异度=TN/(TN+FP)**。
- **重病召回**:真值 G3/G4 的样本中,被判需转诊的比例。
- **QWK(0–4)**:二次加权 Cohen's kappa,`w=((i−j)/4)²`。我们的 tier 先映射回 0–4(No-DR0/Mild1/Moderate2/**弃权→2**/Severe3)再与真值 0–4 算;黑箱直接用 0–4。**已用 sklearn 独立核对**(恒定预测器→0、完美→1、完全反相→−0.26)。
- **置信区间**:对图像做 **1000 次 bootstrap 重采样**,取 2.5/97.5 分位 → 95% CI;"我们 vs X"用**配对 bootstrap**(同一重采样上取 QWK 差值),CI 不含 0 即显著。

---

### 补充实验的口径
- **传统 CNN 参照(C2)**:EfficientNet-B0 / ResNet50(ImageNet 预训练),用**同一批 4081 图 + 0–4 分级**端到端训练(AdamW, 20 epoch, 384², CE),留出 10% 验证。评估:同内部测试 + Messidor-2。
- **忠实天花板**:用**真值病灶**(FGADR mask 派生的 present)→ 走同一透明映射 → tier,与真值 grade 算 QWK = "完美感知下映射能达到的上限"。
- **缺口阶梯(E6)**:在 grounded FGADR(152,有 6 病灶真值)上,用"每个病灶模式取多数真值 grade"的 oracle 上界;分别用 4 病灶 vs 6 病灶(+IRMA/NV),看增益。

---

## 特殊数据解释(为什么有负/0/横杠)

| 现象 | 出现在 | 解释 |
|---|---|---|
| **QWK 为负**(InternVL −0.01、Lingshu −0.02) | 阶段二 Messidor 黑箱 | 这两个模型在自然患病率下**近似恒定输出 grade 2/3**(对健康眼也判中度+),与真值**零相关**;恒定预测器 QWK 恒等于 0,**轻微为负只是 0 附近的噪声**(非计算错误,已用 sklearn 核对)。 |
| **F1/QWK = 0.00**(MedGemma) | 阶段一/二 | 4B 模型极度保守,**几乎全判 absent/grade-0**(退化预测),故 Recall≈0→F1=0、灵敏度=0;经核验它"看得见"眼底,是真实行为非解析 bug。 |
| **lenient 评 ckpt-643 = 0.000** | 阶段一(诊断中) | 该 CoT 以`[Target Evidence]`开头,首句无"Present/Absent"词 → lenient 解析器取不到 → 全判 absent。**是解析器与格式不匹配的假象**,该模型应用 schema(JSON)评,真实 F1=0.600。 |
| **横杠(—)** | 各对比表 | 表示**该实验尚未跑**(不是 0)。例:Lingshu 全流程 Stage-2、第二个干净外部集、部分基模的某项指标。 |
| **子集与全量结论相反**(ckpt-643:子集0.705 / 全量0.600) | 阶段一 Lingshu | 68 图的子集不具代表性、偏乐观;**全量 1108 为准**。同理 Messidor 均衡150 掩盖了黑箱过调,全量才暴露。 |

---

## 第二部分:消融实验 —— 已做 / 待补 / 成本

### 已做的消融/对照(支撑现有结论)
| 消融 | 结果 | 结论 |
|---|---|---|
| 感知:基模零样本 vs 我们微调 | Qwen 0.41→0.68;判present率纠正到≈GT | 微调有效且校准 |
| 框架 开/关(同底座 Qwen) | 黑箱 vs 套框架,自然患病率下框架特异度不崩 | 框架带来鲁棒性 |
| 透明映射 vs 端到端分级头 | 学习头外部过拟合(0.64→0.54) | 透明映射更稳 |
| 弃权 开/关 | 弃权样本强制猜≈掷硬币(0.29/0.42) | 弃权=安全细分 |
| 忠实天花板 / 缺口阶梯 | 完美感知上限0.46;+IRMA/NV仅+0.06 | 差距来自不可验证信号 |
| 底座迁移(Lingshu) | 朴素迁移反降(0.60<0.67) | 方法挑底座(校准层) |
| 传统 CNN 参照 | 同分布0.75–0.90 / 跨集崩0.19 | 精度差距=忠实定价;CNN不泛化 |
| 评估集:均衡150 vs 自然1744 | 均衡掩盖黑箱崩溃 | 须用自然患病率评估 |

### 发表前建议补的消融(按性价比)
| # | 消融实验 | 目的(审稿人会问) | 成本 / 时间 |
|---|---|---|---|
| 1 | **框架组件消融正式化**:{无弃权 / 无感知微调(base审计+映射) / 直接出等级(无CoT) / 学习头}四行,同一 Messidor 全量 + CI | 逐项隔离每个设计的贡献 | **纯计算(现有预测)~1-2h;仅"base审计→映射"需一次推理(Qwen base 在1744 audit,~2-3h GPU)** |
| 2 | **忠实性正面度量**:tier-证据一致性、NV/IRMA 臆造率(自由生成) | 把"忠实"从口号变成数字 | 纯计算,~1h |
| 3 | **映射设计敏感性**:透明映射 vs 若干替代规则(阈值/投票),看 tier 稳健性 | 证明结论不依赖某个特定映射 | 纯计算,~2h |
| 4 | **第二个干净外部集**(DeepDRiD / Messidor-1) | 泛化性只有一个外部集偏薄 | **数据获取 0.5-1 天(下载/注册/对标签)+ 推理 2-4h GPU** |
| 5 | **感知逐病灶误差分析 + SE 弱项归因** | 解释 SE F1 仅 0.43 | 纯计算,~1h |
| 6 | **Lingshu 重标定重训**(减负样本压制/更少 epoch)验证"挑底座"可修 | 把负面发现变成可操作结论(可选) | **训练 ~5-10h + 评估 GPU** |
| 7 | 临床医生对弃权样本的一致性(可选,增强临床说服力) | 弃权的临床合理性 | 需外部医生,数天 |

**优先级建议**:**#1 + #2 + #5 几乎纯计算(半天内完成),是发表前的必备且最省成本**;#3 便宜可加;**#4(第二外部集)是最该补的硬缺口**,但需要先拿到干净数据;#6/#7 为加分项,成本高、可放正文之外或未来工作。

**总结**:核心消融大多已具备或可由现有预测**低成本补齐(半天纯计算)**;真正需要额外资源的是**第二个外部数据集**(数据获取为主)与可选的 **Lingshu 重标定重训**。
