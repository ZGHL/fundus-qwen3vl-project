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

### 1.(b) 医学基模介绍

| 模型 | 厂家 | 架构 | 训练数据 | 备注 |
|---|---|---|---|---|
| **Lingshu-I-8B** | 阿里达摩 | **InternVL3**(InternViT + Qwen2.5-7B) | ~5.05M 医学样本(375 万真实 + 130 万合成),**含眼底**(BRESET/PAPILA/EyePACS/**APTOS**) | <10B 医学 SOTA;论文主力 Lingshu-7B 为 Qwen2.5-VL 版,本变体 "-I" 基于 InternVL3 |
| **QoQ-Med-VL-7B** | (医学) | Qwen2.5-VL | 医学多模态 | 含 DR 相关数据 |
| **MedGemma-4B-IT** | Google | Gemma3 | 医学多模态 | 仅 4B、强安全调校 |
| Qwen3-VL-8B | 阿里 | Qwen3-VL | 通用 | 通用基线 |
| InternVL3.5-8B | 上海 AI Lab | InternVL | 通用 | 通用基线 |

注:Lingshu-I-8B = "医学训练版 InternVL3-8B",与通用 InternVL3.5-8B 天然形成"通用 vs 医学微调"对照。

### 1.(c) 最终结果、结论与分析

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
