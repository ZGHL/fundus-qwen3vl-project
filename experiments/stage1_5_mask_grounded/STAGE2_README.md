# Stage-2:忠实 DR 分诊(faithful triage + 校准弃权)

## 立意
两阶段可解释 DR 诊断:Stage-1.5(病灶 present/absent 感知,v3 已修特异度)→ Stage-2 **透明的 presence→tier 分级 + 校准弃权**。**每个诊断都由可验证的病灶证据支撑;在可见证据无法判定处显式弃权**(不臆造 NV/IRMA、不用计数)。

## 标签空间(5 桶,全部 = 可验证审计的函数)
No-DR / Mild / Moderate / Severe(severe-NPDR-or-PDR)/ **Mod-or-Severe-indeterminate(弃权)**。referable = 后三者。

## tier = 数据拟合的 presence→tier 映射(非手写规则)
在 grounded(FGADR/IDRiD 真 mask + 真 grade)上拟合每个病灶模式的最优 tier;Moderate/Severe 不可分(各<0.6)的模式 → 弃权。映射示例(见 `data/stage2_grade_distribution.json` 的 `fitted_map`):
- none→No-DR;MA→Mild;HE/HESE/单双病灶→Moderate;**HEEX/HEEXSE/MAHEEXSE→Severe**;**MAHEEX(无SE)/EX/EXSE/MAHESE→弃权**(经验上 G2/G3≈50/50)。
- **NV/IRMA 永远 abstained**;**无 count/area**。

## CoT
`[Lesion Audit](MA/HE/EX/SE present/absent;IRMA/NV not assessable—abstained) → [Decision Path](透明套映射;不可分则弃权并注明"需逐象限计数/IRMA/NV,不可视觉评估") → [Conclusion] → [JSON]{dr_tier,referable_dr,lesions_present,abstained,severity_indeterminate,evidence_limited}`。

## 数据(R2: s3://fundusv1/datasets/stage2_grade_20260614.tar.zst,SHA 1a67399383a5c5bb1fc7df0bd1e19d046e1a5fbc8ff7ed92de0a8c265afb9944)
- **TRAIN 4081**:No-DR 1400 / Mild 1000 / Moderate 318 / Severe 633 / Indeterminate 730。referable 1681 vs 非 2400(平衡)。
  - No-DR/Mild = aptos/ddr 的 g0/g1 **grade 派生审计**(g0→全absent,g1→仅MA;临床定义,无噪、无需 v3)。
  - Moderate/Severe/Indeterminate = **grounded 真 mask**(高质量)。
- **TEST 300**(每 tier 60,**全 Adapter1 未见、图像互斥**,验证 ∩=0)。clinical grade 存 meta = 评测参照。
- 可选扩充:`build_stage2_grade.py --v3-preds <{image_id,MA,HE,EX,SE}>`(VM 上用 v3 给 aptos/ddr 的 **g2+** 出伪标审计,主要加厚 Moderate/Severe/Indeterminate;伪标需降权)。

## 训练
warm-start **Stage-1.5 v3**(`stage1_5_v3/checkpoint-400`,create_new_adapter=false);LoRA 同配方、LR 5e-6、1–2 epoch、密集存档、dev 选最优。任务轻(只学审计→tier + CoT 格式)。

## 评测(可解释性核心)
1. **忠实上限对照(novel)**:GT presence→tier 最优映射 = **0.688(4 类)** 作上限;报 我们的忠实 VLM(predicted presence→tier)接近度。
2. **Referable(≥Moderate)sens/spec/PPV**(主临床)。
3. **5 桶 / 4 类 acc·QWK·混淆**;**重病例安全召回**(真 g3/g4 ≥ 被判 referable)。
4. **忠实度**:dr_tier == 映射(审计) 一致率;审计 == v3 实际预测;零 NV/IRMA 臆造;弃权率 vs 准确率。
5. **外部 messidor-2**(g0→No-DR/g1→Mild/g2+→referable)。
6. **(对照实验)学习式分级器**(image→clinical grade)报 acc + 忠实度 → gap = 不忠实记忆的增益 = "DR 分级有多依赖 VLM 无法忠实感知的证据"。

## 诚实 limitation
G2↔G3 在可见证据下 ≈50/50 不可分(数据证明,MAHEEX 无SE 547张 Mod41%/Sev56%);严重度细分需 **逐象限出血计数 / 静脉串珠 / IRMA**——VLM 计数学不会、静脉串珠无标注、IRMA 数据天花板(159)→ 诚实弃权 + 列未来工作(需新标注数据)。
