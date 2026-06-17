# Stage-2 最终方案：忠实可解释 DR 分诊（模仿眼科医生的「先看病灶、再据病灶诊断」）

> 前提：Stage-1.5 v5 已把 MA/HE/EX/SE 四病灶感知做可靠。本方案是**最终设计**，取代旧 Stage-2（旧数值因
> 泄漏/自指/in-sample/选模偏置/忠实度同义反复而作废）。目标：可发表。

---

## 0. 核心哲学：两阶段模仿医生学习过程

1. **Stage-1.5 = 学会看病灶**：单病灶、一个一个看（present/absent）。已由 v5 完成。
2. **Stage-2 = 学会据病灶下诊断**：像住院医学会看病灶后，再学"这组所见 → 这个严重度"的决策——**模型端到端学"审计→走决策树→诊断"**，不是代码查表，也不是自由生成。

**忠实 = 测出来的，不是构造的**：在 held-out 上测 `模型诊断 == tree(模型自己的审计)`（期望 ~0.95）。这同时解决了旧版"忠实度 1.0 同义反复"的发表硬伤。

---

## 1. 标签空间与决策树（透明、数据拟合）

5 个忠实档：`No-DR / Mild / Moderate / Severe / Mod-or-Severe-indeterminate(弃权)`；referable = 后三者。
`tier = f(audit)`，f 是在 🟢 GT-mask 上按 16 个 MA/HE/EX/SE pattern 拟合的透明映射；Moderate/Severe 经验上 ~50/50 不可分的 pattern → **弃权**（校准弃权，弃权 ⊂ referable = 安全转诊）。NV/IRMA **永远弃权**，无 count/area。

f 在 🟢 TRAIN 上拟合后**冻结**；它既是**训练监督**（生成走树 CoT），也是**忠实度判官**（测一致性）与可选**部署护栏**。

---

## 2. 可用数据（精确，对 validated_clean 11783 统计）

### 🟢 mask-grounded（GT presence → 忠实 tier，精确读真 mask）

| tier | FGADR | IDRiD-seg | DDR-seg | 🟢 合计 |
|---|---:|---:|---:|---:|
| No-DR | 105 | 0 | 0 | 105 |
| Mild | 147 | 0 | 88 | **235** |
| Moderate | 334 | 1 | 169 | **504** |
| Severe | 623 | 40 | 251 | **914** |
| 弃权 | 633 | 40 | 249 | **922** |
| 合计 | 1842 | 81 | 757 | **2680** |

### 其余来源
- **grade-only**（aptos 3662 + ddr_grading 5006）：g0 1805（→No-DR，审计空）、g1 874（→Mild，**MA 盲点弱**）、g2-4 5989 referable（→审计定档，RetSAM 可见 ~4572）。
- 三级可靠性：🟢 mask（GT）> 🟡 audit-grounded（v5+RetSAM 交叉 + grade 一致性过滤）> ⚪ grade-rule（g0/g1）。

> 关键：🟢=2680 已足够 → **🟡 几乎不用补**（只给 Moderate 补量），把自蒸馏/伪标风险降到最低。

---

## 3. 数据分配（图像级互斥、leak-free）

铁律：一切按图像切分、三方互斥；**TEST 必须从 Stage-1.5 v5 训练排除**（沿用 `stage2_test_heldout_stems.txt` 机制）。tier 标签 = tree(presence)，永不用原始 grade。

### (a) 训练 TRAIN（~2750，🟢 为主、均衡到 ~600/tier、🟡 极少）
| tier | 🟢 真 mask | ⚪/🟡 补充（降权 0.5×、过滤） | 目标 |
|---|---:|---|---:|
| No-DR | ~85 | g0 审计空 ~515 | ~600 |
| Mild | ~190（真 MA mask） | g1 弱补 ~160（标注弱档） | ~350 |
| Moderate | ~430 | 🟡 ~170（v5+RetSAM 一致 + grade 过滤） | ~600 |
| Severe | ~780 → **cap 600** | — | ~600 |
| 弃权 | ~790 → **cap 600** | — | ~600 |
| 合计 | ~2085 | 🟡/⚪ ~845 | **~2750** |

**🟡 实际只取 ~170**（不是几千）；Severe/弃权 全靠 🟢，No-DR 靠 g0，Mild 靠真 MA mask + 少量 g1。

### (b) 测试 TEST（冻结，绝不进训练/选模/拟合）
| 子集 | 来源 | 规模 | 评什么 |
|---|---|---|---|
| TEST-core | FGADR/IDRiD 有 mask+grade、v5 未见 | ~250 分层 | 审计正确性 + referable/severe + **天花板** + **忠实度** |
| TEST-deploy | grade-only 未见图 | ~300 按 grade 分层 | 部署分布 referable/severe（vs 临床 grade） |
| TEST-external | **Messidor-2** | 全量 | 外部泛化 |

### (c) 验证 DEV（只选 ckpt + 调弃权阈值 θ）
~200（100 mask + 100 grade-only），与 TRAIN/TEST 互斥。

### 映射 / 天花板
f 在 🟢 TRAIN 拟合 → 冻结；**天花板（GT presence→f vs 临床 grade）在 TEST-core 上量 + 给 CI**（非 in-sample）。

---

## 4. 训练设置（学医生的决策，不重学看病灶）

- **起点**：warm-start v5（带着"会看病灶"）。
- **监督目标**：每条 = 图 →「[Lesion Audit] + [Decision Path 走树] + [Diagnosis] + JSON」三段同训。
  - 审计标签：🟢 用 GT mask；🟡 用 v5+RetSAM 交叉（过滤）。
  - tier 标签：**tree(presence)**（不用 grade）→ 监督本身忠实，杜绝 image→grade 捷径。
  - 走树 CoT：树确定性生成。
- **防遗忘**：混入 **~20% Stage-1.5 单病灶 replay**；**低 LR 5e-6、≤2 epoch**、密集存档、**DEV 选 ckpt**（不在 TEST 选）。
- **忠实度**：held-out 测 `模型诊断 == tree(模型审计)`，报两模式：learned(~0.95) 与 guarded(护栏强制=1.0)。
- **必做消融**：(i) v5 + 代码树（不训 Stage-2）vs (ii) Stage-2 端到端训练 → 证明"诊断"该学还是只需树。

---

## 5. CoT 设计（忠实报告）

### 模板
```
[Lesion Audit]   逐病灶 + 可见证据；NV/IRMA 强制弃权
[Decision Path]  按树走、引用规则（不抄 grade）；不可分则显式弃权并注明需 IRMA/逐象限计数
[Diagnosis]      模型生成的诊断（学出来的）
[JSON]           {lesions_present, irma:"abstained", nv:"abstained", pattern,
                  dr_tier, referable, evidence_limited, rationale}
```

### 示例 ①（Severe：HE+EX 无 SE）
```
[Lesion Audit] MA:未见孤立红点→absent · HE:多处暗红出血→present · EX:多灶亮黄硬渗→present
               · SE:未见棉絮灶→absent · IRMA/NV:视觉不可靠→弃权
[Decision Path] 可见 HE+EX、无 SE；该组合数据上以 Severe 为主 → 判 Severe。
[Diagnosis] Severe（severe-NPDR-or-PDR），建议转诊。
[JSON] {"lesions_present":{"MA":false,"HE":true,"EX":true,"SE":false},"irma":"abstained","nv":"abstained","pattern":"HEEX","dr_tier":"Severe","referable":true,"evidence_limited":false}
```

### 示例 ②（弃权：MA+HE+EX 无 SE）
```
[Lesion Audit] MA:散在红点→present · HE:暗红出血→present · EX:亮黄硬渗→present
               · SE:未见→absent · IRMA/NV:视觉不可靠→弃权
[Decision Path] 可见 MA+HE+EX、无 SE；此组合数据上 Moderate/Severe 约各半，单凭可见病灶不可分
                （需逐象限计数/IRMA，视觉不可评估）→ 弃权，按 referable 转诊。
[Diagnosis] Moderate-or-Severe（不确定），建议专科进一步分级。
[JSON] {"lesions_present":{"MA":true,"HE":true,"EX":true,"SE":false},"irma":"abstained","nv":"abstained","pattern":"MAHEEX","dr_tier":"Mod-or-Severe-indeterminate","referable":true,"evidence_limited":true}
```

### 忠实报告流程
模型生成四段 → 代码校验 `tree(JSON.lesions_present) == JSON.dr_tier`：一致→出报告记 faithful=1；不一致(漂移)→标红，可选护栏覆盖。报告里所有诊断级文字都可追溯到 [Lesion Audit] = 忠实报告。

---

## 6. 评估与论文叙事（对临床真值；杀 P1–P5）

| 旧问题 | 规避 |
|---|---|
| P1 泄漏 | 全局图像互斥划分；TEST 从 v5 训练排除 |
| P2 QWK 自指 | 一切指标对**临床 grade**算；QWK 仅辅助、标注非 ICDR |
| P3 天花板 in-sample | f 在 TRAIN 拟合 → TEST-core 量天花板 + CI |
| P4 选模偏置 | DEV 选 ckpt/θ，TEST 只报一次 |
| P5 忠实度同义反复 | 模型学诊断，忠实度=**测量值**（非构造） |

**三系统对比（核心卖点，同一 TEST，对临床 grade）**：
- (i) 黑箱 image→grade 分级器；(ii) 本方案 audit→tree→tier；(iii) gold-audit GT presence→tree（上界）。
- 两个 novel 量：**忠实成本 =(i)−(ii)**、**不可分上界 gap =(iii) 距 100%**（量化 DR 分级多依赖 VLM 看不见的证据）。

**报告指标**：审计逐病灶 F1（TEST-core）/ referable 敏感·特异·PPV（TEST-deploy+Messidor-2）/ 重病安全召回 / 弃权 operating 曲线 / 忠实度 / 天花板+CI。

---

## 7. 诚实局限
- **Mild/MA 盲点**：aptos 域 MA 不可靠 → Mild 天生弱（但 🟢 已有 235 真 MA-mask，远好于旧版）；不靠 g1 充数，列 limitation。
- **模型贡献薄**（审计=v5、树=16 行）→ **novelty 押在忠实成本/天花板的科学发现 + "只需树+可靠审计即可可解释诊断"**，不靠"又训了个分级器"。

---

## 8. 落地步骤
1. v5 验收 → 定全局图像划分（TEST-core/deploy + DEV + MAP-FIT，从 v5 训练排除 TEST）。
2. 读 🟢 真 mask 定 tier（FGADR+IDRiD+DDR-seg，已验证 2680）；g0/g1 grade-rule；g2-4 经 v5+RetSAM+grade 过滤得 🟡(~170)。
3. 均衡到 ~600/tier 抽 ~2750 训练集。
4. f 在 🟢 TRAIN 拟合冻结。
5. warm-start v5 续训（审计+走树+诊断，tier=tree，replay 20%，低 LR，DEV 选）。
6. 评测：TEST-core/deploy/Messidor-2 + 三系统对比 + 天花板+CI + 忠实度。
7. 消融：(i)v5+树 vs (ii)训练；🟢-only vs +🟡；θ 曲线。
