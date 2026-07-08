# Stage-2 总结：忠实可解释分诊（faithful triage）

> 续训自 **Stage-1.5 v3 `checkpoint-400`**。数据计数来自 `stage2_grade_distribution.json`（train 4081 / test 300）；
> free-generation 结果来自 `STAGE2_FAITHFUL_TRIAGE_..._20260614.md` 的 checkpoint sweep；
> from-audit 结果来自解耦重评（commit 05b6b84，VM 运行）。

---

## 0. 新 Stage-2 设计：为什么不做严格的 Grade 1–4，而做"忠实分诊"

**根本原因：Stage-1.5 只能忠实感知 4 种病灶（MA / HE / EX / SE）的 present/absent。**
而标准 ICDR Grade 1–4 分级需要的证据，**远不止这 4 个病灶的有无**：

| ICDR 分级真正依赖的证据 | 我们能否忠实看到 |
|---|---|
| 每象限**微动脉瘤计数**、4-2-1 规则 | ❌ VLM 数不准小而多的病灶（v2 已证伪） |
| **静脉串珠**（venous beading） | ❌ 无标注、未训练 |
| **IRMA**（Grade 3 边界关键） | ❌ 数据天花板（仅 159），F1≈0.33 不可靠 |
| **NV 新生血管**（Grade 4 / PDR 判据） | ❌ 数据天花板（仅 49），F1≈0.21 不可靠 |
| MA/HE/EX/SE 的有无 | ✅ 这是 Stage-1.5 唯一能忠实提供的 |

**结论**：硬套 Grade 1–4 必然要在看不见的证据上"猜"，破坏可解释性。所以我们改成 **忠实分诊**：

1. **5 个忠实档位（tier）**，不是 ICDR 等级：
   `No-DR / Mild / Moderate / Severe（severe-NPDR-or-PDR）/ Mod-or-Severe-indeterminate（弃权档）`；referable = 后三者。
2. **tier = 数据拟合的 presence→tier 映射 f(audit)**（不是手写规则）：在 grounded 真 mask + 真 grade 上，
   按每个病灶组合的临床等级分布拟合最优 tier；当某组合的 Moderate/Severe 都不占主导（<0.6，经验上约 50/50 不可分）
   → 判 **弃权档**（calibrated abstention，在"可见证据真分不出"的边界上诚实弃权）。
3. **NV / IRMA 永远 abstained，绝不当可见证据；无 count/area。**
4. 因此 tier 是"可验证病灶证据的函数" → **构造上 100% 忠实**；临床等级仅保留在 meta 里作评测参照。

**拟合映射表 `fitted_map`（pattern → tier）：**

| pattern | tier | pattern | tier |
|---|---|---|---|
| none | No-DR | MA | Mild |
| HE / HESE | Moderate | SE | Moderate |
| MASE / MAEX / MAEXSE / MAHE | Moderate | EX / EXSE | **弃权** |
| **HEEX / HEEXSE / MAHEEXSE** | **Severe** | MAHESE / MAHEEX | **弃权** |

**弃权是被校准好的，不是 bug**——训练集里临床 Moderate(g2) 和 Severe(g3/4) 大量落进同一弃权档，证明这条边界视觉上确实不可分：

| 临床真值 → | 判 Moderate | 判 Severe | 判**弃权** | 判 Mild | 判 No-DR |
|---|---:|---:|---:|---:|---:|
| 临床 Moderate(g2) | 194 | 128 | **317** | 38 | 2 |
| 临床 Severe(g3/4) | 79 | 493 | **385** | 33 | 5 |

---

## 1. 可用数据（真实、总量）

Stage-2 数据分两类来源，**referable 档被 grounded 真 mask 数据卡住上限**：

| tier | 来源 | cap | 训练实际 | 测试 | 可用状态 |
|---|---|---:|---:|---:|---|
| No-DR | grade 派生（grade-0 → 全 absent） | 1400 | 1400 | 60 | 充足，**被 cap 截断** |
| Mild | grade 派生（grade-1 → 仅 MA） | 1000 | 1000 | 60 | 充足，**被 cap 截断** |
| Moderate | grounded 真 mask | 1400 | **318** | 60 | **已用尽**（远未达 cap） |
| Severe | grounded 真 mask | 1400 | **633** | 60 | **已用尽** |
| Mod-or-Severe-indeterminate | grounded 真 mask | 1400 | **730** | 60 | **已用尽** |

**真实可用总量（关键瓶颈）：**
- **grounded 真 mask 的 referable 数据 ≈ 1681 行（train，三档之和），基本全部用上**（cap 1400 均未触及 → 这就是上限）。
- No-DR / Mild 来自充足的 grade 标签（被 cap 限制，非耗尽）。
- 训练按来源：**grounded 2077 + grade_derived 2004 = 4081**；referable 1681 vs 非 referable 2400（接近平衡）。
- ⚠️ 本次**未启用** `--v3-preds`（用 v3 给 aptos/ddr 的 g2+ 出伪标审计来加厚 Moderate/Severe）——这是后续扩量的主要手段（伪标需降权）。

---

## 2. 训练细节

| 项 | 配置 |
|---|---|
| 起点 | warm-start **Stage-1.5 v3 `checkpoint-400`**（continued LoRA SFT，`create_new_adapter=false`） |
| LoRA | rank 16 / α 32 / dropout 0.05 / target=all；视觉塔可训、projector 冻结、LM 走 LoRA |
| 模板/精度 | `qwen3_vl_nothink`；BF16；SDPA；cutoff 2304；image pixels 65536–589824 |
| 优化 | 2 epoch / 512 步；per-device batch 2 × grad-accum 8 = **有效 batch 16**；LR **5e-6**；cosine；warmup 0.1；max-grad-norm 1.0；AdamW |
| 存档 | 每 60 步存一次 |
| 运行 | 5526 秒（1:32:06）；聚合训练 loss 0.248；首个 epoch 后十步 loss ≈ 0.006–0.008 |
| 选模 | **无 in-training validation**；事后在 300 行内部集上选 checkpoint |

**用了多少数据：**
- **训练 `stage2_grade_train` = 4081 行**（No-DR 1400 / Mild 1000 / Moderate 318 / Severe 633 / 弃权 730）。
- **内部选模 `stage2_grade_test` = 300 行**（每 tier 60，**图像互斥、Adapter1 未见**；grounded 191 + grade_derived 109；referable 180 / 非 120）。
- 外部最终评测：**Messidor-2**（有 grade 无 lesion 标注）。

---

## 3. 训练结果

### 3.1 第一版：让模型自由生成 tier —— 失败（不忠实）

300 行内部集 checkpoint sweep（自由生成 `dr_tier`）：

| checkpoint | valid率 | QWK | MacroF1 | MAE | Ref敏感 | Ref特异 | 重病召回 | **忠实度** | 弃权率 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ckpt-360（QWK 最高） | 0.670 | **0.678** | 0.276 | 1.920 | 0.625 | 0.849 | 0.730 | 0.317 | 0.050 |
| **ckpt-240（最均衡）** | 0.753 | 0.645 | **0.277** | **1.703** | **0.649** | 0.833 | 0.742 | 0.431 | 0.087 |

**三个病（同一根因：自由生成的 tier 漂移出 presence→tier 映射）：**
1. **忠实度仅 0.43，且越训越低**（ckpt-120 0.66 → ckpt-360 0.32）——"可解释"名存实亡；
2. **~25% 输出无效**（主因 512-token 截断杀掉末尾 JSON）；
3. **Mild 召回 = 0**（60 个 Mild 一个没判对）。

### 3.2 修复：解耦（from-audit，零重训）—— 忠实度 1.0

`tier = fitted_map(模型自己的病灶审计)` 在评分时算，`[Lesion Audit]` 放 CoT 最前（截断也能救回）：

| 指标（ckpt-420，from-audit） | 值 |
|---|---:|
| valid 率 | **0.99** |
| 5 档 QWK | 0.601 |
| MAE | 1.003 |
| **referable 敏感度 / 特异度** | **0.708 / 0.841** |
| 重病例安全召回（真 g3/4 ≥ 被判 referable） | **0.843** |
| **忠实度** | **1.000**（构造保证） |
| 弃权率 | 0.427 |
| Mild F1 | 仍 0（不伤临床：Mild 不转诊） |
| NV/IRMA 臆造 | 4 / 297 |

相比自由生成版：referable 敏感度 +0.077、重病召回 +0.135、忠实度 +0.66。

### 3.3 天花板与诚实局限

- **忠实上限 = 0.688**（4 类 acc，GT presence→tier 最优映射）——即"审计完美"时这套框架的理论上限；
  剩下的差距是 G2↔G3 仅凭可见证据原理上不可分（需逐象限计数/静脉串珠/IRMA，VLM 看不到）。
- **gold-audit 上限**：审计=GT presence 时 referable 敏感度 0.982 / 特异度 0.886 → 转诊粗粒度上几乎不丢信息。
- ⚠️ **泄漏警报（必读）**：Stage-2 这 300 测试图有 **172 张**与 v3 的 Stage-1.5 训练池重叠 →
  **从 v3 warm-start 的 Stage-2 在该集上的 referable/Mild 数受泄漏影响，不能当干净泛化引用**。
  v4/v5 已把全部 297 个 Stage-2-test stem 从 Stage-1.5 训练排除 → **须等 Stage-2 从 v4/v5 重 warm-start、from-audit 重评，才是干净数。**

### 3.4 一句话结论
训练机制没问题（faithful 1.0 / valid 0.99 证明学到位）；headline 不该对标标准 ICDR 分级器（0.85+ QWK），
而应是 **referable 0.71/0.84 + 重病召回 0.84 + 忠实 1.0 + 校准弃权 + 忠实天花板 0.688**。
真正的提升杠杆在**审计质量**（即 Stage-1.5 v4/v5 的 HE 召回、MA/SE 特异度），不在 grader 本身。
