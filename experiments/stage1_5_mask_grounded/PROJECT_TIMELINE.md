# 项目尝试时间线（青光眼 + DR 分级）

> 按时间还原我们做过的训练尝试、怎么做的、结果如何。日期为实际运行/提交日期（2026 年）。
> DR 线 = 本仓库；青光眼线 = 同套 Qwen3-VL-8B + LLaMA-Factory + GB10 的并行/前置项目（不在本仓库，作教训记录）。

---

## 一、青光眼线（06-02 → 06-08，5 天，最终搁置）

两阶段链式：Stage-1 解剖感知（CDR / 视盘大小 / 盘沿象限）→ Stage-2 诊断（青光眼 vs 健康，warm-start Stage-1）。
评测集 ORIGA（150 张，39 青光眼 / 111 健康）。核心反复在和**类别不平衡**搏斗。

| 日期 | 版本 | 怎么做 | 结果 |
|---|---|---|---|
| 06-02 晚 | v1 / baseline | 首个两阶段链 | 诊断 eval 崩（CUDA error），促使迭代 |
| 06-03 早 | **bal（再平衡）** | 训练类别再平衡（bal_train 13821） | **塌成全判青光眼**：Acc 0.260 / Sens 1.000 / **Spec 0.000** |
| 06-03 午 | **v2** | 换数据/配方（full 21531，偏不平衡） | **塌成全判健康**：Acc 0.747 / **Sens 0.026** / Spec 1.000 |
| 06-05→06 | **v3** | stage1 用 mix 数据、视觉塔+projector 都训、LR 6e-6；再链 stage2 | stage1 感知尚可，**Stage-2 训练崩溃（rc=1）** |
| 06-07 | v3 感知评 | 单独评 stage1（n=150） | macro **0.573**：盘沿 0.88 / CDR 0.48 / 视盘大小 0.36 |
| 06-08 | v4 | 仅建 config | **未运行，项目搁置** |

训练配方：LoRA r16/α32、视觉塔+projector 均可训、batch 1×16、1 epoch、LR 6e-6(S1)/1.5e-5(S2)、cutoff 1536。
数据变体：imbalanced(21531) / balanced(13821) / mix(13550) / uniform(9196) 四种采样都试过。

**结论**：诊断从未成功——随类别平衡在「全健康 ↔ 全青光眼」间翻转，**负样本不平衡导致感知塌缩**；感知层只有盘沿象限（0.88）可用，CDR / 视盘大小弱。base 零样本本身也几乎全判健康（Acc 0.740 / Sens 0.077 / Spec 0.973）。

---

## 二、DR 分级线（05-21 → 06-17）

### 阶段 A — Stage-1 病灶感知

**① 中文 CoT L3（05-25→26，最早主线）**
- 路径：解剖 warmup（`stage1_pilot`）→ **targeted calibration**（23253 样本，MA/HE/EX/SE 定向校准）→ **six-lesion calibration**（7200 = 6 病灶 ×(600 正+600 负)）。
- 即仓库记录的 `l3_zh_cot_baseline` 线（中文 CoT）。L3 holdout80 micro F1 ≈ 0.795。

**② 英文 CoT Stage-1 = "Adapter1"（06-08→10）**
- 把 L3 感知**重做成英文 CoT、单病灶 present/absent schema** → `stage1_en_cot`（= Adapter1，新主线起点）。
- 加 calibration 变体 → `stage1_en_cot_gentle_calibrated`（ckpt-20 balanced）。06-10 出"统一 Stage-1 报告"。

### 阶段 B — Stage-1.5 mask-grounded（新主线基石，v1.5 → v5）

| 日期 | 版本 | 怎么做 | 结果 |
|---|---|---|---|
| 06-13 | **v1.5 证明** | warm-start Adapter1，FGADR 主四 mask + count/area **桶** + 负样本（1920） | count/area **看着能学**（macro 0.54）——但测试集**泄漏**（Adapter1 见过） |
| 06-13 | **v2** | 干净 Adapter1-未见、图像互斥 split + 修泄漏（9699） | **count/area 证伪**（干净 test ≈ 随机 0.25）；ckpt 不稳 → **砍掉 count/area** |
| 06-13→14 | **v3** | **只做 present/absent** + 猛加负样本（硬负 + grade-0） | **修好过报**：Spec 0.198→0.594、BalAcc 0.711；选定 **ckpt-400（当前最优）** |
| 06-15 | **v4** | 召回再平衡（PRES_CAP 1000→2000）+ 排除 Stage-2 测试 stem（leak-free） | 统一抬正样本（后判定太钝，见 v5） |
| 06-16 | 公平阶梯 | base 零样本 vs ckpt-400 公平 prompt | base BalAcc 0.638 → ckpt-400 0.742 |
| 06-17 | **v5** | **逐病灶**：HE 抬召回、MA/SE 抬特异度、EX 不动 + **三分法 train/dev/test** + DEV 选 ckpt + 预处理提速 | **正在训练** |

v3 ckpt-400 逐病灶（1108 图像互斥 test）：MA R0.955/Sp0.481、HE 0.706/0.810、EX 0.822/0.816、SE 0.950/0.397；macro F1 0.626 / BalAcc 0.742。

### 阶段 C — Stage-2 DR 分诊（06-14 → 17）

| 日期 | 版本 | 怎么做 | 结果 |
|---|---|---|---|
| 06-14 | **v1 自由生成** | warm-start v3 ckpt-400，模型**自由生成 tier**；数据 = 拟合 presence→tier 映射 | **失败**：忠实度 0.43（越训越低）、~25% 无效（截断）、Mild 召回 0 |
| 06-15 | **解耦（Plan A）** | 零重训，评分时 `tier=map(模型审计)`，审计放 CoT 最前 | **忠实 1.0**、valid 0.99、ckpt-420 referable 0.708/0.841、重病召回 0.843 |
| 06-15 | decision-map sweep | 事后调弃权阈值 | 决策 map 非杠杆，弃权是被校准的 |
| 06-17 | **全新设计** | 模型**学走决策树诊断**、忠实度=测量值、三级数据（🟢2680）、三系统对比 | 方案定稿 `STAGE2_REDESIGN.md`，待 v5 后落地 |

---

## 三、贯穿两线的核心教训（时间顺序）

1. **负样本不平衡 → 感知塌缩**（青光眼全健康/全青光眼；DR 早期过报）→ DR v3/v5 精细负样本平衡。
2. **泄漏测试集 = 假阳性**（Stage-1.5 v1.5 count "0.54" → 干净 v2 证伪；Stage-2 test 172/300 泄漏）→ 一律图像级互斥 + 从上游训练排除。
3. **count/area / 计数学不会**（v2 干净评测证；且加 5× 数据无改善 → 能力天花板非数据不足）→ 只做 present/absent → 进而 G2/G3 不可分 → 忠实分诊 + 弃权。
4. **自由生成会漂移**（Stage-2 v1 忠实 0.43）→ 解耦 / 走树 + 忠实度测量。
