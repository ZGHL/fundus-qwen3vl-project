# Stage-1.5 全量数据说明（mask 接地的单病灶结构化感知）

## 任务与 schema
单病灶英文 CoT（与 Adapter 1 兼容，warm-start 续训）。每条样本针对一个病灶输出：
`[Target Evidence] → [Confounder Assessment] → [Attribute Summary] → [Conclusion] → [Structured Output(JSON)]`。
JSON：`{task, target_lesion{name,abbreviation}, present(bool), evidence_state, attributes{count_bucket?, area_bucket?}}`。
**新增能力 vs 原 Stage-1**：present 时输出 `count_bucket` 与 `area_bucket`（absent 不输出）。

## 数据来源（全部真实像素 mask，逐文件计算，非伪标签）
MAIN4 = MA/HE/EX/SE。三套强标注分割集，**crop_box 均为 [0,0,0,0]（零裁剪）**，mask 与图像像素对齐：
| 数据集 | 用法 | 图像 | mask |
|---|---|---|---|
| FGADR Seg-set | 全量新建（全 present + 全负样本） | `FGADR/Seg-set/Original_Images/<id>.png` | `FGADR/Seg-set/<Lesion>_Masks/<id>.png` |
| DDR-seg | 全量新建（全 present + 全负样本） | `DDR-dataset/lesion_segmentation/<split>/image/<id>.jpg` | `.../<label|segmentation label>/<LES>/<id>.tif` |
| IDRiD | 复用现有 Stage-1 S0（图名↔mask 名位数错位，复用 proven-correct 样本 272） | `idrid/images/...` | （复用，不重抽） |

- present/absent 判定：mask 非空像素（阈值 ≥128）→ present；空 → 可靠负样本。
- **count**：连通域计数（min-area ≥5px，去抗锯齿/噪点）。**area**：非零像素占比。
- **count_bucket** = single / few(2–5) / many(>5)（临床阈值，跨集语义一致）。
- **area_bucket** = small/medium/large，按 **每(数据集,病灶) 分位(tertile)归一**（跨集绝对面积不可比，故取相对桶；阈值见 `data/stage1_5_full_distribution.json`）。

## 规模（真实统计，见 `data/stage1_5_full_distribution.json`）
- **train = 9188**（fgadr 6280 / ddr 2636 / idrid 272）
  - present：MA 1786 / HE 1820 / EX 1567 / SE 781
  - absent（全负样本）：MA 514 / HE 476 / EX 729 / **SE 1515**
- **test = 1480**（图像互斥 held-out，fgadr 1088 / ddr 392；按 image_id 哈希 15% 切分，与 train 不重叠）
  - present：MA 279 / HE 303 / EX 265 / SE 119；absent：MA 91 / HE 67 / EX 105 / SE 251

## 文件
- `data/stage1_5_full_train_sft.jsonl`（19MB）、`data/stage1_5_full_test_sft.jsonl`（3MB）
- `data/stage1_5_full_distribution.json`（组成 + 面积分位阈值）
- 构造脚本 `scripts/build_stage1_5_full.py`（VM 上可一键重建，VM 需有 FGADR/DDR-seg masks + validated_clean）

## 已知边界（诚实）
- area 桶为**数据集内相对**（"large" 在不同集对应不同绝对面积）。
- IDRiD 为复用样本（数量小、未含额外负样本）。
- IRMA/NV 未纳入本结构化集（mask 仅 159/49，宜只 present 粗档，另议）。
- RetSAM 弱标注（HE/EX/SE，无 MA）未纳入——如需更大量可加，但须标 weak tier 并降权。

---
# v2（最终，warm-start 用）—— 干净评测版

## 为什么 v2:Adapter1 数据泄漏核查
实测 Adapter1(`fundus_stage1_en_cot_train`)见过 **5395 张唯一图**,覆盖 FGADR/DDR-seg/IDRiD:
- Gold-Test 图像级 ∩Adapter1 = 100/225(44%)、(图,病灶)级 14% → Gold-Test 自身就有泄漏,不宜作主基准。
- IDRiD ∩Adapter1 = 100% → **不能作 warm-start 的外部集**。
- FGADR ~1488/1842 被见过。
→ 结论:warm-start 的干净评测**只能取自 Adapter1 未见图**(FGADR 547 + DDR-seg 246 = 793 张)。

## v2 划分(image_group 互斥 + Adapter1 未见)
- **train = 9699**(fgadr 6656 / ddr 2779 / idrid 264):Adapter1 见过的 FGADR+DDR + 剩余未见图 + IDRiD 复用。
  - present:MA 1951 / HE 2008 / EX 1736 / SE 859;**负样本按病灶封顶 1.5×present**(SE absent 1288,不再淹没)→ 专门补 MA/SE。
- **test = 600**(150 张 **Adapter1 未见**图,整图互斥;FGADR452+DDR148):present MA114/HE115/EX96/SE41,absent 36/35/54/109。**对 Adapter1 与 Stage-1.5 都未见 → 公平、无记忆污染。**
- 验证:test∩Adapter1=0、test∩train=0。
- 数据文件:`stage1_5_v2_{train,test}_sft.jsonl`(注册键同名);构造 `scripts/build_stage1_5_v2.py`(`N_TEST_IMG=150`、`NEG_RATIO=1.5` 可调)。

## 起点决策:warm-start(性价比高)> from-base
warm-start 留住 Adapter1 已验证感知、快、便宜,且有干净未见 test;from-base 更干净但贵、且可能感知回退混淆增益 → 留作消融。IDRiD 在 from-base 下才可作干净外部。
