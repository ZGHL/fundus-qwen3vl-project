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
