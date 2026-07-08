# Stage-1.5 Experiment Bundle (FGADR mask-grounded count/area perception)

本文件夹是 Stage-1.5「证明实验」的完整可检查/可上传包。证明:**把当前 Stage-1 浪费掉的 FGADR 主四病灶(MA/HE/EX/SE)像素 mask 用起来,加 count/area 桶 + 可靠负样本,warm-start Adapter 1**,在 FGADR 图像互斥 held-out 上对照 Adapter1 原样 vs 训练后,看 (a) 是否学会 count/area 桶、(b) 特异度是否因负样本提升。证明有效后,**全量在 VM 上跑**。

## 目录
- `PLAN.md` — 完整方案(数据/CoT/训练/评测/全量设计/风险)。
- `data/`
  - `fgadr_main4_distribution.json` — 本次数据分布(train/test 各病灶 present/absent、count/area 桶分布、FGADR area 分位阈值)。
  - 训练/测试数据本体在 `LLaMA-Factory/data/annotation/fgadr_main4_proof_{train,test}_sft.jsonl`(图像在 `data/FGADR/Seg-set/Original_Images/`)。
- `configs/`
  - `stage1_5_fgadr_proof.yaml` — 证明训练配置(warm-start Adapter1)。
  - `stage1_5_full_TEMPLATE.yaml` — 全量 VM 训练模板(见内注释:加 flash-attn、大 batch、全量数据)。
- `scripts/`
  - `build_fgadr_main4.py` — 从 FGADR mask 构造单病灶 count/area 数据(可改 caps 出全量)。
  - `score_proof.py` — present/absent + count/area 桶准确率打分(baseline vs trained)。
  - `run_overnight.sh` — 编排:train(看门狗+断点续训)→ vLLM 评测 baseline+trained → 打分 → 报告。
  - `lesion_info.json` — 病灶 schema 文案(CoT 复刻用)。
- `eval/` — vLLM 预测输出(baseline_adapter1.jsonl / trained_stage1_5.jsonl)。
- `results/`
  - `run.log` — 运行日志。
  - `PROOF_RESULTS.md` — 最终对照结果(早上生成)。

## 数据来源真实量(逐病灶,本机磁盘 mask 实测)
强标注 present(mask 非空,可抽 count/area):MA 2075 / HE 2137 / EX 1846 / SE 906 / IRMA 159 / NV 49;强负样本(空 mask):MA 605 / HE 542 / EX 834 / SE 1733。
本证明只用 FGADR 主四(present 各 1279–1456),其中 272 张图像留作 held-out test。

## 全量(VM)怎么扩
1. `build_fgadr_main4.py` 提高 `TRAIN_CAP`(用满 present + 负样本),并加入 DDR-seg / IDRiD 强 mask 样本(复用现有 Stage-1 train 里 ddr_mask/strong_mask 的 S0 样本)+ RetSAM 弱补充(标 weak tier)。
2. 用 `stage1_5_full_TEMPLATE.yaml`:装 flash-attn、加大 batch、1–2 epoch、dense ckpt、dev 选模。
3. 评测加真实 Gold-Dev/Test(present/absent 对标 Adapter1 0.66/spec0.385)。

## 关键设计决定(已确认)
- 起点 = Adapter 1 warm-start(非 base、非 ckpt-20):保留已验证感知,换强数据重校准。
- count/area 学**桶**不学精确数(跨集不可比 + 连通域语义噪声 + 临床用阈值);area 桶按数据集内分位归一。
- IRMA/NV 仅 present 粗档(mask 仅 159/49,不过采样);象限仅在 OD/黄斑坐标有效时给(本证明未含象限)。
- GB10 ~9.5s/样本是硬地板 → 证明是少样本 warm-start 校准(1920 train),全量留 VM。
