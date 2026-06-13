# Stage-1.5 Proof Results (FGADR MAIN4 held-out, image-disjoint)

Test set: 240 single-lesion samples.

## Baseline = Adapter 1 (no Stage-1.5) vs Trained = Adapter1 + FGADR count/area+negatives

| Lesion | metric | Adapter1 | Stage1.5 | Δ |
|---|---|---:|---:|---:|
| MA | f1 | 0.753 | 0.714 | -0.039 |
| MA | recall | 0.8 | 0.75 | -0.05 |
| MA | spec | 0.35 | 0.3 | -0.05 |
| MA | count_acc | 0.094 | 0.3 | 0.206 |
| MA | area_acc | 0.125 | 0.367 | 0.242 |
| HE | f1 | 0.83 | 0.776 | -0.054 |
| HE | recall | 0.975 | 0.825 | -0.15 |
| HE | spec | 0.25 | 0.4 | 0.15 |
| HE | count_acc | 0.436 | 0.758 | 0.322 |
| HE | area_acc | 0.41 | 0.545 | 0.135 |
| EX | f1 | 0.816 | 0.784 | -0.032 |
| EX | recall | 0.775 | 0.725 | -0.05 |
| EX | spec | 0.75 | 0.75 | 0.0 |
| EX | count_acc | 0.516 | 0.621 | 0.105 |
| EX | area_acc | 0.323 | 0.517 | 0.194 |
| SE | f1 | 0.704 | 0.725 | 0.021 |
| SE | recall | 0.625 | 0.725 | 0.1 |
| SE | spec | 0.7 | 0.45 | -0.25 |
| SE | count_acc | 0.16 | 0.448 | 0.288 |
| SE | area_acc | 0.28 | 0.172 | -0.108 |
| MACRO | f1 | 0.776 | 0.75 | -0.026 |
| MACRO | recall | 0.794 | 0.756 | -0.038 |
| MACRO | spec | 0.512 | 0.475 | -0.037 |
| MACRO | count_acc | 0.315 | 0.537 | 0.222 |
| MACRO | area_acc | 0.291 | 0.405 | 0.114 |

**Read:** count_acc/area_acc = bucket accuracy on samples both-present (Adapter1 mostly can't emit buckets → near 0/parse-fail = the new capability). spec ↑ = strong negatives fixing over-report.

---
## 解读与结论 (2026-06-13)
设定: warm-start Adapter1, 仅 FGADR 主四 mask (1920 train: present320/absent160 每病灶, 1 epoch, LR 5e-6), GB10. eval=FGADR held-out 240 (图像互斥).

1. [YES] count/area 桶可学(核心假设成立): count_acc 0.315->0.537, area 0.291->0.405, 逐病灶 count 全涨 (HE 0.436->0.758). Adapter1 约等于3类随机(0.33), Stage1.5 明显高于随机 -> 用 mask 教 count/area 有效.
2. [NO] 特异度未提升 (macro spec 0.512->0.475, present/absent F1 -0.026). 归因: 负样本太少(每病灶160, 可用605/542/834/1733), 1 epoch/低LR, 多任务轻微挤占 present/absent. 本次规模不足以重平衡特异度.

上 VM 全量调整: 负样本全用(尤其SE1733); present/absent 与 count/area 分别加权; 全量数据(FGADR+DDR-seg+IDRiD)+2epoch; 评测加真实 Gold-Test(对标 0.66/spec0.385), 特异度为主验收.
环境注意: GB10 triton 曾被 liger-kernel 升级(3.1->3.7)破坏 vLLM, 已从镜像 gb10_pytorch:v1 还原 triton 3.1.0 修复; VM 勿装 liger(且不支持 Qwen3-VL).
