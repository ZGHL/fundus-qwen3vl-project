# Stage-1.5 落地方案(mask 接地的结构化感知)+ 今晚执行计划

> 日期 2026-06-10。目标:把 Stage-1 从"present/absent(弱标注为主)"升级为"present + count桶 + area桶(+象限)的强 mask 接地感知",为 Stage-2 的 burden/分级提供视觉接地。
> 起点 = **Adapter 1**(warm-start);数据 = **真实 mask 抽取**(不再用 RetSAM 伪标签当主力);评测 = **真实 Gold-Dev/Test** + count/area 桶。

---

## A. 数据分配(真实可用量,逐病灶)

强标注 present(mask 非空,带 count+area,可抽象限)/ 强负样本(空 mask):
| 病灶 | present(强) | 负样本(强,空mask) | RetSAM弱present(仅HE/EX/SE) |
|---|---:|---:|---:|
| MA | 2075 | 605 | 0(RetSAM 不产) |
| HE | 2137 | 542 | ~3704(净新增,APTOS+DDR-grading) |
| EX | 1846 | 834 | ~3503 |
| SE | 906 | 1733 | ~1064 |
| IRMA | 159 | (FGADR仅阳性) | 0 |
| NV | 49 | (同上) | 0 |

**桶定义**:
- count_bucket = none / few(1–5) / moderate(6–20) / many(>20) —— 按临床阈值,跨集语义一致;连通域 min‑area ≥5px,丢弃整图 ≤10px 的可疑。
- area_bucket = small / medium / large —— 用 stats 里各病灶 area_fraction 分位(MA/HE/EX/SE 已有 tertile 阈值),**按数据集内归一**。
- quadrant(可选、粗档)= 相对视盘/黄斑的上/下/鼻/颞;**仅在 OD/黄斑坐标 valid 时给**(IDRiD 坐标=0、DDR-grading 部分缺 → 缺则不写象限,不编)。
- 分病灶接地强度:MA/HE 以 count 桶为主;EX 以 area 桶为主(碎裂);SE 粗档;**IRMA/NV 只 present + "few" 粗档**(159/49,不精确数,不过采样填配额——避免重蹈 Adapter1 把 159 撑成 400 的覆辙)。

**tier 标注**(写入 meta,便于加权/ablation):strong_mask(FGADR/DDR/IDRiD)> retsam_weak(仅 HE/EX/SE 补充)> 不用规则当 present。

**划分**:train / 与 train 图像互斥的 count-area 测试集(每病灶留出 ~10%)/ 复用真实 Gold-Dev(596)、Gold-Test(每病灶 225)做 present/absent 对标。

**全量目标(VM)**:present 用满强标注(~6400 主四 + IRMA/NV 少量)+ 强负样本(~3700)+ RetSAM 弱补充(标弱 tier)≈ 单病灶 ~12–15k;present/absent 平衡、count/area 桶平衡。

---

## B. CoT / schema(沿用 Stage-1 英文单病灶,新增量化字段)

- 保持单病灶任务、英文、`[Target Evidence]→[Confounder]→[Attribute Summary]→[Conclusion]→[Structured Output]`。
- **新增**:present 时输出 `count_bucket`、`area_bucket`(+ 有坐标时 `quadrant`);absent 时不写量化。
- JSON:`{task, target_lesion, present, count_bucket, area_bucket, quadrant?, evidence_tier}`。
- 弱来源(RetSAM)样本标 `evidence_tier=weak`,量化措辞降级("approximate")。
- 不泄露 grade(沿用 no_grade_in_model_visible_text)。

---

## C. 训练(VM 全量 / GB10 过夜 pilot)

- 起点:**Adapter 1**(`stage1_en_cot` adapter,从 handoff tar 取),`create_new_adapter=false` warm-start。
- LoRA:rank16/alpha32/dropout0.05/target=all;freeze_vision_tower=false、projector 冻结(与现有一致)。
- template qwen3_vl_nothink、cutoff_len 2304、image 589824、bf16。
- **VM 全量**:LR 6e-6、cosine、1–2 epoch、装 flash-attn、大 batch、dense ckpt、dev 选模。
- **GB10 过夜(12h 预算)**:建**完整** Stage-1.5 数据集,warm-start Adapter1 在完整数据上训练,但**按墙钟时间盒到 ~10h**(image 262144 提速,~0.1 样本/s → ~3500–3800 样本·遍,约 0.3–0.5 epoch;因 warm-start,少 epoch 即可加 count/area + 拉特异度,类比 gentle 校准只 20 步)。LR 5e-6、cosine、save 每 ~40 步、**看门狗**(训练/容器死 → docker restart + resume_from_checkpoint)、到 10h 自动停 → 评测最新 ckpt。shuffle 保证部分 epoch 也是均衡覆盖。

---

## D. 评测

- **present/absent(对标 baseline)**:真实 Gold-Dev(596)+ Gold-Test(MA/HE/EX/SE 各 225),vLLM 推理,算 Macro-F1 / Recall / **Specificity** / per-lesion F1,**与 Adapter1(0.66/0.385)、ckpt-20(0.66/0.385→0.601 balacc)三方对比**。核心看点:**特异度是否因强负样本提升、且 HE/EX/MA present 不退化**。
- **count/area 桶**:在 image-disjoint count-area 测试集上,算每病灶 count_bucket / area_bucket 的准确率 + 混淆(尤其相邻桶)+ 解析率。
- **诚实指标**:IRMA/NV 不纳入 count 评测,只报 present(已知弱)。
- 成功标准:present/absent 不低于 Adapter1 且特异度↑;count/area 桶准确率显著高于随机(≥~0.5,相邻桶容忍)。失败也如实报。

---

## E. 今晚 GB10 执行(经你批准后自动跑)

1. 抽 mask → 建 Stage-1.5 全量数据集 + count-area 测试集 + 注册(GPU-free,~30–60min,**必成**)。
2. 取 Adapter 1、写 pilot config。
3. 启动 pilot 训练(看门狗 + dense ckpt + resume)。
4. 训练完自动 vLLM 评测(Gold-Dev/Test + count/area 测试)。
5. 写 `/sda/zgh/STAGE1_5_RESULTS_<date>.md`:数据组成、pilot 指标 vs baseline、count/area 桶准确率、问题与下一步。
- **交付边界**:数据集+config 必成;pilot 是"少样本校准"性质的初步结果(非全量),全量留 VM。GB10 掉线由看门狗兜底,但若反复崩则保数据、停训、如实报告。

---

## F. 风险与诚实声明
- GB10 ~9.5s/样本是硬地板 → pilot 只能 ~3000 样本,**不是全量 Stage-1.5**,指标是方向性信号不是终值。
- count/area 是**桶**不是精确数(跨集不可比 + 连通域语义噪声 + 临床用阈值)。
- 象限受坐标有效性限制,缺坐标不编。
- IRMA/NV 数据太少,只做 present 粗档,不参与量化评测。
