# 病灶感知全量对比(1108 单病灶,图像互斥,同测试集)

```
Lingshu微调 ckpt-643(最优) | macro Recall 0.504 Spec 0.927 F1 0.600 BalAcc 0.715 | parse_fail 2 | MA:F1=0.66/R=0.61/Sp=0.83 HE:F1=0.66/R=0.52/Sp=0.96 EX:F1=0.71/R=0.58/Sp=0.97 SE:F1=0.36/R=0.30/Sp=0.94
我们Qwen3微调 v3se-270 | macro Recall 0.686 Spec 0.861 F1 0.684 BalAcc 0.773 | parse_fail 0 | MA:F1=0.74/R=0.71/Sp=0.83 HE:F1=0.80/R=0.84/Sp=0.82 EX:F1=0.77/R=0.84/Sp=0.84 SE:F1=0.43/R=0.35/Sp=0.95
Lingshu基模(零样本) | macro Recall 0.860 Spec 0.723 F1 0.673 BalAcc 0.792 | parse_fail 0 | MA:F1=0.74/R=0.80/Sp=0.73 HE:F1=0.76/R=0.76/Sp=0.85 EX:F1=0.78/R=0.96/Sp=0.76 SE:F1=0.41/R=0.93/Sp=0.56
```
