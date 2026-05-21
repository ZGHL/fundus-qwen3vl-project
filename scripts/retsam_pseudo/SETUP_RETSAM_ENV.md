## RetSAM 独立环境（gb10_pytorch 容器内）快速安装指令

约束：**不要复用 LLaMA-Factory 环境**（torch/lightning 可能冲突）。任一依赖失败按原则直接重启容器再来：`docker restart gb10_pytorch`。

### 1) 创建 conda 环境

```bash
conda create -n retsam python=3.10 -y
conda activate retsam
```

### 2) 克隆 RetSAM

```bash
git clone https://github.com/Wzhjerry/RetSAM
cd RetSAM
```

### 3) 安装依赖（严格按 requirements）

先按 RetSAM `requirements.txt` 安装，再处理 torch/cuXX：

```bash
pip install -r requirements.txt
```

torch 优先 `cu121`，若不匹配再换 `cu124`（不要硬装）：

```bash
# cu121（优先）
pip install --index-url https://download.pytorch.org/whl/cu121 torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1

# 若 cu121 安装失败，再用 cu124
pip install --index-url https://download.pytorch.org/whl/cu124 torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1
```

### 4) 下载公开权重到 RetSAM/checkpoints/

```bash
mkdir -p checkpoints
# 示例：huggingface-cli（按你环境可用方式下载）
huggingface-cli download JerryWzh/RetSAM_public --local-dir checkpoints/RetSAM_public --local-dir-use-symlinks False
```

推理时把 `--checkpoint` 指向实际权重文件（通常在 `checkpoints/RetSAM_public/` 下）。

