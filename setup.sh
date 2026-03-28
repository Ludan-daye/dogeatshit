#!/bin/bash
# IQD实验环境搭建脚本 — Ubuntu + RTX 5080 16GB
set -e

echo "========================================="
echo "  IQD 实验环境搭建"
echo "  GPU: RTX 5080 16GB"
echo "========================================="

# 1. 创建conda环境
if ! command -v conda &> /dev/null; then
    echo "[错误] 请先安装 Miniconda/Anaconda"
    echo "  wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
    echo "  bash Miniconda3-latest-Linux-x86_64.sh"
    exit 1
fi

echo "[1/4] 创建conda环境 iqd (Python 3.10)..."
conda create -n iqd python=3.10 -y
eval "$(conda shell.bash hook)"
conda activate iqd

# 2. 安装PyTorch (CUDA 12.x)
echo "[2/4] 安装PyTorch (CUDA 12.x)..."
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 3. 安装其他依赖
echo "[3/4] 安装项目依赖..."
pip install -r requirements.txt

# 4. 验证
echo "[4/4] 验证安装..."
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'VRAM: {torch.cuda.get_device_properties(0).total_mem / 1024**3:.1f} GB')
import mauve
print('MAUVE: OK')
import transformers
print(f'Transformers: {transformers.__version__}')
print('所有依赖安装成功!')
"

# 创建结果目录
mkdir -p results/exp0 results/exp1 results/exp2

echo "========================================="
echo "  环境搭建完成!"
echo "  激活环境: conda activate iqd"
echo "  运行实验: bash run_all.sh"
echo "========================================="
