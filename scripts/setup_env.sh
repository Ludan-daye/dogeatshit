#!/bin/bash
# IQD 实验环境搭建脚本
# 支持：镜像源选择、CUDA 自动检测、conda/venv 自动切换
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "========================================="
echo "  IQD 实验环境搭建"
echo "========================================="

# ── 0. Llama2 模型访问提示 ─────────────────────────────────────────────
echo ""
echo "[提示] 本项目使用 meta-llama/Llama-2-7b-hf 模型。"
echo "  请确保已完成以下步骤："
echo "  1. 在 https://huggingface.co/meta-llama/Llama-2-7b-hf 申请访问权限"
echo "  2. 运行 huggingface-cli login 登录你的 HuggingFace 账号"
echo ""

# ── 1. 选择 pip 镜像源 ──────────────────────────────────────────────────
echo ""
echo "选择 pip 镜像源："
echo "  0) 默认 (PyPI 官方)"
echo "  1) 清华大学"
echo "  2) 阿里云"
echo "  3) 中科大"
echo "  4) 豆瓣"
read -rp "请输入编号 [0-4, 默认 0]: " MIRROR_CHOICE
MIRROR_CHOICE=${MIRROR_CHOICE:-0}

PIP_MIRROR=""
PIP_TRUSTED=""
case "$MIRROR_CHOICE" in
    1)
        PIP_MIRROR="https://pypi.tuna.tsinghua.edu.cn/simple"
        PIP_TRUSTED="pypi.tuna.tsinghua.edu.cn"
        echo ">>> 使用清华镜像"
        ;;
    2)
        PIP_MIRROR="https://mirrors.aliyun.com/pypi/simple"
        PIP_TRUSTED="mirrors.aliyun.com"
        echo ">>> 使用阿里云镜像"
        ;;
    3)
        PIP_MIRROR="https://pypi.mirrors.ustc.edu.cn/simple"
        PIP_TRUSTED="pypi.mirrors.ustc.edu.cn"
        echo ">>> 使用中科大镜像"
        ;;
    4)
        PIP_MIRROR="https://pypi.douban.org/simple"
        PIP_TRUSTED="pypi.douban.org"
        echo ">>> 使用豆瓣镜像"
        ;;
    *)
        echo ">>> 使用 PyPI 官方源"
        ;;
esac

# 构造 pip install 参数
PIP_ARGS=""
if [[ -n "$PIP_MIRROR" ]]; then
    PIP_ARGS="-i $PIP_MIRROR --trusted-host $PIP_TRUSTED"
fi

# ── 2. 检测 CUDA 版本 ───────────────────────────────────────────────────
echo ""
echo "[1/5] 检测 CUDA 版本..."

CUDA_VERSION=""
TORCH_INDEX=""

# 尝试 nvcc
if command -v nvcc &> /dev/null; then
    CUDA_VERSION=$(nvcc --version | grep -oP 'release \K[0-9]+\.[0-9]+')
fi

# fallback: nvidia-smi
if [[ -z "$CUDA_VERSION" ]] && command -v nvidia-smi &> /dev/null; then
    CUDA_VERSION=$(nvidia-smi | grep -oP 'CUDA Version: \K[0-9]+\.[0-9]+')
fi

if [[ -n "$CUDA_VERSION" ]]; then
    echo ">>> 检测到 CUDA $CUDA_VERSION"
    CUDA_MAJOR=$(echo "$CUDA_VERSION" | cut -d. -f1)
    CUDA_MINOR=$(echo "$CUDA_VERSION" | cut -d. -f2)

    if [[ "$CUDA_MAJOR" -eq 11 ]]; then
        TORCH_INDEX="https://download.pytorch.org/whl/cu118"
        echo ">>> 将安装 PyTorch (CUDA 11.8)"
    elif [[ "$CUDA_MAJOR" -eq 12 ]]; then
        if [[ "$CUDA_MINOR" -le 1 ]]; then
            TORCH_INDEX="https://download.pytorch.org/whl/cu121"
            echo ">>> 将安装 PyTorch (CUDA 12.1)"
        elif [[ "$CUDA_MINOR" -le 4 ]]; then
            TORCH_INDEX="https://download.pytorch.org/whl/cu124"
            echo ">>> 将安装 PyTorch (CUDA 12.4)"
        else
            TORCH_INDEX="https://download.pytorch.org/whl/cu126"
            echo ">>> 将安装 PyTorch (CUDA 12.6)"
        fi
    else
        echo "[警告] 未识别的 CUDA 版本 $CUDA_VERSION，尝试 CUDA 12.4"
        TORCH_INDEX="https://download.pytorch.org/whl/cu124"
    fi
else
    echo ">>> 未检测到 CUDA，将安装 CPU 版本 PyTorch"
    TORCH_INDEX="https://download.pytorch.org/whl/cpu"
fi

# ── 3. 创建 Python 环境 ─────────────────────────────────────────────────
echo ""
echo "[2/5] 创建 Python 环境..."

if command -v conda &> /dev/null; then
    ENV_TYPE="conda"
    echo ">>> 检测到 conda，创建 conda 环境 'iqd'"

    # 配置 conda 国内镜像
    if [[ -n "$PIP_MIRROR" ]]; then
        echo ">>> 配置 conda 清华镜像..."
        conda config --set show_channel_urls yes
        conda config --remove-key channels 2>/dev/null || true
        conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
        conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free
        conda config --set custom_channels.pytorch https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
    fi

    # 如果环境已存在则跳过创建
    if conda env list | grep -q "^iqd "; then
        echo ">>> conda 环境 'iqd' 已存在，跳过创建"
    else
        conda create -n iqd python=3.10 -y
    fi
    eval "$(conda shell.bash hook)"
    conda activate iqd
else
    ENV_TYPE="venv"
    echo ">>> 未检测到 conda，使用 venv"

    if [[ ! -d "$PROJECT_DIR/.venv" ]]; then
        python3 -m venv "$PROJECT_DIR/.venv"
    else
        echo ">>> venv 已存在，跳过创建"
    fi
    source "$PROJECT_DIR/.venv/bin/activate"
fi

# ── 4. 安装 PyTorch ─────────────────────────────────────────────────────
echo ""
echo "[3/5] 安装 PyTorch..."
pip install $PIP_ARGS torch torchvision --index-url "$TORCH_INDEX"

# ── 5. 安装项目依赖 ─────────────────────────────────────────────────────
echo ""
echo "[4/5] 安装项目依赖..."
pip install $PIP_ARGS -r "$PROJECT_DIR/requirements.txt"

# ── 6. 创建目录 & 验证 ──────────────────────────────────────────────────
mkdir -p "$PROJECT_DIR/results/exp0" "$PROJECT_DIR/results/exp1" "$PROJECT_DIR/results/exp2"
mkdir -p "$PROJECT_DIR/data"

echo ""
echo "[5/5] 验证安装..."
python -c "
import torch
print(f'PyTorch:        {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU:            {torch.cuda.get_device_name(0)}')
    print(f'VRAM:           {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')
import mauve
print('MAUVE:          OK')
import transformers
print(f'Transformers:   {transformers.__version__}')
import peft
print(f'PEFT:           {peft.__version__}')
print()
print('所有依赖安装成功!')
"

echo ""
echo "========================================="
echo "  环境搭建完成!"
if [[ "$ENV_TYPE" == "conda" ]]; then
    echo "  激活环境: conda activate iqd"
else
    echo "  激活环境: source .venv/bin/activate"
fi
echo "  运行实验: bash scripts/run_all.sh"
echo "========================================="
