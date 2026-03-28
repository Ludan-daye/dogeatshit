#!/bin/bash
# IQD实验一键运行脚本
# 用法: bash run_all.sh [0|1|2|all]
# 示例:
#   bash run_all.sh 0     # 只跑第零层
#   bash run_all.sh 1     # 只跑第一层
#   bash run_all.sh 2     # 只跑第二层（需要GPU）
#   bash run_all.sh 2 2a  # 只跑第二层的实验2a
#   bash run_all.sh all   # 全部跑（默认）

set -e

LAYER=${1:-all}
SUB_EXP=${2:-all}
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

cd "$SCRIPT_DIR/experiments"

# 创建结果目录
mkdir -p "$SCRIPT_DIR/results/exp0"
mkdir -p "$SCRIPT_DIR/results/exp1"
mkdir -p "$SCRIPT_DIR/results/exp2"

echo "========================================="
echo "  IQD 验证实验"
echo "  层级: $LAYER"
echo "========================================="

# 第零层：可控函数环境（CPU）
if [[ "$LAYER" == "all" || "$LAYER" == "0" ]]; then
    echo ""
    echo ">>> 第零层：可控数学函数环境 (CPU)"
    echo "========================================="
    python exp0_toy_function.py
    echo ""
    echo ">>> 第零层完成!"
    echo ">>> 检查 results/exp0/ 中的结果"
    echo ">>> 如果E₃ - E₂ > 0，继续第一层"
    echo "========================================="
fi

# 第一层：线性回归（CPU）
if [[ "$LAYER" == "all" || "$LAYER" == "1" ]]; then
    echo ""
    echo ">>> 第一层：线性回归 + 高斯分布 (CPU)"
    echo "========================================="
    python exp1_linear_regression.py
    echo ""
    echo ">>> 第一层完成!"
    echo ">>> 检查 results/exp1/ 中的结果"
    echo ">>> 如果δ累积和double descent验证通过，继续第二层"
    echo "========================================="
fi

# 第二层：LLM实验（GPU）
if [[ "$LAYER" == "all" || "$LAYER" == "2" ]]; then
    echo ""
    echo ">>> 第二层：LLM多代崩溃 (GPU)"
    echo "========================================="

    # 检查GPU
    if ! python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
        echo "[错误] 未检测到可用GPU，跳过第二层"
        exit 1
    fi

    python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}, VRAM: {torch.cuda.get_device_properties(0).total_mem/1024**3:.1f}GB')"

    python exp2_llm_collapse.py --exp "$SUB_EXP"
    echo ""
    echo ">>> 第二层完成!"
    echo ">>> 检查 results/exp2/ 中的结果"
    echo "========================================="
fi

echo ""
echo "========================================="
echo "  所有实验完成!"
echo "  结果目录: $SCRIPT_DIR/results/"
echo "========================================="
