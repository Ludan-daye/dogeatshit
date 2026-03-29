#!/usr/bin/env bash
# run_llm_experiments.sh
# 按顺序执行所有 LLM 崩溃实验
# 用法：bash run_llm_experiments.sh [--group exp1|exp3a|exp3b|exp3c|exp4|exp5|all]
set -euo pipefail

GRID="src/configs/experiment_grid.csv"
GROUP="${1:-all}"

log() { echo "[$(date '+%H:%M:%S')] $*"; }

# ── 检测 GPU 数量 ────────────────────────────────────────────────────────
N_GPUS=$(python -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo 1)
N_GPUS=${N_GPUS:-1}
log "检测到 $N_GPUS 张 GPU"

mkdir -p results/logs

# ── Step 0: 数据准备 ─────────────────────────────────────────────────────
if [[ ! -f data/real_texts.json ]]; then
    log "准备数据..."
    python src/setup/prepare_data.py
fi

if [[ ! -f results/baselines/baseline_ppl_gpt2.json ]]; then
    log "计算 baseline PPL..."
    python src/setup/baseline_ppl.py --model gpt2
fi

# ── Step 1: 读取网格，按 group 过滤并运行（多卡并行）────────────────────
run_group() {
    local grp="$1"
    log "===== 开始 group: $grp ====="
    # 用 Python 读出该 group 的所有 exp_id
    mapfile -t EXP_IDS < <(python - <<EOF
import csv
with open("$GRID") as f:
    for row in csv.DictReader(f):
        if row["group"] == "$grp":
            print(row["exp_id"])
EOF
)
    local idx=0
    local pids=()

    for exp_id in "${EXP_IDS[@]}"; do
        local gpu=$((idx % N_GPUS))
        log "  运行: $exp_id (GPU $gpu)"
        CUDA_VISIBLE_DEVICES=$gpu python src/train/run_chain.py \
            --exp-id "$exp_id" --grid "$GRID" \
            > "results/logs/${exp_id}.log" 2>&1 &
        pids+=($!)
        idx=$((idx + 1))

        # 每 N_GPUS 个实验等一轮完成
        if (( idx % N_GPUS == 0 )); then
            for pid in "${pids[@]}"; do wait "$pid"; done
            pids=()
        fi
    done
    # 等剩余的
    for pid in "${pids[@]}"; do wait "$pid"; done
    log "===== 完成 group: $grp ====="
}

if [[ "$GROUP" == "all" ]]; then
    for grp in exp1 exp3a exp3b exp5; do
        run_group "$grp"
    done
else
    run_group "$GROUP"
fi

# ── Step 2: 分析（仅在 exp1 完成后）─────────────────────────────────────
if [[ "$GROUP" == "all" || "$GROUP" == "exp1" ]]; then
    log "拟合传递函数 f(·)..."
    python src/analysis/fit_transfer_fn.py \
        --results-dirs results/exp1/*/ \
        --subdir exp2

    log "绘制 δₖ 衰减曲线..."
    python src/analysis/plot_results.py \
        --exp-dir results/exp1 --subdir exp1 --plot delta
fi

if [[ "$GROUP" == "all" || "$GROUP" == exp3* ]]; then
    log "绘制 α 扫描图..."
    python src/analysis/plot_results.py \
        --exp-dir results/exp3a --subdir exp3a --plot alpha_p
    python src/analysis/plot_results.py \
        --exp-dir results/exp3b --subdir exp3b --plot alpha_n
    python src/analysis/plot_results.py \
        --exp-dir results/exp3c --subdir exp3c --plot alpha_model
fi

if [[ "$GROUP" == "all" || "$GROUP" == "exp5" ]]; then
    log "绘制崩溃热力图..."
    python src/analysis/plot_results.py \
        --exp-dir results/exp5 --subdir exp5 --plot heatmap
fi

log "全部完成！结果保存在 results/"
