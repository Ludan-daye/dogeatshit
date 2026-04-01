#!/usr/bin/env bash
# run_llm_experiments.sh
# 按顺序执行所有 LLM 崩溃实验
# 用法：bash run_llm_experiments.sh [--group exp1|exp3a|exp3b|exp3c|exp4|exp5|exp6|exp7a|exp7b|exp8|exp9|all]
set -euo pipefail

# 国内服务器使用 HuggingFace 镜像
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

GRID="experiments/configs/experiment_grid.csv"
GROUP="${1:-all}"

log() { echo "[$(date '+%H:%M:%S')] $*"; }

# ── Step 0: 数据准备 ─────────────────────────────────────────────────────
if [[ ! -f data/real_texts.json ]]; then
    log "准备 OpenWebText 数据..."
    python experiments/setup/prepare_data.py
fi

# 准备 C4 / WikiText-103（Llama 跨数据集实验需要）
if [[ "$GROUP" == "all" || "$GROUP" == exp8* || "$GROUP" == exp9* ]]; then
    if [[ ! -f data/c4_real_texts.json || ! -f data/wiki_real_texts.json ]]; then
        log "准备 C4 / WikiText-103 数据..."
        python experiments/setup/prepare_data_multi.py
    fi
fi

if [[ ! -f results/baselines/baseline_ppl_gpt2.json ]]; then
    log "计算 baseline PPL (GPT-2)..."
    python experiments/setup/baseline_ppl.py --model gpt2
fi

# ── Step 1: 读取网格，按 group 过滤并运行 ────────────────────────────────
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
    for exp_id in "${EXP_IDS[@]}"; do
        log "  运行: $exp_id"
        python experiments/train/run_chain.py --exp-id "$exp_id" --grid "$GRID"
    done
    log "===== 完成 group: $grp ====="
}

if [[ "$GROUP" == "all" ]]; then
    for grp in exp1 exp3a exp3b exp3c exp4 exp5 exp6 exp7a exp7b exp8 exp9; do
        run_group "$grp"
    done
else
    run_group "$GROUP"
fi

# ── Step 2: 分析（仅在对应实验完成后）─────────────────────────────────────
if [[ "$GROUP" == "all" || "$GROUP" == "exp1" ]]; then
    log "拟合传递函数 f(·)..."
    python experiments/analysis/fit_transfer_fn.py \
        --results-dirs results/exp1/*/ \
        --subdir exp2

    log "绘制 δₖ 衰减曲线..."
    python experiments/analysis/plot_results.py \
        --exp-dir results/exp1 --subdir exp1 --plot delta
fi

if [[ "$GROUP" == "all" || "$GROUP" == exp3* ]]; then
    log "绘制 α 扫描图..."
    python experiments/analysis/plot_results.py \
        --exp-dir results/exp3a --subdir exp3a --plot alpha_p
    python experiments/analysis/plot_results.py \
        --exp-dir results/exp3b --subdir exp3b --plot alpha_n
    python experiments/analysis/plot_results.py \
        --exp-dir results/exp3c --subdir exp3c --plot alpha_model
fi

if [[ "$GROUP" == "all" || "$GROUP" == "exp5" ]]; then
    log "绘制崩溃热力图..."
    python experiments/analysis/plot_results.py \
        --exp-dir results/exp5 --subdir exp5 --plot heatmap
fi

# ── Step 3: Llama 实验分析 ───────────────────────────────────────────────
if [[ "$GROUP" == "all" || "$GROUP" == "exp6" ]]; then
    log "绘制 Llama 基准 δₖ 衰减曲线..."
    python experiments/analysis/plot_results.py \
        --exp-dir results/exp6 --subdir exp6 --plot delta
fi

if [[ "$GROUP" == "all" || "$GROUP" == "exp7a" ]]; then
    log "绘制 Llama 混合比例 α 扫描..."
    python experiments/analysis/plot_results.py \
        --exp-dir results/exp7a --subdir exp7a --plot alpha_p
fi

if [[ "$GROUP" == "all" || "$GROUP" == "exp7b" ]]; then
    log "绘制 Llama 模型大小对照..."
    python experiments/analysis/plot_results.py \
        --exp-dir results/exp7b --subdir exp7b --plot alpha_model
fi

# ── Step 4: 跨模型 / 跨数据集对比分析 ───────────────────────────────────
if [[ "$GROUP" == "all" ]]; then
    log "运行跨模型对比分析..."
    python experiments/analysis/compare_models.py --plot all
fi

log "全部完成！结果保存在 results/"
