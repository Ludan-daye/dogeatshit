#!/usr/bin/env bash
# run_exp10.sh — 单代训练 p_syn 扫描实验
# 用法：bash scripts/run_exp10.sh [all|cosmopedia|gptwiki]
set -euo pipefail

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

GRID="src/configs/experiment_grid_exp10.csv"
SYN_SOURCE="${1:-all}"

log() { echo "[$(date '+%H:%M:%S')] $*"; }

# ── Step 0: 准备真实数据 ─────────────────────────────────────────────────
if [[ ! -f data/real_texts.json ]]; then
    log "准备 OpenWebText 数据..."
    python src/setup/prepare_data.py
fi

# ── Step 1: 准备合成数据 ─────────────────────────────────────────────────
if [[ "$SYN_SOURCE" == "all" || "$SYN_SOURCE" == "cosmopedia" ]]; then
    if [[ ! -f data/syn_cosmopedia_texts.json ]]; then
        log "下载 Cosmopedia 合成数据..."
        python src/setup/prepare_data_synthetic.py --dataset cosmopedia
    else
        log "Cosmopedia 数据已存在，跳过"
    fi
fi

if [[ "$SYN_SOURCE" == "all" || "$SYN_SOURCE" == "gptwiki" ]]; then
    if [[ ! -f data/syn_gptwiki_texts.json ]]; then
        log "下载 GPT-wiki-intro 合成数据..."
        python src/setup/prepare_data_synthetic.py --dataset gptwiki
    else
        log "GPT-wiki-intro 数据已存在，跳过"
    fi
fi

# ── Step 2: 逐行运行实验 ─────────────────────────────────────────────────
log "===== 开始 exp10 实验 ====="

mapfile -t EXP_IDS < <(python3 -c "
import csv
with open('$GRID') as f:
    for row in csv.DictReader(f):
        print(row['exp_id'])
")

TOTAL=${#EXP_IDS[@]}
COUNT=0

for exp_id in "${EXP_IDS[@]}"; do
    COUNT=$((COUNT + 1))
    log "[$COUNT/$TOTAL] 运行 $exp_id ..."
    python src/train/run_single_gen.py --exp-id "$exp_id" --grid "$GRID" || {
        log "[!] $exp_id 失败，继续下一个"
        continue
    }
done

# ── Step 3: 分析绘图 ─────────────────────────────────────────────────────
log "生成图表..."
python src/analysis/plot_single_gen.py --exp-dir results/exp10 --subdir exp10 --plot all

log "===== exp10 实验全部完成! ====="
