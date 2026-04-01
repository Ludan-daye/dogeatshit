#!/bin/bash
set -e
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DOWNLOAD_TIMEOUT=300
cd /root/iqd_research

echo '=== Step 1: 检查 GPU ==='
python3 -c 'import torch; print(f"GPU: {torch.cuda.device_count()}x {torch.cuda.get_device_name(0)}")' || { echo 'ERROR: 未检测到GPU，请先挂载显卡'; exit 1; }

echo '=== Step 2: 准备数据 ==='
if [ ! -f data/real_texts.json ]; then
    echo '下载 OpenWebText...'
    python3 experiments/setup/prepare_data.py
fi
if [ ! -f data/c4_real_texts.json ]; then
    echo '下载 C4 / WikiText-103...'
    python3 experiments/setup/prepare_data_multi.py
fi

echo '=== Step 3: 下载 Mistral-7B 模型 ==='
python3 << 'PYEOF'
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "300"
from huggingface_hub import snapshot_download
import time
for attempt in range(10):
    try:
        print(f"[Attempt {attempt+1}] Downloading mistralai/Mistral-7B-v0.1...")
        path = snapshot_download("mistralai/Mistral-7B-v0.1", max_workers=1)
        print(f"MODEL READY: {path}")
        break
    except Exception as e:
        print(f"Failed: {e}")
        print("Retrying in 10s...")
        time.sleep(10)
PYEOF

echo '=== Step 4: 验证环境 ==='
python3 -c '
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer
print(f"PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}")
print("All OK!")
'

echo '=== Step 5: 启动实验 ==='
echo '开始 exp6_r_001 (Mistral-7B 基准崩溃链)...'
nohup python3 experiments/train/run_chain.py \
    --exp-id exp6_r_001 \
    --grid experiments/configs/experiment_grid.csv \
    > exp6_r_001.log 2>&1 &

BGPID=$!
echo "实验已在后台启动 (PID: ${BGPID})"
echo "查看进度: tail -f /root/iqd_research/exp6_r_001.log"
