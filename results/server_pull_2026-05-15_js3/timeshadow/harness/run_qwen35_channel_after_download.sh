#!/usr/bin/env bash
set -euo pipefail

cd /data/timeshadow
source .venv/bin/activate

MODEL="/data/timeshadow/models/Qwen3.5-9B"
LOG="/data/timeshadow/logs/qwen35_channel_after_download.log"

echo "==== $(date -Is) waiting for Qwen3.5-9B download ====" | tee -a "$LOG"

deadline=$((SECONDS + 12 * 3600))
while true; do
  count=$(find "$MODEL" -maxdepth 1 -name '*.safetensors' 2>/dev/null | wc -l | tr -d ' ')
  if [ "$count" -ge 4 ] && [ -f "$MODEL/config.json" ] && [ -f "$MODEL/tokenizer.json" ]; then
    break
  fi
  if [ "$SECONDS" -ge "$deadline" ]; then
    echo "==== $(date -Is) timeout waiting for Qwen3.5-9B; found $count safetensors ====" | tee -a "$LOG"
    exit 2
  fi
  echo "$(date -Is) still waiting; safetensors=$count" >> "$LOG"
  sleep 60
done

OUT="/data/timeshadow/results/channel_probe_qwen35_9b_n30_$(date +%Y%m%d_%H%M%S).json"
echo "==== $(date -Is) starting channel probe -> $OUT ====" | tee -a "$LOG"
CUDA_VISIBLE_DEVICES=0 python harness/run_channel_probe.py \
  --model "$MODEL" \
  --n 30 \
  --max-new-tokens 180 \
  --out "$OUT" 2>&1 | tee -a "$LOG"

echo "==== $(date -Is) finished channel probe ====" | tee -a "$LOG"
