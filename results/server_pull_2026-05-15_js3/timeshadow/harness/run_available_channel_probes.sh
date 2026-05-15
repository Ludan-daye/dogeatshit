#!/usr/bin/env bash
set -euo pipefail

ROOT="${TIMESHADOW_ROOT:-/data/timeshadow}"
LOG="$ROOT/logs/channel_probe_queue.log"
mkdir -p "$ROOT/logs" "$ROOT/results"

log() {
  echo "$(date -Is) $*" | tee -a "$LOG"
}

model_ready() {
  local dir="$1"
  [[ -f "$dir/config.json" ]] || return 1
  [[ -f "$dir/tokenizer.json" || -f "$dir/tokenizer.model" ]] || return 1
  if [[ -f "$dir/model.safetensors.index.json" ]]; then
    python3 - "$dir" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
index = json.loads((root / "model.safetensors.index.json").read_text())
files = set(index.get("weight_map", {}).values())
missing = [name for name in files if not (root / name).exists()]
raise SystemExit(1 if missing else 0)
PY
    return $?
  fi
  find "$dir" -maxdepth 1 -name '*.safetensors' | grep -q .
}

wait_model() {
  local name="$1"
  local dir="$2"
  local deadline=$((SECONDS + 24 * 3600))
  while ! model_ready "$dir"; do
    if [[ "$SECONDS" -ge "$deadline" ]]; then
      log "timeout waiting for $name at $dir"
      return 1
    fi
    local shards
    shards=$(find "$dir" -maxdepth 1 -name '*.safetensors' 2>/dev/null | wc -l | tr -d ' ')
    log "waiting for $name; safetensors=$shards"
    sleep 120
  done
}

run_probe() {
  local slug="$1"
  local name="$2"
  local dir="$3"
  if ls "$ROOT/results/channel_probe_${slug}_n30_"*.json >/dev/null 2>&1; then
    log "skip $name; n30 result already exists"
    return 0
  fi
  wait_model "$name" "$dir"
  local out="$ROOT/results/channel_probe_${slug}_n30_$(date +%Y%m%d_%H%M%S).json"
  log "start $name channel probe -> $out"
  cd "$ROOT"
  source "$ROOT/.venv/bin/activate"
  CUDA_VISIBLE_DEVICES=0 python harness/run_channel_probe.py \
    --model "$dir" \
    --n 30 \
    --max-new-tokens 180 \
    --out "$out" 2>&1 | tee -a "$LOG"
  log "finished $name channel probe -> $out"
}

log "channel probe queue started"
run_probe "llama31_8b" "Llama-3.1-8B-Instruct" "$ROOT/models/Llama-3.1-8B-Instruct"
run_probe "gemma2_9b" "gemma-2-9b-it" "$ROOT/models/gemma-2-9b-it"
run_probe "qwen25_7b" "Qwen2.5-7B-Instruct" "$ROOT/models/Qwen2.5-7B-Instruct"
run_probe "yi15_9b" "Yi-1.5-9B-Chat" "$ROOT/models/Yi-1.5-9B-Chat"
run_probe "mistral7b" "Mistral-7B-Instruct-v0.3" "$ROOT/models/Mistral-7B-Instruct-v0.3"
log "channel probe queue finished"
