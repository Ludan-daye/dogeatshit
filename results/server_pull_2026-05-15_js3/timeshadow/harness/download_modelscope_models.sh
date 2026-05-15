#!/usr/bin/env bash
set -euo pipefail

ROOT="${TIMESHADOW_ROOT:-/data/timeshadow}"
MODEL_DIR="$ROOT/models"
LOG_DIR="$ROOT/logs"
LOG="$LOG_DIR/modelscope_download_queue.log"
LOCK="$LOG_DIR/modelscope_download_queue.lock"
MAX_WORKERS="${MAX_WORKERS:-8}"
EXCLUDES=(
  "original/*"
  "*.pth"
  "*.pt"
  "*.bin"
  "*.ckpt"
  "*.gguf"
  "*.onnx"
)

mkdir -p "$MODEL_DIR" "$LOG_DIR"

if ! mkdir "$LOCK" 2>/dev/null; then
  echo "$(date -Is) queue already running: $LOCK" | tee -a "$LOG"
  exit 0
fi
trap 'rmdir "$LOCK"' EXIT

log() {
  echo "$(date -Is) $*" | tee -a "$LOG"
}

wait_for_qwen35() {
  while ps -eo command | grep -F "modelscope download --model Qwen/Qwen3.5-9B " | grep -v grep >/dev/null; do
    log "waiting for active Qwen3.5 ModelScope download"
    sleep 60
  done
}

download_model() {
  local model_id="$1"
  local target_name="$2"
  local target_dir="$MODEL_DIR/$target_name"
  local marker="$target_dir/.timeshadow_modelscope_complete"

  mkdir -p "$target_dir"
  if [[ -f "$marker" ]]; then
    log "skip complete $model_id -> $target_dir"
    return 0
  fi
  if [[ -f "$target_dir/config.json" ]] &&
    [[ -f "$target_dir/tokenizer.json" || -f "$target_dir/tokenizer.model" ]] &&
    find "$target_dir" -maxdepth 1 -name '*.safetensors' | grep -q .; then
    touch "$marker"
    log "mark existing safetensors complete $model_id -> $target_dir"
    return 0
  fi

  log "downloading $model_id -> $target_dir"
  modelscope download \
    --model "$model_id" \
    --local_dir "$target_dir" \
    --exclude "${EXCLUDES[@]}" \
    --max-workers "$MAX_WORKERS" \
    >> "$LOG" 2>&1

  touch "$marker"
  log "complete $model_id -> $target_dir"
}

log "queue started"
wait_for_qwen35

download_model "LLM-Research/Meta-Llama-3.1-8B-Instruct" "Llama-3.1-8B-Instruct"
download_model "google/gemma-2-9b-it" "gemma-2-9b-it"
download_model "Qwen/Qwen2.5-7B-Instruct" "Qwen2.5-7B-Instruct"
download_model "01ai/Yi-1.5-9B-Chat" "Yi-1.5-9B-Chat"
download_model "mistralai/Mistral-7B-Instruct-v0.3" "Mistral-7B-Instruct-v0.3"

log "queue finished"
