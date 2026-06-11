#!/bin/bash
# dogeatshit 3-chain status dashboard
BASE=$HOME/ludan/reaserch/dogeatshit
CHAINS=("osmix_d1_p00_s42|p_syn=0.0 (baseline)" "osmix_d1_p50_s42|p_syn=0.5 (mid)" "osmix_d1_p100_s42|p_syn=1.0 (full)")

echo "========================================"
echo "  3-CHAIN STATUS @ $(date '+%H:%M:%S')"
echo "========================================"
for entry in "${CHAINS[@]}"; do
  EXP="${entry%%|*}"
  LABEL="${entry##*|}"
  LOG=$HOME/ludan/reaserch/logs/chain_${EXP}.log
  RESDIR=$BASE/results/osmix_d1/$EXP
  METRICS=$RESDIR/metrics.jsonl
  echo ""
  echo "── ${LABEL} [$EXP]"
  # process status
  PID=$(pgrep -f "exp-id $EXP " | head -1)
  if [ -n "$PID" ]; then
    ETIME=$(ps -o etime= -p $PID 2>/dev/null | tr -d ' ')
    echo "   running  pid=$PID  elapsed=$ETIME"
  else
    echo "   NOT RUNNING (check log for completion or error)"
  fi
  # metrics count
  if [ -f "$METRICS" ]; then
    NGEN=$(wc -l < "$METRICS")
    LAST_GEN=$(tail -1 "$METRICS" | python3 -c "import sys,json; d=json.loads(sys.stdin.read()); print(f'gen={d[\"gen\"]} ppl={d[\"ppl_real\"]:.2f} rep={d[\"rep_rate\"]:.3f}')" 2>/dev/null)
    echo "   metrics  $NGEN/11 gens done  | last: $LAST_GEN"
  else
    echo "   metrics  0/11 gens done"
  fi
  # latest meaningful log line (skip progress bars)
  if [ -f "$LOG" ]; then
    LAST=$(grep -aE "===|\[完成\]|\[开始\]|gen[0-9]+ p=|Error|Traceback|ABORT" "$LOG" 2>/dev/null | tail -1 | head -c 150)
    echo "   last log: $LAST"
  fi
done
echo ""
echo "────────────────────────────────────────"
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu,power.draw --format=csv,noheader 2>&1 | awk -F',' '{printf "GPU: %s / %s (%s util, %s)\n", $1, $2, $3, $4}'
df -h /mnt/d5f4cfb6-8afe-40a4-8650-2965046cd208 2>/dev/null | tail -1 | awk '{printf "DISK: %s used of %s (%s)\n", $3, $2, $5}'
