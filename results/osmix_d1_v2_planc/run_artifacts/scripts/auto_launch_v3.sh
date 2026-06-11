# Wait for both data files to reach target sizes (more reliable than pgrep)
echo "watchdog v3 start $(date)"
while true; do
  WT=$(python3 -c "import json; print(len(json.load(open(\"/home/vicuna/ludan/reaserch/dogeatshit/data/train_texts.json\"))))" 2>/dev/null || echo 0)
  CS=$(python3 -c "import json; print(len(json.load(open(\"/home/vicuna/ludan/reaserch/dogeatshit/data/syn_cosmopedia_texts.json\"))))" 2>/dev/null || echo 0)
  echo "[$(date +%H:%M:%S)] WikiText pool: $WT | Cosmopedia pool: $CS"
  if [ "$WT" -ge 30000 ] && [ "$CS" -ge 30000 ]; then break; fi
  sleep 60
done
echo "BOTH POOLS READY $(date) — launching chains"

cd ~/ludan/reaserch/dogeatshit
LOGD=$HOME/ludan/reaserch/logs
for EXP in osmix_d1_p00_s42 osmix_d1_p50_s42 osmix_d1_p100_s42; do
  rm -f $LOGD/chain_${EXP}.log
  nohup python3 src/train/run_chain_osmix.py --exp-id $EXP --grid src/configs/experiment_grid_osmix.csv > $LOGD/chain_${EXP}.log 2>&1 </dev/null & disown
  echo "$EXP -> pid $!"
done
echo "ALL CHAINS v2 LAUNCHED $(date)"
