# Wait for all 3 training chains to exit
while pgrep -af run_chain_osmix > /dev/null; do sleep 60; done
echo "training done $(date)"
sleep 30
cd ~/ludan/reaserch/dogeatshit
for EXP in osmix_d1_p00_s42 osmix_d1_p50_s42 osmix_d1_p100_s42; do
  echo "=== $EXP final MAUVE $(date) ==="
  python3 src/analysis/compute_mauve_offline.py \
    --chain-dir results/osmix_d1/$EXP \
    --ref-file data/real_texts.json \
    --featurize-model ~/ludan/reaserch/models/gpt2-large 2>&1
done
echo "ALL FINAL MAUVE DONE $(date)"
