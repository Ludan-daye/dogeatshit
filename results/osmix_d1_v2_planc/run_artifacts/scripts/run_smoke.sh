set +e
cd ~/ludan/reaserch/dogeatshit || exit 1
export PATH="$HOME/.local/bin:$PATH"
export HF_ENDPOINT=https://hf-mirror.com
export MAUVE_FEATURIZE_MODEL=$HOME/models/gpt2-large
MODELDIR=$HOME/models/Mistral-7B-v0.1

echo "=== [smoke] start $(date) ==="
echo "torch: $(python3 -c 'import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")')"



ls -la data/*.json 2>&1

cat > src/configs/experiment_grid_osmix_smoke.csv <<CSV
exp_id,group,model,real_dataset,ai_dataset,mode,p_syn,n_train,k_max,seed,notes
osmix_smoke,_smoke,$MODELDIR,owt,cosmopedia,constant,0.5,200,2,42,smoke
CSV

echo "=== SMOKE run (2 gen, n_train=200) $(date) ==="
python3 src/train/run_chain_osmix.py --exp-id osmix_smoke --grid src/configs/experiment_grid_osmix_smoke.csv --results-base results/_smoke 2>&1 | tail -60

echo "=== smoke metrics ==="
cat results/_smoke/osmix_smoke/metrics.jsonl 2>&1
echo ""
echo "=== [smoke] DONE $(date) ==="
