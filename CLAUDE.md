# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Research project investigating multi-generational AI model collapse, proposing the **IQD (Information Quality Density)** framework as a unified metric. Builds on "Strong Model Collapse" (Dohmatob et al., ICLR 2025) and related work.

Core research goals:
- Derive δ (teacher-student bias) from first principles instead of assuming it
- Analyze multi-generation bias accumulation via transfer relation δₙ = f(δₙ₋₁)
- Define IQD connecting geometric measures (c²) to information-theoretic quantities
- Use MAUVE scores to empirically track collapse across generations

## Experiment Architecture

The project has a **three-layer experiment design** plus a newer **table-driven LLM experiment system**:

### Layer 0–2 (early experiments, `run_all.sh`)

- **Layer 0** (`experiments/exp0_toy_function.py`): Toy math functions on CPU — validates δ accumulation basics
- **Layer 1** (`experiments/exp1_linear_regression.py`): Linear regression + Gaussian — tests δ accumulation, double descent
- **Layer 2** (`experiments/exp2_llm_collapse.py`): LLM multi-gen collapse — requires GPU

### Table-driven LLM experiments (`run_llm_experiments.sh`)

The main experiment system. All runs are parameterized via `experiments/configs/experiment_grid.csv` with columns: `exp_id, group, model, dataset, p_syn, n_train, k_max, strategy, seed, notes`.

**Pipeline flow:**
1. **Data prep** → `experiments/setup/prepare_data.py` (OpenWebText) and `prepare_data_multi.py` (C4, WikiText-103) → outputs to `data/`
2. **Chain training** → `experiments/train/run_chain.py --exp-id <id> --grid <csv>` — runs a full k-generation chain for one grid row. Calls `train_one_gen.py` (fine-tune + generate) per generation
3. **Evaluation** (per generation, inline in run_chain) → MAUVE (δₖ = 1 − MAUVE), perplexity on real data, n-gram repetition rate. All metrics appended to `metrics.jsonl`
4. **Analysis** → `experiments/analysis/fit_transfer_fn.py` (fits δₙ₊₁ = f(δₙ) with 4 candidate models, selects via AIC/BIC), `plot_results.py` (δ curves, α scans, collapse heatmaps), `compare_models.py` (cross-model/cross-dataset comparison with 4 plot types + summary table)

**Experiment groups:**
- `exp1`: GPT-2 baseline collapse chain (replace vs accumulate, 5 seeds, 15 gens)
- `exp3a/3b/3c`: α sensitivity — sweep p_syn / n_train / model size (GPT-2 vs GPT-2-medium)
- `exp4`: IQD equivalence validation (3 gens only)
- `exp5`: Collapse threshold IQD* identification (p_syn × n_train heatmap, 16 conditions)
- `exp6`: Mistral-7B baseline collapse chain (replace + accumulate, 3 seeds, 10 gens)
- `exp7a/7b`: Mistral-7B α sweeps (p_syn with 6 levels incl. 0.0, model size comparison)
- `exp8`: Cross-dataset — Mistral-7B on C4 and WikiText-103
- `exp9`: Cross-architecture — GPT-2 vs Mistral-7B on C4
- `exp10`: Single-gen p_syn sweep — pre-existing AI data (Cosmopedia/GPT-wiki) mixed with real data, 8 p_syn levels × 3 seeds

**Key design patterns:**
- Each run outputs to `results/<group>/<exp_id>/` with `metrics.jsonl`, `models/gen_k/`, `samples/gen_k.json`
- Supports **checkpoint resume**: skips generations where model + samples already exist
- Two data mixing strategies: `replace` (use only last-gen synthetic) vs `accumulate` (pool all prior synthetic)
- Three metrics per generation: `delta` (1−MAUVE), `ppl_real` (perplexity), `rep_rate` (n-gram repetition)
- `compare_models.py` depends on `plot_results.py` (imports `load_all_metrics`, `_estimate_alpha`)

## Commands

```bash
# Environment setup (GPU server with conda)
bash setup.sh                          # creates conda env "iqd", installs deps

# Remote GPU server quickstart (hardcoded path: /root/iqd_research)
bash start.sh                         # checks GPU, downloads data + Mistral-7B, launches exp6_r_001

# Early experiments (Layer 0-2)
bash run_all.sh 0                      # Layer 0 only (CPU)
bash run_all.sh 1                      # Layer 1 only (CPU)
bash run_all.sh 2                      # Layer 2 only (GPU required)
bash run_all.sh 2 2a                   # Layer 2, sub-experiment 2a only

# Table-driven LLM experiments
bash run_llm_experiments.sh all        # run all groups
bash run_llm_experiments.sh exp1       # run one group

# Single experiment chain
python experiments/train/run_chain.py --exp-id exp1_r_001 --grid experiments/configs/experiment_grid.csv

# Single generation (standalone)
python experiments/train/train_one_gen.py --prev-model gpt2 --train-texts data/train_texts.json --output-dir results/test/gen_0

# Data preparation (standalone)
python experiments/setup/prepare_data.py                        # OpenWebText
python experiments/setup/prepare_data_multi.py                  # C4 + WikiText-103
python experiments/setup/prepare_data_multi.py --dataset c4     # C4 only

# Analysis
python experiments/analysis/plot_results.py --exp-dir results/exp1 --subdir exp1 --plot delta
python experiments/analysis/fit_transfer_fn.py --results-dirs results/exp1/*/ --subdir exp2
python experiments/analysis/compare_models.py --plot all        # cross-model/dataset comparison

# Exp10: single-gen p_syn sweep with pre-existing AI datasets
python src/setup/prepare_data_synthetic.py --dataset cosmopedia  # download Cosmopedia
python src/setup/prepare_data_synthetic.py --dataset gptwiki     # download GPT-wiki-intro
bash scripts/run_exp10.sh                                        # run all exp10
python src/train/run_single_gen.py --exp-id exp10_001 --grid src/configs/experiment_grid_exp10.csv
python src/analysis/plot_single_gen.py --exp-dir results/exp10 --plot all
```

## Environment Notes

- Conda env name: `iqd` (Python 3.10)
- HuggingFace mirror: `HF_ENDPOINT=https://hf-mirror.com` (set in scripts for China servers)
- GPU memory: code aggressively calls `clear_gpu_memory()` after each model use — critical for 16GB VRAM cards
- Auto-selects bf16 on compute capability >= 8 (A100+), fp16 otherwise
- Matplotlib configured for headless (`Agg` backend) with Chinese font fallback
- `start.sh` assumes remote server path `/root/iqd_research` — adjust if deploying elsewhere

## Key Files

- `idea/未命名.md` — Full theoretical framework, math derivations, paper outline
- `ebc042e767de551803ccfcc45e2454f5-Paper-Conference.pdf` — Reference paper (Dohmatob et al.)
- `3.19/` — Earlier experiment results (exp0/exp1/exp2) from initial validation run
- `results/` — Current experiment outputs
- `experiments/utils.py` — Shared utilities: MAUVE computation, GPU memory management, plotting helpers, Timer context manager
