# IQD: Information Quality Density for Quantifying Multi-Generational AI Model Collapse

This research project investigates multi-generational AI model collapse and proposes the **IQD (Information Quality Density)** framework as a unified metric for quantifying it. Building on the "Strong Model Collapse" line of work (Dohmatob et al., ICLR 2025), we derive the origin of bias δ from first principles and use MAUVE scores to empirically track collapse across generations.

## Project Structure

```
.
├── docs/                   # Research notes and theoretical framework
├── references/             # Reference papers
├── src/                    # Experiment source code
│   ├── exp0_*.py           # Layer 0: Toy function experiments
│   ├── exp1_*.py           # Layer 1: Linear regression experiments
│   ├── exp2_*.py           # Layer 2: LLM collapse experiments
│   ├── exp3*.py            # Layer 3: Baseline comparison & MAUVE scan
│   ├── setup/              # Data preparation utilities
│   ├── train/              # Training pipeline
│   ├── eval/               # Evaluation metrics (PPL, MAUVE, diversity)
│   ├── analysis/           # Analysis and visualization
│   └── configs/            # Experiment grid configurations
├── results/                # Experiment outputs
├── figures/                # Publication-quality figures
├── data/                   # Data files
└── scripts/                # Shell scripts for setup and execution
```

## Setup

```bash
bash scripts/setup_env.sh
```

## Running Experiments

```bash
# Run all CPU experiments (Layer 0 + 1)
bash scripts/run_all.sh

# Run specific layer
bash scripts/run_all.sh 0    # Toy function (CPU)
bash scripts/run_all.sh 1    # Linear regression (CPU)
bash scripts/run_all.sh 2    # LLM collapse (GPU required)

# Run LLM experiment grid
bash scripts/run_llm_experiments.sh
```

## Key Experiments

1. **Exp 0** — Toy function: sanity check for bias accumulation
2. **Exp 1** — Linear regression: δ source decomposition, transfer function fitting, double descent
3. **Exp 2** — LLM multi-generational collapse with MAUVE tracking
4. **Exp 3** — Baseline comparison and MAUVE bias scanning

## Dependencies

See `requirements.txt`. Key packages: `torch`, `transformers`, `mauve-text`, `scikit-learn`, `matplotlib`.
