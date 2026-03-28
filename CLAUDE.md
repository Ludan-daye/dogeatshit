# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **research project** (not a software codebase) investigating multi-generational AI model collapse and proposing the **IQD (Information Quality Density)** framework as a unified metric for quantifying it.

## Project Structure

- `docs/theoretical_framework.md` — Main research notes: theoretical framework, math derivations, experiment design, and paper outline
- `docs/experiment_plan.md` — Experiment planning notes
- `references/dohmatob2025_strong_model_collapse.pdf` — Reference paper (Dohmatob et al., "Strong Model Collapse", ICLR 2025)
- `src/` — Experiment source code (exp0–exp3, training pipeline, evaluation, analysis)
- `results/` — Experiment outputs (figures, CSVs, JSONs)
- `scripts/` — Shell scripts for environment setup and running experiments
- `figures/` — Publication-quality figures
- `data/` — Data files (e.g., real_texts.json)

## Research Context

The project builds on the "Strong Model Collapse" line of work and aims to:

1. **Derive the origin of δ** (bias between teacher and student models) from first principles, rather than assuming it
2. **Analyze multi-generation bias accumulation** with a transfer relation δₙ = f(δₙ₋₁) and collapse severity parameter α
3. **Define IQD** as a distribution-distance-based metric connecting geometric measures (c²) to information-theoretic quantities
4. **Use MAUVE scores** to empirically track collapse across generations (claimed as a first in the literature)

## Key Planned Experiments

1. δ source decomposition (vary regularization λ and sample size n)
2. IQD decay curve across generations (MAUVE tracking)
3. IQD equivalence between real and synthetic data
4. Collapse threshold IQD* identification

## Relevant Tools

- `mauve-text` Python package for MAUVE score computation
- Standard ML stack (PyTorch, scikit-learn) for model training
- See `requirements.txt` for full dependencies
