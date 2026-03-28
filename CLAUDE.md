# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **research project** (not a software codebase) investigating multi-generational AI model collapse and proposing the **IQD (Information Quality Density)** framework as a unified metric for quantifying it.

## Contents

- `idea/未命名.md` — Main research notes: theoretical framework, math derivations, experiment design, and paper outline
- `ebc042e767de551803ccfcc45e2454f5-Paper-Conference.pdf` — Reference paper (Dohmatob et al., "Strong Model Collapse", ICLR 2025)
- Saved WordPress page — Blog post on Linux learning (unrelated to the core research)

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

The experiment code (not yet written) is expected to use:
- `mauve-text` Python package for MAUVE score computation
- Standard ML stack for ridge regression / model training
