# TimeShadow Rerun Plan (2026-05-15)

Purpose: rebuild the missing evidence chain on the new A100 server under `/data/timeshadow`.

## Current Server State

- SSH aliases: `timeshadow` (`vipuser`) and `timeshadow-root` (`root`) work without password.
- Project directory: `/data/timeshadow`.
- GPU: NVIDIA A100-SXM4-80GB.
- Existing model cache is mostly VGIC-related Qwen VL / Qwen3.6 assets, not the exact TimeShadow model set.
- Local project snapshot is copied to `/data/timeshadow/local_snapshot`.

## Priority A: Required For A Submission-Ready Paper

1. Rebuild the executable harness with full per-trial logging.
2. Re-run the HERO base-RLHF experiment (`M15v4c2` equivalent):
   - Llama-3.1-8B-Instruct and Gemma-2-9b-it if available.
   - Four trigger conditions, n=90 per condition.
   - Save full response text and strict labels.
3. Re-run Phase-2 cross-OOD (`M30c` equivalent):
   - Six models x seven OOD trigger modes x n=45.
   - If exact models are gated or unavailable, record substitutions explicitly and do not merge with old numbers.
4. Re-run UNSEEN harm-category evaluation (`M15v5c` equivalent):
   - Five models x four conditions x n=40.
   - Full per-trial logging.
5. Re-run cross-model LangGraph-style scheduler attack (`M39` equivalent):
   - Llama / Gemma / Mistral, two conditions, n=21.
6. Re-run judge classification over every new response:
   - Primary strict prompt with DEFENSE / ATTACK / AMBIG.
   - Self-consistency prompt variant over a random subset.
   - Hand-label spot-check must be an actual file, not only a stated claim.

## Priority B: Defense Evidence

1. Recreate the controlled refusal adapter or replace it with a clearly marked fresh adapter.
2. Recreate D3 channel-aware adapter training with preserved training data.
3. Re-run adaptive D3:
   - Qwen-like available model, Llama, Gemma.
   - 50 triggers x 10 reps = 500 per model.
4. Re-run benign utility:
   - 30 benign tasks x 2 reps per configuration.
   - USEFUL / REFUSED / BROKEN v2 CoT-aware judge.

## Priority C: Repair Existing Claims

1. Rejudge old framework raw logs:
   - LangGraph, CrewAI, AutoGen old `ASR_clean` was regex/script-based.
   - Use strict labels and report clean refusals separately.
2. Letta:
   - Only run after a tool-call-capable backend is available.
   - If not run, keep as structural blocker and do not report an ASR number.
3. 4-model x 4-framework matrix:
   - Very high cost.
   - Do not restore this table unless all sixteen cells have raw per-trial files.
4. Mechanism probing:
   - Keep as feature-existence unless causal localization has a benign-preserving positive result.

## Integrity Rules

- Every experiment must save:
  - input metadata,
  - model path / revision,
  - adapter path / revision,
  - condition,
  - seed or deterministic setting,
  - full response text,
  - raw classifier / judge label,
  - summary counts.
- No paper claim should cite a summary-only file when the per-trial raw file is missing.
- New-server reruns must be named separately from old-server runs; do not silently overwrite old experiment IDs.
