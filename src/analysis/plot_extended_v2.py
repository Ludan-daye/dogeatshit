"""扩展指标 plot — distinct-N / token entropy / KL / 长度方差"""
import json, sys
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BASE = PROJECT_ROOT / "results/osmix_d1_v2_planc"

CHAINS = [
    ("osmix_d1_p00_s42", "p_syn=0.0 (baseline)", "#2E7D32"),
    ("osmix_d1_p50_s42", "p_syn=0.5 (mid)",      "#1565C0"),
    ("osmix_d1_p100_s42", "p_syn=1.0 (full syn)", "#C62828"),
]

def load(exp):
    return [json.loads(l) for l in open(BASE / exp / "metrics_full.jsonl") if l.strip()]

data = [(label, color, load(exp)) for exp, label, color in CHAINS]

fig, axs = plt.subplots(2, 3, figsize=(16, 9))
((ax_d1, ax_d2, ax_d3), (ax_H, ax_kl, ax_len)) = axs

for label, color, rows in data:
    gens = [r["gen"] for r in rows]
    ax_d1.plot(gens, [r["distinct_1"] for r in rows], "o-", color=color, label=label, lw=2, ms=7)
    ax_d2.plot(gens, [r["distinct_2"] for r in rows], "o-", color=color, label=label, lw=2, ms=7)
    ax_d3.plot(gens, [r["distinct_3"] for r in rows], "o-", color=color, label=label, lw=2, ms=7)
    ax_H.plot(gens,  [r["token_entropy_bits"] for r in rows], "o-", color=color, label=label, lw=2, ms=7)
    ax_kl.plot(gens, [r["kl_to_real"] for r in rows], "o-", color=color, label=label, lw=2, ms=7)
    ax_len.errorbar(gens, [r["mean_length_words"] for r in rows],
                    yerr=[r["std_length_words"] for r in rows],
                    color=color, label=label, lw=2, marker="o", ms=6, capsize=3)

for ax, title, ylabel in [
    (ax_d1, "distinct-1 (unigram diversity)", "distinct_1"),
    (ax_d2, "distinct-2 (bigram diversity)",  "distinct_2"),
    (ax_d3, "distinct-3 (trigram diversity)", "distinct_3"),
    (ax_H,  "Token Shannon entropy (bits)",   "H (bits)"),
    (ax_kl, "KL(P_gen || P_real) token-freq", "KL"),
    (ax_len, "Mean gen length ± std (words)", "length"),
]:
    ax.set_title(title); ax.set_xlabel("Generation"); ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3); ax.legend(fontsize=8, loc="best")

fig.suptitle("v2 Plan-C 扩展指标:生成多样性 + 分布漂移(独立于 MAUVE)", fontsize=14, y=1.00)
fig.tight_layout()
out = BASE / "plots/extended_metrics.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"saved {out}")
