"""完整 6 dose 点剂量-响应图 (PPL + MAUVE + KL trajectories)"""
import json
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BASE = PROJECT_ROOT / "results/osmix_d1_v2_planc"

CHAINS = [
    ("osmix_d1_p00_s42",  "p=0.0 baseline", "#2E7D32"),
    ("osmix_d1_p10_s42",  "p=0.1",          "#43A047"),
    ("osmix_d1_p20_s42",  "p=0.2",          "#FB8C00"),
    ("osmix_d1_p30_s42",  "p=0.3",          "#E65100"),
    ("osmix_d1_p50_s42",  "p=0.5",          "#1565C0"),
    ("osmix_d1_p100_s42", "p=1.0",          "#C62828"),
]

def load_mauve(exp):
    return [json.loads(l) for l in open(BASE / exp / "metrics_with_mauve.jsonl") if l.strip()]

def load_ext(exp):
    return [json.loads(l) for l in open(BASE / exp / "metrics_full.jsonl") if l.strip()]

fig, axs = plt.subplots(1, 3, figsize=(17, 5.5))
ax_ppl, ax_mauve, ax_kl = axs

for exp, label, color in CHAINS:
    mauve_rows = load_mauve(exp)
    ext_rows = {r["gen"]: r for r in load_ext(exp)}

    gens = [r["gen"] for r in mauve_rows]
    ppls = [r["ppl_real"] for r in mauve_rows]
    mauves = [r["mauve"] for r in mauve_rows]
    kls = [ext_rows.get(r["gen"], {}).get("kl_to_real", None) for r in mauve_rows]

    ax_ppl.plot(gens, ppls, "o-", color=color, label=label, lw=2, ms=7)
    ax_mauve.plot(gens, mauves, "o-", color=color, label=label, lw=2, ms=7)
    ax_kl.plot(gens, [k for k in kls if k is not None], "o-", color=color, label=label, lw=2, ms=7)

ax_ppl.set_yscale("log"); ax_ppl.set_xlabel("Generation"); ax_ppl.set_ylabel("PPL on real")
ax_ppl.set_title("PPL trajectory (log y)"); ax_ppl.grid(True, alpha=0.3); ax_ppl.legend(fontsize=8, loc="upper left")

ax_mauve.set_xlabel("Generation"); ax_mauve.set_ylabel("MAUVE"); ax_mauve.set_ylim(0, 1)
ax_mauve.set_title("MAUVE trajectory"); ax_mauve.grid(True, alpha=0.3); ax_mauve.legend(fontsize=8, loc="lower right")

ax_kl.set_xlabel("Generation"); ax_kl.set_ylabel("KL(P_gen || P_real)")
ax_kl.set_title("KL trajectory (token-freq)"); ax_kl.grid(True, alpha=0.3); ax_kl.legend(fontsize=8, loc="upper right")

fig.suptitle("6-Point Dose-Response: Mistral-7B+LoRA, Plan-C non-overlap", fontsize=14, y=1.02)
fig.tight_layout()
out = BASE / "plots/dose_response_all.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"saved {out}")
