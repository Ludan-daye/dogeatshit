"""
plot_osmix_v2.py — 画 v2 Plan-C 主结果的 3 张核心图

输入:results/osmix_d1_v2_planc/<chain>/metrics_with_mauve.jsonl
输出:results/osmix_d1_v2_planc/plots/{ppl_curves,mauve_curves,ppl_vs_mauve_scatter,combined}.png
"""
import json
import sys
import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[2]

CHAINS = [
    ("osmix_d1_p00_s42", "p_syn = 0.0 (baseline)", "#2E7D32"),
    ("osmix_d1_p50_s42", "p_syn = 0.5 (mid)",     "#1565C0"),
    ("osmix_d1_p100_s42","p_syn = 1.0 (full syn)","#C62828"),
]


def load_chain(base, exp_id):
    rows = []
    for line in open(base / exp_id / "metrics_with_mauve.jsonl"):
        if line.strip():
            rows.append(json.loads(line))
    rows.sort(key=lambda r: r["gen"])
    return rows


def plot_ppl(base, out_dir, data):
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for (exp, label, color), rows in zip(CHAINS, data):
        gens = [r["gen"] for r in rows]
        ppls = [r["ppl_real"] for r in rows]
        ax.plot(gens, ppls, "o-", color=color, label=label, lw=2.2, ms=8)
    ax.axhline(8.32, ls=":", color="gray", alpha=0.5, label="gen 0 starting PPL")
    ax.set_xlabel("Generation", fontsize=12)
    ax.set_ylabel("PPL on held-out real data", fontsize=12)
    ax.set_title("Mistral-7B + LoRA 多代续训:真实数据 PPL 轨迹\n(Plan-C 每代不重叠子集)", fontsize=12)
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(fontsize=10, loc="upper left")
    out = out_dir / "ppl_curves.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")


def plot_mauve(base, out_dir, data):
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for (exp, label, color), rows in zip(CHAINS, data):
        gens = [r["gen"] for r in rows]
        mauves = [r["mauve"] for r in rows]
        ax.plot(gens, mauves, "o-", color=color, label=label, lw=2.2, ms=8)
    ax.set_xlabel("Generation", fontsize=12)
    ax.set_ylabel("MAUVE vs real data (higher = better aligned)", fontsize=12)
    ax.set_title("Mistral-7B + LoRA 多代续训:MAUVE 分布对齐轨迹\n揭示 PPL 看不到的 stealth collapse", fontsize=12)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10, loc="lower left")

    # Annotate the stealth collapse for p=0.5
    p50 = data[1]
    if len(p50) >= 2:
        x_drop = p50[1]["gen"]
        y_drop = p50[1]["mauve"]
        ax.annotate("stealth collapse:\nPPL ≈ 基线\nMAUVE 一代 −73%",
                    xy=(x_drop, y_drop), xytext=(x_drop + 1.5, y_drop + 0.15),
                    fontsize=9, color="#1565C0",
                    arrowprops=dict(arrowstyle="->", color="#1565C0", lw=1.2))

    out = out_dir / "mauve_curves.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")


def plot_scatter(base, out_dir, data):
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for (exp, label, color), rows in zip(CHAINS, data):
        ppls = np.array([r["ppl_real"] for r in rows])
        mauves = np.array([r["mauve"] for r in rows])
        gens = np.array([r["gen"] for r in rows])
        # marker size grows with gen so trajectory direction is visible
        sizes = 30 + gens * 25
        sc = ax.scatter(ppls, mauves, s=sizes, c=color, alpha=0.7, edgecolor="black", lw=0.6, label=label)
        # connect them in order
        ax.plot(ppls, mauves, "-", color=color, alpha=0.4, lw=1.2)
        # label first and last gen
        ax.annotate(f"g{gens[0]}", (ppls[0], mauves[0]), fontsize=7, ha="center", va="center")
        ax.annotate(f"g{gens[-1]}", (ppls[-1], mauves[-1]), fontsize=7, ha="center", va="center", fontweight="bold")

    ax.set_xlabel("PPL on real data (log)", fontsize=12)
    ax.set_ylabel("MAUVE vs real data", fontsize=12)
    ax.set_xscale("log")
    ax.set_title("PPL × MAUVE 联合轨迹 (圆点大小 ∝ gen,每条链一种颜色)", fontsize=12)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(fontsize=10, loc="upper right")
    out = out_dir / "ppl_vs_mauve_scatter.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")


def plot_combined(base, out_dir, data):
    fig, axs = plt.subplots(2, 2, figsize=(13, 9))
    ((ax_ppl, ax_mauve), (ax_rep, ax_scatter)) = axs

    for (exp, label, color), rows in zip(CHAINS, data):
        gens = [r["gen"] for r in rows]
        ppls = [r["ppl_real"] for r in rows]
        mauves = [r["mauve"] for r in rows]
        reps = [r["rep_rate"] for r in rows]

        ax_ppl.plot(gens, ppls, "o-", color=color, label=label, lw=2.2, ms=7)
        ax_mauve.plot(gens, mauves, "o-", color=color, label=label, lw=2.2, ms=7)
        ax_rep.plot(gens, reps, "o-", color=color, label=label, lw=2.2, ms=7)

        sizes = np.array([30 + g * 25 for g in gens])
        ax_scatter.scatter(ppls, mauves, s=sizes, c=color, alpha=0.7, edgecolor="black", lw=0.6, label=label)
        ax_scatter.plot(ppls, mauves, "-", color=color, alpha=0.4, lw=1.0)

    ax_ppl.set_yscale("log"); ax_ppl.set_xlabel("Generation"); ax_ppl.set_ylabel("PPL (real)")
    ax_ppl.set_title("PPL on real data"); ax_ppl.grid(True, alpha=0.3); ax_ppl.legend(fontsize=8)

    ax_mauve.set_xlabel("Generation"); ax_mauve.set_ylabel("MAUVE"); ax_mauve.set_ylim(0, 1)
    ax_mauve.set_title("MAUVE vs real data"); ax_mauve.grid(True, alpha=0.3); ax_mauve.legend(fontsize=8)

    ax_rep.set_xlabel("Generation"); ax_rep.set_ylabel("rep_rate (4-gram)")
    ax_rep.set_title("Repetition rate"); ax_rep.grid(True, alpha=0.3); ax_rep.legend(fontsize=8)

    ax_scatter.set_xscale("log"); ax_scatter.set_xlabel("PPL (log)"); ax_scatter.set_ylabel("MAUVE")
    ax_scatter.set_ylim(0, 1)
    ax_scatter.set_title("PPL × MAUVE (size ∝ gen)"); ax_scatter.grid(True, alpha=0.3, which="both"); ax_scatter.legend(fontsize=8)

    fig.suptitle("v2 Plan-C: 3 chains × {6, 11, 7} gens · Mistral-7B + LoRA · WikiText-103 ⊕ Cosmopedia",
                 fontsize=13, y=1.00)
    out = out_dir / "combined.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=str(PROJECT_ROOT / "results/osmix_d1_v2_planc"))
    args = ap.parse_args()

    base = Path(args.base)
    out_dir = base / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    data = [load_chain(base, exp) for exp, _, _ in CHAINS]
    for (exp, label, _), rows in zip(CHAINS, data):
        print(f"{exp}: {len(rows)} gens loaded ({label})")

    plot_ppl(base, out_dir, data)
    plot_mauve(base, out_dir, data)
    plot_scatter(base, out_dir, data)
    plot_combined(base, out_dir, data)
    print(f"\nall plots in {out_dir}")


if __name__ == "__main__":
    main()
