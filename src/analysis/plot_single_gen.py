"""
单代训练实验（exp10）可视化：delta / PPL / rep_rate vs p_syn

用法：
  python plot_single_gen.py --exp-dir results/exp10 --plot all
  python plot_single_gen.py --exp-dir results/exp10 --plot delta
  python plot_single_gen.py --exp-dir results/exp10 --plot panel
"""

import json
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from src.utils import save_fig


# ── 数据加载 ───────────────────────────────────────────────────────────────

def load_all_metrics(exp_dir: str) -> pd.DataFrame:
    rows = []
    for p in Path(exp_dir).rglob("metrics.jsonl"):
        with open(p) as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))
    return pd.DataFrame(rows)


def _aggregate_by_psyn(df: pd.DataFrame, metric: str):
    """按 p_syn 分组，返回 (p_syn_values, means, stds)"""
    grouped = df.groupby("p_syn")[metric].agg(["mean", "std"]).reset_index()
    grouped = grouped.sort_values("p_syn")
    return (grouped["p_syn"].values,
            grouped["mean"].values,
            grouped["std"].fillna(0).values)


# ── 图 1: delta vs p_syn ─────────────────────────────────────────────────

def plot_delta_vs_psyn(df: pd.DataFrame, subdir: str = "exp10"):
    xs, means, stds = _aggregate_by_psyn(df, "delta")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(xs, means, "o-", color="steelblue", lw=2, markersize=8, label="mean delta")
    ax.fill_between(xs, means - stds, np.clip(means + stds, 0, 1), alpha=0.2, color="steelblue")

    # Garg optimal anchor
    ax.axvline(x=0.38, color="orange", ls="--", lw=1.5, alpha=0.7, label="1-1/phi ~ 0.38 (Garg)")

    ax.set_xlabel("p_syn (synthetic data proportion)")
    ax.set_ylabel("delta = 1 - MAUVE")
    ax.set_title("Model Quality Degradation vs Synthetic Data Proportion")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.05, 1.05)
    save_fig(fig, "delta_vs_psyn", subdir)


# ── 图 2: PPL vs p_syn ───────────────────────────────────────────────────

def plot_ppl_vs_psyn(df: pd.DataFrame, subdir: str = "exp10"):
    xs, means, stds = _aggregate_by_psyn(df, "ppl_real")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(xs, means, "s-", color="crimson", lw=2, markersize=8, label="mean PPL")
    ax.fill_between(xs, means - stds, means + stds, alpha=0.2, color="crimson")

    ax.axvline(x=0.38, color="orange", ls="--", lw=1.5, alpha=0.7, label="1-1/phi ~ 0.38")

    ax.set_xlabel("p_syn (synthetic data proportion)")
    ax.set_ylabel("Perplexity on real data")
    ax.set_title("Perplexity vs Synthetic Data Proportion")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.05, 1.05)
    save_fig(fig, "ppl_vs_psyn", subdir)


# ── 图 3: rep_rate vs p_syn ──────────────────────────────────────────────

def plot_rep_vs_psyn(df: pd.DataFrame, subdir: str = "exp10"):
    xs, means, stds = _aggregate_by_psyn(df, "rep_rate")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(xs, means, "^-", color="forestgreen", lw=2, markersize=8, label="mean rep_rate")
    ax.fill_between(xs, means - stds, means + stds, alpha=0.2, color="forestgreen")

    ax.set_xlabel("p_syn (synthetic data proportion)")
    ax.set_ylabel("4-gram repetition rate")
    ax.set_title("Repetition Rate vs Synthetic Data Proportion")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.05, 1.05)
    save_fig(fig, "rep_vs_psyn", subdir)


# ── 图 4: 三指标面板 ─────────────────────────────────────────────────────

def plot_panel(df: pd.DataFrame, subdir: str = "exp10"):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    metrics_cfg = [
        ("delta",    "delta = 1 - MAUVE", "steelblue", "o"),
        ("ppl_real", "Perplexity",        "crimson",   "s"),
        ("rep_rate", "4-gram rep rate",   "forestgreen","^"),
    ]

    for ax, (metric, ylabel, color, marker) in zip(axes, metrics_cfg):
        xs, means, stds = _aggregate_by_psyn(df, metric)
        ax.plot(xs, means, f"{marker}-", color=color, lw=2, markersize=7)
        ax.fill_between(xs, means - stds, means + stds, alpha=0.2, color=color)
        ax.axvline(x=0.38, color="orange", ls="--", lw=1, alpha=0.5)
        ax.set_xlabel("p_syn")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-0.05, 1.05)

    fig.suptitle("Single-Generation Training: Impact of Synthetic Data Proportion", fontsize=13)
    fig.tight_layout()
    save_fig(fig, "panel_3metrics", subdir)


# ── 图 5: 跨源对比 ───────────────────────────────────────────────────────

def plot_cross_source(df: pd.DataFrame, subdir: str = "exp10"):
    if "syn_source" not in df.columns or df["syn_source"].nunique() < 2:
        print("[跳过] 跨源对比需要至少 2 个 syn_source")
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = {"cosmopedia": "steelblue", "gptwiki": "coral"}

    for source in df["syn_source"].unique():
        sub = df[df["syn_source"] == source]
        xs, means, stds = _aggregate_by_psyn(sub, "delta")
        c = colors.get(source, "gray")
        ax.plot(xs, means, "o-", color=c, lw=2, markersize=7, label=source)
        ax.fill_between(xs, means - stds, np.clip(means + stds, 0, 1), alpha=0.15, color=c)

    ax.set_xlabel("p_syn (synthetic data proportion)")
    ax.set_ylabel("delta = 1 - MAUVE")
    ax.set_title("Cross-Source Comparison: Domain Alignment Effect")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.05, 1.05)
    save_fig(fig, "cross_source_delta", subdir)


# ── 汇总表 ───────────────────────────────────────────────────────────────

def save_summary_table(df: pd.DataFrame, subdir: str = "exp10"):
    group_cols = ["syn_source", "p_syn"] if "syn_source" in df.columns else ["p_syn"]
    summary = df.groupby(group_cols).agg(
        delta_mean=("delta", "mean"),
        delta_std=("delta", "std"),
        ppl_mean=("ppl_real", "mean"),
        ppl_std=("ppl_real", "std"),
        rep_mean=("rep_rate", "mean"),
        n_runs=("exp_id", "count"),
    ).reset_index()

    from src.utils import save_csv
    save_csv(summary.to_dict("list"), "summary_exp10", subdir)


# ── CLI ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-dir", required=True,
                        help="实验结果目录 (如 results/exp10)")
    parser.add_argument("--subdir", default="exp10",
                        help="图表保存子目录")
    parser.add_argument("--plot",
                        choices=["delta", "ppl", "rep", "panel",
                                 "cross_source", "summary", "all"],
                        default="all")
    args = parser.parse_args()

    df = load_all_metrics(args.exp_dir)
    print(f"[*] 加载 {len(df)} 条记录")

    if len(df) == 0:
        print("[!] 无数据，退出")
        return

    if args.plot in ("delta", "all"):
        plot_delta_vs_psyn(df, args.subdir)

    if args.plot in ("ppl", "all"):
        plot_ppl_vs_psyn(df, args.subdir)

    if args.plot in ("rep", "all"):
        plot_rep_vs_psyn(df, args.subdir)

    if args.plot in ("panel", "all"):
        plot_panel(df, args.subdir)

    if args.plot in ("cross_source", "all"):
        plot_cross_source(df, args.subdir)

    if args.plot in ("summary", "all"):
        save_summary_table(df, args.subdir)


if __name__ == "__main__":
    main()
