"""
可视化：δₖ 衰减曲线、α 扫描图、MAUVE vs PPL 散点图

用法：
  # 画实验1 δₖ 曲线
  python plot_results.py --exp-dir results/exp1 --plot delta

  # 画实验3 α vs p 曲线
  python plot_results.py --exp-dir results/exp3a --plot alpha_p

  # 画实验5 崩溃热力图
  python plot_results.py --exp-dir results/exp5 --plot heatmap
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

RESULTS_DIR = PROJECT_ROOT / "results"


# ── 数据加载 ───────────────────────────────────────────────────────────────

def load_all_metrics(exp_dir: str) -> pd.DataFrame:
    rows = []
    for p in Path(exp_dir).rglob("metrics.jsonl"):
        with open(p) as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))
    return pd.DataFrame(rows)


# ── 图 1：δₖ 衰减曲线（Exp 1）─────────────────────────────────────────────

def plot_delta_curves(df: pd.DataFrame, subdir: str = "exp1"):
    strategies = df["strategy"].unique() if "strategy" in df.columns else ["replace"]
    fig, axes  = plt.subplots(1, len(strategies), figsize=(7 * len(strategies), 5))
    if len(strategies) == 1:
        axes = [axes]

    for ax, strategy in zip(axes, strategies):
        sub  = df[df["strategy"] == strategy] if "strategy" in df.columns else df
        gens = sorted(sub["gen"].unique())
        means, stds = [], []
        for g in gens:
            vals = sub[sub["gen"] == g]["delta"].values
            means.append(np.mean(vals))
            stds.append(np.std(vals))
        means, stds = np.array(means), np.array(stds)

        ax.plot(gens, means, "o-", lw=2, label="均值 δₖ")
        ax.fill_between(gens, means - stds, np.clip(means + stds, 0, 1), alpha=0.2)

        # 指数拟合
        valid = [(g, m) for g, m in zip(gens, means) if m > 1e-6]
        if len(valid) >= 4:
            gs, ms   = zip(*valid)
            coeffs   = np.polyfit(gs, np.log(ms), 1)
            alpha_fit = np.exp(coeffs[0])
            g_fit    = np.linspace(min(gs), max(gs), 100)
            ax.plot(g_fit, np.exp(np.polyval(coeffs, g_fit)), "r--", alpha=0.7,
                    label=f"指数拟合 α={alpha_fit:.3f}")

        ax.set_xlabel("代数 k")
        ax.set_ylabel("δₖ = 1 − MAUVE")
        ax.set_title(f"策略: {strategy.upper()}")
        ax.legend(); ax.grid(True, alpha=0.3)

    fig.suptitle("δₖ 跨代演化（均值 ± 1σ，5 seeds）", fontsize=13)
    fig.tight_layout()
    save_fig(fig, "delta_curves", subdir)


# ── 图 2：α̂ vs 超参（Exp 3）─────────────────────────────────────────────

def _estimate_alpha(df_run: pd.DataFrame) -> float:
    """从单条链的 δ 序列估计线性 α"""
    deltas = df_run.sort_values("gen")["delta"].values
    if len(deltas) < 3:
        return np.nan
    pairs = [(deltas[i], deltas[i+1]) for i in range(len(deltas)-1) if deltas[i] > 1e-6]
    if not pairs:
        return np.nan
    ratios = [b / a for a, b in pairs]
    return float(np.median(ratios))


def plot_alpha_vs_param(df: pd.DataFrame, param: str, subdir: str):
    """param: 'p_syn' | 'n_train' | 'model'"""
    records = []
    for (param_val, seed), grp in df.groupby([param, "seed"]):
        alpha = _estimate_alpha(grp)
        records.append({param: param_val, "seed": seed, "alpha": alpha})

    rdf     = pd.DataFrame(records).dropna()
    grouped = rdf.groupby(param)["alpha"].agg(["mean", "std"]).reset_index()

    fig, ax = plt.subplots(figsize=(7, 5))
    xs   = range(len(grouped))
    lbls = [str(v) for v in grouped[param]]
    ax.bar(xs, grouped["mean"], yerr=grouped["std"].fillna(0),
           capsize=5, color="steelblue", alpha=0.8)
    ax.axhline(1.0, color="red", ls="--", lw=1.5, label="α = 1 (崩溃临界)")
    ax.set_xticks(xs); ax.set_xticklabels(lbls)
    ax.set_xlabel(param); ax.set_ylabel("α̂ (偏差传递系数)")
    ax.set_title(f"α̂ vs {param}")
    ax.legend(); ax.grid(True, axis="y", alpha=0.3)
    save_fig(fig, f"alpha_vs_{param}", subdir)


# ── 图 3：崩溃热力图 (Exp 5)───────────────────────────────────────────────

def plot_collapse_heatmap(df: pd.DataFrame, subdir: str = "exp5",
                          baseline_ppl: float = None):
    """
    横轴 n_train，纵轴 p_syn，格值 = 最终代 PPL / baseline_ppl
    颜色：绿=正常，红=崩溃
    """
    if baseline_ppl is None:
        baseline_ppl_path = PROJECT_ROOT / "results" / "baselines" / "baseline_ppl_gpt2.json"
        if baseline_ppl_path.exists():
            baseline_ppl = json.load(open(baseline_ppl_path))["ppl_real"]
        else:
            baseline_ppl = df["ppl_real"].min()  # fallback

    # 取每条链最后一代
    last = df.loc[df.groupby(["p_syn", "n_train"])["gen"].idxmax()].copy()
    last["ppl_ratio"] = last["ppl_real"] / baseline_ppl

    pivot = last.pivot_table(index="p_syn", columns="n_train",
                             values="ppl_ratio", aggfunc="mean")

    fig, ax = plt.subplots(figsize=(9, 6))
    im = ax.imshow(pivot.values, aspect="auto", cmap="RdYlGn_r",
                   vmin=1.0, vmax=2.5)
    plt.colorbar(im, ax=ax, label="PPL / Baseline PPL（>1.5 = 崩溃）")

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f"{v:,}" for v in pivot.columns])
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([f"{v:.2f}" for v in pivot.index])
    ax.set_xlabel("n_train（每代训练样本数）")
    ax.set_ylabel("p_syn（合成数据比例）")
    ax.set_title("崩溃热力图：PPL 比值（IQD* 边界）")

    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        color="white" if val > 1.8 else "black", fontsize=9)

    save_fig(fig, "collapse_heatmap", subdir)


# ── CLI ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-dir", required=True)
    parser.add_argument("--subdir",  default="exp1")
    parser.add_argument("--plot",
                        choices=["delta", "alpha_p", "alpha_n", "alpha_model",
                                 "heatmap", "all"],
                        default="delta")
    args = parser.parse_args()

    df = load_all_metrics(args.exp_dir)
    print(f"[*] 加载 {len(df)} 条记录，{df['exp_id'].nunique() if 'exp_id' in df.columns else '?'} 次运行")

    if args.plot in ("delta", "all"):
        plot_delta_curves(df, args.subdir)

    if args.plot in ("alpha_p", "all") and "p_syn" in df.columns:
        plot_alpha_vs_param(df, "p_syn", args.subdir)

    if args.plot in ("alpha_n", "all") and "n_train" in df.columns:
        plot_alpha_vs_param(df, "n_train", args.subdir)

    if args.plot in ("alpha_model", "all") and "model" in df.columns:
        plot_alpha_vs_param(df, "model", args.subdir)

    if args.plot in ("heatmap", "all"):
        plot_collapse_heatmap(df, args.subdir)


if __name__ == "__main__":
    main()
