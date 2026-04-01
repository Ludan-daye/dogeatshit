"""
跨模型 / 跨数据集对比分析

生成 4 张核心图 + 汇总表：
  1. Delta 曲线对比（GPT-2 vs Llama-1B vs Llama-3B）
  2. Alpha vs 模型大小柱状图
  3. 数据集鲁棒性（OWT vs C4 vs Wiki delta 曲线）
  4. 混合比例对比（GPT-2 vs Llama 的 alpha vs p_syn）

用法：
  python compare_models.py                          # 使用默认目录
  python compare_models.py --results-dir results/   # 指定结果根目录
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
from experiments.utils import save_fig, save_csv
from experiments.analysis.plot_results import load_all_metrics, _estimate_alpha


SUBDIR = "compare"

# 模型显示名和大小映射
MODEL_LABELS = {
    "gpt2":                       ("GPT-2 (124M)", 124),
    "gpt2-medium":                ("GPT-2 Medium (355M)", 355),
    "mistralai/Mistral-7B-v0.1":   ("Mistral-7B", 7000),
}

DATASET_LABELS = {
    "owt":  "OpenWebText",
    "c4":   "C4",
    "wiki": "WikiText-103",
}

COLORS = {
    "gpt2":                       "#1f77b4",
    "gpt2-medium":                "#aec7e8",
    "mistralai/Mistral-7B-v0.1":   "#d62728",
}

DATASET_COLORS = {
    "owt":  "#2ca02c",
    "c4":   "#9467bd",
    "wiki": "#8c564b",
}


def _load_multi(results_dir: Path, groups: list) -> pd.DataFrame:
    """加载多个实验组的 metrics"""
    frames = []
    for grp in groups:
        grp_dir = results_dir / grp
        if grp_dir.exists():
            df = load_all_metrics(str(grp_dir))
            if len(df) > 0:
                frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


# ── 图 1：Delta 曲线 —— 跨模型对比 ─────────────────────────────────────────

def plot_delta_by_model(results_dir: Path):
    """
    对比 GPT-2 / Llama-1B / Llama-3B 的 δₖ 曲线
    数据来源：exp1 (GPT-2), exp6 (Llama-3B), exp7b (Llama-1B + Llama-3B)
    """
    df = _load_multi(results_dir, ["exp1", "exp6", "exp7b"])
    if df.empty:
        print("[跳过] 图1: 无数据")
        return

    # 只取 replace 策略，p_syn=1.0
    df = df[(df["strategy"] == "replace") & (df["p_syn"] >= 0.99)]

    fig, ax = plt.subplots(figsize=(10, 6))

    for model_id, grp in df.groupby("model"):
        label, _ = MODEL_LABELS.get(model_id, (model_id, 0))
        color = COLORS.get(model_id, "gray")
        gens = sorted(grp["gen"].unique())

        means, stds = [], []
        for g in gens:
            vals = grp[grp["gen"] == g]["delta"].values
            means.append(np.mean(vals))
            stds.append(np.std(vals))
        means, stds = np.array(means), np.array(stds)

        ax.plot(gens, means, "o-", color=color, lw=2, ms=5, label=label)
        ax.fill_between(gens, means - stds, np.clip(means + stds, 0, 1),
                        color=color, alpha=0.15)

    ax.set_xlabel("Generation k")
    ax.set_ylabel("δₖ = 1 − MAUVE  (collapse measure)")
    ax.set_title("Model Collapse Across Architectures\n(p_syn=1.0, replace strategy)")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    save_fig(fig, "delta_by_model", SUBDIR)
    print("[保存] 图1: delta_by_model")


# ── 图 2：Alpha vs 模型大小 ─────────────────────────────────────────────────

def plot_alpha_by_model(results_dir: Path):
    """柱状图：各模型的 α̂（偏差传递系数）"""
    df = _load_multi(results_dir, ["exp1", "exp6", "exp7b", "exp3c"])
    if df.empty:
        print("[跳过] 图2: 无数据")
        return

    df = df[(df["strategy"] == "replace") & (df["p_syn"] >= 0.99)]

    records = []
    for (model, seed), grp in df.groupby(["model", "seed"]):
        alpha = _estimate_alpha(grp)
        if not np.isnan(alpha):
            label, size = MODEL_LABELS.get(model, (model, 0))
            records.append({"model": model, "label": label,
                            "size": size, "alpha": alpha})

    rdf = pd.DataFrame(records)
    if rdf.empty:
        print("[跳过] 图2: 无法估计 alpha")
        return

    grouped = rdf.groupby(["model", "label", "size"])["alpha"].agg(
        ["mean", "std"]).reset_index().sort_values("size")

    fig, ax = plt.subplots(figsize=(8, 5))
    xs = range(len(grouped))
    colors = [COLORS.get(m, "gray") for m in grouped["model"]]
    ax.bar(xs, grouped["mean"], yerr=grouped["std"].fillna(0),
           capsize=5, color=colors, alpha=0.85)
    ax.axhline(1.0, color="red", ls="--", lw=1.5, label="α = 1 (collapse threshold)")
    ax.set_xticks(xs)
    ax.set_xticklabels(grouped["label"], rotation=15, ha="right")
    ax.set_ylabel("α̂ (bias transfer coefficient)")
    ax.set_title("Collapse Rate α̂ by Model Size")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    save_fig(fig, "alpha_by_model", SUBDIR)
    print("[保存] 图2: alpha_by_model")


# ── 图 3：跨数据集鲁棒性 ───────────────────────────────────────────────────

def plot_delta_by_dataset(results_dir: Path):
    """
    Llama-3B 在 OWT / C4 / Wiki 上的 δₖ 曲线
    数据来源：exp6 (OWT), exp8 (C4 + Wiki)
    """
    df = _load_multi(results_dir, ["exp6", "exp8"])
    if df.empty:
        print("[跳过] 图3: 无数据")
        return

    df = df[(df["strategy"] == "replace") & (df["p_syn"] >= 0.99)]

    # 推断 dataset 列（如果缺失则从 exp_id 推断）
    if "dataset" not in df.columns:
        def _infer_dataset(exp_id):
            if "exp8" in exp_id:
                # exp8_001-003 = c4, exp8_004-006 = wiki
                num = int(exp_id.split("_")[-1])
                return "c4" if num <= 3 else "wiki"
            return "owt"
        df["dataset"] = df["exp_id"].apply(_infer_dataset)

    fig, ax = plt.subplots(figsize=(10, 6))

    for dataset, grp in df.groupby("dataset"):
        label = DATASET_LABELS.get(dataset, dataset)
        color = DATASET_COLORS.get(dataset, "gray")
        gens = sorted(grp["gen"].unique())

        means, stds = [], []
        for g in gens:
            vals = grp[grp["gen"] == g]["delta"].values
            means.append(np.mean(vals))
            stds.append(np.std(vals))
        means, stds = np.array(means), np.array(stds)

        ax.plot(gens, means, "o-", color=color, lw=2, ms=5, label=label)
        ax.fill_between(gens, means - stds, np.clip(means + stds, 0, 1),
                        color=color, alpha=0.15)

    ax.set_xlabel("Generation k")
    ax.set_ylabel("δₖ = 1 − MAUVE")
    ax.set_title("Dataset Robustness: Llama-3.2-3B Collapse on Different Datasets\n"
                 "(p_syn=1.0, replace strategy)")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    save_fig(fig, "delta_by_dataset", SUBDIR)
    print("[保存] 图3: delta_by_dataset")


# ── 图 4：混合比例对比（GPT-2 vs Llama 的 alpha vs p_syn）──────────────────

def plot_alpha_vs_psyn_comparison(results_dir: Path):
    """
    GPT-2 (exp3a) vs Llama-3B (exp7a) 的 α̂ vs p_syn
    """
    df = _load_multi(results_dir, ["exp3a", "exp7a"])
    if df.empty:
        print("[跳过] 图4: 无数据")
        return

    fig, ax = plt.subplots(figsize=(9, 6))

    for model_id, model_grp in df.groupby("model"):
        label, _ = MODEL_LABELS.get(model_id, (model_id, 0))
        color = COLORS.get(model_id, "gray")

        records = []
        for (p_syn, seed), grp in model_grp.groupby(["p_syn", "seed"]):
            alpha = _estimate_alpha(grp)
            if not np.isnan(alpha):
                records.append({"p_syn": p_syn, "alpha": alpha})

        rdf = pd.DataFrame(records)
        if rdf.empty:
            continue

        grouped = rdf.groupby("p_syn")["alpha"].agg(["mean", "std"]).reset_index()
        ax.errorbar(grouped["p_syn"], grouped["mean"],
                    yerr=grouped["std"].fillna(0),
                    marker="o", lw=2, capsize=4, color=color, label=label)

    ax.axhline(1.0, color="red", ls="--", lw=1.5, alpha=0.5, label="α = 1")
    ax.set_xlabel("p_syn (synthetic data proportion)")
    ax.set_ylabel("α̂ (bias transfer coefficient)")
    ax.set_title("Collapse Rate vs Mixing Ratio: GPT-2 vs Llama-3B")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    save_fig(fig, "alpha_vs_psyn_comparison", SUBDIR)
    print("[保存] 图4: alpha_vs_psyn_comparison")


# ── 汇总表 ──────────────────────────────────────────────────────────────────

def save_summary_table(results_dir: Path):
    """输出 (model, dataset, p_syn, alpha_mean, alpha_std, final_delta_mean) 汇总表"""
    all_groups = ["exp1", "exp3a", "exp3c", "exp6", "exp7a", "exp7b", "exp8", "exp9"]
    df = _load_multi(results_dir, all_groups)
    if df.empty:
        print("[跳过] 汇总表: 无数据")
        return

    records = []
    for (model, p_syn, strategy), grp in df.groupby(["model", "p_syn", "strategy"]):
        alphas = []
        final_deltas = []
        for seed, run in grp.groupby("seed"):
            alpha = _estimate_alpha(run)
            if not np.isnan(alpha):
                alphas.append(alpha)
            last_gen = run.loc[run["gen"].idxmax()]
            final_deltas.append(last_gen["delta"])

        if alphas:
            label, size = MODEL_LABELS.get(model, (model, 0))
            records.append({
                "model": label,
                "model_size_M": size,
                "p_syn": p_syn,
                "strategy": strategy,
                "alpha_mean": np.mean(alphas),
                "alpha_std": np.std(alphas),
                "final_delta_mean": np.mean(final_deltas),
                "final_delta_std": np.std(final_deltas),
                "n_seeds": len(alphas),
            })

    summary = pd.DataFrame(records).sort_values(["model_size_M", "p_syn"])
    save_csv(summary.to_dict("list"), "summary_table", SUBDIR)
    print(f"[保存] 汇总表: {len(summary)} 条记录")


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=str,
                        default=str(PROJECT_ROOT / "results"),
                        help="结果根目录")
    parser.add_argument("--plot",
                        choices=["model", "alpha", "dataset", "mixing", "summary", "all"],
                        default="all")
    args = parser.parse_args()
    results_dir = Path(args.results_dir)

    if args.plot in ("model", "all"):
        plot_delta_by_model(results_dir)

    if args.plot in ("alpha", "all"):
        plot_alpha_by_model(results_dir)

    if args.plot in ("dataset", "all"):
        plot_delta_by_dataset(results_dir)

    if args.plot in ("mixing", "all"):
        plot_alpha_vs_psyn_comparison(results_dir)

    if args.plot in ("summary", "all"):
        save_summary_table(results_dir)


if __name__ == "__main__":
    main()
