"""
拟合偏差传递函数 f(·)：δₙ₊₁ = f(δₙ)

从多条实验链的 metrics.jsonl 中提取 (δₙ, δₙ₊₁) 对，拟合四种候选模型：
  M1 线性：  δₙ₊₁ = α · δₙ
  M2 幂律：  δₙ₊₁ = C · δₙ^γ
  M3 仿射：  δₙ₊₁ = α · δₙ + β
  M4 饱和：  δₙ₊₁ = δ_∞(1 − e^{−δₙ/τ})

用 AIC/BIC 选择最优模型，输出散点图 + 拟合曲线。
"""

import json
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.optimize import curve_fit

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from src.utils import save_fig, ensure_dir

RESULTS_DIR = PROJECT_ROOT / "results"


# ── 候选模型 ───────────────────────────────────────────────────────────────

def _m1(x, alpha):                         return alpha * x
def _m2(x, C, gamma):                      return C * np.power(np.clip(x, 1e-10, None), gamma)
def _m3(x, alpha, beta):                   return alpha * x + beta
def _m4(x, delta_inf, tau):                return delta_inf * (1 - np.exp(-x / np.clip(tau, 1e-10, None)))

MODELS = {
    "M1_linear":   (_m1, ["α"],             1, [1.1],        (0, np.inf)),
    "M2_power":    (_m2, ["C","γ"],          2, [1.0, 1.2],   (0, np.inf)),
    "M3_affine":   (_m3, ["α","β"],          2, [1.0, 0.01],  (-np.inf, np.inf)),
    "M4_saturate": (_m4, ["δ_∞","τ"],       2, [0.5, 0.2],   (0, np.inf)),
}


# ── 数据加载 ───────────────────────────────────────────────────────────────

def collect_pairs(results_dirs: list) -> np.ndarray:
    """从多条链的 metrics.jsonl 提取 (δₙ, δₙ₊₁) 对"""
    pairs = []
    for d in results_dirs:
        p = Path(d) / "metrics.jsonl"
        if not p.exists():
            print(f"  [警告] 找不到 {p}")
            continue
        rows = [json.loads(l) for l in open(p) if l.strip()]
        df   = pd.DataFrame(rows).sort_values("gen").reset_index(drop=True)
        deltas = df["delta"].values
        for i in range(len(deltas) - 1):
            pairs.append((deltas[i], deltas[i+1]))
    return np.array(pairs) if pairs else np.empty((0, 2))


# ── 拟合单模型 ─────────────────────────────────────────────────────────────

def _fit_one(name, func, param_names, n_params, p0, bounds, x, y):
    try:
        popt, _ = curve_fit(func, x, y, p0=p0, bounds=bounds, maxfev=20000)
    except Exception as e:
        print(f"  [{name}] 拟合失败: {e}")
        return None

    y_pred = func(x, *popt)
    sse    = float(np.sum((y - y_pred) ** 2))
    n      = len(y)
    if sse <= 0 or n <= n_params + 1:
        return None

    sigma2  = sse / n
    log_lik = -n/2 * np.log(2 * np.pi * sigma2) - sse / (2 * sigma2)
    k_aic   = n_params + 1          # 参数数 + sigma
    aic     = 2 * k_aic - 2 * log_lik
    bic     = k_aic * np.log(n) - 2 * log_lik
    ss_tot  = float(np.sum((y - np.mean(y)) ** 2))
    r2      = 1 - sse / ss_tot if ss_tot > 0 else 0.0

    return {
        "name":   name,
        "params": dict(zip(param_names, popt.tolist())),
        "sse":    sse,
        "aic":    float(aic),
        "bic":    float(bic),
        "r2":     r2,
        "n":      n,
    }


# ── 主拟合流程 ─────────────────────────────────────────────────────────────

def fit_transfer_fn(pairs: np.ndarray, subdir: str = "exp2") -> list:
    x, y = pairs[:, 0], pairs[:, 1]

    results = []
    for name, (func, pnames, n_p, p0, bounds) in MODELS.items():
        res = _fit_one(name, func, pnames, n_p, p0, bounds, x, y)
        if res:
            results.append(res)
            print(f"  {name:<14}  AIC={res['aic']:8.2f}  BIC={res['bic']:8.2f}  R²={res['r2']:.4f}")

    if not results:
        print("[错误] 所有模型均拟合失败")
        return []

    results.sort(key=lambda r: r["aic"])
    best = results[0]
    print(f"\n[最优 by AIC] {best['name']}  参数={best['params']}  R²={best['r2']:.4f}")

    # 保存 JSON
    out_dir = RESULTS_DIR / subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "transfer_fn_fit.json", "w") as f:
        json.dump({"best_model": best, "all_models": results, "n_pairs": len(pairs)},
                  f, indent=2)

    # 散点图 + 拟合曲线
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(x, y, alpha=0.45, s=22, color="steelblue", zorder=5,
               label=f"数据点 (n={len(pairs)})")

    x_rng  = np.linspace(x.min() * 0.9, x.max() * 1.1, 300)
    colors = ["#E74C3C", "#2ECC71", "#F39C12", "#9B59B6"]
    for i, r in enumerate(results):
        func   = MODELS[r["name"]][0]
        y_fit  = func(x_rng, *list(r["params"].values()))
        style  = "-" if i == 0 else "--"
        ax.plot(x_rng, y_fit, color=colors[i], lw=2, ls=style,
                label=f"{r['name']}  AIC={r['aic']:.1f}  R²={r['r2']:.3f}")

    lim = max(x.max(), y.max()) * 1.05
    ax.plot([0, lim], [0, lim], "k:", alpha=0.3, label="δₙ₊₁ = δₙ（不变）")
    ax.set_xlabel("δₙ（第 n 代偏差）",   fontsize=12)
    ax.set_ylabel("δₙ₊₁（第 n+1 代偏差）", fontsize=12)
    ax.set_title("偏差传递函数 f(·): δₙ₊₁ = f(δₙ)", fontsize=13)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    save_fig(fig, "transfer_fn_scatter", subdir)

    return results


# ── CLI ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dirs", nargs="+", required=True,
                        help="包含 metrics.jsonl 的运行目录列表")
    parser.add_argument("--subdir", default="exp2", help="图片/JSON 输出子目录")
    args = parser.parse_args()

    print("[*] 收集 (δₙ, δₙ₊₁) 数据对 ...")
    pairs = collect_pairs(args.results_dirs)
    print(f"    共 {len(pairs)} 对")

    if len(pairs) < 5:
        print("[错误] 数据点不足，请先完成实验 1")
        return

    print("\n[*] 拟合候选模型 ...")
    fit_transfer_fn(pairs, args.subdir)


if __name__ == "__main__":
    main()
