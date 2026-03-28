"""
baseline 对比实验（含定向偏差版本）

三条链对比：
  - Real 链    ：每代用新鲜真实数据训练，无偏差
  - Syn 链     ：每代用上一代模型输出（synthetic），无额外偏差
  - Biased 链  ：synthetic 标签叠加固定定向偏差 bias_fn(x)，代代累积

定向偏差的含义：
  bias_fn(x) 是确定性函数（非随机），每代方向一致。
  这模拟了真实场景中模型的系统性偏差 δ（如正则化收缩方向固定）。
  区别于随机噪声：随机噪声跨代均值为0，定向偏差跨代单调累积。

指标：Wasserstein 距离（连续值场景的 MAUVE 类比）
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import wasserstein_distance
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge

from experiments.utils import save_fig, save_csv

# ── 参数 ──────────────────────────────────────────────────────────────
N_SAMPLES     = 1000
NOISE_STD     = 0.5
POLY_DEGREE   = 5
RIDGE_ALPHA   = 0.01
N_GENERATIONS = 15
N_REPEATS     = 5
X_RANGE       = (-3, 3)
EVAL_N        = 2000
SUBDIR        = 'baseline_compare'

# 定向偏差扫描值
BIAS_STRENGTHS = [0.0, 0.1, 0.3, 0.5, 1.0, 2.0]

# ── 真实函数 ───────────────────────────────────────────────────────────
def f_star(x):
    return 2 * x**2 + 3 * x + 1

# ── 定向偏差函数（固定方向，非随机）──────────────────────────────────
# 选线性项 bias_fn(x) = BIAS_STRENGTH * x：
#   在 x>0 区间系统性高估，在 x<0 区间系统性低估
#   方向始终一致，每代在上一代偏差基础上再叠加
def bias_fn(x, strength):
    return strength * x

# ── 工具函数 ───────────────────────────────────────────────────────────
def fit(x_train, y_train):
    poly = PolynomialFeatures(POLY_DEGREE)
    X    = poly.fit_transform(x_train)
    m    = Ridge(alpha=RIDGE_ALPHA)
    m.fit(X, y_train)
    return m, poly

def predict(m, poly, x):
    return m.predict(poly.transform(x))

# 用固定网格评估，消除采样方差
X_EVAL = np.linspace(X_RANGE[0], X_RANGE[1], EVAL_N).reshape(-1, 1)
Y_TRUE = f_star(X_EVAL).ravel()

def dist_to_real(m, poly):
    y_pred = predict(m, poly, X_EVAL)
    return wasserstein_distance(Y_TRUE, y_pred)

def sample_real(n, rng):
    x = rng.uniform(X_RANGE[0], X_RANGE[1], size=(n, 1))
    y = f_star(x).ravel() + rng.normal(0, NOISE_STD, n)
    return x, y

def sample_synthetic(m, poly, n, rng, strength=0.0):
    x = rng.uniform(X_RANGE[0], X_RANGE[1], size=(n, 1))
    y = predict(m, poly, x) + bias_fn(x, strength).ravel()
    return x, y


def run_one_chain(strength, rng_seed):
    """单条 biased synthetic 链，返回 (N_REPEATS, N_GENERATIONS+1) 的 W 距离矩阵"""
    dists = np.zeros((N_REPEATS, N_GENERATIONS + 1))
    for r in range(N_REPEATS):
        rng = np.random.RandomState(rng_seed + r)
        x0, y0 = sample_real(N_SAMPLES, rng)
        m, poly = fit(x0, y0)
        dists[r, 0] = dist_to_real(m, poly)
        for k in range(1, N_GENERATIONS + 1):
            xb, yb = sample_synthetic(m, poly, N_SAMPLES, rng, strength)
            m, poly = fit(xb, yb)
            dists[r, k] = dist_to_real(m, poly)
    return dists


# ── 主实验：扫描 BIAS_STRENGTHS ────────────────────────────────────────
def run():
    gens = list(range(N_GENERATIONS + 1))

    # Real baseline（strength=0，每代用真实数据）
    real_dists = np.zeros((N_REPEATS, N_GENERATIONS + 1))
    for r in range(N_REPEATS):
        rng = np.random.RandomState(42 + r)
        x0, y0 = sample_real(N_SAMPLES, rng)
        m, poly = fit(x0, y0)
        real_dists[r, 0] = dist_to_real(m, poly)
        for k in range(1, N_GENERATIONS + 1):
            xr, yr = sample_real(N_SAMPLES, rng)
            m, poly = fit(xr, yr)
            real_dists[r, k] = dist_to_real(m, poly)
    r_m, r_s = real_dists.mean(0), real_dists.std(0)

    # 各 bias strength 的结果
    results = {}
    for strength in BIAS_STRENGTHS:
        d = run_one_chain(strength, rng_seed=42)
        results[strength] = (d.mean(0), d.std(0))
        print(f"  bias={strength:.1f}  gen{N_GENERATIONS} W={d.mean(0)[-1]:.4f} "
              f"gap={d.mean(0)[-1]-r_m[-1]:+.4f}")

    # ── 图1：所有链的绝对 W 距离 ─────────────────────────────────────
    colors = plt.cm.Reds(np.linspace(0.25, 0.95, len(BIAS_STRENGTHS)))
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(gens, r_m, 'g-', lw=2.5, label='Real chain (baseline)', zorder=5)
    ax.fill_between(gens, r_m-r_s, r_m+r_s, color='g', alpha=0.15)
    for (strength, (m_, s_)), c in zip(results.items(), colors):
        ax.plot(gens, m_, color=c, lw=1.8,
                label=f'bias = {strength:.1f}·x')
        ax.fill_between(gens, m_-s_, m_+s_, color=c, alpha=0.08)
    ax.set_xlabel('Generation k')
    ax.set_ylabel('Wasserstein distance to real distribution')
    ax.set_title('Collapse vs directional bias strength\n'
                 '(bias = α·x, each generation adds another α·x to labels)')
    ax.legend(fontsize=8, ncol=2); ax.grid(True, alpha=0.3)
    save_fig(fig, 'bias_scan_absolute', SUBDIR)

    # ── 图2：性能差（相对 real baseline）────────────────────────────
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    for (strength, (m_, s_)), c in zip(results.items(), colors):
        gap = m_ - r_m
        ax2.plot(gens, gap, color=c, lw=1.8,
                 label=f'bias = {strength:.1f}·x  (final gap={gap[-1]:+.2f})')
        ax2.fill_between(gens, gap-s_, gap+s_, color=c, alpha=0.08)
    ax2.axhline(0, color='k', lw=1, ls='--', label='Real baseline (gap=0)')
    ax2.set_xlabel('Generation k')
    ax2.set_ylabel('Performance gap  (W_biased - W_real)')
    ax2.set_title('Performance gap (collapse measure) vs directional bias strength')
    ax2.legend(fontsize=8); ax2.grid(True, alpha=0.3)
    save_fig(fig2, 'bias_scan_gap', SUBDIR)

    # ── 图3：最终代性能差 vs bias strength（α 曲线）─────────────────
    final_gaps = [results[s][0][-1] - r_m[-1] for s in BIAS_STRENGTHS]
    fig3, ax3 = plt.subplots(figsize=(7, 4))
    ax3.plot(BIAS_STRENGTHS, final_gaps, 'ro-', lw=2, ms=7)
    for s, g in zip(BIAS_STRENGTHS, final_gaps):
        ax3.annotate(f'{g:.2f}', (s, g), textcoords='offset points',
                     xytext=(0, 8), ha='center', fontsize=8)
    ax3.set_xlabel('Bias strength α  (bias = α·x per generation)')
    ax3.set_ylabel(f'Performance gap at gen {N_GENERATIONS}')
    ax3.set_title('Collapse severity vs bias strength α')
    ax3.grid(True, alpha=0.3)
    save_fig(fig3, 'bias_scan_final', SUBDIR)

    # ── 保存数据 ──────────────────────────────────────────────────────
    rows = {'generation': gens, 'real_W': r_m.tolist()}
    for s in BIAS_STRENGTHS:
        rows[f'bias_{s}_W']   = results[s][0].tolist()
        rows[f'bias_{s}_gap'] = (results[s][0] - r_m).tolist()
    save_csv(rows, 'bias_scan_results', SUBDIR)

    print(f"\n[最终代性能差汇总]")
    for s, g in zip(BIAS_STRENGTHS, final_gaps):
        print(f"  bias={s:.1f}: gap={g:+.4f}")


if __name__ == '__main__':
    run()
