"""
exp3b_mauve_bias_scan.py

把 exp3_baseline_compare 的定向偏差扫描改用 MAUVE 来度量。

做法：
  - 对每一代模型，在固定 x 网格上采样 (x, y_pred)，作为二维特征向量
  - 用 mauve.compute_mauve(p_features=real_feat, q_features=model_feat)
    计算分布距离（走 pre-computed features 接口，不需要 GPT-2）
  - δₖ = 1 - MAUVE_k   （越高越崩溃）

三条链对比（同 exp3）：
  Real 链    : 每代用真实数据训练
  Syn  链    : 每代用 synthetic（bias=0.0）
  Biased 链  : 每代 synthetic 叠加定向偏差 α·x，扫描 α
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# IMPORTANT: mauve must be imported before sklearn to avoid FAISS/BLAS conflict
import mauve as _mauve_lib

import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge

from src.utils import save_fig, save_csv

# ── 参数 ──────────────────────────────────────────────────────────────
N_SAMPLES     = 1000
NOISE_STD     = 0.5
POLY_DEGREE   = 5
RIDGE_ALPHA   = 0.01
N_GENERATIONS = 15
N_REPEATS     = 3          # MAUVE 较慢，减少重复次数
X_RANGE       = (-3, 3)
FEAT_N        = 5000       # MAUVE 特征样本数（越大越准，越慢）

BIAS_STRENGTHS = [0.0, 0.1, 0.3, 0.5, 1.0, 2.0]
SUBDIR         = 'baseline_compare'

# ── 真实函数 ───────────────────────────────────────────────────────────
def f_star(x):
    return 2 * x**2 + 3 * x + 1

def bias_fn(x, strength):
    return strength * x

# ── 工具 ──────────────────────────────────────────────────────────────
def fit(x_train, y_train):
    poly = PolynomialFeatures(POLY_DEGREE)
    X    = poly.fit_transform(x_train)
    m    = Ridge(alpha=RIDGE_ALPHA)
    m.fit(X, y_train)
    return m, poly

def predict(m, poly, x):
    return m.predict(poly.transform(x))

def sample_real(n, rng):
    x = rng.uniform(X_RANGE[0], X_RANGE[1], size=(n, 1))
    y = f_star(x).ravel() + rng.normal(0, NOISE_STD, n)
    return x, y

def sample_synthetic(m, poly, n, rng, strength=0.0):
    x = rng.uniform(X_RANGE[0], X_RANGE[1], size=(n, 1))
    y = predict(m, poly, x) + bias_fn(x, strength).ravel()
    return x, y

# ── 特征提取：在固定 x 网格上取 y 值，用残差（y - mean_y）作1D特征 ──
# 为避免所有点落在1D流形导致FAISS奇异，使用随机x采样+加微量抖动
_FEAT_RNG = np.random.RandomState(0)
X_FEAT = _FEAT_RNG.uniform(X_RANGE[0], X_RANGE[1], size=(FEAT_N, 1))
Y_REAL_FEAT = (f_star(X_FEAT).ravel()
               + _FEAT_RNG.normal(0, NOISE_STD, FEAT_N))  # 与真实数据同分布

# 真实分布特征：形状 (FEAT_N, 1)
REAL_FEATURES = Y_REAL_FEAT.reshape(-1, 1)

def make_features(m, poly):
    """返回 (FEAT_N, 1) 特征矩阵：在随机 x 网格上的预测值"""
    y_pred = predict(m, poly, X_FEAT)
    return y_pred.reshape(-1, 1)

def compute_mauve_score(m, poly):
    model_feat = make_features(m, poly)
    result = _mauve_lib.compute_mauve(
        p_features=REAL_FEATURES.astype(np.float32),
        q_features=model_feat.astype(np.float32),
        verbose=False,
        num_buckets=50,
    )
    return float(result.mauve)

# ── 单条链 ─────────────────────────────────────────────────────────────
def run_one_chain(strength, is_real=False):
    """返回 (N_REPEATS, N_GENERATIONS+1) 的 MAUVE 分数矩阵"""
    scores = np.zeros((N_REPEATS, N_GENERATIONS + 1))
    for r in range(N_REPEATS):
        rng = np.random.RandomState(42 + r)
        x0, y0 = sample_real(N_SAMPLES, rng)
        m, poly = fit(x0, y0)
        scores[r, 0] = compute_mauve_score(m, poly)
        for k in range(1, N_GENERATIONS + 1):
            if is_real:
                xk, yk = sample_real(N_SAMPLES, rng)
            else:
                xk, yk = sample_synthetic(m, poly, N_SAMPLES, rng, strength)
            m, poly = fit(xk, yk)
            scores[r, k] = compute_mauve_score(m, poly)
            print(f"    chain(bias={strength}, rep={r+1}) gen{k}: "
                  f"MAUVE={scores[r,k]:.4f}  δ={1-scores[r,k]:.4f}")
    return scores

# ── 主实验 ─────────────────────────────────────────────────────────────
def run():
    import warnings; warnings.filterwarnings('ignore')
    gens = list(range(N_GENERATIONS + 1))

    print("[*] Real chain ...")
    real_scores = run_one_chain(0.0, is_real=True)
    r_m, r_s = real_scores.mean(0), real_scores.std(0)
    r_delta   = 1 - r_m   # δ = 1 - MAUVE

    results = {}
    for strength in BIAS_STRENGTHS:
        print(f"\n[*] Bias strength = {strength} ...")
        sc = run_one_chain(strength, is_real=False)
        results[strength] = (sc.mean(0), sc.std(0))

    # δₖ = 1 - MAUVE
    def delta(mean_scores): return 1 - mean_scores

    # ── 图1：MAUVE 随代数变化（越低越崩）────────────────────────────
    colors = plt.cm.Reds(np.linspace(0.25, 0.95, len(BIAS_STRENGTHS)))
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(gens, r_m, 'g-', lw=2.5, label='Real chain (baseline)', zorder=5)
    ax.fill_between(gens, r_m-r_s, r_m+r_s, color='g', alpha=0.15)
    for (strength, (m_, s_)), c in zip(results.items(), colors):
        ax.plot(gens, m_, color=c, lw=1.8, label=f'bias = {strength:.1f}·x')
        ax.fill_between(gens, m_-s_, m_+s_, color=c, alpha=0.08)
    ax.set_xlabel('Generation k')
    ax.set_ylabel('MAUVE score  (higher = closer to real distribution)')
    ax.set_title('MAUVE decay under directional bias\n'
                 '(bias = α·x added to synthetic labels each generation)')
    ax.legend(fontsize=8, ncol=2); ax.grid(True, alpha=0.3)
    save_fig(fig, 'mauve_bias_scan', SUBDIR)

    # ── 图2：δₖ = 1 - MAUVE（越高越崩）──────────────────────────────
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    ax2.plot(gens, r_delta, 'g-', lw=2.5, label='Real chain δ (baseline)', zorder=5)
    ax2.fill_between(gens, r_delta-r_s, r_delta+r_s, color='g', alpha=0.15)
    for (strength, (m_, s_)), c in zip(results.items(), colors):
        d_ = delta(m_)
        ax2.plot(gens, d_, color=c, lw=1.8,
                 label=f'bias={strength:.1f}  δ_final={d_[-1]:.3f}')
        ax2.fill_between(gens, d_-s_, d_+s_, color=c, alpha=0.08)
    ax2.set_xlabel('Generation k')
    ax2.set_ylabel('δₖ = 1 − MAUVE  (collapse measure)')
    ax2.set_title('δₖ under directional bias  (IQD collapse metric)')
    ax2.legend(fontsize=8, ncol=2); ax2.grid(True, alpha=0.3)
    save_fig(fig2, 'delta_bias_scan', SUBDIR)

    # ── 图3：最终代 δ vs bias strength ────────────────────────────────
    final_deltas = [delta(results[s][0])[-1] for s in BIAS_STRENGTHS]
    fig3, ax3 = plt.subplots(figsize=(7, 4))
    ax3.plot(BIAS_STRENGTHS, final_deltas, 'ro-', lw=2, ms=7)
    for s, d in zip(BIAS_STRENGTHS, final_deltas):
        ax3.annotate(f'{d:.3f}', (s, d), textcoords='offset points',
                     xytext=(0, 8), ha='center', fontsize=8)
    ax3.axhline(delta(r_m)[-1], color='g', ls='--',
                label=f'Real baseline δ = {delta(r_m)[-1]:.3f}')
    ax3.set_xlabel('Bias strength α')
    ax3.set_ylabel(f'δ at gen {N_GENERATIONS}  (= 1 − MAUVE)')
    ax3.set_title('Collapse severity (δ) vs bias strength α')
    ax3.legend(); ax3.grid(True, alpha=0.3)
    save_fig(fig3, 'delta_vs_alpha', SUBDIR)

    # ── 保存数据 ──────────────────────────────────────────────────────
    rows = {'generation': gens,
            'real_mauve': r_m.tolist(),
            'real_delta': r_delta.tolist()}
    for s in BIAS_STRENGTHS:
        m_ = results[s][0]
        rows[f'bias_{s}_mauve'] = m_.tolist()
        rows[f'bias_{s}_delta'] = delta(m_).tolist()
    save_csv(rows, 'mauve_bias_scan_results', SUBDIR)

    print(f"\n[δ 最终代汇总]  (real baseline δ = {delta(r_m)[-1]:.4f})")
    for s, d in zip(BIAS_STRENGTHS, final_deltas):
        print(f"  bias={s:.1f}:  δ={d:.4f}")


if __name__ == '__main__':
    run()
