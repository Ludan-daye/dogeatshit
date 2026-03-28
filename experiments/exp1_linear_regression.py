"""
第一层实验：线性回归 + 高斯分布
- 实验1a：正则化强度λ对δ的影响
- 实验1b：样本量n对δ的影响
- 实验1c：20代迭代，验证δ累积
- 实验1d：MAUVE与‖δ‖的关系
- 实验1e：Double descent验证

设定：x ~ N(0, Σ), y = x⊤w* + ε
模型：岭回归 ŵ = (X⊤X + λI)⁻¹X⊤y
硬件：CPU即可
"""

import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

from utils import (
    save_results, save_csv, save_fig, plot_decay_curve,
    plot_double_descent, Timer
)

# ============ 配置 ============

SEED = 42
D = 50                  # 特征维度
SIGMA_NOISE = 0.5       # 噪声标准差
N_GENERATIONS = 20      # 迭代代数
N_REPEATS = 5           # 重复次数

SUBDIR = 'exp1'


def generate_problem(d, rng):
    """生成问题设定：Σ和w*"""
    # 协方差矩阵：特征值递减
    eigenvalues = np.array([1.0 / (i + 1) for i in range(d)])
    Q = linalg.qr(rng.randn(d, d))[0]
    Sigma = Q @ np.diag(eigenvalues) @ Q.T
    w_star = rng.randn(d)
    w_star = w_star / np.linalg.norm(w_star)
    return Sigma, w_star


def generate_data(n, Sigma, w_star, rng):
    """生成数据 x ~ N(0, Σ), y = x⊤w* + ε"""
    L = np.linalg.cholesky(Sigma)
    X = rng.randn(n, len(w_star)) @ L.T
    y = X @ w_star + rng.normal(0, SIGMA_NOISE, n)
    return X, y


def ridge_regression(X, y, lam):
    """岭回归：ŵ = (X⊤X + λI)⁻¹X⊤y"""
    d = X.shape[1]
    XtX = X.T @ X
    w_hat = np.linalg.solve(XtX + lam * np.eye(d), X.T @ y)
    return w_hat


def compute_delta_decomposition(X, y, w_star, lam):
    """
    分解δ = ŵ - w* 为正则化偏置 + 有限样本噪声
    δ = -λ(X⊤X+λI)⁻¹w* + (X⊤X+λI)⁻¹X⊤ε
    """
    d = X.shape[1]
    XtX = X.T @ X
    inv_matrix = np.linalg.inv(XtX + lam * np.eye(d))

    # 正则化偏置
    reg_bias = -lam * inv_matrix @ w_star

    # 有限样本噪声（ε = y - Xw*）
    epsilon = y - X @ w_star
    sample_noise = inv_matrix @ X.T @ epsilon

    return reg_bias, sample_noise


# ============ 实验1a：λ对δ的影响 ============

def exp1a():
    print("\n" + "="*60)
    print("  实验1a：正则化强度λ对δ的影响")
    print("="*60)

    lambdas = [0.001, 0.01, 0.1, 1.0, 10.0]
    n = 1000

    results = {lam: {'delta_norm': [], 'reg_bias_norm': [], 'noise_norm': []}
               for lam in lambdas}

    for repeat in range(N_REPEATS):
        rng = np.random.RandomState(SEED + repeat)
        Sigma, w_star = generate_problem(D, rng)

        for lam in lambdas:
            X, y = generate_data(n, Sigma, w_star, rng)
            w_hat = ridge_regression(X, y, lam)
            delta = w_hat - w_star

            reg_bias, sample_noise = compute_delta_decomposition(X, y, w_star, lam)

            results[lam]['delta_norm'].append(np.linalg.norm(delta))
            results[lam]['reg_bias_norm'].append(np.linalg.norm(reg_bias))
            results[lam]['noise_norm'].append(np.linalg.norm(sample_noise))

    # 画图
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for idx, (key, ylabel) in enumerate([
        ('delta_norm', '‖δ‖'),
        ('reg_bias_norm', '‖正则化偏置‖'),
        ('noise_norm', '‖样本噪声‖')
    ]):
        means = [np.mean(results[lam][key]) for lam in lambdas]
        stds = [np.std(results[lam][key]) for lam in lambdas]
        axes[idx].errorbar(lambdas, means, yerr=stds, marker='o', capsize=5)
        axes[idx].set_xscale('log')
        axes[idx].set_xlabel('λ (正则化强度)')
        axes[idx].set_ylabel(ylabel)
        axes[idx].set_title(f'{ylabel} vs λ (n={n})')
        axes[idx].grid(True, alpha=0.3)

    fig.suptitle('实验1a: λ对δ各分量的影响', fontsize=14)
    fig.tight_layout()
    save_fig(fig, 'exp1a_lambda_effect', SUBDIR)

    summary = {str(lam): {
        'delta_norm': float(np.mean(results[lam]['delta_norm'])),
        'reg_bias_norm': float(np.mean(results[lam]['reg_bias_norm'])),
        'noise_norm': float(np.mean(results[lam]['noise_norm'])),
    } for lam in lambdas}
    save_results(summary, 'exp1a_results', SUBDIR)

    for lam in lambdas:
        print(f"  λ={lam:>6.3f}: ‖δ‖={np.mean(results[lam]['delta_norm']):.4f}, "
              f"‖偏置‖={np.mean(results[lam]['reg_bias_norm']):.4f}, "
              f"‖噪声‖={np.mean(results[lam]['noise_norm']):.4f}")


# ============ 实验1b：n对δ的影响 ============

def exp1b():
    print("\n" + "="*60)
    print("  实验1b：样本量n对δ的影响")
    print("="*60)

    ns = [50, 100, 500, 1000, 5000, 10000]
    lam = 0.1

    results = {n_val: {'delta_norm': [], 'reg_bias_norm': [], 'noise_norm': []}
               for n_val in ns}

    for repeat in range(N_REPEATS):
        rng = np.random.RandomState(SEED + repeat)
        Sigma, w_star = generate_problem(D, rng)

        for n_val in ns:
            X, y = generate_data(n_val, Sigma, w_star, rng)
            w_hat = ridge_regression(X, y, lam)
            delta = w_hat - w_star

            reg_bias, sample_noise = compute_delta_decomposition(X, y, w_star, lam)

            results[n_val]['delta_norm'].append(np.linalg.norm(delta))
            results[n_val]['reg_bias_norm'].append(np.linalg.norm(reg_bias))
            results[n_val]['noise_norm'].append(np.linalg.norm(sample_noise))

    # 画图
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for idx, (key, ylabel) in enumerate([
        ('delta_norm', '‖δ‖'),
        ('reg_bias_norm', '‖正则化偏置‖'),
        ('noise_norm', '‖样本噪声‖')
    ]):
        means = [np.mean(results[n_val][key]) for n_val in ns]
        stds = [np.std(results[n_val][key]) for n_val in ns]
        axes[idx].errorbar(ns, means, yerr=stds, marker='o', capsize=5)
        axes[idx].set_xscale('log')
        axes[idx].set_xlabel('n (样本量)')
        axes[idx].set_ylabel(ylabel)
        axes[idx].set_title(f'{ylabel} vs n (λ={lam})')
        axes[idx].grid(True, alpha=0.3)

    fig.suptitle('实验1b: 样本量n对δ各分量的影响', fontsize=14)
    fig.tight_layout()
    save_fig(fig, 'exp1b_sample_size_effect', SUBDIR)

    for n_val in ns:
        print(f"  n={n_val:>5d}: ‖δ‖={np.mean(results[n_val]['delta_norm']):.4f}, "
              f"‖偏置‖={np.mean(results[n_val]['reg_bias_norm']):.4f}, "
              f"‖噪声‖={np.mean(results[n_val]['noise_norm']):.4f}")


# ============ 实验1c：多代迭代 ============

def exp1c():
    print("\n" + "="*60)
    print("  实验1c：20代迭代，δ累积验证")
    print("="*60)

    n = 1000
    lam = 0.1

    all_delta_norms = np.zeros((N_REPEATS, N_GENERATIONS + 1))
    all_mse = np.zeros((N_REPEATS, N_GENERATIONS + 1))

    for repeat in range(N_REPEATS):
        rng = np.random.RandomState(SEED + repeat)
        Sigma, w_star_original = generate_problem(D, rng)

        # 第0代
        X, y = generate_data(n, Sigma, w_star_original, rng)
        w_hat = ridge_regression(X, y, lam)

        all_delta_norms[repeat, 0] = np.linalg.norm(w_hat - w_star_original)
        X_test, _ = generate_data(500, Sigma, w_star_original, rng)
        y_test_true = X_test @ w_star_original
        all_mse[repeat, 0] = np.mean((X_test @ w_hat - y_test_true) ** 2)

        # 迭代：每代用上一代的w_hat作为新的"真实"权重
        w_current = w_hat
        for gen in range(1, N_GENERATIONS + 1):
            X_new, _ = generate_data(n, Sigma, w_current, rng)
            y_new = X_new @ w_current + rng.normal(0, SIGMA_NOISE, n)

            w_hat = ridge_regression(X_new, y_new, lam)

            all_delta_norms[repeat, gen] = np.linalg.norm(w_hat - w_star_original)
            all_mse[repeat, gen] = np.mean((X_test @ w_hat - y_test_true) ** 2)

            w_current = w_hat

        print(f"  repeat {repeat+1}: ‖δ₀‖={all_delta_norms[repeat, 0]:.4f} → "
              f"‖δ_{N_GENERATIONS}‖={all_delta_norms[repeat, -1]:.4f}")

    gens = list(range(N_GENERATIONS + 1))
    delta_mean = all_delta_norms.mean(axis=0)
    mse_mean = all_mse.mean(axis=0)

    # 画图
    plot_decay_curve(gens, delta_mean.tolist(),
                     ylabel='‖ŵₖ - w*‖', title='实验1c: 偏差累积',
                     fit_exp=True, subdir=SUBDIR, name='exp1c_delta_accumulation')

    plot_decay_curve(gens, mse_mean.tolist(),
                     ylabel='MSE', title='实验1c: 测试误差随代数变化',
                     fit_exp=True, subdir=SUBDIR, name='exp1c_mse_growth')

    save_csv({
        'generation': gens,
        'delta_norm_mean': delta_mean.tolist(),
        'delta_norm_std': all_delta_norms.std(axis=0).tolist(),
        'mse_mean': mse_mean.tolist(),
        'mse_std': all_mse.std(axis=0).tolist(),
    }, 'exp1c_results', SUBDIR)

    # 额外输出 per-repeat metrics.jsonl，供 fit_transfer_fn 使用
    import json as _json
    import os as _os
    results_base = _os.path.join(
        _os.path.dirname(_os.path.dirname(__file__)), 'results', SUBDIR)
    for r in range(N_REPEATS):
        rep_dir = _os.path.join(results_base, f'exp1c_rep{r}')
        _os.makedirs(rep_dir, exist_ok=True)
        with open(_os.path.join(rep_dir, 'metrics.jsonl'), 'w') as fj:
            for g in gens:
                _json.dump({"gen": g, "delta": float(all_delta_norms[r, g])}, fj)
                fj.write('\n')
        print(f"  [保存] {rep_dir}/metrics.jsonl")

    return delta_mean, mse_mean


# ============ 实验1d：MAUVE与δ的关系 ============

def exp1d():
    """
    每代生成高维输出（预测值分布），计算MAUVE
    注意：MAUVE本来用于文本，这里用在连续分布上作为分布距离的代理
    对于连续值，改用KL散度 / Wasserstein距离作为替代
    """
    print("\n" + "="*60)
    print("  实验1d：分布距离与δ的关系")
    print("  (线性回归场景下用Wasserstein距离替代MAUVE)")
    print("="*60)

    from scipy.stats import wasserstein_distance

    n = 1000
    lam = 0.1

    all_delta = np.zeros((N_REPEATS, N_GENERATIONS + 1))
    all_wdist = np.zeros((N_REPEATS, N_GENERATIONS + 1))

    for repeat in range(N_REPEATS):
        rng = np.random.RandomState(SEED + repeat)
        Sigma, w_star = generate_problem(D, rng)

        # 生成参考分布的输出
        X_ref, _ = generate_data(2000, Sigma, w_star, rng)
        y_ref = X_ref @ w_star  # 真实输出分布（无噪声）

        X_train, y_train = generate_data(n, Sigma, w_star, rng)
        w_hat = ridge_regression(X_train, y_train, lam)

        w_current = w_hat
        for gen in range(N_GENERATIONS + 1):
            if gen > 0:
                X_new, _ = generate_data(n, Sigma, w_current, rng)
                y_new = X_new @ w_current + rng.normal(0, SIGMA_NOISE, n)
                w_hat = ridge_regression(X_new, y_new, lam)
                w_current = w_hat

            y_pred = X_ref @ w_current
            all_delta[repeat, gen] = np.linalg.norm(w_current - w_star)
            all_wdist[repeat, gen] = wasserstein_distance(y_ref, y_pred)

    delta_mean = all_delta.mean(axis=0)
    wdist_mean = all_wdist.mean(axis=0)

    # 画散点图：δ vs Wasserstein距离
    fig, ax = plt.subplots(figsize=(8, 5))
    for repeat in range(N_REPEATS):
        ax.scatter(all_delta[repeat], all_wdist[repeat], alpha=0.5, s=20)
    ax.set_xlabel('‖δₖ‖ = ‖ŵₖ - w*‖')
    ax.set_ylabel('Wasserstein距离 (输出分布)')
    ax.set_title('实验1d: 偏差δ与分布距离的关系')
    ax.grid(True, alpha=0.3)

    # 拟合线性关系
    all_d_flat = all_delta.ravel()
    all_w_flat = all_wdist.ravel()
    coeffs = np.polyfit(all_d_flat, all_w_flat, 1)
    r2 = 1 - np.sum((all_w_flat - np.polyval(coeffs, all_d_flat))**2) / \
         np.sum((all_w_flat - np.mean(all_w_flat))**2)
    x_fit = np.linspace(all_d_flat.min(), all_d_flat.max(), 100)
    ax.plot(x_fit, np.polyval(coeffs, x_fit), 'r--',
            label=f'线性拟合 (R²={r2:.4f})')
    ax.legend()
    save_fig(fig, 'exp1d_delta_vs_distribution_distance', SUBDIR)

    print(f"  δ 与 Wasserstein距离的线性相关 R² = {r2:.4f}")

    save_csv({
        'generation': list(range(N_GENERATIONS + 1)),
        'delta_mean': delta_mean.tolist(),
        'wasserstein_mean': wdist_mean.tolist(),
    }, 'exp1d_results', SUBDIR)


# ============ 实验1e：Double Descent ============

def exp1e():
    print("\n" + "="*60)
    print("  实验1e：Double Descent验证")
    print("="*60)

    # p/n 从 0.1 到 10
    n_fixed = 200
    dims = [20, 50, 100, 150, 180, 195, 200, 205, 220, 250, 400, 600, 1000, 2000]
    pn_ratios = [d / n_fixed for d in dims]
    lam = 0.001  # 小正则化，更容易看到double descent

    n_gen = 5  # 每个配置跑5代看崩溃程度

    all_collapse = {d: [] for d in dims}

    for repeat in range(N_REPEATS):
        rng = np.random.RandomState(SEED + repeat)

        for d in dims:
            # 生成问题
            eigenvalues = np.array([1.0 / (i + 1) for i in range(d)])
            Sigma = np.diag(eigenvalues)
            w_star = rng.randn(d)
            w_star = w_star / np.linalg.norm(w_star)

            # 第0代
            X = rng.randn(n_fixed, d) @ np.diag(np.sqrt(eigenvalues))
            y = X @ w_star + rng.normal(0, SIGMA_NOISE, n_fixed)

            try:
                w_hat = ridge_regression(X, y, lam)
            except np.linalg.LinAlgError:
                all_collapse[d].append(float('inf'))
                continue

            mse_0 = np.mean((X @ w_hat - X @ w_star) ** 2)

            # 跑n_gen代
            w_current = w_hat
            for gen in range(n_gen):
                X_new = rng.randn(n_fixed, d) @ np.diag(np.sqrt(eigenvalues))
                y_new = X_new @ w_current + rng.normal(0, SIGMA_NOISE, n_fixed)
                try:
                    w_current = ridge_regression(X_new, y_new, lam)
                except np.linalg.LinAlgError:
                    break

            mse_final = np.mean((X @ w_current - X @ w_star) ** 2)
            collapse_severity = mse_final - mse_0  # 崩溃 = 最终MSE - 初始MSE
            all_collapse[d].append(max(0, collapse_severity))

    # 计算均值
    collapse_means = [np.mean(all_collapse[d]) for d in dims]
    collapse_stds = [np.std(all_collapse[d]) for d in dims]

    # 画图
    plot_double_descent(pn_ratios, collapse_means,
                        ylabel='崩溃严重程度 (MSE增加量)',
                        subdir=SUBDIR, name='exp1e_double_descent')

    # 带误差带的版本
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(pn_ratios, collapse_means, yerr=collapse_stds, marker='o', capsize=4)
    ax.axvline(x=1.0, color='r', linestyle='--', alpha=0.5, label='插值阈值 p/n=1')
    ax.set_xlabel('p/n (参数量/数据量)')
    ax.set_ylabel('崩溃严重程度 (MSE增加量)')
    ax.set_title(f'Double Descent: 崩溃严重程度 vs p/n (n={n_fixed}, {n_gen}代)')
    ax.set_xscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)
    save_fig(fig, 'exp1e_double_descent_with_ci', SUBDIR)

    for d, pn, cm in zip(dims, pn_ratios, collapse_means):
        print(f"  d={d:>4d}, p/n={pn:.2f}: 崩溃量={cm:.6f}")

    save_csv({
        'dim': dims,
        'pn_ratio': pn_ratios,
        'collapse_mean': collapse_means,
        'collapse_std': collapse_stds,
    }, 'exp1e_results', SUBDIR)


# ============ 主函数 ============

if __name__ == '__main__':
    print("="*60)
    print("  第一层实验：线性回归 + 高斯分布")
    print(f"  维度d={D}, 噪声σ={SIGMA_NOISE}")
    print(f"  重复次数: {N_REPEATS}")
    print("="*60)

    with Timer('实验1a: λ对δ的影响'):
        exp1a()

    with Timer('实验1b: n对δ的影响'):
        exp1b()

    with Timer('实验1c: 多代迭代'):
        exp1c()

    with Timer('实验1d: 分布距离与δ的关系'):
        exp1d()

    with Timer('实验1e: Double Descent'):
        exp1e()

    print("\n所有第一层实验完成! 结果保存在 results/exp1/")
