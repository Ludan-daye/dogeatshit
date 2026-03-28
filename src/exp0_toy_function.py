"""
第零层实验：可控数学函数环境
- 实验0a：三组信息损失拆解
- 实验0b：多代迭代（10-20代）

真实函数：f*(x) = 2x² + 3x + 1
模型：多项式回归
硬件：CPU即可
"""

import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

from utils import (
    save_results, save_csv, save_fig, plot_decay_curve,
    plot_comparison_bar, Timer
)

# ============ 配置 ============

SEED = 42
N_SAMPLES = 1000        # 每组样本量
NOISE_STD = 0.5         # 噪声标准差
POLY_DEGREE = 5         # 模型多项式阶数（比真实的2阶高，模拟过参数化）
RIDGE_ALPHA = 0.01      # 正则化强度
N_GENERATIONS = 20      # 迭代代数
N_REPEATS = 5           # 重复次数（取均值和置信区间）
X_RANGE = (-3, 3)       # 输入范围

SUBDIR = 'exp0'


def true_function(x):
    """真实函数 f*(x) = 2x² + 3x + 1"""
    return 2 * x**2 + 3 * x + 1


def generate_real_data(n, rng):
    """从f*加噪采样"""
    x = rng.uniform(X_RANGE[0], X_RANGE[1], size=(n, 1))
    y = true_function(x).ravel() + rng.normal(0, NOISE_STD, n)
    return x, y


def train_model(x_train, y_train, degree=POLY_DEGREE, alpha=RIDGE_ALPHA):
    """训练多项式岭回归"""
    poly = PolynomialFeatures(degree)
    X = poly.fit_transform(x_train)
    model = Ridge(alpha=alpha)
    model.fit(X, y_train)
    return model, poly


def predict(model, poly, x):
    """模型预测"""
    X = poly.transform(x)
    return model.predict(X)


def evaluate_mse(model, poly, x_test, y_test_true):
    """计算相对于f*(x)的MSE"""
    y_pred = predict(model, poly, x_test)
    return mean_squared_error(y_test_true, y_pred)


# ============ 实验0a：三组信息损失拆解 ============

def exp0a():
    """
    A组：真实数据 → 训练 → E₁
    B组：完美synthetic（从f*重新采样）→ 训练 → E₂
    C组：模型生成的synthetic → 训练 → E₃

    E₂ - E₁ = 容量/数据量限制
    E₃ - E₂ = 生成过程偏差（δ的影响）
    """
    print("\n" + "="*60)
    print("  实验0a：三组信息损失拆解")
    print("="*60)

    results = {'E1': [], 'E2': [], 'E3': []}

    for repeat in range(N_REPEATS):
        rng = np.random.RandomState(SEED + repeat)

        # 测试集（固定）
        x_test = np.linspace(X_RANGE[0], X_RANGE[1], 500).reshape(-1, 1)
        y_test_true = true_function(x_test).ravel()

        # A组：真实数据
        x_a, y_a = generate_real_data(N_SAMPLES, rng)
        model_a, poly_a = train_model(x_a, y_a)
        E1 = evaluate_mse(model_a, poly_a, x_test, y_test_true)

        # B组：完美synthetic（从f*重新采样，不同随机种子）
        x_b, y_b = generate_real_data(N_SAMPLES, rng)
        model_b, poly_b = train_model(x_b, y_b)
        E2 = evaluate_mse(model_b, poly_b, x_test, y_test_true)

        # C组：模型生成的synthetic
        # 用A组模型的预测值作为标签（这是synthetic data的本质）
        x_c = rng.uniform(X_RANGE[0], X_RANGE[1], size=(N_SAMPLES, 1))
        y_c = predict(model_a, poly_a, x_c)  # synthetic标签
        model_c, poly_c = train_model(x_c, y_c)
        E3 = evaluate_mse(model_c, poly_c, x_test, y_test_true)

        results['E1'].append(E1)
        results['E2'].append(E2)
        results['E3'].append(E3)
        print(f"  repeat {repeat+1}: E₁={E1:.6f}, E₂={E2:.6f}, E₃={E3:.6f}")

    # 统计
    means = {k: np.mean(v) for k, v in results.items()}
    stds = {k: np.std(v) for k, v in results.items()}

    print(f"\n  平均值:")
    print(f"    E₁ (真实数据)     = {means['E1']:.6f} ± {stds['E1']:.6f}")
    print(f"    E₂ (完美synthetic) = {means['E2']:.6f} ± {stds['E2']:.6f}")
    print(f"    E₃ (模型synthetic) = {means['E3']:.6f} ± {stds['E3']:.6f}")
    print(f"  E₂-E₁ (容量/数据限制) = {means['E2']-means['E1']:.6f}")
    print(f"  E₃-E₂ (生成偏差δ)    = {means['E3']-means['E2']:.6f}")

    # 画图
    plot_comparison_bar(
        groups=['A: 真实数据\n(E₁)', 'B: 完美synthetic\n(E₂)', 'C: 模型synthetic\n(E₃)'],
        values=[means['E1'], means['E2'], means['E3']],
        errors=[stds['E1'], stds['E2'], stds['E3']],
        ylabel='MSE (vs f*)',
        title='实验0a: 三组信息损失拆解',
        subdir=SUBDIR, name='exp0a_decomposition'
    )

    save_results({
        'means': means, 'stds': stds,
        'E2_minus_E1': means['E2'] - means['E1'],
        'E3_minus_E2': means['E3'] - means['E2'],
    }, 'exp0a_results', SUBDIR)

    return results


# ============ 实验0b：多代迭代 ============

def exp0b():
    """
    每代用上一代模型的预测值作为下一代训练标签
    记录：MSE(vs f*), ‖ŵ - w*‖（通过预测差异近似）
    """
    print("\n" + "="*60)
    print("  实验0b：多代迭代")
    print("="*60)

    all_mse = np.zeros((N_REPEATS, N_GENERATIONS + 1))
    all_pred_diff = np.zeros((N_REPEATS, N_GENERATIONS + 1))

    x_test = np.linspace(X_RANGE[0], X_RANGE[1], 500).reshape(-1, 1)
    y_test_true = true_function(x_test).ravel()

    for repeat in range(N_REPEATS):
        rng = np.random.RandomState(SEED + repeat)

        # 第0代：真实数据训练
        x_train, y_train = generate_real_data(N_SAMPLES, rng)
        model, poly = train_model(x_train, y_train)

        mse_0 = evaluate_mse(model, poly, x_test, y_test_true)
        pred_0 = predict(model, poly, x_test)
        pred_diff_0 = np.sqrt(np.mean((pred_0 - y_test_true) ** 2))

        all_mse[repeat, 0] = mse_0
        all_pred_diff[repeat, 0] = pred_diff_0

        # 第1~N代：每代用上一代模型生成synthetic data
        for gen in range(1, N_GENERATIONS + 1):
            x_new = rng.uniform(X_RANGE[0], X_RANGE[1], size=(N_SAMPLES, 1))
            y_new = predict(model, poly, x_new)  # synthetic标签

            model, poly = train_model(x_new, y_new)

            mse_g = evaluate_mse(model, poly, x_test, y_test_true)
            pred_g = predict(model, poly, x_test)
            pred_diff_g = np.sqrt(np.mean((pred_g - y_test_true) ** 2))

            all_mse[repeat, gen] = mse_g
            all_pred_diff[repeat, gen] = pred_diff_g

        print(f"  repeat {repeat+1}: MSE第0代={mse_0:.6f}, MSE第{N_GENERATIONS}代={all_mse[repeat, -1]:.6f}")

    # 均值和标准差
    mse_mean = all_mse.mean(axis=0)
    mse_std = all_mse.std(axis=0)
    pd_mean = all_pred_diff.mean(axis=0)
    pd_std = all_pred_diff.std(axis=0)
    generations = list(range(N_GENERATIONS + 1))

    # 画MSE衰减曲线
    plot_decay_curve(
        generations, mse_mean.tolist(),
        ylabel='MSE (vs f*)', title='实验0b: MSE随代数变化',
        subdir=SUBDIR, name='exp0b_mse_decay'
    )

    # 画预测偏差曲线
    plot_decay_curve(
        generations, pd_mean.tolist(),
        ylabel='RMSE (预测 vs f*)', title='实验0b: 预测偏差随代数变化',
        fit_exp=True, subdir=SUBDIR, name='exp0b_pred_diff_decay'
    )

    # 画带误差带的MSE曲线
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(generations, mse_mean, 'b-o', markersize=5, label='MSE均值')
    ax.fill_between(generations, mse_mean - mse_std, mse_mean + mse_std, alpha=0.2)
    ax.set_xlabel('代数 (k)')
    ax.set_ylabel('MSE (vs f*)')
    ax.set_title(f'实验0b: 多代迭代MSE变化 ({N_REPEATS}次重复)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    save_fig(fig, 'exp0b_mse_with_ci', SUBDIR)

    save_csv({
        'generation': generations,
        'mse_mean': mse_mean.tolist(),
        'mse_std': mse_std.tolist(),
        'pred_diff_mean': pd_mean.tolist(),
        'pred_diff_std': pd_std.tolist(),
    }, 'exp0b_results', SUBDIR)

    return {'mse_mean': mse_mean, 'pred_diff_mean': pd_mean}


# ============ 主函数 ============

if __name__ == '__main__':
    print("="*60)
    print("  第零层实验：可控数学函数环境")
    print(f"  f*(x) = 2x² + 3x + 1")
    print(f"  模型：{POLY_DEGREE}阶多项式岭回归 (λ={RIDGE_ALPHA})")
    print(f"  样本量: {N_SAMPLES}, 噪声: N(0, {NOISE_STD}²)")
    print(f"  重复次数: {N_REPEATS}")
    print("="*60)

    with Timer('实验0a: 三组信息损失拆解'):
        exp0a()

    with Timer('实验0b: 多代迭代'):
        exp0b()

    print("\n所有第零层实验完成! 结果保存在 results/exp0/")
