"""
用MAUVE反向计算崩溃项ζ
核心思路：
  E_test(real) = B + V
  E_test(gen_k) = B + V + ζₖ
  ζₖ = E_test(gen_k) - E_test(real) = PPL_k - PPL_0
  然后找 ζ = f(MAUVE) 的映射关系
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ============ 实验2a原始数据 ============
generations = list(range(11))

mauve      = [0.01710, 0.01509, 0.01501, 0.01211, 0.01244, 0.00920, 0.00869, 0.00724, 0.00578, 0.00615, 0.00617]
perplexity = [21.07,   31.81,   40.44,   47.42,   53.68,   59.74,   64.29,   68.18,   71.04,   72.78,   77.00]
distinct_2 = [0.2829,  0.2235,  0.1818,  0.1549,  0.1377,  0.1122,  0.0895,  0.0674,  0.0496,  0.0485,  0.0455]

# ============ 计算崩溃项 ζ ============
# ζₖ = PPL_k - PPL_0  (PPL_0 是真实数据训练的基准)
PPL_baseline = perplexity[0]  # B + V
zeta = [ppl - PPL_baseline for ppl in perplexity]  # ζₖ

# MAUVE下降量: ΔMAUVE = MAUVE_0 - MAUVE_k
mauve_baseline = mauve[0]
delta_mauve = [mauve_baseline - m for m in mauve]

# Distinct-2 下降量
d2_baseline = distinct_2[0]
delta_d2 = [d2_baseline - d for d in distinct_2]

print("="*60)
print("  各代崩溃项 ζ 和 MAUVE")
print("="*60)
print(f"  {'代数':>4s}  {'MAUVE':>8s}  {'PPL':>8s}  {'ζ(PPL增量)':>12s}  {'ΔMAUVE':>8s}  {'ΔDistinct2':>10s}")
for k in generations:
    print(f"  {k:>4d}  {mauve[k]:>8.4f}  {perplexity[k]:>8.2f}  {zeta[k]:>12.2f}  {delta_mauve[k]:>8.4f}  {delta_d2[k]:>10.4f}")

# ============ 拟合 ζ = f(MAUVE) ============
# 去掉第0代 (ζ=0) ，用第1-10代拟合
mauve_fit_data = np.array(mauve[1:])
zeta_fit_data = np.array(zeta[1:])
delta_mauve_fit = np.array(delta_mauve[1:])

# 拟合1: ζ vs MAUVE (幂律关系 ζ = a * MAUVE^b)
def power_law(x, a, b):
    return a * np.power(x, b)

try:
    popt_power, _ = curve_fit(power_law, mauve_fit_data, zeta_fit_data, p0=[1, -1], maxfev=5000)
    zeta_pred_power = power_law(mauve_fit_data, *popt_power)
    ss_res = np.sum((zeta_fit_data - zeta_pred_power)**2)
    ss_tot = np.sum((zeta_fit_data - np.mean(zeta_fit_data))**2)
    r2_power = 1 - ss_res / ss_tot
    print(f"\n  拟合: ζ = {popt_power[0]:.4f} × MAUVE^({popt_power[1]:.4f}),  R² = {r2_power:.4f}")
except:
    popt_power = None
    r2_power = 0

# 拟合2: ζ vs ΔMAUVE (线性关系)
coeffs_linear = np.polyfit(delta_mauve_fit, zeta_fit_data, 1)
zeta_pred_linear = np.polyval(coeffs_linear, delta_mauve_fit)
ss_res_lin = np.sum((zeta_fit_data - zeta_pred_linear)**2)
ss_tot_lin = np.sum((zeta_fit_data - np.mean(zeta_fit_data))**2)
r2_linear = 1 - ss_res_lin / ss_tot_lin
print(f"  拟合: ζ = {coeffs_linear[0]:.2f} × ΔMAUVE + {coeffs_linear[1]:.2f},  R² = {r2_linear:.4f}")

# 拟合3: ζ vs 代数 (指数增长)
def exp_growth(k, a, b):
    return a * (np.exp(b * k) - 1)

try:
    popt_exp, _ = curve_fit(exp_growth, np.array(generations), np.array(zeta), p0=[10, 0.1])
    zeta_pred_exp = exp_growth(np.array(generations), *popt_exp)
    ss_res_exp = np.sum((np.array(zeta) - zeta_pred_exp)**2)
    ss_tot_exp = np.sum((np.array(zeta) - np.mean(zeta))**2)
    r2_exp = 1 - ss_res_exp / ss_tot_exp
    print(f"  拟合: ζ = {popt_exp[0]:.2f} × (e^({popt_exp[1]:.4f}k) - 1),  R² = {r2_exp:.4f}")
except:
    popt_exp = None
    r2_exp = 0

# ============ 画图 ============
fig = plt.figure(figsize=(18, 14))

# ---- 图1 (左上): 各项指标 vs 代数 ----
ax1 = fig.add_subplot(2, 2, 1)

ax1_mauve = ax1
ax1_ppl = ax1.twinx()

l1, = ax1_mauve.plot(generations, mauve, 'b-o', markersize=7, linewidth=2, label='MAUVE')
l2, = ax1_ppl.plot(generations, perplexity, 'r-s', markersize=7, linewidth=2, label='PPL')

ax1_mauve.set_xlabel('训练代数 (k)', fontsize=12)
ax1_mauve.set_ylabel('MAUVE Score', color='b', fontsize=12)
ax1_ppl.set_ylabel('Perplexity', color='r', fontsize=12)
ax1.set_title('MAUVE & PPL 随代数变化', fontsize=13, fontweight='bold')
ax1.legend(handles=[l1, l2], loc='center right', fontsize=10)
ax1.grid(True, alpha=0.3)

# ---- 图2 (右上): 崩溃项ζ vs 代数 ----
ax2 = fig.add_subplot(2, 2, 2)

ax2.plot(generations, zeta, 'k-D', markersize=8, linewidth=2.5, label='ζₖ = PPLₖ - PPL₀', zorder=3)
ax2.fill_between(generations, 0, zeta, alpha=0.15, color='red')

if popt_exp is not None:
    k_smooth = np.linspace(0, 10, 100)
    ax2.plot(k_smooth, exp_growth(k_smooth, *popt_exp), 'r--', linewidth=1.5,
             label=f'拟合: ζ={popt_exp[0]:.1f}(e$^{{{popt_exp[1]:.3f}k}}$-1)\nR²={r2_exp:.4f}')

# 标注关键值
for k in [0, 3, 6, 10]:
    ax2.annotate(f'ζ={zeta[k]:.1f}', (k, zeta[k]),
                 textcoords="offset points", xytext=(8, 8), fontsize=9,
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))

ax2.set_xlabel('训练代数 (k)', fontsize=12)
ax2.set_ylabel('崩溃项 ζ (PPL增量)', fontsize=12)
ax2.set_title('崩溃项 ζ 随代数增长', fontsize=13, fontweight='bold')
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)

# ---- 图3 (左下): ζ vs MAUVE (核心关系) ----
ax3 = fig.add_subplot(2, 2, 3)

ax3.scatter(mauve[1:], zeta[1:], c=generations[1:], cmap='RdYlGn_r',
            s=120, edgecolors='black', linewidth=1, zorder=3)

# 幂律拟合曲线
if popt_power is not None:
    m_smooth = np.linspace(min(mauve[1:]) * 0.9, max(mauve[1:]) * 1.1, 100)
    ax3.plot(m_smooth, power_law(m_smooth, *popt_power), 'r--', linewidth=2,
             label=f'ζ = {popt_power[0]:.2f} × MAUVE$^{{{popt_power[1]:.2f}}}$\nR² = {r2_power:.4f}')

# 标注代数
for k in range(1, 11):
    ax3.annotate(f'k={k}', (mauve[k], zeta[k]),
                 textcoords="offset points", xytext=(-15, 8), fontsize=8)

ax3.set_xlabel('MAUVE Score', fontsize=12)
ax3.set_ylabel('崩溃项 ζ', fontsize=12)
ax3.set_title('核心发现: ζ = f(MAUVE)', fontsize=13, fontweight='bold')
ax3.legend(fontsize=10)
ax3.grid(True, alpha=0.3)
ax3.invert_xaxis()  # MAUVE越低崩溃越大

# colorbar
sm = plt.cm.ScalarMappable(cmap='RdYlGn_r', norm=plt.Normalize(1, 10))
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax3, shrink=0.8)
cbar.set_label('代数 k', fontsize=10)

# ---- 图4 (右下): 全指标综合 ----
ax4 = fig.add_subplot(2, 2, 4)

# 归一化zeta到[0,1]
zeta_max = max(zeta)
zeta_norm = [z / zeta_max for z in zeta]

# MAUVE衰减率
mauve_decay = [m / mauve_baseline for m in mauve]

# Distinct-2 衰减率
d2_decay = [d / d2_baseline for d in distinct_2]

ax4.plot(generations, mauve_decay, 'b-o', markersize=7, linewidth=2, label='MAUVE / MAUVE₀')
ax4.plot(generations, d2_decay, 'g-^', markersize=7, linewidth=2, label='Distinct-2 / D2₀')
ax4.plot(generations, [1 - z for z in zeta_norm], 'r-s', markersize=7, linewidth=2, label='1 - ζ/ζ_max')

# IQD = 综合
iqd = [(mauve_decay[k] + d2_decay[k] + (1 - zeta_norm[k])) / 3 for k in range(len(generations))]
ax4.plot(generations, iqd, 'k-D', markersize=9, linewidth=2.5, label='IQD 代理')
ax4.fill_between(generations, 0, iqd, alpha=0.1, color='gray')

# 拟合IQD指数衰减
log_iqd = np.log([max(v, 1e-6) for v in iqd])
coeffs_iqd = np.polyfit(generations, log_iqd, 1)
alpha_iqd = np.exp(coeffs_iqd[0])
y0_iqd = np.exp(coeffs_iqd[1])
r2_iqd_vals = log_iqd
r2_iqd = 1 - np.sum((r2_iqd_vals - np.polyval(coeffs_iqd, generations))**2) / \
         np.sum((r2_iqd_vals - np.mean(r2_iqd_vals))**2)

k_s = np.linspace(0, 10, 100)
ax4.plot(k_s, y0_iqd * alpha_iqd**k_s, 'k--', linewidth=1.5, alpha=0.6,
         label=f'IQD拟合: {y0_iqd:.2f}×{alpha_iqd:.3f}$^k$ (R²={r2_iqd:.3f})')

ax4.axhline(y=0.3, color='purple', linestyle=':', alpha=0.5, linewidth=1.5)
ax4.text(0.5, 0.32, 'IQD* 崩溃阈值?', fontsize=10, color='purple')

ax4.set_xlabel('训练代数 (k)', fontsize=12)
ax4.set_ylabel('相对于第0代的比率', fontsize=12)
ax4.set_title('IQD衰减: 从MAUVE反推崩溃全貌', fontsize=13, fontweight='bold')
ax4.legend(fontsize=9, loc='upper right')
ax4.grid(True, alpha=0.3)
ax4.set_ylim(-0.05, 1.1)

# ============ 总标题 ============
fig.suptitle('用MAUVE反向计算崩溃项ζ: ζ = f(MAUVE)\nE_test = B + V + ζ,  其中 ζₖ = PPLₖ - PPL₀',
             fontsize=15, fontweight='bold', y=1.02)

fig.tight_layout()
fig.savefig('/Users/a1-6/importantfile/Research/狗吃屎/3.19/results/zeta_from_mauve.png',
            dpi=200, bbox_inches='tight')
print(f"\n保存到: 3.19/results/zeta_from_mauve.png")

# ============ 输出核心公式 ============
print("\n" + "="*60)
print("  核心发现")
print("="*60)
print(f"  E_test = B + V + ζ")
print(f"  基准 PPL₀ = {PPL_baseline:.2f} (B + V)")
print(f"  ζₖ = PPLₖ - PPL₀")
print(f"")
if popt_power is not None:
    print(f"  ζ 与 MAUVE 的关系:")
    print(f"    ζ = {popt_power[0]:.2f} × MAUVE^({popt_power[1]:.2f}),  R² = {r2_power:.4f}")
print(f"")
print(f"  ζ 与 ΔMAUVE 的关系:")
print(f"    ζ = {coeffs_linear[0]:.2f} × ΔMAUVE + {coeffs_linear[1]:.2f},  R² = {r2_linear:.4f}")
if popt_exp is not None:
    print(f"")
    print(f"  ζ 随代数的增长:")
    print(f"    ζ = {popt_exp[0]:.1f} × (e^({popt_exp[1]:.4f}k) - 1),  R² = {r2_exp:.4f}")
print(f"")
print(f"  IQD指数衰减:")
print(f"    IQD(k) = {y0_iqd:.2f} × {alpha_iqd:.3f}^k,  R² = {r2_iqd:.3f}")
plt.close()
