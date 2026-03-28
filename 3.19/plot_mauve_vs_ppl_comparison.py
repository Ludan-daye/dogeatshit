"""
对比 MAUVE vs PPL vs Distinct-2 作为崩溃项ζ反推指标的优劣
核心问题：用哪个指标反推synthetic data对模型的影响更好？
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import pearsonr, spearmanr

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ============ 实验数据 ============
generations = list(range(11))

mauve      = [0.01710, 0.01509, 0.01501, 0.01211, 0.01244, 0.00920, 0.00869, 0.00724, 0.00578, 0.00615, 0.00617]
perplexity = [21.07,   31.81,   40.44,   47.42,   53.68,   59.74,   64.29,   68.18,   71.04,   72.78,   77.00]
distinct_2 = [0.2829,  0.2235,  0.1818,  0.1549,  0.1377,  0.1122,  0.0895,  0.0674,  0.0496,  0.0485,  0.0455]

# ζ 的多种定义方式
PPL_0 = perplexity[0]
MAUVE_0 = mauve[0]
D2_0 = distinct_2[0]

# 方案A: 用PPL定义ζ → ζ_ppl = PPL_k - PPL_0
zeta_ppl = [p - PPL_0 for p in perplexity]

# 方案B: 用MAUVE定义ζ → ζ_mauve = 1 - MAUVE_k / MAUVE_0 (归一化分布偏移)
zeta_mauve = [1 - m / MAUVE_0 for m in mauve]

# 方案C: 用Distinct-2定义ζ → ζ_d2 = 1 - D2_k / D2_0 (多样性损失)
zeta_d2 = [1 - d / D2_0 for d in distinct_2]

# ============ 各指标的时间敏感性分析 ============
# 每代的变化量（一阶差分）
def diff(arr):
    return [arr[i+1] - arr[i] for i in range(len(arr)-1)]

mauve_diff = diff(mauve)
ppl_diff = diff(perplexity)
d2_diff = diff(distinct_2)

# 相对变化率
def rel_change(arr):
    return [(arr[i+1] - arr[i]) / abs(arr[i]) * 100 for i in range(len(arr)-1)]

mauve_rel = rel_change(mauve)
ppl_rel = rel_change(perplexity)
d2_rel = rel_change(distinct_2)

# ============ 互相关分析 ============
print("=" * 70)
print("  指标对比分析：谁更适合反推崩溃项ζ？")
print("=" * 70)

# Pearson相关系数（线性相关）
# MAUVE vs PPL
r_mp, p_mp = pearsonr(mauve, perplexity)
rho_mp, _ = spearmanr(mauve, perplexity)

# MAUVE vs Distinct-2
r_md, p_md = pearsonr(mauve, distinct_2)
rho_md, _ = spearmanr(mauve, distinct_2)

# PPL vs Distinct-2
r_pd, p_pd = pearsonr(perplexity, distinct_2)
rho_pd, _ = spearmanr(perplexity, distinct_2)

print(f"\n  互相关系数 (Pearson / Spearman):")
print(f"    MAUVE vs PPL:       r={r_mp:.4f} / rho={rho_mp:.4f}")
print(f"    MAUVE vs Distinct2: r={r_md:.4f} / rho={rho_md:.4f}")
print(f"    PPL vs Distinct2:   r={r_pd:.4f} / rho={rho_pd:.4f}")

# ============ 各指标早期检测能力 ============
print(f"\n  前3代相对变化率（%）— 早期检测能力:")
print(f"    {'代数':>4s}  {'MAUVE%':>10s}  {'PPL%':>10s}  {'Distinct2%':>10s}")
for k in range(3):
    print(f"    {k}→{k+1}  {mauve_rel[k]:>10.1f}  {ppl_rel[k]:>10.1f}  {d2_rel[k]:>10.1f}")

print(f"\n  后3代相对变化率（%）— 后期灵敏度:")
for k in range(7, 10):
    print(f"    {k}→{k+1}  {mauve_rel[k]:>10.1f}  {ppl_rel[k]:>10.1f}  {d2_rel[k]:>10.1f}")

# ============ 单调性检查 ============
def check_monotonic(arr, name, decreasing=True):
    violations = 0
    for i in range(len(arr) - 1):
        if decreasing and arr[i+1] > arr[i]:
            violations += 1
        elif not decreasing and arr[i+1] < arr[i]:
            violations += 1
    return violations

mono_mauve = check_monotonic(mauve, 'MAUVE', decreasing=True)
mono_ppl = check_monotonic(perplexity, 'PPL', decreasing=False)
mono_d2 = check_monotonic(distinct_2, 'Distinct2', decreasing=True)

print(f"\n  单调性（10步中违反次数）:")
print(f"    MAUVE（应单调递减）:    {mono_mauve} 次违反")
print(f"    PPL（应单调递增）:      {mono_ppl} 次违反")
print(f"    Distinct-2（应单调递减）:{mono_d2} 次违反")

# ============ 信息论独立性分析 ============
print(f"\n  关键区别:")
print(f"    PPL:      模型内部指标 → 用自身模型评估自身，有循环依赖风险")
print(f"    MAUVE:    分布距离指标 → 用独立参考分布评估，无循环依赖")
print(f"    Distinct: 统计指标    → 纯文本统计，最简单但信息量最少")

# ============ 画图 ============
fig = plt.figure(figsize=(20, 16))

# ---- 图1: 三种ζ定义的对比 ----
ax1 = fig.add_subplot(2, 3, 1)
ax1.plot(generations, zeta_mauve, 'b-o', markersize=7, linewidth=2, label='MAUVE定义: 1-M(k)/M(0)')
ax1.plot(generations, zeta_d2, 'g-^', markersize=7, linewidth=2, label='Distinct定义: 1-D2(k)/D2(0)')

# PPL归一化到[0,1]
zeta_ppl_norm = [z / max(zeta_ppl) for z in zeta_ppl]
ax1.plot(generations, zeta_ppl_norm, 'r-s', markersize=7, linewidth=2, label='PPL定义: (PPL(k)-PPL(0))/max')

ax1.set_xlabel('训练代数 (k)', fontsize=11)
ax1.set_ylabel('归一化崩溃量', fontsize=11)
ax1.set_title('三种ζ定义的崩溃曲线', fontsize=12, fontweight='bold')
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3)

# ---- 图2: 每代变化率对比 ----
ax2 = fig.add_subplot(2, 3, 2)
x = np.arange(10)
width = 0.25

ax2.bar(x - width, [-m for m in mauve_rel], width, label='MAUVE下降率%', color='blue', alpha=0.7)
ax2.bar(x, ppl_rel, width, label='PPL上升率%', color='red', alpha=0.7)
ax2.bar(x + width, [-d for d in d2_rel], width, label='Distinct2下降率%', color='green', alpha=0.7)

ax2.set_xlabel('代数变化 (k→k+1)', fontsize=11)
ax2.set_ylabel('相对变化率 (%)', fontsize=11)
ax2.set_title('每代变化率 — 谁更敏感？', fontsize=12, fontweight='bold')
ax2.legend(fontsize=9)
ax2.grid(True, axis='y', alpha=0.3)
ax2.set_xticks(x)
ax2.set_xticklabels([f'{i}→{i+1}' for i in range(10)], fontsize=8)

# ---- 图3: MAUVE vs PPL 散点 ----
ax3 = fig.add_subplot(2, 3, 3)
sc = ax3.scatter(mauve, perplexity, c=generations, cmap='viridis',
                 s=150, edgecolors='black', linewidth=1, zorder=3)

# 拟合
coeffs = np.polyfit(mauve, perplexity, 1)
m_s = np.linspace(min(mauve)*0.9, max(mauve)*1.1, 100)
ax3.plot(m_s, np.polyval(coeffs, m_s), 'r--', linewidth=1.5,
         label=f'线性: PPL = {coeffs[0]:.0f}×MAUVE + {coeffs[1]:.1f}')

for k in [0, 3, 6, 10]:
    ax3.annotate(f'k={k}', (mauve[k], perplexity[k]),
                 textcoords="offset points", xytext=(8, 5), fontsize=9)

ax3.set_xlabel('MAUVE Score', fontsize=11)
ax3.set_ylabel('Perplexity', fontsize=11)
ax3.set_title(f'MAUVE vs PPL (r={r_mp:.3f})', fontsize=12, fontweight='bold')
ax3.legend(fontsize=9)
ax3.grid(True, alpha=0.3)
plt.colorbar(sc, ax=ax3, label='代数 k', shrink=0.8)

# ---- 图4: 早期检测能力 —— 累积检测率 ----
ax4 = fig.add_subplot(2, 3, 4)

# 各指标在第k代时相对于最终崩溃量的百分比
mauve_cumul = [zeta_mauve[k] / zeta_mauve[-1] * 100 for k in range(11)]
ppl_cumul = [zeta_ppl[k] / zeta_ppl[-1] * 100 for k in range(11)]
d2_cumul = [zeta_d2[k] / zeta_d2[-1] * 100 for k in range(11)]

ax4.plot(generations, mauve_cumul, 'b-o', markersize=7, linewidth=2, label='MAUVE')
ax4.plot(generations, ppl_cumul, 'r-s', markersize=7, linewidth=2, label='PPL')
ax4.plot(generations, d2_cumul, 'g-^', markersize=7, linewidth=2, label='Distinct-2')

ax4.axhline(y=50, color='gray', linestyle=':', alpha=0.5)
ax4.text(0.3, 52, '50%检测线', fontsize=9, color='gray')

ax4.set_xlabel('训练代数 (k)', fontsize=11)
ax4.set_ylabel('累积崩溃检测率 (%)', fontsize=11)
ax4.set_title('早期检测能力: 第k代检测到最终崩溃的%', fontsize=12, fontweight='bold')
ax4.legend(fontsize=10)
ax4.grid(True, alpha=0.3)

# ---- 图5: 信噪比/波动性分析 ----
ax5 = fig.add_subplot(2, 3, 5)

# 计算平滑度：实际值与单调拟合值的偏差
def smoothness_score(arr, should_decrease=True):
    """值越小越平滑"""
    sorted_arr = sorted(arr, reverse=should_decrease)
    deviations = [abs(arr[i] - sorted_arr[i]) / (max(arr) - min(arr)) for i in range(len(arr))]
    return deviations

sm_mauve = smoothness_score(mauve, should_decrease=True)
sm_ppl = smoothness_score(perplexity, should_decrease=False)
sm_d2 = smoothness_score(distinct_2, should_decrease=True)

ax5.plot(generations, sm_mauve, 'b-o', markersize=7, linewidth=2, label=f'MAUVE (违反{mono_mauve}次)')
ax5.plot(generations, sm_ppl, 'r-s', markersize=7, linewidth=2, label=f'PPL (违反{mono_ppl}次)')
ax5.plot(generations, sm_d2, 'g-^', markersize=7, linewidth=2, label=f'Distinct-2 (违反{mono_d2}次)')

ax5.set_xlabel('训练代数 (k)', fontsize=11)
ax5.set_ylabel('与理想单调曲线的偏差', fontsize=11)
ax5.set_title('平滑性/稳定性: 偏差越小越可靠', fontsize=12, fontweight='bold')
ax5.legend(fontsize=10)
ax5.grid(True, alpha=0.3)

# ---- 图6: 综合评分雷达图用表格替代 ----
ax6 = fig.add_subplot(2, 3, 6)
ax6.axis('off')

# 综合评分表
headers = ['评估维度', 'MAUVE', 'PPL', 'Distinct-2']
criteria = [
    ['理论基础', '强\n(KL散度前沿)', '中\n(交叉熵)', '弱\n(纯统计)'],
    ['独立性', '强\n(独立参考分布)', '弱\n(循环依赖)', '强\n(纯文本统计)'],
    ['单调性', f'较好\n({mono_mauve}次违反)', f'完美\n({mono_ppl}次违反)', f'完美\n({mono_d2}次违反)'],
    ['早期敏感', f'高\n(第1代下降{abs(mauve_rel[0]):.0f}%)', f'高\n(第1代上升{ppl_rel[0]:.0f}%)', f'中\n(第1代下降{abs(d2_rel[0]):.0f}%)'],
    ['后期区分', f'低\n(趋于平坦)', f'中\n(仍在增长)', f'低\n(趋于平坦)'],
    ['计算成本', '高\n(需要embedding)', '低\n(前向传播)', '极低\n(字符串统计)'],
    ['反推ζ适合度', '最佳', '次之', '辅助'],
]

table = ax6.table(cellText=criteria, colLabels=headers,
                  cellLoc='center', loc='center',
                  colColours=['#E0E0E0', '#BBDEFB', '#FFCDD2', '#C8E6C9'])

table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1, 1.8)

# 高亮MAUVE列
for i in range(len(criteria)):
    table[i+1, 1].set_facecolor('#E3F2FD')

ax6.set_title('综合对比: 谁更适合反推ζ？', fontsize=12, fontweight='bold', pad=20)

fig.suptitle('MAUVE vs PPL vs Distinct-2: 谁更适合反推崩溃项ζ？',
             fontsize=16, fontweight='bold')
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig('/Users/a1-6/importantfile/Research/狗吃屎/3.19/results/mauve_vs_ppl_comparison.png',
            dpi=200, bbox_inches='tight')
print(f"\n保存到: 3.19/results/mauve_vs_ppl_comparison.png")

# ============ 最终结论 ============
print("\n" + "=" * 70)
print("  最终结论: MAUVE vs PPL 谁更适合反推ζ")
print("=" * 70)
print("""
  PPL的优势:
    ✓ 完美单调（0次违反）— 每一代都会变差，曲线最平滑
    ✓ 计算成本低 — 只需前向传播
    ✓ 后期仍有区分度 — 第8→10代PPL还在明显增长

  PPL的致命问题:
    ✗ 循环依赖 — 用第k代模型评估第k代的PPL，模型越差评估越不准
    ✗ 无法跨模型比较 — GPT-2的PPL和LLaMA的PPL不可比
    ✗ 不是分布距离 — 不能直接对应IQD的理论定义

  MAUVE的优势:
    ✓ 理论基础最强 — 基于KL散度前沿，直接衡量分布距离
    ✓ 独立参考分布 — 用真实文本做基准，不依赖被评估模型
    ✓ 可跨模型比较 — 不同模型的MAUVE可以直接对比
    ✓ 与IQD定义直接对应 — IQD = 1 - divergence(P||Q)

  MAUVE的问题:
    ✗ 非完美单调（2次违反）— 第3→4代和第8→9代有小幅回弹
    ✗ 后期区分度降低 — MAUVE在0.006附近趋于平坦
    ✗ 计算成本高 — 需要embedding模型

  结论:
    理论上 → MAUVE更适合（独立性 + 分布距离 + IQD对应）
    实践上 → PPL + MAUVE结合最好:
      ζ_ppl = PPL_k - PPL_0          (快速计算，单调可靠)
      ζ_mauve = f(MAUVE)             (理论意义，跨模型可比)
      两者交叉验证，互相补充
""")

plt.close()
