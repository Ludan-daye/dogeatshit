"""
多代训练崩溃趋势 + MAUVE变化 — 综合折线图
数据来源：实验2a结果
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ============ 实验2a数据 ============
generations = list(range(11))

mauve =      [0.01710, 0.01509, 0.01501, 0.01211, 0.01244, 0.00920, 0.00869, 0.00724, 0.00578, 0.00615, 0.00617]
perplexity = [21.07,   31.81,   40.44,   47.42,   53.68,   59.74,   64.29,   68.18,   71.04,   72.78,   77.00]
distinct_2 = [0.2829,  0.2235,  0.1818,  0.1549,  0.1377,  0.1122,  0.0895,  0.0674,  0.0496,  0.0485,  0.0455]

# MAUVE指数拟合
log_mauve = np.log(mauve)
coeffs = np.polyfit(generations, log_mauve, 1)
alpha = np.exp(coeffs[0])
y0 = np.exp(coeffs[1])
mauve_fit = [y0 * alpha**k for k in np.linspace(0, 10, 100)]

# ============ 画图 ============
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('多代Synthetic Data训练：模型崩溃全景', fontsize=16, fontweight='bold')

# --- 左上：MAUVE衰减 ---
ax1 = axes[0, 0]
ax1.plot(generations, mauve, 'b-o', markersize=8, linewidth=2, label='MAUVE (实验值)', zorder=3)
ax1.plot(np.linspace(0, 10, 100), mauve_fit, 'r--', linewidth=1.5,
         label=f'拟合: {y0:.4f}×{alpha:.3f}$^k$ (R²=0.95)')
ax1.fill_between(generations, 0, mauve, alpha=0.1, color='blue')
ax1.set_xlabel('训练代数 (k)', fontsize=12)
ax1.set_ylabel('MAUVE Score', fontsize=12)
ax1.set_title('MAUVE 逐代衰减 (α=0.889)', fontsize=13)
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)
ax1.set_xlim(-0.3, 10.3)

# --- 右上：PPL增长 ---
ax2 = axes[0, 1]
ax2.plot(generations, perplexity, 'r-s', markersize=8, linewidth=2, label='Perplexity')
ax2.fill_between(generations, min(perplexity), perplexity, alpha=0.1, color='red')
ax2.set_xlabel('训练代数 (k)', fontsize=12)
ax2.set_ylabel('Perplexity', fontsize=12)
ax2.set_title('Perplexity 逐代增长 (21→77)', fontsize=13)
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)
ax2.set_xlim(-0.3, 10.3)

# --- 左下：多样性衰减 ---
ax3 = axes[1, 0]
ax3.plot(generations, distinct_2, 'g-^', markersize=8, linewidth=2, label='Distinct-2')
ax3.fill_between(generations, 0, distinct_2, alpha=0.1, color='green')
ax3.set_xlabel('训练代数 (k)', fontsize=12)
ax3.set_ylabel('Distinct-2', fontsize=12)
ax3.set_title('生成多样性 逐代丢失 (0.28→0.05)', fontsize=13)
ax3.legend(fontsize=10)
ax3.grid(True, alpha=0.3)
ax3.set_xlim(-0.3, 10.3)

# --- 右下：三指标归一化对比 ---
ax4 = axes[1, 1]

# 归一化到[0,1]
def normalize(arr):
    arr = np.array(arr)
    return (arr - arr.min()) / (arr.max() - arr.min())

mauve_norm = normalize(mauve)
ppl_norm = 1 - normalize(perplexity)   # PPL越低越好，反转
dist_norm = normalize(distinct_2)

ax4.plot(generations, mauve_norm, 'b-o', markersize=7, linewidth=2, label='MAUVE (归一化)')
ax4.plot(generations, ppl_norm, 'r-s', markersize=7, linewidth=2, label='1/PPL (归一化)')
ax4.plot(generations, dist_norm, 'g-^', markersize=7, linewidth=2, label='Distinct-2 (归一化)')

# 三指标平均 = IQD代理
iqd_proxy = (mauve_norm + ppl_norm + dist_norm) / 3
ax4.plot(generations, iqd_proxy, 'k-D', markersize=9, linewidth=2.5, label='IQD代理 (三指标均值)')
ax4.fill_between(generations, 0, iqd_proxy, alpha=0.15, color='gray')

ax4.set_xlabel('训练代数 (k)', fontsize=12)
ax4.set_ylabel('归一化指标值', fontsize=12)
ax4.set_title('三指标同步衰减 → IQD下降', fontsize=13)
ax4.legend(fontsize=9, loc='upper right')
ax4.grid(True, alpha=0.3)
ax4.set_xlim(-0.3, 10.3)
ax4.set_ylim(-0.05, 1.05)

fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig('/Users/a1-6/importantfile/Research/狗吃屎/3.19/results/collapse_trend_overview.png',
            dpi=200, bbox_inches='tight')
print("保存到: 3.19/results/collapse_trend_overview.png")
plt.close()
