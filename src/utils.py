"""
IQD实验共用工具：MAUVE计算、画图、显存管理、日志
"""

import os
import json
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 无GUI环境
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results')


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def save_results(data, name, subdir=''):
    """保存实验结果为JSON"""
    out_dir = ensure_dir(os.path.join(RESULTS_DIR, subdir))
    path = os.path.join(out_dir, f'{name}.json')
    with open(path, 'w') as f:
        json.dump(data, f, indent=2, default=str)
    print(f"[保存] {path}")
    return path


def save_csv(data_dict, name, subdir=''):
    """保存实验结果为CSV"""
    import pandas as pd
    out_dir = ensure_dir(os.path.join(RESULTS_DIR, subdir))
    path = os.path.join(out_dir, f'{name}.csv')
    pd.DataFrame(data_dict).to_csv(path, index=False)
    print(f"[保存] {path}")
    return path


def save_fig(fig, name, subdir=''):
    """保存图表"""
    out_dir = ensure_dir(os.path.join(RESULTS_DIR, subdir))
    path = os.path.join(out_dir, f'{name}.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[保存] {path}")
    return path


# ============ MAUVE 相关 ============

def compute_mauve_score(p_texts, q_texts, device_id=0, max_len=256):
    """
    计算MAUVE分数
    p_texts: 参考文本列表（真实分布）
    q_texts: 生成文本列表（模型分布）
    返回: MAUVE分数 (0~1)
    """
    import mauve
    result = mauve.compute_mauve(
        p_text=p_texts,
        q_text=q_texts,
        device_id=device_id,
        max_text_length=max_len,
        verbose=False,
    )
    return result.mauve


def compute_mauve_from_features(p_features, q_features):
    """从预计算的特征向量算MAUVE（节省重复编码时间）"""
    import mauve
    result = mauve.compute_mauve(
        p_features=p_features,
        q_features=q_features,
        verbose=False,
    )
    return result.mauve


# ============ 显存管理 ============

def clear_gpu_memory():
    """释放GPU显存 — 5080只有16GB，必须勤释放"""
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def gpu_mem_usage():
    """打印当前GPU显存使用"""
    import torch
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        total = torch.cuda.get_device_properties(0).total_mem / 1024**3
        print(f"[GPU] 已分配: {allocated:.1f}GB / 预留: {reserved:.1f}GB / 总计: {total:.1f}GB")


# ============ 画图 ============

def plot_decay_curve(generations, values, ylabel, title, fit_exp=True, subdir='', name='decay'):
    """
    画衰减曲线，可选拟合指数模型 y = y0 * alpha^k
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(generations, values, 'bo-', markersize=8, label='实验值')

    if fit_exp and len(generations) >= 3:
        # 拟合 log(y) = log(y0) + k*log(alpha)
        valid = [(g, v) for g, v in zip(generations, values) if v > 0]
        if len(valid) >= 3:
            gs, vs = zip(*valid)
            log_vs = np.log(vs)
            coeffs = np.polyfit(gs, log_vs, 1)
            alpha = np.exp(coeffs[0])
            y0 = np.exp(coeffs[1])
            r2 = 1 - np.sum((log_vs - np.polyval(coeffs, gs))**2) / np.sum((log_vs - np.mean(log_vs))**2)

            g_fit = np.linspace(min(gs), max(gs), 100)
            v_fit = y0 * alpha ** g_fit
            ax.plot(g_fit, v_fit, 'r--', label=f'拟合: y={y0:.3f}·{alpha:.3f}^k (R²={r2:.4f})')
            ax.set_title(f'{title}\nα={alpha:.4f}, R²={r2:.4f}')
        else:
            ax.set_title(title)
    else:
        ax.set_title(title)

    ax.set_xlabel('代数 (k)')
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.grid(True, alpha=0.3)
    return save_fig(fig, name, subdir)


def plot_comparison_bar(groups, values, errors, ylabel, title, subdir='', name='comparison'):
    """画分组对比柱状图"""
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(groups))
    bars = ax.bar(x, values, yerr=errors, capsize=5, color=['#2196F3', '#4CAF50', '#FF9800', '#F44336'][:len(groups)])
    ax.set_xticks(x)
    ax.set_xticklabels(groups)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, axis='y', alpha=0.3)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f'{val:.4f}', ha='center', va='bottom', fontsize=9)
    return save_fig(fig, name, subdir)


def plot_double_descent(pn_ratios, collapse_values, ylabel, subdir='', name='double_descent'):
    """画double descent曲线"""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(pn_ratios, collapse_values, 'bo-', markersize=6)
    ax.axvline(x=1.0, color='r', linestyle='--', alpha=0.5, label='插值阈值 p/n=1')
    ax.set_xlabel('p/n (参数量/数据量)')
    ax.set_ylabel(ylabel)
    ax.set_title('Double Descent: 崩溃严重程度 vs p/n')
    ax.set_xscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)
    return save_fig(fig, name, subdir)


# ============ 计时 ============

class Timer:
    def __init__(self, name=''):
        self.name = name

    def __enter__(self):
        self.start = time.time()
        print(f"[开始] {self.name}...")
        return self

    def __exit__(self, *args):
        elapsed = time.time() - self.start
        if elapsed < 60:
            print(f"[完成] {self.name} — {elapsed:.1f}秒")
        else:
            print(f"[完成] {self.name} — {elapsed/60:.1f}分钟")
