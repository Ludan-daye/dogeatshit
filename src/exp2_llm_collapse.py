"""
第二层实验：LLM多代崩溃（Llama2-7B + LoRA + WikiText-103）

对齐 Dohmatob ICML24 "A Tale of Tails" 设置：
- 模型：meta-llama/Llama-2-7b-hf + LoRA (r=16, alpha=32)
- 数据：WikiText-103，128-token 块（96 prompt + 32 completion）
- 生成：top-p=0.9, temperature=0.9
- 评估：MAUVE, PPL, repetition rate

子实验：
- 2a: 10代纯synthetic迭代，MAUVE衰减曲线
- 2b: 混合比例实验（p_syn 扫描）
- 2c: IQD等价性验证（四组对比）
- 2d: 崩溃阈值IQD*

适配：A100 40GB × 2
- Llama2-7B fp16 + LoRA 训练 ~22GB
- 两卡并行跑两个实验
"""

import os
import json
import argparse
import numpy as np
import torch
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.train.train_one_gen import finetune, generate_samples
from src.eval.compute_mauve import compute_mauve_score, delta_k
from src.eval.compute_ppl import compute_ppl_on_texts
from src.eval.compute_diversity import compute_repetition_rate
from src.utils import clear_gpu_memory, gpu_mem_usage, Timer, save_results, save_csv, save_fig, plot_decay_curve

# ============ 配置 ============

SEED = 42
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# 模型配置（对齐 Dohmatob ICML24）
BASE_MODEL = 'meta-llama/Llama-2-7b-hf'

# 训练配置
TRAIN_EPOCHS = 1
BATCH_SIZE = 2
GRAD_ACCUM = 16
MAX_LENGTH = 128
LEARNING_RATE = 2e-5
LORA_R = 16
LORA_ALPHA = 32

# 生成配置
N_GENERATE = 2000
TEMPERATURE = 0.9
TOP_P = 0.9

# 评估配置
MAUVE_SAMPLE_SIZE = 2000
PPL_SAMPLE_SIZE = 500
REP_SAMPLE_SIZE = 1000

# 实验配置
N_GENERATIONS = 10
RESULTS_BASE = str(PROJECT_ROOT / 'results' / 'exp2')
DATA_DIR = PROJECT_ROOT / 'data'

SUBDIR = 'exp2'


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_texts():
    """从 data/ 目录读取预处理好的文本"""
    with open(DATA_DIR / 'real_texts.json') as f:
        real_texts = json.load(f)
    with open(DATA_DIR / 'train_texts.json') as f:
        train_texts = json.load(f)
    print(f"  加载 D_real: {len(real_texts)} 条, D_train: {len(train_texts)} 条")
    return real_texts, train_texts


# ============ 实验2a：多代纯synthetic迭代 ============

def exp2a(real_texts, train_texts):
    print("\n" + "=" * 60)
    print("  实验2a: 多代纯synthetic迭代（Llama2-7B + LoRA）")
    print("=" * 60)

    set_seed(SEED)
    ref_texts = real_texts[:MAUVE_SAMPLE_SIZE]

    mauve_scores = []
    delta_scores = []
    ppl_scores = []
    rep_scores = []

    # Gen 0：用真实训练数据 fine-tune
    gen0_dir = os.path.join(RESULTS_BASE, 'models', 'exp2a_gen_0')
    gen0_samples = os.path.join(RESULTS_BASE, 'samples', 'exp2a_gen_0.json')

    if not os.path.exists(os.path.join(gen0_dir, 'config.json')):
        with Timer('Gen 0 fine-tune'):
            finetune(BASE_MODEL, train_texts[:N_GENERATE], gen0_dir,
                     seed=SEED, max_length=MAX_LENGTH, use_lora=True,
                     lora_r=LORA_R, lora_alpha=LORA_ALPHA,
                     batch_size=BATCH_SIZE, grad_accum=GRAD_ACCUM,
                     lr=LEARNING_RATE, epochs=TRAIN_EPOCHS)

    if not os.path.exists(gen0_samples):
        with Timer('Gen 0 生成'):
            synthetic_texts = generate_samples(
                gen0_dir, N_GENERATE,
                prompt_texts=train_texts[:N_GENERATE],
                max_length=MAX_LENGTH, temperature=TEMPERATURE, top_p=TOP_P)
        os.makedirs(os.path.dirname(gen0_samples), exist_ok=True)
        with open(gen0_samples, 'w') as f:
            json.dump(synthetic_texts, f)
    else:
        with open(gen0_samples) as f:
            synthetic_texts = json.load(f)

    # Gen 0 评估
    with Timer('Gen 0 MAUVE'):
        m0 = compute_mauve_score(ref_texts, synthetic_texts[:MAUVE_SAMPLE_SIZE])
    mauve_scores.append(m0)
    delta_scores.append(delta_k(m0))

    ppl0 = compute_ppl_on_texts(gen0_dir, real_texts[:PPL_SAMPLE_SIZE])
    ppl_scores.append(ppl0)

    rep0 = compute_repetition_rate(synthetic_texts[:REP_SAMPLE_SIZE])
    rep_scores.append(rep0)
    clear_gpu_memory()

    print(f"  Gen 0: MAUVE={m0:.4f} δ={1-m0:.4f} PPL={ppl0:.1f} Rep={rep0:.4f}")

    prev_dir = gen0_dir

    # Gen 1...N
    for gen in range(1, N_GENERATIONS + 1):
        gen_dir = os.path.join(RESULTS_BASE, 'models', f'exp2a_gen_{gen}')
        gen_samples_path = os.path.join(RESULTS_BASE, 'samples', f'exp2a_gen_{gen}.json')

        if os.path.exists(os.path.join(gen_dir, 'config.json')) and os.path.exists(gen_samples_path):
            with open(gen_samples_path) as f:
                synthetic_texts = json.load(f)
            prev_dir = gen_dir
            # 仍需计算 metrics
        else:
            with Timer(f'Gen {gen} fine-tune'):
                finetune(prev_dir, synthetic_texts[:N_GENERATE], gen_dir,
                         seed=SEED, max_length=MAX_LENGTH, use_lora=True,
                         lora_r=LORA_R, lora_alpha=LORA_ALPHA,
                         batch_size=BATCH_SIZE, grad_accum=GRAD_ACCUM,
                         lr=LEARNING_RATE, epochs=TRAIN_EPOCHS)

            with Timer(f'Gen {gen} 生成'):
                synthetic_texts = generate_samples(
                    gen_dir, N_GENERATE,
                    prompt_texts=synthetic_texts[:N_GENERATE],
                    max_length=MAX_LENGTH, temperature=TEMPERATURE, top_p=TOP_P)

            os.makedirs(os.path.dirname(gen_samples_path), exist_ok=True)
            with open(gen_samples_path, 'w') as f:
                json.dump(synthetic_texts, f)

            prev_dir = gen_dir

        with Timer(f'Gen {gen} MAUVE'):
            mk = compute_mauve_score(ref_texts, synthetic_texts[:MAUVE_SAMPLE_SIZE])
        mauve_scores.append(mk)
        delta_scores.append(delta_k(mk))

        pplk = compute_ppl_on_texts(gen_dir, real_texts[:PPL_SAMPLE_SIZE])
        ppl_scores.append(pplk)

        repk = compute_repetition_rate(synthetic_texts[:REP_SAMPLE_SIZE])
        rep_scores.append(repk)
        clear_gpu_memory()

        print(f"  Gen {gen}: MAUVE={mk:.4f} δ={1-mk:.4f} PPL={pplk:.1f} Rep={repk:.4f}")

    gens = list(range(N_GENERATIONS + 1))

    # 画 MAUVE 衰减曲线
    plot_decay_curve(gens, mauve_scores,
                     ylabel='MAUVE Score',
                     title='Exp2a: MAUVE Decay over Generations (Llama2-7B)',
                     fit_exp=True, subdir=SUBDIR, name='exp2a_mauve_decay')

    # 画 PPL 增长曲线
    plot_decay_curve(gens, ppl_scores,
                     ylabel='Perplexity',
                     title='Exp2a: PPL over Generations (Llama2-7B)',
                     fit_exp=True, subdir=SUBDIR, name='exp2a_ppl_growth')

    save_csv({
        'generation': gens,
        'mauve': mauve_scores,
        'delta': delta_scores,
        'perplexity': ppl_scores,
        'rep_rate': rep_scores,
    }, 'exp2a_results', SUBDIR)

    return mauve_scores


# ============ 实验2b：混合比例实验 ============

def exp2b(real_texts, train_texts):
    print("\n" + "=" * 60)
    print("  实验2b: 混合比例实验（p_syn 扫描）")
    print("=" * 60)

    set_seed(SEED)
    ref_texts = real_texts[:MAUVE_SAMPLE_SIZE]

    # 先用 D_train 训练基础模型生成 synthetic 数据
    base_dir = os.path.join(RESULTS_BASE, 'models', 'exp2b_base')
    if not os.path.exists(os.path.join(base_dir, 'config.json')):
        with Timer('基础模型 fine-tune'):
            finetune(BASE_MODEL, train_texts[:N_GENERATE], base_dir,
                     seed=SEED, max_length=MAX_LENGTH, use_lora=True,
                     lora_r=LORA_R, lora_alpha=LORA_ALPHA)

    base_samples_path = os.path.join(RESULTS_BASE, 'samples', 'exp2b_base.json')
    if not os.path.exists(base_samples_path):
        with Timer('生成 synthetic 数据'):
            synthetic_all = generate_samples(
                base_dir, len(train_texts),
                prompt_texts=train_texts,
                max_length=MAX_LENGTH, temperature=TEMPERATURE, top_p=TOP_P)
        os.makedirs(os.path.dirname(base_samples_path), exist_ok=True)
        with open(base_samples_path, 'w') as f:
            json.dump(synthetic_all, f)
    else:
        with open(base_samples_path) as f:
            synthetic_all = json.load(f)

    ratios = [0.0, 0.25, 0.5, 0.75, 1.0]
    mauve_scores = []
    ppl_scores = []

    for ratio in ratios:
        n_total = min(len(real_texts), len(synthetic_all), N_GENERATE)
        n_syn = int(n_total * ratio)
        n_real = n_total - n_syn

        mixed = real_texts[:n_real] + synthetic_all[:n_syn]
        np.random.shuffle(mixed)

        model_dir = os.path.join(RESULTS_BASE, 'models', f'exp2b_p{int(ratio*100)}')

        if not os.path.exists(os.path.join(model_dir, 'config.json')):
            with Timer(f'p_syn={ratio:.0%} fine-tune'):
                finetune(BASE_MODEL, mixed, model_dir,
                         seed=SEED, max_length=MAX_LENGTH, use_lora=True,
                         lora_r=LORA_R, lora_alpha=LORA_ALPHA)

        gen_texts = generate_samples(
            model_dir, N_GENERATE,
            prompt_texts=mixed[:N_GENERATE],
            max_length=MAX_LENGTH, temperature=TEMPERATURE, top_p=TOP_P)

        score = compute_mauve_score(ref_texts, gen_texts[:MAUVE_SAMPLE_SIZE])
        mauve_scores.append(score)
        clear_gpu_memory()

        ppl = compute_ppl_on_texts(model_dir, real_texts[:PPL_SAMPLE_SIZE])
        ppl_scores.append(ppl)

        print(f"  p_syn={ratio:.0%}: MAUVE={score:.4f}, PPL={ppl:.1f}")

    import matplotlib.pyplot as plt
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot([r * 100 for r in ratios], mauve_scores, 'b-o', label='MAUVE')
    ax1.set_xlabel('Synthetic Data Ratio (%)')
    ax1.set_ylabel('MAUVE Score', color='b')

    ax2 = ax1.twinx()
    ax2.plot([r * 100 for r in ratios], ppl_scores, 'r-s', label='PPL')
    ax2.set_ylabel('Perplexity', color='r')

    fig.suptitle('Exp2b: Effect of Mixing Ratio (Llama2-7B)')
    fig.legend(loc='upper center', bbox_to_anchor=(0.5, 0.88), ncol=2)
    ax1.grid(True, alpha=0.3)
    save_fig(fig, 'exp2b_mixing_ratio', SUBDIR)

    save_csv({
        'synthetic_ratio': ratios,
        'mauve': mauve_scores,
        'perplexity': ppl_scores,
    }, 'exp2b_results', SUBDIR)


# ============ 实验2c：IQD等价性验证 ============

def exp2c(real_texts, train_texts):
    """
    四组对比：
    A: 真实数据 (高IQD)
    B: 高质量synthetic (低温度, 高IQD) → MAUVE ≈ A
    C: 加噪真实数据 (低IQD)
    D: 低质量synthetic (高温度, 低IQD) → MAUVE ≈ C
    """
    print("\n" + "=" * 60)
    print("  实验2c: IQD等价性验证（四组对比）")
    print("=" * 60)

    set_seed(SEED)
    ref_texts = real_texts[:MAUVE_SAMPLE_SIZE]

    # 训练基础模型
    base_dir = os.path.join(RESULTS_BASE, 'models', 'exp2c_base')
    if not os.path.exists(os.path.join(base_dir, 'config.json')):
        with Timer('基础模型 fine-tune'):
            finetune(BASE_MODEL, train_texts[:N_GENERATE], base_dir,
                     seed=SEED, max_length=MAX_LENGTH, use_lora=True,
                     lora_r=LORA_R, lora_alpha=LORA_ALPHA)

    # A组：真实数据
    group_a = real_texts

    # B组：高质量synthetic（低温度）
    with Timer('B组: 低温度生成 (t=0.7)'):
        group_b = generate_samples(
            base_dir, N_GENERATE, prompt_texts=train_texts[:N_GENERATE],
            max_length=MAX_LENGTH, temperature=0.7, top_p=TOP_P)

    # C组：加噪真实数据
    def add_noise(texts, noise_ratio=0.3):
        rng = np.random.RandomState(SEED)
        noisy = []
        for t in texts:
            words = t.split()
            n_swap = int(len(words) * noise_ratio)
            for _ in range(n_swap):
                if len(words) > 2:
                    i, j = rng.randint(0, len(words), size=2)
                    words[i], words[j] = words[j], words[i]
            noisy.append(' '.join(words))
        return noisy

    group_c = add_noise(real_texts)

    # D组：低质量synthetic（高温度）
    with Timer('D组: 高温度生成 (t=1.5)'):
        group_d = generate_samples(
            base_dir, N_GENERATE, prompt_texts=train_texts[:N_GENERATE],
            max_length=MAX_LENGTH, temperature=1.5, top_p=TOP_P)

    groups = {
        'A_real': group_a,
        'B_hq_syn': group_b,
        'C_noisy_real': group_c,
        'D_lq_syn': group_d,
    }

    mauve_scores = {}
    downstream_ppl = {}

    for name, texts in groups.items():
        with Timer(f'{name} MAUVE'):
            score = compute_mauve_score(ref_texts, texts[:MAUVE_SAMPLE_SIZE])
            mauve_scores[name] = score
            clear_gpu_memory()

        # 用每组数据训练模型，评估下游表现
        model_dir = os.path.join(RESULTS_BASE, 'models', f'exp2c_{name}')
        if not os.path.exists(os.path.join(model_dir, 'config.json')):
            with Timer(f'{name} 训练'):
                finetune(BASE_MODEL, texts[:N_GENERATE], model_dir,
                         seed=SEED, max_length=MAX_LENGTH, use_lora=True,
                         lora_r=LORA_R, lora_alpha=LORA_ALPHA)

        ppl = compute_ppl_on_texts(model_dir, real_texts[:PPL_SAMPLE_SIZE])
        downstream_ppl[name] = ppl

        print(f"  {name}: MAUVE={score:.4f}, PPL={ppl:.1f}")

    import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    labels = list(groups.keys())
    colors = ['#2196F3', '#4CAF50', '#FF9800', '#F44336']

    ax1.bar(range(4), [mauve_scores[n] for n in labels], color=colors)
    ax1.set_xticks(range(4))
    ax1.set_xticklabels(labels, rotation=15)
    ax1.set_ylabel('MAUVE Score')
    ax1.set_title('MAUVE (expect: A≈B > C≈D)')
    ax1.grid(True, axis='y', alpha=0.3)

    ax2.bar(range(4), [downstream_ppl[n] for n in labels], color=colors)
    ax2.set_xticks(range(4))
    ax2.set_xticklabels(labels, rotation=15)
    ax2.set_ylabel('Perplexity')
    ax2.set_title('Downstream PPL (expect: A≈B < C≈D)')
    ax2.grid(True, axis='y', alpha=0.3)

    fig.suptitle('Exp2c: IQD Equivalence — Quality Matters, Not Source', fontsize=13)
    fig.tight_layout()
    save_fig(fig, 'exp2c_iqd_equivalence', SUBDIR)

    save_results({
        'mauve_scores': mauve_scores,
        'downstream_ppl': downstream_ppl,
    }, 'exp2c_results', SUBDIR)


# ============ 实验2d：崩溃阈值IQD* ============

def exp2d(real_texts, train_texts):
    print("\n" + "=" * 60)
    print("  实验2d: 崩溃阈值IQD*")
    print("=" * 60)

    set_seed(SEED)
    ref_texts = real_texts[:MAUVE_SAMPLE_SIZE]

    base_dir = os.path.join(RESULTS_BASE, 'models', 'exp2d_base')
    if not os.path.exists(os.path.join(base_dir, 'config.json')):
        with Timer('基础模型 fine-tune'):
            finetune(BASE_MODEL, train_texts[:N_GENERATE], base_dir,
                     seed=SEED, max_length=MAX_LENGTH, use_lora=True,
                     lora_r=LORA_R, lora_alpha=LORA_ALPHA)

    temperatures = [0.5, 0.7, 0.9, 1.0, 1.2, 1.5]
    mauve_scores = []
    downstream_ppls = []

    for temp in temperatures:
        with Timer(f'temperature={temp}'):
            gen_texts = generate_samples(
                base_dir, N_GENERATE, prompt_texts=train_texts[:N_GENERATE],
                max_length=MAX_LENGTH, temperature=temp, top_p=TOP_P)

            score = compute_mauve_score(ref_texts, gen_texts[:MAUVE_SAMPLE_SIZE])
            mauve_scores.append(score)
            clear_gpu_memory()

            model_dir = os.path.join(RESULTS_BASE, 'models', f'exp2d_t{temp}')
            if not os.path.exists(os.path.join(model_dir, 'config.json')):
                finetune(BASE_MODEL, gen_texts, model_dir,
                         seed=SEED, max_length=MAX_LENGTH, use_lora=True,
                         lora_r=LORA_R, lora_alpha=LORA_ALPHA)

            ppl = compute_ppl_on_texts(model_dir, real_texts[:PPL_SAMPLE_SIZE])
            downstream_ppls.append(ppl)

        print(f"  temp={temp}: MAUVE={score:.4f}, PPL={ppl:.1f}")

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(mauve_scores, downstream_ppls, 'bo-', markersize=8)
    for i, temp in enumerate(temperatures):
        ax.annotate(f't={temp}', (mauve_scores[i], downstream_ppls[i]),
                    textcoords="offset points", xytext=(5, 5), fontsize=8)
    ax.set_xlabel('MAUVE Score (IQD proxy)')
    ax.set_ylabel('Downstream Perplexity')
    ax.set_title('Exp2d: MAUVE vs Downstream PPL — Finding IQD* Threshold')
    ax.grid(True, alpha=0.3)
    ax.invert_xaxis()
    save_fig(fig, 'exp2d_threshold', SUBDIR)

    save_csv({
        'temperature': temperatures,
        'mauve': mauve_scores,
        'downstream_ppl': downstream_ppls,
    }, 'exp2d_results', SUBDIR)


# ============ 主函数 ============

def main():
    parser = argparse.ArgumentParser(description='IQD实验 Layer 2: LLM多代崩溃 (Llama2-7B)')
    parser.add_argument('--exp', type=str, default='all',
                        choices=['all', '2a', '2b', '2c', '2d'],
                        help='运行哪个实验 (默认: all)')
    args = parser.parse_args()

    print("=" * 60)
    print("  第二层实验：LLM多代崩溃")
    print(f"  模型: {BASE_MODEL}")
    print(f"  设备: {DEVICE}")
    print(f"  LoRA: r={LORA_R}, alpha={LORA_ALPHA}")
    print(f"  序列长度: {MAX_LENGTH} (96 prompt + 32 completion)")
    if torch.cuda.is_available():
        gpu_mem_usage()
    print("=" * 60)

    os.makedirs(RESULTS_BASE, exist_ok=True)
    os.makedirs(os.path.join(RESULTS_BASE, 'models'), exist_ok=True)
    os.makedirs(os.path.join(RESULTS_BASE, 'samples'), exist_ok=True)

    with Timer('加载数据'):
        real_texts, train_texts = load_texts()

    if args.exp in ('all', '2a'):
        with Timer('实验2a: 多代纯synthetic迭代'):
            exp2a(real_texts, train_texts)

    if args.exp in ('all', '2b'):
        with Timer('实验2b: 混合比例'):
            exp2b(real_texts, train_texts)

    if args.exp in ('all', '2c'):
        with Timer('实验2c: IQD等价性'):
            exp2c(real_texts, train_texts)

    if args.exp in ('all', '2d'):
        with Timer('实验2d: 崩溃阈值'):
            exp2d(real_texts, train_texts)

    print(f"\n所有第二层实验完成! 结果保存在 {RESULTS_BASE}/")


if __name__ == '__main__':
    main()
