"""
第二层实验：LLM多代崩溃
- 实验2a：10代纯synthetic迭代，MAUVE衰减曲线
- 实验2b：混合比例实验
- 实验2c：IQD等价性验证（四组对比）
- 实验2d：崩溃阈值IQD*
- 实验2e：蒸馏链 vs Synthetic链

适配：RTX 5080 16GB
- 教师模型：GPT-2 Medium (355M) — fine-tune ~6GB
- 学生模型：GPT-2 Small (124M) — fine-tune ~3GB
- fp16 + gradient accumulation
- 训练和MAUVE计算分步执行，不同时占显存
"""

import os
import gc
import json
import argparse
import numpy as np
import torch
from pathlib import Path

# 延迟import，减少启动时间
def lazy_imports():
    global AutoTokenizer, AutoModelForCausalLM, TextDataset
    global DataCollatorForLanguageModeling, Trainer, TrainingArguments
    global load_dataset
    from transformers import (
        AutoTokenizer, AutoModelForCausalLM,
        DataCollatorForLanguageModeling,
        Trainer, TrainingArguments,
    )
    from datasets import load_dataset

from utils import (
    compute_mauve_score, clear_gpu_memory, gpu_mem_usage,
    save_results, save_csv, save_fig, plot_decay_curve,
    plot_comparison_bar, Timer
)

# ============ 配置 ============

SEED = 42
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# 模型配置（适配5080 16GB）
TEACHER_MODEL = 'gpt2-medium'   # 355M, fine-tune ~6GB
STUDENT_MODEL = 'gpt2'          # 124M, fine-tune ~3GB

# 训练配置
TRAIN_EPOCHS = 3
BATCH_SIZE = 4                  # 小batch，省显存
GRAD_ACCUM = 8                  # 等效batch=32
MAX_LENGTH = 256                # 序列长度
LEARNING_RATE = 5e-5

# 生成配置
N_GENERATE = 5000               # 每代生成文本数（5080上不宜太多）
GEN_MAX_LENGTH = 256

# MAUVE配置
MAUVE_SAMPLE_SIZE = 2000        # MAUVE计算用的样本数

# 实验配置
N_GENERATIONS = 10
RESULTS_BASE = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results', 'exp2')

SUBDIR = 'exp2'


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_real_texts(n_samples=10000):
    """加载WikiText-103作为真实文本"""
    from datasets import load_dataset
    dataset = load_dataset('wikitext', 'wikitext-103-raw-v1', split='train')
    texts = [t for t in dataset['text'] if len(t.strip()) > 100]
    texts = texts[:n_samples]
    print(f"  加载真实文本: {len(texts)}条")
    return texts


def finetune_model(model_name, train_texts, output_dir, epochs=TRAIN_EPOCHS):
    """
    Fine-tune GPT-2模型
    显存优化：fp16 + gradient accumulation
    """
    lazy_imports()

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name).to(DEVICE)

    gpu_mem_usage()

    # tokenize
    encodings = tokenizer(
        train_texts, truncation=True, max_length=MAX_LENGTH,
        padding='max_length', return_tensors='pt'
    )

    class TextDataset(torch.utils.data.Dataset):
        def __init__(self, encodings):
            self.encodings = encodings
        def __len__(self):
            return len(self.encodings['input_ids'])
        def __getitem__(self, idx):
            return {
                'input_ids': self.encodings['input_ids'][idx],
                'attention_mask': self.encodings['attention_mask'][idx],
                'labels': self.encodings['input_ids'][idx],
            }

    dataset = TextDataset(encodings)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        fp16=True,  # 5080支持fp16
        learning_rate=LEARNING_RATE,
        warmup_steps=100,
        logging_steps=50,
        save_strategy='no',
        seed=SEED,
        report_to='none',
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
    )

    trainer.train()

    # 保存模型
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    # 释放显存
    del trainer, model, dataset, encodings
    clear_gpu_memory()

    return output_dir


def generate_texts(model_path, n_texts=N_GENERATE, temperature=1.0, top_p=0.95):
    """
    用模型生成文本
    生成完立即释放模型显存
    """
    lazy_imports()

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_path).to(DEVICE)
    model.eval()

    texts = []
    batch_size = 8  # 生成时batch

    with torch.no_grad():
        for i in range(0, n_texts, batch_size):
            current_batch = min(batch_size, n_texts - i)
            input_ids = torch.tensor([[tokenizer.eos_token_id]] * current_batch).to(DEVICE)

            outputs = model.generate(
                input_ids,
                max_length=GEN_MAX_LENGTH,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )

            for output in outputs:
                text = tokenizer.decode(output, skip_special_tokens=True)
                if len(text.strip()) > 20:
                    texts.append(text.strip())

            if (i // batch_size) % 50 == 0:
                print(f"    已生成: {len(texts)}/{n_texts}")

    # 释放
    del model
    clear_gpu_memory()

    print(f"  生成文本: {len(texts)}条 (temperature={temperature})")
    return texts


def compute_perplexity(model_path, texts, max_samples=500):
    """计算perplexity"""
    lazy_imports()

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_path).to(DEVICE)
    model.eval()

    texts = texts[:max_samples]
    total_loss = 0
    total_tokens = 0

    with torch.no_grad():
        for text in texts:
            inputs = tokenizer(text, return_tensors='pt', truncation=True,
                               max_length=MAX_LENGTH).to(DEVICE)
            outputs = model(**inputs, labels=inputs['input_ids'])
            total_loss += outputs.loss.item() * inputs['input_ids'].shape[1]
            total_tokens += inputs['input_ids'].shape[1]

    del model
    clear_gpu_memory()

    ppl = np.exp(total_loss / total_tokens)
    return ppl


def compute_distinct_n(texts, n=2):
    """计算distinct-n（生成多样性）"""
    all_ngrams = set()
    total = 0
    for text in texts:
        words = text.split()
        ngrams = [tuple(words[i:i+n]) for i in range(len(words) - n + 1)]
        all_ngrams.update(ngrams)
        total += len(ngrams)
    return len(all_ngrams) / max(total, 1)


# ============ 实验2a：多代纯synthetic迭代 ============

def exp2a(real_texts):
    print("\n" + "="*60)
    print("  实验2a: 多代纯synthetic迭代")
    print("="*60)

    set_seed(SEED)
    ref_texts = real_texts[:MAUVE_SAMPLE_SIZE]

    mauve_scores = []
    ppl_scores = []
    distinct2_scores = []

    # 第0代：用真实数据fine-tune
    model_dir = os.path.join(RESULTS_BASE, 'models', 'gen_0')
    with Timer('第0代 fine-tune'):
        finetune_model(STUDENT_MODEL, real_texts, model_dir)

    for gen in range(N_GENERATIONS + 1):
        model_dir = os.path.join(RESULTS_BASE, 'models', f'gen_{gen}')

        if gen > 0:
            # 用上一代的synthetic data训练
            with Timer(f'第{gen}代 fine-tune'):
                finetune_model(STUDENT_MODEL, synthetic_texts, model_dir)

        # 生成文本
        with Timer(f'第{gen}代 生成文本'):
            synthetic_texts = generate_texts(model_dir)

        # 计算指标
        with Timer(f'第{gen}代 MAUVE计算'):
            score = compute_mauve_score(
                ref_texts, synthetic_texts[:MAUVE_SAMPLE_SIZE]
            )
            mauve_scores.append(score)
            clear_gpu_memory()

        ppl = compute_perplexity(model_dir, real_texts[:200])
        ppl_scores.append(ppl)

        d2 = compute_distinct_n(synthetic_texts, n=2)
        distinct2_scores.append(d2)

        print(f"  第{gen}代: MAUVE={score:.4f}, PPL={ppl:.1f}, Distinct-2={d2:.4f}")

    gens = list(range(N_GENERATIONS + 1))

    # 画MAUVE衰减曲线
    plot_decay_curve(gens, mauve_scores,
                     ylabel='MAUVE Score',
                     title='实验2a: MAUVE随代数衰减',
                     fit_exp=True, subdir=SUBDIR, name='exp2a_mauve_decay')

    # 画PPL增长曲线
    plot_decay_curve(gens, ppl_scores,
                     ylabel='Perplexity',
                     title='实验2a: Perplexity随代数变化',
                     fit_exp=True, subdir=SUBDIR, name='exp2a_ppl_growth')

    save_csv({
        'generation': gens,
        'mauve': mauve_scores,
        'perplexity': ppl_scores,
        'distinct_2': distinct2_scores,
    }, 'exp2a_results', SUBDIR)

    return mauve_scores


# ============ 实验2b：混合比例实验 ============

def exp2b(real_texts):
    print("\n" + "="*60)
    print("  实验2b: 混合比例实验")
    print("="*60)

    set_seed(SEED)
    ref_texts = real_texts[:MAUVE_SAMPLE_SIZE]

    # 先训练一个模型生成synthetic
    base_dir = os.path.join(RESULTS_BASE, 'models', 'exp2b_base')
    finetune_model(STUDENT_MODEL, real_texts, base_dir)
    synthetic_all = generate_texts(base_dir, n_texts=len(real_texts))

    ratios = [0.0, 0.25, 0.5, 0.75, 1.0]
    mauve_scores = []
    ppl_scores = []

    for ratio in ratios:
        n_total = min(len(real_texts), len(synthetic_all))
        n_synthetic = int(n_total * ratio)
        n_real = n_total - n_synthetic

        mixed_texts = real_texts[:n_real] + synthetic_all[:n_synthetic]
        np.random.shuffle(mixed_texts)

        model_dir = os.path.join(RESULTS_BASE, 'models', f'exp2b_ratio_{int(ratio*100)}')
        with Timer(f'比例{int(ratio*100)}% fine-tune'):
            finetune_model(STUDENT_MODEL, mixed_texts, model_dir)

        gen_texts = generate_texts(model_dir)

        score = compute_mauve_score(ref_texts, gen_texts[:MAUVE_SAMPLE_SIZE])
        mauve_scores.append(score)
        clear_gpu_memory()

        ppl = compute_perplexity(model_dir, real_texts[:200])
        ppl_scores.append(ppl)

        print(f"  synthetic比例={ratio:.0%}: MAUVE={score:.4f}, PPL={ppl:.1f}")

    # 画图
    import matplotlib.pyplot as plt
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot([r*100 for r in ratios], mauve_scores, 'b-o', label='MAUVE')
    ax1.set_xlabel('Synthetic数据比例 (%)')
    ax1.set_ylabel('MAUVE Score', color='b')

    ax2 = ax1.twinx()
    ax2.plot([r*100 for r in ratios], ppl_scores, 'r-s', label='PPL')
    ax2.set_ylabel('Perplexity', color='r')

    fig.suptitle('实验2b: 混合比例对模型质量的影响')
    fig.legend(loc='upper center', bbox_to_anchor=(0.5, 0.88), ncol=2)
    ax1.grid(True, alpha=0.3)
    save_fig(fig, 'exp2b_mixing_ratio', SUBDIR)

    save_csv({
        'synthetic_ratio': ratios,
        'mauve': mauve_scores,
        'perplexity': ppl_scores,
    }, 'exp2b_results', SUBDIR)


# ============ 实验2c：IQD等价性验证 ============

def exp2c(real_texts):
    """
    四组对比：
    A: 真实数据 (高IQD)
    B: 高质量synthetic (低温度, 高IQD) → MAUVE ≈ A
    C: 加噪真实数据 (低IQD)
    D: 低质量synthetic (高温度, 低IQD) → MAUVE ≈ C
    """
    print("\n" + "="*60)
    print("  实验2c: IQD等价性验证（四组对比）")
    print("="*60)

    set_seed(SEED)
    ref_texts = real_texts[:MAUVE_SAMPLE_SIZE]

    # 先训练基础模型
    base_dir = os.path.join(RESULTS_BASE, 'models', 'exp2c_base')
    finetune_model(STUDENT_MODEL, real_texts, base_dir)

    # A组：真实数据
    group_a_texts = real_texts

    # B组：高质量synthetic（低温度）
    with Timer('B组: 低温度生成'):
        group_b_texts = generate_texts(base_dir, temperature=0.7)

    # C组：加噪真实数据（随机打乱单词）
    def add_noise_to_texts(texts, noise_ratio=0.3):
        noisy = []
        rng = np.random.RandomState(SEED)
        for t in texts:
            words = t.split()
            n_swap = int(len(words) * noise_ratio)
            for _ in range(n_swap):
                if len(words) > 2:
                    i, j = rng.randint(0, len(words), size=2)
                    words[i], words[j] = words[j], words[i]
            noisy.append(' '.join(words))
        return noisy

    group_c_texts = add_noise_to_texts(real_texts)

    # D组：低质量synthetic（高温度）
    with Timer('D组: 高温度生成'):
        group_d_texts = generate_texts(base_dir, temperature=1.5)

    # 计算每组的MAUVE
    groups = {
        'A: 真实数据': group_a_texts,
        'B: 高质量synthetic': group_b_texts,
        'C: 加噪真实数据': group_c_texts,
        'D: 低质量synthetic': group_d_texts,
    }

    mauve_scores = {}
    for name, texts in groups.items():
        with Timer(f'{name} MAUVE'):
            score = compute_mauve_score(ref_texts, texts[:MAUVE_SAMPLE_SIZE])
            mauve_scores[name] = score
            clear_gpu_memory()
        print(f"  {name}: MAUVE={score:.4f}")

    # 用每组数据训练模型，评估下游表现
    downstream_ppl = {}
    for name, texts in groups.items():
        model_dir = os.path.join(RESULTS_BASE, 'models',
                                 f'exp2c_{name.split(":")[0].strip()}')
        with Timer(f'{name} 训练'):
            finetune_model(STUDENT_MODEL, texts[:len(real_texts)], model_dir)
        ppl = compute_perplexity(model_dir, real_texts[:200])
        downstream_ppl[name] = ppl
        print(f"  {name}: 下游PPL={ppl:.1f}")

    # 画图
    group_names = list(groups.keys())
    mauve_vals = [mauve_scores[n] for n in group_names]
    ppl_vals = [downstream_ppl[n] for n in group_names]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    colors = ['#2196F3', '#4CAF50', '#FF9800', '#F44336']
    ax1.bar(range(4), mauve_vals, color=colors)
    ax1.set_xticks(range(4))
    ax1.set_xticklabels(['A\n真实', 'B\n高质synthetic', 'C\n加噪真实', 'D\n低质synthetic'])
    ax1.set_ylabel('MAUVE Score')
    ax1.set_title('各组MAUVE (预期: A≈B > C≈D)')
    ax1.grid(True, axis='y', alpha=0.3)

    ax2.bar(range(4), ppl_vals, color=colors)
    ax2.set_xticks(range(4))
    ax2.set_xticklabels(['A\n真实', 'B\n高质synthetic', 'C\n加噪真实', 'D\n低质synthetic'])
    ax2.set_ylabel('Perplexity (下游任务)')
    ax2.set_title('各组下游PPL (预期: A≈B < C≈D)')
    ax2.grid(True, axis='y', alpha=0.3)

    fig.suptitle('实验2c: IQD等价性验证 — 重要的不是真假，而是信息质量', fontsize=13)
    fig.tight_layout()
    save_fig(fig, 'exp2c_iqd_equivalence', SUBDIR)

    save_results({
        'mauve_scores': mauve_scores,
        'downstream_ppl': downstream_ppl,
    }, 'exp2c_results', SUBDIR)


# ============ 实验2d：崩溃阈值IQD* ============

def exp2d(real_texts):
    print("\n" + "="*60)
    print("  实验2d: 崩溃阈值IQD*")
    print("="*60)

    set_seed(SEED)
    ref_texts = real_texts[:MAUVE_SAMPLE_SIZE]

    # 用不同温度生成不同质量的synthetic data
    base_dir = os.path.join(RESULTS_BASE, 'models', 'exp2d_base')
    if not os.path.exists(base_dir):
        finetune_model(STUDENT_MODEL, real_texts, base_dir)

    temperatures = [0.5, 0.7, 0.9, 1.0, 1.2, 1.5, 2.0]
    mauve_scores = []
    downstream_ppls = []

    for temp in temperatures:
        with Timer(f'temperature={temp}'):
            gen_texts = generate_texts(base_dir, temperature=temp)

            score = compute_mauve_score(ref_texts, gen_texts[:MAUVE_SAMPLE_SIZE])
            mauve_scores.append(score)
            clear_gpu_memory()

            # 用这些数据训练模型
            model_dir = os.path.join(RESULTS_BASE, 'models', f'exp2d_temp_{temp}')
            finetune_model(STUDENT_MODEL, gen_texts, model_dir)
            ppl = compute_perplexity(model_dir, real_texts[:200])
            downstream_ppls.append(ppl)

        print(f"  temp={temp}: MAUVE={score:.4f}, 下游PPL={ppl:.1f}")

    # 画 MAUVE vs PPL 散点图，找拐点
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(mauve_scores, downstream_ppls, 'bo-', markersize=8)
    for i, temp in enumerate(temperatures):
        ax.annotate(f't={temp}', (mauve_scores[i], downstream_ppls[i]),
                    textcoords="offset points", xytext=(5, 5), fontsize=8)
    ax.set_xlabel('MAUVE Score (IQD代理)')
    ax.set_ylabel('下游Perplexity')
    ax.set_title('实验2d: MAUVE vs 下游表现 — 寻找IQD*拐点')
    ax.grid(True, alpha=0.3)
    ax.invert_xaxis()  # MAUVE越高越好，从右到左看退化
    save_fig(fig, 'exp2d_threshold', SUBDIR)

    save_csv({
        'temperature': temperatures,
        'mauve': mauve_scores,
        'downstream_ppl': downstream_ppls,
    }, 'exp2d_results', SUBDIR)


# ============ 实验2e：蒸馏链 vs Synthetic链 ============

def exp2e(real_texts):
    print("\n" + "="*60)
    print("  实验2e: 蒸馏链 vs Synthetic链")
    print("="*60)

    set_seed(SEED)
    ref_texts = real_texts[:MAUVE_SAMPLE_SIZE]

    # === 蒸馏链：Medium → Small → DistilGPT2 ===
    distill_models = [
        ('gpt2-medium', '蒸馏链: Medium'),
        ('gpt2', '蒸馏链: Small'),
        ('distilgpt2', '蒸馏链: DistilGPT2'),
    ]
    distill_mauve = []

    for model_name, label in distill_models:
        model_dir = os.path.join(RESULTS_BASE, 'models', f'exp2e_distill_{model_name}')
        with Timer(f'{label} fine-tune'):
            finetune_model(model_name, real_texts, model_dir)
        gen_texts = generate_texts(model_dir)
        score = compute_mauve_score(ref_texts, gen_texts[:MAUVE_SAMPLE_SIZE])
        distill_mauve.append(score)
        clear_gpu_memory()
        print(f"  {label}: MAUVE={score:.4f}")

    # === Synthetic链：Small → synthetic → Small → synthetic → ... ===
    synthetic_chain_mauve = []
    n_chain = len(distill_models)  # 同样步数便于对比

    model_dir = os.path.join(RESULTS_BASE, 'models', 'exp2e_syn_0')
    finetune_model(STUDENT_MODEL, real_texts, model_dir)
    gen_texts = generate_texts(model_dir)
    score = compute_mauve_score(ref_texts, gen_texts[:MAUVE_SAMPLE_SIZE])
    synthetic_chain_mauve.append(score)
    clear_gpu_memory()
    print(f"  Synthetic链 第0步: MAUVE={score:.4f}")

    for step in range(1, n_chain):
        model_dir = os.path.join(RESULTS_BASE, 'models', f'exp2e_syn_{step}')
        with Timer(f'Synthetic链 第{step}步'):
            finetune_model(STUDENT_MODEL, gen_texts, model_dir)
        gen_texts = generate_texts(model_dir)
        score = compute_mauve_score(ref_texts, gen_texts[:MAUVE_SAMPLE_SIZE])
        synthetic_chain_mauve.append(score)
        clear_gpu_memory()
        print(f"  Synthetic链 第{step}步: MAUVE={score:.4f}")

    # 画对比图
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 5))
    steps = list(range(len(distill_models)))
    ax.plot(steps, distill_mauve, 'b-o', markersize=8, label='蒸馏链 (不同模型)')
    ax.plot(steps, synthetic_chain_mauve, 'r-s', markersize=8, label='Synthetic链 (同一模型)')
    ax.set_xticks(steps)
    ax.set_xticklabels([f'第{i}步' for i in steps])
    ax.set_ylabel('MAUVE Score')
    ax.set_title('实验2e: 蒸馏链 vs Synthetic链')
    ax.legend()
    ax.grid(True, alpha=0.3)
    save_fig(fig, 'exp2e_distill_vs_synthetic', SUBDIR)

    save_csv({
        'step': steps,
        'distill_mauve': distill_mauve,
        'synthetic_mauve': synthetic_chain_mauve,
        'distill_labels': [m[1] for m in distill_models],
    }, 'exp2e_results', SUBDIR)


# ============ 主函数 ============

def main():
    parser = argparse.ArgumentParser(description='IQD实验 - 第二层: LLM多代崩溃')
    parser.add_argument('--exp', type=str, default='all',
                        choices=['all', '2a', '2b', '2c', '2d', '2e'],
                        help='运行哪个实验 (默认: all)')
    args = parser.parse_args()

    print("="*60)
    print("  第二层实验：LLM多代崩溃")
    print(f"  教师模型: {TEACHER_MODEL}")
    print(f"  学生模型: {STUDENT_MODEL}")
    print(f"  设备: {DEVICE}")
    if torch.cuda.is_available():
        gpu_mem_usage()
    print("="*60)

    os.makedirs(RESULTS_BASE, exist_ok=True)
    os.makedirs(os.path.join(RESULTS_BASE, 'models'), exist_ok=True)

    # 加载真实文本
    with Timer('加载真实文本'):
        real_texts = load_real_texts()

    if args.exp in ('all', '2a'):
        with Timer('实验2a: 多代纯synthetic迭代'):
            exp2a(real_texts)

    if args.exp in ('all', '2b'):
        with Timer('实验2b: 混合比例'):
            exp2b(real_texts)

    if args.exp in ('all', '2c'):
        with Timer('实验2c: IQD等价性'):
            exp2c(real_texts)

    if args.exp in ('all', '2d'):
        with Timer('实验2d: 崩溃阈值'):
            exp2d(real_texts)

    if args.exp in ('all', '2e'):
        with Timer('实验2e: 蒸馏链vs Synthetic链'):
            exp2e(real_texts)

    print("\n所有第二层实验完成! 结果保存在 results/exp2/")


if __name__ == '__main__':
    main()
