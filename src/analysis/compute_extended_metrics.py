"""
compute_extended_metrics.py — 从保存的 samples 离线算扩展生成质量指标

补充 PPL / MAUVE / rep_rate（这些已在 metrics_with_mauve.jsonl）。
本脚本扫描 samples/gen_*.json，对每代算:

  - distinct_1, distinct_2, distinct_3  unique n-gram / total n-gram，标准多样性
  - ttr                                 type-token ratio (unique words / total words)
  - mean_length                         平均生成长度（词数）
  - std_length                          长度方差
  - token_entropy                       生成文本的 token Shannon 熵 (bits)
  - kl_to_real                          相对参考真实文本的 token 频率 KL
  - dup_rate_pairwise                   随机抽 100 对生成样本算去重比例（粗略 self-overlap）

输出 chain_dir/metrics_full.jsonl，合并 metrics_with_mauve.jsonl 已有字段。

用法：
  python src/analysis/compute_extended_metrics.py \\
      --chain-dir results/osmix_d1_v2_planc/osmix_d1_p50_s42 \\
      --ref-file results/osmix_d1_v2_planc/run_artifacts/real_texts_sample.json
  （ref-file 可选；不传就跳过 kl_to_real）
"""
import json
import sys
import math
import argparse
import random
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def tokenize(text):
    """轻量空格 + 简单清理 tokenizer（不依赖 transformers）"""
    return text.lower().split()


def ngrams(tokens, n):
    return [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]


def distinct_n(samples, n):
    """unique n-grams / total n-grams across all samples"""
    total = 0
    seen = set()
    for s in samples:
        toks = tokenize(s)
        grams = ngrams(toks, n)
        seen.update(grams)
        total += len(grams)
    return len(seen) / total if total else 0.0


def ttr(samples):
    """type-token ratio"""
    all_toks = []
    for s in samples:
        all_toks.extend(tokenize(s))
    return len(set(all_toks)) / len(all_toks) if all_toks else 0.0


def length_stats(samples):
    lens = [len(tokenize(s)) for s in samples]
    if not lens:
        return 0.0, 0.0
    mean = sum(lens) / len(lens)
    var = sum((x - mean) ** 2 for x in lens) / len(lens)
    return mean, math.sqrt(var)


def token_entropy(samples):
    """Shannon entropy of token frequency distribution (bits)"""
    cnt = Counter()
    for s in samples:
        cnt.update(tokenize(s))
    total = sum(cnt.values())
    if total == 0:
        return 0.0
    H = 0.0
    for c in cnt.values():
        p = c / total
        H -= p * math.log2(p)
    return H


def kl_to_real(samples, ref_samples):
    """KL(P_gen || P_real) on token freq, with smoothing"""
    def freq(corpus):
        cnt = Counter()
        for s in corpus:
            cnt.update(tokenize(s))
        return cnt
    cnt_q = freq(samples)
    cnt_p = freq(ref_samples)
    total_q = sum(cnt_q.values())
    total_p = sum(cnt_p.values())
    if total_q == 0 or total_p == 0:
        return -1.0
    vocab = set(cnt_q.keys()) | set(cnt_p.keys())
    # add-one smoothing
    eps = 1.0
    kl = 0.0
    for t in vocab:
        q = (cnt_q[t] + eps) / (total_q + eps * len(vocab))
        p = (cnt_p[t] + eps) / (total_p + eps * len(vocab))
        kl += q * math.log2(q / p)
    return kl


def pairwise_dup_rate(samples, n_pairs=200, seed=42):
    """随机抽 n_pairs 对样本，算 unigram 集合 Jaccard 相似度，>=0.7 视为'近重复'"""
    rng = random.Random(seed)
    if len(samples) < 2:
        return 0.0
    near_dup = 0
    for _ in range(n_pairs):
        a, b = rng.sample(samples, 2)
        sa, sb = set(tokenize(a)), set(tokenize(b))
        if not sa or not sb:
            continue
        jac = len(sa & sb) / len(sa | sb)
        if jac >= 0.7:
            near_dup += 1
    return near_dup / n_pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chain-dir", required=True)
    ap.add_argument("--ref-file", default=None,
                    help="(可选) 真实参考样本 JSON list, 算 kl_to_real")
    ap.add_argument("--n-samp", type=int, default=2000,
                    help="每代用多少样本计算（与 MAUVE 一致默认 2000）")
    args = ap.parse_args()

    chain_dir = Path(args.chain_dir).resolve()
    samples_dir = chain_dir / "samples"

    # 加载已有 metrics (mauve 等)
    existing = {}
    mp = chain_dir / "metrics_with_mauve.jsonl"
    if mp.exists():
        for line in open(mp):
            if line.strip():
                d = json.loads(line)
                existing[d["gen"]] = d

    ref_samples = None
    if args.ref_file and Path(args.ref_file).exists():
        ref_samples = json.load(open(args.ref_file))[:args.n_samp]
        print(f"reference loaded: {len(ref_samples)} samples")

    samp_files = sorted(samples_dir.glob("gen_*.json"),
                        key=lambda p: int(p.stem.replace("gen_", "")))

    out_path = chain_dir / "metrics_full.jsonl"
    with open(out_path, "w") as outf:
        for samp_file in samp_files:
            gen = int(samp_file.stem.replace("gen_", ""))
            samples = json.load(open(samp_file))[:args.n_samp]
            print(f"[gen {gen}] computing on {len(samples)} samples...")

            row = dict(existing.get(gen, {"gen": gen}))
            row["distinct_1"] = round(distinct_n(samples, 1), 6)
            row["distinct_2"] = round(distinct_n(samples, 2), 6)
            row["distinct_3"] = round(distinct_n(samples, 3), 6)
            row["ttr"] = round(ttr(samples), 6)
            ml, sl = length_stats(samples)
            row["mean_length_words"] = round(ml, 3)
            row["std_length_words"] = round(sl, 3)
            row["token_entropy_bits"] = round(token_entropy(samples), 4)
            row["pairwise_near_dup_rate"] = round(pairwise_dup_rate(samples), 4)
            if ref_samples is not None:
                row["kl_to_real"] = round(kl_to_real(samples, ref_samples), 4)
            outf.write(json.dumps(row) + "\n")
            print(f"  d1={row['distinct_1']:.3f} d2={row['distinct_2']:.3f} d3={row['distinct_3']:.3f} "
                  f"ttr={row['ttr']:.3f} H={row['token_entropy_bits']:.2f} "
                  f"len={row['mean_length_words']:.0f}±{row['std_length_words']:.0f} "
                  f"nd={row['pairwise_near_dup_rate']:.3f}"
                  + (f" kl={row['kl_to_real']:.3f}" if 'kl_to_real' in row else ''))

    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
