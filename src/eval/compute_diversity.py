"""
多样性指标：重复率、Self-BLEU
"""

from collections import Counter
import numpy as np


def compute_repetition_rate(texts: list, n: int = 4) -> float:
    """n-gram 重复率：重复出现的 n-gram 占总 n-gram 的比例"""
    total = 0
    repeated = 0
    for text in texts:
        words  = text.lower().split()
        ngrams = [tuple(words[i:i+n]) for i in range(len(words) - n + 1)]
        if not ngrams:
            continue
        counts  = Counter(ngrams)
        total   += len(ngrams)
        repeated += sum(c - 1 for c in counts.values() if c > 1)
    return repeated / max(total, 1)


def compute_self_bleu(texts: list, n_refs: int = 100, n_hyps: int = 100) -> float:
    """
    Self-BLEU：用部分文本互打 BLEU 分
    分数越低 → 生成越多样
    """
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

    smoother = SmoothingFunction().method4
    refs = texts[:n_refs]
    hyps = texts[n_refs:n_refs + n_hyps] or texts[:n_hyps]

    scores = []
    for hyp in hyps:
        hyp_tok  = hyp.lower().split()
        ref_toks = [r.lower().split() for r in refs if r != hyp]
        if not ref_toks:
            continue
        scores.append(sentence_bleu(ref_toks, hyp_tok, smoothing_function=smoother))

    return float(np.mean(scores)) if scores else 0.0
