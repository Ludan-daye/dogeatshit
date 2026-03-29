"""
数据准备：WikiText-103 → 128-token 块，划分 D_real / D_train

对齐 Dohmatob ICML24 "A Tale of Tails" 的数据设置：
- 使用 WikiText-103 数据集
- 用 Llama2 tokenizer 精确分词
- 切成 128-token 块（前 96 token prompt + 后 32 token completion）

输出（项目根目录 data/）：
  data/real_texts.json   — D_real 文本列表（评估参考，不参与训练）
  data/train_texts.json  — D_train 文本列表（Gen 0 初始训练集）
"""

import json
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"

DEFAULT_MODEL   = "meta-llama/Llama-2-7b-hf"
DEFAULT_SEQ_LEN = 128
DEFAULT_N_REAL  = 5000
DEFAULT_N_TRAIN = 5000


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL,
                        help="Tokenizer 对应的模型名")
    parser.add_argument("--seq-len", type=int, default=DEFAULT_SEQ_LEN,
                        help="每条文本的 token 数")
    parser.add_argument("--n-real", type=int, default=DEFAULT_N_REAL,
                        help="D_real 条数")
    parser.add_argument("--n-train", type=int, default=DEFAULT_N_TRAIN,
                        help="D_train 条数")
    args = parser.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # ── 加载 tokenizer ──
    print(f"[*] 加载 tokenizer: {args.model}")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    # ── 加载 WikiText-103 ──
    print("[*] 加载 WikiText-103 ...")
    from datasets import load_dataset
    dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")

    # 过滤空行，拼接全部文本
    all_text = "\n\n".join(t for t in dataset["text"] if t.strip())
    print(f"[*] 原始文本长度: {len(all_text):,} 字符")

    # ── Tokenize → 切块 ──
    print(f"[*] Tokenize 并切成 {args.seq_len}-token 块 ...")
    token_ids = tokenizer.encode(all_text)
    print(f"[*] 总 token 数: {len(token_ids):,}")

    n_chunks = len(token_ids) // args.seq_len
    chunks = [token_ids[i * args.seq_len : (i + 1) * args.seq_len]
              for i in range(n_chunks)]
    print(f"[*] 可用块数: {n_chunks:,}")

    n_needed = args.n_real + args.n_train
    if n_chunks < n_needed:
        print(f"[警告] 可用块数 ({n_chunks}) < 需求 ({n_needed})，将按比例缩减")
        args.n_real = int(n_chunks * args.n_real / n_needed)
        args.n_train = n_chunks - args.n_real

    # Decode 回文本
    print("[*] Decode 块为文本 ...")
    texts = [tokenizer.decode(chunk, skip_special_tokens=True) for chunk in chunks]

    # ── 分割 ──
    real_texts = texts[:args.n_real]
    train_texts = texts[args.n_real : args.n_real + args.n_train]

    print(f"[*] D_real  : {len(real_texts):,} 条 ({args.seq_len} tokens/条)")
    print(f"[*] D_train : {len(train_texts):,} 条 ({args.seq_len} tokens/条)")

    # ── 保存 ──
    for name, data in [("real_texts", real_texts), ("train_texts", train_texts)]:
        path = DATA_DIR / f"{name}.json"
        with open(path, "w") as f:
            json.dump(data, f, ensure_ascii=False)
        print(f"[保存] {path}")


if __name__ == "__main__":
    main()
