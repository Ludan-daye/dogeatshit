"""
数据准备：下载 OpenWebText，划分 D_real / D_train

D_real : 5M tokens，仅用于测量（不参与训练链）
D_train: 2M tokens，作为初始训练集 D₀

输出（项目根目录 data/）：
  data/real_texts.json   — D_real 文本列表
  data/train_texts.json  — D_train 文本列表
"""

import json
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"

DEFAULT_REAL_TOKENS  = 5_000_000
DEFAULT_TRAIN_TOKENS = 2_000_000
DEFAULT_MIN_LEN      = 200


def _approx_tokens(text: str) -> int:
    return len(text) // 4  # 1 token ≈ 4 字符（英文粗估）


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--real-tokens",  type=int, default=DEFAULT_REAL_TOKENS)
    parser.add_argument("--train-tokens", type=int, default=DEFAULT_TRAIN_TOKENS)
    parser.add_argument("--min-length",   type=int, default=DEFAULT_MIN_LEN)
    args = parser.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print("[*] 流式加载 OpenWebText ...")
    from datasets import load_dataset
    dataset = load_dataset("openwebtext", split="train", streaming=True)

    real_texts, train_texts = [], []
    real_tok,   train_tok   = 0, 0

    for item in dataset:
        text = item["text"].strip()
        if len(text) < args.min_length:
            continue
        tok = _approx_tokens(text)

        if real_tok < args.real_tokens:
            real_texts.append(text)
            real_tok += tok
        elif train_tok < args.train_tokens:
            train_texts.append(text)
            train_tok += tok

        if real_tok >= args.real_tokens and train_tok >= args.train_tokens:
            break

        total = len(real_texts) + len(train_texts)
        if total % 5000 == 0:
            print(f"  D_real {real_tok/1e6:.1f}M tok / D_train {train_tok/1e6:.1f}M tok")

    print(f"[*] D_real  : {len(real_texts):,} 篇 (~{real_tok/1e6:.1f}M tokens)")
    print(f"[*] D_train : {len(train_texts):,} 篇 (~{train_tok/1e6:.1f}M tokens)")

    for name, data in [("real_texts", real_texts), ("train_texts", train_texts)]:
        path = DATA_DIR / f"{name}.json"
        with open(path, "w") as f:
            json.dump(data, f)
        print(f"[保存] {path}")


if __name__ == "__main__":
    main()
