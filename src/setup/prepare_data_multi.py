"""
多数据集准备：下载 C4 和 WikiText-103，划分 D_real / D_train

与 prepare_data.py 格式一致，输出到 data/ 目录：
  data/c4_real_texts.json    / data/c4_train_texts.json
  data/wiki_real_texts.json  / data/wiki_train_texts.json

用法：
  python prepare_data_multi.py                     # 准备全部
  python prepare_data_multi.py --dataset c4        # 只准备 C4
  python prepare_data_multi.py --dataset wiki      # 只准备 WikiText-103
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
    return len(text) // 4


def _collect_texts(iterator, text_key, real_tokens, train_tokens, min_len):
    """从数据集迭代器中收集 D_real 和 D_train"""
    real_texts, train_texts = [], []
    real_tok, train_tok = 0, 0

    for item in iterator:
        text = item[text_key].strip()
        if len(text) < min_len:
            continue
        tok = _approx_tokens(text)

        if real_tok < real_tokens:
            real_texts.append(text)
            real_tok += tok
        elif train_tok < train_tokens:
            train_texts.append(text)
            train_tok += tok

        if real_tok >= real_tokens and train_tok >= train_tokens:
            break

        total = len(real_texts) + len(train_texts)
        if total % 5000 == 0:
            print(f"  D_real {real_tok/1e6:.1f}M tok / D_train {train_tok/1e6:.1f}M tok")

    return real_texts, train_texts, real_tok, train_tok


def _save(prefix, real_texts, train_texts):
    for name, data in [(f"{prefix}_real_texts", real_texts),
                       (f"{prefix}_train_texts", train_texts)]:
        path = DATA_DIR / f"{name}.json"
        with open(path, "w") as f:
            json.dump(data, f)
        print(f"[保存] {path}  ({len(data):,} 篇)")


def prepare_c4(real_tokens, train_tokens, min_len):
    """下载 C4 (English, streaming)"""
    print("\n[*] 流式加载 C4 (en) ...")
    from datasets import load_dataset
    dataset = load_dataset("allenai/c4", "en", split="train", streaming=True)

    real_texts, train_texts, real_tok, train_tok = _collect_texts(
        dataset, "text", real_tokens, train_tokens, min_len)

    print(f"[*] C4 D_real  : {len(real_texts):,} 篇 (~{real_tok/1e6:.1f}M tokens)")
    print(f"[*] C4 D_train : {len(train_texts):,} 篇 (~{train_tok/1e6:.1f}M tokens)")
    _save("c4", real_texts, train_texts)


def prepare_wiki(real_tokens, train_tokens, min_len):
    """下载 WikiText-103"""
    print("\n[*] 加载 WikiText-103 ...")
    from datasets import load_dataset
    dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")

    real_texts, train_texts, real_tok, train_tok = _collect_texts(
        dataset, "text", real_tokens, train_tokens, min_len)

    print(f"[*] Wiki D_real  : {len(real_texts):,} 篇 (~{real_tok/1e6:.1f}M tokens)")
    print(f"[*] Wiki D_train : {len(train_texts):,} 篇 (~{train_tok/1e6:.1f}M tokens)")
    _save("wiki", real_texts, train_texts)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="all",
                        choices=["all", "c4", "wiki"],
                        help="准备哪个数据集 (默认: all)")
    parser.add_argument("--real-tokens",  type=int, default=DEFAULT_REAL_TOKENS)
    parser.add_argument("--train-tokens", type=int, default=DEFAULT_TRAIN_TOKENS)
    parser.add_argument("--min-length",   type=int, default=DEFAULT_MIN_LEN)
    args = parser.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if args.dataset in ("all", "c4"):
        prepare_c4(args.real_tokens, args.train_tokens, args.min_length)

    if args.dataset in ("all", "wiki"):
        prepare_wiki(args.real_tokens, args.train_tokens, args.min_length)

    print("\n[*] 数据准备完成!")


if __name__ == "__main__":
    main()
