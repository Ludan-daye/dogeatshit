"""
合成数据准备：从 HuggingFace 下载 AI 生成文本数据集

支持数据集：
  - cosmopedia: HuggingFaceTB/cosmopedia (Mixtral-8x7B 生成)
  - gptwiki:    aadityaubhat/GPT-wiki-intro (GPT-3 生成)

输出格式与 prepare_data.py 一致: data/syn_{name}_texts.json (JSON 字符串列表)

用法：
  python prepare_data_synthetic.py --dataset cosmopedia
  python prepare_data_synthetic.py --dataset gptwiki --n-tokens 2000000
  python prepare_data_synthetic.py --dataset all
"""

import json
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"

DEFAULT_N_TOKENS = 5_000_000
DEFAULT_MIN_LEN = 200

# ── 数据集注册表 ─────────────────────────────────────────────────────────────

SYNTHETIC_DATASETS = {
    "cosmopedia": {
        "hf_path": "HuggingFaceTB/cosmopedia",
        "subset": "web_samples_v2",
        "split": "train",
        "text_key": "text",
        "streaming": True,
    },
    "gptwiki": {
        "hf_path": "aadityaubhat/GPT-wiki-intro",
        "subset": None,
        "split": "train",
        "text_key": "generated_intro",
        "streaming": False,
    },
}


def _approx_tokens(text: str) -> int:
    return len(text) // 4


def collect_synthetic(dataset_name: str, n_tokens: int, min_len: int):
    """从 HuggingFace 下载并收集合成文本"""
    from datasets import load_dataset

    cfg = SYNTHETIC_DATASETS[dataset_name]
    print(f"\n[*] 加载 {cfg['hf_path']} (streaming={cfg['streaming']}) ...")

    kwargs = {
        "path": cfg["hf_path"],
        "split": cfg["split"],
        "streaming": cfg["streaming"],
    }
    if cfg["subset"]:
        kwargs["name"] = cfg["subset"]

    dataset = load_dataset(**kwargs)

    texts = []
    total_tok = 0
    text_key = cfg["text_key"]

    for item in dataset:
        raw = item[text_key]
        # HC3 等数据集的 text_key 可能是 list
        if isinstance(raw, list):
            entries = [t.strip() for t in raw if isinstance(t, str) and len(t.strip()) >= min_len]
        else:
            entries = [raw.strip()] if len(raw.strip()) >= min_len else []

        for text in entries:
            texts.append(text)
            total_tok += _approx_tokens(text)

        if len(texts) % 5000 == 0 and len(texts) > 0:
            print(f"  已收集 {len(texts):,} 篇 (~{total_tok/1e6:.1f}M tokens)")

        if total_tok >= n_tokens:
            break

    print(f"[*] {dataset_name}: 收集 {len(texts):,} 篇 (~{total_tok/1e6:.1f}M tokens)")
    return texts


def save_synthetic(dataset_name: str, texts: list):
    """保存合成文本到 data/syn_{name}_texts.json"""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = DATA_DIR / f"syn_{dataset_name}_texts.json"
    with open(path, "w") as f:
        json.dump(texts, f)
    print(f"[保存] {path}  ({len(texts):,} 篇)")


def main():
    parser = argparse.ArgumentParser(description="下载 HuggingFace AI 生成数据集")
    parser.add_argument("--dataset", type=str, default="cosmopedia",
                        choices=list(SYNTHETIC_DATASETS.keys()) + ["all"],
                        help="数据集名称 (默认: cosmopedia)")
    parser.add_argument("--n-tokens", type=int, default=DEFAULT_N_TOKENS,
                        help=f"目标 token 数 (默认: {DEFAULT_N_TOKENS:,})")
    parser.add_argument("--min-length", type=int, default=DEFAULT_MIN_LEN,
                        help=f"最小文本长度 (默认: {DEFAULT_MIN_LEN})")
    args = parser.parse_args()

    datasets_to_prepare = (
        list(SYNTHETIC_DATASETS.keys()) if args.dataset == "all"
        else [args.dataset]
    )

    for name in datasets_to_prepare:
        texts = collect_synthetic(name, args.n_tokens, args.min_length)
        save_synthetic(name, texts)

    print("\n[*] 合成数据准备完成!")


if __name__ == "__main__":
    main()
