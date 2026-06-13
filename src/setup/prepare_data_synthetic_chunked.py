"""
prepare_data_synthetic_chunked.py — 格式对齐版合成数据准备

跟 prepare_data.py 完全相同的 tokenize + 切块流程:
  1. 加载 HF 合成数据集(默认 Cosmopedia web_samples_v2)
  2. 拼接全部文本(\\n\\n separator)
  3. 用 Llama2 tokenizer 精确分词
  4. 切成 128-token 固定块(跟 WikiText 一样)
  5. Decode 回文本,保存为 data/syn_{name}_chunked_texts.json

用途:做"格式对齐" ablation,排除"格式不一致 → MAUVE/KL 退化"confound。

用法:
  python prepare_data_synthetic_chunked.py --dataset cosmopedia --n-chunks 5000
  python prepare_data_synthetic_chunked.py --dataset cosmopedia --n-chunks 35000  # full
"""
import json
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"

DEFAULT_MODEL    = "/home/vicuna/ludan/reaserch/models/Mistral-7B-v0.1"
DEFAULT_SEQ_LEN  = 128
DEFAULT_N_CHUNKS = 5000
DEFAULT_MAX_DOCS = 100_000  # 取这么多篇原文就够切 5000+ 块

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="cosmopedia",
                        choices=list(SYNTHETIC_DATASETS.keys()))
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL,
                        help="Tokenizer 模型 (必须跟 prepare_data.py 一致)")
    parser.add_argument("--seq-len", type=int, default=DEFAULT_SEQ_LEN,
                        help="每块 token 数 (必须 = WikiText 端的 seq-len)")
    parser.add_argument("--n-chunks", type=int, default=DEFAULT_N_CHUNKS,
                        help="目标块数")
    parser.add_argument("--max-docs", type=int, default=DEFAULT_MAX_DOCS,
                        help="最多取多少原文档(保证够切 n-chunks)")
    args = parser.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # ── tokenizer ──
    print(f"[*] tokenizer: {args.model}")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    # ── 加载合成数据集 ──
    cfg = SYNTHETIC_DATASETS[args.dataset]
    print(f"[*] 加载 {cfg['hf_path']} (streaming={cfg['streaming']}) ...")
    from datasets import load_dataset
    kwargs = {"path": cfg["hf_path"], "split": cfg["split"], "streaming": cfg["streaming"]}
    if cfg["subset"]:
        kwargs["name"] = cfg["subset"]
    ds = load_dataset(**kwargs)

    # ── 收集原文 ──
    raw_texts = []
    for i, item in enumerate(ds):
        if i >= args.max_docs:
            break
        t = item[cfg["text_key"]]
        if isinstance(t, list):
            raw_texts.extend([x for x in t if isinstance(x, str) and x.strip()])
        elif isinstance(t, str) and t.strip():
            raw_texts.append(t.strip())
        if i % 5000 == 0 and i > 0:
            print(f"  已读 {i:,} 篇")
    print(f"[*] 共读 {len(raw_texts):,} 篇原文档")

    # ── 拼接 → tokenize → 切块(完全镜像 prepare_data.py)──
    all_text = "\n\n".join(raw_texts)
    print(f"[*] 拼接长度 {len(all_text):,} 字符")
    token_ids = tokenizer.encode(all_text)
    print(f"[*] tokenize 后 {len(token_ids):,} tokens")

    n_chunks_avail = len(token_ids) // args.seq_len
    n_chunks = min(args.n_chunks, n_chunks_avail)
    print(f"[*] 可切 {n_chunks_avail:,} 块,实际取 {n_chunks:,} 块")

    if n_chunks < args.n_chunks:
        print(f"[警告] 想要 {args.n_chunks} 块但只能切出 {n_chunks},考虑增大 --max-docs")

    chunks = [token_ids[i * args.seq_len : (i + 1) * args.seq_len]
              for i in range(n_chunks)]
    texts = [tokenizer.decode(c, skip_special_tokens=True) for c in chunks]

    # ── 保存 ──
    out = DATA_DIR / f"syn_{args.dataset}_chunked_texts.json"
    with open(out, "w") as f:
        json.dump(texts, f, ensure_ascii=False)
    print(f"[保存] {out}  ({len(texts):,} 块,每块 {args.seq_len} tokens)")


if __name__ == "__main__":
    main()
