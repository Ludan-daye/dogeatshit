"""
计算 GPT-2（原版）在 D_real 上的 baseline PPL
运行一次，存入 results/baselines/baseline_ppl_<model>.json
"""

import json
import argparse
import numpy as np
import torch
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR     = PROJECT_ROOT / "data"
RESULTS_DIR  = PROJECT_ROOT / "results" / "baselines"


def compute_ppl(model_name: str, texts: list, max_length: int = 256) -> float:
    from transformers import AutoTokenizer, AutoModelForCausalLM

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[*] 设备: {device}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    ).to(device)
    model.eval()

    total_loss   = 0.0
    total_tokens = 0

    with torch.no_grad():
        for i, text in enumerate(texts):
            inputs = tokenizer(
                text, return_tensors="pt",
                truncation=True, max_length=max_length
            ).to(device)
            out  = model(**inputs, labels=inputs["input_ids"])
            ntok = inputs["input_ids"].shape[1]
            total_loss   += out.loss.item() * ntok
            total_tokens += ntok
            if (i + 1) % 100 == 0:
                print(f"  {i+1}/{len(texts)}")

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return float(np.exp(total_loss / total_tokens))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",     type=str, default="gpt2")
    parser.add_argument("--n-samples", type=int, default=500)
    parser.add_argument("--max-length",type=int, default=256)
    args = parser.parse_args()

    with open(DATA_DIR / "real_texts.json") as f:
        real_texts = json.load(f)[:args.n_samples]

    print(f"[*] 计算 {args.model} 在 D_real 上的 PPL ({len(real_texts)} 篇) ...")
    ppl = compute_ppl(args.model, real_texts, args.max_length)
    print(f"[*] Baseline PPL = {ppl:.2f}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / f"baseline_ppl_{args.model.replace('/', '_')}.json"
    with open(out_path, "w") as f:
        json.dump({"model": args.model, "ppl_real": ppl,
                   "n_samples": len(real_texts)}, f, indent=2)
    print(f"[保存] {out_path}")


if __name__ == "__main__":
    main()
