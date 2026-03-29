"""
PPL_real：模型在真实数据上的困惑度
"""

import sys
import numpy as np
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils import clear_gpu_memory


def compute_ppl_on_texts(model_path: str, texts: list,
                         max_length: int = 128,
                         batch_size: int = 16) -> float:
    from transformers import AutoTokenizer, AutoModelForCausalLM

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        attn_implementation="flash_attention_2" if device == "cuda" else None,
    ).to(device)
    model.eval()

    total_loss   = 0.0
    total_tokens = 0

    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            inputs = tokenizer(
                batch_texts, return_tensors="pt",
                truncation=True, max_length=max_length,
                padding=True,
            ).to(device)
            out = model(**inputs, labels=inputs["input_ids"])
            # 只统计非 pad token
            mask = inputs["attention_mask"]
            ntok = mask.sum().item()
            total_loss   += out.loss.item() * ntok
            total_tokens += ntok

    del model
    clear_gpu_memory()
    return float(np.exp(total_loss / max(total_tokens, 1)))
