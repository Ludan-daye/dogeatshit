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
                         max_length: int = 256) -> float:
    from transformers import AutoTokenizer, AutoModelForCausalLM

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    ).to(device)
    model.eval()

    total_loss   = 0.0
    total_tokens = 0

    with torch.no_grad():
        for text in texts:
            inputs = tokenizer(
                text, return_tensors="pt",
                truncation=True, max_length=max_length
            ).to(device)
            out  = model(**inputs, labels=inputs["input_ids"])
            ntok = inputs["input_ids"].shape[1]
            total_loss   += out.loss.item() * ntok
            total_tokens += ntok

    del model
    clear_gpu_memory()
    return float(np.exp(total_loss / max(total_tokens, 1)))
