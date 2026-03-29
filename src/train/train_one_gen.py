"""
单代训练：给定上一代模型路径，LoRA fine-tune 并生成样本

对齐 Dohmatob ICML24 "A Tale of Tails" 设置：
- Llama2-7B + LoRA (r=16, alpha=32)
- 128-token 序列（前 96 prompt + 后 32 completion）
- top-p=0.9, temperature=0.9

用法（直接调用，也可作为模块导入）：
  python train_one_gen.py \
      --prev-model meta-llama/Llama-2-7b-hf \
      --train-texts data/train_texts.json \
      --output-dir results/exp1/run_xxx/models/gen_1 \
      --n-gen 2000 --epochs 1
"""

import json
import sys
import os
import argparse
import numpy as np
import torch
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from src.utils import clear_gpu_memory, gpu_mem_usage, Timer


# ── Fine-tune ────────────────────────────────────────────────────────────────

def finetune(
    prev_model: str,
    train_texts: list,
    output_dir: str,
    *,
    epochs: int       = 1,
    batch_size: int   = 4,
    grad_accum: int   = 8,
    lr: float         = 2e-5,
    max_length: int   = 128,
    warmup_steps: int = 100,
    seed: int         = 42,
    use_lora: bool    = True,
    lora_r: int       = 16,
    lora_alpha: int   = 32,
) -> None:
    from transformers import (
        AutoTokenizer, AutoModelForCausalLM,
        DataCollatorForLanguageModeling,
        Trainer, TrainingArguments,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(prev_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        prev_model,
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        attn_implementation="flash_attention_2" if device == "cuda" else None,
    )

    # ── LoRA ──
    if use_lora:
        from peft import LoraConfig, get_peft_model, TaskType
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=0.05,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    model = model.to(device)
    gpu_mem_usage()

    # ── Tokenise ──
    encodings = tokenizer(
        train_texts,
        truncation=True, max_length=max_length,
        padding="max_length", return_tensors="pt",
    )

    class _DS(torch.utils.data.Dataset):
        def __init__(self, enc):
            self.enc = enc
        def __len__(self):
            return len(self.enc["input_ids"])
        def __getitem__(self, i):
            return {
                "input_ids":      self.enc["input_ids"][i],
                "attention_mask": self.enc["attention_mask"][i],
                "labels":         self.enc["input_ids"][i].clone(),
            }

    dataset      = _DS(encodings)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        bf16=(device == "cuda"),
        learning_rate=lr,
        warmup_steps=warmup_steps,
        logging_steps=50,
        save_strategy="no",
        seed=seed,
        report_to="none",
        dataloader_num_workers=0,
        gradient_checkpointing=False,
    )

    trainer = Trainer(
        model=model, args=training_args,
        train_dataset=dataset, data_collator=data_collator,
    )
    trainer.train()

    # ── 保存：LoRA 合并后保存完整模型（确保下一代能正确加载）──
    os.makedirs(output_dir, exist_ok=True)
    if use_lora:
        model = model.merge_and_unload()
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"[保存] 模型 → {output_dir}")

    del trainer, model, dataset, encodings
    clear_gpu_memory()


# ── Generate（prompt-completion 模式）────────────────────────────────────────

def generate_samples(
    model_path: str,
    n_texts: int,
    *,
    prompt_texts: list = None,
    prompt_len: int    = 96,
    completion_len: int = 32,
    temperature: float = 0.9,
    top_p: float       = 0.9,
    max_length: int    = 128,
    gen_batch: int     = 8,
) -> list:
    """
    prompt-completion 生成模式（对齐 Dohmatob ICML24）：
    - 如果提供 prompt_texts，取每条前 prompt_len token 作 prompt，续写 completion_len token
    - 如果不提供 prompt_texts，从 BOS 开始自由生成 max_length token
    """
    from transformers import AutoTokenizer, AutoModelForCausalLM

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        attn_implementation="flash_attention_2" if device == "cuda" else None,
    ).to(device)
    model.eval()

    texts = []
    with torch.no_grad():
        if prompt_texts is not None:
            # prompt-completion 模式
            for i in range(0, min(n_texts, len(prompt_texts)), gen_batch):
                batch_prompts = prompt_texts[i : i + gen_batch]
                enc = tokenizer(
                    batch_prompts, return_tensors="pt",
                    truncation=True, max_length=prompt_len,
                    padding=True,
                ).to(device)
                out = model.generate(
                    **enc,
                    max_new_tokens=completion_len,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=True,
                    pad_token_id=tokenizer.pad_token_id,
                )
                for seq in out:
                    t = tokenizer.decode(seq, skip_special_tokens=True).strip()
                    if len(t) > 20:
                        texts.append(t)
                if i % (gen_batch * 50) == 0:
                    print(f"  已生成 {len(texts)}/{n_texts}")
        else:
            # 自由生成模式（fallback）
            bos = tokenizer.bos_token_id or tokenizer.eos_token_id
            for i in range(0, n_texts, gen_batch):
                curr = min(gen_batch, n_texts - i)
                ids = torch.tensor([[bos]] * curr).to(device)
                out = model.generate(
                    ids,
                    max_length=max_length,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=True,
                    pad_token_id=tokenizer.pad_token_id,
                )
                for seq in out:
                    t = tokenizer.decode(seq, skip_special_tokens=True).strip()
                    if len(t) > 20:
                        texts.append(t)
                if i % (gen_batch * 50) == 0:
                    print(f"  已生成 {len(texts)}/{n_texts}")

    del model
    clear_gpu_memory()
    return texts[:n_texts]


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prev-model",   required=True)
    parser.add_argument("--train-texts",  required=True,
                        help="JSON 文件路径，内含文本列表")
    parser.add_argument("--output-dir",   required=True)
    parser.add_argument("--n-gen",        type=int,   default=2000)
    parser.add_argument("--epochs",       type=int,   default=1)
    parser.add_argument("--batch",        type=int,   default=4)
    parser.add_argument("--grad-accum",   type=int,   default=8)
    parser.add_argument("--lr",           type=float, default=2e-5)
    parser.add_argument("--max-length",   type=int,   default=128)
    parser.add_argument("--temperature",  type=float, default=0.9)
    parser.add_argument("--seed",         type=int,   default=42)
    parser.add_argument("--no-lora",      action="store_true")
    parser.add_argument("--gen-output",   type=str,   default=None,
                        help="若指定，将生成文本保存到此 JSON 路径")
    args = parser.parse_args()

    with open(args.train_texts) as f:
        train_texts = json.load(f)

    with Timer("fine-tune"):
        finetune(
            args.prev_model, train_texts, args.output_dir,
            epochs=args.epochs, batch_size=args.batch,
            grad_accum=args.grad_accum, lr=args.lr,
            max_length=args.max_length, seed=args.seed,
            use_lora=not args.no_lora,
        )

    with Timer("生成样本"):
        samples = generate_samples(
            args.output_dir, args.n_gen,
            prompt_texts=train_texts,
            temperature=args.temperature, max_length=args.max_length,
        )

    if args.gen_output:
        os.makedirs(os.path.dirname(args.gen_output) or ".", exist_ok=True)
        with open(args.gen_output, "w") as f:
            json.dump(samples, f, ensure_ascii=False)
        print(f"[保存] 生成样本 ({len(samples)}) → {args.gen_output}")


if __name__ == "__main__":
    main()
