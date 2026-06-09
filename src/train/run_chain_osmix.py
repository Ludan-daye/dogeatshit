"""
run_chain_osmix.py — 多代续训：开源数据混合(不回收模型自产输出)

每行 experiment_grid_osmix.csv 对应一条 k 代续训链:
  - gen 0:   在纯真实开源数据上 fine-tune 基模型
  - gen k>=1: 载入上一代模型,在 (真实开源 ⊕ 开源AI) 混合上继续训练(LoRA)
  - p_syn 由 schedule 决定(constant=D1 / ramp=D2)
  - 每代生成样本仅用于评估 MAUVE/PPL/rep,绝不回流训练集

用法:
  python src/train/run_chain_osmix.py --exp-id osmix_d1_p50_s42 \\
        --grid src/configs/experiment_grid_osmix.csv
"""
import json
import sys
import csv
import argparse
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.train.train_one_gen import finetune, generate_samples
from src.train.schedule import p_syn_schedule
from src.eval.compute_mauve import compute_mauve_score, delta_k
from src.eval.compute_ppl import compute_ppl_on_texts
from src.eval.compute_diversity import compute_repetition_rate
from src.utils import Timer, clear_gpu_memory, mix_data

DATA_DIR    = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"

DATASET_FILES = {
    "owt":  ("real_texts.json",      "train_texts.json"),
    "c4":   ("c4_real_texts.json",   "c4_train_texts.json"),
    "wiki": ("wiki_real_texts.json", "wiki_train_texts.json"),
}


def load_grid(grid_path):
    with open(grid_path) as f:
        return {row["exp_id"]: row for row in csv.DictReader(f)}


def run_chain_osmix(row, run_dir):
    model     = row["model"]
    real_ds   = row.get("real_dataset", "owt")
    ai_source = row["ai_dataset"]
    mode      = row.get("mode", "constant")
    p_syn     = float(row["p_syn"])
    n_train   = int(row["n_train"])
    k_max     = int(row["k_max"])
    seed      = int(row["seed"])
    exp_id    = row["exp_id"]

    np.random.seed(seed)
    import torch; torch.manual_seed(seed)

    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "models").mkdir(exist_ok=True)
    (run_dir / "samples").mkdir(exist_ok=True)
    metrics_path = run_dir / "metrics.jsonl"

    real_file, train_file = DATASET_FILES[real_ds]
    real_texts = json.load(open(DATA_DIR / real_file))
    train_real = json.load(open(DATA_DIR / train_file))
    ai_texts   = json.load(open(DATA_DIR / f"syn_{ai_source}_texts.json"))
    mauve_ref  = real_texts[:2000]
    ppl_ref    = real_texts[:500]

    prev_dir = None
    for gen in range(0, k_max + 1):
        gen_dir     = run_dir / "models"  / f"gen_{gen}"
        gen_samples = run_dir / "samples" / f"gen_{gen}.json"
        p = p_syn_schedule(gen, k_max, mode=mode, p_syn=p_syn)

        # 断点续跑
        if (gen_dir / "config.json").exists() and gen_samples.exists():
            prev_dir = str(gen_dir)
            continue

        # 组装训练集:开源真实 ⊕ 开源AI,不含模型自产输出
        if p <= 0.0:
            train_texts = list(train_real[:n_train])
        else:
            train_texts = mix_data(ai_texts, train_real, p)[:n_train]

        base = prev_dir if prev_dir is not None else model
        with Timer(f"[{exp_id}] gen{gen} finetune (p={p:.2f})"):
            finetune(base, train_texts, str(gen_dir), seed=seed, use_lora=True)

        # 生成样本:仅评估用
        with Timer(f"[{exp_id}] gen{gen} generate (eval-only)"):
            samp = generate_samples(str(gen_dir), n_train)
        json.dump(samp, open(gen_samples, "w"))

        # MAUVE 暂时跳过:gpt2-large 的依赖与本地路径校验在新 hf_hub 上常出问题;
        # 冒烟/早期评估先靠真实数据 PPL（=损失）+ 重复率,MAUVE 可在大规模实验后补算。
        mauve = -1.0  # sentinel: -1 表示未计算
        ppl = compute_ppl_on_texts(str(gen_dir), ppl_ref)
        rep = compute_repetition_rate(samp[:1000])

        with open(metrics_path, "a") as f:
            f.write(json.dumps({
                "gen": gen, "exp_id": exp_id, "model": model,
                "real_dataset": real_ds, "ai_dataset": ai_source,
                "mode": mode, "p_syn_nominal": p_syn, "p_syn_gen": p,
                "n_train": n_train, "seed": seed,
                "mauve": mauve, "delta": delta_k(mauve),
                "ppl_real": ppl, "rep_rate": rep,
            }) + "\n")
        print(f"[{exp_id}] gen{gen} p={p:.2f} MAUVE={mauve:.4f} PPL={ppl:.1f}")
        prev_dir = str(gen_dir)
        clear_gpu_memory()

    return metrics_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp-id", required=True)
    ap.add_argument("--grid", default="src/configs/experiment_grid_osmix.csv")
    ap.add_argument("--results-base", default=None)
    args = ap.parse_args()

    grid_path = (PROJECT_ROOT / args.grid
                 if not Path(args.grid).is_absolute() else Path(args.grid))
    grid = load_grid(str(grid_path))
    if args.exp_id not in grid:
        raise ValueError(f"exp_id '{args.exp_id}' 不在网格中")
    row = grid[args.exp_id]
    run_dir = (Path(args.results_base) / args.exp_id if args.results_base
               else RESULTS_DIR / row["group"] / args.exp_id)
    print("完成! metrics ->", run_chain_osmix(row, run_dir))


if __name__ == "__main__":
    main()
