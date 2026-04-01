"""
run_chain.py — 表驱动多代训练主程序

每一行 experiment_grid.csv 对应一次完整的代际链训练。
本脚本读取一行参数，运行该条件下的完整 k 代演化，逐代保存 metrics.jsonl。

用法：
  python run_chain.py --exp-id exp1_r_001 \\
                      --grid experiments/configs/experiment_grid.csv

支持断点续跑：若某代的 samples/gen_k.json 和 models/gen_k/config.json 均存在则跳过。
"""

import json
import sys
import os
import argparse
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.train.train_one_gen import finetune, generate_samples
from experiments.eval.compute_mauve  import compute_mauve_score, delta_k
from experiments.eval.compute_ppl    import compute_ppl_on_texts
from experiments.eval.compute_diversity import compute_repetition_rate
from experiments.utils import Timer, clear_gpu_memory

DATA_DIR    = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"


# ── 读取实验网格 ────────────────────────────────────────────────────────────

def load_grid(grid_path: str) -> dict:
    """返回 {exp_id: row_dict} 映射"""
    import csv
    with open(grid_path) as f:
        reader = csv.DictReader(f)
        return {row["exp_id"]: row for row in reader}


# ── 数据混合 ───────────────────────────────────────────────────────────────

def mix_data(syn_texts: list, real_texts: list, p_syn: float) -> list:
    """
    p_syn: synthetic 占比 (0~1)
    返回混合后打乱的文本列表
    """
    if p_syn >= 1.0:
        mixed = list(syn_texts)
    else:
        n     = min(len(real_texts), len(syn_texts)) if syn_texts else len(real_texts)
        n_syn  = int(n * p_syn)
        n_real = n - n_syn
        mixed  = real_texts[:n_real] + syn_texts[:n_syn]
    np.random.shuffle(mixed)
    return mixed


# ── 核心链式训练 ───────────────────────────────────────────────────────────

def run_chain(row: dict, run_dir: Path) -> Path:
    """
    row: experiment_grid.csv 的一行（dict）
    run_dir: 本次运行的输出目录
    返回 metrics.jsonl 路径
    """
    model     = row["model"]
    p_syn     = float(row["p_syn"])
    n_train   = int(row["n_train"])
    k_max     = int(row["k_max"])
    strategy  = row["strategy"]
    seed      = int(row["seed"])
    exp_id    = row["exp_id"]

    np.random.seed(seed)
    import torch; torch.manual_seed(seed)

    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "models").mkdir(exist_ok=True)
    (run_dir / "samples").mkdir(exist_ok=True)
    metrics_path = run_dir / "metrics.jsonl"

    # ── 加载数据（支持多数据集）──
    dataset = row.get("dataset", "owt")
    dataset_files = {
        "owt":  ("real_texts.json",      "train_texts.json"),
        "c4":   ("c4_real_texts.json",   "c4_train_texts.json"),
        "wiki": ("wiki_real_texts.json", "wiki_train_texts.json"),
    }
    real_file, train_file = dataset_files[dataset]

    with open(DATA_DIR / real_file) as f:
        real_texts = json.load(f)
    with open(DATA_DIR / train_file) as f:
        d0_texts = json.load(f)[:n_train]

    mauve_ref = real_texts[:2000]
    ppl_ref   = real_texts[:500]

    # ── 已记录的代数（用于断点续跑）──
    logged_gens = set()
    if metrics_path.exists():
        with open(metrics_path) as f:
            for line in f:
                if line.strip():
                    logged_gens.add(json.loads(line)["gen"])

    # ── Gen 0：在 D₀ 上初始 fine-tune ──
    gen0_dir     = run_dir / "models" / "gen_0"
    gen0_samples = run_dir / "samples" / "gen_0.json"

    if not (gen0_dir / "config.json").exists():
        print(f"\n[{exp_id}] Gen 0 初始训练")
        with Timer("Gen 0 fine-tune"):
            finetune(model, d0_texts, str(gen0_dir), seed=seed)

    if not gen0_samples.exists():
        with Timer("Gen 0 生成"):
            samp = generate_samples(str(gen0_dir), n_train)
        with open(gen0_samples, "w") as f:
            json.dump(samp, f)
    else:
        with open(gen0_samples) as f:
            samp = json.load(f)

    all_syn = list(samp)  # accumulate 模式用

    if 0 not in logged_gens:
        mauve_0 = compute_mauve_score(mauve_ref, samp[:2000])
        ppl_0   = compute_ppl_on_texts(str(gen0_dir), ppl_ref)
        rep_0   = compute_repetition_rate(samp[:1000])
        _append_metrics(metrics_path, {
            "gen": 0, "exp_id": exp_id,
            "model": model, "p_syn": p_syn, "n_train": n_train,
            "strategy": strategy, "seed": seed,
            "mauve": mauve_0, "delta": delta_k(mauve_0),
            "ppl_real": ppl_0, "rep_rate": rep_0,
        })
        print(f"[{exp_id}] Gen 0 — MAUVE={mauve_0:.4f}  δ={1-mauve_0:.4f}  PPL={ppl_0:.1f}")

    prev_dir = str(gen0_dir)
    prev_samp = samp

    # ── Gen 1 … k_max ──
    for gen in range(1, k_max + 1):
        gen_dir     = run_dir / "models"  / f"gen_{gen}"
        gen_samples = run_dir / "samples" / f"gen_{gen}.json"

        if gen_samples.exists() and (gen_dir / "config.json").exists():
            # 断点续跑
            with open(gen_samples) as f:
                prev_samp = json.load(f)
            if strategy == "accumulate":
                all_syn.extend(prev_samp)
            prev_dir = str(gen_dir)
            continue

        # 组装训练集
        if strategy == "replace":
            train_texts = mix_data(prev_samp, real_texts, p_syn)[:n_train]
        else:  # accumulate
            train_texts = mix_data(all_syn,  real_texts, p_syn)[:n_train]

        with Timer(f"[{exp_id}] Gen {gen} fine-tune"):
            finetune(prev_dir, train_texts, str(gen_dir), seed=seed)

        with Timer(f"[{exp_id}] Gen {gen} 生成"):
            prev_samp = generate_samples(str(gen_dir), n_train)

        with open(gen_samples, "w") as f:
            json.dump(prev_samp, f)

        if strategy == "accumulate":
            all_syn.extend(prev_samp)

        # 评估
        with Timer(f"[{exp_id}] Gen {gen} MAUVE"):
            mauve_k = compute_mauve_score(mauve_ref, prev_samp[:2000])
        clear_gpu_memory()

        ppl_k = compute_ppl_on_texts(str(gen_dir), ppl_ref)
        rep_k = compute_repetition_rate(prev_samp[:1000])

        _append_metrics(metrics_path, {
            "gen": gen, "exp_id": exp_id,
            "model": model, "p_syn": p_syn, "n_train": n_train,
            "strategy": strategy, "seed": seed,
            "mauve": mauve_k, "delta": delta_k(mauve_k),
            "ppl_real": ppl_k, "rep_rate": rep_k,
        })
        print(f"[{exp_id}] Gen {gen} — MAUVE={mauve_k:.4f}  δ={1-mauve_k:.4f}  PPL={ppl_k:.1f}")
        prev_dir = str(gen_dir)

    return metrics_path


def _append_metrics(path: Path, row: dict):
    with open(path, "a") as f:
        f.write(json.dumps(row) + "\n")


# ── CLI ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-id", required=True,
                        help="experiment_grid.csv 中的 exp_id")
    parser.add_argument("--grid",
                        default="experiments/configs/experiment_grid.csv",
                        help="实验网格 CSV 路径")
    parser.add_argument("--results-base", default=None,
                        help="结果根目录（默认 results/<group>/<exp_id>）")
    args = parser.parse_args()

    grid = load_grid(PROJECT_ROOT / args.grid
                     if not Path(args.grid).is_absolute() else args.grid)

    if args.exp_id not in grid:
        raise ValueError(f"exp_id '{args.exp_id}' 不在网格中")

    row = grid[args.exp_id]

    if args.results_base:
        run_dir = Path(args.results_base) / args.exp_id
    else:
        run_dir = RESULTS_DIR / row["group"] / args.exp_id

    metrics_path = run_chain(row, run_dir)
    print(f"\n完成! metrics → {metrics_path}")


if __name__ == "__main__":
    main()
