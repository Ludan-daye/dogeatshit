"""
run_single_gen.py — 单代训练：预有合成数据混合实验

与 run_chain.py 不同，本脚本不做多代迭代。对每个实验行：
  1. 加载真实数据 + 预有AI生成数据（来自 HuggingFace）
  2. 按 p_syn 比例混合
  3. 单次 fine-tune 基础模型
  4. 生成样本并评估 MAUVE / PPL / rep_rate

用法：
  python run_single_gen.py --exp-id exp10_001 \\
                           --grid src/configs/experiment_grid_exp10.csv
"""

import json
import sys
import argparse
import csv
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.train.train_one_gen import finetune, generate_samples
from src.eval.compute_mauve import compute_mauve_score, delta_k
from src.eval.compute_ppl import compute_ppl_on_texts
from src.eval.compute_diversity import compute_repetition_rate
from src.utils import Timer, clear_gpu_memory, mix_data

DATA_DIR    = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"


# ── 读取实验网格 ────────────────────────────────────────────────────────────

def load_grid(grid_path: str) -> dict:
    """返回 {exp_id: row_dict} 映射"""
    with open(grid_path) as f:
        reader = csv.DictReader(f)
        return {row["exp_id"]: row for row in reader}


# ── 单代训练 ─────────────────────────────────────────────────────────────────

def run_single_gen(row: dict, run_dir: Path) -> Path:
    """
    row: experiment_grid_exp10.csv 的一行 (dict)
    run_dir: 本次运行的输出目录
    返回 metrics.jsonl 路径
    """
    model      = row["model"]
    p_syn      = float(row["p_syn"])
    n_train    = int(row["n_train"])
    seed       = int(row["seed"])
    exp_id     = row["exp_id"]
    syn_source = row["syn_source"]
    dataset    = row.get("dataset", "owt")

    np.random.seed(seed)
    import torch; torch.manual_seed(seed)

    run_dir.mkdir(parents=True, exist_ok=True)
    model_dir    = run_dir / "models"
    model_dir.mkdir(exist_ok=True)
    metrics_path = run_dir / "metrics.jsonl"

    # ── 断点检查：model + metrics 都存在则跳过 ──
    if (model_dir / "config.json").exists() and metrics_path.exists():
        with open(metrics_path) as f:
            lines = [l for l in f if l.strip()]
        if lines:
            print(f"[{exp_id}] 已完成，跳过")
            return metrics_path

    # ── 加载真实数据 ──
    dataset_files = {
        "owt":  ("real_texts.json",      "train_texts.json"),
        "c4":   ("c4_real_texts.json",   "c4_train_texts.json"),
        "wiki": ("wiki_real_texts.json", "wiki_train_texts.json"),
    }
    real_file, train_file = dataset_files[dataset]

    with open(DATA_DIR / real_file) as f:
        real_texts = json.load(f)
    with open(DATA_DIR / train_file) as f:
        train_texts = json.load(f)

    mauve_ref = real_texts[:2000]
    ppl_ref   = real_texts[:500]

    # ── 加载合成数据 ──
    syn_path = DATA_DIR / f"syn_{syn_source}_texts.json"
    with open(syn_path) as f:
        syn_texts = json.load(f)
    print(f"[{exp_id}] 合成数据: {syn_source} ({len(syn_texts):,} 篇)")

    # ── 混合训练集 ──
    if p_syn <= 0.0:
        mixed = list(train_texts[:n_train])
    else:
        mixed = mix_data(syn_texts, train_texts, p_syn)[:n_train]
    print(f"[{exp_id}] p_syn={p_syn}, 训练集={len(mixed)} 篇")

    # ── Fine-tune ──
    with Timer(f"[{exp_id}] fine-tune (p_syn={p_syn})"):
        finetune(model, mixed, str(model_dir), seed=seed)

    # ── 生成样本 ──
    with Timer(f"[{exp_id}] 生成样本"):
        samples = generate_samples(str(model_dir), n_train)

    samples_path = run_dir / "samples.json"
    with open(samples_path, "w") as f:
        json.dump(samples, f)

    # ── 评估 ──
    with Timer(f"[{exp_id}] MAUVE"):
        mauve = compute_mauve_score(mauve_ref, samples[:2000])
    clear_gpu_memory()

    ppl  = compute_ppl_on_texts(str(model_dir), ppl_ref)
    rep  = compute_repetition_rate(samples[:1000])

    metrics = {
        "exp_id": exp_id,
        "model": model,
        "dataset": dataset,
        "syn_source": syn_source,
        "p_syn": p_syn,
        "n_train": n_train,
        "seed": seed,
        "mauve": mauve,
        "delta": delta_k(mauve),
        "ppl_real": ppl,
        "rep_rate": rep,
    }

    with open(metrics_path, "w") as f:
        f.write(json.dumps(metrics) + "\n")

    print(f"[{exp_id}] MAUVE={mauve:.4f}  delta={1-mauve:.4f}  PPL={ppl:.1f}  rep={rep:.4f}")
    clear_gpu_memory()

    return metrics_path


# ── CLI ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-id", required=True,
                        help="experiment_grid_exp10.csv 中的 exp_id")
    parser.add_argument("--grid",
                        default="src/configs/experiment_grid_exp10.csv",
                        help="实验网格 CSV 路径")
    parser.add_argument("--results-base", default=None,
                        help="结果根目录（默认 results/<group>/<exp_id>）")
    args = parser.parse_args()

    grid_path = (PROJECT_ROOT / args.grid
                 if not Path(args.grid).is_absolute() else Path(args.grid))
    grid = load_grid(str(grid_path))

    if args.exp_id not in grid:
        raise ValueError(f"exp_id '{args.exp_id}' 不在网格中")

    row = grid[args.exp_id]

    if args.results_base:
        run_dir = Path(args.results_base) / args.exp_id
    else:
        run_dir = RESULTS_DIR / row["group"] / args.exp_id

    metrics_path = run_single_gen(row, run_dir)
    print(f"\n完成! metrics -> {metrics_path}")


if __name__ == "__main__":
    main()
