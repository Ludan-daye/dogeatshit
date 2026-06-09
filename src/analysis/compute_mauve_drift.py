"""
compute_mauve_drift.py — 用模型自己的 gen_0 输出作参考,计算每代相对起始状态的 MAUVE 漂移

补充 compute_mauve_offline.py(后者用真实数据作参考):
  - 参考分布 = 同链 gen_0 的生成样本(若 --ref-from-chain 同一目录)
  - 或     = 指定其它链的某代输出(--ref-samples 任意路径)
  - q 分布 = 同链每一代的样本

输出 results/<chain>/metrics_drift.jsonl:
  {gen, ref_label, mauve_drift, delta_drift, ppl_real, ...}

用法：
  # 用同链 gen_0 作参考
  python src/analysis/compute_mauve_drift.py \\
    --chain-dir results/osmix_d1/osmix_d1_p50_s42 \\
    --ref-from-chain --ref-gen 0 \\
    --featurize-model ~/ludan/reaserch/models/gpt2-large

  # 用基线链 gen_5 作参考
  python src/analysis/compute_mauve_drift.py \\
    --chain-dir results/osmix_d1/osmix_d1_p50_s42 \\
    --ref-samples results/osmix_d1/osmix_d1_p00_s42/samples/gen_5.json \\
    --ref-label baseline_gen5 \\
    --featurize-model ~/ludan/reaserch/models/gpt2-large
"""
import json
import sys
import os
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))


def main(chain_dir, ref_samples_path, ref_label, featurize_model, n_ref, n_samp):
    chain_dir = Path(chain_dir).resolve()
    samples_dir = chain_dir / "samples"
    metrics_path = chain_dir / "metrics.jsonl"
    out_path = chain_dir / f"metrics_drift_vs_{ref_label}.jsonl"

    # 已有 PPL metrics
    rows_by_gen = {}
    if metrics_path.exists():
        with open(metrics_path) as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    rows_by_gen[r["gen"]] = r

    # 加载参考样本
    with open(ref_samples_path) as f:
        ref = json.load(f)[:n_ref]
    print(f"reference: {ref_samples_path} ({len(ref)} samples, label={ref_label})")

    samp_files = sorted(samples_dir.glob("gen_*.json"),
                        key=lambda p: int(p.stem.replace("gen_", "")))

    if featurize_model:
        os.environ["MAUVE_FEATURIZE_MODEL"] = str(Path(featurize_model).expanduser())

    from src.eval.compute_mauve import compute_mauve_score

    with open(out_path, "w") as outf:
        for samp_file in samp_files:
            gen = int(samp_file.stem.replace("gen_", ""))
            with open(samp_file) as f:
                samp = json.load(f)
            print(f"[gen {gen}] MAUVE drift vs {ref_label} ...")
            try:
                m = compute_mauve_score(ref, samp[:n_samp])
            except Exception as e:
                print(f"[gen {gen}] FAIL: {e}")
                m = -1.0
            row = dict(rows_by_gen.get(gen, {"gen": gen}))
            row["ref_label"] = ref_label
            row["mauve_drift"] = m
            row["delta_drift"] = (1.0 - m) if m >= 0 else None
            outf.write(json.dumps(row) + "\n")
            print(f"[gen {gen}] mauve_drift = {m:.4f}")
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--chain-dir", required=True)
    ap.add_argument("--ref-from-chain", action="store_true",
                    help="用 --chain-dir 自己的 samples/gen_{ref_gen}.json 作参考")
    ap.add_argument("--ref-gen", type=int, default=0)
    ap.add_argument("--ref-samples", default=None,
                    help="任意 samples JSON 路径(覆盖 --ref-from-chain)")
    ap.add_argument("--ref-label", default=None)
    ap.add_argument("--featurize-model", default=None)
    ap.add_argument("--n-ref", type=int, default=2000)
    ap.add_argument("--n-samp", type=int, default=2000)
    args = ap.parse_args()

    if args.ref_samples:
        ref_path = Path(args.ref_samples).resolve()
        label = args.ref_label or ref_path.parent.parent.name + "_" + ref_path.stem
    elif args.ref_from_chain:
        ref_path = Path(args.chain_dir).resolve() / "samples" / f"gen_{args.ref_gen}.json"
        label = args.ref_label or f"self_gen{args.ref_gen}"
    else:
        raise ValueError("must specify either --ref-from-chain or --ref-samples")

    main(args.chain_dir, ref_path, label, args.featurize_model, args.n_ref, args.n_samp)
