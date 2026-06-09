"""
compute_mauve_offline.py — 事后在保存的 samples 上批量算 MAUVE

训练阶段为了快和稳，run_chain_osmix 只测真实数据 PPL（=损失）+ 重复率，
MAUVE 记一个 sentinel -1.0。本脚本扫描某条链的 samples/，对每代 gen_k.json
跟真实 reference 文本算 MAUVE，写出 metrics_with_mauve.jsonl。

用法：
  python src/analysis/compute_mauve_offline.py \\
      --chain-dir results/_smoke/osmix_smoke \\
      --ref-file data/real_texts.json \\
      --featurize-model ~/ludan/reaserch/models/gpt2-large
"""
import json
import sys
import os
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))


def main(chain_dir, ref_file, featurize_model, n_ref, n_samp):
    chain_dir = Path(chain_dir).resolve()
    metrics_path = chain_dir / "metrics.jsonl"
    samples_dir = chain_dir / "samples"
    out_path = chain_dir / "metrics_with_mauve.jsonl"

    # 读已有 metrics（可能 mauve=-1.0 占位）
    rows_by_gen = {}
    if metrics_path.exists():
        with open(metrics_path) as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    rows_by_gen[r["gen"]] = r
        print(f"loaded {len(rows_by_gen)} existing metric rows")

    samp_files = sorted(samples_dir.glob("gen_*.json"),
                        key=lambda p: int(p.stem.replace("gen_", "")))
    if not samp_files:
        print(f"no samples found in {samples_dir}")
        return

    with open(ref_file) as f:
        ref = json.load(f)[:n_ref]
    print(f"loaded {len(ref)} reference texts from {ref_file}")

    if featurize_model:
        fm = str(Path(featurize_model).expanduser())
        os.environ["MAUVE_FEATURIZE_MODEL"] = fm
        print(f"MAUVE_FEATURIZE_MODEL = {fm}")

    from src.eval.compute_mauve import compute_mauve_score, delta_k

    with open(out_path, "w") as outf:
        for samp_file in samp_files:
            gen = int(samp_file.stem.replace("gen_", ""))
            with open(samp_file) as f:
                samp = json.load(f)
            print(f"[gen {gen}] MAUVE on {min(len(samp), n_samp)} samples vs {len(ref)} ref...")
            try:
                mauve = compute_mauve_score(ref, samp[:n_samp])
                status = "OK"
            except Exception as e:
                print(f"[gen {gen}] MAUVE failed: {e}")
                mauve = -1.0
                status = f"FAIL ({e.__class__.__name__})"

            row = dict(rows_by_gen.get(gen, {"gen": gen}))
            row["mauve"] = mauve
            row["delta"] = (1.0 - mauve) if mauve >= 0 else None
            outf.write(json.dumps(row) + "\n")
            print(f"[gen {gen}] mauve={mauve:.4f} ({status})")

    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--chain-dir", required=True,
                    help="链目录（含 metrics.jsonl 和 samples/）")
    ap.add_argument("--ref-file", required=True,
                    help="真实数据 JSON（列表）")
    ap.add_argument("--featurize-model", default=None,
                    help="gpt2-large 本地路径或 HF id（默认读 MAUVE_FEATURIZE_MODEL 或 gpt2-large）")
    ap.add_argument("--n-ref", type=int, default=2000)
    ap.add_argument("--n-samp", type=int, default=2000)
    args = ap.parse_args()
    main(args.chain_dir, args.ref_file, args.featurize_model, args.n_ref, args.n_samp)
