# Results

实验输出汇总。当前主线:**Mistral-7B + LoRA 多代续训于开源数据混合**（WikiText-103 真实 ⊕ Cosmopedia 开源 AI）。

## 目录

| 路径 | 内容 |
|------|------|
| `osmix_d1_v2_planc/` | **主结果**：Plan-C 实验,每代不重叠子集 + 30k+ 池 + `GEN0_FIXED_SEED=42` |
| `_smoke/` | 冒烟测试(n_train=200, 2 代),验证流水线 |
| `exp0/`、`exp1/`、`exp6/`、`exp10/`、`baseline_compare/` | 历史实验(早期 toy/线性/GPT-2/Mistral 单代),已归档 |
| `archive_2024-03-19/` | 更早期归档,仅作历史参考 |

**早期 v1 实验**(每代复用同一份 5000 条样本)有方法学问题(小集合反复 fine-tune 的过拟合 confound),**不收录到仓库**。直接看 v2 Plan-C 数据。

**模型权重不在仓库中**(每代 ~14GB,见 `.gitignore`)。
样本(`samples/gen_*.json`)和指标(`metrics.jsonl`、`metrics_with_mauve.jsonl`)保留,用于复现和分析。

## 主结果概要(`osmix_d1_v2_planc/`)

3 条链 × p_syn ∈ {0, 0.5, 1.0} × seed=42,Plan-C 不重叠采样。

| Chain | 代数 | 末代 PPL | 末代 MAUVE | 头条 |
|-------|------|----------|------------|------|
| `osmix_d1_p00_s42` | 6 (gen 0–5) | **8.12** | **0.776** | 基线健康学习,PPL ↓ 2.4%,MAUVE ↑ 29% |
| `osmix_d1_p50_s42` | **11** (gen 0–10) | **8.18** | **0.405** | **Stealth collapse**:PPL ≈ 基线,MAUVE 仍差 48% |
| `osmix_d1_p100_s42` | 7 (gen 0–6) | **12.85** | **0.062** | 双崩:PPL +54%,MAUVE 89% 永久损失 |

> 注:不同链代数不同,因为 p_syn=0.5 链每代两个池子各用一半样本,所以训练池(30k)够用更多代;p_syn=0/1.0 只用一个池子,代数受池子大小限制。

### 三大头条发现

1. **Stealth Collapse 存在**:p=0.5 末代 PPL=8.18 跟基线 8.12 只差 0.06,但 MAUVE 永久差 48%。**只看 PPL 漏掉一半退化**。
2. **PPL self-heals, MAUVE scars**:p=0.5 MAUVE gen 1 暴跌 73% 后渐进恢复但**不可达起点**。
3. **rep_rate 跟 MAUVE 解耦**:p=0.5 词频多样性稳定 ~0.017,但 MAUVE 大起大落 → **MAUVE 是 stealth collapse 的唯一可观测信号**。

## 文件 schema

### `metrics.jsonl`（训练时实时产出）
每行一个 JSON,字段:
- `gen`: 代数(0…k_max)
- `p_syn_nominal`: 链配置的 p_syn
- `p_syn_gen`: 该代实际 p_syn(`p_syn_schedule` 决定,gen 0 强制 0)
- `n_train`: 该代训练样本数
- `seed`: 链 seed
- `ppl_real`: 真实数据 hold-out 集 PPL
- `rep_rate`: 4-gram 重复率
- `mauve`: -1.0 (sentinel — 训练时不算 MAUVE)
- `delta`: 1 - mauve

### `metrics_with_mauve.jsonl`（离线 MAUVE 补算）
跟 `metrics.jsonl` 字段一样,但 `mauve` 和 `delta` 用 `src/analysis/compute_mauve_offline.py` 跟真实数据 `data/real_texts.json` 对比算出来的真实值。

### `samples/gen_k.json`
该代模型(从 BOS 无条件)生成的样本列表(每条 ≤256 tokens)。

## 复现

详见 `docs/superpowers/specs/2026-06-08-mistral-opensource-mixing-collapse-design.md`(设计)和
`docs/superpowers/plans/2026-06-08-mistral-osmix-experiment.md`(实施计划)。

数据准备:
```bash
python src/setup/prepare_data.py --model <mistral_path> --n-real 5000 --n-train 30000
python src/setup/prepare_data_synthetic.py --dataset cosmopedia --n-tokens 35000000
```

训练:
```bash
python src/train/run_chain_osmix.py --exp-id osmix_d1_p50_s42 \
  --grid src/configs/experiment_grid_osmix.csv
```

离线 MAUVE:
```bash
python src/analysis/compute_mauve_offline.py \
  --chain-dir results/osmix_d1_v2_planc/osmix_d1_p50_s42 \
  --ref-file data/real_texts.json \
  --featurize-model <gpt2-large_path>
```
