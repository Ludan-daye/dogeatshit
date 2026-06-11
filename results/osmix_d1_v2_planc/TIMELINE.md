# v2 Plan-C 运行时间线

记录主要事件、单代耗时、复现所需信息。基于 `run_artifacts/chain_logs/` 实际日志提取。

## 概览

| 项 | 值 |
|---|---|
| 设备 | NVIDIA A100 80GB PCIe（vicuna@8.138.30.52） |
| 模型 | Mistral-7B-v0.1 + LoRA(r=16, α=32, target=q/k/v/o_proj) |
| 训练规模 | n_train=5000 / gen, batch=4, grad_accum=8, 2 epochs |
| 生成规模 | 5000 samples / gen, temperature=0.9, top_p=0.95, max_len=256 |
| 并发 | 3 链同时跑(共享 GPU,各自独立 process) |
| 优化器 | AdamW, lr=5e-5, warmup=50, bf16 |
| 数据池 | WikiText-103 30000 训练块 + Cosmopedia 35131 块(Plan-C 不重叠子集) |

## 关键时间节点

| 时间 | 事件 |
|---|---|
| 2026-06-10 10:25 | 数据准备完成,3 链启动 |
| 2026-06-10 12:08 | gen 0 全部 finetune+generate 完成,gen 0 metrics 写入 |
| 2026-06-10 19:00 | ~gen 2 完成 |
| 2026-06-11 02:30 | p=0.0 跑到 gen 5(池子用尽,自然停止) |
| 2026-06-11 04:00 | p=1.0 跑到 gen 6(池子用尽) |
| 2026-06-11 08:00 | p=0.5 跑到 gen 10(因为每代两池各用 2500,池容量是其他链 2 倍) |
| 2026-06-11 13:50 | 离线 MAUVE 完成 |
| 2026-06-11 14:00 | 数据 + 图 + 扩展指标推到 GitHub |

## 单代耗时(p=0.0 链作典型,3 进程共享 A100)

```
gen   finetune   generate   合计
 0     67 min     86 min    ~155 min
 1     68 min     85 min    ~155 min
 2     69 min     92 min    ~163 min
 3     76 min     85 min    ~162 min
```

→ **单代约 2.5-3 小时 wall time**(3 进程共享 GPU 拖慢)。单进程独占 GPU 估计 ~70-90 min/代。

## 总计算量

- 24 gens × ~2.7 hr/gen wall ≈ **~65 小时 wall time**(3 链并发)
- 等效单进程 GPU 时间 ≈ **~190 小时 A100**
- 离线 MAUVE 评估:~3 hr
- 离线扩展指标:~5 min(CPU)

## 复现入口

1. 数据准备(~30 min):
   ```bash
   python src/setup/prepare_data.py --model <mistral_path> --n-real 5000 --n-train 30000
   python src/setup/prepare_data_synthetic.py --dataset cosmopedia --n-tokens 35000000
   ```
2. 训练 3 链(~65 hr,A100):
   ```bash
   for E in osmix_d1_p00_s42 osmix_d1_p50_s42 osmix_d1_p100_s42; do
     nohup python src/train/run_chain_osmix.py --exp-id $E \
       --grid src/configs/experiment_grid_osmix.csv &
   done
   ```
3. 离线 MAUVE(~3 hr,串行):
   ```bash
   for E in osmix_d1_p00_s42 osmix_d1_p50_s42 osmix_d1_p100_s42; do
     python src/analysis/compute_mauve_offline.py \
       --chain-dir results/osmix_d1_v2_planc/$E \
       --ref-file data/real_texts.json \
       --featurize-model <gpt2-large_path>
   done
   ```
4. 扩展指标(~5 min,CPU):
   ```bash
   for E in osmix_d1_p00_s42 osmix_d1_p50_s42 osmix_d1_p100_s42; do
     python src/analysis/compute_extended_metrics.py \
       --chain-dir results/osmix_d1_v2_planc/$E \
       --ref-file data/real_texts.json
   done
   ```
5. 画图:`python src/analysis/plot_osmix_v2.py && python src/analysis/plot_extended_v2.py`

## 已知问题 / 经验

- **3 链并发**比单链快 ~2.5×(每链 GPU 时间被瓜分),但内存利用率好(64 GB / 80 GB)
- batch_size=4(而非 8)是为了让 3 个 Mistral-7B + 激活内存能放下;调到 8 会 OOM
- `GEN0_FIXED_SEED=42` 保证不同 seed 链的 gen 0 完全相同(`src/train/run_chain_osmix.py`)
- `data/syn_cosmopedia_texts.json` 用 35M tokens 抽样得到 35131 条;之前用 15M tokens 只得 15087 条(池不够 30k)
- 服务器一度因 transformers 5.10 API 重构无法加载 Trainer → 后退到 transformers 4.46.3 + huggingface_hub 0.26.5
