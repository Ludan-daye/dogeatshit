# Mistral 开源混合多代续训实验 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 让仓库能跑"多代续训 on 开源数据混合(真实 OWT ⊕ 开源 AI 数据 Cosmopedia,不回收模型自产输出)"的 D1/D2 实验,产出每代真实数据损失(PPL/CE)。

**Architecture:** 复用现有 `train_one_gen`(加 LoRA)与 `eval/*` 指标,新增一个多代续训 runner `run_chain_osmix.py`(脱胎于 `run_single_gen.py` 的单代逻辑 + `run_chain.py` 的代际循环);每代用固定外部 AI 语料按 `p_syn` 调度混入真实数据,生成样本仅用于评估。先修好 3 处坏 import。

**Tech Stack:** Python 3.10, PyTorch, HuggingFace transformers + datasets, PEFT(LoRA), mauve-text, pytest。GPU:A100-80GB(训练/生成);纯逻辑任务无需 GPU。

---

## File Structure

| 文件 | 职责 | 动作 |
|------|------|------|
| `src/train/run_chain.py` | 旧自消费链(exp6) | 改:修 import |
| `src/train/train_one_gen.py` | 单代 fine-tune + 生成 | 改:修 import + 加 LoRA |
| `src/analysis/compare_models.py` | 跨模型对比图 | 改:修 import |
| `src/train/schedule.py` | p_syn 代际调度(D1/D2) | **新建**(纯函数,无重依赖) |
| `src/train/run_chain_osmix.py` | 开源混合多代续训 runner | **新建** |
| `src/configs/experiment_grid_osmix.csv` | D1/D2 网格 | **新建** |
| `tests/test_no_stale_imports.py` | 回归:无 `experiments.*` import | **新建** |
| `tests/test_schedule.py` | p_syn 调度单测 | **新建** |
| `tests/test_grid_osmix.py` | 网格 CSV 校验 | **新建** |
| `CLAUDE.md` / `README.md` | 文档对齐 src/ + 新范式 | 改 |

GPU-free 任务(可在任何机器跑测试):1、2、5。需要 A100:3(peft 构造可无 GPU 测,实训需 GPU)、4、6。

---

## Task 1: 修复坏掉的 `experiments.*` import

**Files:**
- Modify: `src/train/run_chain.py:24-28`
- Modify: `src/train/train_one_gen.py:23`
- Modify: `src/analysis/compare_models.py:28-29`
- Test: `tests/test_no_stale_imports.py`

- [ ] **Step 1: 写失败测试**

```python
# tests/test_no_stale_imports.py
import pathlib, re

def test_no_experiments_imports():
    src = pathlib.Path(__file__).resolve().parents[1] / "src"
    pat = re.compile(r'^\s*(from|import)\s+experiments(\.|\s|$)', re.M)
    offenders = [str(p) for p in src.rglob("*.py") if pat.search(p.read_text(encoding="utf-8"))]
    assert not offenders, f"stale 'experiments' imports remain in: {offenders}"
```

- [ ] **Step 2: 跑测试确认失败**

Run: `pytest tests/test_no_stale_imports.py -v`
Expected: FAIL — offenders 含 run_chain.py / train_one_gen.py / compare_models.py

- [ ] **Step 3: 改 `src/train/run_chain.py` 第 24-28 行**

```python
from src.train.train_one_gen import finetune, generate_samples
from src.eval.compute_mauve  import compute_mauve_score, delta_k
from src.eval.compute_ppl    import compute_ppl_on_texts
from src.eval.compute_diversity import compute_repetition_rate
from src.utils import Timer, clear_gpu_memory
```

- [ ] **Step 4: 改 `src/train/train_one_gen.py` 第 23 行**

```python
from src.utils import clear_gpu_memory, gpu_mem_usage, Timer
```

- [ ] **Step 5: 改 `src/analysis/compare_models.py` 第 28-29 行**

```python
from src.utils import save_fig, save_csv
from src.analysis.plot_results import load_all_metrics, _estimate_alpha
```

- [ ] **Step 6: 跑测试确认通过**

Run: `pytest tests/test_no_stale_imports.py -v`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add tests/test_no_stale_imports.py src/train/run_chain.py src/train/train_one_gen.py src/analysis/compare_models.py
git commit -m "fix: repair stale experiments.* imports (renamed to src.*)"
```

---

## Task 2: p_syn 代际调度(D1 固定 / D2 递增)

**Files:**
- Create: `src/train/schedule.py`
- Test: `tests/test_schedule.py`

- [ ] **Step 1: 写失败测试**

```python
# tests/test_schedule.py
import pytest
from src.train.schedule import p_syn_schedule

def test_gen0_is_always_pure_real():
    assert p_syn_schedule(0, 10, mode="constant", p_syn=1.0) == 0.0
    assert p_syn_schedule(0, 10, mode="ramp", p_syn=1.0) == 0.0

def test_constant_mode_returns_fixed_ratio_after_gen0():
    assert p_syn_schedule(1, 10, mode="constant", p_syn=0.5) == 0.5
    assert p_syn_schedule(7, 10, mode="constant", p_syn=0.5) == 0.5

def test_ramp_mode_is_linear_to_p_syn_at_kmax():
    assert p_syn_schedule(5, 10, mode="ramp", p_syn=1.0) == pytest.approx(0.5)
    assert p_syn_schedule(10, 10, mode="ramp", p_syn=1.0) == pytest.approx(1.0)

def test_unknown_mode_raises():
    with pytest.raises(ValueError):
        p_syn_schedule(1, 10, mode="nope", p_syn=1.0)
```

- [ ] **Step 2: 跑测试确认失败**

Run: `pytest tests/test_schedule.py -v`
Expected: FAIL — ModuleNotFoundError: src.train.schedule

- [ ] **Step 3: 实现 `src/train/schedule.py`**

```python
"""p_syn 代际调度：决定第 gen 代训练集中开源 AI 数据的占比。

gen 0 永远是纯真实数据(基模型)。
  - constant (D1): gen>=1 恒为 p_syn
  - ramp     (D2): gen>=1 时线性 0 -> p_syn(在 gen=k_max 处达到 p_syn)
"""

def p_syn_schedule(gen: int, k_max: int, *, mode: str = "constant", p_syn: float = 1.0) -> float:
    if gen <= 0:
        return 0.0
    if mode == "constant":
        return p_syn
    if mode == "ramp":
        return p_syn * gen / k_max
    raise ValueError(f"unknown schedule mode: {mode!r}")
```

- [ ] **Step 4: 跑测试确认通过**

Run: `pytest tests/test_schedule.py -v`
Expected: PASS(4 passed)

- [ ] **Step 5: Commit**

```bash
git add src/train/schedule.py tests/test_schedule.py
git commit -m "feat: add p_syn generation schedule (D1 constant / D2 ramp)"
```

---

## Task 3: 给 `train_one_gen.finetune` 加 LoRA 续训路径

**Files:**
- Modify: `src/train/train_one_gen.py`(新增 `build_lora_config`;`finetune` 加 `use_lora` 参数)
- Test: `tests/test_lora_config.py`

- [ ] **Step 1: 写失败测试**

```python
# tests/test_lora_config.py
import pytest
pytest.importorskip("peft")  # 无 peft 时跳过(纯逻辑校验,A100 环境会装)
from src.train.train_one_gen import build_lora_config

def test_lora_config_defaults():
    cfg = build_lora_config()
    assert cfg.r == 16
    assert cfg.lora_alpha == 32
    assert "q_proj" in cfg.target_modules
    assert "v_proj" in cfg.target_modules
    assert cfg.task_type == "CAUSAL_LM"
```

- [ ] **Step 2: 跑测试确认失败**

Run: `pytest tests/test_lora_config.py -v`
Expected: FAIL — ImportError: cannot import name 'build_lora_config'(或 peft 缺失则 skip)

- [ ] **Step 3: 在 `src/train/train_one_gen.py` 顶部函数区新增**

```python
def build_lora_config():
    """Mistral 注意力投影上的 LoRA 配置。"""
    from peft import LoraConfig
    return LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none", task_type="CAUSAL_LM",
    )
```

- [ ] **Step 4: 修改 `finetune` 签名与模型包装**

在 `finetune(...)` 的关键字参数中加入 `use_lora: bool = False`。在 `model = AutoModelForCausalLM.from_pretrained(...).to(device)` 之后插入:

```python
    if use_lora:
        from peft import get_peft_model
        model = get_peft_model(model, build_lora_config())
        model.print_trainable_parameters()
```

在 `trainer.train()` 之后、`model.save_pretrained(output_dir)` 之前插入(把 LoRA 合并回基座,保证下一代能直接 `from_pretrained` 加载完整模型):

```python
    if use_lora:
        model = model.merge_and_unload()
```

- [ ] **Step 5: 跑测试确认通过**

Run: `pytest tests/test_lora_config.py -v`
Expected: PASS(或在无 peft 机器上 skipped)

- [ ] **Step 6: Commit**

```bash
git add src/train/train_one_gen.py tests/test_lora_config.py
git commit -m "feat: add LoRA continued-training path to finetune (merge before save)"
```

---

## Task 4: 开源混合多代续训 runner `run_chain_osmix.py`

**Files:**
- Create: `src/train/run_chain_osmix.py`
- 依赖:Task 2 的 `p_syn_schedule`、Task 3 的 `finetune(use_lora=True)`、现有 `eval/*` 与 `src.utils.mix_data`

- [ ] **Step 1: 实现 runner(完整代码)**

```python
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
import json, sys, csv, argparse
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

        with Timer(f"[{exp_id}] gen{gen} MAUVE"):
            mauve = compute_mauve_score(mauve_ref, samp[:2000])
        clear_gpu_memory()
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
```

- [ ] **Step 2: 冒烟导入检查(无 GPU 也能跑)**

Run: `python -c "import ast; ast.parse(open('src/train/run_chain_osmix.py').read()); print('syntax ok')"`
Expected: `syntax ok`

- [ ] **Step 3: Commit**

```bash
git add src/train/run_chain_osmix.py
git commit -m "feat: add open-source-mixing multi-gen continued-training runner"
```

---

## Task 5: D1/D2 网格 CSV

**Files:**
- Create: `src/configs/experiment_grid_osmix.csv`
- Test: `tests/test_grid_osmix.py`

- [ ] **Step 1: 写失败测试**

```python
# tests/test_grid_osmix.py
import csv, pathlib

GRID = pathlib.Path(__file__).resolve().parents[1] / "src/configs/experiment_grid_osmix.csv"

def test_grid_schema_and_values():
    rows = list(csv.DictReader(open(GRID)))
    assert rows, "grid is empty"
    need = {"exp_id","group","model","real_dataset","ai_dataset","mode","p_syn","n_train","k_max","seed"}
    assert need.issubset(rows[0].keys()), f"missing columns: {need - set(rows[0].keys())}"
    for r in rows:
        assert r["mode"] in {"constant","ramp"}, r["mode"]
        assert 0.0 <= float(r["p_syn"]) <= 1.0
        assert int(r["k_max"]) >= 1
    ids = [r["exp_id"] for r in rows]
    assert len(ids) == len(set(ids)), "duplicate exp_id"

def test_has_control_and_ramp():
    rows = list(csv.DictReader(open(GRID)))
    assert any(r["mode"]=="constant" and float(r["p_syn"])==0.0 for r in rows), "missing p_syn=0 control (D1)"
    assert any(r["mode"]=="ramp" for r in rows), "missing D2 ramp chain"
```

- [ ] **Step 2: 跑测试确认失败**

Run: `pytest tests/test_grid_osmix.py -v`
Expected: FAIL — FileNotFoundError(网格不存在)

- [ ] **Step 3: 创建 `src/configs/experiment_grid_osmix.csv`**

```csv
exp_id,group,model,real_dataset,ai_dataset,mode,p_syn,n_train,k_max,seed,notes
osmix_d1_p00_s42,osmix_d1,mistralai/Mistral-7B-v0.1,owt,cosmopedia,constant,0.0,5000,10,42,D1 control pure-real
osmix_d1_p10_s42,osmix_d1,mistralai/Mistral-7B-v0.1,owt,cosmopedia,constant,0.1,5000,10,42,D1 10%
osmix_d1_p25_s42,osmix_d1,mistralai/Mistral-7B-v0.1,owt,cosmopedia,constant,0.25,5000,10,42,D1 25%
osmix_d1_p50_s42,osmix_d1,mistralai/Mistral-7B-v0.1,owt,cosmopedia,constant,0.5,5000,10,42,D1 50%
osmix_d1_p75_s42,osmix_d1,mistralai/Mistral-7B-v0.1,owt,cosmopedia,constant,0.75,5000,10,42,D1 75%
osmix_d1_p100_s42,osmix_d1,mistralai/Mistral-7B-v0.1,owt,cosmopedia,constant,1.0,5000,10,42,D1 100%
osmix_d2_ramp_s42,osmix_d2,mistralai/Mistral-7B-v0.1,owt,cosmopedia,ramp,1.0,5000,10,42,D2 ramp 0->100%
osmix_d1_p50_s123,osmix_d1,mistralai/Mistral-7B-v0.1,owt,cosmopedia,constant,0.5,5000,10,123,D1 50% seed123
osmix_d1_p50_s456,osmix_d1,mistralai/Mistral-7B-v0.1,owt,cosmopedia,constant,0.5,5000,10,456,D1 50% seed456
osmix_d2_ramp_s123,osmix_d2,mistralai/Mistral-7B-v0.1,owt,cosmopedia,ramp,1.0,5000,10,123,D2 ramp seed123
osmix_d2_ramp_s456,osmix_d2,mistralai/Mistral-7B-v0.1,owt,cosmopedia,ramp,1.0,5000,10,456,D2 ramp seed456
```

> 备注:这是骨架网格(seed42 全比例 + D2,再补 50%/D2 的 123/456 上 CI)。跑通后可按需把 0.1/0.25/0.75 也补满 3 seed。

- [ ] **Step 4: 跑测试确认通过**

Run: `pytest tests/test_grid_osmix.py -v`
Expected: PASS(2 passed)

- [ ] **Step 5: Commit**

```bash
git add src/configs/experiment_grid_osmix.csv tests/test_grid_osmix.py
git commit -m "feat: add D1/D2 open-source-mixing experiment grid"
```

---

## Task 6: A100 冒烟集成验证(需 GPU)

**Files:** 无新增;验证 Task 1-5 的端到端串联。

- [ ] **Step 1: 准备数据(若未就绪)**

```bash
export HF_ENDPOINT=https://hf-mirror.com
python src/setup/prepare_data.py                       # OWT -> data/real_texts.json, data/train_texts.json
python src/setup/prepare_data_synthetic.py --dataset cosmopedia  # -> data/syn_cosmopedia_texts.json
```
Expected: 三个 json 文件存在于 `data/`。

- [ ] **Step 2: 临时 2 代冒烟(小样本)**

新建临时网格行或直接命令:把 `n_train` 调小到 200、`k_max=2` 跑 `osmix_d1_p50_s42`(可临时复制一行改 n_train/k_max,exp_id 加 `_smoke`)。

Run:
```bash
python src/train/run_chain_osmix.py --exp-id osmix_d1_p50_s42_smoke \
  --grid src/configs/experiment_grid_osmix_smoke.csv --results-base results/_smoke
```
Expected: `results/_smoke/osmix_d1_p50_s42_smoke/metrics.jsonl` 出现 3 行(gen 0/1/2),每行 `ppl_real` 为有限正数,`gen0` 的 `p_syn_gen==0.0`。

- [ ] **Step 3: 校验冒烟结果**

```bash
python - <<'PY'
import json
rows=[json.loads(l) for l in open("results/_smoke/osmix_d1_p50_s42_smoke/metrics.jsonl")]
assert [r["gen"] for r in rows]==[0,1,2]
assert rows[0]["p_syn_gen"]==0.0 and rows[1]["p_syn_gen"]==0.5
assert all(r["ppl_real"]>0 and r["ppl_real"]<1e9 for r in rows)
print("smoke OK", [round(r["ppl_real"],1) for r in rows])
PY
```
Expected: `smoke OK [...]`

- [ ] **Step 4: 清理冒烟产物 + Commit smoke grid**

```bash
rm -rf results/_smoke
git add src/configs/experiment_grid_osmix_smoke.csv
git commit -m "test: add smoke grid for osmix runner integration check"
```

---

## Task 7: 文档对齐(整理实验文档)

**Files:**
- Modify: `CLAUDE.md`(把 `experiments/` → `src/`;新增 osmix runner 说明;注明早期 toy/GPT-2 实验已不再继续)
- Modify: `README.md`(新增"当前实验方向"段,指向 spec 与本 plan)

- [ ] **Step 1: 更新 `CLAUDE.md`**

把所有 `experiments/` 路径改为 `src/`;在命令区追加:

```markdown
# 开源混合多代续训(当前主线)
python src/train/run_chain_osmix.py --exp-id osmix_d1_p50_s42 --grid src/configs/experiment_grid_osmix.csv
```
并在 Overview 注明:早期 exp0/exp1(toy/线性)与 GPT-2 路线**已归档,不再继续**;当前聚焦 Mistral-7B 开源混合。

- [ ] **Step 2: 更新 `README.md`**

在"项目结构"前插入一段"当前实验方向",链接:
`docs/superpowers/specs/2026-06-08-mistral-opensource-mixing-collapse-design.md` 与
`docs/superpowers/plans/2026-06-08-mistral-osmix-experiment.md`。

- [ ] **Step 3: Commit**

```bash
git add CLAUDE.md README.md
git commit -m "docs: align CLAUDE.md/README with src/ layout and osmix direction"
```

---

## Spec 覆盖自检

- 范式(多代续训、开源混合、不回收自产输出)→ Task 4 ✔
- 不回收自产输出(生成仅评估)→ Task 4 生成步注释 + 不进训练集 ✔
- D1 固定 / D2 递增 → Task 2 调度 + Task 5 网格 ✔
- p_syn=0 控制组 → Task 5 `osmix_d1_p00_s42` + `test_has_control_and_ramp` ✔
- Mistral-7B + LoRA → Task 3 ✔
- 损失为主(PPL/CE)+ MAUVE + 重复率 → Task 4 metrics ✔
- 修坏 import 前置 → Task 1 ✔
- 数据来源换成固定 AI 开源语料 → Task 4 用 `syn_cosmopedia` ✔
- 暂缓 IQD → 本 plan 不涉及 IQD 定义(留待结果后另开 spec)✔
- 多 seed 上 CI → Task 5 含 123/456 行 ✔
```
