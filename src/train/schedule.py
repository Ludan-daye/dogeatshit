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
