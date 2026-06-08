"""
MAUVE 计算模块

δₖ = 1 - MAUVE(Gen_k 输出, D_real)
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils import clear_gpu_memory


def compute_mauve_score(ref_texts: list, gen_texts: list,
                        device_id: int = 0, max_len: int = 128,
                        featurize_model_name: str = None) -> float:
    """
    ref_texts : 真实文本列表（分布锚点）
    gen_texts : 模型生成文本列表
    featurize_model_name : MAUVE 特征模型。默认读环境变量 MAUVE_FEATURIZE_MODEL，
                           否则用 gpt2-large。可指向本地 ModelScope 下载目录以离线运行。
    返回 MAUVE ∈ [0, 1]，越高越好
    """
    import mauve as mauve_lib
    feat = featurize_model_name or os.environ.get("MAUVE_FEATURIZE_MODEL") or "gpt2-large"
    result = mauve_lib.compute_mauve(
        p_text=ref_texts,
        q_text=gen_texts,
        device_id=device_id,
        max_text_length=max_len,
        featurize_model_name=feat,
        verbose=False,
    )
    clear_gpu_memory()
    return float(result.mauve)


def delta_k(mauve_score: float) -> float:
    """δₖ = 1 - MAUVE  (0 = 完美对齐，1 = 完全崩溃)"""
    return 1.0 - mauve_score
