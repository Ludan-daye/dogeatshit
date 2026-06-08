import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

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
