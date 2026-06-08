import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

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
