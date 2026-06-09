import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import pytest
from src.utils import get_train_subset_nonoverlap


def test_subset_size():
    pool = list(range(100))
    s = get_train_subset_nonoverlap(pool, 0, 20, seed=42)
    assert len(s) == 20


def test_subset_disjoint_across_gens():
    pool = list(range(100))
    s0 = get_train_subset_nonoverlap(pool, 0, 20, seed=42)
    s1 = get_train_subset_nonoverlap(pool, 1, 20, seed=42)
    s2 = get_train_subset_nonoverlap(pool, 2, 20, seed=42)
    assert set(s0).isdisjoint(s1)
    assert set(s0).isdisjoint(s2)
    assert set(s1).isdisjoint(s2)


def test_subset_uses_pool():
    pool = list(range(100))
    s = get_train_subset_nonoverlap(pool, 3, 20, seed=42)
    assert set(s) <= set(pool)


def test_subset_deterministic_same_seed():
    pool = list(range(100))
    assert get_train_subset_nonoverlap(pool, 0, 20, seed=42) == get_train_subset_nonoverlap(pool, 0, 20, seed=42)


def test_pool_too_small_raises():
    with pytest.raises(ValueError):
        get_train_subset_nonoverlap(list(range(50)), 5, 20, seed=42)
