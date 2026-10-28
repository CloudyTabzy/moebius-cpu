"""Mask generator contract for the M8 eval set (run: pytest tools/eval/tests)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from prepare_eval_set import make_mask  # noqa: E402


def test_deterministic_for_seed():
    for seed in (0, 1, 17, 199):
        assert np.array_equal(make_mask(seed), make_mask(seed))


def test_different_seeds_differ():
    assert not np.array_equal(make_mask(0), make_mask(1))


def test_coverage_in_range_for_seeds_0_to_199():
    for seed in range(200):
        cov = (make_mask(seed) > 0).mean()
        assert 0.05 <= cov <= 0.25, (seed, cov)


def test_format_uint8_binary_512():
    for seed in range(10):
        m = make_mask(seed)
        assert m.shape == (512, 512)
        assert m.dtype == np.uint8
        assert set(np.unique(m)) <= {0, 255}
