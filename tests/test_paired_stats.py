"""Paired seed-by-seed comparison (src.metrics.stats.compare_paired, analysis.paired_by_seed)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from src.analysis import paired_by_seed
from src.metrics.stats import compare_paired


def test_interval_is_the_paired_t_interval():
    rng = np.random.default_rng(0)
    x = rng.normal(1, 1, 12)
    y = x - 0.3 + rng.normal(0, 0.2, 12)
    r = compare_paired(x, y, 0.95, rng)
    ci = stats.ttest_rel(x, y).confidence_interval(0.95)
    assert r.lo == pytest.approx(ci.low) and r.hi == pytest.approx(ci.high)
    assert r.diff == pytest.approx((x - y).mean())


def test_exact_sign_flip_floor_and_null():
    rng = np.random.default_rng(1)
    d = np.linspace(1, 2, 10)  # all positive: only the observed pattern and its mirror reach it
    r = compare_paired(d, np.zeros(10), 0.95, rng)
    assert r.p == pytest.approx(2 / 2**10)
    same = compare_paired(np.arange(10.0), np.arange(10.0), 0.95, rng)
    assert same.diff == 0 and same.p == 1.0


def test_monte_carlo_above_16_pairs_is_close_to_t():
    rng = np.random.default_rng(2)
    x = rng.normal(0, 1, 20)
    y = x + rng.normal(0.4, 1, 20)
    r = compare_paired(x, y, 0.95, rng)
    assert r.n == 20 and 0 <= r.p <= 1
    assert abs(r.p - stats.ttest_rel(x, y).pvalue) < 0.05


def test_paired_by_seed_matches_on_seed_not_row_order():
    df = pd.DataFrame({
        "arm": ["a"] * 4 + ["b"] * 4,
        "seed": [0, 1, 2, 3, 3, 2, 1, 0],
        "m": [1.0, 2.0, 3.0, 4.0, 3.5, 2.5, 1.5, 0.5],
    })
    r = paired_by_seed(df, "a", "b", "m", "exp")
    assert r["n_pairs"] == 4
    assert r["diff_lo_paired"] == pytest.approx(0.5) and r["diff_hi_paired"] == pytest.approx(0.5)


def test_paired_by_seed_refuses_repeated_seeds():
    df = pd.DataFrame({"arm": ["a", "a", "b", "b"], "seed": [0, 0, 0, 1], "m": [1.0, 2.0, 3.0, 4.0]})
    assert paired_by_seed(df, "a", "b", "m", "exp")["n_pairs"] == 0
