"""Bootstrap confidence intervals, effect sizes and multiple-comparison control."""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np
from scipy import stats as sps


@dataclass
class Interval:
    mean: float
    lo: float
    hi: float
    n: int


def bootstrap_mean_ci(
    x: np.ndarray, n_boot: int, confidence: float, rng: np.random.Generator
) -> Interval:
    """Percentile bootstrap CI of the mean."""
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return Interval(float("nan"), float("nan"), float("nan"), 0)
    if len(x) == 1:
        return Interval(float(x[0]), float("nan"), float("nan"), 1)
    means = x[rng.integers(0, len(x), (n_boot, len(x)))].mean(axis=1)
    a = (1 - confidence) / 2
    return Interval(float(x.mean()), float(np.quantile(means, a)), float(np.quantile(means, 1 - a)), len(x))


def bootstrap_curve_band(
    curves: np.ndarray, n_boot: int, confidence: float, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mean and pointwise bootstrap CI across seeds for ``(n_seeds, n_points)`` curves."""
    n = curves.shape[0]
    idx = rng.integers(0, n, (n_boot, n))
    means = curves[idx].mean(axis=1)
    a = (1 - confidence) / 2
    return curves.mean(axis=0), np.quantile(means, a, axis=0), np.quantile(means, 1 - a, axis=0)


def hedges_g(x: np.ndarray, y: np.ndarray) -> float:
    """Bias-corrected standardised mean difference ``(mean x - mean y) / s_pooled``."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    nx, ny = len(x), len(y)
    if nx < 2 or ny < 2:
        return float("nan")
    s = np.sqrt(((nx - 1) * x.var(ddof=1) + (ny - 1) * y.var(ddof=1)) / (nx + ny - 2))
    if s == 0:
        return 0.0 if x.mean() == y.mean() else float("inf") * np.sign(x.mean() - y.mean())
    j = 1 - 3 / (4 * (nx + ny) - 9)
    return float(j * (x.mean() - y.mean()) / s)


@dataclass
class Difference:
    diff: float
    lo: float
    hi: float
    hedges_g: float
    p_perm: float
    n_x: int
    n_y: int

    @property
    def distinguishable(self) -> bool:
        """Whether the bootstrap CI of the difference excludes zero."""
        return bool(np.isfinite(self.lo) and (self.lo > 0 or self.hi < 0))


def compare(
    x: np.ndarray, y: np.ndarray, n_boot: int, confidence: float, rng: np.random.Generator
) -> Difference:
    """Bootstrap CI of ``mean(x) - mean(y)`` (independent samples), Hedges' g, and a
    two-sided permutation p-value."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    x, y = x[np.isfinite(x)], y[np.isfinite(y)]
    d = x.mean() - y.mean()
    bx = x[rng.integers(0, len(x), (n_boot, len(x)))].mean(axis=1)
    by = y[rng.integers(0, len(y), (n_boot, len(y)))].mean(axis=1)
    a = (1 - confidence) / 2
    lo, hi = np.quantile(bx - by, [a, 1 - a])
    pooled = np.concatenate([x, y])
    perm = np.argsort(rng.random((n_boot, len(pooled))), axis=1)
    px = pooled[perm[:, : len(x)]].mean(axis=1) - pooled[perm[:, len(x) :]].mean(axis=1)
    p = (np.sum(np.abs(px) >= abs(d) - 1e-15) + 1) / (n_boot + 1)
    return Difference(float(d), float(lo), float(hi), hedges_g(x, y), float(p), len(x), len(y))


@dataclass
class PairedDifference:
    diff: float
    lo: float
    hi: float
    dz: float
    p: float
    n: int
    corr: float

    @property
    def distinguishable(self) -> bool:
        return bool(np.isfinite(self.lo) and (self.lo > 0 or self.hi < 0))


def compare_paired(
    x: np.ndarray, y: np.ndarray, confidence: float, rng: np.random.Generator, n_perm: int = 20000
) -> PairedDifference:
    """Paired comparison of two arms run on the same seeds (``x[i]`` and ``y[i]`` share seed i).

    Interval: Student t on the per-seed differences (``t_{n-1}``). At 10-20 seeds this
    is wider than a percentile bootstrap, which tends to understate the uncertainty
    there. p: two-sided sign-flip permutation test on the differences, exact (all
    ``2^n`` sign patterns) up to n = 16, Monte Carlo with ``n_perm`` draws above.
    ``dz`` is the mean of the differences over their standard deviation, ``corr`` the
    across-seed correlation of the two arms (how much the pairing helps)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    d = x[ok] - y[ok]
    n = len(d)
    nan = float("nan")
    if n < 2:
        return PairedDifference(float(d.mean()) if n else nan, nan, nan, nan, nan, n, nan)
    mean, sd = float(d.mean()), float(d.std(ddof=1))
    if sd == 0:
        # every seed gives the same difference: no spread to test against
        lo = hi = mean
        p = 1.0 if mean == 0 else 2.0 ** (1 - n)
        dz = 0.0 if mean == 0 else float("inf") * np.sign(mean)
    else:
        half = float(sps.t.ppf(0.5 + confidence / 2, n - 1)) * sd / np.sqrt(n)
        lo, hi = mean - half, mean + half
        if n <= 16:
            signs = np.array(list(itertools.product((1.0, -1.0), repeat=n)))
        else:
            signs = rng.choice((1.0, -1.0), size=(n_perm, n))
            signs[0] = 1.0  # include the observed pattern
        null = np.abs((signs * d).mean(axis=1))
        p = float(np.mean(null >= abs(mean) - 1e-12))
        dz = mean / sd
    sx, sy = x[ok].std(), y[ok].std()
    corr = float(np.corrcoef(x[ok], y[ok])[0, 1]) if sx > 0 and sy > 0 else nan
    return PairedDifference(mean, float(lo), float(hi), float(dz), p, n, corr)


def holm(p_values: list[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values (same order as the input)."""
    p = np.asarray(p_values, float)
    order = np.argsort(p)
    adjusted = np.empty_like(p)
    running = 0.0
    m = len(p)
    for rank, idx in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p[idx]))
        adjusted[idx] = running
    return adjusted.tolist()


def benjamini_hochberg(p_values: list[float]) -> list[float]:
    """Benjamini-Hochberg adjusted p-values (false discovery rate; same order as the input)."""
    p = np.asarray(p_values, float)
    m = len(p)
    order = np.argsort(p)
    scaled = p[order] * m / np.arange(1, m + 1)
    adjusted = np.empty_like(p)
    adjusted[order] = np.minimum(1.0, np.minimum.accumulate(scaled[::-1])[::-1])
    return adjusted.tolist()
