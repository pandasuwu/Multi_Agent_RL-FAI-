"""Analytic and discrete equilibria of the war of attrition (derivations in docs/theory.md)."""

from __future__ import annotations

import numpy as np
from scipy import integrate

from .war_of_attrition import GameA, GameB, TimeGrid


# --------------------------------------------------------------------------- Game A
def exponential_survival(t: np.ndarray | float, prize: float, cost_rate: float) -> np.ndarray:
    """Symmetric mixed NE of Game A: ``P(T > t) = exp(-c t / V)``."""
    return np.exp(-cost_rate * np.asarray(t, dtype=np.float64) / prize)


def discretise_survival(survival_at_grid: np.ndarray) -> np.ndarray:
    """Map a continuous quitting-time law onto the grid by flooring to the grid.

    A player whose continuous quitting time is ``T`` quits at the last grid time
    ``<= T``: grid point ``t_k`` receives ``P(t_k <= T < t_{k+1})`` and the last
    point ``t_max`` receives the whole tail ``P(T >= t_max)``. ``survival_at_grid``
    is ``P(T > t_k)`` evaluated at every grid point (its first entry is 1).

    Flooring is the right discretisation: the exact equilibrium of the discrete
    game coincides with the floored exponential to O(delta^2) (docs/theory.md);
    nearest-point binning leaves an exploitability floor ~20x larger.
    """
    s = np.asarray(survival_at_grid, dtype=np.float64)
    p = s - np.append(s[1:], 0.0)
    return p / p.sum()


def game_a_analytic_policy(game: GameA) -> np.ndarray:
    """The oracle: the analytic exponential equilibrium discretised onto the grid."""
    return discretise_survival(exponential_survival(game.grid.times, game.prize, game.cost_rate))


def solve_symmetric_equilibrium(
    payoff: np.ndarray, tol: float = 1e-12, max_iter: int = 500
) -> np.ndarray:
    """Symmetric Nash equilibrium ``p`` of the symmetric game with row payoff ``payoff``.

    Support refinement: solve the indifference system ``(payoff p)_i = w`` on the
    current support, drop strategies with negative probability, add strategies that
    beat ``w``, and repeat. Terminates with an exact equilibrium (up to ``tol``) or
    raises. Used only on small grids (tens of bins), where this is instant.
    """
    n = payoff.shape[0]
    support = np.ones(n, dtype=bool)
    for _ in range(max_iter):
        idx = np.flatnonzero(support)
        m = len(idx)
        a = np.zeros((m + 1, m + 1))
        a[:m, :m] = payoff[np.ix_(idx, idx)]
        a[:m, m] = -1.0
        a[m, :m] = 1.0
        rhs = np.zeros(m + 1)
        rhs[m] = 1.0
        sol = np.linalg.lstsq(a, rhs, rcond=None)[0]
        p_s, w = sol[:m], sol[m]
        if np.any(p_s < -tol):
            support[idx[np.argmin(p_s)]] = False
            continue
        p = np.zeros(n)
        p[idx] = np.clip(p_s, 0.0, None)
        p /= p.sum()
        gains = payoff @ p
        value = p @ gains
        outside = ~support
        if np.any(outside) and np.max(gains[outside]) > value + 1e-10:
            best = np.flatnonzero(outside)[np.argmax(gains[outside])]
            support[best] = True
            continue
        return p
    raise RuntimeError("support refinement did not converge")


# ---------------------------------------------------- asymmetric-update fixed point
def asymmetric_weights(ratio: float) -> tuple[float, float]:
    """``(alpha_win, alpha_loss)`` with ``alpha_loss / alpha_win = ratio`` and mean 1."""
    if ratio < 1:
        raise ValueError("the loss/win ratio must be >= 1")
    return 2.0 / (1.0 + ratio), 2.0 * ratio / (1.0 + ratio)


def weighted_advantage_matrix(game: GameA, ratio: float, baseline: float) -> np.ndarray:
    """Expected weighted advantage ``alpha(outcome) * (u - b)`` for each pair of times.

    A win (``i > j``) pays ``V - c t_j`` and is weighted ``alpha_win``; a loss pays
    ``-c t_i`` and is weighted ``alpha_loss``. A tie is resolved by a fair coin
    for the prize (expected payoff ``V/2 - c t``, exactly the game's tie rule), so
    its entry is the average of the win and loss entries. Giving ties a separate
    weight instead adds a positive diagonal that makes the modified game unstable
    on a grid (docs/theory.md, section 3).
    """
    a_win, a_loss = asymmetric_weights(ratio)
    n = game.grid.n_bins
    cost = -game.cost_rate * np.minimum(game.grid.times[:, None], game.grid.times[None, :])
    win = a_win * (game.prize + cost - baseline)
    loss = a_loss * (cost - baseline)
    idx = np.arange(n)
    return np.where(
        idx[:, None] > idx[None, :], win, np.where(idx[:, None] < idx[None, :], loss, 0.5 * (win + loss))
    )


def asymmetric_fixed_point(
    game: GameA, ratio: float, damping: float = 0.5, tol: float = 1e-12, max_iter: int = 2000
) -> tuple[np.ndarray, float]:
    """Stationary policy of asymmetric-update REINFORCE with a mean-reward baseline.

    The expected update vanishes where every strategy in the support earns the same
    weighted advantage, i.e. at the symmetric equilibrium of
    :func:`weighted_advantage_matrix` with the baseline ``b = p' U p`` (the mean
    true payoff, which is what a reward-tracking baseline converges to), solved
    self-consistently. Returns ``(p, b)``. ``ratio = 1`` recovers the true game.
    """
    b = 0.0
    for _ in range(max_iter):
        p = solve_symmetric_equilibrium(weighted_advantage_matrix(game, ratio, b))
        b_new = float(p @ game.payoff @ p)
        if abs(b_new - b) < tol:
            return p, b_new
        b = (1 - damping) * b + damping * b_new
    raise RuntimeError("asymmetric fixed point iteration did not converge")


def asymmetric_continuous_survival(
    t: np.ndarray, prize: float, cost_rate: float, ratio: float
) -> tuple[np.ndarray, float]:
    """Closed form of the asymmetric fixed point in continuous time (docs/theory.md).

    Hazard ``rho c / (V + (rho - 1)(c t + b))`` with ``b = V (rho - 1) / (4 rho)``,
    i.e. a Lomax (Pareto II) survival function. Returns ``(survival, b)``.
    """
    t = np.asarray(t, dtype=np.float64)
    if ratio == 1:
        return exponential_survival(t, prize, cost_rate), 0.0
    b = prize * (ratio - 1) / (4 * ratio)
    base = prize + (ratio - 1) * b
    shape = ratio / (ratio - 1)
    return ((base + (ratio - 1) * cost_rate * t) / base) ** (-shape), b


# --------------------------------------------------------------------------- Game B
def game_b_bid(v: np.ndarray | float, value_rate: float, cost_rate: float) -> np.ndarray:
    """Symmetric BNE of Game B for ``v ~ Exp(rate)``: ``b(v) = rate v^2 / (2 c)``.

    General form ``b(v) = (1/c) int_0^v x f(x)/(1-F(x)) dx``; the exponential hazard
    is the constant ``rate``. With ``rate = c = 1`` this is ``v^2 / 2``.
    """
    v = np.asarray(v, dtype=np.float64)
    return value_rate * v**2 / (2.0 * cost_rate)


def spread_onto_grid(grid: TimeGrid, targets: np.ndarray) -> np.ndarray:
    """Row-stochastic matrix putting each target time on its two neighbouring grid
    points, weighted so the mean equals the (clipped) target exactly."""
    x = np.clip(np.asarray(targets, dtype=np.float64), 0.0, grid.t_max) / grid.delta
    lo = np.clip(np.floor(x).astype(int), 0, grid.n_bins - 1)
    hi = np.clip(lo + 1, 0, grid.n_bins - 1)
    frac = x - lo
    out = np.zeros((len(x), grid.n_bins))
    rows = np.arange(len(x))
    np.add.at(out, (rows, lo), 1.0 - frac)
    np.add.at(out, (rows, hi), frac)
    return out


def game_b_analytic_policy(game: GameB) -> np.ndarray:
    """The oracle for Game B: ``b(v_k)`` placed on the grid for every type."""
    return spread_onto_grid(game.grid, game_b_bid(game.values, game.value_rate, game.cost_rate))


def game_b_interim_payoff_continuous(
    z: float, v: float, value_rate: float, cost_rate: float
) -> float:
    """``Pi(z, v)``: type ``v`` mimics type ``z`` while the opponent plays ``b``.

    ``Pi(z, v) = F(z) v - int_0^z c b(x) f(x) dx - (1 - F(z)) c b(z)``, computed by
    quadrature (no grid), used to verify that ``b`` is a best response to itself.
    """
    f = lambda x: value_rate * np.exp(-value_rate * x)  # noqa: E731
    big_f = 1.0 - np.exp(-value_rate * z)
    paid_when_winning, _ = integrate.quad(
        lambda x: cost_rate * game_b_bid(x, value_rate, cost_rate) * f(x), 0.0, z
    )
    paid_when_losing = (1.0 - big_f) * cost_rate * game_b_bid(z, value_rate, cost_rate)
    return float(big_f * v - paid_when_winning - paid_when_losing)
