"""Numerical checks of the asymmetric-update rest point (docs/theory.md section 3).

The rest point of loss-weighted REINFORCE was derived on paper (closed form in
continuous time, ``asymmetric_continuous_survival``) and solved on the grid as the
equilibrium of a modified game (``asymmetric_fixed_point``). Both rest on the same
argument: the expected update vanishes where every quitting time earns the same
weighted advantage. This module checks that argument by running the learning rule
itself, with no reference to the modified game:

* ``mean_field``: the exact expected update of a softmax policy under the rule,
  iterated until it stops moving (no sampling noise);
* ``monte_carlo``: the rule as the agents run it, with sampled matches, coin-flip
  ties, per-sample win/loss weights and a running-mean baseline.

If the derivation is right, both end where the fixed point says.
"""

from __future__ import annotations

import numpy as np

from .equilibrium import asymmetric_weights, weighted_advantage_matrix
from .war_of_attrition import GameA


def _softmax(z: np.ndarray) -> np.ndarray:
    e = np.exp(z - z.max())
    return e / e.sum()


def mean_field(
    game: GameA,
    ratio: float,
    lr: float = 2.0,
    baseline_rate: float = 0.05,
    max_steps: int = 400_000,
    tol: float = 1e-10,
) -> tuple[np.ndarray, float, int]:
    """Expected-update dynamics of loss-weighted REINFORCE on tabular softmax logits.

    Per step: ``theta_k += lr * p_k * (w_k - p.w)`` with ``w = E_{t'~p}[alpha(o)(u - b)]``
    for each own time ``k``, and the baseline tracks the mean true payoff,
    ``b += baseline_rate * (p'Up - b)``. Ties: coin flip for the prize, so a tie is
    half a weighted win and half a weighted loss, as in the trainer. Returns
    ``(p, b, steps)``; stops once the largest logit change and the baseline change are
    both below ``tol``.
    """
    a_win, a_loss = asymmetric_weights(ratio)
    n = game.grid.n_bins
    t = game.grid.times
    cost = -game.cost_rate * np.minimum(t[:, None], t[None, :])
    idx = np.arange(n)
    win = idx[:, None] > idx[None, :]
    loss = idx[:, None] < idx[None, :]
    tie = ~(win | loss)
    # weighted advantage = A0 - b * D (linear in the baseline)
    a0 = np.where(win, a_win * (game.prize + cost), np.where(loss, a_loss * cost, 0.0))
    a0 = a0 + tie * 0.5 * (a_win * (game.prize + cost) + a_loss * cost)
    d = np.where(win, a_win, np.where(loss, a_loss, 0.5 * (a_win + a_loss)))
    theta = np.zeros(n)
    b = 0.0
    for step in range(1, max_steps + 1):
        p = _softmax(theta)
        w = (a0 - b * d) @ p
        g = p * (w - p @ w)
        db = baseline_rate * (p @ game.payoff @ p - b)
        theta += lr * g
        b += db
        if np.max(np.abs(lr * g)) < tol and abs(db) < tol:
            break
    return _softmax(theta), float(b), step


def natural_gradient_field(game: GameA, ratio: float, lr: float = 0.2, steps: int = 200_000, baseline_rate: float = 0.05) -> tuple[np.ndarray, float]:
    """Expected update preconditioned by the inverse softmax Fisher: ``theta += lr (w - p.w)``.

    Same rest points as the plain expected update (every used quitting time earns the
    same weighted advantage), but unused times are removed exponentially fast instead
    of like 1/steps, so it reaches the rest point to machine precision.
    """
    theta, b = np.zeros(game.grid.n_bins), 0.0
    for _ in range(steps):
        p = _softmax(theta)
        w = weighted_advantage_matrix(game, ratio, b) @ p
        theta += lr * (w - p @ w)
        b += baseline_rate * (p @ game.payoff @ p - b)
    return _softmax(theta), float(b)


def monte_carlo(
    game: GameA,
    ratio: float,
    seed: int,
    steps: int = 20_000,
    batch: int = 4096,
    lr: float = 0.5,
    lr_final_fraction: float = 0.02,
    baseline_rate: float = 0.05,
    average_last: float = 0.25,
) -> tuple[np.ndarray, float]:
    """Sampled loss-weighted REINFORCE in self-play, tabular softmax, plain SGD.

    Each step draws ``batch`` matches of the policy against itself, settles ties by a
    coin flip, weights each sample's ``(u - b)`` by ``alpha_win`` or ``alpha_loss`` and
    takes one step on the logits. The learning rate decays linearly to
    ``lr * lr_final_fraction``. Returns the policy averaged over the last
    ``average_last`` of training and the final baseline.
    """
    rng = np.random.default_rng(seed)
    a_win, a_loss = asymmetric_weights(ratio)
    n = game.grid.n_bins
    t = game.grid.times
    theta = np.zeros(n)
    b = 0.0
    avg = np.zeros(n)
    n_avg = 0
    start_avg = int(steps * (1 - average_last))
    for step in range(steps):
        p = _softmax(theta)
        own = rng.choice(n, size=batch, p=p)
        opp = rng.choice(n, size=batch, p=p)
        coin = rng.random(batch) < 0.5
        won = (own > opp) | ((own == opp) & coin)
        paid = game.cost_rate * t[np.minimum(own, opp)]
        u = np.where(won, game.prize, 0.0) - paid
        weight = np.where(won, a_win, a_loss)
        adv = weight * (u - b)
        grad = np.bincount(own, weights=adv, minlength=n) / batch - p * adv.mean()
        frac = step / max(1, steps - 1)
        theta += lr * (1 - (1 - lr_final_fraction) * frac) * grad
        b += baseline_rate * (u.mean() - b)
        if step >= start_avg:
            avg += p
            n_avg += 1
    return avg / n_avg, float(b)


def discrete_hazard(p: np.ndarray, delta: float) -> np.ndarray:
    """Hazard of a grid law in continuous-time units: ``-ln(1 - p_k / S_k) / delta``,
    with ``S_k = P(T >= t_k)``. For the floored exponential this is exactly the rate."""
    surv = np.cumsum(p[::-1])[::-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        q = np.where(surv > 0, p / surv, np.nan)
        return -np.log1p(-np.minimum(q, 1 - 1e-15)) / delta


def continuous_hazard(t: np.ndarray, prize: float, cost_rate: float, ratio: float) -> np.ndarray:
    """Closed-form hazard ``rho c / (V + (rho - 1)(c t + b))``, ``b = V (rho-1)/(4 rho)``."""
    b = prize * (ratio - 1) / (4 * ratio)
    return ratio * cost_rate / (prize + (ratio - 1) * (cost_rate * np.asarray(t, float) + b))
