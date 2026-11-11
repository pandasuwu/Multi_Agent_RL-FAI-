"""Social welfare of a war-of-attrition strategy, against the social optimum.

Both players pay ``c * min(T1, T2)``, so the fight itself is pure waste: the social
optimum is no fight at all, with the prize going to whoever values it most.

Game A (common prize V): optimum welfare V. Total welfare of a symmetric strategy
``p`` is ``V - 2c E[min(T1, T2)]``. At the mixed equilibrium each player expects 0,
so the whole prize is dissipated.

Game B (private values): optimum welfare ``E[max(v1, v2)]`` (highest value takes the
prize, nobody fights). Welfare splits into the value actually allocated,
``E[v_winner]``, minus the fight cost ``2c E[min]``. The equilibrium bid
``b(v) = v^2 / 2`` allocates efficiently (the higher value waits longer) and pays for
it with the fight.

All quantities are per match (both players together). ``welfare_a`` / ``welfare_b``
take one strategy played by both sides, which is right for a single shared policy
(exact mode, the oracle). A population of ``M`` independent agents needs
:func:`population_welfare`: an agent never meets itself, so the match is between two
*different* agents and the average runs over those pairs only. Feeding the population
mixture to ``welfare_a`` instead counts every agent against itself in ``1/M`` of the
matches, which overstates the fight when agents differ (a pair that has split into
"quit at once" and "hold on" hardly fights, but its mixture against itself does).
"""

from __future__ import annotations

import numpy as np

from ..games import GameA, GameB


def _min_matrix(times: np.ndarray) -> np.ndarray:
    return np.minimum(times[:, None], times[None, :])


def welfare_a(game: GameA, p: np.ndarray, q: np.ndarray | None = None) -> dict[str, float]:
    """Welfare of a Game A match between strategies ``p`` and ``q`` (shape ``(n_bins,)``).

    ``q`` defaults to ``p``: one strategy played by both sides."""
    p = np.asarray(p, dtype=np.float64).reshape(-1)
    q = p if q is None else np.asarray(q, dtype=np.float64).reshape(-1)
    e_min = float(p @ _min_matrix(game.grid.times) @ q)
    optimum = game.prize
    welfare = game.prize - 2 * game.cost_rate * e_min
    return {
        "optimum": optimum,
        "welfare": welfare,
        "fight_cost": 2 * game.cost_rate * e_min,
        "allocated_value": game.prize,
        "dissipation": 1 - welfare / optimum,   # share of the achievable welfare lost
        "mean_fight": e_min,
    }


def welfare_b(game: GameB, p: np.ndarray, q: np.ndarray | None = None) -> dict[str, float]:
    """Welfare of a Game B match between type-contingent strategies ``p`` and ``q``
    (each ``(n_types, n_bins)``). ``q`` defaults to ``p``."""
    p = np.asarray(p, dtype=np.float64)
    q = p if q is None else np.asarray(q, dtype=np.float64)
    w, v = game.type_weights, game.values
    win = p @ game.win @ q.T                      # win[k, l] = Pr(p's type k outlasts q's type l), ties 1/2
    pair = w[:, None] * w[None, :]
    allocated = float(np.sum(pair * (v[:, None] * win + v[None, :] * (1.0 - win))))
    e_min = float(np.sum(pair * (p @ _min_matrix(game.grid.times) @ q.T)))
    optimum = float(np.sum(pair * np.maximum(v[:, None], v[None, :])))
    welfare = allocated - 2 * game.cost_rate * e_min
    return {
        "optimum": optimum,
        "welfare": welfare,
        "fight_cost": 2 * game.cost_rate * e_min,
        "allocated_value": allocated,
        "dissipation": 1 - welfare / optimum,
        "mean_fight": e_min,
    }


def welfare(game: GameA | GameB, p: np.ndarray, q: np.ndarray | None = None) -> dict[str, float]:
    return welfare_a(game, p, q) if isinstance(game, GameA) else welfare_b(game, p, q)


def population_welfare(game: GameA | GameB, probs: np.ndarray) -> dict[str, float]:
    """Welfare per match of a randomly re-matched population.

    ``probs`` is ``(M, K, n_bins)``, one policy per agent (``K = 1`` in Game A). A
    match is between two different agents drawn uniformly, so every quantity is the
    mean over the ``M (M - 1) / 2`` pairs. ``M = 1`` is one shared policy in self-play
    and reduces to :func:`welfare`.
    """
    probs = np.asarray(probs, dtype=np.float64)
    if isinstance(game, GameA):
        probs = probs[:, 0, :]
    m = probs.shape[0]
    if m == 1:
        return welfare(game, probs[0])
    pairs = [welfare(game, probs[i], probs[j]) for i in range(m) for j in range(i + 1, m)]
    return {key: float(np.mean([w[key] for w in pairs])) for key in pairs[0]}
