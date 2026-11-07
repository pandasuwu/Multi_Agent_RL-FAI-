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

All quantities are per match (both players together) and use the symmetric strategy
``p``; for a learned population ``p`` is the population mixture.
"""

from __future__ import annotations

import numpy as np

from ..games import GameA, GameB


def _min_matrix(times: np.ndarray) -> np.ndarray:
    return np.minimum(times[:, None], times[None, :])


def welfare_a(game: GameA, p: np.ndarray) -> dict[str, float]:
    """Welfare of the symmetric strategy ``p`` (shape ``(n_bins,)``) in Game A."""
    p = np.asarray(p, dtype=np.float64).reshape(-1)
    e_min = float(p @ _min_matrix(game.grid.times) @ p)
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


def welfare_b(game: GameB, p: np.ndarray) -> dict[str, float]:
    """Welfare of the symmetric type-contingent strategy ``p`` (``(n_types, n_bins)``) in Game B."""
    p = np.asarray(p, dtype=np.float64)
    w, v = game.type_weights, game.values
    win = p @ game.win @ p.T                      # win[k, l] = Pr(type k outlasts type l), ties 1/2
    pair = w[:, None] * w[None, :]
    allocated = float(np.sum(pair * (v[:, None] * win + v[None, :] * win.T)))
    e_min = float(np.sum(pair * (p @ _min_matrix(game.grid.times) @ p.T)))
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


def welfare(game: GameA | GameB, p: np.ndarray) -> dict[str, float]:
    return welfare_a(game, p) if isinstance(game, GameA) else welfare_b(game, p)
