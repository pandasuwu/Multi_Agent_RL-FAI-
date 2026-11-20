"""Discretised war of attrition: payoff engine for Games A, B and C.

Conventions (see docs/theory.md):

* Quitting times live on the grid ``t_k = k * delta``, ``k = 0..n_bins-1``,
  ``delta = t_max / (n_bins - 1)``.
* Cost accrues at rate ``c`` for both players until the first quit, so both pay
  ``c * min(t_i, t_j)``. The later quitter receives the prize; on a tie each
  receives half of it.
* Row player ``i`` quits at ``t_i``, column player at ``t_j``::

      u(t_i, t_j) = V * W[i, j] - c * min(t_i, t_j),
      W[i, j] = 1 if i > j, 1/2 if i == j, 0 otherwise.

  Game B replaces the common prize ``V`` by the row player's private value ``v``.
  Game C computes that private value from two private cues, ``v = h * f``.

Everything here is float64 numpy: this module is the ground truth that metrics and
tests are computed against.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class TimeGrid:
    """Uniform grid of quitting times ``{0, delta, ..., t_max}``."""

    t_max: float
    n_bins: int

    def __post_init__(self) -> None:
        if self.n_bins < 2:
            raise ValueError("n_bins must be at least 2")
        if self.t_max <= 0:
            raise ValueError("t_max must be positive")

    @property
    def delta(self) -> float:
        return self.t_max / (self.n_bins - 1)

    @property
    def times(self) -> np.ndarray:
        return np.linspace(0.0, self.t_max, self.n_bins)


def win_matrix(n_bins: int) -> np.ndarray:
    """``W[i, j]``: prize share of the row player (1 later, 1/2 tie, 0 earlier)."""
    idx = np.arange(n_bins)
    w = (idx[:, None] > idx[None, :]).astype(np.float64)
    w[idx, idx] = 0.5
    return w


def cost_matrix(times: np.ndarray, cost_rate: float) -> np.ndarray:
    """``C[i, j] = -c * min(t_i, t_j)``: what both players pay."""
    return -cost_rate * np.minimum(times[:, None], times[None, :])


def pairwise_payoff(
    t_self: np.ndarray, t_other: np.ndarray, value_self: np.ndarray | float, cost_rate: float
) -> np.ndarray:
    """Realised payoff of a player quitting at ``t_self`` against ``t_other``.

    Works elementwise on arrays of realised quitting times (not grid indices).
    """
    t_self = np.asarray(t_self, dtype=np.float64)
    t_other = np.asarray(t_other, dtype=np.float64)
    share = np.where(t_self > t_other, 1.0, np.where(t_self == t_other, 0.5, 0.0))
    return np.asarray(value_self) * share - cost_rate * np.minimum(t_self, t_other)


@dataclass
class GameA:
    """Complete-information symmetric war of attrition with common prize ``V``."""

    prize: float
    cost_rate: float
    grid: TimeGrid
    name: str = "A"
    payoff: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.prize <= 0 or self.cost_rate <= 0:
            raise ValueError("prize and cost_rate must be positive")
        self.payoff = self.prize * win_matrix(self.grid.n_bins) + cost_matrix(
            self.grid.times, self.cost_rate
        )

    def expected_payoffs(self, opponent: np.ndarray) -> np.ndarray:
        """Payoff of each pure quitting time against the mixed strategy ``opponent``."""
        return self.payoff @ opponent

    def expected_payoff(self, own: np.ndarray, opponent: np.ndarray) -> float:
        return float(own @ self.payoff @ opponent)


@dataclass
class GameB:
    """Incomplete-information war of attrition with private values ``v ~ Exp(rate)``.

    Values are discretised to ``n_types`` equally likely types placed at the
    quantile midpoints ``F^{-1}((k + 1/2) / n_types)``. A strategy is an
    ``(n_types, n_bins)`` row-stochastic matrix ``P[k, i] = Pr(quit at t_i | v_k)``.
    """

    value_rate: float
    cost_rate: float
    grid: TimeGrid
    n_types: int
    name: str = "B"
    values: np.ndarray = field(init=False, repr=False)
    type_weights: np.ndarray = field(init=False, repr=False)
    win: np.ndarray = field(init=False, repr=False)
    cost: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.value_rate <= 0 or self.cost_rate <= 0:
            raise ValueError("value_rate and cost_rate must be positive")
        u = (np.arange(self.n_types) + 0.5) / self.n_types
        self.values = -np.log1p(-u) / self.value_rate
        self.type_weights = np.full(self.n_types, 1.0 / self.n_types)
        self.win = win_matrix(self.grid.n_bins)
        self.cost = cost_matrix(self.grid.times, self.cost_rate)

    def marginal(self, policy: np.ndarray) -> np.ndarray:
        """Distribution of the quitting time of a player whose type is unknown."""
        return self.type_weights @ policy

    def interim_payoffs(self, opponent_marginal: np.ndarray) -> np.ndarray:
        """``U[k, i]``: payoff of type ``v_k`` quitting at ``t_i`` vs the opponent marginal."""
        win_prob = self.win @ opponent_marginal
        cost = self.cost @ opponent_marginal
        return self.values[:, None] * win_prob[None, :] + cost[None, :]

    def ex_ante_payoff(self, own: np.ndarray, opponent: np.ndarray) -> float:
        u = self.interim_payoffs(self.marginal(opponent))
        return float(self.type_weights @ np.sum(own * u, axis=1))

    # the network's input for each type: the standardised value (v - E v) / sd(v),
    # which is rate * v - 1 for v ~ Exp(rate); see training/evaluate.py
    input_dim = 1

    def features(self) -> np.ndarray:
        """``(n_types, input_dim)`` input features, one row per type."""
        return (self.value_rate * self.values - 1.0)[:, None]

    def bne_bids(self) -> np.ndarray:
        """Symmetric BNE quitting time of every type (continuous, before the grid)."""
        from .equilibrium import game_b_bid

        return game_b_bid(self.values, self.value_rate, self.cost_rate)


class GameC(GameB):
    """Game B with the value computed from two private cues: ``v = h * f``.

    Ventimiglia et al. 2026 found that a fly's give-up rate shifts with its hunger and
    with the presence of food. Here each fly privately knows its hunger
    ``h ~ Exp(hunger_rate)`` and its own reading of the food quality ``f``, drawn
    uniformly from ``food_levels``, and values the prize at ``v = h * f``. The network
    gets ``h`` and ``f`` as separate inputs, so it has to combine them multiplicatively
    before it can bid; in Game B it is handed ``v`` itself.

    Types: ``n_hunger`` quantile midpoints of ``h`` times every food level, all equally
    likely (the sampled-mode trainer draws types uniformly), ordered food-major.
    Values are independent across players, so this is a private-value war of attrition
    whose value distribution ``G`` is a mixture of exponentials, with symmetric BNE
    ``b(v) = (1/c) int_0^v x g(x) / (1 - G(x)) dx`` (``game_c_bid``).

    ``input_code``:

    * ``cues``: two inputs, standardised ``h`` and standardised ``f``.
    * ``population``: ``h`` as a population code over ``n_population`` Gaussian tuning
      curves (centres evenly spaced over the hunger types, width one spacing), plus
      standardised ``f``. The network has to decode a distributed signal first.
    """

    INPUT_CODES = ("cues", "population")

    def __init__(
        self,
        cost_rate: float,
        grid: TimeGrid,
        hunger_rate: float = 1.0,
        food_levels: tuple[float, ...] = (0.5, 1.0, 1.5, 2.0),
        n_hunger: int = 25,
        input_code: str = "cues",
        n_population: int = 16,
    ) -> None:
        if hunger_rate <= 0 or cost_rate <= 0 or min(food_levels) <= 0:
            raise ValueError("hunger_rate, cost_rate and every food level must be positive")
        if input_code not in self.INPUT_CODES:
            raise ValueError(f"input_code must be one of {self.INPUT_CODES}, got {input_code!r}")
        self.name = "C"
        self.value_rate = float("nan")  # no single exponential rate: use bne_bids()
        self.cost_rate = float(cost_rate)
        self.grid = grid
        self.hunger_rate = float(hunger_rate)
        self.food_levels = tuple(float(f) for f in food_levels)
        self.n_hunger = int(n_hunger)
        self.input_code = input_code
        self.n_population = int(n_population)
        u = (np.arange(self.n_hunger) + 0.5) / self.n_hunger
        self.hunger = -np.log1p(-u) / self.hunger_rate
        food = np.asarray(self.food_levels)
        # food-major: type k = j * n_hunger + i has food f_j and hunger h_i
        self.type_food = np.repeat(food, self.n_hunger)
        self.type_hunger = np.tile(self.hunger, len(food))
        self.values = self.type_hunger * self.type_food
        self.n_types = len(self.values)
        self.type_weights = np.full(self.n_types, 1.0 / self.n_types)
        self.win = win_matrix(self.grid.n_bins)
        self.cost = cost_matrix(self.grid.times, self.cost_rate)
        self.input_dim = 2 if input_code == "cues" else self.n_population + 1

    def __repr__(self) -> str:
        return (f"GameC(food_levels={self.food_levels}, n_hunger={self.n_hunger}, "
                f"input_code={self.input_code!r}, grid={self.grid})")

    def features(self) -> np.ndarray:
        food = np.asarray(self.food_levels)
        f_std = (self.type_food - food.mean()) / (food.std() if food.std() > 0 else 1.0)
        if self.input_code == "cues":
            # same standardisation as Game B: (h - E h) / sd(h) = rate * h - 1
            return np.stack([self.hunger_rate * self.type_hunger - 1.0, f_std], axis=1)
        centres = np.linspace(self.hunger[0], self.hunger[-1], self.n_population)
        width = centres[1] - centres[0]
        pop = np.exp(-0.5 * ((self.type_hunger[:, None] - centres[None, :]) / width) ** 2)
        return np.concatenate([pop, f_std[:, None]], axis=1)

    def bne_bids(self) -> np.ndarray:
        """Computed by quadrature once per game and cached (evaluation asks often)."""
        if getattr(self, "_bne", None) is None:
            from .equilibrium import game_c_bid

            self._bne = game_c_bid(self.values, self.food_levels, self.hunger_rate, self.cost_rate)
            self._bne.setflags(write=False)
        return self._bne


Game = GameA | GameB


def build_game(cfg: dict[str, Any]) -> Game:
    """Construct a game from the ``game`` section of an experiment config."""
    grid = TimeGrid(t_max=float(cfg["t_max"]), n_bins=int(cfg["n_bins"]))
    if cfg["name"] == "A":
        return GameA(prize=float(cfg["prize"]), cost_rate=float(cfg["cost_rate"]), grid=grid)
    if cfg["name"] == "B":
        return GameB(
            value_rate=float(cfg["value_rate"]),
            cost_rate=float(cfg["cost_rate"]),
            grid=grid,
            n_types=int(cfg["n_types"]),
        )
    if cfg["name"] == "C":
        return GameC(
            cost_rate=float(cfg["cost_rate"]),
            grid=grid,
            hunger_rate=float(cfg.get("hunger_rate", 1.0)),
            food_levels=tuple(cfg.get("food_levels", (0.5, 1.0, 1.5, 2.0))),
            n_hunger=int(cfg.get("n_hunger", 25)),
            input_code=str(cfg.get("input_code", "cues")),
            n_population=int(cfg.get("n_population", 16)),
        )
    raise ValueError(f"unknown game {cfg['name']!r}")
