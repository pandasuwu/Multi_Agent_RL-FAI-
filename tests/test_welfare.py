import numpy as np

from src.games import GameA, GameB, TimeGrid, game_a_analytic_policy, game_b_analytic_policy
from src.metrics.welfare import welfare_a, welfare_b


def _games():
    a = GameA(prize=1.0, cost_rate=1.0, grid=TimeGrid(8.0, 64))
    b = GameB(value_rate=1.0, cost_rate=1.0, grid=TimeGrid(12.0, 64), n_types=50)
    return a, b


def test_welfare_is_twice_the_symmetric_payoff():
    a, b = _games()
    pa = game_a_analytic_policy(a)
    assert np.isclose(welfare_a(a, pa)["welfare"], 2 * a.expected_payoff(pa, pa))
    pb = game_b_analytic_policy(b)
    assert np.isclose(welfare_b(b, pb)["welfare"], 2 * b.ex_ante_payoff(pb, pb))


def test_no_fight_and_optimum():
    a, b = _games()
    now = np.zeros(64)
    now[0] = 1
    w = welfare_a(a, now)
    assert w["welfare"] == w["optimum"] == 1.0 and w["dissipation"] == 0
    wb = welfare_b(b, np.tile(now, (50, 1)))
    assert np.isclose(wb["welfare"], b.values.mean())   # coin flip, no fight
    assert wb["optimum"] > wb["welfare"]


def test_equilibrium_dissipates_most_of_the_prize():
    a, b = _games()
    assert welfare_a(a, game_a_analytic_policy(a))["dissipation"] > 0.8
    wb = welfare_b(b, game_b_analytic_policy(b))
    assert wb["allocated_value"] > 0.98 * wb["optimum"]   # equilibrium sorts by value
    assert wb["welfare"] < b.values.mean()                # but loses to a coin flip
