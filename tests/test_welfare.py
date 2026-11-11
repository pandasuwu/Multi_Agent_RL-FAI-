import numpy as np

from src.games import GameA, GameB, TimeGrid, game_a_analytic_policy, game_b_analytic_policy
from src.metrics.welfare import population_welfare, welfare_a, welfare_b


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


def test_population_welfare_leaves_out_self_matches():
    # a pair split into "quit at once" and "hold to the end" never fights: welfare is
    # the whole prize. The pair's mixture against itself would fight a quarter of the
    # time for the full length, which is what the mixture formula reports.
    a, _ = _games()
    n = a.grid.n_bins
    quit_now, hold = np.zeros(n), np.zeros(n)
    quit_now[0], hold[-1] = 1.0, 1.0
    pair = np.stack([quit_now, hold])[:, None, :]
    w = population_welfare(a, pair)
    assert np.isclose(w["welfare"], 1.0) and np.isclose(w["fight_cost"], 0.0)
    assert welfare_a(a, pair[:, 0].mean(axis=0))["welfare"] < -2.9


def test_population_welfare_reduces_to_symmetric_cases():
    a, b = _games()
    rng = np.random.default_rng(0)
    pa, pb = game_a_analytic_policy(a), game_b_analytic_policy(b)
    # one shared policy, and a population of identical agents, are the symmetric case
    for m in (1, 4):
        assert np.isclose(population_welfare(a, np.tile(pa, (m, 1, 1)))["welfare"], welfare_a(a, pa)["welfare"])
        assert np.isclose(population_welfare(b, np.tile(pb, (m, 1, 1)))["welfare"], welfare_b(b, pb)["welfare"])
    # different agents: the average over pairs of distinct agents
    pols = rng.dirichlet(np.ones(a.grid.n_bins), size=(3, 1))
    t = a.grid.times
    mn = np.minimum(t[:, None], t[None, :])
    e = np.mean([pols[i, 0] @ mn @ pols[j, 0] for i in range(3) for j in range(3) if i != j])
    assert np.isclose(population_welfare(a, pols)["welfare"], 1 - 2 * e)
    # Game B: a match's welfare does not depend on which agent is listed first
    pb2 = rng.dirichlet(np.ones(b.grid.n_bins), size=(2, b.n_types))
    w12, w21 = welfare_b(b, pb2[0], pb2[1]), welfare_b(b, pb2[1], pb2[0])
    assert all(np.isclose(w12[k], w21[k]) for k in w12)
    assert np.isclose(population_welfare(b, pb2)["welfare"], w12["welfare"])
