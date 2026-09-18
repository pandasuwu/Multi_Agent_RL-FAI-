"""Payoff engine, equilibria and exact metrics."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import integrate

from src.games import (
    GameA,
    GameB,
    TimeGrid,
    asymmetric_continuous_survival,
    asymmetric_fixed_point,
    game_a_analytic_policy,
    game_b_analytic_policy,
    game_b_bid,
    game_b_interim_payoff_continuous,
    pairwise_payoff,
    solve_symmetric_equilibrium,
)
from src.metrics.exploitability import (
    bid_l2,
    bid_mse_by_decile,
    exploitability_a,
    population_utility_loss_a,
    relative_utility_loss_b,
    total_variation,
    utility_loss_b,
    wasserstein_1,
)


@pytest.fixture
def game_a() -> GameA:
    return GameA(prize=1.0, cost_rate=1.0, grid=TimeGrid(t_max=8.0, n_bins=64))


@pytest.fixture
def game_b() -> GameB:
    return GameB(value_rate=1.0, cost_rate=1.0, grid=TimeGrid(t_max=12.0, n_bins=64), n_types=50)


# ---------------------------------------------------------------- payoff engine
@pytest.mark.parametrize(
    "t_i, t_j, value, cost, expected",
    [
        (3.0, 1.0, 1.0, 1.0, 1.0 - 1.0),  # i outlasts j: V - c t_j
        (1.0, 3.0, 1.0, 1.0, -1.0),  # i quits first: -c t_i
        (2.0, 2.0, 1.0, 1.0, 0.5 - 2.0),  # tie: V/2 - c t
        (0.0, 0.0, 1.0, 1.0, 0.5),  # tie at zero: V/2
        (0.0, 5.0, 1.0, 1.0, 0.0),  # quit immediately and lose: nothing paid
        (4.0, 0.5, 3.0, 2.0, 3.0 - 1.0),  # V = 3, c = 2: 3 - 2 * 0.5
        (0.5, 4.0, 3.0, 2.0, -1.0),  # loser pays c t_i = 1
    ],
)
def test_pairwise_payoff_hand_cases(t_i, t_j, value, cost, expected):
    assert pairwise_payoff(t_i, t_j, value, cost) == pytest.approx(expected)


def test_winner_and_loser_pay_the_same(game_a):
    t = game_a.grid.times
    for i in range(0, 64, 7):
        for j in range(0, 64, 5):
            if i == j:
                continue
            ui = pairwise_payoff(t[i], t[j], 1.0, 1.0)
            uj = pairwise_payoff(t[j], t[i], 1.0, 1.0)
            # payoffs sum to V minus both paying c * min(t_i, t_j)
            assert ui + uj == pytest.approx(1.0 - 2 * min(t[i], t[j]))


def test_payoff_matrix_matches_pairwise(game_a):
    t = game_a.grid.times
    expected = pairwise_payoff(t[:, None], t[None, :], game_a.prize, game_a.cost_rate)
    np.testing.assert_allclose(game_a.payoff, expected)


def test_expected_payoff_hand_computed():
    grid = TimeGrid(t_max=2.0, n_bins=3)  # times 0, 1, 2
    game = GameA(prize=1.0, cost_rate=1.0, grid=grid)
    p = np.array([0.5, 0.25, 0.25])
    # quit at 1 vs p: 0.5 * (1 - 0) + 0.25 * (0.5 - 1) + 0.25 * (-1) = 0.125
    assert game.expected_payoffs(p)[1] == pytest.approx(0.125)
    # quit at 0 vs p: 0.5 * 0.5 = 0.25 ; quit at 2: 0.5*1 + 0.25*(1-1) + 0.25*(0.5-2)
    assert game.expected_payoffs(p)[0] == pytest.approx(0.25)
    assert game.expected_payoffs(p)[2] == pytest.approx(0.5 - 0.375)


def test_game_b_interim_payoffs_hand_computed():
    grid = TimeGrid(t_max=2.0, n_bins=3)
    game = GameB(value_rate=1.0, cost_rate=1.0, grid=grid, n_types=2)
    q = np.array([0.0, 1.0, 0.0])  # opponent always quits at t = 1
    u = game.interim_payoffs(q)
    v = game.values
    np.testing.assert_allclose(u[:, 0], 0.0)  # quit at 0: lose, pay 0
    np.testing.assert_allclose(u[:, 1], v / 2 - 1.0)  # tie at 1
    np.testing.assert_allclose(u[:, 2], v - 1.0)  # win, pay 1


# --------------------------------------------------------------- equilibria
def test_oracle_near_zero_exploitability(game_a):
    p = game_a_analytic_policy(game_a)
    assert p.sum() == pytest.approx(1.0)
    assert exploitability_a(game_a, p) < 2e-3


def test_oracle_equilibrium_payoff_is_order_delta(game_a):
    """Continuous equilibrium payoff is exactly 0; on the grid it is O(delta)."""
    p = game_a_analytic_policy(game_a)
    assert abs(game_a.expected_payoff(p, p)) < game_a.grid.delta


def test_discrete_equilibrium_is_exact_and_matches_oracle(game_a):
    q = solve_symmetric_equilibrium(game_a.payoff)
    assert exploitability_a(game_a, q) < 1e-10
    assert total_variation(q, game_a_analytic_policy(game_a)) < 5e-3


def test_oracle_error_shrinks_with_grid():
    losses = []
    for n in (32, 64, 128):
        g = GameA(prize=1.0, cost_rate=1.0, grid=TimeGrid(t_max=8.0, n_bins=n))
        losses.append(exploitability_a(g, game_a_analytic_policy(g)))
    assert losses[0] > losses[1] > losses[2]


def test_exploitability_nonnegative_and_positive_off_equilibrium(game_a):
    rng = np.random.default_rng(0)
    for _ in range(20):
        p = rng.dirichlet(np.ones(64))
        assert exploitability_a(game_a, p) > 1e-3
    uniform = np.full(64, 1 / 64)
    assert exploitability_a(game_a, uniform) > 0.1


def test_population_loss_reduces_to_self_play(game_a):
    p = game_a_analytic_policy(game_a)
    assert population_utility_loss_a(game_a, p[None]) == pytest.approx(exploitability_a(game_a, p))
    # a population whose members all play the oracle is also near equilibrium
    assert population_utility_loss_a(game_a, np.stack([p] * 4)) < 2e-3


def test_game_b_bid_is_best_response_to_itself_continuous():
    """b(v) = v^2/2 maximises Pi(z, v) over z, with no grid involved."""
    z_grid = np.linspace(0.0, 6.0, 601)
    for v in (0.25, 0.7, 1.0, 1.9, 3.3):
        payoffs = [game_b_interim_payoff_continuous(z, v, 1.0, 1.0) for z in z_grid]
        assert z_grid[int(np.argmax(payoffs))] == pytest.approx(v, abs=0.01)
        # first-order condition dPi/dz = f(z)(v - z) vanishes at z = v
        h = 1e-4
        d = (
            game_b_interim_payoff_continuous(v + h, v, 1.0, 1.0)
            - game_b_interim_payoff_continuous(v - h, v, 1.0, 1.0)
        ) / (2 * h)
        assert abs(d) < 1e-6


def test_game_b_bid_closed_form():
    v = np.linspace(0, 4, 9)
    np.testing.assert_allclose(game_b_bid(v, 1.0, 1.0), v**2 / 2)
    general, _ = integrate.quad(lambda x: x * 2.0, 0, 1.5)  # rate 2: hazard 2
    assert game_b_bid(1.5, 2.0, 1.0) == pytest.approx(general)


def test_game_b_oracle_on_grid(game_b):
    p = game_b_analytic_policy(game_b)
    np.testing.assert_allclose(p.sum(axis=1), 1.0)
    assert bid_l2(game_b, p) < 1e-12
    assert np.all(bid_mse_by_decile(game_b, p) < 1e-20)
    assert relative_utility_loss_b(game_b, p) == pytest.approx(0.0, abs=1e-12)
    u_star = game_b.ex_ante_payoff(p, p)
    assert utility_loss_b(game_b, p) < 0.02 * u_star


def test_game_b_truthful_deviation_is_worse(game_b):
    """Bidding the wrong function has a larger utility loss than the BNE."""
    bne = game_b_analytic_policy(game_b)
    from src.games.equilibrium import spread_onto_grid

    linear = spread_onto_grid(game_b.grid, game_b.values)  # b(v) = v
    assert utility_loss_b(game_b, linear) > 5 * utility_loss_b(game_b, bne)


# ------------------------------------------------------- asymmetric fixed point
def test_asymmetric_ratio_one_is_the_equilibrium(game_a):
    p, _ = asymmetric_fixed_point(game_a, 1.0)
    assert exploitability_a(game_a, p) < 1e-9


def test_asymmetric_fixed_point_is_exploitable_and_monotone(game_a):
    expl = [exploitability_a(game_a, asymmetric_fixed_point(game_a, r)[0]) for r in (1, 2, 4, 8)]
    assert all(a < b for a, b in zip(expl, expl[1:]))
    assert expl[1] > 0.02


def test_asymmetric_continuous_baseline_self_consistent():
    """The closed-form baseline b = V(rho-1)/(4 rho) equals V/2 - c E[min(T1, T2)]."""
    for rho in (2.0, 4.0, 8.0):
        surv = lambda t: asymmetric_continuous_survival(np.array([t]), 1.0, 1.0, rho)[0][0]  # noqa: E731
        _, b = asymmetric_continuous_survival(np.array([0.0]), 1.0, 1.0, rho)
        e_min, _ = integrate.quad(lambda t: surv(t) ** 2, 0, np.inf, limit=400)
        assert 0.5 - e_min == pytest.approx(b, rel=1e-6)


# ------------------------------------------------------------ distance metrics
def test_wasserstein_and_tv_hand_cases():
    t = np.array([0.0, 1.0, 2.0])
    p = np.array([1.0, 0.0, 0.0])
    q = np.array([0.0, 0.0, 1.0])
    assert wasserstein_1(p, q, t) == pytest.approx(2.0)
    assert total_variation(p, q) == pytest.approx(1.0)
    assert wasserstein_1(p, p, t) == 0.0


def test_discrete_equilibrium_is_geometric_with_closed_form_ratio(game_a):
    """p_{k+1}/p_k = sqrt(1 + x^2) - x with x = c*delta/V (docs/theory.md, 1.3)."""
    q = solve_symmetric_equilibrium(game_a.payoff)
    x = game_a.cost_rate * game_a.grid.delta / game_a.prize
    r = np.sqrt(1 + x**2) - x
    # the T_max truncation adds an alternating mode (-1/r)^k that decays away from
    # the boundary, so check the head of the distribution
    np.testing.assert_allclose(q[1:12] / q[:11], r, rtol=5e-5)
    assert game_a.expected_payoff(q, q) == pytest.approx(0.5 * game_a.prize * (1 - r))


def test_grid_game_is_stable(game_a):
    """Symmetric part of U is negative definite on the simplex tangent space."""
    n = game_a.grid.n_bins
    basis = np.linalg.qr(np.eye(n) - 1.0 / n)[0][:, : n - 1]
    sym = 0.5 * (game_a.payoff + game_a.payoff.T)
    assert np.linalg.eigvalsh(basis.T @ sym @ basis).max() < 0
