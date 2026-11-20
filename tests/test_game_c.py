"""Game C: value from two private cues (v = hunger x food). The equilibrium is checked
independently of the formula that produces it, and the game is checked to need both cues."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from scipy import integrate

from src.agents.factory import build_policy, input_dim
from src.config import load_config
from src.games import GameB, GameC, TimeGrid, build_game, game_b_analytic_policy, game_c_bid, mixture_hazard
from src.games.equilibrium import spread_onto_grid
from src.metrics import exploitability as mx
from src.training.trainer import ExactObjective, SampledSelfPlay

FOOD = (0.5, 1.0, 1.5, 2.0)


@pytest.fixture(scope="module")
def game() -> GameC:
    return GameC(cost_rate=1.0, grid=TimeGrid(20.0, 64))


def survival(x, food=FOOD, rate=1.0):
    return np.mean([np.exp(-rate * x / f) for f in food], axis=0)


def density(x, food=FOOD, rate=1.0):
    return np.mean([(rate / f) * np.exp(-rate * x / f) for f in food], axis=0)


# ------------------------------------------------------------------ the equilibrium
def test_one_food_level_is_game_b():
    v = np.array([0.1, 0.5, 1.0, 2.0, 4.0])
    np.testing.assert_allclose(game_c_bid(v, (1.0,), 1.0, 1.0), v**2 / 2, rtol=1e-9)
    np.testing.assert_allclose(game_c_bid(v, (2.0,), 1.0, 1.0), v**2 / 4, rtol=1e-9)  # Exp(1/2): rate v^2 / 2
    c = GameC(cost_rate=1.0, grid=TimeGrid(12.0, 64), food_levels=(1.0,), n_hunger=50)
    b = GameB(value_rate=1.0, cost_rate=1.0, grid=TimeGrid(12.0, 64), n_types=50)
    np.testing.assert_allclose(c.values, b.values)
    np.testing.assert_allclose(c.bne_bids(), b.bne_bids(), rtol=1e-9)
    np.testing.assert_allclose(c.features()[:, 0], b.features()[:, 0])


def test_hazard_is_density_over_survival():
    x = np.linspace(0.0, 12.0, 50)
    np.testing.assert_allclose(mixture_hazard(x, FOOD, 1.0), density(x) / survival(x), rtol=1e-10)
    # falls from the mean rate towards the slowest component's rate
    h = mixture_hazard(np.array([0.0, 50.0]), FOOD, 1.0)
    assert h[0] == pytest.approx(np.mean([1 / f for f in FOOD]))
    assert h[1] == pytest.approx(1 / max(FOOD), rel=1e-3)


@pytest.mark.parametrize("v", [0.3, 1.0, 2.0, 3.5, 5.0, 7.0])
def test_truthful_bid_is_a_best_response(v):
    """Payoff of type ``v`` bidding ``b(z)`` against an opponent playing ``b``, by quadrature:

    ``Pi(z, v) = G(z) v - int_0^z c b(x) g(x) dx - (1 - G(z)) c b(z)``.

    Uses only G, g and b, not the integral formula behind b, and asks that ``z = v``
    maximises it: no type gains by bidding like another type.
    """
    b = lambda x: float(game_c_bid(np.array([x]), FOOD, 1.0, 1.0)[0])  # noqa: E731

    def payoff(z):
        paid_win, _ = integrate.quad(lambda x: b(x) * density(x), 0.0, z, limit=200)
        return (1 - survival(z)) * v - paid_win - survival(z) * b(z)

    zs = np.linspace(0.0, 10.0, 401)
    vals = np.array([payoff(z) for z in zs])
    assert abs(zs[vals.argmax()] - v) <= zs[1] - zs[0] + 1e-9
    assert payoff(v) >= vals.max() - 1e-9


def test_bid_is_increasing_and_between_the_one_level_bounds():
    v = np.linspace(0.05, 7.5, 60)
    b = game_c_bid(v, FOOD, 1.0, 1.0)
    assert np.all(np.diff(b) > 0)
    # the hazard lies between 1/max f and mean 1/f, so b does too, scaled by v^2/2
    assert np.all(b <= np.mean([1 / f for f in FOOD]) * v**2 / 2 + 1e-12)
    assert np.all(b >= (1 / max(FOOD)) * v**2 / 2 - 1e-12)


# ------------------------------------------------------------------ the grid game
def test_types_are_equally_likely_and_food_major(game):
    assert game.n_types == 100 and np.allclose(game.type_weights, 0.01)
    assert np.all(game.type_food[:25] == 0.5) and np.all(game.type_food[-25:] == 2.0)
    np.testing.assert_allclose(game.values, game.type_hunger * game.type_food)
    assert game.bne_bids().max() < game.grid.t_max  # t_max 20 holds every BNE bid


def test_the_game_needs_both_cues(game):
    """Near-equilibrium play needs hunger and food combined multiplicatively."""
    oracle = mx.utility_loss_b(game, game_b_analytic_policy(game))
    b = game.bne_bids()
    pol = lambda bids: spread_onto_grid(game.grid, bids)  # noqa: E731
    fbar = np.mean(FOOD)
    ignore_food = mx.utility_loss_b(game, pol(game_c_bid(game.type_hunger * fbar, FOOD, 1.0, 1.0)))
    hi = np.unique(game.type_hunger, return_inverse=True)[1]
    fi = np.unique(game.type_food, return_inverse=True)[1]
    x = np.concatenate([np.eye(hi.max() + 1)[hi], np.eye(fi.max() + 1)[fi][:, 1:]], axis=1)
    additive = x @ np.linalg.lstsq(x, b, rcond=None)[0]  # best alpha(h) + beta(f)
    additive_loss = mx.utility_loss_b(game, pol(additive))
    assert oracle < 0.01
    assert ignore_food > 5 * oracle
    assert additive_loss > 5 * oracle


@pytest.mark.parametrize("code,dim", [("cues", 2), ("population", 17)])
def test_features(code, dim):
    g = GameC(cost_rate=1.0, grid=TimeGrid(20.0, 64), input_code=code)
    f = g.features()
    assert f.shape == (100, dim) and g.input_dim == dim and input_dim(g) == dim
    assert np.isfinite(f).all()
    assert len({tuple(r) for r in np.round(f, 12)}) == 100  # every type has its own input
    if code == "population":
        # the most active hunger channel tracks hunger
        peak = f[:25, :16].argmax(axis=1)
        assert np.all(np.diff(peak) >= 0) and peak[0] == 0 and peak[-1] == 15
    else:
        assert abs(f[:, 0].mean()) < 0.2 and abs(f[:, 1].mean()) < 1e-12


def test_build_game_reads_the_config():
    g = build_game({"name": "C", "cost_rate": 1.0, "t_max": 20.0, "n_bins": 64, "input_code": "population"})
    assert isinstance(g, GameC) and isinstance(g, GameB) and g.input_dim == 17
    with pytest.raises(ValueError):
        GameC(cost_rate=1.0, grid=TimeGrid(20.0, 64), input_code="spikes")


# ------------------------------------------------------------------ training on it
def test_exact_objective_is_the_grid_payoff(game):
    rng = np.random.default_rng(0)
    p = rng.dirichlet(np.ones(64), size=game.n_types)

    class Fixed(torch.nn.Module):
        n_agents = 1

        def forward(self, feats, types):
            return torch.log(torch.as_tensor(p, dtype=torch.float32)).unsqueeze(0)

    loss = ExactObjective(game, 1).loss(Fixed())
    assert -float(loss) == pytest.approx(game.ex_ante_payoff(p, p), rel=1e-4)


@pytest.mark.parametrize("name", ["dense", "connectome"])
def test_networks_take_the_cue_inputs_and_learn(name):
    cfg = load_config("configs/gamec_exact.yaml")
    game = build_game(cfg["game"])
    from src.training.runner import circuit_for, parameter_budget

    circuit = circuit_for(cfg, {"name": name}, 0) if name != "dense" else None
    pol = build_policy({"name": name}, game, 1, cfg["network"], 0, circuit, parameter_budget(cfg))
    obj = ExactObjective(game, 1)
    opt = torch.optim.Adam(pol.parameters(), lr=0.01, betas=(0.0, 0.999))
    from src.training.evaluate import policy_probs

    start = mx.utility_loss_b(game, policy_probs(pol, game)[0])
    for _ in range(150):
        opt.zero_grad()
        obj.loss(pol).backward()
        opt.step()
    end = mx.utility_loss_b(game, policy_probs(pol, game)[0])
    assert end < 0.5 * start


def test_sampled_mode_runs_on_game_c(game):
    cfg = load_config("configs/gamec_sampled.yaml")
    pol = build_policy({"name": "tabular"}, game, 8, cfg["network"], 0)
    sp = SampledSelfPlay(game, 8, cfg["training"], torch.Generator().manual_seed(0))
    out = sp.rollout(pol)
    # each agent draws its own type (hunger and food) for every match, values are h * f
    np.testing.assert_allclose(sp.type_values.numpy(), game.values.astype(np.float32))
    assert out["types"].shape == (8, 128) and int(out["types"].min()) >= 0 and int(out["types"].max()) < 100
    assert len(torch.unique(out["types"])) > 90  # 1,024 draws cover nearly all 100 types
    assert torch.isfinite(out["advantage"]).all() and out["logp_all"].requires_grad
