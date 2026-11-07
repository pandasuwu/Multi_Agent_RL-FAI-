"""Checks added after the project-review feedback: population Nash metric, the
asymmetric rest point against simulations of the rule, input-output paths in the
control graphs, and the shared learning-rate sweep."""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from src.agents.factory import CircuitSpec
from src.config import REPO_ROOT, load_config
from src.games import GameA, TimeGrid, asymmetric_fixed_point, game_a_analytic_policy
from src.games.asym_dynamics import monte_carlo, natural_gradient_field
from src.metrics.exploitability import exploitability_a, total_variation
from src.training.evaluate import evaluate
from src.training.runner import expand_jobs, learning_rate

GAME = GameA(1.0, 1.0, TimeGrid(8.0, 64))


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ------------------------------------------------------------ population Nash
def test_split_population_is_individually_fine_but_not_a_population_nash():
    """Two agents that split into 'quit now' and 'never quit' each best-respond to the
    other (individual loss 0), but their mixture is far from the symmetric equilibrium."""
    early = np.zeros(64)
    early[0] = 1.0
    late = np.zeros(64)
    late[-1] = 1.0
    out = evaluate(GAME, np.stack([early, late])[:, None, :])
    assert out["exploitability"] < 1e-9
    assert out["mixture_exploitability"] > 0.1
    assert out["agent_tv_mean"] > 0.5


def test_agents_at_equilibrium_score_zero_on_every_population_metric():
    eq = game_a_analytic_policy(GAME)
    out = evaluate(GAME, np.stack([eq] * 4)[:, None, :])
    assert out["exploitability"] < 3e-3
    assert out["mixture_exploitability"] < 3e-3
    assert out["agent_tv_mean"] == pytest.approx(0.0, abs=1e-12)


# --------------------------------------------------- asymmetric rest point
@pytest.mark.parametrize("ratio", [1.0, 4.0])
def test_expected_dynamics_of_the_rule_reach_the_grid_fixed_point(ratio):
    fixed, b_fixed = asymmetric_fixed_point(GAME, ratio)
    p, b = natural_gradient_field(GAME, ratio, steps=100_000)
    assert total_variation(p, fixed) < 1e-3
    assert b == pytest.approx(b_fixed, abs=1e-4)


def test_sampled_rule_lands_nearer_the_shifted_point_than_the_equilibrium():
    ratio = 8.0
    fixed, b_fixed = asymmetric_fixed_point(GAME, ratio)
    p, b = monte_carlo(GAME, ratio, seed=0, steps=6_000, batch=2048)
    eq = game_a_analytic_policy(GAME)
    assert total_variation(p, fixed) < total_variation(p, eq)
    assert b == pytest.approx(b_fixed, abs=0.06)
    assert exploitability_a(GAME, p) > 0.08  # the rest point is exploitable


# ------------------------------------------------------ input-output paths
def test_path_diagnostics_on_a_toy_graph():
    diag = _load_script("path_diagnostics").diagnostics
    # 0 -> 2 -> 3 -> 4 (output), 1 -> 4 directly; 5 is an unreachable output
    spec = CircuitSpec(
        n_neurons=6,
        edges=np.array([[0, 2], [2, 3], [3, 4], [1, 4]]),
        input_nodes=np.array([0, 1]),
        output_nodes=np.array([4, 5]),
        node_sign=np.ones(6),
        graph_id="toy",
        synapses=np.ones(4),
    )
    d = diag(spec, depth=2)
    assert d["outputs_reachable"] == 1
    assert d["pairs_reachable"] == 2
    assert d["pairs_reachable_in_time"] == 1  # 0 -> 4 needs 3 synapses
    assert d["shortest_min"] == 1
    assert d["walks_in_time"] == 1  # only 1 -> 4 fits in two steps


# --------------------------------------------------- shared-lr sweep config
def test_lr_sweep_runs_every_arm_at_the_swept_rate():
    cfg = load_config(REPO_ROOT / "configs" / "lrsweep_game_b_exact.yaml")
    jobs = expand_jobs(cfg)
    rates = cfg["sweep"]["training.shared_lr"]
    assert len(jobs) == len(cfg["arms"]) * len(rates) * cfg["seeds"]["count"]
    for job in jobs:
        assert learning_rate(job["cfg"], job["arm"]) == job["overrides"]["training.shared_lr"]
