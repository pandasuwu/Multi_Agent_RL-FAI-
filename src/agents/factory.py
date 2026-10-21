"""Build the policy for an arm, with parameter-count matching across arms 2-5."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from ..games.war_of_attrition import Game, GameA
from .flygm import FlyGMPolicy, flygm_param_count, match_flygm_width
from .policies import DenseMLP, Policy, SparseCircuitPolicy, TabularPolicy, dense_param_count, match_dense_width

CIRCUIT_ARMS = ("connectome", "rewired", "random_sparse")
# FlyGM arms: same three graphs, but the synapses are frozen and the neurons train
# (src/agents/flygm.py). Each one maps to the graph its weights come from.
FLYGM_GRAPH = {"flygm": "connectome", "flygm_rewired": "rewired", "flygm_random_sparse": "random_sparse"}
FLYGM_ARMS = tuple(FLYGM_GRAPH)
GRAPH_ARMS = (*CIRCUIT_ARMS, *FLYGM_ARMS)  # arms that need a CircuitSpec
NETWORK_ARMS = ("dense", *CIRCUIT_ARMS, *FLYGM_ARMS)
ALL_ARMS = ("oracle", "tabular", *NETWORK_ARMS)
D_IN = 1  # context feature: V/c in Game A, the private value v in Game B


def graph_name(arm_name: str) -> str:
    """The graph an arm is built on: FlyGM arms reuse the three existing graphs."""
    return FLYGM_GRAPH.get(arm_name, arm_name)


def arm_label(arm: dict[str, Any]) -> str:
    return arm["name"] + ("+sign" if arm.get("sign_constrained") else "")


def n_contexts(game: Game) -> int:
    return 1 if isinstance(game, GameA) else game.n_types


@dataclass
class CircuitSpec:
    """Everything a circuit arm needs: an edge list plus the node roles."""

    n_neurons: int
    edges: np.ndarray  # (E, 2) pre, post
    input_nodes: np.ndarray
    output_nodes: np.ndarray
    node_sign: np.ndarray  # +1 excitatory, -1 inhibitory, 0 unknown/modulatory (Dale's law)
    graph_id: str  # checksum of the edge set actually used
    # Synapse count per edge, used only by the FlyGM arm (whose weights are frozen
    # synapse counts). The trained-edge arms 3-5 ignore it: their weights are learned.
    synapses: np.ndarray | None = None

    @property
    def edge_sign(self) -> np.ndarray:
        return self.node_sign[self.edges[:, 0]]


def circuit_param_count(spec: CircuitSpec, n_bins: int) -> int:
    """Trainable parameters of one :class:`SparseCircuitPolicy` agent."""
    return (
        D_IN * len(spec.input_nodes)
        + len(spec.edges)
        + spec.n_neurons
        + len(spec.output_nodes) * n_bins
        + n_bins
    )


def flygm_hidden_width(net_cfg: dict[str, Any], target_params: int | None, n_bins: int, spec: CircuitSpec) -> int:
    """Width of ``f_psi``: explicit, or solved to match ``target_params``."""
    fly = net_cfg["flygm"]
    width = fly["hidden_width"]
    if width == "matched":
        if target_params is None:
            raise ValueError("flygm hidden_width 'matched' needs the circuit parameter budget")
        width = match_flygm_width(
            target_params,
            spec.n_neurons,
            D_IN,
            len(spec.input_nodes),
            len(spec.output_nodes),
            n_bins,
            int(fly["descriptor_dim"]),
            int(fly["hidden_layers"]),
        )
    return int(width)


def dense_hidden_sizes(net_cfg: dict[str, Any], target_params: int | None, n_bins: int) -> list[int]:
    """Hidden widths of the dense arm: explicit, or matched to ``target_params``."""
    layers = int(net_cfg["dense_hidden_layers"])
    width = net_cfg["dense_width"]
    if width == "matched":
        if target_params is None:
            raise ValueError("dense_width 'matched' needs the circuit parameter budget")
        width = match_dense_width(target_params, D_IN, layers, n_bins)
    return [int(width)] * layers


def build_policy(
    arm: dict[str, Any],
    game: Game,
    n_agents: int,
    net_cfg: dict[str, Any],
    seed: int,
    circuit: CircuitSpec | None = None,
    budget: int | None = None,
) -> Policy:
    """Instantiate the policy of ``arm`` (not the oracle, which has no parameters)."""
    gen = torch.Generator().manual_seed(seed)
    n_bins = game.grid.n_bins
    name = arm["name"]
    if name == "tabular":
        return TabularPolicy(n_agents, n_contexts(game), n_bins)
    if name == "dense":
        hidden = dense_hidden_sizes(net_cfg, budget, n_bins)
        return DenseMLP(n_agents, D_IN, hidden, n_bins, net_cfg["activation"], gen)
    if name in FLYGM_ARMS:
        if circuit is None:
            raise ValueError(f"arm {name!r} needs a circuit")
        if circuit.synapses is None:
            raise ValueError(f"arm {name!r} needs the synapse counts of its graph")
        fly = net_cfg["flygm"]
        return FlyGMPolicy(
            n_agents=n_agents,
            d_in=D_IN,
            n_neurons=circuit.n_neurons,
            edges=circuit.edges,
            synapses=circuit.synapses,
            node_sign=circuit.node_sign,
            input_nodes=circuit.input_nodes,
            output_nodes=circuit.output_nodes,
            n_bins=n_bins,
            unroll_steps=int(net_cfg["unroll_steps"]),
            activation=net_cfg["activation"],
            descriptor_dim=int(fly["descriptor_dim"]),
            hidden_width=flygm_hidden_width(net_cfg, budget, n_bins, circuit),
            hidden_layers=int(fly["hidden_layers"]),
            unsigned_sign=float(fly["unsigned_sign"]),
            eta_init_std=float(fly["eta_init_std"]),
            generator=gen,
        )
    if name in CIRCUIT_ARMS:
        if circuit is None:
            raise ValueError(f"arm {name!r} needs a circuit")
        return SparseCircuitPolicy(
            n_agents=n_agents,
            d_in=D_IN,
            n_neurons=circuit.n_neurons,
            edges=circuit.edges,
            input_nodes=circuit.input_nodes,
            output_nodes=circuit.output_nodes,
            n_bins=n_bins,
            unroll_steps=int(net_cfg["unroll_steps"]),
            activation=net_cfg["activation"],
            init_gain=float(net_cfg["init_gain"]),
            generator=gen,
            edge_sign=circuit.edge_sign,
            sign_constrained=bool(arm.get("sign_constrained", False)),
            sparse_max_batch=int(net_cfg["sparse_max_batch"]),
        )
    raise ValueError(f"unknown arm {name!r}")


def assert_matched(counts: dict[str, int], tolerance: float) -> None:
    """Fail loudly if trainable parameter counts of the network arms differ by > tolerance.

    ``counts`` maps arm label -> parameters per agent, for arms 2-5 only.
    """
    lo, hi = min(counts.values()), max(counts.values())
    if hi > lo * (1 + tolerance):
        raise AssertionError(
            f"parameter counts differ by more than {tolerance:.0%}: {counts}. Arms with "
            "mismatched parameter counts must not be compared."
        )


__all__ = [
    "ALL_ARMS",
    "CIRCUIT_ARMS",
    "FLYGM_ARMS",
    "GRAPH_ARMS",
    "CircuitSpec",
    "NETWORK_ARMS",
    "arm_label",
    "assert_matched",
    "build_policy",
    "circuit_param_count",
    "dense_hidden_sizes",
    "dense_param_count",
    "flygm_hidden_width",
    "flygm_param_count",
    "graph_name",
]
