"""FlyGM arm: frozen connectome synapses, learned neurons (Jin et al. 2026, arXiv:2602.17997).

The circuit arms (3-5) keep the connectome's *sparsity* and train one weight per
edge. The FlyGM arm keeps the connectome's *weights* and trains the neurons
instead. The synaptic matrix

    W[post, pre] = sign(pre) * synapse_count(pre -> post),   rescaled to spectral radius 1

is a fixed buffer: it is never a parameter and never receives a gradient. What
trains is

* ``eta``: a descriptor in ``R^D`` per neuron (``D = descriptor_dim``), the only
  thing that distinguishes one neuron from another;
* ``f_psi``: one shared update MLP applied at every neuron and every step,
  ``h_i <- act(f_psi(h_i, m_i, eta_i))`` with messages ``m = W h + drive``;
* the input encoder and gate on the input (PPL1) neurons,
  ``drive_i = enc_i(x) * sigmoid(gate_i(x))``;
* the readout from the output (AVLP727m) neurons to the ``n_bins`` logits,
  zero-initialised like every other arm so training starts from the uniform policy.
  ``readout_nodes: all`` reads out from every neuron through a rank-``readout_rank``
  map instead (:class:`~src.agents.policies.GraphReadoutMixin`);
* optionally (``w_gain``), one trainable scalar ``g`` per agent on the frozen
  matrix, ``m = g * W h + drive``, initialised to 1. ``W`` itself stays frozen.

``unroll_steps`` is shared with the circuit arms (6), so the two architectures see
the graph for the same number of synaptic hops.

**Sign of unsigned neurons.** 26 of the 258 neurons have no transmitter
prediction (``node_sign == 0``), including all 16 dopaminergic PPL1 input
neurons. With ``sign = 0`` those neurons have no output at all in ``W``, which
disconnects the input neurons from the circuit entirely (``input_output_gain``
is then exactly 0 for every graph). ``unsigned_sign`` chooses what they get; it
is recorded per run, with the two diagnostics below.

**Diagnostics** (per graph, logged on every run):

* ``spectral_radius_raw``: the radius of ``W`` *before* the global rescaling
  (about 1230 for the real graph, a few hundred for the controls). The real
  circuit's radius is dominated by a strong recurrent core, so dividing by it
  may be what crushes the input-to-output path.
* ``input_output_gain``: ``sum_{k=1..unroll_steps} |(W^k)[output, input]|`` on
  the rescaled ``W`` (about 0.008 for the real graph, 1-13 for the controls).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch
from torch import nn

from .policies import ACTIVATIONS, GraphReadoutMixin, Policy, readout_param_count


def signed_weight_matrix(
    n_neurons: int,
    edges: np.ndarray,
    synapses: np.ndarray,
    node_sign: np.ndarray,
    unsigned_sign: float,
) -> np.ndarray:
    """``W[post, pre] = sign(pre) * synapse_count``, before any rescaling."""
    sign = np.asarray(node_sign, dtype=np.float64).copy()
    if float(unsigned_sign) not in (0.0, 1.0):
        raise ValueError("unsigned_sign must be 0 or +1")
    sign[sign == 0] = float(unsigned_sign)
    edges = np.asarray(edges, dtype=np.int64)
    w = np.zeros((n_neurons, n_neurons), dtype=np.float64)
    w[edges[:, 1], edges[:, 0]] = sign[edges[:, 0]] * np.asarray(synapses, dtype=np.float64)
    return w


def spectral_radius(w: np.ndarray) -> float:
    """``max |lambda_i|`` of a square matrix."""
    if w.size == 0:
        return 0.0
    return float(np.max(np.abs(np.linalg.eigvals(w))))


def input_output_gain(
    w: np.ndarray, input_nodes: np.ndarray, output_nodes: np.ndarray, n_steps: int
) -> float:
    """``sum_{k=1..n_steps} |(W^k)[output, input]|``: how much input reaches the readout."""
    total, power = 0.0, np.eye(len(w))
    for _ in range(n_steps):
        power = power @ w
        total += float(np.abs(power[np.ix_(np.asarray(output_nodes), np.asarray(input_nodes))]).sum())
    return total


def flygm_param_count(
    n_neurons: int,
    d_in: int,
    n_in: int,
    n_out: int,
    n_bins: int,
    descriptor_dim: int,
    width: int,
    layers: int,
    readout_nodes: str = "output",
    readout_rank: int = 0,
    w_gain: bool = False,
) -> int:
    """Trainable parameters of one :class:`FlyGMPolicy` agent (``W`` is not one)."""
    sizes = [2 + descriptor_dim, *([width] * layers), 1]
    mlp = sum(a * b + b for a, b in zip(sizes[:-1], sizes[1:]))
    encoder = 2 * (d_in * n_in + n_in)  # encoder and gate, each with a bias
    readout = readout_param_count(n_neurons, n_out, n_bins, readout_nodes, readout_rank)
    return n_neurons * descriptor_dim + encoder + mlp + readout + int(w_gain)


def match_flygm_width(
    target: int,
    n_neurons: int,
    d_in: int,
    n_in: int,
    n_out: int,
    n_bins: int,
    descriptor_dim: int,
    layers: int,
    readout_nodes: str = "output",
    readout_rank: int = 0,
    w_gain: bool = False,
) -> int:
    """Width of ``f_psi`` whose trainable count is closest to ``target``."""
    best, best_gap = 1, float("inf")
    for h in range(1, 4096):
        count = flygm_param_count(
            n_neurons, d_in, n_in, n_out, n_bins, descriptor_dim, h, layers, readout_nodes, readout_rank, w_gain
        )
        gap = abs(count - target)
        if gap < best_gap:
            best, best_gap = h, gap
    return best


class FlyGMPolicy(GraphReadoutMixin, Policy):
    """Frozen synapses, learned per-neuron descriptors and one shared update MLP."""

    def __init__(
        self,
        n_agents: int,
        d_in: int,
        n_neurons: int,
        edges: np.ndarray,
        synapses: np.ndarray,
        node_sign: np.ndarray,
        input_nodes: np.ndarray,
        output_nodes: np.ndarray,
        n_bins: int,
        unroll_steps: int,
        activation: str,
        descriptor_dim: int,
        hidden_width: int,
        hidden_layers: int,
        unsigned_sign: float,
        eta_init_std: float,
        generator: torch.Generator,
        readout_nodes: str = "output",
        readout_rank: int = 0,
        w_gain: bool = False,
    ) -> None:
        super().__init__()
        self.n_agents, self.n_bins, self.n_neurons = n_agents, n_bins, n_neurons
        self.unroll_steps = int(unroll_steps)
        self.descriptor_dim = int(descriptor_dim)
        self.act = ACTIVATIONS[activation]

        raw = signed_weight_matrix(n_neurons, edges, synapses, node_sign, unsigned_sign)
        radius = spectral_radius(raw)
        scaled = raw / radius if radius > 0 else raw
        self._diagnostics: dict[str, Any] = {
            "unsigned_sign": float(unsigned_sign),
            "spectral_radius_raw": radius,
            "input_output_gain": input_output_gain(scaled, input_nodes, output_nodes, self.unroll_steps),
            "descriptor_dim": self.descriptor_dim,
            "hidden_width": int(hidden_width),
            "frozen_synapses": int(len(np.asarray(edges))),
        }
        # W is a buffer, never a parameter: no gradient ever reaches a synapse.
        self.register_buffer("w", torch.as_tensor(scaled, dtype=torch.float32))
        self.register_buffer("input_nodes", torch.as_tensor(np.asarray(input_nodes, dtype=np.int64)))
        self.register_buffer("output_nodes", torch.as_tensor(np.asarray(output_nodes, dtype=np.int64)))

        n_in, n_out = len(input_nodes), len(output_nodes)
        self.eta = nn.Parameter(
            torch.randn(n_agents, n_neurons, self.descriptor_dim, generator=generator) * float(eta_init_std)
        )
        self.w_enc = nn.Parameter(
            torch.randn(n_agents, d_in, n_in, generator=generator) / math.sqrt(max(d_in, 1))
        )
        self.b_enc = nn.Parameter(torch.zeros(n_agents, 1, n_in))
        self.w_gate = nn.Parameter(
            torch.randn(n_agents, d_in, n_in, generator=generator) / math.sqrt(max(d_in, 1))
        )
        self.b_gate = nn.Parameter(torch.zeros(n_agents, 1, n_in))

        sizes = [2 + self.descriptor_dim, *([int(hidden_width)] * int(hidden_layers)), 1]
        self.psi_weights = nn.ParameterList()
        self.psi_biases = nn.ParameterList()
        for a, b in zip(sizes[:-1], sizes[1:]):
            w0 = torch.randn(n_agents, a, b, generator=generator) / math.sqrt(a)
            self.psi_weights.append(nn.Parameter(w0))
            self.psi_biases.append(nn.Parameter(torch.zeros(n_agents, 1, b)))

        # Built last, so with the defaults every initialisation is bit-identical to
        # what it was before these options existed.
        self._init_readout(n_agents, n_neurons, n_out, n_bins, readout_nodes, readout_rank, generator)
        self.w_gain = nn.Parameter(torch.ones(n_agents, 1, 1)) if w_gain else None

    @property
    def diagnostics(self) -> dict[str, Any]:
        """Per-graph diagnostics fixed at construction, plus the trained gain on ``W``
        (mean over agents) when there is one: read after training, it is the final value."""
        out = dict(self._diagnostics)
        if self.w_gain is not None:
            out["w_gain"] = float(self.w_gain.detach().mean())
        return out

    def _psi(self, z: torch.Tensor) -> torch.Tensor:
        """Shared update MLP on ``(M, P, 2 + D)`` features, returning ``(M, P, 1)``."""
        n = len(self.psi_weights)
        for k, (w, b) in enumerate(zip(self.psi_weights, self.psi_biases)):
            z = torch.baddbmm(b, z, w)
            if k < n - 1:
                z = self.act(z)
        return z

    def forward(self, features: torch.Tensor, type_index: torch.Tensor) -> torch.Tensor:
        m, b, _ = features.shape
        n, d = self.n_neurons, self.descriptor_dim
        enc = torch.baddbmm(self.b_enc, features, self.w_enc)
        gate = torch.sigmoid(torch.baddbmm(self.b_gate, features, self.w_gate))
        drive = features.new_zeros(m, b, n).index_copy(2, self.input_nodes, enc * gate)
        eta = self.eta.unsqueeze(1).expand(m, b, n, d)
        h = features.new_zeros(m, b, n)
        for _ in range(self.unroll_steps):
            recurrent = torch.matmul(h, self.w.t())
            if self.w_gain is not None:
                recurrent = recurrent * self.w_gain  # (M, 1, 1)
            msg = recurrent + drive
            z = torch.cat([h.unsqueeze(-1), msg.unsqueeze(-1), eta], dim=-1).reshape(m, b * n, 2 + d)
            h = self.act(self._psi(z).reshape(m, b, n))
        return self._read(h)


__all__ = [
    "FlyGMPolicy",
    "flygm_param_count",
    "input_output_gain",
    "match_flygm_width",
    "signed_weight_matrix",
    "spectral_radius",
]
