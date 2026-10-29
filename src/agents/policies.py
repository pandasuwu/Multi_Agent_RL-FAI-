"""Policy parameterisations for every arm, vectorised over a population of agents.

Every policy maps a context to logits over the quitting-time grid and carries a
leading agent dimension ``M`` so that a whole population trains in one batched
forward pass. Parameters of different agents never interact (Adam is elementwise),
so this is exactly ``M`` independent learners.

Inputs to :meth:`Policy.forward`:
    features: ``(M, B, d_in)`` float tensor (used by networks),
    type_index: ``(M, B)`` long tensor (used by the tabular arm).
Output: logits ``(M, B, n_bins)``.

The final (readout) layer of every arm is zero-initialised, so every arm starts
from exactly the uniform policy: initial conditions are identical across arms.
"""

from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ACTIVATIONS = {"tanh": torch.tanh, "relu": F.relu}


class Policy(nn.Module):
    n_agents: int
    n_bins: int

    def forward(self, features: torch.Tensor, type_index: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def trainable_parameters_per_agent(self) -> int:
        """Effective trainable parameters of one agent (masked-out entries excluded)."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad) // self.n_agents


class TabularPolicy(Policy):
    """Softmax logits directly over the grid, one row per context (type)."""

    def __init__(self, n_agents: int, n_contexts: int, n_bins: int) -> None:
        super().__init__()
        self.n_agents, self.n_bins = n_agents, n_bins
        self.logits = nn.Parameter(torch.zeros(n_agents, n_contexts, n_bins))

    def forward(self, features: torch.Tensor, type_index: torch.Tensor) -> torch.Tensor:
        idx = type_index.unsqueeze(-1).expand(-1, -1, self.n_bins)
        return torch.gather(self.logits, 1, idx)


class DenseMLP(Policy):
    """Conventional fully connected policy: ``d_in -> H -> ... -> H -> n_bins``."""

    def __init__(
        self,
        n_agents: int,
        d_in: int,
        hidden: list[int],
        n_bins: int,
        activation: str,
        generator: torch.Generator,
    ) -> None:
        super().__init__()
        self.n_agents, self.n_bins = n_agents, n_bins
        self.act = ACTIVATIONS[activation]
        sizes = [d_in, *hidden, n_bins]
        self.weights = nn.ParameterList()
        self.biases = nn.ParameterList()
        for k, (a, b) in enumerate(zip(sizes[:-1], sizes[1:])):
            last = k == len(sizes) - 2
            w = torch.zeros(n_agents, a, b)
            if not last:
                w.normal_(0.0, 1.0 / math.sqrt(a), generator=generator)
            self.weights.append(nn.Parameter(w))
            self.biases.append(nn.Parameter(torch.zeros(n_agents, 1, b)))

    def forward(self, features: torch.Tensor, type_index: torch.Tensor) -> torch.Tensor:
        h = features
        n = len(self.weights)
        for k, (w, b) in enumerate(zip(self.weights, self.biases)):
            h = torch.baddbmm(b, h, w)
            if k < n - 1:
                h = self.act(h)
        return h


READOUTS = ("output", "all")


def readout_param_count(n_neurons: int, n_out: int, n_bins: int, nodes: str = "output", rank: int = 0) -> int:
    """Trainable parameters of a graph arm's readout (logit bias included).

    ``output``: a full ``n_out x n_bins`` map from the output (AVLP727m) neurons.
    ``all``: a rank-``rank`` map from every neuron, ``h @ U @ V`` with ``U`` of
    ``n_neurons x rank`` and ``V`` of ``rank x n_bins``. A full map from all 258
    neurons would be 16,512 parameters, three times the whole budget."""
    if nodes == "output":
        return n_out * n_bins + n_bins
    if nodes == "all":
        return n_neurons * rank + rank * n_bins + n_bins
    raise ValueError(f"readout must be one of {READOUTS}, got {nodes!r}")


class GraphReadoutMixin:
    """Readout of a graph arm, from the output neurons or (low rank) from all neurons.

    Either way the last factor (``readout``) is zero-initialised, so the arm still
    starts from the uniform policy. In the low-rank case ``readout_in`` is random
    (``N(0, 1/n_neurons)``): with both factors at zero neither would get a gradient.
    The parameters live on the policy itself, so names and state dicts of the
    default readout are what they always were."""

    def _init_readout(
        self, n_agents: int, n_neurons: int, n_out: int, n_bins: int, nodes: str, rank: int,
        generator: torch.Generator,
    ) -> None:
        if nodes not in READOUTS:
            raise ValueError(f"readout must be one of {READOUTS}, got {nodes!r}")
        self.readout_nodes = nodes
        width = n_out
        if nodes == "all":
            if rank < 1:
                raise ValueError("a readout from all neurons needs readout_rank >= 1")
            self.readout_in = nn.Parameter(
                torch.randn(n_agents, n_neurons, rank, generator=generator) / math.sqrt(n_neurons)
            )
            width = rank
        self.readout = nn.Parameter(torch.zeros(n_agents, width, n_bins))
        self.readout_bias = nn.Parameter(torch.zeros(n_agents, 1, n_bins))

    def _read(self, h: torch.Tensor) -> torch.Tensor:
        """``(M, B, N)`` neuron states to ``(M, B, n_bins)`` logits."""
        if self.readout_nodes == "all":
            z = torch.bmm(h, self.readout_in)
        else:
            z = h.index_select(2, self.output_nodes)
        return torch.baddbmm(self.readout_bias, z, self.readout)


def dense_param_count(d_in: int, hidden: list[int], n_bins: int) -> int:
    sizes = [d_in, *hidden, n_bins]
    return sum(a * b + b for a, b in zip(sizes[:-1], sizes[1:]))


def match_dense_width(target: int, d_in: int, n_layers: int, n_bins: int) -> int:
    """Width ``H`` of an ``n_layers``-hidden-layer MLP whose count is closest to ``target``."""
    best, best_gap = 1, float("inf")
    for h in range(1, 4096):
        gap = abs(dense_param_count(d_in, [h] * n_layers, n_bins) - target)
        if gap < best_gap:
            best, best_gap = h, gap
    return best


class SparseCircuitPolicy(GraphReadoutMixin, Policy):
    """Recurrent network whose neuron-to-neuron weights live only on a fixed edge set.

    ``N`` neurons, directed edges ``(pre -> post)``. Context enters through a dense
    projection onto the *input* neurons; after ``unroll_steps`` synchronous updates

        h <- act(A h + W_in x + bias),     A[post, pre] = w_e for e in edges,

    the logits are read out from the *output* neurons. Gradient descent changes the
    ``E`` edge weights; entries of ``A`` off the edge set are structurally zero and
    are not parameters, so the network can never create a connection the graph
    does not have.

    ``edge_sign`` (+1, -1, 0 per edge) activates sign constraints when
    ``sign_constrained``: +1 edges are ``softplus(raw) >= 0``, -1 edges are
    ``-softplus(raw) <= 0``, 0 (modulatory / unknown transmitter) edges are free.
    """

    def __init__(
        self,
        n_agents: int,
        d_in: int,
        n_neurons: int,
        edges: np.ndarray,
        input_nodes: np.ndarray,
        output_nodes: np.ndarray,
        n_bins: int,
        unroll_steps: int,
        activation: str,
        init_gain: float,
        generator: torch.Generator,
        edge_sign: np.ndarray | None = None,
        sign_constrained: bool = False,
        sparse_max_batch: int = 8,
        readout_nodes: str = "output",
        readout_rank: int = 0,
    ) -> None:
        super().__init__()
        self.n_agents, self.n_bins, self.n_neurons = n_agents, n_bins, n_neurons
        self.sparse_max_batch = sparse_max_batch
        self.unroll_steps = unroll_steps
        self.act = ACTIVATIONS[activation]
        edges = np.asarray(edges, dtype=np.int64)
        if edges.ndim != 2 or edges.shape[1] != 2:
            raise ValueError("edges must be an (E, 2) array of (pre, post)")
        if len({(int(a), int(b)) for a, b in edges}) != len(edges):
            raise ValueError("duplicate edges")
        pre, post = edges[:, 0], edges[:, 1]
        self.register_buffer("flat_index", torch.as_tensor(post * n_neurons + pre))
        self.register_buffer("input_nodes", torch.as_tensor(np.asarray(input_nodes, dtype=np.int64)))
        self.register_buffer("output_nodes", torch.as_tensor(np.asarray(output_nodes, dtype=np.int64)))

        in_degree = np.bincount(post, minlength=n_neurons).astype(np.float64)
        std = init_gain / np.sqrt(np.maximum(in_degree[post], 1.0))
        w0 = torch.randn(n_agents, len(edges), generator=generator) * torch.as_tensor(std, dtype=torch.float32)

        sign = np.zeros(len(edges)) if edge_sign is None else np.asarray(edge_sign, dtype=np.float64)
        self.sign_constrained = bool(sign_constrained)
        self.register_buffer("edge_sign", torch.as_tensor(sign, dtype=torch.float32))
        if self.sign_constrained:
            signed = self.edge_sign != 0
            mag = w0.abs().clamp_min(1e-4)
            raw = torch.where(signed, torch.log(torch.expm1(mag)), w0)  # softplus^-1(|w0|)
            self.edge_raw = nn.Parameter(raw)
        else:
            self.edge_raw = nn.Parameter(w0)

        n_in = len(input_nodes)
        self.w_in = nn.Parameter(
            torch.randn(n_agents, d_in, n_in, generator=generator) / math.sqrt(max(d_in, 1))
        )
        self.bias = nn.Parameter(torch.zeros(n_agents, 1, n_neurons))
        # Built last, so the default readout draws nothing from the generator and every
        # earlier initialisation is bit-identical with or without the low-rank option.
        self._init_readout(n_agents, n_neurons, len(output_nodes), n_bins, readout_nodes, readout_rank, generator)

    def edge_weights(self) -> torch.Tensor:
        if not self.sign_constrained:
            return self.edge_raw
        signed = self.edge_sign != 0
        return torch.where(signed, self.edge_sign * F.softplus(self.edge_raw), self.edge_raw)

    def adjacency(self) -> torch.Tensor:
        """Dense ``(M, N, N)`` view of the weights, ``A[m, post, pre]``."""
        m, n = self.n_agents, self.n_neurons
        flat = self.edge_raw.new_zeros(m, n * n)
        flat = flat.index_put((torch.arange(m).unsqueeze(1), self.flat_index.unsqueeze(0)), self.edge_weights())
        return flat.view(m, n, n)

    def forward(self, features: torch.Tensor, type_index: torch.Tensor) -> torch.Tensor:
        """Edge-list message passing: ``(A h)[post] = sum_e w_e h[pre_e]``.

        Numerically identical to ``act(h @ A^T + drive)`` with the dense adjacency
        (``test_sparse_forward_matches_dense``). O(E) per step, which is faster for
        few contexts; for many contexts (Game B) the dense matmul is faster, so we
        switch at ``sparse_max_batch``. Only speed differs, never the function.
        """
        m, b, _ = features.shape
        if b > self.sparse_max_batch:
            return self.forward_dense(features)
        w = self.edge_weights().unsqueeze(1)  # (M, 1, E)
        pre = self.flat_index % self.n_neurons
        post = self.flat_index // self.n_neurons
        drive = features.new_zeros(m, b, self.n_neurons)
        drive = drive.index_copy(2, self.input_nodes, torch.bmm(features, self.w_in))
        drive = drive + self.bias
        h = features.new_zeros(m, b, self.n_neurons)
        for _ in range(self.unroll_steps):
            msg = h.index_select(2, pre) * w
            h = self.act(features.new_zeros(m, b, self.n_neurons).index_add(2, post, msg) + drive)
        return self._read(h)

    def forward_dense(self, features: torch.Tensor) -> torch.Tensor:
        """Reference implementation through the dense adjacency (tests only)."""
        m, b, _ = features.shape
        a_t = self.adjacency().transpose(1, 2)
        drive = features.new_zeros(m, b, self.n_neurons)
        drive = drive.index_copy(2, self.input_nodes, torch.bmm(features, self.w_in)) + self.bias
        h = features.new_zeros(m, b, self.n_neurons)
        for _ in range(self.unroll_steps):
            h = self.act(torch.bmm(h, a_t) + drive)
        return self._read(h)
