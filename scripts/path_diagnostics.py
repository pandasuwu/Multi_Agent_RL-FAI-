"""Input-to-output connectivity of the real circuit and every control graph.

Degree-preserving rewiring keeps each neuron's in- and out-degree but does not, by
itself, guarantee that the rewired graph still connects the PPL1 input neurons to the
AVLP727m output neurons. This script checks it for exactly the graphs the experiments
used (the real circuit and the rewired / random graphs of seeds 0-19, rebuilt through
the same code path as the runs, and matched to the runs by graph checksum).

Only paths that fit inside the unrolled network can carry the input: the context
enters PPL1 on the first of the ``unroll_steps`` updates and the policy is read from
AVLP727m after the last one, so a path needs at most ``unroll_steps - 1`` synapses.

Per graph:
* ``outputs_reachable``: AVLP727m neurons reachable from any PPL1 neuron (any length);
* ``outputs_reachable_in_time``: the same within ``unroll_steps - 1`` synapses;
* ``pairs_reachable_in_time``: (PPL1, AVLP727m) pairs connected within that many;
* ``shortest_min`` / ``shortest_median``: shortest PPL1 -> AVLP727m path over pairs;
* ``walks_in_time``: number of directed walks of length 1..unroll_steps-1 from
  PPL1 to AVLP727m (sum of the binary adjacency powers), a count of routes;
* ``reach_one_hop_hidden``: neurons a PPL1 neuron synapses onto directly.

Then, for the control arms, the Spearman correlation across seeds between each graph's
connectivity and that run's final exploitability, to see whether runs on better-
connected control graphs do better.

Writes ``results/tables/path_diagnostics.csv`` (one row per graph),
``results/tables/path_summary.csv`` (per graph family) and
``results/tables/path_vs_performance.csv``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import REPO_ROOT, load_config  # noqa: E402
from src.connectome.circuit import circuit_spec_for_arm  # noqa: E402

SEEDS = range(20)
PERF_EXPERIMENTS = (
    "game_b_exact",
    "robust_game_b_exact_shared_lr",
    "game_b_sampled",
    "robust_game_b_sampled_shared_lr",
    "ppo_game_b_sampled",
    "readout_game_b_exact",
    "flygm_game_b_exact",
    "flygm_game_b_sampled",
)
CONNECTIVITY = ["pairs_reachable_in_time", "walks_in_time", "shortest_median"]


def diagnostics(spec, depth: int) -> dict[str, float]:
    n = spec.n_neurons
    adj = np.zeros((n, n), dtype=np.int64)  # adj[pre, post]
    adj[spec.edges[:, 0], spec.edges[:, 1]] = 1
    src, dst = spec.input_nodes, spec.output_nodes
    # BFS distances from every input neuron
    dist = np.full((len(src), n), np.inf)
    for i, s in enumerate(src):
        dist[i, s] = 0
        frontier, d = np.array([s]), 0
        while len(frontier):
            d += 1
            nxt = np.flatnonzero(adj[frontier].any(axis=0) & np.isinf(dist[i]))
            dist[i, nxt] = d
            frontier = nxt
    pair = dist[:, dst]  # (inputs, outputs)
    walks = 0.0
    reach = np.zeros((len(src), n), dtype=float)
    reach[np.arange(len(src)), src] = 1.0
    for _ in range(depth):
        reach = reach @ adj
        walks += reach[:, dst].sum()
    finite = pair[np.isfinite(pair)]
    return {
        "outputs_reachable": int(np.isfinite(pair).any(axis=0).sum()),
        "outputs_reachable_in_time": int((pair <= depth).any(axis=0).sum()),
        "pairs_reachable": int(np.isfinite(pair).sum()),
        "pairs_reachable_in_time": int((pair <= depth).sum()),
        "n_pairs": int(pair.size),
        "shortest_min": float(finite.min()) if finite.size else np.inf,
        "shortest_median": float(np.median(pair)) if finite.size else np.inf,
        "walks_in_time": walks,
        "log10_walks_in_time": float(np.log10(walks)) if walks > 0 else -np.inf,
        "reach_one_hop_hidden": int(adj[src].any(axis=0).sum()),
    }


def main() -> None:
    cfg = load_config(REPO_ROOT / "configs" / "base.yaml")
    depth = int(cfg["network"]["unroll_steps"]) - 1
    rows = []
    for graph, seeds in (("connectome", [0]), ("rewired", SEEDS), ("rewired_block", SEEDS), ("connectome_shortcut", SEEDS), ("rewired_noshortcut", SEEDS), ("random_sparse", SEEDS)):
        for seed in seeds:
            spec = circuit_spec_for_arm(cfg["connectome"], graph, seed)
            rows.append({"graph": graph, "seed": seed, "graph_id": spec.graph_id, "max_path_length_used": depth, **diagnostics(spec, depth)})
    per_graph = pd.DataFrame(rows)

    summary = []
    for graph, sub in per_graph.groupby("graph", sort=False):
        row = {"graph": graph, "n_graphs": len(sub)}
        for col in ("outputs_reachable_in_time", "pairs_reachable_in_time", "shortest_min", "shortest_median", "log10_walks_in_time", "reach_one_hop_hidden"):
            row[f"{col}_min"] = sub[col].min()
            row[f"{col}_median"] = sub[col].median()
            row[f"{col}_max"] = sub[col].max()
        row["all_outputs_reachable_in_time"] = bool((sub["outputs_reachable_in_time"] == 5).all())
        summary.append(row)
    summary = pd.DataFrame(summary)

    # does connectivity of a control graph predict how well its run did?
    perf_rows = []
    by_id = per_graph.set_index("graph_id")
    for exp in PERF_EXPERIMENTS:
        path = REPO_ROOT / "results" / exp / "summary.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path)
        df = df[df["graph_id"].isin(by_id.index)]
        for arm, sub in df.groupby("arm"):
            if sub["graph_id"].nunique() < 5:
                continue  # the real circuit is one graph: nothing to correlate
            merged = sub.join(by_id[CONNECTIVITY], on="graph_id")
            for col in CONNECTIVITY:
                if merged[col].nunique() < 2:
                    continue
                rho, p = stats.spearmanr(merged[col], merged["final_exploitability"])
                perf_rows.append({"experiment": exp, "arm": arm, "connectivity": col, "n": len(merged), "spearman_rho": rho, "p": p})
    perf = pd.DataFrame(perf_rows)
    if len(perf):
        from src.metrics.stats import holm

        perf["p_holm"] = holm(perf["p"].tolist())

    tables = REPO_ROOT / "results" / "tables"
    per_graph.to_csv(tables / "path_diagnostics.csv", index=False, float_format="%.6g")
    summary.to_csv(tables / "path_summary.csv", index=False, float_format="%.6g")
    perf.to_csv(tables / "path_vs_performance.csv", index=False, float_format="%.6g")
    pd.set_option("display.width", 200)
    print(summary.T.to_string())
    if len(perf):
        print(perf.sort_values("p").head(12).to_string(index=False))


if __name__ == "__main__":
    main()
