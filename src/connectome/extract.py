"""Extract the fly give-up circuit subgraph from MaleCNS v1.0 and cache it with provenance.

Definition (all parameters in the ``connectome`` config section; rationale in
docs/circuit.md):

1. Core cell types, resolved by exact type name / regex against the MaleCNS
   ``type`` annotation: PPL1 dopaminergic neurons, the V2 MBONs MBON18/MBON19 and
   AVLP727m (the MaleCNS type of Tk-GAL4^FruM neurons, per Ventimiglia et al. 2026).
2. Connecting neurons: every neuron on a directed path of at most
   ``path.max_synapses`` synapses from the ``path.source`` population to the
   ``path.target`` population, every edge having at least ``min_synapses``
   synapses (the paper's threshold). Together with 1 this is the exact circuit.
3. One-hop expansion (only if the exact circuit has fewer than
   ``expansion.target_range[0]`` neurons): add every neuron that exchanges at
   least ``expansion.min_exchanged_synapses`` synapses (in + out, over edges of at
   least ``min_synapses``) with the exact circuit.
4. The subgraph is the induced graph on the selected neurons, keeping edges with
   at least ``min_synapses`` synapses. Self-loops are dropped.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import subprocess
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.feather as feather

from ..config import REPO_ROOT
from .source import ConnectomeDataError, ensure_raw_file, sha256_file


def graph_checksum(node_ids: np.ndarray, edges: np.ndarray) -> str:
    h = hashlib.sha256()
    h.update(np.asarray(node_ids, dtype=np.int64).tobytes())
    h.update(np.asarray(edges, dtype=np.int64).tobytes())
    return h.hexdigest()


def resolve_types(annotations: pd.DataFrame, spec: dict[str, Any]) -> pd.DataFrame:
    """Rows of the annotation table matching a population spec (types list or regex)."""
    types = annotations["type"].astype("string")
    if "types" in spec:
        mask = types.isin(spec["types"])
    elif "type_regex" in spec:
        mask = types.str.match(spec["type_regex"]).fillna(False)
    else:
        raise ValueError(f"population spec needs 'types' or 'type_regex': {spec}")
    rows = annotations[mask.to_numpy(dtype=bool)]
    if rows.empty:
        raise ConnectomeDataError(f"no MaleCNS neurons match {spec}")
    return rows


def path_neurons(edges: pd.DataFrame, source: set[int], target: set[int], max_synapses: int) -> set[int]:
    """Neurons (excluding endpoints) on directed paths source -> ... -> target of length <= max_synapses."""
    succ = edges.groupby("body_pre")["body_post"].apply(set).to_dict()
    pred = edges.groupby("body_post")["body_pre"].apply(set).to_dict()
    # forward distances from source and backward distances to target (BFS, capped)
    def bfs(start: set[int], nbrs: dict[int, set[int]]) -> dict[int, int]:
        dist = {n: 0 for n in start}
        frontier = set(start)
        for d in range(1, max_synapses + 1):
            nxt = set()
            for n in frontier:
                for m in nbrs.get(n, ()):
                    if m not in dist:
                        dist[m] = d
                        nxt.add(m)
            frontier = nxt
        return dist

    fwd, bwd = bfs(source, succ), bfs(target, pred)
    on_path = {n for n in fwd if n in bwd and fwd[n] + bwd[n] <= max_synapses}
    return on_path - source - target


def build_subgraph(conn_cfg: dict[str, Any], download: bool = False) -> dict[str, Any]:
    """Run the extraction and write the cache + provenance. Returns the provenance."""
    ann_path = ensure_raw_file(conn_cfg, "annotations", download)
    nt_path = ensure_raw_file(conn_cfg, "neurotransmitters", download)
    w_path = ensure_raw_file(conn_cfg, "weights", download)

    annotations = feather.read_table(ann_path, columns=["bodyId", "type", "instance", "superclass", "status"]).to_pandas()
    weights = feather.read_table(w_path, columns=["body_pre", "body_post", "weight"]).to_pandas()
    min_syn = int(conn_cfg["min_synapses"])
    strong = weights[weights["weight"] >= min_syn]

    populations: dict[str, pd.DataFrame] = {
        name: resolve_types(annotations, spec) for name, spec in conn_cfg["core"].items()
    }
    core = set().union(*[set(df["bodyId"]) for df in populations.values()])

    path_cfg = conn_cfg["path"]
    connecting = path_neurons(
        strong,
        set(populations[path_cfg["source"]]["bodyId"]),
        set(populations[path_cfg["target"]]["bodyId"]),
        int(path_cfg["max_synapses"]),
    )
    exact_circuit = core | connecting

    exp_cfg = conn_cfg["expansion"]
    lo, hi = exp_cfg["target_range"]
    expanded: set[int] = set()
    if len(exact_circuit) < lo:
        ids = np.fromiter(exact_circuit, dtype=np.int64)
        into = strong[strong["body_post"].isin(ids) & ~strong["body_pre"].isin(ids)].groupby("body_pre")["weight"].sum()
        out = strong[strong["body_pre"].isin(ids) & ~strong["body_post"].isin(ids)].groupby("body_post")["weight"].sum()
        exchanged = into.add(out, fill_value=0)
        expanded = set(exchanged[exchanged >= int(exp_cfg["min_exchanged_synapses"])].index.astype(np.int64))
    selected = np.array(sorted(exact_circuit | expanded), dtype=np.int64)
    if not lo <= len(selected) <= hi:
        raise ConnectomeDataError(
            f"subgraph has {len(selected)} neurons, outside the configured range {exp_cfg['target_range']}"
        )

    sub = strong[strong["body_pre"].isin(selected) & strong["body_post"].isin(selected)]
    n_self = int((sub["body_pre"] == sub["body_post"]).sum())
    sub = sub[sub["body_pre"] != sub["body_post"]].sort_values(["body_pre", "body_post"])
    index = {int(b): k for k, b in enumerate(selected)}
    edges = np.stack([sub["body_pre"].map(index).to_numpy(), sub["body_post"].map(index).to_numpy()], axis=1)
    synapses = sub["weight"].to_numpy(dtype=np.int64)

    # node table: identity, stage, role, transmitter, sign (Dale's law, by presynaptic neuron)
    table = annotations.set_index("bodyId").reindex(selected)
    nt = feather.read_table(nt_path, columns=["body", "consensus_nt"]).to_pandas().set_index("body")["consensus_nt"]
    transmitter = nt.reindex(selected).fillna("unclear").astype(str).to_numpy()
    sign_map = {k: int(v) for k, v in conn_cfg["sign_map"].items()}
    node_sign = np.array([sign_map.get(t, 0) for t in transmitter], dtype=np.int64)
    population_of = {int(b): name for name, df in populations.items() for b in df["bodyId"]}
    stage = np.array(
        [
            "core:" + population_of[int(b)] if int(b) in population_of else ("connecting" if int(b) in connecting else "expansion")
            for b in selected
        ]
    )
    roles = conn_cfg["roles"]
    role = np.array(
        [
            "input" if population_of.get(int(b)) == roles["input"] else "output" if population_of.get(int(b)) == roles["output"] else "hidden"
            for b in selected
        ]
    )
    nodes = pd.DataFrame(
        {
            "index": np.arange(len(selected)),
            "bodyId": selected,
            "type": table["type"].astype(str).to_numpy(),
            "instance": table["instance"].astype(str).to_numpy(),
            "superclass": table["superclass"].astype(str).to_numpy(),
            "stage": stage,
            "role": role,
            "transmitter": transmitter,
            "sign": node_sign,
            "in_degree": np.bincount(edges[:, 1], minlength=len(selected)),
            "out_degree": np.bincount(edges[:, 0], minlength=len(selected)),
        }
    )

    cache = REPO_ROOT / conn_cfg["cache"]
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache,
        node_ids=selected,
        edges=edges,
        synapses=synapses,
        node_sign=node_sign,
        role=role,
        stage=stage,
        node_type=nodes["type"].to_numpy().astype(str),
    )
    nodes.to_csv(cache.with_name(cache.stem + "_nodes.csv"), index=False)
    pd.DataFrame(
        {"pre": edges[:, 0], "post": edges[:, 1], "pre_bodyId": selected[edges[:, 0]], "post_bodyId": selected[edges[:, 1]], "synapses": synapses}
    ).to_csv(cache.with_name(cache.stem + "_edges.csv"), index=False)

    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=REPO_ROOT).stdout.strip()
    except OSError:
        commit = "unknown"
    provenance = {
        "dataset": conn_cfg["dataset"],
        "source_url": conn_cfg["base_url"],
        "license": "CC-BY 4.0 (Berg et al., Cell 2026, MaleCNS v1.0)",
        "raw_files": {
            key: {"name": conn_cfg["files"][key]["name"], "bytes": int(conn_cfg["files"][key]["bytes"]), "sha256": sha256_file(p)}
            for key, p in (("annotations", ann_path), ("neurotransmitters", nt_path), ("weights", w_path))
        },
        "query": {k: conn_cfg[k] for k in ("min_synapses", "core", "path", "expansion", "roles", "sign_map")},
        "resolved_core": {
            name: sorted(f"{t} ({int(b)})" for t, b in zip(df["type"].astype(str), df["bodyId"]))
            for name, df in populations.items()
        },
        "counts": {
            "core_neurons": len(core),
            "connecting_neurons": len(connecting),
            "exact_circuit_neurons": len(exact_circuit),
            "expansion_neurons": len(expanded),
            "nodes": int(len(selected)),
            "edges": int(len(edges)),
            "self_loops_dropped": n_self,
            "synapses_in_edges": int(synapses.sum()),
            "input_nodes": int((role == "input").sum()),
            "output_nodes": int((role == "output").sum()),
            "signed_excitatory_nodes": int((node_sign > 0).sum()),
            "signed_inhibitory_nodes": int((node_sign < 0).sum()),
            "unsigned_nodes": int((node_sign == 0).sum()),
        },
        "subgraph_sha256": graph_checksum(selected, edges),
        "created": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "code_commit": commit,
    }
    with open(REPO_ROOT / conn_cfg["provenance"], "w", encoding="utf-8") as fh:
        json.dump(provenance, fh, indent=2)
    return provenance


def type_counts(nodes: pd.DataFrame, stage_prefix: str) -> dict[str, int]:
    sub = nodes[nodes["stage"].str.startswith(stage_prefix)]
    return sub["type"].value_counts().to_dict()


__all__ = ["build_subgraph", "graph_checksum", "path_neurons", "resolve_types", "type_counts"]
