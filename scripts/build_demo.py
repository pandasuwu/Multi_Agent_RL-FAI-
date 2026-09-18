"""Build demo/index.html from scripts/demo_template.html.

Everything the page shows that is not simulated live comes from the project's own
files, never typed in: the circuit (data/connectome), the equilibria and predicted
asymmetric rest points (src/games), graph statistics of the circuit and its controls
(src/connectome), and the study's findings (results/tables).

    python scripts/build_demo.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config  # noqa: E402
from src.connectome.circuit import circuit_spec_for_arm, load_circuit  # noqa: E402
from src.games import GameA, TimeGrid, asymmetric_fixed_point, game_a_analytic_policy  # noqa: E402

TEMPLATE = ROOT / "scripts" / "demo_template.html"
OUT = ROOT / "demo" / "index.html"
TABLES = ROOT / "results" / "tables"
ARMS = ["tabular", "dense", "connectome", "connectome+sign", "rewired", "rewired+sign", "random_sparse", "random_sparse+sign"]


def graph_stats(conn_cfg: dict) -> list[dict]:
    real = circuit_spec_for_arm(conn_cfg, "connectome", 0)
    real_set = {tuple(e) for e in real.edges}
    rows = []
    for arm, label in (("connectome", "Real fly wiring"), ("rewired", "Shuffled fly"), ("random_sparse", "Random sparse")):
        spec = circuit_spec_for_arm(conn_cfg, arm, 0)
        g = nx.DiGraph()
        g.add_nodes_from(range(spec.n_neurons))
        g.add_edges_from(map(tuple, spec.edges))
        indeg = np.bincount(spec.edges[:, 1], minlength=spec.n_neurons)
        rows.append({
            "label": label,
            "edges": int(len(spec.edges)),
            "reciprocity": round(float(nx.reciprocity(g)), 3),
            "clustering": round(float(nx.average_clustering(g.to_undirected())), 3),
            "max_in": int(indeg.max()),
            "overlap": round(len({tuple(e) for e in spec.edges} & real_set) / len(real_set), 3),
            "degrees_kept": arm != "random_sparse",
        })
    return rows


def findings() -> dict:
    per = pd.read_csv(TABLES / "per_arm.csv")
    out: dict = {"per_arm": {}, "contrasts": [], "asym": []}
    for exp in ("game_a_exact", "game_a_sampled", "game_b_exact", "game_b_sampled"):
        sub = per[per["experiment"] == exp].set_index("arm")
        out["per_arm"][exp] = {
            "oracle": float(sub.loc["oracle", "final_exploitability"]),
            "n_seeds": int(sub["n_seeds"].max()),
            "arms": [
                {"arm": a, "mean": float(sub.loc[a, "final_exploitability"]), "lo": float(sub.loc[a, "final_exploitability_lo"]),
                 "hi": float(sub.loc[a, "final_exploitability_hi"])}
                for a in ARMS if a in sub.index
            ],
        }
    pw = pd.read_csv(TABLES / "pairwise.csv")
    pw = pw[pw["metric"] == "final_exploitability"]
    settings = [
        ("game_a_exact", "Game A, exact"),
        ("robust_game_a_sampled_shared_lr", "Game A, sampled"),
        ("game_b_exact", "Game B, exact"),
        ("robust_game_b_sampled_shared_lr", "Game B, sampled"),
    ]
    for exp, label in settings:
        for contrast, name in (("connectome - rewired", "vs shuffled"), ("connectome - random_sparse", "vs random"),
                               ("connectome+sign - rewired+sign", "vs shuffled (signed)"),
                               ("connectome+sign - random_sparse+sign", "vs random (signed)")):
            row = pw[(pw["experiment"] == exp) & (pw["contrast"] == contrast)]
            if row.empty:
                continue
            r = row.iloc[0]
            out["contrasts"].append({"setting": label, "contrast": name, "diff": float(r["diff"]), "lo": float(r["diff_lo"]),
                                     "hi": float(r["diff_hi"]), "p": float(r["p_holm"]), "sig": bool(r["distinguishable"])})
    asym = pd.read_csv(TABLES / "asym_per_arm.csv")
    for rho, sub in asym.groupby("loss_win_ratio"):
        out["asym"].append({
            "rho": float(rho),
            "predicted": float(sub["predicted_exploitability"].iloc[0]),
            "mean": float(sub["final_exploitability"].mean()),
            "min": float(sub["final_exploitability"].min()),
            "max": float(sub["final_exploitability"].max()),
            "loser": float(sub["wl_dmu_per_loss_all"].mean()),
        })
    return out


def main() -> None:
    cfg = load_config("configs/base.yaml")
    conn = cfg["connectome"]
    data = load_circuit(conn)
    g = cfg["game"]
    game = GameA(g["prize"], g["cost_rate"], TimeGrid(g["t_max"], g["n_bins"]))
    group = [{"core:ppl1": "ppl1", "core:mbon": "mbon", "core:tk": "tk"}.get(str(s), str(s)) for s in data["stage"]]
    prov = json.loads((ROOT / conn["provenance"]).read_text(encoding="utf-8"))
    nodes = pd.read_csv(ROOT / "data" / "connectome" / "give_up_circuit_nodes.csv")
    core = nodes[nodes["stage"].str.startswith("core")]
    identities = (
        core.groupby("type")
        .agg(count=("bodyId", "size"), transmitter=("transmitter", "first"), ids=("bodyId", lambda s: ", ".join(map(str, s))))
        .reset_index()
        .to_dict("records")
    )
    payload = {
        "n": int(len(data["node_ids"])),
        "edges": data["edges"].astype(int).ravel().tolist(),
        "group": group,
        "tMax": g["t_max"],
        "nBins": g["n_bins"],
        "ideal": np.round(game_a_analytic_policy(game), 8).tolist(),
        "predicted": {str(int(r)): np.round(asymmetric_fixed_point(game, float(r))[0], 8).tolist() for r in (1, 2, 4, 8)},
        "graphStats": graph_stats(conn),
        "identities": identities,
        "counts": prov["counts"],
        "sha": prov["subgraph_sha256"][:12],
        "findings": findings(),
    }
    html = TEMPLATE.read_text(encoding="utf-8").replace("/*__DATA__*/null", json.dumps(payload, separators=(",", ":")))
    OUT.write_text(html, encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)} ({len(html) / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
