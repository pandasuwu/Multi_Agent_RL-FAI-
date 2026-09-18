"""Build demo/index.html: the browser demo with the real circuit and predicted curves embedded.

    python scripts/build_demo.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config  # noqa: E402
from src.connectome.circuit import load_circuit  # noqa: E402
from src.games import GameA, TimeGrid, asymmetric_fixed_point, game_a_analytic_policy  # noqa: E402


def main() -> None:
    cfg = load_config("configs/base.yaml")
    data = load_circuit(cfg["connectome"])
    g = cfg["game"]
    game = GameA(g["prize"], g["cost_rate"], TimeGrid(g["t_max"], g["n_bins"]))
    stage = [str(s) for s in data["stage"]]
    group = []
    for s in stage:
        group.append({"core:ppl1": "ppl1", "core:mbon": "mbon", "core:tk": "tk"}.get(s, s))
    payload = {
        "n": int(len(data["node_ids"])),
        "edges": data["edges"].astype(int).ravel().tolist(),
        "group": group,
        "type": [str(t) for t in data["node_type"]],
        "tMax": g["t_max"],
        "nBins": g["n_bins"],
        "ideal": np.round(game_a_analytic_policy(game), 8).tolist(),
        "predicted": {
            str(int(r)): np.round(asymmetric_fixed_point(game, float(r))[0], 8).tolist() for r in (1, 2, 4, 8)
        },
    }
    template = (ROOT / "demo" / "template.html").read_text(encoding="utf-8")
    out = template.replace("/*__DATA__*/null", json.dumps(payload, separators=(",", ":")))
    (ROOT / "demo" / "index.html").write_text(out, encoding="utf-8")
    print(f"wrote demo/index.html ({len(out) / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
