"""Welfare between two different agents, for sampled runs made before runs stored it.

A sampled run record holds only the population *mixture* as its final policy, and the
welfare of a match between two different agents cannot be computed from a mixture (see
``src/metrics/welfare.py``). Runs made from now on store that welfare themselves. For
the committed runs this script retrains each one from the config snapshot in its record
(same arm, seed and learning rate, so the same trajectory up to floating point) and
keeps only the welfare of its final population, next to the final exploitability of
the retrained run and of the committed one, so any drift between the two is visible.

    python scripts/welfare_rerun.py --workers 6
    python scripts/welfare_rerun.py --only popsize_game_a_sampled game_a_sampled

Per-run results go to ``results/_scratch/welfare_rerun/`` (resumable, not committed)
and are collected into ``results/tables/welfare_population.csv``, which
``scripts/welfare.py`` reads.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402
import torch  # noqa: E402

from src.agents.factory import GRAPH_ARMS, build_policy  # noqa: E402
from src.games import build_game  # noqa: E402
from src.training.runner import (  # noqa: E402
    circuit_for,
    learning_rate,
    n_agents_for,
    needs_budget,
    parameter_budget,
)
from src.training.trainer import train_run  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
SCRATCH = RES / "_scratch" / "welfare_rerun"
TABLE = RES / "tables" / "welfare_population.csv"

# cheapest first; the last entries are hours each
ORDER = [
    "popsize_game_a_sampled",
    "robust_game_a_sampled_shared_lr",
    "popnash_game_a_sampled",
    "game_a_sampled",
    "game_b_sampled",
    "robust_game_b_sampled_shared_lr",
    "flygm_game_a_sampled",
    "robust_ppo_game_a_sampled_shared_lr",
    "robust_ppo_game_b_sampled_shared_lr",
    "ppo_game_a_sampled",
    "ppo_game_b_sampled",
    "flygm_game_b_sampled",
]


def arm_from_label(label: str) -> dict:
    name, _, flag = label.partition("+")
    return {"name": name, "sign_constrained": True} if flag == "sign" else {"name": name}


def rerun(path: str) -> dict:
    torch.set_num_threads(1)
    with open(path, encoding="utf-8") as fh:
        rec = json.load(fh)
    cfg, seed, arm = rec["config"], rec["seed"], arm_from_label(rec["arm"])
    game = build_game(cfg["game"])
    # the same steps as src.training.runner.run_job
    circuit = circuit_for(cfg, arm, seed) if arm["name"] in GRAPH_ARMS else None
    budget = parameter_budget(cfg) if needs_budget(cfg, arm["name"]) else None
    torch.manual_seed(seed)
    policy = build_policy(arm, game, n_agents_for(cfg), cfg["network"], seed, circuit, budget)
    started = time.perf_counter()
    out = train_run(game, policy, cfg["training"], cfg["metrics"], learning_rate(cfg, arm), seed)
    row = {
        "experiment": rec["experiment"],
        "run_id": rec["run_id"],
        "arm": rec["arm"],
        "seed": seed,
        **{f"welfare_{k}" if k != "welfare" else k: v for k, v in out.welfare.items()},
        "rerun_final_exploitability": out.summary["final_exploitability"],
        "committed_final_exploitability": rec["summary"]["final_exploitability"],
        "same_graph": (circuit.graph_id if circuit is not None else None) == rec.get("graph_id"),
        "torch": torch.__version__,
        "wall_seconds": time.perf_counter() - started,
    }
    out_dir = SCRATCH / rec["experiment"]
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp = out_dir / f"{rec['run_id']}.json.partial"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(row, fh)
    os.replace(tmp, out_dir / f"{rec['run_id']}.json")
    return row


def collect() -> pd.DataFrame:
    rows = []
    for path in sorted(SCRATCH.glob("*/*.json")):
        with open(path, encoding="utf-8") as fh:
            rows.append(json.load(fh))
    table = pd.DataFrame(rows)
    if len(table):
        table = table.sort_values(["experiment", "run_id"]).reset_index(drop=True)
        TABLE.parent.mkdir(exist_ok=True)
        table.to_csv(TABLE, index=False, float_format="%.10g")
    return table


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", nargs="*", default=None, help="experiments to do (default: all sampled ones)")
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    for exp in args.only or ORDER:
        runs = sorted((RES / exp / "runs").glob("*.json"))
        done = {p.stem for p in (SCRATCH / exp).glob("*.json")} if (SCRATCH / exp).exists() else set()
        todo = [str(p) for p in runs if "oracle" not in p.stem and p.stem not in done]
        print(f"[{exp}] {len(todo)} runs to redo on {args.workers} workers", flush=True)
        started = time.perf_counter()
        if todo:
            with mp.get_context("spawn").Pool(args.workers) as pool:
                for k, _ in enumerate(pool.imap_unordered(rerun, todo), 1):
                    if k == len(todo) or k % max(1, len(todo) // 10) == 0:
                        print(f"[{exp}] {k}/{len(todo)}, {time.perf_counter() - started:.0f}s", flush=True)
        table = collect()
        sub = table[table["experiment"] == exp]
        drift = (sub["rerun_final_exploitability"] - sub["committed_final_exploitability"]).abs()
        print(f"[{exp}] done: {len(sub)} runs, identical to committed {int((drift == 0).sum())}, "
              f"max drift in final exploitability {drift.max():.2g}", flush=True)


if __name__ == "__main__":
    main()
