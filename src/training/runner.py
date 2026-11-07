"""Expand an experiment config into runs, execute them in parallel, and write results.

Layout of ``results/<experiment>/``:

* ``runs/<run_id>.json``: one file per run with the *complete resolved config
  snapshot*, the seed, the arm, the parameter count, the summary, the evaluation
  curve and (sampled Game A) the winner/loser statistics.
* ``summary.csv``: one row per run.
* ``curves.csv``: long format, one row per (run, evaluation step).
* ``config.yaml``: the experiment config as loaded (after inheritance).
"""

from __future__ import annotations

import copy
import itertools
import json
import multiprocessing as mp
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml

from ..agents.factory import (
    FLYGM_ARMS,
    GRAPH_ARMS,
    NETWORK_ARMS,
    arm_label,
    assert_matched,
    build_policy,
    circuit_param_count,
    graph_name,
    readout_settings,
)
from ..config import REPO_ROOT, config_hash, deep_merge
from ..games.equilibrium import game_a_analytic_policy, game_b_analytic_policy
from ..games.war_of_attrition import GameA, build_game
from .evaluate import evaluate
from .trainer import summarise, train_run

CURVE_COLUMNS = (
    "step",
    "exploitability",
    "mixture_exploitability",
    "tv",
    "w1",
    "mean_quit",
    "relative_utility_loss",
    "l2",
)


def set_nested(cfg: dict[str, Any], dotted: str, value: Any) -> None:
    node = cfg
    keys = dotted.split(".")
    for key in keys[:-1]:
        node = node.setdefault(key, {})
    node[keys[-1]] = value


def expand_jobs(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    """One job per (sweep point, arm, seed)."""
    sweep = cfg.get("sweep") or {}
    keys = list(sweep)
    points = list(itertools.product(*[sweep[k] for k in keys])) if keys else [()]
    seeds = range(int(cfg["seeds"]["start"]), int(cfg["seeds"]["start"]) + int(cfg["seeds"]["count"]))
    jobs = []
    for point in points:
        overrides = dict(zip(keys, point))
        for arm in cfg["arms"]:
            if arm["name"] == "oracle" and point != points[0]:
                continue  # the oracle does not depend on training settings
            for seed in seeds:
                run_cfg = copy.deepcopy(cfg)
                for k, v in overrides.items():
                    set_nested(run_cfg, k, v)
                tag = "_".join(f"{k.split('.')[-1]}={v}" for k, v in overrides.items())
                run_id = f"{arm_label(arm)}{'__' + tag if tag else ''}__seed{seed}"
                jobs.append({"cfg": run_cfg, "arm": arm, "seed": seed, "overrides": overrides, "run_id": run_id})
    return jobs


def learning_rate(cfg: dict[str, Any], arm: dict[str, Any]) -> float:
    table = cfg["lr"]
    label = arm_label(arm)
    for key in (label, arm["name"], "default"):
        if key in table:
            return float(table[key])
    raise KeyError(f"no learning rate for arm {label!r} in config 'lr'")


def n_agents_for(cfg: dict[str, Any]) -> int:
    return 1 if cfg["training"]["mode"] == "exact" else int(cfg["training"]["population_size"])


def circuit_for(cfg: dict[str, Any], arm: dict[str, Any], seed: int):
    from ..connectome.circuit import circuit_spec_for_arm  # heavy import only when needed

    return circuit_spec_for_arm(cfg["connectome"], graph_name(arm["name"]), seed)


def needs_budget(cfg: dict[str, Any], arm_name: str) -> bool:
    """Whether this arm's size is solved from the connectome arm's parameter count."""
    net = cfg["network"]
    if arm_name == "dense":
        return net["dense_width"] == "matched"
    if arm_name in FLYGM_ARMS:
        return net.get("flygm", {}).get("hidden_width") == "matched"
    return False


def parameter_budget(cfg: dict[str, Any]) -> int | None:
    """Trainable parameters of the connectome arm; the budget other arms match."""
    if not any(needs_budget(cfg, a["name"]) for a in cfg["arms"]):
        return None
    spec = circuit_for(cfg, {"name": "connectome"}, 0)
    return circuit_param_count(spec, int(cfg["game"]["n_bins"]), *readout_settings(cfg["network"], "budget_readout"))


def check_parameter_matching(cfg: dict[str, Any]) -> dict[str, int]:
    """Build every network arm once and assert matched trainable parameter counts."""
    game = build_game(cfg["game"])
    budget = parameter_budget(cfg)
    counts: dict[str, int] = {}
    for arm in cfg["arms"]:
        if arm["name"] not in NETWORK_ARMS:
            continue
        circuit = circuit_for(cfg, arm, 0) if arm["name"] in GRAPH_ARMS else None
        policy = build_policy(arm, game, 1, cfg["network"], 0, circuit, budget)
        counts[arm_label(arm)] = policy.trainable_parameters_per_agent()
    if counts:
        # always compare against the connectome arm's count, even in FlyGM-only configs
        counts["connectome(budget)"] = budget if budget is not None else circuit_param_count(
            circuit_for(cfg, {"name": "connectome"}, 0), int(cfg["game"]["n_bins"]), *readout_settings(cfg["network"])
        )
        # and against dense
        if budget is not None and "dense" not in counts and cfg["network"]["dense_width"] == "matched":
            dense = build_policy({"name": "dense"}, game, 1, cfg["network"], 0, None, budget)
            counts["dense(budget)"] = dense.trainable_parameters_per_agent()
    if len(counts) > 1:
        assert_matched(counts, float(cfg["network"]["param_tolerance"]))
    return counts


def run_job(job: dict[str, Any]) -> dict[str, Any]:
    # one thread per worker
    torch.set_num_threads(1)
    cfg, arm, seed = job["cfg"], job["arm"], job["seed"]
    game = build_game(cfg["game"])
    record: dict[str, Any] = {
        "run_id": job["run_id"],
        "experiment": cfg["experiment"],
        "arm": arm_label(arm),
        "seed": seed,
        "overrides": {k.split(".")[-1]: v for k, v in job["overrides"].items()},
        "config_hash": config_hash(cfg),
        "config": cfg,
    }
    started = time.perf_counter()
    if arm["name"] == "oracle":
        policy = game_a_analytic_policy(game) if isinstance(game, GameA) else game_b_analytic_policy(game)
        probs = policy[None, None, :] if isinstance(game, GameA) else policy[None]
        row = {"step": 0, **evaluate(game, probs)}
        summary = summarise([row], [probs], 0, float(cfg["metrics"]["threshold"]), 1)
        record.update(
            params_per_agent=0,
            graph_id=None,
            curve=[row],
            summary=summary,
            winner_loser={},
            final_policy=probs[0].tolist(),
        )
    else:
        circuit = circuit_for(cfg, arm, seed) if arm["name"] in GRAPH_ARMS else None
        budget = parameter_budget(cfg) if needs_budget(cfg, arm["name"]) else None
        torch.manual_seed(seed)
        policy = build_policy(arm, game, n_agents_for(cfg), cfg["network"], seed, circuit, budget)
        out = train_run(game, policy, cfg["training"], cfg["metrics"], learning_rate(cfg, arm), seed)
        record.update(
            params_per_agent=policy.trainable_parameters_per_agent(),
            graph_id=circuit.graph_id if circuit is not None else None,
            curve=out.curve,
            summary=out.summary,
            winner_loser=out.winner_loser,
            final_policy=out.final_policy,
            # arm-specific diagnostics (FlyGM: unsigned_sign used, spectral radius
            # before scaling, input-to-output gain of the frozen W)
            diagnostics=dict(getattr(policy, "diagnostics", {}) or {}),
        )
    record["summary"]["wall_seconds"] = time.perf_counter() - started
    out_dir = Path(job["out_dir"]) / "runs"
    out_dir.mkdir(parents=True, exist_ok=True)
    # write to a temp file then rename, so a killed run never leaves a half-written JSON
    final = out_dir / f"{job['run_id']}.json"
    tmp = final.with_suffix(".json.partial")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(record, fh, default=_json_default)
    os.replace(tmp, final)
    return {"run_id": job["run_id"], "wall": record["summary"]["wall_seconds"]}


def _json_default(o: Any) -> Any:
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


def collect(out_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Aggregate every ``runs/*.json`` into summary and curve tables."""
    rows, curves = [], []
    for path in sorted((out_dir / "runs").glob("*.json")):
        with open(path, encoding="utf-8") as fh:
            rec = json.load(fh)
        base = {"run_id": rec["run_id"], "arm": rec["arm"], "seed": rec["seed"], **rec["overrides"]}
        rows.append(
            {
                **base,
                "params_per_agent": rec["params_per_agent"],
                "graph_id": rec["graph_id"],
                "config_hash": rec["config_hash"],
                **rec["summary"],
                **{f"wl_{k}": v for k, v in rec["winner_loser"].items()},
                **{f"diag_{k}": v for k, v in rec.get("diagnostics", {}).items()},
            }
        )
        for row in rec["curve"]:
            curves.append({**base, **{k: row[k] for k in CURVE_COLUMNS if k in row}})
    return pd.DataFrame(rows), pd.DataFrame(curves)


def run_experiment(
    cfg: dict[str, Any], workers: int, resume: bool = True, out_root: Path | None = None
) -> Path:
    from .tuning import resolve_lr_table

    cfg = copy.deepcopy(cfg)
    cfg["lr"] = resolve_lr_table(cfg)  # the snapshot records the rates actually used
    out_dir = (out_root or REPO_ROOT / "results") / cfg["experiment"]
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "config.yaml", "w", encoding="utf-8") as fh:
        yaml.safe_dump(cfg, fh, sort_keys=False)
    counts = check_parameter_matching(cfg)
    if counts:
        print(f"[{cfg['experiment']}] trainable parameters per agent: {counts}")
    jobs = expand_jobs(cfg)
    for job in jobs:
        job["out_dir"] = str(out_dir)
    if resume:
        # one JSON per finished run; leftover .partial files get overwritten
        done = {p.stem for p in (out_dir / "runs").glob("*.json")} if (out_dir / "runs").exists() else set()
        jobs = [j for j in jobs if j["run_id"] not in done]
    print(f"[{cfg['experiment']}] {len(jobs)} runs to do on {workers} workers")
    started = time.perf_counter()
    if workers <= 1:
        for k, job in enumerate(jobs):
            run_job(job)
            _progress(cfg["experiment"], k + 1, len(jobs), started)
    else:
        ctx = mp.get_context("spawn")
        with ctx.Pool(workers) as pool:
            for k, _ in enumerate(pool.imap_unordered(run_job, jobs)):
                _progress(cfg["experiment"], k + 1, len(jobs), started)
    summary, curves = collect(out_dir)
    summary.to_csv(out_dir / "summary.csv", index=False)
    curves.to_csv(out_dir / "curves.csv", index=False)
    return out_dir


def _progress(name: str, done: int, total: int, started: float) -> None:
    if done == total or done % max(1, total // 10) == 0:
        el = time.perf_counter() - started
        print(f"[{name}] {done}/{total} runs, {el:.0f}s elapsed", flush=True)


def default_workers() -> int:
    return max(1, (os.cpu_count() or 2) - 2)


__all__ = ["collect", "deep_merge", "expand_jobs", "run_experiment", "run_job"]
