"""Learning-rate selection with an identical budget for every arm.

For each experiment and each trainable arm we try every learning rate in the
configured grid on the *tuning seeds* (disjoint from the evaluation seeds), and
pick the one with the lowest mean ``log10_expl_auc`` (the mean of log10
exploitability over the whole training curve), which rewards both speed and the
final level. Ties go to the smaller learning rate. Sweep experiments are tuned at
their first sweep point (``loss_win_ratio = 1``, the symmetric control), so the
asymmetric conditions never get to pick their own step size.

An experiment may set ``tuning.steps`` to screen the grid on shorter runs than the
experiment itself uses. The criterion is unchanged (mean log10 exploitability over
the curve); only the curve is shorter. This is a compute reduction, used for the
settings where a full-length grid does not fit the laptop budget, and it is recorded
in the experiment's config and in RESULTS.md.

The chosen rates are written to ``configs/tuned_lr.yaml``, which experiment
configs read through ``lr_from``.
"""

from __future__ import annotations

import copy
from typing import Any

import numpy as np
import pandas as pd
import yaml

from ..agents.factory import CIRCUIT_ARMS, FLYGM_ARMS, NETWORK_ARMS, arm_label
from ..config import CONFIG_DIR, REPO_ROOT
from .runner import run_experiment

TUNED_FILE = CONFIG_DIR / "tuned_lr.yaml"


def lr_grid(cfg: dict[str, Any], arm: dict[str, Any]) -> list[float]:
    grid = cfg["tuning"]["lr_grid"][cfg["training"]["mode"]]
    return [float(x) for x in grid["tabular" if arm["name"] == "tabular" else "network"]]


def tune_experiment(cfg: dict[str, Any], workers: int) -> tuple[dict[str, float], pd.DataFrame]:
    """Run the tuning grid for one experiment config and return ``{arm label: lr}``."""
    base = copy.deepcopy(cfg)
    for key in list(base.get("sweep") or {}):
        # tune at the first sweep point only (the symmetric control)
        first = base["sweep"][key][0]
        node = base
        parts = key.split(".")
        for p in parts[:-1]:
            node = node[p]
        node[parts[-1]] = first
    base["sweep"] = {}
    base.pop("lr_from", None)
    base["seeds"] = dict(cfg["tuning"]["seeds"])
    if "steps" in cfg["tuning"]:
        base["training"]["steps"] = int(cfg["tuning"]["steps"])
    arms = [a for a in cfg["arms"] if a["name"] == "tabular" or a["name"] in NETWORK_ARMS]
    rows = []
    scratch = REPO_ROOT / "results" / "_scratch" / "tuning"
    grids = sorted({lr for a in arms for lr in lr_grid(base, a)})
    for lr in grids:
        run_cfg = copy.deepcopy(base)
        run_cfg["arms"] = [a for a in arms if lr in lr_grid(base, a)]
        run_cfg["lr"] = {"default": lr}
        run_cfg["experiment"] = f"{cfg['experiment']}__lr={lr:g}"
        out = run_experiment(run_cfg, workers=workers, resume=True, out_root=scratch)
        summary = pd.read_csv(out / "summary.csv")
        summary["lr"] = lr
        rows.append(summary)
    table = pd.concat(rows, ignore_index=True)
    score = (
        table.groupby(["arm", "lr"])
        .agg(
            criterion=("log10_expl_auc", "mean"),
            final_exploitability=("final_exploitability", "mean"),
            n=("seed", "count"),
        )
        .reset_index()
    )
    chosen = {}
    for arm in arms:
        label = arm_label(arm)
        sub = score[score["arm"] == label].sort_values(["criterion", "lr"])
        chosen[label] = float(sub.iloc[0]["lr"])
    score["experiment"] = cfg["experiment"]
    score["chosen"] = [chosen.get(a) == lr for a, lr in zip(score["arm"], score["lr"])]
    return chosen, score


def write_tuned(experiment: str, chosen: dict[str, float]) -> None:
    current: dict[str, Any] = {}
    if TUNED_FILE.exists():
        current = yaml.safe_load(TUNED_FILE.read_text(encoding="utf-8")) or {}
    current[experiment] = chosen
    header = "# Written by scripts/tune_lr.py; do not edit by hand.\n"
    TUNED_FILE.write_text(header + yaml.safe_dump(current, sort_keys=True), encoding="utf-8")


def resolve_lr_table(cfg: dict[str, Any]) -> dict[str, Any]:
    """The ``lr`` map of a config, taken from ``configs/tuned_lr.yaml`` if requested."""
    source = cfg.get("lr_from")
    if not source:
        return cfg["lr"]
    path = CONFIG_DIR / source["file"]
    table = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    key = source["experiment"]
    if key not in table:
        raise KeyError(f"{path} has no tuned learning rates for {key!r}; run scripts/tune_lr.py first")
    lrs = dict(table[key])
    if source.get("share_circuit_lr"):
        # Robustness protocol: within one architecture family the arms differ only in
        # the graph, so give the whole family one step size, its most common tuned
        # choice (ties to the smaller rate). Removes tuning noise from the wiring
        # contrasts. The trained-edge arms 3-5 and the FlyGM arms are separate
        # families: they are different architectures and are never compared directly
        # at a shared rate.
        for family in (CIRCUIT_ARMS, FLYGM_ARMS):
            present = [v for k, v in lrs.items() if k.split("+")[0] in family]
            if not present:
                continue
            values, counts = np.unique(present, return_counts=True)
            shared = float(values[np.argmax(counts)])
            for k in lrs:
                if k.split("+")[0] in family:
                    lrs[k] = shared
    return lrs
