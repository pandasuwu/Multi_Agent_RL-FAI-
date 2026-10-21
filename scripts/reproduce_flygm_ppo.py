"""Run the FlyGM and PPO experiments, recording the wall time of each step.

    python scripts/reproduce_flygm_ppo.py                      # every step, in order
    python scripts/reproduce_flygm_ppo.py --only run-ppo-a     # one or more steps
    python scripts/reproduce_flygm_ppo.py --list

These experiments are additive: they write new ``results/<experiment>/`` directories
and never touch the ones produced by ``scripts/reproduce.py``. The steps are separate
because the grid does not fit one sitting on a laptop CPU (the FlyGM arm in sampled
Game B costs about an hour of CPU per run), and because each one's learning rates
must be tuned before its runs start.

Every step appends ``<name>: <seconds> s`` to ``results/flygm_ppo_log.txt``, which is
where the wall times quoted in RESULTS.md come from.

Restartable. Each run writes its own ``results/<experiment>/runs/<run_id>.json`` and
every step resumes, so re-running the same command after a crash, a sleep or a reboot
redoes only the runs that were in flight. Re-running a finished step costs seconds.

Single-threaded workers: the pool is the parallelism, so every child gets
``OMP/MKL/OpenBLAS/numexpr = 1`` thread (``src.training.runner.run_job`` also calls
``torch.set_num_threads(1)``) and ``--workers`` is the only knob that decides how many
cores are busy.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
LOG = ROOT / "results" / "flygm_ppo_log.txt"
# BLAS must stay single-threaded in every child: a fresh process otherwise takes one
# thread per physical core, and `--workers` of those would oversubscribe the machine.
CHILD_ENV = {
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}

# name -> (script, configs). "tune" steps pick learning rates on the tuning seeds and
# write configs/tuned_lr.yaml; "run" steps run the evaluation seeds.
STEPS: list[tuple[str, str, list[str]]] = [
    (
        "tune-flygm-fast",
        "tune_lr",
        ["flygm_game_a_exact", "flygm_game_a_sampled", "flygm_game_b_exact"],
    ),
    (
        "run-flygm-fast",
        "run_experiment",
        ["flygm_game_a_exact", "flygm_game_a_sampled", "flygm_game_b_exact"],
    ),
    ("eval1", "run_experiment", ["game_a_exact_eval1"]),
    ("tune-ppo-a", "tune_lr", ["ppo_game_a_sampled"]),
    ("run-ppo-a", "run_experiment", ["ppo_game_a_sampled"]),
    ("tune-flygm-b-sampled", "tune_lr", ["flygm_game_b_sampled"]),
    ("run-flygm-b-sampled", "run_experiment", ["flygm_game_b_sampled"]),
    ("tune-ppo-b", "tune_lr", ["ppo_game_b_sampled"]),
    ("run-ppo-b", "run_experiment", ["ppo_game_b_sampled"]),
    ("run-ppo-b-flygm", "run_experiment", ["ppo_game_b_sampled_flygm"]),
    (
        "robust",
        "run_experiment",
        [
            "robust_flygm_game_a_sampled_shared_lr",
            "robust_flygm_game_b_exact_shared_lr",
            "robust_flygm_game_b_sampled_shared_lr",
        ],
    ),
    ("asym-ppo", "run_experiment", ["ppo_game_a_sampled_asym"]),
    ("analyze", "analyze", []),
]


def note(line: str) -> None:
    print(line, flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def run(args: list[str]) -> None:
    print("+", " ".join(args), flush=True)
    subprocess.run([PY, *args], cwd=ROOT, check=True, env={**os.environ, **CHILD_ENV})


def sharing_changes_nothing(path: Path) -> bool:
    """True if a ``share_circuit_lr`` config would run at the rates already measured.

    A shared-rate robustness run is only informative when the arms were tuned to
    *different* rates. If they all landed on the same one, the main experiment already
    is the shared-rate run and re-running it would duplicate it exactly.
    """
    from src.agents.factory import arm_label
    from src.config import load_config
    from src.training.tuning import resolve_lr_table

    cfg = load_config(path)
    source = cfg.get("lr_from") or {}
    if not source.get("share_circuit_lr"):
        return False
    try:
        shared = resolve_lr_table(cfg)
        plain = resolve_lr_table({**cfg, "lr_from": {k: v for k, v in source.items() if k != "share_circuit_lr"}})
    except KeyError:
        return False  # not tuned yet: let the run itself report the problem
    return all(shared.get(arm_label(a)) == plain.get(arm_label(a)) for a in cfg["arms"])


def run_step(name: str, script: str, configs: list[str], workers: int) -> None:
    paths = [f"configs/{c}.yaml" for c in configs]
    missing = [p for p in paths if not (ROOT / p).exists()]
    if missing:
        note(f"{name}: SKIPPED, no config yet ({', '.join(missing)})")
        return
    if script == "run_experiment":
        keep = [q for q in paths if not sharing_changes_nothing(ROOT / q)]
        for dropped in [q for q in paths if q not in keep]:
            note(f"{name}: SKIPPED {Path(dropped).stem}, every arm tuned to the same rate "
                 "(its main experiment already is the shared-rate run)")
        paths = keep
        if not paths:
            return
    started = time.perf_counter()
    if script == "analyze":
        run(["scripts/analyze.py"])
        run(["scripts/make_figures.py"])
    else:
        run([f"scripts/{script}.py", *paths, "--workers", str(workers)])
    note(f"{name}: {time.perf_counter() - started:.0f} s wall ({workers} workers)")


def main() -> None:
    sys.path.insert(0, str(ROOT))
    from src.training.runner import default_workers

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workers", type=int, default=default_workers())
    parser.add_argument("--only", nargs="+", default=None, help="step names to run")
    parser.add_argument("--list", action="store_true", help="print the step names and exit")
    args = parser.parse_args()
    if args.list:
        for name, script, configs in STEPS:
            print(f"{name:22s} {script:15s} {' '.join(configs)}")
        return
    steps = STEPS if args.only is None else [s for s in STEPS if s[0] in set(args.only)]
    unknown = set(args.only or []) - {s[0] for s in STEPS}
    if unknown:
        raise SystemExit(f"unknown step(s): {sorted(unknown)}")
    note(f"=== {time.strftime('%Y-%m-%d %H:%M:%S')} steps: {', '.join(s[0] for s in steps)} ===")
    started = time.perf_counter()
    for name, script, configs in steps:
        run_step(name, script, configs, args.workers)
    note(f"=== total {(time.perf_counter() - started) / 60:.1f} min wall ===")


if __name__ == "__main__":
    main()
