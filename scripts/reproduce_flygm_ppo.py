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

Unattended runs: ``--log-file PATH`` makes the driver write its own output, and every
child's, to that file instead of to a console, and it ignores Ctrl-C and Ctrl-Break.
Started with ``pythonw.exe`` there is then no console window anywhere in the tree, so
nothing in the user's session can close one and take the run down with it. A run of
this length needs that: the longest single run here is about 4 hours, and `resume`
redoes any run that was in flight.
"""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

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

# Worker cap for the FlyGM Game B steps. Measured on this machine (8 physical cores):
# a FlyGM Game B run takes 462 s alone, 420 s with 6 running at once (no contention at
# all), and 1752 s with 14 -- so 14 workers deliver *half* the throughput of 6. Game B
# puts 50 contexts x 8 agents x 258 neurons through f_psi every step, and past ~6
# processes they evict each other's working sets. Game A is 50x smaller per step and
# does not need the cap.
HEAVY_WORKERS = 6
# run-ppo-b-flygm is exactly 9 runs of about 4 hours each. At 6 workers that is two
# waves (~7.6 h) and the second wave leaves the machine nearly idle; at 9 it is one
# wave, which wins unless the per-run slowdown at 9 exceeds 1.9x. The measured
# contention-free point is 6 and the collapse was at 14, so 9 is the better bet here.
FLYGM_PPO_WORKERS = 9

# name -> (script, configs, worker cap or None). "tune" steps pick learning rates on
# the tuning seeds and write configs/tuned_lr.yaml; "run" steps run the evaluation seeds.
STEPS: list[tuple[str, str, list[str], int | None]] = [
    (
        "tune-flygm-fast",
        "tune_lr",
        ["flygm_game_a_exact", "flygm_game_a_sampled", "flygm_game_b_exact"],
        None,
    ),
    (
        "run-flygm-fast",
        "run_experiment",
        ["flygm_game_a_exact", "flygm_game_a_sampled", "flygm_game_b_exact"],
        None,
    ),
    ("eval1", "run_experiment", ["game_a_exact_eval1"], None),
    ("tune-ppo-a", "tune_lr", ["ppo_game_a_sampled"], None),
    ("run-ppo-a", "run_experiment", ["ppo_game_a_sampled"], None),
    ("tune-flygm-b-sampled", "tune_lr", ["flygm_game_b_sampled"], HEAVY_WORKERS),
    ("run-flygm-b-sampled", "run_experiment", ["flygm_game_b_sampled"], HEAVY_WORKERS),
    ("tune-ppo-b", "tune_lr", ["ppo_game_b_sampled"], None),
    ("run-ppo-b", "run_experiment", ["ppo_game_b_sampled"], None),
    ("run-ppo-b-flygm", "run_experiment", ["ppo_game_b_sampled_flygm"], FLYGM_PPO_WORKERS),
    ("robust-ppo-a", "run_experiment", ["robust_ppo_game_a_sampled_shared_lr"], None),
    ("robust-ppo-b", "run_experiment", ["robust_ppo_game_b_sampled_shared_lr"], None),
    (
        "robust",
        "run_experiment",
        [
            "robust_flygm_game_a_sampled_shared_lr",
            "robust_flygm_game_b_exact_shared_lr",
            "robust_flygm_game_b_sampled_shared_lr",
        ],
        HEAVY_WORKERS,
    ),
    ("asym-ppo", "run_experiment", ["ppo_game_a_sampled_asym"], None),
    ("analyze", "analyze", [], None),
]


def note(line: str) -> None:
    print(line, flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")


_CHILD_OUTPUT: Any = None  # file object when --log-file is used, else inherit


def detach_from_console(path: Path) -> None:
    """Send our own output, and our children's, to ``path``; ignore console signals.

    Under ``pythonw.exe`` ``sys.stdout`` is None, so this also gives the driver (and
    anything it prints through ``note``) somewhere real to write.
    """
    global _CHILD_OUTPUT
    path.parent.mkdir(parents=True, exist_ok=True)
    _CHILD_OUTPUT = open(path, "a", buffering=1, encoding="utf-8", errors="replace")
    sys.stdout = sys.stderr = _CHILD_OUTPUT
    for name in ("SIGINT", "SIGBREAK"):
        sig = getattr(signal, name, None)
        if sig is not None:
            try:
                signal.signal(sig, signal.SIG_IGN)
            except (ValueError, OSError):
                pass


def run(args: list[str]) -> None:
    print("+", " ".join(args), flush=True)
    subprocess.run(
        [PY, *args],
        cwd=ROOT,
        check=True,
        env={**os.environ, **CHILD_ENV},
        stdout=_CHILD_OUTPUT,
        stderr=_CHILD_OUTPUT,
    )


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


def run_step(name: str, script: str, configs: list[str], workers: int, cap: int | None) -> None:
    if cap is not None and workers > cap:
        note(f"{name}: capped at {cap} workers (measured: more than that cuts throughput here)")
        workers = cap
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
    parser.add_argument(
        "--log-file",
        type=Path,
        default=None,
        help="write all output here and ignore console signals (for pythonw.exe runs)",
    )
    args = parser.parse_args()
    if args.log_file is not None:
        detach_from_console(args.log_file)
    if args.list:
        for name, script, configs, cap in STEPS:
            tag = f"  [<= {cap} workers]" if cap else ""
            print(f"{name:22s} {script:15s} {' '.join(configs)}{tag}")
        return
    steps = STEPS if args.only is None else [s for s in STEPS if s[0] in set(args.only)]
    unknown = set(args.only or []) - {s[0] for s in STEPS}
    if unknown:
        raise SystemExit(f"unknown step(s): {sorted(unknown)}")
    note(f"=== {time.strftime('%Y-%m-%d %H:%M:%S')} steps: {', '.join(s[0] for s in steps)} ===")
    started = time.perf_counter()
    for name, script, configs, cap in steps:
        run_step(name, script, configs, args.workers, cap)
    note(f"=== total {(time.perf_counter() - started) / 60:.1f} min wall ===")


if __name__ == "__main__":
    main()
