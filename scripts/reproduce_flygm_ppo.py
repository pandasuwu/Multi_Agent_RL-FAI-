"""Run the FlyGM and PPO experiments step by step and log each step's wall time.

    python scripts/reproduce_flygm_ppo.py                      # all steps
    python scripts/reproduce_flygm_ppo.py --only run-ppo-a     # selected steps
    python scripts/reproduce_flygm_ppo.py --list

Writes new results/<experiment>/ dirs and doesn't touch the ones from reproduce.py.
Split into steps because the whole grid takes too long for one sitting on a laptop.
Wall times go to results/flygm_ppo_log.txt.

Each run saves its own JSON, so rerunning after a crash only redoes unfinished runs.
Workers are single-threaded (BLAS threads = 1), so --workers sets the core count.
--log-file sends all output to a file and ignores Ctrl-C (for long runs under pythonw).
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
# keep BLAS single-threaded in children so workers don't oversubscribe
CHILD_ENV = {
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}

# FlyGM Game B is memory-bound: on the laptop (8 cores) 14 workers gave half the
# throughput of 6. On a 4-core cloud box 4 was best for this and 5 for run-ppo-b-flygm.
HEAVY_WORKERS = 6
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
    # not run (too expensive), kept for completeness
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
    # readout / W-gain fixes (RESULTS.md sec. 12)
    ("tune-readout-b", "tune_lr", ["readout_game_b_exact", "wgain_game_b_exact"], HEAVY_WORKERS),
    ("run-readout-b", "run_experiment", ["readout_game_b_exact", "wgain_game_b_exact"], HEAVY_WORKERS),
    ("tune-readout-a", "tune_lr", ["readout_game_a_exact", "wgain_game_a_exact"], HEAVY_WORKERS),
    ("run-readout-a", "run_experiment", ["readout_game_a_exact", "wgain_game_a_exact"], HEAVY_WORKERS),
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
    different rates. If they all landed on the same one, the main experiment already
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
