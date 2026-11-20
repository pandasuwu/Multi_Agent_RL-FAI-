"""Rerun every experiment in RESULTS.md and the report, in one go, on one machine.

    python scripts/reproduce_all.py --workers 32                 # everything, from scratch
    python scripts/reproduce_all.py --workers 32 --skip-tuning   # keep configs/tuned_lr.yaml
    python scripts/reproduce_all.py --list                       # print the steps
    python scripts/reproduce_all.py --resume                     # continue after a crash or time limit

This is the script behind hpc/run_all.slurm. It chains what was run in pieces before
(reproduce.py, reproduce_flygm_ppo.py, the review checks, the population-size sweep and
the welfare analysis), so every committed number comes out of a single run of the code
at a single commit. Order matters: tuning writes configs/tuned_lr.yaml, and later
experiments read their learning rates from it.

Without --resume it first deletes every result directory it is about to regenerate
(plus results/tables, results/tuning and the scratch dirs), so nothing stale survives.
With --resume it keeps them: each run and tuning run is its own JSON, so only
unfinished runs are redone.

Writes results/run_manifest.json (commit, host, versions, worker counts, wall time per
step) and appends wall times to results/reproduce_all_log.txt.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
LOG = ROOT / "results" / "reproduce_all_log.txt"
MANIFEST = ROOT / "results" / "run_manifest.json"
CHILD_ENV = {k: "1" for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")}

MAIN = ["game_a_exact", "game_b_exact", "game_a_sampled", "game_b_sampled"]

# (name, kind, configs, heavy). kind: tune | run | script. heavy steps (FlyGM / readout
# in Game B) are memory-bound and use --heavy-workers instead of --workers.
STEPS: list[tuple[str, str, list[str], bool]] = [
    # --- main grid (RESULTS 3-8) ---
    ("tune-main", "tune", MAIN, False),
    ("run-main", "run", MAIN, False),
    ("run-robust", "run", ["robust_game_a_sampled_shared_lr", "robust_game_b_exact_shared_lr",
                           "robust_game_b_sampled_shared_lr"], False),
    # --- FlyGM, PPO, eval_every 1 (RESULTS 9-11) ---
    ("tune-flygm-fast", "tune", ["flygm_game_a_exact", "flygm_game_a_sampled", "flygm_game_b_exact"], False),
    ("run-flygm-fast", "run", ["flygm_game_a_exact", "flygm_game_a_sampled", "flygm_game_b_exact"], False),
    ("run-eval1", "run", ["game_a_exact_eval1"], False),
    ("tune-ppo-a", "tune", ["ppo_game_a_sampled"], False),
    ("run-ppo-a", "run", ["ppo_game_a_sampled"], False),
    ("tune-flygm-b-sampled", "tune", ["flygm_game_b_sampled"], True),
    ("run-flygm-b-sampled", "run", ["flygm_game_b_sampled"], True),
    ("tune-ppo-b", "tune", ["ppo_game_b_sampled"], False),
    ("run-ppo-b", "run", ["ppo_game_b_sampled"], False),
    ("run-robust-ppo", "run", ["robust_ppo_game_a_sampled_shared_lr", "robust_ppo_game_b_sampled_shared_lr"], False),
    ("run-robust-flygm", "run", ["robust_flygm_game_a_sampled_shared_lr", "robust_flygm_game_b_exact_shared_lr",
                                 "robust_flygm_game_b_sampled_shared_lr"], True),
    # --- readout / W gain (RESULTS 12) ---
    ("tune-readout-a", "tune", ["readout_game_a_exact", "wgain_game_a_exact"], True),
    ("run-readout-a", "run", ["readout_game_a_exact", "wgain_game_a_exact"], True),
    ("tune-readout-b", "tune", ["readout_game_b_exact", "wgain_game_b_exact"], True),
    ("run-readout-b", "run", ["readout_game_b_exact", "wgain_game_b_exact"], True),
    # --- population size (RESULTS 14) and review checks (RESULTS 15) ---
    ("run-popsize", "run", ["popsize_game_a_sampled"], False),
    ("run-review", "run", ["popnash_game_a_sampled", "lrsweep_game_a_exact", "lrsweep_game_b_exact"], False),
    # --- second review: activation scale and how the circuit is cut (RESULTS 15.5-15.6) ---
    ("run-scale-sensitivity", "run", ["scalenorm_game_a_exact", "scalenorm_game_b_exact", "scalenorm_game_b_sampled",
                                      "sensitivity_syn3_game_b_exact", "sensitivity_syn10_game_b_exact"], False),
    # --- task that needs computation (Game C) and the remaining review checks (RESULTS 16) ---
    ("run-unroll-seeds20", "run", ["unroll_game_b_exact", "seeds20_game_b_sampled"], False),
    ("tune-gamec", "tune", ["gamec_exact", "gamec_pop_exact", "gamec_sampled", "gamec_pop_sampled"], False),
    ("run-gamec", "run", ["gamec_exact", "gamec_pop_exact", "gamec_sampled", "gamec_pop_sampled"], False),
    ("run-gamec-checks", "run", ["gamec_shared_exact", "gamec_pop_shared_exact", "gamec_readout_all_exact",
                                 "gamec_unroll_exact", "gamec_robust_sampled", "gamec_pop_robust_sampled"], False),
    ("run-block-long", "run", ["block_gamec_sampled", "block_gamec_pop_sampled", "block_gamec_exact",
                               "block_game_b_sampled", "long_gamec_sampled", "long_gamec_pop_sampled"], False),
    # --- what in the coarse layout: input -> output shortcuts, E/I labels, dose-response ---
    ("run-shortcut", "run", ["shortcut_gamec_sampled", "shortcut_gamec_pop_sampled", "shortcut_game_b_sampled",
                             "dose_gamec_sampled", "dose_game_b_sampled"], False),
    ("run-signs", "run", ["signs_gamec_sampled", "signs_gamec_pop_sampled", "signs_game_b_sampled"], False),
    ("validate-asymmetric", "script", ["validate_asymmetric.py"], False),
    ("path-diagnostics", "script", ["path_diagnostics.py"], False),
    # --- tables and figures ---
    ("welfare-from-records", "script", ["welfare_rerun.py", "--from-records"], False),
    ("analyze", "script", ["analyze.py"], False),
    ("welfare", "script", ["welfare.py"], False),
    ("figures", "script", ["make_figures.py"], False),
]
TUNED_FILE = ROOT / "configs" / "tuned_lr.yaml"
# Steps where a shared-rate config is dropped when every arm was tuned to one rate (it
# would duplicate its main experiment), as reproduce_flygm_ppo.py did. Not the others:
# robust_game_a_sampled, popsize and popnash also use share_circuit_lr, but always run
# (popsize and popnash vary something else, and reproduce.py ran the robust set as is).
SKIP_IF_SAME_RATE = {"run-robust-ppo", "run-robust-flygm"}


def sharing_changes_nothing(path: Path) -> bool:
    from reproduce_flygm_ppo import sharing_changes_nothing as check

    return check(path)


def note(line: str) -> None:
    print(line, flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def run(args: list[str]) -> None:
    print("+", " ".join(args), flush=True)
    subprocess.run([PY, *args], cwd=ROOT, check=True, env={**os.environ, **CHILD_ENV})


def git(*args: str) -> str:
    try:
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def clean(steps: list[tuple[str, str, list[str], bool]], tuning: bool) -> None:
    """Delete every output the selected steps regenerate."""
    for _, kind, configs, _ in steps:
        if kind == "run":
            for c in configs:
                shutil.rmtree(ROOT / "results" / c, ignore_errors=True)
    for d in ("tables", "_scratch/welfare_rerun"):
        shutil.rmtree(ROOT / "results" / d, ignore_errors=True)
    if tuning:
        shutil.rmtree(ROOT / "results" / "tuning", ignore_errors=True)
        shutil.rmtree(ROOT / "results" / "_scratch" / "tuning", ignore_errors=True)
        TUNED_FILE.unlink(missing_ok=True)


def main() -> None:
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "scripts"))
    from src.training.runner import default_workers

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workers", type=int, default=default_workers())
    parser.add_argument("--heavy-workers", type=int, default=None,
                        help="workers for the memory-bound FlyGM/readout Game B steps (default: --workers)")
    parser.add_argument("--skip-tuning", action="store_true", help="keep the committed configs/tuned_lr.yaml")
    parser.add_argument("--rebuild-connectome", action="store_true",
                        help="rebuild the subgraph from raw MaleCNS (needs data/raw or internet) and check its hash")
    parser.add_argument("--resume", action="store_true", help="keep finished runs instead of starting clean")
    parser.add_argument("--only", nargs="+", default=None, help="step names to run")
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()

    if args.list:
        for name, kind, configs, heavy in STEPS:
            print(f"{name:22s} {kind:6s} {' '.join(configs)}{'  [heavy]' if heavy else ''}")
        return
    unknown = set(args.only or []) - {s[0] for s in STEPS}
    if unknown:
        raise SystemExit(f"unknown step(s): {sorted(unknown)}")
    steps = [s for s in STEPS if args.only is None or s[0] in set(args.only)]
    if args.skip_tuning:
        steps = [s for s in steps if s[1] != "tune"]
    if args.skip_tuning and not TUNED_FILE.exists():
        raise SystemExit("--skip-tuning needs configs/tuned_lr.yaml")
    heavy_workers = args.heavy_workers or args.workers

    manifest = {
        # compute nodes may have no git: pass GIT_COMMIT from the login node
        "commit": os.environ.get("GIT_COMMIT") or git("rev-parse", "HEAD"),
        "host": platform.node(),
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "python": sys.version.split()[0],
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "workers": args.workers,
        "heavy_workers": heavy_workers,
        "tuning": not args.skip_tuning,
        "resume": args.resume,
        "started": time.strftime("%Y-%m-%d %H:%M:%S"),
        "steps": {},
    }
    if args.resume and MANIFEST.exists():
        manifest["steps"] = json.loads(MANIFEST.read_text(encoding="utf-8")).get("steps", {})
    try:
        import numpy
        import torch

        manifest.update(torch=torch.__version__, numpy=numpy.__version__)
    except ImportError:
        pass

    if args.rebuild_connectome:
        prov = ROOT / "data/connectome/give_up_circuit.provenance.json"
        committed = json.loads(prov.read_text(encoding="utf-8"))["subgraph_sha256"]
        run(["scripts/build_connectome.py", "--download"])
        rebuilt = json.loads(prov.read_text(encoding="utf-8"))["subgraph_sha256"]
        if rebuilt != committed:
            raise SystemExit(f"rebuilt subgraph {rebuilt} differs from committed {committed}")
        manifest["connectome_sha256"] = rebuilt
    if not args.resume:
        clean(steps, tuning=not args.skip_tuning)

    note(f"=== {manifest['started']} commit {manifest['commit'][:10]} on {manifest['host']} "
         f"({manifest['cpu_count']} cpus), steps: {', '.join(s[0] for s in steps)} ===")
    started = time.perf_counter()
    for name, kind, configs, heavy in steps:
        w = heavy_workers if heavy else args.workers
        t0 = time.perf_counter()
        if kind == "script":
            extra = ["--workers", str(w)] if configs[0] == "welfare_rerun.py" else []
            run([f"scripts/{configs[0]}", *configs[1:], *extra])
        else:
            if kind == "run" and name in SKIP_IF_SAME_RATE:
                # a shared-rate config whose arms were all tuned to one rate would only
                # duplicate its main experiment (same rule as reproduce_flygm_ppo.py)
                skipped = [c for c in configs if sharing_changes_nothing(ROOT / "configs" / f"{c}.yaml")]
                for c in skipped:
                    note(f"{name}: SKIPPED {c}, every arm tuned to the same rate")
                configs = [c for c in configs if c not in skipped]
            script = "tune_lr.py" if kind == "tune" else "run_experiment.py"
            if configs:
                run([f"scripts/{script}", *[f"configs/{c}.yaml" for c in configs], "--workers", str(w)])
        secs = time.perf_counter() - t0
        manifest["steps"][name] = {"seconds": round(secs, 1), "workers": w, "configs": configs}
        note(f"{name}: {secs:.0f} s wall ({w} workers)")
        MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    manifest["total_minutes"] = round((time.perf_counter() - started) / 60, 1)
    manifest["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    note(f"=== total {manifest['total_minutes']} min wall ===")


if __name__ == "__main__":
    main()
