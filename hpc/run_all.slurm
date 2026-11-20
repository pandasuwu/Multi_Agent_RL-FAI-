#!/bin/bash
# Full rerun of every experiment on one cluster node. See hpc/README.md.
#
#   sbatch hpc/run_all.slurm                    # from scratch, tuning included
#   SKIP_TUNING=1 sbatch hpc/run_all.slurm      # keep configs/tuned_lr.yaml
#   RESUME=1 sbatch hpc/run_all.slurm           # continue after a time limit
#   RESUME=1 ONLY="analyze figures" sbatch ...  # only some steps (see --list)
#   GIT_COMMIT=$(git rev-parse HEAD) sbatch ... # compute nodes have no git
#
# Edit the partition / time / cpus lines for your cluster, or override them on the
# command line: sbatch -A <account> -p small -c 48 hpc/run_all.slurm. CPU only.
#SBATCH --job-name=fly-woa-all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=3-00:00:00
#SBATCH --output=hpc/logs/%x-%j.out
#SBATCH --partition=small

# the venv was made with this module's Python (PARAM-Ananta); harmless elsewhere.
# Before `set -u`: Lmod's shell function reads unset variables.
module load python/3.12 2>/dev/null || true
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
mkdir -p hpc/logs   # must also exist before sbatch, or SLURM can't open the log
source .venv/bin/activate

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
WORKERS=${WORKERS:-${SLURM_CPUS_PER_TASK:-32}}
# FlyGM / readout in Game B are memory-bound; on the laptop more than 6 workers was slower.
# Leave at WORKERS on a server node, lower it if those steps crawl.
HEAVY=${HEAVY_WORKERS:-$WORKERS}

ARGS=(--workers "$WORKERS" --heavy-workers "$HEAVY")
[[ -n "${SKIP_TUNING:-}" ]] && ARGS+=(--skip-tuning)
[[ -n "${RESUME:-}" ]] && ARGS+=(--resume)
[[ -n "${ONLY:-}" ]] && ARGS+=(--only $ONLY)

echo "job $SLURM_JOB_ID on $(hostname), $WORKERS workers ($HEAVY heavy), commit ${GIT_COMMIT:-$(git rev-parse --short HEAD 2>/dev/null || echo unknown)}"
python -m pytest -q
python scripts/reproduce_all.py "${ARGS[@]}"
python scripts/compare_to_committed.py --out results/tables/rerun_vs_committed.csv || true
