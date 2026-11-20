# Full rerun on the cluster

The committed results were made in pieces on three machines (laptop, a cloud VM, and
another machine for §14–§15). This reruns all of them in one job, from one commit, so
every number in the report comes from the same code run once.

## Setup (login node, once)

```bash
git clone https://github.com/pandasuwu/Multi_Agent_RL-FAI-.git
cd Multi_Agent_RL-FAI-
git checkout <branch or commit to run>
# module load python/3.12   # if your cluster uses modules; needs Python 3.10+
python -m venv .venv
source .venv/bin/activate
pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
python -m pytest -q
```

The job doesn't need internet: the extracted connectome subgraph is committed in
`data/connectome/`. (To also rebuild it from the raw MaleCNS files, run
`python scripts/build_connectome.py --download` on the login node first and pass
`--rebuild-connectome`; it checks the hash against the committed one.)

## Run

Set the partition, CPU count and time limit at the top of `run_all.slurm`, then:

```bash
sbatch hpc/run_all.slurm
```

The job runs the tests, then `scripts/reproduce_all.py` (tune → run → analyse →
figures; `--list` shows the 26 steps), then compares the new numbers with the
committed ones into `results/tables/rerun_vs_committed.csv`.

- From scratch, tuning included (default). Learning rates are re-chosen on the tuning
  seeds, so some numbers will move. `SKIP_TUNING=1 sbatch ...` keeps the committed
  `configs/tuned_lr.yaml` instead.
- If the job hits its time limit, `RESUME=1 sbatch hpc/run_all.slurm` continues: every
  run is saved as its own JSON, so only unfinished runs are redone.
- `HEAVY_WORKERS=<n>` sets the worker count for the memory-bound FlyGM / readout Game B
  steps if they scale badly at full core count.

Compute: the pieces took about 200 core-hours in total on the earlier machines. On 32
cores expect roughly 8–12 hours; the FlyGM and readout Game B steps dominate.

## Afterwards

Copy back or commit `results/`, `figures/`, `configs/tuned_lr.yaml` and
`hpc/logs/`. `results/run_manifest.json` records the commit, host, versions and wall
time of every step. `results/tables/rerun_vs_committed.csv` lists which numbers changed.
