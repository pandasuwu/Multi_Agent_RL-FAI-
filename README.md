# Fly give-up-circuit wiring vs. equilibrium learning in the war of attrition

**Question.** When two learning agents play a war of attrition, does constraining each
agent's policy network to the wiring of the real *Drosophila* give-up circuit change
how well or how fast they converge to the known equilibrium, compared to conventional
networks of matched size? The circuit is PPL1 → V2 MBONs (MBON18/19) → Tk-GAL4^FruM
(AVLP727m), from the MaleCNS v1.0 connectome, following Ventimiglia et al. 2026.

**Answer:** connectome wiring made no difference to how well the agents converged.
The details, including small speed effects of inconsistent sign, are in
[`RESULTS.md`](RESULTS.md).
**Live demo:** [Give-Up Game Lab](https://claude.ai/artifact/7RDFGua3hKQyYSKt8W7RKN), or open [`demo/index.html`](demo/index.html) in any browser. Players learn both games in real time, with brains wired like the fly circuit (all 4,952 real connections) or its controls. Rebuild it with `python scripts/build_demo.py` (source: `scripts/demo_template.html`).
**Theory:** [`docs/theory.md`](docs/theory.md).
**Circuit extraction:** [`docs/circuit.md`](docs/circuit.md).

## What is compared

| # | Arm | What it is |
|---|---|---|
| 0 | Oracle | the analytic equilibrium floored onto the grid; yardstick and payoff-engine check |
| 1 | Tabular | softmax logits per context; sanity baseline |
| 2 | Dense MLP | 2 hidden layers of width 48 (5,584 parameters) |
| 3 | Connectome | recurrent circuit on the real 258-neuron / 4,952-edge MaleCNS subgraph (5,610 parameters); only existing edges have weights |
| 4 | Degree-preserving rewire | same neurons, same in- and out-degree of every neuron, wiring shuffled by double-edge swaps, redrawn per seed |
| 5 | Random sparse | same neurons and edge count, uniformly random edges |
| 6 | FlyGM, real circuit | the same graph with its **synapse counts frozen** as the weights (`W[post, pre] = sign(pre)·synapses`, rescaled to spectral radius 1, never trained). What trains is a 9-dimensional descriptor per neuron, one shared update MLP `f_ψ(h_i, m_i, η_i)` with `m = W h`, the encoder and gate on PPL1, and the readout (5,638 parameters) |
| 7 | FlyGM, rewire | the same, on the degree-preserving rewire |
| 8 | FlyGM, random sparse | the same, on the random sparse graph |

Arms 3–5 also run **sign-constrained** (Dale's law from MaleCNS transmitter predictions),
giving 9 trained-edge arms, and 12 with the FlyGM arms. Arms 2–8 match in trainable
parameters within 5% (5,584 / 5,610 / 5,638). The runner asserts this before any run
starts, and it is tested (`tests/test_connectome.py`, `tests/test_flygm.py`).
Arms 3–8 share node roles: context enters through the 16 PPL1 neurons, and the policy
is read out from the 5 AVLP727m neurons.

Arms 3–5 ask whether the connectome's *sparsity* helps when every edge is free to
learn its weight; arms 6–8 ask whether its *weights* help when they are kept exactly
and the neurons learn instead (after Jin et al. 2026, arXiv:2602.17997). A control
graph carries a seeded permutation of the real synapse-count multiset, so it matches
the real circuit in sparsity, degree sequence (arm 7) and weight distribution, and
differs only in which pair of neurons each count sits on.

## Games and protocol

* **Game A** (primary): complete information, prize `V = 1`, cost `c = 1`. The
  symmetric equilibrium is `T ~ Exp(c/V)`.
* **Game B**: private values `v ~ Exp(1)`. The BNE is `b(v) = v²/2`.

Quitting times lie on a 64-point grid, `T_max = 8` for Game A and `12` for Game B.
Exploitability (ex-ante utility loss, Bichler et al. 2021) is computed **exactly** on
the grid.

* **Exact-gradient self-play:** one shared policy ascends its exact expected payoff
  against a detached copy of itself. Noise-free.
* **Sampled self-play:** a population of 8 independent agents, randomly re-matched
  every round, 128 matches per agent per round. Agents learn only from their own
  sampled quitting times and payoffs (REINFORCE with a running-mean baseline). Ties
  are fair coin flips for the prize. Win/loss asymmetry scales the per-sample
  gradient by `α_loss` or `α_win`, with the ratio swept over {1, 2, 4, 8}.
* **PPO** (`training.sampled_algorithm: ppo`) is the alternative sampled learner:
  4 clipped updates per round on the same batch, clip 0.2, no entropy bonus, the same
  running-mean baseline as critic, and the same win/loss weights. It runs for every
  arm, FlyGM included.
* **Optimizer:** the same for every arm (Adam with β₁ = 0, cosine decay). Learning
  rates are picked per arm from a fixed grid on tuning seeds 1000+, which are
  disjoint from evaluation seeds 0–19 (`configs/tuned_lr.yaml`,
  `results/tuning/*.csv`).
* **Robustness:** the circuit arms are re-run at **one shared learning rate**
  (`configs/robust_*_shared_lr.yaml`), so that the connectome and its controls differ
  only in their mask. Per-arm tuning on 2 seeds had produced a learning-rate confound
  that the controls exposed; see RESULTS.md §5.
* **Seeds:** 20 per configuration in exact mode, 10 in sampled mode (compute budget;
  see RESULTS.md). Every reported number is a mean with a 95% bootstrap CI. Every
  arm-vs-arm contrast has a bootstrap CI on the difference, Hedges' g and a
  Holm-adjusted permutation p-value.

## Reproduce

CPU only. Requires Python ≥ 3.10.

```bash
python -m venv .venv
.venv/Scripts/pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv/Scripts/pip install -r requirements.txt
.venv/Scripts/python -m pytest
.venv/Scripts/python scripts/reproduce.py
```

On Linux/macOS use `.venv/bin/`. `make reproduce` does the same where `make` exists.

`scripts/reproduce.py` runs the whole pipeline:

1. Downloads the three public MaleCNS files (about 566 MB, CC-BY) if absent,
   rebuilds the subgraph, and checks it against the committed checksum.
2. Tunes learning rates.
3. Runs the four experiments.
4. Writes every statistic to `results/tables/`.
5. Redraws every figure in `figures/`.

`--skip-connectome` and `--skip-tuning` reuse the committed subgraph and learning
rates. The measured wall time is in RESULTS.md.

The FlyGM and PPO experiments (RESULTS.md §9–§11) are a second, much longer grid:

```bash
.venv/Scripts/python scripts/reproduce_flygm_ppo.py --list
.venv/Scripts/python scripts/reproduce_flygm_ppo.py --workers 14
```

It runs in named steps, and appends each step's measured wall time to
`results/flygm_ppo_log.txt`.

## Layout

```
src/games/        payoff engine, equilibria (analytic, grid, asymmetric-update rest point)
src/agents/       tabular / dense / sparse-circuit policies, arm factory, parameter matching
src/connectome/   MaleCNS download + extraction + provenance, rewired / random controls
src/training/     exact & sampled self-play, runner, learning-rate tuning
src/metrics/      exact exploitability & distances, bootstrap statistics
src/analysis.py   every number in RESULTS.md
configs/          one YAML per experiment (+ base.yaml, tuned_lr.yaml); no magic numbers in code
scripts/          entry points (reproduce, build_connectome, tune_lr, run_experiment, analyze, make_figures)
data/connectome/  the extracted subgraph, node/edge tables, provenance record (committed)
results/          per-run JSON (with full config snapshot and seed), summary/curve CSVs, tables
figures/          PNG + PDF
tests/            pytest (payoffs, oracle, tabular convergence, b(v) best response,
                  parameter matching, degree preservation, no synthetic fallback, ...)
```

## Data and citations

* MaleCNS v1.0: Berg, S. et al. *Sexual dimorphism in the complete Drosophila male
  central nervous system connectome.* Cell 189 (2026). CC-BY 4.0.
* Ventimiglia, D. et al. *Neural circuits regulating social dominance implement a
  strategy predicted by evolutionary game theory.* bioRxiv 2026.02.20.706896.
* Bichler, M., Fichtl, M., Heidekrüger, S., Kohring, N., Sutterer, P. *Learning
  equilibria in symmetric auction games using artificial neural networks.* Nature
  Machine Intelligence 3, 687–695 (2021).
