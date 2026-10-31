# Fly give-up circuit and the war of attrition

Two agents play a war of attrition and learn by self-play. I wanted to know whether
building their policy networks out of the real Drosophila give-up circuit changes how
close (or how fast) they get to the equilibrium, compared with ordinary networks of the
same size. The circuit is PPL1 → V2 MBONs (MBON18/19) → Tk-GAL4^FruM (AVLP727m), taken
from the MaleCNS v1.0 connectome, following Ventimiglia et al. 2026.

Short version: the wiring didn't matter. Full numbers are in [RESULTS.md](RESULTS.md);
section numbers below point there.

- With trained edges, the real circuit ends up at the same exploitability as a
  degree-preserving rewire or a random sparse graph, in both games and both training
  modes (§3–§4).
- With frozen synapse counts as weights (FlyGM), Game A works and shows no wiring effect
  (§9). Game B collapses to a single bid on all three graphs. That turned out to be the
  5-neuron readout, not weak synapses: a trainable gain on W does nothing, but reading
  out from all 258 neurons fixes it (0.33 → 0.011, all 60 seeds bid by value). Still no
  wiring effect after that, and FlyGM stays behind the trained-edge arms (§12).
- PPO gets to the threshold up to 4x sooner in Game A. Final accuracy depends on the
  game: it helps dense and the signed circuits in Game B and hurts the connectome arms
  in Game A. One Game B contrast (real circuit worse than random) passes Holm within its
  experiment but not across all 108 wiring tests in the project, so I don't count it (§10).
- The readout matters more than the wiring. Reading out from every neuron improves every
  circuit arm in Game B and removes most of the gap to the dense MLP, along with the
  earlier sign-constraint advantage (§12).

Other docs: [docs/theory.md](docs/theory.md) (equilibria, exploitability),
[docs/circuit.md](docs/circuit.md) (how the subgraph was extracted).

There's also a small browser demo in `demo/index.html` where you can watch agents learn
both games with the fly wiring or its controls. Rebuild it with
`python scripts/build_demo.py` (template in `scripts/demo_template.html`).

## Arms

| # | Arm | Notes |
|---|---|---|
| 0 | Oracle | analytic equilibrium floored onto the grid, used as a check |
| 1 | Tabular | softmax logits per context |
| 2 | Dense MLP | 2 hidden layers of 48 (5,584 params) |
| 3 | Connectome | recurrent net on the 258-neuron / 4,952-edge MaleCNS subgraph, weights only on real edges (5,610 params) |
| 4 | Rewire | same neurons and in/out degrees, edges shuffled by double-edge swaps, new draw per seed |
| 5 | Random sparse | same neurons and edge count, edges uniform at random |
| 6 | FlyGM, real | synapse counts frozen as weights, `W[post, pre] = sign(pre)·synapses`, scaled to spectral radius 1. Trained: 9-dim descriptor per neuron, a shared update MLP `f(h_i, m_i, η_i)` with `m = W h`, encoder and gate on PPL1, readout (5,638 params) |
| 7 | FlyGM, rewire | same on the rewired graph |
| 8 | FlyGM, random | same on the random graph |

Arms 3–5 also have a sign-constrained version (Dale's law, using the MaleCNS
transmitter predictions), so there are 9 trained-edge arms and 12 counting FlyGM.
Arms 2–8 are within 5% of each other in trainable parameters; the runner checks this
before starting and there's a test for it. In arms 3–8 the context goes in through the
16 PPL1 neurons and the policy comes out of the 5 AVLP727m neurons. §12 also reruns
arms 2–8 with a rank-4 readout from all 258 neurons (`network.readout: all`, 6,578
params).

Arms 3–5 test whether the connectome's sparsity pattern helps when each edge learns its
own weight. Arms 6–8 test whether its actual weights help when they're kept fixed and
the neurons learn instead (the FlyGM setup from Jin et al. 2026, arXiv:2602.17997). The
control graphs get a seeded permutation of the real synapse counts, so they match on
sparsity, degree sequence (arm 7) and weight distribution and only differ in which
neuron pair each count lands on.

## Games and training

Game A: complete information, V = 1, c = 1, symmetric equilibrium T ~ Exp(c/V).
Game B: private values v ~ Exp(1), BNE b(v) = v²/2.

Quit times are on a 64-point grid (T_max = 8 for A, 12 for B). Exploitability is the
ex-ante utility loss from Bichler et al. 2021, computed exactly on the grid.

- Exact mode: one shared policy follows the exact gradient of its payoff against a
  detached copy of itself.
- Sampled mode: 8 agents, re-matched randomly each round, 128 matches per agent per
  round, REINFORCE with a running-mean baseline. Ties are a coin flip. For the
  asymmetry experiments the per-sample gradient is scaled by α_loss or α_win, ratio in
  {1, 2, 4, 8}.
- PPO (`training.sampled_algorithm: ppo`): 4 clipped updates per round on the same
  batch, clip 0.2, no entropy bonus, same baseline and win/loss weights. Runs for every
  arm including FlyGM.
- Adam with β₁ = 0 and cosine decay for everything. Learning rates come from a fixed
  grid, tuned on seeds 1000+ (eval seeds are 0–19). See `configs/tuned_lr.yaml`.
- Per-arm tuning on 2 seeds gave a learning-rate confound (RESULTS §5), so the circuit
  arms are also rerun at one shared learning rate (`configs/robust_*_shared_lr.yaml`).
- 20 seeds in exact mode, 10 in sampled mode. Means come with 95% bootstrap CIs;
  pairwise contrasts with a bootstrap CI, Hedges' g and a Holm-adjusted permutation p.

## Running it

CPU only, Python 3.10+.

```bash
python -m venv .venv
.venv/Scripts/pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv/Scripts/pip install -r requirements.txt
.venv/Scripts/python -m pytest
.venv/Scripts/python scripts/reproduce.py
```

(`.venv/bin/` on Linux/macOS, or `make reproduce`.)

`reproduce.py` downloads the three MaleCNS files (~566 MB, CC-BY) if they aren't there,
rebuilds the subgraph and checks its checksum, tunes learning rates, runs the four
experiments, and writes `results/tables/` and `figures/`. Use `--skip-connectome` and
`--skip-tuning` to reuse the committed subgraph and learning rates.

The FlyGM/PPO grid (§9–§11) is separate and much longer:

```bash
.venv/Scripts/python scripts/reproduce_flygm_ppo.py --list
.venv/Scripts/python scripts/reproduce_flygm_ppo.py --workers 14
```

Wall times get appended to `results/flygm_ppo_log.txt`.

## Layout

```
src/games/        payoffs, equilibria
src/agents/       tabular, dense, circuit and FlyGM policies, arm factory
src/connectome/   MaleCNS download and extraction, rewired/random controls
src/training/     exact and sampled self-play, runner, LR tuning
src/metrics/      exploitability, bootstrap stats
src/analysis.py   numbers in RESULTS.md
configs/          one YAML per experiment
scripts/          entry points
data/connectome/  extracted subgraph and provenance
results/          per-run JSON, summary CSVs, tables
figures/          PNG + PDF
tests/            pytest
```

## References

- Berg, S. et al. Sexual dimorphism in the complete Drosophila male central nervous
  system connectome. Cell 189 (2026). Data CC-BY 4.0.
- Ventimiglia, D. et al. Neural circuits regulating social dominance implement a
  strategy predicted by evolutionary game theory. bioRxiv 2026.02.20.706896.
- Bichler, M., Fichtl, M., Heidekrüger, S., Kohring, N., Sutterer, P. Learning
  equilibria in symmetric auction games using artificial neural networks. Nature
  Machine Intelligence 3, 687–695 (2021).
- Jin et al. 2026, arXiv:2602.17997.
