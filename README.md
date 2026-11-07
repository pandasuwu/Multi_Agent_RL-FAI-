# Fly give-up circuit and the war of attrition

Question: does building a learning agent out of a real neural circuit change how close
it gets to equilibrium in a strategic game?

The game is the war of attrition. Two players contest a prize, both pay for every
second the contest lasts, and whoever quits first loses. Male fruit flies fighting over
food play it, and Ventimiglia et al. 2026 found the circuit that makes them give up:
PPL1 → V2 MBONs (MBON18/19) → Tk-GAL4^FruM (AVLP727m). I cut that circuit out of the
MaleCNS v1.0 connectome and used it as the policy network of learning agents, against
size-matched controls with the same neurons but different wiring.

- Agents: 8 independent learners (2 to 16 in §14), each choosing a quitting time.
- Known and unknown: the equilibrium is known in closed form, but the agents don't know
  it. They either follow the exact payoff gradient (exact mode) or learn only from the
  fights they play (sampled mode, multi-agent REINFORCE or PPO). In Game B each agent
  also doesn't know its opponent's value.
- Baselines: degree-preserving rewire, random sparse graph, dense MLP and a look-up
  table, all matched in trainable parameters.
- What changes: information (Game A vs B), the learning rule (losses counting 1 to 8
  times more than wins), REINFORCE vs PPO, trained vs frozen synapses, the readout, and
  the number of agents.
- Metrics: exact exploitability (how much a best responder could gain), episodes to
  threshold, and welfare against the social optimum (§13).

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
- The equilibrium is socially wasteful. In Game A it burns the whole prize in fighting
  (88% on the grid); in Game B it gets the prize to the higher value but pays more for
  that than a no-fight coin flip would lose. Learners that weight losses more than wins
  quit earlier, which raises exploitability and roughly doubles welfare (§13).
- Number of agents changes the outcome more than wiring does. A fixed pair usually
  splits roles (one gives in at once), which is an equilibrium with almost no fighting.
  From 8 agents up the population settles on the symmetric mixed equilibrium, and at 16
  every arm, real wiring included, ends at the same exploitability (§14).

- Review checks (§15). The population mixture is further from Nash than each agent
  against the others, but no wiring conclusion changes on that metric. The derived
  rest point of loss-weighted learning matches simulations of the rule. Every control
  graph keeps all PPL1 → AVLP727m paths. A sweep over shared learning rates shows the
  Game A speed-up is a wiring effect, not tuning, and finds nothing else.

Other docs: [docs/theory.md](docs/theory.md) (equilibria, exploitability),
[docs/circuit.md](docs/circuit.md) (how the subgraph was extracted).

There's also a small browser demo at https://pandasuwu.github.io/Multi_Agent_RL-FAI-/ where you can watch agents learn
both games with the fly wiring or its controls. It is built from
`scripts/demo_template.html` by `python scripts/build_demo.py`, and a GitHub Action
republishes it on every push to `main` that touches the demo.

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
- Sampled mode: 8 agents (2 to 16 in §14), re-matched randomly each round, 128 matches per agent per
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

The welfare analysis and the population-size sweep (§13–§14):

```bash
.venv/Scripts/python scripts/run_experiment.py configs/popsize_game_a_sampled.yaml --workers 2
.venv/Scripts/python scripts/welfare.py
.venv/Scripts/python scripts/make_figures.py
```

## Layout

```
src/games/        payoffs, equilibria
src/agents/       tabular, dense, circuit and FlyGM policies, arm factory
src/connectome/   MaleCNS download and extraction, rewired/random controls
src/training/     exact and sampled self-play, runner, LR tuning
src/metrics/      exploitability, welfare, bootstrap stats
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
