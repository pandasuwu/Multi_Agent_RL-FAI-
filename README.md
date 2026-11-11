# Fly give-up circuit and the war of attrition

CS329 Foundations of AI: Multi-Agent Systems, IIT Gandhinagar, 2026. Project by Suhani,
Mayank Choudhary, Krina Paghadar and Nakshatra Lad.

Question: does building a learning agent out of a real neural circuit change how close
it gets to equilibrium in a strategic game?

The game is the war of attrition. Two players contest a prize, both pay for every
second the contest lasts, and whoever quits first loses. Male fruit flies fighting over
food play it, and Ventimiglia et al. 2026 found the circuit that makes them give up:
PPL1 → V2 MBONs (MBON18/19) → Tk-GAL4^FruM (AVLP727m). We cut that circuit out of the
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

Short version: we found no effect of the wiring on how close the agents get to
equilibrium. Two caveats. The games ask little of the network (one constant input in
Game A, one number in Game B), and in sampled Game B the 10-seed confidence intervals
are wide enough to hide an effect of a quarter of the loss (§8 and the Summary). Full
numbers are in [RESULTS.md](RESULTS.md); section numbers below point there.

- With trained edges, the real circuit ends at the same exploitability as a
  degree-preserving rewire or a random sparse graph, in both games and both training
  modes (§3–§4). The one clear difference is stability: in Game B at three times its
  tuned learning rate, the unsigned real circuit converges on 7 of 20 seeds against 15
  and 16 for its controls.
- With frozen synapse counts as weights (FlyGM), Game A works and shows no wiring
  effect. Game B collapses to a single bid on all three graphs, because of the 5-neuron
  readout: reading out from all 258 neurons fixes it (0.33 → 0.011). Still no wiring
  effect after that (§9, §12).
- The readout matters more than the wiring. The wider readout improves every circuit
  arm in Game B and removes most of the gap to the dense MLP (§12).
- PPO reaches the threshold up to 4x sooner in Game A; its effect on final accuracy
  depends on the game (§10).
- The equilibrium is socially wasteful. Learners that weight losses more than wins quit
  earlier, which raises exploitability and nearly triples welfare (0.26 to 0.72 of the
  prize, §13).
- The number of agents changes the outcome more than wiring does. A fixed pair usually
  splits roles and hardly fights; from 8 agents up the population settles on the
  symmetric equilibrium, and at 16 every arm ends at the same exploitability (§14).
- Review checks (§15): population-level Nash, the loss-weighted rest point, input-output
  paths in the controls, and a shared learning-rate sweep. None changes the conclusion.

![Final exploitability by arm in the four main settings](figures/fig3b_main_four_settings.png)

What would count as an answer: a difference between the real circuit and its
degree-preserving rewire that holds at a shared learning rate and survives Holm
correction would be a wiring effect; confidence intervals tight around zero would be
evidence of none. The exact-gradient settings give the second. The sampled settings
give neither firmly.

The project report is [final_report.pdf](final_report.pdf).

Other docs: [docs/theory.md](docs/theory.md) (equilibria, exploitability),
[docs/circuit.md](docs/circuit.md) (how the subgraph was extracted).

There's also a small [browser demo](https://pandasuwu.github.io/Multi_Agent_RL-FAI-/)
where you can watch agents learn both games with the fly wiring or its controls
(`python scripts/build_demo.py`).

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
transmitter predictions). Arms 2–8 are matched in trainable parameters; the runner
checks this before starting. In arms 3–8 the context goes in through the 16 PPL1 neurons
and the policy comes out of the 5 AVLP727m neurons; §12 reruns them with a rank-4
readout from all 258 neurons.

Arms 3–5 test whether the connectome's sparsity pattern helps when each edge learns its
own weight. Arms 6–8 test whether its actual weights help when they're kept fixed and
the neurons learn instead (the FlyGM setup from Jin et al. 2026).

## Games and training

Game A: complete information, V = 1, c = 1, symmetric equilibrium T ~ Exp(c/V).
Game B: private values v ~ Exp(1), BNE b(v) = v²/2.

Quit times are on a 64-point grid (T_max = 8 for A, 12 for B). Exploitability is the
ex-ante utility loss from Bichler et al. 2021, computed exactly on the grid.

- Exact mode: one shared policy follows the exact gradient of its payoff against a
  detached copy of itself.
- Sampled mode: 8 agents (2 to 16 in §14), re-matched randomly each round, 128 matches
  per agent per round, REINFORCE with a running-mean baseline, or PPO. For the asymmetry
  experiments the gradient is scaled by α_loss or α_win, ratio in {1, 2, 4, 8}.
- Learning rates are tuned per arm on separate seeds (`configs/tuned_lr.yaml`). That
  gave a learning-rate confound (RESULTS §5), so the circuit arms are also rerun at one
  shared rate.
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

On Linux/macOS use `.venv/bin/` instead of `.venv/Scripts/`. `requirements-lock.txt` is
the exact laptop environment; with it the laptop runs reproduce bit for bit. The runs of
§14 and §15 were made on another machine and reproduce to about 0.01 in final
exploitability on single seeds.

`reproduce.py` downloads the three MaleCNS files (~566 MB, CC-BY) if they aren't there,
rebuilds the subgraph, tunes learning rates, runs the four main experiments, and writes
`results/tables/` and `figures/`. Use `--skip-connectome` and `--skip-tuning` to reuse
the committed subgraph and learning rates.

The other experiments:

```bash
.venv/Scripts/python scripts/reproduce_flygm_ppo.py --workers 6
```

runs the FlyGM, PPO and readout experiments (§9–§12; much longer). The population-size
sweep and welfare analysis (§13–§14):

```bash
.venv/Scripts/python scripts/run_experiment.py configs/popsize_game_a_sampled.yaml --workers 2
.venv/Scripts/python scripts/analyze.py
.venv/Scripts/python scripts/welfare.py
.venv/Scripts/python scripts/make_figures.py
```

The checks after the project review (§15): `make review`, then `make figures`.

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
- Jin, Z., Zhu, Y., Zhang, C., Sui, Y. Whole-brain connectomic graph model enables
  whole-body locomotion control in fruit fly. arXiv:2602.17997 (2026).
