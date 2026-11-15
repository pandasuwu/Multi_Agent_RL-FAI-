# Fly give-up circuit and the war of attrition

CS329 Foundations of AI: Multi-Agent Systems, IIT Gandhinagar, 2026. Project by Suhani,
Mayank Choudhary, Krina Paghadar and Nakshatra Lad.

Question: when independent agents learn the war of attrition from their own fights, what
decides how close they get to equilibrium: the learning rule, the number of agents, what
the learner is given, or how its policy network is wired?

The game is the war of attrition. Two players contest a prize, both pay for every
second the contest lasts, and whoever quits first loses. Male fruit flies fighting over
food play it, and Ventimiglia et al. 2026 found the circuit that makes them give up:
PPL1 → V2 MBONs (MBON18/19) → Tk-GAL4^FruM (AVLP727m). That circuit is our wiring
factor. We cut it out of the MaleCNS v1.0 connectome and used it as the policy network
of learning agents, against size-matched controls with the same neurons but different
wiring.

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

Short version: the learning rule, the number of agents, the learning signal and the
readout each move the outcome. The wiring does not, in the one setting that tests it
fairly. Full numbers are in [RESULTS.md](RESULTS.md); section numbers below point there.

| factor | what we varied | final exploitability | § |
|---|---|---|---|
| Learning rule | losses weighted 1, 2, 4, 8 times wins (sampled Game A, mean of 8 arms) | 0.048, 0.102, 0.130, 0.146 (rest point predicted at 0, 0.066, 0.108, 0.130) | §6, §15.2 |
| Number of agents | 2, 4, 8, 16 (sampled Game A, range over 5 arms) | 0.04–0.08, 0.06–0.11, 0.03–0.07, 0.033–0.035 | §14 |
| Learning signal | exact payoff gradient vs own sampled fights (connectome arm) | Game A 0.0013 vs 0.045; Game B 0.0082 vs 0.154 | §3 |
| Readout | 5 output neurons vs all 258 (Game B exact) | trained edges 0.0082 → 0.0039; frozen synapses 0.33 → 0.011 | §12 |
| Wiring | real circuit vs degree-preserving rewire (Game B exact) | 0.0082 vs 0.0078, difference +0.0004 [−0.0002, 0.0015] | §4 |

- Weighting losses more than wins moves the learners to a different rest point: the
  equilibrium of a modified game. We derive it (a Lomax quitting law instead of the
  exponential) and check it against simulations of the learning rule. Exploitability
  rises with the weight on every arm and sits 0.02–0.05 above the prediction, about the
  sampling-noise floor (§6, §15.2).
- The equilibrium is socially wasteful, and learners further from it waste less. In
  Game A the equilibrium burns almost the whole prize. Loss-weighted learners quit
  earlier, which raises exploitability and nearly triples welfare (0.26 to 0.72 of the
  prize, §13).
- The number of agents changes what is learned. A fixed pair usually splits roles (the
  asymmetric equilibrium) and hardly fights. Four agents learn worst. From 8 up the
  population approaches the symmetric equilibrium, and at 16 every arm ends at the same
  exploitability (§14).
- Learning from sampled fights ends well above the exact-gradient result. In Game B no
  network arm reaches the 0.05 threshold under REINFORCE. PPO gets the dense network and
  two signed circuits there, and reaches the threshold up to 4x sooner in Game A
  (§3, §10).
- The readout matters more than the wiring. Reading the policy from all 258 neurons
  instead of 5 improves every circuit arm in Game B, removes most of the gap to the
  dense MLP, and fixes the frozen-synapse (FlyGM) arms, which otherwise collapse to a
  single bid on all three graphs (§9, §12).
- Wiring: no effect found on final exploitability, with trained edges or frozen
  synapses (§3–§4, §9, §12). How much that says depends on the setting:
  - Game B exact is the fair test: the learners converge and the policy has to depend
    on the input. The intervals on real-minus-control are 10–21% of the loss.
  - Game A cannot show a wiring effect on the final policy. Its input is a constant, so
    the 64 output biases can represent any policy whatever the graph. Wiring can only
    change the path there, and a small speed-up is what we see (15–20 episodes out of
    about 217, §11).
  - Sampled Game B compares learners that have not converged, and its 10-seed intervals
    are wide enough to hide an effect of a quarter of the loss (§8).
  - The one clear wiring difference goes against the real circuit: in Game B at three
    times its tuned learning rate, the unsigned real circuit converges on 7 of 20 seeds
    against 15 and 16 for its controls (§4).
- Review checks (§15): population-level Nash, the loss-weighted rest point, input-output
  paths in the controls, and a shared learning-rate sweep. None changes the conclusion.

![Final exploitability by arm in the four main settings](figures/fig3b_main_four_settings.png)

What would count as a wiring effect: a difference between the real circuit and its
degree-preserving rewire that holds at a shared learning rate and survives Holm
correction. Confidence intervals tight around zero would be evidence of none. Game B
exact gives the second. Game A exact gives it too, but would for any wiring. The sampled
settings give neither firmly.

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
