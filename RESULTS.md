# Results

## In thirty seconds

**Connectome wiring made no difference to how well the agents converged.** In all four
settings (Game A and Game B, exact and sampled self-play), the final exploitability of
the policy constrained to the real fly give-up circuit cannot be distinguished from
two controls with the same parameter count: a degree-preserving rewire of the same
circuit and a random sparse graph. Every such contrast has a bootstrap CI that
includes zero, or fails Holm correction.

Headline, Game A (the game that matches the fly data), exact self-play, 20 seeds.
Final exploitability, mean [95% CI]:

* real connectome: 0.00127 [0.00118, 0.00144]
* degree-preserving rewire: 0.00127 [0.00126, 0.00128]
* random sparse: 0.00130 [0.00128, 0.00133]
* connectome − rewire: **+0.000002 [−0.00009, +0.00018]**, Hedges' g = 0.01

**On speed, the wiring effects are small, inconsistent in sign and near the
measurement resolution.**

* Game A, exact, unsigned: the connectome reached the threshold 19–24 episodes
  earlier than its controls (226 vs 245 and 250, about 8–10% faster; Holm p = 0.001).
  That gap is under one evaluation interval (25 episodes), and the sign-constrained
  version does not show it.
* Game B, exact, signed: the connectome was 45–50 episodes (about 9%) *slower* than
  its signed controls. It also learned a less accurate bid function for the
  highest-value types (bid L2 +0.10 to +0.23).

No setting shows fly wiring *helping* convergence robustly. The one direction that
recurs in Game B is a small *hurt*.

What *did* matter:

* **Architecture class.** The dense MLP beat every circuit arm in 3 of 4 settings.
  The circuit arms beat dense only in Game A, exact.
* **Sign constraints (Dale's law).** They improved all three circuit graphs in Game B,
  real and control alike.
* **The fly-like asymmetric learning rule.** It moved every arm away from equilibrium,
  exactly as the theory predicts, and amplified a loser effect that **wiring did not
  modulate**.

---

## 1. Setup in brief

| | |
|---|---|
| Arms | oracle, tabular, dense MLP (5,584 params), connectome, degree-preserving rewire, random sparse (5,610 params each; ±Dale's-law sign constraint) |
| Circuit | MaleCNS v1.0: PPL1 (16) → MBON18/19 (6) → AVLP727m = Tk-GAL4^FruM (5), with 54 path neurons and 177 one-hop partners; 258 neurons and 4,952 edges ([docs/circuit.md](docs/circuit.md)) |
| Games | A: `V = c = 1`, NE `Exp(1)`. B: `v ~ Exp(1)`, BNE `v²/2`. 64-point grids ([docs/theory.md](docs/theory.md)) |
| Exact mode | one shared policy, exact expected-payoff gradient; 20 seeds; 3,000 (A) or 4,000 (B) episodes |
| Sampled mode | 8 independent agents, randomly re-matched, 128 matches per agent per episode, REINFORCE with baseline; 10 seeds; 3,000 episodes; `α_loss/α_win ∈ {1, 2, 4, 8}` (Game A) |
| Primary metric | exact ex-ante utility loss (exploitability) on the grid, Bichler et al. 2021 conventions, absolute units (Game A equilibrium payoff = 0) |
| Threshold | 0.01 (exact), 0.05 (sampled); time to threshold censored at the training length |
| Statistics | mean with 95% percentile bootstrap CI (10,000 resamples). Each contrast: bootstrap CI of the difference, Hedges' g, permutation p, Holm-adjusted within each experiment × metric family. **"Distinguishable" = CI excludes 0 and Holm p < 0.05.** |

All tables are in `results/tables/`: `per_arm.csv`, `pairwise.csv`, `asym_*.csv` and
`game_b_deciles.csv`. Per-run records with the full config snapshot and seed are in
`results/<experiment>/runs/`.

To watch the setup run, open the live demo, [Give-Up Game Lab](https://claude.ai/artifact/7RDFGua3hKQyYSKt8W7RKN)
(offline copy: [`demo/index.html`](demo/index.html)). It trains the same arms on both games in the
browser, with a smaller setup (10 value levels in Game B, no seeds or statistics). It illustrates
the method; none of the numbers in this report come from it.

## 2. Did every piece work? (sanity checks)

* **Oracle:** grid exploitability 0.00149 in Game A and 0.0058 in Game B. These are the
  discretisation floors; the exact grid equilibrium has 0.
* **Tabular, Game A exact:** converges to 0.0032 and crosses 0.01 at episode 650. Its
  learned survival curve lies on `exp(−t)` (`figures/fig1_*`).
* **`b(v) = v²/2` is a best response to itself:** confirmed by quadrature, with no grid
  (`tests/test_games.py`).
* **Parameter counts:** matched at 5,584 vs 5,610 (0.5%). This is asserted before every
  run, and every rewired control preserves every in- and out-degree exactly (asserted).
* **Rebuild:** the subgraph rebuilt from the raw MaleCNS files reproduced the committed
  SHA-256 exactly.

## 3. Final exploitability by arm

Mean [95% CI]. Source: `results/tables/per_arm.csv`. Figure: `figures/fig3_main_comparison.png`.

| arm | A exact (20) | A sampled, ρ=1 (10) | B exact (20) | B sampled (10) |
|---|---|---|---|---|
| oracle | 0.00149 | 0.00149 | 0.0058 | 0.0058 |
| tabular | 0.00318 | 0.065 [0.056, 0.077] | 0.0095 | 0.053 [0.051, 0.054] |
| dense MLP | 0.00159 [0.00157, 0.00160] | **0.033** [0.031, 0.037] | **0.0029** [0.0026, 0.0033] | 0.072 [0.061, 0.084] |
| connectome | 0.00127 [0.00118, 0.00144] | 0.045 [0.042, 0.049] | 0.0082 [0.0077, 0.0093] | 0.154 [0.133, 0.171] |
| connectome, signed | 0.00127 [0.00125, 0.00130] | 0.045 [0.041, 0.049] | 0.0060 [0.0051, 0.0070] | 0.094 [0.088, 0.101] |
| rewire | 0.00127 [0.00126, 0.00128] | 0.057 [0.050, 0.065]* | 0.0078 [0.0078, 0.0079] | 0.095 [0.088, 0.102]* |
| rewire, signed | 0.00125 [0.00122, 0.00128] | 0.047 [0.042, 0.052] | 0.0052 [0.0049, 0.0056] | 0.090 [0.070, 0.121] |
| random sparse | 0.00130 [0.00128, 0.00133] | 0.049 [0.043, 0.056] | 0.064 [0.009, 0.146]† | 0.143 [0.116, 0.172] |
| random sparse, signed | 0.00132 [0.00125, 0.00145] | 0.044 [0.040, 0.048] | 0.0054 [0.0049, 0.0062] | 0.079 [0.068, 0.090] |

\* Tuned to a different learning rate from the connectome (§5). †Three of 20 seeds ended far from equilibrium (0.12, 0.28, 0.72), and 4 never reached the threshold.

Circuit arms in Game A exact (about 0.00127) sit *below* the oracle. They learn the
exact grid equilibrium (exploitability 0) better than the floored analytic law does.

## 4. The research question: specific wiring vs controls

Differences in final exploitability, connectome minus control, with 95% CI [Holm p].
Negative means the connectome is better. Source: `results/tables/pairwise.csv`.

| setting | vs rewire (unsigned) | vs rewire (signed) | vs random (unsigned) | vs random (signed) |
|---|---|---|---|---|
| A exact | +0.000002 [−0.00009, 0.00018] (1.0) | +0.00002 [−0.00002, 0.00006] (1.0) | −0.00003 [−0.00013, 0.00015] (1.0) | −0.00005 [−0.00018, 0.00003] (1.0) |
| A sampled, shared lr | +0.0004 [−0.0054, 0.0061] (1.0) | −0.0015 [−0.0082, 0.0050] (1.0) | −0.0031 [−0.0109, 0.0038] (1.0) | +0.0010 [−0.0052, 0.0069] (1.0) |
| B exact | +0.0004 [−0.0002, 0.0015] (1.0) | +0.0007 [−0.0002, 0.0019] (0.76) | −0.056 [−0.140, −0.002] (0.22) | +0.0005 [−0.0007, 0.0018] (1.0) |
| B sampled, shared lr | −0.027 [−0.069, 0.012] (0.92) | +0.004 [−0.027, 0.026] (1.0) | +0.011 [−0.025, 0.044] (1.0) | +0.015 [0.003, 0.028] (0.22) |

**No cell is distinguishable from zero.** The effect sizes are near zero wherever the
CIs are tight (Game A exact: |g| ≤ 0.3).

Where speed or secondary metrics *are* distinguishable at a matched learning rate:

| setting | contrast | effect | Holm p |
|---|---|---|---|
| A exact | connectome − rewire (unsigned), episodes to 0.01 | −19 [−24, −14] (g = −2.2) | 0.001 |
| A exact | connectome − random (unsigned), episodes to 0.01 | −24 [−28, −20] (g = −3.3) | 0.001 |
| A exact | same contrasts, signed | +15 [5, 25] and +10 [0, 20] | 0.08 / 0.36, not distinguishable |
| B exact | connectome − random (signed), episodes to 0.01 | +50 [18, 83] (g = 0.94) | 0.035 |
| B exact | connectome − rewire (signed), episodes to 0.01 | +45 [17, 75] (g = 0.89) | 0.050 (0.036 in the shared-lr run) |
| B exact | connectome − rewire (signed), bid L2 | +0.109 [0.039, 0.189] (g = 0.85) | 0.022 |
| B sampled | connectome − random (signed), bid L2 | +0.227 [0.102, 0.341] (g = 1.5) | 0.025 |

The Game B L2 gap is concentrated in the **top value decile**: bid MSE 0.99 for the
signed connectome vs 0.24 (rewire) and 0.28 (random). Those types bid 7–10 time units,
where the interim payoff is nearly flat, so a weak gradient there barely moves utility
loss (`figures/fig6_game_b_bids.png`). Deciles 1–9 are indistinguishable across all
signed arms.

**Reading.** Degree structure (rewire vs random) never mattered for final loss either.
The only wiring-specific effects are:

* a sub-evaluation-interval speed-up in Game A exact that vanishes under sign
  constraints;
* a small, consistent *slow-down plus worse high-value bids* for the signed real
  circuit in Game B.

Neither changes how well the agents end up playing.

## 5. A confound that the controls caught

Per-arm learning-rate tuning, with 2 tuning seeds per grid point, gave the unsigned
rewired control a rate **10× higher** than the connectome in sampled Game A (0.03 vs
0.003) and 3× *lower* in sampled Game B (0.003 vs 0.01). At those rates the "wiring"
contrasts looked significant:

* Game B sampled: connectome worse by +0.059 [0.036, 0.078], Holm p = 0.004.
* Game A sampled: connectome better on TV and on the loser effect (§6).

We re-ran every circuit arm at **one shared learning rate** (the modal tuned value,
`configs/robust_*_shared_lr.yaml`), so that the arms differ only in their mask. Every
such effect disappears:

* Game B sampled: −0.027 [−0.069, 0.012].
* Game A sampled: +0.0004 [−0.0054, 0.0061].

The shared-lr runs are reported alongside the per-arm-tuned ones in every table. We
treat only effects that survive both protocols as real.

## 6. Asymmetric win/loss updates and the loser effect (sampled Game A)

Source: `results/tables/asym_*.csv`, `figures/fig4_asymmetric_sweep.png`.

**The rest point moves exactly where the theory says.** With a reward-tracking
baseline, loss-weighted REINFORCE converges to the equilibrium of a *modified* game, not
the real one (docs/theory.md §3). The prediction is a decreasing hazard, more immediate
quits and a heavy tail, with an exploitability floor. Mean over arms:

| ρ = α_loss/α_win | 1 | 2 | 4 | 8 |
|---|---|---|---|---|
| predicted exploitability of the rest point | 0 | 0.066 | 0.108 | 0.130 |
| observed final exploitability (mean of 8 arms) | 0.048 | 0.102 | 0.130 | 0.146 |

Every arm's final exploitability rises monotonically with ρ. Every increase over ρ = 1
is distinguishable (Holm p ≤ 0.0024, g = 1.5–10). Observed values run about 0.015–0.04
above the prediction, which is roughly the symmetric sampling-noise floor. Asymmetry
**hurts convergence in every architecture**. No wiring contrast at any ρ is
distinguishable (`asym_pairwise.csv`). Speed cannot be compared at ρ ≥ 2 because almost
no run gets under 0.05.

**Winner/loser effect.** For every agent and round we regressed the *exact* change in
its mean quitting time caused by its own update on the fraction of that round's 128
matches it lost. This is a within-agent measure, free of between-agent heterogeneity.

* Losing lowers the next-round quitting time in **every arm, even at ρ = 1**: about
  −6·10⁻⁶ per lost match relative to a won one. So an all-loss round vs an all-win
  round moves the mean by about −0.0008, against a mean near 0.9. This is mechanical:
  winners' long quitting times get reinforced by a positive advantage, and losers'
  quitting times carry no such push.
* Asymmetric updates amplify it about 1.8×, to about −1.1·10⁻⁵ per loss at ρ = 4–8.
  That is distinguishable from ρ = 1 in 7 of 8 arms at ρ = 4 and 8.
* **Wiring does not modulate it.** With signed arms, or with the shared learning rate,
  every connectome-vs-control difference is ≤ 10⁻⁶ with a CI that includes 0. The
  apparent unsigned connectome-vs-rewire difference is the learning-rate artefact of §5:
  at the shared rate, rewired gives −7·10⁻⁶, the same as the others.
* The *raw* behavioural measure (next-round sampled quitting time vs loss fraction) is
  about 10³× larger, −0.015 to −0.047, because agents that lose are agents that quit early.
  That confound is exactly why the within-agent measure is the one reported.

So the learning agents do show a loser effect of the fly kind. It is small, it grows
with loss-weighting, and it comes from the learning rule, not from the fly wiring.

## 7. Other findings

* **Sign constraints help in Game B.** Signed minus unsigned final loss:
  * exact: −0.0023 (connectome), −0.0026 (rewire), −0.058 (random); all Holm p ≤ 0.006;
  * sampled, shared-lr run: −0.059, −0.090, −0.064; all Holm p ≤ 0.016.

  Unsigned circuits are also less stable at high learning rates. In the shared-lr
  Game B exact run at lr 0.03, only 35% of unsigned-connectome seeds reached the
  threshold. In Game A the constraint makes no difference.
* **Dense vs circuit.** Dense is better in A sampled (0.033 vs 0.045), B exact (0.0029
  vs 0.0060) and B sampled (0.072 vs 0.094); all Holm p ≤ 0.04. The circuits are
  better and faster in A exact: 0.00127 vs 0.00159, and 226 vs 341 episodes. Plausible
  cause: the circuit reads out from only 5 AVLP727m neurons, which suits a
  one-context distribution but is a bottleneck for Game B's value-dependent bids. This
  is a property of the whole architecture, shared with its controls, not of the wiring.
* **Stability.** Late-training policy variance is tiny in exact mode (< 2·10⁻⁶ for
  every arm in Game A). No arm orbits the equilibrium. In sampled mode the
  variance reflects sampling noise.

## 8. What failed, and what we could not establish

* **Game A has a single context, so any policy is a reparameterisation of one
  distribution.** All circuit arms share the readout bias path, which limits how much
  wiring can matter there by construction. The six circuit variants in Game A exact
  agree to within about 6%. Game B, where the network must compute a value-dependent
  bid, is the stronger test, and there too no final-loss effect survives.
* **Sampled REINFORCE never reaches near-zero exploitability.** Tabular and every
  network stall at 0.03–0.05 in Game A: exploitability is first-order in policy noise
  at a mixed equilibrium (docs/theory.md §5). In Game B sampled, no network arm reached
  the 0.05 threshold (tabular: 2 of 10 seeds), so speed there is uninformative.
* **Compute reductions** (the full grid must run on a laptop CPU in under 2 h):
  * sampled modes use the minimum 10 seeds and 3,000 episodes (exact modes use 20);
  * learning rates were tuned on 2 seeds per grid point;
  * several tuned rates sit on a grid edge: dense 0.0003 in A sampled, and tabular 0.03
    in both sampled games, so better rates may exist outside the grid.

  Measured wall time for the whole pipeline on an 8-core laptop (AMD Ryzen AI 7 350,
  12 workers):
  * 59.7 min for the connectome rebuild, tuning, the four experiments, analysis and
    figures (`results/reproduce_log.txt`);
  * +23.7 min for the three shared-learning-rate robustness experiments;
  * **83 min in total.**
* **Only one circuit definition was tested.** It uses the 5-synapse threshold, ≤3-synapse
  MBON→AVLP727m paths, and a one-hop expansion at ≥400 exchanged synapses. We did not
  test sensitivity to these choices, to the unroll depth (6), or to the choice of
  PPL1 as input and AVLP727m as output. Kenyon cells are excluded by the per-neuron
  strength criterion (docs/circuit.md §3).
* **Transmitter signs are EM-based predictions** (26 of 258 neurons are unsigned); we
  did not test alternative sign maps.
* **Independent learners partly specialise.** Deviations in which agents differ are
  unstable at rate ∝ 1/(M−1) (docs/theory.md §5). With 8 agents in sampled Game A,
  the SD of per-agent mean quitting time ends near 0.4–0.5 (dense 0.41, circuits
  0.50–0.51; 1.2 for the rewired arm at lr 0.03). We report individual utility loss;
  population-mixture exploitability is in the per-run files.
* **Not established:**
  * whether a *larger* or differently cut subgraph, or a task where the circuit's
    inputs carry real sensory structure, would reveal a wiring effect;
  * whether the Game B top-decile deficit of the signed connectome reflects its
    reciprocity and clustering (0.43 and 0.39, vs 0.10 and 0.21 after rewiring) or
    something else;
  * ~~whether the Game A exact speed-up (unsigned) is real beyond its 25-episode
    evaluation resolution~~. **Settled in §11**: it is real, at 15 to 20 episodes
    rather than 19 to 24.

---

## 9. What happens if you keep the real synapse strengths?

### The idea, in plain terms

Arms 3 to 5 keep the *shape* of the fly circuit. They know which neuron connects to
which, and then they learn a strength for every one of those 4,952 connections from
scratch. The real synapse counts are thrown away.

The FlyGM arm (arms 6 to 8, after Jin et al. 2026) keeps the real numbers as well:

* Every connection is fixed at its measured synapse count, with a plus or minus sign
  taken from what the sending neuron releases. This matrix `W` never changes. It is
  stored as a buffer, not a parameter, so no gradient can reach it.
* The whole matrix is divided by one number so that its spectral radius is 1. Without
  that, the weights are far too large to pass a signal through sensibly.
* What learns instead is the *neurons*: a 9-number description per neuron, one small
  network shared by all 258 neurons that updates them, the input layer on the 16 PPL1
  neurons, and the readout from the 5 AVLP727m neurons.
* Size is matched like every other arm: **5,638 trainable numbers against 5,610** for
  the connectome arm, a difference of 0.5%. The runner checks this before every run.
* All three graphs get the same treatment, so "real wiring" can still be compared with
  "rewired" and "random".

So arms 3 to 5 ask whether the connectome's *shape* helps. Arms 6 to 8 ask whether its
*strengths* help.

### Two numbers logged for every run

| graph | spectral radius before scaling | input to output gain |
|---|---|---|
| real circuit | 1,230.4 | **0.0079** |
| degree-preserving rewire | 170 to 920 (varies by seed) | 1.4 to 12.8 |
| random sparse | 350 to 450 (varies by seed) | 7.5 to 12.9 |

The gain is `sum over k = 1..6 of |(W^k)[AVLP727m, PPL1]|`. In plain terms: if you put a
signal into the input neurons, how much of it arrives at the output neurons within the
6 steps the network runs for. **The real circuit passes about 1,000 times less than its
own controls do.** The reason is in the first column: the real circuit's radius is
dominated by a strong loop elsewhere in the graph, so dividing by it shrinks the
input-to-output path far more than it shrinks the controls' paths.

One setting matters a lot here. 26 of the 258 neurons have no transmitter prediction,
and **all 16 PPL1 input neurons are among them**. If those are given sign 0, they have
no outgoing weight at all and the gain is exactly 0 on every graph, so the policy cannot
see its input. `unsigned_sign` is therefore set to +1, and every run records which value
it used (`results/tables/flygm_diagnostics.csv`).

### Game A: the arm works, and the wiring makes no difference

Final utility loss, 20 seeds, mean with 95% CI. Source: `results/tables/per_arm.csv`.

| arm | final loss | episodes to 0.01 |
|---|---|---|
| FlyGM, real circuit | 0.001398 [0.001363, 0.001433] | 262.5 |
| FlyGM, rewire | 0.001396 [0.001361, 0.001431] | 263.8 |
| FlyGM, random sparse | 0.001391 [0.001357, 0.001428] | 262.5 |
| oracle (the grid floor) | 0.001488 | n/a |

* No contrast between the three graphs is distinguishable. The largest is 2.4e-06 with
  an effect size of 0.03 and a Holm p of 1.0.
* Speed is the same too: real minus rewire is **-1.25 episodes [-11.3, +8.8]**.
* This is worth comparing with §11. With trained edges, the real circuit crosses the
  threshold 15 episodes before its rewire, and that survives full time resolution. With
  the synapses frozen, that advantage is gone. So the speed effect in §11 comes from the
  *trainable* weights, not from the real connection pattern.
* As with the circuit arms, all three sit just below the oracle, because they learn the
  exact grid equilibrium rather than the floored analytic law.

In Game A sampled the picture is the same: 0.0443, 0.0396 and 0.0439 for real, rewire
and random, and no contrast is distinguishable (smallest Holm p is 0.35).

### Game B: the arm stops working, on all three graphs

This is the main finding of this section, and it is not about wiring.

| arm | final loss | seeds reaching the 0.01 threshold |
|---|---|---|
| FlyGM, real circuit | 0.3311 [0.33112, 0.33114] | **0 of 20** |
| FlyGM, rewire | 0.3290 | 0 of 20 |
| FlyGM, random sparse | 0.3263 | 0 of 20 |
| connectome, trained edges | 0.0082 | 20 of 20 |
| dense MLP | 0.0029 | 20 of 20 |

A confidence interval that narrow across 20 seeds is not noise. Every seed lands in the
same place. Looking at what the policies actually do explains it. In Game B each player
gets a private value `v` and should quit later when `v` is larger. Here is the mean
quitting time each arm chooses for the smallest and the largest value:

| arm | at the smallest value | at the largest value | spread | match to the correct answer |
|---|---|---|---|---|
| FlyGM, real circuit | 0.921 | 0.921 | **0.0000** | -0.00 |
| FlyGM, rewire | 0.920 | 0.920 | **0.0000** | 0.69 |
| FlyGM, random sparse | 0.923 | 0.923 | **0.0000** | 0.33 |
| connectome, trained edges | 0.002 | 7.535 | 7.53 | 0.96 |
| dense MLP | 0.000 | 10.569 | 10.57 | 1.00 |
| correct answer, `b(v) = v²/2` | 0.000 | 10.604 | 10.60 | 1.00 |

**The FlyGM arm plays exactly the same thing no matter what value it holds.** The stored
policies are identical row for row, to the last digit. It is not bidding badly, it is
not bidding at all.

Why? The context does enter the network, but it does not arrive where the policy is read
from. Measuring the spread across the 50 values at every stage of one forward pass:

| graph | gain of `W` | spread across all 258 neurons | spread at the 5 readout neurons |
|---|---|---|---|
| real circuit | 0.008 | 0.92 | **0.000016** |
| rewire | 12.8 | 0.93 | 0.039 |
| random sparse | 12.9 | 0.83 | 0.042 |

* The input signal is strong inside the network (spread of about 0.9) and almost absent
  at the 5 neurons the policy is read from.
* For the first two of the six steps the spread at the readout is exactly 0, because
  those neurons are at least 3 connections away from the input.
* The gain diagnostic predicts this well: 0.008 goes with 1.6e-05, and 12.8 goes with
  0.039. **That is the diagnostic earning its place.**
* But the controls fail too, with 1,000 times more signal to work with. So a weak path is
  not the whole story. With `W` frozen and one update network shared by all neurons,
  nothing in this architecture can strengthen the input to readout path on its own, and
  the readout can only rescale what little it receives.

What this means for reading the numbers:

* The Game B rows above measure an arm that cannot condition on its input. They are not
  a statement about fly wiring, and they should not be compared with dense or with the
  trained-edge circuits as though they were.
* The wiring comparison inside Game B is still "no difference", but for a degenerate
  reason, so it carries no weight.
* Game A is unaffected. It has a single context, so there is nothing to condition on,
  and the arm performs like the circuit arms there.

**Sampled Game B behaves the same way.** 10 seeds per graph, 3,000 episodes:

| arm | final loss | bid L2 | seeds reaching the 0.05 threshold | bid spread across the 50 values |
|---|---|---|---|---|
| FlyGM, real circuit | 0.2805 [0.2672, 0.2941] | 2.14 | 0 of 10 | **0.0000** |
| FlyGM, rewire | 0.2761 [0.2514, 0.3056] | 2.23 | 0 of 10 | **0.0000** |
| FlyGM, random sparse | 0.2551 [0.2408, 0.2709] | 2.50 | 0 of 10 | **0.0000** |
| oracle (the grid floor) | 0.0058 | 0.0 | 10 of 10 | n/a |

* Every one of the 30 runs plays one policy for all 50 values. The collapse is not
  specific to exact gradients.
* No wiring contrast is distinguishable. The closest is real against random sparse,
  +0.0253 [0.0048, 0.0449], Holm p = 0.089.
* Every arm is about 45 times the grid floor, and no seed got near the threshold.

### FlyGM against the arms that train their edges

Same game, same seeds, each arm at its own tuned rate. Source:
`results/tables/cross_experiment.csv`.

| setting | comparison | difference in final loss | episodes |
|---|---|---|---|
| A exact | FlyGM vs connectome (same graph) | +0.00013, not distinguishable | **+36.3**, distinguishable |
| A exact | FlyGM rewire vs rewire | +0.00013, distinguishable | +18.8, distinguishable |
| A exact | FlyGM vs dense MLP | **-0.00019**, distinguishable | **-78.8**, distinguishable |
| A sampled | FlyGM vs connectome (same graph) | -0.0011, not distinguishable | +20, not distinguishable |
| A sampled | FlyGM vs dense MLP | +0.011, distinguishable | -5, not distinguishable |

* In Game A exact, freezing the synapses costs a little speed against trained edges
  (about 36 episodes) but not accuracy.
* The FlyGM arm still beats the dense MLP there, on both measures.
* In Game A sampled it is as good as the trained-edge connectome and worse than dense,
  which is the same pattern the circuit arms show (§7).

### Compute and caveats

* Tuning: 2 seeds per grid point, as everywhere else. The grid for sampled mode was
  extended down to 0.0001 for this work. Every FlyGM graph picked the **same rate in
  every setting** (0.03, 0.003, 0.01, 0.03), so this arm has none of the per-arm
  learning-rate confound described in §5, and its shared-rate robustness runs are
  no-ops that the driver skips.
* The rate for Game B sampled needs a longer note, below.
* Game B sampled was split across two machines because of its cost: 23 runs on a
  4-core cloud box and the last 7 on the laptop. The per-run wall times in its
  `summary.csv` are therefore a mix of two machines and are not one measurement. The
  laptop logged 8,200 s at 6 workers for its share.
* Measured wall times on the laptop (AMD Ryzen AI 7 350, 8 physical cores), from
  `results/flygm_ppo_log.txt`: tuning for Game A exact, Game A sampled and Game B exact
  took 1,518 s at 14 workers; the runs for those three settings took 8,483 s at 14
  workers; tuning for Game B sampled took 10,555 s at 6 workers. The last of these ran
  while the machine was also running a game, which cost about 1.9x, so it is an upper
  bound.
* Worker counts are not a free choice for this arm. One FlyGM Game B run takes 462 s
  alone, 420 s with 6 running at once, and 1,752 s with 14. **Fourteen workers deliver
  half the throughput of six**, because each step pushes 50 contexts by 8 agents by 258
  neurons through the shared update network and they crowd each other out of cache.

### A problem with how the Game B sampled rate was chosen

This one is worth spelling out, because the fix for one compute problem created
another.

To keep the cost down, `flygm_game_b_sampled.yaml` screens learning rates on
**1,000 steps** (`tuning.steps`) but the real runs are **3,000 steps**. On a short
horizon the rate that drops fastest wins. That is not always the rate that ends up in
the best place.

That is exactly what happened. Looking at every completed run's own curve:

| arm | runs | best value reached | final value | ends worse than its own best by | best value reached in the first half |
|---|---|---|---|---|---|
| FlyGM, real circuit | 10 | 0.2511 | 0.2806 | 12.1% | 10 of 10 |
| FlyGM, rewire | 10 | 0.2426 | 0.2761 | 14.1% | 6 of 10 |
| FlyGM, random sparse | 10 | 0.2457 | 0.2545 | 3.6% | 6 of 10 |

* Nothing is NaN or infinite, so this is not numerical blow-up. It is overshoot: the
  runs descend quickly, reach their best point early, then drift back up and stay there.
* Some are extreme. One seed reached its best value at **step 50 of 3,000** and got
  worse for the remaining 2,950.
* So the number reported for this arm is not its best under this setting. Reported is
  about 0.28; the best reached during training is about 0.24.

Two further points, both against our own earlier reading:

* The grid for this setting stopped at 0.03 and final loss was still falling there, so
  we extended the screen to 0.1 and 0.3 afterwards. 0.1 came out best, beating 0.03 by
  3.5% (`results/tuning/flygm_game_b_sampled_grid_extension.csv`). But that extension
  used the **same 1,000-step screen**, so it carries the same bias. It shows the grid
  was truncated. It does not show which rate is best over 3,000 steps.
* The honest summary is therefore: the rate used here was picked on a shorter horizon
  than the runs it was used for, and the runs overshoot as a result.

What it does not change:

* The arm plays the same policy for every value, at every rate we have tried, between
  0.001 and 0.3. A step size cannot fix an arm that does not read its input.
* All three graphs share the rate, so the comparison between them is unaffected.
* Game A is unaffected: its rates were screened on the full run length, and its runs do
  not overshoot.

This was found by the cloud session running the cell, from the shape of the curves, not
by the tuner.

---

## 10. PPO instead of REINFORCE

### The idea, in plain terms

Sampled self-play so far used REINFORCE: play 128 matches, take one gradient step, throw
the matches away. PPO reuses them:

* The same batch of matches is used for **4 updates instead of 1**.
* Each update is clipped, with epsilon 0.2, so one batch cannot move the policy too far.
* The critic is the running-mean baseline the project already had. There is no entropy
  bonus.
* The win/loss weighting still multiplies the update, so asymmetric learning still works.
* Everything else is shared: same rollout, same optimizer, same cosine decay, one
  scheduler step per round rather than per update.

It is available to every arm, including FlyGM, by setting
`training.sampled_algorithm: ppo`.

### PPO is much faster to the threshold, and that is the clearest effect

Episodes to reach the convergence threshold, 10 seeds, censored at the training length.

| arm | Game A, REINFORCE | Game A, PPO | Game B, REINFORCE | Game B, PPO |
|---|---|---|---|---|
| dense MLP | 720 | 460 | 3,000 | **1,860** |
| connectome | 695 | **175** | 3,000 | 3,000 |
| connectome, signed | 680 | **160** | 3,000 | 2,675 |
| rewire | 1,030 | 450 | 3,000 | 3,000 |
| random sparse | 690 | 450 | 3,000 | 3,000 |
| random sparse, signed | 710 | **155** | 3,000 | 2,350 |
| FlyGM, real circuit | 715 | 460 | n/a | n/a |
| tabular | 425 | 975 | 2,890 | 3,000 |

* Every network arm reaches the threshold sooner under PPO in Game A, and the circuit
  arms reach it about 4 times sooner.
* In Game B, where almost nothing reached the threshold before, dense now does so at
  1,860 episodes and two signed circuits get there as well.
* The tabular baseline is the exception. It is slower under PPO in both games.

### On final accuracy, PPO depends on the game

Final utility loss, 10 seeds. A negative change means PPO ended closer to equilibrium.
Bold rows are distinguishable (CI excludes 0 and Holm p < 0.05).

| game | arm | REINFORCE | PPO | change |
|---|---|---|---|---|
| A | tabular | 0.0655 | 0.0351 | **-0.0304** |
| A | dense MLP | 0.0333 | 0.0338 | +0.0004 |
| A | connectome | 0.0454 | 0.0636 | +0.0182 |
| A | connectome, signed | 0.0451 | 0.0596 | **+0.0145** |
| A | rewire | 0.0568 | 0.0503 | -0.0064 |
| A | FlyGM, rewire | 0.0396 | 0.0544 | **+0.0148** |
| B | tabular | 0.0529 | 0.0811 | **+0.0282** |
| B | dense MLP | 0.0716 | 0.0491 | **-0.0225** |
| B | connectome, signed | 0.0942 | 0.0564 | **-0.0379** |
| B | random sparse, signed | 0.0790 | 0.0510 | **-0.0280** |
| B | connectome | 0.1536 | 0.1364 | -0.0172 |

* In **Game A**, PPO helps the tabular baseline a lot and hurts the connectome arms.
* In **Game B**, it is the other way round. PPO helps dense and the signed circuits and
  hurts tabular.
* A simple reading: PPO moves the policy faster per round. Where the limit is how fast
  you can move (Game B, where nothing had converged), that helps. Where the limit is
  sampling noise at a mixed equilibrium (Game A, see docs/theory.md §5), moving faster
  just settles sooner in a worse place.
* Late-training policy variance rises under PPO almost everywhere, which fits that
  reading: it is noisier at the end.

### Does any wiring difference show up under PPO?

This needed care, because §5 of this report documents a trap: if two arms tune to
different learning rates, a difference between them may be the rate rather than the
wiring. Two contrasts looked distinguishable at first, so both PPO grids were re-run
with one shared rate per family.

| contrast | at each arm's own rate | at one shared rate | verdict |
|---|---|---|---|
| A sampled, connectome signed minus rewire signed | +0.0159, Holm p = **0.032** | +0.0139, Holm p = 0.170 | does not survive |
| B sampled, connectome signed minus rewire signed | -0.0294, Holm p = **0.035** | +0.0049, Holm p = 0.690 | **gone, and the sign flips** |
| B sampled, connectome minus random sparse | +0.0417, Holm p = 0.015 | **+0.0417, Holm p = 0.008** | **survives** |

* The first two were the §5 trap again. In Game B the two signed arms had tuned to 0.003
  and 0.03, and once both run at 0.01 the difference disappears and changes direction.
* The third is different. The connectome, rewire and random sparse arms had **all**
  tuned to 0.01 in Game B sampled, so they were already sharing a rate and those runs
  are identical in both grids. There is no rate confound to remove.
* So under PPO in Game B sampled, the real circuit is worse than random wiring by
  **+0.0417 [0.021, 0.061]**, effect size 1.68, Holm p = 0.008. Against the
  degree-preserving rewire it is +0.0249 with p = 0.198, which is not distinguishable.
* Read it carefully. This is "wiring and degrees together", not "the specific wiring",
  because the rewire control is not distinguishable. It is one cell of the grid with 10
  seeds. It is the first contrast anywhere in this project to survive the shared-rate
  check, and it points the same way as §4: where the real wiring does anything at all in
  Game B, it makes things slightly worse.

### Compute and caveats

* 10 seeds per arm, 3,000 episodes, symmetric updates. Learning rates were tuned for PPO
  separately, with the sampled grid extended down to 0.0001.
* The shared-rate re-runs (`configs/robust_ppo_game_*_shared_lr.yaml`) were added because
  of what the data showed, not because the plan called for them.
* The FlyGM arms in Game B sampled under PPO are a separate experiment at **3 seeds**,
  not 10, and they reuse the rate tuned for the same arms under REINFORCE rather than
  getting their own. At about 4 hours of CPU per run it was the most expensive cell in
  the project. Given §9, it measures an arm that cannot condition on its input, so it is
  reported for completeness rather than as evidence about wiring. Those numbers are added
  when the cloud job finishes.
* Wall times for `tune-ppo-a` (6,906 s) and `run-ppo-a` (2,841 s) were measured while the
  machine was also running a game. They are upper bounds and not comparable with the
  other steps. The rest, all at 6 workers on the laptop: `tune-ppo-b` 3,135 s,
  `run-ppo-b` 7,366 s, `robust-ppo-a` 3,862 s, `robust-ppo-b` 5,777 s.

---

## 11. Does the Game A speed-up survive full time resolution?

§4 reported the unsigned connectome crossing the 0.01 threshold 19–24 episodes before
its controls, and §8 listed as not established whether that is real beyond the
25-episode evaluation interval. `configs/game_a_exact_eval1.yaml` reruns the whole
Game A exact grid (the same 9 arms, the same 20 evaluation seeds, the same tuned
learning rates in `configs/tuned_lr.yaml` under `game_a_exact`) with `eval_every: 1`.

**Compute.** 180 runs, **306 s wall** on 8 workers (CPU only), run alongside the FlyGM
tuning. Evaluating after every episode makes the per-run records 3,001 points instead
of 121, so `results/game_a_exact_eval1/runs/` and its `curves.csv` are 168 MB and 68 MB
and are *not* committed (`.gitignore` says so, with the command to rebuild them); the
committed `summary.csv` holds every crossing time this section uses.

**The comparison is exactly paired.** Evaluation reads the policy under `no_grad` and
draws no random numbers, so for a given seed the two runs are the *same* training
trajectory, observed 3,001 times instead of 121. Verified per seed: every coarse
crossing time is the exact one rounded up to the next multiple of 25 (gap in [0, 25)
for all 180 runs), and final exploitability agrees. Mean episodes to 0.01, 20 seeds:

| arm | `eval_every: 1` | `eval_every: 25` | rounding gap |
|---|---|---|---|
| connectome | **216.6** | 226.2 | 9.7 |
| connectome, signed | 239.5 | 251.2 | 11.8 |
| rewire | 231.6 | 245.0 | 13.4 |
| rewire, signed | 227.3 | 236.2 | 8.9 |
| random sparse | 236.7 | 250.0 | 13.3 |
| random sparse, signed | 228.9 | 241.2 | 12.3 |
| dense MLP | 327.4 | 341.2 | 13.9 |
| tabular | 629.0 | 650.0 | 21.0 |

**The speed-up is real, and about a fifth smaller than the coarse grid suggested.**
Differences in episodes to 0.01, 95% CI [Holm p]; source `results/tables/pairwise.csv`
(experiment `game_a_exact_eval1`).

| contrast | `eval_every: 1` | `eval_every: 25` (§4) |
|---|---|---|
| connectome − rewire | **−15.1 [−18.4, −11.7]**, g = −2.7 [0.001] | −18.8 [−23.8, −13.8] [0.001] |
| connectome − random sparse | **−20.1 [−25.1, −15.5]**, g = −2.5 [0.001] | −23.8 [−27.5, −20.0] [0.001] |
| connectome − dense | −110.8 [−114.6, −106.9] [0.001] | −115.0 [−120.0, −108.8] [0.001] |
| connectome, signed − rewire, signed | +12.2 [3.0, 21.9] [0.088], not distinguishable | +15.0 [5.0, 25.0] [0.093] |
| connectome, signed − random, signed | +10.6 [1.8, 19.6] [0.115], not distinguishable | +10.0 [0.0, 20.0] [0.321] |
| connectome, signed − connectome | +23.0 [15.0, 31.9] [0.001] | +25.0 [16.3, 33.8] [0.001] |

The coarse grid inflated the gap because the rounding is not the same for every arm:
the connectome's mean rounding gap is 9.7 episodes against 13.4 (rewire) and 13.3
(random), so roughly 4 of the reported 19–24 episodes were an artefact of where the
evaluation grid happened to fall. What remains, 15 and 20 episodes or 6 to 8 percent,
is distinguishable with a tight CI and a large effect size.

**Caveats.** Three things this does not change.

* It is a *speed* result only. The trajectories are identical, so final exploitability
  is identical too, and §3–§4's conclusion that no final-loss contrast is
  distinguishable is untouched.
* The sign-constrained version still shows nothing, and with the opposite sign. The
  effect does not survive Dale's law, which is the §4 reading unchanged.
* 15 episodes out of 217 is small in a setting where the dense MLP needs 327 and the
  tabular baseline 629, and Game A has a single context, so any policy there is a
  reparameterisation of one distribution (§8).
