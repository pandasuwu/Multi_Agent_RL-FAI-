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
  * whether the Game A exact speed-up (unsigned) is real beyond its 25-episode
    evaluation resolution.
