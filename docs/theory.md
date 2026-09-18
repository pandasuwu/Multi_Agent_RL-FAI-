# Theory

Notation: prize `V > 0`, cost rate `c > 0`. A fight ends when the first player
quits; **both** players pay `c·min(t_i, t_j)`, and the later quitter also gets the prize.
Code: `src/games/`. Every numbered claim below has a test in `tests/test_games.py`.

---

## 1. Game A: complete information, symmetric

### 1.1 Payoffs

```
u_i(t_i, t_j) = V − c·t_j     if t_i > t_j
              = −c·t_i        if t_i < t_j
              = V/2 − c·t     if t_i = t_j = t
```

### 1.2 Symmetric mixed equilibrium

Suppose the opponent's quitting time has density `g` and survival `G(t) = P(T > t)`.
Quitting at `t` earns

```
π(t) = ∫₀ᵗ (V − c s) g(s) ds  −  c t G(t).
```

The first term covers the opponent quitting first: I win `V` and pay the cost up to their
quitting time. The second covers me quitting first: I pay up to `t`. (Ties have
probability zero for a continuous `G`.) In a mixed equilibrium with full support on
`[0, ∞)` the player must be indifferent, so `π'(t) = 0`:

```
π'(t) = (V − c t) g(t) − c G(t) − c t G'(t)
      = (V − c t) g(t) − c G(t) + c t g(t)        (G' = −g)
      = V g(t) − c G(t) = 0.
```

So the hazard is constant, `g(t)/G(t) = c/V`, and

```
G(t) = P(T > t) = exp(−c t / V),    i.e.  T ~ Exponential(rate = c/V),   E[T] = V/c.
```

**Equilibrium payoff.** Every `t` in the support earns the same payoff, and `π(0) = 0`
(quit at once, lose, pay nothing), so the expected equilibrium payoff is exactly **0**.
The prize is fully dissipated by the expected cost. This is the classical result
(Maynard Smith 1974; Bishop & Cannings 1978). The symmetric equilibrium is unique and
is an ESS.

**Asymmetric equilibria exist.** "Player 1 quits at 0, player 2 never quits" is also a
Nash equilibrium, and so is a continuum of asymmetric mixed profiles. By Selten's (1980)
theorem, a role-asymmetric game has no mixed ESS, so two *independently parameterised*
learners locked in a fixed pair can drift to one of these dominance equilibria. Our
protocols avoid that confound in two ways (see `README.md`, *Protocol*). Exact mode
uses a single shared policy in self-play, the convention of Bichler et al. (2021) for
symmetric games. Sampled mode uses a randomly re-matched population, the setting in
which the exponential is the ESS.

### 1.3 The game on a grid

Quitting times are restricted to `t_k = kΔ`, `k = 0..n−1`, `Δ = T_max/(n−1)`
(`configs/base.yaml`: `T_max = 8`, `n = 64`). The payoff matrix is
`U = V·W + C`, where `W[i,j] = 1, ½, 0` for `i >, =, < j` and `C[i,j] = −c·min(t_i, t_j)`.
Against a mixed strategy `p`, the expected payoff of every pure time is the vector `U p`.

**The grid game is strictly stable.** The symmetric part of `U` is
`V/2·𝟙𝟙ᵀ + C`. The constant part vanishes on the tangent space
`{z : Σz = 0}`, and `−min(t_i,t_j)` is a negative-definite kernel (min is the covariance
of Brownian motion). So `U` is negative definite on the tangent space: a *stable game*
in the sense of Hofbauer & Sandholm (2009). Consequences:

1. The symmetric equilibrium is unique.
2. It globally attracts replicator, best-response and projection dynamics, and
   gradient play in self-play. A learner that fails to converge therefore signals
   a bug or a step-size problem, not a hard game.

This is checked numerically: the largest nonzero tangent eigenvalue is about −0.032 at
the default grid.

**Exact equilibrium on the grid.** Indifference between adjacent times `t_k` and `t_{k+1}`
(against `p_k` the switch turns a tie into a win, against `p_{k+1}` it turns a loss into a
tie, and against later quitters it costs `cΔ` more) gives

```
(V/2)(p_k + p_{k+1}) = cΔ · P(T ≥ t_{k+1}).
```

A geometric law `p_k = (1−r) r^k` solves this with

```
r = √(1 + (cΔ/V)²) − cΔ/V  =  e^{−cΔ/V} + (cΔ/V)³/6 + O(Δ⁴).
```

So the discrete equilibrium is, to O(Δ³) per step, the exponential **floored onto the
grid**: grid time `t_k` gets `P(t_k ≤ T < t_{k+1})`, and `t_max` gets the tail. The
support solver (`solve_symmetric_equilibrium`) confirms this. The ratio
`p_1/p_0 = 0.88104` equals `r`, and apart from the truncated tail the solver and the
floored exponential agree to total variation `1.3·10⁻³`. The grid equilibrium payoff is
`π(0) = (V/2)·p_0 = (V/2)(1−r) ≈ cΔ/2 = 0.059`. That is not 0, because on a grid a tie
at `t = 0` has positive probability; it vanishes as `Δ → 0`.

**The oracle** (arm 0) plays the floored exponential. Its exact exploitability at the
default grid is `1.5·10⁻³`. This is the *discretisation floor*. No learner can be
distinguished from equilibrium below it, and the convergence threshold is set well above
it. Nearest-point binning, the obvious alternative, has a floor of `2.9·10⁻²`, about 20×
worse, because it gives `t = 0` half a bin.

---

## 2. Game B: incomplete information, private values

Each player privately draws `v ~ F` with density `f`. A strategy is a bid function
`b(v)`, meaning "quit at time `b(v)`". Look for a symmetric, strictly increasing
equilibrium `b`.

### 2.1 Derivation

Suppose the opponent uses `b`, and a player of type `v` quits at `b(z)`, mimicking type
`z`. They win exactly when the opponent's type `x < z`. Then they pay `c·b(x)` and
receive `v`. Otherwise they lose and pay `c·b(z)`. The interim expected payoff is

```
Π(z, v) = F(z)·v − ∫₀ᶻ c·b(x) f(x) dx − (1 − F(z))·c·b(z).
```

Differentiate in `z`:

```
∂Π/∂z = f(z)·v − c·b(z) f(z) + c·b(z) f(z) − (1 − F(z))·c·b'(z)
      = f(z)·v − (1 − F(z))·c·b'(z).
```

Truthful mimicry `z = v` must satisfy the first-order condition, so

```
c·b'(v) = v · f(v) / (1 − F(v))          ⟹     b(v) = (1/c) ∫₀ᵛ x · f(x)/(1−F(x)) dx,
```

using `b(0) = 0`, since a type with zero value never pays to fight. For `v ~ Exp(1)` the
hazard `f/(1−F) ≡ 1`, and with `c = 1`

```
b(v) = v² / 2.
```

**Global optimality, not just the FOC.** Substitute `b'(z) = z` back in:
`∂Π/∂z = f(z)(v − z)`. This is positive for `z < v` and negative for `z > v`, so `z = v`
is the global maximiser of `Π(·, v)`. `b` is a best response to itself.

**Numerical check** (`test_game_b_bid_is_best_response_to_itself_continuous`). With no
grid, `Π(z, v)` is computed by adaptive quadrature for `v ∈ {0.25, 0.7, 1, 1.9, 3.3}`. The
argmax over a 601-point `z` grid lands within 0.01 of `v`, and the central-difference
derivative at `z = v` is below `10⁻⁶`.

### 2.2 The game on a grid

Types are discretised to `K = 50` equally likely values at the quantile midpoints
`v_k = F⁻¹((k+½)/K)`. The largest is `ln 100 ≈ 4.61`, whose equilibrium bid is `10.6`. So
Game B uses `T_max = 12` (`configs/game_b_*.yaml`) and no type's bid is clipped. A
strategy is a `K × n` row-stochastic matrix `P`. The opponent's quitting-time marginal
is `q = Σ_k w_k P_k`, and the interim payoff matrix is
`U_B[k, i] = v_k · (W q)_i + (C q)_i`. The oracle places each `b(v_k)` on its two
neighbouring grid points, weighted so that the mean is exact. Its utility loss is
`5.8·10⁻³`, which is 1.2% of the equilibrium payoff `u* = 0.488`.

---

## 3. Asymmetric win/loss updates: what the learner converges to

Sampled self-play uses REINFORCE with a baseline `b` that tracks the mean reward. Real
flies take losses harder than wins, and we model this by scaling the per-sample policy
gradient by `α_loss` after a loss and `α_win` after a win, with ratio `ρ = α_loss/α_win ≥ 1`.
The weights are normalised to mean 1, `α_win = 2/(1+ρ)` and `α_loss = 2ρ/(1+ρ)`, so
changing `ρ` does not change the overall step size.

### 3.1 The rest point is not the equilibrium

The expected update of a softmax policy vanishes when every time in the support earns
the same *weighted advantage* `E[α(outcome)(u − b)]`. That is the symmetric equilibrium of
a modified game with payoff `α(outcome)·(u − b)`. In continuous time, the payoff of
quitting at `t` becomes

```
π̃(t) = ∫₀ᵗ α_w (V − c s − b) g(s) ds − α_l (c t + b) G(t).
```

Setting `π̃'(t) = 0` gives `g(t)[α_w(V − ct − b) + α_l(ct + b)] = α_l c G(t)`, i.e. a
**decreasing hazard**

```
h(t) = ρ c / (V + (ρ − 1)(c t + b)).
```

The survival function is Lomax (Pareto II),
`G(t) = (1 + (ρ−1)c t / (V + (ρ−1)b))^{−ρ/(ρ−1)}`, not exponential. The baseline must
equal the mean true payoff under this law, `b = V/2 − c·E[min(T₁,T₂)]`, and
`E[min] = ∫G² = (V + (ρ−1)b)/(c(ρ+1))`. Solving gives

```
b = V (ρ − 1) / (4ρ).
```

`ρ = 1` gives `b = 0` and the exponential back. Checked by quadrature in
`test_asymmetric_continuous_baseline_self_consistent`.

**Prediction.** Loss-weighted learning does not just change convergence *speed*. It moves
the rest point to a strictly exploitable profile: more immediate quits, a heavier tail,
and a positive mean payoff. On the default grid, `asymmetric_fixed_point` gives:

| ρ | baseline b | exploitability of the rest point | TV to equilibrium | P(quit at 0) |
|---|-----------:|----:|----:|----:|
| 1 | 0.060 | 0.000 | 0.001 | 0.119 |
| 2 | 0.182 | 0.067 | 0.161 | 0.191 |
| 4 | 0.241 | 0.108 | 0.246 | 0.249 |
| 8 | 0.270 | 0.130 | 0.295 | 0.285 |

These are the floors the asymmetric arms should approach, if they converge at all. The
experiment tests whether they do, and whether wiring changes how closely they do.

### 3.2 Ties are coin flips

On a grid, ties have positive probability, so the update needs an outcome label for
them. We resolve a tie by a fair coin for the prize. The realised payoffs are
`V − ct` / `−ct`, so the expected tie payoff is exactly the game's `V/2 − ct`, and every
round is a clean win or loss. Giving ties their own weight (for example the mean
weight 1) instead adds `(ρ−1)/(ρ+1)·V/2` to the diagonal of the modified game. That makes
the modified game *unstable* on the grid (positive tangent eigenvalues). The mean-field
dynamics then break into atoms that chase each other and never settle, a grid artefact
rather than a property of the learning rule. With coin-flip ties, the symmetric part of
the modified game is `α_w V/2·𝟙𝟙ᵀ − c·min − b`, negative definite on the tangent space
for every `ρ`, so the rest point is stable.

---

## 4. Metrics (definitions; code in `src/metrics/exploitability.py`)

Conventions follow Bichler et al., *Nature Machine Intelligence* 3, 687–695 (2021).

* **Exploitability / utility loss** `ℓ(π; π_opp) = max_t E[u(t, π_opp)] − E[u(π, π_opp)]`.
  This is their eq. (2), ex ante. It is exact on the grid: the max is over the 64 grid
  times (per type in Game B), and the expectations are finite sums. **No sampling.** It
  is reported in absolute payoff units, because Game A's equilibrium payoff is 0 and
  their relative form `1 − u/(u+ℓ)` would divide by roughly 0.
  * Exact mode, one shared policy: `π_opp = π`.
  * Sampled mode, population of `M` agents: each agent's opponent is the leave-one-out
    mixture of the others, and we average over agents. We also report the
    exploitability of the population mixture `π̄`, which is what a random opponent faces
    and what the fly survival curve measures.
* **Relative utility loss vs the BNE** (Game B), `L = 1 − u(β, β*)/u(β*, β*)`: their eq. (10).
* **L2** (Game B): probability-weighted RMSE between the learned mean quitting time per
  type and `b(v)`. Their eq. (11). We also report MSE per value decile.
* **TV and W1** between the learned quitting-time law and the floored exponential. W1 on
  the grid is `Σ_k |F_p(t_k) − F_q(t_k)| Δ`, which is exact.
* **Convergence speed**: the first evaluated episode at which exploitability falls below
  the configured threshold. Runs that never reach it are reported as censored.
* **Late-training stability**: total variance of the policy across the snapshots taken in
  the final 10% of episodes.
