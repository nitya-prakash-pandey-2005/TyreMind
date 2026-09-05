# Statistical architecture

## The model

For driver *d* on session lap *t*, running tyre set *r*:

```
y[d,t] = alpha[r]           run intercept: car pace, setup, starting fuel mass
       - A * g(t)           track evolution, shared across the whole field
       + theta[t]           residual track wobble (tightly bounded)
       + s[d,t]             latent tyre performance loss   <- the deliverable
       - phi_kappa * L[d,t] fuel burn-off, L = laps completed this run
       + gamma * TI[d,t]    traffic
       + eps                observation noise, heavy-tailed
```

with `g(t) = 1 - exp(-k*t)`, and the latent tyre evolving as a local linear trend
in **tyre age**, not in session lap:

```
s[d, a+da]    = s[d,a] + da * rate[d,a] + noise
rate[d, a+da] = rate[d,a] + noise
```

`rate` is the reported quantity: instantaneous degradation in seconds per lap.

**A cliff needs no special machinery.** Because the rate is itself a free state,
degradation that accelerates simply appears as `rate` rising. The same structure
represents a plateau or a recovery. Nothing assumes a cliff exists — which
matters, because not every tyre has one.

## Why a Kalman filter and not MCMC

Conditional on the variance hyperparameters the model is linear-Gaussian, so the
marginal likelihood is available in closed form through the prediction error
decomposition. That buys three things a sampler does not:

1. **An exact likelihood number**, usable directly for hyperparameter estimation
   (L-BFGS-B over six log-variances) and for model comparison.
2. **Analytic posteriors** for every state, including the coefficients. No
   sampling error, no convergence diagnostics.
3. **An incremental mode.** The forward pass alone *is* the real-time estimator.
   A sampler has no such mode, and re-running one every lap is not an option on a
   pit wall. Measured: 0.22 ms per lap, flat in session length.

## Coefficients as states

`phi_kappa`, `gamma`, the track amplitude and the per-compound baseline rates are
represented as states with **zero process noise** rather than as free parameters
in the optimiser.

For a constant, "state with no process noise" *is* the Bayesian posterior. The
consequence that matters: physical priors enter as `P0`, and the filter
propagates their uncertainty into the tyre posterior automatically. The width of
the fuel prior widens every degradation interval, with no bootstrap needed.

## The hierarchy

At each tyre change, a stint's degradation rate is drawn around its compound's
pooled baseline:

```
rate[new stint] ~ N(compound_baseline[c], stint_rate_sd^2)
```

`stint_rate_sd` is estimated, so the data chooses how far individual stints
shrink towards their compound mean. This is what pools evidence across the whole
grid, and it is the direct answer to Cappello and Hoegh finding compounds
statistically indistinguishable from a single driver.

## Identifiability

See `01_RESEARCH_AUDIT.md` §3 for the derivation. In short:

| Collinearity | Resolved by | Evidence or assumption? |
|---|---|---|
| Fuel vs degradation, within a run | Physical prior, 0.030 s/kg × 2.7 kg/lap | **Assumption** |
| Track evolution vs uniform rate shift | Saturating basis + informative amplitude prior | **Assumption** |
| Tyre age vs session lap | Fitting the whole field; run stagger | **Evidence** |

`exp02_prior_sensitivity` quantifies what the assumptions cost if wrong: a
full-standard-deviation error in the fuel prior moves the recovered rate by about
0.02 s/lap, which remains 4.9× better than the naive method at 0.0966.

## Numerical implementation

- **Sequential scalar updates** within a time step (Durbin and Koopman 2012,
  section 6.4). No matrix inverse is ever formed; cost falls from O(n³) to O(n²)
  per observation. It is also what makes true online operation possible.
- **A batched per-step path** for fitting — mathematically identical, pinned by
  test to 1e-9, but issuing far fewer NumPy calls. At several hundred likelihood
  evaluations per fit, interpreter dispatch dominates the arithmetic.
- **A structured transition.** The transition matrix is identity everywhere
  except two rows per car, so propagation is done by row operations in O(n²)
  instead of a dense O(n³) product. Roughly a hundredfold difference at a full grid.
- **Forced symmetry** after each rank-1 downdate. Left alone, mantissa-level
  asymmetry accumulates over a few thousand updates until an eigenvalue goes
  negative and the log-likelihood silently becomes NaN.

Together these took a session fit from 25 s to 5.8 s with identical results.

## Verification

The recursive likelihood is checked against an **independently constructed joint
multivariate normal** over all observations, agreeing to 1e-9.

That test exists because a Kalman recursion that is subtly wrong does not crash.
It returns a plausible number and quietly biases every hyperparameter downstream,
so a second, structurally different route to the same quantity was necessary
rather than nice to have.

## Model comparison

Five rungs, scored on identical expanding-window chronological folds. A random
split would let a model see lap 40 while predicting lap 20 of the same stint, and
every metric would improve for a reason that does not exist on a Sunday.

Two scores are reported because they measure different things and they disagree:
**lap-time prediction** (which any model can do, and which fuel dominates) and
**degradation recovery** (which is the product's purpose, and which can only be
scored where the truth is known).

The disagreement is the finding. LightGBM wins the first and cannot compete in
the second, because it has no parameter that means "degradation rate".
