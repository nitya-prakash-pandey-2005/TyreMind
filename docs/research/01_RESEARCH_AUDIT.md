# Research audit and feasibility assessment

*Written before implementation, revised after it. Where a prediction made here
turned out wrong, the entry says so rather than being quietly corrected.*

---

## 1. What already exists

The brief's framing — "AI predicts tyre degradation" — describes work that is
published, peer-reviewed and in some cases running live at Grands Prix. Building
that again would have produced a competent reimplementation of someone else's
result.

| Work | What it does | What it leaves open |
|---|---|---|
| **Cappello & Hoegh, [arXiv:2512.00640](https://arxiv.org/abs/2512.00640)** (Nov 2025) — *A State-Space Approach to Modeling Tire Degradation in Formula 1* | Bayesian state-space model. Lap time as a function of fuel mass and latent tyre pace, pit stops as state resets, compound-specific rates, skewed-t observation model. FastF1 data. | **Single driver, single race.** No traffic, no track evolution, no telemetry, no physics. Reports compounds as statistically indistinguishable — a statistical power problem that pooling across the field would address. |
| **Pitwall, [arXiv:2607.06495](https://arxiv.org/abs/2607.06495)** (2026) | Calibrated Monte Carlo race simulator (2,000 sims, fitted on 191k green laps) plus an LLM briefing layer gated on verified claims. Ran live at the 2026 Austrian and British GPs. | Own stated gaps: **no car telemetry**, fuel mass approximate, tyre thermals absorbed into noise. Predicts race *outcomes*; does not attribute observed pace to causes. |
| **[arXiv:2501.04067](https://arxiv.org/abs/2501.04067)** — *Explainable Time Series Prediction of Tyre Energy in F1* | XGBoost and deep models predicting tyre energy, with SHAP and counterfactual explanations. | Uses **private Mercedes-AMG telemetry**. Not reproducible from public data. |
| **MegaRide / [Sci.Direct S0043164824000565](https://www.sciencedirect.com/science/article/pii/S0043164824000565)** | Tyre wear model fusing rubber viscoelasticity, road roughness and thermodynamics. | Commercial; requires rig and FE data that public sources do not contain. |
| **Heilmeier et al. / [TUMFTM race-simulation](https://github.com/TUMFTM/race-simulation)** | Open-source race simulator and pit-stop optimisation. The reference implementation. | Takes a degradation rate as **given**. |
| **Rubber wear review, [PMC12915245](https://pmc.ncbi.nlm.nih.gov/articles/PMC12915245/)** | Survey of tyre wear modelling; Archard as baseline, energy-based formulations as the better-supported family. | Confirms that a constant Archard coefficient is inadequate for rubber, and that temperature belongs in the coefficient. |

**Conclusion.** Degradation prediction and race simulation are solved-enough
problems. Rebuilding either was ruled out.

---

## 2. The gap

Every model in the table above is validated on **lap-time prediction error**.

That metric cannot detect the failure that matters here. Fuel burn-off, track
evolution and tyre degradation all move lap time smoothly with lap number, so
**many wrong decompositions sum to the same right total.** A model can predict
lap times almost perfectly while attributing the change to entirely the wrong
cause, and no lap-time metric will notice.

The brief asks to *isolate true tyre wear from confounding variables*. That is an
attribution problem, and attribution is untested in this literature.

**So TyreMind's contribution is to test it.** Generate sessions where the true
degradation rate is known, bury it under realistic confounding, and measure
recovery — separately for each cause, not just in aggregate.

---

## 3. Identifiability: the actual hard part

Working through the algebra before building revealed three collinearities. Only
the first was anticipated by the brief.

### 3.1 Fuel against degradation, within a run

Both are linear in laps completed. Fuel makes the car faster at roughly
0.030 s/kg × 2.7 kg/lap ≈ **0.081 s/lap**; the tyre makes it slower by an unknown
amount. Within one run these are not separable at all.

The unknown *starting* fuel mass is irrelevant — it shifts the run's intercept,
which a run-level effect absorbs. Only the slope matters, and the slope is
confounded.

**Resolution:** pin the fuel coefficient with an informative physical prior and
carry its uncertainty as state, so it widens every published interval.

**Verified empirically.** On 25 synthetic sessions the naive estimator's bias is
**−0.0966 s/lap** — almost exactly the fuel slope, in the direction theory
predicts. On real 2024 races it reports *negative* degradation.

### 3.2 Track evolution against a uniform shift in degradation

**Not anticipated.** Found by diagnosing a stubborn −0.013 s/lap bias that
survived removing the cliff, scrubbed sets and traffic from the generator.

Shift every degradation rate by *c* and the track-evolution slope by *−c*. For a
lap at session lap *L* with tyre age *a = a₀ + (L − L₀)*:

```
degradation moves by  +c·(a₀ + L − L₀)
track       moves by  −c·L
net                    c·(a₀ − L₀)      ← constant WITHIN a run
```

A constant within a run is exactly what the run intercept absorbs. The two are
structurally indistinguishable, and scrubbed sets do not help — the difference is
still constant within the run.

**Resolution:** model track evolution parametrically as a saturating curve
`A·(1 − exp(−kL))` with an informative prior on the amplitude, rather than as a
free random walk. Rubber deposition genuinely saturates, so this is the more
physically correct model as well as the identifying one. Bias fell from −0.013 to
**+0.0012 s/lap**.

### 3.3 Tyre age against session lap, within a run

They advance together. A single car cannot distinguish an ageing tyre from a
changing session.

**Resolution — and the only one that is free.** Fit the whole field at once. Cars
change tyres on different laps, so at any session lap the grid spans a wide range
of tyre ages. Run stagger is a natural experiment requiring no prior, only the
whole grid instead of one car. This is the structural difference from Cappello &
Hoegh, who model one driver.

**Two of three are resolved by assumption, not by data.** No amount of
single-session data will resolve them. `exp02_prior_sensitivity` reports how far
the answer moves when those priors move.

---

## 4. Feasibility, as assessed and as it turned out

| Component | Assessed | Outcome |
|---|---|---|
| FastF1 ingestion + quality engine | GREEN | Built. 454 raw laps → 143 usable at Monza FP2, every exclusion counted. |
| Hierarchical state-space estimator | YELLOW | Built. Hand-written Kalman + RTS smoother; 5.8s per session fit. |
| Ground-truth synthetic benchmark | GREEN | Built. The headline result. |
| Practice → race validation | YELLOW | Built. 5 events, 10 comparisons. |
| Model ladder with time-aware CV | GREEN | Built. Five rungs, chronological folds. |
| Real-time online estimation | GREEN | Built. 0.22 ms/lap mean. |
| Curvature → per-corner energy | YELLOW | Built and independently validated (§5). |
| Reduced-order thermal model | YELLOW | Built. Estimated states, never claimed as temperatures. |
| Energy-based wear + energy clock | YELLOW | Built. **Hypothesis failed** — see §6. |
| Monte Carlo strategy + regret | GREEN | Built. 10,000 races × 5 strategies in 0.04s. |
| Cross-domain transfer | YELLOW | Built on NASA C-MAPSS with real ground truth. RUL RMSE 26.5 cycles over 40 held-out engines. |
| **Full Pacejka identification** | **RED** | **Cut.** Slip angle and slip ratio are not observable from public telemetry. Identifying a production Pacejka set would have consumed the build and produced parameters no data could constrain. |
| **PyMC / Stan** | **RED** | **Cut.** Windows compile risk, and no incremental mode — which would have made the real-time claim impossible. |
| **Neural state-space / TFT** | **RED** | **Cut** as originally specified. A tuned multi-layer perceptron was added to the ladder instead, so the "have you tried deep learning" question is answered by measurement rather than opinion — see §7. A torch model would add a 2 GB dependency to an offline demo and change the answer only by overfitting harder: on a few hundred laps with six features, data is the binding constraint, not depth. |

---

## 5. Independent validation of the physics layer

The physics chain — GPS trace → curvature → lateral acceleration → load transfer
→ per-corner energy — needed a check that was not circular.

**Test:** a clockwise circuit is mostly right-hand corners, and cornering right
throws load onto the *left* tyres. So a clockwise circuit must show a left-side
energy share above 50%. Circuit rotation direction is published fact and is never
given to any code under test.

**Result: 7 of 8 circuits recovered correctly.** Clockwise circuits show 21–35%
left-turn energy; correctly-classified anti-clockwise circuits show 60–75%.
Austin misses at 46.2%, which is genuinely borderline — COTA's esses alternate.

This required curvature sign, load transfer and energy integration to all be
right simultaneously.

---

## 6. A hypothesis that failed

**Predicted:** replacing "tyre age in laps" with "cumulative tyre energy" as the
degradation clock would improve transfer between sessions that load the tyre
differently.

**Result: no meaningful difference.** Mean R² gain −0.0005 across six stints.

**Why, and this is the useful part:** per-lap energy varies by only **1.9%**
within a stint (CV 0.019). F1 drivers are extremely consistent, so cumulative
energy is very nearly proportional to lap count, and no improvement is *possible*
within a stint. The cross-session version of the hypothesis remains untested.

Reported rather than buried. A negative result with a mechanism is more useful
than a quiet retreat.

---

## 7. Where a flexible model beats us, and why it does not matter

On lap-time prediction over chronological folds, **LightGBM wins**: CRPS 0.677
against the state-space model's 0.949.

It also:

- has **no parameter meaning "degradation rate"**, so there is nothing to report
  to an engineer and nothing to carry from Friday to Sunday;
- is badly overconfident, with 60% coverage on nominal 95% intervals;
- **cannot extrapolate.** Bias drift across successive folds is +0.340 for
  LightGBM and +1.29 for the naive baseline, against **−0.136** for the
  state-space model — the only rung whose error does not grow as it forecasts
  further past its training window.

A tuned MLP finishes last on lap-time prediction (CRPS 1.643) and has the *most
unstable* extrapolation of any rung, with bias drift of −3.324 — its error swings
by more than three seconds between folds. It was tuned first, across five
configurations on held-out folds, because beating a badly-configured competitor
would prove nothing.

That last column is the physics-informed argument demonstrated rather than
asserted: encoding fuel as physics lets a model extrapolate it; learning fuel as
a pattern does not, and the more flexible the learner, the worse the
extrapolation behaves.

---

## 8. Honest claims

**We claim:**

- The model recovers a known degradation rate under controlled confounding,
  95.5% better than the standard method.
- The identifiability structure of the problem is characterised, and two of three
  collinearities are resolved by stated assumption.
- Physics recovers circuit geometry it was never told, on 7 of 8 circuits.
- Practice-derived degradation predicts race degradation at 0.0518 s/lap MAE,
  56% better than naive, with 90% interval coverage.
- The same estimator transfers to turbofan degradation with real ground truth.

**We do not claim:**

- To measure tread depth. Public telemetry does not contain it.
- To measure tyre temperature. The thermal model produces *estimated states*.
- Causal identification in the interventional sense. This is structural
  attribution under a stated model.
- Any validated fleet result. No public dataset pairs tread depth with
  telematics.
- To beat purpose-built prognostics models on C-MAPSS. Published RMSE there is
  12–20 cycles; ours is 26.5.

---

## Sources

- [arXiv:2512.00640](https://arxiv.org/abs/2512.00640) — State-space F1 tyre degradation
- [arXiv:2607.06495](https://arxiv.org/abs/2607.06495) — Pitwall
- [arXiv:2501.04067](https://arxiv.org/abs/2501.04067) — Explainable tyre energy
- [PMC12915245](https://pmc.ncbi.nlm.nih.gov/articles/PMC12915245/) — Rubber wear review
- [S0043164824000565](https://www.sciencedirect.com/science/article/pii/S0043164824000565) — MegaRide tyre wear
- [TUMFTM/race-simulation](https://github.com/TUMFTM/race-simulation) — Heilmeier race simulator
- [NASA PCoE](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/) — C-MAPSS
- [docs.fastf1.dev](https://docs.fastf1.dev/) — FastF1
