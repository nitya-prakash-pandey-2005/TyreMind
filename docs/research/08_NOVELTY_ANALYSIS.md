# Novelty analysis

Feature by feature: what already exists, what is ours, and how to phrase the
claim without overstating it.

---

## 1. Testing whether the attribution is correct

**Exists?** No. Every model surveyed in `01_RESEARCH_AUDIT.md` is validated on
lap-time prediction error.

**Why that gap exists and matters.** Fuel burn-off, track evolution and tyre
degradation all move lap time smoothly with lap number. Many wrong decompositions
sum to the same right total, so a model can predict lap times almost perfectly
while blaming entirely the wrong cause — and no lap-time metric will notice.

**Ours.** Generate sessions where the true degradation rate is set by us, bury it
under realistic confounding, and measure recovery per cause.

**Honest claim:** *"We test whether the attribution is right, not just whether
the prediction is close. On synthetic sessions with a known hidden rate, our
error is 0.0044 s/lap against the standard method's 0.0966."*

**Do not say:** that we have validated attribution on real F1 data. Public data
contains no measured wear, so that test is not available to anyone.

---

## 2. The track-evolution collinearity

**Exists?** Not that we found. Cappello and Hoegh model fuel and latent tyre pace
and do not include track evolution; the collinearity therefore does not arise for
them.

**Ours.** Shifting every degradation rate by *c* and the track slope by *−c*
leaves a difference constant *within a run* — exactly what a run intercept
absorbs. The two are structurally indistinguishable, and scrubbed sets do not
help. Resolved by modelling track evolution as a saturating curve with an
informative amplitude prior.

**Honest claim:** *"We characterise a second collinearity in this problem and
resolve it with a physically motivated basis. It was found by chasing a residual
bias, not derived in advance."*

**Do not say:** that we discovered something the field missed. It is a
consequence of adding a term others did not model.

---

## 3. Whole-field pooling

**Exists?** Partially. Multilevel modelling of F1 driver and constructor
performance is established (Bristol/White Rose, 1950–2014). Applying it to tyre
degradation with pit-stagger as the identifying variation is, as far as we found,
not published.

**Ours.** Cars change tyres on different laps, so at any session lap the grid
spans a wide range of tyre ages. That stagger decorrelates tyre age from session
lap. It is the one identification here that costs nothing — no prior, just the
whole grid instead of one car.

**Honest claim:** *"Pit stagger across the field is a natural experiment that
separates tyre age from session lap. A single-driver model cannot use it."*

---

## 4. Per-corner energy from public position telemetry

**Exists?** The physics is textbook. Applying it to *public* F1 position
telemetry, and validating it against circuit geometry, we did not find published.
MegaRide's work needs rig data; the tyre-energy paper uses private Mercedes
telemetry.

**Ours.** Curvature from X/Y, lateral acceleration, load transfer, per-corner
frictional energy — then checked by recovering circuit rotation direction on 7 of
8 circuits, having never been told it.

**Honest claim:** *"Public position telemetry supports a per-corner loading
estimate good enough to recover circuit geometry it was never given."*

**Do not say:** that we measure tyre loads. Frictional power is a proxy; slip
velocity is not observable.

---

## 5. Real-time online estimation

**Exists?** Pitwall runs live. Kalman filtering is decades old.

**Ours.** Not the filter — the *architectural consequence*. Choosing a recursive
estimator over a sampler means the same model that produces the retrospective
curve runs forward-only during the session at 0.22 ms per lap, cost flat in
session length, and agreeing with the batch implementation to 0.0006 s/lap at
worst. (Those are two separate implementations. The filtered and smoothed
estimates of a *single* fit are identical at the final step by construction, so
comparing those would prove nothing.)
Filtered and smoothed estimates are kept strictly separate.

**Honest claim:** *"The same estimator serves both modes because it is recursive.
An MCMC formulation could not."*

**Do not say:** that real-time tyre estimation is new.

---

## 6. Cross-domain transfer with real ground truth

**Exists?** C-MAPSS is a heavily-studied benchmark with a large literature.

**Ours.** Using it to *validate a motorsport model's generalisation*, because
motorsport itself cannot supply ground truth. The `AssetProfile` abstraction
makes the claim a property of the code rather than a slide.

**Honest claim:** *"The identical estimator, with no tyre-specific code, predicts
turbofan remaining life at 26.5 cycles RMSE against published labels."*

**Do not say:** that we are competitive on C-MAPSS. Purpose-built models reach
12–20 cycles. We are demonstrating transfer, not entering a leaderboard.

---

## 7. Race strategy simulation

**Exists?** Yes, thoroughly. Heilmeier's simulator is the open-source reference;
Pitwall runs a calibrated one live.

**Ours.** Only the input. Published simulators take a degradation rate as given,
usually a single fitted slope. Ours arrives as a posterior whose width came from
a stated identifiability argument, and is **sampled per simulated race** rather
than fixed — so outcome spread reflects genuine tyre uncertainty, and a strategy
that only wins under a confident estimate visibly stops winning.

**Honest claim:** *"We do not claim novelty in race simulation. What differs is
that the degradation rate enters as a distribution."*

---

## 8. What is NOT novel, stated plainly

- Predicting tyre degradation with a state-space model. **Published, Nov 2025.**
- Monte Carlo pit-stop optimisation. **Heilmeier et al.**
- LLM-narrated race strategy. **Pitwall, with a stricter faithfulness protocol
  than ours.**
- Gradient boosting on tyre features. **Standard practice.**
- Kalman filtering, Archard wear, Pacejka, load transfer. **Textbook.**

---

## The one-sentence version

> Existing work predicts lap times well and never checks whether the reason it
> gives is the right one. We characterise why that check is hard — three
> collinearities, two resolvable only by assumption — build an estimator that
> handles them, and test it against a truth we control, on a second asset class
> with real ground truth, and on the practice-to-race transfer the challenge
> actually asks for.

---

## Where a competitor beats us

Stated because a novelty claim that hides its losses is not credible.

**LightGBM predicts lap times better than we do** — CRPS 0.677 against 0.949.
It also has no degradation parameter, is badly overconfident (60% coverage on
nominal 95%), and cannot extrapolate: bias drift +0.340 against our −0.136.

We report this in the product, not just here.
