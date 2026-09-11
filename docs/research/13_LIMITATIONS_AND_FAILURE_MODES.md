# Limitations and failure modes

*Where TyreMind is wrong, or would be. A tool that cannot say this is not
engineering software.*

---

## 1. The assumptions that carry the result

Two of the three collinearities in this problem are resolved by prior, not by
data. No amount of single-session data will resolve them.

| Assumption | Value | If it is wrong by 1 sd |
|---|---|---|
| Fuel effect | 0.030 ± 0.005 s/kg | Recovered rate moves ~0.020 s/lap |
| Fuel burn rate | 2.7 ± 0.3 kg/lap | (combined into the above) |
| Track evolution total | 0.90 ± 0.45 s | Recovered rate moves ~0.009 s/lap |

Even at the worst perturbation tested, error stays at 0.0199 s/lap against the
naive method's 0.0966 — but the direction of the answer is ours, not the data's.

**A team with real fuel telemetry should replace the prior.** The architecture
supports it directly, and doing so would remove the largest assumption in the
method.

---

## 2. Calibration is imperfect, in a specific direction

The synthetic benchmark reports **100% interval coverage over 75 comparisons**
against a nominal 95%. That is not a perfect score — it means the intervals are
**conservative**, wider than strictly necessary.

For decision support that is the safer error, but it is a miscalibration and is
reported as one. Practice-to-race coverage went the *other* way and on a sample
large enough to mean something: 76% over 62 comparisons against a nominal 95%.
That is overconfidence, and it is the more dangerous direction, because a
strategist can plan around a wide interval and cannot plan around a wrong one.

The cause is not a broken filter. The posterior sd correctly answers "how well do
these practice laps pin down the practice rate", which is not the question the
number gets used for; nothing about the transfer to Sunday is inside it. Split
conformal calibration fixes it to 95% at the honest cost of a wider interval
(+-0.25 s/lap against the Gaussian's +-0.14). An earlier version of this document
reported 90% over 10 comparisons; that sample was too small to have detected the
problem.

**And it goes the other way on lap-time prediction.** In the model ladder our
95% lap-time intervals cover only **82%** of observations, measured across
66,606 held-out laps from twenty races. That is *under*-confidence's opposite —
intervals too narrow — and pooled regression does better at **85%**. Every rung
in that ladder undercovers, so it is partly a property of the task, but ours is
not the best of them and the direction is the opposite of the conservatism
reported above.

The two are not in conflict: the degradation *rate* is a slowly-varying pooled
state with a wide prior, while a single lap time carries driver noise the model
deliberately does not try to explain. But a reader is entitled to be told both
numbers, not only the flattering one.

**This is now fixed, by a different method from the one used for degradation
rates, and the difference is the interesting part.** Split conformal repaired the
practice→race intervals because one rate per compound per event is exchangeable —
the events could be shuffled without changing anything. Lap times inside a race
are the opposite: the car burns fuel, the track rubbers in, a safety car
rearranges the order. A fixed quantile calibrated on the first half of a race is
calibrated for a race that no longer exists.

Adaptive Conformal Inference (Gibbs & Candès, 2021) drops the exchangeability
assumption rather than hoping it holds, moving the working miss-rate by
`γ(α − err)` after every lap. Pooled over every rung it reaches **95.2%** against
the Gaussian's 75.5%, at a median width of 7.37 s. The nonconformity score also
had to change: the *studentised* score that won for degradation rates is wrong
here, because dividing by a posterior sd that is itself badly wrong amplifies the
miscalibration instead of correcting it.

Two honest caveats. A wider interval is not a better model — this makes the
uncertainty truthful, it does not make the prediction sharper. And the *live*
filter was never as badly off as the batch ladder: its innovation variance is a
real predictive variance and already covered 90–95%, so calibration holds it at
95% rather than rescuing it.

---

## 3. Situations where the model should not be trusted

| Situation | What goes wrong | What the product does |
|---|---|---|
| **Wet or drying track** | Wet-compound degradation is a different physical process; the priors do not describe it. | Wet compounds excluded at ingestion. Silverstone 2024 (mixed conditions) shows 0.82 s residual noise against Monza's 0.42 s — visible in diagnostics. |
| **Extrapolating past observed tyre age** | The local linear trend extends a straight line into a cliff it cannot see. | Applicability score decays past the oldest observed age; shown on every projection. Below 50% the UI says the model is extrapolating. |
| **A compound run only briefly** | The estimate is dominated by its prior. | Lap count per compound shown; `assess_applicability` flags it explicitly. |
| **Short stints** | Under ~5 laps cannot show a trend. | Runs under 4 laps dropped and counted. |
| **Safety car, red flag, VSC** | Slow laps corrupt the trend. | Removed by a robust median-absolute-deviation threshold and counted. |
| **Single-car analysis** | Loses the run-stagger identification. | Measured: halving the field moves error from 0.0047 to 0.0073. |
| **Attributing degradation to a driver** | Driver and car are perfectly confounded, and the per-stint rate spread the model fits sits at the floor of its search range. | Not attempted. Tested in `exp15`; the result is reported there rather than shipped as a feature. |
| **Puncture, debris, damage** | Represented as smooth degradation; a step change is not in the model. | Innovation z-scores are surfaced in the live monitor — a run of large same-signed innovations means the model is being surprised. |
| **New circuit with no telemetry analysed** | Per-corner energy unavailable. | The twin says so rather than showing an even split as a result. |
| **Sprint or heavily disrupted sessions** | Few long runs. | Session quality score surfaced; low scores indicate the session cannot support a conclusion. |

---

## 4. Known systematic biases

**Practice over-predicts race degradation by +0.042 s/lap**, in 42 of 62
comparisons across 27 events in 2024 and 2023.

An earlier version of this section proposed a physical cause — practice race-sims
holding high fuel throughout — and called the cross-session test untested. **It
has since been run, and that explanation is not supported.** Nine candidate
mechanisms were tested against the signed error with a Benjamini–Hochberg
correction across all nine. Two survive: mean practice stint length (ρ +0.37,
p 0.003) and pit stops per driver (ρ +0.33, p 0.009). The fuel-load reading is
not among them, and neither are the temperature gap, traffic, or the model's own
posterior sd.

The causal story those two implied — practice running deeper into the wear curve
than a pitted race — was then built and **refuted**: practice runs are on average
4.4 laps *shallower* than race stints, and forcing a common tyre-age window made
both bias and error worse.

So the bias is reported and corrected as a **measured offset, not an explained
one**, and the interval is widened by conformal calibration until it covers what
it claims. A bias whose mechanism is unknown is worth stating plainly; a
mechanism asserted without evidence is worth less than nothing.

---

## 4b. The cross-domain number is now like-for-like -- it was not before

This section used to read: *"our C-MAPSS figure of 26.5 cycles RUL RMSE is scored
on 40 of the 100 engines... making it comparable is a tractable engineering
problem -- the fit would need to be batched per engine rather than joint -- and
it is not done."*

**It is done.** The estimator fits the test set jointly, so cost grows faster
than linearly and a 100-engine run had been abandoned after 108 CPU-minutes
without converging. Fitting in batches of 25 removes the limit, because each
engine's level and rate are its own states driven by its own observations; the
only quantity pooled across engines is the shared degradation baseline, which is
already estimated from the training engines.

The figure is now **22.7 cycles RUL RMSE over all 100 FD001 test engines** --
the same set the published 12-20 cycles is quoted over. MAE 17.9, and 44%
predicted early, which is the safe direction for a prognostics model.

Two things worth stating about that, because the result improved and an
improvement is exactly when a reader should be most suspicious:

- **Batching does not flatter it.** Re-running the original 40 engines in one
  batch reproduces 26.5/21.1/1893 exactly. At batch size 20 the same 40 give
  26.6/21.4/1870 -- a 0.4% difference. The improvement from 26.5 to 22.7 comes
  from scoring the other 60 engines, not from how the fit was split.
- **The NASA score is a sum, not a mean.** 2415 over 100 engines against 1893
  over 40 looks worse and is better: 24.2 per engine against 47.3. Any comparison
  of that score between runs with different engine counts is meaningless, and the
  experiment now prints the per-engine figure alongside it.

We still do not claim to be competitive. 22.7 against a published 12-20 is a tyre
model pointed at engines with no retuning.

---

## 5. Things deliberately not built

| Not built | Why |
|---|---|
| Full Pacejka identification | Slip angle and slip ratio are not observable from public telemetry. The parameters would be unconstrained by any available data. |
| MCMC / PyMC | No incremental mode, which would have made the real-time claim impossible. Windows compile risk. The exact likelihood is available in closed form anyway. |
| Neural state-space / temporal fusion transformer | Public data cannot support a model that would beat a six-parameter linear SSM. LightGBM already makes the point about flexible models. |
| Validated fleet product | No public dataset pairs tread depth with telematics. Building a fleet claim without one would be dishonest. |
| Tyre pressure modelling | Not in public data. |

---

## 6. What would most improve this

In order of estimated uncertainty reduction (`models/trust.value_of_information`
— these are engineering judgements, not measurements):

1. **Tyre surface temperature (−35%).** Would separate thermal degradation from
   mechanical wear, which the model currently absorbs into one rate.
2. **Actual fuel mass (−30%).** Would remove the largest assumption entirely.
3. **Measured tread depth (−20%).** Would let the model estimate physical wear
   rather than performance loss, and would make the whole product's central
   caveat unnecessary.
4. **Tyre pressure (−12%).** Would explain part of the within-stint variation now
   treated as noise.

---

## 7. The nine bugs found during development, and what they say

Listed because each one was invisible to normal use and caught only by a test or
a cross-check — which is the argument for having them.

| Bug | Symptom | Found by |
|---|---|---|
| Falsy-zero on `session_lap` | Track basis pinned at zero; all track evolution leaked into degradation as a uniform −0.025 s/lap offset. Invisible on real F1 data, which numbers laps from 1. | Live-vs-batch agreement test |
| `session_progress` recomputed per frame | Chronological test blocks measured progress from their own start. No error, no leak — just wrong by tens of seconds. Made two benchmark rungs look far worse than they are. | Investigating an implausible 33 s MAE |
| Unregularised pooled regression | One Silverstone fold at 176 s MAE while every other fold was under a second. | Per-fold inspection |
| `include_router` silently adding nothing | Eight API endpoints registered zero routes, with no error. Every one would have 404'd. | Route listing after registration |
| FastF1 resolving "Interlagos" to Zandvoort | Analysis of an entirely different circuit, warning-only. | Two circuits producing byte-identical results |
| Inverted confidence interval | `competitive_life_interval()` returned (8, 1). | Reading the output |
| Traffic index identically zero | Grouped by lap number, but two drivers' "lap 5" can be twenty minutes apart. The confounder was in the model and contributing nothing. | Inspecting the fitted coefficient |
| Frozen chart clip path | An interrupted entry animation left a line's expanding clip frozen part-way, so a 43-lap stint rendered as 14 with the axis still scaled to 43 — indistinguishable from a data bug. | Reading pixels back from the canvas |
| Stale experiment result | A committed result file predated the 125-cycle RUL cap added to its own script, overstating cross-domain error at 56 cycles instead of 26.5. | Cross-checking a chart against the script that produced it |

---

## 8. The honest summary

TyreMind separates tyre degradation from its confounders better than the standard
method, by a large and measured margin, under assumptions it states and whose
cost it quantifies.

It does not measure tyre wear. It cannot be validated against real tyre wear,
because no such public data exists. It should never be used for a safety
decision. And on the narrow task of predicting lap times, a gradient-boosted tree
beats it — while being unable to answer the question the product exists to
answer.
