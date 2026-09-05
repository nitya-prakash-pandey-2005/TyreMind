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
reported as one. Practice-to-race coverage is 90% over 10 comparisons, which is
closer to nominal but on far too small a sample to conclude much.

**And it goes the other way on lap-time prediction.** In the model ladder our
95% lap-time intervals cover only **73%** of observations. That is
*under*-confidence's opposite — intervals too narrow — and pooled regression does
better at 79%. Every rung in that ladder undercovers on lap time, so it is partly
a property of the task, but ours is not the best of them and the direction is the
opposite of the conservatism reported above.

The two are not in conflict: the degradation *rate* is a slowly-varying pooled
state with a wide prior, while a single lap time carries driver noise the model
deliberately does not try to explain. But a reader is entitled to be told both
numbers, not only the flattering one.

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
| **Puncture, debris, damage** | Represented as smooth degradation; a step change is not in the model. | Innovation z-scores are surfaced in the live monitor — a run of large same-signed innovations means the model is being surprised. |
| **New circuit with no telemetry analysed** | Per-corner energy unavailable. | The twin says so rather than showing an even split as a result. |
| **Sprint or heavily disrupted sessions** | Few long runs. | Session quality score surfaced; low scores indicate the session cannot support a conclusion. |

---

## 4. Known systematic biases

**Practice over-predicts race degradation by +0.047 s/lap**, in 9 of 10
comparisons across five 2024 events.

Most likely physical cause: practice race-sim runs hold high fuel throughout
while a race stint averages lower, putting more load through the tyre on Friday.
An energy-based degradation clock should absorb it — but `exp04` showed per-lap
energy varies only 1.9% *within* a stint, so the within-stint version of that fix
cannot work. The cross-session version is untested.

Reported rather than corrected, because a bias that is understood is more useful
than one that has been tuned away.

---

## 4b. The cross-domain number is not a like-for-like comparison

Our C-MAPSS figure of **26.5 cycles RUL RMSE** is scored on **40 of the 100
engines in the FD001 test set**, taken in unit-id order. Published figures --
including the 12-20 cycles we quote for purpose-built deep models -- are computed
over all 100.

This is a runtime limit rather than a selection. The estimator fits the test set
jointly, so cost grows faster than linearly in the number of engines: 40 engines
fit in about three minutes, while a full 100-engine run was attempted and
abandoned after 108 CPU-minutes without converging.

The honest statement is therefore that the transfer works and the number is
indicative, not that it sits at a particular distance from the state of the art.
Making it comparable is a tractable engineering problem -- the fit would need to
be batched per engine rather than joint -- and it is not done.

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

## 7. Bugs found during development, and what they say

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
