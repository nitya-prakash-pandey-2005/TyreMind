# Model card — TyreMind latent tyre-state estimator

**Version** 0.1.0 · **Updated** 2026-09-05

---

## What it does

Estimates the **latent performance state of a tyre** from lap times and
telemetry, separating degradation from the conditions that also change lap times.

Output is a degradation rate in **seconds per lap**, with a credible interval,
per compound and per individual tyre set.

## What it does not do

**It does not measure tread depth.** Public Formula 1 telemetry contains no
physical wear measurement. There is no sensor reading, no tread gauge, no rubber
mass. The model estimates how much *performance* a tyre has lost, which is
related to physical wear but is not the same quantity and is not calibrated
against it.

**It does not measure tyre temperature.** The thermal model produces estimated
states calibrated so their output correlates with observed degradation. They are
not validated against any temperature sensor, because none is available. Absolute
values in degrees are not trustworthy; the relative structure is what is used.

**It does not perform causal identification.** The decomposition is exact
arithmetic on an assumed structural model. Two of its three identifying
assumptions are priors, not evidence. "Structural attribution under the stated
model" is the accurate description.

---

## Intended use

- Post-session analysis of tyre degradation by race engineers and strategists.
- Estimating race degradation from practice running.
- Real-time degradation monitoring during a session, as decision *support*.
- Research into degradation estimation under confounding.

## Out of scope

- **Any safety decision.** This is not a roadworthiness assessment and must never
  be used as one. For road or commercial vehicles, physical inspection and
  certified TPMS remain authoritative.
- Wet-weather running. Wet compounds are excluded; their degradation is a
  different physical process the priors do not describe.
- Circuits, compounds or conditions outside the fitted session, beyond the
  applicability limits the model reports itself.
- Regulatory, scrutineering or commercial decisions of any kind.

---

## Data

**Source.** [FastF1](https://docs.fastf1.dev/), which reads Formula 1's official
timing API. Lap times, tyre compound, tyre age, stint numbering, weather, and —
for the physics layer — car and position telemetry at roughly 4–10 Hz.

**Filtering.** From a raw session, removed: laps with no time, pit in/out laps,
laps FastF1 flags as inaccurate, laps with unknown compound, wet compounds, laps
more than 3 robust deviations slower than the session median (safety cars,
flags, severe traffic), and runs shorter than 4 laps. At 2024 Monza FP2 this is
454 laps in, 143 out. Every exclusion is counted and reported.

**Synthetic data** is used *only* for validating recovery of a known truth, and
is never mixed with real telemetry.

**Cross-domain data:** NASA C-MAPSS turbofan degradation (Saxena & Goebel, 2008).

---

## Assumptions

Listed in descending order of how much the result depends on them.

1. **Fuel costs 0.030 ± 0.005 s/kg, burned at 2.7 ± 0.3 kg/lap.** The largest
   assumption. Fuel and degradation are perfectly collinear within a run, so this
   prior is what makes degradation identifiable at all. Uncertainty is carried as
   state and widens every published interval.
2. **Track evolution follows a saturating curve, total 0.90 ± 0.45 s.** Resolves
   the second collinearity. Rubber deposition genuinely saturates, so the shape
   is physically motivated, but the amplitude is a prior.
3. **Degradation is smooth in tyre age.** The rate is a random walk, so it can
   accelerate (a cliff) or plateau, but it cannot jump discontinuously. A
   puncture or debris strike is not represented.
4. **Traffic effect is proportional to a saturating gap index.** Cars more than
   2s apart are treated as being in clean air.
5. **Observation noise is heavy-tailed.** Lap times contain lock-ups and
   mistakes, which a Gaussian model would absorb into the tyre state.
6. **A run's intercept absorbs car pace, setup and starting fuel mass.** Anything
   constant within a run is not attributed to the tyre.

---

## Performance

### Recovery of a known degradation rate — 25 synthetic sessions

| | Naive (lap time vs tyre age) | TyreMind |
|---|---:|---:|
| Mean absolute error | 0.0966 s/lap | **0.0044 s/lap** |
| Bias | −0.0966 | **+0.0012** |
| 95% interval coverage | — | **100%** |

The naive bias equals the fuel slope, in the direction theory predicts.

**On coverage:** 100% over 75 comparisons against a nominal 95% suggests the
intervals are slightly *conservative* — wider than strictly necessary. For a
decision-support tool that is the safer error, but it is a known
miscalibration, not a perfect result.

### Practice → race transfer — 2024, 5 events, 10 compound comparisons

| | Naive | TyreMind |
|---|---:|---:|
| MAE | 0.1166 s/lap | **0.0518 s/lap** |
| 95% coverage | — | 90% |
| Bias | — | **+0.0472 s/lap** |

**The bias is systematic** — practice over-predicts race degradation in 9 of 10
comparisons. Most likely cause: practice race-sim runs hold high fuel throughout
while a race stint averages lower, putting more load through the tyre on Friday.
A known, consistent bias is correctable; this one is reported rather than tuned
away.

### Lap-time prediction — 4 real races, chronological folds

| Model | CRPS | Coverage | Bias drift |
|---|---:|---:|---:|
| LightGBM | **0.677** | 60% | +0.340 |
| Pooled regression (ridge) | 0.813 | 79% | +0.663 |
| TyreMind state-space | 0.949 | 73% | **−0.136** |
| Naive | 1.119 | 69% | +1.280 |
| Fuel-corrected regression | 1.120 | 69% | +1.293 |
| Neural network (MLP) | 1.643 | 69% | −3.324 |

**LightGBM predicts lap times better than we do, and cannot answer the
question.** Neither it nor the neural network has a parameter meaning
"degradation rate", so there is nothing to hand an engineer and nothing to carry
from Friday to Sunday.

Bias drift measures how much a model's error grows as it forecasts further past
its training window. TyreMind is the only rung whose error does not grow. The MLP
is the most unstable of all at −3.324 — its bias swings wildly between folds,
which is what unconstrained extrapolation looks like.

The MLP was tuned before being compared (five configurations on held-out folds;
disabling early stopping, which was validating on ~30 rows, was worth roughly a
full second of CRPS). Beating a badly-configured competitor would prove nothing.

### Sensitivity to the assumptions

With the fuel prior wrong by a full standard deviation — the single largest
assumption — error is **0.0199 s/lap**, still 5× better than naive at 0.0966.
Doubling both prior widths roughly doubles the posterior standard deviation, as
it should. Coverage holds at 100% across every perturbation.

### Cross-domain — NASA C-MAPSS FD001

RUL RMSE **56 cycles** over 60 engines, 48% predicted early. Purpose-built deep
prognostics models reach 12–20 on this dataset. This is a tyre model pointed at
engines with no retuning, so it demonstrates transfer, not competitiveness.

---

## Known failure modes

| Situation | What happens | Mitigation in the product |
|---|---|---|
| **Wet or drying track** | Priors do not describe wet compounds; the model would report nonsense. | Wet compounds excluded at ingestion. Silverstone 2024 (mixed conditions) shows residual noise of 0.82s against Monza's 0.42s — the diagnostic is visible. |
| **Very short stints** | Fewer than ~5 laps cannot show a trend; the estimate is mostly prior. | Runs under 4 laps dropped; run count and length reported. |
| **Extrapolating past observed tyre age** | The local linear trend extends a straight line into a cliff it cannot see. | Applicability score decays past the oldest observed age and is shown alongside every projection. |
| **A compound run only briefly** | Estimate dominated by the compound prior. | Lap count per compound shown; `assess_applicability` flags it. |
| **Safety car / red flag** | Slow laps corrupt the trend. | Removed as outliers via a robust threshold and counted. |
| **Single-car analysis** | Loses the run-stagger identification; uncertainty rises. | Measured: halving the field degrades error from 0.0047 to 0.0073. |
| **A prior that is simply wrong** | The answer shifts. | Quantified in exp02: a full-sd error in the fuel prior costs 0.0199 s/lap. |
| **New circuit, no telemetry analysed** | Per-corner energy unavailable. | The twin says so rather than showing an even split as a result. |

---

## Uncertainty

Every published estimate carries a posterior standard deviation. Sources:

- **Aleatoric** — lap-time scatter from driver variation, modelled with a
  heavy-tailed observation distribution.
- **Epistemic** — state uncertainty from the Kalman recursion, plus the width of
  the physical priors, which is carried as state and therefore propagates into
  every derived quantity including strategy outcomes.

Two estimates are reported and never conflated: **filtered** (conditioned only on
laps so far — what the pit wall could legitimately know) and **smoothed**
(conditioned on the whole session — what engineers know afterwards).

---

## Ethical and safety notes

- **Decision support only.** Never an autonomous or safety-certifying system.
- Fleet and passenger-vehicle applications shown in the product are
  **architecture, not evidence**, and are labelled as such throughout.
- The narration layer never lets a language model compute a number. Templates
  generate the text from model output; an LLM may only rewrite prose, and any
  rewrite introducing an unverifiable number is discarded.
- No personal data is processed. All inputs are publicly published timing data.

---

## Reproducing these numbers

```bash
python experiments/exp01_ground_truth_recovery.py --n-seeds 25
python experiments/exp02_prior_sensitivity.py
python experiments/exp03_practice_to_race.py --year 2024
python experiments/exp04_energy_clock.py
python experiments/exp05_model_ladder.py
python experiments/exp06_circuit_asymmetry.py
python experiments/exp07_cross_domain.py --subset FD001
```

Results are written to `experiments/results/*.json` and read from there by the
dashboard. No figure in this document or in the product is typed by hand.
