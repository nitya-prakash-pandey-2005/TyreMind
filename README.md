<div align="center">

# TYREMIND

### Causal Tyre Intelligence

**Observed performance is not the same thing as tyre degradation.**

</div>

---

## The problem, in one example

Here is a real stint from the 2024 Italian Grand Prix. Over forty laps the car
got **1.78 seconds faster**.

A stopwatch says the tyre is fine. It is not. Over those same laps the hard tyre
lost **2.15 seconds** of performance — the car got quicker because it burned off
3.27 seconds of fuel weight, and the fuel gain was larger than the tyre loss.

Read lap times alone and you miss a dying tyre completely.

That is not an edge case. Fit the standard method — a straight line through lap
time against tyre age — to a real race and it reports **negative degradation**:
tyres apparently getting faster the longer they run. It happens on every race we
tested, because fuel burn-off is worth about 0.08 s/lap and is simply bigger than
the effect being measured.

## What TyreMind does

Estimates the **latent performance state of a tyre** underneath a confounded
observation, separating degradation from fuel burn-off, track evolution and
traffic — and reports how sure it is about each.

```bash
python -m tyremind.serve             # dashboard, one command, no network needed
python -m tyremind.mcp_server        # the same model as tools an AI agent can call
```

Nine screens, dark and light, including a **3D circuit** coloured by the
frictional load the physics layer computes at each point — so you can see *where
on the lap* a tyre gets used up, not just how fast it goes away.

---

## Results

Every figure below is produced by a script in `experiments/` and read from
`experiments/results/*.json`. Nothing here is typed by hand.

### Can it recover a degradation rate it was never shown?

25 synthetic sessions with a known hidden rate, buried under realistic
confounding:

| | Naive (lap time vs tyre age) | **TyreMind** |
|---|---:|---:|
| Mean absolute error | 0.0966 s/lap | **0.0044 s/lap** |
| Bias | −0.0966 | **+0.0012** |
| 95% interval coverage | — | **100%** |

**95.5% error reduction.** The naive bias equals the fuel slope, in the direction
theory predicts — the collinearity showing up as a measured quantity.

### Does a Friday curve predict Sunday?

2024, five events, ten compound comparisons. No race data reaches the practice
fit:

| | Naive | **TyreMind** |
|---|---:|---:|
| MAE | 0.1166 s/lap | **0.0518 s/lap** |
| 95% coverage | — | 90% |

There is a **systematic +0.047 s/lap bias** — practice over-predicts race
degradation in 9 of 10 comparisons. Reported, not tuned away.

### Does the physics compute what it claims?

The pipeline goes GPS trace → curvature → lateral acceleration → per-corner load.
It is never told which way a circuit runs. A clockwise circuit must load its
*left* tyres more.

**7 of 8 circuits recovered correctly.** Clockwise circuits show 21–35% left-turn
energy; anti-clockwise 60–75%. Austin misses at 46.2% and is reported as a miss.

### Does it work on something that is not a tyre?

Public F1 data has no measured tyre wear, so motorsport cannot supply ground
truth. NASA's C-MAPSS turbofan benchmark does.

**Same estimator, no tyre-specific code:** 56-cycle RUL error over 60 engines,
48% predicted early. Purpose-built deep models reach 12–20 on that dataset — this
demonstrates transfer, not competitiveness.

### Where we lose

| Model | CRPS (lap time) | Coverage | Bias drift |
|---|---:|---:|---:|
| LightGBM | **0.677** | 60% | +0.340 |
| TyreMind | 0.949 | 73% | **−0.136** |

**LightGBM predicts lap times better than we do.** It also has no parameter
meaning "degradation rate", is badly overconfident, and cannot extrapolate — bias
drift measures how much a model's error grows as it forecasts further past its
training window, and TyreMind is the only model tested whose error does not grow.

---

## Why this is hard

Three causes push lap time the same way, so many wrong decompositions sum to the
same right total.

| Collinearity | Resolved by | Evidence or assumption? |
|---|---|---|
| Fuel vs degradation, within a run | Physical prior, 0.030 s/kg × 2.7 kg/lap | **Assumption** |
| Track evolution vs a uniform rate shift | Saturating basis + amplitude prior | **Assumption** |
| Tyre age vs session lap | Fitting the whole field — pit stagger | **Evidence** |

Only one of the three is resolved by data. `exp02_prior_sensitivity` measures what
the other two cost if wrong: with the fuel prior off by a full standard deviation,
error is 0.0199 s/lap — still 5× better than naive.

The second collinearity was not anticipated. It was found by chasing a −0.013
s/lap bias that survived removing the cliff, scrubbed sets and traffic from the
generator. Shift every degradation rate by *c* and the track slope by *−c*, and
the difference is constant *within a run* — exactly what the run intercept
absorbs.

---

## Scientific honesty

TyreMind does **not** measure tread depth. Public telemetry does not contain it.
It estimates a *latent performance state*.

It does **not** measure tyre temperature — the thermal model produces estimated
states, calibrated against degradation rather than any sensor.

It does **not** perform causal identification. The decomposition is exact
arithmetic on an assumed structural model, two of whose assumptions are priors.

See [`docs/model_card.md`](docs/model_card.md) and
[`docs/research/13_LIMITATIONS_AND_FAILURE_MODES.md`](docs/research/13_LIMITATIONS_AND_FAILURE_MODES.md).

---

## Quickstart

```bash
py -3.11 -m venv .venv
.venv\Scripts\activate                  # Windows
pip install -r requirements-dev.txt
pip install -e .

pytest                                  # 81 tests
python -m tyremind.serve                # dashboard at http://127.0.0.1:8077
```

Eight sessions are committed as Parquet, so a fresh clone runs offline. To cache
more:

```bash
python scripts/build_demo.py --events Monza Suzuka --year 2024
```

Frontend development:

```bash
cd apps/web && npm install && npm run dev
```

---

## Reproducing every number

```bash
python experiments/exp01_ground_truth_recovery.py --n-seeds 25
python experiments/exp02_prior_sensitivity.py
python experiments/exp03_practice_to_race.py --year 2024
python experiments/exp04_energy_clock.py
python experiments/exp05_model_ladder.py
python experiments/exp06_circuit_asymmetry.py
python experiments/exp07_cross_domain.py --subset FD001
```

---

## Structure

```
src/tyremind/
  data/       FastF1 ingestion, quality engine, synthetic ground truth
  models/ssm/ Kalman kernel and the tyre state-space model
  models/     baselines, evaluation harness, trust layer, validation
  physics/    dynamics, thermal, wear
  causal/     decomposition, counterfactuals, projection
  simulate/   Monte Carlo race and strategy
  assets/     AssetProfile abstraction, C-MAPSS adapter
  stream/     online estimator and replay
  explain/    narration templates, business value
  api/        FastAPI service
apps/web/     Vite + React dashboard
experiments/  reproducible scripts; results/ holds the JSON the UI reads
docs/         research audit, model card, limitations, demo guide
```

---

## Documentation

| | |
|---|---|
| [Research audit](docs/research/01_RESEARCH_AUDIT.md) | Prior art, feasibility, what was cut and why |
| [Data availability](docs/research/03_DATA_AVAILABILITY.md) | What public F1 data does and does not contain |
| [Physics foundation](docs/research/04_PHYSICS_FOUNDATION.md) | Dynamics, thermal, wear — and their validation |
| [Statistical architecture](docs/research/05_STATISTICAL_ARCHITECTURE.md) | The model, and why a Kalman filter |
| [Novelty analysis](docs/research/08_NOVELTY_ANALYSIS.md) | What is ours, what is not |
| [Limitations](docs/research/13_LIMITATIONS_AND_FAILURE_MODES.md) | Where it is wrong |
| [Model card](docs/model_card.md) | Intended use, assumptions, performance |
| [Demo guide](docs/DEMO_STORY.md) | Seven-minute run-through |
| [Judge questions](docs/JUDGE_QUESTIONS.md) | Anticipated questions, honest answers |
| [Integrations](docs/INTEGRATIONS.md) | MCP and RAG — what each is for, and its limits |

---

## Licence

MIT
