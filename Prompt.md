# MASTER BUILD PROMPT — TYREMIND

## Role

You are acting as a combined:

* Principal ML Engineer
* Vehicle Dynamics Engineer
* Tire/Tyre Physics Researcher
* Statistical Modelling Researcher
* Simulation Engineer
* Data Scientist
* MLOps Engineer
* Backend Engineer
* Frontend/Product Engineer
* Technical Product Architect
* Startup CTO
* Hackathon Competition Strategist

We are a 4-person engineering team participating in a high-level motorsport/AI innovation challenge.

We have approximately 24 hours for the actual build, but we want to prepare the architecture, research, data strategy, prompts, schemas and development plan beforehand.

Our objective is NOT to build a superficial hackathon dashboard.

We want to build a prototype that makes expert judges think:

> "This is much deeper than a tyre degradation predictor. This is a serious race-engineering intelligence platform."

The product should also have a credible path beyond motorsport into:

* automotive engineering
* EVs
* commercial fleets
* tyre manufacturers
* predictive maintenance
* remaining-useful-life estimation
* vehicle testing
* mobility
* industrial wear/degradation monitoring

The product name is:

# TYREMIND

### Causal Tyre Intelligence & Digital Twin Platform

Core positioning:

> **TyreMind estimates the hidden health and performance state of a tyre from noisy vehicle observations, separates true degradation from confounding effects, predicts future performance, simulates counterfactual scenarios, and converts the result into explainable operational decisions.**

Do NOT reduce this to:

> "AI predicts tyre degradation."

That is too generic and already exists.

The central intellectual proposition is:

> **Observed vehicle performance is not the same thing as tyre degradation.**

We want to estimate the latent tyre state underneath a noisy, confounded observation process.

---

# 0. CRITICAL INSTRUCTION — AUDIT THIS ENTIRE PROMPT BEFORE BUILDING

Before writing substantial code, critically validate this entire specification.

You MUST:

1. Identify which components are realistically implementable in a 24-hour prototype.
2. Identify which components require unavailable/private data.
3. Identify which physics equations can actually be parameterized using public data.
4. Identify which claims would be scientifically invalid if made with public F1 data.
5. Identify which components already exist in academic literature or commercial products.
6. Identify where this architecture is genuinely differentiated.
7. Identify redundant or low-value features.
8. Identify features that sound impressive but cannot be validated.
9. Propose better alternatives where necessary.
10. Check every mathematical/physics assumption.
11. Check whether the proposed model architecture is computationally realistic.
12. Check whether the data schema supports every requested model.
13. Check whether the simulation is identifiable from the available data.
14. Check whether the multi-industry extension is technically credible.
15. Check whether any feature would create misleading claims about "actual tyre wear."
16. Search current research and existing solutions before finalizing the architecture.

Do NOT blindly follow this prompt.

You are explicitly authorized to modify, replace, remove or add components.

After auditing, produce:

### A. Feasibility assessment

Rate every proposed component:

* GREEN = realistic in 24 hours
* YELLOW = possible but simplified
* RED = not realistically implementable/validated in 24 hours

### B. Research novelty assessment

For every major feature:

* Existing research?
* Existing commercial solution?
* Our differentiation?
* How to phrase the claim honestly?

### C. Recommended architecture

Produce your improved architecture.

### D. 24-hour MVP architecture

Identify the minimum system that MUST work.

### E. Stretch architecture

Identify components that should only be implemented if the MVP is stable.

ONLY AFTER THIS AUDIT SHOULD YOU START BUILDING.

---

# 1. PRODUCT VISION

Build TyreMind as a multimodal engineering intelligence system.

The high-level pipeline should be:

RAW DATA

↓

DATA QUALITY / SYNCHRONIZATION

↓

PHYSICAL FEATURE EXTRACTION

↓

BASELINE VEHICLE PERFORMANCE MODEL

↓

CONFOUNDING-EFFECT ESTIMATION

↓

LATENT TYRE STATE ESTIMATION

↓

PHYSICS-CONSTRAINED DEGRADATION MODEL

↓

UNCERTAINTY QUANTIFICATION

↓

FUTURE TYRE PERFORMANCE FORECAST

↓

COUNTERFACTUAL ENGINE

↓

RACE / VEHICLE STRATEGY OPTIMIZER

↓

POST-EVENT VALIDATION

↓

DIGITAL TWIN

↓

CROSS-INDUSTRY TYRE HEALTH PLATFORM

---

# 2. IMPORTANT SCIENTIFIC POSITIONING

Do not claim:

> "We directly measure physical tread wear."

Unless actual tread-depth measurements exist.

With public F1 telemetry we generally infer a latent performance/degradation state.

Therefore distinguish:

### Physical wear

Actual physical loss of rubber/material.

### Performance degradation

Loss of useful tyre performance caused by wear, thermal state, pressure, surface condition, etc.

### Latent tyre state

A hidden state inferred from observations.

TyreMind's initial public-data product should primarily estimate:

> **Latent tyre performance state and degradation rate.**

When physical wear measurements are available in another industry dataset, the same architecture can estimate actual wear/RUL.

This distinction is critical for scientific credibility.

---

# 3. RESEARCH FOUNDATION

Before implementation, research and document the relevant literature.

At minimum investigate:

## Tyre dynamics

* Pacejka Magic Formula
* longitudinal slip
* lateral slip angle
* combined slip
* vertical load
* camber
* friction coefficient
* relaxation/transient effects

## Tyre thermodynamics

Investigate:

* carcass temperature
* tread/surface temperature
* bulk temperature
* heat generation
* frictional heating
* hysteresis losses
* convection
* conduction
* cooling to road
* cooling to air

## Wear mechanics

Investigate:

* Archard wear
* Reye's hypothesis
* energy-based wear
* frictional work
* frictional power
* contact pressure
* slip distance
* slip velocity
* temperature-dependent wear coefficients
* rubber viscoelasticity
* abrasive wear
* adhesive wear
* thermal degradation

## Statistical models

Investigate:

* Bayesian state-space models
* Kalman filters
* Extended Kalman Filters
* Unscented Kalman Filters
* particle filters
* hierarchical Bayesian models
* Gaussian processes
* mixed-effects models
* survival analysis
* change-point detection
* Bayesian online change-point detection

## Machine learning

Investigate:

* XGBoost
* LightGBM
* Random Forest
* temporal CNN
* LSTM
* GRU
* Transformer
* temporal fusion transformer
* neural state-space models
* physics-informed neural networks
* physics-guided neural networks
* neural ODEs
* differentiable simulators

## Explainability

Investigate:

* SHAP
* Integrated Gradients
* permutation importance
* counterfactual explanations
* feature attribution
* uncertainty decomposition

## Degradation / RUL

Investigate:

* remaining useful life
* health index
* degradation trajectories
* hazard models
* survival probability
* confidence intervals
* conformal prediction

## Motorsport strategy

Investigate:

* pit-stop optimization
* tyre strategy
* undercut/overcut
* stint optimization
* lap-time degradation
* tyre energy
* race simulation
* Monte Carlo strategy

---

# 4. RESEARCH SOURCES TO START FROM

Use authoritative/primary sources wherever possible.

At minimum investigate:

1. Pacejka / Magic Formula literature.
2. Thermo Racing Tyre research.
3. Multiphysical tyre thermal/wear modelling.
4. Recent 2026 Bayesian state-space F1 tyre degradation research.
5. Recent F1 tyre-energy prediction research.
6. Recent tyre wear reviews.
7. Recent energy-based tyre wear models.
8. Recent temperature-dependent Archard models.
9. Recent physics-informed tribology research.
10. Recent sensorless passenger-vehicle tyre-health research.
11. Recent commercial-vehicle tyre-life/RUL research.
12. OpenF1 documentation.
13. FastF1 documentation.
14. FIA regulations and motorsport technical documentation where relevant.

Do not simply collect papers.

Build a research matrix:

| Research area | Existing method | Inputs | Outputs | Strength | Limitation | TyreMind extension |
| ------------- | --------------- | ------ | ------- | -------- | ---------- | ------------------ |

---

# 5. DATA STRATEGY

Use public data first.

Primary motorsport data sources should include:

* OpenF1
* FastF1
* historical race timing
* telemetry
* stint information
* tyre compound
* tyre age
* weather
* track status
* pit stops
* sector/lap times
* position
* speed
* throttle
* brake
* RPM
* gear

Do not assume data exists simply because it would be useful.

Create a formal:

# DATA AVAILABILITY MATRIX

| Variable                    | Public?               | Source        | Resolution | Reliability | Used for            |
| --------------------------- | --------------------- | ------------- | ---------- | ----------- | ------------------- |
| Lap time                    | Yes                   | OpenF1/FastF1 | Lap        | High        | target              |
| Tyre compound               | Yes                   | OpenF1        | Stint      | High        | tyre state          |
| Tyre age                    | Yes                   | OpenF1        | Lap/stint  | High        | degradation         |
| Speed                       | Yes                   | telemetry     | ~3.7Hz     | High        | dynamics            |
| Throttle                    | Yes                   | telemetry     | ~3.7Hz     | High        | energy/load proxy   |
| Brake                       | Yes                   | telemetry     | ~3.7Hz     | High        | thermal/wear proxy  |
| Fuel mass                   | Not directly          | inferred      | —          | Low/medium  | confounder          |
| Tyre temperature            | generally unavailable | —             | —          | —           | latent/synthetic    |
| Tyre pressure               | generally unavailable | —             | —          | —           | latent/synthetic    |
| Physical tread depth        | unavailable           | —             | —          | —           | external validation |
| Internal tyre carcass state | unavailable           | —             | —          | —           | latent model        |

Never hide missing data.

---

# 6. DATA INGESTION LAYER

Build a robust ingestion system.

Requirements:

* API ingestion
* local CSV ingestion
* JSON ingestion
* Parquet support
* caching
* retries
* schema validation
* timestamp normalization
* missing-data detection
* duplicate detection
* data-quality report

Suggested structure:

data/

```
raw/
interim/
processed/
synthetic/
validation/
```

Create:

`DataLoader`

`SessionLoader`

`TelemetryLoader`

`StintLoader`

`WeatherLoader`

`RaceEventLoader`

---

# 7. DATA QUALITY ENGINE

Before any ML model:

Calculate:

* missingness
* duplicate rate
* timestamp gaps
* sampling rate
* impossible values
* outliers
* telemetry discontinuities
* pit-lane anomalies
* safety-car laps
* yellow-flag laps
* invalid laps
* installation laps
* in/out laps
* traffic-heavy laps

Create a:

# DATA QUALITY SCORE

Example:

```text
Data Quality
92.4 / 100

Telemetry completeness     97%
Timing completeness        99%
Stint consistency          100%
Weather completeness       84%
Anomaly rate               3.1%
```

---

# 8. SYNCHRONIZATION ENGINE

Different signals have different timestamps and sampling frequencies.

Build a synchronization layer.

Inputs:

* telemetry timestamps
* lap timestamps
* sector timestamps
* weather timestamps
* stint boundaries
* pit events

Perform:

* interpolation where scientifically acceptable
* forward fill only where justified
* aggregation
* resampling
* lap alignment
* sector alignment

Never blindly interpolate physical signals.

Document every transformation.

---

# 9. FEATURE ENGINEERING

Create multiple classes of features.

## Basic

* tyre age
* compound
* lap number
* stint number
* track position
* lap time
* sector times
* gap
* speed

## Driving

* mean speed
* maximum speed
* throttle %
* braking intensity
* braking duration
* acceleration
* deceleration
* cornering proxies
* speed variance

## Slip/load proxies

Where direct variables are unavailable, construct carefully documented proxies.

Potential features:

* longitudinal acceleration
* lateral acceleration
* speed gradients
* braking events
* cornering severity
* throttle transitions

Do not pretend these are direct tyre slip measurements.

---

# 10. PHYSICS FEATURE ENGINE

Create physically meaningful derived quantities.

Where possible:

### Kinetic energy

E = 1/2 m v²

### Longitudinal acceleration

a_x = dv/dt

### Approximate longitudinal force

F_x ≈ m a_x

### Approximate power

P ≈ F_x v

### Frictional work

W = ∫ F_friction ds

When exact friction force is unavailable, construct a proxy and label it explicitly:

`friction_work_proxy`

### Approximate tyre energy exposure

Integrate a physically motivated proxy over distance/time.

Do not call it "true tyre energy" unless directly measured.

---

# 11. TYRE DYNAMICS MODEL

Implement a simplified tyre-force model.

Use the Pacejka/Magic Formula concept as a physics layer where inputs are available or can be approximated.

General form:

y(x) =
D sin(C atan(Bx - E(Bx - atan(Bx))))

Allow:

* longitudinal force
* lateral force
* combined-slip approximation
* load sensitivity
* friction coefficient

But do NOT spend the entire hackathon attempting to fully identify a production-grade Pacejka parameter set from public F1 data.

Instead create:

# Physics-Informed Tyre Response Layer

that produces physically meaningful latent features.

---

# 12. THERMAL MODEL

Create a reduced-order tyre thermal model.

Use a state such as:

T_tire(t)

with:

dT/dt =
Q_generation
------------

## Q_track

## Q_air

Q_internal

A simplified implementation may be:

dT/dt =
(alpha * friction_power)
------------------------

## (beta * (T - T_track))

(gamma * (T - T_air))

Where alpha, beta and gamma are calibrated/learned parameters.

Create:

* estimated surface temperature
* estimated bulk temperature
* thermal stress
* cooling rate
* heating rate
* thermal exposure

Important:

These are **estimated states**, not measured temperatures.

---

# 13. WEAR MODEL

Implement a physics-inspired wear model.

Start from Archard-style reasoning:

V ∝ K F_N L / H

and energy-based formulations:

wear ∝ dissipated frictional energy.

Use a generalized formulation:

dW/dt =
K(T, compound, surface)
*
P_contact
*
v_slip

or an energy formulation:

dW/dt =
K_w(T, compound)
*
P_friction

where appropriate.

Make K_w learnable or calibrated.

Add:

* temperature dependence
* compound dependence
* load proxy
* slip proxy
* speed
* distance
* frictional-energy exposure

The purpose is NOT to claim exact physical tread depth.

The purpose is to impose physically meaningful structure on the latent degradation model.

---

# 14. CORE RESEARCH INNOVATION

Now build the most important part.

# CAUSAL / STRUCTURAL DECOMPOSITION ENGINE

Observed performance should be represented approximately as:

Observed Lap Time =
Baseline Car/Driver Pace
+
Tyre Effect
+
Fuel Effect
+
Traffic Effect
+
Track Evolution
+
Weather Effect
+
Energy Deployment
+
Driver Variation
+
Random Error

The exact formulation should be validated by you before implementation.

Create separate latent effects.

Example:

```text
Observed slowdown: +0.420 s

Tyre degradation       +0.117 s
Traffic                +0.221 s
Fuel                   -0.041 s
Track evolution        -0.029 s
Driver variation       +0.064 s
Energy deployment      +0.052 s
Residual               +0.036 s
```

Most importantly:

# Do not simply use SHAP and call it causal.

SHAP gives feature attribution, not causality.

If the system claims causal decomposition, implement a structural/statistical framework appropriate to the data and clearly label causal assumptions.

Possible approaches:

* hierarchical regression
* mixed-effects model
* Bayesian state-space model
* latent variable model
* dynamic linear model
* Kalman filter
* particle filter
* causal graph assumptions
* orthogonalization / residualization
* double machine learning where appropriate
* counterfactual model

You must evaluate which is actually identifiable from the data.

---

# 15. BASELINE MODELS

Build multiple models.

## Model 1 — naive degradation

Simple:

lap_time ~ tyre_age

This becomes the baseline.

## Model 2 — multivariate statistical model

lap_time ~ tyre_age + compound + weather + track + traffic + driver + fuel_proxy

## Model 3 — Gradient boosting

XGBoost or LightGBM.

## Model 4 — State-space model

Latent tyre state evolves over time:

z_t = z_(t-1) + degradation_t + process_noise

Observation:

y_t = f(z_t, confounders) + observation_noise

## Model 5 — Physics-guided model

Combine:

physics-derived features
+
ML residual correction

## Model 6 — optional neural state-space model

Only if time permits.

---

# 16. MODEL ENSEMBLE

Do not automatically assume the neural network is best.

Build a champion/challenger system.

Example:

```text
Naive baseline
      ↓
Statistical model
      ↓
XGBoost
      ↓
Bayesian state-space
      ↓
Physics-guided model
```

Compare them using:

* MAE
* RMSE
* MAPE where appropriate
* R²
* negative log likelihood
* CRPS
* calibration
* interval coverage
* rolling-origin validation

---

# 17. TIME-AWARE VALIDATION

NEVER use random train/test split across laps if that leaks future information.

Use:

* chronological split
* rolling-origin cross-validation
* race-level holdout
* driver-level holdout
* circuit-level holdout if possible

Example:

Train:

2023–2025

Validate:

early 2026

Or:

Train races 1–15

Test races 16–19

Make leakage prevention a first-class component.

---

# 18. GENERALIZATION TESTS

Test:

### Same driver, new race

### New driver, known track

### New track

### New compound

### Different weather

### Different traffic conditions

### Different race phase

Report degradation in performance.

This is much more impressive than showing one cherry-picked race.

---

# 19. LATENT TYRE STATE

Create:

# TYRE HEALTH INDEX

Example:

```text
TYRE HEALTH

76.4 / 100
```

But explain what this means.

Potential state vector:

```text
x_t =

[performance_state,
 thermal_state,
 degradation_rate,
 wear_state,
 cliff_risk]
```

Estimate:

```text
performance_state
degradation_rate
thermal_state
```

and optionally:

```text
wear_proxy
```

Use a probabilistic state estimator.

---

# 20. UNCERTAINTY ENGINE

This is essential.

Never output:

> Degradation = 0.082 sec/lap

without uncertainty.

Output:

```text
Estimated degradation:
0.082 s/lap

95% credible interval:
0.061 – 0.104 s/lap

Confidence:
87%
```

Use appropriate methods such as:

* Bayesian posterior intervals
* bootstrap
* conformal prediction
* ensemble uncertainty
* Monte Carlo simulation

Distinguish:

### Aleatoric uncertainty

Noise inherent in the system.

### Epistemic uncertainty

Model/data uncertainty.

---

# 21. TYRE CLIFF DETECTION

Create a change-point/cliff detector.

Possible methods:

* Bayesian change-point detection
* CUSUM
* piecewise regression
* spline model
* state-space transition
* survival/hazard model

Output:

```text
TYRE CLIFF RISK

18%

Likely cliff window:
Lap 34–37

Current trend:
Stable

Acceleration:
Low
```

Do not assume every tyre has a sudden cliff.

The system should be capable of:

* gradual degradation
* accelerated degradation
* plateau
* recovery
* thermal-induced performance loss

---

# 22. REMAINING COMPETITIVE LIFE

Estimate:

# Remaining Competitive Life

Not simply "remaining physical life."

Example:

```text
Current tyre age: 21 laps

Competitive life:
7–9 laps

Probability of performance threshold breach:

5 laps: 4%
7 laps: 19%
9 laps: 47%
11 laps: 81%
```

Define the threshold explicitly.

For example:

> Performance degradation exceeding X seconds/lap relative to baseline.

Allow threshold customization.

---

# 23. COUNTERFACTUAL ENGINE

This should be one of the signature features.

User can ask:

### What if traffic were removed?

```text
Actual:
91.42 s

Estimated clean-air:
91.19 s

Traffic penalty:
+0.23 s
```

### What if tyre were fresh?

```text
Actual:
91.42

Estimated fresh-tyre equivalent:
90.96

Tyre penalty:
+0.46 s
```

### What if tyre age were +3 laps?

Predict future.

### What if pit now?

### What if pit in 2 laps?

### What if stay out?

Every counterfactual must clearly indicate:

> This is a model-based estimate, not observed reality.

---

# 24. RACE STRATEGY SIMULATOR

Create a Monte Carlo race simulator.

Inputs:

* current lap
* tyre state
* degradation distribution
* pit loss
* pit strategy
* traffic
* track evolution
* expected pace
* uncertainty

Simulate thousands of possible race futures.

Example:

```text
Strategy             Expected Finish

Pit now               P6.1
Pit +2                 P4.7
Pit +4                 P5.3
Stay out               P7.2
```

Also calculate:

* probability of finishing in each position
* expected race time
* variance
* downside risk
* upside potential

---

# 25. DECISION ENGINE

Convert predictions into action.

Example:

```text
RECOMMENDATION

PIT IN 2 LAPS

Expected gain:
+1.4 positions

Probability:
72%

Risk:
Medium

Primary reason:
Current pace loss is more traffic-driven than tyre-driven.

Secondary reason:
Tyre cliff probability rises sharply after lap 35.

Confidence:
84%
```

The system must explain:

# WHY?

Never make unexplained recommendations.

---

# 26. STRATEGY OBJECTIVE FUNCTION

Define a transparent objective.

For example:

Minimize:

Expected total race time

or maximize:

Expected finishing position

subject to:

* tyre constraints
* pit-stop constraints
* available compounds
* stint constraints
* uncertainty
* race rules

Use:

* Monte Carlo
* dynamic programming
* constrained optimization
* Bayesian decision theory

Choose the simplest robust method that can be implemented in the available time.

---

# 27. POST-RACE VALIDATION ENGINE

This is mandatory.

Compare:

```text
PREDICTED

degradation:
0.082 s/lap

cliff:
lap 35

pit window:
31–33

expected position:
P4
```

against:

```text
ACTUAL

degradation:
0.079 s/lap

cliff:
lap 35

pit:
lap 32

finish:
P4
```

Calculate:

* degradation MAE
* degradation bias
* forecast error
* interval coverage
* pit-window accuracy
* strategy regret
* position prediction error

Create a:

# MODEL VALIDATION SCORE

---

# 28. STRATEGY REGRET

A very valuable metric.

If the recommended strategy was not selected, calculate:

> How much performance was theoretically lost?

Example:

```text
Recommended:
Pit Lap 32

Actual:
Pit Lap 35

Estimated strategy regret:
+3.8 seconds
```

This converts model accuracy into business value.

---

# 29. BUSINESS VALUE ENGINE

Do not just report ML metrics.

Report:

### Time saved

### Race-position improvement

### Expected lap-time gain

### Reduced unnecessary pit stops

### Increased tyre utilization

### Reduced tyre consumption

### Improved testing efficiency

Potential output:

```text
Estimated engineering value

Strategy improvement:
+0.18% race-time efficiency

Potential tyre utilization:
+7.4%

Unnecessary pit decisions avoided:
2

Model confidence:
81%
```

Make these values clearly model-estimated.

---

# 30. DIGITAL TWIN

Build a simplified:

# Tyre Digital Twin

Represent each tyre as a continuously updated state.

```text
TYRE DIGITAL TWIN

Compound: MEDIUM
Age: 21 laps

Health: 76%

Thermal state:
OPTIMAL

Degradation:
0.082 s/lap

Wear proxy:
0.61

Cliff probability:
18%

Remaining competitive life:
7–9 laps

Confidence:
87%
```

Create a timeline:

```text
Lap 1
│
│ grip
│
Lap 10
│
│ thermal build
│
Lap 20
│
│ degradation
│
Lap 30
│
│ cliff probability
▼
```

---

# 31. DIGITAL TWIN STATE TRANSITIONS

The twin should update after every lap.

Conceptually:

x_(t+1) =
f(x_t, u_t, environment_t) + process_noise

where:

x = tyre health state

u = operating conditions

environment = track/weather/etc.

Observation:

y_t =
g(x_t, vehicle_state_t, driver_state_t) + observation_noise

This state-space formulation should be documented.

---

# 32. PHYSICS + ML HYBRID DESIGN

Do not choose between:

> Physics

and

> AI.

Use:

# Physics-guided ML

Architecture:

```text
Telemetry
    ↓
Physics Feature Engine
    ↓
Physical State Estimates
    ↓
ML Residual Model
    ↓
Latent Tyre State
    ↓
Forecast
```

Potential formulation:

Prediction =
PhysicsModel(inputs)
+
MLResidual(inputs)

This allows:

* interpretability
* better extrapolation
* less data dependence
* physical consistency

---

# 33. PHYSICAL CONSISTENCY CONSTRAINTS

Where appropriate enforce:

### Degradation should generally not become negative without explanation.

### Increased severe frictional exposure should not randomly reduce wear.

### Thermal state should evolve continuously.

### Tyre age should not decrease within a stint.

### State estimates should not jump unrealistically.

### Confidence should decrease under extrapolation.

Do not hard-code physically false assumptions.

Allow exceptions such as:

* cooling
* track evolution
* changing operating conditions
* graining recovery
* thermal recovery

---

# 34. ABLATION STUDY

Create a research panel.

Compare:

### Model A

ML only.

### Model B

Physics features + ML.

### Model C

Latent state model.

### Model D

Physics + latent state + ML residual.

Display:

```text
                    RMSE      Calibration

ML only             0.241     71%
Physics + ML        0.208     82%
State-space         0.192     89%
Hybrid model        0.167     94%
```

Use real measured values from your experiment.

Never fabricate results.

---

# 35. CONFUSION / FAILURE ANALYSIS

Create:

# Where TyreMind is Wrong

Examples:

* unusual safety-car period
* wet-to-dry transition
* missing telemetry
* unknown fuel load
* unusual driver behaviour
* unseen circuit
* new tyre compound
* sparse observations

This is extremely important for credibility.

A serious engineering product understands its failure modes.

---

# 36. MODEL CARD

Create a model card.

Include:

* training data
* validation data
* assumptions
* limitations
* known biases
* uncertainty
* intended use
* prohibited use
* extrapolation limits

---

# 37. MULTI-INDUSTRY EXTENSION

Now extend the architecture beyond motorsport.

The abstract problem is:

> **Infer hidden degradation/health from noisy operational signals.**

This applies to:

## 1. Passenger vehicle tyres

Inputs:

* wheel speed
* acceleration
* braking
* steering
* mileage
* pressure if available
* temperature
* vehicle load

Outputs:

* tyre health
* wear estimate
* RUL
* anomaly detection

## 2. Commercial fleets

Inputs:

* telematics
* mileage
* route
* load
* speed
* braking
* temperature

Outputs:

* tyre health
* replacement timing
* retread suitability
* fleet cost optimization

## 3. EVs

Inputs:

* torque
* acceleration
* regenerative braking
* battery power
* vehicle mass
* temperature

Outputs:

* tyre stress
* degradation
* efficiency impact

## 4. Tyre manufacturers

Inputs:

* test telemetry
* compound
* load
* temperature
* friction
* wear

Outputs:

* compound comparison
* wear modelling
* product development

## 5. Automotive testing

Inputs:

* proving-ground telemetry
* vehicle dynamics
* thermal state
* tyre state

Outputs:

* durability
* tyre characterization
* performance prediction

---

# 38. INDUSTRY-AGNOSTIC CORE

Architect the backend around:

```text
Asset
    ↓
Operational Context
    ↓
Observed Signals
    ↓
Physics Layer
    ↓
Latent Health State
    ↓
Degradation Model
    ↓
Forecast
    ↓
Decision
```

Do NOT hard-code everything around Formula 1.

Create an abstraction:

```python
AssetProfile
```

with:

* asset type
* physical parameters
* operating limits
* sensor availability
* degradation model
* environment model

Then:

```text
F1 TyreProfile
PassengerTyreProfile
TruckTyreProfile
EVTyreProfile
IndustrialWearProfile
```

This is important for future commercialization.

---

# 39. PRODUCT ARCHITECTURE

Recommended stack:

## Frontend

* Next.js / React
* TypeScript
* Tailwind
* shadcn/ui
* Recharts / Plotly / ECharts
* Framer Motion where useful

## Backend

* Python
* FastAPI
* Pydantic
* NumPy
* Pandas / Polars
* SciPy
* scikit-learn
* XGBoost / LightGBM
* PyMC / Stan if feasible
* PyTorch if neural models are justified

## Data

* Parquet
* DuckDB
* SQLite/PostgreSQL

Prefer DuckDB + Parquet for prototype analytics.

## Visualization

Use:

* Plotly
* ECharts
* SVG
* Canvas where useful

## Simulation

* NumPy
* SciPy
* custom Monte Carlo engine

## ML tracking

If practical:

* MLflow

Do not add infrastructure simply for the sake of looking sophisticated.

---

# 40. API DESIGN

Create endpoints such as:

```text
POST /api/data/upload

POST /api/data/ingest/openf1

GET /api/session/{id}

GET /api/session/{id}/quality

GET /api/tyre/{id}/state

GET /api/tyre/{id}/degradation

GET /api/tyre/{id}/forecast

GET /api/tyre/{id}/uncertainty

POST /api/counterfactual

POST /api/simulate

GET /api/strategy/recommendation

GET /api/validation/{race_id}

GET /api/model/metrics
```

Use typed schemas.

---

# 41. FRONTEND — PRODUCT EXPERIENCE

The frontend should feel like:

# Bloomberg Terminal + F1 Race Engineering + Modern AI Product

NOT:

> college ML dashboard.

Design system:

* dark engineering interface
* strong typography
* restrained accent colors
* dense but readable information
* high-quality charts
* subtle animations
* clear hierarchy

---

# 42. MAIN SCREENS

Build these screens.

## Screen 1 — Executive Overview

Show:

```text
TYREMIND

Race:
Monaco GP

Driver:
#44

Current tyre:
Medium — 21 laps

Tyre Health:
76%

Degradation:
0.082 s/lap

Remaining life:
7–9 laps

Cliff probability:
18%

Recommendation:
STAY OUT
```

---

# 43. SCREEN 2 — LIVE TYRE DIGITAL TWIN

Visualize:

* tyre health
* thermal state
* degradation
* wear proxy
* confidence
* tyre age
* operating regime

Use a visually impressive tyre representation.

---

# 44. SCREEN 3 — "WHY IS THE CAR SLOW?"

This is the signature screen.

Title:

# WHY DID PERFORMANCE DROP?

Show waterfall decomposition.

Example:

```text
Observed:
+0.420 s

Traffic:
+0.221

Tyre:
+0.117

Driver:
+0.064

Energy:
+0.052

Fuel:
-0.041

Track:
-0.029
```

Then:

# KEY INSIGHT

> Only 28% of the observed slowdown is attributable to tyre degradation.

This is the feature judges should remember.

---

# 45. SCREEN 4 — DEGRADATION LAB

Allow users to manipulate:

```text
Tyre age
Temperature
Traffic
Fuel proxy
Track evolution
Driver pace
```

Display:

### Observed performance

vs

### Clean performance

vs

### Counterfactual performance

This should feel like a scientific laboratory.

---

# 46. SCREEN 5 — COUNTERFACTUAL ENGINE

Interface:

```text
WHAT IF?

○ Fresh tyre
○ No traffic
○ +3 laps tyre age
○ Pit now
○ Pit +2
○ Push harder
○ Reduce thermal load
```

Then show the predicted consequence.

---

# 47. SCREEN 6 — STRATEGY SIMULATOR

Display:

```text
SIMULATE 10,000 RACE FUTURES

Strategy      Expected   P(top 5)   Risk

Pit now       P6.1       54%        High
Pit +2        P4.7       72%        Medium
Pit +4        P5.3       63%        Medium
Stay out      P7.2       39%        High
```

Include a distribution visualization.

---

# 48. SCREEN 7 — POST-RACE VALIDATION

Show:

```text
MODEL PREDICTION
       vs
ACTUAL RACE
```

Charts:

* predicted vs actual degradation
* predicted vs actual lap pace
* predicted vs actual cliff
* strategy recommendation vs actual
* cumulative error

---

# 49. SCREEN 8 — MODEL SCIENCE

For technically sophisticated judges.

Show:

### Models

* baseline
* XGBoost
* state-space
* hybrid physics ML

### Metrics

* RMSE
* MAE
* CRPS
* calibration
* coverage

### Ablation

* ML only
* physics only
* hybrid

### Uncertainty

* posterior
* prediction interval

---

# 50. SCREEN 9 — BUSINESS IMPACT

Show:

```text
Potential Engineering Value

Race-time improvement
+0.18%

Tyre utilization
+7.4%

Avoidable strategy errors
2

Estimated engineering review time saved
31%
```

Only show values generated from actual model outputs or clearly labelled scenario estimates.

---

# 51. SCREEN 10 — CROSS-INDUSTRY MODE

This is a major differentiator.

Allow switching:

```text
[ Motorsport ]

[ Passenger Vehicle ]

[ EV Fleet ]

[ Commercial Fleet ]

[ Tyre R&D ]
```

When selected, show the same underlying platform applied to a different asset.

This demonstrates that the fundamental technology is not merely an F1 dashboard.

---

# 52. DEMO MODE

Create a deterministic demo dataset.

DO NOT rely on internet connectivity during the final presentation.

Create:

```text
demo/
    race_session.json
    telemetry.parquet
    tyre_states.json
    simulation_results.json
```

The demo should be reproducible.

---

# 53. THE PERFECT DEMO STORY

Build one carefully engineered scenario.

Start with:

```text
Lap 23

Car:
P5

Tyre:
Medium

Age:
18 laps

Observed slowdown:
+0.42 sec
```

Ask:

# "Is the tyre dying?"

Naive model:

> Yes.

TyreMind:

> No.

Show:

```text
Traffic             +0.22
Tyre degradation    +0.12
Driver variation    +0.06
Energy              +0.05
Fuel                -0.04
Track               -0.03
```

Then say:

> **Only 29% of the slowdown is tyre-driven.**

Then:

# "Should we pit?"

Run simulation.

```text
Pit now:
P6

Pit +2:
P4

Stay out:
P7
```

Recommendation:

# PIT +2

Then show:

> Model confidence: 84%.

Finally:

# POST-RACE

```text
Prediction:
P4

Actual:
P4

Predicted degradation:
0.081

Actual:
0.078
```

This should be the final proof.

---

# 54. SCIENTIFIC DEMONSTRATION

Create a controlled synthetic experiment.

Generate a hidden true degradation process.

Example:

```text
True degradation:
0.080 s/lap
```

Then introduce:

```text
Traffic
Fuel
Track evolution
Driver noise
Energy deployment
Weather
```

The observed lap-time signal becomes heavily confounded.

Compare:

### Naive model

```text
Estimated degradation:
0.143
```

### TyreMind

```text
Estimated degradation:
0.084
```

Then show:

```text
Ground truth:
0.080

Error:
5%
```

IMPORTANT:

This experiment must be clearly labelled:

# Synthetic validation

not real-world validation.

This proves the architecture can recover a hidden signal under controlled confounding.

---

# 55. REAL-DATA VALIDATION

Separately validate on real F1 data.

Do not claim physical tread wear.

Evaluate:

* lap-time prediction
* degradation-state forecasting
* uncertainty calibration
* cross-race generalization
* strategy prediction

Clearly state:

> Public telemetry does not expose all internal tyre variables; TyreMind therefore estimates latent tyre performance/degradation rather than directly measuring tread wear.

This honesty will improve credibility.

---

# 56. MODEL BENCHMARK TABLE

Create automatically generated benchmark results.

Example structure:

| Model           | MAE | RMSE | CRPS | Coverage | Generalization |
| --------------- | --: | ---: | ---: | -------: | -------------: |
| Naive age model |     |      |      |          |                |
| Regression      |     |      |      |          |                |
| XGBoost         |     |      |      |          |                |
| State-space     |     |      |      |          |                |
| Physics-guided  |     |      |      |          |                |
| Hybrid          |     |      |      |          |                |

Never hard-code invented numbers.

---

# 57. RESEARCH TRACEABILITY

Create:

```text
docs/research/
```

with:

```text
literature_review.md
physics_foundation.md
statistical_methods.md
data_sources.md
model_comparison.md
limitations.md
novelty_analysis.md
```

Every important modelling decision should have a short rationale.

Example:

```text
Why state-space?

Because tyre performance is latent and evolves over time.

Why physics layer?

Because wear/thermal behaviour is constrained by physical relationships.

Why ML residual?

Because simplified physics models are incomplete.
```

---

# 58. MODEL CARD

Create:

```text
docs/model_card.md
```

Include:

* intended use
* data
* assumptions
* limitations
* uncertainty
* failure cases
* ethical/safety limitations
* extrapolation limitations

---

# 59. BUSINESS MODEL

Design TyreMind as:

# Tyre Intelligence Infrastructure

Potential products:

### TyreMind Race

For motorsport teams.

### TyreMind Sim

For simulation/esports.

### TyreMind Fleet

For commercial vehicles.

### TyreMind R&D

For tyre manufacturers and automotive testing.

### TyreMind API

For third-party vehicle platforms.

---

# 60. BUSINESS VALUE PROPOSITION

Motorsport:

> Better tyre decisions.

Fleet:

> Fewer unexpected tyre failures and optimized replacement.

Commercial trucking:

> Better casing utilization and retread timing.

EV:

> Understand tyre stress under high torque/regenerative braking.

Tyre manufacturers:

> Faster compound development and test interpretation.

Automotive OEM:

> Vehicle/tyre characterization and durability prediction.

---

# 61. POTENTIAL LONG-TERM TECHNOLOGY MOAT

The long-term moat should NOT be:

> "We use XGBoost."

Instead:

```text
Operational data
        ↓
Physics models
        ↓
Latent-state estimation
        ↓
Asset-specific calibration
        ↓
Historical degradation database
        ↓
Better predictions
        ↓
More customers
        ↓
More data
```

Potential moat:

* proprietary degradation datasets
* asset-specific priors
* physics-informed state models
* calibration engine
* uncertainty calibration
* failure signatures
* cross-vehicle transfer learning

---

# 62. TRANSFER LEARNING

Design the architecture so that a model learned on one asset can initialize another.

Example:

```text
F1 tyre
     ↓
General tyre representation
     ↓
Passenger tyre
     ↓
Truck tyre
```

But do not assume transfer automatically works.

Evaluate domain shift.

Use:

* feature normalization
* domain adaptation
* hierarchical Bayesian priors
* transfer learning

where justified.

---

# 63. SENSORLESS MODE

This could be a future major product.

Create a mode:

# SENSORLESS TYRE HEALTH

Inputs:

* vehicle telemetry
* wheel-speed-derived features
* acceleration
* braking
* steering
* mileage
* environment

Infer:

* tyre health
* degradation
* anomaly
* RUL

Clearly distinguish it from direct TPMS/tread measurement.

---

# 64. DIGITAL-TWIN API

Create a generic schema:

```json
{
  "asset_id": "...",
  "timestamp": "...",
  "health_state": 0.76,
  "degradation_rate": 0.082,
  "remaining_life": {
    "lower": 7,
    "upper": 9
  },
  "uncertainty": 0.13,
  "operating_regime": "high_thermal_load",
  "recommended_action": "stay_out"
}
```

This is the basis of a future SaaS/API.

---

# 65. SAFETY / ENGINEERING GUARDRAILS

Never recommend:

> "Safe to continue driving"

based only on this prototype.

The product should be described as:

> decision support

not:

> autonomous safety certification.

For passenger/commercial applications, physical inspections and certified systems remain authoritative.

---

# 66. AGENT DEVELOPMENT STRATEGY

You are working with four human engineers.

Use AI coding agents as force multipliers.

Divide work into:

### Agent A — Data/Research

Research and implement:

* ingestion
* preprocessing
* features
* synthetic data
* experiments

### Agent B — Physics/ML

Implement:

* physics features
* thermal model
* wear model
* state-space
* ML models
* uncertainty

### Agent C — Backend/Simulation

Implement:

* APIs
* simulation
* strategy engine
* validation
* model serving

### Agent D — Frontend/Product

Implement:

* dashboard
* visualizations
* digital twin
* counterfactual UI
* business UI

Human team owns:

* architecture
* scientific correctness
* integration
* final decisions
* demo
* pitch

---

# 67. GIT STRUCTURE

Use:

```text
tyremind/
│
├── apps/
│   ├── web/
│   └── api/
│
├── ml/
│   ├── baseline/
│   ├── state_space/
│   ├── physics_guided/
│   ├── uncertainty/
│   └── evaluation/
│
├── physics/
│   ├── tyre_dynamics/
│   ├── thermal/
│   ├── wear/
│   └── features/
│
├── simulation/
│   ├── race/
│   ├── counterfactual/
│   └── strategy/
│
├── data/
│   ├── raw/
│   ├── processed/
│   ├── synthetic/
│   └── demo/
│
├── experiments/
│
├── docs/
│   ├── research/
│   ├── model_card/
│   └── architecture/
│
├── tests/
│
└── scripts/
```

---

# 68. TESTING

Implement tests for:

* data ingestion
* timestamp synchronization
* feature calculations
* physics equations
* state updates
* model predictions
* uncertainty
* counterfactuals
* simulation
* API
* frontend critical paths

Include physical sanity tests.

Examples:

```text
Increasing tyre age should not automatically imply
the same degradation under every operating regime.

Increasing thermal load should affect the thermal model.

Increasing traffic should not be attributed entirely to tyre state.

Counterfactual traffic removal should modify traffic contribution,
not tyre age.
```

---

# 69. PERFORMANCE

The demo should run quickly.

Target:

* ingestion: seconds
* feature engineering: seconds
* model inference: milliseconds/seconds
* counterfactual: <5 seconds
* 10,000 race simulations: preferably <5 seconds
* dashboard update: <1 second where possible

Cache expensive computations.

---

# 70. OBSERVABILITY

Create logging:

```text
model_version
dataset_version
experiment_id
prediction_timestamp
input_hash
```

This makes the system feel like engineering software rather than a notebook.

---

# 71. EXPERIMENT TRACKING

Every model experiment should record:

```text
dataset
features
model
hyperparameters
metrics
validation split
random seed
timestamp
```

Do not manually copy metrics.

Generate them automatically.

---

# 72. CONFIGURATION

Avoid hard-coded constants.

Use:

```text
configs/
    default.yaml
    f1.yaml
    passenger_vehicle.yaml
    fleet.yaml
```

Physics parameters and model settings should be configurable.

---

# 73. DOCUMENTATION

Create:

### README

Explain:

* problem
* solution
* architecture
* science
* demo
* business

### Architecture diagram

### Data dictionary

### Research report

### Model card

### Validation report

### Limitations

### Demo guide

---

# 74. WHAT NOT TO DO

Do NOT:

* fabricate accuracy
* fabricate physical tyre measurements
* claim direct causal identification without assumptions
* claim actual tyre temperature when estimated
* claim actual tread depth from F1 telemetry
* use random train/test splits that leak future information
* build a generic chatbot and call it AI race engineering
* add 50 useless dashboard charts
* prioritize UI before scientific core
* spend 10 hours trying to perfectly reproduce a research-grade FE tyre model
* build a huge distributed architecture for a 24-hour prototype
* use an LLM as the numerical prediction engine

LLMs may explain results.

They should not replace the scientific models.

---

# 75. LLM ROLE

Use an LLM only as an orchestration/explanation layer.

Example:

Structured model output:

```json
{
  "tyre_effect": 0.117,
  "traffic_effect": 0.221,
  "confidence": 0.84,
  "recommendation": "stay_out"
}
```

Then the LLM can generate:

> "The current slowdown appears to be dominated by traffic rather than tyre degradation..."

The LLM must NEVER invent numerical values.

---

# 76. AI EXPLANATION CONTRACT

Every generated explanation must reference actual structured model outputs.

Example:

```text
WHY?

Tyre contribution: 0.117 s
Traffic contribution: 0.221 s
Confidence: 84%

Therefore:
traffic currently explains more observed performance loss
than tyre degradation.
```

No hallucinated reasoning.

---

# 77. ADVANCED FEATURE — REGIME DETECTION

Detect operating regimes:

```text
OPTIMAL
THERMALLY HOT
UNDERHEATED
HIGH SLIP
HIGH LOAD
TRAFFIC LIMITED
FUEL LIMITED
DATA UNCERTAIN
```

Use clustering or classification.

This makes the tyre digital twin much more intelligent.

---

# 78. ADVANCED FEATURE — DRIVER NORMALIZATION

Separate:

```text
Car performance
Driver performance
Tyre performance
```

Use:

* hierarchical effects
* driver baseline
* sector-specific pace
* historical driver performance

Do not overclaim causal driver attribution.

---

# 79. ADVANCED FEATURE — TRACK EVOLUTION

Estimate track evolution as a latent/session-level state.

Conceptually:

TrackState_t

evolves slowly:

TrackState_(t+1) =
TrackState_t + small process noise

Use it to avoid attributing all pace improvement to tyres.

---

# 80. ADVANCED FEATURE — FUEL EFFECT

Exact fuel load is generally unavailable publicly.

Therefore:

Option A:

Infer a rough fuel trend from race phase.

Option B:

Use a latent fuel effect.

Option C:

Treat it as an uncertainty component.

Do not present inferred fuel mass as ground truth.

---

# 81. ADVANCED FEATURE — TRAFFIC

Traffic should be estimated using:

* gap
* nearby car positions
* sector/lap interactions
* speed loss relative to clean-air baseline
* track position

Create:

```text
Traffic Index
0–100
```

Then estimate its contribution to pace.

---

# 82. ADVANCED FEATURE — CLEAN-AIR PACE

This is highly valuable.

Estimate:

# Clean-Air Equivalent Pace

Question:

> "How fast would the car have been under equivalent conditions without traffic?"

This is a powerful bridge between causal decomposition and strategy.

---

# 83. ADVANCED FEATURE — TYRE EQUIVALENT LAP TIME

Estimate:

> What would this tyre have done under standardized conditions?

For example:

```text
Observed:
91.42

Clean-air equivalent:
91.19

Fresh-tyre equivalent:
90.96
```

This makes the latent tyre state understandable.

---

# 84. ADVANCED FEATURE — TYRE ENERGY EXPOSURE

Use a physics-derived energy exposure index.

Example:

```text
Cumulative tyre energy exposure
██████████████░░

Current:
82%

Projected cliff:
High after 8 laps
```

This bridges physics and ML.

Do not call it direct tyre energy if it is inferred.

---

# 85. ADVANCED FEATURE — MODEL CONSENSUS

Show:

```text
MODEL CONSENSUS

XGBoost:
0.079

State-space:
0.082

Physics model:
0.085

Hybrid:
0.081

Consensus:
0.082

Confidence:
High
```

This is visually powerful and scientifically useful.

---

# 86. ADVANCED FEATURE — DISAGREEMENT DETECTOR

If models disagree:

```text
⚠ MODEL DISAGREEMENT

State-space:
0.081

XGBoost:
0.123

Physics model:
0.078

Reason:
Operating regime outside training distribution.

Recommendation:
Do not make aggressive strategy decision.
```

This is an excellent trust feature.

---

# 87. ADVANCED FEATURE — OUT-OF-DISTRIBUTION DETECTION

Estimate whether current conditions are outside training distribution.

Examples:

* new circuit
* extreme weather
* unusual tyre compound
* abnormal traffic
* sparse telemetry

Output:

```text
Model applicability:
74%

OOD risk:
Medium
```

This is much more enterprise-grade.

---

# 88. ADVANCED FEATURE — DECISION CONFIDENCE

Separate:

### Prediction confidence

How certain is degradation estimate?

### Decision confidence

How robust is the strategy recommendation?

Example:

```text
Tyre estimate confidence:
91%

Strategy confidence:
73%
```

This distinction is important.

---

# 89. ADVANCED FEATURE — VALUE OF INFORMATION

If time permits, calculate:

> What additional measurement would most reduce uncertainty?

For example:

```text
Current uncertainty:
0.031 s/lap

Most valuable additional signal:
tyre temperature

Expected uncertainty reduction:
21%
```

This creates a bridge toward real-world sensor planning and tyre R&D.

---

# 90. CROSS-INDUSTRY PRODUCT DEMO

At the end, demonstrate:

```text
MOTORSPORT
    ↓
Same engine
    ↓
COMMERCIAL FLEET
```

Change the inputs:

```text
Telemetry
Mileage
Load
Temperature
Braking
Route
```

And output:

```text
Tyre Health:
81%

Estimated remaining life:
18,000 km

High-risk operating pattern:
Repeated heavy braking

Recommended inspection:
Within 1,500 km
```

If this is synthetic, label it:

# Conceptual Fleet Demonstration

Do not pretend it has real fleet validation.

---

# 91. 24-HOUR EXECUTION PRIORITY

Priority 1:

### Data pipeline

Priority 2:

### Baseline + validation

Priority 3:

### Latent tyre state

Priority 4:

### Physics feature layer

Priority 5:

### Causal/confounder decomposition

Priority 6:

### Counterfactual

Priority 7:

### Strategy simulation

Priority 8:

### Dashboard

Priority 9:

### Cross-industry demo

Priority 10:

### Advanced neural models

Do not reverse this order.

---

# 92. 24-HOUR PLAN

## Hour 0–2

Architecture lock.

Research verification.

Data acquisition.

Environment setup.

Repository.

---

## Hour 2–5

Data ingestion.

Cleaning.

Feature pipeline.

Synthetic generator.

---

## Hour 5–8

Baseline models.

Naive degradation.

Regression.

XGBoost.

Evaluation framework.

---

## Hour 8–11

State-space / latent tyre model.

Uncertainty.

Rolling validation.

---

## Hour 11–14

Physics layer.

Thermal model.

Wear/energy proxy.

Physics-guided residual model.

---

## Hour 14–17

Causal decomposition.

Traffic.

Track evolution.

Fuel proxy.

Driver effects.

Clean-air pace.

---

## Hour 17–19

Counterfactual engine.

Race simulation.

Strategy optimization.

---

## Hour 19–22

Frontend.

Digital twin.

Explain-lap screen.

Strategy simulator.

---

## Hour 22–23

Validation.

Bug fixing.

Demo scenario.

---

## Hour 23–24

Pitch.

Screenshots.

Backup demo.

Final testing.

NO major new features after Hour 22.

---

# 93. DEFINITION OF DONE

The prototype is considered successful only if:

### Data

We can ingest real public F1 data.

### Physics

We have physically motivated features/models.

### ML

We have baseline and advanced models.

### Statistics

We have uncertainty and time-aware validation.

### Latent state

We estimate tyre state rather than simply fitting lap time vs tyre age.

### Confounding

We explicitly estimate major confounders.

### Counterfactual

We can ask "what if?"

### Strategy

We can recommend a decision.

### Validation

We can compare predictions with actual race outcomes.

### UI

The product looks like professional engineering software.

### Business

We can explain how the same engine expands into other industries.

---

# 94. JUDGE QUESTIONS YOU MUST PREPARE FOR

Create answers for:

### "Isn't tyre degradation prediction already solved?"

### "What is actually novel?"

### "How is this different from existing F1 strategy tools?"

### "Where does your tyre data come from?"

### "How do you know the degradation is real?"

### "How do you separate causation from correlation?"

### "Why should a team trust your model?"

### "What happens when the model is wrong?"

### "Can this work outside F1?"

### "Who pays for this?"

### "What is your moat?"

### "Why can't a team build this internally?"

### "What happens with a new circuit?"

### "How do you deal with missing telemetry?"

### "Can this work on road cars?"

### "Can tyre manufacturers use it?"

### "What would you build with real team telemetry?"

Prepare technically honest answers.

---

# 95. FINAL PRODUCT POSITIONING

The product should NOT be:

> AI tyre degradation predictor.

It should be:

# TYREMIND

## Causal Tyre Intelligence

> **From noisy vehicle telemetry to a continuously updated tyre digital twin, uncertainty-aware degradation forecast, counterfactual simulation and actionable decisions.**

The conceptual architecture:

```text
                   RAW VEHICLE DATA
                          │
                          ▼
                ┌──────────────────┐
                │ DATA QUALITY      │
                │ & SYNCHRONIZATION │
                └────────┬─────────┘
                         │
                         ▼
                ┌──────────────────┐
                │ PHYSICS ENGINE    │
                │                  │
                │ Dynamics         │
                │ Thermal          │
                │ Energy           │
                │ Wear             │
                └────────┬─────────┘
                         │
                         ▼
                ┌──────────────────┐
                │ CONFOUNDER       │
                │ DECOMPOSITION    │
                │                  │
                │ Traffic          │
                │ Fuel            │
                │ Track            │
                │ Driver           │
                │ Energy           │
                └────────┬─────────┘
                         │
                         ▼
                ┌──────────────────┐
                │ LATENT TYRE      │
                │ STATE ESTIMATOR  │
                └────────┬─────────┘
                         │
             ┌───────────┼────────────┐
             ▼           ▼            ▼
        Degradation   Thermal      Cliff/RUL
           State       State         Risk
             │           │            │
             └───────────┼────────────┘
                         ▼
                ┌──────────────────┐
                │ UNCERTAINTY      │
                │ ENGINE           │
                └────────┬─────────┘
                         │
                         ▼
                ┌──────────────────┐
                │ COUNTERFACTUAL   │
                │ ENGINE           │
                └────────┬─────────┘
                         │
                         ▼
                ┌──────────────────┐
                │ DECISION /       │
                │ STRATEGY ENGINE  │
                └────────┬─────────┘
                         │
                         ▼
                ┌──────────────────┐
                │ POST-RACE        │
                │ VALIDATION       │
                └────────┬─────────┘
                         │
                         ▼
                ┌──────────────────┐
                │ DIGITAL TWIN     │
                │ PLATFORM         │
                └────────┬─────────┘
                         │
             ┌───────────┼────────────┐
             ▼           ▼            ▼
         Motorsport     Fleet       Tyre R&D
             │           │            │
             └───────────┼────────────┘
                         ▼
                  TYRE INTELLIGENCE
                    INFRASTRUCTURE
```

---

# 96. MOST IMPORTANT PRINCIPLE

The product's central intelligence should be:

> **Observed performance ≠ true degradation.**

Everything else should support that idea.

The physics explains what could physically cause the signal.

The statistical model separates latent states.

Machine learning learns nonlinear residual behaviour.

Bayesian/state-space methods quantify uncertainty.

Counterfactuals answer "what would have happened?"

Simulation answers "what should we do?"

Post-race validation answers "were we right?"

The digital twin remembers the evolving asset state.

That is the complete intelligence loop.

---

# 97. YOUR FIRST TASK

DO NOT immediately generate the entire application.

First produce the following files/reports:

```text
01_RESEARCH_AUDIT.md
02_COMPETITOR_ANALYSIS.md
03_DATA_AVAILABILITY.md
04_PHYSICS_FOUNDATION.md
05_STATISTICAL_ARCHITECTURE.md
06_ML_ARCHITECTURE.md
07_SYSTEM_ARCHITECTURE.md
08_NOVELTY_ANALYSIS.md
09_BUSINESS_MODEL.md
10_24_HOUR_BUILD_PLAN.md
11_MODEL_VALIDATION_PLAN.md
12_DEMO_STORY.md
13_RISK_REGISTER.md
```

Then generate:

```text
ARCHITECTURE_DECISION.md
```

containing your final recommended architecture after critically auditing my proposal.

Only after that should you begin implementation.

---

# 98. FINAL INSTRUCTION TO CLAUDE

You are allowed—and explicitly encouraged—to disagree with this specification.

If you find a better:

* mathematical formulation
* statistical method
* physics model
* ML architecture
* uncertainty method
* simulation method
* validation method
* product architecture
* business positioning

replace my proposal.

But every change must answer:

1. Why is it better?
2. Is it implementable in 24 hours?
3. Is it scientifically defensible?
4. Is it supported by research?
5. Does it improve judge impact?
6. Does it improve business value?
7. Does it improve long-term scalability?

Do not optimize merely for number of features.

Optimize for:

# SCIENTIFIC CREDIBILITY

*

# NOVELTY

*

# DEMONSTRABLE INTELLIGENCE

*

# BUSINESS VALUE

*

# 24-HOUR EXECUTABILITY

The final prototype should feel like:

> **A research-grade concept compressed into a polished commercial product prototype.**

Not a hackathon toy.

Begin with the complete research/feasibility audit before writing the production implementation.
