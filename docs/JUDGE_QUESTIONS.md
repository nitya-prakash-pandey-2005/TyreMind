# Anticipated questions

Short, honest answers. Where the honest answer is "we cannot", it says so.

---

### "Isn't tyre degradation prediction already solved?"

Prediction largely is. A Bayesian state-space model for F1 tyre degradation was
published in November 2025 (arXiv:2512.00640), and Pitwall ran a calibrated race
simulator live at two 2026 Grands Prix.

What is *not* solved is attribution. Every one of those models is validated on
lap-time prediction error, and that metric cannot detect the failure that
matters: fuel, track evolution and degradation all move lap time smoothly with
lap number, so many wrong decompositions sum to the same right total. A model can
predict lap times perfectly while blaming the wrong cause. We built the test for
that and ran it.

---

### "What is actually novel?"

Three things, in order of how much they matter:

1. **The attribution test.** Synthetic sessions with a known hidden rate, buried
   under realistic confounding, scored per cause. Nobody in this literature does
   this.
2. **A collinearity we characterise and resolve** — track evolution against a
   uniform shift in degradation. They differ by a constant within a run, which
   the run intercept absorbs exactly.
3. **Whole-field pooling using pit stagger** as the natural experiment that
   separates tyre age from session lap. A single-driver model cannot use it.

Not novel: the Kalman filter, Archard wear, Monte Carlo race simulation, LLM
narration. `08_NOVELTY_ANALYSIS.md` states plainly which is which.

---

### "How do you know the degradation you report is real?"

On real F1 data, we do not — and neither does anyone else, because public
telemetry contains no measured tyre wear. That is why we do three other things:

- **Synthetic ground truth.** We set the true rate, hide it, and measure
  recovery: 0.0044 s/lap error against the standard method's 0.0966.
- **Practice → race.** Estimate from Friday, score against Sunday, no leakage,
  across 42 events and 94 compound comparisons in 2023, 2024 and 2025: 0.0807 MAE
  against naive 0.1471, and 95% coverage once conformally calibrated (76%
  before). On five events this read 0.0518; the small sample was flattering, and
  we quote the number that replicated.
- **A second asset class.** NASA C-MAPSS turbofans, which *do* have run-to-failure
  ground truth. Same estimator, no tyre-specific code.

---

### "How do you separate causation from correlation?"

We do not claim causal identification. The decomposition is exact arithmetic on
an assumed structural model, and two of its three identifying assumptions are
priors rather than evidence. The accurate phrase is "structural attribution under
the stated model", and that is what the UI says.

What we *can* do is quantify what the assumptions cost: a full-standard-deviation
error in the fuel prior — our largest assumption — moves the answer by about
0.02 s/lap, which is still 4.9× better than the naive method.

---

### "Why should a team trust this?"

They should trust it exactly as far as the intervals say. Specifically:

- Every number carries a credible interval, drawn to scale rather than printed as
  a footnote.
- The applicability score falls when a query runs past what the session contains,
  and the UI says the model is extrapolating.
- Filtered and smoothed estimates are never conflated — we do not show a
  retrospective number as if it had been available live.
- We report where we lose. LightGBM beats us on lap-time prediction, and we say
  so in the product.

---

### "What happens when the model is wrong?"

`13_LIMITATIONS_AND_FAILURE_MODES.md` lists nine failure modes with what the
product does about each. The short version: wet running is excluded, short stints
are dropped, extrapolation is flagged, safety-car laps are removed and counted,
and a run of large innovations in the live monitor means the model is being
surprised.

There is also a known systematic bias — practice over-predicts race degradation
by 0.047 s/lap — which we report rather than tune away, because a bias that is
understood is more useful than one that has been hidden.

---

### "Can this work outside Formula 1?"

Validated on one other asset class, with real ground truth: NASA C-MAPSS
turbofans, 56-cycle RUL error, using the identical estimator with no tyre-specific
code. Purpose-built models on that dataset reach 12–20 cycles, so we are
demonstrating transfer, not competitiveness.

For road and commercial vehicles: the architecture is there and is **not
validated**, because no public dataset pairs tyre tread depth with telematics. We
searched; it does not exist. The product labels that section "architecture only".

---

### "Why can't a team build this internally?"

They can, and a top team probably has something comparable — with better inputs
than ours, since they have real fuel telemetry and tyre sensors.

The defensible position is not the model. It is the identifiability analysis and
the validation protocol: knowing *which* quantities are recoverable from which
data, and having a test that catches a confidently wrong attribution. That
transfers to any team, any series, and any asset class with the same structure.

---

### "What happens with a new circuit?"

Degradation estimation works — it needs only timing data and enough run stagger.
The per-corner physics needs position telemetry to have been analysed for that
circuit; if it has not, the twin says so rather than showing an even split as if
it were a result.

**Predicting a circuit before anyone has run there is a different problem, and we
tested it and failed.** Leave-one-*circuit*-out across 193 stint estimates at 26
venues, so the model has never seen the venue in any form — exactly Thursday's
situation. Circuit geometry **does not beat the label mean** (−3.3%, p 0.22), and
adding compound identity on top is significantly worse.

On two seasons we put this more strongly — geometry then measured −7.5% at
p 0.010, significantly harmful — and four seasons walked it back to "does
nothing". The product conclusion is the same either way, since a family that
does not earn its place is excluded regardless, but the claims differ and only
the weaker one holds.

The mechanism is worth the answer. **Pirelli nominates to equalise** — harder
rubber for abrasive venues, softer for gentle ones — so the weekend's
HARD/MEDIUM/SOFT label already carries circuit severity *by design*. Geometry
laid on top re-fits severity that is already spoken for, and overfits.

So the honest position is that pre-event prediction is the right product goal
and circuit geometry is not the route to it. We would rather say that than ship
a feature that looks principled and is not.

---

### "How do you deal with missing telemetry?"

Timing data alone is enough for the core estimate; telemetry is only needed for
the physics layer. Missing laps are handled natively — the filter propagates the
state through a gap without an observation, and a tyre in the garage correctly
does not age.

---

### "What would you build with real team telemetry?"

Replace the fuel prior with measured fuel mass. That single change removes the
largest assumption in the method and would tighten every published interval. It
is the top-ranked item in the value-of-information panel, at an estimated 30%
uncertainty reduction.

Second: real tyre temperature, which would let the model separate thermal
degradation from mechanical wear instead of absorbing both into one rate.

---

### "Who pays for this, and what is the moat?"

Motorsport is the proving ground, not the market — a handful of teams, all with
internal capability. The transferable asset is the estimator plus the
identifiability method, which applies to any degrading asset observed through a
confounded signal: commercial fleets, tyre manufacturers doing test
interpretation, industrial predictive maintenance.

The honest caveat is that we have validated transfer to *one* other asset class,
on a public benchmark. That is evidence the approach generalises. It is not a
commercial product.

---

### "Isn't a neural network going to beat this?"

On lap-time prediction, **two simpler models already beat us** — and over twenty
races the better of them is plain pooled regression, CRPS 0.396, with LightGBM
second at 0.469 and our state-space model third at 0.645. We report it.

Neither of the two above us has a parameter meaning "degradation rate", so there
is nothing to hand an engineer and nothing to carry from Friday to Sunday. They
are not last in the degradation table; they are **absent from it**, because there
is nothing in them to score. On that task we are first, 0.0041 s/lap against
pooled regression's 0.0068 and the naive method's 0.0748.

LightGBM is also badly overconfident (62% coverage on nominal 95%). On
extrapolation, ours falls by 0.432 as each fold forecasts further past its
training window — by a wide margin the largest fall of any rung. On the previous
corpus the lap-time leader's error *grew* (+0.262) and we said so; across four
seasons it is essentially flat (+0.031), so that half of the argument no longer
holds and is withdrawn rather than restated.

That last number is the argument for encoding physics rather than learning it.

---

### "Why is your interval coverage 100% when you designed for 95%?"

On the synthetic benchmark, because the intervals are slightly conservative —
wider than strictly necessary. For decision support that is the safer direction
to err, but it is a miscalibration and we report it as one rather than as a
perfect score.

**The practice→race intervals erred the other way, which is the dangerous one.**
They covered **76% of 94 comparisons while labelled 95%**. An earlier version of
our own documents reported 90% on ten comparisons; that sample was simply too
small to have detected the problem.

The cause is not a broken filter. The posterior standard deviation correctly
answers *how well do these practice laps pin down the practice rate* — which is
not the question the number gets used for. Nothing about the transfer to Sunday,
fuel and track state and driving style and tyre management, is inside it. The
interval was honest and answering the wrong question.

We fixed it with **split conformal prediction**, which gives distribution-free
finite-sample coverage without assuming the errors are Gaussian or the model
correctly specified. It lands on **95%**, at the honest cost of a wider interval
(±0.25 vs ±0.14 s/lap). Calibration folds are grouped by *event*, because two
compounds at one Grand Prix share a track, a weather window and a fuel
correction and are not exchangeable with each other.

It is shipped, not just measured: `tyremind.models.conformal`, served by the API
as `race_projection`, and visible on each compound card in the dashboard.
