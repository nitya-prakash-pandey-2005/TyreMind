# Demo guide

**One command, no network required:**

```bash
python -m tyremind.serve
```

Warms the cached sessions, serves the dashboard at `http://127.0.0.1:8077`, opens
a browser. Eight sessions (four events × FP2 + race) are committed to the repo as
Parquet, so a fresh clone works with the wifi unplugged.

---

## The seven-minute run

### 1. Open on **Start here**, Monza race (60 s)

Point at the two numbers side by side:

> The standard method — fit a line through lap time against tyre age — says the
> hard tyre degraded at **−0.004 s/lap** on this race. Negative. It says the tyre
> got *faster* the longer it ran.
>
> That is obviously impossible, and it is not a one-off: the naive method comes out
> negative on 3 of the 4 races we analysed, on 4 of 11 compound-stints. The car
> burns fuel, gets lighter, and speeds up by about 0.08 s a lap — which is bigger
> than the tyre's degradation, so the tyre effect is buried under it and comes out
> with the wrong sign.
>
> (If asked: Barcelona is the race where it does not go negative. That is not the
> method working — it is the confounders happening not to swamp the signal there.
> Whether the standard approach gets the sign right is a matter of luck.)

Then TyreMind's number: **+0.074 s/lap** for the hard, **+0.113** for the medium,
correctly ordered, each with an interval.

### 2. Click through to **Why is the car slow** (90 s)

Let the peel-away animation run. Grey line is what the stopwatch saw; each step
removes one cause.

Then read the plain-language panel out loud — it is generated from model output:

> *"Car RIC was 1.78 seconds FASTER on lap 53 than on lap 13, so a stopwatch says
> the tyre is fine. It is not. Strip out the things that changed around the car
> and the hard tyre has actually lost 2.15 seconds over those laps. The car got
> faster because it burned off fuel, and the fuel gain was bigger than the tyre
> loss. Watching lap times alone would miss a degrading tyre completely."*

**This is the moment.** A car that got a second and a half faster while its tyre
was dying. Nothing that reads lap times can see it.

### 3. **Tyre twin** (60 s)

The car diagram. Left tyres carry 55% of the frictional energy at Monza.

> That asymmetry is computed from GPS traces — curvature gives lateral
> acceleration, which throws load to the outside of the car. We never told the
> model which way Monza runs. It worked out *clockwise* from the loading alone,
> and it gets that right on 7 of 8 circuits.

Then the health timeline with its widening uncertainty band, and remaining
competitive life with the applicability column falling off as the projection
reaches past what the session contains.

### 4. **When to pit** (60 s)

Move the lap slider. Show the outcome distributions overlapping.

> Five thousand simulated races per option. We draw a *different* degradation rate
> from the posterior for every simulated race, so this spread is real uncertainty
> about the tyre, not just lap-time noise. Where the curves overlap, the model is
> telling you the choice genuinely does not matter — which is as useful as being
> told it does.

Point at the regret number: stopping five laps late costs **4.0 seconds**. That is
what model accuracy is worth in the only unit a pit wall uses.

### 5. **Live monitor** — press start (60 s)

> This is the same model running forward-only, one lap at a time, with no access
> to the future. **0.22 milliseconds per update.** Cost is flat in session length —
> lap 60 costs what lap 1 cost.
>
> That is why we chose a Kalman filter over MCMC. A sampler has no incremental
> mode; you cannot re-run one every lap on a pit wall.

By the end, the online estimator agrees with the batch fit to 0.0006 s/lap at
worst — two separate implementations of the same model.

### 6. **Does it work** (90 s)

Lead with the identifiability panel — the thing most tools omit.

> Three causes push lap time the same way, so many wrong answers add up to the
> same right total. Only one of the three is resolved by evidence. We say which.

Then the numbers:

- **Ground truth recovery:** 0.0044 s/lap against naive 0.0966. 95.5% better,
  100% interval coverage, 25 sessions.
- **Practice → race:** 0.0518 vs 0.1166 MAE, 90% coverage — and a systematic
  +0.047 s/lap bias we report rather than tune away.

### 7. **Beyond racing** (60 s)

> F1 cannot prove this works, because public F1 data has no measured tyre wear —
> there is nothing to check against. So we ran the *identical* estimator on NASA's
> turbofan benchmark, which does have run-to-failure ground truth.
>
> 26.5-cycle RUL error on 40 of the 100 FD001 test engines, 32% predicted early.
> (If asked: published figures use all 100, so this is indicative, not like-for-like.)
> Purpose-built deep models
> get 12–20 on that dataset, so we are not competitive — but we are a tyre model
> pointed at jet engines with no retuning, and it works.

Close on the three asset profiles, with commercial fleet explicitly labelled
**architecture only**.

---

## If asked to prove nothing is hard-coded

```bash
python experiments/exp01_ground_truth_recovery.py --n-seeds 5
```

Runs in about a minute and writes `experiments/results/*.json`, which is what the
dashboard reads. No number in the product is typed by hand.

---

## Backup plan

| Failure | Response |
|---|---|
| No network | Everything already runs offline. Say so; it is a feature. |
| Server will not start | `pytest -q` (81 tests) shows the science is intact; walk the code. |
| Browser trouble | `/docs` gives the interactive API. Every endpoint returns real numbers. |
| Asked something not built | Say so. `13_LIMITATIONS_AND_FAILURE_MODES.md` lists what was cut and why. |

---

## The sentence to leave them with

> Everyone in this field validates lap-time prediction. Nobody checks whether the
> *reason* is right — and that is the actual question, because many wrong
> decompositions give the same right total. We built the test, ran it, and
> reported what it said, including where we lose.
