<div align="center">

# TYREMIND

### Causal Tyre Intelligence

**Observed performance is not the same thing as tyre degradation.**

![Python](https://img.shields.io/badge/python-3.11%20%E2%80%93%203.13-1d7d9c?style=flat-square)
![Tests](https://img.shields.io/badge/tests-326%20passing-1f8a5c?style=flat-square)
![Offline](https://img.shields.io/badge/runs-fully%20offline-c4501f?style=flat-square)
![Licence](https://img.shields.io/badge/licence-MIT-6b7780?style=flat-square)

```
pip install -r requirements.txt && pip install -e . && python -m tyremind.serve
```

Python 3.11–3.13 &nbsp;·&nbsp; no Node, no network, no API key

[**Run it**](#run-it) &nbsp;·&nbsp;
[Results](#results) &nbsp;·&nbsp;
[Why this is hard](#why-this-is-hard) &nbsp;·&nbsp;
[Technical dossier (PDF)](docs/pitch/TyreMind_Technical_Dossier.pdf) &nbsp;·&nbsp;
[Pitch deck (PDF)](docs/pitch/TyreMind_Pitch_Deck.pdf)

</div>

<br>

![TyreMind dashboard](docs/images/overview.png)

---

## The problem, in one example

Here is a real stint from the 2024 Italian Grand Prix — Ricciardo, 41 laps on the
hard tyre, laps 13 to 53. Between those two laps the car got **1.78 seconds
faster**.

A stopwatch says the tyre is fine. It is not. Across that stint the hard tyre
lost **2.15 seconds** of performance — the car got quicker because it burned off
3.27 seconds of fuel weight, and the fuel gain was larger than the tyre loss.
(The set was not new: it went on with 3 laps already on it, which the model
accounts for and a lap-count-from-zero reading would not.)

Read lap times alone and you miss a dying tyre completely.

That is not an edge case. Fit the standard method — a straight line through lap
time against tyre age — and it reports **negative degradation**: tyres apparently
getting faster the longer they run. Counted across **every dry race in the corpus,
61 of them**, it does this in **70% of races** and in **51% of the 164
compound-stints** — more often than not, at the level of an individual compound.

The cause is not subtlety. Fuel burn-off is worth about 0.08 s/lap and is simply
bigger than the effect being measured. Where the standard method happens to get
the sign right, it is because the confounders did not swamp the signal that
weekend, which is the point: **it is a matter of luck, and it is not on your side
half the time.**

**TyreMind estimates the latent performance state of a tyre underneath that
confounded observation** — separating degradation from fuel burn-off, track
evolution and traffic, and reporting how sure it is about each.

---

## What it looks like

<table>
<tr>
<td width="50%">

![Confounders peeled away](docs/images/explain.png)

**Why is the car slow** — confounders lifted off a stint one at a time. The
dashed line is the model's degradation estimate; the solid line is the measured
lap times with fuel, track and traffic removed.

</td>
<td width="50%">

![3D circuit coloured by tyre load](docs/images/circuit.png)

**Where it wears** — the real racing line in 3D, coloured by the frictional load
the physics layer computes at each point, with the friction envelope beside it.

</td>
</tr>
<tr>
<td width="50%">

![Pit strategy](docs/images/strategy.png)

**When to pit** — five thousand simulated races per option, with the degradation
rate resampled from its posterior for every race, plus the full pit-lap sweep.

</td>
<td width="50%">

**Six more screens.** A tyre twin with per-corner condition and what-if
counterfactuals. A live monitor streaming lap by lap with no access to future
data, where the interval visibly collapses as evidence arrives. The validation
evidence. A retrieval search over the project's own research corpus. And the same
estimator pointed at NASA turbofan engines.

Dark and light themes. Every estimate drawn as an interval, never printed as a
bare number with a plus-or-minus appended.

</td>
</tr>
</table>

---

## Results

Every figure below is produced by a script in `experiments/` and read from
`experiments/results/*.json`. Nothing here is typed by hand.

The evidence base is **113 sessions and 61,396 laps** across 2024, 2023 and 2022
(`scripts/build_corpus.py` builds it into `data/season/`, which is gitignored
because it is large and rebuildable). The eight sessions committed under
`data/demo/` are a presentation set chosen to span circuit types, so a fresh
clone runs offline — they are not the evidence.

Five hypotheses were tested and **not** supported, and are reported as such
rather than dropped: compound identity ([exp08](experiments/exp08_compound_identity.py)),
circuit geometry transfer ([exp09](experiments/exp09_circuit_transfer.py)), the
practice-to-race temperature gap and traffic ([exp10](experiments/exp10_bias_mechanism.py)),
stint-depth matching ([exp11](experiments/exp11_depth_matched.py)), and the
driver effect ([exp15](experiments/exp15_driver_effect.py)).

### Is there a driver effect?

Everyone in the paddock will tell you driving style changes tyre life. Across
**1,152 driver-races, 31 drivers, 61 races**, that splits into two answers that
disagree — and the disagreement is the finding.

**Reliable, but only just.** Split the corpus into random halves and the ranking
of drivers reproduces: split-half r = **+0.24**, with a 5th percentile of
**+0.008** across 500 random partitions. That lower bound clears zero and does
not clear it comfortably. On an earlier corpus it read +0.36 with a 5th
percentile of +0.09; correcting the fuel counter and adding 2025 weakened it, and
the weaker figure is the one quoted.

**Not useful.** Predicting a driver's deviation at a held-out race from their own
history scores 0.0248 s/lap against **0.0246 for just predicting the field
mean** — very slightly worse. The spread of driver means is 0.0113 s/lap while
the race-to-race scatter is 0.0248: the trait is real and about *half the size of
the noise it has to be read through*.

**And it may not be the driver.** Driver and car are perfectly confounded.
Teammates share a car, so the within-team difference removes it — and that
contrast is not stable. Even the reliable part cannot be pinned on the driver
rather than the machinery.

A per-driver term would add a parameter, a maintenance burden and a persuasive
story, and would not improve a forecast. **Not built, deliberately.**

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

2024 and 2023, **27 events, 62 compound comparisons**. No race data reaches the
practice fit:

| | Naive | **TyreMind** |
|---|---:|---:|
| MAE | 0.1440 s/lap | **0.0858 s/lap** |
| 95% coverage, as fitted | — | 76% |
| 95% coverage, calibrated | — | **95%** |

**40% error reduction.** This number got *worse* as the evidence grew, and that
is worth stating plainly: on five events it was 0.0518 s/lap. The small sample
was flattering. What held is the comparison that matters — the naive method
worsened too, 0.1166 → 0.1440, and the gap between the two stayed at roughly
40%. A claim that survives a six-fold increase in evidence is worth more than a
prettier one resting on ten comparisons.

Three events are excluded because they were not dry-degradation sessions at all
(Canada 2024 ran 67% of its race laps on wet rubber). The rule is applied to
both sides of every comparison and was calibrated, not chosen to flatter.

There is a **systematic +0.043 s/lap bias** — practice over-predicts race
degradation in 42 of 62 comparisons. Nine candidate explanations were tested
with a multiplicity correction. **One survives** — mean practice stint length
(ρ +0.39, p 0.002) — and a second, pit stops per driver, misses by a hair
(p 0.0113 against a threshold of 0.0111). The causal story built on them was
then tested and **refuted** ([exp11](experiments/exp11_depth_matched.py)).
So the bias is not explained away. It is corrected as a measured offset, and the
interval is widened by conformal calibration until it covers what it claims.

### Does the physics compute what it claims?

The pipeline goes GPS trace → curvature → lateral acceleration → per-corner load.
It is never told which way a circuit runs. A clockwise circuit must load its
*left* tyres more.

**7 of 8 circuits recovered correctly.** Clockwise circuits show 21–35% left-turn
energy; anti-clockwise 60–75%. Austin misses at 46.2% and is reported as a miss.

### Does it work on something that is not a tyre?

Public F1 data has no measured tyre wear, so motorsport cannot supply ground
truth. NASA's C-MAPSS turbofan benchmark does.

**Same estimator, no tyre-specific code:** **22.7-cycle RUL error**, 44%
predicted early — the safe direction. Purpose-built deep models reach 12–20 on
that dataset, so this demonstrates transfer, not competitiveness.

It is scored on **all 100 engines in the FD001 test set**, which is what
published figures are quoted on, so the comparison is now like-for-like. An
earlier version scored 40 because the estimator fits the test set jointly and a
100-engine run would not converge; fitting in batches removes the limit, and the
result *improved* — 26.5 → 22.7 cycles — because the 40-engine subset was the
harder end of the set, not because batching flatters it.

### Where we lose

Six models, **twenty races**, identical expanding-window chronological folds:

| Model | CRPS (lap time) | Coverage | Bias drift | Degradation rate MAE |
|---|---:|---:|---:|---:|
| Pooled regression | **0.487** | 84% | +0.262 | 0.0068 |
| LightGBM | 0.537 | 62% | −0.021 | *no such parameter* |
| TyreMind | 0.757 | 81% | **−0.452** | **0.0041** |
| Naive | 0.974 | 76% | +0.398 | 0.0748 |

**Two models predict lap times better than we do**, and on twenty races the
better of them is plain pooled regression, not LightGBM. Both beat us on CRPS.
We are third.

That is the whole argument, stated against ourselves. **Neither of them has a
parameter that means "degradation rate" at all** — LightGBM and the neural
network cannot be scored on the degradation task because there is nothing in
them to score. On the task that matters, recovering a known hidden rate,
TyreMind is first at 0.0041 s/lap against pooled regression's 0.0068 and the
naive method's 0.0748. *Predicting lap times well is not the same as
understanding the tyre.*

Bias drift measures how much a model's error grows as it forecasts further past
its training window. TyreMind's shrinks the most of any usable rung (−0.452)
while pooled regression's grows (+0.262). An earlier version of this README
claimed TyreMind was **the only** model whose error does not grow; at twenty
races that is no longer true — LightGBM is roughly flat at −0.021 — and the
claim has been corrected rather than quietly restated.

**Our own lap-time intervals were undercovered too.** Across **66,606 held-out
laps**, every rung undercovers and none reaches 85%: LightGBM 63%, the neural
network 71%, naive 76%, TyreMind 81%, pooled regression 84%. Ours is neither the
worst nor the best, and that is the opposite direction of error from the
conservative 100% we report on the degradation task.

**Fixed** ([exp13](experiments/exp13_lap_time_calibration.py)). Split conformal
is the wrong tool here — it needs the calibration set to be exchangeable with
the test point, and lap times inside a race are the opposite of that, because the
car burns fuel, the track rubbers in and a safety car rearranges everything.
Adaptive Conformal Inference drops the assumption instead of hoping it holds,
retuning the working miss-rate after every lap:

| | Coverage | Median width |
|---|---:|---:|
| Gaussian, as reported | 75.1% | — |
| Split conformal | 89.3% | 6.11 s |
| **Adaptive conformal** | **95.4%** | 7.12 s |

It lifts LightGBM from 63% to 95% at only 3.5 s of width — its lap-time
*predictions* were always fine, its *sd* was wrong.

**And the shape is right, not just the size**
([exp16](experiments/exp16_calibration_shape.py)). A model can hit 95% exactly
while being far too confident in the middle and too timid in the tails. The PIT
histogram — where the truth lands inside its own predicted distribution — comes
out **U-shaped for five of the six rungs**, which is overconfidence seen directly
rather than inferred.

The sixth is ours, and it fails the *opposite* way: the state-space model is
**hump-shaped — intervals too wide, underconfident**. That is the same
conservatism as the 100% coverage on the synthetic benchmark, and it is the safer
direction to err, but it is still a miscalibration.

Sweeping the nominal level from 50% to 99%, the Gaussian sits a mean 0.195 from
the reliability diagonal and adaptive conformal sits 0.006 away: **33× closer, at
every level and not only at 95%.** The live monitor now carries the same machinery and reports the
coverage it has actually achieved, so the claimed 95% is auditable in flight
rather than after the race.

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
error is 0.0199 s/lap — still 4.9× better than naive. Shifting the track prior by
±1sd moves the bias monotonically (+0.0000, +0.0043, +0.0091), about 0.0045 s/lap
per prior standard deviation.

That experiment re-fits eight variants and so runs on 4 sessions rather than the
headline benchmark's 25, which is why its baseline reads 0.0047 against the 0.0044
above — different samples, not different answers.

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

## Run it

**Python 3.11 to 3.13, and nothing else.** The dashboard ships built, eight
sessions ship cached, and no step needs a network connection. Verified on a clean
3.11 install; 3.14 is excluded because numpy has no wheel for it yet and produces
an install that succeeds and then fails to import.

```bash
git clone https://github.com/nitya-prakash-pandey-2005/TyreMind.git
cd TyreMind

python -m venv .venv
.venv\Scripts\activate                 # Windows
# source .venv/bin/activate            # macOS / Linux

pip install -r requirements.txt        # dependencies
pip install -e .                       # the tyremind package itself

python -m tyremind.serve               # opens http://127.0.0.1:8077
```

That is the whole thing. The last command starts the API, serves the dashboard
from the same process, pre-fits the cached sessions so no click in the demo is
the slow one, and opens a browser.

> [!IMPORTANT]
> **Both install lines are needed, in that order.** `pyproject.toml` declares no
> dependency list, so `pip install -e .` installs the package but none of what it
> imports. The first line fixes the versions; the second makes `tyremind`
> importable.

Expect roughly this on startup:

```
  TYREMIND
  Causal tyre intelligence

  sessions      8 (8 cached locally)
  warming     fitting cached sessions…
              8 ready in 22.4s

  dashboard   http://127.0.0.1:8077
  api docs    http://127.0.0.1:8077/docs
```

Warming takes 10–40 s depending on the machine. Pass `--no-warm` to start in
about a second and pay the ~6 s cost on first use of each session instead.

| Flag | |
|---|---|
| `--port 9000` | Use a different port if 8077 is taken |
| `--no-browser` | Do not open a browser — for a remote or headless machine |
| `--no-warm` | Skip pre-fitting |
| `--host 0.0.0.0` | Serve to other machines on the network |

**Where to start once it is open.** The left rail is ordered as an argument, so
reading top to bottom works. **Start here** shows why the obvious method fails on
this session's real numbers; **Live monitor** is the one to press play on,
because it shows the estimator running forward-only, one lap at a time, with the
interval visibly collapsing as evidence arrives. Every view is deep-linkable —
`#/circuit`, `#/strategy`, and so on.

**Verify the install:**

```bash
pytest              # 326 tests, about 30 s
ruff check .        # lint
```

If those pass, everything in this README is reproducible on your machine.

---

<details>
<summary><b>Optional extras</b> — MCP server, LLM narration, more sessions, frontend development</summary>

<br>

None of these are needed to run the dashboard.

### The MCP server — the estimator as tools an AI assistant can call

```bash
pip install -r requirements-optional.txt

python -m tyremind.mcp_server           # stdio, for Claude Desktop
python -m tyremind.mcp_server --http    # streamable HTTP
```

For Claude Desktop, add to `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "tyremind": {
      "command": "python",
      "args": ["-m", "tyremind.mcp_server"],
      "cwd": "/absolute/path/to/TyreMind"
    }
  }
}
```

Seven read-only tools: `list_sessions`, `get_degradation`, `explain_lap`,
`project_tyre_life`, `recommend_strategy`, `assess_trust`, `search_documentation`.

### LLM narration

Entirely optional. Without a key, TyreMind narrates from deterministic templates
with every number computed before the text is written — which is the safer
default, not a fallback.

```bash
cp .env.example .env                    # then paste your key into .env
```

`.env` is gitignored. Never commit a key.

### Caching more sessions

Eight sessions are committed as Parquet, so a fresh clone runs offline. To add
more you need network access once:

```bash
python scripts/build_demo.py --events Monza Suzuka --year 2024
python scripts/build_track_geometry.py --circuits Suzuka   # for the 3D view
```

### Working on the frontend

The dashboard is committed pre-built, so this is only needed if you change it.
Requires Node 20 or newer.

```bash
cd apps/web
npm install
npm run dev                             # hot reload on :5173, proxies the API
npm run build                           # rebuild dist/ — commit the result
```

</details>

<details>
<summary><b>Reproducing every number</b> — the seven experiment scripts</summary>

<br>

Every figure in the interface, the model card and the pitch documents is read
from a JSON file under `experiments/results/`. Regenerating those files
regenerates the claims.

**Offline — these run against the committed demo sessions:**

```bash
python experiments/exp01_ground_truth_recovery.py --n-seeds 25   # ~4 min
python experiments/exp02_prior_sensitivity.py                    # ~6 min
python experiments/exp04_energy_clock.py
python experiments/exp05_model_ladder.py                         # ~8 min
python experiments/exp13_lap_time_calibration.py --corpus demo   # ~6 min
```

Both of those also take `--corpus season` once the corpus below exists, which
scores the model ladder and the interval calibration on twenty races instead of
four.

**Needs the season corpus.** It is ~113 sessions and 61,396 laps, gitignored
because it is large and rebuildable. Build it once, then everything below is
offline too:

```bash
python scripts/build_corpus.py --years 2024 2023 --sessions R FP2   # hours
python scripts/build_session_conditions.py --years 2024 2023 --sessions R FP2
python scripts/build_circuit_features.py --year 2024 --geometry-only
python scripts/build_lineups.py --years 2024 2023

python experiments/exp03_practice_to_race.py --years 2024 2023 --all  # ~25 min
python experiments/exp08_compound_identity.py --years 2024 2023 2022  # ~8 min
python experiments/exp09_circuit_transfer.py
python experiments/exp10_bias_mechanism.py
python experiments/exp11_depth_matched.py                             # ~35 min
python experiments/exp12_conformal_intervals.py
python experiments/exp13_lap_time_calibration.py --corpus season
python experiments/exp14_naive_failure_rate.py
python experiments/exp15_driver_effect.py --refit             # ~20 min, then cached
python experiments/exp16_calibration_shape.py                 # ~10 min
python experiments/exp17_degradation_cliff.py --refit         # ~20 min, then cached
python experiments/exp18_identifiability_bound.py --limit 12
```

Or run the whole set in dependency order, which is what the ordering note below
is about:

```bash
bash scripts/rerun_all.sh
```

**Order matters in two places.** `exp09`, `exp10` and `exp11` read the result
files of `exp08` and `exp03`, and `exp12` reads `exp03`. Running them against a
stale result file is the single easiest way to produce a number that disagrees
with the rest of the project — which is why `scripts/rerun_all.sh` exists and
why `scripts/check_results_fresh.py` will tell you when it has happened.

**Check the results still match their scripts.** Every figure in the README, the
model card, the deck and the dashboard is read from `experiments/results/`, so a
script edited after its result was written leaves the documents quoting a number
the current code would no longer produce:

```bash
python scripts/check_results_fresh.py
```

`exp06` and `exp07` download their own data on first run and are cached
afterwards:

```bash
python experiments/exp06_circuit_asymmetry.py
python experiments/exp07_cross_domain.py --subset FD001          # ~3 min
```

</details>

<details>
<summary><b>If something goes wrong</b></summary>

<br>

| Symptom | Cause and fix |
|---|---|
| `ModuleNotFoundError: No module named 'tyremind'` | `pip install -e .` was skipped, or the virtualenv is not active |
| `ModuleNotFoundError: No module named 'fastf1'` (or numpy, pandas…) | `pip install -r requirements.txt` was skipped — `pip install -e .` alone installs no dependencies |
| `OverflowError: cannot convert longdouble infinity to integer` on `import numpy` | You are on Python 3.14. numpy has no wheel for it and installs a broken build. Use 3.11–3.13; `pyproject.toml` now refuses 3.14 outright |
| `Dashboard not built. The API will run without it.` | `apps/web/dist/` is missing. It is committed, so this means a partial clone — or run `npm --prefix apps/web run build` |
| `[Errno 10048] address already in use` | Something else holds port 8077. Use `--port 9000` |
| `No sessions are cached` | `data/demo/` is missing. It is committed; with network access, `python scripts/build_demo.py` rebuilds it |
| Charts render but are blank or monochrome | A stale build. `npm --prefix apps/web run build`, then hard-reload the browser |
| Slow first click on a session | Expected with `--no-warm`. A session fit takes about 6 s and is then cached for the process lifetime |

</details>

---

## Structure

```
src/tyremind/
  data/          FastF1 ingestion, quality engine, synthetic ground truth
  models/ssm/    Kalman kernel and the tyre state-space model
  models/        baselines, evaluation harness, trust layer, validation
  physics/       dynamics, thermal, wear
  causal/        decomposition, counterfactuals, projection
  simulate/      Monte Carlo race and strategy
  assets/        AssetProfile abstraction, C-MAPSS adapter
  stream/        online estimator and replay
  explain/       narration templates, business value
  rag/           hybrid retrieval index over the project's own documents
  api/           FastAPI service
  mcp_server.py  seven read-only tools for an AI agent

apps/web/        React dashboard — src/ plus a committed dist/, so a clone
                 runs without Node
data/demo/       eight sessions as Parquet, plus circuit geometry, committed
experiments/     reproducible scripts; results/ holds the JSON the UI reads
docs/            research audit, model card, limitations, demo guide
docs/pitch/      technical dossier and pitch deck, with their HTML sources
scripts/         cache builders, run once with network access
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

### Presentation material

| | |
|---|---|
| [**Technical dossier**](docs/pitch/TyreMind_Technical_Dossier.pdf) — PDF, 36pp | Problem, prior art, the identifiability derivation, model, physics, architecture, all seven experiments, uniqueness matrix, limitations, industry impact, scaling, roadmap, references |
| [**Pitch deck**](docs/pitch/TyreMind_Pitch_Deck.pdf) — PDF, 24 slides | The same argument at presentation pace |
| [`docs/pitch/deck_web.html`](docs/pitch/deck_web.html) | The pitch as one scrolling page, for sharing a link rather than a file |
| [**Plan of action**](docs/plan/ROADMAP.pdf) — PDF, 8pp | Audit against the five judged metrics, the eight gaps a rival would exploit, and the five moves to close them |

Both PDFs are generated from the HTML sources beside them, so they are
regenerated rather than edited:

```bash
chrome --headless --no-pdf-header-footer \
  --print-to-pdf=docs/pitch/TyreMind_Technical_Dossier.pdf \
  docs/pitch/dossier.html
```

---

## Licence

MIT
