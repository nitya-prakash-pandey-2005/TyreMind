<div align="center">

# TYREMIND

### Causal Tyre Intelligence & Digital Twin Platform

**Observed performance is not the same thing as tyre degradation.**

</div>

---

## The problem

A car is 0.42 s/lap slower than it was ten laps ago. How much of that is the tyre?

In a practice session the honest answer is *nobody knows*, because at least six things
changed at once: the fuel load dropped, the track rubbered in, the driver caught traffic,
the air cooled, the engine mode changed — and yes, the tyre wore. Read the lap time
naively and you will attribute all of it to the tyre. You will be wrong by more than the
effect you are trying to measure.

TyreMind estimates the **latent tyre state** underneath that confounded observation,
separates it from the things that are not the tyre, quantifies how sure it is, and turns
the result into a decision you can defend.

## What makes this different

Existing work predicts lap times well. **None of it checks whether the attribution is
correct** — and a model can predict lap time perfectly while blaming entirely the wrong
cause. That is the actual problem, and it is untested in the literature.

So we test it. See [`docs/research/08_NOVELTY_ANALYSIS.md`](docs/research/08_NOVELTY_ANALYSIS.md).

## Scientific honesty

TyreMind does **not** measure tread depth. Public F1 telemetry does not contain it.
What we estimate is a *latent performance/degradation state*. Every number the system
publishes carries an uncertainty interval and an epistemic tag. Where a quantity is
inferred rather than observed, the UI says so.

See [`docs/model_card.md`](docs/model_card.md) for intended use, limitations and failure modes.

## Status

Under active development. See [`docs/research/10_24_HOUR_BUILD_PLAN.md`](docs/research/10_24_HOUR_BUILD_PLAN.md).

## Quickstart

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements-dev.txt
pip install -e .
pytest
```

## Licence

MIT
