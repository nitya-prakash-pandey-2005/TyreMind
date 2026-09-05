"""Strategy, trust, business-value and cross-industry endpoints.

Kept separate from `main` so the core session routes stay readable. Everything
here converts a fitted model into a decision, a caveat, or a number a
non-specialist can act on.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException

from tyremind.api.store import SessionStore
from tyremind.assets.profile import PROFILES
from tyremind.causal.counterfactual import project_tyre
from tyremind.causal.decomposition import decompose_lap
from tyremind.explain.business import build_value_report, fleet_value_estimate
from tyremind.explain.narrate import (
    decomposition_narration,
    projection_narration,
    strategy_narration,
)
from tyremind.models.ssm.tyre_ssm import TyreSSMPriors, fit_tyre_ssm
from tyremind.models.trust import (
    REGIME_MEANING,
    assess_applicability,
    build_consensus,
    detect_regime,
    value_of_information,
)
from tyremind.simulate.race import (
    DEFAULT_PIT_LOSS_S,
    RaceState,
    TyreModel,
    recommend,
    strategy_regret,
)

logger = logging.getLogger(__name__)

#: Shared with `main` so a session is fitted once, not once per module.
store = SessionStore()

EXPERIMENTS_DIR = Path("experiments/results")

#: Perturbed-prior fits, cached per session. Each one is a full re-fit taking a
#: few seconds, they are identical on every request, and without this the trust
#: panel leaves the UI blank for fifteen seconds each time it is opened.
_PERTURBATION_CACHE: dict[tuple[str, str], dict[str, tuple[float, float]]] = {}

#: Where a compound's degradation typically starts accelerating, as a fraction of
#: the oldest age observed for it. Used only to give the simulator a cliff point
#: when the session itself does not contain one.
CLIFF_FRACTION = 0.85


def _load(session_id: str):
    try:
        return store.get(session_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


def _tyre_models(loaded) -> dict[str, TyreModel]:
    """Build simulator inputs from the fitted degradation posterior.

    The posterior standard deviation is passed through rather than discarded --
    it is what makes the simulated outcome spread reflect genuine tyre
    uncertainty rather than only lap-time noise.
    """
    fit = loaded.fit
    lap_table = loaded.lap_table
    reference = float(lap_table["lap_time"].median())

    models = {}
    for compound, (rate, sd) in fit.compound_rates().items():
        laps = lap_table[lap_table["compound"] == compound]
        if laps.empty:
            continue
        # Pace of this compound relative to the session, from fresh-tyre laps.
        fresh = laps[laps["tyre_age"] <= 3]
        base = float(fresh["lap_time"].median() - reference) if len(fresh) >= 3 else 0.0
        max_age = float(laps["tyre_age"].max())
        models[compound] = TyreModel(
            compound=compound,
            base_pace_s=base,
            degradation_rate=float(rate),
            degradation_rate_sd=float(sd),
            cliff_lap=max(max_age * CLIFF_FRACTION, 8.0),
        )
    return models


def strategy(
    session_id: str,
    driver: str,
    lap: int,
    n_sims: int = 5000,
) -> dict:
    """Simulate pit strategies from a car's position at a given lap.

    Returns the ranked options, the recommendation, its decision confidence, and
    a plain-language justification built only from simulated quantities.
    """
    loaded = _load(session_id)
    lap_table = loaded.lap_table

    row = lap_table[(lap_table["driver"] == driver) & (lap_table["session_lap"] == lap)]
    if row.empty:
        raise HTTPException(
            status_code=400, detail=f"{driver} has no valid lap {lap} in this session"
        )
    row = row.iloc[0]

    tyres = _tyre_models(loaded)
    if str(row["compound"]) not in tyres:
        raise HTTPException(
            status_code=400,
            detail=f"no degradation estimate for compound {row['compound']!r}",
        )

    total_laps = int(lap_table["session_lap"].max())
    state = RaceState(
        current_lap=int(lap),
        total_laps=total_laps,
        position=0,
        current_compound=str(row["compound"]),
        current_tyre_age=float(row["tyre_age"]),
        gap_ahead_s=float(2.0 - 1.6 * float(row.get("traffic_index", 0.0))),
        gap_behind_s=2.0,
        base_lap_time_s=float(lap_table["lap_time"].median()),
        pit_loss_s=DEFAULT_PIT_LOSS_S,
    )

    result = recommend(state, tyres, n_sims=n_sims)
    narration = strategy_narration(result)

    return {
        **result.to_dict(),
        "state": {
            "current_lap": state.current_lap,
            "total_laps": state.total_laps,
            "laps_remaining": state.laps_remaining,
            "compound": state.current_compound,
            "tyre_age": state.current_tyre_age,
            "pit_loss_s": state.pit_loss_s,
        },
        "narration": narration.to_dict(),
        "distributions": {
            outcome.strategy.label: _histogram(outcome.race_times)
            for outcome in result.alternatives
        },
    }


def _histogram(values: np.ndarray, bins: int = 28) -> dict:
    """Bin simulated race times for a distribution plot.

    The distribution is the point of running a simulation at all -- a bar chart of
    expected values hides exactly the overlap that tells a strategist two options
    are not really distinguishable.
    """
    counts, edges = np.histogram(values, bins=bins)
    centres = (edges[:-1] + edges[1:]) / 2.0
    return {
        "centres": centres.tolist(),
        "counts": counts.tolist(),
        "mean": float(values.mean()),
        "p10": float(np.quantile(values, 0.1)),
        "p90": float(np.quantile(values, 0.9)),
    }


def regret(
    session_id: str,
    driver: str,
    lap: int,
    recommended_lap: int,
    actual_lap: int,
) -> dict:
    """How much time a mis-timed stop cost, in seconds."""
    loaded = _load(session_id)
    lap_table = loaded.lap_table
    row = lap_table[(lap_table["driver"] == driver) & (lap_table["session_lap"] == lap)]
    if row.empty:
        raise HTTPException(status_code=400, detail=f"{driver} has no valid lap {lap}")
    row = row.iloc[0]

    tyres = _tyre_models(loaded)
    alternatives = [c for c in tyres if c != str(row["compound"])]

    state = RaceState(
        current_lap=int(lap),
        total_laps=int(lap_table["session_lap"].max()),
        position=0,
        current_compound=str(row["compound"]),
        current_tyre_age=float(row["tyre_age"]),
        gap_ahead_s=2.0,
        gap_behind_s=2.0,
        base_lap_time_s=float(lap_table["lap_time"].median()),
    )
    return strategy_regret(
        state,
        tyres,
        recommended_lap,
        actual_lap,
        new_compound=alternatives[0] if alternatives else None,
    )


def trust(
    session_id: str,
    compound: str | None = None,
    tyre_age: float = 20.0,
) -> dict:
    """Whether the model should be believed about this situation.

    Combines a robustness ensemble across perturbed assumptions, an
    out-of-distribution check, and the ranked value of additional measurements.
    """
    loaded = _load(session_id)
    lap_table = loaded.lap_table
    compound = compound or str(lap_table["compound"].mode().iloc[0])

    # Consensus across PERTURBED ASSUMPTIONS of the same model, not across the
    # naive baselines. Comparing against methods already shown to be biased would
    # only re-measure their bias; what matters is whether the conclusion survives
    # someone disagreeing with our priors.
    base = TyreSSMPriors()
    perturbations = {
        "baseline": base,
        "fuel prior +1sd": TyreSSMPriors(
            fuel_slope_mean=base.fuel_slope_mean + base.fuel_slope_sd
        ),
        "fuel prior -1sd": TyreSSMPriors(
            fuel_slope_mean=base.fuel_slope_mean - base.fuel_slope_sd
        ),
        "track prior +1sd": TyreSSMPriors(
            track_amplitude_mean=base.track_amplitude_mean + base.track_amplitude_sd
        ),
    }

    estimates: dict[str, dict[str, tuple[float, float]]] = {}
    for name, priors in perturbations.items():
        if name == "baseline":
            estimates[name] = loaded.fit.compound_rates()
            continue

        key = (session_id, name)
        if key not in _PERTURBATION_CACHE:
            try:
                _PERTURBATION_CACHE[key] = fit_tyre_ssm(lap_table, priors=priors).compound_rates()
            except Exception:  # noqa: BLE001 - a failed perturbation is data, not a crash
                logger.warning("perturbation %s failed for %s", name, session_id)
                continue
        estimates[name] = _PERTURBATION_CACHE[key]

    consensus = build_consensus(estimates) if len(estimates) >= 2 else {}
    applicability = assess_applicability(lap_table, compound=compound, tyre_age=tyre_age)

    degradation = loaded.fit.degradation()
    regimes = []
    sizes = lap_table.groupby(["driver", "run_id"]).size().sort_values(ascending=False)
    for (driver, run_id), laps in list(sizes.items())[:10]:
        summary = detect_regime(degradation, lap_table, driver, int(run_id))
        regimes.append(
            {
                "driver": driver,
                "run_id": int(run_id),
                "laps": int(laps),
                **summary.to_dict(),
                "meaning": REGIME_MEANING.get(summary.regime, ""),
            }
        )

    return {
        "consensus": {c: v.to_dict() for c, v in consensus.items()},
        "applicability": applicability.to_dict(),
        "regimes": regimes,
        "value_of_information": value_of_information(loaded.fit),
    }


def narrate(session_id: str, driver: str, lap: int, use_llm: bool = False) -> dict:
    """Plain-language explanations of what the model found for one lap.

    Written for someone with no motorsport background. Numbers come only from the
    model; a language model, if enabled, may rewrite the prose but any rewrite
    that introduces an unverifiable number is discarded.
    """
    loaded = _load(session_id)
    try:
        decomposition = decompose_lap(loaded.fit, driver, lap)
        projection = project_tyre(loaded.fit, driver, lap, horizon=15)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    narrations = {
        "decomposition": decomposition_narration(decomposition),
        "projection": projection_narration(projection),
    }

    if use_llm:
        from tyremind.explain.narrate import rewrite_with_llm

        narrations = {k: rewrite_with_llm(v) for k, v in narrations.items()}

    return {k: v.to_dict() for k, v in narrations.items()}


def business(
    naive_error: float = 0.0966,
    model_error: float = 0.0044,
    degradation_rate: float = 0.10,
    regret_seconds: float = 4.0,
) -> dict:
    """Model quality translated into operational terms.

    Defaults are the measured figures from exp01. Every returned estimate carries
    a confidence tag -- measured, estimated or illustrative -- and its derivation.
    """
    try:
        report = build_value_report(
            naive_error=naive_error,
            model_error=model_error,
            degradation_rate=degradation_rate,
            regret_seconds=regret_seconds,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return report.to_dict()


def cross_industry(
    fleet_size: int = 500,
    annual_km: float = 120_000.0,
) -> dict:
    """The same engine applied to other asset classes.

    Returns the asset profiles, the real cross-domain result on NASA C-MAPSS, and
    a clearly-labelled illustrative fleet calculation.

    The C-MAPSS number is the one that carries weight: it is real data with
    published ground truth, produced by the identical estimator. The fleet figures
    are arithmetic about an opportunity, not evidence, and are labelled so.
    """
    cmapss_path = EXPERIMENTS_DIR / "exp07_cross_domain.json"
    cmapss = json.loads(cmapss_path.read_text()) if cmapss_path.exists() else None

    return {
        "profiles": [p.to_dict() for p in PROFILES.values()],
        "validated_transfer": cmapss,
        "fleet_illustration": fleet_value_estimate(
            fleet_size=fleet_size, annual_km_per_vehicle=annual_km
        ).to_dict(),
        "honest_summary": (
            "The estimator has been validated on a second asset class with real "
            "ground truth (NASA C-MAPSS turbofans). It has NOT been validated on "
            "vehicle fleets, because no public dataset pairs tyre tread depth with "
            "telematics. The fleet figures show the arithmetic of the opportunity, "
            "not a result."
        ),
    }


def validation(session_id: str) -> dict:
    """Practice-to-race validation for this event, if it has been run.

    Read from recorded experiment output rather than recomputed, so the dashboard
    shows what was actually measured.
    """
    path = EXPERIMENTS_DIR / "exp03_practice_to_race.json"
    if not path.exists():
        raise HTTPException(
            status_code=404,
            detail="practice-to-race validation has not been run. "
            "Run experiments/exp03_practice_to_race.py.",
        )

    data = json.loads(path.read_text())
    loaded = _load(session_id)
    event = loaded.ref.grand_prix.lower()

    matching = [
        r for r in data.get("reports", []) if event in str(r.get("event", "")).lower()
    ]
    return {
        "overall": data.get("overall"),
        "this_event": matching[0] if matching else None,
        "all_events": data.get("reports", []),
    }


def health_timeline(session_id: str, driver: str, run_id: int) -> dict:
    """Lap-by-lap tyre health for one stint, for the digital-twin view."""
    loaded = _load(session_id)
    degradation = loaded.fit.degradation()
    stint = degradation[
        (degradation["driver"] == driver) & (degradation["run_id"] == run_id)
    ].sort_values("tyre_age")

    if stint.empty:
        raise HTTPException(status_code=400, detail=f"{driver} has no run {run_id}")

    # Same anchoring as the live monitor: 1.5 s/lap of accumulated loss reads as
    # zero health. A convention, shown alongside the seconds it came from.
    health = np.clip(100.0 * (1.0 - stint["level"].to_numpy() / 1.5), 0.0, 100.0)

    return {
        "driver": driver,
        "run_id": run_id,
        "compound": str(stint["compound"].iloc[0]),
        "rows": [
            {
                "session_lap": int(r.session_lap),
                "tyre_age": float(r.tyre_age),
                "level": float(r.level),
                "level_sd": float(r.level_sd),
                "rate": float(r.rate),
                "rate_sd": float(r.rate_sd),
                "health": float(h),
            }
            for r, h in zip(stint.itertuples(index=False), health, strict=True)
        ],
        "health_anchor_note": (
            "Health is a rescaling of accumulated performance loss: 100 is a fresh "
            "set, 0 is 1.5 s/lap slower than fresh. It is a convention for reading "
            "at a glance, not a measurement of tread depth."
        ),
    }


def pit_window(
    session_id: str,
    driver: str,
    lap: int,
    n_sims: int = 1200,
) -> dict:
    """Expected race time for every possible pit lap, not just a few candidates.

    The single most useful strategy picture. A table of four options tells you
    which is best; sweeping the whole window tells you how *sharp* the optimum
    is -- whether being two laps late costs a tenth or costs the race. A flat
    curve means the decision does not matter much, which is as actionable as a
    steep one.

    Simulated with common random numbers across the sweep, so the shape is the
    signal and not Monte Carlo noise.
    """
    loaded = _load(session_id)
    lap_table = loaded.lap_table

    row = lap_table[(lap_table["driver"] == driver) & (lap_table["session_lap"] == lap)]
    if row.empty:
        raise HTTPException(
            status_code=400, detail=f"{driver} has no valid lap {lap} in this session"
        )
    row = row.iloc[0]

    tyres = _tyre_models(loaded)
    current = str(row["compound"])
    if current not in tyres:
        raise HTTPException(status_code=400, detail=f"no estimate for compound {current!r}")

    alternatives = [c for c in tyres if c != current]
    fresh = alternatives[0] if alternatives else current

    total_laps = int(lap_table["session_lap"].max())
    state = RaceState(
        current_lap=int(lap),
        total_laps=total_laps,
        position=0,
        current_compound=current,
        current_tyre_age=float(row["tyre_age"]),
        gap_ahead_s=float(2.0 - 1.6 * float(row.get("traffic_index", 0.0))),
        gap_behind_s=2.0,
        base_lap_time_s=float(lap_table["lap_time"].median()),
        pit_loss_s=DEFAULT_PIT_LOSS_S,
    )

    from tyremind.simulate.race import Strategy, simulate_strategy

    stay = simulate_strategy(state, Strategy("stay out", None), tyres, n_sims=n_sims, seed=11)

    candidates = list(range(lap + 1, total_laps + 1))
    if not candidates:
        raise HTTPException(status_code=400, detail="no laps remain to pit on")

    sweep = []
    for pit_lap in candidates:
        outcome = simulate_strategy(
            state,
            Strategy(f"pit lap {pit_lap}", pit_lap, fresh),
            tyres,
            n_sims=n_sims,
            seed=11,  # common random numbers: the shape is signal, not noise
        )
        sweep.append(
            {
                "pit_lap": pit_lap,
                "expected_time": outcome.expected_time,
                "downside": outcome.downside,
                "best_case": outcome.quantile(0.1),
                "runs_past_cliff": outcome.ran_out_of_tyre,
            }
        )

    best = min(sweep, key=lambda r: r["expected_time"])
    # How wide is the window that costs less than a second against the optimum?
    tolerable = [r["pit_lap"] for r in sweep if r["expected_time"] <= best["expected_time"] + 1.0]

    return {
        "driver": driver,
        "from_lap": int(lap),
        "total_laps": total_laps,
        "new_compound": fresh,
        "stay_out_expected_time": stay.expected_time,
        "optimum_lap": best["pit_lap"],
        "optimum_expected_time": best["expected_time"],
        "window_within_1s": [min(tolerable), max(tolerable)] if tolerable else None,
        "sweep": sweep,
        "n_sims": n_sims,
        "note": (
            "Model estimate. The width of the window matters as much as its "
            "centre: a flat curve means the exact lap is not critical."
        ),
    }


def corner_energy(circuit: str) -> dict:
    """Measured per-corner frictional energy share for a circuit.

    Read from `exp06_circuit_asymmetry`, which computes it from position
    telemetry. Returns `measured: false` when a circuit has not been analysed,
    so the dashboard can say so rather than showing an even split as if it were
    a result.
    """
    path = EXPERIMENTS_DIR / "exp06_circuit_asymmetry.json"
    if not path.exists():
        return {
            "circuit": circuit,
            "measured": False,
            "reason": "physics validation has not been run (experiments/exp06_circuit_asymmetry.py)",
        }

    data = json.loads(path.read_text())
    match = next(
        (c for c in data.get("circuits", []) if c["circuit"].lower() == circuit.lower()), None
    )
    if not match or "corner_share" not in match:
        analysed = [c["circuit"] for c in data.get("circuits", [])]
        return {
            "circuit": circuit,
            "measured": False,
            "reason": (
                f"{circuit} has not been analysed. Circuits with telemetry results: "
                f"{', '.join(analysed) if analysed else 'none'}."
            ),
        }

    return {
        "circuit": circuit,
        "measured": True,
        "corner_share": match["corner_share"],
        "left_side_energy_share": match["left_side_energy_share"],
        "front_axle_energy_share": match["front_axle_energy_share"],
        "peak_lateral_g": match["peak_lateral_g"],
        "published_direction": match["published_direction"],
        "predicted_direction": match["predicted_direction"],
        "n_laps": match["n_laps"],
    }


#: The retrieval corpus is built once, lazily. Indexing the docs and result
#: files takes a moment and never changes while the server is up.
_CORPUS = None


def track_geometry(circuit: str) -> dict:
    """Racing line and per-point tyre loading for the 3D circuit view.

    Precomputed by scripts/build_track_geometry.py, because the dashboard runs
    from cached timing data which does not carry position telemetry.
    """
    path = Path("data/demo") / f"track_{circuit.lower()}.json"
    if not path.exists():
        available = sorted(
            p.stem.replace("track_", "") for p in Path("data/demo").glob("track_*.json")
        )
        raise HTTPException(
            status_code=404,
            detail=(
                f"No track geometry for {circuit!r}. Available: {available}. "
                "Run scripts/build_track_geometry.py to add more."
            ),
        )
    return json.loads(path.read_text())


def ask(q: str, k: int = 4) -> dict:
    """Hybrid retrieval over the project's documentation and recorded results.

    Returns passages with their sources. Generates nothing -- the point is that
    every answer can be traced to a document or an experiment output.
    """
    global _CORPUS
    from tyremind.rag.index import build_corpus

    if _CORPUS is None:
        _CORPUS = build_corpus()

    try:
        hits = _CORPUS.index.search(q, k=k)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {
        "query": q,
        "results": [h.to_dict() for h in hits],
        "corpus": _CORPUS.stats(),
        "note": (
            "Retrieved passages, not generated text. Every figure here traces to a "
            "document or a recorded experiment result."
        ),
    }


def register(app: FastAPI) -> None:
    """Attach these routes to the application.

    Registered with `app.get` directly rather than through an APIRouter.
    In fastapi 0.141.1 with starlette 1.6.0, `include_router` silently adds
    nothing -- a two-line reproduction confirms it -- so a router registered
    that way disappears with no error and every endpoint on it 404s. Binding
    to the app avoids the broken path entirely.
    """
    app.get("/api/session/{session_id}/strategy")(strategy)
    app.get("/api/session/{session_id}/regret")(regret)
    app.get("/api/session/{session_id}/trust")(trust)
    app.get("/api/session/{session_id}/narrate")(narrate)
    app.get("/api/business")(business)
    app.get("/api/cross-industry")(cross_industry)
    app.get("/api/session/{session_id}/validation")(validation)
    app.get("/api/session/{session_id}/health-timeline")(health_timeline)
    app.get("/api/session/{session_id}/pit-window")(pit_window)
    app.get("/api/physics/corner-energy")(corner_energy)
    app.get("/api/physics/track-geometry")(track_geometry)
    app.get("/api/ask")(ask)
