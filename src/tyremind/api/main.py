"""FastAPI service for TyreMind.

Serves the fitted model to the dashboard, and streams a session lap by lap over
a WebSocket so the live estimator can be shown working rather than described.

Every numeric response carries its uncertainty, and every derived quantity says
whether it is filtered (knowable at the time) or smoothed (knowable afterwards).
That is a product decision as much as a scientific one: a strategy tool that
reports a point estimate with no interval invites exactly the overconfidence it
should be protecting against.
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from tyremind.api.store import SessionStore
from tyremind.causal.counterfactual import (
    DEFAULT_PERFORMANCE_THRESHOLD_S,
    counterfactuals,
    project_tyre,
)
from tyremind.causal.decomposition import decompose_lap, decompose_run
from tyremind.data.synthetic import naive_degradation_estimate
from tyremind.stream.live import LiveTyreMonitor, replay

logger = logging.getLogger(__name__)

EXPERIMENTS_DIR = Path("experiments/results")
WEB_DIST = Path("apps/web/dist")

app = FastAPI(
    title="TyreMind API",
    version="0.1.0",
    description="Causal tyre intelligence: latent tyre-state estimation from confounded telemetry.",
)

# The dashboard is served from this same process in the packaged demo, but runs
# on a Vite dev server during development. Permissive CORS is appropriate for a
# local analysis tool and would not be for a deployed one.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

store = SessionStore()


@app.get("/api/health")
def health() -> dict:
    """Liveness check, plus what the service can serve without a network."""
    catalogue = store.catalogue()
    return {
        "status": "ok",
        "sessions_available": len(catalogue),
        "sessions_cached": sum(1 for r in catalogue if r.cached),
        "offline_ready": all(r.cached for r in catalogue) and bool(catalogue),
    }


@app.get("/api/sessions")
def list_sessions() -> list[dict]:
    """Sessions the API can serve."""
    return [r.to_dict() for r in store.catalogue()]


def _load(session_id: str):
    try:
        return store.get(session_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.get("/api/session/{session_id}")
def session_summary(session_id: str) -> dict:
    """Headline numbers for a session: what was estimated, and how sure we are.

    Includes the naive per-compound estimate alongside the model's, because the
    gap between them is the entire argument for the platform and hiding it would
    be a strange choice.
    """
    loaded = _load(session_id)
    fit = loaded.fit

    fuel_mean, fuel_sd = fit.fuel_slope()
    track_mean, track_sd = fit.track_amplitude()
    traffic_mean, traffic_sd = fit.traffic_coefficient()
    naive = naive_degradation_estimate(loaded.lap_table)

    return {
        "session": loaded.ref.to_dict(),
        "quality": loaded.quality,
        "n_laps": int(len(loaded.lap_table)),
        "n_drivers": int(loaded.lap_table["driver"].nunique()),
        "n_runs": int(loaded.lap_table["run_id"].nunique()),
        "compounds": {
            compound: {
                "degradation_rate": mean,
                "degradation_rate_sd": sd,
                "ci95": [mean - 1.96 * sd, mean + 1.96 * sd],
                "naive_estimate": naive.get(compound),
                "laps": int((loaded.lap_table["compound"] == compound).sum()),
            }
            for compound, (mean, sd) in sorted(fit.compound_rates().items())
        },
        "confounders": {
            "fuel_slope": {
                "mean": fuel_mean,
                "sd": fuel_sd,
                "prior_mean": fit.priors.fuel_slope_mean,
                "prior_sd": fit.priors.fuel_slope_sd,
                "note": (
                    "Pinned by a physical prior. Fuel and degradation are collinear "
                    "within a run, so this is an assumption, not a measurement."
                ),
            },
            "track_evolution": {
                "mean": track_mean,
                "sd": track_sd,
                "prior_mean": fit.priors.track_amplitude_mean,
                "prior_sd": fit.priors.track_amplitude_sd,
                "note": (
                    "Total lap time the track gained over the session. Identified "
                    "by its saturating shape plus an informative prior."
                ),
            },
            "traffic": {
                "mean": traffic_mean,
                "sd": traffic_sd,
                "note": (
                    "Seconds lost at maximum traffic. Genuinely identified from "
                    "data -- traffic varies independently of tyre age."
                ),
            },
        },
        "diagnostics": {
            "loglik": fit.loglik,
            "aic": fit.aic(),
            "bic": fit.bic(),
            "converged": fit.converged,
            "observation_noise_sd": float(2.718281828459045 ** fit.hyper.log_obs_sd),
            "n_states": fit.index.size,
        },
    }


@app.get("/api/session/{session_id}/degradation")
def degradation(
    session_id: str,
    smoothed: Annotated[bool, Query(description="Full-session posterior vs live estimate")] = True,
) -> dict:
    """Per-driver, per-lap latent tyre state."""
    loaded = _load(session_id)
    frame = loaded.fit.degradation(smoothed=smoothed)
    return {
        "estimate_type": "smoothed" if smoothed else "filtered",
        "rows": json.loads(frame.to_json(orient="records")),
    }


@app.get("/api/session/{session_id}/track")
def track_evolution(session_id: str) -> dict:
    """Estimated track evolution across the session."""
    loaded = _load(session_id)
    frame = loaded.fit.track_evolution()
    return {"rows": json.loads(frame.to_json(orient="records"))}


@app.get("/api/session/{session_id}/runs")
def runs(session_id: str) -> list[dict]:
    """Every run in the session, longest first.

    The unit a degradation conclusion is actually drawn from, so this is what the
    UI offers for selection rather than raw driver names.
    """
    loaded = _load(session_id)
    grouped = (
        loaded.lap_table.groupby(["driver", "run_id", "compound"])
        .agg(
            laps=("session_lap", "size"),
            first_lap=("session_lap", "min"),
            last_lap=("session_lap", "max"),
            start_age=("tyre_age", "min"),
            end_age=("tyre_age", "max"),
            median_lap_time=("lap_time", "median"),
        )
        .reset_index()
        .sort_values("laps", ascending=False)
    )
    return json.loads(grouped.to_json(orient="records"))


@app.get("/api/session/{session_id}/decompose")
def decompose(
    session_id: str,
    driver: str,
    lap: int,
    reference_lap: int | None = None,
) -> dict:
    """Attribute one lap's time change to its named causes."""
    loaded = _load(session_id)
    try:
        return decompose_lap(loaded.fit, driver, lap, reference_lap).to_dict()
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/session/{session_id}/decompose-run")
def decompose_run_endpoint(session_id: str, driver: str, run_id: int) -> dict:
    """Lap-by-lap decomposition across a whole run, for the stacked-area view."""
    loaded = _load(session_id)
    try:
        frame = decompose_run(loaded.fit, driver, run_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"rows": json.loads(frame.to_json(orient="records"))}


@app.get("/api/session/{session_id}/counterfactual")
def counterfactual(session_id: str, driver: str, lap: int) -> dict:
    """What this lap would have been in clean air, or on a fresh set."""
    loaded = _load(session_id)
    try:
        results = counterfactuals(loaded.fit, driver, lap)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {
        "scenarios": [c.to_dict() for c in results],
        "disclaimer": (
            "Model-based estimates of laps that were never driven, under the "
            "identifying assumptions in the model card."
        ),
    }


@app.get("/api/session/{session_id}/projection")
def projection(
    session_id: str,
    driver: str,
    lap: int,
    horizon: int = 20,
    threshold: float = DEFAULT_PERFORMANCE_THRESHOLD_S,
) -> dict:
    """Project a tyre forward, with remaining competitive life."""
    loaded = _load(session_id)
    try:
        return project_tyre(
            loaded.fit, driver, lap, horizon=horizon, threshold=threshold
        ).to_dict()
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/experiments")
def experiments() -> dict:
    """Recorded experiment results.

    Read from disk rather than recomputed, and never hard-coded. If an experiment
    has not been run, it is absent -- the dashboard shows a gap rather than a
    plausible number.
    """
    out = {}
    if EXPERIMENTS_DIR.exists():
        for path in sorted(EXPERIMENTS_DIR.glob("*.json")):
            try:
                out[path.stem] = json.loads(path.read_text())
            except json.JSONDecodeError:
                logger.error("could not parse experiment result %s", path)
    return out


@app.websocket("/ws/replay/{session_id}")
async def replay_session(websocket: WebSocket, session_id: str) -> None:
    """Stream a session lap by lap through the live estimator.

    The estimator sees one lap at a time in order, with no access to the future,
    exactly as it would during a live session. Every frame carries the filtered
    state and the per-update timing, so the real-time claim is visible in the UI
    rather than asserted in a caption.

    Query parameter `speed` sets the delay between laps in seconds.
    """
    await websocket.accept()

    try:
        loaded = store.get(session_id)
    except (KeyError, RuntimeError) as exc:
        await websocket.send_json({"type": "error", "detail": str(exc)})
        await websocket.close()
        return

    try:
        delay = float(websocket.query_params.get("speed", "0.08"))
    except ValueError:
        delay = 0.08

    lap_table = loaded.lap_table
    monitor = LiveTyreMonitor(
        drivers=sorted(lap_table["driver"].unique().tolist()),
        compounds=sorted(lap_table["compound"].unique().tolist()),
        hyper=loaded.fit.hyper,
        max_runs_per_driver=int(lap_table.groupby("driver")["run_id"].nunique().max()) + 2,
        reference_time=float(lap_table["lap_time"].median()),
    )

    await websocket.send_json(
        {
            "type": "start",
            "session": loaded.ref.to_dict(),
            "total_laps": int(len(lap_table)),
            "drivers": monitor.drivers,
            "compounds": monitor.compounds,
        }
    )

    try:
        for index, (obs, state) in enumerate(replay(lap_table, monitor=monitor)):
            await websocket.send_json(
                {
                    "type": "lap",
                    "index": index,
                    "observation": {
                        "driver": obs.driver,
                        "session_lap": obs.session_lap,
                        "lap_time": obs.lap_time,
                        "compound": obs.compound,
                        "tyre_age": obs.tyre_age,
                        "traffic_index": obs.traffic_index,
                    },
                    "state": state.to_dict(),
                    "compound_rates": {
                        c: {"mean": m, "sd": s}
                        for c, (m, s) in monitor.compound_rates().items()
                    },
                }
            )
            # Yield to the event loop even at zero delay, so a fast replay cannot
            # starve the server or block the disconnect handler.
            await asyncio.sleep(delay)

        await websocket.send_json(
            {
                "type": "complete",
                "performance": monitor.performance_summary(),
                "final_states": [s.to_dict() for s in monitor.all_states()],
            }
        )
    except WebSocketDisconnect:
        logger.info("replay client disconnected from %s", session_id)
    except Exception as exc:  # noqa: BLE001
        logger.exception("replay of %s failed", session_id)
        try:
            await websocket.send_json({"type": "error", "detail": str(exc)})
        except (WebSocketDisconnect, RuntimeError):
            pass


# The built dashboard is mounted last so that it does not shadow /api routes.
if WEB_DIST.exists():
    app.mount("/assets", StaticFiles(directory=WEB_DIST / "assets"), name="assets")

    # response_model=None: the return type is a union of Response subclasses,
    # which FastAPI would otherwise try (and fail) to turn into a Pydantic model.
    @app.get("/{full_path:path}", response_model=None)
    def serve_dashboard(full_path: str) -> FileResponse | JSONResponse:
        """Serve the dashboard, falling back to index.html for client-side routes."""
        candidate = WEB_DIST / full_path
        if full_path and candidate.is_file():
            return FileResponse(candidate)
        index = WEB_DIST / "index.html"
        if index.exists():
            return FileResponse(index)
        return JSONResponse({"detail": "dashboard not built"}, status_code=404)
