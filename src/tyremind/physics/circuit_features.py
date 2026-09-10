"""Describe a circuit by what it does to a tyre, before anyone has run there.

The product question a team actually asks is not "explain Friday" -- it is "we
arrive on Thursday, what will the tyres do?" Answering that needs a description
of the circuit that exists *independently of any stint run on it*, so a model
fitted on other circuits has something to predict from.

Two families of feature, and the distinction is load-bearing:

  GEOMETRIC   corner count, how tight the corners are, how much of the lap is
              spent turning. Available from FastF1's circuit info with no
              telemetry download, and identical for every session at that venue.

  ENERGETIC   lateral acceleration and the frictional energy it implies, from a
              single fast lap's position and speed trace. Closer to the physics
              that actually removes rubber, but needs telemetry and varies a
              little with the lap chosen.

Nothing here reads a degradation estimate. That would be circular: these features
exist to predict degradation, so they must be computable for a circuit whose
degradation is unknown.
"""

from __future__ import annotations

import warnings
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

#: Corners tighter than this are treated as the slow, high-slip variety that
#: punishes a tyre differently from a fast sweeper. Degrees of direction change.
TIGHT_CORNER_DEG = 90.0

#: Lateral acceleration above which a corner is "loaded". Chosen well inside the
#: 7 g clip the dynamics layer applies, so a GPS glitch cannot manufacture one.
LOADED_G = 3.0


@dataclass(frozen=True)
class CircuitFeatures:
    """What a circuit asks of a tyre, computed without any tyre data."""

    circuit: str
    year: int

    # Geometric -- cheap, session-independent
    n_corners: int
    n_tight_corners: int
    mean_corner_angle: float
    median_corner_spacing_m: float
    lap_length_m: float

    # Energetic -- needs one fast lap of telemetry
    mean_abs_lateral_g: float | None = None
    p95_lateral_g: float | None = None
    loaded_fraction: float | None = None
    lateral_energy_proxy: float | None = None
    top_speed_kmh: float | None = None

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def has_energetics(self) -> bool:
        return self.lateral_energy_proxy is not None


def geometric_features(corners: pd.DataFrame, lap_length_m: float) -> dict:
    """Corner statistics from FastF1 circuit info.

    Args:
        corners: Frame with `Angle` in degrees and `X`, `Y` in position units.
        lap_length_m: Track length, metres.
    """
    angles = np.abs(np.asarray(corners["Angle"], dtype=float))
    # FastF1 reports the angle as a heading, so wrap into a 0-180 turn magnitude.
    turn = np.minimum(angles % 360.0, 360.0 - (angles % 360.0))

    xy = np.column_stack([corners["X"].to_numpy(float), corners["Y"].to_numpy(float)])
    # A single corner has no gap to measure. Say so with NaN rather than asking
    # numpy to take the median of an empty or all-NaN array, which is the same
    # answer plus a warning.
    spacing = (
        np.linalg.norm(np.diff(xy, axis=0), axis=1) if len(xy) > 1 else np.array([])
    )
    median_spacing = float(np.nanmedian(spacing)) if spacing.size else float("nan")

    return {
        "n_corners": int(len(corners)),
        "n_tight_corners": int((turn >= TIGHT_CORNER_DEG).sum()),
        "mean_corner_angle": float(np.nanmean(turn)) if len(turn) else float("nan"),
        "median_corner_spacing_m": median_spacing,
        "lap_length_m": float(lap_length_m),
    }


def energetic_features(telemetry: pd.DataFrame) -> dict:
    """Lateral loading from one fast lap.

    Reuses the same curvature chain the 3D view is drawn from, so the numbers a
    judge sees on screen and the numbers the transfer model consumes come from
    one implementation rather than two that can drift apart.

    Args:
        telemetry: Frame with `X`, `Y` position and `Speed` in km/h.
    """
    from tyremind.physics.dynamics import lateral_acceleration, path_curvature

    x = np.asarray(telemetry["X"], dtype=float)
    y = np.asarray(telemetry["Y"], dtype=float)
    speed_ms = np.asarray(telemetry["Speed"], dtype=float) / 3.6

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        kappa = path_curvature(x, y)
        lat_g = lateral_acceleration(speed_ms, kappa) / 9.81

    abs_g = np.abs(lat_g)
    finite = abs_g[np.isfinite(abs_g)]
    if finite.size == 0:
        return {}

    # Frictional energy scales with load times slip, and slip is not observable.
    # What survives is the ordering, so this is a proxy: mean of v * a_y over the
    # lap, which is dimensionally a specific power and ranks circuits sensibly.
    power_proxy = np.nanmean(speed_ms * np.abs(lat_g) * 9.81)

    return {
        "mean_abs_lateral_g": float(np.nanmean(finite)),
        "p95_lateral_g": float(np.nanpercentile(finite, 95)),
        "loaded_fraction": float((finite >= LOADED_G).mean()),
        "lateral_energy_proxy": float(power_proxy),
        "top_speed_kmh": float(np.nanmax(telemetry["Speed"])),
    }


def from_session(session, *, with_telemetry: bool = True) -> CircuitFeatures:
    """Build features from a loaded FastF1 session.

    Args:
        session: A FastF1 session with laps loaded.
        with_telemetry: Whether to compute the energetic half. Skipping it keeps
            the call cheap when only geometry is needed.

    Raises:
        ValueError: If the session exposes no circuit information.
    """
    info = session.get_circuit_info()
    if info is None or info.corners is None or info.corners.empty:
        raise ValueError("session exposes no circuit corner information")

    try:
        lap_length = float(session.laps.pick_fastest().telemetry["Distance"].max())
    except Exception:  # noqa: BLE001 - length is nice to have, not load-bearing
        lap_length = float("nan")

    fields = geometric_features(info.corners, lap_length)
    fields["circuit"] = str(session.event["Location"])
    fields["year"] = int(session.event.year)

    if with_telemetry:
        try:
            fastest = session.laps.pick_fastest()
            telemetry = fastest.get_telemetry()
            if lap_length != lap_length:  # NaN check without importing math
                fields["lap_length_m"] = float(telemetry["Distance"].max())
            fields.update(energetic_features(telemetry))
        except Exception:  # noqa: BLE001 - a circuit without telemetry still has geometry
            pass

    return CircuitFeatures(**fields)
