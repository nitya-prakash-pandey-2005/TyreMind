"""Extract circuit geometry and per-point tyre loading for the 3D view.

The dashboard runs from cached timing data, which does not carry position
telemetry -- that is a much heavier download used by the experiments. But the 3D
circuit view needs the actual racing line, so this precomputes it once and
commits the result: a few thousand points per circuit, a few hundred kilobytes.

What gets stored is not just geometry. Each point on the line carries the lateral
acceleration, the speed, and the per-corner energy share at that instant, so the
3D view can colour the track by what it was doing to the tyres rather than by
elevation or an arbitrary gradient.

    python scripts/build_track_geometry.py --circuits Monza Zandvoort

Writes data/demo/track_<circuit>.json.
"""

from __future__ import annotations

import argparse
import json
import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from tyremind.physics.dynamics import (
    VehicleParameters,
    corner_loads,
    frictional_power_proxy,
    lateral_acceleration,
    longitudinal_acceleration,
    path_curvature,
)

OUT_DIR = Path("data/demo")

#: Points kept per lap. The raw trace is 4-10 Hz over a 60-120 s lap, so a few
#: hundred to a thousand samples. Downsampling to this keeps the payload small
#: while preserving every corner -- an F1 circuit has 10-20 of them.
TARGET_POINTS = 420


def resample(values: np.ndarray, n: int) -> np.ndarray:
    """Resample a trace to `n` points by linear interpolation along its index."""
    if len(values) <= n:
        return values
    source = np.linspace(0.0, 1.0, len(values))
    target = np.linspace(0.0, 1.0, n)
    return np.interp(target, source, values)


def extract(session, params: VehicleParameters) -> dict | None:
    """Build the geometry payload from the fastest representative lap.

    The fastest lap is used because it is the cleanest single trace of the
    racing line -- a slower lap may include a lift, an off, or a different line
    through traffic, all of which would show up as spurious geometry.
    """
    quick = session.laps.pick_quicklaps()
    if quick.empty:
        return None

    lap = quick.pick_fastest()
    telemetry = lap.get_telemetry()
    if len(telemetry) < 100 or "X" not in telemetry.columns:
        return None

    time_column = "Time" if "Time" in telemetry.columns else "SessionTime"
    t = pd.to_timedelta(telemetry[time_column]).dt.total_seconds().to_numpy()
    t = t - t[0]

    speed_ms = telemetry["Speed"].to_numpy(dtype=float) / 3.6
    # FastF1 supplies position in tenths of a metre.
    x = telemetry["X"].to_numpy(dtype=float) / 10.0
    y = telemetry["Y"].to_numpy(dtype=float) / 10.0
    z = (
        telemetry["Z"].to_numpy(dtype=float) / 10.0
        if "Z" in telemetry.columns
        else np.zeros_like(x)
    )

    kappa = path_curvature(x, y)
    a_lat = lateral_acceleration(speed_ms, kappa)
    a_long = longitudinal_acceleration(speed_ms, t)

    loads = corner_loads(speed_ms, a_long, a_lat, params)
    power = frictional_power_proxy(loads, speed_ms, a_long, a_lat)
    total_power = sum(power.values())

    # Trim the smoothing edges, where the convolution biases curvature to zero.
    edge = 8
    keep = slice(edge, -edge)

    n = min(TARGET_POINTS, len(x[keep]))
    payload = {
        "x": resample(x[keep], n).round(2).tolist(),
        "y": resample(y[keep], n).round(2).tolist(),
        "z": resample(z[keep], n).round(2).tolist(),
        "speed_kmh": (resample(speed_ms[keep], n) * 3.6).round(1).tolist(),
        "lateral_g": (resample(a_lat[keep], n) / 9.81).round(3).tolist(),
        "longitudinal_g": (resample(a_long[keep], n) / 9.81).round(3).tolist(),
        # Normalised so the 3D view can colour by relative tyre loading without
        # needing to know the arbitrary units of the frictional power proxy.
        "tyre_load": (
            resample(total_power[keep], n) / max(float(total_power[keep].max()), 1e-9)
        )
        .round(3)
        .tolist(),
        "throttle": (
            resample(telemetry["Throttle"].to_numpy(dtype=float)[keep], n).round(0).tolist()
            if "Throttle" in telemetry.columns
            else []
        ),
        "brake": (
            resample(telemetry["Brake"].to_numpy(dtype=float)[keep], n).round(2).tolist()
            if "Brake" in telemetry.columns
            else []
        ),
    }

    payload["stats"] = {
        "lap_time_s": float(lap["LapTime"].total_seconds())
        if pd.notna(lap["LapTime"])
        else None,
        "driver": str(lap["Driver"]),
        # 99th percentile rather than the maximum. A single GPS glitch produces
        # one impossible sample, and reporting the max would quote that glitch
        # as the circuit's peak load. The percentile is what the car actually
        # sustains through its hardest corner.
        "peak_lateral_g": float(np.percentile(np.abs(a_lat[keep]), 99) / 9.81),
        "peak_braking_g": float(np.percentile(np.abs(np.minimum(a_long[keep], 0)), 99) / 9.81),
        "top_speed_kmh": float(telemetry["Speed"].max()),
        "n_points": n,
        "track_length_m": float(np.sum(speed_ms[keep] * np.gradient(t[keep]))),
    }
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--year", type=int, default=2024)
    parser.add_argument(
        "--circuits",
        nargs="*",
        default=["Monza", "Silverstone", "Zandvoort", "Barcelona"],
    )
    args = parser.parse_args()

    warnings.filterwarnings("ignore")
    logging.getLogger("fastf1").setLevel(logging.ERROR)

    from tyremind.data.f1_loader import load_session

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    params = VehicleParameters(fuel_mass_kg=40.0)
    written = 0

    for circuit in args.circuits:
        print(f"  {circuit}…", end=" ", flush=True)
        try:
            session = load_session(args.year, circuit, "R", telemetry=True)
            payload = extract(session, params)
        except Exception as exc:  # noqa: BLE001 - one circuit failing must not stop the rest
            print(f"failed: {type(exc).__name__}: {exc}")
            continue

        if payload is None:
            print("no usable telemetry")
            continue

        payload["circuit"] = circuit
        payload["year"] = args.year

        path = OUT_DIR / f"track_{circuit.lower()}.json"
        path.write_text(json.dumps(payload))
        written += 1
        stats = payload["stats"]
        print(
            f"{stats['n_points']} points, {stats['track_length_m'] / 1000:.2f} km, "
            f"peak {stats['peak_lateral_g']:.1f} g -> {path.name} "
            f"({path.stat().st_size / 1024:.0f} kB)"
        )

    print(f"\nwrote {written} circuits to {OUT_DIR}")
    return 0 if written else 1


if __name__ == "__main__":
    raise SystemExit(main())
