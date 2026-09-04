"""Experiment 06 -- does the physics layer recover circuit geometry it was never told?

An independent check on the whole dynamics pipeline: GPS trace to curvature to
lateral acceleration to per-corner load. Nothing in that chain knows which way a
circuit runs, so the direction is recoverable only if curvature sign, load
transfer and energy integration are all correct together.

The test is falsifiable against published fact. A clockwise circuit is mostly
right-hand corners, and cornering right throws load onto the LEFT-hand tyres, so
a clockwise circuit must show a left-side energy share above 50%. An
anti-clockwise circuit must show the opposite. Circuit rotation direction is not
in the telemetry and is not used by any code under test.

    python experiments/exp06_circuit_asymmetry.py

Writes experiments/results/exp06_circuit_asymmetry.json.

Averaged over many laps and drivers, because a single lap is a sample of one and
a driver taking an unusual line can move the number.
"""

from __future__ import annotations

import argparse
import json
import logging
import warnings
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from tyremind.physics.dynamics import (
    VehicleParameters,
    corner_loads,
    frictional_power_proxy,
    lateral_acceleration,
    longitudinal_acceleration,
    path_curvature,
)

RESULTS = Path(__file__).parent / "results" / "exp06_circuit_asymmetry.json"

#: Circuit rotation direction, from published circuit maps. Ground truth for this
#: test, and deliberately not derived from any data the pipeline touches.
CIRCUIT_DIRECTION: dict[str, str] = {
    "Monza": "clockwise",
    "Zandvoort": "clockwise",
    "Silverstone": "clockwise",
    "Barcelona": "clockwise",
    "Interlagos": "anti-clockwise",
    "Austin": "anti-clockwise",
    "Imola": "anti-clockwise",
    "Singapore": "anti-clockwise",
}


def analyse_lap(telemetry, params: VehicleParameters) -> dict | None:
    """Per-corner energy split for one lap. None if the telemetry is unusable."""
    if len(telemetry) < 50 or "X" not in telemetry.columns:
        return None

    time_column = "Time" if "Time" in telemetry.columns else "SessionTime"
    import pandas as pd

    t = pd.to_timedelta(telemetry[time_column]).dt.total_seconds().to_numpy()
    t = t - t[0]

    speed = telemetry["Speed"].to_numpy(dtype=float) / 3.6
    x = telemetry["X"].to_numpy(dtype=float) / 10.0
    y = telemetry["Y"].to_numpy(dtype=float) / 10.0

    kappa = path_curvature(x, y)
    a_lat = lateral_acceleration(speed, kappa)
    a_long = longitudinal_acceleration(speed, t)

    loads = corner_loads(speed, a_long, a_lat, params)
    power = frictional_power_proxy(loads, speed, a_long, a_lat)

    edge = 6
    energy = {
        corner: float(np.trapezoid(p[edge:-edge], t[edge:-edge]))
        for corner, p in power.items()
    }
    total = sum(energy.values())
    if total <= 0:
        return None

    # Net rotation over the lap. A closed circuit turns through a full circle, so
    # the sign of the integrated curvature is the direction the car went round.
    net_rotation = float(np.sum(kappa[edge:-edge]))

    # Energy-weighted split of cornering into left-hand and right-hand turns.
    weight = (np.abs(a_lat) * speed)[edge:-edge]
    interior = a_lat[edge:-edge]
    left_turn = float(weight[interior > 0].sum())
    right_turn = float(weight[interior < 0].sum())
    turning = left_turn + right_turn

    return {
        "left_side_energy_share": (energy["FL"] + energy["RL"]) / total,
        "front_axle_energy_share": (energy["FL"] + energy["FR"]) / total,
        "left_turn_energy_share": left_turn / turning if turning > 0 else float("nan"),
        "net_rotation": net_rotation,
        "peak_lateral_g": float(np.abs(a_lat[edge:-edge]).max() / 9.81),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--year", type=int, default=2024)
    parser.add_argument("--circuits", nargs="*", default=list(CIRCUIT_DIRECTION))
    parser.add_argument("--laps-per-circuit", type=int, default=12)
    args = parser.parse_args()

    warnings.filterwarnings("ignore")
    logging.getLogger("fastf1").setLevel(logging.ERROR)

    from tyremind.data.f1_loader import load_session

    params = VehicleParameters(fuel_mass_kg=50.0)
    rows = []

    for circuit in args.circuits:
        expected = CIRCUIT_DIRECTION.get(circuit, "unknown")
        print(f"  {circuit} ({expected})…", end=" ", flush=True)
        try:
            session = load_session(args.year, circuit, "R", telemetry=True)
        except Exception as exc:  # noqa: BLE001
            print(f"could not load: {type(exc).__name__}")
            continue

        # Spread laps across drivers so one unusual line cannot dominate.
        quick = session.laps.pick_quicklaps().reset_index(drop=True)
        if quick.empty:
            print("no representative laps")
            continue
        sample = quick.iloc[:: max(1, len(quick) // args.laps_per_circuit)].head(
            args.laps_per_circuit
        )

        analyses = []
        for _, lap in sample.iterrows():
            try:
                result = analyse_lap(lap.get_telemetry(), params)
            except Exception:  # noqa: BLE001
                continue
            if result:
                analyses.append(result)

        if len(analyses) < 3:
            print(f"only {len(analyses)} usable laps")
            continue

        left_share = float(np.mean([a["left_side_energy_share"] for a in analyses]))
        left_turn = float(np.mean([a["left_turn_energy_share"] for a in analyses]))
        rotation = float(np.median([a["net_rotation"] for a in analyses]))

        # A clockwise circuit loads the left-hand tyres, because the car spends
        # most of its cornering turning right.
        predicted = "clockwise" if left_share > 0.5 else "anti-clockwise"
        rows.append(
            {
                "circuit": circuit,
                "published_direction": expected,
                "predicted_direction": predicted,
                "correct": predicted == expected,
                "left_side_energy_share": left_share,
                "left_turn_energy_share": left_turn,
                "front_axle_energy_share": float(
                    np.mean([a["front_axle_energy_share"] for a in analyses])
                ),
                "net_rotation": rotation,
                "peak_lateral_g": float(np.mean([a["peak_lateral_g"] for a in analyses])),
                "n_laps": len(analyses),
            }
        )
        mark = "OK " if predicted == expected else "MISS"
        print(f"{mark} left-side {left_share:.1%}, left-turn energy {left_turn:.1%}")

    if not rows:
        raise SystemExit("no circuit produced usable telemetry")

    correct = sum(r["correct"] for r in rows)

    print("\n" + "=" * 88)
    print("CIRCUIT GEOMETRY RECOVERED FROM GPS TRACES ALONE")
    print("=" * 88)
    print(
        f"{'circuit':<14}{'published':<16}{'predicted':<16}"
        f"{'left-side':>11}{'left-turn':>11}{'peak g':>9}"
    )
    for r in rows:
        flag = "" if r["correct"] else "   <- MISS"
        print(
            f"{r['circuit']:<14}{r['published_direction']:<16}{r['predicted_direction']:<16}"
            f"{r['left_side_energy_share']:>10.1%}{r['left_turn_energy_share']:>11.1%}"
            f"{r['peak_lateral_g']:>9.2f}{flag}"
        )
    print("-" * 88)
    print(f"direction recovered correctly: {correct} of {len(rows)} circuits")
    print(
        "\nThe pipeline is never told which way a circuit runs. Recovering it from\n"
        "GPS traces requires the curvature sign, the load transfer and the energy\n"
        "integration all to be right at once."
    )
    if correct < len(rows):
        missed = [r["circuit"] for r in rows if not r["correct"]]
        print(
            f"\nMissed: {', '.join(missed)}. Reported rather than dropped -- a validation\n"
            "that only lists its successes is not a validation."
        )
    print("=" * 88)

    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    RESULTS.write_text(
        json.dumps(
            {
                "experiment": "exp06_circuit_asymmetry",
                "generated_at": datetime.now(UTC).isoformat(),
                "year": args.year,
                "n_circuits": len(rows),
                "n_correct": correct,
                "accuracy": correct / len(rows),
                "circuits": rows,
            },
            indent=2,
            default=str,
        )
    )
    print(f"\nwrote {RESULTS}")


if __name__ == "__main__":
    main()
