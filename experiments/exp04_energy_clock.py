"""Experiment 04 -- is cumulative tyre energy a better degradation clock than lap count?

The physics layer's central bet, stated so it can fail:

    A tyre does not know how many laps it has done. It knows how much energy has
    gone through its contact patch. So replacing "tyre age in laps" with
    "cumulative tyre energy" should give a degradation curve that transfers
    better between sessions that load the tyre differently.

This matters because exp03 found practice over-predicts race degradation by a
systematic +0.047 s/lap, and the most likely cause is a load difference: FP2
race-sim runs hold high fuel throughout while a race stint averages lower. If
that is the mechanism, an energy clock should absorb it.

Requires telemetry, which is a much heavier download than timing data, so this
runs over a small number of stints by default.

    python experiments/exp04_energy_clock.py --session 2024-monza-R --n-runs 6

Writes experiments/results/exp04_energy_clock.json.

The result is reported either way. A negative result here is a real finding
about what public telemetry can support, and is more useful than a quiet
retreat to lap count.
"""

from __future__ import annotations

import argparse
import json
import logging
import warnings
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from tyremind.physics.dynamics import VehicleParameters
from tyremind.physics.wear import cumulative_energy_clock, lap_wear

RESULTS = Path(__file__).parent / "results" / "exp04_energy_clock.json"


def collect_stint_physics(session, driver: str, run_laps: pd.DataFrame) -> list:
    """Run the physics pass over every lap of one stint, carrying thermal state.

    Carrying `initial_bulk_c` forward is what gives the model thermal memory. A
    carcass that is already hot on lap 12 starts lap 13 hot, and pretending
    otherwise removes the main mechanism by which a stint runs away from itself.
    """
    wears = []
    carry_bulk: float | None = None

    for row in run_laps.sort_values("session_lap").itertuples(index=False):
        try:
            lap = session.laps.pick_drivers(driver).pick_laps(int(row.session_lap))
            telemetry = lap.get_telemetry()
        except Exception:  # noqa: BLE001 - a missing lap is normal, not fatal
            continue

        if len(telemetry) < 12 or "X" not in telemetry.columns:
            continue

        try:
            result = lap_wear(
                telemetry,
                driver=driver,
                lap_number=int(row.session_lap),
                compound=str(row.compound),
                vehicle_params=VehicleParameters(fuel_mass_kg=60.0),
                initial_bulk_c=carry_bulk,
            )
        except ValueError:
            continue

        carry_bulk = result.mean_bulk_c
        wears.append(result)

    return wears


def fit_slope(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Least-squares slope and its R^2. Returns NaN if the clock does not move."""
    if len(x) < 4 or np.ptp(x) == 0:
        return float("nan"), float("nan")
    slope, intercept = np.polyfit(x, y, 1)
    fitted = slope * x + intercept
    ss_res = float(((y - fitted) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return float(slope), float(r2)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--year", type=int, default=2024)
    parser.add_argument("--grand-prix", default="Monza")
    parser.add_argument("--session", default="R")
    parser.add_argument("--n-runs", type=int, default=6)
    parser.add_argument("--min-laps", type=int, default=10)
    args = parser.parse_args()

    warnings.filterwarnings("ignore")
    logging.getLogger("fastf1").setLevel(logging.ERROR)

    from tyremind.data.f1_loader import build_lap_table, load_session

    print(f"  loading {args.year} {args.grand_prix} {args.session} with telemetry…")
    session = load_session(
        args.year, args.grand_prix, args.session, telemetry=True
    )
    lap_table, quality = build_lap_table(session)
    print(f"  {len(lap_table)} usable laps, quality {quality.score():.0f}/100")

    sizes = (
        lap_table.groupby(["driver", "run_id"])
        .size()
        .sort_values(ascending=False)
        .reset_index(name="laps")
    )
    candidates = sizes[sizes["laps"] >= args.min_laps].head(args.n_runs)

    if candidates.empty:
        raise SystemExit(f"no run has at least {args.min_laps} laps in this session")

    stints = []
    for row in candidates.itertuples(index=False):
        run_laps = lap_table[
            (lap_table["driver"] == row.driver) & (lap_table["run_id"] == row.run_id)
        ]
        print(f"    {row.driver} run {row.run_id} ({row.laps} laps)…", end=" ", flush=True)

        wears = collect_stint_physics(session, row.driver, run_laps)
        if len(wears) < args.min_laps:
            print(f"only {len(wears)} laps had usable telemetry, skipped")
            continue

        lap_numbers = [w.lap_number for w in wears]
        matched = run_laps[run_laps["session_lap"].isin(lap_numbers)].sort_values("session_lap")

        # Fuel-correct the lap times first. Without this both clocks are fitted
        # against a signal dominated by fuel, and the comparison measures nothing
        # about tyres.
        lap_time = matched["lap_time"].to_numpy(dtype=float)
        corrected = lap_time + 0.081 * matched["lap_in_run"].to_numpy(dtype=float)

        age_clock = matched["tyre_age"].to_numpy(dtype=float)
        energy_clock = cumulative_energy_clock(wears)

        age_slope, age_r2 = fit_slope(age_clock, corrected)
        energy_slope, energy_r2 = fit_slope(energy_clock, corrected)

        stints.append(
            {
                "driver": row.driver,
                "run_id": int(row.run_id),
                "compound": wears[0].compound,
                "laps": len(wears),
                "age_slope": age_slope,
                "age_r2": age_r2,
                "energy_slope": energy_slope,
                "energy_r2": energy_r2,
                "r2_gain": energy_r2 - age_r2,
                "mean_energy_mj": float(np.mean([w.total_energy_mj for w in wears])),
                "energy_cv": float(
                    np.std([w.total_energy_mj for w in wears])
                    / max(np.mean([w.total_energy_mj for w in wears]), 1e-9)
                ),
                "thermal_regimes": pd.Series([w.thermal_regime for w in wears])
                .value_counts()
                .to_dict(),
                "mean_fraction_in_window": float(
                    np.mean([w.fraction_in_window for w in wears])
                ),
                "front_rear_ratio": float(
                    np.mean(
                        [
                            (w.energy_mj["FL"] + w.energy_mj["FR"]) / max(w.total_energy_mj, 1e-9)
                            for w in wears
                        ]
                    )
                ),
                "left_right_ratio": float(
                    np.mean(
                        [
                            (w.energy_mj["FL"] + w.energy_mj["RL"]) / max(w.total_energy_mj, 1e-9)
                            for w in wears
                        ]
                    )
                ),
            }
        )
        print(f"R2 age {age_r2:.3f} vs energy {energy_r2:.3f}")

    if not stints:
        raise SystemExit("no stint had enough usable telemetry")

    frame = pd.DataFrame(stints)
    gains = frame["r2_gain"].dropna()
    energy_wins = int((gains > 0).sum())

    print("\n" + "=" * 84)
    print("DEGRADATION CLOCK -- lap count vs cumulative tyre energy")
    print("=" * 84)
    print(f"{'car':<6}{'compound':<9}{'laps':>6}{'R2 age':>9}{'R2 energy':>11}{'gain':>9}{'energy CV':>11}")
    for row in frame.itertuples(index=False):
        print(
            f"{row.driver:<6}{row.compound:<9}{row.laps:>6}{row.age_r2:>9.3f}"
            f"{row.energy_r2:>11.3f}{row.r2_gain:>+9.3f}{row.energy_cv:>11.3f}"
        )
    print("-" * 84)
    print(f"stints where the energy clock fits better : {energy_wins} of {len(gains)}")
    print(f"mean R2 gain                              : {gains.mean():+.4f}")
    print(f"mean lap-to-lap energy variation (CV)     : {frame['energy_cv'].mean():.3f}")
    print(f"mean front-axle share of energy           : {frame['front_rear_ratio'].mean():.3f}")
    print(f"mean left-side share of energy            : {frame['left_right_ratio'].mean():.3f}")

    verdict = (
        "energy clock is better"
        if gains.mean() > 0.01
        else "no meaningful difference"
        if gains.mean() > -0.01
        else "lap count is better"
    )
    print(f"\nVERDICT: {verdict}")
    if frame["energy_cv"].mean() < 0.05:
        print(
            "Note: per-lap energy varies by under 5% within a stint, so the two\n"
            "clocks are nearly proportional here and little difference is expected.\n"
            "The clocks should diverge across sessions with different fuel loads,\n"
            "which is what exp03 measures."
        )
    print("=" * 84)

    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    RESULTS.write_text(
        json.dumps(
            {
                "experiment": "exp04_energy_clock",
                "generated_at": datetime.now(UTC).isoformat(),
                "session": f"{args.year} {args.grand_prix} {args.session}",
                "n_stints": len(frame),
                "energy_clock_wins": energy_wins,
                "mean_r2_gain": float(gains.mean()),
                "mean_energy_cv": float(frame["energy_cv"].mean()),
                "verdict": verdict,
                "stints": frame.to_dict(orient="records"),
            },
            indent=2,
            default=str,
        )
    )
    print(f"\nwrote {RESULTS}")


if __name__ == "__main__":
    main()
