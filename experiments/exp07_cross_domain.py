"""Experiment 07 -- does the same estimator work on something that is not a tyre?

Runs the identical state-space model on NASA's C-MAPSS turbofan degradation
benchmark and scores remaining-useful-life predictions against published ground
truth.

This exists because motorsport cannot supply the proof. Public F1 telemetry
contains no measured tyre wear, so there is no real-world number to check a
degradation estimate against. C-MAPSS has run-to-failure data with published RUL
labels and a large comparable literature, and it is the same abstract problem:
recover a hidden degradation state from noisy multivariate signals under varying
operating conditions.

No tyre-specific code runs here. Engine cycles are translated into the
estimator's vocabulary through `AssetProfile`, and everything downstream is the
same code that fits a Grand Prix.

    python experiments/exp07_cross_domain.py --subset FD001 --n-units 40

Writes experiments/results/exp07_cross_domain.json.
"""

from __future__ import annotations

import argparse
import json
import warnings
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from tyremind.assets.cmapss import (
    build_health_index,
    load_subset,
    select_trending_sensors,
    to_observations,
)
from tyremind.assets.profile import TURBOFAN, to_lap_table
from tyremind.models.ssm.tyre_ssm import TyreSSMPriors, fit_tyre_ssm

RESULTS = Path(__file__).parent / "results" / "exp07_cross_domain.json"

#: Piecewise-linear RUL cap, the standard convention in the C-MAPSS literature.
RUL_CAP = 125.0


def rul_score(error: np.ndarray) -> float:
    """NASA's asymmetric prognostics score.

    Late predictions are penalised far more heavily than early ones, because
    predicting an engine has more life than it does is the failure that grounds
    aircraft. Lower is better.

        s = exp(-d/13) - 1  for d < 0 (early)
        s = exp( d/10) - 1  for d >= 0 (late)

    Included because it is the metric the C-MAPSS literature reports, so results
    here are comparable to published work rather than to a metric of our own
    choosing.

    It is a SUM over engines, not a mean, so it is only comparable between runs
    that scored the same number of them. Published FD001 figures are quoted over
    all 100, which is why this experiment now scores all 100.
    """
    return float(
        np.sum(np.where(error < 0, np.exp(-error / 13.0) - 1.0, np.exp(error / 10.0) - 1.0))
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subset", default="FD001", choices=["FD001", "FD002", "FD003", "FD004"])
    # How many test engines to score. 0 means all of them.
    #
    # This used to default to 40 because the estimator fits the whole test set
    # jointly and cost grows faster than linearly in the number of engines: a
    # 100-engine run was abandoned after 108 CPU-minutes without converging,
    # where 40 finished in three. That mattered, because published C-MAPSS RMSE
    # figures are quoted over all 100, so ours was not a like-for-like number.
    #
    # Batching removes the limit. Each engine's level and rate are its own
    # states, driven by its own observations; the only quantity pooled across
    # engines in a fit is the shared degradation baseline, and that is already
    # estimated from the TRAINING engines. Fitting the test set in batches is
    # therefore close to fitting it whole, and `--batch-size 0` still fits it
    # jointly for anyone who wants to check that claim rather than take it.
    parser.add_argument("--n-units", type=int, default=0)
    parser.add_argument(
        "--batch-size", type=int, default=25,
        help="engines per test fit; 0 fits them all jointly",
    )
    parser.add_argument(
        "--train-units", type=int, default=40,
        help="training engines used for the pooled baseline and health index",
    )
    args = parser.parse_args()

    warnings.filterwarnings("ignore")

    print(f"  loading C-MAPSS {args.subset}…")
    data = load_subset(args.subset)
    print(
        f"  {data.train['unit'].nunique()} training engines, "
        f"{data.test['unit'].nunique()} test engines, "
        f"{data.n_conditions} operating condition(s)"
    )

    sensors = select_trending_sensors(data.train)
    print(f"  {len(sensors)} of 21 sensors carry a usable trend: {', '.join(sensors[:6])}…")

    # Health index fitted on TRAINING units only, then applied to test units.
    # Fitting it on the test set would leak the very degradation being predicted.
    train_health = build_health_index(data.train, sensors)
    test_health = build_health_index(data.test, sensors, reference=data.train)

    observations = to_observations(data.train, train_health, max_units=args.train_units)
    lap_table = to_lap_table(observations, TURBOFAN)
    print(f"  translated {len(lap_table)} engine-cycles into the estimator's schema")

    # Priors matched to the asset, not to tyres. Turbofans have no fuel burn-off
    # and no traffic, so those confounders are switched off by giving them zero
    # prior width -- the estimator then leaves them at zero rather than fitting
    # noise into terms that do not exist for this asset.
    priors = TyreSSMPriors(
        fuel_slope_mean=0.0,
        fuel_slope_sd=1e-6,
        track_amplitude_mean=0.0,
        track_amplitude_sd=1e-6,
        traffic_coef_sd=1e-6,
        compound_rate_mean=TURBOFAN.degradation_prior_mean,
        compound_rate_sd=TURBOFAN.degradation_prior_sd,
        run_intercept_sd=1.0,
        initial_level_sd=0.02,
    )

    print("  fitting the state-space model…")
    fit = fit_tyre_ssm(lap_table, priors=priors)
    print(f"  converged={fit.converged}  loglik={fit.loglik:.1f}  states={fit.index.size}")

    rates = {c: r for c, (r, _) in fit.compound_rates().items()}
    mean_rate = float(np.mean(list(rates.values()))) if rates else float("nan")
    print(f"  estimated degradation rate: {mean_rate:.5f} health units per cycle")

    # --- calibrate the failure threshold from run-to-failure data ---------
    # The training engines all run to failure, so the health index at their last
    # cycle IS the failure point. Reading it off the data beats picking a round
    # number: the health index is a constructed scale with no natural units, and
    # an arbitrary threshold produced a 45-cycle over-prediction of engine life,
    # which for a prognostics model is the dangerous direction.
    healthy = float(train_health.loc[data.train.groupby("unit").head(5).index].mean())
    at_failure = float(train_health.loc[data.train.groupby("unit").tail(3).index].mean())
    failure_level = healthy - at_failure

    print(
        f"  failure threshold calibrated from run-to-failure data: "
        f"health {healthy:.3f} when new -> {at_failure:.3f} at failure "
        f"({failure_level:.3f} of accumulated degradation)"
    )

    # --- predict RUL on held-out engines ---------------------------------
    # The estimator is run on the TEST engines' truncated trajectories -- which
    # uses no RUL labels -- and its own latent state is extrapolated to the
    # calibrated failure level. This is the same calculation as remaining
    # competitive tyre life, on a different asset.
    all_units = sorted(data.test["unit"].unique())
    wanted = all_units[: args.n_units] if args.n_units else all_units
    batch_size = args.batch_size or len(wanted)

    # Fitted in batches so every engine can be scored. See --batch-size.
    states = []
    for start in range(0, len(wanted), batch_size):
        batch = wanted[start : start + batch_size]
        batch_rows = data.test[data.test["unit"].isin(batch)]
        batch_health = test_health.loc[batch_rows.index]
        batch_table = to_lap_table(
            to_observations(batch_rows, batch_health, max_units=0), TURBOFAN
        )
        batch_fit = fit_tyre_ssm(batch_table, priors=priors)
        states.append(batch_fit.degradation())
        print(f"    engines {batch[0]:>3}-{batch[-1]:>3}  converged={batch_fit.converged}")
    test_state = pd.concat(states, ignore_index=True)

    predictions, truths = [], []

    for unit in wanted:
        asset = f"engine_{int(unit):03d}"
        engine = test_state[test_state["driver"] == asset].sort_values("session_lap")
        if len(engine) < 10:
            continue

        level = float(engine["level"].iloc[-1])
        rate = float(engine["rate"].iloc[-1])

        if rate <= 1e-6:
            # Not yet degrading measurably. Fall back to the fleet rate rather
            # than dividing by ~zero and predicting an absurd life.
            rate = abs(mean_rate) if np.isfinite(mean_rate) and mean_rate else 0.002

        remaining = (failure_level - level) / rate
        # Cap at 125 cycles, the piecewise-linear RUL convention used throughout
        # the C-MAPSS literature. Engines show essentially no degradation early
        # in life, so an unbounded extrapolation from a near-flat trend produces
        # meaningless numbers; published results assume the same cap, so applying
        # it keeps these figures comparable rather than flattering.
        predictions.append(float(np.clip(remaining, 0.0, RUL_CAP)))
        truths.append(float(data.test_rul.loc[unit]))

    predictions_arr = np.array(predictions)
    truths_arr = np.array(truths)
    error = predictions_arr - truths_arr

    rmse = float(np.sqrt((error**2).mean()))
    mae = float(np.abs(error).mean())
    score = rul_score(error)

    print("\n" + "=" * 82)
    print(f"CROSS-DOMAIN: TyreMind estimator on NASA C-MAPSS {args.subset}")
    print("=" * 82)
    print(f"engines scored              : {len(predictions_arr)}")
    print(f"RUL RMSE                    : {rmse:.1f} cycles")
    print(f"RUL MAE                     : {mae:.1f} cycles")
    print(f"NASA prognostics score      : {score:.0f}  (lower is better; a SUM over "
          f"{len(predictions_arr)} engines, {score / len(predictions_arr):.1f} each)")
    print(f"mean prediction / truth     : {predictions_arr.mean():.0f} / {truths_arr.mean():.0f} cycles")
    print(f"fraction predicted early    : {(error < 0).mean():.0%}  (safer than late)")
    print("-" * 82)
    print(
        "Published RMSE on FD001 spans roughly 12-20 cycles for purpose-built deep\n"
        "prognostics models trained on this dataset. The number above comes from a\n"
        "tyre degradation model pointed at engines with no retuning, so it is a\n"
        "generalisation check rather than a leaderboard entry -- and it is reported\n"
        "as such."
    )
    print("=" * 82)

    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    RESULTS.write_text(
        json.dumps(
            {
                "experiment": "exp07_cross_domain",
                "generated_at": datetime.now(UTC).isoformat(),
                "dataset": f"NASA C-MAPSS {args.subset}",
                "asset_profile": TURBOFAN.to_dict(),
                "n_train_engines": int(data.train["unit"].nunique()),
                "n_engines_scored": len(predictions_arr),
                "n_test_engines_available": int(data.test["unit"].nunique()),
                "batch_size": int(args.batch_size),
                "n_sensors_used": len(sensors),
                "sensors_used": sensors,
                "estimated_degradation_rate": mean_rate,
                "converged": fit.converged,
                "rul_rmse": rmse,
                "rul_mae": mae,
                "nasa_score": score,
                "nasa_score_per_engine": float(score / len(predictions_arr)),
                "rul_cap": RUL_CAP,
                "fraction_early": float((error < 0).mean()),
                "predictions": predictions_arr.tolist(),
                "truths": truths_arr.tolist(),
                "note": (
                    "Same estimator as the F1 model, no tyre-specific code. "
                    "Published deep-learning RMSE on FD001 is roughly 12-20 cycles; "
                    "this is a generalisation check, not a leaderboard entry."
                ),
            },
            indent=2,
            default=str,
        )
    )
    print(f"\nwrote {RESULTS}")


if __name__ == "__main__":
    main()
