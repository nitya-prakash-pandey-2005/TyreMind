"""Experiment 01 -- can the model recover a degradation rate it was never shown?

Generates confounded synthetic sessions with a known hidden degradation rate,
runs both the naive estimator and the state-space model, and measures how close
each gets. Repeated over many seeds so the answer is a distribution, not an
anecdote.

This is the experiment the platform's central claim rests on. Run it before
believing anything else.

    python experiments/exp01_ground_truth_recovery.py --n-seeds 20

Writes experiments/results/exp01_ground_truth_recovery.json.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from tyremind.data.synthetic import (
    SessionConfig,
    generate_session,
    naive_degradation_estimate,
)
from tyremind.models.ssm.tyre_ssm import fit_tyre_ssm

RESULTS = Path(__file__).parent / "results" / "exp01_ground_truth_recovery.json"


def run_one_seed(seed: int) -> dict:
    """Generate one session, estimate it both ways, return per-compound errors."""
    session = generate_session(SessionConfig(seed=seed))
    truth = session.truth.compound_rates

    started = time.perf_counter()
    result = fit_tyre_ssm(session.lap_table)
    fit_seconds = time.perf_counter() - started

    ssm_rates = result.compound_rates()
    naive_rates = naive_degradation_estimate(session.lap_table)

    per_compound = {}
    for compound, true_rate in truth.items():
        ssm_mean, ssm_sd = ssm_rates.get(compound, (np.nan, np.nan))
        naive = naive_rates.get(compound, np.nan)

        # Does the 95% credible interval actually contain the truth? Coverage is
        # the property that decides whether the uncertainty numbers mean anything.
        lo, hi = ssm_mean - 1.96 * ssm_sd, ssm_mean + 1.96 * ssm_sd
        per_compound[compound] = {
            "true_rate": true_rate,
            "ssm_rate": float(ssm_mean),
            "ssm_sd": float(ssm_sd),
            "ssm_error": float(ssm_mean - true_rate),
            "ssm_covered": bool(lo <= true_rate <= hi),
            "naive_rate": float(naive),
            "naive_error": float(naive - true_rate),
        }

    fuel_mean, fuel_sd = result.fuel_slope()
    traffic_mean, traffic_sd = result.traffic_coefficient()

    return {
        "seed": seed,
        "n_laps": int(len(session.lap_table)),
        "converged": result.converged,
        "fit_seconds": fit_seconds,
        "loglik": result.loglik,
        "per_compound": per_compound,
        "fuel_slope": {
            "true": session.truth.fuel_slope,
            "posterior_mean": fuel_mean,
            "posterior_sd": fuel_sd,
            "prior_mean": result.priors.fuel_slope_mean,
            "prior_sd": result.priors.fuel_slope_sd,
        },
        "traffic_coefficient": {
            "true": session.truth.traffic_coefficient,
            "posterior_mean": traffic_mean,
            "posterior_sd": traffic_sd,
        },
        "estimated_obs_sd": float(np.exp(result.hyper.log_obs_sd)),
    }


def summarise(runs: list[dict]) -> dict:
    """Aggregate across seeds into the headline numbers."""
    compounds = sorted(runs[0]["per_compound"])
    summary: dict[str, dict] = {}

    for compound in compounds:
        ssm_err = np.array([r["per_compound"][compound]["ssm_error"] for r in runs])
        naive_err = np.array([r["per_compound"][compound]["naive_error"] for r in runs])
        covered = np.array([r["per_compound"][compound]["ssm_covered"] for r in runs])
        true_rate = runs[0]["per_compound"][compound]["true_rate"]

        summary[compound] = {
            "true_rate": true_rate,
            "ssm_mae": float(np.abs(ssm_err).mean()),
            "ssm_bias": float(ssm_err.mean()),
            "ssm_rmse": float(np.sqrt((ssm_err**2).mean())),
            "naive_mae": float(np.abs(naive_err).mean()),
            "naive_bias": float(naive_err.mean()),
            "naive_rmse": float(np.sqrt((naive_err**2).mean())),
            "error_reduction_pct": float(
                100.0 * (1.0 - np.abs(ssm_err).mean() / np.abs(naive_err).mean())
            ),
            "interval_coverage_95": float(covered.mean()),
        }

    all_ssm = np.array(
        [r["per_compound"][c]["ssm_error"] for r in runs for c in compounds]
    )
    all_naive = np.array(
        [r["per_compound"][c]["naive_error"] for r in runs for c in compounds]
    )
    all_covered = np.array(
        [r["per_compound"][c]["ssm_covered"] for r in runs for c in compounds]
    )

    return {
        "per_compound": summary,
        "overall": {
            "ssm_mae": float(np.abs(all_ssm).mean()),
            "naive_mae": float(np.abs(all_naive).mean()),
            "ssm_bias": float(all_ssm.mean()),
            "naive_bias": float(all_naive.mean()),
            "error_reduction_pct": float(
                100.0 * (1.0 - np.abs(all_ssm).mean() / np.abs(all_naive).mean())
            ),
            "interval_coverage_95": float(all_covered.mean()),
            "mean_fit_seconds": float(np.mean([r["fit_seconds"] for r in runs])),
            "convergence_rate": float(np.mean([r["converged"] for r in runs])),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-seeds", type=int, default=20)
    parser.add_argument("--start-seed", type=int, default=1000)
    args = parser.parse_args()

    runs = []
    for i in range(args.n_seeds):
        seed = args.start_seed + i
        run = run_one_seed(seed)
        runs.append(run)
        overall = {
            c: f"{v['ssm_error']:+.4f}" for c, v in run["per_compound"].items()
        }
        print(f"  seed {seed}: {run['fit_seconds']:5.2f}s  ssm error {overall}")

    summary = summarise(runs)

    print("\n" + "=" * 72)
    print("GROUND-TRUTH DEGRADATION RECOVERY")
    print("=" * 72)
    print(f"{'compound':<10} {'true':>8} {'naive MAE':>11} {'TyreMind MAE':>14} {'cover95':>9}")
    for compound, s in summary["per_compound"].items():
        print(
            f"{compound:<10} {s['true_rate']:>8.4f} {s['naive_mae']:>11.4f} "
            f"{s['ssm_mae']:>14.4f} {s['interval_coverage_95']:>8.0%}"
        )
    o = summary["overall"]
    print("-" * 72)
    print(
        f"{'OVERALL':<10} {'':>8} {o['naive_mae']:>11.4f} {o['ssm_mae']:>14.4f} "
        f"{o['interval_coverage_95']:>8.0%}"
    )
    print(f"\nError reduction vs naive : {o['error_reduction_pct']:.1f}%")
    print(f"Naive bias               : {o['naive_bias']:+.4f} s/lap")
    print(f"TyreMind bias            : {o['ssm_bias']:+.4f} s/lap")
    print(f"Mean fit time            : {o['mean_fit_seconds']:.2f} s")
    print("=" * 72)

    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    RESULTS.write_text(
        json.dumps(
            {
                "experiment": "exp01_ground_truth_recovery",
                "generated_at": datetime.now(UTC).isoformat(),
                "n_seeds": args.n_seeds,
                "summary": summary,
                "runs": runs,
            },
            indent=2,
        )
    )
    print(f"\nwrote {RESULTS}")


if __name__ == "__main__":
    main()
