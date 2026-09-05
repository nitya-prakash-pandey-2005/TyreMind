"""Experiment 02 -- how much does the answer depend on the assumptions?

Two of the three collinearities in this problem are resolved by prior, not by
data. That is stated plainly everywhere in the codebase, but stating it is not
enough: the honest follow-up is to show how far the answer moves when those
priors move, so a reader can judge whether the conclusion survives disagreeing
with us.

The perturbations are not arbitrary. Each shifts one assumption by a full prior
standard deviation, or removes half the evidence, and re-fits:

    fuel prior +/- 1 sd      the physical fuel correction (0.081 +/- 0.016 s/lap)
    track prior +/- 1 sd     total track evolution (0.90 +/- 0.45 s)
    wide priors              both doubled in width -- "we are much less sure"
    half the field           evidence halved, testing the pooling argument
    live at 75%              filtered only, three-quarters through the session

An estimate that survives all of these is one whose conclusion is driven by the
data. One that does not is being driven by us, and should be reported as such.

    python experiments/exp02_prior_sensitivity.py

Writes experiments/results/exp02_prior_sensitivity.json.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import warnings
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from tyremind.data.synthetic import SessionConfig, generate_session
from tyremind.models.ssm.tyre_ssm import TyreSSMPriors, fit_tyre_ssm
from tyremind.models.trust import build_consensus

RESULTS = Path(__file__).parent / "results" / "exp02_prior_sensitivity.json"

BASE = TyreSSMPriors()

#: Where the "live" variant is sampled. The filtered estimate at the final step
#: equals the smoothed one by construction, so a live-vs-retrospective comparison
#: has to be taken before the session ends to mean anything.
LIVE_FRACTION = 0.75


def variants() -> dict[str, tuple[TyreSSMPriors, str]]:
    """The perturbations, each with the question it answers."""
    return {
        "baseline": (BASE, "Default priors, as documented in configs/physics.yaml."),
        "fuel prior +1sd": (
            TyreSSMPriors(fuel_slope_mean=BASE.fuel_slope_mean + BASE.fuel_slope_sd),
            "What if fuel burn-off costs more lap time than we assumed?",
        ),
        "fuel prior -1sd": (
            TyreSSMPriors(fuel_slope_mean=BASE.fuel_slope_mean - BASE.fuel_slope_sd),
            "What if it costs less?",
        ),
        "track prior +1sd": (
            TyreSSMPriors(
                track_amplitude_mean=BASE.track_amplitude_mean + BASE.track_amplitude_sd
            ),
            "What if the track rubbers in more than we assumed?",
        ),
        "track prior -1sd": (
            TyreSSMPriors(
                track_amplitude_mean=BASE.track_amplitude_mean - BASE.track_amplitude_sd
            ),
            "What if it rubbers in less?",
        ),
        "wide priors": (
            TyreSSMPriors(
                fuel_slope_sd=BASE.fuel_slope_sd * 2,
                track_amplitude_sd=BASE.track_amplitude_sd * 2,
            ),
            "What if we are much less confident in both physical assumptions?",
        ),
    }


def run_session(lap_table: pd.DataFrame) -> dict[str, dict[str, tuple[float, float]]]:
    """Fit every variant to one session and collect its compound rates."""
    out: dict[str, dict[str, tuple[float, float]]] = {}

    for name, (priors, _) in variants().items():
        try:
            out[name] = fit_tyre_ssm(lap_table, priors=priors).compound_rates()
        except Exception:  # noqa: BLE001 - a failed variant is data, not a crash
            continue

    # Halving the field tests the pooling argument directly: run stagger across
    # cars is what identifies tyre age against session lap, so with fewer cars
    # the estimate should get less certain, not merely noisier.
    drivers = sorted(lap_table["driver"].unique())
    half = lap_table[lap_table["driver"].isin(drivers[: max(2, len(drivers) // 2)])]
    if half["run_id"].nunique() >= 3:
        # A variant that fails to converge is data about the variant, not a crash.
        with contextlib.suppress(Exception):
            out["half the field"] = fit_tyre_ssm(half).compound_rates()

    # What was knowable PART-WAY THROUGH, which is the only version of this
    # comparison that carries information.
    #
    # The obvious implementation -- filtered estimate at the final step against
    # smoothed estimate at the final step -- is worthless, and worse, it looks
    # like a result. The RTS smoother is initialised from the filtered estimate
    # at time T, so for any state the two are identical at T to machine
    # precision. Reporting that as "live matches retrospective" would be
    # reporting an algebraic identity as though it were evidence.
    #
    # The real question is how good the live estimate was BEFORE the session
    # finished, so this reads the filtered state three-quarters of the way
    # through and scores it against the same truth. That is a number the
    # smoother genuinely improves on.
    try:
        fit = fit_tyre_ssm(lap_table)
        idx = fit.index
        var = np.einsum("tii->ti", fit.filtered.P_filt)
        cut = max(1, int(fit.filtered.a_filt.shape[0] * LIVE_FRACTION) - 1)
        out["live at 75% of the session"] = {
            compound: (
                float(fit.filtered.a_filt[cut, i]),
                float(np.sqrt(max(var[cut, i], 0.0))),
            )
            for compound, i in idx.compound_rate.items()
        }
    except Exception:  # noqa: BLE001
        pass

    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-seeds", type=int, default=4)
    parser.add_argument(
        "--real-sessions",
        nargs="*",
        default=["2024-monza-R", "2024-zandvoort-R"],
    )
    args = parser.parse_args()

    warnings.filterwarnings("ignore")

    # ---------- synthetic: is the TRUE rate still recovered? ----------
    print("  synthetic sessions (true rate known)")
    synthetic_rows = []
    for i in range(args.n_seeds):
        session = generate_session(SessionConfig(seed=7000 + i))
        results = run_session(session.lap_table)
        for variant, rates in results.items():
            for compound, truth in session.truth.compound_rates.items():
                mean, sd = rates.get(compound, (np.nan, np.nan))
                if np.isfinite(mean):
                    synthetic_rows.append(
                        {
                            "seed": 7000 + i,
                            "variant": variant,
                            "compound": compound,
                            "true_rate": truth,
                            "estimate": float(mean),
                            "sd": float(sd),
                            "error": float(mean - truth),
                            "covered": bool(abs(mean - truth) <= 1.96 * sd),
                        }
                    )

    synthetic = pd.DataFrame(synthetic_rows)
    summary = (
        synthetic.assign(abs_error=synthetic["error"].abs())
        .groupby("variant")
        .agg(
            mae=("abs_error", "mean"),
            bias=("error", "mean"),
            mean_sd=("sd", "mean"),
            coverage=("covered", "mean"),
        )
        .reset_index()
        .sort_values("mae")
    )

    print("\n" + "=" * 86)
    print("SENSITIVITY TO THE ASSUMPTIONS -- synthetic, true rate known")
    print("=" * 86)
    print(f"{'variant':<22}{'rate MAE':>11}{'bias':>11}{'posterior sd':>15}{'coverage':>11}")
    for row in summary.itertuples(index=False):
        print(
            f"{row.variant:<22}{row.mae:>11.4f}{row.bias:>+11.4f}"
            f"{row.mean_sd:>15.4f}{row.coverage:>10.0%}"
        )

    baseline_mae = float(summary[summary["variant"] == "baseline"]["mae"].iloc[0])
    worst = summary.iloc[-1]
    print("-" * 86)
    print(
        f"Worst case is {worst['variant']!r} at {worst['mae']:.4f} s/lap against a "
        f"baseline of {baseline_mae:.4f}."
    )
    print(
        "A full standard deviation of error in the fuel prior -- the single largest\n"
        "assumption in the method -- moves the recovered degradation rate by about\n"
        f"{abs(float(summary[summary['variant'] == 'fuel prior +1sd']['bias'].iloc[0])):.4f} s/lap."
    )

    # ---------- real sessions: robustness ensemble ----------
    real_rows = []
    for session_id in args.real_sessions:
        path = Path("data/demo") / f"{session_id}.parquet"
        if not path.exists():
            continue
        print(f"\n  {session_id}")
        results = run_session(pd.read_parquet(path))
        consensus = build_consensus(results)
        for compound, con in consensus.items():
            print(
                f"    {compound:<8} consensus {con.consensus:+.4f} s/lap  "
                f"spread across assumptions {con.spread:.4f}  agreement {con.agreement:.2f}"
            )
            real_rows.append({"session": session_id, **con.to_dict()})

    print("\n" + "=" * 86)
    print("ROBUSTNESS ON REAL SESSIONS")
    print("=" * 86)
    print(
        "Spread here is across PERTURBED ASSUMPTIONS of the same model, not across\n"
        "different models. Comparing against the naive baselines would only re-measure\n"
        "their known bias; what matters is whether our own conclusion survives\n"
        "someone disagreeing with our priors."
    )
    if real_rows:
        spreads = [r["spread"] for r in real_rows]
        print(f"\nmean spread across assumptions: {np.mean(spreads):.4f} s/lap")
        print(f"largest spread                : {np.max(spreads):.4f} s/lap")
    print("=" * 86)

    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    RESULTS.write_text(
        json.dumps(
            {
                "experiment": "exp02_prior_sensitivity",
                "generated_at": datetime.now(UTC).isoformat(),
                "variants": {k: v[1] for k, v in variants().items()},
                "synthetic_summary": summary.to_dict(orient="records"),
                "synthetic_detail": synthetic.to_dict(orient="records"),
                "real_sessions": real_rows,
            },
            indent=2,
            default=str,
        )
    )
    print(f"\nwrote {RESULTS}")


if __name__ == "__main__":
    main()
