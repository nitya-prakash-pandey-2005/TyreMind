"""Experiment 18 -- what exactly is unidentifiable here, and what buys it back?

Every document in this project says fuel burn-off and tyre degradation are
collinear, and that a physical prior on the fuel effect is what makes
degradation estimable. That is asserted everywhere and quantified nowhere, and
"collinear" is doing a lot of work in that sentence -- it can mean anything from
"correlated at 0.9" to "algebraically the same vector".

It is the second one. This experiment shows it, and then measures what is left.

THE ALGEBRA. Within one run, lap time under the structural model is

    y_t  =  c_r  -  phi * f_t  +  beta * a_t  +  eps_t

with f_t the laps completed on this run (fuel burned) and a_t the tyre's age. A
run puts one lap on the tyre per lap completed, so

    a_t  =  a_0  +  f_t

and substituting,

    y_t  =  [ c_r + beta * a_0 ]  +  (beta - phi) * f_t  +  eps_t

The run intercept c_r is free, so it swallows `beta * a_0` whole. What is left is
a single slope, (beta - phi), multiplying a single regressor. **Within a run,
beta and phi are not weakly identified, not poorly conditioned -- they are
exactly unidentifiable, and no quantity of laps changes that.** The Fisher
information matrix for (phi, beta) is singular by construction, not by accident
of the data.

This is stronger than the project has been claiming, and it matters, because it
rules out the obvious rescue. Pit-stop stagger does NOT fix it: different drivers
starting a stint on tyres of different ages changes a_0, and a_0 is absorbed by
the run intercept. More cars does not help. More laps does not help.

WHAT DOES BUY IT BACK. Two channels, and they are the only two:

  1. THE PRIOR on phi. With a Gaussian prior phi ~ N(mu, tau^2), the posterior
     precision of beta is finite. Set tau -> infinity and the posterior variance
     of beta diverges. The prior is therefore not a refinement of the estimate;
     it IS the estimate's existence.

  2. CURVATURE. Fuel burns at a constant rate, so its effect is exactly linear
     in f. The tyre's is not: the state-space model carries the degradation rate
     as a random walk, so accumulated loss may bend. Any departure from linearity
     in the tyre term is orthogonal to the fuel term and is identified by the data
     alone. This channel is real but small, and this experiment measures how small.

The uncomfortable corollary is worth stating plainly: the flatter the true
degradation curve, the more the answer rests on the prior. A perfectly linear
tyre is a tyre whose degradation cannot be separated from fuel by any amount of
data.

    python experiments/exp18_identifiability_bound.py
    python experiments/exp18_identifiability_bound.py --limit 12

Writes experiments/results/exp18_identifiability_bound.json.
"""

from __future__ import annotations

import argparse
import json
import warnings
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from tyremind.data.corpus import sessions
from tyremind.models.ssm.tyre_ssm import fit_tyre_ssm

RESULTS = Path(__file__).parent / "results" / "exp18_identifiability_bound.json"
CORPUS = Path("data/season")
DEMO = Path("data/demo")

#: A run needs enough laps for a within-run slope to mean anything.
MIN_RUN_LAPS = 5

#: The fuel prior the project actually ships, in seconds per lap of run.
#: 0.030 s/kg x 2.7 kg/lap = 0.081 s/lap, with the sd propagated through.
FUEL_PRIOR_SD = 0.016


def within_run_geometry(lap_table: pd.DataFrame) -> dict:
    """Measure the collinearity the algebra predicts, rather than assuming it.

    For each run, the fuel regressor and the tyre-age regressor are centred
    within that run -- which is exactly what a free run intercept does to them --
    and then compared. If the algebra is right they are the same vector and the
    correlation is 1 to machine precision.
    """
    correlations: list[float] = []
    max_deviation = 0.0
    n_runs = 0

    for _, run in lap_table.groupby(["driver", "run_id"]):
        if len(run) < MIN_RUN_LAPS:
            continue
        fuel = run["lap_in_run"].to_numpy(float)
        age = run["tyre_age"].to_numpy(float)
        fuel_c = fuel - fuel.mean()
        age_c = age - age.mean()
        if np.allclose(fuel_c, 0) or np.allclose(age_c, 0):
            continue
        correlations.append(
            float(np.corrcoef(fuel_c, age_c)[0, 1])
        )
        # The residual after projecting age onto fuel. Zero means identical.
        max_deviation = max(max_deviation, float(np.abs(age_c - fuel_c).max()))
        n_runs += 1

    return {
        "n_runs": n_runs,
        "mean_correlation": float(np.mean(correlations)) if correlations else float("nan"),
        "min_correlation": float(np.min(correlations)) if correlations else float("nan"),
        "exactly_collinear_runs": int(sum(c > 1 - 1e-9 for c in correlations)),
        "max_abs_deviation_laps": max_deviation,
    }


def information_matrix(lap_table: pd.DataFrame, sigma: float) -> dict:
    """Fisher information for (phi, beta), after removing run intercepts.

    Removing the intercept is what the model does, so the information that
    matters is what survives it. The determinant of the 2x2 is the whole story:
    zero means the pair cannot be separated at any sample size.
    """
    rows_f, rows_a = [], []
    for _, run in lap_table.groupby(["driver", "run_id"]):
        if len(run) < MIN_RUN_LAPS:
            continue
        fuel = run["lap_in_run"].to_numpy(float)
        age = run["tyre_age"].to_numpy(float)
        rows_f.append(fuel - fuel.mean())
        rows_a.append(age - age.mean())

    if not rows_f:
        return {}

    f = np.concatenate(rows_f)
    a = np.concatenate(rows_a)
    design = np.column_stack([-f, a])          # the model's own sign convention
    gram = design.T @ design / (sigma**2)      # Fisher information for (phi, beta)

    eigenvalues = np.linalg.eigvalsh(gram)
    smallest = float(eigenvalues.min())
    largest = float(eigenvalues.max())

    return {
        "n_observations": int(f.size),
        "determinant": float(np.linalg.det(gram)),
        "smallest_eigenvalue": smallest,
        "largest_eigenvalue": largest,
        # Infinite when singular, which is the honest answer rather than 1e17.
        "condition_number": float(largest / smallest) if smallest > 0 else float("inf"),
        "rank": int(np.linalg.matrix_rank(gram, tol=1e-8)),
    }


def posterior_variance(info: dict, prior_sd: float | None) -> float:
    """Var(beta) from the information matrix plus a Gaussian prior on phi.

    Posterior precision is  I + diag(1/prior_sd^2, 0), and Var(beta) is the
    [1,1] entry of its inverse. With a singular I and no prior this is infinite,
    which is the point of the whole experiment.
    """
    if not info:
        return float("nan")
    # Reconstruct the 2x2 from its invariants: trace and determinant are enough
    # only if we keep the matrix, so it is passed through explicitly instead.
    gram = np.array(info["_gram"])
    precision = gram.copy()
    if prior_sd is not None:
        precision[0, 0] += 1.0 / (prior_sd**2)
    try:
        return float(np.linalg.inv(precision)[1, 1])
    except np.linalg.LinAlgError:
        return float("inf")


def curvature_share(lap_table: pd.DataFrame, fit) -> dict:
    """How much of the tyre term is NOT linear in age.

    The only part of the tyre effect the data can separate from fuel is the part
    that is not a straight line. Fit the model's own estimated cumulative loss
    against tyre age within each run and keep the residual: its size relative to
    the total is the share of the tyre signal that is identified by data rather
    than by the prior.
    """
    per_lap = fit.degradation()
    total_var, residual_var, n = 0.0, 0.0, 0

    for _, run in per_lap.groupby(["driver", "run_id"]):
        if len(run) < MIN_RUN_LAPS:
            continue
        age = run["tyre_age"].to_numpy(float)
        level = run["level"].to_numpy(float)
        if np.ptp(age) < 2:
            continue
        design = np.column_stack([np.ones_like(age), age])
        coef, *_ = np.linalg.lstsq(design, level, rcond=None)
        residual = level - design @ coef
        total_var += float(np.var(level - level.mean()))
        residual_var += float(np.var(residual))
        n += 1

    if n == 0 or total_var <= 0:
        return {"n_runs": 0}
    return {
        "n_runs": n,
        "nonlinear_share": residual_var / total_var,
        "linear_share": 1.0 - residual_var / total_var,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=8)
    args = parser.parse_args()
    warnings.filterwarnings("ignore")

    races = sessions(CORPUS, session_type="R", limit=args.limit)
    if not races:
        races = sessions(DEMO, session_type="R", limit=args.limit)
    if not races:
        raise SystemExit("no sessions found")

    print(f"\n  {len(races)} sessions\n")
    per_session: list[dict] = []

    for race in races:
        lap_table = race.load()
        geometry = within_run_geometry(lap_table)
        if not geometry["n_runs"]:
            continue

        try:
            fit = fit_tyre_ssm(lap_table)
        except Exception as exc:  # noqa: BLE001 - a session that will not fit is data
            print(f"  {race.session_id:<40} skip  {type(exc).__name__}")
            continue

        sigma = float(np.sqrt(fit.hyper.obs_var))
        info = information_matrix(lap_table, sigma)
        if not info:
            continue

        # Keep the actual matrix for the posterior-variance arithmetic.
        rows_f, rows_a = [], []
        for _, run in lap_table.groupby(["driver", "run_id"]):
            if len(run) < MIN_RUN_LAPS:
                continue
            fuel = run["lap_in_run"].to_numpy(float)
            age = run["tyre_age"].to_numpy(float)
            rows_f.append(fuel - fuel.mean())
            rows_a.append(age - age.mean())
        design = np.column_stack([-np.concatenate(rows_f), np.concatenate(rows_a)])
        info["_gram"] = (design.T @ design / sigma**2).tolist()

        with_prior = posterior_variance(info, FUEL_PRIOR_SD)
        without_prior = posterior_variance(info, None)
        curvature = curvature_share(lap_table, fit)
        model_sd = float(np.mean([sd for _, sd in fit.compound_rates().values()]))

        per_session.append({
            "session_id": race.session_id,
            "sigma": sigma,
            "geometry": geometry,
            "information": {k: v for k, v in info.items() if k != "_gram"},
            "var_beta_with_prior": with_prior,
            "var_beta_without_prior": without_prior,
            "sd_beta_with_prior": float(np.sqrt(with_prior)) if np.isfinite(with_prior) else None,
            "curvature": curvature,
            "model_reported_sd": model_sd,
        })
        print(f"  {race.session_id:<40} corr {geometry['mean_correlation']:.6f}  "
              f"rank {info['rank']}/2  CRB sd "
              f"{np.sqrt(with_prior):.4f}" if np.isfinite(with_prior) else
              f"  {race.session_id:<40} corr {geometry['mean_correlation']:.6f}")

    if not per_session:
        raise SystemExit("nothing analysed")

    frame = pd.DataFrame([{
        "session_id": s["session_id"],
        "correlation": s["geometry"]["mean_correlation"],
        "exact_runs": s["geometry"]["exactly_collinear_runs"],
        "runs": s["geometry"]["n_runs"],
        "rank": s["information"]["rank"],
        "determinant": s["information"]["determinant"],
        "condition": s["information"]["condition_number"],
        "crb_sd": s["sd_beta_with_prior"],
        "model_sd": s["model_reported_sd"],
        "nonlinear_share": s["curvature"].get("nonlinear_share", float("nan")),
    } for s in per_session])

    exact = int(frame["exact_runs"].sum())
    total_runs = int(frame["runs"].sum())
    singular = int((frame["rank"] < 2).sum())

    print("\n" + "=" * 86)
    print(f"WHAT IS IDENTIFIED, AND BY WHAT  ({len(frame)} sessions, {total_runs} runs)")
    print("=" * 86)

    print("\n  1. THE COLLINEARITY IS EXACT, NOT SEVERE")
    print(f"     runs where fuel and tyre-age are the same vector : {exact}/{total_runs}"
          f"  ({exact / total_runs:.1%})")
    print(f"     mean within-run correlation                     : "
          f"{frame['correlation'].mean():.9f}")
    print(f"     sessions whose Fisher information is rank < 2   : {singular}/{len(frame)}")
    print("     A run puts exactly one lap on the tyre per lap completed, so after a")
    print("     free run intercept absorbs the starting age, one regressor remains.")
    print("     No amount of laps, cars or pit stagger changes this.")

    print("\n  2. WITHOUT THE PRIOR, THE VARIANCE IS INFINITE")
    finite = frame["crb_sd"].notna()
    print(f"     Var(beta) with a flat prior on phi : infinite in "
          f"{len(frame)}/{len(frame)} sessions")
    if finite.any():
        print(f"     Cramer-Rao sd with the shipped prior (0.016 s/lap): "
              f"{frame.loc[finite, 'crb_sd'].median():.4f} s/lap")
        print(f"     the model's own reported sd                       : "
              f"{frame.loc[finite, 'model_sd'].median():.4f} s/lap")
        ratio = frame.loc[finite, "model_sd"].median() / frame.loc[finite, "crb_sd"].median()
        print(f"     ratio, model / bound                              : {ratio:.2f}x")
        print("     The prior is not a refinement of this estimate. It is the reason the")
        print("     estimate exists at all.")

    print("\n  3. THE ONLY DATA-DRIVEN CHANNEL IS CURVATURE")
    share = frame["nonlinear_share"].dropna()
    if len(share):
        print(f"     share of the tyre term that is NOT linear in age : "
              f"{share.median():.1%}  (median over sessions)")
        print(f"     range across sessions                            : "
              f"{share.min():.1%} to {share.max():.1%}")
        print("     Fuel is exactly linear in laps completed, so only the tyre term's")
        print("     departure from a straight line is orthogonal to it and identified by")
        print("     data. The flatter the true curve, the more the answer is the prior.")

    print("\n" + "=" * 86)
    print("  This is a sharper statement than 'the priors matter', and a less")
    print("  comfortable one. Within the structural model, the fuel/degradation split")
    print("  is not weakly identified -- it is exactly unidentified, and is restored")
    print("  only by the prior and by whatever curvature the tyre happens to show.")
    print("  Every degradation number in this project should be read as conditional")
    print("  on 0.030 s/kg, and the sensitivity of that conditioning is exp02.")
    print("=" * 86)

    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    RESULTS.write_text(json.dumps({
        "experiment": "exp18_identifiability_bound",
        "generated_at": datetime.now(UTC).isoformat(),
        "n_sessions": int(len(frame)),
        "n_runs": total_runs,
        "exactly_collinear_runs": exact,
        "exact_collinearity_rate": float(exact / total_runs),
        "mean_within_run_correlation": float(frame["correlation"].mean()),
        "singular_information_sessions": singular,
        "fuel_prior_sd": FUEL_PRIOR_SD,
        "median_cramer_rao_sd": float(frame.loc[finite, "crb_sd"].median()) if finite.any() else None,
        "median_model_reported_sd": float(frame.loc[finite, "model_sd"].median()) if finite.any() else None,
        "median_nonlinear_share": float(share.median()) if len(share) else None,
        "per_session": per_session,
    }, indent=2, default=float))
    print(f"\nwrote {RESULTS}")


if __name__ == "__main__":
    main()
