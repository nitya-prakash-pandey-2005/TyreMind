"""Experiment 09 -- can we predict a circuit we have never run?

Everything TyreMind does today needs the car to have already run. A team's real
question arrives on Thursday: we get here tomorrow, what will the tyres do? That
is a different problem, and it is the one that separates an analysis tool from a
product.

The test is leave-one-CIRCUIT-out, not leave-one-event-out. Holding out a circuit
means the model has never seen that venue in any form, which is exactly the
situation on Thursday.

Four predictors, each harder to beat than the last:

  global mean      every stint gets the same number. The floor.
  label mean       what exp08 established as surprisingly strong, because Pirelli
                   nominates to equalise and the label already carries severity.
  label + circuit  the label, plus what the geometry says this venue asks of a tyre.
  compound + circuit   the same, on the compound actually fitted.

The last two are the interesting ones. If neither beats the label mean, then
circuit geometry adds nothing we do not already get for free from Pirelli's
nomination -- which would be worth knowing, and worth saying.

An ablation runs alongside, scoring each feature family on its own through the
identical procedure. The headline test throws geometry and temperature in
together and so cannot say which is carrying the result -- or, as it turns out,
which is carrying the damage. It also answers a question asked directly of the
thermal layer elsewhere in this project: does it improve a number, or is it
decoration? Every family, including temperature on its own, makes the label-mean
baseline significantly WORSE.

    python experiments/exp09_circuit_transfer.py

Writes experiments/results/exp09_circuit_transfer.json.
"""

from __future__ import annotations

import argparse
import json
import statistics as st
import warnings
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

RESULTS = Path(__file__).parent / "results" / "exp09_circuit_transfer.json"
ESTIMATES = Path(__file__).parent / "results" / "exp08_compound_identity.json"
FEATURES = Path("data/reference/circuit_features.json")
CONDITIONS = Path("data/reference/session_conditions.json")

#: Geometric descriptors available without any telemetry download. Deliberately
#: few: 58 observations cannot support a wide model, and a ridge that memorises
#: circuits would pass every in-sample check and fail the only test that matters.
GEOMETRIC = ["n_corners", "n_tight_corners", "mean_corner_angle", "median_corner_spacing_m"]

#: Energetic descriptors, present only once telemetry has been pulled.
ENERGETIC = ["mean_abs_lateral_g", "p95_lateral_g", "loaded_fraction", "lateral_energy_proxy"]

#: Thermal descriptors. Rubber wear is strongly temperature-dependent and the
#: dependence is NOT monotone -- our own wear model puts the multiplier up on both
#: sides of the working window, because a cold tyre is slid and torn (graining)
#: while a hot one abrades. A linear term cannot express that, so the squared
#: term is included and the ridge is allowed to find the curvature. Las Vegas ran
#: at 17.7 C and degraded fastest all season; Monza ran at 49.3 C and did not.
THERMAL = ["track_temp_c", "track_temp_sq"]


def load_frame() -> tuple[pd.DataFrame, list[str]]:
    """Join per-stint rate estimates to their circuit's description."""
    if not ESTIMATES.exists():
        raise SystemExit("run exp08_compound_identity.py first -- it produces the rate estimates")
    if not FEATURES.exists():
        raise SystemExit("run scripts/build_circuit_features.py first")

    estimates = pd.DataFrame(json.loads(ESTIMATES.read_text())["estimates"])
    features = pd.DataFrame(json.loads(FEATURES.read_text()))

    merged = estimates.merge(
        features, left_on=["year", "round"], right_on=["year", "round"], how="inner",
        suffixes=("", "_circuit"),
    )

    # Drop sessions that were not dry-degradation sessions at all. Canada 2024 ran
    # two thirds of its laps on wet rubber and yielded -0.18 s/lap: the model was
    # reading a drying track as a tyre improving. Keeping it would let one broken
    # session set the scale every predictor is scored against.
    if CONDITIONS.exists():
        conditions = pd.DataFrame(json.loads(CONDITIONS.read_text()))
        # The conditions file holds a row per SESSION, so an event with both a
        # race and an FP2 row matches twice on (year, round). Left-joining that
        # silently duplicated every race stint, doubled the apparent sample and
        # attached Friday afternoon's track temperature to Sunday's stints. The
        # rate estimates being joined are fitted from races, so only the race row
        # is the right weather.
        conditions = conditions[conditions["session"] == "R"]
        merged = merged.merge(
            conditions[["year", "round", "track_temp_c", "air_temp_c", "humidity_pct",
                        "dry_session", "rain_advisory"]],
            on=["year", "round"], how="left", validate="many_to_one",
        )
        before = len(merged)
        merged = merged[merged["dry_session"].fillna(True)]
        if before != len(merged):
            print(f"  dropped {before - len(merged)} stint estimates from wet sessions")
        merged["track_temp_sq"] = merged["track_temp_c"] ** 2

    # Use the energetic features only if they are present for every circuit;
    # a column that is null for half the venues would silently drop those rows.
    available = list(GEOMETRIC)
    if all(c in merged.columns and merged[c].notna().all() for c in THERMAL):
        available += THERMAL
    if all(c in merged.columns and merged[c].notna().all() for c in ENERGETIC):
        available += ENERGETIC
    merged = merged.dropna(subset=available + ["rate"])
    return merged, available


def leave_one_circuit_out(df: pd.DataFrame, feature_cols: list[str]) -> dict:
    """Score four predictors, each held out one whole venue at a time."""
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler

    circuits = sorted(df["circuit"].unique())
    rows: list[dict] = []

    for held in circuits:
        train = df[df["circuit"] != held]
        test = df[df["circuit"] == held]
        if train.empty or test.empty:
            continue

        global_mean = float(train["rate"].mean())
        label_means = train.groupby("label")["rate"].mean().to_dict()

        # Bind the fold's frames explicitly. Closing over the loop variable works
        # here only because these are called immediately, which is exactly the
        # kind of accident that breaks when someone later defers the call.
        def fit_ridge(group_col: str, fold: pd.DataFrame = train) -> tuple:
            design = pd.get_dummies(fold[group_col], prefix=group_col, dtype=float)
            columns = list(design.columns)
            X = np.column_stack([design.to_numpy(float), fold[feature_cols].to_numpy(float)])
            scaler = StandardScaler().fit(X)
            # Ridge rather than OLS: the circuit descriptors are correlated with
            # one another (tight corners come with more corners) and 58 rows will
            # not separate them. Shrinkage keeps the fit honest.
            model = Ridge(alpha=5.0).fit(scaler.transform(X), fold["rate"].to_numpy(float))
            return model, scaler, columns

        def predict_ridge(group_col: str, bundle, fold: pd.DataFrame = test) -> np.ndarray:
            model, scaler, columns = bundle
            design = pd.get_dummies(fold[group_col], prefix=group_col, dtype=float)
            design = design.reindex(columns=columns, fill_value=0.0)
            X = np.column_stack([design.to_numpy(float), fold[feature_cols].to_numpy(float)])
            return model.predict(scaler.transform(X))

        label_ridge = predict_ridge("label", fit_ridge("label"))
        compound_ridge = predict_ridge("compound_id", fit_ridge("compound_id"))

        for i, (_, r) in enumerate(test.iterrows()):
            rows.append(
                {
                    "circuit": held,
                    "event": r["event"],
                    "label": r["label"],
                    "compound_id": r["compound_id"],
                    "actual": float(r["rate"]),
                    "global_mean": global_mean,
                    "label_mean": float(label_means.get(r["label"], global_mean)),
                    "label_plus_circuit": float(label_ridge[i]),
                    "compound_plus_circuit": float(compound_ridge[i]),
                }
            )

    return {"n": len(rows), "cases": rows}


def score(cases: list[dict]) -> dict:
    """MAE per predictor, plus a paired test against the label-mean baseline."""
    from scipy import stats

    predictors = ["global_mean", "label_mean", "label_plus_circuit", "compound_plus_circuit"]
    errors = {p: [abs(c[p] - c["actual"]) for c in cases] for p in predictors}
    baseline = errors["label_mean"]

    out: dict[str, dict] = {}
    for p in predictors:
        e = errors[p]
        entry = {
            "mae": st.fmean(e),
            "rmse": (st.fmean(x * x for x in e)) ** 0.5,
            "vs_label_mean_pct": 100.0 * (st.fmean(baseline) - st.fmean(e)) / st.fmean(baseline),
        }
        if p != "label_mean":
            _, p_value = stats.wilcoxon(e, baseline)
            entry["wilcoxon_p_vs_label_mean"] = float(p_value)
            entry["beats_baseline"] = bool(st.fmean(e) < st.fmean(baseline) and p_value < 0.05)
        out[p] = entry
    return out


def ablation(df: pd.DataFrame, available: list[str]) -> dict:
    """Score each feature family on its own, against the label-mean baseline.

    The headline test throws geometry and temperature in together, which cannot
    say which of them is carrying the result -- or, as it turns out, which is
    carrying the damage. The roadmap asks a specific question of the thermal
    layer: does it improve a number, or is it decoration? Bundling it with four
    geometric features is not an answer to that question.

    Each family is run through the identical leave-one-circuit-out procedure, so
    the only thing that changes between rows is which descriptors the ridge is
    allowed to see.
    """
    from scipy import stats

    families = {
        "geometry only": [c for c in GEOMETRIC if c in available],
        "thermal only": [c for c in THERMAL if c in available],
        "energetics only": [c for c in ENERGETIC if c in available],
        "everything": list(available),
    }

    out: dict[str, dict] = {}
    for name, columns in families.items():
        if not columns:
            continue
        cases = leave_one_circuit_out(df, columns)["cases"]
        if len(cases) < 20:
            continue
        errors = [abs(c["label_plus_circuit"] - c["actual"]) for c in cases]
        baseline = [abs(c["label_mean"] - c["actual"]) for c in cases]
        _, p_value = stats.wilcoxon(errors, baseline)
        out[name] = {
            "features": columns,
            "mae": st.fmean(errors),
            "baseline_mae": st.fmean(baseline),
            "vs_label_mean_pct": 100.0 * (st.fmean(baseline) - st.fmean(errors)) / st.fmean(baseline),
            "wilcoxon_p": float(p_value),
            "helps": bool(st.fmean(errors) < st.fmean(baseline) and p_value < 0.05),
            "hurts": bool(st.fmean(errors) > st.fmean(baseline) and p_value < 0.05),
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--alpha", type=float, default=5.0)
    parser.parse_args()
    warnings.filterwarnings("ignore")

    df, feature_cols = load_frame()
    print(f"\n  {len(df)} stint estimates across {df['circuit'].nunique()} circuits")
    print(f"  circuit descriptors: {', '.join(feature_cols)}\n")

    loco = leave_one_circuit_out(df, feature_cols)
    if loco["n"] < 20:
        raise SystemExit(f"only {loco['n']} scored cases -- too few to conclude")
    scores = score(loco["cases"])

    print("=" * 82)
    print(f"PREDICTING A CIRCUIT NEVER RUN  ({loco['n']} stints, leave-one-circuit-out)")
    print("=" * 82)
    print(f"\n  {'predictor':<26} {'MAE':>9} {'RMSE':>9} {'vs label':>10} {'p':>8}")
    for name, s in scores.items():
        p = s.get("wilcoxon_p_vs_label_mean")
        print(f"  {name:<26} {s['mae']:>9.4f} {s['rmse']:>9.4f} "
              f"{s['vs_label_mean_pct']:>+9.1f}% {('' if p is None else f'{p:>8.3f}')}")

    families = ablation(df, feature_cols)
    if families:
        print("\n  Each family on its own, same leave-one-circuit-out procedure:")
        print(f"  {'feature family':<20}{'MAE':>9}{'vs label':>11}{'p':>9}   verdict")
        for name, f in families.items():
            verdict = "helps" if f["helps"] else ("HURTS" if f["hurts"] else "no effect")
            print(f"  {name:<20}{f['mae']:>9.4f}{f['vs_label_mean_pct']:>+10.1f}%"
                  f"{f['wilcoxon_p']:>9.3f}   {verdict}")

    best = min(scores.items(), key=lambda kv: kv[1]["mae"])
    winner = [n for n, s in scores.items() if s.get("beats_baseline")]
    # A predictor that is significantly WORSE than the baseline is not a null
    # result. It is evidence that the extra structure is fitting noise, and it
    # deserves to be reported as the finding it is.
    harmful = [
        n for n, s in scores.items()
        if s.get("wilcoxon_p_vs_label_mean", 1.0) < 0.05
        and s["mae"] > scores["label_mean"]["mae"]
    ]
    print()
    if winner:
        print(f"  VERDICT: {', '.join(winner)} beats the label-mean baseline significantly.")
        print("           Circuit description carries information the nomination does not.")
    elif harmful:
        print(f"  VERDICT: the label mean wins ({best[1]['mae']:.4f} s/lap), and adding circuit")
        print(f"           description makes it significantly WORSE: {', '.join(harmful)}.")
        print("           This is a real negative result, not an underpowered one. Pirelli")
        print("           nominates to equalise -- harder rubber for abrasive venues -- so the")
        print("           weekend label already carries circuit severity. Geometry on top of")
        print("           it re-fits severity that is spoken for, and overfits.")
    else:
        print(f"  VERDICT: nothing beats the label mean significantly. Lowest MAE is "
              f"{best[0]} ({best[1]['mae']:.4f}),")
        print("           but not by enough to claim.")
    print("=" * 82)

    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    RESULTS.write_text(
        json.dumps(
            {
                "experiment": "exp09_circuit_transfer",
                "generated_at": datetime.now(UTC).isoformat(),
                "n_stints": loco["n"],
                "n_circuits": int(df["circuit"].nunique()),
                "feature_columns": feature_cols,
                "has_energetics": any(c in feature_cols for c in ENERGETIC),
                "scores": scores,
                "harmful_predictors": harmful,
                "feature_family_ablation": families,
                "cases": loco["cases"],
            },
            indent=2,
        )
    )
    print(f"\nwrote {RESULTS}")


if __name__ == "__main__":
    main()
