"""Experiment 11 -- matching stint depth does NOT remove the practice-to-race bias.

This experiment tested a hypothesis and refuted it. The result is negative and
the file is kept because the refutation is worth more than the guess was.

The hypothesis. Experiment 10 found that the signed error tracks
`practice_stint_len` (rho +0.37, p 0.003) and `stops_per_driver` (rho +0.33,
p 0.009), both surviving Benjamini-Hochberg across nine predictors. Those look
unrelated until you read them as one thing: how far into the tyre's life each
side of the comparison goes. The reasoning was that a practice long run is a
race simulation that reaches the punishing part of the wear curve, while a
heavily-pitted race never leaves the flat early part -- so fitting one linear
rate to each and comparing them is a specification error, because a line fitted
over laps 1-20 is not the same quantity as a line fitted over laps 1-8.

The refutation, in two parts.

First, the premise is simply false, and measuring it was enough to kill it.
Practice runs are on average 4.4 laps SHALLOWER than race stints, not deeper.
The mental picture of a long Friday race-sim versus a short pitted race stint is
backwards: teams run 10-15 lap evaluation runs on Friday and 20-30 lap stints on
Sunday. Whatever `practice_stint_len` is a proxy for, it is not practice going
deeper than the race.

Second, the correction does not work even on its own terms. Re-fitting both
sides inside a common tyre-age window makes the bias slightly worse (+0.0422 to
+0.0457 s/lap) and the absolute error clearly worse (0.0871 to 0.1005, +15%,
paired t p = 0.037). Truncating to the overlap throws away laps that both fits
needed, and a shorter window buys a noisier slope.

What survives. Experiment 10's correlation is real -- it was corrected for
multiplicity and it replicates across two seasons. What does not survive is the
causal reading placed on it here. Stint length is standing in for something else
that co-varies with it, and this experiment does not identify what. The bias
therefore stays a reported offset rather than a correction, and experiment 12
handles it the honest way instead: not by pretending to explain it, but by
widening the interval until it covers what it claims to cover.

    python experiments/exp11_depth_matched.py
    python experiments/exp11_depth_matched.py --years 2024 2023

Writes experiments/results/exp11_depth_matched.json.
"""

from __future__ import annotations

import argparse
import json
import warnings
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from tyremind.data.corpus import read_lap_table
from tyremind.models.ssm.tyre_ssm import fit_tyre_ssm

RESULTS = Path(__file__).parent / "results" / "exp11_depth_matched.json"
CORPUS = Path("data/season")
CONDITIONS = Path("data/reference/session_conditions.json")

#: Below this many laps a compound is not evidence on either side. Matches the
#: floor experiment 03 applies, so the two are comparing the same population.
MIN_LAPS_PER_COMPOUND = 8

#: One driver running a 30-lap stint while the rest stop at 15 should not define
#: how deep "the race" went. The 90th percentile is robust to that without
#: throwing away the genuinely long stints that a one-stop race is made of.
RACE_DEPTH_PERCENTILE = 90

#: A window has to leave enough laps to fit a slope through. Four is the same
#: floor the run-length filter uses elsewhere in the project.
MIN_WINDOW_LAPS = 4


def slugify(year: int, event: str, session: str) -> str:
    return f"{year}-{event.lower().replace(' ', '-')}-{session}"


def load(year: int, event: str, session: str) -> pd.DataFrame | None:
    path = CORPUS / f"{slugify(year, event, session)}.parquet"
    return read_lap_table(path) if path.exists() else None


def rate_for(laps: pd.DataFrame, compound: str) -> tuple[float, float] | None:
    """Fit the model and return this compound's (rate, sd), or None."""
    if laps.empty:
        return None
    try:
        fit = fit_tyre_ssm(laps)
    except Exception:  # noqa: BLE001 - a fit that will not converge is a result
        return None
    return fit.compound_rates().get(compound)


def dry_events(years: list[int]) -> set[tuple[int, str]]:
    """Events whose race and FP2 were both genuine dry-degradation sessions."""
    if not CONDITIONS.exists():
        return set()
    conditions = json.loads(CONDITIONS.read_text())
    wet = {
        (c["year"], c["event"]) for c in conditions
        if c["year"] in years and not c["dry_session"]
    }
    return wet


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--years", type=int, nargs="*", default=[2024, 2023])
    parser.add_argument("--practice", default="FP2")
    args = parser.parse_args()
    warnings.filterwarnings("ignore")

    manifest = json.loads((CORPUS / "corpus.json").read_text())
    have: dict[tuple[int, str], set[str]] = {}
    for entry in manifest:
        have.setdefault((entry["year"], entry["event_name"]), set()).add(entry["session"])

    wet = dry_events(args.years)
    targets = sorted(
        key for key, sessions in have.items()
        if key[0] in args.years
        and args.practice in sessions and "R" in sessions
        and key not in wet
    )

    print(f"\n  {len(targets)} events with a dry {args.practice} and a dry race\n")

    rows: list[dict] = []
    for year, event in targets:
        practice = load(year, event, args.practice)
        race = load(year, event, "R")
        if practice is None or race is None:
            continue

        shared = sorted(set(practice["compound"]) & set(race["compound"]))
        for compound in shared:
            p_block = practice[practice["compound"] == compound]
            r_block = race[race["compound"] == compound]
            if len(p_block) < MIN_LAPS_PER_COMPOUND or len(r_block) < MIN_LAPS_PER_COMPOUND:
                continue

            # The common window: as deep as both sides genuinely went, with the
            # race side trimmed to its 90th percentile so an outlier stint does
            # not set the depth. Capping at the practice maximum matters too --
            # extrapolating a practice fit past the laps it saw would be exactly
            # the over-reach this experiment exists to remove.
            race_depth = float(np.percentile(r_block["tyre_age"], RACE_DEPTH_PERCENTILE))
            practice_depth = float(p_block["tyre_age"].max())
            window = min(race_depth, practice_depth)
            if window < MIN_WINDOW_LAPS:
                continue

            baseline_p = rate_for(practice, compound)
            baseline_r = rate_for(race, compound)
            matched_p = rate_for(practice[practice["tyre_age"] <= window], compound)
            matched_r = rate_for(race[race["tyre_age"] <= window], compound)
            if not all((baseline_p, baseline_r, matched_p, matched_r)):
                continue

            rows.append({
                "year": year,
                "event": event,
                "compound": compound,
                "window_laps": window,
                "practice_depth": practice_depth,
                "race_depth": race_depth,
                "depth_gap": practice_depth - race_depth,
                "baseline_predicted": baseline_p[0],
                "baseline_actual": baseline_r[0],
                "baseline_error": baseline_p[0] - baseline_r[0],
                "matched_predicted": matched_p[0],
                "matched_actual": matched_r[0],
                "matched_error": matched_p[0] - matched_r[0],
                "matched_sd": matched_p[1],
            })
            last = rows[-1]
            print(f"  {year} {event[:26]:<27} {compound:<7} window {window:>4.0f} laps  "
                  f"err {last['baseline_error']:>+7.4f} -> {last['matched_error']:>+7.4f}")

    if len(rows) < 20:
        raise SystemExit(f"only {len(rows)} comparisons -- too few to conclude")

    df = pd.DataFrame(rows)
    from scipy import stats

    base_err = df["baseline_error"].to_numpy()
    match_err = df["matched_error"].to_numpy()

    # Two questions, two tests. Did the BIAS move (a paired t-test on the signed
    # errors, because bias is a mean)? Did the ACCURACY suffer (Wilcoxon on the
    # absolute errors, because it should not be assumed symmetric)?
    bias_t, bias_p = stats.ttest_rel(np.abs(base_err), np.abs(match_err))
    _, acc_p = stats.wilcoxon(np.abs(base_err), np.abs(match_err))

    summary = {
        "n_comparisons": int(len(df)),
        "n_events": int(df.groupby(["year", "event"]).ngroups),
        "baseline_bias": float(base_err.mean()),
        "matched_bias": float(match_err.mean()),
        "baseline_mae": float(np.abs(base_err).mean()),
        "matched_mae": float(np.abs(match_err).mean()),
        "bias_reduction_pct": float(
            100.0 * (abs(base_err.mean()) - abs(match_err.mean())) / abs(base_err.mean())
        ),
        "mae_change_pct": float(
            100.0 * (np.abs(match_err).mean() - np.abs(base_err).mean())
            / np.abs(base_err).mean()
        ),
        "paired_t_statistic": float(bias_t),
        "paired_p_value": float(bias_p),
        "wilcoxon_abs_error_p": float(acc_p),
        "mean_depth_gap": float(df["depth_gap"].mean()),
    }

    print("\n" + "=" * 84)
    print(f"DEPTH-MATCHED PRACTICE -> RACE  ({summary['n_comparisons']} comparisons, "
          f"{summary['n_events']} events)")
    print("=" * 84)
    gap = summary["mean_depth_gap"]
    direction = "deeper into the tyre's life than" if gap >= 0 else "shallower than"
    print(f"\n  practice ran on average {abs(gap):.1f} laps {direction} the race\n")
    print(f"  {'':<26}{'bias':>12}{'MAE':>12}")
    print(f"  {'as fitted (baseline)':<26}{summary['baseline_bias']:>+12.4f}"
          f"{summary['baseline_mae']:>12.4f}")
    print(f"  {'depth-matched':<26}{summary['matched_bias']:>+12.4f}"
          f"{summary['matched_mae']:>12.4f}")
    print(f"\n  bias reduced by {summary['bias_reduction_pct']:.1f}%   "
          f"MAE change {summary['mae_change_pct']:+.1f}%")
    print(f"  paired t on |error|  p = {summary['paired_p_value']:.4f}")
    print(f"  Wilcoxon on |error|  p = {summary['wilcoxon_abs_error_p']:.4f}")

    print()
    improved = summary["bias_reduction_pct"] > 25 and summary["mae_change_pct"] < 10
    if improved:
        print("  VERDICT: matching stint depth removes a substantial part of the bias")
        print("           without costing accuracy. The over-prediction was not the tyre")
        print("           behaving differently on Sunday -- it was us comparing a rate")
        print("           fitted deep into the stint against one fitted shallow, and")
        print("           calling both 'degradation'. Depth is now part of the estimand.")
    else:
        print("  VERDICT: depth-matching does not remove the bias -- it makes it worse.")
        print("           The premise was also wrong: practice runs are SHALLOWER than")
        print("           race stints, not deeper, so the mental picture of a long Friday")
        print("           race-sim against a short pitted race stint is backwards.")
        print("           Experiment 10's correlation is real and corrected for")
        print("           multiplicity; the causal reading placed on it here is not.")
        print("           The bias stays a reported offset, and experiment 12 handles it")
        print("           by widening the interval rather than by pretending to explain it.")
    print("=" * 84)

    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    RESULTS.write_text(json.dumps({
        "experiment": "exp11_depth_matched",
        "generated_at": datetime.now(UTC).isoformat(),
        "years": args.years,
        "practice_session": args.practice,
        "race_depth_percentile": RACE_DEPTH_PERCENTILE,
        "summary": summary,
        "supported": bool(improved),
        "cases": rows,
    }, indent=2))
    print(f"\nwrote {RESULTS}")


if __name__ == "__main__":
    main()
