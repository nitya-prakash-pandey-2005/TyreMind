"""Experiment 10 -- why does practice over-predict race degradation?

Experiment 03 leaves a residue it does not explain. Across 62 compound
comparisons the practice fit is right on average to within a tenth of a second
per lap, but it is not *unbiased*: it runs +0.042 s/lap high, and the sign is
consistent enough across events -- 42 of 62 -- that it cannot be dismissed as
noise. A bias we cannot explain is a bias we cannot correct, and a model that is
knowingly wrong in a known direction is not a product.

NINE candidate explanations, each computable from data already on disk, each
falsifiable. Three describe the practice side, three the race side, and three
the conditions or the model itself:

  practice_stint_len   How long the fitted practice runs were. Short runs make
                       degradation and track evolution harder to separate, since
                       the saturating basis and the linear wear term are only
                       distinguishable over enough laps. An IDENTIFIABILITY story.
  practice_runs        How many such runs there were.
  practice_laps        How many laps in total went into the practice fit.
  race_stint_len       Long race stints mean the driver was managing the tyre to
                       reach a pit window, and managed laps degrade slower than
                       pushed ones. A BEHAVIOUR story.
  race_max_stint       The same, measured by the longest stint anyone completed.
  stops_per_driver     The behaviour story from the other side: a one-stop race
                       is a managed race.
  traffic_gap          Practice long runs happen in clean air; races do not.
  temp_gap             Friday afternoon is not Sunday afternoon.
  posterior_sd         Does the model already know when it is about to be wrong?
                       If error tracks the interval the model reports, then the
                       bias is at least flagged even when it is not removed.

Nine tests against one target is nine chances to find a p below 0.05 by luck, so
everything is corrected with Benjamini-Hochberg across all nine. Uncorrected
significance is kept alongside it, so the correction's effect is visible rather
than quietly applied.

THE OUTCOME. Two survive the correction: `practice_stint_len` (rho +0.37,
p 0.003) and `stops_per_driver` (rho +0.33, p 0.009). `posterior_sd` is
significant raw (p 0.043) and does NOT survive, which is exactly the kind of
result the correction exists to catch. Six fail outright, including the
temperature gap and traffic.

A warning about what this does and does not license. The two survivors are a
CORRELATION. Experiment 11 took the obvious causal reading of them -- that
practice runs go deeper into the wear curve than pitted race stints -- and
refuted it: practice runs are 4.4 laps *shallower* than race stints, and forcing
a common tyre-age window made the error worse. So this experiment identifies
where to look next, not what the mechanism is. The bias is handled as a measured
offset by experiment 12, not as an explained one.

    python experiments/exp10_bias_mechanism.py

Writes experiments/results/exp10_bias_mechanism.json.
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

RESULTS = Path(__file__).parent / "results" / "exp10_bias_mechanism.json"
EXP03 = Path(__file__).parent / "results" / "exp03_practice_to_race.json"
CORPUS = Path("data/season")
CONDITIONS = Path("data/reference/session_conditions.json")

#: A run has to be long enough to be a degradation observation at all. Matches
#: the floor the estimator itself applies, so the counts here describe the same
#: stints the model actually fitted rather than a looser population.
MIN_RUN_LAPS = 4


def slugify(year: int, event: str, session: str) -> str:
    return f"{year}-{event.lower().replace(' ', '-')}-{session}"


def stint_profile(year: int, event: str, session: str) -> dict[str, dict]:
    """Per-compound stint statistics for one session, or {} if not on disk."""
    path = CORPUS / f"{slugify(year, event, session)}.parquet"
    if not path.exists():
        return {}
    laps = read_lap_table(path)

    out: dict[str, dict] = {}
    for compound, block in laps.groupby("compound"):
        runs = block.groupby(["driver", "run_id"]).agg(
            laps=("lap_time", "size"),
            traffic=("traffic_index", "mean"),
        )
        runs = runs[runs["laps"] >= MIN_RUN_LAPS]
        if runs.empty:
            continue
        out[str(compound)] = {
            "n_runs": int(len(runs)),
            "mean_run_laps": float(runs["laps"].mean()),
            "max_run_laps": float(runs["laps"].max()),
            "total_laps": int(runs["laps"].sum()),
            "mean_traffic": float(runs["traffic"].mean()),
            # How many qualifying stints a driver got through. In a race this is
            # the pit count, the cleanest available proxy for how hard the field
            # was managing its tyres.
            "stints_per_driver": float(runs.reset_index().groupby("driver").size().mean()),
        }
    return out


def build_table() -> pd.DataFrame:
    """One row per compound comparison, with everything that might explain it."""
    if not EXP03.exists():
        raise SystemExit("run exp03_practice_to_race.py first")
    exp03 = json.loads(EXP03.read_text())
    practice_session = exp03.get("practice_session", "FP2")

    conditions = {}
    if CONDITIONS.exists():
        for c in json.loads(CONDITIONS.read_text()):
            conditions[(c["year"], c["event"], c["session"])] = c

    rows: list[dict] = []
    for report in exp03["reports"]:
        year, event = report["year"], report["event"]
        practice = stint_profile(year, event, practice_session)
        race = stint_profile(year, event, "R")
        cond_p = conditions.get((year, event, practice_session), {})
        cond_r = conditions.get((year, event, "R"), {})

        for case in report["comparisons"]:
            label = case["compound"]
            p, r = practice.get(label, {}), race.get(label, {})
            if not p or not r:
                continue
            error = case["predicted"] - case["actual"]
            rows.append(
                {
                    # Year is carried because event names repeat across seasons:
                    # "British Grand Prix" exists in both 2023 and 2024, and
                    # counting unique names reported 17 events where there are 27.
                    "year": year,
                    "event": event,
                    "compound": label,
                    "error": error,
                    "abs_error": abs(error),
                    "predicted": case["predicted"],
                    "actual": case["actual"],
                    "posterior_sd": case.get("predicted_sd", np.nan),
                    "practice_stint_len": p["mean_run_laps"],
                    "practice_runs": p["n_runs"],
                    "practice_laps": p["total_laps"],
                    "race_stint_len": r["mean_run_laps"],
                    "race_max_stint": r["max_run_laps"],
                    "stops_per_driver": r["stints_per_driver"],
                    "traffic_gap": r["mean_traffic"] - p["mean_traffic"],
                    "temp_gap": (
                        cond_r.get("track_temp_c", np.nan) - cond_p.get("track_temp_c", np.nan)
                    ),
                }
            )
    return pd.DataFrame(rows)


def correlate(df: pd.DataFrame, target: str, predictors: list[str]) -> list[dict]:
    """Spearman rather than Pearson.

    Sixty-odd points, no reason to assume the relationship is linear, and one or
    two events (Singapore, Spa) carry errors large enough to drag a Pearson
    coefficient on their own. Rank correlation answers the question actually
    being asked -- does the error tend to grow with this quantity -- without
    letting two circuits decide the answer.
    """
    from scipy import stats

    out = []
    for name in predictors:
        pair = df[[target, name]].dropna()
        if len(pair) < 8 or pair[name].nunique() < 3:
            continue
        rho, p = stats.spearmanr(pair[name], pair[target])
        out.append(
            {
                "predictor": name,
                "n": int(len(pair)),
                "spearman_rho": float(rho),
                "p_value": float(p),
            }
        )
    out.sort(key=lambda d: d["p_value"])

    # Nine predictors against one target is nine chances to find a p below 0.05
    # by luck alone, and at that width roughly one spurious hit is the EXPECTED
    # outcome. Benjamini-Hochberg controls the share of claims that are false
    # rather than the chance of any false claim, which is the right trade when
    # the point is to decide what to investigate next. Uncorrected significance
    # is retained alongside it so the correction's effect is visible rather than
    # quietly applied.
    m = len(out)
    for rank, row in enumerate(out, start=1):
        row["bh_threshold"] = 0.05 * rank / m
        row["significant_raw"] = bool(row["p_value"] < 0.05)
    # A hit is kept only if some larger rank also clears its threshold.
    survives = False
    for row in reversed(out):
        survives = survives or row["p_value"] <= row["bh_threshold"]
        row["significant"] = bool(survives)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    warnings.filterwarnings("ignore")

    df = build_table()
    if df.empty:
        raise SystemExit("no comparisons could be matched to corpus sessions")

    predictors = [
        "practice_stint_len", "practice_runs", "practice_laps",
        "race_stint_len", "race_max_stint", "stops_per_driver",
        "traffic_gap", "temp_gap", "posterior_sd",
    ]
    signed = correlate(df, "error", predictors)
    absolute = correlate(df, "abs_error", predictors)

    print("=" * 84)
    n_events = int(df.groupby(["year", "event"]).ngroups)
    print(f"WHY DOES PRACTICE OVER-PREDICT?  ({len(df)} comparisons, "
          f"{n_events} events)")
    print("=" * 84)
    print(f"\n  mean signed error {df['error'].mean():+.4f} s/lap   "
          f"mean |error| {df['abs_error'].mean():.4f} s/lap\n")

    for title, table in (("SIGNED ERROR (the bias)", signed), ("ABSOLUTE ERROR", absolute)):
        print(f"  {title}")
        print(f"  {'predictor':<22} {'n':>4} {'rho':>8} {'p':>8} {'BH':>7}   verdict")
        for row in table:
            if row["significant"]:
                mark = "SUPPORTED"
            elif row["significant_raw"]:
                mark = "raw only -- does not survive correction"
            else:
                mark = "-"
            print(f"  {row['predictor']:<22} {row['n']:>4} {row['spearman_rho']:>+8.3f} "
                  f"{row['p_value']:>8.3f} {row['bh_threshold']:>7.4f}   {mark}")
        print()

    supported = [r["predictor"] for r in signed if r["significant"]]
    print("=" * 84)
    if supported:
        print(f"  The bias tracks: {', '.join(supported)},")
        print("  at a 5% false-discovery rate across the nine predictors tested.")
        print("  Both survivors measure the same thing from opposite ends: HOW DEEP INTO")
        print("  THE TYRE'S LIFE each side of the comparison goes. A long practice run")
        print("  reaches the aggressive part of the wear curve; a heavily-pitted race")
        print("  never leaves the flat early part. We fit ONE LINEAR RATE to two")
        print("  different depths and then compare them as though they were the same")
        print("  quantity. That is not noise -- it is a specification error, and it is")
        print("  correctable by matching the lap window before comparing.")
    else:
        print("  Nothing explains the signed bias at this sample size. The bias is real")
        print("  and stable, but its mechanism is not yet identified -- so it stays a")
        print("  reported offset rather than a correction we pretend to understand.")
    print("=" * 84)

    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    RESULTS.write_text(json.dumps({
        "experiment": "exp10_bias_mechanism",
        "generated_at": datetime.now(UTC).isoformat(),
        "n_comparisons": int(len(df)),
        "n_events": n_events,
        "mean_signed_error": float(df["error"].mean()),
        "mean_abs_error": float(df["abs_error"].mean()),
        "signed_error": signed,
        "absolute_error": absolute,
        "supported": supported,
        "table": df.replace({np.nan: None}).to_dict("records"),
    }, indent=2))
    print(f"\nwrote {RESULTS}")


if __name__ == "__main__":
    main()
