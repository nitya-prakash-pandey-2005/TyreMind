"""Experiment 15 -- is there a driver effect, and does it generalise?

Every strategist will tell you that driving style changes tyre life, and every
tyre model in this project pools the whole field into one compound baseline and
never attributes any of it to the driver. Believing the strategist is not the
same as measuring the effect, and an effect that exists but does not *transfer*
is useless: a per-driver term fitted on Sunday that says nothing about next
Sunday is a parameter, not knowledge.

So the question is not "do drivers differ" -- with enough laps, everything
differs. It is whether driver identity carries signal that survives being taken
to a race the model has not seen.

Three tests, each answering a different question:

  A. SPLIT-HALF RELIABILITY. Split the races at random into two halves, take each
     driver's mean deviation in each half, and correlate across drivers. A real,
     stable trait shows up as a positive correlation; noise does not. Repeated
     over many splits so the answer does not depend on one lucky partition.

  B. OUT-OF-SAMPLE PREDICTION, leave-one-race-out. Predict a driver's deviation
     at the held-out race from their mean deviation at every other race, and
     score it against the honest null of predicting zero -- the race mean, which
     is what the model does today. This is the test that decides whether the
     term is worth adding, and it is the same shape as experiments 08 and 09.

  C. TEAMMATE CONTRAST. Driver and car are perfectly confounded: "this driver is
     hard on tyres" and "this car is hard on tyres" make identical predictions.
     Teammates share a car, so the within-team difference removes it.

The deviation is taken within (race, compound), which removes circuit severity,
weather, and compound identity in one step. What remains is who was driving --
and, until test C, which car they were driving.

WHAT THE FULL CORPUS SAYS, and it inverted partway through. On two seasons the
raw deviation was stable and the teammate contrast was not, which reads as
"we measured the car". On four seasons it is the other way round: the raw
deviation no longer clears zero (5th percentile -0.079) while the teammate
contrast does (+0.031 over 21 pairs).

Removing a nuisance should not IMPROVE a signal unless the nuisance was large.
The car is exactly that -- it changes between seasons, it changes with upgrades
inside one, and pooling across the field buries a small driver contrast
underneath it. Holding the car fixed is what makes the contrast legible, which
is a better-identified result than the one this experiment started with.

It is still not useful. Test B is unmoved: a driver's history does not beat the
field mean at a held-out race. Real, identified, and too small to act on.

    python experiments/exp15_driver_effect.py
    python experiments/exp15_driver_effect.py --limit 40

Writes experiments/results/exp15_driver_effect.json.
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

RESULTS = Path(__file__).parent / "results" / "exp15_driver_effect.json"
#: Per-stint slopes, cached because extracting them costs twenty minutes and
#: the analysis built on them costs seconds.
STINT_CACHE = Path("data/reference/driver_stint_slopes.parquet")
CORPUS = Path("data/season")
CONDITIONS = Path("data/reference/session_conditions.json")
LINEUPS = Path("data/reference/lineups.json")

#: A driver needs this many laps on a compound before their stint rate is an
#: estimate rather than a prior. Below it the posterior is mostly the pooled
#: baseline, which would manufacture agreement between drivers for free.
MIN_LAPS_PER_STINT = 8

#: A driver has to appear at this many races before their mean deviation means
#: anything. Two races of noise averages to noise.
MIN_RACES_PER_DRIVER = 5

#: Random splits for the split-half test.
N_SPLITS = 500


def dry_races(limit: int) -> list:
    """Races that were genuine dry-degradation sessions."""
    dry = {}
    if CONDITIONS.exists():
        dry = {
            (c["year"], c["event"]): bool(c["dry_session"])
            for c in json.loads(CONDITIONS.read_text())
            if c["session"] == "R"
        }
    races = sessions(CORPUS, session_type="R", limit=0)
    keep = [r for r in races if dry.get((r.year, r.event), True)]
    return keep[:limit] if limit else keep


def load_lineups() -> dict[tuple[int, str], dict[str, str]]:
    if not LINEUPS.exists():
        return {}
    return {
        (r["year"], r["event"]): r["lineup"] for r in json.loads(LINEUPS.read_text())
    }


def free_stint_slopes(fit, lap_table: pd.DataFrame) -> pd.DataFrame:
    """Per-stint degradation slope, fitted freely rather than read off the model.

    Reading the model's own per-driver rate cannot answer this question, and
    finding that out was the first result of this experiment. The estimator
    optimises a per-stint rate spread, `stint_rate_sd`, and on every session
    tried the likelihood drove it to the FLOOR of its search range -- 1e-3, a
    variance of 1e-6. The per-driver rates that come back therefore differ by
    around 1e-5 s/lap, which is numerically zero. The model is not saying drivers
    are identical; it is structurally unable to say otherwise, because whatever
    per-stint variation exists is being absorbed elsewhere or shrunk away.

    So the confounders are taken from the model -- which is what it is good at,
    and what makes the tyre slope identifiable at all -- and the tyre slope
    itself is then fitted by ordinary least squares within each stint, with no
    prior on it whatsoever:

        corrected = lap_time + fuel_slope * lap_in_run
                             - track_effect(session_lap)
                             - traffic_coef * traffic_index

        corrected ~ a + b * tyre_age,  per driver-run

    The run intercept and the driver's starting tyre state are constant within a
    stint, so they fall into `a` and cannot bias `b`.
    """
    fuel_slope = fit.fuel_slope()[0]
    traffic_coef = fit.traffic_coefficient()[0]
    track = fit.track_evolution().set_index("session_lap")["track_effect"]

    frame = lap_table.copy()
    frame["corrected"] = (
        frame["lap_time"]
        + fuel_slope * frame["lap_in_run"]
        - frame["session_lap"].map(track).fillna(0.0)
        - traffic_coef * frame["traffic_index"]
    )

    rows: list[dict] = []
    for (driver, run_id), stint in frame.groupby(["driver", "run_id"]):
        if len(stint) < MIN_LAPS_PER_STINT:
            continue
        age = stint["tyre_age"].to_numpy(float)
        if np.ptp(age) < 3.0:
            continue
        slope, _ = np.polyfit(age, stint["corrected"].to_numpy(float), 1)
        rows.append({
            "driver": str(driver),
            "run_id": int(run_id),
            "compound": str(stint["compound"].iloc[0]),
            "laps": int(len(stint)),
            "slope": float(slope),
        })
    return pd.DataFrame(rows)


def driver_deviations(limit: int) -> pd.DataFrame:
    """One row per (race, driver, compound): how far from the field that driver ran.

    The deviation is taken within (race, compound), so circuit severity, weather
    and compound identity are all differenced away before anything is compared.
    """
    lineups = load_lineups()
    rows: list[dict] = []

    for race in dry_races(limit):
        lap_table = race.load()
        try:
            fit = fit_tyre_ssm(lap_table)
        except Exception as exc:  # noqa: BLE001 - a session that will not fit is data
            print(f"  {race.session_id:<40} skip  {type(exc).__name__}")
            continue

        stints = free_stint_slopes(fit, lap_table)
        if stints.empty:
            continue
        stints = stints.groupby(["driver", "compound"], as_index=False).agg(
            slope=("slope", "median"), laps=("laps", "sum")
        )

        # Deviation from the field, within compound.
        stints["deviation"] = stints["slope"] - stints.groupby("compound")["slope"].transform("mean")
        # A compound only one or two drivers ran gives a deviation that is mostly
        # an artefact of the small group it is measured against.
        sizes = stints.groupby("compound")["slope"].transform("size")
        stints = stints[sizes >= 3]
        if stints.empty:
            continue

        lineup = lineups.get((race.year, race.event), {})
        for row in stints.itertuples(index=False):
            rows.append({
                "session_id": race.session_id,
                "year": race.year,
                "event": race.event,
                "driver": row.driver,
                "team": lineup.get(row.driver),
                "compound": row.compound,
                "rate": float(row.slope),
                "deviation": float(row.deviation),
                "laps": int(row.laps),
            })
        print(f"  {race.session_id:<40} {len(stints)} driver-compound stints")

    return pd.DataFrame(rows)


def split_half(per_race: pd.DataFrame, rng: np.random.Generator) -> list[float]:
    """Correlate each driver's deviation between two random halves of the season."""
    races = per_race["session_id"].unique()
    out: list[float] = []
    for _ in range(N_SPLITS):
        shuffled = rng.permutation(races)
        left = set(shuffled[: len(shuffled) // 2])
        a = per_race[per_race["session_id"].isin(left)].groupby("driver")["deviation"].mean()
        b = per_race[~per_race["session_id"].isin(left)].groupby("driver")["deviation"].mean()
        shared = a.index.intersection(b.index)
        if len(shared) < 8:
            continue
        r = np.corrcoef(a[shared], b[shared])[0, 1]
        if np.isfinite(r):
            out.append(float(r))
    return out


def leave_one_race_out(per_race: pd.DataFrame) -> dict:
    """Predict a driver's deviation at a held-out race from their other races."""
    records: list[dict] = []
    for held in per_race["session_id"].unique():
        train = per_race[per_race["session_id"] != held]
        test = per_race[per_race["session_id"] == held]
        history = train.groupby("driver")["deviation"].mean()
        for row in test.itertuples(index=False):
            if row.driver not in history:
                continue
            records.append({
                "session_id": held,
                "driver": row.driver,
                "actual": row.deviation,
                "predicted": float(history[row.driver]),
            })

    if len(records) < 30:
        return {"n": len(records)}

    frame = pd.DataFrame(records)
    from scipy import stats

    driver_err = (frame["predicted"] - frame["actual"]).abs()
    # The honest null: predict the field mean, which after differencing is zero.
    # This is exactly what the model does today.
    null_err = frame["actual"].abs()
    _, p = stats.wilcoxon(driver_err, null_err)
    return {
        "n": int(len(frame)),
        "n_drivers": int(frame["driver"].nunique()),
        "driver_mae": float(driver_err.mean()),
        "field_mean_mae": float(null_err.mean()),
        "improvement_pct": float(100.0 * (null_err.mean() - driver_err.mean()) / null_err.mean()),
        "wilcoxon_p": float(p),
        "beats_null": bool(driver_err.mean() < null_err.mean() and p < 0.05),
    }


def teammate_contrast(per_race: pd.DataFrame) -> dict:
    """Split-half reliability of the within-team difference, which has no car in it."""
    known = per_race.dropna(subset=["team"])
    if known.empty:
        return {"available": False, "reason": "no lineup data; run scripts/build_lineups.py"}

    by_race_team = known.groupby(["session_id", "team", "driver"])["deviation"].mean()
    gaps: list[dict] = []
    for (session_id, team), block in by_race_team.groupby(level=[0, 1]):
        drivers = block.index.get_level_values("driver").tolist()
        if len(drivers) != 2:
            continue
        first, second = sorted(drivers)
        gap = float(block.xs(first, level="driver").iloc[0] - block.xs(second, level="driver").iloc[0])
        gaps.append({"session_id": session_id, "team": team, "pair": f"{first}-{second}", "gap": gap})

    if len(gaps) < 20:
        return {"available": False, "reason": f"only {len(gaps)} teammate pairs"}

    frame = pd.DataFrame(gaps)
    counts = frame["pair"].value_counts()
    frame = frame[frame["pair"].isin(counts[counts >= MIN_RACES_PER_DRIVER].index)]
    if frame["pair"].nunique() < 5:
        return {"available": False, "reason": "too few pairs with enough races"}

    rng = np.random.default_rng(17)
    correlations: list[float] = []
    races = frame["session_id"].unique()
    for _ in range(N_SPLITS):
        shuffled = rng.permutation(races)
        left = set(shuffled[: len(shuffled) // 2])
        a = frame[frame["session_id"].isin(left)].groupby("pair")["gap"].mean()
        b = frame[~frame["session_id"].isin(left)].groupby("pair")["gap"].mean()
        shared = a.index.intersection(b.index)
        if len(shared) < 5:
            continue
        r = np.corrcoef(a[shared], b[shared])[0, 1]
        if np.isfinite(r):
            correlations.append(float(r))

    if not correlations:
        return {"available": False, "reason": "no usable splits"}

    return {
        "available": True,
        "n_pairs": int(frame["pair"].nunique()),
        "n_observations": int(len(frame)),
        "mean_split_half_r": float(np.mean(correlations)),
        "p05": float(np.percentile(correlations, 5)),
        "p95": float(np.percentile(correlations, 95)),
        "stable": bool(np.percentile(correlations, 5) > 0.0),
        "largest_gaps": frame.groupby("pair")["gap"].mean().abs().nlargest(5).round(4).to_dict(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=0, help="0 means every dry race")
    parser.add_argument(
        "--refit", action="store_true",
        help="re-fit every race instead of reusing the cached per-stint slopes",
    )
    args = parser.parse_args()
    warnings.filterwarnings("ignore")

    # Extracting the slopes means fitting every dry race, which is twenty
    # minutes. The analysis built on them takes seconds. Caching the expensive
    # half means a change to the *reasoning* can be re-run immediately, which is
    # the difference between revisiting a verdict and not bothering to.
    if STINT_CACHE.exists() and not args.refit and not args.limit:
        stints = pd.read_parquet(STINT_CACHE)
        print(f"\n  reusing {len(stints)} cached stints from {STINT_CACHE}")
        print("  pass --refit to re-fit every race\n")
    else:
        print(f"\n  fitting dry races from {CORPUS}\n")
        stints = driver_deviations(args.limit)
        if not args.limit and not stints.empty:
            STINT_CACHE.parent.mkdir(parents=True, exist_ok=True)
            stints.to_parquet(STINT_CACHE, index=False)

    if stints.empty:
        raise SystemExit("no usable stints")

    # One deviation per driver per race, so a driver who ran three compounds does
    # not count three times against one who ran one.
    per_race = stints.groupby(["session_id", "driver"], as_index=False).agg(
        deviation=("deviation", "mean"),
        team=("team", "first"),
        year=("year", "first"),
    )
    appearances = per_race["driver"].value_counts()
    per_race = per_race[per_race["driver"].isin(appearances[appearances >= MIN_RACES_PER_DRIVER].index)]

    rng = np.random.default_rng(11)
    correlations = split_half(per_race, rng)
    reliability = {
        "n_splits": len(correlations),
        "mean_r": float(np.mean(correlations)) if correlations else float("nan"),
        "p05": float(np.percentile(correlations, 5)) if correlations else float("nan"),
        "p95": float(np.percentile(correlations, 95)) if correlations else float("nan"),
    }
    reliability["stable"] = bool(correlations and reliability["p05"] > 0.0)

    prediction = leave_one_race_out(per_race)
    teammates = teammate_contrast(per_race)

    spread = float(per_race.groupby("driver")["deviation"].mean().std())

    print("\n" + "=" * 84)
    print(f"IS THERE A DRIVER EFFECT?  ({len(per_race)} driver-races, "
          f"{per_race['driver'].nunique()} drivers, {per_race['session_id'].nunique()} races)")
    print("=" * 84)

    print(f"\n  spread of driver means            : {spread:.4f} s/lap")
    print("\n  A. SPLIT-HALF RELIABILITY")
    print(f"     mean r across {reliability['n_splits']} random splits : "
          f"{reliability['mean_r']:+.3f}  (5-95%: {reliability['p05']:+.3f} to "
          f"{reliability['p95']:+.3f})")
    print(f"     stable                         : {reliability['stable']}")

    print("\n  B. LEAVE-ONE-RACE-OUT PREDICTION")
    if prediction.get("n", 0) < 30:
        print(f"     only {prediction.get('n', 0)} scored cases -- too few to conclude")
    else:
        print(f"     predicting from driver history : {prediction['driver_mae']:.4f} s/lap")
        print(f"     predicting the field mean      : {prediction['field_mean_mae']:.4f} s/lap")
        print(f"     improvement                    : {prediction['improvement_pct']:+.1f}%"
              f"   Wilcoxon p = {prediction['wilcoxon_p']:.4f}")

    print("\n  C. TEAMMATE CONTRAST (the car differenced out)")
    if not teammates.get("available"):
        print(f"     unavailable: {teammates.get('reason')}")
    else:
        print(f"     {teammates['n_pairs']} pairs, {teammates['n_observations']} observations")
        print(f"     split-half r of the teammate gap: {teammates['mean_split_half_r']:+.3f}"
              f"  (5-95%: {teammates['p05']:+.3f} to {teammates['p95']:+.3f})")
        print(f"     stable                          : {teammates['stable']}")

    print("\n" + "=" * 84)
    reliable = bool(reliability["stable"])
    useful = bool(prediction.get("beats_null"))
    car_free = bool(teammates.get("stable"))

    if useful and car_free:
        print("  VERDICT: a driver effect is real, transfers out of sample, and survives")
        print("           differencing the car away. It is worth modelling.")
    elif useful:
        print("  VERDICT: the effect transfers out of sample, but the teammate contrast")
        print("           does NOT separate it from the car. What has been measured is a")
        print("           car-and-driver effect, and calling it a driver effect would be")
        print("           claiming an identification the data does not support.")
    elif reliable:
        print("  VERDICT: the driver effect is RELIABLE but not USEFUL, and the distinction")
        print("           is the finding.")
        print()
        print(f"           Reliable: split-half r = {reliability['mean_r']:+.2f} with a 5th")
        print(f"           percentile of {reliability['p05']:+.2f}, so the ranking of drivers")
        print("           reproduces across independent halves of the corpus. Some drivers")
        print("           genuinely are harder on tyres than others, measurably so.")
        print()
        print("           Not useful: knowing it does not help. Predicting a driver's")
        print("           deviation at a held-out race from their own history scores")
        print(f"           {prediction['driver_mae']:.4f} s/lap against {prediction['field_mean_mae']:.4f}")
        print(f"           for simply predicting the field mean (p = {prediction['wilcoxon_p']:.2f}).")
        print()
        print("           The reason is the ratio. The spread of driver means is")
        print(f"           {spread:.4f} s/lap; the race-to-race scatter around them is")
        print(f"           {prediction['field_mean_mae']:.4f}. The trait is real and it is")
        print("           roughly half the size of the noise it has to be read through, so")
        print("           on any single race the history is worth less than the average.")
        print()
        print("           A per-driver term would therefore add a parameter, a maintenance")
        print("           burden and a story, and would not improve a forecast. It is not")
        print("           built.")
    elif car_free:
        print("  VERDICT: the raw driver deviation is NOT stable, and the teammate")
        print("           contrast IS. That inversion is the finding.")
        print()
        print("           Pooled across the field, a driver's deviation reproduces at")
        print(f"           r = {reliability['mean_r']:+.2f} with a 5th percentile of")
        print(f"           {reliability['p05']:+.3f} -- it does not clear zero. Difference out")
        print("           the car by comparing teammates and the same measurement")
        print(f"           reproduces at r = {teammates['mean_split_half_r']:+.2f}, 5th percentile")
        print(f"           {teammates['p05']:+.3f}, over {teammates['n_pairs']} pairs.")
        print()
        print("           Removing a nuisance should not IMPROVE a signal unless the")
        print("           nuisance was large. The car is: it changes between seasons, it")
        print("           changes with upgrades inside one, and pooling across the field")
        print("           buries a small driver contrast underneath it. Holding the car")
        print("           fixed is what makes the contrast legible.")
        print()
        print("           It is still not USEFUL. Predicting a driver's deviation at a")
        print("           held-out race from their history scores")
        print(f"           {prediction['driver_mae']:.4f} s/lap against")
        print(f"           {prediction['field_mean_mae']:.4f} for the field mean")
        print(f"           (p = {prediction['wilcoxon_p']:.2f}). The spread of driver means is")
        print(f"           {spread:.4f} s/lap against race-to-race scatter of")
        print(f"           {prediction['field_mean_mae']:.4f}. Real, identified, and too small")
        print("           to act on. Not built.")
    else:
        print("  VERDICT: driver identity does not beat the field mean out of sample, and")
        print("           the trait is not stable across halves of the corpus either.")
        print("           Drivers differ within a race -- that is not in doubt -- but the")
        print("           difference does not carry to a race the model has not seen.")

    if reliable and not car_free:
        print()
        print("           One further caution: the teammate contrast, which differences the")
        print("           car away, is NOT stable. So even the reliable part cannot be")
        print("           attributed to the driver rather than to the car they sit in.")
    print("=" * 84)

    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    RESULTS.write_text(json.dumps({
        "experiment": "exp15_driver_effect",
        "generated_at": datetime.now(UTC).isoformat(),
        "n_driver_races": int(len(per_race)),
        "n_drivers": int(per_race["driver"].nunique()),
        "n_races": int(per_race["session_id"].nunique()),
        "driver_mean_spread": spread,
        "split_half": reliability,
        "prediction": prediction,
        "teammates": teammates,
        "reliable_not_useful": bool(reliability["stable"] and not prediction.get("beats_null")),
        "teammate_contrast_cleaner_than_raw": bool(
            teammates.get("stable") and not reliability["stable"]
        ),
        "signal_to_noise": (
            spread / prediction["field_mean_mae"] if prediction.get("field_mean_mae") else None
        ),
        "driver_means": per_race.groupby("driver")["deviation"].mean().round(5).to_dict(),
    }, indent=2))
    print(f"\nwrote {RESULTS}")


if __name__ == "__main__":
    main()
