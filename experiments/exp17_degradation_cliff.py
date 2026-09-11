"""Experiment 17 -- is the degradation curve a line, or does it have a cliff?

The brief asks for *degradation curves*. What this project actually produces is
close to a straight line: the state-space model carries a degradation rate as a
random walk, so the curve can bend, but nothing in it represents the thing every
strategist means by the word. A tyre does not fade linearly and then stop. It
holds, and then it goes -- the cliff -- and the lap that matters is the one where
that happens.

A linear rate answers "how fast is this tyre losing performance". It cannot
answer "how many more laps have I got", which is the question that actually gets
asked on a pit wall, and the two are only the same if the curve is a line.

THE MODEL. A broken-stick, fitted on de-confounded lap time against tyre age:

    L(a) = alpha + beta * a                                    for a <= tau
    L(a) = alpha + beta * tau + (beta + delta) * (a - tau)     for a >  tau

  tau    the cliff lap: tyre age at which the regime changes
  beta   degradation before it, s/lap
  delta  the STEP UP in rate after it, s/lap. delta > 0 is a cliff; delta < 0 is
         a tyre coming back to the driver, which happens after a graining phase
         clears and is a real effect rather than a fitting artefact.

Continuity at tau is imposed, not fitted -- performance does not teleport. That
costs a parameter and buys a model that cannot produce a discontinuity nobody has
ever observed.

The confounders come from the state-space model, which is what it is good at and
what makes the tyre term identifiable at all; the curve on top is fitted freely,
with no prior on tau, beta or delta. tau is found by exhaustive search over
admissible laps rather than by gradient descent, because the least-squares
surface in tau is piecewise-smooth with many local minima and a gradient method
lands in whichever one it started nearest.

THREE QUESTIONS, and they are different:

  1. DOES a cliff exist? Compare against a straight line by BIC, which charges
     the broken stick for its two extra parameters. A model that fits better
     because it has more knobs has not found anything.

  2. Does modelling it PREDICT BETTER? Fit on the first 70% of a stint and
     forecast the rest. This is the only test that matters, because the cliff is
     precisely a claim about extrapolation, and it is where a linear fit is
     guaranteed to be optimistic.

  3. Is the cliff PREDICTABLE before it happens? A cliff you can only see
     afterwards is a post-race curiosity. One whose onset is consistent for a
     compound is a pit call.

    python experiments/exp17_degradation_cliff.py
    python experiments/exp17_degradation_cliff.py --limit 20

Writes experiments/results/exp17_degradation_cliff.json.
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

RESULTS = Path(__file__).parent / "results" / "exp17_degradation_cliff.json"
CACHE = Path("data/reference/cliff_stints.parquet")
CORPUS = Path("data/season")
CONDITIONS = Path("data/reference/session_conditions.json")

#: A changepoint needs data on both sides of it. Twelve laps leaves room for a
#: cliff with at least four laps either side, which is the minimum that could
#: distinguish a regime change from two outliers.
MIN_STINT_LAPS = 12

#: Laps that must remain on each side of a candidate cliff. Below this the
#: "cliff" is fitted to a handful of points and will find one in pure noise.
MIN_SEGMENT = 4

#: Fraction of a stint used to fit, when forecasting the remainder.
FIT_FRACTION = 0.70

#: A cliff has to be a real change in rate to count, not a rounding error. Two
#: hundredths of a second per lap, per lap -- over a five-lap window that is a
#: tenth, which is the scale a strategist would act on.
MEANINGFUL_DELTA = 0.02

#: A tyre losing time faster than this before the break is degrading, not warming
#: up. Below minus this it is genuinely getting quicker, which a degrading tyre
#: does not do, so the break is a warm-up boundary rather than a cliff. The two
#: are separated here because the broken stick finds both and they are opposite
#: physical events -- a first pass that pooled them reported a "cliff" 37% of the
#: way through a stint, which is a tyre coming to temperature.
WARMUP_SLOPE = 0.005


def corrected_stints(limit: int) -> pd.DataFrame:
    """De-confounded lap time against tyre age, one row per lap, per stint.

    The model supplies the fuel, track-evolution and traffic corrections; the
    curve is then fitted on what is left, free of any prior about its shape.
    """
    dry = {}
    if CONDITIONS.exists():
        dry = {
            (c["year"], c["event"]): bool(c["dry_session"])
            for c in json.loads(CONDITIONS.read_text())
            if c["session"] == "R"
        }
    races = [
        r for r in sessions(CORPUS, session_type="R", limit=0)
        if dry.get((r.year, r.event), True)
    ]
    if limit:
        races = races[:limit]

    rows: list[pd.DataFrame] = []
    for race in races:
        lap_table = race.load()
        try:
            fit = fit_tyre_ssm(lap_table)
        except Exception as exc:  # noqa: BLE001 - a session that will not fit is data
            print(f"  {race.session_id:<40} skip  {type(exc).__name__}")
            continue

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
        frame["session_id"] = race.session_id
        frame["year"] = race.year
        frame["event"] = race.event
        rows.append(frame[[
            "session_id", "year", "event", "driver", "run_id",
            "compound", "tyre_age", "corrected",
        ]])
        print(f"  {race.session_id:<40} {frame['run_id'].nunique()} stints")

    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def fit_linear(age: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float]:
    """Ordinary least squares. Returns (prediction, residual sum of squares)."""
    design = np.column_stack([np.ones_like(age), age])
    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    prediction = design @ coef
    return coef, float(((y - prediction) ** 2).sum())


def broken_stick_design(age: np.ndarray, tau: float) -> np.ndarray:
    """[1, a, max(a - tau, 0)] -- continuous at tau by construction.

    The third column is the hinge. Its coefficient is `delta`, the step up in
    slope after the cliff, so continuity is a property of the basis rather than
    a constraint that has to be enforced afterwards.
    """
    return np.column_stack([np.ones_like(age), age, np.maximum(age - tau, 0.0)])


def fit_broken_stick(age: np.ndarray, y: np.ndarray) -> dict | None:
    """Search every admissible cliff lap and keep the best by least squares."""
    candidates = np.unique(age)
    candidates = candidates[
        (candidates >= np.min(age) + MIN_SEGMENT) & (candidates <= np.max(age) - MIN_SEGMENT)
    ]
    if candidates.size == 0:
        return None

    best = None
    for tau in candidates:
        design = broken_stick_design(age, float(tau))
        coef, *_ = np.linalg.lstsq(design, y, rcond=None)
        rss = float(((y - design @ coef) ** 2).sum())
        if best is None or rss < best["rss"]:
            best = {"tau": float(tau), "coef": coef, "rss": rss}
    return best


def bic(rss: float, n: int, k: int) -> float:
    """Gaussian BIC. Lower is better.

    The broken stick is charged for tau as a free parameter even though it is
    found by search rather than by derivative -- searching a grid is still
    fitting, and pretending otherwise is how a changepoint model comes to look
    better than it is.
    """
    if rss <= 0 or n <= k:
        return float("inf")
    return n * np.log(rss / n) + k * np.log(n)


def analyse_stint(age: np.ndarray, y: np.ndarray) -> dict | None:
    """Fit both models to one stint, in-sample and as a forecast."""
    order = np.argsort(age)
    age, y = age[order], y[order]

    linear_coef, linear_rss = fit_linear(age, y)
    stick = fit_broken_stick(age, y)
    if stick is None:
        return None

    n = age.size
    linear_bic = bic(linear_rss, n, 3)        # intercept, slope, variance
    stick_bic = bic(stick["rss"], n, 5)       # + tau, delta

    # Forecast: fit on the opening stretch, score on the rest. The cliff is a
    # claim about what happens next, so this is where it has to earn its place.
    cut = int(np.floor(n * FIT_FRACTION))
    forecast: dict[str, float] = {}
    if cut >= MIN_SEGMENT * 2 and n - cut >= 2:
        train_age, train_y = age[:cut], y[:cut]
        test_age, test_y = age[cut:], y[cut:]

        lin_coef, _ = fit_linear(train_age, train_y)
        lin_pred = np.column_stack([np.ones_like(test_age), test_age]) @ lin_coef

        train_stick = fit_broken_stick(train_age, train_y)
        if train_stick is not None:
            design = broken_stick_design(test_age, train_stick["tau"])
            stick_pred = design @ train_stick["coef"]
            forecast = {
                "linear_mae": float(np.abs(lin_pred - test_y).mean()),
                "stick_mae": float(np.abs(stick_pred - test_y).mean()),
                "linear_bias": float((lin_pred - test_y).mean()),
                "stick_bias": float((stick_pred - test_y).mean()),
                "n_forecast": int(test_age.size),
            }

    delta = float(stick["coef"][2])
    slope_before = float(stick["coef"][1])
    fraction = float((stick["tau"] - age[0]) / max(age[-1] - age[0], 1e-9))

    # Classify the regime change, because the broken stick finds two physically
    # different things and averaging them together describes neither.
    #
    #   WARM-UP   the tyre is getting FASTER before the break (slope_before < 0)
    #             and then starts degrading. This is a tyre coming to temperature,
    #             or a graining phase clearing. It is early in the stint and it is
    #             not a cliff, however much the fit looks like one.
    #   CLIFF     the tyre is already degrading and then degrades markedly faster.
    #             This is the thing a pit wall means by the word.
    #   RECOVERY  the rate DROPS after the break. A tyre coming back to the driver.
    if not (stick_bic < linear_bic):
        regime = "linear"
    elif delta < -MEANINGFUL_DELTA:
        regime = "recovery"
    elif slope_before < -WARMUP_SLOPE and delta > MEANINGFUL_DELTA:
        regime = "warm-up"
    elif delta > MEANINGFUL_DELTA:
        regime = "cliff"
    else:
        regime = "linear"

    return {
        "regime": regime,
        "n_laps": int(n),
        "first_age": float(age[0]),
        "last_age": float(age[-1]),
        "linear_slope": float(linear_coef[1]),
        "cliff_lap": stick["tau"],
        "cliff_fraction": fraction,
        "slope_before": slope_before,
        "delta": delta,
        "slope_after": slope_before + delta,
        "linear_bic": linear_bic,
        "stick_bic": stick_bic,
        "stick_wins_bic": bool(stick_bic < linear_bic),
        "meaningful_cliff": bool(stick_bic < linear_bic and delta > MEANINGFUL_DELTA),
        **forecast,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=0, help="0 means every dry race")
    parser.add_argument("--refit", action="store_true", help="ignore the cached lap series")
    args = parser.parse_args()
    warnings.filterwarnings("ignore")

    if CACHE.exists() and not args.refit and not args.limit:
        laps = pd.read_parquet(CACHE)
        print(f"\n  reusing cached lap series from {CACHE} ({len(laps):,} laps)")
        print("  pass --refit to re-fit every race\n")
    else:
        print(f"\n  fitting dry races from {CORPUS}\n")
        laps = corrected_stints(args.limit)
        if not args.limit and not laps.empty:
            CACHE.parent.mkdir(parents=True, exist_ok=True)
            laps.to_parquet(CACHE, index=False)

    if laps.empty:
        raise SystemExit("no usable laps")

    records: list[dict] = []
    for (session_id, driver, run_id), stint in laps.groupby(["session_id", "driver", "run_id"]):
        if len(stint) < MIN_STINT_LAPS:
            continue
        result = analyse_stint(
            stint["tyre_age"].to_numpy(float), stint["corrected"].to_numpy(float)
        )
        if result is None:
            continue
        result |= {
            "session_id": session_id,
            "driver": driver,
            "run_id": int(run_id),
            "compound": str(stint["compound"].iloc[0]),
        }
        records.append(result)

    if len(records) < 30:
        raise SystemExit(f"only {len(records)} usable stints")

    df = pd.DataFrame(records)
    from scipy import stats

    wins = df["stick_wins_bic"]
    cliffs = df[df["regime"] == "cliff"]
    warmups = df[df["regime"] == "warm-up"]

    forecast_block: dict = {"n": 0}
    per_regime_forecast: dict = {}
    if "linear_mae" in df:
        forecast = df.dropna(subset=["linear_mae", "stick_mae"])
        if len(forecast) >= 20:
            _, p = stats.wilcoxon(forecast["stick_mae"], forecast["linear_mae"])
            forecast_block = {
                "n": int(len(forecast)),
                "linear_mae": float(forecast["linear_mae"].mean()),
                "stick_mae": float(forecast["stick_mae"].mean()),
                "improvement_pct": float(
                    100.0 * (forecast["linear_mae"].mean() - forecast["stick_mae"].mean())
                    / forecast["linear_mae"].mean()
                ),
                "linear_bias": float(forecast["linear_bias"].mean()),
                "stick_bias": float(forecast["stick_bias"].mean()),
                "wilcoxon_p": float(p),
                "stick_forecasts_better": bool(
                    forecast["stick_mae"].mean() < forecast["linear_mae"].mean() and p < 0.05
                ),
            }
            # Pooling every regime hides the only question worth asking: does
            # knowing about a CLIFF help, on the stints that actually have one?
            for name, block in forecast.groupby("regime"):
                if len(block) < 10:
                    continue
                _, rp = stats.wilcoxon(block["stick_mae"], block["linear_mae"])
                per_regime_forecast[str(name)] = {
                    "n": int(len(block)),
                    "linear_mae": float(block["linear_mae"].mean()),
                    "stick_mae": float(block["stick_mae"].mean()),
                    "improvement_pct": float(
                        100.0 * (block["linear_mae"].mean() - block["stick_mae"].mean())
                        / block["linear_mae"].mean()
                    ),
                    "linear_bias": float(block["linear_bias"].mean()),
                    "wilcoxon_p": float(rp),
                    "helps": bool(
                        block["stick_mae"].mean() < block["linear_mae"].mean() and rp < 0.05
                    ),
                }

    by_compound = {}
    for compound, block in cliffs.groupby("compound"):
        if len(block) < 8:
            continue
        by_compound[str(compound)] = {
            "n": int(len(block)),
            "median_cliff_lap": float(block["cliff_lap"].median()),
            "iqr": float(block["cliff_lap"].quantile(0.75) - block["cliff_lap"].quantile(0.25)),
            "median_delta": float(block["delta"].median()),
            "median_fraction_through_stint": float(block["cliff_fraction"].median()),
        }

    print("\n" + "=" * 86)
    print(f"IS THE DEGRADATION CURVE A LINE?  ({len(df)} stints, "
          f"{df['session_id'].nunique()} races)")
    print("=" * 86)

    print("\n  1. WHAT SHAPE IS IT? (broken stick vs a line, BIC-penalised)")
    print(f"     broken stick beats a line : {wins.sum()}/{len(df)}  ({wins.mean():.0%})")
    print("     ...but it finds two physically opposite things, so they are split:\n")
    print(f"     {'regime':<12}{'n':>5}{'share':>8}{'slope before':>14}{'step':>10}{'position':>12}")
    for name in ("linear", "warm-up", "cliff", "recovery"):
        block = df[df["regime"] == name]
        if block.empty:
            continue
        if name == "linear":
            print(f"     {name:<12}{len(block):>5}{len(block) / len(df):>8.0%}"
                  f"{'-':>14}{'-':>10}{'-':>12}")
        else:
            position = f"{block['cliff_fraction'].median():.0%} in"
            print(f"     {name:<12}{len(block):>5}{len(block) / len(df):>8.0%}"
                  f"{block['slope_before'].median():>+14.4f}"
                  f"{block['delta'].median():>+10.4f}{position:>12}")

    if not warmups.empty:
        print(f"\n     WARM-UP is the tyre getting FASTER "
              f"({warmups['slope_before'].median():+.4f} s/lap) and then starting")
        print("     to degrade -- coming to temperature, or a graining phase clearing.")
        print(f"     Median at {warmups['cliff_fraction'].median():.0%} through the stint. "
              f"It is not a cliff.")
    if not cliffs.empty:
        print(f"\n     CLIFF is the tyre already degrading "
              f"({cliffs['slope_before'].median():+.4f} s/lap) and then")
        print(f"     degrading {cliffs['delta'].median():+.4f} s/lap faster. Median at "
              f"{cliffs['cliff_fraction'].median():.0%} through the stint.")

    print(f"\n  2. DOES IT FORECAST BETTER? (fit {FIT_FRACTION:.0%}, predict the rest)")
    if forecast_block["n"] < 20:
        print("     too few stints long enough to split")
    else:
        print(f"     {'regime':<12}{'n':>5}{'line MAE':>11}{'stick MAE':>11}"
              f"{'gain':>9}{'p':>8}{'line bias':>12}")
        print(f"     {'ALL':<12}{forecast_block['n']:>5}{forecast_block['linear_mae']:>11.4f}"
              f"{forecast_block['stick_mae']:>11.4f}{forecast_block['improvement_pct']:>+8.1f}%"
              f"{forecast_block['wilcoxon_p']:>8.3f}{forecast_block['linear_bias']:>+12.4f}")
        for name, block in per_regime_forecast.items():
            print(f"     {name:<12}{block['n']:>5}{block['linear_mae']:>11.4f}"
                  f"{block['stick_mae']:>11.4f}{block['improvement_pct']:>+8.1f}%"
                  f"{block['wilcoxon_p']:>8.3f}{block['linear_bias']:>+12.4f}")

    print("\n  3. IS THE CLIFF LAP CONSISTENT WITHIN A COMPOUND?")
    if not by_compound:
        print("     too few cliffed stints per compound to say")
    else:
        print(f"     {'compound':<10}{'n':>5}{'median lap':>12}{'IQR':>8}{'step':>10}")
        for name, block in by_compound.items():
            print(f"     {name:<10}{block['n']:>5}{block['median_cliff_lap']:>12.0f}"
                  f"{block['iqr']:>8.1f}{block['median_delta']:>+10.4f}")

    print("\n" + "=" * 86)
    cliff_rate = len(cliffs) / len(df)
    cliff_helps = per_regime_forecast.get("cliff", {}).get("helps", False)
    if cliff_rate > 0.10 and cliff_helps:
        print("  VERDICT: the curve is not a line, and saying so predicts better. A straight")
        print("           fit is systematically optimistic about the end of a stint, which")
        print("           is the part a pit wall is actually asking about.")
    elif cliff_rate > 0.10:
        print(f"  VERDICT: a genuine cliff appears in {cliff_rate:.0%} of stints and is large when")
        print("           it appears, but locating it from the first 70% of a stint does NOT")
        print("           forecast the rest better. The cliff is real and arrives too late in")
        print("           the stint to be identified before it matters -- a statement about")
        print("           what is knowable, not about the physics.")
    else:
        print("  VERDICT: once the confounders are removed, most stints are adequately")
        print("           described by a straight line over the tyre ages a race reaches.")

    if not warmups.empty:
        print()
        print(f"  A separate finding, and the one that nearly went unnoticed: "
              f"{len(warmups)} stints")
        print(f"  ({len(warmups) / len(df):.0%}) show the opposite shape -- the tyre getting "
              f"quicker and then")
        print("  turning. A first pass pooled these with the cliffs and reported a 'cliff'")
        print("  37% of the way through a stint, which is a tyre coming to temperature.")
    print("=" * 86)

    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    RESULTS.write_text(json.dumps({
        "experiment": "exp17_degradation_cliff",
        "generated_at": datetime.now(UTC).isoformat(),
        "n_stints": int(len(df)),
        "n_races": int(df["session_id"].nunique()),
        "min_stint_laps": MIN_STINT_LAPS,
        "meaningful_delta_threshold": MEANINGFUL_DELTA,
        "bic_wins": int(wins.sum()),
        "bic_win_rate": float(wins.mean()),
        "regimes": df["regime"].value_counts().to_dict(),
        "cliff_rate": float(len(cliffs) / len(df)),
        "warmup_rate": float(len(warmups) / len(df)),
        "median_cliff_delta": float(cliffs["delta"].median()) if len(cliffs) else None,
        "median_cliff_fraction": float(cliffs["cliff_fraction"].median()) if len(cliffs) else None,
        "forecast": forecast_block,
        "forecast_by_regime": per_regime_forecast,
        "by_compound": by_compound,
        "stints": records,
    }, indent=2, default=float))
    print(f"\nwrote {RESULTS}")


if __name__ == "__main__":
    main()
