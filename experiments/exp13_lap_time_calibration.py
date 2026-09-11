"""Experiment 13 -- the lap-time intervals undercover, and time is the reason.

The model ladder's remaining embarrassment. Every rung's nominal 95% lap-time
interval covers far less than 95% of observations. Across 66,606 held-out laps
from twenty races the pooled figure is 75%, and no rung reaches 86%: LightGBM
64%, the state-space model 82%, naive 76%, the neural network 70%.
Under-coverage is the dangerous direction, and being beaten on it by pooled
regression is worse.

Experiment 12 fixed the same disease for degradation rates with split conformal
prediction. Pointing that machinery at lap times is NOT a copy-paste job, and
the reason is worth being precise about.

Split conformal needs the calibration set to be exchangeable with the test
point. For degradation rates that was defensible: one rate per compound per
event, and the events can be shuffled without changing anything. Lap times
inside a session are the opposite of exchangeable. The car burns fuel and gets
faster; the track rubbers in and gets faster; a safety car rearranges
everything. A residual from lap 8 and a residual from lap 48 are not draws from
one distribution, and pretending otherwise would produce a guarantee that is
formally stated and practically false.

So two methods, and the comparison is the experiment:

  split       Calibrate on residuals from folds STRICTLY EARLIER in the session
              and apply to the next fold. Never uses the future to calibrate the
              past. Exchangeability is still violated by drift, so its guarantee
              is approximate here -- which is precisely why it is measured rather
              than assumed.

  adaptive    Adaptive Conformal Inference (Gibbs & Candes, NeurIPS 2021).
              Rather than fixing a quantile, it adjusts the working miss-rate
              after every observation:

                  alpha_{t+1} = alpha_t + gamma * (alpha - err_t)

              where err_t is 1 if the observation fell outside the interval. Miss
              too often and the interval widens; miss too rarely and it tightens.
              Long-run coverage converges to 1 - alpha under ARBITRARY
              distribution shift, with no exchangeability assumption at all. That
              is the correct tool for a quantity that drifts, and drift is the
              defining feature of a race.

The falsifiable claim: adaptive should beat both the Gaussian interval and split
conformal on coverage error, and should do so without inflating the interval to
uselessness. If it does not, this file says so.

    python experiments/exp13_lap_time_calibration.py
    python experiments/exp13_lap_time_calibration.py --corpus season --limit 40

Writes experiments/results/exp13_lap_time_calibration.json.
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

from tyremind.data.corpus import load_frames
from tyremind.models.baselines import model_ladder
from tyremind.models.conformal import AdaptiveConformal, conformal_quantile
from tyremind.models.evaluation import rolling_origin_folds

RESULTS = Path(__file__).parent / "results" / "exp13_lap_time_calibration.json"
DEMO_DIR = Path("data/demo")
SEASON_DIR = Path("data/season")

#: Gaussian two-sided 95% multiplier, the interval the ladder reports today.
Z_95 = 1.959964

#: ACI step size. Gibbs & Candes use 0.005-0.05; the trade-off is that a larger
#: gamma tracks a regime change faster and jitters more between laps. 0.02 moves
#: the working alpha by two percentage points per miss, which crosses a race
#: stint in a handful of laps without visibly twitching.
ACI_GAMMA = 0.02

#: Nonconformity scores compared here. See `scale_for` for why the choice that
#: won in experiment 12 is not automatically the right one for lap times.
SCORES = ("absolute", "studentised")

#: Below this many earlier residuals, a conformal quantile at alpha=0.05 does not
#: exist and the method silently returns an infinite interval. Such laps are
#: scored on the Gaussian interval and counted, rather than dropped.
MIN_CALIBRATION = 19


def load_sessions(directory: Path, limit: int) -> dict[str, pd.DataFrame]:
    """Race sessions to score, most recent season first and then in calendar order.

    The selection rule lives in `tyremind.data.corpus` rather than here, because
    which races an experiment runs on is a methodological claim shared with the
    model ladder, and two copies of it would be two claims.
    """
    return load_frames(directory, session_type="R", limit=limit)


def residual_stream(lap_table: pd.DataFrame, model, n_folds: int = 4) -> pd.DataFrame:
    """Per-observation predictions for one model, in session order.

    Returns one row per held-out lap with the model's mean, its posterior sd and
    the truth, tagged by fold so a calibrator can be restricted to the past.
    """
    folds = rolling_origin_folds(lap_table, n_folds=n_folds)
    rows: list[pd.DataFrame] = []
    for i, (train, test) in enumerate(folds):
        model.fit(train)
        mean, sd = model.predict(test)
        rows.append(
            pd.DataFrame(
                {
                    "fold": i,
                    "session_lap": test["session_lap"].to_numpy(),
                    "actual": test["lap_time"].to_numpy(dtype=float),
                    "mean": np.asarray(mean, dtype=float),
                    "sd": np.asarray(sd, dtype=float),
                }
            )
        )
    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True)
    return out.sort_values(["fold", "session_lap"]).reset_index(drop=True)


def score_gaussian(stream: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    half = Z_95 * stream["sd"].to_numpy()
    covered = np.abs(stream["actual"] - stream["mean"]).to_numpy() <= half
    return covered, 2.0 * half


def scale_for(sd: np.ndarray, score: str) -> np.ndarray:
    """The per-observation multiplier a nonconformity score is expressed in.

    The choice is not cosmetic, and LightGBM is the cautionary tale. Studentising
    divides by the model's own posterior sd, which is the right thing when that
    sd carries information about which predictions are shaky. When it does not --
    LightGBM's intervals are so under-dispersed that they cover 62% while
    claiming 95% -- dividing by it amplifies the miscalibration instead of
    correcting it, and the resulting intervals are absurd rather than merely
    wide. The absolute score ignores the sd entirely and cannot do that.
    """
    if score == "studentised":
        return np.where(sd > 0, sd, np.nan)
    return np.ones_like(sd)


def score_split(
    stream: pd.DataFrame, alpha: float, score: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split conformal calibrated only on strictly earlier folds.

    Returns (covered, width, calibrated), where `calibrated` marks the laps that
    actually received a conformal interval. The first fold cannot: there is no
    earlier residual to calibrate on, which is a real property of the setting and
    not an artefact. Those laps are scored on the Gaussian interval and counted,
    because silently dropping the opening quarter of every session would flatter
    the method. Reporting both numbers lets the warm-up cost be seen rather than
    hidden either way.
    """
    residual = np.abs(stream["actual"] - stream["mean"]).to_numpy()
    sd = stream["sd"].to_numpy()
    scale = scale_for(sd, score)
    folds = stream["fold"].to_numpy()

    covered = np.zeros(len(stream), dtype=bool)
    width = np.zeros(len(stream))
    calibrated = np.zeros(len(stream), dtype=bool)

    for fold in np.unique(folds):
        here = folds == fold
        past = folds < fold
        if past.sum() < MIN_CALIBRATION:
            half = Z_95 * sd[here]
        else:
            calibrated[here] = True
            scores = residual[past] / scale[past]
            finite = scores[np.isfinite(scores)]
            q = conformal_quantile(finite, alpha)
            # An unrepresentable quantile means "wider than anything seen", not
            # "revert to the Gaussian" -- see score_adaptive for why that sign
            # matters.
            q = q if np.isfinite(q) else float(finite.max())
            # A lap whose own sd is zero has no studentised scale; it keeps the
            # Gaussian half-width rather than becoming a NaN interval that would
            # silently count as uncovered.
            half = np.where(np.isfinite(scale[here]), q * scale[here], Z_95 * sd[here])
        covered[here] = residual[here] <= half
        width[here] = 2.0 * half
    return covered, width, calibrated


def score_adaptive(stream: pd.DataFrame, alpha: float, gamma: float, score: str) -> tuple:
    """Score one stream with Adaptive Conformal Inference.

    The algorithm itself lives in `tyremind.models.conformal.AdaptiveConformal`
    rather than here. A calibration method that only exists inside an experiment
    is a paper; the live monitor is where an interval that retunes itself every
    lap actually earns its keep, and a second copy of the update rule would drift
    from the tested one.
    """
    calibrator = AdaptiveConformal(
        alpha=alpha, gamma=gamma, score=score, warmup=MIN_CALIBRATION, fallback_z=Z_95
    )

    covered = np.zeros(len(stream), dtype=bool)
    width = np.zeros(len(stream))
    trace = np.zeros(len(stream))

    mean = stream["mean"].to_numpy()
    sd = stream["sd"].to_numpy()
    actual = stream["actual"].to_numpy()

    for t in range(len(stream)):
        trace[t] = calibrator.working_alpha
        width[t] = 2.0 * calibrator.half_width(sd[t])
        covered[t] = calibrator.update(mean[t], sd[t], actual[t])

    return covered, width, trace


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", choices=["demo", "season"], default="season")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--gamma", type=float, default=ACI_GAMMA)
    parser.add_argument("--folds", type=int, default=4)
    args = parser.parse_args()

    warnings.filterwarnings("ignore")
    logging.getLogger("fastf1").setLevel(logging.ERROR)

    directory = SEASON_DIR if args.corpus == "season" else DEMO_DIR
    sessions = load_sessions(directory, args.limit)
    if not sessions:
        raise SystemExit(f"no race sessions found in {directory}")

    target = 1.0 - args.alpha
    print(f"\n  {len(sessions)} race sessions from {directory}\n")

    records: list[dict] = []
    for session_id, lap_table in sessions.items():
        for model in model_ladder():
            try:
                stream = residual_stream(lap_table, model, n_folds=args.folds)
            except Exception as exc:  # noqa: BLE001 - a rung that will not fit is data
                print(f"  {session_id:<32} {model.name:<26} skip {type(exc).__name__}")
                continue
            if stream.empty or len(stream) < MIN_CALIBRATION * 2:
                continue

            g_cov, g_w = score_gaussian(stream)
            row = {
                "session": session_id,
                "model": model.name,
                "n": int(len(stream)),
                "gaussian_coverage": float(g_cov.mean()),
                "gaussian_width": float(np.nanmean(g_w)),
            }
            for score in SCORES:
                s_cov, s_w, s_cal = score_split(stream, args.alpha, score)
                a_cov, a_w, _ = score_adaptive(stream, args.alpha, args.gamma, score)
                row[f"split_{score}_coverage"] = float(s_cov.mean())
                row[f"split_{score}_width"] = float(np.nanmean(s_w))
                row[f"split_{score}_median_width"] = float(np.nanmedian(s_w))
                # Coverage on the laps that actually received a conformal
                # interval, excluding the unavoidable warm-up fold.
                row[f"split_{score}_coverage_calibrated"] = (
                    float(s_cov[s_cal].mean()) if s_cal.any() else None
                )
                row["n_calibrated"] = int(s_cal.sum())
                row[f"adaptive_{score}_coverage"] = float(a_cov.mean())
                row[f"adaptive_{score}_width"] = float(np.nanmean(a_w))
                row[f"adaptive_{score}_median_width"] = float(np.nanmedian(a_w))
            records.append(row)
        print(f"  {session_id:<32} {len(lap_table):>5} laps  "
              f"{sum(1 for r in records if r['session'] == session_id)} rungs scored")

    if not records:
        raise SystemExit("nothing scored")

    df = pd.DataFrame(records)
    methods = ["gaussian"] + [f"{m}_{s}" for m in ("split", "adaptive") for s in SCORES]

    # Weight every lap equally rather than every session, so a 1,300-lap race
    # does not count the same as a 250-lap one.
    def weighted(frame: pd.DataFrame, column: str) -> float:
        return float(np.average(frame[column], weights=frame["n"]))

    summary: list[dict] = []
    for name, block in df.groupby("model", sort=False):
        row = {"model": name, "n_sessions": int(len(block)), "n_laps": int(block["n"].sum())}
        for method in methods:
            row[f"{method}_coverage"] = weighted(block, f"{method}_coverage")
            row[f"{method}_width"] = weighted(block, f"{method}_width")
            row[f"{method}_coverage_error"] = abs(row[f"{method}_coverage"] - target)
            if f"{method}_median_width" in block:
                row[f"{method}_median_width"] = weighted(block, f"{method}_median_width")
        summary.append(row)

    overall = {}
    for method in methods:
        overall[method] = {
            "coverage": float(np.average(df[f"{method}_coverage"], weights=df["n"])),
            "mean_width": float(np.average(df[f"{method}_width"], weights=df["n"])),
            "coverage_error": 0.0,
        }
        if f"{method}_median_width" in df:
            overall[method]["median_width"] = float(
                np.average(df[f"{method}_median_width"], weights=df["n"])
            )
        overall[method]["coverage_error"] = abs(overall[method]["coverage"] - target)

    # The winner is the construction closest to nominal coverage, with the
    # narrower MEDIAN interval breaking ties. Median rather than mean, because a
    # single rung with pathological intervals should not decide the comparison
    # for the other five -- and on this data exactly that happens.
    best = min(
        methods,
        key=lambda m: (
            round(overall[m]["coverage_error"], 3),
            overall[m].get("median_width", overall[m]["mean_width"]),
        ),
    )

    print("\n" + "=" * 96)
    print(f"LAP-TIME INTERVAL CALIBRATION  (target {target:.0%}, "
          f"{df['n'].sum():,} scored laps)")
    print("=" * 96)
    print(f"\n  {'construction':<26}{'coverage':>10}{'mean width':>13}{'median width':>15}"
          f"{'|error|':>10}")
    for method in methods:
        o = overall[method]
        median = o.get("median_width")
        print(f"  {method:<26}{o['coverage']:>9.1%}{o['mean_width']:>12.2f}s"
              f"{(f'{median:.2f}s' if median else '--'):>15}"
              f"{o['coverage_error']:>10.1%}")

    print(f"\n  per rung, best construction ({best}):")
    print(f"  {'model':<28}{'Gaussian':>12}{'calibrated':>13}{'width':>10}")
    for row in sorted(summary, key=lambda r: r[f"{best}_coverage_error"]):
        print(f"  {row['model']:<28}{row['gaussian_coverage']:>11.0%}"
              f"{row[f'{best}_coverage']:>12.0%}"
              f"{row.get(f'{best}_median_width', row[f'{best}_width']):>9.2f}s")

    print("\n" + "=" * 96)
    family = best.split("_")[0]
    if family == "adaptive":
        print("  VERDICT: adaptive conformal wins. Lap times inside a session are not")
        print("           exchangeable -- fuel burns, the track rubbers in, a safety car")
        print("           rearranges everything -- so a fixed quantile calibrated on the")
        print("           past is calibrated for a session that no longer exists. ACI")
        print("           retunes after every lap and assumes no exchangeability at all.")
    elif family == "split":
        print("  VERDICT: split conformal is enough. The drift within a session is mild")
        print("           enough that a fixed quantile from earlier folds still holds,")
        print("           and the extra machinery of ACI is not earning its place.")
    else:
        print("  VERDICT: neither conformal method beats the Gaussian interval. That is")
        print("           a surprise worth chasing rather than a result to report, and")
        print("           it is reported as a failure of this experiment's hypothesis.")

    if best.endswith("absolute"):
        # Named from the data rather than written in. The worst-calibrated rung
        # is the one that makes the argument, and which rung that is -- and by
        # how much -- moves when the corpus does.
        worst = min(summary, key=lambda r: r["gaussian_coverage"])
        studentised = f"{best.split('_')[0]}_studentised"
        print()
        print("  On the score: the ABSOLUTE score wins, and the studentised one -- which")
        print("  won for degradation rates in experiment 12 -- is the wrong tool here.")
        print("  Studentising divides by the model's own posterior sd, which helps only")
        print(f"  when that sd knows something. {worst['model']}'s intervals cover "
              f"{worst['gaussian_coverage']:.0%} while")
        print("  claiming 95%, so dividing by them amplifies the miscalibration rather")
        print(f"  than correcting it: mean width {overall[studentised]['mean_width']:.1f}s "
              f"against {overall[best]['mean_width']:.1f}s for the absolute score.")
        print("  Same machinery, opposite answer, because the question changed.")
    print("=" * 96)

    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    RESULTS.write_text(json.dumps({
        "experiment": "exp13_lap_time_calibration",
        "generated_at": datetime.now(UTC).isoformat(),
        "corpus": args.corpus,
        "alpha": args.alpha,
        "target_coverage": target,
        "aci_gamma": args.gamma,
        "n_sessions": int(df["session"].nunique()),
        "n_scored_laps": int(df["n"].sum()),
        "scores": list(SCORES),
        "per_model": summary,
        "overall": overall,
        "best_method": best,
        "cases": records,
    }, indent=2))
    print(f"\nwrote {RESULTS}")


if __name__ == "__main__":
    main()
