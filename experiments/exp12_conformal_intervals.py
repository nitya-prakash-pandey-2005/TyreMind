"""Experiment 12 -- make the 95% interval mean 95%.

Experiment 03 reports a 95% credible interval around every practice-derived
degradation rate, and across 62 comparisons the race rate falls inside it 76% of
the time. That is not a rounding error. It is the model telling a strategist it
is more certain than it is, which is worse than telling them nothing, because a
strategist can plan around a wide interval and cannot plan around a wrong one.

The cause is not a bug in the filter. The posterior standard deviation the
Kalman/RTS pass produces is the right answer to the question it was asked: given
these practice laps and this model, how well is the practice degradation rate
pinned down? But that is not the question the number gets used for. The user is
asking how well the practice rate predicts SUNDAY, and that transfer carries its
own error -- fuel loads, track state, driving style, tyre management -- none of
which is inside the practice posterior. The interval is honest and answers the
wrong question.

Split conformal prediction fixes exactly this. Given any point predictor and any
nonconformity score, it produces intervals with finite-sample marginal coverage
at least 1 - alpha, with no distributional assumption whatsoever -- not
Gaussianity, not correct model specification, not even that the model is any
good. The only requirement is exchangeability of the calibration set with the
test point.

Two scores, because the choice is a real trade-off:

  absolute    |predicted - actual|. Gives every event the same width. Robust,
              and completely uninformative about which predictions are shaky.
  studentised |predicted - actual| / sd. Width scales with the model's own
              posterior, so a thin practice session gets a wide interval. Only
              beats the absolute score if the posterior sd carries real signal
              about the error, which experiment 10 suggests is marginal.

Calibration is LEAVE-ONE-EVENT-OUT, not leave-one-comparison-out. Two compounds
at the same Grand Prix share a track, a weather window and a fuel correction,
so they are not exchangeable with each other; calibrating on one to predict the
other would leak and would report a coverage we could not reproduce on Sunday.

    python experiments/exp12_conformal_intervals.py
    python experiments/exp12_conformal_intervals.py --alpha 0.10

Writes experiments/results/exp12_conformal_intervals.json.
"""

from __future__ import annotations

import argparse
import json
import warnings
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

RESULTS = Path(__file__).parent / "results" / "exp12_conformal_intervals.json"
EXP03 = Path(__file__).parent / "results" / "exp03_practice_to_race.json"

#: The Gaussian two-sided 95% multiplier the uncalibrated interval uses.
Z_95 = 1.959964

#: Conformal needs enough calibration points that the required quantile exists at
#: all. At alpha=0.05 the ceiling index exceeds the sample until n >= 19, and
#: below that the method silently returns an infinite interval.
MIN_CALIBRATION = 19


def load_cases() -> pd.DataFrame:
    if not EXP03.exists():
        raise SystemExit("run exp03_practice_to_race.py first")
    payload = json.loads(EXP03.read_text())

    rows = []
    for report in payload["reports"]:
        for c in report["comparisons"]:
            sd = c["predicted_sd"]
            if not np.isfinite(sd) or sd <= 0:
                continue
            rows.append({
                "year": report["year"],
                "event": report["event"],
                "compound": c["compound"],
                "predicted": c["predicted"],
                "actual": c["actual"],
                "sd": sd,
                # The race estimate has an interval of its own. Experiment 03's
                # coverage figure combines the two, which answers "are these two
                # estimates consistent?". That is a fair question and a different
                # one from the question here, which is the only one a strategist
                # can act on: does Sunday land inside the interval Friday gave
                # them? Both are reported so the two numbers cannot look like a
                # contradiction between our own documents.
                "combined_sd": float(np.hypot(sd, c["actual_sd"])),
                "error": c["predicted"] - c["actual"],
            })
    return pd.DataFrame(rows)


def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    """The finite-sample-corrected (1-alpha) quantile of the calibration scores.

    The correction is the whole reason conformal has a guarantee rather than an
    asymptotic hope: taking the ceil((n+1)(1-alpha))-th order statistic, rather
    than the plain empirical quantile, is what makes coverage hold at every n.
    """
    n = scores.size
    k = int(np.ceil((n + 1) * (1.0 - alpha)))
    if k > n:
        return float("inf")
    return float(np.sort(scores)[k - 1])


def evaluate(df: pd.DataFrame, alpha: float, *, debias: bool) -> dict:
    """Leave-one-event-out coverage and width for each interval construction."""
    events = sorted({(r.year, r.event) for r in df.itertuples()})
    records: list[dict] = []

    for held in events:
        mask = [(r.year, r.event) == held for r in df.itertuples()]
        test = df[mask]
        calib = df[[not m for m in mask]]
        if len(calib) < MIN_CALIBRATION:
            continue

        # The bias correction is itself fitted on the calibration fold only.
        # Estimating it on everything and then measuring coverage on the same
        # rows would be marking our own homework.
        shift = float(calib["error"].mean()) if debias else 0.0
        calib_err = np.abs(calib["error"].to_numpy() - shift)

        q_abs = conformal_quantile(calib_err, alpha)
        q_stu = conformal_quantile(calib_err / calib["sd"].to_numpy(), alpha)

        for r in test.itertuples():
            point = r.predicted - shift
            residual = abs(point - r.actual)
            records.append({
                "year": r.year,
                "event": r.event,
                "compound": r.compound,
                "residual": residual,
                "gaussian_half_width": Z_95 * r.sd,
                "gaussian_covered": bool(residual <= Z_95 * r.sd),
                "consistency_half_width": Z_95 * r.combined_sd,
                "consistency_covered": bool(residual <= Z_95 * r.combined_sd),
                "absolute_half_width": q_abs,
                "absolute_covered": bool(residual <= q_abs),
                "studentised_half_width": q_stu * r.sd,
                "studentised_covered": bool(residual <= q_stu * r.sd),
            })

    out = pd.DataFrame(records)
    summary = {}
    for name in ("gaussian", "consistency", "absolute", "studentised"):
        summary[name] = {
            "coverage": float(out[f"{name}_covered"].mean()),
            "mean_half_width": float(out[f"{name}_half_width"].mean()),
            "median_half_width": float(out[f"{name}_half_width"].median()),
        }
    summary["n_scored"] = int(len(out))
    summary["n_events"] = int(out.groupby(["year", "event"]).ngroups)
    return {"summary": summary, "cases": records}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--alpha", type=float, default=0.05)
    args = parser.parse_args()
    warnings.filterwarnings("ignore")

    df = load_cases()
    target = 1.0 - args.alpha

    raw = evaluate(df, args.alpha, debias=False)
    fixed = evaluate(df, args.alpha, debias=True)

    print("=" * 84)
    print(f"CALIBRATING THE INTERVAL  ({raw['summary']['n_scored']} comparisons, "
          f"{raw['summary']['n_events']} events, target {target:.0%})")
    print("=" * 84)

    for title, block in (("point prediction as-is", raw),
                         ("with leave-one-event-out bias correction", fixed)):
        s = block["summary"]
        print(f"\n  {title}")
        print(f"  {'construction':<26}{'coverage':>10}{'mean +-':>12}{'median +-':>12}")
        for name in ("gaussian", "consistency", "absolute", "studentised"):
            flag = "" if abs(s[name]["coverage"] - target) < 0.05 else "  <-- off target"
            print(f"  {name:<26}{s[name]['coverage']:>9.0%}"
                  f"{s[name]['mean_half_width']:>12.4f}"
                  f"{s[name]['median_half_width']:>12.4f}{flag}")

    best = min(
        (("absolute", fixed), ("studentised", fixed), ("absolute", raw), ("studentised", raw)),
        key=lambda kv: (
            abs(kv[1]["summary"][kv[0]]["coverage"] - target),
            kv[1]["summary"][kv[0]]["mean_half_width"],
        ),
    )
    name, block = best
    chosen = block["summary"][name]

    print("\n" + "=" * 84)
    print(f"  The Gaussian interval covers {raw['summary']['gaussian']['coverage']:.0%} "
          f"of the time while claiming {target:.0%}. It is answering the wrong")
    print("  question: it describes how well practice pins down the practice rate, not")
    print("  how well the practice rate predicts Sunday.")
    print("\n  'consistency' is experiment 03's criterion -- practice and race intervals")
    print(f"  combined, {raw['summary']['consistency']['coverage']:.0%}. Looser, because it "
          f"is allowed to use Sunday's own")
    print("  uncertainty, which nobody has on Friday.")
    print(f"\n  Conformal ({name}) covers {chosen['coverage']:.0%} at "
          f"+-{chosen['mean_half_width']:.4f} s/lap on average -- a wider interval that")
    print("  is worth what it says it is worth. The guarantee is distribution-free and")
    print("  finite-sample: it does not assume the model is Gaussian, or even correct.")
    print("=" * 84)

    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    RESULTS.write_text(json.dumps({
        "experiment": "exp12_conformal_intervals",
        "generated_at": datetime.now(UTC).isoformat(),
        "alpha": args.alpha,
        "target_coverage": target,
        "min_calibration": MIN_CALIBRATION,
        "uncalibrated": raw["summary"],
        "bias_corrected": fixed["summary"],
        "recommended": {"score": name, "debiased": block is fixed, **chosen},
        "cases": fixed["cases"],
    }, indent=2))
    print(f"\nwrote {RESULTS}")


if __name__ == "__main__":
    main()
