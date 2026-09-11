"""Experiment 16 -- coverage is right on average; is it right everywhere?

Experiment 13 fixed the headline number: lap-time intervals covered 75% while
claiming 95%, and adaptive conformal brings that to 95.2%. A single coverage
figure is a weak check, though, and it is weak in a specific way. Coverage at
one nominal level says nothing about the *shape* of the predictive distribution.
A model can hit 95% exactly while being far too confident in the middle and far
too timid in the tails, and every number reported so far would look perfect.

Two diagnostics, because they answer different questions.

  PIT HISTOGRAM. The probability integral transform: for each observation, where
  in its own predicted distribution did the truth land? Under a correctly
  specified predictive distribution those values are uniform on [0, 1]. The shape
  of the departure names the fault:

      U-shaped        too many observations in the tails -- intervals too NARROW
      hump-shaped     too many in the middle -- intervals too WIDE
      sloped          the mean is biased, not the spread
      spike at an end one-sided bias

  This applies to the Gaussian predictive distribution, which is a full
  distribution and can therefore be transformed.

  RELIABILITY CURVE. Conformal does not produce a distribution, only an interval
  at a chosen level, so the PIT is not defined for it. The equivalent check is to
  sweep the nominal level from 50% to 99% and measure empirical coverage at each.
  A calibrated method traces the diagonal. A method tuned to look right at 95%
  and wrong everywhere else does not, and this is the only test that would catch
  it.

The point of running both is that they can disagree, and the disagreement is
informative: a method can trace the reliability diagonal while its PIT is still
lumpy, because coverage is an integral and shape is not.

    python experiments/exp16_calibration_shape.py
    python experiments/exp16_calibration_shape.py --limit 20

Writes experiments/results/exp16_calibration_shape.json.
"""

from __future__ import annotations

import argparse
import json
import warnings
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from tyremind.data.corpus import load_frames
from tyremind.models.baselines import model_ladder
from tyremind.models.conformal import AdaptiveConformal
from tyremind.models.evaluation import rolling_origin_folds

RESULTS = Path(__file__).parent / "results" / "exp16_calibration_shape.json"
CORPUS = Path("data/season")

#: Nominal levels swept for the reliability curve. Dense near the top, because
#: that is where decisions are made and where a miscalibration costs most.
LEVELS = [0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.95, 0.98, 0.99]

#: Bins in the PIT histogram. Twenty gives 5% resolution, which is enough to see
#: a U or a hump without turning sampling noise into a diagnosis.
PIT_BINS = 20


def residual_stream(lap_table: pd.DataFrame, model, n_folds: int = 4) -> pd.DataFrame:
    """One row per held-out lap: the model's mean, its sd, and the truth."""
    rows: list[pd.DataFrame] = []
    for i, (train, test) in enumerate(rolling_origin_folds(lap_table, n_folds=n_folds)):
        model.fit(train)
        mean, sd = model.predict(test)
        rows.append(pd.DataFrame({
            "fold": i,
            "session_lap": test["session_lap"].to_numpy(),
            "actual": test["lap_time"].to_numpy(dtype=float),
            "mean": np.asarray(mean, dtype=float),
            "sd": np.asarray(sd, dtype=float),
        }))
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True).sort_values(["fold", "session_lap"])


def pit_values(stream: pd.DataFrame) -> np.ndarray:
    """Where the truth landed inside its own Gaussian predictive distribution."""
    usable = stream[stream["sd"] > 0]
    if usable.empty:
        return np.array([])
    return stats.norm.cdf(
        (usable["actual"].to_numpy() - usable["mean"].to_numpy()) / usable["sd"].to_numpy()
    )


def diagnose_pit(pit: np.ndarray) -> dict:
    """Name the fault from the shape, rather than leaving a histogram to the reader."""
    if pit.size < 100:
        return {"n": int(pit.size), "verdict": "too few observations"}

    counts, _ = np.histogram(pit, bins=PIT_BINS, range=(0.0, 1.0))
    share = counts / counts.sum()
    expected = 1.0 / PIT_BINS

    # Tail mass against middle mass. Under uniformity both equal their width.
    tails = float(share[:2].sum() + share[-2:].sum())
    middle = float(share[PIT_BINS // 2 - 2 : PIT_BINS // 2 + 2].sum())
    expected_tails = 4 * expected
    expected_middle = 4 * expected

    # Kolmogorov-Smirnov against uniform: does the departure exceed sampling noise?
    ks_stat, ks_p = stats.kstest(pit, "uniform")
    # Mean away from 0.5 is a bias in the mean, not the spread.
    bias = float(pit.mean() - 0.5)

    if tails > expected_tails * 1.25:
        verdict = "U-shaped: intervals too NARROW, overconfident"
    elif middle > expected_middle * 1.25:
        verdict = "hump-shaped: intervals too WIDE, underconfident"
    elif abs(bias) > 0.05:
        verdict = f"sloped: the mean is biased {'high' if bias < 0 else 'low'}"
    else:
        verdict = "close to uniform"

    return {
        "n": int(pit.size),
        "tail_mass": tails,
        "expected_tail_mass": expected_tails,
        "middle_mass": middle,
        "mean": float(pit.mean()),
        "bias_from_centre": bias,
        "ks_statistic": float(ks_stat),
        "ks_p": float(ks_p),
        "uniform": bool(ks_p > 0.05),
        "histogram": share.round(4).tolist(),
        "verdict": verdict,
    }


def reliability(streams: list[pd.DataFrame], levels: list[float]) -> dict:
    """Empirical coverage at each nominal level, for Gaussian and adaptive conformal.

    The adaptive calibrator is run PER RACE and the hits pooled, rather than over
    one long concatenation of every race. That matters: ACI carries state, and
    letting it run across a race boundary would hand it a head start it would
    never have on a Sunday morning. Each race starts it cold, exactly as the live
    monitor does.
    """
    gaussian_hits = {level: [] for level in levels}
    adaptive_hits = {level: [] for level in levels}

    for stream in streams:
        residual = np.abs(stream["actual"] - stream["mean"]).to_numpy()
        sd = stream["sd"].to_numpy()
        mean = stream["mean"].to_numpy()
        actual = stream["actual"].to_numpy()

        for level in levels:
            z = stats.norm.ppf(0.5 + level / 2.0)
            gaussian_hits[level].extend((residual <= z * sd).tolist())

            # A fresh calibrator per level and per race: its state IS the level
            # it is chasing, and it must not inherit the previous race's.
            aci = AdaptiveConformal(alpha=1.0 - level, score="absolute")
            adaptive_hits[level].extend(
                aci.update(mean[t], sd[t], actual[t]) for t in range(len(stream))
            )

    return {
        "levels": levels,
        "gaussian": [float(np.mean(gaussian_hits[v])) for v in levels],
        "adaptive": [float(np.mean(adaptive_hits[v])) for v in levels],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=10, help="races to use")
    parser.add_argument("--folds", type=int, default=4)
    args = parser.parse_args()
    warnings.filterwarnings("ignore")

    sessions = load_frames(CORPUS, session_type="R", limit=args.limit)
    if not sessions:
        raise SystemExit(f"no races in {CORPUS}. Run scripts/build_corpus.py first.")
    print(f"\n  {len(sessions)} races\n")

    per_model: dict[str, list[pd.DataFrame]] = {}
    for session_id, lap_table in sessions.items():
        for model in model_ladder():
            try:
                stream = residual_stream(lap_table, model, n_folds=args.folds)
            except Exception as exc:  # noqa: BLE001 - a rung that will not fit is data
                print(f"  {session_id:<34} {model.name:<28} skip {type(exc).__name__}")
                continue
            if not stream.empty:
                per_model.setdefault(model.name, []).append(stream)
        print(f"  {session_id:<34} {len(per_model)} rungs so far")

    if not per_model:
        raise SystemExit("nothing scored")

    findings: dict[str, dict] = {}
    for name, frames in per_model.items():
        stream = pd.concat(frames, ignore_index=True)
        findings[name] = {
            "n_laps": int(len(stream)),
            "pit": diagnose_pit(pit_values(stream)),
            "reliability": reliability(frames, LEVELS),
        }

    print("\n" + "=" * 88)
    print("IS THE INTERVAL THE RIGHT SHAPE, NOT JUST THE RIGHT SIZE?")
    print("=" * 88)

    print(f"\n  PIT of the Gaussian predictive distribution ({PIT_BINS} bins, uniform if correct)")
    print(f"  {'model':<28}{'tail mass':>11}{'expected':>10}{'KS p':>9}   diagnosis")
    for name, f in findings.items():
        pit = f["pit"]
        if "tail_mass" not in pit:
            continue
        print(f"  {name:<28}{pit['tail_mass']:>11.3f}{pit['expected_tail_mass']:>10.3f}"
              f"{pit['ks_p']:>9.3f}   {pit['verdict']}")

    print("\n  Reliability: empirical coverage at each nominal level")
    header = "  " + "level".ljust(28) + "".join(f"{int(v * 100):>6}" for v in LEVELS)
    print(header)
    for name, f in findings.items():
        rel = f["reliability"]
        print("  " + f"{name} · gaussian".ljust(28)
              + "".join(f"{v * 100:>6.0f}" for v in rel["gaussian"]))
        print("  " + f"{name} · adaptive".ljust(28)
              + "".join(f"{v * 100:>6.0f}" for v in rel["adaptive"]))

    # Mean absolute gap from the diagonal, across every level and every rung.
    def diagonal_gap(key: str) -> float:
        gaps = [
            abs(v - level)
            for f in findings.values()
            for level, v in zip(f["reliability"]["levels"], f["reliability"][key], strict=True)
        ]
        return float(np.mean(gaps))

    gauss_gap, adaptive_gap = diagonal_gap("gaussian"), diagonal_gap("adaptive")
    overconfident = [n for n, f in findings.items() if "NARROW" in f["pit"].get("verdict", "")]

    print("\n" + "=" * 88)
    print("  mean distance from the reliability diagonal, over all levels and rungs:")
    print(f"    gaussian  {gauss_gap:.3f}")
    print(f"    adaptive  {adaptive_gap:.3f}")
    print()
    if overconfident:
        print(f"  The PIT is U-shaped for {len(overconfident)} of {len(findings)} rungs, which is")
        print("  overconfidence -- too many observations land in the tails of their own")
        print("  predicted distribution. That is the same fault the coverage number found,")
        print("  and seeing it in the shape rather than at one level is the point: the")
        print("  intervals are not merely mis-sized, the distribution is the wrong width.")
    else:
        print("  No rung shows a U-shaped PIT, so the Gaussian spread is not the problem.")
    print()
    if adaptive_gap < gauss_gap:
        print(f"  Adaptive conformal tracks the diagonal {gauss_gap / max(adaptive_gap, 1e-9):.1f}x")
        print("  closer across the whole sweep, not only at 95%. A method tuned to look")
        print("  right at one level and wrong everywhere else would fail here, and this")
        print("  is the only test in the project that would have caught it.")
    else:
        print("  Adaptive conformal does NOT track the diagonal better across the sweep.")
        print("  Its 95% figure is therefore a point result rather than a calibrated")
        print("  method, and should be quoted as one.")
    print("=" * 88)

    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    RESULTS.write_text(json.dumps({
        "experiment": "exp16_calibration_shape",
        "generated_at": datetime.now(UTC).isoformat(),
        "n_races": len(sessions),
        "pit_bins": PIT_BINS,
        "levels": LEVELS,
        "gaussian_diagonal_gap": gauss_gap,
        "adaptive_diagonal_gap": adaptive_gap,
        "overconfident_rungs": overconfident,
        "per_model": findings,
    }, indent=2))
    print(f"\nwrote {RESULTS}")


if __name__ == "__main__":
    main()
