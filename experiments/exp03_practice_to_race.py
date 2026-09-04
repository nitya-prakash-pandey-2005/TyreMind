"""Experiment 03 -- does a Friday degradation curve predict Sunday?

Estimates degradation from each event's practice session, then scores it against
the same event's race. No race data reaches the practice fit.

This is the hardest honest test in the project: practice and race differ in fuel
load, traffic density, track state and driving style all at once, so it is a
genuine out-of-distribution transfer rather than a holdout split.

    python experiments/exp03_practice_to_race.py --year 2024 --events Bahrain Monza Silverstone

Writes experiments/results/exp03_practice_to_race.json.

Needs network access on first run; afterwards it replays from the FastF1 cache.
"""

from __future__ import annotations

import argparse
import json
import logging
import warnings
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from tyremind.models.validation import summarise_reports, validate_practice_to_race

RESULTS = Path(__file__).parent / "results" / "exp03_practice_to_race.json"

#: A spread of circuit types rather than a convenient handful: low-degradation
#: power tracks, high-degradation abrasive ones, a street circuit and a couple of
#: high-energy aerodynamic ones. Cherry-picking three friendly events would
#: produce a better-looking table and a worse result.
DEFAULT_EVENTS = [
    "Bahrain",       # abrasive, high degradation
    "Barcelona",     # high lateral energy, the traditional tyre test
    "Silverstone",   # high-speed corners, big lateral loads
    "Monza",         # low degradation, fuel-effect dominated
    "Zandvoort",     # banked, sustained lateral load
    "Suzuka",        # high energy, both directions
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--year", type=int, default=2024)
    parser.add_argument("--events", nargs="*", default=DEFAULT_EVENTS)
    parser.add_argument("--practice", default="FP2")
    args = parser.parse_args()

    warnings.filterwarnings("ignore")
    logging.getLogger("fastf1").setLevel(logging.ERROR)

    reports = []
    failures: dict[str, str] = {}

    for event in args.events:
        try:
            report = validate_practice_to_race(
                args.year, event, practice_session=args.practice
            )
        except Exception as exc:  # noqa: BLE001 - an event failing must not stop the sweep
            failures[event] = f"{type(exc).__name__}: {exc}"
            print(f"  {event:<14} SKIPPED -- {type(exc).__name__}: {exc}")
            continue

        reports.append(report)
        print(
            f"  {event:<14} {len(report.comparisons)} compounds  "
            f"MAE {report.mae:.4f}  naive {report.naive_mae:.4f}  "
            f"bias {report.bias:+.4f}  coverage {report.coverage:.0%}"
        )

    if not reports:
        raise SystemExit(f"no events could be validated. Failures: {failures}")

    table = summarise_reports(reports)

    all_errors = np.array([c.error for r in reports for c in r.comparisons])
    all_naive = np.array(
        [c.naive_error for r in reports for c in r.comparisons if np.isfinite(c.naive_error)]
    )
    all_covered = np.array([c.covered for r in reports for c in r.comparisons])

    print("\n" + "=" * 78)
    print(f"PRACTICE ({args.practice}) -> RACE VALIDATION, {args.year}")
    print("=" * 78)
    print(f"{'event':<16}{'compound':<10}{'predicted':>12}{'actual':>10}{'error':>10}{'in 95%':>9}")
    for report in reports:
        for c in sorted(report.comparisons, key=lambda x: x.compound):
            print(
                f"{report.event[:15]:<16}{c.compound:<10}"
                f"{c.predicted:>+9.4f}+-{c.predicted_sd:<4.3f}"
                f"{c.actual:>+10.4f}{c.error:>+10.4f}{'yes' if c.covered else 'NO':>9}"
            )

    print("-" * 78)
    print(f"events validated        : {len(reports)}")
    print(f"compound comparisons    : {all_errors.size}")
    print(f"TyreMind MAE            : {np.abs(all_errors).mean():.4f} s/lap")
    if all_naive.size:
        print(f"naive practice MAE      : {np.abs(all_naive).mean():.4f} s/lap")
    print(f"bias                    : {all_errors.mean():+.4f} s/lap")
    print(f"95% interval coverage   : {all_covered.mean():.0%}")
    if failures:
        print(f"events skipped          : {', '.join(failures)}")
    print("=" * 78)

    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    RESULTS.write_text(
        json.dumps(
            {
                "experiment": "exp03_practice_to_race",
                "generated_at": datetime.now(UTC).isoformat(),
                "year": args.year,
                "practice_session": args.practice,
                "overall": {
                    "n_events": len(reports),
                    "n_comparisons": int(all_errors.size),
                    "mae": float(np.abs(all_errors).mean()),
                    "naive_mae": float(np.abs(all_naive).mean()) if all_naive.size else None,
                    "bias": float(all_errors.mean()),
                    "coverage_95": float(all_covered.mean()),
                },
                "per_event": table.to_dict(orient="records"),
                "reports": [r.to_dict() for r in reports],
                "failures": failures,
            },
            indent=2,
            default=str,
        )
    )
    print(f"\nwrote {RESULTS}")


if __name__ == "__main__":
    main()
