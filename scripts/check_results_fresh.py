"""Check that every recorded result was produced by the script as it stands now.

Every figure in the README, the model card, the deck and the dashboard is read
from a file under `experiments/results/`. That is the right design -- nothing is
typed by hand -- but it moves the failure mode rather than removing it. If a
script is edited after its result was written, the documents keep quoting a
number the current code would no longer produce, and nothing complains.

This has bitten the project twice. Once when a committed result predated a cap
introduced in its own script, and once when a headline was quoted from a
five-event sample long after the corpus had grown to twenty-seven.

    python scripts/check_results_fresh.py
    python scripts/check_results_fresh.py --quiet   # exit code only, for CI

Exit code 1 if anything is out of date, so it can gate a commit.

A caveat worth stating, because the alternative is a tool nobody trusts: this
compares timestamps, not behaviour. Editing a docstring marks a result stale
even though no number could have changed. It is deliberately the cautious
direction -- a false alarm costs a re-run, a missed one costs a wrong claim in
front of a judge.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

RESULTS = Path("experiments/results")
EXPERIMENTS = Path("experiments")


def script_for(result: Path) -> Path | None:
    """The experiment script that writes this result file."""
    candidate = EXPERIMENTS / f"{result.stem}.py"
    if candidate.exists():
        return candidate
    matches = sorted(EXPERIMENTS.glob(f"{result.stem}*.py"))
    return matches[0] if matches else None


def generated_at(result: Path) -> dt.datetime | None:
    try:
        stamp = json.loads(result.read_text()).get("generated_at")
    except (json.JSONDecodeError, OSError):
        return None
    if not stamp:
        return None
    try:
        return dt.datetime.fromisoformat(stamp).astimezone(dt.UTC)
    except ValueError:
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quiet", action="store_true", help="print nothing, exit code only")
    args = parser.parse_args()

    rows: list[tuple[str, str, float]] = []
    for result in sorted(RESULTS.glob("exp*.json")):
        script = script_for(result)
        if script is None:
            rows.append((result.stem, "no script found", 0.0))
            continue

        written = generated_at(result)
        if written is None:
            rows.append((result.stem, "no generated_at stamp", 0.0))
            continue

        # Both sides in UTC. Comparing a UTC stamp against a local mtime reports
        # every result as stale by the machine's offset, which is a fact about
        # the timezone and not about any experiment.
        edited = dt.datetime.fromtimestamp(script.stat().st_mtime, dt.UTC)
        if edited > written:
            rows.append((result.stem, "stale", (edited - written).total_seconds() / 3600))
        else:
            rows.append((result.stem, "ok", 0.0))

    problems = [r for r in rows if r[1] != "ok"]

    if not args.quiet:
        width = max((len(r[0]) for r in rows), default=10)
        for name, status, lag in rows:
            detail = f"script edited {lag:.1f}h later" if status == "stale" else status
            mark = "  " if status == "ok" else "->"
            print(f"{mark} {name:<{width}}  {detail}")
        print()
        if problems:
            print(f"  {len(problems)} of {len(rows)} results are older than their script.")
            print("  Re-run them before quoting their numbers anywhere.")
        else:
            print(f"  all {len(rows)} results were produced by the scripts as they stand.")

    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
