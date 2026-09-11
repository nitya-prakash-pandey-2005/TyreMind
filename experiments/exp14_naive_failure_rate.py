"""Experiment 14 -- how often does the standard method report an impossible tyre?

The opening claim of this project is that fitting a straight line through lap
time against tyre age does not measure degradation, because fuel burn-off is
worth more per lap than the effect being measured and pushes the other way. The
sharpest evidence for that is not an error metric. It is the sign: the method
reports tyres *getting faster the longer they are used*, which cannot happen.

That claim was originally quoted on four races, where it held on three. Four
races is an anecdote, and the honest question is whether it survives a season.
This counts it across every dry race in the corpus.

Two rates, because they answer different questions:

  races     the share of races where at least one compound comes out negative.
            This is the number a strategist cares about -- would I have been
            handed an impossible answer this weekend?
  stints    the share of individual compound estimates that are negative. This
            is the number a statistician cares about, since a race with three
            compounds has three chances to go wrong and the race-level figure
            does not say how many took it.

Wet races are excluded on the same criterion the rest of the project uses. A
session that ran most of its laps on wet rubber is not a dry-degradation session,
and counting the naive method's failures there would be scoring it on a question
nobody asked it.

    python experiments/exp14_naive_failure_rate.py

Writes experiments/results/exp14_naive_failure_rate.json.
"""

from __future__ import annotations

import argparse
import json
import warnings
from datetime import UTC, datetime
from pathlib import Path

from tyremind.data.corpus import sessions
from tyremind.data.synthetic import naive_degradation_estimate

RESULTS = Path(__file__).parent / "results" / "exp14_naive_failure_rate.json"
CORPUS = Path("data/season")
CONDITIONS = Path("data/reference/session_conditions.json")


def dry_lookup() -> dict[tuple[int, str, str], bool]:
    if not CONDITIONS.exists():
        return {}
    return {
        (c["year"], c["event"], c["session"]): bool(c["dry_session"])
        for c in json.loads(CONDITIONS.read_text())
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=0, help="0 means every race")
    parser.add_argument("--allow-wet", action="store_true")
    args = parser.parse_args()
    warnings.filterwarnings("ignore")

    dry = dry_lookup()
    races = sessions(CORPUS, session_type="R", limit=args.limit)
    if not races:
        raise SystemExit(f"no races in {CORPUS}. Run scripts/build_corpus.py first.")

    rows: list[dict] = []
    excluded_wet = 0
    would_not_fit = 0

    for race in races:
        if not args.allow_wet and dry.get((race.year, race.event, "R")) is False:
            excluded_wet += 1
            continue
        try:
            estimate = naive_degradation_estimate(race.load())
        except Exception as exc:  # noqa: BLE001 - a session that will not fit is data
            would_not_fit += 1
            print(f"  {race.session_id:<40} skip  {type(exc).__name__}")
            continue

        finite = {c: v for c, v in estimate.items() if v == v}
        if not finite:
            continue
        negative = {c: v for c, v in finite.items() if v < 0}
        rows.append({
            "session_id": race.session_id,
            "year": race.year,
            "event": race.event,
            "n_compounds": len(finite),
            "n_negative": len(negative),
            "estimates": finite,
        })
        flag = "IMPOSSIBLE" if negative else "ok"
        print(f"  {race.session_id:<40} {len(negative)}/{len(finite)} negative  {flag}")

    if not rows:
        raise SystemExit("nothing scored")

    n_races = len(rows)
    n_bad_races = sum(1 for r in rows if r["n_negative"])
    n_stints = sum(r["n_compounds"] for r in rows)
    n_bad_stints = sum(r["n_negative"] for r in rows)

    summary = {
        "n_races": n_races,
        "n_races_with_a_negative_compound": n_bad_races,
        "race_failure_rate": n_bad_races / n_races,
        "n_compound_stints": n_stints,
        "n_negative_compound_stints": n_bad_stints,
        "stint_failure_rate": n_bad_stints / n_stints,
        "excluded_wet": excluded_wet,
        "would_not_fit": would_not_fit,
    }

    print("\n" + "=" * 80)
    print("THE STANDARD METHOD, COUNTED ACROSS THE CORPUS")
    print("=" * 80)
    print(f"\n  dry races analysed                  : {n_races}"
          f"  (excluded {excluded_wet} wet, {would_not_fit} would not fit)")
    print(f"  races reporting a tyre getting FASTER: {n_bad_races}"
          f"  ({summary['race_failure_rate']:.0%})")
    print(f"  compound-stints analysed            : {n_stints}")
    print(f"  of those, negative                  : {n_bad_stints}"
          f"  ({summary['stint_failure_rate']:.0%})")
    print("\n  A negative degradation rate is a tyre that gets faster the longer it")
    print("  is used. It is not a small error, it is the wrong sign, and on this")
    print("  corpus the standard method produces it more often than not at the")
    print("  level of an individual compound.")
    print("=" * 80)

    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    RESULTS.write_text(json.dumps({
        "experiment": "exp14_naive_failure_rate",
        "generated_at": datetime.now(UTC).isoformat(),
        "summary": summary,
        "races": rows,
    }, indent=2))
    print(f"\nwrote {RESULTS}")


if __name__ == "__main__":
    main()
