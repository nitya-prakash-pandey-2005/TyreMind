"""Build the research corpus: every session we can get, not just the demo set.

`data/demo/` stays what it is -- eight curated sessions, committed, so a clone
runs offline. This writes somewhere else entirely: `data/season/`, a much larger
and deliberately gitignored corpus that the experiments fit against.

The separation matters. The demo set is a presentation asset chosen to span
circuit types; the corpus is evidence, and evidence should be everything we can
lay hands on. Our entire real-data claim currently rests on ten comparisons
across five events of one season, which is thin enough to be attacked.

    python scripts/build_corpus.py                       # 2024, races + FP2
    python scripts/build_corpus.py --years 2024 2023
    python scripts/build_corpus.py --sessions R FP2 FP1 FP3

Work is ordered by priority and skips anything already on disk, so it survives
being interrupted -- which it will be, because a full season is hours of
downloads and FastF1 fails on individual sessions for reasons that have nothing
to do with us (sessions that never ran, timing feeds that were never published,
red-flagged practice that produced no laps).
"""

from __future__ import annotations

import argparse
import json
import logging
import time
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

CORPUS_DIR = Path("data/season")
MANIFEST = CORPUS_DIR / "corpus.json"

#: Default session priority when --sessions is not given. Races anchor every
#: validation; FP2 carries the long race-simulation runs the practice-to-race
#: claim depends on. An explicit --sessions overrides this order.
SESSION_ORDER = ["R", "FP2", "FP3", "FP1", "Q"]


@dataclass(frozen=True)
class CorpusEntry:
    session_id: str
    year: int
    round_number: int
    event_name: str
    location: str
    country: str
    session: str
    n_laps: int
    n_drivers: int
    quality_score: float
    compounds: list[str]


def targets(years: list[int], sessions: list[str]) -> list[tuple[int, int, str, str, str, str]]:
    """(year, round, event_name, location, country, session), in priority order."""
    import fastf1

    out: list[tuple[int, int, str, str, str, str]] = []
    for year in years:
        try:
            schedule = fastf1.get_event_schedule(year, include_testing=False)
        except Exception as exc:  # noqa: BLE001
            print(f"  {year}: schedule unavailable -- {type(exc).__name__}: {exc}")
            continue
        for _, ev in schedule.iterrows():
            for session in sessions:
                out.append(
                    (
                        year,
                        int(ev["RoundNumber"]),
                        str(ev["EventName"]),
                        str(ev["Location"]),
                        str(ev.get("Country", "")),
                        session,
                    )
                )
    # Priority: the order the caller asked for, then most recent year, then round.
    # Honouring --sessions matters: practice is what the brief is about, so
    # "--sessions FP2 R" has to mean FP2 first rather than being silently
    # re-sorted into a default that puts races ahead of it.
    rank = {s: i for i, s in enumerate(sessions)}
    out.sort(key=lambda t: (rank.get(t[5], 99), -t[0], t[1]))
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--years", type=int, nargs="*", default=[2024])
    parser.add_argument("--sessions", nargs="*", default=SESSION_ORDER[:2])
    parser.add_argument("--out", type=Path, default=CORPUS_DIR)
    parser.add_argument("--limit", type=int, default=0, help="stop after N new sessions")
    parser.add_argument("--delay", type=float, default=2.0, help="seconds between sessions")
    parser.add_argument("--retries", type=int, default=4, help="backoff attempts on a rate limit")
    args = parser.parse_args()

    warnings.filterwarnings("ignore")
    logging.getLogger("fastf1").setLevel(logging.ERROR)

    from tyremind.api.store import slug
    from tyremind.data.f1_loader import load_lap_table

    args.out.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out / "corpus.json"
    existing: dict[str, dict] = {}
    if manifest_path.exists():
        existing = {e["session_id"]: e for e in json.loads(manifest_path.read_text())}

    plan = targets(args.years, args.sessions)
    print(f"  {len(plan)} candidate sessions across {len(args.years)} season(s)")
    print(f"  {len(existing)} already in the corpus\n")

    added = 0
    failures: dict[str, str] = {}
    started = time.perf_counter()

    def attempt_session(year, rnd, event, location, country, session) -> bool:
        """Fetch one session into the corpus. True if it landed."""
        nonlocal added
        session_id = slug(year, event, session)
        parquet = args.out / f"{session_id}.parquet"
        if session_id in existing and parquet.exists():
            return True

        # The upstream timing API rate-limits, and a first pass over three
        # seasons hit it 181 times. Those are not missing sessions -- they are the
        # same sessions, refused. Back off and retry rather than recording a
        # failure that is really our own request rate.
        lap_table = quality = None
        for attempt in range(args.retries):
            try:
                lap_table, quality = load_lap_table(year, event, session)
                break
            except Exception as exc:  # noqa: BLE001 - a missing session is data, not a crash
                if "RateLimit" in type(exc).__name__ and attempt < args.retries - 1:
                    backoff = args.delay * (4 ** (attempt + 1))
                    print(f"  {session_id:<38} rate limited, waiting {backoff:.0f}s")
                    time.sleep(backoff)
                    continue
                failures[session_id] = f"{type(exc).__name__}: {exc}"
                print(f"  {session_id:<38} skip   {type(exc).__name__}")
                break
        if lap_table is None:
            return False
        time.sleep(args.delay)

        if lap_table.empty:
            failures[session_id] = "no usable laps after quality filtering"
            print(f"  {session_id:<38} skip   no usable laps")
            return False

        lap_table.to_parquet(parquet, index=False)
        entry = CorpusEntry(
            session_id=session_id,
            year=year,
            round_number=rnd,
            event_name=event,
            location=location,
            country=country,
            session=session,
            n_laps=int(len(lap_table)),
            n_drivers=int(lap_table["driver"].nunique()),
            quality_score=float(quality.score()),
            compounds=sorted(lap_table["compound"].unique().tolist()),
        )
        existing[session_id] = asdict(entry)
        manifest_path.write_text(json.dumps(list(existing.values()), indent=2))
        added += 1
        failures.pop(session_id, None)
        print(
            f"  {session_id:<38} {entry.n_laps:>5} laps  {entry.n_drivers:>2} cars  "
            f"q{entry.quality_score:>3.0f}  {'/'.join(entry.compounds)}"
        )
        return True

    for target in plan:
        if args.limit and added >= args.limit:
            break
        attempt_session(*target)

    # A second pass over whatever the upstream refused.
    #
    # This is not belt-and-braces, it is the difference between a corpus and a
    # corpus with holes in it. The first 2025 scrape recorded 6 races of 24 and
    # then moved on to practice sessions, because each rate-limited race was
    # written off after its retries and never revisited. Nothing errored. The
    # corpus simply had gaps where the API had said no, and the gaps were in the
    # most valuable sessions because those were attempted first, while the limit
    # was at its tightest.
    #
    # Refusals are transient by definition, so they are retried once more at a
    # deliberately slower pace rather than recorded as missing data.
    retryable = [
        target for target in plan
        if slug(target[0], target[2], target[5]) in failures
        and "RateLimit" in failures[slug(target[0], target[2], target[5])]
    ]
    if retryable and not (args.limit and added >= args.limit):
        print(f"\n  second pass over {len(retryable)} rate-limited sessions, "
              f"at {args.delay * 3:.0f}s between requests\n")
        slow = args.delay
        args.delay = slow * 3
        for target in retryable:
            if args.limit and added >= args.limit:
                break
            attempt_session(*target)
        args.delay = slow

    elapsed = time.perf_counter() - started
    print(f"\n  added {added} sessions in {elapsed / 60:.1f} min")
    print(f"  corpus now holds {len(existing)} sessions")
    if failures:
        rate_limited = sum(1 for reason in failures.values() if "RateLimit" in reason)
        print(f"  {len(failures)} unavailable (expected: sessions that never ran, "
              "or no timing feed)")
        if rate_limited:
            # Worth separating. A session that never ran is data; a session the
            # API refused twice is a gap, and re-running this script will try it
            # again because the corpus is resumable.
            print(f"  of which {rate_limited} were STILL rate-limited after a second "
                  "pass -- re-run to pick them up")

    if existing:
        df = pd.DataFrame(list(existing.values()))
        # Grouped by (year, event), not by name. Event names repeat across
        # seasons -- there is a British Grand Prix every year -- so counting
        # unique names reports a quarter of the evidence actually held.
        n_events = df.groupby(["year", "event_name"]).ngroups
        print(f"\n  {df['n_laps'].sum():,} laps \u00b7 {n_events} event-seasons \u00b7 "
              f"{df['year'].nunique()} season(s)")
        print(f"  manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
