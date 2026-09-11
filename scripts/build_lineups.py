"""Record which driver drove for which team at each event.

Needed to separate a driver effect from a car effect, which are otherwise
perfectly confounded. "This driver degrades tyres faster" and "the car this
driver sits in degrades tyres faster" make identical predictions until you
compare two drivers in the *same* car -- which is what a teammate is for.

Lineups change mid-season (reserve drivers, mid-year swaps), so this is stored
per event rather than per season. 2024 alone had several.

    python scripts/build_lineups.py --years 2024 2023

Writes data/reference/lineups.json.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
import warnings
from pathlib import Path

CORPUS = Path("data/season")
OUT = Path("data/reference/lineups.json")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--years", type=int, nargs="*", default=[2024, 2023])
    parser.add_argument("--delay", type=float, default=1.0)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()

    warnings.filterwarnings("ignore")
    logging.getLogger("fastf1").setLevel(logging.ERROR)

    import fastf1

    fastf1.Cache.enable_cache("cache/fastf1")

    manifest = json.loads((CORPUS / "corpus.json").read_text())
    events = sorted(
        {(e["year"], e["round_number"], e["event_name"])
         for e in manifest if e["year"] in args.years and e["session"] == "R"}
    )

    existing: dict[str, dict] = {}
    if args.out.exists():
        existing = {f"{r['year']}-{r['round']}": r for r in json.loads(args.out.read_text())}

    print(f"\n  {len(events)} events, {len(existing)} already on file\n")
    added = 0

    for year, rnd, event in events:
        key = f"{year}-{rnd}"
        if key in existing:
            continue
        try:
            session = fastf1.get_session(year, event, "R")
            session.load(telemetry=False, weather=False, messages=False)
            results = session.results
        except Exception as exc:  # noqa: BLE001 - an event without results is data
            print(f"  {year} {event:<32} skip  {type(exc).__name__}")
            continue

        if results is None or "TeamName" not in results:
            print(f"  {year} {event:<32} skip  no team information")
            continue

        lineup = {
            str(row.Abbreviation): str(row.TeamName)
            for row in results.itertuples()
            if isinstance(row.Abbreviation, str) and isinstance(row.TeamName, str)
        }
        if not lineup:
            continue

        existing[key] = {"year": year, "round": rnd, "event": event, "lineup": lineup}
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(list(existing.values()), indent=2))
        added += 1
        print(f"  {year} {event:<32} {len(lineup)} drivers, "
              f"{len(set(lineup.values()))} teams")
        time.sleep(args.delay)

    print(f"\n  recorded {added} events, {len(existing)} on file")
    print(f"  wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
