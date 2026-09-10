"""Describe every circuit in the corpus, so a model can predict one it has not seen.

Geometry is cheap: FastF1's circuit info comes with the session data we already
cache, so `--geometry-only` costs nothing beyond what the corpus scrape already
pulled. Energetics need a fast lap's telemetry, which is a much heavier download
and competes with the scrape for the same rate limit -- hence the flag, and hence
the default being the cheap half.

    python scripts/build_circuit_features.py --geometry-only
    python scripts/build_circuit_features.py --with-telemetry --year 2024

Writes data/reference/circuit_features.json.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
import warnings
from pathlib import Path

CORPUS = Path("data/season")
OUT = Path("data/reference/circuit_features.json")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--year", type=int, default=2024)
    parser.add_argument("--with-telemetry", action="store_true")
    parser.add_argument("--geometry-only", action="store_true")
    parser.add_argument("--delay", type=float, default=1.5)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()

    warnings.filterwarnings("ignore")
    logging.getLogger("fastf1").setLevel(logging.ERROR)

    import fastf1

    from tyremind.physics.circuit_features import from_session

    fastf1.Cache.enable_cache("cache/fastf1")
    with_telemetry = args.with_telemetry and not args.geometry_only

    manifest = json.loads((CORPUS / "corpus.json").read_text())
    events = sorted(
        {(e["year"], e["round_number"], e["event_name"], e["location"])
         for e in manifest if e["year"] == args.year and e["session"] == "R"}
    )

    existing: dict[str, dict] = {}
    if args.out.exists():
        existing = {f"{r['year']}-{r['circuit']}": r for r in json.loads(args.out.read_text())}

    print(f"\n  {len(events)} circuits from the {args.year} corpus"
          f"  ({'geometry + energetics' if with_telemetry else 'geometry only'})\n")

    added = 0
    for year, rnd, event, location in events:
        key = f"{year}-{location}"
        # Only recompute when the cheap version is being upgraded to the full one.
        if key in existing and not (with_telemetry and not existing[key].get("lateral_energy_proxy")):
            continue
        try:
            session = fastf1.get_session(year, event, "R")
            session.load(telemetry=with_telemetry, weather=False, messages=False)
            features = from_session(session, with_telemetry=with_telemetry)
        except Exception as exc:  # noqa: BLE001 - a circuit we cannot describe is data
            print(f"  {location:<26} skip  {type(exc).__name__}")
            continue

        row = features.to_dict()
        row["round"] = rnd
        row["event"] = event
        existing[key] = row
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(list(existing.values()), indent=2))
        added += 1

        energy = (f"  energy {features.lateral_energy_proxy:>7.0f}"
                  if features.has_energetics else "")
        print(f"  {location:<26} {features.n_corners:>2} corners "
              f"({features.n_tight_corners} tight)  "
              f"{features.lap_length_m / 1000:>5.2f} km{energy}")
        time.sleep(args.delay)

    print(f"\n  described {added} circuits, {len(existing)} on file")
    print(f"  wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
