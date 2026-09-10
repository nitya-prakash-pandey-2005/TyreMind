"""Record the conditions each corpus session ran in.

Two things came out of experiment 09 that this exists to serve.

First, a correctness problem. Canada 2024 ran 846 of its 1,272 laps on wet tyres.
We exclude wet compounds and fit whatever dry laps remain -- but those laps sit on
a drying track, where the circuit improves far faster and for longer than the
saturating basis can represent. The model attributes that improvement to the tyre
and reports -0.18 s/lap. A negative degradation rate is precisely the impossible
number we criticise the naive method for producing, and we were producing it
ourselves, silently.

Second, a missing predictor. Las Vegas ran at 17.7 C track temperature and showed
the season's highest degradation; Monza ran at 49.3 C and was unremarkable. That
is the cold-graining side of the bidirectional wear model we already built and
never fed to the estimator.

    python scripts/build_session_conditions.py --year 2024
    python scripts/build_session_conditions.py --years 2024 2023 --sessions R FP2

Writes data/reference/session_conditions.json.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
import warnings
from pathlib import Path

CORPUS = Path("data/season")
OUT = Path("data/reference/session_conditions.json")

#: Above this share of laps on wet or intermediate rubber, the session is not a
#: dry-degradation session and any rate fitted from the dry remainder describes a
#: drying track rather than a tyre.
#:
#: Calibrated against 2024 rather than guessed. Canada ran 66.5% of its laps on
#: wet rubber and produced -0.18 s/lap, an impossible number. Britain ran 24.4%
#: and produced +0.12 to +0.15, which is entirely ordinary. So the boundary sits
#: between them, and a first attempt at 0.20 would have thrown away a perfectly
#: good race. Rainfall alone is NOT a criterion: Spain and Austria both recorded
#: rain with zero wet-tyre laps and gave unremarkable estimates, so rain that
#: never puts anyone on wets is carried as an advisory flag, not an exclusion.
WET_LAP_FRACTION_LIMIT = 0.40


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--years", type=int, nargs="*", default=[2024])
    parser.add_argument("--sessions", nargs="*", default=["R"])
    parser.add_argument("--delay", type=float, default=2.0)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()

    warnings.filterwarnings("ignore")
    logging.getLogger("fastf1").setLevel(logging.ERROR)

    import fastf1

    fastf1.Cache.enable_cache("cache/fastf1")

    manifest = json.loads((CORPUS / "corpus.json").read_text())
    wanted = [
        e for e in manifest
        if e["year"] in args.years and e["session"] in args.sessions
    ]
    wanted.sort(key=lambda e: (e["year"], e["round_number"]))

    existing: dict[str, dict] = {}
    if args.out.exists():
        existing = {r["session_id"]: r for r in json.loads(args.out.read_text())}

    print(f"\n  {len(wanted)} sessions to describe, {len(existing)} already on file\n")
    added = 0

    for entry in wanted:
        if entry["session_id"] in existing:
            continue
        try:
            session = fastf1.get_session(entry["year"], entry["event_name"], entry["session"])
            session.load(telemetry=False, weather=True, messages=False)
            weather = session.weather_data
            laps = session.laps
        except Exception as exc:  # noqa: BLE001 - a session without weather is data
            print(f"  {entry['session_id']:<40} skip  {type(exc).__name__}")
            continue

        wet_laps = int(laps["Compound"].isin(["INTERMEDIATE", "WET"]).sum())
        total_laps = int(len(laps))
        wet_fraction = wet_laps / total_laps if total_laps else 0.0
        rained = bool(weather["Rainfall"].any()) if weather is not None and len(weather) else False

        row = {
            "session_id": entry["session_id"],
            "year": entry["year"],
            "round": entry["round_number"],
            "event": entry["event_name"],
            "session": entry["session"],
            "track_temp_c": float(weather["TrackTemp"].mean()),
            "track_temp_min_c": float(weather["TrackTemp"].min()),
            "track_temp_max_c": float(weather["TrackTemp"].max()),
            "air_temp_c": float(weather["AirTemp"].mean()),
            "humidity_pct": float(weather["Humidity"].mean()),
            "wind_speed_ms": float(weather["WindSpeed"].mean()),
            "rainfall": rained,
            # Advisory only. Rain that never forced a wet tyre did not stop the
            # session being a dry-degradation session.
            "rain_advisory": bool(rained and wet_fraction < WET_LAP_FRACTION_LIMIT),
            "wet_laps": wet_laps,
            "total_laps": total_laps,
            "wet_lap_fraction": wet_fraction,
            # The verdict, computed once here so every consumer agrees on it.
            "dry_session": bool(wet_fraction < WET_LAP_FRACTION_LIMIT),
        }
        existing[entry["session_id"]] = row
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(list(existing.values()), indent=2))
        added += 1

        flag = "WET*" if not row["dry_session"] else ("dry~" if row["rain_advisory"] else "DRY ")
        print(f"  {entry['session_id']:<40} {flag} track {row['track_temp_c']:>5.1f}C  "
              f"wet laps {wet_fraction:>5.1%}")
        time.sleep(args.delay)

    dry = sum(1 for r in existing.values() if r["dry_session"])
    print(f"\n  described {added} sessions, {len(existing)} on file")
    print(f"  {dry} usable as dry-degradation sessions, {len(existing) - dry} excluded")
    print(f"  wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
