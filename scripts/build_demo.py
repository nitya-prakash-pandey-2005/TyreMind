"""Cache sessions to disk so the demo never needs a network.

Run this once with connectivity. Afterwards the API, the dashboard and the live
replay all work from `data/demo/` with the network unplugged.

    python scripts/build_demo.py
    python scripts/build_demo.py --events Monza Silverstone --year 2024

Venue wifi is the single most reliable point of failure in a live demo, and it
fails at the worst moment. Nothing in the presentation path is allowed to depend
on it.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import warnings
from pathlib import Path

from tyremind.api.store import DEMO_DIR, MANIFEST, SessionRef, save_session, slug
from tyremind.data.f1_loader import load_lap_table

#: The demo set. Each event contributes a practice session and its race, which is
#: what the practice-to-race story needs. Chosen to span circuit types rather
#: than to flatter the model.
DEFAULT_EVENTS = ["Monza", "Silverstone", "Zandvoort", "Barcelona"]
DEFAULT_SESSIONS = ["FP2", "R"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--year", type=int, default=2024)
    parser.add_argument("--events", nargs="*", default=DEFAULT_EVENTS)
    parser.add_argument("--sessions", nargs="*", default=DEFAULT_SESSIONS)
    parser.add_argument("--out", type=Path, default=DEMO_DIR)
    args = parser.parse_args()

    warnings.filterwarnings("ignore")
    logging.getLogger("fastf1").setLevel(logging.ERROR)

    args.out.mkdir(parents=True, exist_ok=True)
    manifest: list[dict] = []
    failures: dict[str, str] = {}

    for event in args.events:
        for session in args.sessions:
            session_id = slug(args.year, event, session)
            ref = SessionRef(
                session_id=session_id,
                year=args.year,
                grand_prix=event,
                session=session,
                label=f"{args.year} {event} - {session}",
            )
            try:
                lap_table, quality = load_lap_table(args.year, event, session)
            except Exception as exc:  # noqa: BLE001
                failures[session_id] = f"{type(exc).__name__}: {exc}"
                print(f"  {session_id:<28} FAILED  {type(exc).__name__}: {exc}")
                continue

            save_session(lap_table, quality, ref, args.out)
            manifest.append(ref.to_dict())
            print(
                f"  {session_id:<28} {len(lap_table):>4} laps, "
                f"{quality.n_drivers:>2} drivers, quality {quality.score():.0f}/100"
            )

    if not manifest:
        print("\nnothing cached. Check network access and event names.", file=sys.stderr)
        if failures:
            print(json.dumps(failures, indent=2), file=sys.stderr)
        return 1

    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(manifest, indent=2))

    print(f"\ncached {len(manifest)} sessions to {args.out}")
    print(f"manifest: {MANIFEST}")
    if failures:
        print(f"failed: {', '.join(failures)}")
    print("\nthe demo now runs offline.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
