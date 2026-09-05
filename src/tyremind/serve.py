"""Start TyreMind: one command, no network required.

    python -m tyremind.serve

Serves the API and the built dashboard from a single process, warms the fitted
models so the first click is not the slow one, and opens a browser.

If the dashboard has not been built, the API still runs and the command says so
rather than failing silently -- a working API with no UI is a recoverable
situation, and worth distinguishing from a broken install.
"""

from __future__ import annotations

import argparse
import logging
import sys
import threading
import time
import webbrowser
from pathlib import Path

WEB_DIST = Path("apps/web/dist")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8077)
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--no-warm", action="store_true", help="skip pre-fitting sessions")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)-8s %(message)s")
    logging.getLogger("fastf1").setLevel(logging.ERROR)

    import uvicorn

    from tyremind.api.main import app, store

    catalogue = store.catalogue()
    cached = [r for r in catalogue if r.cached]

    print("\n  TYREMIND")
    print("  Causal tyre intelligence\n")
    print(f"  sessions      {len(catalogue)} ({len(cached)} cached locally)")

    if not cached:
        print("\n  No sessions are cached. Run this while you have network access:")
        print("      python scripts/build_demo.py\n")
    if not WEB_DIST.exists():
        print("\n  Dashboard not built. The API will run without it. To build:")
        print("      cd apps/web && npm install && npm run build\n")

    if not args.no_warm and cached:
        print("  warming     fitting cached sessions…")
        started = time.perf_counter()
        warmed = store.warm()
        print(f"              {len(warmed)} ready in {time.perf_counter() - started:.1f}s")

    url = f"http://{args.host}:{args.port}"
    print(f"\n  dashboard   {url}")
    print(f"  api docs    {url}/docs\n")

    if not args.no_browser:
        threading.Timer(1.2, lambda: webbrowser.open(url)).start()

    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
    return 0


if __name__ == "__main__":
    sys.exit(main())
