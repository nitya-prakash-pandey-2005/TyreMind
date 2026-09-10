"""Render the HTML documents to the PDFs that get handed to judges.

These PDFs were previously produced by hand, which is how a PDF ends up
disagreeing with the HTML it came from. Every figure in those documents is read
from `experiments/results/*.json`, so when an experiment is re-run at a larger
sample the HTML changes and any stale PDF becomes a document that contradicts
its own source.

    python scripts/render_pdfs.py            # all of them
    python scripts/render_pdfs.py --only roadmap

Needs Chromium via Playwright, the same dependency `scripts/record_demo.py`
already uses:

    python -m playwright install chromium
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

#: (key, source HTML, destination PDF, print background, margin).
#: The deck is landscape slides and the dossier and roadmap are portrait
#: documents, so they cannot share one page setup.
DOCUMENTS = [
    ("deck", Path("docs/pitch/deck.html"), Path("docs/pitch/TyreMind_Pitch_Deck.pdf")),
    ("dossier", Path("docs/pitch/dossier.html"),
     Path("docs/pitch/TyreMind_Technical_Dossier.pdf")),
    ("roadmap", Path("docs/plan/ROADMAP.html"), Path("docs/plan/ROADMAP.pdf")),
]


def render(source: Path, destination: Path, *, timeout_ms: int = 60_000) -> int:
    """Print one HTML file to PDF, returning the resulting size in bytes."""
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        browser = p.chromium.launch()
        try:
            page = browser.new_page()
            page.goto(source.resolve().as_uri(), wait_until="networkidle",
                      timeout=timeout_ms)
            # The documents set their own @page size and margins in CSS. Passing
            # prefer_css_page_size keeps the deck's landscape slides landscape
            # instead of silently reflowing them onto A4 portrait.
            destination.parent.mkdir(parents=True, exist_ok=True)
            page.pdf(
                path=str(destination),
                print_background=True,
                prefer_css_page_size=True,
            )
        finally:
            browser.close()
    return destination.stat().st_size


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--only", nargs="*", choices=[k for k, _, _ in DOCUMENTS],
        help="render a subset; default is all of them",
    )
    args = parser.parse_args()

    wanted = set(args.only) if args.only else {k for k, _, _ in DOCUMENTS}
    failures = 0

    for key, source, destination in DOCUMENTS:
        if key not in wanted:
            continue
        if not source.exists():
            print(f"  {key:<10} SKIP   {source} does not exist")
            failures += 1
            continue
        try:
            size = render(source, destination)
        except Exception as exc:  # noqa: BLE001 - report and keep going
            print(f"  {key:<10} FAIL   {type(exc).__name__}: {exc}")
            failures += 1
            continue
        print(f"  {key:<10} {size / 1_000_000:>5.2f} MB  {destination}")

    if failures:
        print(f"\n  {failures} document(s) not rendered")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
