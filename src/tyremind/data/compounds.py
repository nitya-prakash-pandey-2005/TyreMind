"""Resolve a weekend's relative tyre label to the compound actually fitted.

Pirelli nominates three of the C1-C5 slick range for each Grand Prix and labels
them HARD / MEDIUM / SOFT *for that weekend*. The labels are relative, so the
same word means different rubber at different circuits:

    Monza 2024        C3 / C4 / C5   -> "MEDIUM" is C4
    Silverstone 2024  C1 / C2 / C3   -> "MEDIUM" is C2

Pooling those two as one compound is a specification error, not a simplification.
It biases every cross-circuit comparison, throws away the evidence that a given
compound accumulates across a season, and makes predicting an unseen circuit
impossible -- there is nothing to predict *for*, because "MEDIUM" is not a
physical thing.

This module turns the label into the compound. Everything downstream that wants
to pool across events should pool on `compound_id`, not on the label.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import pandas as pd

ALLOCATION_PATH = Path("data/reference/compound_allocation.json")

#: Relative label -> offset from the weekend's hardest nomination. Pirelli has
#: nominated three consecutive compounds since 2023, so one number per event is
#: enough to place all three.
LABEL_OFFSET = {"HARD": 0, "MEDIUM": 1, "SOFT": 2}

#: Compounds outside the dry range. They are a different physical process with
#: different priors, and are excluded upstream rather than mapped.
NON_SLICK = {"INTERMEDIATE", "WET", "UNKNOWN", "TEST_UNKNOWN"}


@dataclass(frozen=True)
class Allocation:
    """One event's nomination."""

    year: int
    round_number: int
    event: str
    hardest: int
    confidence: str
    source: str

    def compound_id(self, label: str) -> str | None:
        """The C-number worn under `label` at this event, e.g. "C4"."""
        offset = LABEL_OFFSET.get(label.upper())
        if offset is None:
            return None
        return f"C{self.hardest + offset}"


@lru_cache(maxsize=1)
def load_allocations(path: Path | None = None) -> dict[tuple[int, int], Allocation]:
    """Allocations keyed by (year, round).

    Raises:
        FileNotFoundError: If the reference table is missing.
        ValueError: If a nomination is outside the range that can be consecutive.
    """
    target = path or ALLOCATION_PATH
    if not target.exists():
        raise FileNotFoundError(
            f"{target} is missing. It is committed reference data, not generated."
        )
    raw = json.loads(target.read_text(encoding="utf-8"))

    out: dict[tuple[int, int], Allocation] = {}
    for row in raw["allocations"]:
        hardest = int(row["hardest"])
        # Three consecutive compounds from C1-C5 means the hardest can only be
        # C1, C2 or C3. Anything else means either a typo or a genuine change in
        # how Pirelli nominates, and both deserve to fail loudly.
        if not 1 <= hardest <= 3:
            raise ValueError(
                f"{row['year']} round {row['round']}: hardest=C{hardest} cannot start a "
                "consecutive triplet within C1-C5"
            )
        out[(int(row["year"]), int(row["round"]))] = Allocation(
            year=int(row["year"]),
            round_number=int(row["round"]),
            event=str(row["event"]),
            hardest=hardest,
            confidence=str(row["confidence"]),
            source=str(row["source"]),
        )
    return out


def resolve(year: int, round_number: int, label: str) -> str | None:
    """Compound actually fitted, or None if the event or label is unknown."""
    if label.upper() in NON_SLICK:
        return None
    allocation = load_allocations().get((year, round_number))
    return allocation.compound_id(label) if allocation else None


def annotate(lap_table: pd.DataFrame, year: int, round_number: int) -> pd.DataFrame:
    """Add a `compound_id` column carrying the compound actually fitted.

    Rows whose event is not in the reference table, or whose label is not a dry
    slick, get None. That is deliberate: a missing allocation should show up as a
    gap to be filled, never as a silent fallback to the relative label, because
    the fallback is exactly the error this module exists to remove.

    Args:
        lap_table: Lap table carrying a `compound` column.
        year: Season.
        round_number: Championship round.

    Returns:
        A copy with `compound_id` added.
    """
    out = lap_table.copy()
    allocation = load_allocations().get((year, round_number))
    if allocation is None:
        out["compound_id"] = None
        return out
    out["compound_id"] = out["compound"].map(
        lambda c: allocation.compound_id(c) if str(c).upper() not in NON_SLICK else None
    )
    return out


def coverage() -> pd.DataFrame:
    """What the reference table covers, for auditing gaps before trusting it."""
    rows = [
        {
            "year": a.year,
            "round": a.round_number,
            "event": a.event,
            "hard": a.compound_id("HARD"),
            "medium": a.compound_id("MEDIUM"),
            "soft": a.compound_id("SOFT"),
            "confidence": a.confidence,
        }
        for a in sorted(load_allocations().values(), key=lambda x: (x.year, x.round_number))
    ]
    return pd.DataFrame(rows)
