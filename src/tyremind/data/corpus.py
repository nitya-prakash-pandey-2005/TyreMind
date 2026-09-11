"""Choosing which sessions an experiment runs on, in one place and on the record.

Which races an experiment scores is a methodological claim, not a loading
detail. "Twenty races" invites the obvious question -- which twenty, and who
chose them? -- and the only good answer is a rule stated up front that nobody
could have tuned after seeing a result.

An earlier version of this selection took `reversed(sorted(paths))`, which reads
as "most recent first" and is not. Sorting filenames alphabetically and walking
backwards happened to put 2024 ahead of 2023, but inside a season it ordered by
event *name*, so asking for twenty races silently dropped Abu Dhabi, Australia
and Austria for no reason anyone could defend.

The rule here is: most recent season first, and within a season in calendar
order. A limit therefore means "the N most recent races", which is a sentence
that can be written in a paper.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

CORPUS_DIR = Path("data/season")
DEMO_DIR = Path("data/demo")
MANIFEST = "corpus.json"

#: A race with fewer laps than this cannot be split into several chronological
#: folds and still leave enough in each block to score anything.
MIN_LAPS = 200


@dataclass(frozen=True)
class CorpusSession:
    """One session on disk, with enough identity to order it."""

    session_id: str
    path: Path
    year: int
    round_number: int
    event: str
    session: str

    def load(self) -> pd.DataFrame:
        return pd.read_parquet(self.path)


def _from_manifest(directory: Path) -> list[CorpusSession]:
    manifest = directory / MANIFEST
    if not manifest.exists():
        return []
    out: list[CorpusSession] = []
    for entry in json.loads(manifest.read_text()):
        path = directory / f"{entry['session_id']}.parquet"
        if path.exists():
            out.append(
                CorpusSession(
                    session_id=entry["session_id"],
                    path=path,
                    year=int(entry["year"]),
                    round_number=int(entry["round_number"]),
                    event=str(entry["event_name"]),
                    session=str(entry["session"]),
                )
            )
    return out


def _from_filenames(directory: Path) -> list[CorpusSession]:
    """Fallback for a directory with no manifest, such as the committed demo set.

    Round number is unknown here, so ordering inside a season falls back to the
    name. That is stated rather than disguised: the demo set is eight curated
    sessions and its order is a presentation choice, not evidence.
    """
    out: list[CorpusSession] = []
    for path in sorted(directory.glob("*.parquet")):
        stem = path.stem
        year = int(stem[:4]) if stem[:4].isdigit() else 0
        out.append(
            CorpusSession(
                session_id=stem,
                path=path,
                year=year,
                round_number=0,
                event=stem,
                session=stem.rsplit("-", 1)[-1],
            )
        )
    return out


def sessions(
    directory: Path = CORPUS_DIR,
    *,
    session_type: str | None = "R",
    limit: int = 0,
    min_laps: int = MIN_LAPS,
    years: list[int] | None = None,
) -> list[CorpusSession]:
    """Sessions to run an experiment on, most recent first.

    Args:
        directory: Corpus directory to read.
        session_type: Keep only this session code, e.g. "R". None keeps all.
        limit: Keep at most this many. Zero means all of them.
        min_laps: Drop sessions with fewer laps than this, since they cannot
            support a chronological split.
        years: Restrict to these seasons.

    Returns:
        Sessions ordered by season descending then round ascending, so that a
        limit selects the most recent races in calendar order rather than an
        alphabetical accident.
    """
    found = _from_manifest(directory) or _from_filenames(directory)

    if session_type is not None:
        found = [s for s in found if s.session == session_type]
    if years:
        found = [s for s in found if s.year in years]

    found.sort(key=lambda s: (-s.year, s.round_number, s.session_id))

    kept: list[CorpusSession] = []
    for entry in found:
        if limit and len(kept) >= limit:
            break
        # Reading the frame is the only way to know its lap count, and the cost
        # is paid once here rather than by every caller.
        if min_laps and len(entry.load()) < min_laps:
            continue
        kept.append(entry)
    return kept


def load_frames(*args, **kwargs) -> dict[str, pd.DataFrame]:
    """`sessions` with the frames already read, keyed by session id."""
    return {s.session_id: s.load() for s in sessions(*args, **kwargs)}
