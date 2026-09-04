"""Session catalogue and fitted-model cache behind the API.

Two jobs.

**Offline operation.** Sessions are read from Parquet under `data/demo/` if
present, and only fetched from FastF1 if not. A venue with unreliable network is
the expected environment, not the exceptional one, so the demo path must never
depend on reaching the internet. `scripts/build_demo.py` populates the cache.

**Fit reuse.** Fitting takes a few seconds; serving a chart must not. Fits are
computed once per session and held in memory, so every endpoint after the first
is immediate.
"""

from __future__ import annotations

import json
import logging
import threading
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from tyremind.data.f1_loader import SessionQuality
from tyremind.models.ssm.tyre_ssm import TyreSSMResult, fit_tyre_ssm

logger = logging.getLogger(__name__)

DEMO_DIR = Path("data/demo")
MANIFEST = DEMO_DIR / "manifest.json"


@dataclass(frozen=True)
class SessionRef:
    """A session the API can serve.

    Attributes:
        session_id: Stable slug, e.g. "2024-monza-R".
        year: Season.
        grand_prix: Event name.
        session: Session code -- FP1, FP2, FP3, Q, S or R.
        label: Human-readable name.
        cached: Whether a local Parquet copy exists, meaning it can be served
            with no network access.
    """

    session_id: str
    year: int
    grand_prix: str
    session: str
    label: str
    cached: bool = False

    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "year": self.year,
            "grand_prix": self.grand_prix,
            "session": self.session,
            "label": self.label,
            "cached": self.cached,
        }


@dataclass
class LoadedSession:
    """A session together with its fitted model."""

    ref: SessionRef
    lap_table: pd.DataFrame
    quality: dict
    fit: TyreSSMResult


def slug(year: int, grand_prix: str, session: str) -> str:
    """Stable identifier for a session."""
    return f"{year}-{grand_prix.lower().replace(' ', '-')}-{session}"


class SessionStore:
    """Loads sessions and caches their fits.

    Thread-safe: FastAPI serves requests concurrently, and two simultaneous
    requests for an uncached session would otherwise both pay for the fit. A
    per-session lock means the second waits for the first instead.
    """

    def __init__(self, demo_dir: Path | None = None) -> None:
        self.demo_dir = demo_dir or DEMO_DIR
        self._cache: dict[str, LoadedSession] = {}
        self._locks: dict[str, threading.Lock] = {}
        self._global_lock = threading.Lock()

    def catalogue(self) -> list[SessionRef]:
        """Sessions available to serve, cached ones first.

        Reads the demo manifest if present. Falls back to a small default list
        that will need network access on first use.
        """
        if MANIFEST.exists():
            entries = json.loads(MANIFEST.read_text())
            return [
                SessionRef(
                    session_id=e["session_id"],
                    year=e["year"],
                    grand_prix=e["grand_prix"],
                    session=e["session"],
                    label=e["label"],
                    cached=(self.demo_dir / f"{e['session_id']}.parquet").exists(),
                )
                for e in entries
            ]

        logger.warning(
            "no demo manifest at %s; falling back to a default catalogue that "
            "requires network access. Run scripts/build_demo.py to cache sessions.",
            MANIFEST,
        )
        return [
            SessionRef(slug(2024, "Monza", "R"), 2024, "Monza", "R", "2024 Italian GP - Race"),
            SessionRef(slug(2024, "Monza", "FP2"), 2024, "Monza", "FP2", "2024 Italian GP - FP2"),
        ]

    def _lock_for(self, session_id: str) -> threading.Lock:
        with self._global_lock:
            return self._locks.setdefault(session_id, threading.Lock())

    def get(self, session_id: str) -> LoadedSession:
        """Load a session and its fit, using the cache where possible.

        Args:
            session_id: Slug from `catalogue`.

        Returns:
            The loaded session.

        Raises:
            KeyError: If the id is not in the catalogue.
            RuntimeError: If the session is neither cached nor fetchable.
        """
        if session_id in self._cache:
            return self._cache[session_id]

        with self._lock_for(session_id):
            # Another thread may have finished while we waited.
            if session_id in self._cache:
                return self._cache[session_id]

            ref = next((r for r in self.catalogue() if r.session_id == session_id), None)
            if ref is None:
                available = [r.session_id for r in self.catalogue()]
                raise KeyError(f"unknown session {session_id!r}; available: {available}")

            parquet = self.demo_dir / f"{session_id}.parquet"
            quality_path = self.demo_dir / f"{session_id}.quality.json"

            if parquet.exists():
                logger.info("loading %s from local cache", session_id)
                lap_table = pd.read_parquet(parquet)
                quality = (
                    json.loads(quality_path.read_text()) if quality_path.exists() else {}
                )
            else:
                logger.info("fetching %s from FastF1 (no local cache)", session_id)
                try:
                    from tyremind.data.f1_loader import load_lap_table

                    lap_table, session_quality = load_lap_table(
                        ref.year, ref.grand_prix, ref.session
                    )
                    quality = session_quality.to_dict()
                except Exception as exc:  # noqa: BLE001
                    raise RuntimeError(
                        f"{session_id} is not cached locally and could not be fetched "
                        f"({type(exc).__name__}: {exc}). Run scripts/build_demo.py "
                        "while you have network access."
                    ) from exc

            loaded = LoadedSession(
                ref=ref,
                lap_table=lap_table,
                quality=quality,
                fit=fit_tyre_ssm(lap_table),
            )
            self._cache[session_id] = loaded
            return loaded

    def warm(self, session_ids: list[str] | None = None) -> list[str]:
        """Pre-load sessions so the first request is not the slow one.

        Args:
            session_ids: Sessions to load. Defaults to every cached session.

        Returns:
            The ids successfully loaded. Failures are logged, not raised -- one
            bad session must not stop a demo from starting.
        """
        targets = session_ids or [r.session_id for r in self.catalogue() if r.cached]
        loaded = []
        for session_id in targets:
            try:
                self.get(session_id)
                loaded.append(session_id)
            except Exception as exc:  # noqa: BLE001
                logger.error("could not warm %s: %s", session_id, exc)
        return loaded


def save_session(
    lap_table: pd.DataFrame, quality: SessionQuality, ref: SessionRef, demo_dir: Path | None = None
) -> Path:
    """Write a session to the demo cache.

    Args:
        lap_table: The reduced lap table.
        quality: Its quality report.
        ref: Session reference.
        demo_dir: Target directory.

    Returns:
        Path of the written Parquet file.
    """
    target = demo_dir or DEMO_DIR
    target.mkdir(parents=True, exist_ok=True)

    parquet = target / f"{ref.session_id}.parquet"
    lap_table.to_parquet(parquet, index=False)
    (target / f"{ref.session_id}.quality.json").write_text(
        json.dumps(quality.to_dict(), indent=2, default=str)
    )
    return parquet
