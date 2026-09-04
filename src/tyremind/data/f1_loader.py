"""Load real F1 sessions and reduce them to the estimator's lap table.

Wraps FastF1, which reads F1's official timing API. Everything here is about
getting from "what the API returns" to "laps we are willing to draw a
degradation conclusion from", and the second is a much smaller set than the first.

The filtering is the substantive part. A raw session contains in-laps, out-laps,
safety-car laps, laps set on a damaged car and laps where the driver simply gave
up on the corner. Every one of them is a real lap time, and every one of them
will corrupt a degradation estimate if left in. `SessionQuality` records what was
removed and why, so that the exclusions are auditable rather than hidden inside
a chain of boolean masks.
"""

from __future__ import annotations

import logging
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

#: FastF1 marks a lap's compound UNKNOWN when timing data is incomplete. Such a
#: lap cannot be assigned to a tyre set, so it cannot inform degradation.
_UNUSABLE_COMPOUNDS = {"UNKNOWN", "TEST_UNKNOWN", "nan", "None", ""}

#: Wet-weather compounds. Degradation on a wet tyre is a different physical
#: process (thermal, driven by water clearance and overheating on a drying line)
#: and the model's priors do not describe it. Excluded, and reported as excluded.
_WET_COMPOUNDS = {"INTERMEDIATE", "WET"}

DEFAULT_CACHE = Path("cache/fastf1")


@dataclass
class SessionQuality:
    """An audit trail for what was dropped between the API and the lap table.

    Every degradation number the platform publishes rests on the surviving laps,
    so how many there were, and what happened to the rest, is part of the result
    rather than a debugging detail.

    Attributes:
        session_name: Human-readable session identifier.
        total_laps: Laps returned by the API.
        exclusions: Reason to number of laps removed. Ordered by application.
        retained_laps: Laps in the final table.
        n_drivers: Drivers with at least one retained lap.
        n_runs: Distinct tyre-set runs in the final table.
        compounds: Retained lap count per compound.
        median_lap_time: Session reference pace, s.
        longest_run: Most laps in a single retained run. Degradation is estimated
            from within-run trends, so this is the practical limit on what the
            session can say about a cliff.
    """

    session_name: str
    total_laps: int = 0
    exclusions: dict[str, int] = field(default_factory=dict)
    retained_laps: int = 0
    n_drivers: int = 0
    n_runs: int = 0
    compounds: dict[str, int] = field(default_factory=dict)
    median_lap_time: float = float("nan")
    longest_run: int = 0

    @property
    def retention_rate(self) -> float:
        """Fraction of laps that survived filtering."""
        return self.retained_laps / self.total_laps if self.total_laps else 0.0

    def score(self) -> float:
        """A 0-100 summary of whether this session can support a conclusion.

        Deliberately blunt, and deliberately shown to the user. It rewards the
        three things that actually determine whether degradation is estimable:
        enough laps survived, runs are long enough to show a trend, and enough
        cars ran to break the tyre-age/session-lap collinearity.
        """
        retention = min(self.retention_rate / 0.6, 1.0)
        run_length = min(self.longest_run / 12.0, 1.0)
        field_size = min(self.n_drivers / 15.0, 1.0)
        return float(100.0 * (0.3 * retention + 0.4 * run_length + 0.3 * field_size))

    def to_dict(self) -> dict:
        out = asdict(self)
        out["retention_rate"] = self.retention_rate
        out["quality_score"] = self.score()
        return out


def configure_cache(path: str | Path | None = None) -> Path:
    """Point FastF1 at an on-disk cache and make sure it exists.

    Caching is not an optimisation here, it is what makes the demo work at a
    venue with no usable network. Sessions fetched once are replayed from disk.

    Args:
        path: Cache directory. Falls back to $TYREMIND_FASTF1_CACHE, then to
            ./cache/fastf1.

    Returns:
        The directory now in use.
    """
    import fastf1

    chosen = Path(path or os.environ.get("TYREMIND_FASTF1_CACHE") or DEFAULT_CACHE)
    chosen.mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(str(chosen))
    logger.info("FastF1 cache at %s", chosen)
    return chosen


def load_session(
    year: int,
    grand_prix: str | int,
    session: str,
    *,
    cache: str | Path | None = None,
    telemetry: bool = False,
):
    """Fetch one session from FastF1.

    Args:
        year: Season, e.g. 2024.
        grand_prix: Event name ("Monza") or round number.
        session: "FP1", "FP2", "FP3", "Q", "S" or "R".
        cache: Cache directory. See `configure_cache`.
        telemetry: Whether to load car and position telemetry as well. Adds tens
            of megabytes and a lot of time, so it is off unless the physics layer
            actually needs it.

    Returns:
        The loaded `fastf1.core.Session`.
    """
    import fastf1

    configure_cache(cache)
    loaded = fastf1.get_session(year, grand_prix, session)
    loaded.load(laps=True, telemetry=telemetry, weather=True, messages=False)
    return loaded


def _mark_runs(laps: pd.DataFrame) -> pd.DataFrame:
    """Assign a run id to each contiguous block of laps on one tyre set.

    A run breaks when the driver changes, the stint number changes, the compound
    changes, or tyre age steps backwards -- the last of which catches a set swap
    that the stint numbering missed.
    """
    laps = laps.sort_values(["Driver", "LapNumber"]).reset_index(drop=True)

    age = laps["TyreLife"].to_numpy(dtype=float)
    new_run = (
        (laps["Driver"] != laps["Driver"].shift())
        | (laps["Stint"] != laps["Stint"].shift())
        | (laps["Compound"] != laps["Compound"].shift())
        | (np.diff(age, prepend=age[0]) < 0)
    )
    laps["run_id"] = new_run.cumsum().astype(int)
    return laps


#: Gap to the car ahead, in seconds, beyond which a car is treated as being in
#: clean air. A following car loses roughly 15-20% of downforce within about a
#: second; by two seconds the wake effect is essentially gone.
CLEAN_AIR_GAP_S = 2.0

#: Gap at or below which traffic is treated as maximally compromising.
FULL_TRAFFIC_GAP_S = 0.4


def _traffic_index(laps: pd.DataFrame) -> pd.Series:
    """Estimate how compromised each lap was by the car ahead, on a 0-1 scale.

    Two cars that cross the start line n seconds apart are n seconds apart on
    track for the lap that follows, so the gap to the car ahead is recoverable
    from lap start times alone -- no positional telemetry needed.

    Crucially the comparison is across *session time*, not lap number. In a
    practice session two drivers' "lap 5" can be forty minutes apart, so grouping
    by lap number compares cars that were never on track together. Sorting every
    lap in the session by start time and looking at the immediately preceding one
    is what actually identifies who was behind whom.

    The mapping from gap to index saturates at both ends. We deliberately do not
    convert gap into seconds lost -- the model estimates that coefficient from
    data. This function only has to be monotone in "how much traffic" and leave
    the magnitude to be inferred.

    Args:
        laps: Lap rows carrying `Driver` and `LapStartTime`.

    Returns:
        Traffic index per lap in [0, 1], aligned to `laps.index`.
    """
    if "LapStartTime" not in laps.columns or laps["LapStartTime"].isna().all():
        return pd.Series(0.0, index=laps.index)

    start = pd.to_timedelta(laps["LapStartTime"]).dt.total_seconds()
    drivers = laps["Driver"].to_numpy()

    order = np.argsort(start.to_numpy(dtype=float), kind="stable")
    ordered_times = start.to_numpy(dtype=float)[order]
    ordered_drivers = drivers[order]

    gaps = np.full(len(order), np.inf)
    for position in range(1, len(order)):
        # Walk back to the most recent lap started by a *different* car. Skipping
        # the same driver matters: a car's own previous lap start is one lap time
        # earlier, and treating that as a gap would mark every lap as clear.
        look_back = position - 1
        while look_back >= 0 and ordered_drivers[look_back] == ordered_drivers[position]:
            look_back -= 1
        if look_back >= 0:
            gaps[position] = ordered_times[position] - ordered_times[look_back]

    span = CLEAN_AIR_GAP_S - FULL_TRAFFIC_GAP_S
    scaled = np.clip((CLEAN_AIR_GAP_S - gaps) / span, 0.0, 1.0)
    scaled[~np.isfinite(gaps)] = 0.0

    index = pd.Series(0.0, index=laps.index)
    index.iloc[order] = scaled
    return index


def build_lap_table(
    session,
    *,
    min_run_laps: int = 4,
    outlier_sd: float = 3.0,
    include_wet: bool = False,
) -> tuple[pd.DataFrame, SessionQuality]:
    """Reduce a FastF1 session to the lap table the estimator consumes.

    The exclusions, in the order applied:

      * **No lap time** -- the driver did not complete a timed lap.
      * **Pit in/out laps** -- a lap entering or leaving the pit lane is
        dominated by the pit lane, not the tyre.
      * **Not accurate** -- FastF1's own flag for laps its timing reconstruction
        does not trust.
      * **Unknown compound** -- cannot be assigned to a tyre set.
      * **Wet compounds** -- a different degradation physics; see module notes.
      * **Slow laps** -- more than `outlier_sd` robust deviations above the
        session median. Catches safety cars, yellow flags, traffic so severe the
        lap says nothing about the tyre, and aborted push laps. A robust
        threshold is used because the mean and standard deviation of a session's
        lap times are themselves wrecked by these laps.
      * **Short runs** -- a run of fewer than `min_run_laps` laps cannot show a
        trend, and contributes an intercept the model has to pay for with no
        information in return.

    Args:
        session: A loaded `fastf1.core.Session`.
        min_run_laps: Shortest run to retain.
        outlier_sd: Robust deviations above the median beyond which a lap is
            treated as compromised.
        include_wet: Retain intermediate and wet compounds. The model's priors do
            not describe them, so this is off by default.

    Returns:
        A ``(lap_table, quality)`` pair. The lap table matches the schema
        `fit_tyre_ssm` expects; `quality` records what was dropped.

    Raises:
        ValueError: If no laps survive filtering.
    """
    name = f"{getattr(session, 'event', {}).get('EventName', '?')} {getattr(session, 'name', '?')}"
    laps = session.laps.copy()
    quality = SessionQuality(session_name=name, total_laps=len(laps))

    def drop(mask: pd.Series, reason: str) -> pd.DataFrame:
        nonlocal laps
        removed = int(mask.sum())
        if removed:
            quality.exclusions[reason] = removed
        laps = laps.loc[~mask].copy()
        return laps

    drop(laps["LapTime"].isna(), "no_lap_time")
    drop(laps["PitInTime"].notna() | laps["PitOutTime"].notna(), "pit_in_out_lap")

    if "IsAccurate" in laps.columns:
        drop(~laps["IsAccurate"].fillna(False).astype(bool), "flagged_inaccurate")

    compound = laps["Compound"].astype(str).str.upper()
    drop(compound.isin(_UNUSABLE_COMPOUNDS), "unknown_compound")

    if not include_wet:
        drop(laps["Compound"].astype(str).str.upper().isin(_WET_COMPOUNDS), "wet_compound")

    if laps.empty:
        raise ValueError(f"no usable laps remain for {name} after filtering")

    seconds = laps["LapTime"].dt.total_seconds()
    median = float(seconds.median())
    # Median absolute deviation, scaled to be comparable to a standard deviation
    # for normal data. Robust because the very laps we are trying to remove would
    # otherwise inflate an ordinary standard deviation and hide themselves.
    mad = float((seconds - median).abs().median()) * 1.4826
    threshold = median + outlier_sd * max(mad, 0.05)
    drop(seconds > threshold, "slow_lap_safety_car_or_traffic")

    if laps.empty:
        raise ValueError(f"no usable laps remain for {name} after outlier removal")

    laps = _mark_runs(laps)

    run_sizes = laps.groupby("run_id")["LapNumber"].transform("size")
    drop(run_sizes < min_run_laps, "run_too_short")

    if laps.empty:
        raise ValueError(
            f"no runs of at least {min_run_laps} laps remain for {name}; "
            "this session cannot support a degradation estimate"
        )

    laps = _mark_runs(laps)  # re-number so run ids stay contiguous

    lap_table = pd.DataFrame(
        {
            "driver": laps["Driver"].astype(str).to_numpy(),
            "session_lap": laps["LapNumber"].astype(int).to_numpy(),
            "run_id": laps["run_id"].astype(int).to_numpy(),
            "tyre_age": laps["TyreLife"].astype(float).to_numpy(),
            "lap_time": laps["LapTime"].dt.total_seconds().to_numpy(),
            "compound": laps["Compound"].astype(str).str.upper().to_numpy(),
            "traffic_index": _traffic_index(laps).to_numpy(),
        }
    )

    # Laps completed on this run, which drives the fuel term. Distinct from tyre
    # age: a scrubbed set arrives with laps already on it, and that difference is
    # one of the few things that helps separate fuel from degradation.
    lap_table["lap_in_run"] = lap_table.groupby("run_id").cumcount()

    lap_table = lap_table.sort_values(["session_lap", "driver"]).reset_index(drop=True)

    quality.retained_laps = len(lap_table)
    quality.n_drivers = int(lap_table["driver"].nunique())
    quality.n_runs = int(lap_table["run_id"].nunique())
    quality.compounds = lap_table["compound"].value_counts().to_dict()
    quality.median_lap_time = float(lap_table["lap_time"].median())
    quality.longest_run = int(lap_table.groupby("run_id").size().max())

    return lap_table, quality


def load_lap_table(
    year: int,
    grand_prix: str | int,
    session: str,
    *,
    cache: str | Path | None = None,
    **kwargs,
) -> tuple[pd.DataFrame, SessionQuality]:
    """Fetch a session and reduce it to a lap table in one call.

    Args:
        year: Season.
        grand_prix: Event name or round number.
        session: "FP1", "FP2", "FP3", "Q", "S" or "R".
        cache: FastF1 cache directory.
        **kwargs: Passed through to `build_lap_table`.

    Returns:
        A ``(lap_table, quality)`` pair.
    """
    return build_lap_table(load_session(year, grand_prix, session, cache=cache), **kwargs)
