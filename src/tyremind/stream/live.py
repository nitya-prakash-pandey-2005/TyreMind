"""Online tyre-state estimation: one lap in, updated state out.

The same model that produces a smoothed degradation curve on Monday runs
forward-only during the session. That is not a second implementation -- it is
the same recursion with the backward pass omitted, which is what a Kalman filter
is *for*. The choice of a recursive estimator over an MCMC one was made largely
for this: a sampler has no incremental mode, and re-running it every lap is not
an option on a pit wall.

Cost per lap is one propagate and one rank-one update, both O(n^2) in the state
dimension and independent of how long the session has been running. Lap 60 costs
exactly what lap 1 cost.


Two estimates, never conflated
------------------------------
    filtered  -- conditioned on laps up to now. What the pit wall can legitimately
                 know at this moment.
    smoothed  -- conditioned on the whole session. What the engineers know
                 afterwards, and always the tighter of the two.

Reporting a smoothed number as if it had been available live is the single
easiest way to make a strategy tool look prescient in review and be useless in
practice. This module only ever produces the filtered estimate, and the API
labels it as such.


Hyperparameters
---------------
Variance hyperparameters are *not* re-optimised per lap; that would cost seconds
and defeat the point. They are supplied from a prior fit -- practice for a race,
or a comparable historical session -- and held fixed. `recalibrate` exists for
when a session drifts far enough to warrant a refit, and is intended to be
called occasionally and off the hot path.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from tyremind.models.ssm.kalman import _symmetrise
from tyremind.models.ssm.tyre_ssm import (
    DIFFUSE_VARIANCE,
    TyreSSMHyper,
    TyreSSMPriors,
    track_basis,
)


@dataclass(frozen=True)
class LapObservation:
    """One completed lap, as it arrives.

    Attributes:
        driver: Car identifier.
        session_lap: Session lap counter. Must not decrease across the stream.
        lap_time: Lap time, seconds.
        compound: Tyre compound in use.
        tyre_age: Laps on this set, including any from a previous session.
        lap_in_run: Laps completed on the current run. Drives the fuel term.
        run_index: Which run this is for this driver, counting from zero.
        traffic_index: Traffic severity in [0, 1]. Zero if unknown.
        wall_clock: Optional real timestamp, for replay pacing.
    """

    driver: str
    session_lap: int
    lap_time: float
    compound: str
    tyre_age: float
    lap_in_run: int
    run_index: int
    traffic_index: float = 0.0
    wall_clock: float | None = None


@dataclass
class LiveTyreState:
    """The current filtered estimate for one car.

    Attributes:
        driver: Car identifier.
        session_lap: Lap this estimate is current as of.
        compound: Compound in use.
        tyre_age: Laps on the set.
        performance_loss: Estimated seconds lost relative to this set when new.
        performance_loss_sd: Its posterior standard deviation.
        degradation_rate: Estimated instantaneous degradation, s/lap.
        degradation_rate_sd: Its posterior standard deviation.
        laps_observed: Laps folded in for this car so far.
        innovation: Latest prediction error -- observed minus predicted lap
            time, seconds. A run of large same-signed innovations means the
            model is being surprised, which is worth surfacing.
        innovation_z: That error in units of its own predicted standard
            deviation. Above about 3 the lap was not what the model expected.
    """

    driver: str
    session_lap: int
    compound: str
    tyre_age: float
    performance_loss: float
    performance_loss_sd: float
    degradation_rate: float
    degradation_rate_sd: float
    laps_observed: int
    innovation: float = float("nan")
    innovation_z: float = float("nan")

    @property
    def health_index(self) -> float:
        """A 0-100 summary of how much competitive performance the tyre retains.

        Anchored so that a fresh set reads 100 and a set that has lost 1.5 s/lap
        reads 0. That anchor is a *convention*, not a measurement, and the number
        is a rescaling of `performance_loss` with no additional information in
        it. It exists because "76" is easier to act on at a glance than
        "0.36 s/lap of accumulated loss", and it is always displayed alongside
        the seconds it came from.

        This is emphatically not tread depth. Public telemetry contains no such
        thing.
        """
        return float(np.clip(100.0 * (1.0 - self.performance_loss / 1.5), 0.0, 100.0))

    def to_dict(self) -> dict:
        return {
            "driver": self.driver,
            "session_lap": int(self.session_lap),
            "compound": self.compound,
            "tyre_age": self.tyre_age,
            "performance_loss": self.performance_loss,
            "performance_loss_sd": self.performance_loss_sd,
            "degradation_rate": self.degradation_rate,
            "degradation_rate_sd": self.degradation_rate_sd,
            "health_index": self.health_index,
            "laps_observed": self.laps_observed,
            "innovation": self.innovation,
            "innovation_z": self.innovation_z,
            "estimate_type": "filtered",
        }


class LiveTyreMonitor:
    """Recursive tyre-state estimator for a session in progress.

    Pre-allocates the state vector for a known entry list and a run budget, so
    that no reallocation is needed mid-session. The layout mirrors the batch
    model exactly, which is what lets `test_live_matches_batch_filter` assert the
    two produce identical filtered estimates.

    Args:
        drivers: Entry list.
        compounds: Compounds expected this session.
        hyper: Variance hyperparameters, from a prior fit.
        priors: Prior specification.
        max_runs_per_driver: Run intercepts reserved per car. Exceeding it raises
            rather than silently reusing a slot, which would pool two unrelated
            runs into one intercept and corrupt both.
        reference_time: Lap time the model works relative to, for conditioning.
            Defaults to the first observed lap.
    """

    def __init__(
        self,
        drivers: list[str],
        compounds: list[str],
        hyper: TyreSSMHyper | None = None,
        priors: TyreSSMPriors | None = None,
        *,
        max_runs_per_driver: int = 8,
        reference_time: float | None = None,
    ) -> None:
        self.drivers = list(drivers)
        self.compounds = list(compounds)
        self.hyper = hyper or TyreSSMHyper()
        self.priors = priors or TyreSSMPriors()
        self.max_runs_per_driver = max_runs_per_driver
        self.reference_time = reference_time

        cursor = 0

        def take() -> int:
            nonlocal cursor
            i = cursor
            cursor += 1
            return i

        self.i_track_amplitude = take()
        self.i_track_resid = take()
        self.i_fuel_slope = take()
        self.i_traffic_coef = take()
        self.i_compound_rate = {c: take() for c in self.compounds}
        self.i_level: dict[str, int] = {}
        self.i_rate: dict[str, int] = {}
        for d in self.drivers:
            self.i_level[d] = take()
            self.i_rate[d] = take()
        self.i_run: dict[tuple[str, int], int] = {}
        for d in self.drivers:
            for r in range(max_runs_per_driver):
                self.i_run[(d, r)] = take()

        self.n_state = cursor

        self.a = np.zeros(self.n_state)
        self.P = np.zeros((self.n_state, self.n_state))
        self._initialise_prior()

        self.session_start_lap: int | None = None
        self.current_lap: int | None = None
        self._active_run: dict[str, int] = {}
        self._prev_age: dict[str, float] = {}
        self._laps_seen: dict[str, int] = dict.fromkeys(self.drivers, 0)
        self._last_state: dict[str, LiveTyreState] = {}
        self.total_laps = 0
        self.update_times_ms: list[float] = []

    def _initialise_prior(self) -> None:
        p = self.priors
        self.a[self.i_track_amplitude] = p.track_amplitude_mean
        self.P[self.i_track_amplitude, self.i_track_amplitude] = p.track_amplitude_sd**2
        self.P[self.i_track_resid, self.i_track_resid] = 1e-4

        self.a[self.i_fuel_slope] = p.fuel_slope_mean
        self.P[self.i_fuel_slope, self.i_fuel_slope] = p.fuel_slope_sd**2

        self.P[self.i_traffic_coef, self.i_traffic_coef] = p.traffic_coef_sd**2

        for i in self.i_compound_rate.values():
            self.a[i] = p.compound_rate_mean
            self.P[i, i] = p.compound_rate_sd**2

        for i in self.i_run.values():
            self.P[i, i] = p.run_intercept_sd**2

        for d in self.drivers:
            self.P[self.i_level[d], self.i_level[d]] = DIFFUSE_VARIANCE
            self.P[self.i_rate[d], self.i_rate[d]] = DIFFUSE_VARIANCE

    def _advance_session_lap(self) -> None:
        """Carry the track residual forward one lap. All other states hold."""
        self.P[self.i_track_resid, self.i_track_resid] += self.hyper.q_track

    def _fit_new_set(self, driver: str, compound: str) -> None:
        """Apply a tyre change: loss returns to zero, rate redraws around the compound.

        Expressed exactly as in the batch model, so the two agree.
        """
        i_level, i_rate = self.i_level[driver], self.i_rate[driver]
        i_base = self.i_compound_rate[compound]

        # Level goes to a known zero: clear its row and column, then set variance.
        self.a[i_level] = 0.0
        self.P[i_level, :] = 0.0
        self.P[:, i_level] = 0.0
        self.P[i_level, i_level] = self.priors.initial_level_sd**2

        # Rate is drawn around the compound baseline, inheriting that state's
        # uncertainty and its correlations. This is the hierarchical shrinkage.
        self.a[i_rate] = self.a[i_base]
        self.P[i_rate, :] = self.P[i_base, :]
        self.P[:, i_rate] = self.P[:, i_base]
        self.P[i_rate, i_rate] = self.P[i_base, i_base] + self.hyper.stint_rate_var

    def _advance_tyre(self, driver: str, delta_age: float) -> None:
        """Age a tyre by `delta_age` laps under the local linear trend."""
        if delta_age <= 0:
            return
        i_level, i_rate = self.i_level[driver], self.i_rate[driver]

        # Level absorbs the rate: a row operation, not a full matrix product.
        self.a[i_level] += delta_age * self.a[i_rate]
        self.P[i_level, :] += delta_age * self.P[i_rate, :]
        self.P[:, i_level] += delta_age * self.P[:, i_rate]

        self.P[i_level, i_level] += self.hyper.q_level * delta_age
        self.P[i_rate, i_rate] += self.hyper.q_rate * delta_age
        self.P = _symmetrise(self.P)

    def _design_row(self, obs: LapObservation) -> np.ndarray:
        z = np.zeros(self.n_state)
        # Explicit None check, not `or`: session_lap is legitimately 0 in
        # synthetic sessions, and a falsy-zero fallback here silently pins the
        # track basis at zero for the whole session, pushing every bit of track
        # evolution into the degradation rates instead.
        start = self.session_start_lap if self.session_start_lap is not None else obs.session_lap
        lap_offset = float(obs.session_lap - start)
        z[self.i_track_amplitude] = float(track_basis(np.array([lap_offset]), self.hyper.track_shape)[0])
        z[self.i_track_resid] = 1.0
        z[self.i_run[(obs.driver, obs.run_index)]] = 1.0
        z[self.i_level[obs.driver]] = 1.0
        z[self.i_fuel_slope] = -float(obs.lap_in_run)
        z[self.i_traffic_coef] = float(obs.traffic_index)
        return z

    def observe(self, obs: LapObservation) -> LiveTyreState:
        """Fold one completed lap into the estimate.

        Args:
            obs: The lap.

        Returns:
            The car's updated filtered state.

        Raises:
            ValueError: If the driver or compound was not declared at
                construction, the run index exceeds the reserved budget, or the
                session lap goes backwards.
        """
        started = time.perf_counter()

        if obs.driver not in self.i_level:
            raise ValueError(
                f"{obs.driver} was not in the entry list this monitor was built for "
                f"({sorted(self.i_level)})"
            )
        if obs.compound not in self.i_compound_rate:
            raise ValueError(
                f"compound {obs.compound!r} was not declared; known compounds are "
                f"{sorted(self.i_compound_rate)}"
            )
        if obs.run_index >= self.max_runs_per_driver:
            raise ValueError(
                f"{obs.driver} is on run {obs.run_index} but only "
                f"{self.max_runs_per_driver} run slots were reserved. Rebuild the "
                "monitor with a larger max_runs_per_driver."
            )

        if self.session_start_lap is None:
            self.session_start_lap = obs.session_lap
            self.current_lap = obs.session_lap
        if self.reference_time is None:
            self.reference_time = obs.lap_time

        if self.current_lap is not None and obs.session_lap < self.current_lap:
            raise ValueError(
                f"session lap went backwards: got {obs.session_lap} after "
                f"{self.current_lap}. The stream must be ordered."
            )

        # Transition. Advance the session clock, then this car's tyre.
        while self.current_lap is not None and self.current_lap < obs.session_lap:
            self._advance_session_lap()
            self.current_lap += 1

        if self._active_run.get(obs.driver) != obs.run_index:
            self._fit_new_set(obs.driver, obs.compound)
            self._active_run[obs.driver] = obs.run_index
        else:
            self._advance_tyre(obs.driver, obs.tyre_age - self._prev_age.get(obs.driver, 0.0))
        self._prev_age[obs.driver] = obs.tyre_age

        # Update.
        z = self._design_row(obs)
        y = obs.lap_time - (self.reference_time or 0.0)

        Pz = self.P @ z
        F = float(z @ Pz + self.hyper.obs_var)
        v = float(y - z @ self.a)

        K = Pz / F
        self.a = self.a + K * v
        self.P = _symmetrise(self.P - np.outer(K, Pz))

        self._laps_seen[obs.driver] += 1
        self.total_laps += 1
        self.update_times_ms.append((time.perf_counter() - started) * 1000.0)

        state = LiveTyreState(
            driver=obs.driver,
            session_lap=obs.session_lap,
            compound=obs.compound,
            tyre_age=obs.tyre_age,
            performance_loss=float(self.a[self.i_level[obs.driver]]),
            performance_loss_sd=float(np.sqrt(max(self.P[self.i_level[obs.driver]][self.i_level[obs.driver]], 0.0))),
            degradation_rate=float(self.a[self.i_rate[obs.driver]]),
            degradation_rate_sd=float(np.sqrt(max(self.P[self.i_rate[obs.driver]][self.i_rate[obs.driver]], 0.0))),
            laps_observed=self._laps_seen[obs.driver],
            innovation=v,
            innovation_z=v / np.sqrt(F),
        )
        self._last_state[obs.driver] = state
        return state

    def state(self, driver: str) -> LiveTyreState | None:
        """Latest filtered state for one car, or None if it has not run."""
        return self._last_state.get(driver)

    def all_states(self) -> list[LiveTyreState]:
        """Latest filtered state for every car that has completed a lap."""
        return list(self._last_state.values())

    def compound_rates(self) -> dict[str, tuple[float, float]]:
        """Current pooled degradation baseline per compound, s/lap, with sd."""
        return {
            c: (float(self.a[i]), float(np.sqrt(max(self.P[i, i], 0.0))))
            for c, i in self.i_compound_rate.items()
        }

    def performance_summary(self) -> dict:
        """Timing statistics, so the real-time claim is measured rather than asserted."""
        times = np.array(self.update_times_ms) if self.update_times_ms else np.array([0.0])
        return {
            "total_laps_processed": self.total_laps,
            "n_states": self.n_state,
            "mean_update_ms": float(times.mean()),
            "p95_update_ms": float(np.percentile(times, 95)),
            "max_update_ms": float(times.max()),
        }


def replay(
    lap_table: pd.DataFrame,
    *,
    speed: float = 0.0,
    monitor: LiveTyreMonitor | None = None,
) -> Iterator[tuple[LapObservation, LiveTyreState]]:
    """Stream a stored session through the live estimator, lap by lap.

    Replays a cached session in session order so that the live path can be
    demonstrated, tested and driven from a UI without a network connection or a
    running race weekend. The estimator sees exactly what it would see live: one
    lap at a time, in order, with no access to the future.

    Args:
        lap_table: A session in the standard schema.
        speed: Seconds to pause between laps. Zero replays as fast as possible;
            a small value paces a live-looking demo.
        monitor: An existing monitor to feed. One is built from the session's
            entry list if omitted.

    Yields:
        Each ``(observation, updated_state)`` pair as it is processed.
    """
    df = lap_table.sort_values(["session_lap", "driver"]).reset_index(drop=True)

    if monitor is None:
        runs_per_driver = df.groupby("driver")["run_id"].nunique().max()
        monitor = LiveTyreMonitor(
            drivers=sorted(df["driver"].unique().tolist()),
            compounds=sorted(df["compound"].unique().tolist()),
            max_runs_per_driver=int(runs_per_driver) + 2,
        )

    # Map each driver's run ids onto contiguous per-driver indices, which is what
    # a live feed would provide (a car knows it is on its third set, not that it
    # is on the session's forty-first).
    run_index: dict[tuple[str, int], int] = {}
    for driver, group in df.groupby("driver"):
        for i, run in enumerate(sorted(group["run_id"].unique().tolist())):
            run_index[(driver, int(run))] = i

    for row in df.itertuples(index=False):
        obs = LapObservation(
            driver=str(row.driver),
            session_lap=int(row.session_lap),
            lap_time=float(row.lap_time),
            compound=str(row.compound),
            tyre_age=float(row.tyre_age),
            lap_in_run=int(row.lap_in_run),
            run_index=run_index[(str(row.driver), int(row.run_id))],
            traffic_index=float(getattr(row, "traffic_index", 0.0)),
        )
        state = monitor.observe(obs)
        if speed:
            time.sleep(speed)
        yield obs, state
