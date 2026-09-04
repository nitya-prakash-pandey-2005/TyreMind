"""Tests for the online estimator.

Two properties matter here and neither is about the arithmetic being pretty.

First, the live estimator must agree with the batch one. They share a derivation
but not a code path -- batch builds transition matrices and folds a whole lap in
at once, live applies row operations in place and folds one car in at a time.
Nothing but a test keeps them from drifting apart, and if they drift the platform
is quietly showing the pit wall a different model from the one it validated.

Second, cost per lap must not grow with session length. A filter that degrades as
the race goes on is not a real-time system, and the claim is easy to make and
easy to get wrong.
"""

from __future__ import annotations

import numpy as np
import pytest

from tyremind.data.synthetic import SessionConfig, generate_session
from tyremind.models.ssm.tyre_ssm import TyreSSMHyper, fit_tyre_ssm
from tyremind.stream.live import LapObservation, LiveTyreMonitor, replay


@pytest.fixture(scope="module")
def session():
    return generate_session(SessionConfig(seed=77, n_drivers=8, session_slots=40))


class TestAgreementWithBatch:
    def test_live_converges_to_the_batch_estimate(self, session) -> None:
        """After the whole session, live and batch must agree on compound rates.

        Not to machine precision -- the batch model carries one extra track-noise
        step before its first observation -- but to far inside the posterior
        uncertainty. Disagreement beyond that means the two implementations have
        genuinely diverged.
        """
        fit = fit_tyre_ssm(session.lap_table)

        monitor = LiveTyreMonitor(
            drivers=sorted(session.lap_table["driver"].unique().tolist()),
            compounds=sorted(session.lap_table["compound"].unique().tolist()),
            hyper=fit.hyper,
            max_runs_per_driver=int(session.lap_table.groupby("driver")["run_id"].nunique().max())
            + 2,
        )
        for _ in replay(session.lap_table, monitor=monitor):
            pass

        live = monitor.compound_rates()
        batch = fit.compound_rates()

        for compound, (batch_mean, batch_sd) in batch.items():
            live_mean, _ = live[compound]
            assert abs(live_mean - batch_mean) < 0.5 * batch_sd, (
                f"{compound}: live {live_mean:.4f} vs batch {batch_mean:.4f}, "
                f"more than half a posterior sd apart"
            )

    def test_live_recovers_the_true_degradation_rate(self, session) -> None:
        """The online path must be useful, not merely self-consistent."""
        fit = fit_tyre_ssm(session.lap_table)
        monitor = LiveTyreMonitor(
            drivers=sorted(session.lap_table["driver"].unique().tolist()),
            compounds=sorted(session.lap_table["compound"].unique().tolist()),
            hyper=fit.hyper,
            max_runs_per_driver=8,
        )
        for _ in replay(session.lap_table, monitor=monitor):
            pass

        for compound, (estimate, _) in monitor.compound_rates().items():
            truth = session.truth.compound_rates[compound]
            assert abs(estimate - truth) < 0.04, (
                f"{compound}: live estimate {estimate:.4f} vs true {truth:.4f}"
            )


class TestRealTimeProperties:
    def test_update_cost_does_not_grow_with_session_length(self, session) -> None:
        """Lap 200 must cost what lap 10 cost.

        The whole justification for a recursive estimator over a sampler is O(1)
        updates. If this regressed -- an accidental full-history recompute, say --
        the system would still produce correct numbers and stop being real-time,
        which is exactly the kind of failure that survives every other test.
        """
        monitor = LiveTyreMonitor(
            drivers=sorted(session.lap_table["driver"].unique().tolist()),
            compounds=sorted(session.lap_table["compound"].unique().tolist()),
            max_runs_per_driver=8,
        )
        for _ in replay(session.lap_table, monitor=monitor):
            pass

        times = np.array(monitor.update_times_ms)
        assert times.size > 100

        first_quarter = times[: times.size // 4].mean()
        last_quarter = times[-times.size // 4 :].mean()

        assert last_quarter < 3.0 * first_quarter, (
            f"per-lap cost grew from {first_quarter:.3f} ms to {last_quarter:.3f} ms"
        )

    def test_updates_are_fast_enough_to_be_called_real_time(self, session) -> None:
        """A lap takes ~90 seconds. An update budget of 50 ms is generous."""
        monitor = LiveTyreMonitor(
            drivers=sorted(session.lap_table["driver"].unique().tolist()),
            compounds=sorted(session.lap_table["compound"].unique().tolist()),
            max_runs_per_driver=8,
        )
        for _ in replay(session.lap_table, monitor=monitor):
            pass

        assert monitor.performance_summary()["p95_update_ms"] < 50.0

    def test_uncertainty_shrinks_as_laps_accumulate(self, session) -> None:
        """More evidence must mean a tighter estimate, monotonically enough to matter."""
        monitor = LiveTyreMonitor(
            drivers=sorted(session.lap_table["driver"].unique().tolist()),
            compounds=sorted(session.lap_table["compound"].unique().tolist()),
            max_runs_per_driver=8,
        )

        early: dict[str, float] = {}
        late: dict[str, float] = {}
        for i, (_, state) in enumerate(replay(session.lap_table, monitor=monitor)):
            target = early if i < 30 else late
            target.setdefault(state.driver, state.degradation_rate_sd)
            if i >= 30:
                late[state.driver] = state.degradation_rate_sd

        shared = set(early) & set(late)
        assert shared
        assert np.mean([late[d] for d in shared]) < np.mean([early[d] for d in shared])


class TestGuardrails:
    def _monitor(self) -> LiveTyreMonitor:
        return LiveTyreMonitor(
            drivers=["A", "B"], compounds=["SOFT", "MEDIUM"], max_runs_per_driver=2
        )

    def _obs(self, **kwargs) -> LapObservation:
        base = dict(
            driver="A",
            session_lap=1,
            lap_time=90.0,
            compound="SOFT",
            tyre_age=1.0,
            lap_in_run=0,
            run_index=0,
        )
        base.update(kwargs)
        return LapObservation(**base)

    def test_rejects_an_unknown_driver(self) -> None:
        with pytest.raises(ValueError, match="entry list"):
            self._monitor().observe(self._obs(driver="Z"))

    def test_rejects_an_unknown_compound(self) -> None:
        with pytest.raises(ValueError, match="not declared"):
            self._monitor().observe(self._obs(compound="WET"))

    def test_rejects_running_out_of_run_slots(self) -> None:
        """Silently reusing a slot would pool two unrelated runs into one intercept."""
        with pytest.raises(ValueError, match="run slots were reserved"):
            self._monitor().observe(self._obs(run_index=5))

    def test_rejects_an_out_of_order_stream(self) -> None:
        monitor = self._monitor()
        monitor.observe(self._obs(session_lap=5))
        with pytest.raises(ValueError, match="went backwards"):
            monitor.observe(self._obs(session_lap=3))

    def test_health_index_is_anchored_and_bounded(self) -> None:
        """A convention, but a bounded one: a fresh tyre reads 100, a dead one 0."""
        monitor = self._monitor()
        state = monitor.observe(self._obs())

        state.performance_loss = 0.0
        assert state.health_index == pytest.approx(100.0)

        state.performance_loss = 1.5
        assert state.health_index == pytest.approx(0.0)

        state.performance_loss = 10.0
        assert state.health_index == 0.0  # clipped, never negative
