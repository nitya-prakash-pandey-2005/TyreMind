"""Tests for the Monte Carlo race simulator and decision engine.

The simulator's job is to turn a degradation posterior into a decision, so the
properties worth pinning are the ones a wrong answer would violate: a pit stop
must actually cost pit-lane time, a worse tyre must produce a worse race, and
the spread of outcomes must widen when the degradation estimate is less certain.

That last one is the reason the whole thing exists. Published race simulators
take a degradation rate as a fixed number; here it arrives with a posterior, and
a strategy that only wins under a confident estimate must visibly stop winning
when the estimate is honest.
"""

from __future__ import annotations

import numpy as np
import pytest

from tyremind.simulate.race import (
    RaceState,
    Strategy,
    TyreModel,
    recommend,
    simulate_strategy,
    strategy_regret,
)


@pytest.fixture
def tyres() -> dict[str, TyreModel]:
    return {
        "MEDIUM": TyreModel("MEDIUM", 0.0, 0.113, 0.023, cliff_lap=22),
        "HARD": TyreModel("HARD", 0.45, 0.074, 0.018, cliff_lap=32),
    }


@pytest.fixture
def state() -> RaceState:
    return RaceState(
        current_lap=30,
        total_laps=53,
        position=5,
        current_compound="MEDIUM",
        current_tyre_age=18,
        gap_ahead_s=3.0,
        gap_behind_s=2.4,
        base_lap_time_s=84.0,
    )


class TestSimulation:
    def test_pitting_actually_costs_pit_lane_time(self, state, tyres) -> None:
        """Guards an off-by-one that made 'box now' silently never pit.

        The loop starts at current_lap + 1, so a pit_lap equal to current_lap is
        never reached. The bug produced a 'box now' outcome identical to staying
        out, which looked plausible and was completely wrong.
        """
        fresh = TyreModel("HARD", 0.0, 0.0, 0.0, cliff_lap=999)
        no_deg = {"MEDIUM": TyreModel("MEDIUM", 0.0, 0.0, 0.0, cliff_lap=999), "HARD": fresh}

        stay = simulate_strategy(state, Strategy("stay", None), no_deg, n_sims=2000, seed=1)
        box = simulate_strategy(
            state, Strategy("box", state.current_lap + 1, "HARD"), no_deg, n_sims=2000, seed=1
        )

        # With zero degradation a stop is pure loss, and must cost about pit_loss.
        assert box.expected_time - stay.expected_time == pytest.approx(
            state.pit_loss_s, rel=0.25
        )

    def test_a_pit_lap_before_the_current_lap_never_fires(self, state, tyres) -> None:
        """Documents the boundary rather than leaving it to be rediscovered.

        Compared loosely rather than bit-for-bit: naming a new compound draws its
        sampled degradation rate from the generator even when the stop never
        happens, so the two random streams diverge slightly. The property that
        matters is that no pit-lane time was paid.
        """
        past = simulate_strategy(
            state, Strategy("past", state.current_lap - 5, "HARD"), tyres, n_sims=4000, seed=2
        )
        stay = simulate_strategy(state, Strategy("stay", None), tyres, n_sims=4000, seed=2)

        assert abs(past.expected_time - stay.expected_time) < 0.1 * state.pit_loss_s

    def test_worse_degradation_makes_a_slower_race(self, state, tyres) -> None:
        gentle = {**tyres, "MEDIUM": TyreModel("MEDIUM", 0.0, 0.05, 0.01, cliff_lap=40)}
        harsh = {**tyres, "MEDIUM": TyreModel("MEDIUM", 0.0, 0.25, 0.01, cliff_lap=40)}

        slow = simulate_strategy(state, Strategy("stay", None), harsh, n_sims=2000, seed=3)
        fast = simulate_strategy(state, Strategy("stay", None), gentle, n_sims=2000, seed=3)

        assert slow.expected_time > fast.expected_time

    def test_degradation_uncertainty_widens_the_outcome_spread(self, state) -> None:
        """The point of carrying a posterior instead of a point estimate.

        Two tyres with the same mean degradation but different confidence must
        produce different outcome spreads. If they did not, the uncertainty work
        upstream would be decorative.
        """
        confident = {"MEDIUM": TyreModel("MEDIUM", 0.0, 0.12, 0.005, cliff_lap=40)}
        uncertain = {"MEDIUM": TyreModel("MEDIUM", 0.0, 0.12, 0.060, cliff_lap=40)}

        tight = simulate_strategy(state, Strategy("stay", None), confident, n_sims=4000, seed=4)
        wide = simulate_strategy(state, Strategy("stay", None), uncertain, n_sims=4000, seed=4)

        assert wide.time_sd > tight.time_sd
        assert tight.expected_time == pytest.approx(wide.expected_time, rel=0.02)

    def test_traffic_costs_time(self, state, tyres) -> None:
        clear = RaceState(**{**state.__dict__, "gap_ahead_s": 5.0})
        stuck = RaceState(**{**state.__dict__, "gap_ahead_s": 0.5})

        free = simulate_strategy(clear, Strategy("stay", None), tyres, n_sims=2000, seed=5)
        held = simulate_strategy(stuck, Strategy("stay", None), tyres, n_sims=2000, seed=5)

        assert held.expected_time > free.expected_time

    def test_is_reproducible(self, state, tyres) -> None:
        """A recommendation shown to a user must not change on refresh."""
        a = simulate_strategy(state, Strategy("stay", None), tyres, n_sims=1000, seed=9)
        b = simulate_strategy(state, Strategy("stay", None), tyres, n_sims=1000, seed=9)
        np.testing.assert_allclose(a.race_times, b.race_times)

    def test_finished_race_returns_zero(self, tyres) -> None:
        done = RaceState(53, 53, 1, "MEDIUM", 20, 3.0, 3.0, 84.0)
        assert simulate_strategy(done, Strategy("stay", None), tyres, n_sims=100).expected_time == 0.0

    def test_ten_thousand_races_are_fast_enough_to_be_interactive(self, state, tyres) -> None:
        import time

        started = time.perf_counter()
        simulate_strategy(state, Strategy("stay", None), tyres, n_sims=10_000, seed=0)
        assert time.perf_counter() - started < 2.0


class TestRecommendation:
    def test_recommends_pitting_when_the_tyre_is_finished(self, state) -> None:
        dead = {
            "MEDIUM": TyreModel("MEDIUM", 0.0, 0.30, 0.02, cliff_lap=15),
            "HARD": TyreModel("HARD", 0.3, 0.06, 0.01, cliff_lap=40),
        }
        assert recommend(state, dead, n_sims=3000).best.strategy.pit_lap is not None

    def test_recommends_staying_out_near_the_end(self, tyres) -> None:
        """Two laps from the flag, a stop can never pay for itself."""
        late = RaceState(51, 53, 5, "MEDIUM", 20, 3.0, 2.0, 84.0)
        assert recommend(late, tyres, n_sims=3000).best.strategy.pit_lap is None

    def test_decision_confidence_is_a_probability(self, state, tyres) -> None:
        result = recommend(state, tyres, n_sims=3000)
        assert 0.0 <= result.decision_confidence <= 1.0

    def test_close_calls_are_flagged_as_close(self, state) -> None:
        """A pit wall needs to know when the model has no real preference.

        Constructed as two stops one lap apart on a barely-degrading tyre, which
        genuinely cannot be separated. Comparing stay-out against a stop would
        not be close -- the pit loss dominates -- so the candidates are passed
        explicitly rather than left to the default set.
        """
        flat = {
            "MEDIUM": TyreModel("MEDIUM", 0.0, 0.002, 0.0005, cliff_lap=999),
            "HARD": TyreModel("HARD", 0.0, 0.002, 0.0005, cliff_lap=999),
        }
        result = recommend(
            state,
            flat,
            strategies=[
                Strategy("Box in 2 laps", state.current_lap + 3, "HARD"),
                Strategy("Box in 3 laps", state.current_lap + 4, "HARD"),
            ],
            n_sims=4000,
        )

        assert result.margin_s < 0.5
        assert any("inside the noise" in r or "defensible" in r for r in result.reasons)

    def test_every_reason_is_non_empty_prose(self, state, tyres) -> None:
        result = recommend(state, tyres, n_sims=2000)
        assert result.reasons
        assert all(isinstance(r, str) and len(r) > 20 for r in result.reasons)

    def test_uncertain_degradation_is_called_out(self, state) -> None:
        vague = {
            "MEDIUM": TyreModel("MEDIUM", 0.0, 0.10, 0.08, cliff_lap=25),
            "HARD": TyreModel("HARD", 0.4, 0.07, 0.05, cliff_lap=35),
        }
        result = recommend(state, vague, n_sims=2000)
        assert any("uncertain" in r for r in result.reasons)


class TestRegret:
    def test_regret_is_never_negative(self, state, tyres) -> None:
        """Regret measures what a choice cost, so a better actual choice scores zero."""
        result = strategy_regret(state, tyres, recommended_lap=31, actual_lap=45, n_sims=2000)
        assert result["regret_s"] >= 0.0

    def test_stopping_far_from_the_recommendation_costs_time(self, state, tyres) -> None:
        result = strategy_regret(
            state, tyres, recommended_lap=31, actual_lap=48, new_compound="HARD", n_sims=3000
        )
        assert result["regret_s"] > 0.0

    def test_following_the_recommendation_costs_nothing(self, state, tyres) -> None:
        result = strategy_regret(
            state, tyres, recommended_lap=33, actual_lap=33, new_compound="HARD", n_sims=1000
        )
        assert result["regret_s"] == pytest.approx(0.0, abs=1e-9)
