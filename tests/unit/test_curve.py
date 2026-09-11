"""A curve fitter that finds a cliff in a straight line is worse than none.

The classification is the part worth testing, not the least squares. Warm-up and
cliff are the same arithmetic -- a changepoint with a positive step in slope --
and opposite physical events, separated only by the sign of the slope before the
break. An earlier version of this analysis pooled them and reported a "cliff" a
third of the way through a stint, which was a tyre coming to temperature.

Each shape below is generated with a known answer and noise small enough that
the right verdict is not in doubt, then again with noise large enough that the
fitter should back off to "linear" rather than invent structure.
"""

from __future__ import annotations

import numpy as np
import pytest

from tyremind.models.curve import (
    MIN_STINT_LAPS,
    CurveShape,
    fit_curve_shape,
)


def stint(slope, n=24, delta=0.0, tau=None, noise=0.0, seed=0, start=0.0):
    """A synthetic stint with a known shape."""
    rng = np.random.default_rng(seed)
    age = np.arange(n, dtype=float) + start
    y = slope * (age - age[0])
    if delta and tau is not None:
        y = y + delta * np.maximum(age - tau, 0.0)
    return age, y + rng.normal(0.0, noise, n)


class TestLinear:
    def test_a_straight_line_is_called_linear(self):
        shape = fit_curve_shape(*stint(0.08, noise=0.01))
        assert shape.regime == "linear"
        assert shape.slope == pytest.approx(0.08, abs=0.01)

    def test_no_changepoint_is_reported_for_a_line(self):
        shape = fit_curve_shape(*stint(0.08, noise=0.01))
        assert shape.changepoint_age is None
        assert shape.delta is None
        assert shape.is_linear

    def test_pure_noise_does_not_manufacture_a_cliff(self):
        """The failure that matters. A changepoint model will always fit better
        in-sample; BIC has to stop it claiming so."""
        rng = np.random.default_rng(5)
        age = np.arange(24, dtype=float)
        for seed in range(8):
            noise = np.random.default_rng(seed).normal(0.0, 0.3, 24)
            shape = fit_curve_shape(age, noise)
            assert shape.regime in {"linear", "recovery", "cliff", "warm-up"}
        # Across many draws the overwhelming majority must come back linear.
        verdicts = [
            fit_curve_shape(age, rng.normal(0.0, 0.3, 24)).regime for _ in range(40)
        ]
        assert verdicts.count("linear") >= 30


class TestCliff:
    def test_a_late_step_up_in_rate_is_a_cliff(self):
        shape = fit_curve_shape(*stint(0.05, n=30, delta=0.25, tau=20, noise=0.02))
        assert shape.regime == "cliff"

    def test_the_changepoint_is_found_near_the_truth(self):
        shape = fit_curve_shape(*stint(0.05, n=30, delta=0.25, tau=20, noise=0.02))
        assert shape.changepoint_age == pytest.approx(20, abs=2)

    def test_the_step_and_the_rate_after_it_are_recovered(self):
        shape = fit_curve_shape(*stint(0.05, n=30, delta=0.25, tau=20, noise=0.02))
        assert shape.delta == pytest.approx(0.25, abs=0.05)
        assert shape.slope_after == pytest.approx(0.30, abs=0.05)

    def test_a_cliff_sits_late_in_the_stint(self):
        shape = fit_curve_shape(*stint(0.05, n=30, delta=0.25, tau=22, noise=0.02))
        assert shape.position > 0.5


class TestWarmUp:
    def test_a_tyre_getting_faster_then_turning_is_warm_up_not_cliff(self):
        """The regression this module exists for. Same positive step in slope as
        a cliff; the tyre is IMPROVING before the break, so it is not one."""
        shape = fit_curve_shape(*stint(-0.09, n=28, delta=0.17, tau=7, noise=0.02))
        assert shape.regime == "warm-up"
        assert shape.slope < 0

    def test_warm_up_sits_early_in_the_stint(self):
        shape = fit_curve_shape(*stint(-0.09, n=28, delta=0.17, tau=7, noise=0.02))
        assert shape.position < 0.5

    def test_the_settled_rate_after_warm_up_is_positive(self):
        shape = fit_curve_shape(*stint(-0.09, n=28, delta=0.17, tau=7, noise=0.02))
        assert shape.slope_after > 0


class TestRecovery:
    def test_a_rate_that_drops_is_recovery(self):
        shape = fit_curve_shape(*stint(0.25, n=28, delta=-0.20, tau=10, noise=0.02))
        assert shape.regime == "recovery"
        assert shape.delta < 0

    def test_recovery_is_not_reported_as_a_cliff(self):
        shape = fit_curve_shape(*stint(0.25, n=28, delta=-0.20, tau=10, noise=0.02))
        assert shape.regime != "cliff"


class TestGuards:
    def test_a_stint_too_short_to_split_returns_nothing(self):
        assert fit_curve_shape(*stint(0.08, n=MIN_STINT_LAPS - 1)) is None

    def test_a_stint_with_no_spread_in_age_returns_nothing(self):
        assert fit_curve_shape(np.full(20, 5.0), np.arange(20.0)) is None

    def test_non_finite_rows_are_dropped_rather_than_propagated(self):
        age, y = stint(0.08, n=26, noise=0.01)
        y[3] = np.nan
        age[7] = np.nan
        shape = fit_curve_shape(age, y)
        assert shape is not None
        assert shape.n_laps == 24

    def test_unsorted_input_gives_the_same_answer(self):
        age, y = stint(0.05, n=30, delta=0.25, tau=20, noise=0.02)
        order = np.random.default_rng(2).permutation(len(age))
        assert fit_curve_shape(age[order], y[order]).regime == fit_curve_shape(age, y).regime

    def test_a_scrubbed_set_starting_at_age_ten_is_handled(self):
        shape = fit_curve_shape(*stint(0.05, n=30, delta=0.25, tau=30, noise=0.02, start=10))
        assert shape.regime == "cliff"
        assert shape.changepoint_age > 10


class TestDescription:
    def test_every_regime_has_a_sentence_without_statistics(self):
        cases = [
            stint(0.08, noise=0.01),
            stint(0.05, n=30, delta=0.25, tau=20, noise=0.02),
            stint(-0.09, n=28, delta=0.17, tau=7, noise=0.02),
            stint(0.25, n=28, delta=-0.20, tau=10, noise=0.02),
        ]
        for age, y in cases:
            shape = fit_curve_shape(age, y)
            assert shape.description
            assert "bic" not in shape.description.lower()
            assert shape.description.endswith(".")

    def test_the_dict_carries_the_sentence(self):
        shape = fit_curve_shape(*stint(0.08, noise=0.01))
        assert shape.to_dict()["description"] == shape.description

    def test_is_linear_agrees_with_the_regime(self):
        assert CurveShape(
            regime="linear", n_laps=20, slope=0.08, changepoint_age=None,
            delta=None, slope_after=None, position=None,
            linear_bic=1.0, broken_bic=2.0, rmse=0.1,
        ).is_linear
