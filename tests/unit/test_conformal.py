"""A calibrated interval has to earn its label, including when the model is bad.

The point of conformal prediction is a guarantee that survives a misspecified
model, so the tests do not check that our intervals are narrow -- they check that
coverage holds when the error distribution is skewed, heavy-tailed, or when the
point predictor is deliberately terrible. An interval that is only correct for
well-behaved Gaussian residuals would be the posterior sd we already had.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from tyremind.models.conformal import (
    CalibrationUnavailableError,
    ConformalCalibrator,
    conformal_quantile,
)


class TestQuantile:
    def test_uses_the_finite_sample_corrected_order_statistic(self):
        # n=19, alpha=0.05 -> ceil(20 * 0.95) = 19 -> the largest score.
        scores = np.arange(1.0, 20.0)
        assert conformal_quantile(scores, 0.05) == pytest.approx(19.0)

    def test_returns_infinity_when_the_sample_cannot_represent_the_level(self):
        # n=18, alpha=0.05 -> ceil(19 * 0.95) = 19 > 18.
        assert conformal_quantile(np.arange(1.0, 19.0), 0.05) == float("inf")

    def test_an_empty_calibration_set_is_infinitely_uncertain(self):
        assert conformal_quantile(np.array([]), 0.05) == float("inf")

    def test_non_finite_scores_are_dropped_not_propagated(self):
        clean = conformal_quantile(np.arange(1.0, 21.0), 0.10)
        dirty = conformal_quantile(
            np.concatenate([np.arange(1.0, 21.0), [np.nan, np.inf]]), 0.10
        )
        assert dirty == pytest.approx(clean)

    def test_a_looser_level_needs_a_smaller_threshold(self):
        scores = np.arange(1.0, 101.0)
        assert conformal_quantile(scores, 0.20) < conformal_quantile(scores, 0.05)


class TestCoverageGuarantee:
    """The property that matters, checked against distributions the model is wrong about."""

    def empirical_coverage(self, rng, draw, alpha=0.05, n_cal=200, n_test=2000):
        cal, test = draw(n_cal), draw(n_test)
        sd = np.ones(n_cal)
        calibrator = ConformalCalibrator.fit(
            predicted=np.zeros(n_cal), actual=cal, posterior_sd=sd,
            alpha=alpha, score="absolute",
        )
        # physical=False: these are synthetic residuals centred on zero, not
        # degradation rates, so the non-negativity constraint does not hold and
        # applying it would reject perfectly valid negative outcomes.
        return np.mean([calibrator.covers(0.0, 1.0, y, physical=False) for y in test])

    def test_covers_under_gaussian_errors(self):
        rng = np.random.default_rng(11)
        cov = self.empirical_coverage(rng, lambda n: rng.normal(0.0, 1.0, n))
        assert cov >= 0.93

    def test_covers_under_heavy_tails_where_a_gaussian_interval_would_not(self):
        rng = np.random.default_rng(12)
        cov = self.empirical_coverage(rng, lambda n: rng.standard_t(df=2, size=n))
        assert cov >= 0.93

    def test_covers_under_skew(self):
        rng = np.random.default_rng(13)
        cov = self.empirical_coverage(rng, lambda n: rng.exponential(1.0, n))
        assert cov >= 0.93

    def test_a_useless_predictor_gets_enormous_intervals_rather_than_false_ones(self):
        """The honest failure mode. Coverage is kept; the width tells the truth."""
        rng = np.random.default_rng(14)
        cov = self.empirical_coverage(rng, lambda n: rng.normal(0.0, 50.0, n))
        assert cov >= 0.93

        calibrator = ConformalCalibrator.fit(
            predicted=np.zeros(200), actual=rng.normal(0.0, 50.0, 200),
            posterior_sd=np.ones(200), score="absolute",
        )
        assert calibrator.half_width(1.0) > 50.0


class TestFit:
    def sample(self, n=60, seed=3):
        rng = np.random.default_rng(seed)
        predicted = rng.normal(0.12, 0.04, n)
        actual = predicted - 0.05 + rng.normal(0.0, 0.03, n)
        sd = rng.uniform(0.02, 0.15, n)
        return predicted, actual, sd

    def test_recovers_a_known_bias(self):
        predicted, actual, sd = self.sample()
        cal = ConformalCalibrator.fit(predicted, actual, sd, score="absolute")
        assert cal.bias == pytest.approx(0.05, abs=0.01)

    def test_the_bias_is_removed_from_the_interval_centre(self):
        cal = ConformalCalibrator(
            score="absolute", alpha=0.05, quantile=0.2, bias=0.05,
            n_calibration=60, n_events=27,
        )
        low, high = cal.interval(predicted=0.30, posterior_sd=0.04)
        assert (low + high) / 2 == pytest.approx(0.25)

    def test_studentised_width_scales_with_the_posterior_but_absolute_does_not(self):
        predicted, actual, sd = self.sample()
        stu = ConformalCalibrator.fit(predicted, actual, sd, score="studentised")
        abs_ = ConformalCalibrator.fit(predicted, actual, sd, score="absolute")
        assert stu.half_width(0.10) > stu.half_width(0.02)
        assert abs_.half_width(0.10) == abs_.half_width(0.02)

    def test_events_are_counted_separately_from_comparisons(self):
        """62 comparisons across 27 events is 27 pieces of evidence, not 62."""
        predicted, actual, sd = self.sample(n=60)
        events = np.array([f"event-{i // 3}" for i in range(60)])
        cal = ConformalCalibrator.fit(predicted, actual, sd, events=events)
        assert cal.n_calibration == 60
        assert cal.n_events == 20

    def test_refuses_a_calibration_set_too_small_for_the_level(self):
        predicted, actual, sd = self.sample(n=10)
        with pytest.raises(ValueError, match="cannot represent"):
            ConformalCalibrator.fit(predicted, actual, sd, alpha=0.05)

    def test_rejects_mismatched_input_lengths(self):
        with pytest.raises(ValueError, match="same length"):
            ConformalCalibrator.fit(np.zeros(5), np.zeros(4), np.ones(5))

    def test_rejects_an_unknown_score(self):
        predicted, actual, sd = self.sample()
        with pytest.raises(ValueError, match="unknown score"):
            ConformalCalibrator.fit(predicted, actual, sd, score="vibes")

    def test_drops_rows_with_a_zero_posterior_under_the_studentised_score(self):
        predicted, actual, sd = self.sample(n=60)
        sd[:5] = 0.0
        cal = ConformalCalibrator.fit(predicted, actual, sd, score="studentised")
        assert cal.n_calibration == 55
        assert np.isfinite(cal.quantile)


class TestPersistence:
    def calibrator(self):
        return ConformalCalibrator(
            score="studentised", alpha=0.05, quantile=3.2, bias=0.042,
            n_calibration=62, n_events=27, seasons=[2023, 2024],
            generated_at="2026-09-10T00:00:00+00:00",
        )

    def test_survives_a_round_trip_through_disk(self, tmp_path):
        path = self.calibrator().save(tmp_path / "cal.json")
        assert ConformalCalibrator.load(path) == self.calibrator()

    def test_ignores_unknown_fields_so_the_artefact_can_grow(self, tmp_path):
        path = tmp_path / "cal.json"
        payload = self.calibrator().to_dict() | {"future_field": "added later"}
        path.write_text(json.dumps(payload))
        assert ConformalCalibrator.load(path).quantile == pytest.approx(3.2)

    def test_a_missing_calibration_raises_rather_than_falling_back_silently(self, tmp_path):
        """Returning the Gaussian interval here would relabel a 76% interval as
        95%, which is the precise failure this module exists to prevent."""
        with pytest.raises(CalibrationUnavailableError, match="exp12"):
            ConformalCalibrator.load(tmp_path / "absent.json")


class TestPhysicalConstraint:
    """A tyre that gets faster with age does not exist, and the interval may say so."""

    def calibrator(self):
        return ConformalCalibrator(
            score="absolute", alpha=0.05, quantile=0.20, bias=0.0,
            n_calibration=62, n_events=27,
        )

    def test_clips_a_negative_lower_bound_at_zero(self):
        low, high = self.calibrator().interval(predicted=0.05, posterior_sd=0.02)
        assert low == pytest.approx(0.0)
        assert high == pytest.approx(0.25)

    def test_leaves_an_already_positive_interval_untouched(self):
        low, high = self.calibrator().interval(predicted=0.50, posterior_sd=0.02)
        assert low == pytest.approx(0.30)
        assert high == pytest.approx(0.70)

    def test_clipping_can_only_help_coverage_never_hurt_it(self):
        """Intersecting a valid interval with a region known to contain the truth
        removes non-covering mass only. Every actual degradation rate is positive,
        so no true value can be clipped away."""
        cal = self.calibrator()
        rng = np.random.default_rng(21)
        actuals = np.abs(rng.normal(0.12, 0.06, 500))
        predicted = actuals + rng.normal(0.0, 0.05, 500)

        clipped = np.mean([
            cal.covers(p, 0.02, a) for p, a in zip(predicted, actuals, strict=True)
        ])
        unclipped = np.mean([
            cal.covers(p, 0.02, a, physical=False)
            for p, a in zip(predicted, actuals, strict=True)
        ])
        assert clipped >= unclipped

    def test_the_constraint_can_be_switched_off_for_non_rate_quantities(self):
        low, _ = self.calibrator().interval(0.05, 0.02, physical=False)
        assert low < 0.0


class TestConstruction:
    def test_rejects_an_impossible_alpha(self):
        with pytest.raises(ValueError, match="alpha"):
            ConformalCalibrator(
                score="absolute", alpha=1.5, quantile=1.0, bias=0.0,
                n_calibration=20, n_events=10,
            )

    def test_rejects_an_infinite_quantile(self):
        """An infinite interval is technically valid and operationally useless."""
        with pytest.raises(ValueError, match="finite"):
            ConformalCalibrator(
                score="absolute", alpha=0.05, quantile=float("inf"), bias=0.0,
                n_calibration=5, n_events=5,
            )

    def test_confidence_is_the_complement_of_alpha(self):
        cal = ConformalCalibrator(
            score="absolute", alpha=0.05, quantile=1.0, bias=0.0,
            n_calibration=20, n_events=10,
        )
        assert cal.confidence == pytest.approx(0.95)
