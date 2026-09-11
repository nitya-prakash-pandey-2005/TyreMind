"""Adaptive conformal has to hold coverage while the ground moves under it.

Split conformal is tested against distributions it was never told about. ACI
claims something stronger and is tested accordingly: coverage in the long run
under *drift*, which is the case where a fixed quantile is provably wrong. The
scenarios below are deliberately nastier than a race -- a mid-stream variance
jump, a steady ramp, a heavy-tailed regime -- because the guarantee is supposed
to survive arbitrary shift, not merely gentle shift.
"""

from __future__ import annotations

import numpy as np
import pytest

from tyremind.models.conformal import AdaptiveConformal


def run(stream_sd, n=4000, alpha=0.05, gamma=0.02, score="absolute", seed=0):
    """Push a drifting stream through ACI and report achieved coverage.

    `stream_sd` maps a step index to the true noise scale at that step, while the
    model always claims sd=1.0. So the model is not merely uncertain, it is
    *wrong about its own uncertainty*, in a way that changes over time.
    """
    rng = np.random.default_rng(seed)
    aci = AdaptiveConformal(alpha=alpha, gamma=gamma, score=score)
    for t in range(n):
        actual = rng.normal(0.0, stream_sd(t))
        aci.update(predicted=0.0, posterior_sd=1.0, actual=actual)
    return aci


class TestCoverageUnderDrift:
    def test_holds_coverage_on_a_stationary_stream(self):
        aci = run(lambda t: 1.0)
        assert aci.empirical_coverage == pytest.approx(0.95, abs=0.03)

    def test_holds_coverage_across_a_sudden_variance_jump(self):
        """The case a fixed quantile cannot survive: calibrate on the quiet half,
        then the world gets ten times noisier."""
        aci = run(lambda t: 1.0 if t < 2000 else 10.0)
        assert aci.empirical_coverage == pytest.approx(0.95, abs=0.04)

    def test_holds_coverage_on_a_steady_ramp(self):
        aci = run(lambda t: 1.0 + 9.0 * t / 4000)
        assert aci.empirical_coverage == pytest.approx(0.95, abs=0.04)

    def test_holds_coverage_when_the_stream_gets_quieter(self):
        """Drift in the other direction. Over-wide intervals should tighten, not
        be left alone -- an interval that always covers is uninformative."""
        aci = run(lambda t: 10.0 if t < 2000 else 1.0)
        assert aci.empirical_coverage == pytest.approx(0.95, abs=0.04)

    def test_holds_coverage_under_heavy_tails(self):
        rng = np.random.default_rng(7)
        aci = AdaptiveConformal(alpha=0.05, gamma=0.02)
        for _ in range(4000):
            aci.update(0.0, 1.0, float(rng.standard_t(df=2)))
        assert aci.empirical_coverage == pytest.approx(0.95, abs=0.04)

    def test_a_looser_target_is_actually_looser(self):
        tight = run(lambda t: 1.0, alpha=0.05)
        loose = run(lambda t: 1.0, alpha=0.20)
        assert loose.empirical_coverage < tight.empirical_coverage


class TestAdaptation:
    def test_the_working_alpha_falls_when_the_interval_keeps_missing(self):
        """Missing lowers the working alpha, which widens the next interval."""
        aci = AdaptiveConformal(alpha=0.05, gamma=0.02, warmup=5)
        for _ in range(10):
            aci.update(predicted=0.0, posterior_sd=1.0, actual=1000.0)
        assert aci.working_alpha < 0.05

    def test_the_working_alpha_rises_when_nothing_ever_misses(self):
        aci = AdaptiveConformal(alpha=0.05, gamma=0.02, warmup=5)
        for _ in range(10):
            aci.update(predicted=0.0, posterior_sd=1.0, actual=0.0)
        assert aci.working_alpha > 0.05

    def test_a_persistent_miss_streak_widens_the_interval(self):
        aci = AdaptiveConformal(alpha=0.05, gamma=0.05, warmup=5)
        for value in np.linspace(-1.0, 1.0, 40):
            aci.update(0.0, 1.0, float(value))
        before = aci.half_width(1.0)
        for _ in range(40):
            aci.update(0.0, 1.0, 500.0)
        assert aci.half_width(1.0) > before

    def test_an_unrepresentable_quantile_widens_rather_than_reverting(self):
        """The bug this guards against. Driving the working alpha below what the
        sample can represent must not silently return the Gaussian half-width --
        that is the narrowest interval available, and the alpha fell precisely
        because the interval was too narrow."""
        aci = AdaptiveConformal(alpha=0.05, gamma=0.2, warmup=5)
        for value in np.linspace(-3.0, 3.0, 30):
            aci.update(0.0, 1.0, float(value))
        for _ in range(30):
            aci.update(0.0, 1.0, 100.0)
        assert aci.working_alpha <= 1e-3
        assert aci.half_width(1.0) > 1.96

    def test_a_larger_gamma_reacts_faster(self):
        slow = AdaptiveConformal(alpha=0.05, gamma=0.005, warmup=5)
        fast = AdaptiveConformal(alpha=0.05, gamma=0.05, warmup=5)
        for aci in (slow, fast):
            for _ in range(10):
                aci.update(0.0, 1.0, 1000.0)
        assert fast.working_alpha < slow.working_alpha


class TestMechanics:
    def test_the_interval_is_centred_on_the_prediction(self):
        aci = AdaptiveConformal()
        low, high = aci.interval(predicted=92.4, posterior_sd=0.5)
        assert (low + high) / 2 == pytest.approx(92.4)

    def test_before_warmup_it_reports_the_gaussian_interval(self):
        aci = AdaptiveConformal(warmup=19)
        assert aci.half_width(2.0) == pytest.approx(1.959964 * 2.0)

    def test_scoring_never_uses_the_outcome_it_is_scoring(self):
        """The interval at step t must not depend on the truth at step t."""
        aci = AdaptiveConformal(alpha=0.05, warmup=5)
        for value in np.linspace(-1.0, 1.0, 30):
            aci.update(0.0, 1.0, float(value))
        expected = aci.half_width(1.0)
        covered = aci.update(0.0, 1.0, expected * 0.99)
        assert covered
        aci2 = AdaptiveConformal(alpha=0.05, warmup=5)
        for value in np.linspace(-1.0, 1.0, 30):
            aci2.update(0.0, 1.0, float(value))
        assert aci2.update(0.0, 1.0, expected * 1.01) is False

    def test_absolute_width_ignores_the_posterior_but_studentised_does_not(self):
        rng = np.random.default_rng(5)
        absolute = AdaptiveConformal(score="absolute", warmup=5)
        studentised = AdaptiveConformal(score="studentised", warmup=5)
        for aci in (absolute, studentised):
            for _ in range(40):
                aci.update(0.0, 1.0, float(rng.normal()))
        assert absolute.half_width(0.5) == pytest.approx(absolute.half_width(4.0))
        assert studentised.half_width(4.0) > studentised.half_width(0.5)

    def test_empirical_coverage_is_undefined_before_anything_is_seen(self):
        assert np.isnan(AdaptiveConformal().empirical_coverage)

    def test_counts_what_it_has_seen_and_missed(self):
        aci = AdaptiveConformal(warmup=1000)
        for _ in range(7):
            aci.update(0.0, 1.0, 0.0)
        for _ in range(3):
            aci.update(0.0, 1.0, 500.0)
        assert aci.n_seen == 10
        assert aci.n_missed == 3
        assert aci.empirical_coverage == pytest.approx(0.7)


class TestConstruction:
    @pytest.mark.parametrize("alpha", [0.0, 1.0, -0.1, 1.5])
    def test_rejects_an_impossible_alpha(self, alpha):
        with pytest.raises(ValueError, match="alpha"):
            AdaptiveConformal(alpha=alpha)

    def test_rejects_a_non_positive_step_size(self):
        with pytest.raises(ValueError, match="gamma"):
            AdaptiveConformal(gamma=0.0)

    def test_rejects_an_unknown_score(self):
        with pytest.raises(ValueError, match="unknown score"):
            AdaptiveConformal(score="vibes")
