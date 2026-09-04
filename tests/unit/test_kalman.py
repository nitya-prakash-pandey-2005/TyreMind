"""Correctness tests for the linear-Gaussian state-space kernel.

The central test here is `test_loglik_matches_brute_force_mvn`. A Kalman filter
computes the marginal likelihood recursively, and a recursion that is subtly
wrong still returns a plausible-looking number -- it does not crash, it just
quietly biases every hyperparameter the optimiser goes on to estimate. So we
check it against a completely independent route to the same quantity: build the
full joint covariance of the observations by hand and evaluate a multivariate
normal density. If the recursion and the direct construction agree to machine
precision, the recursion is right.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import multivariate_normal

from tyremind.models.ssm.kalman import (
    Observations,
    StateSpaceModel,
    filter_ssm,
    filter_ssm_batched,
    smooth_ssm,
    standardised_residuals,
)


def _local_linear_trend(n_steps: int, q_level: float, q_slope: float) -> StateSpaceModel:
    """A 2-state local linear trend: [level, slope], slope integrating into level.

    This is the exact structure TyreMind uses for a tyre: `level` is accumulated
    performance loss and `slope` is the instantaneous degradation rate.
    """
    T = np.tile(np.array([[1.0, 1.0], [0.0, 1.0]]), (n_steps - 1, 1, 1))
    Q = np.tile(np.diag([q_level, q_slope]), (n_steps - 1, 1, 1))
    return StateSpaceModel(
        T=T,
        Q=Q,
        a0=np.zeros(2),
        P0=np.diag([1e2, 1e2]),
    )


def _brute_force_loglik(model: StateSpaceModel, obs: Observations) -> float:
    """Marginal log-likelihood via the explicit joint MVN over observations.

    Independent of the filter recursion by construction: it propagates the state
    moments forward, assembles every cross-covariance Cov(x_s, x_t) explicitly,
    maps them through the observation rows, and calls scipy. O(m^2 n^2) and only
    usable on toy sizes, which is exactly what a reference implementation should be.
    """
    n_steps, n = model.n_steps, model.n_state

    # Marginal state moments.
    m = np.zeros((n_steps, n))
    V = np.zeros((n_steps, n, n))
    m[0], V[0] = model.a0, model.P0
    for t in range(n_steps - 1):
        m[t + 1] = model.T[t] @ m[t] + (model.c[t] if model.c is not None else 0.0)
        V[t + 1] = model.T[t] @ V[t] @ model.T[t].T + model.Q[t]

    # State transition products Phi(s, t) = T[t-1] @ ... @ T[s], with Phi(s, s) = I.
    Phi = np.zeros((n_steps, n_steps, n, n))
    for s in range(n_steps):
        Phi[s, s] = np.eye(n)
        for t in range(s + 1, n_steps):
            Phi[s, t] = model.T[t - 1] @ Phi[s, t - 1]

    mean = np.array([obs.Z[j] @ m[obs.t_index[j]] for j in range(obs.n_obs)])

    cov = np.zeros((obs.n_obs, obs.n_obs))
    for j in range(obs.n_obs):
        tj = int(obs.t_index[j])
        for k in range(obs.n_obs):
            tk = int(obs.t_index[k])
            # Cov(x_s, x_t) = V[s] Phi(s, t)' for s <= t. Written as an
            # if/else rather than a ternary because the ordering convention
            # is the substance of the line.
            if tj <= tk:  # noqa: SIM108
                cross = V[tj] @ Phi[tj, tk].T
            else:
                cross = (V[tk] @ Phi[tk, tj].T).T
            cov[j, k] = obs.Z[j] @ cross @ obs.Z[k]
        cov[j, j] += obs.H[j]

    return float(multivariate_normal(mean=mean, cov=cov, allow_singular=True).logpdf(obs.y))


class TestLogLikelihood:
    def test_loglik_matches_brute_force_mvn(self) -> None:
        """The recursive likelihood must equal the explicit joint MVN density."""
        rng = np.random.default_rng(20260904)
        n_steps = 9
        model = _local_linear_trend(n_steps, q_level=0.01, q_slope=0.002)

        # Two "cars" observing the same latent trend, at ragged time steps, so the
        # test exercises the long-form path rather than a tidy one-per-step panel.
        t_index = np.array([0, 0, 1, 2, 2, 3, 5, 5, 6, 7, 8, 8])
        Z = np.tile(np.array([1.0, 0.0]), (len(t_index), 1))
        y = rng.normal(size=len(t_index))
        H = np.full(len(t_index), 0.04)

        obs = Observations(y=y, t_index=t_index, Z=Z, H=H)

        assert filter_ssm(model, obs).loglik == pytest.approx(
            _brute_force_loglik(model, obs), rel=1e-9, abs=1e-9
        )

    def test_loglik_matches_brute_force_with_state_intercepts(self) -> None:
        """Intercepts shift the state mean; the likelihood must track that shift.

        Guards the tyre-set-reset path, which is expressed purely as a state
        intercept.
        """
        rng = np.random.default_rng(7)
        n_steps = 6
        base = _local_linear_trend(n_steps, q_level=0.02, q_slope=0.005)
        c = rng.normal(scale=0.3, size=(n_steps - 1, 2))
        model = StateSpaceModel(T=base.T, Q=base.Q, a0=base.a0, P0=base.P0, c=c)

        t_index = np.arange(n_steps)
        obs = Observations(
            y=rng.normal(size=n_steps),
            t_index=t_index,
            Z=np.tile(np.array([1.0, 0.0]), (n_steps, 1)),
            H=np.full(n_steps, 0.09),
        )

        assert filter_ssm(model, obs).loglik == pytest.approx(
            _brute_force_loglik(model, obs), rel=1e-9, abs=1e-9
        )

    def test_time_steps_with_no_observations_are_handled(self) -> None:
        """Sessions have gaps -- in-laps, out-laps, pit windows, red flags.

        A step with no observation should propagate the state and contribute
        nothing to the likelihood, not raise or silently drop the step.
        """
        model = _local_linear_trend(6, q_level=0.01, q_slope=0.001)
        # Nothing observed at steps 1, 3 or 4.
        obs = Observations(
            y=np.array([0.1, 0.4, 0.9]),
            t_index=np.array([0, 2, 5]),
            Z=np.tile(np.array([1.0, 0.0]), (3, 1)),
            H=np.full(3, 0.01),
        )

        result = filter_ssm(model, obs)

        assert np.isfinite(result.loglik)
        assert result.loglik == pytest.approx(_brute_force_loglik(model, obs), rel=1e-9)


class TestSmoother:
    def test_smoothed_equals_filtered_at_final_step(self) -> None:
        """At the last step there is no future to condition on, so they coincide."""
        rng = np.random.default_rng(11)
        n_steps = 12
        model = _local_linear_trend(n_steps, q_level=0.01, q_slope=0.001)
        obs = Observations(
            y=np.cumsum(rng.normal(scale=0.1, size=n_steps)),
            t_index=np.arange(n_steps),
            Z=np.tile(np.array([1.0, 0.0]), (n_steps, 1)),
            H=np.full(n_steps, 0.02),
        )

        filtered = filter_ssm(model, obs)
        smoothed = smooth_ssm(model, filtered)

        np.testing.assert_allclose(smoothed.a_smooth[-1], filtered.a_filt[-1], atol=1e-10)
        np.testing.assert_allclose(smoothed.P_smooth[-1], filtered.P_filt[-1], atol=1e-10)

    def test_smoothing_never_increases_variance(self) -> None:
        """Conditioning on more data cannot make a posterior less certain.

        This is the property that justifies reporting smoothed intervals as
        *tighter* than live ones in the UI, so it is worth asserting rather than
        assuming.
        """
        rng = np.random.default_rng(3)
        n_steps = 15
        model = _local_linear_trend(n_steps, q_level=0.02, q_slope=0.002)
        obs = Observations(
            y=np.cumsum(rng.normal(scale=0.1, size=n_steps)),
            t_index=np.arange(n_steps),
            Z=np.tile(np.array([1.0, 0.0]), (n_steps, 1)),
            H=np.full(n_steps, 0.05),
        )

        filtered = filter_ssm(model, obs)
        smoothed = smooth_ssm(model, filtered)

        filt_var = np.einsum("tii->ti", filtered.P_filt)
        smooth_var = np.einsum("tii->ti", smoothed.P_smooth)
        assert np.all(smooth_var <= filt_var + 1e-9)

    def test_covariances_stay_positive_semidefinite(self) -> None:
        """Sequential rank-1 downdates must not destroy positive-definiteness.

        Runs long enough to accumulate the drift that `_symmetrise` exists to
        prevent.
        """
        rng = np.random.default_rng(99)
        n_steps = 300
        model = _local_linear_trend(n_steps, q_level=1e-4, q_slope=1e-6)
        t_index = np.repeat(np.arange(n_steps), 4)  # four cars per step
        obs = Observations(
            y=rng.normal(size=t_index.size),
            t_index=t_index,
            Z=np.tile(np.array([1.0, 0.0]), (t_index.size, 1)),
            H=np.full(t_index.size, 0.01),
        )

        smoothed = smooth_ssm(model, filter_ssm(model, obs))

        eigenvalues = np.linalg.eigvalsh(smoothed.P_smooth)
        assert eigenvalues.min() > -1e-9


class TestRecovery:
    def test_recovers_a_known_linear_trend(self) -> None:
        """End-to-end: hide a known slope in noise, check the filter finds it.

        A direct rehearsal of the platform's core claim at the smallest possible
        scale -- with the degradation rate fixed at 0.08 s/lap, roughly what an
        F1 medium compound actually does.
        """
        rng = np.random.default_rng(2024)
        n_steps, true_slope, noise_sd = 40, 0.08, 0.15

        truth = true_slope * np.arange(n_steps)
        y = truth + rng.normal(scale=noise_sd, size=n_steps)

        # Near-deterministic slope: we assert the trend is close to linear.
        model = _local_linear_trend(n_steps, q_level=1e-6, q_slope=1e-8)
        obs = Observations(
            y=y,
            t_index=np.arange(n_steps),
            Z=np.tile(np.array([1.0, 0.0]), (n_steps, 1)),
            H=np.full(n_steps, noise_sd**2),
        )

        smoothed = smooth_ssm(model, filter_ssm(model, obs))
        estimated_slope = smoothed.a_smooth[:, 1].mean()

        assert estimated_slope == pytest.approx(true_slope, abs=0.01)

    def test_standardised_residuals_are_approximately_standard_normal(self) -> None:
        """Under a correctly specified model the innovations are white noise.

        This is the specification check the diagnostics screen surfaces, so it
        needs to actually hold when the model *is* correct.
        """
        rng = np.random.default_rng(555)
        n_steps, q_level, noise_var = 400, 0.01, 0.04

        level = np.cumsum(rng.normal(scale=np.sqrt(q_level), size=n_steps))
        y = level + rng.normal(scale=np.sqrt(noise_var), size=n_steps)

        model = StateSpaceModel(
            T=np.ones((n_steps - 1, 1, 1)),
            Q=np.full((n_steps - 1, 1, 1), q_level),
            a0=np.array([0.0]),
            P0=np.array([[1e4]]),
        )
        obs = Observations(
            y=y,
            t_index=np.arange(n_steps),
            Z=np.ones((n_steps, 1)),
            H=np.full(n_steps, noise_var),
        )

        resid = standardised_residuals(filter_ssm(model, obs))[1:]  # skip diffuse start

        assert abs(resid.mean()) < 0.15
        assert resid.std() == pytest.approx(1.0, abs=0.12)


class TestValidation:
    def test_rejects_mismatched_state_dimension(self) -> None:
        model = _local_linear_trend(4, 0.01, 0.001)
        obs = Observations(
            y=np.array([1.0]),
            t_index=np.array([0]),
            Z=np.ones((1, 3)),  # model has 2 states
            H=np.array([0.01]),
        )
        with pytest.raises(ValueError, match="state columns"):
            filter_ssm(model, obs)

    def test_rejects_observation_beyond_horizon(self) -> None:
        model = _local_linear_trend(4, 0.01, 0.001)
        obs = Observations(
            y=np.array([1.0]),
            t_index=np.array([9]),  # horizon is 4 steps
            Z=np.ones((1, 2)),
            H=np.array([0.01]),
        )
        with pytest.raises(ValueError, match="out of range"):
            filter_ssm(model, obs)

    def test_rejects_non_positive_observation_variance(self) -> None:
        with pytest.raises(ValueError, match="strictly positive"):
            Observations(
                y=np.array([1.0]),
                t_index=np.array([0]),
                Z=np.ones((1, 2)),
                H=np.array([0.0]),
            )


class TestBatchedFilter:
    """The batched filter is an optimisation, so it must be provably equivalent."""

    def test_batched_matches_sequential(self) -> None:
        """Same likelihood and same filtered moments, to machine precision.

        If this ever drifts, every fitted hyperparameter in the platform is
        suspect, because the optimiser runs against the batched path while the
        live estimator runs against the sequential one.
        """
        rng = np.random.default_rng(4242)
        n_steps = 25
        model = _local_linear_trend(n_steps, q_level=0.02, q_slope=0.001)

        # Ragged panel: a varying number of "cars" report on each step.
        t_index = np.concatenate(
            [np.full(int(rng.integers(0, 5)), t) for t in range(n_steps)]
        ).astype(int)
        obs = Observations(
            y=rng.normal(size=t_index.size),
            t_index=t_index,
            Z=np.tile(np.array([1.0, 0.0]), (t_index.size, 1)),
            H=rng.uniform(0.01, 0.1, size=t_index.size),
        )

        sequential = filter_ssm(model, obs)
        batched = filter_ssm_batched(model, obs)

        assert batched.loglik == pytest.approx(sequential.loglik, rel=1e-9, abs=1e-9)
        np.testing.assert_allclose(batched.a_filt, sequential.a_filt, atol=1e-9)
        np.testing.assert_allclose(batched.P_filt, sequential.P_filt, atol=1e-9)

    def test_batched_matches_brute_force_with_varying_design_rows(self) -> None:
        """Independent check, with each observation loading the states differently."""
        rng = np.random.default_rng(8)
        n_steps = 7
        model = _local_linear_trend(n_steps, q_level=0.03, q_slope=0.004)

        t_index = np.array([0, 0, 1, 1, 1, 2, 4, 4, 5, 6])
        Z = rng.normal(size=(t_index.size, 2))
        obs = Observations(
            y=rng.normal(size=t_index.size),
            t_index=t_index,
            Z=Z,
            H=rng.uniform(0.02, 0.2, size=t_index.size),
        )

        assert filter_ssm_batched(model, obs).loglik == pytest.approx(
            _brute_force_loglik(model, obs), rel=1e-9, abs=1e-9
        )
