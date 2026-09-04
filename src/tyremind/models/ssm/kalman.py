"""Linear-Gaussian state-space inference: filtering, smoothing, exact likelihood.

This is the numerical kernel the whole platform rests on, so it is deliberately
general and has no knowledge of tyres. It implements the model

    x_0      ~ N(a0, P0)
    x_{t+1}   = T_t x_t + c_t + eta_t,     eta_t ~ N(0, Q_t)
    y_j       = Z_j x_{t(j)} + eps_j,      eps_j ~ N(0, H_j)

Observations are supplied in *long* form -- a flat array of scalar observations
each tagged with the time step it belongs to. Lap-time panels are ragged (a
different set of drivers sets a valid lap on each lap of a session), and long
form handles that without padding or masking.

Scalar observations are processed **sequentially** within a time step, following
Durbin & Koopman (2012) section 6.4. Two reasons, both of which matter here:

  * Cost drops from O(n^3) to O(n^2) per observation, because no matrix inverse
    is ever formed. With ~100 states and ~20 cars per lap this is the difference
    between a 30-second fit and a sub-second one, and the fit runs inside an
    optimiser loop.
  * It is what makes true online operation possible. A lap time can be folded in
    the instant it arrives, rather than waiting for every car to complete the lap.

The same code path serves both modes: `filter_ssm` alone is the real-time
estimator, and `smooth_ssm` on top of it is the retrospective one.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

_LOG_2PI = float(np.log(2.0 * np.pi))


@runtime_checkable
class Transition(Protocol):
    """State dynamics, as the filter needs to see them.

    Expressed as three operations rather than as matrices so that models with
    exploitable structure can override the propagation. The tyre model's
    transition is identity everywhere except two rows per car, and going through
    a dense matrix product there costs O(n^3) per step to compute something that
    is available in O(n^2) -- roughly a hundredfold difference at realistic
    field sizes, inside an optimiser loop.

    `transition_matrix` is only ever called by the smoother, which runs once,
    so a structured implementation may materialise a dense matrix there.
    """

    @property
    def n_state(self) -> int: ...

    @property
    def n_steps(self) -> int: ...

    def propagate_mean(self, a: np.ndarray, t: int) -> np.ndarray:
        """Return T[t] a + c[t], the state mean carried into step t + 1."""
        ...

    def propagate_cov(self, P: np.ndarray, t: int) -> np.ndarray:
        """Return T[t] P T[t]' + Q[t], the state covariance carried into step t + 1."""
        ...

    def transition_matrix(self, t: int) -> np.ndarray:
        """Return T[t] densely, shape (n, n). Used by the smoother only."""
        ...


@dataclass(frozen=True)
class Observations:
    """Scalar observations in long form.

    Attributes:
        y: Observed values, shape (m,).
        t_index: Time step each observation belongs to, shape (m,), integer dtype.
            Need not be sorted; filtering visits time steps in order regardless.
        Z: Design row per observation, shape (m, n_state).
        H: Observation noise variance per observation, shape (m,). Strictly positive.
    """

    y: np.ndarray
    t_index: np.ndarray
    Z: np.ndarray
    H: np.ndarray

    def __post_init__(self) -> None:
        m = self.y.shape[0]
        if self.t_index.shape != (m,):
            raise ValueError(f"t_index must have shape ({m},), got {self.t_index.shape}")
        if self.Z.ndim != 2 or self.Z.shape[0] != m:
            raise ValueError(f"Z must have shape ({m}, n_state), got {self.Z.shape}")
        if self.H.shape != (m,):
            raise ValueError(f"H must have shape ({m},), got {self.H.shape}")
        if m and not np.all(self.H > 0):
            raise ValueError("observation variances H must be strictly positive")

    @property
    def n_obs(self) -> int:
        return int(self.y.shape[0])

    @property
    def n_state(self) -> int:
        return int(self.Z.shape[1])


@dataclass(frozen=True)
class StateSpaceModel:
    """Time-varying linear-Gaussian state dynamics.

    Attributes:
        T: Transition matrices, shape (n_steps - 1, n, n). ``T[t]`` maps step t
            to step t + 1.
        Q: Process noise covariances, shape (n_steps - 1, n, n).
        a0: Prior state mean at step 0, shape (n,).
        P0: Prior state covariance at step 0, shape (n, n). A large diagonal
            entry expresses near-ignorance about a state. We use that rather
            than exact diffuse initialisation because the data are scaled to
            O(1) seconds, so a variance of 1e4 is already uninformative by four
            orders of magnitude -- and it avoids a second filter pass.
        c: Optional state intercepts, shape (n_steps - 1, n), for the
            deterministic part of a transition such as a tyre-set change.
    """

    T: np.ndarray
    Q: np.ndarray
    a0: np.ndarray
    P0: np.ndarray
    c: np.ndarray | None = None

    def __post_init__(self) -> None:
        n = self.a0.shape[0]
        if self.T.ndim != 3 or self.T.shape[1:] != (n, n):
            raise ValueError(f"T must have shape (n_steps-1, {n}, {n}), got {self.T.shape}")
        if self.Q.shape != self.T.shape:
            raise ValueError(f"Q must have the same shape as T, got {self.Q.shape}")
        if self.P0.shape != (n, n):
            raise ValueError(f"P0 must have shape ({n}, {n}), got {self.P0.shape}")
        if self.c is not None and self.c.shape != (self.T.shape[0], n):
            raise ValueError(f"c must have shape ({self.T.shape[0]}, {n}), got {self.c.shape}")

    @property
    def n_state(self) -> int:
        return int(self.a0.shape[0])

    @property
    def n_steps(self) -> int:
        return int(self.T.shape[0] + 1)

    def propagate_mean(self, a: np.ndarray, t: int) -> np.ndarray:
        out = self.T[t] @ a
        return out if self.c is None else out + self.c[t]

    def propagate_cov(self, P: np.ndarray, t: int) -> np.ndarray:
        Tm = self.T[t]
        return _symmetrise(Tm @ P @ Tm.T + self.Q[t])

    def transition_matrix(self, t: int) -> np.ndarray:
        return self.T[t]


@dataclass
class FilterResult:
    """Output of a forward pass.

    ``a_pred``/``P_pred`` hold the one-step-ahead moments *before* any
    observation at that step was seen; ``a_filt``/``P_filt`` hold them after.
    The smoother needs both. At step 0 the predicted moments are the prior.
    """

    a_pred: np.ndarray  # (n_steps, n)
    P_pred: np.ndarray  # (n_steps, n, n)
    a_filt: np.ndarray  # (n_steps, n)
    P_filt: np.ndarray  # (n_steps, n, n)
    loglik: float
    residuals: np.ndarray  # (m,) prediction errors v, in observation order
    residual_var: np.ndarray  # (m,) their variances F


@dataclass
class SmootherResult:
    """Output of the backward pass: marginal posteriors given *all* data."""

    a_smooth: np.ndarray  # (n_steps, n)
    P_smooth: np.ndarray  # (n_steps, n, n)

    def std(self) -> np.ndarray:
        """Marginal posterior standard deviation of each state, shape (n_steps, n)."""
        var = np.einsum("tii->ti", self.P_smooth)
        return np.sqrt(np.maximum(var, 0.0))


def _symmetrise(P: np.ndarray) -> np.ndarray:
    """Force exact symmetry.

    Sequential rank-1 downdates accumulate asymmetry in the last bits of the
    mantissa. Left alone over a few thousand updates that drift is enough to
    push an eigenvalue slightly negative, at which point the log-likelihood
    silently becomes NaN. Costs nothing to prevent.
    """
    return 0.5 * (P + P.T)


def filter_ssm(model: Transition, obs: Observations) -> FilterResult:
    """Run the Kalman filter, returning filtered moments and the exact log-likelihood.

    The log-likelihood is the marginal likelihood of the data with all states
    integrated out -- the prediction error decomposition. It is exact rather than
    a bound, which is what lets it be used directly both for hyperparameter
    estimation and for comparison across the model ladder.

    Args:
        model: State dynamics.
        obs: Observations in long form.

    Returns:
        FilterResult with per-step moments, the log-likelihood, and the raw
        prediction errors needed for diagnostics.

    Raises:
        ValueError: If observation and model state dimensions disagree, or an
            observation refers to a time step outside the model horizon.
        FloatingPointError: If the state covariance loses positive-definiteness.
    """
    n = model.n_state
    n_steps = model.n_steps

    if obs.n_state != n:
        raise ValueError(f"Z has {obs.n_state} state columns but model has {n} states")
    if obs.n_obs and (obs.t_index.min() < 0 or obs.t_index.max() >= n_steps):
        raise ValueError(
            f"t_index out of range [0, {n_steps - 1}]: "
            f"got [{obs.t_index.min()}, {obs.t_index.max()}]"
        )

    # Group observation indices by time step once, rather than rescanning per step.
    by_step: list[list[int]] = [[] for _ in range(n_steps)]
    for j, t in enumerate(obs.t_index):
        by_step[int(t)].append(j)

    a_pred = np.zeros((n_steps, n))
    P_pred = np.zeros((n_steps, n, n))
    a_filt = np.zeros((n_steps, n))
    P_filt = np.zeros((n_steps, n, n))
    residuals = np.full(obs.n_obs, np.nan)
    residual_var = np.full(obs.n_obs, np.nan)

    a = np.asarray(model.a0, dtype=float).copy()
    P = _symmetrise(np.asarray(model.P0, dtype=float).copy())
    loglik = 0.0

    for t in range(n_steps):
        if t > 0:
            a = model.propagate_mean(a, t - 1)
            P = _symmetrise(model.propagate_cov(P, t - 1))

        a_pred[t] = a
        P_pred[t] = P

        for j in by_step[t]:
            z = obs.Z[j]
            Pz = P @ z                      # (n,)
            F = float(z @ Pz + obs.H[j])    # scalar innovation variance
            if F <= 0.0 or not np.isfinite(F):
                raise FloatingPointError(
                    f"non-positive innovation variance F={F} at observation {j}; "
                    "the state covariance has lost positive-definiteness"
                )
            v = float(obs.y[j] - z @ a)

            residuals[j] = v
            residual_var[j] = F

            K = Pz / F                      # (n,) Kalman gain
            a = a + K * v
            P = _symmetrise(P - np.outer(K, Pz))

            loglik += -0.5 * (_LOG_2PI + np.log(F) + v * v / F)

        a_filt[t] = a
        P_filt[t] = P

    return FilterResult(
        a_pred=a_pred,
        P_pred=P_pred,
        a_filt=a_filt,
        P_filt=P_filt,
        loglik=float(loglik),
        residuals=residuals,
        residual_var=residual_var,
    )


def smooth_ssm(model: Transition, filtered: FilterResult) -> SmootherResult:
    """Run the Rauch-Tung-Striebel backward pass.

    Turns filtered estimates (each conditioned only on the past) into smoothed
    ones (conditioned on the whole session). For TyreMind that is the difference
    between what the pit wall could have known at lap 12 and what the engineers
    know on Monday morning. Both are useful, and the platform reports them
    separately rather than passing one off as the other.

    Args:
        model: The same model passed to `filter_ssm`.
        filtered: Result of `filter_ssm`.

    Returns:
        SmootherResult with the marginal posterior mean and covariance per step.
    """
    n_steps, n = filtered.a_filt.shape
    a_smooth = np.zeros((n_steps, n))
    P_smooth = np.zeros((n_steps, n, n))

    a_smooth[-1] = filtered.a_filt[-1]
    P_smooth[-1] = filtered.P_filt[-1]

    for t in range(n_steps - 2, -1, -1):
        Tm = model.transition_matrix(t)
        P_next_pred = filtered.P_pred[t + 1]

        # J = P_filt[t] T' inv(P_pred[t+1]), solved rather than inverted.
        # P_pred is near-singular whenever a state is deterministic (a run
        # intercept has exactly zero process noise), so fall back to a
        # least-squares solve, which degrades gracefully instead of raising.
        rhs = (filtered.P_filt[t] @ Tm.T).T
        try:
            J = np.linalg.solve(P_next_pred, rhs).T
        except np.linalg.LinAlgError:
            J = (np.linalg.lstsq(P_next_pred, rhs, rcond=None)[0]).T

        a_smooth[t] = filtered.a_filt[t] + J @ (a_smooth[t + 1] - filtered.a_pred[t + 1])
        P_smooth[t] = _symmetrise(
            filtered.P_filt[t] + J @ (P_smooth[t + 1] - P_next_pred) @ J.T
        )

    return SmootherResult(a_smooth=a_smooth, P_smooth=P_smooth)


def filter_ssm_batched(model: Transition, obs: Observations) -> FilterResult:
    """Kalman filter that folds in every observation at a time step at once.

    Mathematically identical to `filter_ssm` -- `test_batched_matches_sequential`
    pins them together to 1e-9 -- but it trades many small updates for one joint
    one per step.

    The flop counts are almost the same; what differs is interpreter overhead. A
    full grid produces around twenty scalar updates per lap, each issuing a
    handful of NumPy calls, and at a few hundred likelihood evaluations per fit
    that dispatch cost dominates the arithmetic entirely. Batching cuts it by
    roughly the field size.

    Use this for fitting. Use `filter_ssm` for live operation, where observations
    genuinely arrive one at a time and there is nothing to batch.

    Args:
        model: State dynamics.
        obs: Observations in long form.

    Returns:
        FilterResult, with `residuals`/`residual_var` holding the *marginal*
        prediction error and variance for each observation (the diagonal of the
        joint innovation covariance), which is what the diagnostics want.

    Raises:
        ValueError: If dimensions disagree or a time index is out of range.
        FloatingPointError: If an innovation covariance is not positive-definite.
    """
    n, n_steps = model.n_state, model.n_steps

    if obs.n_state != n:
        raise ValueError(f"Z has {obs.n_state} state columns but model has {n} states")
    if obs.n_obs and (obs.t_index.min() < 0 or obs.t_index.max() >= n_steps):
        raise ValueError(
            f"t_index out of range [0, {n_steps - 1}]: "
            f"got [{obs.t_index.min()}, {obs.t_index.max()}]"
        )

    by_step: list[list[int]] = [[] for _ in range(n_steps)]
    for j, t in enumerate(obs.t_index):
        by_step[int(t)].append(j)

    a_pred = np.zeros((n_steps, n))
    P_pred = np.zeros((n_steps, n, n))
    a_filt = np.zeros((n_steps, n))
    P_filt = np.zeros((n_steps, n, n))
    residuals = np.full(obs.n_obs, np.nan)
    residual_var = np.full(obs.n_obs, np.nan)

    a = np.asarray(model.a0, dtype=float).copy()
    P = _symmetrise(np.asarray(model.P0, dtype=float).copy())
    loglik = 0.0

    for t in range(n_steps):
        if t > 0:
            a = model.propagate_mean(a, t - 1)
            P = _symmetrise(model.propagate_cov(P, t - 1))

        a_pred[t] = a
        P_pred[t] = P

        rows = by_step[t]
        if rows:
            idx = np.asarray(rows)
            Zt = obs.Z[idx]                       # (k, n)
            v = obs.y[idx] - Zt @ a               # (k,)
            PZt = P @ Zt.T                        # (n, k)
            F = Zt @ PZt                          # (k, k)
            F[np.diag_indices_from(F)] += obs.H[idx]
            F = 0.5 * (F + F.T)

            try:
                L = np.linalg.cholesky(F)
            except np.linalg.LinAlgError as exc:
                raise FloatingPointError(
                    f"innovation covariance at step {t} is not positive-definite; "
                    "the state covariance has lost positive-definiteness"
                ) from exc

            residuals[idx] = v
            residual_var[idx] = np.diag(F)

            # Solve rather than invert: K = P Z' F^-1, and the quadratic form
            # v' F^-1 v comes free from the same triangular solve.
            alpha = np.linalg.solve(L, v)                     # (k,)
            K = np.linalg.solve(L.T, np.linalg.solve(L, PZt.T)).T  # (n, k)

            a = a + K @ v
            P = _symmetrise(P - K @ PZt.T)

            loglik += -0.5 * (
                len(rows) * _LOG_2PI + 2.0 * np.log(np.diag(L)).sum() + float(alpha @ alpha)
            )

        a_filt[t] = a
        P_filt[t] = P

    return FilterResult(
        a_pred=a_pred,
        P_pred=P_pred,
        a_filt=a_filt,
        P_filt=P_filt,
        loglik=float(loglik),
        residuals=residuals,
        residual_var=residual_var,
    )


def standardised_residuals(filtered: FilterResult) -> np.ndarray:
    """Prediction errors scaled to unit variance, shape (m,).

    Under a correctly specified model these are i.i.d. standard normal, which
    makes them the primary specification check: autocorrelation means the state
    dynamics are wrong, heavy tails mean the Gaussian observation model is wrong
    (which, for lap times, it is -- see `robust.py`).
    """
    return filtered.residuals / np.sqrt(filtered.residual_var)
