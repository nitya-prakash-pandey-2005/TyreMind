"""The TyreMind latent tyre-state model.

Casts the question "how much of this lap time is the tyre?" as inference in a
linear-Gaussian state-space model, fitted jointly across the whole field.

The observation equation for driver d on session lap t, running tyre set r:

    y[d,t] = alpha[r]              run intercept: car pace, setup, starting fuel
           - A * g(t)              track evolution, SHARED across the field
           + theta[t]              residual track wobble (small)
           + s[d,t]                latent tyre performance loss
           - phi_kappa * L[d,t]    fuel burn-off, L = laps completed this run
           + gamma * TI[d,t]       traffic
           + eps                   observation noise

and the latent tyre evolves as a local linear trend in *tyre age*:

    s[d, a + da]  = s[d,a] + da * rate[d,a] + noise
    rate[d, a+da] = rate[d,a] + noise

`rate` is the deliverable: instantaneous degradation in seconds per lap. Because
it is free to drift, a tyre cliff needs no special machinery -- it simply appears
as `rate` accelerating, and the model can equally represent a plateau or a
recovery. Nothing here assumes a cliff exists.


Three collinearities, and what pins each one
--------------------------------------------
Isolating degradation from a practice session is hard for a specific, structural
reason: several causes push lap time in the same monotone direction, and many
wrong decompositions sum to the same right total. There are exactly three, and
being explicit about how each is resolved is the difference between a defensible
estimate and a confident-looking one.

**1. Fuel against degradation, within a run.** Both are linear in laps completed.
Fuel makes the car faster by about 0.081 s/lap; the tyre makes it slower by an
unknown amount. From a single run these are not separable at all.
*Pinned by:* an informative physical prior on `phi_kappa`, the product of
0.030 s/kg and 2.7 kg/lap (configs/physics.yaml). The prior's width is carried
as state uncertainty, so it widens every degradation interval we publish.

**2. Track evolution against a uniform shift in degradation.** Shift every
degradation rate by c and the track evolution slope by -c: for a lap at session
lap L with tyre age a = a0 + (L - L0), degradation moves by c*(a0 + L - L0) and
track by -c*L, leaving c*(a0 - L0) -- a constant *within the run*, which the run
intercept absorbs exactly. The two are structurally indistinguishable, and
scrubbed sets do not help.
*Pinned by:* modelling track evolution parametrically as a saturating curve
`A * (1 - exp(-k*L))` with an informative prior on the amplitude A, rather than
as a free random walk. Rubber deposition genuinely saturates, so this is the
more physically correct model as well as the identifying one. A small residual
random walk is retained for wobble (wind, track temperature) but is bounded
tightly enough that it cannot absorb a trend.

**3. Tyre age against session lap, within a run.** They advance together.
*Pinned by:* fitting the whole field at once. Cars change tyres on different
laps, so at any given session lap the field spans a wide range of tyre ages.
Run stagger is the natural experiment, and it is the one thing here that is
genuinely free -- it needs no prior, only the whole grid instead of one car.

Collinearities 1 and 2 are resolved by assumption, not by data, and no amount of
data from a single session will resolve them. We say so in the UI, and
`experiments/exp02_prior_sensitivity.py` reports how far the answer moves when
those priors move.


Coefficients as states
----------------------
`phi_kappa`, `gamma`, the track amplitude and the per-compound baseline rates are
all represented as states with zero process noise rather than as free parameters
in the optimiser. For a constant, "state with no process noise" *is* the Bayesian
posterior, and it buys two things:

  * Physical priors enter as P0, and the filter propagates their uncertainty into
    the tyre posterior automatically. No bootstrap needed.
  * The optimiser is left with six variance parameters instead of a dozen mixed
    ones, which makes the fit fast and well behaved.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from tyremind.models.ssm.kalman import (
    FilterResult,
    Observations,
    SmootherResult,
    _symmetrise,
    filter_ssm_batched,
    smooth_ssm,
)

REQUIRED_COLUMNS = (
    "driver",
    "session_lap",
    "run_id",
    "tyre_age",
    "lap_in_run",
    "lap_time",
    "compound",
)

#: Variance used for states we are essentially ignorant about. Observations are
#: scaled to seconds-relative-to-session-reference, so they sit around 1.0; 1e4
#: is uninformative by four orders of magnitude without the conditioning problems
#: a true diffuse prior would bring.
DIFFUSE_VARIANCE = 1.0e4


@dataclass(frozen=True)
class TyreSSMPriors:
    """Prior beliefs entering the model, in seconds unless stated.

    Attributes:
        fuel_slope_mean: Lap-time gain per lap from fuel burn-off, s/lap. The
            product of 0.030 s/kg and 2.7 kg/lap; configs/physics.yaml sources both.
        fuel_slope_sd: Uncertainty on that product. Resolves collinearity 1.
        track_amplitude_mean: Total lap time the track is expected to gain over
            the session as it rubbers in, s. Resolves collinearity 2.
        track_amplitude_sd: Uncertainty on it. This is the single most
            consequential prior in the model after the fuel slope; see
            exp02_prior_sensitivity for how much the answer depends on it.
        traffic_coef_sd: Prior scale for the traffic coefficient. Weak, because
            traffic is genuinely identified -- it varies within a driver
            independently of tyre age.
        run_intercept_sd: Prior scale for a run intercept, which absorbs car
            pace, setup and unknown starting fuel mass. Deliberately wide.
        compound_rate_mean: Prior mean degradation rate for a fresh set, s/lap.
        compound_rate_sd: Prior scale around it. Wide enough to let the data
            speak; the point of pooling is that it usually does.
        initial_level_sd: Prior scale on tyre performance loss at the moment a
            set is fitted. Small but non-zero -- a scrubbed set is not new.
    """

    fuel_slope_mean: float = 0.081
    fuel_slope_sd: float = 0.016

    track_amplitude_mean: float = 0.90
    track_amplitude_sd: float = 0.45

    traffic_coef_sd: float = 1.0
    run_intercept_sd: float = 5.0
    compound_rate_mean: float = 0.05
    compound_rate_sd: float = 0.20
    initial_level_sd: float = 0.05


@dataclass(frozen=True)
class TyreSSMHyper:
    """Variance hyperparameters, on the log scale so the optimiser is unconstrained.

    Attributes:
        log_q_track: Process noise on the *residual* track wobble per lap. Bounded
            small: its job is wind and track temperature, not trend. Letting it
            grow would reopen collinearity 2.
        log_q_level: Process noise on tyre performance loss per lap of tyre age.
        log_q_rate: Process noise on the degradation *rate*. Governs how sharply
            a cliff is allowed to develop.
        log_obs_sd: Observation noise standard deviation, s.
        log_stint_rate_sd: Spread of per-stint degradation rates around their
            compound baseline. This is the hierarchical shrinkage parameter.
        log_track_shape: Rubbering-in rate k in `1 - exp(-k*lap)`. Larger means
            the track matures earlier in the session.
    """

    log_q_track: float = np.log(1e-5)
    log_q_level: float = np.log(1e-4)
    log_q_rate: float = np.log(1e-5)
    log_obs_sd: float = np.log(0.25)
    log_stint_rate_sd: float = np.log(0.05)
    log_track_shape: float = np.log(0.055)

    def to_vector(self) -> np.ndarray:
        return np.array(
            [
                self.log_q_track,
                self.log_q_level,
                self.log_q_rate,
                self.log_obs_sd,
                self.log_stint_rate_sd,
                self.log_track_shape,
            ]
        )

    @classmethod
    def from_vector(cls, v: np.ndarray) -> TyreSSMHyper:
        return cls(*(float(x) for x in v))

    @property
    def q_track(self) -> float:
        return float(np.exp(self.log_q_track))

    @property
    def q_level(self) -> float:
        return float(np.exp(self.log_q_level))

    @property
    def q_rate(self) -> float:
        return float(np.exp(self.log_q_rate))

    @property
    def obs_var(self) -> float:
        return float(np.exp(2.0 * self.log_obs_sd))

    @property
    def stint_rate_var(self) -> float:
        return float(np.exp(2.0 * self.log_stint_rate_sd))

    @property
    def track_shape(self) -> float:
        return float(np.exp(self.log_track_shape))


#: Optimiser bounds. Without them the observation noise is driven to zero and
#: every lap is explained by a bespoke wobble in the tyre state -- a perfect fit
#: that means nothing. The tight ceiling on q_track is load-bearing: see
#: collinearity 2 in the module docstring.
HYPER_BOUNDS: tuple[tuple[float, float], ...] = (
    (np.log(1e-9), np.log(1e-3)),    # q_track  -- residual wobble only
    (np.log(1e-8), np.log(1e-1)),    # q_level
    (np.log(1e-10), np.log(1e-2)),   # q_rate
    (np.log(0.01), np.log(3.0)),     # obs_sd
    (np.log(1e-3), np.log(1.0)),     # stint_rate_sd
    (np.log(0.005), np.log(0.5)),    # track_shape
)


@dataclass(frozen=True)
class StateIndex:
    """Names and positions of every state in the vector.

    Explicit rather than implied by construction order, because every reported
    contribution in the decomposition is traced back through this mapping.
    """

    track_amplitude: int
    track_resid: int
    fuel_slope: int
    traffic_coef: int
    compound_rate: dict[str, int] = field(default_factory=dict)
    level: dict[str, int] = field(default_factory=dict)
    rate: dict[str, int] = field(default_factory=dict)
    run_intercept: dict[int, int] = field(default_factory=dict)
    size: int = 0

    def names(self) -> list[str]:
        """Human-readable name for each state position, for diagnostics."""
        out = ["?"] * self.size
        out[self.track_amplitude] = "track_amplitude"
        out[self.track_resid] = "track_resid"
        out[self.fuel_slope] = "fuel_slope"
        out[self.traffic_coef] = "traffic_coef"
        for c, i in self.compound_rate.items():
            out[i] = f"compound_rate[{c}]"
        for d, i in self.level.items():
            out[i] = f"tyre_level[{d}]"
        for d, i in self.rate.items():
            out[i] = f"tyre_rate[{d}]"
        for r, i in self.run_intercept.items():
            out[i] = f"run_intercept[{r}]"
        return out


# --------------------------------------------------------------------------- #
# Structured transition
# --------------------------------------------------------------------------- #

_ADVANCE = 0
_RESET = 1


@dataclass
class TyreTransition:
    """Sparse state dynamics for the tyre model.

    The transition matrix is the identity everywhere except two rows per car:
    the tyre level, which absorbs the rate, and the rate itself, which is
    overwritten when a set is fitted. Propagating a dense matrix would cost
    O(n^3) per step to compute something available in O(n^2). At a full grid
    (~150 states) that is a hundredfold difference, repeated for every one of
    the several hundred likelihood evaluations an optimiser run needs.

    Implements the `Transition` protocol from `kalman`.

    Attributes:
        ops: Per transition index, the row rewrites to apply. Each entry is
            ``(_ADVANCE, i_level, i_rate, da)`` or ``(_RESET, i_level, i_rate, i_base)``.
        q_diag: Diagonal process noise per transition, shape (n_steps - 1, n).
            Every noise source in this model is independent, so a diagonal
            suffices and the off-diagonal work can be skipped entirely.
    """

    _n_state: int
    _n_steps: int
    a0: np.ndarray
    P0: np.ndarray
    ops: list[list[tuple[int, int, int, float]]]
    q_diag: np.ndarray

    @property
    def n_state(self) -> int:
        return self._n_state

    @property
    def n_steps(self) -> int:
        return self._n_steps

    def propagate_mean(self, a: np.ndarray, t: int) -> np.ndarray:
        out = a.copy()
        for kind, i_level, i_rate, param in self.ops[t]:
            if kind == _ADVANCE:
                out[i_level] = a[i_level] + param * a[i_rate]
            else:
                out[i_level] = 0.0
                out[i_rate] = a[int(param)]
        return out

    def _apply_rows(self, X: np.ndarray, t: int) -> np.ndarray:
        """Return T[t] @ X, touching only the rows T actually rewrites."""
        out = X.copy()
        for kind, i_level, i_rate, param in self.ops[t]:
            if kind == _ADVANCE:
                out[i_level] = X[i_level] + param * X[i_rate]
            else:
                out[i_level] = 0.0
                out[i_rate] = X[int(param)]
        return out

    def propagate_cov(self, P: np.ndarray, t: int) -> np.ndarray:
        # T P T' computed as T (T P)', valid because P is symmetric.
        out = self._apply_rows(self._apply_rows(P, t).T, t)
        out[np.diag_indices_from(out)] += self.q_diag[t]
        return _symmetrise(out)

    def transition_matrix(self, t: int) -> np.ndarray:
        """Materialise T[t]. Called by the smoother only, once per step."""
        T = np.eye(self._n_state)
        for kind, i_level, i_rate, param in self.ops[t]:
            if kind == _ADVANCE:
                T[i_level, i_rate] = param
            else:
                T[i_level, :] = 0.0
                T[i_rate, :] = 0.0
                T[i_rate, int(param)] = 1.0
        return T


def _validate(lap_table: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_COLUMNS if c not in lap_table.columns]
    if missing:
        raise ValueError(f"lap_table is missing required columns: {missing}")
    if lap_table.empty:
        raise ValueError("lap_table is empty; nothing to estimate")
    if lap_table["lap_time"].isna().any():
        raise ValueError("lap_table contains NaN lap times; filter to valid laps first")
    monotone = lap_table.groupby("run_id")["tyre_age"].apply(lambda s: s.is_monotonic_increasing)
    if not monotone.all():
        bad = list(monotone[~monotone].index)
        raise ValueError(f"tyre_age must be non-decreasing within a run; violated by runs {bad}")


def build_state_index(lap_table: pd.DataFrame) -> StateIndex:
    """Lay out the state vector for a given session."""
    drivers = sorted(lap_table["driver"].unique().tolist())
    compounds = sorted(lap_table["compound"].unique().tolist())
    runs = sorted(lap_table["run_id"].unique().tolist())

    cursor = 0

    def take() -> int:
        nonlocal cursor
        i = cursor
        cursor += 1
        return i

    track_amplitude, track_resid = take(), take()
    fuel_slope, traffic_coef = take(), take()
    compound_rate = {c: take() for c in compounds}

    level: dict[str, int] = {}
    rate: dict[str, int] = {}
    for d in drivers:
        level[d] = take()
        rate[d] = take()

    run_intercept = {int(r): take() for r in runs}

    return StateIndex(
        track_amplitude=track_amplitude,
        track_resid=track_resid,
        fuel_slope=fuel_slope,
        traffic_coef=traffic_coef,
        compound_rate=compound_rate,
        level=level,
        rate=rate,
        run_intercept=run_intercept,
        size=cursor,
    )


@dataclass(frozen=True)
class SessionDesign:
    """Everything about a session that does not depend on the hyperparameters.

    Built once and reused across every optimiser iteration.

    Attributes:
        index: State layout.
        obs: Observation rows. The track-amplitude column is rewritten per
            evaluation because it depends on the shape hyperparameter.
        n_steps: Number of filter steps (session laps plus a synthetic step 0).
        reference_time: Session median lap time, subtracted for conditioning.
        lap_offset: Session lap of each observation, zeroed to the session start.
        ops: Per-transition row rewrites, ready for `TyreTransition`.
        noise_slots: Per transition, the (state index, hyperparameter key,
            multiplier) triples that populate the diagonal process noise.
        lap_table: The (sorted) input, retained for reporting.
        step_of_lap: Session lap to filter step.
    """

    index: StateIndex
    obs: Observations
    n_steps: int
    reference_time: float
    lap_offset: np.ndarray
    ops: list[list[tuple[int, int, int, float]]]
    noise_slots: list[list[tuple[int, str, float]]]
    lap_table: pd.DataFrame
    step_of_lap: dict[int, int]


def build_design(lap_table: pd.DataFrame) -> SessionDesign:
    """Assemble observation rows and tyre-age bookkeeping for a session.

    Time steps are session laps offset by one: step 0 is a synthetic
    initialisation step carrying no observations, so that every real lap --
    including the first -- has a transition on which a tyre-set fit can be
    expressed.

    Args:
        lap_table: One row per valid green lap. See REQUIRED_COLUMNS. An optional
            `traffic_index` column is used if present and treated as zero if not.

    Returns:
        A SessionDesign holding the observation matrix and per-step bookkeeping.

    Raises:
        ValueError: If `lap_table` is malformed.
    """
    _validate(lap_table)
    df = lap_table.sort_values(["session_lap", "driver"]).reset_index(drop=True)

    index = build_state_index(df)
    reference_time = float(df["lap_time"].median())

    lap_values = sorted(df["session_lap"].unique().tolist())
    step_of_lap = {lap: i + 1 for i, lap in enumerate(lap_values)}
    n_steps = len(lap_values) + 1
    session_start = float(min(lap_values))

    traffic = (
        df["traffic_index"].to_numpy(dtype=float)
        if "traffic_index" in df.columns
        else np.zeros(len(df))
    )

    m = len(df)
    Z = np.zeros((m, index.size))
    t_index = np.zeros(m, dtype=int)
    y = np.zeros(m)
    lap_offset = np.zeros(m)

    for j, row in enumerate(df.itertuples(index=False)):
        t_index[j] = step_of_lap[row.session_lap]
        y[j] = float(row.lap_time) - reference_time
        lap_offset[j] = float(row.session_lap) - session_start

        Z[j, index.track_resid] = 1.0
        Z[j, index.run_intercept[int(row.run_id)]] = 1.0
        Z[j, index.level[row.driver]] = 1.0
        Z[j, index.fuel_slope] = -float(row.lap_in_run)
        Z[j, index.traffic_coef] = float(traffic[j])
        # index.track_amplitude column is filled in per hyperparameter evaluation.

    # Tyre-age bookkeeping. For each driver, walk their laps in order; the
    # transition INTO a lap either fits a new set or advances the existing one by
    # the elapsed tyre age. Laps a driver sits out leave their tyre untouched,
    # which is correct -- a tyre in the garage does not degrade.
    ops: list[list[tuple[int, int, int, float]]] = [[] for _ in range(n_steps - 1)]
    noise_slots: list[list[tuple[int, str, float]]] = [[] for _ in range(n_steps - 1)]

    for driver, group in df.groupby("driver", sort=True):
        i_level, i_rate = index.level[driver], index.rate[driver]
        prev_run: int | None = None
        prev_age: float | None = None

        for row in group.sort_values("session_lap").itertuples(index=False):
            tr = step_of_lap[row.session_lap] - 1  # transition INTO this lap
            run = int(row.run_id)

            if run != prev_run:
                i_base = index.compound_rate[str(row.compound)]
                ops[tr].append((_RESET, i_level, i_rate, float(i_base)))
                noise_slots[tr].append((i_level, "initial_level_var", 1.0))
                noise_slots[tr].append((i_rate, "stint_rate_var", 1.0))
            else:
                da = max(float(row.tyre_age) - float(prev_age), 0.0)
                ops[tr].append((_ADVANCE, i_level, i_rate, da))
                noise_slots[tr].append((i_level, "q_level", da))
                noise_slots[tr].append((i_rate, "q_rate", da))

            prev_run, prev_age = run, float(row.tyre_age)

    obs = Observations(y=y, t_index=t_index, Z=Z, H=np.ones(m))
    return SessionDesign(
        index=index,
        obs=obs,
        n_steps=n_steps,
        reference_time=reference_time,
        lap_offset=lap_offset,
        ops=ops,
        noise_slots=noise_slots,
        lap_table=df,
        step_of_lap=step_of_lap,
    )


def track_basis(lap_offset: np.ndarray, shape: float) -> np.ndarray:
    """Saturating track-evolution basis, `-(1 - exp(-k * lap))`.

    Negative because track evolution makes the car *faster*, so a positive
    amplitude corresponds to a lap-time gain. Saturating because rubber
    deposition saturates -- and, usefully, because a saturating curve is not
    collinear with a uniform shift in degradation rate the way a straight line is.
    """
    return -(1.0 - np.exp(-shape * lap_offset))


def build_model(
    design: SessionDesign, hyper: TyreSSMHyper, priors: TyreSSMPriors
) -> TyreTransition:
    """Instantiate the transition and priors for one hyperparameter setting."""
    idx = design.index
    n, n_steps = idx.size, design.n_steps

    variances = {
        "q_level": hyper.q_level,
        "q_rate": hyper.q_rate,
        "stint_rate_var": hyper.stint_rate_var,
        "initial_level_var": priors.initial_level_sd**2,
    }

    q_diag = np.zeros((n_steps - 1, n))
    q_diag[:, idx.track_resid] = hyper.q_track
    for tr, slots in enumerate(design.noise_slots):
        for state_i, key, multiplier in slots:
            q_diag[tr, state_i] += variances[key] * multiplier

    a0 = np.zeros(n)
    P0 = np.zeros((n, n))

    # Track evolution: informative prior on how much the track gains overall.
    # This is what resolves collinearity 2. See the module docstring.
    a0[idx.track_amplitude] = priors.track_amplitude_mean
    P0[idx.track_amplitude, idx.track_amplitude] = priors.track_amplitude_sd**2
    P0[idx.track_resid, idx.track_resid] = 1e-4

    # The physical fuel prior. Resolves collinearity 1; its width is what keeps
    # the published degradation intervals honest.
    a0[idx.fuel_slope] = priors.fuel_slope_mean
    P0[idx.fuel_slope, idx.fuel_slope] = priors.fuel_slope_sd**2

    P0[idx.traffic_coef, idx.traffic_coef] = priors.traffic_coef_sd**2

    for i in idx.compound_rate.values():
        a0[i] = priors.compound_rate_mean
        P0[i, i] = priors.compound_rate_sd**2

    for i in idx.run_intercept.values():
        P0[i, i] = priors.run_intercept_sd**2

    # Tyre states at step 0 are placeholders; every run opens with a reset
    # transition that overwrites them.
    for driver in idx.level:
        P0[idx.level[driver], idx.level[driver]] = DIFFUSE_VARIANCE
        P0[idx.rate[driver], idx.rate[driver]] = DIFFUSE_VARIANCE

    return TyreTransition(
        _n_state=n,
        _n_steps=n_steps,
        a0=a0,
        P0=P0,
        ops=design.ops,
        q_diag=q_diag,
    )


def build_observations(
    design: SessionDesign, hyper: TyreSSMHyper
) -> Observations:
    """Observation rows for one hyperparameter setting.

    Only the track-amplitude column and the noise variance depend on the
    hyperparameters, so the rest of Z is reused as built.
    """
    Z = design.obs.Z.copy()
    Z[:, design.index.track_amplitude] = track_basis(design.lap_offset, hyper.track_shape)
    return replace(design.obs, Z=Z, H=np.full(design.obs.n_obs, hyper.obs_var))


@dataclass
class TyreSSMResult:
    """Fitted model, posterior states, and the quantities the product reports."""

    hyper: TyreSSMHyper
    priors: TyreSSMPriors
    index: StateIndex
    model: TyreTransition
    obs: Observations
    filtered: FilterResult
    smoothed: SmootherResult
    design: SessionDesign
    loglik: float
    n_obs: int
    converged: bool

    @property
    def n_params(self) -> int:
        """Hyperparameters estimated by the optimiser. Used for AIC/BIC."""
        return len(self.hyper.to_vector())

    def aic(self) -> float:
        return 2.0 * self.n_params - 2.0 * self.loglik

    def bic(self) -> float:
        return self.n_params * np.log(self.n_obs) - 2.0 * self.loglik

    def _state(self, i: int, *, smoothed: bool = True) -> tuple[np.ndarray, np.ndarray]:
        if smoothed:
            return self.smoothed.a_smooth[:, i], self.smoothed.std()[:, i]
        var = np.einsum("tii->ti", self.filtered.P_filt)[:, i]
        return self.filtered.a_filt[:, i], np.sqrt(np.maximum(var, 0.0))

    def _scalar(self, i: int) -> tuple[float, float]:
        mean, sd = self._state(i)
        return float(mean[-1]), float(sd[-1])

    def fuel_slope(self) -> tuple[float, float]:
        """Posterior mean and sd of the fuel burn-off slope, s/lap.

        Comparing this against the prior says whether the data moved it. In
        practice it barely does, which is collinearity 1 showing up empirically
        rather than merely being asserted.
        """
        return self._scalar(self.index.fuel_slope)

    def track_amplitude(self) -> tuple[float, float]:
        """Posterior mean and sd of total track evolution over the session, s."""
        return self._scalar(self.index.track_amplitude)

    def traffic_coefficient(self) -> tuple[float, float]:
        """Posterior mean and sd of the traffic penalty at traffic index 1.0, s."""
        return self._scalar(self.index.traffic_coef)

    def compound_rates(self) -> dict[str, tuple[float, float]]:
        """Pooled baseline degradation rate per compound, s/lap, with posterior sd."""
        return {c: self._scalar(i) for c, i in self.index.compound_rate.items()}

    def track_evolution(self) -> pd.DataFrame:
        """Estimated track evolution over the session, s relative to its start."""
        amp_mean, amp_sd = self.track_amplitude()
        laps = sorted(self.design.lap_table["session_lap"].unique().tolist())
        offset = np.array(laps, dtype=float) - float(min(laps))
        basis = track_basis(offset, self.hyper.track_shape)

        resid_mean, _ = self._state(self.index.track_resid)
        return pd.DataFrame(
            {
                "session_lap": laps,
                "track_effect": basis * amp_mean + resid_mean[1:],
                "track_effect_sd": np.abs(basis) * amp_sd,
            }
        )

    def degradation(self, *, smoothed: bool = True) -> pd.DataFrame:
        """Per-driver, per-lap latent tyre state and degradation rate.

        The model's primary output. `rate` is instantaneous degradation in s/lap;
        `level` is cumulative performance loss on the current set.

        Args:
            smoothed: If True use the full-session posterior. If False use the
                filtered estimate -- what was knowable in real time at that lap.

        Returns:
            One row per observed lap, with posterior means and standard deviations.
        """
        df = self.design.lap_table
        level_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        rate_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}

        for driver in self.index.level:
            level_cache[driver] = self._state(self.index.level[driver], smoothed=smoothed)
            rate_cache[driver] = self._state(self.index.rate[driver], smoothed=smoothed)

        rows = []
        for row in df.itertuples(index=False):
            step = self.design.step_of_lap[row.session_lap]
            lvl_mean, lvl_sd = level_cache[row.driver]
            rt_mean, rt_sd = rate_cache[row.driver]
            rows.append(
                {
                    "driver": row.driver,
                    "session_lap": row.session_lap,
                    "run_id": row.run_id,
                    "compound": row.compound,
                    "tyre_age": row.tyre_age,
                    "level": lvl_mean[step],
                    "level_sd": lvl_sd[step],
                    "rate": rt_mean[step],
                    "rate_sd": rt_sd[step],
                }
            )
        return pd.DataFrame(rows)


def _negative_loglik(theta: np.ndarray, design: SessionDesign, priors: TyreSSMPriors) -> float:
    """Objective for hyperparameter estimation: the exact marginal likelihood.

    Returns a large finite penalty rather than raising if a covariance goes
    non-positive-definite, so the optimiser backs away from a bad region instead
    of dying in it.
    """
    hyper = TyreSSMHyper.from_vector(theta)
    try:
        model = build_model(design, hyper, priors)
        return -filter_ssm_batched(model, build_observations(design, hyper)).loglik
    except (FloatingPointError, np.linalg.LinAlgError):
        return 1.0e12


def fit_tyre_ssm(
    lap_table: pd.DataFrame,
    priors: TyreSSMPriors | None = None,
    initial: TyreSSMHyper | None = None,
    *,
    maxiter: int = 200,
) -> TyreSSMResult:
    """Fit the latent tyre-state model to one session.

    Hyperparameters are estimated by maximising the exact marginal likelihood
    with L-BFGS-B on the log scale. The states -- including the fuel slope,
    track amplitude, traffic coefficient and compound baselines -- are then
    integrated out exactly by the filter, so their posteriors are analytic rather
    than sampled.

    Args:
        lap_table: One row per valid green lap. See REQUIRED_COLUMNS.
        priors: Prior specification. Defaults are sourced in configs/physics.yaml.
        initial: Starting hyperparameters.
        maxiter: Optimiser iteration cap.

    Returns:
        A TyreSSMResult carrying the fit, the posterior states, and the reporting
        helpers.

    Raises:
        ValueError: If `lap_table` is malformed. See `_validate`.
    """
    priors = priors or TyreSSMPriors()
    initial = initial or TyreSSMHyper()

    design = build_design(lap_table)

    opt = minimize(
        _negative_loglik,
        initial.to_vector(),
        args=(design, priors),
        method="L-BFGS-B",
        bounds=list(HYPER_BOUNDS),
        options={"maxiter": maxiter},
    )

    hyper = TyreSSMHyper.from_vector(opt.x)
    model = build_model(design, hyper, priors)
    obs = build_observations(design, hyper)
    filtered = filter_ssm_batched(model, obs)
    smoothed = smooth_ssm(model, filtered)

    return TyreSSMResult(
        hyper=hyper,
        priors=priors,
        index=design.index,
        model=model,
        obs=obs,
        filtered=filtered,
        smoothed=smoothed,
        design=design,
        loglik=float(filtered.loglik),
        n_obs=obs.n_obs,
        converged=bool(opt.success),
    )
