"""Counterfactuals and forward projection on the fitted tyre state.

Two related questions, both answered by evaluating the same fitted structure
under altered inputs rather than by any separate model:

  * **What would this lap have been if...** the car had been in clean air, or on
    a fresh tyre, or three laps further into the stint. Because the observation
    equation is additive in named states, removing a term is exact arithmetic.
  * **How long can this tyre still do a job.** The degradation rate is a state
    with its own dynamics, so projecting it forward is the model's own extrapolation
    with its own growing uncertainty, not a curve fitted after the fact.

Everything here is a model-based estimate of an unobserved quantity. The
counterfactual lap time was never driven. The UI labels these as estimates
throughout, and `Projection.applicability` degrades explicitly once a query
runs past the tyre ages the model actually saw.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from tyremind.models.ssm.tyre_ssm import TyreSSMResult

#: Default threshold for "no longer competitive": performance loss relative to a
#: fresh set of the same compound, in seconds per lap. Configurable, because what
#: counts as too slow depends entirely on the race situation -- a car in clean
#: air defending a gap tolerates far more than one trying to make an undercut work.
DEFAULT_PERFORMANCE_THRESHOLD_S = 0.8


@dataclass(frozen=True)
class Counterfactual:
    """One "what if" evaluated against a lap that was actually driven.

    Attributes:
        scenario: Machine-readable scenario key.
        label: Human-readable description.
        actual_lap_time: The lap time that was actually set, s.
        estimated_lap_time: What the model estimates it would have been, s.
        delta: estimated minus actual, s. Negative means the counterfactual is faster.
        sd: Posterior standard deviation of the delta.
        note: What assumption the estimate rests on.
    """

    scenario: str
    label: str
    actual_lap_time: float
    estimated_lap_time: float
    delta: float
    sd: float
    note: str

    @property
    def ci95(self) -> tuple[float, float]:
        return (self.delta - 1.96 * self.sd, self.delta + 1.96 * self.sd)

    def to_dict(self) -> dict:
        return {
            "scenario": self.scenario,
            "label": self.label,
            "actual_lap_time": self.actual_lap_time,
            "estimated_lap_time": self.estimated_lap_time,
            "delta": self.delta,
            "sd": self.sd,
            "ci95": list(self.ci95),
            "note": self.note,
            "is_model_estimate": True,
        }


@dataclass(frozen=True)
class Projection:
    """Forward projection of a tyre's performance.

    Attributes:
        driver: Car.
        from_lap: Lap the projection starts from.
        tyre_age: Tyre age at that lap.
        compound: Compound in use.
        horizon: Laps ahead projected, one entry per step.
        loss: Projected cumulative performance loss at each horizon, s.
        loss_sd: Posterior standard deviation of that loss.
        rate: Projected instantaneous degradation rate at each horizon, s/lap.
        rate_sd: Its standard deviation.
        threshold: Performance-loss threshold used for competitive life, s.
        breach_probability: Probability the loss exceeds the threshold, per horizon.
        applicability: How far inside the observed data each horizon sits, 1.0 down
            to 0.0. Falls away once the projection runs past the oldest tyre age
            the model has actually seen on this compound.
    """

    driver: str
    from_lap: int
    tyre_age: float
    compound: str
    horizon: np.ndarray
    loss: np.ndarray
    loss_sd: np.ndarray
    rate: np.ndarray
    rate_sd: np.ndarray
    threshold: float
    breach_probability: np.ndarray
    applicability: np.ndarray

    def competitive_life(self, confidence: float = 0.5) -> float:
        """Laps until the tyre is projected to breach the performance threshold.

        Args:
            confidence: Breach probability at which the tyre is called done.
                0.5 is the median expectation.

        Returns:
            Laps remaining, or the projection horizon if no breach is projected
            inside it (in which case the answer is "more than this", not "never").
        """
        beyond = np.nonzero(self.breach_probability >= confidence)[0]
        return float(self.horizon[beyond[0]]) if beyond.size else float(self.horizon[-1])

    def competitive_life_interval(self) -> tuple[float, float]:
        """Pessimistic and optimistic bounds on remaining competitive life, laps.

        Read as: the tyre could be done as early as the lower bound, and is
        unlikely to still be competitive past the upper one.

        Note the direction. Breach probability *rises* with horizon, so demanding
        high confidence of a breach before calling the tyre finished yields the
        *longer* life, not the shorter one. The pessimistic bound is therefore the
        low-confidence query and the optimistic bound the high-confidence one.
        """
        return (self.competitive_life(0.2), self.competitive_life(0.8))

    def to_dict(self) -> dict:
        lower, upper = self.competitive_life_interval()
        return {
            "driver": self.driver,
            "from_lap": int(self.from_lap),
            "tyre_age": self.tyre_age,
            "compound": self.compound,
            "threshold_s": self.threshold,
            "competitive_life_laps": self.competitive_life(),
            "competitive_life_lower": lower,
            "competitive_life_upper": upper,
            "horizon": self.horizon.tolist(),
            "loss": self.loss.tolist(),
            "loss_sd": self.loss_sd.tolist(),
            "rate": self.rate.tolist(),
            "rate_sd": self.rate_sd.tolist(),
            "breach_probability": self.breach_probability.tolist(),
            "applicability": self.applicability.tolist(),
            "is_model_estimate": True,
        }


def _lap_row(result: TyreSSMResult, driver: str, session_lap: int):
    df = result.design.lap_table
    match = df[(df["driver"] == driver) & (df["session_lap"] == session_lap)]
    if match.empty:
        raise ValueError(f"{driver} has no valid lap {session_lap} in this session")
    return match.iloc[0]


def counterfactuals(
    result: TyreSSMResult, driver: str, session_lap: int
) -> list[Counterfactual]:
    """Evaluate the standard set of "what if" scenarios for one lap.

    Each scenario removes or alters exactly one additive term of the fitted
    observation equation and leaves the rest untouched. That isolation is the
    point: removing traffic must not silently change the tyre contribution, and
    because the terms are separate states here, it cannot.

    Args:
        result: A fitted model.
        driver: Car.
        session_lap: Lap to reason about.

    Returns:
        Counterfactuals for clean air, a fresh tyre, and both together.

    Raises:
        ValueError: If the lap is not a valid lap for that driver.
    """
    idx = result.index
    row = _lap_row(result, driver, session_lap)
    step = result.design.step_of_lap[session_lap]

    smooth = result.smoothed.a_smooth
    sd = result.smoothed.std()
    actual = float(row["lap_time"])

    traffic_index = float(row.get("traffic_index", 0.0))
    traffic_effect = float(smooth[step, idx.traffic_coef]) * traffic_index
    traffic_sd = float(sd[step, idx.traffic_coef]) * abs(traffic_index)

    tyre_effect = float(smooth[step, idx.level[driver]])
    tyre_sd = float(sd[step, idx.level[driver]])

    return [
        Counterfactual(
            scenario="clean_air",
            label="In clean air",
            actual_lap_time=actual,
            estimated_lap_time=actual - traffic_effect,
            delta=-traffic_effect,
            sd=traffic_sd,
            note=(
                "Removes the estimated traffic penalty for this lap. The tyre "
                "state is left untouched -- a car in clean air still has the "
                "tyre it has."
            ),
        ),
        Counterfactual(
            scenario="fresh_tyre",
            label="On a fresh set of the same compound",
            actual_lap_time=actual - tyre_effect,
            estimated_lap_time=actual - tyre_effect,
            delta=-tyre_effect,
            sd=tyre_sd,
            note=(
                "Removes accumulated tyre performance loss, holding fuel load, "
                "traffic and track state fixed. Not a pit stop: it does not pay "
                "the pit-lane time loss."
            ),
        ),
        Counterfactual(
            scenario="clean_air_fresh_tyre",
            label="Clean air on a fresh set",
            actual_lap_time=actual,
            estimated_lap_time=actual - traffic_effect - tyre_effect,
            delta=-(traffic_effect + tyre_effect),
            sd=float(np.hypot(traffic_sd, tyre_sd)),
            note=(
                "The car's underlying pace at this fuel load and track state. "
                "This is the number to compare across cars, because it is the "
                "one with the situational noise taken out."
            ),
        ),
    ]


def project_tyre(
    result: TyreSSMResult,
    driver: str,
    session_lap: int,
    *,
    horizon: int = 20,
    threshold: float = DEFAULT_PERFORMANCE_THRESHOLD_S,
) -> Projection:
    """Project a tyre's performance forward from a given lap.

    Runs the fitted state dynamics forward with no further observations. Loss
    accumulates as the rate integrates, and the rate itself keeps drifting, so
    uncertainty grows super-linearly with the horizon -- which is correct, and is
    why a twenty-lap projection is honestly much vaguer than a five-lap one.

    Under the local linear trend, projecting h laps ahead gives

        loss(h)  = level + h * rate
        var(h)   = var(level) + h^2 var(rate) + 2h cov(level, rate)
                   + h * q_level + (h^3/3) * q_rate

    where the last two terms are accumulated process noise. The cubic term is
    what makes far-horizon projections widen sharply, and it is a property of
    the model rather than an added safety margin.

    Args:
        result: A fitted model.
        driver: Car.
        session_lap: Lap to project from.
        horizon: Laps ahead to project.
        threshold: Performance loss beyond which the tyre is no longer
            competitive, s.

    Returns:
        A Projection with per-horizon loss, rate, breach probability and
        applicability.

    Raises:
        ValueError: If the lap is not valid for that driver, or horizon < 1.
    """
    if horizon < 1:
        raise ValueError(f"horizon must be at least 1 lap, got {horizon}")

    idx = result.index
    row = _lap_row(result, driver, session_lap)
    step = result.design.step_of_lap[session_lap]

    i_level, i_rate = idx.level[driver], idx.rate[driver]
    P = result.smoothed.P_smooth[step]

    level = float(result.smoothed.a_smooth[step, i_level])
    rate = float(result.smoothed.a_smooth[step, i_rate])
    var_level = float(P[i_level, i_level])
    var_rate = float(P[i_rate, i_rate])
    cov = float(P[i_level, i_rate])

    h = np.arange(1, horizon + 1, dtype=float)

    loss = level + h * rate
    var = (
        var_level
        + h**2 * var_rate
        + 2.0 * h * cov
        + h * result.hyper.q_level
        + (h**3 / 3.0) * result.hyper.q_rate
    )
    loss_sd = np.sqrt(np.maximum(var, 0.0))

    rate_forward = np.full_like(h, rate)
    rate_sd = np.sqrt(np.maximum(var_rate + h * result.hyper.q_rate, 0.0))

    # Probability the projected loss exceeds the threshold, under the Gaussian
    # posterior at each horizon.
    from scipy.stats import norm

    breach = norm.sf(threshold, loc=loss, scale=np.maximum(loss_sd, 1e-9))

    # Applicability: how far past the oldest tyre age the model actually observed
    # on this compound are we asking it to speak. Beyond that, the projection is
    # pure extrapolation of a trend and deserves to be flagged as such.
    compound = str(row["compound"])
    observed = result.design.lap_table
    same_compound = observed[observed["compound"] == compound]["tyre_age"]
    max_observed_age = float(same_compound.max()) if not same_compound.empty else 0.0

    projected_age = float(row["tyre_age"]) + h
    overshoot = np.maximum(projected_age - max_observed_age, 0.0)
    applicability = np.exp(-overshoot / 5.0)

    return Projection(
        driver=driver,
        from_lap=int(session_lap),
        tyre_age=float(row["tyre_age"]),
        compound=compound,
        horizon=h,
        loss=loss,
        loss_sd=loss_sd,
        rate=rate_forward,
        rate_sd=rate_sd,
        threshold=threshold,
        breach_probability=breach,
        applicability=applicability,
    )
