"""Attribute an observed lap-time change to the causes that produced it.

This answers the question the whole platform exists for: a car is four tenths
slower than it was ten laps ago -- how much of that is actually the tyre?

**This is not feature attribution.** SHAP and permutation importance answer "how
much does this input move the prediction", which is a statement about the model,
not about the car. Here the model is *additive in named latent states* by
construction, so the decomposition is exact arithmetic on the fitted structure:

    y[lap] - y[ref] = sum over terms of ( Z[lap,i]*x[lap,i] - Z[ref,i]*x[ref,i] )

Every term is a quantity with physical meaning and its own posterior. Nothing is
approximated, and nothing is sampled.

What it is *not* is causal in the interventional sense. These are contributions
under an assumed structural model whose identifying assumptions are stated in
`tyre_ssm` -- two of the three collinearities are resolved by prior, not by data.
The honest label is "structural attribution under the stated model", and that is
what the UI says. Calling it causal identification would be overclaiming, and it
is the specific overclaim this field is prone to.


Choice of reference
-------------------
Attribution is always relative to something. The default reference is the *start
of the car's current run*, which is the one comparison where every term is exact:
tyre performance loss is zero there by definition, laps-completed-this-run is
zero, and the run intercept cancels. Comparing against an arbitrary earlier lap
is also supported, and is what the "why is the car slow now" view uses, but the
tyre term's interval is then conservative -- see `_tyre_delta_sd`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from tyremind.models.ssm.tyre_ssm import TyreSSMResult, track_basis

#: Which terms count as "the tyre" when computing the headline share. Everything
#: else is a confounder -- something that changed the lap time without the tyre
#: having changed at all.
TYRE_TERMS = frozenset({"tyre"})


@dataclass(frozen=True)
class Contribution:
    """One named cause of a lap-time change.

    Attributes:
        key: Stable machine identifier.
        label: Human-readable name for display.
        seconds: Contribution to the lap-time delta. Positive means slower.
        sd: Posterior standard deviation of that contribution.
        is_tyre: Whether this term is genuine tyre degradation as opposed to a
            confounder.
    """

    key: str
    label: str
    seconds: float
    sd: float
    is_tyre: bool = False

    @property
    def ci95(self) -> tuple[float, float]:
        """95% credible interval for this contribution."""
        return (self.seconds - 1.96 * self.sd, self.seconds + 1.96 * self.sd)


@dataclass
class LapDecomposition:
    """A full accounting of why one lap differed from a reference lap.

    Attributes:
        driver: Car identifier.
        session_lap: The lap being explained.
        reference_lap: The lap it is measured against.
        observed_delta: Actual lap-time difference, s. Positive means slower.
        contributions: Named causes, largest absolute effect first.
        residual: Part of the delta the model does not explain, s. Driver noise,
            a lock-up, a lift -- anything outside the structure.
        tyre_age: Tyre age at the lap being explained, laps.
        compound: Compound in use.
    """

    driver: str
    session_lap: int
    reference_lap: int
    observed_delta: float
    contributions: list[Contribution]
    residual: float
    tyre_age: float
    compound: str

    @property
    def tyre_seconds(self) -> float:
        """Seconds of the delta attributable to tyre degradation."""
        return sum(c.seconds for c in self.contributions if c.is_tyre)

    @property
    def confounder_seconds(self) -> float:
        """Seconds attributable to everything that is not the tyre."""
        return sum(c.seconds for c in self.contributions if not c.is_tyre)

    @property
    def tyre_share(self) -> float:
        """Fraction of the observed slowdown that is genuinely the tyre.

        Returns NaN when the car did not actually slow down, because a share of a
        non-slowdown is not a meaningful quantity and rendering one would invite
        a false reading.
        """
        if self.observed_delta <= 0:
            return float("nan")
        return float(self.tyre_seconds / self.observed_delta)

    def to_dict(self) -> dict:
        return {
            "driver": self.driver,
            "session_lap": int(self.session_lap),
            "reference_lap": int(self.reference_lap),
            "observed_delta": self.observed_delta,
            "residual": self.residual,
            "tyre_age": self.tyre_age,
            "compound": self.compound,
            "tyre_seconds": self.tyre_seconds,
            "confounder_seconds": self.confounder_seconds,
            "tyre_share": self.tyre_share,
            "contributions": [
                {
                    "key": c.key,
                    "label": c.label,
                    "seconds": c.seconds,
                    "sd": c.sd,
                    "ci95": list(c.ci95),
                    "is_tyre": c.is_tyre,
                }
                for c in self.contributions
            ],
        }


def _lap_row(result: TyreSSMResult, driver: str, session_lap: int) -> pd.Series:
    df = result.design.lap_table
    match = df[(df["driver"] == driver) & (df["session_lap"] == session_lap)]
    if match.empty:
        raise ValueError(f"{driver} has no valid lap {session_lap} in this session")
    return match.iloc[0]


def _tyre_delta_sd(
    result: TyreSSMResult, driver: str, step_a: int, step_b: int
) -> float:
    """Posterior sd of the change in tyre performance loss between two steps.

    The RTS smoother returns marginal covariances but not the cross-covariance
    between two time steps, so Var(x_a - x_b) = Var(a) + Var(b) - 2Cov(a,b) is not
    directly available. The two are strongly positively correlated -- they are
    the same tyre minutes apart -- so dropping the covariance term overstates the
    spread.

    We take that trade deliberately: the alternative is a lag-covariance smoother
    for a quantity that only sets the width of an error bar, and an interval that
    is too wide is a far safer failure than one that is too narrow.

    When the reference is the run start, tyre loss is zero with negligible
    variance and the result is exact anyway, which is the default path.
    """
    i = result.index.level[driver]
    sd = result.smoothed.std()[:, i]
    return float(np.hypot(sd[step_a], sd[step_b]))


def decompose_lap(
    result: TyreSSMResult,
    driver: str,
    session_lap: int,
    reference_lap: int | None = None,
) -> LapDecomposition:
    """Explain why one lap was slower or faster than a reference lap.

    Args:
        result: A fitted model.
        driver: Car to explain.
        session_lap: The lap in question.
        reference_lap: Lap to compare against. Defaults to the first lap of the
            car's current run, which is the exact-arithmetic case.

    Returns:
        A LapDecomposition whose contributions plus residual sum exactly to the
        observed delta.

    Raises:
        ValueError: If either lap is not a valid lap for this driver, or the two
            laps are on different runs (which would make the run intercepts fail
            to cancel and the decomposition meaningless).
    """
    idx = result.index
    df = result.design.lap_table

    row = _lap_row(result, driver, session_lap)
    run_id = int(row["run_id"])

    if reference_lap is None:
        run_laps = df[(df["driver"] == driver) & (df["run_id"] == run_id)]
        reference_lap = int(run_laps["session_lap"].min())

    ref = _lap_row(result, driver, reference_lap)

    if int(ref["run_id"]) != run_id:
        raise ValueError(
            f"lap {session_lap} and reference lap {reference_lap} are on different "
            f"runs ({run_id} and {int(ref['run_id'])}). Their run intercepts would "
            "not cancel, so the decomposition would not be interpretable. Compare "
            "laps within a run, or use the run start as the reference."
        )

    step = result.design.step_of_lap[session_lap]
    step_ref = result.design.step_of_lap[reference_lap]

    smooth = result.smoothed.a_smooth
    sd = result.smoothed.std()

    contributions: list[Contribution] = []

    # --- tyre: the quantity of interest -----------------------------------
    i_level = idx.level[driver]
    tyre_seconds = float(smooth[step, i_level] - smooth[step_ref, i_level])
    contributions.append(
        Contribution(
            key="tyre",
            label="Tyre degradation",
            seconds=tyre_seconds,
            sd=_tyre_delta_sd(result, driver, step, step_ref),
            is_tyre=True,
        )
    )

    # --- fuel: the car got lighter, so this is normally a gain -------------
    d_laps = float(row["lap_in_run"]) - float(ref["lap_in_run"])
    fuel_mean = float(smooth[step, idx.fuel_slope])
    contributions.append(
        Contribution(
            key="fuel",
            label="Fuel burn-off",
            seconds=-fuel_mean * d_laps,
            sd=abs(d_laps) * float(sd[step, idx.fuel_slope]),
        )
    )

    # --- track evolution: shared across the field --------------------------
    session_start = float(df["session_lap"].min())
    basis = track_basis(
        np.array([session_lap - session_start, reference_lap - session_start], dtype=float),
        result.hyper.track_shape,
    )
    d_basis = float(basis[0] - basis[1])
    amp_mean = float(smooth[step, idx.track_amplitude])
    resid = float(smooth[step, idx.track_resid] - smooth[step_ref, idx.track_resid])
    contributions.append(
        Contribution(
            key="track",
            label="Track evolution",
            seconds=d_basis * amp_mean + resid,
            sd=abs(d_basis) * float(sd[step, idx.track_amplitude]),
        )
    )

    # --- traffic ------------------------------------------------------------
    d_traffic = float(row.get("traffic_index", 0.0)) - float(ref.get("traffic_index", 0.0))
    traffic_mean = float(smooth[step, idx.traffic_coef])
    contributions.append(
        Contribution(
            key="traffic",
            label="Traffic",
            seconds=traffic_mean * d_traffic,
            sd=abs(d_traffic) * float(sd[step, idx.traffic_coef]),
        )
    )

    observed_delta = float(row["lap_time"] - ref["lap_time"])
    residual = observed_delta - sum(c.seconds for c in contributions)

    contributions.sort(key=lambda c: abs(c.seconds), reverse=True)

    return LapDecomposition(
        driver=driver,
        session_lap=int(session_lap),
        reference_lap=int(reference_lap),
        observed_delta=observed_delta,
        contributions=contributions,
        residual=residual,
        tyre_age=float(row["tyre_age"]),
        compound=str(row["compound"]),
    )


def decompose_run(result: TyreSSMResult, driver: str, run_id: int) -> pd.DataFrame:
    """Decompose every lap of one run against that run's opening lap.

    The lap-by-lap version of `decompose_lap`, shaped for plotting a stacked
    area chart of where the time is going as a stint develops.

    Args:
        result: A fitted model.
        driver: Car to explain.
        run_id: Run to walk through.

    Returns:
        One row per lap, with a column per contribution plus `residual`,
        `observed_delta` and `tyre_share`.

    Raises:
        ValueError: If the driver has no laps on that run.
    """
    df = result.design.lap_table
    run = df[(df["driver"] == driver) & (df["run_id"] == run_id)].sort_values("session_lap")
    if run.empty:
        raise ValueError(f"{driver} has no laps on run {run_id}")

    rows = []
    for lap in run["session_lap"]:
        d = decompose_lap(result, driver, int(lap))
        record = {
            "session_lap": d.session_lap,
            "tyre_age": d.tyre_age,
            "compound": d.compound,
            "observed_delta": d.observed_delta,
            "residual": d.residual,
            "tyre_share": d.tyre_share,
        }
        record.update({c.key: c.seconds for c in d.contributions})
        record.update({f"{c.key}_sd": c.sd for c in d.contributions})
        rows.append(record)

    return pd.DataFrame(rows)
