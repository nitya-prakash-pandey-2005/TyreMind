"""What shape a stint's degradation curve actually has.

The rest of this project estimates a degradation *rate* -- one number, seconds
per lap. That answers "how fast is this tyre losing performance" and cannot
answer "how many laps have I got left", and the two are only the same question
if the curve is a line.

It usually is. Across the corpus roughly half of all stints are described
adequately by a straight line once fuel, track evolution and traffic have been
removed. The other half are not, and they are not all failing in the same way:

    LINEAR    a straight line, within the noise.
    WARM-UP   the tyre gets FASTER and then turns. Coming to temperature, or a
              graining phase clearing. Early in the stint.
    CLIFF     the tyre is already degrading and then degrades markedly faster.
              The thing a pit wall means by the word. Late in the stint.
    RECOVERY  the rate DROPS partway through. A tyre coming back to the driver.

WARM-UP and CLIFF have the same arithmetic signature -- a changepoint with a
positive step in slope -- and opposite physical meanings. A model that reports
only "a changepoint at lap 9" describes neither, and an early version of the
analysis behind this module did exactly that, reporting a "cliff" a third of the
way through a stint that was a tyre coming to temperature.

THE MODEL. A continuous broken stick in tyre age:

    L(a) = alpha + beta * a                                   for a <= tau
    L(a) = alpha + beta * tau + (beta + delta) * (a - tau)    for a >  tau

fitted as ordinary least squares on the basis [1, a, max(a - tau, 0)], which is
continuous at tau by construction rather than by constraint. tau is found by
exhaustive search over admissible laps, because the least-squares surface in tau
is piecewise-smooth with many local minima and a gradient method lands in
whichever one it started nearest.

WHAT THIS IS NOT FOR. The changepoint is descriptive, not predictive. Fitting the
opening 70% of a stint does not locate a cliff that has not happened yet, and
experiment 17 measures that directly: on cliffed stints a broken stick fitted
early forecasts no better than a straight line. What it does establish is the
*direction* of the error -- a straight line is systematically optimistic about
the end of a stint, by around four tenths of a second on the stints that cliff.
Use this to read a stint that has happened, not to predict one that has not.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

#: A changepoint needs data either side of it. Four laps is the minimum that
#: could distinguish a regime change from a pair of outliers.
MIN_SEGMENT = 4

#: Below this many laps a stint cannot support a changepoint at all.
MIN_STINT_LAPS = MIN_SEGMENT * 3

#: A step in rate smaller than this is a rounding error, not a regime. Over a
#: five-lap window it is a tenth of a second, which is the scale a strategist
#: would act on.
MEANINGFUL_DELTA = 0.02

#: A tyre losing time more slowly than this before the break is not degrading.
#: Below minus this it is genuinely getting quicker, which a degrading tyre does
#: not do, so the break is a warm-up boundary rather than a cliff.
WARMUP_SLOPE = 0.005


@dataclass(frozen=True)
class CurveShape:
    """The fitted shape of one stint's degradation curve.

    Attributes:
        regime: One of "linear", "warm-up", "cliff", "recovery".
        n_laps: Laps the fit used.
        slope: Degradation before any changepoint, s/lap. For a linear stint
            this is the whole story.
        changepoint_age: Tyre age at the regime change, or None if linear.
        delta: Step UP in rate at the changepoint, s/lap. Positive is a cliff.
        slope_after: Rate after the changepoint.
        position: How far through the stint the changepoint sits, 0 to 1.
        linear_bic: BIC of the straight-line fit. Lower is better.
        broken_bic: BIC of the broken stick, charged for its two extra
            parameters -- including tau, which is fitted by searching a grid and
            is still fitting however it is found.
        rmse: Residual root-mean-square of the chosen model, seconds.
    """

    regime: str
    n_laps: int
    slope: float
    changepoint_age: float | None
    delta: float | None
    slope_after: float | None
    position: float | None
    linear_bic: float
    broken_bic: float
    rmse: float

    @property
    def is_linear(self) -> bool:
        return self.regime == "linear"

    @property
    def description(self) -> str:
        """One sentence a strategist can read without the statistics."""
        if self.regime == "linear":
            return f"Losing {self.slope:.3f} s/lap, steadily."
        if self.regime == "warm-up":
            return (
                f"Came in over the first {self.changepoint_age:.0f} laps, then settled "
                f"into {self.slope_after:.3f} s/lap."
            )
        if self.regime == "cliff":
            return (
                f"Held at {self.slope:.3f} s/lap to lap {self.changepoint_age:.0f}, "
                f"then fell away to {self.slope_after:.3f} s/lap."
            )
        return (
            f"Degraded at {self.slope:.3f} s/lap, then recovered to "
            f"{self.slope_after:.3f} s/lap."
        )

    def to_dict(self) -> dict:
        return asdict(self) | {"description": self.description}


def _bic(rss: float, n: int, k: int) -> float:
    """Gaussian BIC. Lower is better."""
    if rss <= 0 or n <= k:
        return float("inf")
    return float(n * np.log(rss / n) + k * np.log(n))


def _fit_linear(age: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float]:
    design = np.column_stack([np.ones_like(age), age])
    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    return coef, float(((y - design @ coef) ** 2).sum())


def _fit_broken(age: np.ndarray, y: np.ndarray) -> dict | None:
    """Best continuous broken stick, by exhaustive search over the changepoint."""
    candidates = np.unique(age)
    candidates = candidates[
        (candidates >= age.min() + MIN_SEGMENT) & (candidates <= age.max() - MIN_SEGMENT)
    ]
    if candidates.size == 0:
        return None

    best: dict | None = None
    for tau in candidates:
        design = np.column_stack([np.ones_like(age), age, np.maximum(age - tau, 0.0)])
        coef, *_ = np.linalg.lstsq(design, y, rcond=None)
        rss = float(((y - design @ coef) ** 2).sum())
        if best is None or rss < best["rss"]:
            best = {"tau": float(tau), "coef": coef, "rss": rss}
    return best


def fit_curve_shape(tyre_age, performance) -> CurveShape | None:
    """Fit both shapes to one stint and return whichever the evidence supports.

    Args:
        tyre_age: Tyre age per lap. Need not be sorted.
        performance: De-confounded lap time, or any monotone measure of
            accumulated loss, aligned to `tyre_age`. Passing raw lap time here
            would fold fuel burn-off into the slope and is the caller's mistake
            to avoid -- this function has no way to detect it.

    Returns:
        A CurveShape, or None if the stint is too short to say anything.
    """
    age = np.asarray(tyre_age, dtype=float)
    y = np.asarray(performance, dtype=float)
    keep = np.isfinite(age) & np.isfinite(y)
    age, y = age[keep], y[keep]
    if age.size < MIN_STINT_LAPS or np.ptp(age) < MIN_SEGMENT:
        return None

    order = np.argsort(age)
    age, y = age[order], y[order]

    linear_coef, linear_rss = _fit_linear(age, y)
    n = age.size
    linear_bic = _bic(linear_rss, n, 3)
    broken = _fit_broken(age, y)

    if broken is None:
        return CurveShape(
            regime="linear", n_laps=n, slope=float(linear_coef[1]),
            changepoint_age=None, delta=None, slope_after=None, position=None,
            linear_bic=linear_bic, broken_bic=float("inf"),
            rmse=float(np.sqrt(linear_rss / n)),
        )

    broken_bic = _bic(broken["rss"], n, 5)
    slope_before = float(broken["coef"][1])
    delta = float(broken["coef"][2])
    position = float((broken["tau"] - age[0]) / max(age[-1] - age[0], 1e-9))

    # The classification, and the reason this module exists rather than a bare
    # changepoint fitter: warm-up and cliff are the same arithmetic and opposite
    # events, and only the sign of the slope BEFORE the break separates them.
    if broken_bic >= linear_bic:
        regime = "linear"
    elif delta < -MEANINGFUL_DELTA:
        regime = "recovery"
    elif slope_before < -WARMUP_SLOPE and delta > MEANINGFUL_DELTA:
        regime = "warm-up"
    elif delta > MEANINGFUL_DELTA:
        regime = "cliff"
    else:
        regime = "linear"

    if regime == "linear":
        return CurveShape(
            regime="linear", n_laps=n, slope=float(linear_coef[1]),
            changepoint_age=None, delta=None, slope_after=None, position=None,
            linear_bic=linear_bic, broken_bic=broken_bic,
            rmse=float(np.sqrt(linear_rss / n)),
        )

    return CurveShape(
        regime=regime, n_laps=n, slope=slope_before,
        changepoint_age=broken["tau"], delta=delta,
        slope_after=slope_before + delta, position=position,
        linear_bic=linear_bic, broken_bic=broken_bic,
        rmse=float(np.sqrt(broken["rss"] / n)),
    )


def deconfounded_lap_time(lap_table: pd.DataFrame, fit) -> pd.Series:
    """Lap time with fuel, track evolution and traffic removed.

    The curve is only meaningful on this. Fuel burn-off is worth more per lap
    than the degradation being measured and pushes the other way, so a shape
    fitted to raw lap time describes the fuel load.

    Args:
        lap_table: The session, in the standard schema.
        fit: A fitted `TyreSSMResult` for that session.

    Returns:
        De-confounded lap time, aligned to `lap_table`.
    """
    track = fit.track_evolution().set_index("session_lap")["track_effect"]
    return (
        lap_table["lap_time"]
        + fit.fuel_slope()[0] * lap_table["lap_in_run"]
        - lap_table["session_lap"].map(track).fillna(0.0)
        - fit.traffic_coefficient()[0] * lap_table["traffic_index"]
    )


def session_curve_shapes(lap_table: pd.DataFrame, fit) -> dict[tuple[str, int], CurveShape]:
    """Curve shape for every stint in a session, keyed by (driver, run_id)."""
    corrected = deconfounded_lap_time(lap_table, fit)
    frame = lap_table.assign(_corrected=corrected)

    shapes: dict[tuple[str, int], CurveShape] = {}
    for (driver, run_id), stint in frame.groupby(["driver", "run_id"]):
        shape = fit_curve_shape(stint["tyre_age"], stint["_corrected"])
        if shape is not None:
            shapes[(str(driver), int(run_id))] = shape
    return shapes
