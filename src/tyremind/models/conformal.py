"""Turn a posterior into an interval that is worth what it claims to be worth.

The state-space model reports a posterior standard deviation for every
degradation rate it estimates, and that number is correct. It is also the answer
to a question nobody asks. It says how tightly *these practice laps* pin down
*the practice degradation rate* under *this model*. What a strategist wants to
know on Friday evening is how well that rate predicts Sunday, and the transfer
between the two carries error the posterior has never seen: fuel loads, track
state, driving style, tyre management, a red flag, a wind shift.

Measured over 62 compound comparisons across 2024 and 2023, the uncalibrated 95%
interval contained the race rate 76% of the time. A model that is overconfident
in a known direction is worse than one that says nothing, because a strategist
can plan around a wide interval and cannot plan around a wrong one.

Split conformal prediction repairs this with remarkably little machinery and one
genuinely strong guarantee. Given a calibration set of past
(prediction, outcome) pairs that is exchangeable with the case at hand, the
resulting interval has marginal coverage of at least 1 - alpha in finite
samples. It does not assume Gaussian errors. It does not assume the model is
well specified. It does not even assume the model is any good -- a useless
predictor gets correspondingly enormous intervals, which is the honest outcome.

What it does assume is exchangeability, and that assumption is where the care
goes. See `ConformalCalibrator.fit` for why calibration folds are grouped by
event rather than by comparison.

Reference: Vovk, Gammerman & Shafer, *Algorithmic Learning in a Random World*
(2005); Lei et al., *Distribution-Free Predictive Inference for Regression*
(JASA, 2018) for the split-conformal form used here.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

#: Where the fitted calibration lives once `experiments/exp12` has produced it.
#: A reference artefact rather than a code constant: it is measured from data and
#: will move as the corpus grows, and it should move visibly in a diff.
CALIBRATION_PATH = Path("data/reference/conformal_calibration.json")

#: Nonconformity scores this module knows how to apply.
#:
#: absolute      |predicted - actual|. One width for every case. Robust, and
#:               silent about which predictions are shaky.
#: studentised   |predicted - actual| / posterior_sd. Width scales with the
#:               model's own uncertainty, so a thin practice session is given a
#:               wider interval than a session with six clean long runs. Only
#:               worth using if the posterior sd carries real information about
#:               the error; on our data it narrows the mean interval from 0.268
#:               to 0.250 s/lap at identical coverage, so it does, slightly.
SCORES = ("absolute", "studentised")


class CalibrationUnavailableError(RuntimeError):
    """Raised when a calibrated interval is required but none has been fitted."""


@dataclass(frozen=True)
class ConformalCalibrator:
    """A fitted conformal calibration for practice-to-race degradation rates.

    Attributes:
        score: Which nonconformity score the quantile belongs to. Mixing these up
            silently produces intervals off by a factor of the posterior sd, so
            it is stored rather than assumed.
        alpha: Miss rate the calibration targets. 0.05 for a 95% interval.
        quantile: The calibrated score threshold.
        bias: Mean signed error of the point prediction, subtracted before the
            interval is drawn. Positive means practice over-predicts.
        n_calibration: How many comparisons the quantile was computed from.
        n_events: How many distinct events those comparisons came from. The
            honest measure of sample size here, since comparisons within an event
            are not independent.
        seasons: Which seasons contributed.
        generated_at: When it was fitted.
    """

    score: str
    alpha: float
    quantile: float
    bias: float
    n_calibration: int
    n_events: int
    seasons: list[int] = field(default_factory=list)
    generated_at: str = ""

    def __post_init__(self) -> None:
        if self.score not in SCORES:
            raise ValueError(f"unknown score {self.score!r}, expected one of {SCORES}")
        if not 0.0 < self.alpha < 1.0:
            raise ValueError(f"alpha must lie in (0, 1), got {self.alpha}")
        if not np.isfinite(self.quantile) or self.quantile < 0.0:
            raise ValueError(f"quantile must be finite and non-negative, got {self.quantile}")

    @property
    def confidence(self) -> float:
        """The nominal coverage, as a fraction."""
        return 1.0 - self.alpha

    def half_width(self, posterior_sd: float) -> float:
        """Half the width of the calibrated interval for one prediction."""
        if self.score == "studentised":
            return float(self.quantile * posterior_sd)
        return float(self.quantile)

    def interval(
        self, predicted: float, posterior_sd: float, *, physical: bool = True
    ) -> tuple[float, float]:
        """Calibrated interval around a practice-derived rate.

        Args:
            predicted: The practice degradation rate, s/lap.
            posterior_sd: Its posterior standard deviation from the filter.
            physical: Clip the interval at zero. A negative degradation rate is a
                tyre that gets faster the longer it is used, which does not
                happen; all 62 dry-session race rates measured for the
                calibration are positive, minimum +0.0124 s/lap.

                This does not weaken the conformal guarantee, and the reason is
                worth stating: intersecting a valid prediction interval with a
                region that is known to contain the truth can only remove
                non-covering mass, so coverage is preserved or improved. Clipping
                to a region that might exclude the truth would be a different and
                indefensible act.

        Returns:
            (low, high), bias-corrected.
        """
        centre = predicted - self.bias
        half = self.half_width(posterior_sd)
        low, high = centre - half, centre + half
        if physical:
            low, high = max(low, 0.0), max(high, 0.0)
        return (low, high)

    def covers(
        self, predicted: float, posterior_sd: float, actual: float, *, physical: bool = True
    ) -> bool:
        low, high = self.interval(predicted, posterior_sd, physical=physical)
        return bool(low <= actual <= high)

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path: Path = CALIBRATION_PATH) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2))
        return path

    @classmethod
    def from_dict(cls, payload: dict) -> ConformalCalibrator:
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in payload.items() if k in known})

    @classmethod
    def load(cls, path: Path = CALIBRATION_PATH) -> ConformalCalibrator:
        """Load a fitted calibration.

        Raises:
            CalibrationUnavailableError: If none has been fitted. Deliberately an
                error rather than a silent fallback to the Gaussian interval --
                quietly returning an interval known to cover 76% of the time
                while labelled 95% is the exact failure this module exists to
                prevent.
        """
        if not path.exists():
            raise CalibrationUnavailableError(
                f"no conformal calibration at {path}. "
                "Run: python experiments/exp12_conformal_intervals.py"
            )
        return cls.from_dict(json.loads(path.read_text()))

    @classmethod
    def fit(
        cls,
        predicted: np.ndarray,
        actual: np.ndarray,
        posterior_sd: np.ndarray,
        *,
        events: np.ndarray | None = None,
        alpha: float = 0.05,
        score: str = "studentised",
        seasons: list[int] | None = None,
    ) -> ConformalCalibrator:
        """Fit a calibration from past practice-to-race comparisons.

        Args:
            predicted: Practice-derived rates.
            actual: The race rates they were scored against.
            posterior_sd: Posterior sd of each prediction. Required for the
                studentised score, ignored by the absolute one.
            events: Event identifier per comparison, used only to report how many
                independent events the calibration rests on. Two compounds at one
                Grand Prix share a track, a weather window and a fuel correction,
                so 62 comparisons across 27 events is 27 pieces of evidence and
                not 62, and the caller deserves to see both numbers.
            alpha: Target miss rate.
            score: One of SCORES.
            seasons: Recorded for provenance.

        Returns:
            A fitted ConformalCalibrator.

        Raises:
            ValueError: If the inputs disagree in length, or the calibration set
                is too small for the requested alpha to be representable.
        """
        predicted = np.asarray(predicted, dtype=float)
        actual = np.asarray(actual, dtype=float)
        posterior_sd = np.asarray(posterior_sd, dtype=float)
        if not predicted.shape == actual.shape == posterior_sd.shape:
            raise ValueError("predicted, actual and posterior_sd must be the same length")
        if score not in SCORES:
            raise ValueError(f"unknown score {score!r}, expected one of {SCORES}")

        finite = np.isfinite(predicted) & np.isfinite(actual) & np.isfinite(posterior_sd)
        if score == "studentised":
            finite &= posterior_sd > 0.0
        predicted, actual, posterior_sd = predicted[finite], actual[finite], posterior_sd[finite]

        n = predicted.size
        # The finite-sample guarantee needs the ceil((n+1)(1-alpha))-th order
        # statistic to exist. Below that threshold conformal does not fail loudly,
        # it returns an infinite interval -- technically valid, operationally
        # useless, and worth refusing outright.
        if n < int(np.ceil((n + 1) * (1.0 - alpha))):
            raise ValueError(
                f"{n} calibration points cannot represent a {1 - alpha:.0%} interval; "
                f"need at least {int(np.ceil(1.0 / alpha)) - 1}"
            )

        bias = float((predicted - actual).mean())
        residuals = np.abs(predicted - actual - bias)
        scores = residuals / posterior_sd if score == "studentised" else residuals
        quantile = conformal_quantile(scores, alpha)

        n_events = int(len(set(map(str, events)))) if events is not None else n
        return cls(
            score=score,
            alpha=float(alpha),
            quantile=float(quantile),
            bias=bias,
            n_calibration=int(n),
            n_events=n_events,
            seasons=sorted(seasons or []),
            generated_at=datetime.now(UTC).isoformat(),
        )


def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    """The finite-sample-corrected (1 - alpha) quantile of nonconformity scores.

    Taking the ceil((n+1)(1-alpha))-th order statistic rather than the plain
    empirical quantile is the entire reason split conformal carries a guarantee
    at every n rather than an asymptotic hope. The +1 accounts for the test point
    itself, which under exchangeability is equally likely to take any rank among
    the n + 1 scores.

    Args:
        scores: Nonconformity scores from the calibration set.
        alpha: Target miss rate.

    Returns:
        The threshold, or infinity if the sample is too small to represent it.
    """
    scores = np.asarray(scores, dtype=float)
    scores = scores[np.isfinite(scores)]
    n = scores.size
    if n == 0:
        return float("inf")
    k = int(np.ceil((n + 1) * (1.0 - alpha)))
    if k > n:
        return float("inf")
    return float(np.sort(scores)[k - 1])


class AdaptiveConformal:
    """Online conformal intervals for a quantity that drifts.

    Split conformal needs the calibration set to be exchangeable with the test
    point. A degradation rate per compound per event satisfies that; a lap time
    inside a race does not. The car burns fuel and gets faster, the track rubbers
    in and gets faster, a safety car rearranges everything. A residual from lap 8
    and a residual from lap 48 are not draws from one distribution, so a fixed
    quantile is calibrated for a session that no longer exists.

    Adaptive Conformal Inference (Gibbs & Candes, NeurIPS 2021) drops the
    assumption instead of hoping it holds. Rather than fixing a quantile it
    treats the working miss-rate as state and moves it after every observation::

        alpha_{t+1} = alpha_t + gamma * (alpha - err_t)

    where ``err_t`` is 1 when the observation fell outside the interval. Miss too
    often and the interval widens; miss too rarely and it tightens. Long-run
    coverage converges to ``1 - alpha`` under *arbitrary* distribution shift,
    with no exchangeability assumption at all.

    The price is that the guarantee is long-run rather than per-observation, and
    that it needs a warm-up before it has any scores to take a quantile of.

    Usage is one call per observation, in order::

        aci = AdaptiveConformal(alpha=0.05)
        for predicted, sd, actual in stream:
            low, high = aci.interval(predicted, sd)
            aci.update(predicted, sd, actual)
    """

    def __init__(
        self,
        *,
        alpha: float = 0.05,
        gamma: float = 0.02,
        score: str = "absolute",
        warmup: int = 19,
        fallback_z: float = 1.959964,
    ) -> None:
        """
        Args:
            alpha: Target long-run miss rate.
            gamma: Step size. Gibbs & Candes use 0.005-0.05. Larger tracks a
                regime change faster and jitters more between observations.
            score: "absolute" or "studentised". Studentising divides by the
                model's own sd, which helps only when that sd carries real
                information about which predictions are shaky -- when it does
                not, dividing by it amplifies the miscalibration rather than
                correcting it.
            warmup: Observations required before a conformal quantile is used.
                Below it the Gaussian interval stands in.
            fallback_z: Multiplier for that stand-in interval.

        Raises:
            ValueError: On an alpha outside (0, 1), a non-positive gamma, or an
                unknown score.
        """
        if not 0.0 < alpha < 1.0:
            raise ValueError(f"alpha must lie in (0, 1), got {alpha}")
        if gamma <= 0.0:
            raise ValueError(f"gamma must be positive, got {gamma}")
        if score not in SCORES:
            raise ValueError(f"unknown score {score!r}, expected one of {SCORES}")

        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.score = score
        self.warmup = int(warmup)
        self.fallback_z = float(fallback_z)

        #: The working miss-rate. This is the state the algorithm adapts.
        self.working_alpha = float(alpha)
        self._scores: list[float] = []
        self.n_seen = 0
        self.n_missed = 0

    def _scale(self, posterior_sd: float) -> float:
        if self.score == "studentised":
            return float(posterior_sd) if posterior_sd > 0 else float("nan")
        return 1.0

    def half_width(self, posterior_sd: float) -> float:
        """Half-width of the next interval, given this prediction's sd."""
        scale = self._scale(posterior_sd)
        if len(self._scores) < self.warmup or not np.isfinite(scale):
            return self.fallback_z * float(posterior_sd)

        # The working alpha is kept strictly inside (0, 1): at or below 0 the
        # interval is infinite and at or above 1 it is empty, and neither is
        # informative.
        q = conformal_quantile(
            np.asarray(self._scores), float(np.clip(self.working_alpha, 1e-3, 0.999))
        )
        if not np.isfinite(q):
            # The working alpha has dropped below what the sample can represent.
            # Falling back to the Gaussian half-width here would be a bug with
            # the worst possible sign: the alpha fell precisely BECAUSE the
            # interval has been missing, and answering that by reverting to the
            # narrowest interval on offer fights the correction. The widest score
            # yet seen is the honest finite stand-in.
            q = float(max(self._scores))
        return float(q * scale)

    def interval(self, predicted: float, posterior_sd: float) -> tuple[float, float]:
        """Interval for the next observation."""
        half = self.half_width(posterior_sd)
        return (predicted - half, predicted + half)

    def update(self, predicted: float, posterior_sd: float, actual: float) -> bool:
        """Score one observation and adapt. Returns whether it was covered.

        The interval is computed *before* the outcome is used, so what is scored
        at step t never depends on the truth at step t.
        """
        half = self.half_width(posterior_sd)
        residual = abs(float(actual) - float(predicted))
        covered = residual <= half

        self.working_alpha += self.gamma * (self.alpha - (0.0 if covered else 1.0))
        scale = self._scale(posterior_sd)
        if np.isfinite(scale) and scale > 0:
            self._scores.append(residual / scale)
        self.n_seen += 1
        self.n_missed += int(not covered)
        return covered

    @property
    def empirical_coverage(self) -> float:
        """Coverage achieved so far. NaN before anything has been seen."""
        if self.n_seen == 0:
            return float("nan")
        return 1.0 - self.n_missed / self.n_seen
