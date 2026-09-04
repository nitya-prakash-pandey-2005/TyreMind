"""Scoring the model ladder, without leaking the future into the past.

Two things are measured, and they are not the same thing.

**Lap-time prediction** is what every model can do and what most papers report.
It is also the easier and less interesting question: fuel dominates the signal,
so a model that gets fuel roughly right will look good regardless of whether it
understands tyres at all.

**Degradation-rate recovery** is what the product is for, and it can only be
scored where the true rate is known -- which means synthetic data. That is a
limitation worth stating plainly rather than working around: public F1 telemetry
contains no measured tyre wear, so there is no real-world ground truth to score
against. What real data *can* do is test whether a practice-derived rate predicts
the race, which is `models/validation.py`.

Splits are chronological throughout. A random split across laps would let a model
see lap 40 while predicting lap 20 of the same stint, and every metric would
improve for a reason that does not exist on a Sunday.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.stats import norm

from tyremind.models.baselines import DegradationModel


def crps_gaussian(y: np.ndarray, mean: np.ndarray, sd: np.ndarray) -> np.ndarray:
    """Continuous ranked probability score for a Gaussian forecast, per observation.

    CRPS scores the whole predictive distribution rather than just its centre, so
    a model cannot improve it by being confidently wrong. Lower is better, and it
    reduces to absolute error as the spread goes to zero -- which is exactly why
    it is the right headline metric for a tool whose intervals are the point.

    Closed form for a normal forecast:

        CRPS = sd * [ z(2*Phi(z) - 1) + 2*phi(z) - 1/sqrt(pi) ],  z = (y - mean)/sd
    """
    sd = np.maximum(sd, 1e-9)
    z = (y - mean) / sd
    return sd * (z * (2.0 * norm.cdf(z) - 1.0) + 2.0 * norm.pdf(z) - 1.0 / np.sqrt(np.pi))


@dataclass
class FoldScore:
    """Metrics for one held-out block."""

    fold: int
    n_train: int
    n_test: int
    mae: float
    rmse: float
    crps: float
    coverage_95: float
    interval_width_95: float
    #: Mean signed error. Positive means the model predicted slower than reality.
    #: Tracked per fold because its *drift* across folds is diagnostic: a model
    #: that cannot extrapolate the fuel trend gets steadily more wrong as the
    #: forecast reaches further past its training window, while one that encodes
    #: fuel physically stays flat.
    bias: float = float("nan")

    def to_dict(self) -> dict:
        return {
            "fold": self.fold,
            "n_train": self.n_train,
            "n_test": self.n_test,
            "mae": self.mae,
            "rmse": self.rmse,
            "crps": self.crps,
            "coverage_95": self.coverage_95,
            "interval_width_95": self.interval_width_95,
            "bias": self.bias,
        }


@dataclass
class ModelScore:
    """A model's performance across every fold."""

    model: str
    folds: list[FoldScore] = field(default_factory=list)
    failed: str | None = None
    fit_seconds: float = 0.0

    def _mean(self, attribute: str) -> float:
        if not self.folds:
            return float("nan")
        return float(np.mean([getattr(f, attribute) for f in self.folds]))

    @property
    def mae(self) -> float:
        return self._mean("mae")

    @property
    def rmse(self) -> float:
        return self._mean("rmse")

    @property
    def crps(self) -> float:
        return self._mean("crps")

    @property
    def coverage_95(self) -> float:
        return self._mean("coverage_95")

    @property
    def interval_width_95(self) -> float:
        return self._mean("interval_width_95")

    @property
    def bias(self) -> float:
        return self._mean("bias")

    @property
    def bias_drift(self) -> float:
        """How much the bias grows from the first fold to the last, seconds.

        The single clearest indicator of whether a model can extrapolate. Each
        successive fold forecasts further past its training window, so a model
        that has learned the fuel trend as a pattern rather than as physics gets
        progressively more wrong, and this number says by how much.
        """
        if len(self.folds) < 2:
            return float("nan")
        return float(self.folds[-1].bias - self.folds[0].bias)

    def to_dict(self) -> dict:
        return {
            "model": self.model,
            "failed": self.failed,
            "fit_seconds": self.fit_seconds,
            "mae": self.mae,
            "rmse": self.rmse,
            "crps": self.crps,
            "coverage_95": self.coverage_95,
            "interval_width_95": self.interval_width_95,
            "bias": self.bias,
            "bias_drift": self.bias_drift,
            "folds": [f.to_dict() for f in self.folds],
        }


def rolling_origin_folds(
    lap_table: pd.DataFrame, n_folds: int = 4, min_train_fraction: float = 0.4
) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    """Expanding-window splits by session lap.

    Each fold trains on everything up to a cut-off and tests on the block that
    follows. The training window grows; the future is never visible. This is the
    setting the model actually operates in -- at lap 30 you have laps 1 to 29 and
    nothing else.

    Args:
        lap_table: A session.
        n_folds: Number of test blocks.
        min_train_fraction: Fraction of the session reserved for the first
            training window. Too small and the first fold has too few runs to
            estimate anything, which measures nothing useful.

    Returns:
        ``(train, test)`` pairs. Folds whose test block is empty are dropped.
    """
    laps = np.sort(lap_table["session_lap"].unique())
    if len(laps) < n_folds + 2:
        return []

    start = int(len(laps) * min_train_fraction)
    edges = np.linspace(start, len(laps), n_folds + 1).astype(int)

    folds = []
    for i in range(n_folds):
        cut, end = laps[edges[i] - 1], laps[min(edges[i + 1] - 1, len(laps) - 1)]
        train = lap_table[lap_table["session_lap"] <= cut]
        test = lap_table[(lap_table["session_lap"] > cut) & (lap_table["session_lap"] <= end)]
        if len(test) >= 5 and len(train) >= 20:
            folds.append((train, test))
    return folds


def score_fold(
    model: DegradationModel, train: pd.DataFrame, test: pd.DataFrame, fold: int
) -> FoldScore:
    """Fit on the training block and score the held-out one."""
    model.fit(train)
    mean, sd = model.predict(test)
    y = test["lap_time"].to_numpy(dtype=float)

    error = y - mean
    lower, upper = mean - 1.96 * sd, mean + 1.96 * sd

    return FoldScore(
        fold=fold,
        n_train=len(train),
        n_test=len(test),
        mae=float(np.abs(error).mean()),
        rmse=float(np.sqrt((error**2).mean())),
        crps=float(crps_gaussian(y, mean, sd).mean()),
        coverage_95=float(((y >= lower) & (y <= upper)).mean()),
        interval_width_95=float((upper - lower).mean()),
        bias=float((mean - y).mean()),
    )


def evaluate_ladder(
    lap_table: pd.DataFrame,
    models: list[DegradationModel],
    *,
    n_folds: int = 4,
) -> list[ModelScore]:
    """Run every model through the same chronological folds.

    A model that raises is recorded as failed rather than crashing the sweep --
    a benchmark where one bad rung hides all the others is not much of a
    benchmark.

    Args:
        lap_table: A session.
        models: Models to score. Each is refit per fold.
        n_folds: Number of held-out blocks.

    Returns:
        One ModelScore per model, in the order given.

    Raises:
        ValueError: If the session is too short to split at all.
    """
    import time

    folds = rolling_origin_folds(lap_table, n_folds=n_folds)
    if not folds:
        raise ValueError(
            f"session has only {lap_table['session_lap'].nunique()} distinct laps, "
            "too few for a chronological split"
        )

    scores = []
    for model in models:
        score = ModelScore(model=model.name)
        started = time.perf_counter()
        try:
            for i, (train, test) in enumerate(folds):
                score.folds.append(score_fold(model, train, test, i))
        except Exception as exc:  # noqa: BLE001
            score.failed = f"{type(exc).__name__}: {exc}"
        score.fit_seconds = time.perf_counter() - started
        scores.append(score)

    return scores


def score_rate_recovery(
    models: list[DegradationModel],
    lap_table: pd.DataFrame,
    true_rates: dict[str, float],
) -> pd.DataFrame:
    """Score how close each model gets to a known degradation rate.

    Only meaningful on synthetic data, where the truth was set rather than
    inferred. Models with no degradation parameter are reported with a null
    estimate -- which is the finding, not a gap in the table.

    Args:
        models: Models to score. Each is fitted on the full lap table.
        lap_table: The session.
        true_rates: True degradation rate per compound, s/lap.

    Returns:
        One row per model and compound, with the estimate, its error, and
        whether the 95% interval covered the truth.
    """
    rows = []
    for model in models:
        try:
            model.fit(lap_table)
            estimates = model.compound_rates()
        except Exception as exc:  # noqa: BLE001
            rows.append({"model": model.name, "compound": None, "failed": str(exc)})
            continue

        if not estimates:
            rows.append(
                {
                    "model": model.name,
                    "compound": None,
                    "estimate": None,
                    "error": None,
                    "note": "no degradation parameter to report",
                }
            )
            continue

        for compound, truth in true_rates.items():
            mean, sd = estimates.get(compound, (np.nan, np.nan))
            covered = (
                bool(abs(mean - truth) <= 1.96 * sd)
                if np.isfinite(mean) and np.isfinite(sd)
                else None
            )
            rows.append(
                {
                    "model": model.name,
                    "compound": compound,
                    "true_rate": truth,
                    "estimate": float(mean) if np.isfinite(mean) else None,
                    "estimate_sd": float(sd) if np.isfinite(sd) else None,
                    "error": float(mean - truth) if np.isfinite(mean) else None,
                    "covered_95": covered,
                }
            )

    return pd.DataFrame(rows)


def ladder_table(scores: list[ModelScore]) -> pd.DataFrame:
    """Fold the scores into the benchmark table, best CRPS first."""
    frame = pd.DataFrame([s.to_dict() for s in scores]).drop(columns=["folds"])
    return frame.sort_values("crps", na_position="last").reset_index(drop=True)
