"""The model ladder: everything TyreMind has to beat, and why each rung exists.

A state-space model is more complicated than a regression. That complexity has to
earn its place against simpler alternatives on the same data with the same
validation, or it is just expensive. So each rung here is a real method someone
would reasonably use, not a straw man:

  1. `NaiveAgeModel`      lap time against tyre age. What a lap chart and a ruler
                          give you, and still the most common approach.
  2. `FuelCorrectedModel` the same, after subtracting the textbook fuel
                          correction. This is the actual paddock standard, and
                          the honest baseline to beat.
  3. `PooledRegression`   least squares with driver, compound, traffic and a
                          session trend. Everything the SSM has, minus the
                          latent state.
  4. `GradientBoosted`    LightGBM on engineered features. Represents "throw ML
                          at it", and will predict lap times well.
  5. `TyreStateModel`     the state-space model.

Rung 4 is the interesting comparison. Gradient boosting will likely predict lap
times *better* than the SSM, because it can exploit any pattern in the data
including ones with no causal reading. What it cannot do is answer the question
the product is for -- it has no parameter that means "degradation rate", so
there is nothing to report and nothing to carry to Sunday. Documenting that is
more useful than pretending the SSM wins everywhere.

Every model exposes the same two things: a predictive distribution over lap
times, and a per-compound degradation rate where one is even definable.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

from tyremind.models.ssm.tyre_ssm import TyreSSMPriors, fit_tyre_ssm

#: Physical fuel correction: 0.030 s/kg x 2.7 kg/lap. Same value the SSM uses as
#: a prior, so rungs 2 and 5 differ in method rather than in assumption.
FUEL_SLOPE_S_PER_LAP = 0.081


class DegradationModel(ABC):
    """Common interface, so the evaluation harness never special-cases a model."""

    name: str = "model"

    @abstractmethod
    def fit(self, lap_table: pd.DataFrame) -> DegradationModel:
        """Fit to a lap table. Returns self."""

    @abstractmethod
    def predict(self, lap_table: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Predictive mean and standard deviation of lap time, seconds.

        The standard deviation is required, not optional. A model that cannot say
        how sure it is cannot be scored on calibration, and calibration is what
        separates a usable strategy input from a plausible-looking one.
        """

    def compound_rates(self) -> dict[str, tuple[float, float]]:
        """Degradation rate per compound, s/lap, with a standard deviation.

        Empty when the model has no such parameter. That is a meaningful answer,
        not a failure -- see the note about gradient boosting in the module
        docstring.
        """
        return {}


def _design(lap_table: pd.DataFrame, origin: float | None = None) -> pd.DataFrame:
    """Features shared by the regression rungs.

    Args:
        lap_table: Laps to build features for.
        origin: Session lap that `session_progress` is measured from. This MUST
            be the value captured at fit time, not recomputed from whatever frame
            is being transformed.

            Recomputing it per frame is a subtle and destructive bug: in a
            chronological split the test block starts later than the training
            block, so a locally-derived origin makes lap 60 look like lap 0 and
            the session-trend coefficient gets applied to the wrong value. It
            does not raise, it does not leak the future -- it just makes the
            predictions wrong, by tens of seconds in a race.
    """
    df = lap_table.copy()
    df["traffic_index"] = df.get("traffic_index", pd.Series(0.0, index=df.index)).fillna(0.0)
    start = float(df["session_lap"].min()) if origin is None else float(origin)
    df["session_progress"] = df["session_lap"].astype(float) - start
    return df


class NaiveAgeModel(DegradationModel):
    """Lap time regressed on tyre age, per compound.

    The estimate a reasonable person makes without any of this machinery, and the
    one that is wrong in a knowable direction: fuel burn-off pulls every slope
    towards zero by about 0.081 s/lap, which on a medium compound is larger than
    the effect being measured. On real race data it routinely produces *negative*
    degradation -- tyres apparently getting faster with age.
    """

    name = "Naive (lap time vs tyre age)"

    def __init__(self) -> None:
        self._slopes: dict[str, float] = {}
        self._intercepts: dict[str, float] = {}
        self._sd: float = 1.0
        self._rate_sd: dict[str, float] = {}

    def fit(self, lap_table: pd.DataFrame) -> NaiveAgeModel:
        residuals = []
        for compound, group in lap_table.groupby("compound"):
            age = group["tyre_age"].to_numpy(dtype=float)
            y = group["lap_time"].to_numpy(dtype=float)
            if len(group) < 3 or np.ptp(age) == 0:
                self._slopes[str(compound)] = 0.0
                self._intercepts[str(compound)] = float(y.mean())
                self._rate_sd[str(compound)] = float("nan")
                continue
            slope, intercept = np.polyfit(age, y, 1)
            self._slopes[str(compound)] = float(slope)
            self._intercepts[str(compound)] = float(intercept)

            fitted = slope * age + intercept
            residuals.append(y - fitted)
            # Textbook standard error of an OLS slope.
            dof = max(len(age) - 2, 1)
            se = np.sqrt(((y - fitted) ** 2).sum() / dof / ((age - age.mean()) ** 2).sum())
            self._rate_sd[str(compound)] = float(se)

        self._sd = float(np.concatenate(residuals).std()) if residuals else 1.0
        return self

    def predict(self, lap_table: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        mean = np.array(
            [
                self._slopes.get(str(r.compound), 0.0) * float(r.tyre_age)
                + self._intercepts.get(str(r.compound), float("nan"))
                for r in lap_table.itertuples(index=False)
            ]
        )
        fallback = np.nanmedian(list(self._intercepts.values())) if self._intercepts else 0.0
        return np.nan_to_num(mean, nan=fallback), np.full(len(lap_table), max(self._sd, 1e-3))

    def compound_rates(self) -> dict[str, tuple[float, float]]:
        return {c: (s, self._rate_sd.get(c, float("nan"))) for c, s in self._slopes.items()}


class FuelCorrectedModel(NaiveAgeModel):
    """The paddock standard: subtract the fuel effect, then read the slope.

    Corrects lap times by the known physical fuel rate before regressing on tyre
    age. This removes the naive model's dominant bias and is a genuinely
    reasonable method -- it is what fuel-corrected lap charts have done for years.

    What it still cannot do is separate track evolution from degradation, or
    account for traffic, or pool across the field. Those are what the remaining
    rungs add.
    """

    name = "Fuel-corrected regression"

    def fit(self, lap_table: pd.DataFrame) -> FuelCorrectedModel:
        corrected = lap_table.copy()
        corrected["lap_time"] = corrected["lap_time"] + FUEL_SLOPE_S_PER_LAP * corrected[
            "lap_in_run"
        ].astype(float)
        super().fit(corrected)
        return self

    def predict(self, lap_table: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        mean, sd = super().predict(lap_table)
        # Undo the correction so predictions are on the observed scale.
        return mean - FUEL_SLOPE_S_PER_LAP * lap_table["lap_in_run"].to_numpy(dtype=float), sd


class PooledRegression(DegradationModel):
    """Least squares with everything the SSM has, except the latent state.

    Driver effects, compound-specific age slopes, traffic, a session trend and
    the fuel correction, all fitted jointly. This is the strongest thing you can
    build without a state-space formulation, and it isolates what the latent
    state actually buys: the ability for the degradation rate to *change* within
    a stint, which is what a cliff is.
    """

    name = "Pooled regression"

    #: Ridge strength. Small enough not to bias the effects meaningfully on a
    #: full session, large enough to stop a rank-deficient fold from producing a
    #: coefficient that explodes on extrapolation.
    RIDGE = 1.0

    def __init__(self) -> None:
        self._coef: np.ndarray | None = None
        self._columns: list[str] = []
        self._sd = 1.0
        self._compounds: list[str] = []
        self._drivers: list[str] = []
        self._rate_sd: dict[str, float] = {}
        self._origin: float | None = None

    def _matrix(self, df: pd.DataFrame) -> np.ndarray:
        parts = [np.ones((len(df), 1))]
        for driver in self._drivers[1:]:  # first driver absorbed by the intercept
            parts.append((df["driver"] == driver).to_numpy(dtype=float)[:, None])
        for compound in self._compounds:
            is_c = (df["compound"] == compound).to_numpy(dtype=float)
            parts.append(is_c[:, None])
            parts.append((is_c * df["tyre_age"].to_numpy(dtype=float))[:, None])
        parts.append(df["traffic_index"].to_numpy(dtype=float)[:, None])
        parts.append(df["session_progress"].to_numpy(dtype=float)[:, None])
        parts.append(-df["lap_in_run"].to_numpy(dtype=float)[:, None])
        return np.hstack(parts)

    def fit(self, lap_table: pd.DataFrame) -> PooledRegression:
        self._origin = float(lap_table["session_lap"].min())
        df = _design(lap_table, self._origin)
        self._drivers = sorted(df["driver"].unique().tolist())
        self._compounds = sorted(df["compound"].unique().tolist())

        X = self._matrix(df)
        y = df["lap_time"].to_numpy(dtype=float)

        # Ridge rather than plain least squares. Within a single fold a compound
        # can appear on only a couple of laps at nearly one tyre age, leaving its
        # age slope all but unidentifiable; ordinary least squares then returns a
        # huge coefficient that is harmless in-sample and catastrophic when
        # extrapolated. Left unregularised this produced a 176-second fold on
        # Silverstone while every other fold sat under a second.
        #
        # A competent practitioner would regularise here, so the benchmark does
        # too -- beating a brittle competitor would prove nothing. The intercept
        # is left unpenalised so the ridge shrinks effects, not the mean lap time.
        penalty = np.sqrt(self.RIDGE) * np.eye(X.shape[1])
        penalty[0, 0] = 0.0
        X_augmented = np.vstack([X, penalty])
        y_augmented = np.concatenate([y, np.zeros(X.shape[1])])
        self._coef, *_ = np.linalg.lstsq(X_augmented, y_augmented, rcond=None)

        residual = y - X @ self._coef
        dof = max(len(y) - X.shape[1], 1)
        sigma2 = float((residual**2).sum() / dof)
        self._sd = float(np.sqrt(sigma2))

        # Standard errors from the usual sigma^2 (X'X)^-1.
        try:
            cov = sigma2 * np.linalg.pinv(X.T @ X + self.RIDGE * np.eye(X.shape[1]))
            for i, compound in enumerate(self._compounds):
                slope_col = 1 + (len(self._drivers) - 1) + 2 * i + 1
                self._rate_sd[compound] = float(np.sqrt(max(cov[slope_col, slope_col], 0.0)))
        except np.linalg.LinAlgError:
            self._rate_sd = dict.fromkeys(self._compounds, float("nan"))

        return self

    def predict(self, lap_table: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        if self._coef is None:
            raise RuntimeError("fit() must be called before predict()")
        df = _design(lap_table, self._origin)
        # Unseen drivers fall back to the reference level rather than raising --
        # a car that did not run in training is a normal event, not an error.
        df = df.assign(driver=df["driver"].where(df["driver"].isin(self._drivers), self._drivers[0]))
        df = df.assign(
            compound=df["compound"].where(df["compound"].isin(self._compounds), self._compounds[0])
        )
        return self._matrix(df) @ self._coef, np.full(len(df), max(self._sd, 1e-3))

    def compound_rates(self) -> dict[str, tuple[float, float]]:
        if self._coef is None:
            return {}
        out = {}
        for i, compound in enumerate(self._compounds):
            slope_col = 1 + (len(self._drivers) - 1) + 2 * i + 1
            out[compound] = (float(self._coef[slope_col]), self._rate_sd.get(compound, float("nan")))
        return out


class GradientBoosted(DegradationModel):
    """LightGBM on engineered features.

    Included to make a specific point rather than to win. Boosting will predict
    lap times well -- probably better than the state-space model -- because it can
    exploit any regularity in the data, including ones with no causal reading.

    But `compound_rates` returns nothing, and that is the entire issue. There is
    no parameter here that means "degradation rate", so there is nothing to
    report to an engineer, nothing to carry from practice to the race, and no way
    to say how much of a slowdown was the tyre. A better lap-time predictor that
    cannot answer the question is not a better tyre model.

    Prediction intervals come from quantile regression at the 16th and 84th
    percentiles -- roughly one standard deviation for a symmetric distribution --
    because point-predicting boosters have no native notion of spread.
    """

    name = "LightGBM"

    FEATURES = [
        "tyre_age",
        "lap_in_run",
        "session_progress",
        "traffic_index",
        "compound_code",
        "driver_code",
    ]

    #: Driver and compound are genuinely categorical, and letting LightGBM know
    #: that matters -- as plain integers it would learn spurious orderings
    #: ("driver 7 is between driver 6 and driver 8"). Configured properly so the
    #: comparison is against a well-set-up competitor rather than a strawman.
    CATEGORICAL = ["compound_code", "driver_code"]

    def __init__(self, **params) -> None:
        self._params = {
            "n_estimators": 200,
            "learning_rate": 0.05,
            "num_leaves": 15,
            "min_child_samples": 20,
            "subsample": 0.9,
            "subsample_freq": 1,
            "colsample_bytree": 0.9,
            "reg_lambda": 1.0,
            "verbose": -1,
            **params,
        }
        self._mean = None
        self._lower = None
        self._upper = None
        self._compound_codes: dict[str, int] = {}
        self._driver_codes: dict[str, int] = {}
        self._origin: float | None = None

    def _features(self, lap_table: pd.DataFrame) -> pd.DataFrame:
        df = _design(lap_table, self._origin)
        df["compound_code"] = df["compound"].map(self._compound_codes).fillna(-1).astype(int)
        df["driver_code"] = df["driver"].map(self._driver_codes).fillna(-1).astype(int)
        out = df[self.FEATURES].copy()
        for column in self.CATEGORICAL:
            out[column] = out[column].astype("category")
        return out

    def fit(self, lap_table: pd.DataFrame) -> GradientBoosted:
        import lightgbm as lgb

        self._origin = float(lap_table["session_lap"].min())
        self._compound_codes = {
            c: i for i, c in enumerate(sorted(lap_table["compound"].unique()))
        }
        self._driver_codes = {d: i for i, d in enumerate(sorted(lap_table["driver"].unique()))}

        X = self._features(lap_table)
        y = lap_table["lap_time"].to_numpy(dtype=float)

        self._mean = lgb.LGBMRegressor(**self._params).fit(X, y, categorical_feature=self.CATEGORICAL)
        self._lower = lgb.LGBMRegressor(objective="quantile", alpha=0.16, **self._params).fit(
            X, y, categorical_feature=self.CATEGORICAL
        )
        self._upper = lgb.LGBMRegressor(objective="quantile", alpha=0.84, **self._params).fit(
            X, y, categorical_feature=self.CATEGORICAL
        )
        return self

    def predict(self, lap_table: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        if self._mean is None:
            raise RuntimeError("fit() must be called before predict()")
        X = self._features(lap_table)
        mean = self._mean.predict(X)
        spread = (self._upper.predict(X) - self._lower.predict(X)) / 2.0
        return mean, np.maximum(spread, 1e-3)


class NeuralNetwork(DegradationModel):
    """A multi-layer perceptron on the same engineered features.

    Included because "have you tried deep learning" is the first question any
    modelling choice has to answer, and answering it with a measurement beats
    answering it with an opinion.

    A scikit-learn MLP rather than a torch model: on a few hundred laps with six
    features, network *depth* is not the binding constraint -- data is. A larger
    architecture would change the answer only by overfitting harder, and it would
    add a two-gigabyte dependency to a project whose demo has to run offline.

    Like gradient boosting, it has no parameter meaning "degradation rate", so
    `compound_rates` stays empty. Prediction intervals come from the ensemble
    spread across differently-seeded networks, which captures model uncertainty
    but not observation noise -- so its intervals are expected to be too narrow,
    and the benchmark measures how much.
    """

    name = "Neural network (MLP)"

    FEATURES = GradientBoosted.FEATURES

    def __init__(self, n_models: int = 5, **params) -> None:
        # Chosen by comparing five configurations on held-out folds, because
        # beating a badly-tuned network would prove nothing. Early stopping is
        # OFF deliberately: it holds out 10% for validation, which on a few
        # hundred laps is ~30 rows, and stopping on that is noise -- it cost
        # roughly a full second of CRPS. Strong L2 (alpha=10) does the
        # regularising instead.
        self._params = {
            "hidden_layer_sizes": (48, 24),
            "activation": "relu",
            "max_iter": 2000,
            "early_stopping": False,
            "alpha": 10.0,
            **params,
        }
        self._n_models = n_models
        self._models: list = []
        self._scaler = None
        self._compound_codes: dict[str, int] = {}
        self._driver_codes: dict[str, int] = {}
        self._origin: float | None = None
        self._residual_sd = 1.0

    def _features(self, lap_table: pd.DataFrame) -> np.ndarray:
        df = _design(lap_table, self._origin)
        df["compound_code"] = df["compound"].map(self._compound_codes).fillna(-1).astype(float)
        df["driver_code"] = df["driver"].map(self._driver_codes).fillna(-1).astype(float)
        return df[self.FEATURES].to_numpy(dtype=float)

    def fit(self, lap_table: pd.DataFrame) -> NeuralNetwork:
        from sklearn.neural_network import MLPRegressor
        from sklearn.preprocessing import StandardScaler

        self._origin = float(lap_table["session_lap"].min())
        self._compound_codes = {
            c: i for i, c in enumerate(sorted(lap_table["compound"].unique()))
        }
        self._driver_codes = {d: i for i, d in enumerate(sorted(lap_table["driver"].unique()))}

        X = self._features(lap_table)
        y = lap_table["lap_time"].to_numpy(dtype=float)

        # Scaling is not optional for an MLP: lap times are ~90 and tyre age ~20,
        # and without it the optimiser spends its budget on the scale difference.
        self._scaler = StandardScaler().fit(X)
        Xs = self._scaler.transform(X)

        self._models = [
            MLPRegressor(random_state=seed, **self._params).fit(Xs, y)
            for seed in range(self._n_models)
        ]

        predictions = np.mean([m.predict(Xs) for m in self._models], axis=0)
        self._residual_sd = float(np.std(y - predictions))
        return self

    def predict(self, lap_table: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        if not self._models:
            raise RuntimeError("fit() must be called before predict()")
        Xs = self._scaler.transform(self._features(lap_table))
        stacked = np.array([m.predict(Xs) for m in self._models])
        mean = stacked.mean(axis=0)
        # Ensemble disagreement plus in-sample residual scatter. The first is
        # epistemic, the second stands in for aleatoric; neither is a posterior.
        spread = np.sqrt(stacked.var(axis=0) + self._residual_sd**2)
        return mean, np.maximum(spread, 1e-3)


class TyreStateModel(DegradationModel):
    """The TyreMind state-space model, wrapped for the benchmark harness."""

    name = "TyreMind state-space"

    def __init__(self, priors: TyreSSMPriors | None = None) -> None:
        self._priors = priors
        self._fit = None

    def fit(self, lap_table: pd.DataFrame) -> TyreStateModel:
        self._fit = fit_tyre_ssm(lap_table, priors=self._priors)
        return self

    def predict(self, lap_table: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Predict lap times, projecting the latent state forward where needed.

        For laps inside the fitted session this reads the smoothed state. For
        laps beyond it -- the held-out fold in a rolling-origin split -- the tyre
        state is projected forward from the last observed lap of that car's run,
        with variance growing accordingly. That is a genuine forecast, not a
        lookup.
        """
        if self._fit is None:
            raise RuntimeError("fit() must be called before predict()")

        fit = self._fit
        idx = fit.index
        smooth = fit.smoothed.a_smooth
        std = fit.smoothed.std()
        last_step = len(smooth) - 1

        seen_laps = fit.design.lap_table
        session_start = float(seen_laps["session_lap"].min())

        from tyremind.models.ssm.tyre_ssm import track_basis

        # Latest observed tyre age per (driver, run), so a forecast knows how far
        # forward it is reaching.
        latest_age = (
            seen_laps.groupby(["driver", "run_id"])["tyre_age"].max().to_dict()
        )

        means, sds = [], []
        for row in _design(lap_table, session_start).itertuples(index=False):
            driver = str(row.driver)
            if driver not in idx.level:
                means.append(float(seen_laps["lap_time"].median()))
                sds.append(float(seen_laps["lap_time"].std()))
                continue

            step = fit.design.step_of_lap.get(int(row.session_lap), last_step)

            level = float(smooth[step, idx.level[driver]])
            rate = float(smooth[step, idx.rate[driver]])
            level_sd = float(std[step, idx.level[driver]])

            # Project forward if this lap is beyond what the fit saw.
            known_age = latest_age.get((driver, int(row.run_id)))
            if known_age is not None and float(row.tyre_age) > known_age:
                ahead = float(row.tyre_age) - known_age
                level += ahead * rate
                level_sd = float(
                    np.sqrt(
                        level_sd**2
                        + ahead**2 * float(std[step, idx.rate[driver]]) ** 2
                        + ahead * fit.hyper.q_level
                        + (ahead**3 / 3.0) * fit.hyper.q_rate
                    )
                )

            run_state = idx.run_intercept.get(int(row.run_id))
            intercept = float(smooth[step, run_state]) if run_state is not None else 0.0

            basis = float(
                track_basis(
                    np.array([float(row.session_lap) - session_start]), fit.hyper.track_shape
                )[0]
            )
            track = basis * float(smooth[step, idx.track_amplitude]) + float(
                smooth[step, idx.track_resid]
            )
            fuel = -float(smooth[step, idx.fuel_slope]) * float(row.lap_in_run)
            traffic = float(smooth[step, idx.traffic_coef]) * float(row.traffic_index)

            means.append(fit.design.reference_time + intercept + track + level + fuel + traffic)
            sds.append(float(np.sqrt(level_sd**2 + fit.hyper.obs_var)))

        return np.array(means), np.maximum(np.array(sds), 1e-3)

    def compound_rates(self) -> dict[str, tuple[float, float]]:
        return self._fit.compound_rates() if self._fit else {}


def model_ladder() -> list[DegradationModel]:
    """Every rung, in increasing order of what it accounts for."""
    return [
        NaiveAgeModel(),
        FuelCorrectedModel(),
        PooledRegression(),
        GradientBoosted(),
        NeuralNetwork(),
        TyreStateModel(),
    ]
