"""Synthetic practice sessions with a known hidden truth.

This module exists to answer the one question the literature does not ask.

Published F1 tyre models are validated on lap-time prediction error. But a model
can predict lap times almost perfectly while attributing the slowdown to entirely
the wrong cause -- fuel, traffic and degradation all push lap time in tidy
monotone directions, and many wrong decompositions add up to the same right
total. Prediction accuracy simply cannot detect that failure.

The only way to test attribution is to know the answer in advance. So we generate
sessions where the true degradation rate, the true track evolution curve, the
true fuel slope and the true traffic penalty are all *set by us*, bury them under
realistic confounding, and then measure how close an estimator gets to each one
separately.

Everything here is clearly synthetic and is never mixed with real telemetry.
A model that cannot recover a signal it was handed under controlled conditions
has no business being trusted on a real session -- and one that can has earned
a claim the lap-time-error literature cannot make.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

#: Baseline degradation in s/lap for each compound, before the cliff term.
#: Ordered softest-to-hardest: softer rubber degrades faster and cliffs sooner.
DEFAULT_COMPOUND_RATES: dict[str, float] = {
    "SOFT": 0.115,
    "MEDIUM": 0.072,
    "HARD": 0.041,
}

#: Tyre age at which degradation starts accelerating, per compound.
DEFAULT_CLIFF_ONSET: dict[str, float] = {
    "SOFT": 12.0,
    "MEDIUM": 20.0,
    "HARD": 30.0,
}


@dataclass(frozen=True)
class SessionConfig:
    """Controls for a synthetic session.

    Defaults describe a plausible 60-minute FP2: twenty cars, a handful of runs
    each, a track that improves by about 0.9 s as it rubbers in, and lap times
    scattered by a couple of tenths of driver noise.

    Attributes:
        n_drivers: Cars on track.
        session_slots: Length of the session in lap slots. A driver occupies one
            slot per lap they run, and is idle in the rest.
        runs_per_driver: Number of separate runs (tyre sets) each driver completes.
        min_run_laps: Shortest run length, in laps.
        max_run_laps: Longest run length, in laps.
        base_lap_time: Reference lap time in seconds.
        driver_pace_sd: Spread of true car/driver pace across the field, s.
        compound_rates: True degradation rate per compound, s/lap.
        cliff_onset: True tyre age at which degradation accelerates, per compound.
        cliff_severity: Quadratic coefficient on age past the cliff onset.
        fuel_slope: True lap-time gain per lap from fuel burn-off, s/lap.
        track_evolution_total: Total lap-time gain from track rubbering in, s
            (positive means the track gets faster).
        track_evolution_shape: Exponential rate of the rubbering-in curve. Larger
            means the track matures earlier in the session.
        traffic_probability: Probability that any given lap is compromised.
        traffic_coefficient: Lap time lost at a traffic index of 1.0, s.
        observation_noise_sd: Scale of per-lap driver noise, s.
        observation_noise_df: Student-t degrees of freedom for that noise. Low
            values produce the occasional lock-up or scruffy lap that real
            sessions contain and Gaussian models handle badly.
        scrubbed_set_probability: Probability a run starts on a used tyre set.
        seed: Random seed. The same seed always yields the same session.
    """

    n_drivers: int = 20
    session_slots: int = 60
    runs_per_driver: int = 3
    min_run_laps: int = 6
    max_run_laps: int = 16
    base_lap_time: float = 92.0
    driver_pace_sd: float = 0.55

    compound_rates: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_COMPOUND_RATES))
    cliff_onset: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_CLIFF_ONSET))
    cliff_severity: float = 0.004

    fuel_slope: float = 0.081
    track_evolution_total: float = 0.90
    track_evolution_shape: float = 0.055

    traffic_probability: float = 0.18
    traffic_coefficient: float = 1.30

    observation_noise_sd: float = 0.16
    observation_noise_df: float = 5.0

    scrubbed_set_probability: float = 0.25
    seed: int = 20260904


@dataclass(frozen=True)
class GroundTruth:
    """What actually generated the session. Never visible to any estimator.

    Attributes:
        compound_rates: True baseline degradation rate per compound, s/lap.
        cliff_onset: True cliff onset age per compound, laps.
        fuel_slope: True fuel burn-off slope, s/lap.
        traffic_coefficient: True traffic penalty at index 1.0, s.
        driver_pace: True car/driver pace offset per driver, s.
        track_evolution: Session lap to true track effect, s.
        lap_truth: Per-lap decomposition. One column per causal term, so that
            attribution error can be measured term by term rather than only in
            aggregate.
    """

    compound_rates: dict[str, float]
    cliff_onset: dict[str, float]
    fuel_slope: float
    traffic_coefficient: float
    driver_pace: dict[str, float]
    track_evolution: pd.DataFrame
    lap_truth: pd.DataFrame

    def true_rate_at(self, compound: str, tyre_age: float, cliff_severity: float) -> float:
        """Instantaneous true degradation rate, s/lap, at a given tyre age.

        The derivative of the cumulative loss curve: constant at
        `compound_rates[compound]` until the cliff onset, then rising linearly.
        """
        base = self.compound_rates[compound]
        past_cliff = max(0.0, tyre_age - self.cliff_onset[compound])
        return base + 2.0 * cliff_severity * past_cliff


@dataclass(frozen=True)
class SyntheticSession:
    """A generated session: what an estimator sees, and what really happened.

    Attributes:
        lap_table: The observable data, in the schema `fit_tyre_ssm` expects.
            Contains no ground-truth columns.
        truth: The hidden generating process.
        config: The configuration used, for reproducibility.
    """

    lap_table: pd.DataFrame
    truth: GroundTruth
    config: SessionConfig


def _true_tyre_loss(age: float, compound: str, cfg: SessionConfig) -> float:
    """Cumulative performance loss from the tyre at a given age, seconds.

    Linear degradation up to the cliff onset, then a quadratic term on top. The
    quadratic is what makes this a real test: an estimator that assumes linear
    degradation will fit the early laps and miss the cliff entirely, and we want
    that failure to be visible rather than assumed away.
    """
    base = cfg.compound_rates[compound] * age
    past_cliff = max(0.0, age - cfg.cliff_onset[compound])
    return base + cfg.cliff_severity * past_cliff**2


def _track_effect(session_lap: int, cfg: SessionConfig) -> float:
    """Track evolution at a given session lap, seconds (negative = faster).

    Saturating exponential: the track gains grip quickly at first as rubber goes
    down, then plateaus. Shares its monotone-in-lap-number shape with fuel
    burn-off, which is precisely why the two are hard to separate and why the
    model needs a physical prior on one of them.
    """
    progress = 1.0 - np.exp(-cfg.track_evolution_shape * session_lap)
    return -cfg.track_evolution_total * float(progress)


def generate_session(config: SessionConfig | None = None) -> SyntheticSession:
    """Generate one confounded practice session with a recorded ground truth.

    The observable lap time is built as

        lap_time = base + driver_pace
                 + tyre_loss(age, compound)      <- the thing we want back
                 - fuel_slope * lap_in_run       <- confounder, monotone in run
                 + track_effect(session_lap)     <- confounder, monotone in session
                 + traffic_coefficient * traffic <- confounder, sporadic
                 + heavy-tailed noise

    Runs are staggered across the field, which is what makes tyre age and session
    lap separable at all. Some runs start on scrubbed sets, so tyre age and
    laps-completed-this-run are not the same variable.

    Args:
        config: Session parameters. Defaults describe a typical FP2.

    Returns:
        A SyntheticSession holding the observable table and the hidden truth.
    """
    cfg = config or SessionConfig()
    rng = np.random.default_rng(cfg.seed)

    drivers = [f"CAR{i + 1:02d}" for i in range(cfg.n_drivers)]
    driver_pace = {d: float(rng.normal(0.0, cfg.driver_pace_sd)) for d in drivers}
    compounds = list(cfg.compound_rates)

    rows: list[dict] = []
    truth_rows: list[dict] = []
    run_counter = 0

    for driver in drivers:
        # Lay the driver's runs out across the session with idle gaps between
        # them, so that different cars are at different tyre ages on any given lap.
        cursor = int(rng.integers(0, max(1, cfg.session_slots // 6)))

        for _ in range(cfg.runs_per_driver):
            run_length = int(rng.integers(cfg.min_run_laps, cfg.max_run_laps + 1))
            if cursor + run_length >= cfg.session_slots:
                break

            run_counter += 1
            compound = str(rng.choice(compounds))

            # A scrubbed set arrives with laps already on it. This decouples tyre
            # age from laps-completed-this-run, which is one of the few things
            # that breaks the fuel/degradation collinearity from within a run.
            starting_age = (
                float(rng.integers(3, 10))
                if rng.random() < cfg.scrubbed_set_probability
                else 0.0
            )

            for lap_in_run in range(run_length):
                session_lap = cursor + lap_in_run
                tyre_age = starting_age + lap_in_run

                tyre_term = _true_tyre_loss(tyre_age, compound, cfg)
                fuel_term = -cfg.fuel_slope * lap_in_run
                track_term = _track_effect(session_lap, cfg)

                traffic_index = (
                    float(rng.beta(2.0, 3.0))
                    if rng.random() < cfg.traffic_probability
                    else 0.0
                )
                traffic_term = cfg.traffic_coefficient * traffic_index

                # Student-t noise: mostly tidy laps, occasionally a scruffy one.
                noise = float(
                    rng.standard_t(cfg.observation_noise_df) * cfg.observation_noise_sd
                )

                lap_time = (
                    cfg.base_lap_time
                    + driver_pace[driver]
                    + tyre_term
                    + fuel_term
                    + track_term
                    + traffic_term
                    + noise
                )

                rows.append(
                    {
                        "driver": driver,
                        "session_lap": session_lap,
                        "run_id": run_counter,
                        "tyre_age": float(tyre_age),
                        "lap_in_run": lap_in_run,
                        "lap_time": lap_time,
                        "compound": compound,
                        "traffic_index": traffic_index,
                    }
                )
                truth_rows.append(
                    {
                        "driver": driver,
                        "session_lap": session_lap,
                        "run_id": run_counter,
                        "compound": compound,
                        "tyre_age": float(tyre_age),
                        "true_tyre": tyre_term,
                        "true_fuel": fuel_term,
                        "true_track": track_term,
                        "true_traffic": traffic_term,
                        "true_driver": driver_pace[driver],
                        "true_noise": noise,
                        "true_rate": cfg.compound_rates[compound]
                        + 2.0 * cfg.cliff_severity * max(0.0, tyre_age - cfg.cliff_onset[compound]),
                    }
                )

            # Idle time in the garage before the next run.
            cursor += run_length + int(rng.integers(2, 6))

    if not rows:
        raise ValueError(
            "generated an empty session; session_slots is too small for the "
            "requested runs_per_driver and run length range"
        )

    lap_table = pd.DataFrame(rows).sort_values(["session_lap", "driver"]).reset_index(drop=True)
    lap_truth = pd.DataFrame(truth_rows).sort_values(["session_lap", "driver"]).reset_index(
        drop=True
    )

    session_laps = sorted(lap_table["session_lap"].unique().tolist())
    track_evolution = pd.DataFrame(
        {
            "session_lap": session_laps,
            "true_track_effect": [_track_effect(x, cfg) for x in session_laps],
        }
    )

    truth = GroundTruth(
        compound_rates=dict(cfg.compound_rates),
        cliff_onset=dict(cfg.cliff_onset),
        fuel_slope=cfg.fuel_slope,
        traffic_coefficient=cfg.traffic_coefficient,
        driver_pace=driver_pace,
        track_evolution=track_evolution,
        lap_truth=lap_truth,
    )
    return SyntheticSession(lap_table=lap_table, truth=truth, config=cfg)


def naive_degradation_estimate(lap_table: pd.DataFrame) -> dict[str, float]:
    """The estimate a reasonable person would make without any of this machinery.

    Regress lap time on tyre age within each run, then average the slopes by
    compound. This is the honest baseline -- it is what a lap-chart-and-a-ruler
    analysis gives you, and it is what the platform has to beat to justify itself.

    It is biased, and the direction is knowable in advance: fuel burn-off makes
    the car faster by ~0.081 s per lap of a run, which sits on top of the tyre's
    slowdown and pulls every slope towards zero. On a compound degrading at
    0.072 s/lap that bias is larger than the signal.

    Args:
        lap_table: Same schema as `generate_session` produces.

    Returns:
        Estimated degradation rate per compound, s/lap.
    """
    slopes: dict[str, list[float]] = {}

    for (_, compound), run in lap_table.groupby(["run_id", "compound"]):
        if len(run) < 3:
            continue
        age = run["tyre_age"].to_numpy(dtype=float)
        if np.ptp(age) == 0:
            continue
        slope = float(np.polyfit(age, run["lap_time"].to_numpy(dtype=float), 1)[0])
        slopes.setdefault(str(compound), []).append(slope)

    return {c: float(np.mean(v)) for c, v in slopes.items() if v}
