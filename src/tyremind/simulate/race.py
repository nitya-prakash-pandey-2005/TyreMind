"""Monte Carlo race simulation and pit-strategy decisions.

Turns a degradation estimate into the decision it exists to support: stay out, or
box, and when.

The simulator is deliberately modest. Race simulation is the best-covered ground
in this literature -- Heilmeier et al. built the reference open-source simulator,
and Pitwall (arXiv:2607.06495) runs a calibrated one live at Grands Prix. There is
no point re-deriving that work, and claiming novelty here would be wrong.

What is different is the *input*. Every published simulator takes a degradation
rate as given, usually a single fitted slope with no uncertainty attached. Here
the degradation rate arrives as a posterior -- a distribution whose width came
from a stated identifiability argument -- and that width propagates all the way
to the finishing-position spread. A strategy that is only better under a
confident degradation estimate stops looking better once the estimate is honest,
and that is exactly the case a pit wall needs flagged.

Every number this module produces is a model estimate about a race that has not
happened. The UI labels them so.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

#: Time lost driving the pit lane and stopping, seconds. Circuit-specific; measured
#: from pit in/out laps where available, otherwise this prior.
DEFAULT_PIT_LOSS_S = 21.0

#: Seconds a following car loses per lap when stuck behind another. Dirty air
#: costs a following car roughly 15-20% of its downforce within about a second.
DEFAULT_DIRTY_AIR_S = 0.45

#: Per-lap probability of a safety car. Roughly one per two races over a
#: 55-lap distance.
DEFAULT_SAFETY_CAR_RATE = 0.009

#: Fraction of the usual pit loss paid when stopping under a safety car -- the
#: field is slowed, so the stop costs far less. This is why safety cars decide races.
SAFETY_CAR_PIT_DISCOUNT = 0.45


@dataclass(frozen=True)
class TyreModel:
    """Degradation for one compound, as a distribution rather than a number.

    Attributes:
        compound: Compound label.
        base_pace_s: Lap time on a fresh set relative to the reference compound, s.
            Negative is faster.
        degradation_rate: Mean degradation, s/lap.
        degradation_rate_sd: Posterior standard deviation of that rate. This is
            the number that makes the whole simulation honest -- it is sampled
            per race, not averaged away.
        cliff_lap: Tyre age beyond which degradation accelerates.
        cliff_severity: Quadratic growth in loss past the cliff.
    """

    compound: str
    base_pace_s: float
    degradation_rate: float
    degradation_rate_sd: float
    cliff_lap: float = 25.0
    cliff_severity: float = 0.004

    def loss(self, age: np.ndarray, rate: np.ndarray) -> np.ndarray:
        """Cumulative performance loss at a given tyre age, seconds.

        Args:
            age: Tyre age, shape (n_sims,) or scalar-broadcastable.
            rate: Sampled degradation rate per simulation, shape (n_sims,).

        Returns:
            Loss in seconds, same shape.
        """
        past_cliff = np.maximum(0.0, age - self.cliff_lap)
        return rate * age + self.cliff_severity * past_cliff**2


@dataclass(frozen=True)
class RaceState:
    """The situation a decision is being made from.

    Attributes:
        current_lap: Lap just completed.
        total_laps: Race distance.
        position: Current track position.
        current_compound: Compound in use.
        current_tyre_age: Laps on the current set.
        gap_ahead_s: Gap to the car ahead, seconds. Determines whether the car is
            in traffic now.
        gap_behind_s: Gap to the car behind, seconds. Determines whether a pit
            stop loses track position.
        base_lap_time_s: Reference lap time on a fresh set with no degradation.
        pit_loss_s: Time lost for a pit stop at this circuit.
    """

    current_lap: int
    total_laps: int
    position: int
    current_compound: str
    current_tyre_age: float
    gap_ahead_s: float
    gap_behind_s: float
    base_lap_time_s: float
    pit_loss_s: float = DEFAULT_PIT_LOSS_S

    @property
    def laps_remaining(self) -> int:
        return max(self.total_laps - self.current_lap, 0)


@dataclass(frozen=True)
class Strategy:
    """A candidate plan.

    Attributes:
        label: Human-readable name.
        pit_lap: Lap to pit on. None means run to the end on the current set.
        new_compound: Compound to fit. Ignored when `pit_lap` is None.
    """

    label: str
    pit_lap: int | None
    new_compound: str | None = None


@dataclass
class StrategyOutcome:
    """Simulated distribution of results for one strategy.

    Attributes:
        strategy: The plan simulated.
        race_times: Total remaining race time per simulation, seconds.
        n_sims: Simulations run.
        ran_out_of_tyre: Fraction of simulations where the tyre went past its
            cliff before the end. Not a failure, but worth surfacing -- a
            strategy that is fast on average by running a tyre into the ground
            carries a risk the mean does not show.
    """

    strategy: Strategy
    race_times: np.ndarray
    n_sims: int
    ran_out_of_tyre: float = 0.0

    @property
    def expected_time(self) -> float:
        return float(self.race_times.mean())

    @property
    def time_sd(self) -> float:
        return float(self.race_times.std())

    def quantile(self, q: float) -> float:
        return float(np.quantile(self.race_times, q))

    @property
    def downside(self) -> float:
        """The bad case: 90th percentile race time, seconds.

        A strategy is not just its average. A plan that is 0.5 s better on
        average but two seconds worse when it goes wrong is a different
        proposition, and this is the number that says so.
        """
        return self.quantile(0.9)

    def to_dict(self) -> dict:
        return {
            "label": self.strategy.label,
            "pit_lap": self.strategy.pit_lap,
            "new_compound": self.strategy.new_compound,
            "expected_time": self.expected_time,
            "time_sd": self.time_sd,
            "best_case": self.quantile(0.1),
            "downside": self.downside,
            "ran_out_of_tyre": self.ran_out_of_tyre,
            "n_sims": self.n_sims,
        }


def simulate_strategy(
    state: RaceState,
    strategy: Strategy,
    tyres: dict[str, TyreModel],
    *,
    n_sims: int = 5000,
    lap_time_noise_s: float = 0.25,
    dirty_air_s: float = DEFAULT_DIRTY_AIR_S,
    safety_car_rate: float = DEFAULT_SAFETY_CAR_RATE,
    seed: int = 0,
) -> StrategyOutcome:
    """Simulate the remainder of a race under one strategy.

    Fully vectorised across simulations -- every lap advances all `n_sims` races at
    once, so ten thousand futures cost about as much as one loop over the
    remaining laps. That is what keeps the strategy screen interactive.

    The degradation rate is **sampled per simulation** from its posterior rather
    than fixed at the mean. This is the point of the whole exercise: the spread
    in outcomes reflects genuine uncertainty about the tyre, not just lap-time
    noise, so a strategy that only wins under a confident estimate will show a
    wide and overlapping distribution here.

    Args:
        state: Situation to simulate forward from.
        strategy: The plan.
        tyres: Degradation model per compound. Must contain the current compound
            and, if pitting, the new one.
        n_sims: Number of race futures.
        lap_time_noise_s: Per-lap driver and traffic noise, s.
        dirty_air_s: Time lost per lap while stuck behind another car.
        safety_car_rate: Per-lap safety-car probability.
        seed: Random seed. The same seed gives the same answer, so a
            recommendation shown to a user is reproducible.

    Returns:
        A StrategyOutcome.

    Raises:
        KeyError: If a needed compound is missing from `tyres`.
    """
    rng = np.random.default_rng(seed)
    laps = state.laps_remaining

    current = tyres[state.current_compound]
    if laps == 0:
        return StrategyOutcome(strategy, np.zeros(n_sims), n_sims)

    # Sample the degradation rate once per simulated race. Sampling per lap would
    # average the uncertainty away and produce falsely tight outcomes.
    current_rate = rng.normal(current.degradation_rate, current.degradation_rate_sd, n_sims)
    new_tyre = tyres[strategy.new_compound] if strategy.new_compound else None
    new_rate = (
        rng.normal(new_tyre.degradation_rate, new_tyre.degradation_rate_sd, n_sims)
        if new_tyre
        else None
    )

    total = np.zeros(n_sims)
    age = np.full(n_sims, float(state.current_tyre_age))
    on_new_tyre = np.zeros(n_sims, dtype=bool)
    past_cliff = np.zeros(n_sims, dtype=bool)

    # Safety car per lap, shared across the race rather than per simulation-lap,
    # so a safety car is an event with consequences rather than noise.
    safety_car = rng.random((n_sims, laps)) < safety_car_rate

    # A car in traffic now stays in traffic until it pits or the car ahead does.
    in_traffic = state.gap_ahead_s < 1.2

    for i in range(laps):
        lap_number = state.current_lap + i + 1

        if strategy.pit_lap is not None and lap_number == strategy.pit_lap:
            discount = np.where(safety_car[:, i], SAFETY_CAR_PIT_DISCOUNT, 1.0)
            total += state.pit_loss_s * discount
            age[:] = 0.0
            on_new_tyre[:] = True
            # Fresh rubber and clear air: pitting drops the car out of the
            # dirty-air train it was in.
            in_traffic = False

        age += 1.0

        model = new_tyre if new_tyre is not None else current
        rate = new_rate if new_rate is not None else current_rate
        active_model = np.where(on_new_tyre, 1.0, 0.0)

        loss_current = current.loss(age, current_rate)
        if new_tyre is not None:
            loss_new = new_tyre.loss(age, new_rate)
            loss = np.where(on_new_tyre, loss_new, loss_current)
            base = np.where(on_new_tyre, new_tyre.base_pace_s, current.base_pace_s)
            cliff = np.where(on_new_tyre, new_tyre.cliff_lap, current.cliff_lap)
        else:
            loss, base, cliff = loss_current, current.base_pace_s, current.cliff_lap
            del active_model, model, rate

        past_cliff |= age > cliff

        lap_time = state.base_lap_time_s + base + loss
        if in_traffic:
            lap_time = lap_time + dirty_air_s
        lap_time = lap_time + rng.normal(0.0, lap_time_noise_s, n_sims)

        # A safety-car lap is much slower but costs everyone the same, so it
        # compresses differences between strategies rather than creating them.
        lap_time = np.where(safety_car[:, i], lap_time * 1.35, lap_time)

        total += lap_time

    return StrategyOutcome(
        strategy=strategy,
        race_times=total,
        n_sims=n_sims,
        ran_out_of_tyre=float(past_cliff.mean()),
    )


@dataclass
class Recommendation:
    """A ranked decision, with the reasoning that produced it.

    Attributes:
        best: The recommended strategy's outcome.
        alternatives: Every strategy considered, best expected time first.
        margin_s: How much better the best is than the runner-up, in expected
            seconds.
        decision_confidence: Probability the recommended strategy actually beats
            the runner-up, estimated across paired simulations. Distinct from how
            confident the model is about degradation -- a very certain tyre
            estimate can still leave two strategies genuinely too close to call.
        reasons: Plain-language justification, generated from the numbers.
    """

    best: StrategyOutcome
    alternatives: list[StrategyOutcome]
    margin_s: float
    decision_confidence: float
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "recommended": self.best.strategy.label,
            "margin_s": self.margin_s,
            "decision_confidence": self.decision_confidence,
            "reasons": self.reasons,
            "alternatives": [o.to_dict() for o in self.alternatives],
            "is_model_estimate": True,
        }


def recommend(
    state: RaceState,
    tyres: dict[str, TyreModel],
    strategies: list[Strategy] | None = None,
    *,
    n_sims: int = 5000,
    seed: int = 0,
) -> Recommendation:
    """Simulate candidate strategies and recommend one, with reasons.

    Uses **common random numbers**: every strategy is simulated with the same
    seed, so paired differences are not swamped by simulation noise. Without this
    a five-thousand-simulation comparison of two strategies half a second apart
    would be mostly noise, and the recommendation would change on every refresh.

    Args:
        state: Current race situation.
        tyres: Degradation model per compound.
        strategies: Candidates. A sensible default set is generated if omitted.
        n_sims: Simulations per strategy.
        seed: Shared random seed.

    Returns:
        A Recommendation.

    Raises:
        ValueError: If no strategy could be simulated.
    """
    if strategies is None:
        alternatives = [c for c in tyres if c != state.current_compound]
        fresh = alternatives[0] if alternatives else state.current_compound
        # Offsets are measured from the NEXT lap, because the simulation begins
        # at current_lap + 1. "Box now" means the stop happens on the very next
        # lap; a pit_lap equal to current_lap would never be reached and the stop
        # would silently never happen.
        strategies = [Strategy("Stay out", None)]
        for offset in (0, 2, 5, 8):
            lap = state.current_lap + 1 + offset
            if lap <= state.total_laps:
                label = "Box now" if offset == 0 else f"Box in {offset} laps"
                strategies.append(Strategy(label, lap, fresh))

    outcomes = [
        simulate_strategy(state, s, tyres, n_sims=n_sims, seed=seed) for s in strategies
    ]
    if not outcomes:
        raise ValueError("no strategies could be simulated")

    outcomes.sort(key=lambda o: o.expected_time)
    best, runner_up = outcomes[0], outcomes[1] if len(outcomes) > 1 else outcomes[0]

    margin = runner_up.expected_time - best.expected_time
    # Paired comparison, valid because of common random numbers.
    confidence = (
        float((best.race_times < runner_up.race_times).mean())
        if best is not runner_up
        else 1.0
    )

    return Recommendation(
        best=best,
        alternatives=outcomes,
        margin_s=float(margin),
        decision_confidence=confidence,
        reasons=_explain(state, best, runner_up, margin, confidence, tyres),
    )


def _explain(
    state: RaceState,
    best: StrategyOutcome,
    runner_up: StrategyOutcome,
    margin: float,
    confidence: float,
    tyres: dict[str, TyreModel],
) -> list[str]:
    """Build the justification from the simulated numbers.

    Deterministic and template-driven. Every sentence cites a quantity the
    simulator actually produced, so the explanation cannot drift from the
    result it is explaining. An LLM may rephrase these; it never invents them.
    """
    reasons = []
    current = tyres.get(state.current_compound)

    if current is not None:
        loss_now = float(
            current.loss(
                np.array([state.current_tyre_age]), np.array([current.degradation_rate])
            )[0]
        )
        reasons.append(
            f"The current {state.current_compound.lower()} set is {state.current_tyre_age:.0f} laps "
            f"old and is costing about {loss_now:.2f} s a lap against a fresh one."
        )
        if current.degradation_rate_sd > 0.4 * abs(current.degradation_rate):
            reasons.append(
                f"That degradation estimate is uncertain "
                f"({current.degradation_rate:.3f} ± {current.degradation_rate_sd:.3f} s/lap), "
                "so treat the margin between these strategies as soft."
            )

    if margin < 0.5:
        reasons.append(
            f"{best.strategy.label} and {runner_up.strategy.label} are within "
            f"{margin:.2f} s over the remaining {state.laps_remaining} laps. That is "
            "inside the noise -- either is defensible."
        )
    else:
        reasons.append(
            f"{best.strategy.label} is {margin:.1f} s faster than "
            f"{runner_up.strategy.label} over the remaining {state.laps_remaining} laps."
        )

    reasons.append(
        f"Across {best.n_sims:,} simulated races it wins {confidence:.0%} of the time."
    )

    if best.ran_out_of_tyre > 0.25:
        reasons.append(
            f"In {best.ran_out_of_tyre:.0%} of those races the tyre runs past its cliff, "
            "so the downside is worse than the average suggests."
        )

    if state.gap_ahead_s < 1.2:
        reasons.append(
            f"The car is within {state.gap_ahead_s:.1f} s of the one ahead, so part of "
            "the current pace loss is dirty air rather than the tyre."
        )

    return reasons


def strategy_regret(
    state: RaceState,
    tyres: dict[str, TyreModel],
    recommended_lap: int | None,
    actual_lap: int | None,
    *,
    new_compound: str | None = None,
    n_sims: int = 5000,
) -> dict:
    """How much time the strategy actually taken cost against the recommended one.

    Converts model accuracy into something a team can act on. "Our degradation
    MAE was 0.05 s/lap" means nothing to a sporting director; "stopping three laps
    late cost 3.8 seconds" does.

    Args:
        state: Race situation the decision was made from.
        tyres: Degradation model per compound.
        recommended_lap: Lap the model recommended pitting on. None for stay out.
        actual_lap: Lap the team actually pitted on. None if they stayed out.
        new_compound: Compound fitted.
        n_sims: Simulations per strategy.

    Returns:
        Expected times for both choices and the regret between them, in seconds.
        Regret is non-negative by construction: if the actual choice was better,
        the recommendation was simply not costly.
    """
    recommended = Strategy("recommended", recommended_lap, new_compound)
    actual = Strategy("actual", actual_lap, new_compound)

    rec = simulate_strategy(state, recommended, tyres, n_sims=n_sims, seed=7)
    act = simulate_strategy(state, actual, tyres, n_sims=n_sims, seed=7)

    return {
        "recommended_lap": recommended_lap,
        "actual_lap": actual_lap,
        "recommended_expected_time": rec.expected_time,
        "actual_expected_time": act.expected_time,
        "regret_s": max(0.0, act.expected_time - rec.expected_time),
        "n_sims": n_sims,
        "is_model_estimate": True,
    }
