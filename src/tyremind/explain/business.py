"""Translating model quality into the units a decision-maker uses.

"Our degradation MAE is 0.052 s/lap" means nothing to a sporting director, a
fleet manager or an investor. "Stopping three laps later than recommended cost
3.8 seconds, which at this circuit is a track position" means something.

Everything here is an *estimate derived from model output*, and every figure
carries that label. The distinction matters: a number like "7% better tyre
utilisation" is trivially easy to state and impossible to verify, and stating it
without provenance is how technical work gets dismissed as marketing.

Nothing in this module invents a number. Each figure names the model quantity it
came from and the assumption used to convert it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class ValueEstimate:
    """One quantified benefit, with its derivation attached.

    Attributes:
        metric: What is being estimated.
        value: The figure.
        unit: Its units.
        derivation: Exactly how it was computed, in one sentence a sceptic can check.
        source_quantity: The model output it came from.
        confidence: "measured", "estimated" or "illustrative".
            measured     -- computed directly from model output on real data
            estimated    -- model output converted using a stated assumption
            illustrative -- a scenario, shown to convey scale only
    """

    metric: str
    value: float
    unit: str
    derivation: str
    source_quantity: str
    confidence: str

    def to_dict(self) -> dict:
        return {
            "metric": self.metric,
            "value": self.value,
            "unit": self.unit,
            "derivation": self.derivation,
            "source_quantity": self.source_quantity,
            "confidence": self.confidence,
        }


@dataclass
class ValueReport:
    """The business case, assembled from measured quantities.

    Attributes:
        estimates: Individual figures.
        caveats: What would have to be true for these to hold in practice.
    """

    estimates: list[ValueEstimate] = field(default_factory=list)
    caveats: list[str] = field(default_factory=list)

    def by_confidence(self, level: str) -> list[ValueEstimate]:
        return [e for e in self.estimates if e.confidence == level]

    def to_dict(self) -> dict:
        return {
            "estimates": [e.to_dict() for e in self.estimates],
            "caveats": self.caveats,
            "measured_count": len(self.by_confidence("measured")),
            "estimated_count": len(self.by_confidence("estimated")),
            "illustrative_count": len(self.by_confidence("illustrative")),
        }


def build_value_report(
    *,
    naive_error: float,
    model_error: float,
    degradation_rate: float,
    regret_seconds: float | None = None,
    laps_per_stint: int = 25,
    pit_loss_s: float = 21.0,
    seconds_per_position: float = 8.0,
) -> ValueReport:
    """Convert measured model performance into operational terms.

    Args:
        naive_error: Degradation error of the naive method, s/lap. Measured.
        model_error: Degradation error of TyreMind, s/lap. Measured.
        degradation_rate: Typical degradation rate, s/lap. Measured.
        regret_seconds: Time lost by a mis-timed stop, from `strategy_regret`.
        laps_per_stint: Stint length used to convert per-lap error into per-stint.
        pit_loss_s: Pit-lane time loss at this circuit.
        seconds_per_position: Typical race-time gap between finishing positions.
            The weakest assumption here, and flagged as such -- it varies hugely
            between circuits and race situations.

    Returns:
        A ValueReport.

    Raises:
        ValueError: If the error figures are not positive.
    """
    if naive_error <= 0 or model_error <= 0:
        raise ValueError(
            f"error figures must be positive; got naive={naive_error}, model={model_error}"
        )

    report = ValueReport()

    # --- measured -------------------------------------------------------
    reduction = 1.0 - model_error / naive_error
    report.estimates.append(
        ValueEstimate(
            metric="Degradation estimate error reduced",
            value=100.0 * reduction,
            unit="%",
            derivation=(
                f"Naive method errs by {naive_error:.4f} s/lap against a known truth; "
                f"TyreMind by {model_error:.4f}. Both measured on the same sessions."
            ),
            source_quantity="exp01_ground_truth_recovery",
            confidence="measured",
        )
    )

    # An error in the per-lap rate compounds over a stint: that is what a
    # strategist is actually mis-planning by.
    stint_error_naive = naive_error * laps_per_stint
    stint_error_model = model_error * laps_per_stint
    report.estimates.append(
        ValueEstimate(
            metric="Stint-end pace error avoided",
            value=stint_error_naive - stint_error_model,
            unit="seconds",
            derivation=(
                f"A per-lap rate error compounds over a stint: "
                f"({naive_error:.4f} - {model_error:.4f}) s/lap x {laps_per_stint} laps."
            ),
            source_quantity="measured degradation error x stint length",
            confidence="measured",
        )
    )

    if regret_seconds is not None:
        report.estimates.append(
            ValueEstimate(
                metric="Time recovered by correcting one mis-timed stop",
                value=regret_seconds,
                unit="seconds",
                derivation=(
                    "Difference in expected race time between the recommended pit lap "
                    "and the one actually taken, over 5,000 simulated races."
                ),
                source_quantity="simulate.race.strategy_regret",
                confidence="measured",
            )
        )

    # --- estimated ------------------------------------------------------
    # How many laps of stint length a strategist would mis-plan by, given the
    # error in the rate. Directly actionable: it is the pit-window width.
    if degradation_rate > 0:
        window_error = stint_error_naive / max(degradation_rate * laps_per_stint, 1e-6)
        report.estimates.append(
            ValueEstimate(
                metric="Pit-window mis-placement avoided",
                value=window_error,
                unit="laps",
                derivation=(
                    f"Stint-end pace error of {stint_error_naive:.2f} s divided by a "
                    f"degradation rate of {degradation_rate:.3f} s/lap gives how many "
                    "laps of tyre life the naive estimate mistakes."
                ),
                source_quantity="measured degradation error / measured rate",
                confidence="estimated",
            )
        )

    if regret_seconds is not None and seconds_per_position > 0:
        report.estimates.append(
            ValueEstimate(
                metric="Track positions at stake",
                value=regret_seconds / seconds_per_position,
                unit="positions",
                derivation=(
                    f"{regret_seconds:.1f} s of strategy regret divided by an assumed "
                    f"{seconds_per_position:.0f} s between finishing positions."
                ),
                source_quantity="strategy_regret / assumed position gap",
                confidence="estimated",
            )
        )

    # --- illustrative ---------------------------------------------------
    report.estimates.append(
        ValueEstimate(
            metric="Unnecessary stop avoided",
            value=pit_loss_s,
            unit="seconds",
            derivation=(
                f"A stop costs {pit_loss_s:.0f} s of pit-lane time. Correctly "
                "attributing a slowdown to traffic rather than the tyre avoids "
                "paying it. Shown to convey scale; frequency is not measured here."
            ),
            source_quantity="circuit pit-lane loss",
            confidence="illustrative",
        )
    )

    report.caveats = [
        "Every figure is derived from model output, not from a team's accounts.",
        "The degradation error figures are measured against a SYNTHETIC ground "
        "truth, because public F1 data contains no measured tyre wear. The "
        "practice-to-race test is the closest real-world check available.",
        f"Converting seconds into positions assumes {seconds_per_position:.0f} s per "
        "position, which varies enormously by circuit and race situation. It is the "
        "weakest link in this chain.",
        "No claim is made about championship points, prize money, or revenue. Those "
        "depend on far more than tyre strategy.",
    ]
    return report


def fleet_value_estimate(
    *,
    fleet_size: int,
    annual_km_per_vehicle: float,
    tyres_per_vehicle: int = 6,
    tyre_cost: float = 400.0,
    life_extension_pct: float = 5.0,
) -> ValueReport:
    """Scale of the opportunity in commercial fleets. **Illustrative only.**

    Marked illustrative throughout and deliberately so. TyreMind has no fleet
    validation, because no public dataset pairs tyre tread depth with telematics.
    What this shows is the arithmetic of the opportunity, not a result.

    Presenting this as evidence would be the single most dishonest thing this
    project could do, so the report labels every figure and carries the caveat
    with it into the API and the UI.

    Args:
        fleet_size: Vehicles.
        annual_km_per_vehicle: Annual distance per vehicle.
        tyres_per_vehicle: Tyres carried.
        tyre_cost: Replacement cost per tyre.
        life_extension_pct: Assumed life extension from better replacement timing.

    Returns:
        A ValueReport, entirely illustrative.
    """
    annual_tyres = fleet_size * tyres_per_vehicle * (annual_km_per_vehicle / 60_000.0)
    saving = annual_tyres * tyre_cost * (life_extension_pct / 100.0)

    report = ValueReport()
    report.estimates.append(
        ValueEstimate(
            metric="Annual tyre spend addressed",
            value=annual_tyres * tyre_cost,
            unit="currency units",
            derivation=(
                f"{fleet_size:,} vehicles x {tyres_per_vehicle} tyres x "
                f"{annual_km_per_vehicle:,.0f} km / 60,000 km life x {tyre_cost:.0f} each."
            ),
            source_quantity="assumed fleet parameters",
            confidence="illustrative",
        )
    )
    report.estimates.append(
        ValueEstimate(
            metric="Saving at the assumed life extension",
            value=saving,
            unit="currency units per year",
            derivation=f"{life_extension_pct:.0f}% of the addressed spend.",
            source_quantity="assumed life extension",
            confidence="illustrative",
        )
    )
    report.caveats = [
        "ILLUSTRATIVE ARITHMETIC, NOT A RESULT. TyreMind has no fleet validation.",
        "No public dataset pairs tyre tread depth with vehicle telematics, so there "
        "is nothing to score a fleet claim against. We looked; it does not exist.",
        "The life-extension figure is an assumption supplied by the reader, not "
        "something this system has demonstrated.",
        "What IS demonstrated is that the same estimator transfers to a different "
        "asset class: see the C-MAPSS turbofan result, which has real ground truth.",
    ]
    return report
