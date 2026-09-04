"""Practice-to-race validation: does a Friday degradation curve survive Sunday?

This is the challenge's second explicit ask -- "post-race validation tools to
compare predicted wear against actual race-day pace" -- and it is also the
hardest honest test in the project, which is why it is the headline result
rather than a held-out split.

A random or chronological holdout inside one session asks the model to
interpolate among laps it has essentially already seen. Practice-to-race asks
something much harder, because the two settings differ in almost every way that
matters:

  * **Fuel.** Practice runs are short and variably fuelled. A race stint starts
    heavy and runs to the end of the tank.
  * **Traffic.** Practice cars are spread out; a race field is packed.
  * **Track.** The circuit is greener on Friday than on Sunday.
  * **Driving.** Practice long runs are managed. Racing is not, except when it is
    -- and the difference between those two is itself unobservable.

If a Friday-derived degradation curve predicts Sunday, that is evidence the
model captured something about the tyre rather than something about the session.
If it does not, that is a finding worth reporting, and this module is built to
report it either way.

The comparison is deliberately on *degradation rate*, not lap time. Predicting
race lap times well is easy for the wrong reasons -- fuel dominates the signal
and would flatter any model that gets fuel roughly right. The rate is the
quantity the practice session was supposed to tell us, so it is the quantity we
score.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from tyremind.data.synthetic import naive_degradation_estimate
from tyremind.models.ssm.tyre_ssm import TyreSSMPriors, fit_tyre_ssm


@dataclass
class CompoundComparison:
    """Predicted against realised degradation for one compound.

    Attributes:
        compound: Compound label.
        predicted: Degradation rate estimated from practice, s/lap.
        predicted_sd: Its posterior standard deviation.
        actual: Degradation rate estimated from the race, s/lap.
        actual_sd: Its posterior standard deviation.
        naive_predicted: What the naive practice estimate would have said, s/lap.
        practice_laps: Laps on this compound in practice.
        race_laps: Laps on this compound in the race.
    """

    compound: str
    predicted: float
    predicted_sd: float
    actual: float
    actual_sd: float
    naive_predicted: float
    practice_laps: int
    race_laps: int

    @property
    def error(self) -> float:
        """Predicted minus actual, s/lap."""
        return self.predicted - self.actual

    @property
    def naive_error(self) -> float:
        return self.naive_predicted - self.actual

    @property
    def covered(self) -> bool:
        """Whether the practice 95% interval contains the race estimate.

        Both sides carry uncertainty, so the comparison uses the combined
        standard deviation. Treating the race figure as a known constant would
        make the test look harder than it honestly is.
        """
        combined = float(np.hypot(self.predicted_sd, self.actual_sd))
        return abs(self.error) <= 1.96 * combined

    def to_dict(self) -> dict:
        return {
            "compound": self.compound,
            "predicted": self.predicted,
            "predicted_sd": self.predicted_sd,
            "actual": self.actual,
            "actual_sd": self.actual_sd,
            "error": self.error,
            "naive_predicted": self.naive_predicted,
            "naive_error": self.naive_error,
            "covered_95": self.covered,
            "practice_laps": self.practice_laps,
            "race_laps": self.race_laps,
        }


@dataclass
class ValidationReport:
    """Outcome of validating one event's practice prediction against its race.

    Attributes:
        event: Event label.
        year: Season.
        practice_session: Session the prediction came from.
        comparisons: Per-compound results, for compounds present in both.
        practice_quality: Data-quality score of the practice session, 0-100.
        race_quality: Data-quality score of the race.
        skipped: Compounds present in only one of the two sessions, and why.
    """

    event: str
    year: int
    practice_session: str
    comparisons: list[CompoundComparison] = field(default_factory=list)
    practice_quality: float = float("nan")
    race_quality: float = float("nan")
    skipped: dict[str, str] = field(default_factory=dict)

    @property
    def mae(self) -> float:
        """Mean absolute error of the practice-derived prediction, s/lap."""
        if not self.comparisons:
            return float("nan")
        return float(np.mean([abs(c.error) for c in self.comparisons]))

    @property
    def naive_mae(self) -> float:
        if not self.comparisons:
            return float("nan")
        return float(np.mean([abs(c.naive_error) for c in self.comparisons]))

    @property
    def bias(self) -> float:
        """Signed mean error. Persistent sign means practice systematically
        under- or over-states race degradation, which is a physical finding
        rather than a modelling failure."""
        if not self.comparisons:
            return float("nan")
        return float(np.mean([c.error for c in self.comparisons]))

    @property
    def coverage(self) -> float:
        """Fraction of compounds whose race value fell inside the practice interval."""
        if not self.comparisons:
            return float("nan")
        return float(np.mean([c.covered for c in self.comparisons]))

    def to_dict(self) -> dict:
        return {
            "event": self.event,
            "year": self.year,
            "practice_session": self.practice_session,
            "mae": self.mae,
            "naive_mae": self.naive_mae,
            "bias": self.bias,
            "coverage_95": self.coverage,
            "practice_quality": self.practice_quality,
            "race_quality": self.race_quality,
            "skipped": self.skipped,
            "comparisons": [c.to_dict() for c in self.comparisons],
        }


def validate_practice_to_race(
    year: int,
    grand_prix: str | int,
    *,
    practice_session: str = "FP2",
    priors: TyreSSMPriors | None = None,
    min_laps_per_compound: int = 8,
) -> ValidationReport:
    """Estimate degradation from a practice session and score it against the race.

    Both sessions are fitted with the *same* model and the *same* priors. No race
    information reaches the practice fit -- that separation is the entire point,
    and it is enforced by simply never passing one to the other.

    The race estimate is treated as the reference rather than as ground truth. It
    is itself a model output with its own interval, and `CompoundComparison.covered`
    accounts for that. Public data contains no measured tyre wear, so a true
    ground truth does not exist here; the synthetic benchmark in
    `exp01_ground_truth_recovery` is where that claim gets tested.

    Args:
        year: Season.
        grand_prix: Event name or round number.
        practice_session: Which practice session to predict from. FP2 by default,
            because it is where long runs on representative fuel usually happen.
        priors: Prior specification, shared by both fits.
        min_laps_per_compound: Compounds with fewer laps than this on either side
            are skipped and recorded in `skipped`, rather than being compared on
            evidence too thin to mean anything.

    Returns:
        A ValidationReport.

    Raises:
        ValueError: If neither session yields any comparable compound.
    """
    from tyremind.data.f1_loader import load_lap_table

    practice_laps, practice_quality = load_lap_table(year, grand_prix, practice_session)
    race_laps, race_quality = load_lap_table(year, grand_prix, "R")

    practice_fit = fit_tyre_ssm(practice_laps, priors=priors)
    race_fit = fit_tyre_ssm(race_laps, priors=priors)

    practice_rates = practice_fit.compound_rates()
    race_rates = race_fit.compound_rates()
    naive_rates = naive_degradation_estimate(practice_laps)

    practice_counts = practice_laps["compound"].value_counts().to_dict()
    race_counts = race_laps["compound"].value_counts().to_dict()

    report = ValidationReport(
        event=str(practice_quality.session_name).rsplit(" ", 2)[0],
        year=year,
        practice_session=practice_session,
        practice_quality=practice_quality.score(),
        race_quality=race_quality.score(),
    )

    for compound in sorted(set(practice_rates) | set(race_rates)):
        p_laps = int(practice_counts.get(compound, 0))
        r_laps = int(race_counts.get(compound, 0))

        if compound not in practice_rates:
            report.skipped[compound] = f"not run in {practice_session}"
            continue
        if compound not in race_rates:
            report.skipped[compound] = "not run in the race"
            continue
        if p_laps < min_laps_per_compound or r_laps < min_laps_per_compound:
            report.skipped[compound] = (
                f"too few laps to compare ({p_laps} in {practice_session}, {r_laps} in race; "
                f"need {min_laps_per_compound})"
            )
            continue

        p_mean, p_sd = practice_rates[compound]
        r_mean, r_sd = race_rates[compound]
        report.comparisons.append(
            CompoundComparison(
                compound=compound,
                predicted=p_mean,
                predicted_sd=p_sd,
                actual=r_mean,
                actual_sd=r_sd,
                naive_predicted=float(naive_rates.get(compound, np.nan)),
                practice_laps=p_laps,
                race_laps=r_laps,
            )
        )

    if not report.comparisons:
        raise ValueError(
            f"{year} {grand_prix}: no compound had at least {min_laps_per_compound} "
            f"laps in both {practice_session} and the race. Reasons: {report.skipped}"
        )

    return report


def summarise_reports(reports: list[ValidationReport]) -> pd.DataFrame:
    """Fold a set of event reports into one row per event, for the results table."""
    return pd.DataFrame(
        [
            {
                "year": r.year,
                "event": r.event,
                "compounds": len(r.comparisons),
                "mae": r.mae,
                "naive_mae": r.naive_mae,
                "bias": r.bias,
                "coverage_95": r.coverage,
                "practice_quality": r.practice_quality,
            }
            for r in reports
        ]
    )
