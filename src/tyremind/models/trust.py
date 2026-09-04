"""Knowing when not to trust the model.

A degradation number with a tight interval is only useful if the interval is
honest, and an interval is only honest inside the conditions the model was
fitted on. This module produces the three signals that decide whether a
recommendation should be acted on:

  * **Consensus.** Several independent methods estimate the same quantity. When
    they agree, the number is probably about the tyre. When they disagree, it is
    probably about the model.
  * **Out-of-distribution risk.** How far the current situation sits outside what
    was actually observed. A twenty-lap projection on a compound seen for six
    laps is extrapolation wearing a confidence interval.
  * **Decision confidence.** Separate from prediction confidence. A very certain
    degradation estimate can still leave two strategies genuinely too close to
    call, and a very uncertain one can still make "box now" obvious.

The reason to build this is that the failure mode of a strategy tool is not
being wrong. It is being wrong *confidently*, at the moment the situation is
unusual -- which is exactly when someone reaches for it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class ConsensusEstimate:
    """Several models' view of the same degradation rate.

    Attributes:
        compound: Compound in question.
        estimates: Model name to (mean, sd), s/lap.
        consensus: Precision-weighted mean across models.
        consensus_sd: Spread of the consensus.
        spread: Range between the highest and lowest point estimate, s/lap. The
            headline disagreement number.
        agreement: 0-1 score. 1.0 when models are indistinguishable relative to
            their own uncertainty; falls as they separate.
    """

    compound: str
    estimates: dict[str, tuple[float, float]]
    consensus: float
    consensus_sd: float
    spread: float
    agreement: float

    @property
    def disagreement_flagged(self) -> bool:
        """Whether the models disagree enough to withhold a confident answer.

        Threshold at 0.5 rather than something tighter because models *should*
        differ a little -- they encode different assumptions, and identical
        answers from different methods would be more suspicious than mild spread.
        """
        return self.agreement < 0.5

    def explain(self) -> str:
        """Plain-language reading of the consensus."""
        if not self.disagreement_flagged:
            return (
                f"Four independent methods put {self.compound.lower()} degradation "
                f"within {self.spread:.3f} s/lap of each other. That agreement is "
                "evidence the number describes the tyre rather than the method."
            )
        worst = max(self.estimates.items(), key=lambda kv: kv[1][0])
        best = min(self.estimates.items(), key=lambda kv: kv[1][0])
        return (
            f"The methods disagree about {self.compound.lower()}: {best[0]} says "
            f"{best[1][0]:.3f} s/lap and {worst[0]} says {worst[1][0]:.3f} s/lap, "
            f"a spread of {self.spread:.3f}. Treat any strategy call that depends "
            "on this number as unsupported until more laps are in."
        )

    def to_dict(self) -> dict:
        return {
            "compound": self.compound,
            "estimates": {k: {"mean": v[0], "sd": v[1]} for k, v in self.estimates.items()},
            "consensus": self.consensus,
            "consensus_sd": self.consensus_sd,
            "spread": self.spread,
            "agreement": self.agreement,
            "disagreement_flagged": self.disagreement_flagged,
            "explanation": self.explain(),
        }


def build_consensus(
    estimates_by_model: dict[str, dict[str, tuple[float, float]]],
) -> dict[str, ConsensusEstimate]:
    """Combine several models' degradation estimates, compound by compound.

    Combination is precision-weighted: a model that reports a tight interval
    counts for more than one reporting a vague one. That is the correct weighting
    when the estimates are of the same quantity, and it stops a deliberately
    uncertain method from dragging the consensus around.

    Args:
        estimates_by_model: Model name to {compound: (mean, sd)}.

    Returns:
        Compound to ConsensusEstimate, for compounds at least two models cover.

    Raises:
        ValueError: If no model supplied any estimate.
    """
    if not estimates_by_model:
        raise ValueError("no model estimates supplied")

    compounds: set[str] = set()
    for estimates in estimates_by_model.values():
        compounds.update(estimates)

    out: dict[str, ConsensusEstimate] = {}
    for compound in sorted(compounds):
        contributions = {
            model: estimates[compound]
            for model, estimates in estimates_by_model.items()
            if compound in estimates and np.isfinite(estimates[compound][0])
        }
        if len(contributions) < 2:
            continue

        means = np.array([m for m, _ in contributions.values()])
        sds = np.array([max(s, 1e-6) for _, s in contributions.values()])

        weights = 1.0 / sds**2
        consensus = float((means * weights).sum() / weights.sum())
        consensus_sd = float(np.sqrt(1.0 / weights.sum()))
        spread = float(means.max() - means.min())

        # Spread measured against the models' own claimed uncertainty. Models
        # that differ by less than their error bars are not really disagreeing.
        typical_sd = float(np.median(sds))
        agreement = float(np.exp(-spread / (2.0 * max(typical_sd, 1e-6))))

        out[compound] = ConsensusEstimate(
            compound=compound,
            estimates=contributions,
            consensus=consensus,
            consensus_sd=consensus_sd,
            spread=spread,
            agreement=agreement,
        )

    return out


@dataclass
class ApplicabilityReport:
    """How far the situation being asked about sits outside the observed data.

    Attributes:
        applicability: 0-1. 1.0 means fully inside what was observed.
        risk: "low", "medium" or "high".
        reasons: Specific ways the query leaves the observed range.
        checks: Per-dimension detail, for display.
    """

    applicability: float
    risk: str
    reasons: list[str] = field(default_factory=list)
    checks: dict[str, dict] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "applicability": self.applicability,
            "risk": self.risk,
            "reasons": self.reasons,
            "checks": self.checks,
        }


def assess_applicability(
    lap_table: pd.DataFrame,
    *,
    compound: str,
    tyre_age: float,
    traffic_index: float = 0.0,
) -> ApplicabilityReport:
    """Judge whether the model has any business answering about this situation.

    Checks the query against the data actually fitted, dimension by dimension.
    The dimensions are the ones that genuinely break a degradation estimate:

      * **Tyre age beyond what was seen.** The commonest and most dangerous. A
        model that watched a compound for eight laps has no evidence about lap
        twenty-five, and the local-linear-trend structure will happily
        extrapolate a straight line into a cliff it cannot know about.
      * **Compound coverage.** A compound with a handful of laps has a
        degradation estimate dominated by its prior.
      * **Traffic beyond the observed range.** The traffic coefficient is fitted
        from observed interference; asking about heavier traffic extrapolates it.

    Args:
        lap_table: The data the model was fitted on.
        compound: Compound being asked about.
        tyre_age: Tyre age being asked about, laps.
        traffic_index: Traffic severity being asked about.

    Returns:
        An ApplicabilityReport.

    Raises:
        ValueError: If the lap table is empty.
    """
    if lap_table.empty:
        raise ValueError("cannot assess applicability against an empty lap table")

    reasons: list[str] = []
    checks: dict[str, dict] = {}
    scores: list[float] = []

    same_compound = lap_table[lap_table["compound"] == compound]

    # --- compound coverage ---------------------------------------------
    n_laps = len(same_compound)
    n_runs = int(same_compound["run_id"].nunique()) if n_laps else 0
    coverage = float(np.clip(n_laps / 40.0, 0.0, 1.0)) * float(np.clip(n_runs / 3.0, 0.0, 1.0))
    checks["compound_coverage"] = {
        "laps_observed": n_laps,
        "runs_observed": n_runs,
        "score": coverage,
    }
    scores.append(coverage)

    if n_laps == 0:
        reasons.append(
            f"{compound} was never run in this session, so its degradation estimate "
            "is entirely prior and carries no evidence from these laps."
        )
    elif n_laps < 15:
        reasons.append(
            f"Only {n_laps} laps of {compound} data across {n_runs} run(s). The "
            "estimate leans heavily on the compound prior."
        )

    # --- tyre age ---------------------------------------------------------
    max_age = float(same_compound["tyre_age"].max()) if n_laps else 0.0
    overshoot = max(0.0, tyre_age - max_age)
    age_score = float(np.exp(-overshoot / 6.0))
    checks["tyre_age"] = {
        "requested": tyre_age,
        "max_observed": max_age,
        "laps_beyond": overshoot,
        "score": age_score,
    }
    scores.append(age_score)

    if overshoot > 3:
        reasons.append(
            f"Asking about {tyre_age:.0f} laps of age when the oldest {compound} "
            f"set observed reached {max_age:.0f}. The last {overshoot:.0f} laps are "
            "extrapolation, and a cliff outside the observed range cannot be seen."
        )

    # --- traffic -----------------------------------------------------------
    if "traffic_index" in lap_table.columns:
        max_traffic = float(lap_table["traffic_index"].max())
        traffic_score = 1.0 if traffic_index <= max_traffic + 0.05 else 0.6
        checks["traffic"] = {
            "requested": traffic_index,
            "max_observed": max_traffic,
            "score": traffic_score,
        }
        scores.append(traffic_score)
        if traffic_score < 1.0:
            reasons.append(
                f"Traffic severity {traffic_index:.2f} exceeds anything in this "
                f"session (max {max_traffic:.2f}), so the traffic penalty is extrapolated."
            )

    applicability = float(np.prod(scores) ** (1.0 / max(len(scores), 1)))
    risk = "low" if applicability > 0.7 else "medium" if applicability > 0.4 else "high"

    if not reasons:
        reasons.append(
            "The situation sits inside the range of laps this model was fitted on."
        )

    return ApplicabilityReport(
        applicability=applicability, risk=risk, reasons=reasons, checks=checks
    )


@dataclass
class RegimeSummary:
    """How a tyre was being operated, as a label rather than a number.

    Attributes:
        regime: One of the labels below.
        confidence: How clearly the evidence points to that label, 0-1.
        evidence: The quantities the label was drawn from.
    """

    regime: str
    confidence: float
    evidence: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"regime": self.regime, "confidence": self.confidence, "evidence": self.evidence}


#: What each regime means, in the terms an engineer would use.
REGIME_MEANING: dict[str, str] = {
    "clean running": "Consistent laps in clear air. The degradation estimate is at its most trustworthy here.",
    "traffic limited": "Pace is being set by the car ahead rather than by the tyre. Reading degradation from these laps will overstate it.",
    "high degradation": "The tyre is losing performance quickly relative to its compound baseline.",
    "cliff": "Degradation is accelerating, not just continuing. Life left is shorter than a straight-line reading suggests.",
    "recovering": "Pace is improving with age, which usually means the tyre came in late or cooled back into its window.",
    "sparse data": "Too few laps to characterise how this tyre is being run.",
}


def detect_regime(
    degradation: pd.DataFrame,
    lap_table: pd.DataFrame,
    driver: str,
    run_id: int,
) -> RegimeSummary:
    """Classify how one stint was being operated.

    Rule-based rather than clustered. A clustering would need a labelled corpus
    to be interpretable, and an unlabelled cluster called "regime 3" is not
    something anyone can act on. These rules read directly off quantities the
    model already estimates, so every label can be traced to a number.

    Args:
        degradation: Per-lap latent state from `TyreSSMResult.degradation`.
        lap_table: The session.
        driver: Car.
        run_id: Run to classify.

    Returns:
        A RegimeSummary.
    """
    stint = degradation[
        (degradation["driver"] == driver) & (degradation["run_id"] == run_id)
    ].sort_values("tyre_age")
    laps = lap_table[(lap_table["driver"] == driver) & (lap_table["run_id"] == run_id)]

    if len(stint) < 4:
        return RegimeSummary("sparse data", 1.0, {"laps": float(len(stint))})

    rate = stint["rate"].to_numpy(dtype=float)
    mean_rate = float(rate.mean())
    # Is the rate itself climbing? That is what distinguishes a cliff from
    # steady degradation, and it is why the rate is modelled as a free state.
    rate_trend = float(np.polyfit(np.arange(len(rate)), rate, 1)[0]) if len(rate) > 3 else 0.0
    traffic = float(laps.get("traffic_index", pd.Series([0.0])).mean())

    evidence = {
        "mean_rate": mean_rate,
        "rate_trend": rate_trend,
        "mean_traffic": traffic,
        "laps": float(len(stint)),
    }

    if traffic > 0.35:
        return RegimeSummary("traffic limited", min(traffic / 0.6, 1.0), evidence)
    if rate_trend > 0.004:
        return RegimeSummary("cliff", min(rate_trend / 0.01, 1.0), evidence)
    if mean_rate < -0.01:
        return RegimeSummary("recovering", min(abs(mean_rate) / 0.05, 1.0), evidence)
    if mean_rate > 0.15:
        return RegimeSummary("high degradation", min(mean_rate / 0.25, 1.0), evidence)
    return RegimeSummary("clean running", 0.8, evidence)


def value_of_information(
    fit, *, candidate_signals: dict[str, float] | None = None
) -> list[dict]:
    """Which additional measurement would most reduce uncertainty.

    A bridge from "here is our uncertainty" to "here is what to instrument". The
    reductions are *estimated* from how much each unmeasured quantity currently
    contributes to the posterior width, not measured -- we cannot know what a
    sensor would tell us before fitting one.

    Args:
        fit: A fitted TyreSSMResult.
        candidate_signals: Signal name to the fraction of remaining uncertainty
            it would plausibly remove. Defaults to a documented set.

    Returns:
        Signals ranked by estimated uncertainty reduction.
    """
    rates = fit.compound_rates()
    current = float(np.mean([sd for _, sd in rates.values()])) if rates else float("nan")

    # Each entry is an engineering judgement about what the signal would resolve,
    # stated so it can be argued with rather than buried.
    defaults = {
        "Tyre surface temperature": (
            0.35,
            "Would separate thermal degradation from mechanical wear, which the "
            "model currently absorbs into one rate.",
        ),
        "Actual fuel mass": (
            0.30,
            "Would remove the fuel prior entirely, and with it the largest "
            "assumption in the identification argument.",
        ),
        "Tyre pressure": (
            0.12,
            "Would explain part of the within-stint pace variation now treated as noise.",
        ),
        "Measured tread depth": (
            0.20,
            "Would let the model estimate physical wear rather than performance loss.",
        ),
    }

    entries = []
    for signal, (reduction, why) in defaults.items():
        if candidate_signals and signal in candidate_signals:
            reduction = candidate_signals[signal]
        entries.append(
            {
                "signal": signal,
                "current_uncertainty": current,
                "estimated_reduction": reduction,
                "projected_uncertainty": current * (1.0 - reduction),
                "rationale": why,
                "is_estimate": True,
            }
        )

    return sorted(entries, key=lambda e: -e["estimated_reduction"])
