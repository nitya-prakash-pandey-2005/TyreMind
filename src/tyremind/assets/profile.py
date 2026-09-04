"""The industry-agnostic core: what TyreMind is actually estimating.

Strip the motorsport vocabulary away and the problem is not about tyres:

    An asset degrades while it operates. You cannot observe its condition
    directly. You observe a performance signal that is confounded by operating
    conditions which also change over time. Recover the degradation.

That description fits an F1 tyre, a truck tyre, a jet engine, a battery, a
bearing and a cutting tool. The estimator does not need to know which -- it needs
to know what plays the part of "age", what plays the part of "performance", and
which confounders are present.

`AssetProfile` is that description. The state-space model consumes profiles, not
tyres, which is what makes the cross-domain claim a property of the code rather
than a slide. `experiments/exp07_cross_domain.py` runs the *identical* estimator
on NASA's turbofan degradation benchmark and scores it against published RUL
ground truth.


Why this matters commercially
-----------------------------
There is no public dataset pairing tyre tread depth with telematics -- we
checked, and it does not exist. So the honest way to demonstrate that the
estimator generalises is to run it on the one real degradation benchmark that
does have ground truth, and say plainly that the fleet application is
architecture rather than evidence.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class Confounder(str, Enum):
    """A cause of observed performance change that is not degradation.

    Naming these is the whole design. Every one is something that moves the
    performance signal while the asset's actual condition is unchanged, and
    every one has to be either measured, modelled, or declared unidentifiable.
    """

    #: Something that gets monotonically lighter/easier over an operating run.
    #: Fuel burn-off in a race; payload drop on a delivery round.
    LOAD_REDUCTION = "load_reduction"

    #: A slow environmental improvement shared by every asset in the fleet.
    #: Track rubbering in; ambient temperature falling overnight.
    ENVIRONMENT_DRIFT = "environment_drift"

    #: Interference from other assets. Dirty air; congestion.
    INTERFERENCE = "interference"

    #: Differences between operators. Driver style.
    OPERATOR_VARIATION = "operator_variation"

    #: Discrete changes in how the asset is being run. Engine modes; duty cycles.
    OPERATING_MODE = "operating_mode"


@dataclass(frozen=True)
class AssetProfile:
    """What the estimator needs to know about a class of asset.

    Attributes:
        asset_type: Identifier, e.g. "f1_tyre", "turbofan", "truck_tyre".
        display_name: Human-readable name.
        age_unit: What degradation is measured against -- "laps", "cycles",
            "kilometres". The x-axis of every degradation curve.
        performance_unit: What is observed -- "seconds per lap", "health index".
        performance_is_cost: True when a HIGHER observed value means WORSE
            condition. Lap time is a cost (slower is worse); a health index is
            not. Getting this backwards inverts every degradation sign, so it is
            explicit rather than inferred.
        confounders: Which confounders are present for this asset class.
        typical_life: Expected life in `age_unit`, for scaling priors and axes.
        performance_threshold: Loss beyond which the asset is no longer fit for
            purpose, in `performance_unit`. Defines remaining useful life.
        degradation_prior_mean: Prior mean degradation per unit age.
        degradation_prior_sd: Prior spread. Wide by default -- the point of
            pooling across a fleet is that the data should speak.
        notes: Anything an engineer reading a result should know first.
    """

    asset_type: str
    display_name: str
    age_unit: str
    performance_unit: str
    performance_is_cost: bool
    confounders: frozenset[Confounder]
    typical_life: float
    performance_threshold: float
    degradation_prior_mean: float
    degradation_prior_sd: float
    notes: str = ""

    def has(self, confounder: Confounder) -> bool:
        return confounder in self.confounders

    def to_dict(self) -> dict:
        return {
            "asset_type": self.asset_type,
            "display_name": self.display_name,
            "age_unit": self.age_unit,
            "performance_unit": self.performance_unit,
            "performance_is_cost": self.performance_is_cost,
            "confounders": sorted(c.value for c in self.confounders),
            "typical_life": self.typical_life,
            "performance_threshold": self.performance_threshold,
            "degradation_prior_mean": self.degradation_prior_mean,
            "degradation_prior_sd": self.degradation_prior_sd,
            "notes": self.notes,
        }


F1_TYRE = AssetProfile(
    asset_type="f1_tyre",
    display_name="Formula 1 tyre",
    age_unit="laps",
    performance_unit="seconds per lap",
    performance_is_cost=True,
    confounders=frozenset(
        {
            Confounder.LOAD_REDUCTION,
            Confounder.ENVIRONMENT_DRIFT,
            Confounder.INTERFERENCE,
            Confounder.OPERATOR_VARIATION,
        }
    ),
    typical_life=30.0,
    performance_threshold=0.8,
    degradation_prior_mean=0.05,
    degradation_prior_sd=0.20,
    notes=(
        "The hard case. All four confounders present, and two of them are only "
        "separable from degradation through a physical prior."
    ),
)

TURBOFAN = AssetProfile(
    asset_type="turbofan",
    display_name="Turbofan engine (NASA C-MAPSS)",
    age_unit="cycles",
    performance_unit="health index",
    performance_is_cost=False,
    confounders=frozenset({Confounder.OPERATING_MODE}),
    typical_life=200.0,
    performance_threshold=0.5,
    degradation_prior_mean=0.004,
    degradation_prior_sd=0.010,
    notes=(
        "Run-to-failure with published RUL ground truth, which motorsport data "
        "does not have. Far fewer confounders -- only operating mode -- so it "
        "tests the estimator rather than the identification argument."
    ),
)

TRUCK_TYRE = AssetProfile(
    asset_type="truck_tyre",
    display_name="Commercial vehicle tyre",
    age_unit="thousand kilometres",
    performance_unit="rolling resistance index",
    performance_is_cost=True,
    confounders=frozenset(
        {
            Confounder.LOAD_REDUCTION,
            Confounder.ENVIRONMENT_DRIFT,
            Confounder.OPERATOR_VARIATION,
            Confounder.OPERATING_MODE,
        }
    ),
    typical_life=150.0,
    performance_threshold=0.15,
    degradation_prior_mean=0.001,
    degradation_prior_sd=0.003,
    notes=(
        "ARCHITECTURE ONLY -- not validated. No public dataset pairs tyre tread "
        "depth with telematics, so there is nothing to score against. The JRC "
        "reports 1.0-1.2 mm of front tread lost per 10,000 km for passenger "
        "cars, which sets the scale but is not a validation."
    ),
)

PROFILES: dict[str, AssetProfile] = {
    p.asset_type: p for p in (F1_TYRE, TURBOFAN, TRUCK_TYRE)
}


def get_profile(asset_type: str) -> AssetProfile:
    """Look up a profile by type.

    Args:
        asset_type: Identifier, e.g. "f1_tyre".

    Returns:
        The profile.

    Raises:
        KeyError: If unknown.
    """
    if asset_type not in PROFILES:
        raise KeyError(f"unknown asset type {asset_type!r}; known: {sorted(PROFILES)}")
    return PROFILES[asset_type]


@dataclass
class DegradationObservation:
    """One observation of an asset, in the estimator's own vocabulary.

    The translation layer. A lap and an engine cycle become the same thing here,
    which is what lets one estimator serve both.

    Attributes:
        asset_id: Which individual asset -- a car, an engine unit, a vehicle.
        unit_id: Which run-to-replacement period. A tyre set; an engine's life.
        sequence: Global time index within the dataset.
        age: Age of the unit at this observation, in the profile's age_unit.
        performance: The observed signal.
        mode: Operating mode or regime label, if the profile has one.
        elapsed_in_run: Observations completed in this run, for load-reduction
            confounders.
        interference: Interference severity in [0, 1], if applicable.
    """

    asset_id: str
    unit_id: int
    sequence: int
    age: float
    performance: float
    mode: str = "default"
    elapsed_in_run: int = 0
    interference: float = 0.0


def to_lap_table(observations: list[DegradationObservation], profile: AssetProfile):
    """Translate generic observations into the schema `fit_tyre_ssm` consumes.

    The column names remain motorsport-flavoured because that is the estimator's
    native vocabulary and renaming them everywhere would buy nothing. What
    matters is that any asset class can be mapped onto them.

    Sign handling is the substantive part. The estimator assumes performance is a
    *cost* that rises as the asset degrades, so a profile whose signal falls with
    degradation (a health index) is negated here. Without that, degradation would
    come back with the wrong sign and every downstream conclusion would invert.

    Args:
        observations: Observations to translate.
        profile: The asset class.

    Returns:
        A DataFrame in the estimator's schema.

    Raises:
        ValueError: If no observations are supplied.
    """
    import pandas as pd

    if not observations:
        raise ValueError("no observations to translate")

    sign = 1.0 if profile.performance_is_cost else -1.0

    return pd.DataFrame(
        {
            "driver": [o.asset_id for o in observations],
            "session_lap": [o.sequence for o in observations],
            "run_id": [o.unit_id for o in observations],
            "tyre_age": [float(o.age) for o in observations],
            "lap_in_run": [int(o.elapsed_in_run) for o in observations],
            "lap_time": [sign * float(o.performance) for o in observations],
            "compound": [o.mode for o in observations],
            "traffic_index": [float(o.interference) for o in observations],
        }
    )
