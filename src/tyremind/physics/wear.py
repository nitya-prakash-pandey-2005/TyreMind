"""Energy-based tyre wear, and the degradation clock it defines.

The central hypothesis of the physics layer, stated so it can be falsified:

    A tyre does not know how many laps it has done. It knows how much energy has
    been put through its contact patch, and how hot it was while that happened.
    So cumulative tyre energy should be a better degradation clock than lap count.

If that is true, a degradation curve measured on Friday transfers to Sunday even
though the two sessions load the tyre differently -- which is exactly the problem
the challenge poses, and exactly where the practice-to-race validation currently
shows a systematic +0.047 s/lap bias.

`experiments/exp04_energy_clock.py` tests it. The result is reported either way.


Model
-----
Archard's law is the standard starting point:

    V = K * F_N * L / H          volume worn ~ load x sliding distance / hardness

It is used because it is simple, not because it is right for rubber: it assumes
a constant coefficient and ignores viscoelasticity, roughness and temperature
entirely (see the rubber-wear review at PMC12915245). For tyres the energy
formulation is the better-supported one --

    dW/dt = K(T, compound) * P_friction

-- because rubber wear tracks dissipated frictional energy, and the coefficient
is where temperature enters. Both are implemented: Archard as the documented
baseline for the ablation, energy-based as the model actually used.

Units are deliberately arbitrary. Nothing here is calibrated against a measured
tread depth, because no such measurement exists in public data. Only *relative*
energy matters -- which lap was harder, which corner worked hardest, which stint
put more through the rubber -- and the unknown constant divides out of every
comparison the platform makes.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from tyremind.physics.thermal import (
    DEFAULT_WINDOW_C,
    ThermalParameters,
    simulate_thermal,
    temperature_wear_multiplier,
)

#: Relative wear rate by compound at a common energy input. Softer rubber gives
#: more grip and wears faster; the ordering is certain, the spacing is a
#: calibrated guess and is treated as such.
COMPOUND_WEAR_FACTOR: dict[str, float] = {
    "SOFT": 1.55,
    "MEDIUM": 1.00,
    "HARD": 0.68,
    "INTERMEDIATE": 1.20,
    "WET": 1.10,
}


@dataclass(frozen=True)
class WearParameters:
    """Coefficients for the energy-based wear law.

    Attributes:
        reference_coefficient: Wear per unit frictional energy for a medium
            compound in its working window. Arbitrary scale -- only ratios are used.
        temperature_sensitivity: Growth in wear rate per K^2 outside the window.
        window_c: Working temperature range.
        compound_factors: Relative wear rate per compound.
    """

    reference_coefficient: float = 1.0e-8
    temperature_sensitivity: float = 0.025
    window_c: tuple[float, float] = DEFAULT_WINDOW_C
    compound_factors: dict[str, float] | None = None

    def factor_for(self, compound: str) -> float:
        factors = self.compound_factors or COMPOUND_WEAR_FACTOR
        return factors.get(str(compound).upper(), 1.0)


@dataclass
class LapWear:
    """Wear and energy accounting for one lap.

    Attributes:
        driver: Car.
        lap_number: Lap.
        compound: Compound in use.
        energy_mj: Frictional energy proxy per corner, MJ-equivalent.
        wear_increment: Wear accumulated on this lap, arbitrary units.
        thermal_regime: Coarse operating description from the thermal model.
        peak_surface_c: Highest estimated tread temperature, degrees C.
        mean_bulk_c: Mean estimated carcass temperature, degrees C.
        fraction_in_window: Share of the lap spent in the working range.
        thermal_stress: Mean squared excursion outside the window, K^2.
    """

    driver: str
    lap_number: int
    compound: str
    energy_mj: dict[str, float]
    wear_increment: float
    thermal_regime: str
    peak_surface_c: float
    mean_bulk_c: float
    fraction_in_window: float
    thermal_stress: float

    @property
    def total_energy_mj(self) -> float:
        return float(sum(self.energy_mj.values()))

    def to_dict(self) -> dict:
        return {
            "driver": self.driver,
            "lap_number": int(self.lap_number),
            "compound": self.compound,
            "energy_mj": self.energy_mj,
            "total_energy_mj": self.total_energy_mj,
            "wear_increment": self.wear_increment,
            "thermal_regime": self.thermal_regime,
            "peak_surface_c": self.peak_surface_c,
            "mean_bulk_c": self.mean_bulk_c,
            "fraction_in_window": self.fraction_in_window,
            "thermal_stress": self.thermal_stress,
        }


def archard_wear(
    normal_force_n: np.ndarray,
    sliding_distance_m: np.ndarray,
    hardness: float = 1.0,
    coefficient: float = 1.0e-8,
) -> float:
    """Archard wear volume. **Baseline for the ablation, not the model in use.**

    V = K * F_N * L / H.

    Retained so the energy formulation has something to be compared against.
    Archard assumes a constant wear coefficient and is blind to temperature, so
    it cannot represent graining or thermal degradation at all -- which is
    precisely the comparison worth making.

    Args:
        normal_force_n: Contact load per sample, N.
        sliding_distance_m: Sliding distance per sample, m.
        hardness: Material hardness, arbitrary units.
        coefficient: Dimensionless wear coefficient K.

    Returns:
        Total worn volume in arbitrary units.
    """
    return float(
        coefficient
        * np.sum(np.asarray(normal_force_n, dtype=float) * np.asarray(sliding_distance_m, dtype=float))
        / max(hardness, 1e-9)
    )


def energy_wear_rate(
    frictional_power_w: np.ndarray,
    surface_c: np.ndarray,
    compound: str,
    params: WearParameters | None = None,
) -> np.ndarray:
    """Instantaneous wear rate per sample, arbitrary units per second.

        dW/dt = K_ref * f(compound) * m(T) * P_friction

    The temperature multiplier `m(T)` is what distinguishes this from Archard.
    Two laps with identical frictional energy wear the tyre differently if one
    was run outside the working window, and that difference is often larger than
    the difference in energy.

    Args:
        frictional_power_w: Frictional power per sample, W.
        surface_c: Estimated tread temperature per sample, degrees C.
        compound: Compound label.
        params: Wear coefficients.

    Returns:
        Wear rate per sample.
    """
    params = params or WearParameters()
    multiplier = temperature_wear_multiplier(
        surface_c, params.window_c, params.temperature_sensitivity
    )
    return (
        params.reference_coefficient
        * params.factor_for(compound)
        * multiplier
        * np.maximum(np.asarray(frictional_power_w, dtype=float), 0.0)
    )


def lap_wear(
    telemetry: pd.DataFrame,
    driver: str,
    lap_number: int,
    compound: str,
    *,
    vehicle_params=None,
    thermal_params: ThermalParameters | None = None,
    wear_params: WearParameters | None = None,
    initial_bulk_c: float | None = None,
) -> LapWear:
    """Full physics pass over one lap: dynamics, thermal state, then wear.

    Args:
        telemetry: Samples for a single lap. See `dynamics.lap_energy`.
        driver: Car identifier.
        lap_number: Lap number.
        compound: Compound in use.
        vehicle_params: `dynamics.VehicleParameters`.
        thermal_params: Thermal coefficients.
        wear_params: Wear coefficients.
        initial_bulk_c: Carcass temperature carried in from the previous lap.
            Passing this through a stint is what gives the model thermal memory --
            without it every lap starts from the blankets, which is wrong after
            lap one.

    Returns:
        A LapWear summary.

    Raises:
        ValueError: If the telemetry is unusable. See `dynamics.lap_energy`.
    """
    from tyremind.physics.dynamics import (
        VehicleParameters,
        corner_loads,
        frictional_power_proxy,
        lateral_acceleration,
        longitudinal_acceleration,
        path_curvature,
    )

    vehicle_params = vehicle_params or VehicleParameters()
    wear_params = wear_params or WearParameters()

    required = {"Speed", "X", "Y"}
    missing = required - set(telemetry.columns)
    if missing:
        raise ValueError(f"telemetry is missing required columns: {sorted(missing)}")
    if len(telemetry) < 12:
        raise ValueError(
            f"only {len(telemetry)} samples for {driver} lap {lap_number}; "
            "need at least 12 to differentiate position twice"
        )

    time_column = "Time" if "Time" in telemetry.columns else "SessionTime"
    t = pd.to_timedelta(telemetry[time_column]).dt.total_seconds().to_numpy()
    t = t - t[0]

    speed_ms = telemetry["Speed"].to_numpy(dtype=float) / 3.6
    x = telemetry["X"].to_numpy(dtype=float) / 10.0
    y = telemetry["Y"].to_numpy(dtype=float) / 10.0

    a_lat = lateral_acceleration(speed_ms, path_curvature(x, y))
    a_long = longitudinal_acceleration(speed_ms, t)

    loads = corner_loads(speed_ms, a_long, a_lat, vehicle_params)
    power = frictional_power_proxy(loads, speed_ms, a_long, a_lat)
    total_power = sum(power.values())

    thermal = simulate_thermal(
        total_power, speed_ms, t, thermal_params, initial_bulk_c=initial_bulk_c
    )

    edge = 4
    energy = {
        corner: float(np.trapezoid(p[edge:-edge], t[edge:-edge]) / 1.0e6)
        for corner, p in power.items()
    }
    rate = energy_wear_rate(
        total_power[edge:-edge], thermal.surface_c[edge:-edge], compound, wear_params
    )

    return LapWear(
        driver=driver,
        lap_number=lap_number,
        compound=compound,
        energy_mj=energy,
        wear_increment=float(np.trapezoid(rate, t[edge:-edge])),
        thermal_regime=thermal.regime(),
        peak_surface_c=thermal.peak_surface_c,
        mean_bulk_c=thermal.mean_bulk_c,
        fraction_in_window=thermal.fraction_in_window(),
        thermal_stress=thermal.thermal_stress(),
    )


def cumulative_energy_clock(lap_wears: list[LapWear]) -> np.ndarray:
    """Cumulative tyre energy across a stint, normalised to a lap-equivalent scale.

    This is the alternative degradation clock. Dividing by the stint's mean
    per-lap energy makes it directly comparable to lap count: a value of 10 means
    "as much energy as ten average laps of this stint", so a degradation rate
    expressed against it is still in seconds per lap-equivalent and remains
    readable by an engineer.

    Without that normalisation the clock would be in arbitrary MJ and every
    downstream rate would change units, which is a good way to make a better
    model look worse.

    Args:
        lap_wears: Per-lap physics results, in stint order.

    Returns:
        Cumulative lap-equivalent energy, one value per lap. Empty input gives an
        empty array.
    """
    if not lap_wears:
        return np.array([])

    energies = np.array([w.total_energy_mj for w in lap_wears], dtype=float)
    mean_energy = float(energies.mean())
    if mean_energy <= 0:
        # Degenerate telemetry: fall back to lap count, which is what the clock
        # would reduce to anyway if every lap cost the same.
        return np.arange(1.0, len(energies) + 1.0)
    return np.cumsum(energies / mean_energy)
