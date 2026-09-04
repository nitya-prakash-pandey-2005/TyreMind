"""Reduced-order tyre thermal model.

Tyre degradation is not a function of distance. It is a function of the energy
put through the rubber and the temperature the rubber was at while that happened,
and temperature is the reason two stints of identical length can end in
completely different places.

Public telemetry contains no tyre temperature. What it contains is speed, the
racing line, and therefore the frictional work being done. This module turns that
into a two-state thermal estimate:

    surface   the tread, which heats and cools in seconds and is what actually
              grips the road
    bulk      the carcass, which heats and cools over minutes and is what carries
              thermal history from one lap into the next

    C_s dTs/dt = Q_friction - k_sb(Ts - Tb) - h(v)(Ts - T_air) - k_road(Ts - T_track)
    C_b dTb/dt = k_sb(Ts - Tb) - h_b(Tb - T_air)

Two states rather than one because the distinction is the whole point: a driver
can drop surface temperature in a single cool-down lap, but an overheated carcass
stays overheated, and that is what ends stints.


What this is, and is not
------------------------
These are ESTIMATED states, not measurements. The coefficients are calibrated so
that the model's output correlates with observed degradation, not against any
temperature sensor -- there is no such sensor in the data. The absolute numbers in
degrees are therefore not trustworthy as temperatures, and the platform never
displays them as if they were.

What *is* trustworthy is the relative structure: which laps put more heat into the
tyre, whether the carcass was trending up or down across a stint, and how long a
tyre spent outside its working window. Those are the quantities the degradation
model consumes, and they only require the shape to be right.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: Pirelli slick working range, degrees C. Outside it degradation accelerates in
#: BOTH directions -- too cold gives graining, too hot gives thermal degradation.
#: That bidirectional shape is the physically correct one; a wear law that is
#: monotone in temperature is simply wrong about rubber.
DEFAULT_WINDOW_C = (90.0, 110.0)


@dataclass(frozen=True)
class ThermalParameters:
    """Lumped thermal coefficients for one tyre.

    Calibrated, not measured. See `experiments/exp04_energy_clock.py` for how
    they are fitted and configs/physics.yaml for their epistemic status.

    Attributes:
        surface_capacity: Heat capacity of the tread layer, J/K. Small, so the
            surface responds within a corner.
        bulk_capacity: Heat capacity of the carcass, J/K. Roughly four times the
            surface, so it responds over a stint.
        surface_to_bulk: Conductance between tread and carcass, W/K.
        convection_per_speed: Convective cooling to air, W/K per m/s. Scales with
            airspeed, which is why a slow lap is a hot lap.
        conduction_to_track: Conduction into the road surface, W/K.
        bulk_convection: Direct carcass cooling to air, W/K.
        ambient_c: Air temperature, degrees C.
        track_c: Track surface temperature, degrees C.
        window_c: Working range in which the compound performs, degrees C.
    """

    surface_capacity: float = 12_000.0
    bulk_capacity: float = 48_000.0
    surface_to_bulk: float = 55.0
    convection_per_speed: float = 2.4
    conduction_to_track: float = 30.0
    bulk_convection: float = 8.0
    ambient_c: float = 25.0
    track_c: float = 35.0
    window_c: tuple[float, float] = DEFAULT_WINDOW_C

    @property
    def window_centre(self) -> float:
        return 0.5 * (self.window_c[0] + self.window_c[1])


@dataclass
class ThermalTrace:
    """Estimated thermal state through a lap.

    Attributes:
        surface_c: Tread temperature estimate per sample, degrees C.
        bulk_c: Carcass temperature estimate per sample, degrees C.
        time_s: Sample times, seconds.
        window_c: The working range used.
    """

    surface_c: np.ndarray
    bulk_c: np.ndarray
    time_s: np.ndarray
    window_c: tuple[float, float]

    @property
    def peak_surface_c(self) -> float:
        return float(self.surface_c.max())

    @property
    def mean_bulk_c(self) -> float:
        return float(self.bulk_c.mean())

    def fraction_in_window(self) -> float:
        """Share of the lap the tread spent inside its working range.

        The single most actionable thermal number: a tyre outside its window is
        degrading faster than its age suggests, in one direction or the other.
        """
        lo, hi = self.window_c
        return float(((self.surface_c >= lo) & (self.surface_c <= hi)).mean())

    def thermal_stress(self) -> float:
        """Mean squared distance outside the working window, in K^2.

        Squared because the penalty for being outside the window is not linear --
        rubber tolerates a few degrees and does not tolerate twenty. Zero when the
        tyre stayed in range for the whole lap.
        """
        lo, hi = self.window_c
        excursion = np.maximum(0.0, np.maximum(lo - self.surface_c, self.surface_c - hi))
        return float((excursion**2).mean())

    def regime(self) -> str:
        """A one-word description of how the tyre was operating.

        Deliberately coarse. The estimate is not precise enough to support
        anything finer, and a label an engineer can act on beats a number they
        cannot check.
        """
        lo, hi = self.window_c
        mean_surface = float(self.surface_c.mean())
        if mean_surface < lo - 12:
            return "underheated"
        if mean_surface < lo:
            return "below window"
        if mean_surface > hi + 12:
            return "overheating"
        if mean_surface > hi:
            return "above window"
        return "in window"


def simulate_thermal(
    frictional_power_w: np.ndarray,
    speed_ms: np.ndarray,
    time_s: np.ndarray,
    params: ThermalParameters | None = None,
    *,
    initial_surface_c: float | None = None,
    initial_bulk_c: float | None = None,
) -> ThermalTrace:
    """Integrate the two-state thermal model over one lap of telemetry.

    Uses explicit Euler with the timestep taken from the telemetry itself. At the
    4-10 Hz public telemetry provides, and with time constants of seconds
    (surface) and minutes (bulk), explicit integration is comfortably stable --
    the stiffest mode has a time constant of roughly `surface_capacity /
    (surface_to_bulk + cooling)`, which is several seconds against a timestep of
    0.1-0.25 s.

    The step is nonetheless clamped, because telemetry contains gaps and a single
    two-second hole would otherwise produce a temperature spike that propagates
    into every downstream figure.

    Args:
        frictional_power_w: Heat generated at the contact patch per sample, W.
            Typically from `dynamics.frictional_power_proxy`.
        speed_ms: Speed per sample, m/s. Drives convective cooling.
        time_s: Sample times, seconds.
        params: Thermal coefficients.
        initial_surface_c: Starting tread temperature. Defaults to the middle of
            the working window, which is where a tyre leaves the blankets.
        initial_bulk_c: Starting carcass temperature. Defaults to slightly below
            the surface.

    Returns:
        A ThermalTrace.

    Raises:
        ValueError: If the inputs are not the same length or are too short.
    """
    params = params or ThermalParameters()

    power = np.asarray(frictional_power_w, dtype=float)
    speed = np.asarray(speed_ms, dtype=float)
    t = np.asarray(time_s, dtype=float)

    if not (len(power) == len(speed) == len(t)):
        raise ValueError(
            f"inputs must be the same length; got power={len(power)}, "
            f"speed={len(speed)}, time={len(t)}"
        )
    if len(t) < 3:
        raise ValueError(f"need at least 3 samples to integrate, got {len(t)}")

    surface = np.zeros(len(t))
    bulk = np.zeros(len(t))
    surface[0] = (
        params.window_centre if initial_surface_c is None else float(initial_surface_c)
    )
    bulk[0] = (
        surface[0] - 8.0 if initial_bulk_c is None else float(initial_bulk_c)
    )

    dt = np.diff(t, prepend=t[0])
    # Clamp against telemetry gaps. A dropout would otherwise integrate a whole
    # second of frictional heating in one step.
    dt = np.clip(dt, 0.0, 0.5)

    for i in range(1, len(t)):
        convection = params.convection_per_speed * max(speed[i], 0.0)

        d_surface = (
            power[i]
            - params.surface_to_bulk * (surface[i - 1] - bulk[i - 1])
            - convection * (surface[i - 1] - params.ambient_c)
            - params.conduction_to_track * (surface[i - 1] - params.track_c)
        ) / params.surface_capacity

        d_bulk = (
            params.surface_to_bulk * (surface[i - 1] - bulk[i - 1])
            - params.bulk_convection * (bulk[i - 1] - params.ambient_c)
        ) / params.bulk_capacity

        surface[i] = surface[i - 1] + d_surface * dt[i]
        bulk[i] = bulk[i - 1] + d_bulk * dt[i]

    return ThermalTrace(
        surface_c=surface, bulk_c=bulk, time_s=t, window_c=params.window_c
    )


def temperature_wear_multiplier(
    surface_c: np.ndarray,
    window_c: tuple[float, float] = DEFAULT_WINDOW_C,
    sensitivity: float = 0.025,
) -> np.ndarray:
    """How much faster the tyre wears at a given temperature, relative to ideal.

    Returns 1.0 inside the working window and rises quadratically outside it, in
    both directions. The bidirectional shape is the physically important part:

      * **Too cold.** The rubber is stiff, cannot generate grip through
        hysteresis, and the driver has to slide it to make it work -- which
        tears the surface. This is graining.
      * **Too hot.** The rubber softens past its useful range and abrades and
        blisters. This is thermal degradation.

    An Arrhenius form would be the textbook choice, but Arrhenius is monotone in
    temperature and so cannot represent the cold side at all. A symmetric
    quadratic penalty around the window is the simplest form that has the right
    shape, and with public data there is no basis for anything more elaborate.

    Args:
        surface_c: Tread temperature per sample, degrees C.
        window_c: Working range.
        sensitivity: Multiplier growth per K^2 outside the window.

    Returns:
        Wear rate multiplier per sample. At least 1.0 everywhere.
    """
    lo, hi = window_c
    surface = np.asarray(surface_c, dtype=float)
    excursion = np.maximum(0.0, np.maximum(lo - surface, surface - hi))
    return 1.0 + sensitivity * excursion**2
