"""Vehicle dynamics recovered from public F1 telemetry.

Public telemetry gives speed, throttle, brake, gear, RPM and -- crucially -- the
car's X/Y position on track at roughly 4-10 Hz. Position is what makes real
dynamics recoverable rather than guessed at: differentiating the racing line
twice gives path curvature, and curvature with speed gives lateral acceleration
directly, with no vehicle model in between.

That single quantity unlocks most of what follows. Combined with longitudinal
acceleration from the speed trace it yields a g-g trace, and load transfer turns
that into a per-corner picture of what each of the four tyres was asked to do.
Which is the actual physical driver of degradation: a tyre does not care how many
laps it has done, it cares how much energy has been put through its contact patch.


What is real and what is a proxy
--------------------------------
The distinction matters, and every function here states which side it is on.

**Recovered from data, no vehicle model:** speed, longitudinal acceleration,
path curvature, lateral acceleration, distance.

**Model-derived, and therefore only as good as the parameters in
configs/physics.yaml:** normal load per corner (needs mass, weight distribution,
CG height, downforce), frictional power (needs load), energy exposure.

**Not available at all:** slip angle, slip ratio, contact patch pressure, tyre
temperature, tread depth. These require sensors that public telemetry does not
carry, and nothing here pretends otherwise. Functions returning a proxy say
`_proxy` in the name and carry the caveat in the docstring.

The purpose of this layer is not to claim measurement. It is to give the
degradation model a physically motivated *clock* -- energy through the tyre --
in place of a naive one -- laps elapsed -- and then to test whether that clock
predicts better. `exp04_energy_clock` is where that hypothesis is settled.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

#: Minimum speed below which curvature is not computed. At low speed the
#: positional sampling is too coarse relative to the distance travelled, and the
#: second derivative becomes numerical noise -- pit lane and slow zones produce
#: spectacular fake cornering loads if this is not enforced.
MIN_SPEED_FOR_CURVATURE_MS = 8.0

#: Physical ceiling on lateral acceleration, in g. Modern F1 cars peak around
#: 5-6 g in high-speed corners. Anything beyond this is a telemetry artefact --
#: a GPS jump or a dropped sample -- not a driver discovering new physics.
MAX_PHYSICAL_LATERAL_G = 7.0


@dataclass(frozen=True)
class VehicleParameters:
    """Vehicle constants needed to turn kinematics into forces.

    Defaults describe a 2024-generation F1 car. Every value is sourced in
    configs/physics.yaml with an epistemic tag; the ones here are duplicated as
    defaults so the module is usable standalone.

    Attributes:
        mass_kg: Car plus driver, excluding fuel.
        fuel_mass_kg: Current fuel load, which moves with the race.
        weight_distribution_front: Static front mass fraction.
        cg_height_m: Centre of gravity height. Drives load transfer.
        wheelbase_m: Axle separation.
        track_width_m: Lateral wheel separation.
        drag_area_m2: CdA.
        lift_area_m2: ClA, the downforce equivalent. Positive means downforce.
        air_density_kg_m3: Ambient air density.
    """

    mass_kg: float = 798.0
    fuel_mass_kg: float = 0.0
    weight_distribution_front: float = 0.455
    cg_height_m: float = 0.28
    wheelbase_m: float = 3.60
    track_width_m: float = 2.00
    drag_area_m2: float = 1.20
    lift_area_m2: float = 5.40
    air_density_kg_m3: float = 1.225

    @property
    def total_mass_kg(self) -> float:
        return self.mass_kg + self.fuel_mass_kg


def path_curvature(x: np.ndarray, y: np.ndarray, smooth_window: int = 7) -> np.ndarray:
    """Signed curvature of the racing line, 1/m.

    Uses the standard parametric formula

        kappa = (x' y'' - y' x'') / (x'^2 + y'^2)^(3/2)

    Positive is a left-hand turn, negative a right-hand one, which is what makes
    the left/right tyre asymmetry in `corner_loads` recoverable.

    Second derivatives amplify noise brutally, and positional telemetry is noisy,
    so the coordinates are smoothed first. The window is a real trade-off:
    too short and straights sprout phantom corners, too long and genuine
    direction changes get flattened into one. Seven samples at ~4-10 Hz covers
    roughly one to two seconds, which preserves corners while killing jitter.

    Args:
        x: X position, metres.
        y: Y position, metres.
        smooth_window: Samples in the moving-average window. Forced odd.

    Returns:
        Curvature per sample, 1/m. Zero where the path is degenerate.
    """
    if smooth_window % 2 == 0:
        smooth_window += 1

    if len(x) < smooth_window + 2:
        return np.zeros_like(x, dtype=float)

    kernel = np.ones(smooth_window) / smooth_window
    # 'same' mode biases the ends towards zero; those samples are dropped by the
    # caller when aggregating per lap, so the edge effect never reaches a result.
    xs = np.convolve(np.asarray(x, dtype=float), kernel, mode="same")
    ys = np.convolve(np.asarray(y, dtype=float), kernel, mode="same")

    dx, dy = np.gradient(xs), np.gradient(ys)
    ddx, ddy = np.gradient(dx), np.gradient(dy)

    denominator = (dx**2 + dy**2) ** 1.5
    with np.errstate(divide="ignore", invalid="ignore"):
        kappa = (dx * ddy - dy * ddx) / denominator

    return np.nan_to_num(kappa, nan=0.0, posinf=0.0, neginf=0.0)


def lateral_acceleration(speed_ms: np.ndarray, curvature: np.ndarray) -> np.ndarray:
    """Lateral acceleration from speed and path curvature, m/s^2.

    `a_y = v^2 * kappa`, straight from kinematics -- no vehicle model, no tyre
    model, no assumption about grip. This is the most physically trustworthy
    derived quantity available from public telemetry.

    Clipped at `MAX_PHYSICAL_LATERAL_G` and zeroed below
    `MIN_SPEED_FOR_CURVATURE_MS`, because positional glitches otherwise produce
    impossible loads that then propagate into every energy figure downstream.

    Args:
        speed_ms: Speed, m/s.
        curvature: Path curvature, 1/m.

    Returns:
        Signed lateral acceleration, m/s^2. Sign follows the curvature.
    """
    a_y = np.asarray(speed_ms, dtype=float) ** 2 * np.asarray(curvature, dtype=float)
    a_y[np.asarray(speed_ms) < MIN_SPEED_FOR_CURVATURE_MS] = 0.0
    ceiling = MAX_PHYSICAL_LATERAL_G * 9.81
    return np.clip(a_y, -ceiling, ceiling)


def longitudinal_acceleration(speed_ms: np.ndarray, time_s: np.ndarray) -> np.ndarray:
    """Longitudinal acceleration, m/s^2, by differentiating the speed trace.

    Args:
        speed_ms: Speed, m/s.
        time_s: Sample times, seconds. Need not be evenly spaced -- telemetry
            sampling is irregular, and `np.gradient` handles that correctly.

    Returns:
        Longitudinal acceleration, m/s^2. Positive accelerating.
    """
    t = np.asarray(time_s, dtype=float)
    v = np.asarray(speed_ms, dtype=float)
    if len(t) < 2:
        return np.zeros_like(v)

    # Guard against duplicate timestamps, which telemetry does contain and which
    # would otherwise divide by zero.
    dt = np.diff(t)
    if np.any(dt <= 0):
        t = np.maximum.accumulate(t + np.arange(len(t)) * 1e-9)

    return np.gradient(v, t)


def aerodynamic_downforce(speed_ms: np.ndarray, params: VehicleParameters) -> np.ndarray:
    """Aerodynamic downforce, newtons.

    `F = 0.5 * rho * ClA * v^2`. Model-derived: ClA is a literature estimate with
    real uncertainty (see configs/physics.yaml), and it varies with the
    aerodynamic setup a team ran that weekend, which is not public.

    It matters anyway. At 300 km/h an F1 car generates several times its own
    weight in downforce, so ignoring it would understate tyre loads in exactly
    the high-speed corners that do the most damage.

    Args:
        speed_ms: Speed, m/s.
        params: Vehicle parameters.

    Returns:
        Downforce, newtons. Always non-negative.
    """
    v = np.asarray(speed_ms, dtype=float)
    return 0.5 * params.air_density_kg_m3 * params.lift_area_m2 * v**2


def corner_loads(
    speed_ms: np.ndarray,
    a_long: np.ndarray,
    a_lat: np.ndarray,
    params: VehicleParameters,
) -> dict[str, np.ndarray]:
    """Vertical load on each tyre, newtons.

    Static weight plus aerodynamic downforce, redistributed by longitudinal and
    lateral load transfer:

        longitudinal transfer = m * a_x * h / wheelbase      (braking loads the front)
        lateral transfer      = m * a_y * h / track_width    (cornering loads the outside)

    This is a quasi-static rigid-body treatment. It ignores suspension
    compliance, anti-roll distribution, aerodynamic balance shift with ride
    height, and the transient in which load actually migrates. Those matter for
    setup work; for ranking which corner of the car worked hardest over a lap,
    the quasi-static split captures the dominant effect.

    Loads are floored at zero. A tyre can be unloaded, but it cannot pull down.

    Args:
        speed_ms: Speed, m/s.
        a_long: Longitudinal acceleration, m/s^2.
        a_lat: Lateral acceleration, m/s^2. Positive turning left.
        params: Vehicle parameters.

    Returns:
        Load per corner in newtons, keyed FL, FR, RL, RR.
    """
    mass = params.total_mass_kg
    weight = mass * 9.81
    downforce = aerodynamic_downforce(speed_ms, params)

    total_vertical = weight + downforce
    front_static = total_vertical * params.weight_distribution_front
    rear_static = total_vertical * (1.0 - params.weight_distribution_front)

    # Braking (negative a_x) shifts load forward.
    long_transfer = mass * np.asarray(a_long, dtype=float) * params.cg_height_m / params.wheelbase_m
    front_axle = front_static - long_transfer
    rear_axle = rear_static + long_transfer

    # Cornering left (positive a_y) loads the right-hand tyres.
    lat_transfer = mass * np.asarray(a_lat, dtype=float) * params.cg_height_m / params.track_width_m

    return {
        "FL": np.maximum(front_axle * 0.5 - lat_transfer * 0.5, 0.0),
        "FR": np.maximum(front_axle * 0.5 + lat_transfer * 0.5, 0.0),
        "RL": np.maximum(rear_axle * 0.5 - lat_transfer * 0.5, 0.0),
        "RR": np.maximum(rear_axle * 0.5 + lat_transfer * 0.5, 0.0),
    }


def frictional_power_proxy(
    loads: dict[str, np.ndarray],
    speed_ms: np.ndarray,
    a_long: np.ndarray,
    a_lat: np.ndarray,
) -> dict[str, np.ndarray]:
    """Rate of frictional energy dissipation per tyre, watts. **A proxy.**

    True frictional power at the contact patch is `mu * F_normal * v_slip`, and
    slip velocity is not observable from public telemetry -- it needs wheel-speed
    sensors the API does not expose. So slip is approximated as proportional to
    the demanded acceleration: a tyre being asked for more grip is slipping more.

    That approximation is the weakest link in this module, and it is a real one.
    What survives it is the *relative* ordering -- which corner worked hardest,
    which lap was more demanding, which circuit puts more through the rubber --
    because the unknown constant of proportionality divides out of every
    comparison the platform actually makes. Absolute values in watts are not
    meaningful and are never reported as such.

    Args:
        loads: Per-corner vertical loads, newtons.
        speed_ms: Speed, m/s.
        a_long: Longitudinal acceleration, m/s^2.
        a_lat: Lateral acceleration, m/s^2.

    Returns:
        Frictional power proxy per corner, keyed FL, FR, RL, RR.
    """
    v = np.asarray(speed_ms, dtype=float)
    # Combined acceleration magnitude: the tyre does not distinguish between
    # grip spent turning and grip spent braking, it only has a friction circle.
    demand = np.hypot(np.asarray(a_long, dtype=float), np.asarray(a_lat, dtype=float))
    slip_proxy = demand / 9.81  # dimensionless, in g

    return {corner: load * slip_proxy * v for corner, load in loads.items()}


@dataclass
class LapEnergy:
    """Per-corner energy exposure for a single lap.

    Attributes:
        driver: Car.
        lap_number: Lap.
        energy_mj: Frictional energy proxy per corner, in MJ-equivalent units.
            Relative, not absolute -- see `frictional_power_proxy`.
        peak_lateral_g: Highest lateral acceleration on the lap, g.
        peak_longitudinal_g: Largest braking deceleration on the lap, g.
        mean_speed_kmh: Mean speed.
        distance_m: Lap distance covered by the telemetry.
        n_samples: Telemetry samples used.
    """

    driver: str
    lap_number: int
    energy_mj: dict[str, float]
    peak_lateral_g: float
    peak_longitudinal_g: float
    mean_speed_kmh: float
    distance_m: float
    n_samples: int

    @property
    def total_energy_mj(self) -> float:
        """Summed across all four corners."""
        return float(sum(self.energy_mj.values()))

    @property
    def front_rear_ratio(self) -> float:
        """Front axle energy as a fraction of the total.

        Above 0.5 means a front-limited lap, which is where understeer and front
        graining come from; below means rear-limited.
        """
        front = self.energy_mj["FL"] + self.energy_mj["FR"]
        return float(front / self.total_energy_mj) if self.total_energy_mj else float("nan")

    @property
    def left_right_ratio(self) -> float:
        """Left-side energy as a fraction of the total.

        Strongly circuit-dependent, and a good sanity check on the whole
        pipeline: a clockwise circuit with mostly right-hand corners must load
        the left tyres more, and if this number does not reflect that, the
        curvature sign convention is wrong somewhere.
        """
        left = self.energy_mj["FL"] + self.energy_mj["RL"]
        return float(left / self.total_energy_mj) if self.total_energy_mj else float("nan")

    def to_dict(self) -> dict:
        return {
            "driver": self.driver,
            "lap_number": int(self.lap_number),
            "energy_mj": self.energy_mj,
            "total_energy_mj": self.total_energy_mj,
            "front_rear_ratio": self.front_rear_ratio,
            "left_right_ratio": self.left_right_ratio,
            "peak_lateral_g": self.peak_lateral_g,
            "peak_longitudinal_g": self.peak_longitudinal_g,
            "mean_speed_kmh": self.mean_speed_kmh,
            "distance_m": self.distance_m,
            "n_samples": self.n_samples,
        }


def lap_energy(
    telemetry: pd.DataFrame,
    driver: str,
    lap_number: int,
    params: VehicleParameters | None = None,
) -> LapEnergy:
    """Compute per-corner energy exposure for one lap of telemetry.

    Args:
        telemetry: Samples for a single lap, with columns `Speed` (km/h),
            `X`, `Y` (position, tenths of a metre as FastF1 supplies them) and
            either `Time` or `SessionTime`.
        driver: Car identifier, for labelling.
        lap_number: Lap number, for labelling.
        params: Vehicle parameters. Defaults to a 2024-generation car.

    Returns:
        A LapEnergy summary.

    Raises:
        ValueError: If required columns are missing or there are too few samples
            to differentiate.
    """
    params = params or VehicleParameters()

    required = {"Speed", "X", "Y"}
    missing = required - set(telemetry.columns)
    if missing:
        raise ValueError(f"telemetry is missing required columns: {sorted(missing)}")

    if len(telemetry) < 12:
        raise ValueError(
            f"only {len(telemetry)} telemetry samples for {driver} lap {lap_number}; "
            "need at least 12 to differentiate position twice"
        )

    time_column = "Time" if "Time" in telemetry.columns else "SessionTime"
    if time_column not in telemetry.columns:
        raise ValueError("telemetry needs a 'Time' or 'SessionTime' column")

    t = pd.to_timedelta(telemetry[time_column]).dt.total_seconds().to_numpy()
    t = t - t[0]

    speed_ms = telemetry["Speed"].to_numpy(dtype=float) / 3.6
    # FastF1 supplies X/Y in tenths of a metre.
    x = telemetry["X"].to_numpy(dtype=float) / 10.0
    y = telemetry["Y"].to_numpy(dtype=float) / 10.0

    kappa = path_curvature(x, y)
    a_lat = lateral_acceleration(speed_ms, kappa)
    a_long = longitudinal_acceleration(speed_ms, t)

    loads = corner_loads(speed_ms, a_long, a_lat, params)
    power = frictional_power_proxy(loads, speed_ms, a_long, a_lat)

    # Integrate power over time. Trim the edges, where the convolution used for
    # smoothing biases curvature towards zero.
    edge = 4
    dt = np.gradient(t)
    energy = {
        corner: float(np.trapezoid(p[edge:-edge], t[edge:-edge]) / 1.0e6)
        for corner, p in power.items()
    }

    distance = float(np.sum(speed_ms * dt))

    return LapEnergy(
        driver=driver,
        lap_number=lap_number,
        energy_mj=energy,
        peak_lateral_g=float(np.abs(a_lat[edge:-edge]).max() / 9.81),
        peak_longitudinal_g=float(np.abs(a_long[edge:-edge].min()) / 9.81),
        mean_speed_kmh=float(telemetry["Speed"].mean()),
        distance_m=distance,
        n_samples=len(telemetry),
    )
