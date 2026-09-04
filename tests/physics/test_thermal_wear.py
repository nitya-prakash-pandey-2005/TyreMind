"""Tests for the thermal and wear models.

These check physical behaviour rather than specific numbers, because the
coefficients are calibrated and the absolute temperatures carry no claim. What
must hold is the *shape*: heat goes in when work is done, cooling scales with
airspeed, the carcass lags the surface, and wear rises on both sides of the
working window rather than just the hot one.

The bidirectional wear curve is the one worth guarding hardest. An Arrhenius
form -- the obvious textbook choice -- is monotone in temperature and cannot
represent graining at all, so a regression to it would silently remove the cold
half of tyre behaviour while still looking like physics.
"""

from __future__ import annotations

import numpy as np
import pytest

from tyremind.physics.thermal import (
    DEFAULT_WINDOW_C,
    ThermalParameters,
    simulate_thermal,
    temperature_wear_multiplier,
)
from tyremind.physics.wear import (
    COMPOUND_WEAR_FACTOR,
    WearParameters,
    archard_wear,
    cumulative_energy_clock,
    energy_wear_rate,
)


def _trace(power_w: float, speed_ms: float, seconds: float = 90.0, hz: float = 10.0):
    n = int(seconds * hz)
    t = np.linspace(0.0, seconds, n)
    return (
        np.full(n, power_w),
        np.full(n, speed_ms),
        t,
    )


class TestThermalModel:
    def test_work_heats_the_tyre(self) -> None:
        power, speed, t = _trace(power_w=45_000.0, speed_ms=60.0)
        trace = simulate_thermal(power, speed, t)
        assert trace.surface_c[-1] > trace.surface_c[0]

    def test_no_work_cools_the_tyre(self) -> None:
        """A tyre doing nothing sheds heat to the air and the road."""
        power, speed, t = _trace(power_w=0.0, speed_ms=40.0)
        trace = simulate_thermal(power, speed, t, initial_surface_c=110.0)
        assert trace.surface_c[-1] < trace.surface_c[0]

    def test_surface_responds_faster_than_the_carcass(self) -> None:
        """The two-state split only earns its place if the states differ.

        Surface temperature is what grips; carcass temperature is what carries
        history between laps. If they moved together, one state would do.
        """
        power, speed, t = _trace(power_w=60_000.0, speed_ms=55.0)
        trace = simulate_thermal(power, speed, t, initial_surface_c=90.0, initial_bulk_c=90.0)

        surface_rise = trace.surface_c[-1] - trace.surface_c[0]
        bulk_rise = trace.bulk_c[-1] - trace.bulk_c[0]

        assert surface_rise > bulk_rise > 0

    def test_faster_airflow_cools_more(self) -> None:
        """Convective cooling scales with speed, which is why a slow lap runs hot."""
        power, _, t = _trace(power_w=30_000.0, speed_ms=0.0)

        slow = simulate_thermal(power, np.full(len(t), 20.0), t, initial_surface_c=120.0)
        fast = simulate_thermal(power, np.full(len(t), 80.0), t, initial_surface_c=120.0)

        assert fast.surface_c[-1] < slow.surface_c[-1]

    def test_carries_thermal_history_between_laps(self) -> None:
        """Starting hot must end hotter. Without this a stint has no memory."""
        power, speed, t = _trace(power_w=40_000.0, speed_ms=60.0)

        cold_start = simulate_thermal(power, speed, t, initial_bulk_c=60.0)
        hot_start = simulate_thermal(power, speed, t, initial_bulk_c=110.0)

        assert hot_start.bulk_c[-1] > cold_start.bulk_c[-1]
        assert hot_start.surface_c[-1] > cold_start.surface_c[-1]

    def test_stays_finite_with_telemetry_gaps(self) -> None:
        """Real telemetry drops samples. A gap must not integrate a heat spike."""
        n = 400
        t = np.linspace(0, 40, n)
        t[200:] += 3.0  # a three-second dropout
        power = np.full(n, 50_000.0)
        speed = np.full(n, 60.0)

        trace = simulate_thermal(power, speed, t)

        assert np.all(np.isfinite(trace.surface_c))
        assert trace.surface_c.max() < 400.0

    def test_regime_labels_track_temperature(self) -> None:
        power, speed, t = _trace(power_w=0.0, speed_ms=30.0)

        cold = simulate_thermal(
            power, speed, t, ThermalParameters(ambient_c=5.0, track_c=5.0), initial_surface_c=40.0
        )
        assert cold.regime() in {"underheated", "below window"}

        hot = simulate_thermal(
            np.full(len(t), 200_000.0), speed, t, initial_surface_c=140.0
        )
        assert hot.regime() in {"overheating", "above window"}

    def test_rejects_mismatched_input_lengths(self) -> None:
        with pytest.raises(ValueError, match="same length"):
            simulate_thermal(np.zeros(10), np.zeros(5), np.arange(10.0))


class TestWearTemperatureCurve:
    def test_no_penalty_inside_the_working_window(self) -> None:
        lo, hi = DEFAULT_WINDOW_C
        inside = np.linspace(lo, hi, 20)
        np.testing.assert_allclose(temperature_wear_multiplier(inside), 1.0)

    def test_penalty_rises_on_BOTH_sides_of_the_window(self) -> None:
        """The load-bearing property.

        Rubber wears faster when too cold (graining, from sliding a stiff tyre)
        and when too hot (thermal degradation). A monotone-in-temperature law --
        Arrhenius, say -- gets the hot side right and silently removes the cold
        side, which is half of real tyre behaviour.
        """
        lo, hi = DEFAULT_WINDOW_C
        too_cold = temperature_wear_multiplier(np.array([lo - 25.0]))[0]
        just_right = temperature_wear_multiplier(np.array([(lo + hi) / 2]))[0]
        too_hot = temperature_wear_multiplier(np.array([hi + 25.0]))[0]

        assert too_cold > just_right
        assert too_hot > just_right
        assert just_right == pytest.approx(1.0)

    def test_penalty_grows_faster_than_linearly(self) -> None:
        """Rubber tolerates a few degrees out of window and not twenty."""
        hi = DEFAULT_WINDOW_C[1]
        small = temperature_wear_multiplier(np.array([hi + 10.0]))[0] - 1.0
        large = temperature_wear_multiplier(np.array([hi + 20.0]))[0] - 1.0
        assert large > 3.0 * small

    def test_multiplier_never_below_one(self) -> None:
        temps = np.linspace(-20.0, 250.0, 200)
        assert np.all(temperature_wear_multiplier(temps) >= 1.0)


class TestWearRate:
    def test_scales_with_frictional_power(self) -> None:
        surface = np.full(50, 100.0)
        low = energy_wear_rate(np.full(50, 10_000.0), surface, "MEDIUM")
        high = energy_wear_rate(np.full(50, 20_000.0), surface, "MEDIUM")
        assert np.allclose(high, 2.0 * low)

    def test_softer_compounds_wear_faster(self) -> None:
        power, surface = np.full(20, 30_000.0), np.full(20, 100.0)
        soft = energy_wear_rate(power, surface, "SOFT").sum()
        medium = energy_wear_rate(power, surface, "MEDIUM").sum()
        hard = energy_wear_rate(power, surface, "HARD").sum()
        assert soft > medium > hard

    def test_out_of_window_wears_faster_than_in_window(self) -> None:
        """Two laps with identical energy wear differently if one ran hot."""
        power = np.full(30, 40_000.0)
        in_window = energy_wear_rate(power, np.full(30, 100.0), "MEDIUM").sum()
        overheating = energy_wear_rate(power, np.full(30, 135.0), "MEDIUM").sum()
        assert overheating > in_window

    def test_negative_power_does_not_reduce_wear(self) -> None:
        """Wear is irreversible. A numerical artefact must not un-wear a tyre."""
        rate = energy_wear_rate(np.full(10, -5_000.0), np.full(10, 100.0), "MEDIUM")
        assert np.all(rate >= 0.0)

    def test_compound_factors_are_ordered_softest_first(self) -> None:
        assert (
            COMPOUND_WEAR_FACTOR["SOFT"]
            > COMPOUND_WEAR_FACTOR["MEDIUM"]
            > COMPOUND_WEAR_FACTOR["HARD"]
        )

    def test_archard_scales_with_load_and_distance(self) -> None:
        """The documented baseline, retained for the ablation."""
        base = archard_wear(np.full(10, 4_000.0), np.full(10, 1.0))
        assert archard_wear(np.full(10, 8_000.0), np.full(10, 1.0)) == pytest.approx(2 * base)
        assert archard_wear(np.full(10, 4_000.0), np.full(10, 2.0)) == pytest.approx(2 * base)


class TestEnergyClock:
    def _wear(self, energies: list[float]):
        from tyremind.physics.wear import LapWear

        return [
            LapWear(
                driver="X",
                lap_number=i,
                compound="MEDIUM",
                energy_mj={"FL": e / 4, "FR": e / 4, "RL": e / 4, "RR": e / 4},
                wear_increment=e,
                thermal_regime="in window",
                peak_surface_c=105.0,
                mean_bulk_c=95.0,
                fraction_in_window=1.0,
                thermal_stress=0.0,
            )
            for i, e in enumerate(energies)
        ]

    def test_identical_laps_give_a_clock_equal_to_lap_count(self) -> None:
        """Normalisation must keep the clock readable in lap-equivalents.

        If every lap costs the same energy, the energy clock and lap count are
        the same thing, and a degradation rate expressed against either is in the
        same units. Without that the two clocks would not be comparable at all.
        """
        clock = cumulative_energy_clock(self._wear([10.0] * 6))
        np.testing.assert_allclose(clock, np.arange(1.0, 7.0))

    def test_harder_laps_advance_the_clock_faster(self) -> None:
        clock = cumulative_energy_clock(self._wear([5.0, 5.0, 20.0, 5.0]))
        steps = np.diff(clock)
        assert steps[1] > steps[0]

    def test_empty_input_returns_empty(self) -> None:
        assert cumulative_energy_clock([]).size == 0

    def test_degenerate_energy_falls_back_to_lap_count(self) -> None:
        """Broken telemetry must not produce a divide-by-zero clock."""
        clock = cumulative_energy_clock(self._wear([0.0, 0.0, 0.0]))
        np.testing.assert_allclose(clock, np.arange(1.0, 4.0))


class TestWearParameters:
    def test_unknown_compound_falls_back_to_neutral(self) -> None:
        assert WearParameters().factor_for("TEST_UNKNOWN") == 1.0
