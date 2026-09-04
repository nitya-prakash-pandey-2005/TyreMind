"""Physical sanity tests for the vehicle dynamics layer.

These check the maths against cases where the answer is known in closed form --
a perfect circle has curvature 1/r, a car cornering left loads its right-hand
tyres, braking loads the front axle. Numerical differentiation of noisy
positional data is exactly the kind of code that produces confident nonsense
when a sign convention is wrong, and a sign error here would silently invert the
left/right tyre asymmetry without changing any total.
"""

from __future__ import annotations

import numpy as np
import pytest

from tyremind.physics.dynamics import (
    MAX_PHYSICAL_LATERAL_G,
    VehicleParameters,
    aerodynamic_downforce,
    corner_loads,
    frictional_power_proxy,
    lateral_acceleration,
    longitudinal_acceleration,
    path_curvature,
)


class TestCurvature:
    def test_circle_has_curvature_one_over_radius(self) -> None:
        """The defining property. A 100 m circle must give kappa = 0.01 /m."""
        radius = 100.0
        theta = np.linspace(0, 2 * np.pi, 400)
        x, y = radius * np.cos(theta), radius * np.sin(theta)

        kappa = path_curvature(x, y)

        # Trim the ends, where 'same'-mode smoothing biases towards zero.
        interior = np.abs(kappa[30:-30])
        assert interior.mean() == pytest.approx(1.0 / radius, rel=0.05)

    def test_straight_line_has_zero_curvature(self) -> None:
        x = np.linspace(0, 1000, 300)
        y = np.zeros_like(x)

        kappa = path_curvature(x, y)

        assert np.abs(kappa[30:-30]).max() < 1e-6

    def test_curvature_sign_distinguishes_left_from_right(self) -> None:
        """Left and right turns must produce opposite signs.

        This is what makes the left/right tyre load asymmetry recoverable. If the
        sign convention collapsed, every circuit would look symmetric and the
        per-corner energy split would be meaningless -- while every total stayed
        correct, so nothing else would catch it.
        """
        theta = np.linspace(0, np.pi, 300)
        left_x, left_y = 50 * np.cos(theta), 50 * np.sin(theta)
        right_x, right_y = 50 * np.cos(theta), -50 * np.sin(theta)

        left = path_curvature(left_x, left_y)[30:-30].mean()
        right = path_curvature(right_x, right_y)[30:-30].mean()

        assert np.sign(left) == -np.sign(right)
        assert abs(left) == pytest.approx(abs(right), rel=0.05)

    def test_short_input_returns_zeros_rather_than_raising(self) -> None:
        """Out-laps and truncated telemetry are common; they must not crash a session."""
        kappa = path_curvature(np.array([0.0, 1.0, 2.0]), np.array([0.0, 0.0, 0.0]))
        assert np.all(kappa == 0.0)


class TestLateralAcceleration:
    def test_matches_v_squared_over_r(self) -> None:
        """A car at 200 km/h on a 100 m radius pulls v^2/r.

        (55.6 m/s)^2 / 100 m = 30.9 m/s^2, which is 3.15 g -- a realistic
        medium-speed corner.
        """
        speed_ms = np.full(100, 200 / 3.6)
        curvature = np.full(100, 1.0 / 100.0)

        a_lat = lateral_acceleration(speed_ms, curvature)

        assert a_lat.mean() == pytest.approx((200 / 3.6) ** 2 / 100.0, rel=1e-9)
        assert a_lat.mean() / 9.81 == pytest.approx(3.15, abs=0.01)

    def test_clips_physically_impossible_values(self) -> None:
        """A GPS jump must not become a 40 g corner and poison every energy figure."""
        speed_ms = np.full(50, 90.0)
        curvature = np.full(50, 0.05)  # would imply ~41 g

        a_lat = lateral_acceleration(speed_ms, curvature)

        assert np.abs(a_lat).max() <= MAX_PHYSICAL_LATERAL_G * 9.81 + 1e-9

    def test_zeroed_at_low_speed(self) -> None:
        """Pit lane and slow zones give meaningless curvature; it must not leak through."""
        speed_ms = np.full(50, 3.0)
        curvature = np.full(50, 0.02)

        assert np.all(lateral_acceleration(speed_ms, curvature) == 0.0)


class TestLongitudinalAcceleration:
    def test_constant_acceleration_is_recovered(self) -> None:
        t = np.linspace(0, 10, 200)
        speed = 20.0 + 3.0 * t  # 3 m/s^2

        a = longitudinal_acceleration(speed, t)

        assert a.mean() == pytest.approx(3.0, rel=1e-6)

    def test_duplicate_timestamps_do_not_divide_by_zero(self) -> None:
        """Telemetry does contain repeated timestamps. It must degrade, not explode."""
        t = np.array([0.0, 0.1, 0.1, 0.2, 0.3, 0.4])
        speed = np.array([50.0, 51.0, 51.0, 52.0, 53.0, 54.0])

        assert np.all(np.isfinite(longitudinal_acceleration(speed, t)))


class TestCornerLoads:
    def test_static_loads_sum_to_weight_at_rest(self) -> None:
        """Stationary, with no downforce, the four corners carry exactly the weight."""
        params = VehicleParameters(fuel_mass_kg=100.0)
        zero = np.zeros(10)

        loads = corner_loads(zero, zero, zero, params)
        total = sum(v for v in loads.values())

        np.testing.assert_allclose(total, params.total_mass_kg * 9.81, rtol=1e-9)

    def test_static_front_rear_split_matches_weight_distribution(self) -> None:
        params = VehicleParameters(weight_distribution_front=0.455)
        zero = np.zeros(5)

        loads = corner_loads(zero, zero, zero, params)
        front = loads["FL"] + loads["FR"]
        total = front + loads["RL"] + loads["RR"]

        np.testing.assert_allclose(front / total, 0.455, rtol=1e-9)

    def test_cornering_left_loads_the_right_hand_tyres(self) -> None:
        """The core sign convention. Turn left, weight goes right."""
        params = VehicleParameters()
        speed = np.full(10, 60.0)
        a_lat = np.full(10, 3.0 * 9.81)  # 3 g left

        loads = corner_loads(speed, np.zeros(10), a_lat, params)

        assert np.all(loads["FR"] > loads["FL"])
        assert np.all(loads["RR"] > loads["RL"])

    def test_braking_loads_the_front_axle(self) -> None:
        params = VehicleParameters()
        speed = np.full(10, 80.0)
        a_long = np.full(10, -4.0 * 9.81)  # 4 g braking

        loads = corner_loads(speed, a_long, np.zeros(10), params)
        front = loads["FL"] + loads["FR"]
        rear = loads["RL"] + loads["RR"]

        assert np.all(front > rear)

    def test_loads_never_go_negative(self) -> None:
        """A tyre can lift off the road. It cannot pull the car down."""
        params = VehicleParameters()
        speed = np.full(10, 20.0)
        a_lat = np.full(10, 6.0 * 9.81)

        loads = corner_loads(speed, np.full(10, -5.0 * 9.81), a_lat, params)

        assert all(np.all(v >= 0.0) for v in loads.values())

    def test_downforce_dominates_at_speed(self) -> None:
        """At 300 km/h an F1 car makes several times its weight in downforce.

        If this were not true, high-speed corners would look benign and the
        energy model would badly understate what actually wears tyres out.
        """
        params = VehicleParameters()
        speed = np.array([300 / 3.6])

        ratio = aerodynamic_downforce(speed, params)[0] / (params.total_mass_kg * 9.81)

        assert 2.0 < ratio < 8.0


class TestFrictionalPower:
    def test_zero_when_the_car_is_not_working_the_tyres(self) -> None:
        """Coasting in a straight line dissipates no frictional energy in this proxy."""
        loads = {c: np.full(10, 4000.0) for c in ("FL", "FR", "RL", "RR")}
        power = frictional_power_proxy(loads, np.full(10, 70.0), np.zeros(10), np.zeros(10))

        assert all(np.allclose(v, 0.0) for v in power.values())

    def test_increases_with_load_speed_and_demand(self) -> None:
        """Monotone in all three inputs, which is what makes the ordering usable."""
        base_load = {c: np.array([4000.0]) for c in ("FL", "FR", "RL", "RR")}
        speed, a_long, a_lat = np.array([70.0]), np.array([0.0]), np.array([2.0 * 9.81])

        base = frictional_power_proxy(base_load, speed, a_long, a_lat)["FL"][0]

        heavier = frictional_power_proxy(
            {c: v * 2 for c, v in base_load.items()}, speed, a_long, a_lat
        )["FL"][0]
        faster = frictional_power_proxy(base_load, speed * 2, a_long, a_lat)["FL"][0]
        harder = frictional_power_proxy(base_load, speed, a_long, a_lat * 2)["FL"][0]

        assert heavier > base
        assert faster > base
        assert harder > base

    def test_combined_slip_uses_the_friction_circle(self) -> None:
        """Braking and cornering together demand more than either alone.

        The tyre has one friction circle; it does not have a separate budget for
        turning and for stopping. Summing the magnitudes in quadrature is what
        encodes that.
        """
        loads = {c: np.array([4000.0]) for c in ("FL", "FR", "RL", "RR")}
        speed = np.array([60.0])
        g = 9.81

        only_lat = frictional_power_proxy(loads, speed, np.array([0.0]), np.array([2 * g]))["FL"][0]
        combined = frictional_power_proxy(
            loads, speed, np.array([-2 * g]), np.array([2 * g])
        )["FL"][0]

        assert combined == pytest.approx(only_lat * np.sqrt(2), rel=1e-9)
