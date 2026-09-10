"""Circuit description must be computable without any tyre data, and must not lie.

These features exist to predict degradation at a venue nobody has run yet, so the
one property that cannot be compromised is that nothing here reads a lap time, a
compound or a degradation estimate. The rest is arithmetic that has to survive
the shapes FastF1 actually hands over: heading angles that wrap past 360, a
single-corner frame, and telemetry with gaps in it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tyremind.physics.circuit_features import (
    LOADED_G,
    TIGHT_CORNER_DEG,
    CircuitFeatures,
    energetic_features,
    geometric_features,
)


def corners(angles: list[float], spacing: float = 500.0) -> pd.DataFrame:
    """A corner frame laid out along a straight line at fixed spacing."""
    return pd.DataFrame(
        {
            "Angle": angles,
            "X": np.arange(len(angles), dtype=float) * spacing,
            "Y": np.zeros(len(angles)),
        }
    )


class TestGeometry:
    def test_counts_every_corner(self):
        out = geometric_features(corners([30.0, 60.0, 120.0]), lap_length_m=5000.0)
        assert out["n_corners"] == 3

    def test_a_tight_corner_is_one_at_or_past_the_threshold(self):
        out = geometric_features(
            corners([TIGHT_CORNER_DEG - 1, TIGHT_CORNER_DEG, TIGHT_CORNER_DEG + 1]),
            lap_length_m=5000.0,
        )
        assert out["n_tight_corners"] == 2

    def test_headings_past_360_wrap_to_a_turn_magnitude(self):
        """FastF1 reports a heading, not a turn. 350 degrees is a 10 degree kink."""
        out = geometric_features(corners([350.0]), lap_length_m=5000.0)
        assert out["mean_corner_angle"] == pytest.approx(10.0)
        assert out["n_tight_corners"] == 0

    def test_a_heading_beyond_a_full_turn_still_wraps(self):
        out = geometric_features(corners([730.0]), lap_length_m=5000.0)
        assert out["mean_corner_angle"] == pytest.approx(10.0)

    def test_negative_headings_are_the_same_corner_the_other_way(self):
        left = geometric_features(corners([-120.0]), lap_length_m=5000.0)
        right = geometric_features(corners([120.0]), lap_length_m=5000.0)
        assert left["mean_corner_angle"] == pytest.approx(right["mean_corner_angle"])

    def test_spacing_is_the_median_gap_between_corners(self):
        out = geometric_features(corners([30.0, 30.0, 30.0], spacing=250.0), 5000.0)
        assert out["median_corner_spacing_m"] == pytest.approx(250.0)

    def test_a_single_corner_has_no_spacing_but_does_not_crash(self):
        out = geometric_features(corners([90.0]), lap_length_m=5000.0)
        assert out["n_corners"] == 1
        assert np.isnan(out["median_corner_spacing_m"])

    def test_lap_length_is_carried_through_untouched(self):
        out = geometric_features(corners([90.0]), lap_length_m=7004.0)
        assert out["lap_length_m"] == pytest.approx(7004.0)


class TestEnergetics:
    def circle(self, radius_m: float, speed_kmh: float, n: int = 400) -> pd.DataFrame:
        """A constant-radius, constant-speed corner: lateral g is v^2/r throughout."""
        theta = np.linspace(0.0, 2.0 * np.pi, n)
        return pd.DataFrame(
            {
                "X": radius_m * np.cos(theta),
                "Y": radius_m * np.sin(theta),
                "Speed": np.full(n, speed_kmh),
            }
        )

    def test_a_constant_radius_corner_gives_the_textbook_lateral_g(self):
        radius, speed_kmh = 100.0, 180.0
        expected = (speed_kmh / 3.6) ** 2 / radius / 9.81
        out = energetic_features(self.circle(radius, speed_kmh))
        assert out["mean_abs_lateral_g"] == pytest.approx(expected, rel=0.05)

    def test_a_tighter_corner_at_the_same_speed_loads_the_tyre_harder(self):
        tight = energetic_features(self.circle(50.0, 150.0))
        open_ = energetic_features(self.circle(200.0, 150.0))
        assert tight["mean_abs_lateral_g"] > open_["mean_abs_lateral_g"]

    def test_loaded_fraction_is_all_or_nothing_on_a_constant_corner(self):
        # Well above the threshold everywhere, so every sample counts as loaded.
        hot = energetic_features(self.circle(60.0, 220.0))
        assert hot["mean_abs_lateral_g"] > LOADED_G
        assert hot["loaded_fraction"] == pytest.approx(1.0, abs=0.02)

        # Well below it everywhere, so none do.
        cold = energetic_features(self.circle(600.0, 90.0))
        assert cold["mean_abs_lateral_g"] < LOADED_G
        assert cold["loaded_fraction"] == pytest.approx(0.0, abs=0.02)

    def test_top_speed_is_reported_in_the_units_it_arrived_in(self):
        out = energetic_features(self.circle(100.0, 180.0))
        assert out["top_speed_kmh"] == pytest.approx(180.0)

    def test_a_straight_line_loads_the_tyre_not_at_all(self):
        straight = pd.DataFrame(
            {"X": np.zeros(50), "Y": np.linspace(0.0, 1000.0, 50), "Speed": np.full(50, 200.0)}
        )
        out = energetic_features(straight)
        assert out["mean_abs_lateral_g"] == pytest.approx(0.0, abs=1e-6)
        assert out["loaded_fraction"] == pytest.approx(0.0)
        assert out["lateral_energy_proxy"] == pytest.approx(0.0, abs=1e-6)

    def test_telemetry_with_no_finite_curvature_at_all_returns_nothing(self):
        """Two points cannot define a curvature. Returning {} beats returning NaNs
        that would later be silently dropped from a model's design matrix."""
        degenerate = pd.DataFrame({"X": [np.nan], "Y": [np.nan], "Speed": [np.nan]})
        assert energetic_features(degenerate) == {}


class TestFeatureContract:
    def test_energetics_are_optional_and_advertised_as_such(self):
        geometry_only = CircuitFeatures(
            circuit="Nowhere", year=2024, n_corners=12, n_tight_corners=5,
            mean_corner_angle=70.0, median_corner_spacing_m=300.0, lap_length_m=5000.0,
        )
        assert not geometry_only.has_energetics

        full = CircuitFeatures(
            circuit="Nowhere", year=2024, n_corners=12, n_tight_corners=5,
            mean_corner_angle=70.0, median_corner_spacing_m=300.0, lap_length_m=5000.0,
            lateral_energy_proxy=1234.0,
        )
        assert full.has_energetics

    def test_nothing_in_the_feature_set_names_a_tyre(self):
        """The whole point is predicting a venue nobody has run. A feature derived
        from tyre behaviour would make that circular, and would pass every
        in-sample check while doing it."""
        fields = CircuitFeatures(
            circuit="Nowhere", year=2024, n_corners=1, n_tight_corners=0,
            mean_corner_angle=10.0, median_corner_spacing_m=1.0, lap_length_m=1.0,
        ).to_dict()
        forbidden = ("rate", "degradation", "compound", "tyre", "lap_time", "wear")
        assert not [f for f in fields if any(word in f for word in forbidden)]
