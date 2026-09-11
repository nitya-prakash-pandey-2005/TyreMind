"""The fuel regressor must count laps the driver drove, not rows that survived.

This pins a defect found by working the identifiability algebra through. The
fuel term was `groupby("run_id").cumcount()` -- a positional counter over rows
remaining *after* the quality filter. When a safety car or an outlier threshold
removes laps from the middle of a stint, the driver still completed them and
still burned that fuel, but the counter resumed at the next surviving lap.

The error does not stay in the fuel term. Under-counting fuel means
under-subtracting a correction that makes the car faster, and the shortfall is
attributed to the tyre. On 2025 Miami, where 163 laps sit in gapped runs, the
HARD degradation estimate moved 0.0389 -> 0.0669 s/lap when this was corrected.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tyremind.data.f1_loader import laps_completed_in_run


def stint(driver: str, run_id: int, ages: list[float]) -> pd.DataFrame:
    return pd.DataFrame({
        "driver": driver,
        "run_id": run_id,
        "tyre_age": [float(a) for a in ages],
    })


class TestContiguousStints:
    def test_a_new_set_counts_from_zero(self):
        frame = stint("VER", 0, [0, 1, 2, 3, 4])
        assert laps_completed_in_run(frame).tolist() == [0, 1, 2, 3, 4]

    def test_a_scrubbed_set_also_counts_from_zero(self):
        """A set arriving with three laps on it has still completed zero of THIS
        run. Fuel is burned by laps driven now, not by the tyre's history."""
        frame = stint("VER", 0, [3, 4, 5, 6, 7])
        assert laps_completed_in_run(frame).tolist() == [0, 1, 2, 3, 4]


class TestGappedStints:
    def test_laps_removed_mid_stint_are_still_counted_as_fuel(self):
        """The regression. Laps 2-5 of the set were filtered out; the driver
        completed them, so lap six of the run is the sixth lap of fuel burned,
        not the second."""
        frame = stint("DOO", 18, [2, 3, 8, 9, 10, 11])
        assert laps_completed_in_run(frame).tolist() == [0, 1, 6, 7, 8, 9]

    def test_the_old_positional_counter_would_have_been_wrong_here(self):
        frame = stint("DOO", 18, [2, 3, 8, 9, 10, 11])
        positional = frame.groupby("run_id").cumcount().tolist()
        assert positional == [0, 1, 2, 3, 4, 5]
        assert laps_completed_in_run(frame).tolist() != positional

    def test_a_gap_at_the_start_of_a_run_shifts_nothing(self):
        frame = stint("VER", 0, [5, 6, 7])
        assert laps_completed_in_run(frame).tolist() == [0, 1, 2]


class TestMultipleRunsAndDrivers:
    def test_each_run_restarts_its_own_count(self):
        frame = pd.concat([
            stint("VER", 0, [0, 1, 2]),
            stint("VER", 1, [0, 1, 2]),
        ], ignore_index=True)
        assert laps_completed_in_run(frame).tolist() == [0, 1, 2, 0, 1, 2]

    def test_two_drivers_sharing_a_run_id_are_not_pooled(self):
        """run_id is only unique within a driver in some sources. Grouping by
        run_id alone would splice two cars' stints into one counter."""
        frame = pd.concat([
            stint("VER", 0, [0, 1, 2]),
            stint("HAM", 0, [10, 11, 12]),
        ], ignore_index=True)
        assert laps_completed_in_run(frame).tolist() == [0, 1, 2, 0, 1, 2]

    def test_row_order_does_not_change_the_answer(self):
        frame = pd.concat([
            stint("VER", 0, [0, 1, 2]),
            stint("HAM", 0, [10, 11, 12]),
        ], ignore_index=True)
        shuffled = frame.sample(frac=1.0, random_state=3)
        assert laps_completed_in_run(shuffled).loc[frame.index[0]] == 0
        assert laps_completed_in_run(shuffled).sort_index().tolist() == [0, 1, 2, 0, 1, 2]


class TestAgainstTheIdentifiabilityAlgebra:
    def test_the_result_is_exactly_tyre_age_minus_its_start(self):
        """Experiment 18 rests on this: within a run the fuel regressor and the
        tyre-age regressor differ by a constant, so a free run intercept absorbs
        the difference and the two become the same vector. If this identity ever
        stopped holding, that analysis would be describing a different model."""
        ages = [3, 4, 9, 10, 11]
        frame = stint("VER", 0, ages)
        fuel = laps_completed_in_run(frame).to_numpy()
        age = np.asarray(ages, dtype=float)
        assert np.allclose(age - fuel, age[0])

    def test_centred_within_a_run_the_two_regressors_coincide(self):
        frame = stint("VER", 0, [3, 4, 9, 10, 11])
        fuel = laps_completed_in_run(frame).to_numpy()
        age = frame["tyre_age"].to_numpy()
        assert np.allclose(fuel - fuel.mean(), age - age.mean())


class TestTypes:
    def test_returns_floats_so_a_gap_cannot_be_silently_truncated(self):
        frame = stint("VER", 0, [0, 1, 2])
        assert laps_completed_in_run(frame).dtype == float

    def test_a_single_lap_run_is_zero_not_empty(self):
        assert laps_completed_in_run(stint("VER", 0, [7])).tolist() == [0.0]

    def test_fractional_tyre_ages_survive(self):
        frame = stint("VER", 0, [0.0, 1.0, 2.5])
        assert laps_completed_in_run(frame).tolist() == pytest.approx([0.0, 1.0, 2.5])
