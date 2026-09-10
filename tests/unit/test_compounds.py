"""The compound reference table is data a wrong row can silently corrupt.

Every downstream claim that pools across events depends on this mapping being
right. A typo would not crash anything -- it would quietly attribute one
compound's degradation to another and bias the pooled baseline. So the table is
tested like code.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from tyremind.data.compounds import (
    ALLOCATION_PATH,
    annotate,
    coverage,
    load_allocations,
    resolve,
)


class TestReferenceTable:
    def test_table_is_present_and_parses(self) -> None:
        assert ALLOCATION_PATH.exists(), "committed reference data is missing"
        assert load_allocations(), "table parsed but is empty"

    def test_every_nomination_can_start_a_consecutive_triplet(self) -> None:
        """Pirelli has nominated three consecutive compounds since 2023.

        That makes C1, C2 and C3 the only possible hardest choices. If this ever
        fails it means either a typo or that the nomination convention changed,
        and the table needs revisiting either way.
        """
        for key, allocation in load_allocations().items():
            assert 1 <= allocation.hardest <= 3, f"{key}: C{allocation.hardest} cannot start a triplet"
            assert allocation.compound_id("SOFT") in {"C3", "C4", "C5"}

    def test_labels_map_to_distinct_ascending_compounds(self) -> None:
        for allocation in load_allocations().values():
            hard = allocation.compound_id("HARD")
            medium = allocation.compound_id("MEDIUM")
            soft = allocation.compound_id("SOFT")
            assert hard != medium != soft
            assert int(hard[1]) < int(medium[1]) < int(soft[1])

    def test_no_duplicate_events(self) -> None:
        raw = json.loads(ALLOCATION_PATH.read_text(encoding="utf-8"))
        keys = [(r["year"], r["round"]) for r in raw["allocations"]]
        assert len(keys) == len(set(keys)), "an event is listed twice"

    def test_every_row_cites_a_source_that_exists(self) -> None:
        """A claim with a dangling citation is worse than one with none."""
        raw = json.loads(ALLOCATION_PATH.read_text(encoding="utf-8"))
        known = set(raw["sources"])
        for row in raw["allocations"]:
            assert row["source"] in known, f"{row['event']} cites unknown source {row['source']}"
            assert row["confidence"] in {"primary", "secondary"}

    def test_2024_season_is_complete(self) -> None:
        """Twenty-four rounds, no gaps -- a missing round silently drops a race."""
        rounds = {a.round_number for a in load_allocations().values() if a.year == 2024}
        assert rounds == set(range(1, 25)), f"2024 is missing rounds {sorted(set(range(1, 25)) - rounds)}"


class TestResolution:
    def test_the_same_label_is_different_rubber_at_different_events(self) -> None:
        """The whole reason this module exists."""
        monza = resolve(2024, 16, "MEDIUM")      # C3/C4/C5 weekend
        silverstone = resolve(2024, 12, "MEDIUM")  # C1/C2/C3 weekend
        assert monza == "C4"
        assert silverstone == "C2"
        assert monza != silverstone

    def test_wet_compounds_do_not_resolve(self) -> None:
        assert resolve(2024, 16, "INTERMEDIATE") is None
        assert resolve(2024, 16, "WET") is None

    def test_unknown_event_returns_none_rather_than_guessing(self) -> None:
        assert resolve(1999, 1, "SOFT") is None

    @pytest.mark.parametrize("label", ["hard", "Hard", "HARD"])
    def test_label_case_is_ignored(self, label: str) -> None:
        assert resolve(2024, 1, label) == "C1"


class TestAnnotate:
    def test_adds_compound_id_per_row(self) -> None:
        laps = pd.DataFrame({"compound": ["HARD", "MEDIUM", "SOFT"], "lap_time": [90.0, 90.5, 89.5]})
        out = annotate(laps, 2024, 16)  # Monza: C3/C4/C5
        assert out["compound_id"].tolist() == ["C3", "C4", "C5"]
        assert "compound" in out.columns, "the original label must survive"

    def test_unknown_event_yields_none_not_the_label(self) -> None:
        """Falling back to the label would reintroduce exactly the bug."""
        laps = pd.DataFrame({"compound": ["HARD"], "lap_time": [90.0]})
        out = annotate(laps, 1999, 1)
        assert out["compound_id"].isna().all()

    def test_does_not_mutate_the_input(self) -> None:
        laps = pd.DataFrame({"compound": ["HARD"], "lap_time": [90.0]})
        annotate(laps, 2024, 1)
        assert "compound_id" not in laps.columns


class TestCoverage:
    def test_coverage_reports_every_row(self) -> None:
        table = coverage()
        assert len(table) == len(load_allocations())
        assert set(table.columns) >= {"year", "round", "event", "hard", "medium", "soft", "confidence"}
