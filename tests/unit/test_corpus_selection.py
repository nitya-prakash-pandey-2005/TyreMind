"""Which sessions an experiment runs on is a claim, and claims get tested.

The rule is "most recent season first, and within a season in calendar order",
so that `--limit 20` means the twenty most recent races. The bug this replaced
sorted filenames alphabetically and walked backwards, which looks like recency
and is not: it kept 2024 ahead of 2023 by accident of the digits, then ordered
inside the season by event name and silently dropped Abu Dhabi, Australia and
Austria from a twenty-race request.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from tyremind.data.corpus import sessions


def write_session(directory, session_id, n_laps=400):
    pd.DataFrame(
        {
            "driver": ["CAR1"] * n_laps,
            "session_lap": range(n_laps),
            "run_id": [0] * n_laps,
            "tyre_age": range(n_laps),
            "lap_time": [90.0] * n_laps,
            "compound": ["HARD"] * n_laps,
            "traffic_index": [0.0] * n_laps,
            "lap_in_run": range(n_laps),
        }
    ).to_parquet(directory / f"{session_id}.parquet", index=False)


@pytest.fixture
def corpus(tmp_path):
    """Two seasons whose calendar order disagrees with their alphabetical order."""
    entries = [
        # (year, round, event) -- Abu Dhabi is round 24 but sorts first by name.
        (2024, 1, "Bahrain Grand Prix"),
        (2024, 2, "Australian Grand Prix"),
        (2024, 24, "Abu Dhabi Grand Prix"),
        (2023, 1, "Bahrain Grand Prix"),
        (2023, 22, "Abu Dhabi Grand Prix"),
    ]
    manifest = []
    for year, rnd, event in entries:
        session_id = f"{year}-{event.lower().replace(' ', '-')}-R"
        write_session(tmp_path, session_id)
        manifest.append(
            {
                "session_id": session_id,
                "year": year,
                "round_number": rnd,
                "event_name": event,
                "session": "R",
            }
        )
    # A short session, to prove the lap floor is applied.
    write_session(tmp_path, "2024-sprint-grand-prix-R", n_laps=50)
    manifest.append(
        {
            "session_id": "2024-sprint-grand-prix-R",
            "year": 2024,
            "round_number": 3,
            "event_name": "Sprint Grand Prix",
            "session": "R",
        }
    )
    # A practice session, to prove session filtering is applied.
    write_session(tmp_path, "2024-bahrain-grand-prix-FP2")
    manifest.append(
        {
            "session_id": "2024-bahrain-grand-prix-FP2",
            "year": 2024,
            "round_number": 1,
            "event_name": "Bahrain Grand Prix",
            "session": "FP2",
        }
    )
    (tmp_path / "corpus.json").write_text(json.dumps(manifest))
    return tmp_path


class TestOrdering:
    def test_newest_season_comes_first(self, corpus):
        years = [s.year for s in sessions(corpus)]
        assert years == sorted(years, reverse=True)

    def test_within_a_season_the_order_is_the_calendar(self, corpus):
        rounds = [s.round_number for s in sessions(corpus) if s.year == 2024]
        assert rounds == sorted(rounds)

    def test_a_limit_takes_the_most_recent_races_not_an_alphabetical_slice(self, corpus):
        """The regression. Abu Dhabi sorts first by name and is the LAST race of
        the season; a two-race request must not be decided by its spelling."""
        picked = [s.event for s in sessions(corpus, limit=2)]
        assert picked == ["Bahrain Grand Prix", "Australian Grand Prix"]

    def test_a_limit_larger_than_the_corpus_returns_everything(self, corpus):
        assert len(sessions(corpus, limit=999)) == len(sessions(corpus))

    def test_zero_means_no_limit(self, corpus):
        assert len(sessions(corpus, limit=0)) == 5


class TestFiltering:
    def test_only_the_requested_session_type_is_returned(self, corpus):
        assert {s.session for s in sessions(corpus, session_type="R")} == {"R"}
        assert {s.session for s in sessions(corpus, session_type="FP2")} == {"FP2"}

    def test_none_keeps_every_session_type(self, corpus):
        assert {s.session for s in sessions(corpus, session_type=None)} == {"R", "FP2"}

    def test_sessions_too_short_to_split_are_dropped(self, corpus):
        assert "2024-sprint-grand-prix-R" not in {s.session_id for s in sessions(corpus)}
        assert "2024-sprint-grand-prix-R" in {
            s.session_id for s in sessions(corpus, min_laps=0)
        }

    def test_the_lap_floor_is_applied_before_the_limit(self, corpus):
        """Otherwise a limit would be spent on sessions that are then discarded,
        quietly returning fewer races than asked for."""
        assert len(sessions(corpus, limit=3)) == 3

    def test_years_can_be_restricted(self, corpus):
        assert {s.year for s in sessions(corpus, years=[2023])} == {2023}


class TestFallback:
    def test_a_directory_without_a_manifest_still_works(self, tmp_path):
        write_session(tmp_path, "2024-monza-R")
        write_session(tmp_path, "2023-monza-R")
        found = sessions(tmp_path)
        assert [s.year for s in found] == [2024, 2023]

    def test_a_missing_directory_yields_nothing_rather_than_raising(self, tmp_path):
        assert sessions(tmp_path / "absent") == []

    def test_a_manifest_entry_with_no_file_on_disk_is_skipped(self, corpus):
        manifest = json.loads((corpus / "corpus.json").read_text())
        manifest.append(
            {
                "session_id": "2024-ghost-grand-prix-R",
                "year": 2024,
                "round_number": 5,
                "event_name": "Ghost Grand Prix",
                "session": "R",
            }
        )
        (corpus / "corpus.json").write_text(json.dumps(manifest))
        assert "2024-ghost-grand-prix-R" not in {s.session_id for s in sessions(corpus)}
