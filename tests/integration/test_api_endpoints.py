"""Every endpoint the dashboard depends on, exercised against real cached data.

The unit tests cover the estimator, the calibration and the physics. Nothing
covered the thing a judge actually opens. Twenty-three routes served the
dashboard with no automated test between them and a regression, and two of the
defects found in this project -- a WebSocket frame carrying a bare NaN, and a
cursor pointing at a number the panel no longer rendered -- were exactly the
kind that a route-level test catches and a unit test does not.

These run against the committed demo sessions, so they need no network. They are
deliberately shallow per route and broad across routes: the contract being
protected is "every screen still has data to draw", not the arithmetic, which is
tested where it lives.
"""

from __future__ import annotations

import json
import math

import pytest
from fastapi.testclient import TestClient

from tyremind.api.main import app

RACE = "2024-monza-R"
PRACTICE = "2024-monza-FP2"


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="module")
def a_run(client) -> dict:
    """The session's longest stint, which every per-run route can answer for."""
    runs = client.get(f"/api/session/{RACE}/runs").json()
    assert runs, "no runs in the demo session"
    return runs[0]


@pytest.fixture(scope="module")
def a_lap(a_run) -> dict:
    """A driver and a lap PART WAY through a stint.

    Deliberately not the last lap. Several routes forecast forwards, and asked
    about the final lap of a run they correctly refuse -- `pit-window` answers
    400 "no laps remain to pit on", which is right and is not what this suite is
    trying to exercise.
    """
    midpoint = int(a_run["first_lap"]) + (int(a_run["last_lap"]) - int(a_run["first_lap"])) // 2
    return {"driver": a_run["driver"], "lap": midpoint}


def assert_json_is_strict(payload) -> None:
    """No NaN or Infinity anywhere in the response.

    Python's encoder emits bare NaN/Infinity tokens, which are not valid JSON and
    which strict parsers reject -- and one such field poisons the whole frame,
    not only itself. This has already happened once, on the live WebSocket.
    """

    def boom(token: str):
        raise AssertionError(f"response contains a non-finite JSON token: {token}")

    json.loads(json.dumps(payload), parse_constant=boom)


class TestServiceLevel:
    def test_health_reports_what_it_can_serve_offline(self, client):
        body = client.get("/api/health").json()
        assert body["status"] == "ok"
        assert body["sessions_available"] > 0
        assert body["offline_ready"] is True

    def test_the_session_catalogue_is_not_empty(self, client):
        sessions = client.get("/api/sessions").json()
        assert sessions
        assert {s["session_id"] for s in sessions} >= {RACE, PRACTICE}

    def test_every_experiment_result_is_served(self, client):
        experiments = client.get("/api/experiments").json()
        # The dashboard reads its figures from here; an empty payload means every
        # evidence panel silently renders its "not yet run" state.
        assert len(experiments) >= 15
        assert "exp03_practice_to_race" in experiments

    def test_an_unknown_session_is_a_404_not_a_500(self, client):
        assert client.get("/api/session/not-a-session").status_code == 404


class TestSessionRoutes:
    @pytest.mark.parametrize(
        "route", ["", "/degradation", "/track", "/runs", "/trust", "/validation"]
    )
    def test_route_answers_with_finite_json(self, client, route):
        response = client.get(f"/api/session/{RACE}{route}")
        assert response.status_code == 200, response.text
        payload = response.json()
        assert payload not in (None, [], {})
        assert_json_is_strict(payload)

    @pytest.mark.parametrize(
        "route", ["/decompose", "/projection", "/pit-window", "/strategy"]
    )
    def test_per_lap_route_answers_with_finite_json(self, client, route, a_lap):
        response = client.get(f"/api/session/{RACE}{route}", params=a_lap)
        assert response.status_code == 200, response.text
        assert_json_is_strict(response.json())

    def test_regret_scores_a_pit_call_against_the_recommendation(self, client, a_run, a_lap):
        response = client.get(
            f"/api/session/{RACE}/regret",
            params=a_lap | {
                "recommended_lap": a_lap["lap"] + 2,
                "actual_lap": int(a_run["last_lap"]),
            },
        )
        assert response.status_code == 200, response.text
        body = response.json()
        assert_json_is_strict(body)
        assert body["regret_s"] >= 0.0

    def test_asking_to_pit_on_the_final_lap_is_refused_not_answered(self, client, a_run):
        """A forecast with nothing left to forecast should say so rather than
        return an empty result the UI would render as a recommendation."""
        response = client.get(
            f"/api/session/{RACE}/pit-window",
            params={"driver": a_run["driver"], "lap": int(a_run["last_lap"])},
        )
        assert response.status_code == 400
        assert "no laps remain" in response.text

    def test_health_timeline_answers_for_a_run(self, client, a_run):
        response = client.get(
            f"/api/session/{RACE}/health-timeline",
            params={"driver": a_run["driver"], "run_id": a_run["run_id"]},
        )
        assert response.status_code == 200, response.text
        assert_json_is_strict(response.json())

    def test_summary_carries_a_rate_and_an_interval_per_compound(self, client):
        body = client.get(f"/api/session/{RACE}").json()
        assert body["compounds"]
        for compound, estimate in body["compounds"].items():
            assert math.isfinite(estimate["degradation_rate"]), compound
            low, high = estimate["ci95"]
            assert low <= estimate["degradation_rate"] <= high

    def test_a_race_gets_no_race_projection_and_practice_does(self, client):
        race = client.get(f"/api/session/{RACE}").json()["compounds"]
        practice = client.get(f"/api/session/{PRACTICE}").json()["compounds"]
        assert all(v["race_projection"] is None for v in race.values())
        assert any(v["race_projection"] is not None for v in practice.values())

    def test_the_race_projection_brackets_its_own_point_estimate(self, client):
        for estimate in client.get(f"/api/session/{PRACTICE}").json()["compounds"].values():
            projection = estimate["race_projection"]
            if projection is None:
                continue
            low, high = projection["interval"]
            assert low <= high
            assert projection["calibrated_on"]["events"] > 0

    def test_runs_carry_a_curve_shape(self, client):
        runs = client.get(f"/api/session/{RACE}/runs").json()
        assert runs
        shaped = [r for r in runs if r["curve"]]
        assert shaped, "no stint long enough to fit a curve, which would be a data problem"
        assert {r["curve"]["regime"] for r in shaped} <= {
            "linear", "warm-up", "cliff", "recovery",
        }
        for run in shaped:
            assert run["curve"]["description"].endswith(".")

    def test_decompose_run_explains_a_real_stint(self, client):
        run = client.get(f"/api/session/{RACE}/runs").json()[0]
        response = client.get(
            f"/api/session/{RACE}/decompose-run",
            params={"driver": run["driver"], "run_id": run["run_id"]},
        )
        assert response.status_code == 200, response.text
        assert_json_is_strict(response.json())

    def test_counterfactual_answers_for_a_driver_and_lap(self, client):
        run = client.get(f"/api/session/{RACE}/runs").json()[0]
        response = client.get(
            f"/api/session/{RACE}/counterfactual",
            params={"driver": run["driver"], "lap": int(run["last_lap"])},
        )
        assert response.status_code == 200, response.text
        assert_json_is_strict(response.json())


class TestSupportingRoutes:
    @pytest.mark.parametrize("route", ["/api/business", "/api/cross-industry"])
    def test_route_answers(self, client, route):
        response = client.get(route)
        assert response.status_code == 200, response.text
        assert_json_is_strict(response.json())

    def test_retrieval_returns_something_for_a_question_about_the_project(self, client):
        response = client.get("/api/ask", params={"q": "how is fuel separated from degradation"})
        assert response.status_code == 200, response.text
        assert_json_is_strict(response.json())


class TestLiveReplay:
    def test_the_websocket_streams_laps_and_finishes(self, client):
        """The live path, end to end. This is where the NaN defect lived: the
        frame is built by hand rather than by a response model, so nothing else
        would have caught an unserialisable field."""
        seen_laps = 0
        with client.websocket_connect(f"/ws/replay/{RACE}?speed=0") as ws:
            # The full session is hundreds of frames and the contract is proved
            # long before the end, so this stops early rather than replaying a
            # whole race inside the test suite.
            while seen_laps < 80:
                message = ws.receive_json()
                if message["type"] == "lap":
                    seen_laps += 1
                    assert message["state"]["estimate_type"] == "filtered"
                    assert_json_is_strict(message)
                elif message["type"] == "complete":
                    assert_json_is_strict(message)
                    break
        assert seen_laps > 0

    def test_the_stream_reports_its_own_interval_coverage(self, client):
        """The number that makes the live interval auditable in flight."""
        last = None
        seen = 0
        with client.websocket_connect(f"/ws/replay/{RACE}?speed=0") as ws:
            while seen < 80:
                message = ws.receive_json()
                if message["type"] == "lap":
                    last = message["state"]
                    seen += 1
                elif message["type"] == "complete":
                    break
        assert last is not None
        assert 0.0 <= last["interval_coverage"] <= 1.0
