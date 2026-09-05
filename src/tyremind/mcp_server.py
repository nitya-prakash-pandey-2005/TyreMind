"""TyreMind as an MCP server: the model as a tool an AI agent can call.

Turns the estimator into something a race engineer can reach through a
conversation rather than a dashboard. Ask Claude "how is car 44's tyre doing and
should we box?" and it calls these tools, gets real posterior estimates, and
answers from them.

    python -m tyremind.mcp_server              # stdio, for Claude Desktop
    python -m tyremind.mcp_server --http       # streamable HTTP

**Seven tools, deliberately.** The most common way to ruin an MCP server is to
expose everything: a server that dumps forty tools into the context window
degrades the agent before it has done anything. These seven cover the questions
someone actually asks, and each returns a compact structured result rather than
a data dump.

**Every tool returns uncertainty.** An agent that receives "degradation is 0.113"
will state it as fact. One that receives "0.113 plus or minus 0.023, and the
model is extrapolating past what it observed" can hedge correctly. The interval
is not optional in any response here.
"""

from __future__ import annotations

import argparse
import json
import logging
from typing import Annotated

from pydantic import Field

logger = logging.getLogger(__name__)

_INSTRUCTIONS = """
TyreMind estimates Formula 1 tyre degradation, separating it from the things
that also change lap times: fuel burn-off, track evolution and traffic.

Key facts to convey accurately when using these tools:

- TyreMind estimates LATENT PERFORMANCE STATE, not physical tread depth. Public
  F1 telemetry contains no wear measurement. Never describe its output as
  measured tyre wear.
- Every estimate has a credible interval. Always report it. A degradation rate
  without its uncertainty is misleading.
- Two of the three identifying assumptions are physical priors rather than
  evidence (the fuel coefficient and the track-evolution amplitude). If a user
  asks how confident to be, say so.
- Use `assess_trust` before giving strategy advice about an unusual situation.
  Its applicability score falls when the question runs past what the data covers.
- `search_documentation` retrieves passages from the project's own research
  audit, model card and recorded experiment results. Prefer citing it over
  recalling numbers.
""".strip()


def build_server():
    """Construct the MCP server with its tools registered.

    Imports are deferred into this function so that `import tyremind.mcp_server`
    stays cheap: fitting a session pulls in the whole scientific stack, and a
    client listing tools should not pay for that.
    """
    from mcp.server.mcpserver import MCPServer

    from tyremind.api.store import SessionStore
    from tyremind.causal.counterfactual import project_tyre
    from tyremind.causal.decomposition import decompose_lap
    from tyremind.explain.narrate import decomposition_narration
    from tyremind.models.trust import assess_applicability
    from tyremind.simulate.race import DEFAULT_PIT_LOSS_S, RaceState, TyreModel, recommend

    server = MCPServer(
        name="tyremind",
        title="TyreMind — causal tyre intelligence",
        description=(
            "Estimate Formula 1 tyre degradation separated from fuel burn-off, "
            "track evolution and traffic, with calibrated uncertainty."
        ),
        instructions=_INSTRUCTIONS,
    )
    store = SessionStore()

    def _load(session_id: str):
        try:
            return store.get(session_id)
        except KeyError as exc:
            raise ValueError(
                f"Unknown session {session_id!r}. Call list_sessions first."
            ) from exc

    # ---------------------------------------------------------------- 1
    @server.tool(
        description=(
            "List the race and practice sessions TyreMind can analyse. Call this "
            "first to get valid session ids."
        )
    )
    def list_sessions() -> str:
        catalogue = store.catalogue()
        return json.dumps(
            {
                "sessions": [
                    {
                        "session_id": r.session_id,
                        "event": f"{r.year} {r.grand_prix}",
                        "session": "race" if r.session == "R" else r.session,
                        "available_offline": r.cached,
                    }
                    for r in catalogue
                ]
            },
            indent=2,
        )

    # ---------------------------------------------------------------- 2
    @server.tool(
        description=(
            "Degradation rate per tyre compound for a session, in seconds per lap, "
            "with credible intervals. Also returns what the naive lap-time-versus-"
            "tyre-age method would have said, for comparison."
        )
    )
    def get_degradation(
        session_id: Annotated[str, Field(description="Session id from list_sessions")],
    ) -> str:
        from tyremind.data.synthetic import naive_degradation_estimate

        loaded = _load(session_id)
        naive = naive_degradation_estimate(loaded.lap_table)
        fuel_mean, fuel_sd = loaded.fit.fuel_slope()

        return json.dumps(
            {
                "session": loaded.ref.label,
                "compounds": {
                    compound: {
                        "degradation_s_per_lap": round(mean, 4),
                        "uncertainty_sd": round(sd, 4),
                        "credible_interval_95": [
                            round(mean - 1.96 * sd, 4),
                            round(mean + 1.96 * sd, 4),
                        ],
                        "naive_method_would_say": round(naive[compound], 4)
                        if compound in naive
                        else None,
                        "laps_observed": int(
                            (loaded.lap_table["compound"] == compound).sum()
                        ),
                    }
                    for compound, (mean, sd) in sorted(loaded.fit.compound_rates().items())
                },
                "fuel_effect_s_per_lap": round(fuel_mean, 4),
                "note": (
                    "Latent performance state, not physical tread wear. The naive "
                    "figure is often negative on real races, which would mean tyres "
                    "getting faster with age -- it is an artefact of fuel burn-off."
                ),
            },
            indent=2,
        )

    # ---------------------------------------------------------------- 3
    @server.tool(
        description=(
            "Explain why a specific lap was faster or slower than earlier in the "
            "stint, splitting the change into tyre degradation, fuel burn-off, "
            "track evolution and traffic."
        )
    )
    def explain_lap(
        session_id: Annotated[str, Field(description="Session id")],
        driver: Annotated[str, Field(description="Driver code, e.g. VER, HAM, NOR")],
        lap: Annotated[int, Field(description="Session lap number to explain")],
    ) -> str:
        loaded = _load(session_id)
        try:
            decomposition = decompose_lap(loaded.fit, driver.upper(), lap)
        except ValueError as exc:
            raise ValueError(str(exc)) from exc

        return json.dumps(
            {
                **decomposition.to_dict(),
                "plain_language": decomposition_narration(decomposition).text,
            },
            indent=2,
        )

    # ---------------------------------------------------------------- 4
    @server.tool(
        description=(
            "Project how many competitive laps a tyre has left, with the "
            "probability it has passed a performance threshold at each horizon."
        )
    )
    def project_tyre_life(
        session_id: Annotated[str, Field(description="Session id")],
        driver: Annotated[str, Field(description="Driver code")],
        lap: Annotated[int, Field(description="Lap to project from")],
        threshold_s: Annotated[
            float,
            Field(description="Seconds per lap slower than fresh at which the tyre is done"),
        ] = 0.8,
    ) -> str:
        loaded = _load(session_id)
        try:
            projection = project_tyre(
                loaded.fit, driver.upper(), lap, horizon=15, threshold=threshold_s
            )
        except ValueError as exc:
            raise ValueError(str(exc)) from exc

        lower, upper = projection.competitive_life_interval()
        return json.dumps(
            {
                "driver": projection.driver,
                "compound": projection.compound,
                "tyre_age_laps": projection.tyre_age,
                "competitive_laps_remaining": round(projection.competitive_life(), 1),
                "range": [round(lower, 1), round(upper, 1)],
                "threshold_s_per_lap": threshold_s,
                "breach_probability_by_horizon": {
                    f"+{int(h)}": round(float(p), 3)
                    for h, p in zip(
                        projection.horizon[:10],
                        projection.breach_probability[:10],
                        strict=False,
                    )
                },
                "applicability_at_horizon": {
                    f"+{int(h)}": round(float(a), 2)
                    for h, a in zip(
                        projection.horizon[:10], projection.applicability[:10], strict=False
                    )
                },
                "note": (
                    "Applicability falls below 1.0 once the projection reaches past "
                    "the oldest tyre age observed. Below 0.5 the model is "
                    "extrapolating a trend, and a cliff outside the observed range "
                    "cannot be seen."
                ),
            },
            indent=2,
        )

    # ---------------------------------------------------------------- 5
    @server.tool(
        description=(
            "Recommend whether to pit and when, by simulating the rest of the race "
            "several thousand times with the degradation rate drawn from its "
            "posterior. Returns the recommendation, how often it wins, and why."
        )
    )
    def recommend_strategy(
        session_id: Annotated[str, Field(description="Session id")],
        driver: Annotated[str, Field(description="Driver code")],
        lap: Annotated[int, Field(description="Lap to decide from")],
    ) -> str:
        loaded = _load(session_id)
        lap_table = loaded.lap_table

        row = lap_table[
            (lap_table["driver"] == driver.upper()) & (lap_table["session_lap"] == lap)
        ]
        if row.empty:
            raise ValueError(f"{driver.upper()} has no valid lap {lap} in this session")
        row = row.iloc[0]

        reference = float(lap_table["lap_time"].median())
        tyres = {}
        for compound, (rate, sd) in loaded.fit.compound_rates().items():
            laps = lap_table[lap_table["compound"] == compound]
            fresh = laps[laps["tyre_age"] <= 3]
            tyres[compound] = TyreModel(
                compound=compound,
                base_pace_s=float(fresh["lap_time"].median() - reference)
                if len(fresh) >= 3
                else 0.0,
                degradation_rate=float(rate),
                degradation_rate_sd=float(sd),
                cliff_lap=max(float(laps["tyre_age"].max()) * 0.85, 8.0),
            )

        state = RaceState(
            current_lap=int(lap),
            total_laps=int(lap_table["session_lap"].max()),
            position=0,
            current_compound=str(row["compound"]),
            current_tyre_age=float(row["tyre_age"]),
            gap_ahead_s=float(2.0 - 1.6 * float(row.get("traffic_index", 0.0))),
            gap_behind_s=2.0,
            base_lap_time_s=reference,
            pit_loss_s=DEFAULT_PIT_LOSS_S,
        )
        result = recommend(state, tyres, n_sims=4000)

        return json.dumps(
            {
                "recommendation": result.best.strategy.label,
                "pit_on_lap": result.best.strategy.pit_lap,
                "wins_against_next_best": round(result.decision_confidence, 2),
                "margin_seconds": round(result.margin_s, 2),
                "reasons": result.reasons,
                "options": [
                    {
                        "label": o.strategy.label,
                        "expected_time_s": round(o.expected_time, 1),
                        "worse_case_s": round(o.downside, 1),
                        "runs_past_cliff": round(o.ran_out_of_tyre, 2),
                    }
                    for o in result.alternatives
                ],
                "note": "Simulated estimate of a race that has not happened.",
            },
            indent=2,
        )

    # ---------------------------------------------------------------- 6
    @server.tool(
        description=(
            "Check whether the model should be trusted about a situation: how far "
            "the question sits outside the data actually observed. Call this before "
            "giving confident advice about an unusual case."
        )
    )
    def assess_trust(
        session_id: Annotated[str, Field(description="Session id")],
        compound: Annotated[str, Field(description="Compound, e.g. SOFT, MEDIUM, HARD")],
        tyre_age: Annotated[float, Field(description="Tyre age being asked about, in laps")],
    ) -> str:
        loaded = _load(session_id)
        report = assess_applicability(
            loaded.lap_table, compound=compound.upper(), tyre_age=tyre_age
        )
        return json.dumps(
            {
                "applicability": round(report.applicability, 2),
                "risk": report.risk,
                "reasons": report.reasons,
                "interpretation": (
                    "Above 0.7 the question sits inside the observed data. Below 0.4 "
                    "the model is extrapolating and its intervals understate the risk."
                ),
            },
            indent=2,
        )

    # ---------------------------------------------------------------- 7
    @server.tool(
        description=(
            "Search TyreMind's own research audit, model card, limitations register "
            "and recorded experiment results. Use this to answer questions about "
            "method, validation, accuracy or limitations, and cite what it returns "
            "rather than recalling numbers."
        )
    )
    def search_documentation(
        query: Annotated[str, Field(description="Natural-language question")],
        max_results: Annotated[int, Field(description="Passages to return", ge=1, le=8)] = 4,
    ) -> str:
        from tyremind.rag.index import build_corpus

        nonlocal _corpus
        if _corpus is None:
            _corpus = build_corpus()

        hits = _corpus.index.search(query, k=max_results)
        if not hits:
            return json.dumps(
                {
                    "query": query,
                    "results": [],
                    "note": "Nothing relevant found. Do not invent an answer.",
                }
            )

        return json.dumps(
            {
                "query": query,
                "results": [
                    {"source": h.passage.citation, "text": h.passage.text} for h in hits
                ],
                "note": "Cite the source alongside any figure taken from these passages.",
            },
            indent=2,
        )

    _corpus = None
    return server


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--http",
        action="store_true",
        help="serve over streamable HTTP instead of stdio",
    )
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING)

    server = build_server()
    if args.http:
        server.run(transport="streamable-http", port=args.port)
    else:
        server.run(transport="stdio")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
