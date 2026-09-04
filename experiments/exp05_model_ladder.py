"""Experiment 05 -- the model ladder, scored two ways.

Runs every rung through identical chronological folds on real sessions, then
scores degradation-rate recovery on synthetic sessions where the truth is known.

The two tables usually disagree, and that disagreement is the point. A model can
top the lap-time table while having nothing to say about tyres.

    python experiments/exp05_model_ladder.py

Writes experiments/results/exp05_model_ladder.json.
"""

from __future__ import annotations

import argparse
import json
import logging
import warnings
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from tyremind.data.synthetic import SessionConfig, generate_session
from tyremind.models.baselines import model_ladder
from tyremind.models.evaluation import evaluate_ladder, score_rate_recovery

RESULTS = Path(__file__).parent / "results" / "exp05_model_ladder.json"
DEMO_DIR = Path("data/demo")


def load_sessions(session_ids: list[str]) -> dict[str, pd.DataFrame]:
    out = {}
    for session_id in session_ids:
        path = DEMO_DIR / f"{session_id}.parquet"
        if path.exists():
            out[session_id] = pd.read_parquet(path)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sessions",
        nargs="*",
        default=["2024-monza-R", "2024-silverstone-R", "2024-zandvoort-R", "2024-barcelona-R"],
    )
    parser.add_argument("--n-seeds", type=int, default=6)
    parser.add_argument("--n-folds", type=int, default=4)
    args = parser.parse_args()

    warnings.filterwarnings("ignore")
    logging.getLogger("fastf1").setLevel(logging.ERROR)

    sessions = load_sessions(args.sessions)
    if not sessions:
        raise SystemExit(
            f"no cached sessions found in {DEMO_DIR}. Run scripts/build_demo.py first."
        )

    # ---------------- lap-time prediction, real data ----------------
    per_session = {}
    for session_id, lap_table in sessions.items():
        print(f"\n  {session_id} ({len(lap_table)} laps)")
        scores = evaluate_ladder(lap_table, model_ladder(), n_folds=args.n_folds)
        per_session[session_id] = [s.to_dict() for s in scores]
        for s in scores:
            status = f"FAILED {s.failed}" if s.failed else (
                f"CRPS {s.crps:.4f}  MAE {s.mae:.4f}  coverage {s.coverage_95:.0%}"
            )
            print(f"    {s.model:<34} {status}")

    combined = []
    for model_name in [m.name for m in model_ladder()]:
        rows = [
            s
            for session in per_session.values()
            for s in session
            if s["model"] == model_name and not s["failed"]
        ]
        if not rows:
            continue
        combined.append(
            {
                "model": model_name,
                "crps": float(np.mean([r["crps"] for r in rows])),
                "mae": float(np.mean([r["mae"] for r in rows])),
                "rmse": float(np.mean([r["rmse"] for r in rows])),
                "coverage_95": float(np.mean([r["coverage_95"] for r in rows])),
                "interval_width_95": float(np.mean([r["interval_width_95"] for r in rows])),
                "bias": float(np.mean([r["bias"] for r in rows])),
                "bias_drift": float(np.mean([r["bias_drift"] for r in rows])),
                "fit_seconds": float(np.mean([r["fit_seconds"] for r in rows])),
                "n_sessions": len(rows),
            }
        )
    combined.sort(key=lambda r: r["crps"])

    print("\n" + "=" * 92)
    print("LAP-TIME PREDICTION -- real sessions, expanding-window chronological folds")
    print("=" * 92)
    print(
        f"{'model':<34}{'CRPS':>9}{'MAE':>9}{'cover95':>10}{'width':>8}{'bias':>9}{'drift':>9}"
    )
    for row in combined:
        print(
            f"{row['model']:<34}{row['crps']:>9.4f}{row['mae']:>9.4f}"
            f"{row['coverage_95']:>9.0%}{row['interval_width_95']:>8.2f}"
            f"{row['bias']:>+9.3f}{row['bias_drift']:>+9.3f}"
        )
    print(
        "\nbias drift = how much a model's error grows from the first fold to the last.\n"
        "Each fold forecasts further past its training window, so this measures whether\n"
        "a model can extrapolate the fuel trend or has merely memorised it."
    )

    # ---------------- degradation recovery, synthetic ----------------
    recovery_frames = []
    for i in range(args.n_seeds):
        session = generate_session(SessionConfig(seed=5000 + i))
        recovery_frames.append(
            score_rate_recovery(model_ladder(), session.lap_table, session.truth.compound_rates)
        )
    recovery = pd.concat(recovery_frames, ignore_index=True)

    scored = recovery.dropna(subset=["error"])
    summary = (
        scored.assign(abs_error=scored["error"].abs())
        .groupby("model")
        .agg(
            rate_mae=("abs_error", "mean"),
            rate_bias=("error", "mean"),
            coverage=("covered_95", "mean"),
            n=("error", "size"),
        )
        .reset_index()
        .sort_values("rate_mae")
    )

    no_parameter = sorted(
        recovery[recovery["compound"].isna()]["model"].dropna().unique().tolist()
    )

    print("\n" + "=" * 92)
    print(f"DEGRADATION RECOVERY -- {args.n_seeds} synthetic sessions with a known true rate")
    print("=" * 92)
    print(f"{'model':<34}{'rate MAE':>11}{'bias':>11}{'cover95':>10}{'n':>6}")
    for row in summary.itertuples(index=False):
        print(
            f"{row.model:<34}{row.rate_mae:>11.4f}{row.rate_bias:>+11.4f}"
            f"{row.coverage:>9.0%}{row.n:>6d}"
        )
    for model in no_parameter:
        print(f"{model:<34}{'—':>11}{'—':>11}{'—':>10}{'—':>6}  no degradation parameter")
    print("=" * 92)

    if combined and not summary.empty:
        best_laptime = combined[0]["model"]
        best_rate = summary.iloc[0]["model"]
        if best_laptime != best_rate:
            print(
                f"\nBest lap-time predictor is {best_laptime!r};"
                f" best degradation estimator is {best_rate!r}."
            )
            print("Predicting lap times well is not the same as understanding the tyre.")

    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    RESULTS.write_text(
        json.dumps(
            {
                "experiment": "exp05_model_ladder",
                "generated_at": datetime.now(UTC).isoformat(),
                "sessions": list(sessions),
                "n_synthetic_seeds": args.n_seeds,
                "lap_time_prediction": combined,
                "degradation_recovery": summary.to_dict(orient="records"),
                "models_without_degradation_parameter": no_parameter,
                "per_session": per_session,
            },
            indent=2,
            default=str,
        )
    )
    print(f"\nwrote {RESULTS}")


if __name__ == "__main__":
    main()
