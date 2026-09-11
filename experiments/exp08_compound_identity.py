"""Experiment 08 -- is a tyre label a compound, or is it just a colour?

Pirelli nominates three of the C1-C5 range per Grand Prix and calls them HARD,
MEDIUM and SOFT for that weekend. Every tyre model we surveyed, including ours,
pools on those labels. But "MEDIUM" is C4 at Monza and C2 at Silverstone -- it is
not one compound, it is a position in a weekend's ranking.

If that distinction is real, pooling on the compound actually fitted should
transfer better than pooling on the label. If it is not -- if the label captures
whatever matters, perhaps because Pirelli nominates to equalise behaviour -- then
we have been right by accident and should say so.

Two tests, deliberately different in kind:

  A. Variance decomposition. Group the per-event rate estimates by label, then by
     compound. If compound identity is the better organising principle, the
     spread WITHIN a compound is smaller than the spread within a label.

  B. Leave-one-event-out prediction. For a held-out race, predict each stint's
     degradation rate from the other races, once using the mean of that LABEL and
     once using the mean of that COMPOUND. This is the test that matters, because
     it is out of sample and it is the same shape as predicting a circuit we have
     never run.

    python experiments/exp08_compound_identity.py
    python experiments/exp08_compound_identity.py --year 2024

Writes experiments/results/exp08_compound_identity.json.
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics as st
import time
import warnings
from datetime import UTC, datetime
from pathlib import Path

from tyremind.data.compounds import load_allocations
from tyremind.data.corpus import read_lap_table
from tyremind.models.ssm.tyre_ssm import fit_tyre_ssm

CORPUS = Path("data/season")
RESULTS = Path(__file__).parent / "results" / "exp08_compound_identity.json"

#: A stint estimate is only worth pooling if the session actually saw the
#: compound run. Below this the posterior is mostly prior, and including it would
#: flatter both predictors equally while adding noise to neither's credit.
MIN_LAPS_PER_COMPOUND = 25


def fit_events(years: list[int], session: str) -> list[dict]:
    """Fit every corpus session in `years`, returning per-label rate estimates.

    An event without a Pirelli nomination on file still yields a usable rate --
    it just cannot contribute to the compound-versus-label comparison. Skipping
    it entirely would throw away the circuit coverage that experiment 09 needs,
    so the allocation is attached where known and left null where it is not.
    """
    manifest = json.loads((CORPUS / "corpus.json").read_text())
    entries = sorted(
        (e for e in manifest if e["year"] in years and e["session"] == session),
        key=lambda e: (e["year"], e["round_number"]),
    )
    allocations = load_allocations()

    rows: list[dict] = []
    for entry in entries:
        key = (entry["year"], entry["round_number"])
        allocation = allocations.get(key)

        parquet = CORPUS / f"{entry['session_id']}.parquet"
        if not parquet.exists():
            continue
        lap_table = read_lap_table(parquet)

        started = time.perf_counter()
        try:
            fit = fit_tyre_ssm(lap_table)
        except Exception as exc:  # noqa: BLE001 - a failed fit is data about the session
            print(f"  {entry['event_name']:<30} skip  {type(exc).__name__}: {exc}")
            continue

        laps_per_compound = lap_table.groupby("compound").size().to_dict()
        kept = 0
        for label, (rate, sd) in fit.compound_rates().items():
            compound_id = allocation.compound_id(label) if allocation else None
            n_laps = int(laps_per_compound.get(label, 0))
            if n_laps < MIN_LAPS_PER_COMPOUND:
                continue
            rows.append(
                {
                    "year": entry["year"],
                    "circuit": entry["location"],
                    "round": entry["round_number"],
                    "event": entry["event_name"],
                    "label": label,
                    "compound_id": compound_id,
                    "rate": float(rate),
                    "sd": float(sd),
                    "n_laps": n_laps,
                }
            )
            kept += 1
        nomination = (
            "/".join(allocation.compound_id(x) for x in ("HARD", "MEDIUM", "SOFT"))
            if allocation else "no nomination on file"
        )
        print(
            f"  {entry['year']} {entry['event_name']:<30} {kept} compounds  "
            f"{nomination:<22} {time.perf_counter() - started:.1f}s"
        )
    return rows


def variance_decomposition(rows: list[dict]) -> dict:
    """Spread of rate estimates within label groups against within compound groups."""
    rows = [r for r in rows if r["compound_id"]]

    def within_group_sd(key: str) -> tuple[float, int]:
        groups: dict[str, list[float]] = {}
        for r in rows:
            groups.setdefault(r[key], []).append(r["rate"])
        usable = [v for v in groups.values() if len(v) >= 3]
        if not usable:
            return float("nan"), 0
        # Pooled within-group standard deviation, weighted by group size.
        num = sum((len(v) - 1) * st.variance(v) for v in usable)
        den = sum(len(v) - 1 for v in usable)
        return (num / den) ** 0.5, len(usable)

    label_sd, n_label = within_group_sd("label")
    compound_sd, n_compound = within_group_sd("compound_id")
    overall = st.pstdev([r["rate"] for r in rows]) if len(rows) > 1 else float("nan")

    return {
        "overall_sd": overall,
        "within_label_sd": label_sd,
        "within_compound_sd": compound_sd,
        "n_label_groups": n_label,
        "n_compound_groups": n_compound,
        "compound_is_tighter": bool(compound_sd < label_sd),
        "reduction_pct": float(100.0 * (label_sd - compound_sd) / label_sd) if label_sd else 0.0,
    }


def leave_one_event_out(rows: list[dict]) -> dict:
    """Predict a held-out race's rates from the others, by label and by compound.

    Restricted to stints whose event has a nomination on file. Without this an
    unallocated row carries compound_id None, and pooling on None would silently
    group every unallocated stint into one bogus "compound".
    """
    rows = [r for r in rows if r["compound_id"]]
    events = sorted({(r["year"], r["round"]) for r in rows})
    per_case: list[dict] = []

    for held in events:
        train = [r for r in rows if (r["year"], r["round"]) != held]
        test = [r for r in rows if (r["year"], r["round"]) == held]
        if not train or not test:
            continue

        def group_mean(key: str, value: str, pool: list[dict] = train) -> float | None:
            vals = [r["rate"] for r in pool if r[key] == value]
            return st.fmean(vals) if vals else None

        for r in test:
            by_label = group_mean("label", r["label"])
            by_compound = group_mean("compound_id", r["compound_id"])
            if by_label is None or by_compound is None:
                # Only score cases where BOTH predictors are available, so the
                # comparison is like for like rather than a coverage artefact.
                continue
            per_case.append(
                {
                    "event": r["event"],
                    "label": r["label"],
                    "compound_id": r["compound_id"],
                    "actual": r["rate"],
                    "pred_label": by_label,
                    "pred_compound": by_compound,
                    "err_label": abs(by_label - r["rate"]),
                    "err_compound": abs(by_compound - r["rate"]),
                }
            )

    if not per_case:
        return {"n": 0}

    mae_label = st.fmean(c["err_label"] for c in per_case)
    mae_compound = st.fmean(c["err_compound"] for c in per_case)
    wins = sum(1 for c in per_case if c["err_compound"] < c["err_label"])

    # The errors are paired -- same held-out stint, two predictors -- so a paired
    # test is the right one, and a non-parametric one because rate errors are not
    # remotely Gaussian. Without this a 6% gap reads as a finding when it is noise.
    from scipy import stats

    stat, p_value = stats.wilcoxon(
        [c["err_compound"] for c in per_case], [c["err_label"] for c in per_case]
    )

    return {
        "n": len(per_case),
        "mae_by_label": mae_label,
        "mae_by_compound": mae_compound,
        "mean_paired_difference": st.fmean(
            c["err_compound"] - c["err_label"] for c in per_case
        ),
        "wilcoxon_p": float(p_value),
        "significant_at_5pct": bool(p_value < 0.05),
        "improvement_pct": float(100.0 * (mae_label - mae_compound) / mae_label) if mae_label else 0.0,
        "compound_wins": wins,
        "compound_win_rate": wins / len(per_case),
        "cases": per_case,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--year", type=int, default=2024)
    parser.add_argument(
        "--years", type=int, nargs="*",
        help="fit several seasons at once; overrides --year",
    )
    parser.add_argument("--session", default="R")
    args = parser.parse_args()

    warnings.filterwarnings("ignore")
    logging.getLogger("fastf1").setLevel(logging.ERROR)

    years = args.years or [args.year]
    seasons = "/".join(str(y) for y in years)
    print(f"\nfitting {seasons} {args.session} sessions from the corpus\n")
    rows = fit_events(years, args.session)
    if len(rows) < 6:
        raise SystemExit(f"only {len(rows)} usable estimates -- corpus too small to conclude")

    decomposition = variance_decomposition(rows)
    loeo = leave_one_event_out(rows)

    print("\n" + "=" * 78)
    print(f"COMPOUND IDENTITY vs RELATIVE LABEL  ({len(rows)} stint estimates, "
          f"{len({(r['year'], r['round']) for r in rows})} events)")
    print("=" * 78)
    print("\nA. Variance decomposition -- which grouping is tighter?")
    print(f"   spread across all estimates       : {decomposition['overall_sd']:.4f} s/lap")
    print(f"   within relative label (H/M/S)     : {decomposition['within_label_sd']:.4f} s/lap"
          f"  ({decomposition['n_label_groups']} groups)")
    print(f"   within compound identity (C1-C5)  : {decomposition['within_compound_sd']:.4f} s/lap"
          f"  ({decomposition['n_compound_groups']} groups)")
    print(f"   -> compound is tighter            : {decomposition['compound_is_tighter']}"
          f"  ({decomposition['reduction_pct']:+.1f}%)")

    print("\nB. Leave-one-event-out prediction -- which grouping transfers?")
    print(f"   cases scored                      : {loeo['n']}")
    print(f"   MAE predicting by label           : {loeo['mae_by_label']:.4f} s/lap")
    print(f"   MAE predicting by compound        : {loeo['mae_by_compound']:.4f} s/lap")
    print(f"   difference (compound - label)     : {loeo['mean_paired_difference']:+.4f} s/lap")
    print(f"   compound wins                     : {loeo['compound_wins']}/{loeo['n']}"
          f"  ({loeo['compound_win_rate']:.0%})")
    print(f"   Wilcoxon paired p                 : {loeo['wilcoxon_p']:.3f}")
    print()
    if loeo["significant_at_5pct"]:
        better = "compound identity" if loeo["mae_by_compound"] < loeo["mae_by_label"] else "the relative label"
        print(f"   VERDICT: {better} transfers better, significantly.")
    else:
        print("   VERDICT: no detectable difference. Pooling on the compound actually")
        print("            fitted does NOT beat pooling on the weekend's relative label.")
        print("            A plausible mechanism: Pirelli nominates to equalise -- harder")
        print("            compounds for abrasive circuits, softer for gentle ones -- so the")
        print("            label already carries circuit severity, by design.")
    print("=" * 78)

    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    RESULTS.write_text(
        json.dumps(
            {
                "experiment": "exp08_compound_identity",
                "generated_at": datetime.now(UTC).isoformat(),
                "year": years[0] if len(years) == 1 else None,
                "years": years,
                "session": args.session,
                "min_laps_per_compound": MIN_LAPS_PER_COMPOUND,
                "n_estimates": len(rows),
                "n_events": len({(r["year"], r["round"]) for r in rows}),
                "variance_decomposition": decomposition,
                "leave_one_event_out": {k: v for k, v in loeo.items() if k != "cases"},
                "estimates": rows,
                "cases": loeo.get("cases", []),
            },
            indent=2,
        )
    )
    print(f"\nwrote {RESULTS}")


if __name__ == "__main__":
    main()
