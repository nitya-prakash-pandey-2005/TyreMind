#!/usr/bin/env bash
# Re-run every experiment that touches the corpus, in dependency order.
#
# Order matters in two places and getting it wrong produces numbers that
# disagree with each other rather than an error:
#   exp09 and exp10 read exp08's estimates
#   exp10 and exp12 read exp03's comparisons
#
# Cheap experiments that only re-analyse a cached artefact are run last, so a
# failure early does not waste an hour of fitting.
set -u
PY=.venv/Scripts/python.exe
LOG_DIR="${1:-.}"

run () {
  local name="$1"; shift
  echo "=== $name  $(date +%H:%M:%S)"
  if "$PY" -u "$@" > "$LOG_DIR/$name.log" 2>&1; then
    echo "    ok"
  else
    echo "    FAILED (see $LOG_DIR/$name.log)"
  fi
}

run exp03 experiments/exp03_practice_to_race.py --years 2025 2024 2023 --all
run exp08 experiments/exp08_compound_identity.py --years 2025 2024 2023 2022
run exp09 experiments/exp09_circuit_transfer.py
run exp10 experiments/exp10_bias_mechanism.py
run exp12 experiments/exp12_conformal_intervals.py
run exp18 experiments/exp18_identifiability_bound.py --limit 12
run exp14 experiments/exp14_naive_failure_rate.py
run exp15 experiments/exp15_driver_effect.py --refit
run exp17 experiments/exp17_degradation_cliff.py --refit
run exp11 experiments/exp11_depth_matched.py --years 2025 2024 2023
run exp05 experiments/exp05_model_ladder.py --corpus season --limit 20
run exp13 experiments/exp13_lap_time_calibration.py --corpus season --limit 20
run exp16 experiments/exp16_calibration_shape.py --limit 10

echo "=== done  $(date +%H:%M:%S)"
