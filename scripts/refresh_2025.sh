#!/usr/bin/env bash
# Wait for the corpus scrape to finish, then bring the reference data up to date
# and re-run everything that depends on it.
#
# The reference files are not optional extras. Without session conditions the
# wet-session guard cannot exclude a race that ran two thirds on wets; without
# lineups the driver effect cannot difference out the car; without circuit
# features the transfer test has no descriptors to score.
set -u
PY=.venv/Scripts/python.exe
LOG_DIR="${1:-.}"

echo "=== waiting for the scrape  $(date +%H:%M:%S)"
while pgrep -f build_corpus > /dev/null 2>&1 || \
      powershell -NoProfile -Command \
        "if (Get-CimInstance Win32_Process -Filter \"Name like 'python%'\" | Where-Object { \$_.CommandLine -match 'build_corpus' }) { exit 0 } else { exit 1 }" 2>/dev/null; do
  sleep 60
done
echo "=== scrape done  $(date +%H:%M:%S)"

run () {
  local name="$1"; shift
  echo "=== $name  $(date +%H:%M:%S)"
  if "$PY" -u "$@" > "$LOG_DIR/$name.log" 2>&1; then echo "    ok"; else echo "    FAILED"; fi
}

run conditions scripts/build_session_conditions.py --years 2025 --sessions R FP2 --delay 3.0
run lineups    scripts/build_lineups.py --years 2025 --delay 2.0
run circuits   scripts/build_circuit_features.py --year 2025 --geometry-only --delay 1.5

echo "=== reference data refreshed  $(date +%H:%M:%S)"
