#!/usr/bin/env bash
#
# run.sh — run the parallel labeling tool and log the full process output.
#
# Usage:
#   ./run.sh --dataset-dir <DIR> [--config config.yaml] [--output-dir <DIR>] [extra flags...]
#
# All arguments are forwarded to `python -m parallel_labeling.cli`.
# stdout and stderr are shown live AND appended to a timestamped log under logs/.

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Project convention: Bangkok timezone, %Y-%m-%d %H:%M:%S timestamps.
export TZ="Asia/Bangkok"

LOG_DIR="${LOG_DIR:-logs}"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/run_$(date '+%Y-%m-%d_%H-%M-%S').log"

# Activate the project virtualenv if present (created per CLAUDE.md setup).
if [[ -f ".venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source ".venv/bin/activate"
fi

# Always record the final outcome in the log, even on failure.
on_exit() {
  local code=$?
  if [[ $code -ne 0 ]]; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] run.sh FAILED with exit code ${code}" | tee -a "$LOG_FILE"
  fi
  exit "$code"
}
trap on_exit EXIT

{
  echo "=================================================================="
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] starting parallel labeling run"
  echo "  python : $(command -v python) ($(python --version 2>&1))"
  echo "  args   : $*"
  echo "  log    : $LOG_FILE"
  echo "=================================================================="
} | tee -a "$LOG_FILE"

# Stream stdout+stderr live to the console and append to the log.
PYTHONUNBUFFERED=1 python -m parallel_labeling.cli "$@" 2>&1 | tee -a "$LOG_FILE"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] run.sh completed successfully" | tee -a "$LOG_FILE"
