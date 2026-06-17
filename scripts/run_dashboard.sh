#!/usr/bin/env bash
# Launch the +EV Soccer Streamlit dashboard.
#
# Usage:
#   ./scripts/run_dashboard.sh
#
# Environment:
#   PORT  - override the default 8501
#   HEADLESS - "true" (default, safe for servers) or "false" (auto-open browser)
set -euo pipefail

# Resolve project root from this script's location.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

# Activate venv if it exists and we're not already inside it.
if [[ -z "${VIRTUAL_ENV:-}" && -f "${PROJECT_ROOT}/.venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "${PROJECT_ROOT}/.venv/bin/activate"
fi

PORT="${PORT:-8501}"
HEADLESS="${HEADLESS:-true}"

exec streamlit run dashboard/app.py \
  --server.port "${PORT}" \
  --server.headless "${HEADLESS}" \
  --server.address 0.0.0.0 \
  --browser.gatherUsageStats false
