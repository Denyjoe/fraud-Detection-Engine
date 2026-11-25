#!/usr/bin/env bash
# Starts the pamostore Fraud Detection Engine (macOS / Linux).
#
# Usage:
#   bash run.sh                 # http://127.0.0.1:8000
#   PORT=8080 bash run.sh       # custom port
#   HOST=0.0.0.0 bash run.sh    # expose on your network (trusted networks only:
#                               # the API has no authentication yet)
set -euo pipefail
cd "$(dirname "$0")"

VENV_PY=".venv/bin/python"
PORT="${PORT:-8000}"
HOST="${HOST:-127.0.0.1}"

if [[ ! -x "$VENV_PY" ]]; then
  echo "ERROR: Virtual environment not found. Run setup first:  bash setup.sh" >&2
  exit 1
fi

if [[ ! -f "models/fraud_model.joblib" ]]; then
  echo "No trained model found - training one now (takes a few seconds)..."
  "$VENV_PY" generate_synthetic_data.py
fi

echo "Starting server on http://$HOST:$PORT  (press Ctrl+C to stop)"
echo "First start downloads the vision model (~600 MB) and can take a few minutes."
exec "$VENV_PY" -m uvicorn main:app --host "$HOST" --port "$PORT"
