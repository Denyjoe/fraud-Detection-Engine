#!/usr/bin/env bash
# One-command setup for the pamostore Fraud Detection Engine (macOS / Linux).
#
#   1. Verifies Python 3.10+ is installed.
#   2. Creates an isolated virtual environment in .venv
#   3. Installs dependencies (CPU-only PyTorch on Linux to save ~2 GB)
#   4. Pre-trains the cold-start fraud model (models/fraud_model.joblib)
#
# Usage:
#   bash setup.sh            # normal setup
#   bash setup.sh --retrain  # force re-training of the fraud model
#   USE_GPU=1 bash setup.sh  # Linux: install the default (CUDA) PyTorch build
set -euo pipefail
cd "$(dirname "$0")"

MIN_MAJOR=3
MIN_MINOR=10
VENV_DIR=".venv"
VENV_PY="$VENV_DIR/bin/python"
MODEL_FILE="models/fraud_model.joblib"
TORCH_CPU_INDEX="https://download.pytorch.org/whl/cpu"
RETRAIN=0
[[ "${1:-}" == "--retrain" ]] && RETRAIN=1

# Print a formatted step header. Args: $1 = message
step() { printf '\n\033[36m==> %s\033[0m\n' "$1"; }

# Print an error and exit. Args: $1 = message explaining the fix
fail() { printf '\n\033[31mERROR: %s\033[0m\n' "$1" >&2; exit 1; }

# Echo the first python command that is version 3.10+, or nothing.
find_python() {
  for candidate in python3 python; do
    command -v "$candidate" >/dev/null 2>&1 || continue
    if "$candidate" -c "import sys; sys.exit(0 if sys.version_info >= ($MIN_MAJOR, $MIN_MINOR) else 1)" 2>/dev/null; then
      echo "$candidate"
      return
    fi
  done
}

step "Checking for Python $MIN_MAJOR.$MIN_MINOR or newer"
PYTHON="$(find_python)"
[[ -n "$PYTHON" ]] || fail "Python $MIN_MAJOR.$MIN_MINOR+ not found. Install it (https://www.python.org/downloads/ or your package manager) and re-run."
echo "Using: $("$PYTHON" --version)"

step "Creating virtual environment in $VENV_DIR"
if [[ -x "$VENV_PY" ]]; then
  echo "Virtual environment already exists - reusing it."
else
  "$PYTHON" -m venv "$VENV_DIR" || fail "Could not create the virtual environment (on Debian/Ubuntu: sudo apt install python3-venv)."
fi

step "Installing dependencies (first run downloads ~1 GB, please be patient)"
"$VENV_PY" -m pip install --upgrade pip || fail "Could not upgrade pip. Check your internet connection."
if [[ "$(uname -s)" == "Linux" && "${USE_GPU:-0}" != "1" ]]; then
  echo "Installing CPU-only PyTorch (set USE_GPU=1 to skip this)."
  "$VENV_PY" -m pip install torch --index-url "$TORCH_CPU_INDEX" || fail "PyTorch installation failed."
fi
"$VENV_PY" -m pip install -r requirements.txt || fail "Dependency installation failed. Check your connection and free disk space (~3 GB)."

step "Preparing the fraud model"
if [[ -f "$MODEL_FILE" && "$RETRAIN" -eq 0 ]]; then
  echo "Model already exists at $MODEL_FILE - skipping training (use --retrain to rebuild)."
else
  "$VENV_PY" generate_synthetic_data.py || fail "Model pre-training failed. See the log output above."
fi

printf '\n\033[32mSetup complete!\033[0m\n'
echo "Start the server with:   bash run.sh"
echo "Then open:               http://127.0.0.1:8000"
