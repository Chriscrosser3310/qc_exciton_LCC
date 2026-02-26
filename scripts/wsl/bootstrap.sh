#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

VENV_DIR="${VENV_DIR:-.venv-wsl}"
PLATFORM_LABEL="${PLATFORM_LABEL:-WSL}"
PY_BIN="${PY_BIN:-python3}"

if ! command -v "$PY_BIN" >/dev/null 2>&1; then
  echo "Error: $PY_BIN not found in $PLATFORM_LABEL. Install Python 3 first."
  exit 1
fi

if [ ! -d "$VENV_DIR" ]; then
  "$PY_BIN" -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip
python -m pip install -e ".[chem,dev,qiskit]"

echo "$PLATFORM_LABEL environment ready at $ROOT_DIR/$VENV_DIR"
echo "Run: VENV_DIR=$VENV_DIR PLATFORM_LABEL=$PLATFORM_LABEL bash ./scripts/wsl/run.sh mos2"
