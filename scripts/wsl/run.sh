#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

TASK="${1:-help}"
shift || true

if [ "$TASK" = "bootstrap" ]; then
  bash ./scripts/wsl/bootstrap.sh
  exit 0
fi

VENV_DIR=".venv-wsl"
if [ ! -d "$VENV_DIR" ]; then
  echo "WSL virtualenv not found. Run: bash ./scripts/wsl/bootstrap.sh"
  exit 1
fi

source "$VENV_DIR/bin/activate"
export PYTHONPATH="$ROOT_DIR/src"

case "$TASK" in
  help)
    echo "Usage: bash ./scripts/wsl/run.sh <task>"
    echo "Tasks:"
    echo "  bootstrap  - install/update dependencies in WSL venv"
    echo "  test       - run pytest"
    echo "  example    - run generic example workflow"
    echo "  mos2       - run MoS2 LMO + screening example"
    echo "  cmd ...    - run an arbitrary command inside the WSL venv"
    ;;
  test)
    python -m pytest
    ;;
  example)
    python ./examples/exciton_workflow.py
    ;;
  mos2)
    python ./examples/mos2_lmo_workflow.py
    ;;
  cmd)
    if [ "$#" -eq 0 ]; then
      echo "Usage: bash ./scripts/wsl/run.sh cmd <command...>"
      exit 1
    fi
    "$@"
    ;;
  *)
    echo "Unknown task: $TASK"
    echo "Run: bash ./scripts/wsl/run.sh help"
    exit 1
    ;;
esac
