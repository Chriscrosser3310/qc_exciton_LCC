#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

VENV_DIR="${VENV_DIR:-.venv-linux}"
PLATFORM_LABEL="${PLATFORM_LABEL:-Linux}"

VENV_DIR="$VENV_DIR" PLATFORM_LABEL="$PLATFORM_LABEL" bash ./scripts/wsl/bootstrap.sh
