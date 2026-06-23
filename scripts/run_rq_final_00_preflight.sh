#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
cd "$(dirname "$0")/.."

"${PYTHON_BIN}" -m final3.cli check preflight \
  --experiment all \
  --output-dir paper/checks/preflight_all

