#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
RUN_DIR="${RUN_DIR:-paper/experiments/earlyeval_lightgbm}"
cd "$(dirname "$0")/.."

"${PYTHON_BIN}" -m earlyeval.cli experiment paper-suite \
  --stage lightgbm-summary \
  --config configs/earlyeval.yaml \
  --output-dir "${RUN_DIR}"
