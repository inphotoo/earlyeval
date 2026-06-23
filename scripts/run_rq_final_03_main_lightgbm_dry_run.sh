#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/home/ugproj/anaconda3/envs/swebench/bin/python}"
RUN_DIR="${RUN_DIR:-paper/experiments/rq_final_lightgbm_17}"
cd "$(dirname "$0")/.."

"${PYTHON_BIN}" -m final3.cli experiment rq-final \
  --stage lightgbm-main \
  --config configs/rq_final.yaml \
  --output-dir "${RUN_DIR}"

