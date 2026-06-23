#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/home/ugproj/anaconda3/envs/swebench/bin/python}"
RUN_DIR="${RUN_DIR:-paper/experiments/rq_final_smoke}"
MAX_FOLDS="${MAX_FOLDS:-2}"
cd "$(dirname "$0")/.."

"${PYTHON_BIN}" -m final3.cli experiment rq-final \
  --stage make-splits \
  --config configs/rq_final.yaml \
  --output-dir "${RUN_DIR}" \
  --datasets sweverify \
  --max-folds "${MAX_FOLDS}"

