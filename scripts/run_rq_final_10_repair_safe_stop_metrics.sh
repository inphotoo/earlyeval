#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/home/ugproj/anaconda3/envs/swebench/bin/python}"
RUN_DIR="${RUN_DIR:-paper/experiments/rq_final_lightgbm_17}"
DRY_RUN="${DRY_RUN:-0}"
EXTRA_ARGS=()

if [[ "${DRY_RUN}" == "1" ]]; then
  EXTRA_ARGS+=(--dry-run)
else
  EXTRA_ARGS+=(--refresh-lightgbm-summary)
  EXTRA_ARGS+=(--audit-json "${RUN_DIR}/safe_stop_metric_repair_manifest.json")
fi

cd "$(dirname "$0")/.."

"${PYTHON_BIN}" -m final3.experiments.repair_safe_stop_metrics \
  --root "${RUN_DIR}" \
  --config configs/rq_final.yaml \
  --output-dir "${RUN_DIR}" \
  "${EXTRA_ARGS[@]}"
