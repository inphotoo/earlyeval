#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
RUN_DIR="${RUN_DIR:-paper/experiments/rq_final_lightgbm_17}"
RUN_SUBDIR="${RUN_SUBDIR:-sweverify_ablation_smoke}"
PROFILES="${PROFILES:-feature_groups}"
VARIANTS="${VARIANTS:-i no_task_tfidf}"
MAX_FOLDS="${MAX_FOLDS:-1}"
SMOKE_TRAJECTORIES_PER_SPLIT="${SMOKE_TRAJECTORIES_PER_SPLIT:-8}"
THREADS="${THREADS:-2}"
EXTRA_ARGS=()

if [[ "${FORCE:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--force)
fi

cd "$(dirname "$0")/.."

SWE_MAX_CPU_THREADS="${THREADS}" \
OMP_NUM_THREADS="${THREADS}" \
OPENBLAS_NUM_THREADS="${THREADS}" \
MKL_NUM_THREADS="${THREADS}" \
NUMEXPR_NUM_THREADS="${THREADS}" \
"${PYTHON_BIN}" -m final3.experiments.rq_final_ablation \
  --config configs/rq_final.yaml \
  --output-dir "${RUN_DIR}" \
  --run-subdir "${RUN_SUBDIR}" \
  --profiles ${PROFILES} \
  --variants ${VARIANTS} \
  --max-folds "${MAX_FOLDS}" \
  --smoke-trajectories-per-split "${SMOKE_TRAJECTORIES_PER_SPLIT}" \
  --threads "${THREADS}" \
  --policy-min-steps 0 \
  --consecutive 1 \
  --success-thresholds 0.00 \
  --failure-thresholds 0.00 \
  --score-modes calibrated \
  --min-valid-decision-acc 0.0 \
  --fallback-min-save-pct 0.0 \
  --execute \
  "${EXTRA_ARGS[@]}"
