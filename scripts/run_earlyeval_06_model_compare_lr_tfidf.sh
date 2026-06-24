#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
RUN_DIR="${RUN_DIR:-paper/experiments/earlyeval_lightgbm}"
THREADS="${THREADS:-4}"
MAX_FOLDS="${MAX_FOLDS:-}"
MAX_INSTANCES="${MAX_INSTANCES:-500}"
SMOKE_TRAJECTORIES_PER_SPLIT="${SMOKE_TRAJECTORIES_PER_SPLIT:-0}"
VARIANTS="${VARIANTS:-dense_af tfidf_af}"
RUN_SUBDIR="${RUN_SUBDIR:-lr_tfidf_baselines}"
# Optional: restrict the run to a comma- or space-separated list of test
# models. Use this to split the held-out-agent sweep across two tmux sessions
# without races: pick a different RUN_SUBDIR for each session and pass
# disjoint TEST_MODELS lists. Per-fold output is always safe across
# processes; RUN_SUBDIR isolation is just to keep summary files separate.
TEST_MODELS="${TEST_MODELS:-}"
FORCE="${FORCE:-0}"
EXTRA_ARGS=()

if [[ -n "${MAX_FOLDS}" ]]; then
  EXTRA_ARGS+=(--max-folds "${MAX_FOLDS}")
fi
if [[ -n "${TEST_MODELS}" ]]; then
  # accept comma- or whitespace-separated list
  TEST_MODELS_ARR=( ${TEST_MODELS//,/ } )
  EXTRA_ARGS+=(--test-models "${TEST_MODELS_ARR[@]}")
fi
if [[ "${FORCE}" == "1" ]]; then
  EXTRA_ARGS+=(--force)
fi

cd "$(dirname "$0")/.."

SWE_MAX_CPU_THREADS="${THREADS}" \
OMP_NUM_THREADS="${THREADS}" \
OPENBLAS_NUM_THREADS="${THREADS}" \
MKL_NUM_THREADS="${THREADS}" \
NUMEXPR_NUM_THREADS="${THREADS}" \
"${PYTHON_BIN}" -m earlyeval.experiments.lr_tfidf_baselines \
  --config configs/earlyeval.yaml \
  --output-dir "${RUN_DIR}" \
  --run-subdir "${RUN_SUBDIR}" \
  --max-cpu-threads "${THREADS}" \
  --max-instances "${MAX_INSTANCES}" \
  --smoke-trajectories-per-split "${SMOKE_TRAJECTORIES_PER_SPLIT}" \
  --variants ${VARIANTS} \
  --execute \
  "${EXTRA_ARGS[@]}"
