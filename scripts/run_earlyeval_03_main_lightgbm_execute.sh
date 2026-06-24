#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
RUN_DIR="${RUN_DIR:-paper/experiments/earlyeval_lightgbm}"

# Concurrency knobs:
# - MAX_PARALLEL_FOLDS = how many test_model folds run in parallel.
# - LGBM_THREADS_PER_FOLD = LightGBM thread cap inside each fold.
#
# Memory note: the legacy trainer file-locks the load-prefix-table-with-text
# + FeatureEngineer.fit phase (`--ram-peak-lock-path`), so only ONE fold has
# the full text-column prefix table in RAM at a time even when several
# folds are scheduled in parallel. The other folds either wait for the lock
# or have already dropped text after their own fit (matrix building runs
# from disk in streaming batches).
#
# Safe starting point: 2 parallel folds, 8 threads each (16 cores).
# Tested ceiling on a 48-core / 188 GB machine: 4 parallel, 8 threads each
# (32 cores) keeps peak RAM well under 60 GB. Going higher works but watch
# `top` / `free -g` the first time.
MAX_PARALLEL_FOLDS="${MAX_PARALLEL_FOLDS:-2}"
LGBM_THREADS_PER_FOLD="${LGBM_THREADS_PER_FOLD:-8}"

EXTRA_ARGS=()
if [[ "${FORCE:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--force)
fi

cd "$(dirname "$0")/.."

EARLYEVAL_LGBM_THREADS_PER_FOLD="${LGBM_THREADS_PER_FOLD}" \
SWE_MAX_CPU_THREADS="${LGBM_THREADS_PER_FOLD}" \
OMP_NUM_THREADS="${LGBM_THREADS_PER_FOLD}" \
OPENBLAS_NUM_THREADS="${LGBM_THREADS_PER_FOLD}" \
MKL_NUM_THREADS="${LGBM_THREADS_PER_FOLD}" \
NUMEXPR_NUM_THREADS="${LGBM_THREADS_PER_FOLD}" \
"${PYTHON_BIN}" -m earlyeval.cli experiment paper-suite \
  --stage lightgbm-main \
  --config configs/earlyeval.yaml \
  --output-dir "${RUN_DIR}" \
  --max-parallel-folds "${MAX_PARALLEL_FOLDS}" \
  --execute \
  "${EXTRA_ARGS[@]}"
