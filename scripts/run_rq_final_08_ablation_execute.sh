#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/home/ugproj/anaconda3/envs/swebench/bin/python}"
RUN_DIR="${RUN_DIR:-paper/experiments/rq_final_lightgbm_17}"
RUN_SUBDIR="${RUN_SUBDIR:-sweverify_ablation_full}"
PROFILES="${PROFILES:-feature_groups component_with_model_id component_default_reg}"
# THREADS = per-fold LightGBM thread cap.
# MAX_PARALLEL_FOLDS = concurrent fold subprocesses. The legacy trainer
# file-locks the load+fit phase (`--ram-peak-lock-path`), so peak RAM is
# bounded to one full prefix table at a time regardless of parallelism.
# Safe default: 2 parallel × 8 threads = 16 cores. Bump to 3-4 once you
# verify `free -g` headroom is comfortable.
THREADS="${THREADS:-8}"
MAX_PARALLEL_FOLDS="${MAX_PARALLEL_FOLDS:-2}"
MAX_FOLDS="${MAX_FOLDS:-}"
SAMPLE_FOLDS="${SAMPLE_FOLDS:-}"
SAMPLE_SEED="${SAMPLE_SEED:-42}"
TEST_MODELS="${TEST_MODELS:-}"
VARIANTS="${VARIANTS:-}"
EXTRA_ARGS=()

if [[ -n "${MAX_FOLDS}" ]]; then
  EXTRA_ARGS+=(--max-folds "${MAX_FOLDS}")
fi
if [[ -n "${SAMPLE_FOLDS}" ]]; then
  EXTRA_ARGS+=(--sample-folds "${SAMPLE_FOLDS}" --sample-seed "${SAMPLE_SEED}")
fi
if [[ -n "${TEST_MODELS}" ]]; then
  EXTRA_ARGS+=(--test-models ${TEST_MODELS})
fi
if [[ -n "${VARIANTS}" ]]; then
  EXTRA_ARGS+=(--variants ${VARIANTS})
fi
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
  --threads "${THREADS}" \
  --max-parallel-folds "${MAX_PARALLEL_FOLDS}" \
  --policy-min-steps 0 5 10 \
  --consecutive 1 2 \
  --success-thresholds 0.80 0.90 0.95 \
  --failure-thresholds 0.80 0.90 0.95 \
  --score-modes raw calibrated \
  --execute \
  "${EXTRA_ARGS[@]}"
