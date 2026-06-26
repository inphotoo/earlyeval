#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-python}"
CONFIG_PATH="${CONFIG_PATH:-configs/harness_debug_cross_agent.yaml}"
OUTPUT_DIR="${OUTPUT_DIR:-paper/experiments/cross_agent_harness}"
RUN_SUBDIR="${RUN_SUBDIR:-terminalbench_cross_agent_leave_one_unit}"
PARQUET_BATCH_SIZE="${PARQUET_BATCH_SIZE:-8192}"
MAX_CPU_THREADS="${MAX_CPU_THREADS:-1}"
VMEM_GB="${VMEM_GB:-48}"
FOLD_TIMEOUT="${FOLD_TIMEOUT:-8h}"
ONLY_TEST_MODELS="${ONLY_TEST_MODELS:-}"
TOKEN_PREFIX_CACHE="${TOKEN_PREFIX_CACHE:-}"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export BLIS_NUM_THREADS=1
export ARROW_NUM_THREADS=1
export SWE_MAX_CPU_THREADS=1
export MALLOC_ARENA_MAX=1

vmem_kb=$((VMEM_GB * 1024 * 1024))

args=(
  -m earlyeval.experiments.robustness_15pct
  --config "${CONFIG_PATH}"
  --output-dir "${OUTPUT_DIR}"
  --run-subdir "${RUN_SUBDIR}"
  --datasets terminalbench_harness_core16
  --feature-preset rich_af_gold
  --split-mode leave_one_model
  --execute
  --max-cpu-threads "${MAX_CPU_THREADS}"
  --parquet-batch-size "${PARQUET_BATCH_SIZE}"
  --smoke-trajectories-per-split "${SMOKE_TRAJECTORIES_PER_SPLIT:-0}"
  --max-train-rows "${MAX_TRAIN_ROWS:-0}"
  --max-valid-rows "${MAX_VALID_ROWS:-0}"
  --max-test-rows "${MAX_TEST_ROWS:-0}"
  --num-boost-round "${NUM_BOOST_ROUND:-250}"
  --tfidf-max-features "${TFIDF_MAX_FEATURES:-30000}"
  --tfidf-min-df "${TFIDF_MIN_DF:-5}"
  --tfidf-svd-dim "${TFIDF_SVD_DIM:-64}"
  --no-save-models
)

if [[ -n "${ONLY_TEST_MODELS}" ]]; then
  # shellcheck disable=SC2206
  only_models=( ${ONLY_TEST_MODELS} )
  args+=(--only-test-models "${only_models[@]}")
fi

prlimit "--as=$((vmem_kb * 1024))" -- timeout --signal=TERM --kill-after=120s "${FOLD_TIMEOUT}" \
  "${PYTHON_BIN}" "${args[@]}"

summary_args=(
  -m earlyeval.experiments.harness_debug_terminalbench_summary
  --run-dir "${OUTPUT_DIR}/${RUN_SUBDIR}"
  --output-dir "${OUTPUT_DIR}/${RUN_SUBDIR}/summary/fixed_thresholds_main_aligned"
)

if [[ -n "${TOKEN_PREFIX_CACHE}" ]]; then
  summary_args+=(--token-prefix-cache "${TOKEN_PREFIX_CACHE}")
fi

"${PYTHON_BIN}" "${summary_args[@]}"
