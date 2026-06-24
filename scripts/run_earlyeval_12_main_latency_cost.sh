#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${THREADS:-1}"
export OPENBLAS_NUM_THREADS="${THREADS:-1}"
export MKL_NUM_THREADS="${THREADS:-1}"
export VECLIB_MAXIMUM_THREADS="${THREADS:-1}"
export NUMEXPR_NUM_THREADS="${THREADS:-1}"
export BLIS_NUM_THREADS="${THREADS:-1}"
export SWE_MAX_CPU_THREADS="${THREADS:-1}"
export MALLOC_ARENA_MAX="${MALLOC_ARENA_MAX:-2}"

PYTHON_BIN="${PYTHON_BIN:-python}"
RUN_DIR="${RUN_DIR:-paper/experiments/earlyeval_lightgbm/lightgbm_main}"
OUTPUT_DIR="${OUTPUT_DIR:-${RUN_DIR}/latency_cost}"
REPEATS="${REPEATS:-5}"

EXTRA_ARGS=()
if [[ -n "${PRICE_PER_MILLION_TOTAL_TOKENS:-}" ]]; then
  EXTRA_ARGS+=(--price-per-million-total-tokens "${PRICE_PER_MILLION_TOTAL_TOKENS}")
fi
if [[ -n "${PRICE_PER_MILLION_INPUT_TOKENS:-}" ]]; then
  EXTRA_ARGS+=(--price-per-million-input-tokens "${PRICE_PER_MILLION_INPUT_TOKENS}")
fi
if [[ "${NO_BENCHMARK:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--no-benchmark)
fi

"${PYTHON_BIN}" -m earlyeval.experiments.main_latency_cost_audit \
  --run-dir "${RUN_DIR}" \
  --output-dir "${OUTPUT_DIR}" \
  --benchmark-repeats "${REPEATS}" \
  "${EXTRA_ARGS[@]}"
