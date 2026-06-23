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
export NUMEXPR_MAX_THREADS="${THREADS:-1}"
export BLIS_NUM_THREADS="${THREADS:-1}"
export SWE_MAX_CPU_THREADS="${THREADS:-1}"
export MALLOC_ARENA_MAX="${MALLOC_ARENA_MAX:-2}"

PYTHON_BIN="${PYTHON_BIN:-python}"
RUN_DIR="${RUN_DIR:-paper/experiments/rq_final_lightgbm_17}"
THREADS="${THREADS:-1}"
VMEM_KB="${SWE_ABLATION_VMEM_KB:-52428800}"

"${PYTHON_BIN}" -m final3.experiments.run_missing_ablation \
  --root "${RUN_DIR}" \
  --threads "${THREADS}" \
  --vmem-kb "${VMEM_KB}" \
  --audit-json "${RUN_DIR}/ablation_missing_resume_manifest.json" \
  --refresh-reporting-detail \
  --execute
