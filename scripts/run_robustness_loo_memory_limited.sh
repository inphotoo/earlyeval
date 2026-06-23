#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export BLIS_NUM_THREADS=1
export SWE_MAX_CPU_THREADS=1
export MALLOC_ARENA_MAX=2
PYTHON_BIN="${PYTHON_BIN:-python}"
PARQUET_BATCH_SIZE="${SWE_ROBUSTNESS_LOO_BATCH_SIZE:-4096}"

# Default cap is 64 GiB virtual memory per Python process. Override by setting
# SWE_ROBUSTNESS_LOO_VMEM_KB before launching this script.
ulimit -v "${SWE_ROBUSTNESS_LOO_VMEM_KB:-67108864}"

"${PYTHON_BIN}" -m final3.experiments.robustness_15pct \
  --config configs/rq_final.yaml \
  --output-dir paper/experiments/rq_final_lightgbm_17 \
  --run-subdir robustness_loo_model_holdout_process_memory_limited \
  --datasets toolathlon terminalbench \
  --feature-preset process \
  --split-mode leave_one_model \
  --execute \
  --max-cpu-threads 1 \
  --parquet-batch-size "${PARQUET_BATCH_SIZE}" \
  --no-save-models

"${PYTHON_BIN}" -m final3.experiments.robustness_15pct \
  --config configs/rq_final.yaml \
  --output-dir paper/experiments/rq_final_lightgbm_17 \
  --run-subdir robustness_loo_model_holdout_rich_af_gold_memory_limited \
  --datasets toolathlon terminalbench \
  --feature-preset rich_af_gold \
  --split-mode leave_one_model \
  --execute \
  --max-cpu-threads 1 \
  --parquet-batch-size "${PARQUET_BATCH_SIZE}" \
  --no-save-models
