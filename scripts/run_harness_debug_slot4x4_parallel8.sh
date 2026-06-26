#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-python}"
OUTPUT_DIR="${OUTPUT_DIR:-paper/experiments/harness_debug_exclusion_20260626}"
DATASET="${DATASET:-terminalbench_slot4x4}"
LOG_DIR="${OUTPUT_DIR}/logs/slot4x4_manual_parallel8"

PARQUET_BATCH_SIZE="${HARNESS_DEBUG_PARQUET_BATCH_SIZE:-256}"
VMEM_GB="${HARNESS_DEBUG_VMEM_GB:-32}"
VMEM_BYTES="$((VMEM_GB * 1024 * 1024 * 1024))"
FOLD_TIMEOUT="${HARNESS_DEBUG_FOLD_TIMEOUT:-4h}"
NUM_BOOST_ROUND="${NUM_BOOST_ROUND:-250}"
TFIDF_MAX_FEATURES="${TFIDF_MAX_FEATURES:-30000}"
TFIDF_MIN_DF="${TFIDF_MIN_DF:-5}"
TFIDF_SVD_DIM="${TFIDF_SVD_DIM:-64}"

mkdir -p "${LOG_DIR}"

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

"${PYTHON_BIN}" - <<'PY'
import lightgbm
print(f"[slot4x4] lightgbm={lightgbm.__version__}")
PY

run_fold() {
  local setting="$1"
  local config="$2"
  local run_subdir="$3"
  local heldout="$4"
  local log_path="${LOG_DIR}/${setting}__${heldout}.log"

  {
    echo "[slot4x4] start $(date '+%F %T') setting=${setting} heldout=${heldout}"
    echo "[slot4x4] timeout=${FOLD_TIMEOUT} vmem_gb=${VMEM_GB} batch=${PARQUET_BATCH_SIZE}"
  } > "${log_path}"

  nice -n 15 ionice -c3 \
    prlimit "--as=${VMEM_BYTES}" -- \
    timeout --signal=TERM --kill-after=120s "${FOLD_TIMEOUT}" \
    "${PYTHON_BIN}" -m earlyeval.experiments.robustness_15pct \
      --config "${config}" \
      --output-dir "${OUTPUT_DIR}" \
      --run-subdir "${run_subdir}" \
      --datasets "${DATASET}" \
      --feature-preset rich_af_gold \
      --split-mode leave_one_model \
      --only-test-models "${heldout}" \
      --execute \
      --max-cpu-threads 1 \
      --parquet-batch-size "${PARQUET_BATCH_SIZE}" \
      --smoke-trajectories-per-split 0 \
      --max-train-rows 0 \
      --max-valid-rows 0 \
      --max-test-rows 0 \
      --num-boost-round "${NUM_BOOST_ROUND}" \
      --tfidf-max-features "${TFIDF_MAX_FEATURES}" \
      --tfidf-min-df "${TFIDF_MIN_DF}" \
      --tfidf-svd-dim "${TFIDF_SVD_DIM}" \
      --no-save-models >> "${log_path}" 2>&1

  echo "[slot4x4] end $(date '+%F %T') setting=${setting} heldout=${heldout}" >> "${log_path}"
}

pids=()

for heldout in \
  gpt-5_openai \
  gpt-5-mini_openai \
  gemini-2.5-pro_gemini \
  claude-haiku-4-5-20251001_anthropic
do
  run_fold \
    "leave_model" \
    "configs/harness_debug_slot4x4_leave_model.yaml" \
    "terminalbench_slot4x4_leave_model" \
    "${heldout}" &
  pids+=("$!")
done

for heldout in \
  terminus-2 \
  mini-swe-agent \
  openhands \
  native-cli
do
  run_fold \
    "leave_agent" \
    "configs/harness_debug_slot4x4_leave_agent.yaml" \
    "terminalbench_slot4x4_leave_agent" \
    "${heldout}" &
  pids+=("$!")
done

failures=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    failures=$((failures + 1))
  fi
done

echo "[slot4x4] done $(date '+%F %T') failures=${failures}"
exit "${failures}"
