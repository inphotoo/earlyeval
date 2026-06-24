#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PACKAGE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${PACKAGE_ROOT}/.." && pwd)"
source "${SCRIPT_DIR}/_earlyeval_sweverify_holdout_models.sh"

PYTHON_BIN="${PYTHON_BIN:-python}"
THREADS="${THREADS:-64}"
MAX_PARALLEL_FOLDS="${MAX_PARALLEL_FOLDS:-1}"
TEXT_BATCH_SIZE="${TEXT_BATCH_SIZE:-65536}"
LOW_MEMORY="${LOW_MEMORY:-0}"
CACHE_MATRICES="${CACHE_MATRICES:-1}"
REBUILD_MATRIX_CACHE="${REBUILD_MATRIX_CACHE:-0}"
MAX_TRAIN_ROWS_PER_HEAD="${MAX_TRAIN_ROWS_PER_HEAD:-0}"
VARIANTS="${VARIANTS:-i j}"
HIDDEN="${HIDDEN:-64}"
MAX_ITER="${MAX_ITER:-20}"
BATCH_SIZE="${BATCH_SIZE:-512}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PACKAGE_ROOT}/paper/experiments/earlyeval_lightgbm/model_compare/direct_mlp_sweverify_ij}"
TEST_MODELS="${TEST_MODELS:-$(earlyeval_sweverify_holdout_models_string)}"
EXCLUDED_TRAIN_MODELS="${EXCLUDED_TRAIN_MODELS:-$(earlyeval_excluded_models_string)}"
VERIFIED_JSONL="${VERIFIED_JSONL:-${REPO_ROOT}/data/swe_verify_500/offical_answer/test.jsonl}"
SAMPLE_WEIGHT_MODE="${SAMPLE_WEIGHT_MODE:-none}"
WEIGHTED_RESAMPLE_SIZE_PER_HEAD="${WEIGHTED_RESAMPLE_SIZE_PER_HEAD:-0}"
BALANCED_ROW_SAMPLE="${BALANCED_ROW_SAMPLE:-1}"

mkdir -p "${OUTPUT_ROOT}/logs" "${OUTPUT_ROOT}/folds"

EXTRA_ARGS=()
if [[ "${BALANCED_ROW_SAMPLE}" == "0" ]]; then
  EXTRA_ARGS+=(--no-balanced-row-sample)
fi
if [[ "${LOW_MEMORY}" == "1" ]]; then
  EXTRA_ARGS+=(--low-memory)
fi
if [[ "${CACHE_MATRICES}" == "1" ]]; then
  EXTRA_ARGS+=(--cache-matrices)
fi
if [[ "${REBUILD_MATRIX_CACHE}" == "1" ]]; then
  EXTRA_ARGS+=(--rebuild-matrix-cache)
fi

run_one_fold() {
  local model_id="$1"
  local fold_dir="${OUTPUT_ROOT}/folds/${model_id}"
  local log_path="${OUTPUT_ROOT}/logs/${model_id}.log"
  if [[ -f "${fold_dir}/safe_stop_test_selected.csv" && "${FORCE:-0}" != "1" ]]; then
    echo "[mlp] skip existing ${model_id}"
    return 0
  fi
  mkdir -p "${fold_dir}" "$(dirname "${log_path}")"
  echo "[mlp] start ${model_id}; log=${log_path}"
  (
    cd "${REPO_ROOT}"
    export SWE_MAX_CPU_THREADS="${THREADS}"
    export OMP_NUM_THREADS="${THREADS}"
    export OPENBLAS_NUM_THREADS="${THREADS}"
    export MKL_NUM_THREADS="${THREADS}"
    export NUMEXPR_NUM_THREADS="${THREADS}"
    export NUMEXPR_MAX_THREADS="${THREADS}"
    "${PYTHON_BIN}" "${PACKAGE_ROOT}/earlyeval/vendor/architecture_baselines/train_direct_dual_head_mlp.py" \
      --run-name model_holdout_answer_calibrated_full \
      --verified-jsonl "${VERIFIED_JSONL}" \
      --holdout-models "${model_id}" \
      --exclude-train-models ${EXCLUDED_TRAIN_MODELS} \
      --max-instances 500 \
      --split-strategy per_instance_model \
      --valid-models-per-instance 3 \
      --feature-set variants \
      --variants ${VARIANTS} \
      --mask-train-model-id \
      --safe-label-min-step 10 \
      --success-thresholds 0.80 0.90 0.95 inf \
      --failure-thresholds 0.80 0.90 0.95 inf \
      --policy-min-steps 0 5 10 15 \
      --consecutive 1 2 3 \
      --score-modes raw calibrated \
      --max-valid-abs-drop-pp 2.0 \
      --min-valid-decision-acc 0.90 \
      --hidden ${HIDDEN} \
      --alpha 0.001 \
      --max-iter "${MAX_ITER}" \
      --batch-size "${BATCH_SIZE}" \
      --max-train-rows-per-head "${MAX_TRAIN_ROWS_PER_HEAD}" \
      --sample-weight-mode "${SAMPLE_WEIGHT_MODE}" \
      --weighted-resample-size-per-head "${WEIGHTED_RESAMPLE_SIZE_PER_HEAD}" \
      --text-batch-size "${TEXT_BATCH_SIZE}" \
      --max-cpu-threads "${THREADS}" \
      --output-dir "${fold_dir}" \
      "${EXTRA_ARGS[@]}"
  ) >"${log_path}" 2>&1
}

declare -a pids=()
declare -a names=()
failures=0

wait_for_oldest() {
  local pid="${pids[0]}"
  local name="${names[0]}"
  if ! wait "${pid}"; then
    echo "[mlp] failed ${name}; see ${OUTPUT_ROOT}/logs/${name}.log"
    failures=$((failures + 1))
  else
    echo "[mlp] done ${name}"
  fi
  pids=("${pids[@]:1}")
  names=("${names[@]:1}")
}

for model_id in ${TEST_MODELS}; do
  while (( ${#pids[@]} >= MAX_PARALLEL_FOLDS )); do
    wait_for_oldest
  done
  run_one_fold "${model_id}" &
  pids+=("$!")
  names+=("${model_id}")
done

while (( ${#pids[@]} > 0 )); do
  wait_for_oldest
done

if (( failures > 0 )); then
  exit 1
fi

echo "[mlp] all requested folds finished: ${OUTPUT_ROOT}"
