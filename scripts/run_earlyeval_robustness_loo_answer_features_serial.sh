#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-python}"
OUTPUT_DIR="${OUTPUT_DIR:-paper/experiments/earlyeval_lightgbm}"
RUN_SUBDIR="${RUN_SUBDIR:-robustness_loo_model_holdout_rich_af_gold_memory_limited}"
DATASETS="${DATASETS:-toolathlon terminalbench}"
ONLY_TEST_MODELS="${ONLY_TEST_MODELS:-}"

PARQUET_BATCH_SIZE="${SWE_ROBUSTNESS_LOO_BATCH_SIZE:-256}"
VMEM_KB="${SWE_ROBUSTNESS_LOO_VMEM_KB:-67108864}"
VMEM_BYTES="$((VMEM_KB * 1024))"
RSS_KILL_GB="${SWE_RSS_KILL_GB:-55}"
RSS_KILL_KB="$((RSS_KILL_GB * 1024 * 1024))"
FOLD_TIMEOUT="${SWE_ROBUSTNESS_LOO_FOLD_TIMEOUT:-4h}"
MONITOR_INTERVAL_SECONDS="${SWE_MONITOR_INTERVAL_SECONDS:-5}"
STOP_ON_FAILURE="${STOP_ON_FAILURE:-1}"
MIN_FREE_GB="${SWE_MIN_FREE_GB:-30}"

LOG_DIR="${OUTPUT_DIR}/logs/robustness_loo_answer_features_serial"
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

check_free_space() {
  local path="$1"
  local min_gb="$2"
  local available_kb
  available_kb="$(df -Pk "${path}" | awk 'NR == 2 {print $4}')"
  local required_kb=$((min_gb * 1024 * 1024))
  if (( available_kb < required_kb )); then
    echo "ERROR: ${path} has less than ${min_gb} GiB free. Set SWE_MIN_FREE_GB lower only if you accept the risk." >&2
    exit 3
  fi
}

check_free_space "${ROOT_DIR}" "${MIN_FREE_GB}"
check_free_space /tmp "${MIN_FREE_GB}"

echo "[one-by-one] generating command_index for ${DATASETS}"
"${PYTHON_BIN}" -m earlyeval.experiments.robustness_15pct \
  --config configs/earlyeval.yaml \
  --output-dir "${OUTPUT_DIR}" \
  --run-subdir "${RUN_SUBDIR}" \
  --datasets ${DATASETS} \
  --feature-preset rich_af_gold \
  --split-mode leave_one_model \
  --max-cpu-threads 1 \
  --parquet-batch-size "${PARQUET_BATCH_SIZE}" \
  --no-save-models

COMMAND_INDEX="${OUTPUT_DIR}/${RUN_SUBDIR}/command_index.csv"
MODEL_LIST="$(mktemp)"
trap 'rm -f "${MODEL_LIST}"' EXIT

"${PYTHON_BIN}" - "${COMMAND_INDEX}" "${DATASETS}" "${ONLY_TEST_MODELS}" > "${MODEL_LIST}" <<'PY'
import csv
import sys
from pathlib import Path

path, datasets_arg, only_arg = sys.argv[1:4]
datasets = set(datasets_arg.split()) if datasets_arg.strip() else None
only = set(only_arg.split()) if only_arg.strip() else None

with open(path, newline="", encoding="utf-8") as handle:
    for row in csv.DictReader(handle):
        dataset = str(row["dataset"])
        model = str(row["test_model"])
        output_dir = str(row["output_dir"])
        if datasets and dataset not in datasets:
            continue
        if only and model not in only:
            continue
        safe_model = Path(output_dir).name
        print(f"{dataset}\t{model}\t{output_dir}\t{safe_model}")
PY

rss_for_pgid_kb() {
  local pgid="$1"
  ps -eo pgid=,rss= | awk -v pgid="${pgid}" '$1 == pgid {sum += $2} END {print sum + 0}'
}

run_one_fold() {
  local dataset="$1"
  local model="$2"
  local fold_dir="$3"
  local safe_model="$4"
  local fold_log="${LOG_DIR}/${dataset}__${safe_model}.log"
  local success_path="${fold_dir}/_SUCCESS"

  if [[ -f "${success_path}" ]]; then
    echo "[one-by-one] skip existing dataset=${dataset} model=${model}"
    return 0
  fi

  {
    echo ""
    echo "[one-by-one] start $(date '+%F %T') dataset=${dataset} model=${model}"
    echo "[one-by-one] settings batch=${PARQUET_BATCH_SIZE} vmem_kb=${VMEM_KB} rss_kill_gb=${RSS_KILL_GB} timeout=${FOLD_TIMEOUT}"
  } | tee -a "${fold_log}"

  local cmd=(
    nice -n 15
    ionice -c3
    env
    PYTHONUNBUFFERED=1
    OMP_NUM_THREADS=1
    OPENBLAS_NUM_THREADS=1
    MKL_NUM_THREADS=1
    VECLIB_MAXIMUM_THREADS=1
    NUMEXPR_NUM_THREADS=1
    BLIS_NUM_THREADS=1
    ARROW_NUM_THREADS=1
    SWE_MAX_CPU_THREADS=1
    MALLOC_ARENA_MAX=1
    prlimit "--as=${VMEM_BYTES}" --
    timeout --signal=TERM --kill-after=120s "${FOLD_TIMEOUT}"
    "${PYTHON_BIN}" -m earlyeval.experiments.robustness_15pct
    --config configs/earlyeval.yaml
    --output-dir "${OUTPUT_DIR}"
    --run-subdir "${RUN_SUBDIR}"
    --datasets "${dataset}"
    --feature-preset rich_af_gold
    --split-mode leave_one_model
    --only-test-models "${model}"
    --execute
    --max-cpu-threads 1
    --parquet-batch-size "${PARQUET_BATCH_SIZE}"
    --no-save-models
  )

  setsid "${cmd[@]}" >> "${fold_log}" 2>&1 &
  local runner_pid="$!"
  local max_rss_kb=0
  local killed_by_watchdog=0

  while kill -0 "${runner_pid}" 2>/dev/null; do
    local rss_kb
    rss_kb="$(rss_for_pgid_kb "${runner_pid}")"
    if (( rss_kb > max_rss_kb )); then
      max_rss_kb="${rss_kb}"
    fi
    if (( rss_kb > RSS_KILL_KB )); then
      killed_by_watchdog=1
      {
        echo "[one-by-one] RSS watchdog kill at $(date '+%F %T'): rss_kb=${rss_kb} limit_kb=${RSS_KILL_KB}"
      } | tee -a "${fold_log}"
      kill -TERM "-${runner_pid}" 2>/dev/null || true
      sleep 120
      kill -KILL "-${runner_pid}" 2>/dev/null || true
      break
    fi
    sleep "${MONITOR_INTERVAL_SECONDS}"
  done

  local status=0
  wait "${runner_pid}" || status="$?"
  {
    echo "[one-by-one] end $(date '+%F %T') dataset=${dataset} model=${model} status=${status} max_rss_kb=${max_rss_kb} killed_by_watchdog=${killed_by_watchdog}"
  } | tee -a "${fold_log}"

  if [[ "${status}" != "0" ]]; then
    echo "[one-by-one] FAILED dataset=${dataset} model=${model}; log=${fold_log}" >&2
    if [[ "${STOP_ON_FAILURE}" == "1" ]]; then
      exit "${status}"
    fi
  fi
}

while IFS=$'\t' read -r dataset model fold_dir safe_model; do
  [[ -n "${dataset}" && -n "${model}" && -n "${fold_dir}" ]] || continue
  run_one_fold "${dataset}" "${model}" "${fold_dir}" "${safe_model}"
done < "${MODEL_LIST}"

echo "[one-by-one] refreshing command_index"
"${PYTHON_BIN}" -m earlyeval.experiments.robustness_15pct \
  --config configs/earlyeval.yaml \
  --output-dir "${OUTPUT_DIR}" \
  --run-subdir "${RUN_SUBDIR}" \
  --datasets ${DATASETS} \
  --feature-preset rich_af_gold \
  --split-mode leave_one_model \
  --max-cpu-threads 1 \
  --parquet-batch-size "${PARQUET_BATCH_SIZE}" \
  --no-save-models

success_count="$(find "${OUTPUT_DIR}/${RUN_SUBDIR}" -name _SUCCESS | wc -l)"
echo "[one-by-one] done $(date '+%F %T') success_count=${success_count}"
