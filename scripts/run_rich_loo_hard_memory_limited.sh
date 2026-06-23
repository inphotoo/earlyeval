#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-/home/ugproj/anaconda3/envs/swebench/bin/python}"
OUTPUT_DIR="${OUTPUT_DIR:-paper/experiments/rq_final_lightgbm_17}"
RUN_SUBDIR="${RUN_SUBDIR:-robustness_loo_model_holdout_rich_af_gold_memory_limited}"
DATASETS="${DATASETS:-toolathlon terminalbench}"
ONLY_TEST_MODELS="${ONLY_TEST_MODELS:-}"

PARQUET_BATCH_SIZE="${SWE_ROBUSTNESS_LOO_BATCH_SIZE:-512}"
VMEM_KB="${SWE_ROBUSTNESS_LOO_VMEM_KB:-33554432}"
VMEM_BYTES="$((VMEM_KB * 1024))"
FOLD_TIMEOUT="${SWE_ROBUSTNESS_LOO_FOLD_TIMEOUT:-8h}"

CGROUP_MEMORY_HIGH="${SWE_CGROUP_MEMORY_HIGH:-24G}"
CGROUP_MEMORY_MAX="${SWE_CGROUP_MEMORY_MAX:-32G}"
CGROUP_SWAP_MAX="${SWE_CGROUP_SWAP_MAX:-0}"
CGROUP_CPU_QUOTA="${SWE_CGROUP_CPU_QUOTA:-100%}"
CGROUP_IO_WEIGHT="${SWE_CGROUP_IO_WEIGHT:-10}"

LOG_DIR="${OUTPUT_DIR}/logs/rich_loo_hard_memory_limited"
mkdir -p "${LOG_DIR}"

MIN_FREE_GB="${SWE_MIN_FREE_GB:-30}"

check_free_space() {
  local path="$1"
  local min_gb="$2"
  local available_kb
  available_kb="$(df -Pk "${path}" | awk 'NR == 2 {print $4}')"
  local required_kb=$((min_gb * 1024 * 1024))
  if (( available_kb < required_kb )); then
    cat >&2 <<EOF
ERROR: ${path} has less than ${min_gb} GiB free.

rich_af_gold is heavy on parquet reads/writes. Running near a full filesystem can
stall the machine even if memory is capped. Free space first, lower
SWE_MIN_FREE_GB, or set SWE_SKIP_DISK_CHECK=1 if you deliberately accept that risk.
EOF
    exit 3
  fi
}

if [[ "${SWE_SKIP_DISK_CHECK:-0}" != "1" ]]; then
  check_free_space /data3 "${MIN_FREE_GB}"
  check_free_space /tmp "${MIN_FREE_GB}"
fi

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

SYSTEMD_SCOPE=(
  systemd-run --user --scope -q
  -p "MemoryHigh=${CGROUP_MEMORY_HIGH}"
  -p "MemoryMax=${CGROUP_MEMORY_MAX}"
  -p "MemorySwapMax=${CGROUP_SWAP_MAX}"
  -p "TasksMax=64"
  -p "CPUQuota=${CGROUP_CPU_QUOTA}"
  -p "IOWeight=${CGROUP_IO_WEIGHT}"
)

if "${SYSTEMD_SCOPE[@]}" /bin/true >/dev/null 2>&1; then
  HARD_LIMIT_AVAILABLE=1
else
  HARD_LIMIT_AVAILABLE=0
fi

if [[ "${HARD_LIMIT_AVAILABLE}" != "1" && "${SWE_ALLOW_SOFT_LIMIT:-0}" != "1" ]]; then
  cat >&2 <<EOF
ERROR: cannot create a user systemd cgroup scope, so a hard memory cap is not available in this shell.

This script refuses to run rich_af_gold without a kernel-level MemoryMax cap.
Try from a normal SSH/tmux login shell, or run this quick check:

  systemd-run --user --scope -p MemoryMax=1G /bin/true

If you intentionally accept the weaker prlimit-only fallback, set:

  SWE_ALLOW_SOFT_LIMIT=1
EOF
  exit 2
fi

"${PYTHON_BIN}" -m final3.experiments.robustness_15pct \
  --config configs/rq_final.yaml \
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

path, datasets_arg, only_arg = sys.argv[1:4]
datasets = set(datasets_arg.split()) if datasets_arg.strip() else None
only = set(only_arg.split()) if only_arg.strip() else None

with open(path, newline="", encoding="utf-8") as handle:
    for row in csv.DictReader(handle):
        dataset = str(row["dataset"])
        model = str(row["test_model"])
        if datasets and dataset not in datasets:
            continue
        if only and model not in only:
            continue
        print(f"{dataset}\t{model}")
PY

run_one_fold() {
  local dataset="$1"
  local model="$2"
  local safe_model
  safe_model="$(printf '%s' "${model}" | tr -c 'A-Za-z0-9_.@=-' '_')"
  local fold_log="${LOG_DIR}/${dataset}__${safe_model}.log"

  local cmd=(
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
    "${PYTHON_BIN}" -m final3.experiments.robustness_15pct
    --config configs/rq_final.yaml
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

  echo "[rich-loo] $(date '+%F %T') dataset=${dataset} model=${model}" | tee -a "${fold_log}"
  if [[ "${HARD_LIMIT_AVAILABLE}" == "1" ]]; then
    /usr/bin/time -v nice -n 15 ionice -c3 "${SYSTEMD_SCOPE[@]}" "${cmd[@]}" 2>&1 | tee -a "${fold_log}"
  else
    /usr/bin/time -v nice -n 15 ionice -c3 "${cmd[@]}" 2>&1 | tee -a "${fold_log}"
  fi
}

while IFS=$'\t' read -r dataset model; do
  [[ -n "${dataset}" && -n "${model}" ]] || continue
  success_path="${OUTPUT_DIR}/${RUN_SUBDIR}/${dataset}/${model//[^A-Za-z0-9_.-]/_}/_SUCCESS"
  if [[ -f "${success_path}" ]]; then
    echo "[rich-loo] skip existing dataset=${dataset} model=${model}"
    continue
  fi
  run_one_fold "${dataset}" "${model}"
done < "${MODEL_LIST}"

"${PYTHON_BIN}" -m final3.experiments.robustness_15pct \
  --config configs/rq_final.yaml \
  --output-dir "${OUTPUT_DIR}" \
  --run-subdir "${RUN_SUBDIR}" \
  --datasets ${DATASETS} \
  --feature-preset rich_af_gold \
  --split-mode leave_one_model \
  --max-cpu-threads 1 \
  --parquet-batch-size "${PARQUET_BATCH_SIZE}" \
  --no-save-models

echo "[rich-loo] done $(date '+%F %T')"
