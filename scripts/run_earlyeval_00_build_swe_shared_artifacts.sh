#!/usr/bin/env bash
set -euo pipefail

# Build the SWE-bench Verified shared artifacts used by the final experiments:
# step_table, prefix_table, prefix_table_filtered, answer-enriched prefix table,
# and FeatureEngineer pickles. This is the from-raw-data stage that must run
# before the LightGBM/MLP/BERT folds when those artifacts are not already
# materialized.
#
# Required:
#   SWE_PARQUET_DIR=/path/to/tool-parquet-directory
#
# Optional:
#   VERIFIED_JSONL=/path/to/swe_verify_500/offical_answer/test.jsonl
#   PYTHON_BIN=/path/to/python
#   MAX_INSTANCES=500
#   RUN_NAME=model_holdout_answer_calibrated_full
#   SHARED_DIR=../data/prefix_predict_model_holdout_answer/model_holdout_answer_shared
#   ARTIFACT_MODEL_DIR=../artifacts/model_holdout_answer_calibrated_full/models

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-python}"
RUN_NAME="${RUN_NAME:-model_holdout_answer_calibrated_full}"
SHARED_DIR="${SHARED_DIR:-../data/prefix_predict_model_holdout_answer/model_holdout_answer_shared}"
ARTIFACT_MODEL_DIR="${ARTIFACT_MODEL_DIR:-../artifacts/model_holdout_answer_calibrated_full/models}"
VERIFIED_JSONL="${VERIFIED_JSONL:-../data/swe_verify_500/offical_answer/test.jsonl}"
MAX_INSTANCES="${MAX_INSTANCES:-500}"
SKIP_LGBM="${SKIP_LGBM:-1}"
REUSE_EXISTING="${REUSE_EXISTING:-0}"
NO_GPU_LGBM="${NO_GPU_LGBM:-0}"

if [[ -z "${SWE_PARQUET_DIR:-}" ]]; then
  cat >&2 <<'EOF'
ERROR: SWE_PARQUET_DIR is required.

Set it to the directory containing the raw SWE trajectory tool-*.parquet files:

  SWE_PARQUET_DIR=/path/to/tool-parquets bash scripts/run_earlyeval_00_build_swe_shared_artifacts.sh
EOF
  exit 2
fi

mkdir -p "${SHARED_DIR}" "${ARTIFACT_MODEL_DIR}"

export FINAL3_VENDOR_RUNTIME_ROOT="${FINAL3_VENDOR_RUNTIME_ROOT:-${ROOT_DIR}/outputs/vendor_runtime/prefix_predict_model_holdout_answer}"
export SWE_PREFIX_SKIP_INSTANCE_DEDUP="${SWE_PREFIX_SKIP_INSTANCE_DEDUP:-1}"
export PYTHONUNBUFFERED=1

args=(
  "${ROOT_DIR}/final3/vendor/prefix_predict_model_holdout_answer/run_all.py"
  --run-name "${RUN_NAME}"
  --data-dir "${SWE_PARQUET_DIR}"
  --split-by model_holdout
  --holdout-models auto_mid3
  --max-instances "${MAX_INSTANCES}"
  --verified-jsonl "${VERIFIED_JSONL}"
)

if [[ "${SKIP_LGBM}" == "1" ]]; then
  args+=(--skip-lgbm --skip-ablation)
fi
if [[ "${REUSE_EXISTING}" == "1" ]]; then
  args+=(--reuse-answer-enriched --reuse-feature-engineers)
fi
if [[ "${NO_GPU_LGBM}" == "1" ]]; then
  args+=(--no-gpu-lgbm)
fi

echo "[build-swe-shared] runtime=${FINAL3_VENDOR_RUNTIME_ROOT}"
"${PYTHON_BIN}" "${args[@]}"

RUN_ROOT="${ROOT_DIR}/final3/vendor/prefix_predict_model_holdout_answer/runs/${RUN_NAME}"

cp -f "${RUN_ROOT}/data/step_table.parquet" "${SHARED_DIR}/step_table.parquet"
cp -f "${RUN_ROOT}/data/prefix_table.parquet" "${SHARED_DIR}/prefix_table.parquet"
cp -f "${RUN_ROOT}/data/prefix_table_filtered.parquet" "${SHARED_DIR}/prefix_table_filtered.parquet"
if [[ -f "${RUN_ROOT}/data/prefix_table_answer_enriched.parquet" ]]; then
  cp -f "${RUN_ROOT}/data/prefix_table_answer_enriched.parquet" "${SHARED_DIR}/prefix_table_answer_enriched.parquet"
fi
cp -f "${RUN_ROOT}/models/feature_engineer_with_model.pkl" "${ARTIFACT_MODEL_DIR}/feature_engineer_with_model.pkl"
if [[ -f "${RUN_ROOT}/models/feature_engineer_no_model.pkl" ]]; then
  cp -f "${RUN_ROOT}/models/feature_engineer_no_model.pkl" "${ARTIFACT_MODEL_DIR}/feature_engineer_no_model.pkl"
fi

cat > "${SHARED_DIR}/BUILD_MANIFEST.txt" <<EOF
Built by: scripts/run_earlyeval_00_build_swe_shared_artifacts.sh
Run name: ${RUN_NAME}
Raw SWE_PARQUET_DIR: ${SWE_PARQUET_DIR}
Verified JSONL: ${VERIFIED_JSONL}
Runtime root: ${FINAL3_VENDOR_RUNTIME_ROOT}
Shared data dir: ${SHARED_DIR}
Artifact model dir: ${ARTIFACT_MODEL_DIR}
EOF

echo "[build-swe-shared] wrote shared parquet artifacts to ${SHARED_DIR}"
echo "[build-swe-shared] wrote FeatureEngineer pickles to ${ARTIFACT_MODEL_DIR}"
