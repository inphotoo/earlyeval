#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/home/ugproj/anaconda3/envs/swebench/bin/python}"
RUN_DIR="${RUN_DIR:-paper/experiments/rq_final_lightgbm_17}"
FEATURE_PRESET="${FEATURE_PRESET:-rich_af_gold}"
if [[ "${FEATURE_PRESET}" == "rich_af_gold" ]]; then
  RUN_SUBDIR="${RUN_SUBDIR:-robustness_15pct_model_holdout_rich_af_gold_no_length}"
else
  RUN_SUBDIR="${RUN_SUBDIR:-robustness_15pct_model_holdout_no_length}"
fi
DATASETS="${DATASETS:-toolathlon terminalbench}"
THREADS="${THREADS:-2}"
TEST_MODEL_RATIO="${TEST_MODEL_RATIO:-0.15}"
TFIDF_MAX_FEATURES="${TFIDF_MAX_FEATURES:-30000}"
TFIDF_MIN_DF="${TFIDF_MIN_DF:-5}"
TFIDF_SVD_DIM="${TFIDF_SVD_DIM:-64}"
TFIDF_NGRAM_MAX="${TFIDF_NGRAM_MAX:-2}"
SMOKE_TRAJECTORIES_PER_SPLIT="${SMOKE_TRAJECTORIES_PER_SPLIT:-0}"
MAX_TRAIN_ROWS="${MAX_TRAIN_ROWS:-0}"
MAX_VALID_ROWS="${MAX_VALID_ROWS:-0}"
MAX_TEST_ROWS="${MAX_TEST_ROWS:-0}"
EXTRA_ARGS=()

if [[ "${FORCE:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--force)
fi

cd "$(dirname "$0")/.."

SWE_MAX_CPU_THREADS="${THREADS}" \
OMP_NUM_THREADS="${THREADS}" \
OPENBLAS_NUM_THREADS="${THREADS}" \
MKL_NUM_THREADS="${THREADS}" \
NUMEXPR_NUM_THREADS="${THREADS}" \
"${PYTHON_BIN}" -m final3.experiments.robustness_15pct \
  --config configs/rq_final.yaml \
  --output-dir "${RUN_DIR}" \
  --run-subdir "${RUN_SUBDIR}" \
  --datasets ${DATASETS} \
  --feature-preset "${FEATURE_PRESET}" \
  --test-model-ratio "${TEST_MODEL_RATIO}" \
  --max-cpu-threads "${THREADS}" \
  --tfidf-max-features "${TFIDF_MAX_FEATURES}" \
  --tfidf-min-df "${TFIDF_MIN_DF}" \
  --tfidf-svd-dim "${TFIDF_SVD_DIM}" \
  --tfidf-ngram-max "${TFIDF_NGRAM_MAX}" \
  --smoke-trajectories-per-split "${SMOKE_TRAJECTORIES_PER_SPLIT}" \
  --max-train-rows "${MAX_TRAIN_ROWS}" \
  --max-valid-rows "${MAX_VALID_ROWS}" \
  --max-test-rows "${MAX_TEST_ROWS}" \
  --execute \
  "${EXTRA_ARGS[@]}"
