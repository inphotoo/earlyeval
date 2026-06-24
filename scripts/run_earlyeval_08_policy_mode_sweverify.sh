#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_earlyeval_sweverify_holdout_models.sh"

PYTHON_BIN="${PYTHON_BIN:-python}"
RUN_DIR="${RUN_DIR:-paper/experiments/earlyeval_lightgbm/lightgbm_main}"
OUTPUT_DIR="${OUTPUT_DIR:-paper/experiments/earlyeval_lightgbm/policy_ablation/sweverify_policy_mode}"
TEST_MODELS="${TEST_MODELS:-$(earlyeval_sweverify_holdout_models_string)}"

EXTRA_ARGS=()
if [[ "${ALLOW_MISSING:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--allow-missing)
fi

cd "${SCRIPT_DIR}/.."
"${PYTHON_BIN}" -m earlyeval.experiments.policy_mode_ablation \
  --config configs/earlyeval.yaml \
  --run-dir "${RUN_DIR}" \
  --output-dir "${OUTPUT_DIR}" \
  --test-models ${TEST_MODELS} \
  --predictors I_LightGBM_Dense_AF \
  --score-modes raw calibrated \
  --policy-modes success_only failure_only dual \
  --success-thresholds 0.80 0.90 0.95 \
  --failure-thresholds 0.80 0.90 0.95 \
  --policy-min-steps 0 5 10 \
  --consecutive 1 2 \
  "${EXTRA_ARGS[@]}"
