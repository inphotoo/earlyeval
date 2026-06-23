#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_rq_final_full16_models.sh"

# Regularization ablation over the final 16 SWEVerify folds.
# Same I feature family as the main LightGBM run, but lgbm_preset=default.
RUN_SUBDIR="${RUN_SUBDIR:-sweverify_ablation_default_reg_full16}"
PROFILES="${PROFILES:-component_default_reg}"
TEST_MODELS="${TEST_MODELS:-$(rq_final_full16_models_string)}"

export RUN_SUBDIR PROFILES TEST_MODELS

cd "${SCRIPT_DIR}/.."
bash scripts/run_rq_final_08_ablation_execute.sh
