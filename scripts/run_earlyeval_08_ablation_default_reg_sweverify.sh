#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_earlyeval_sweverify_holdout_models.sh"

# Regularization ablation over the paper SWE-bench Verified folds.
# Same I feature family as the main LightGBM run, but lgbm_preset=default.
RUN_SUBDIR="${RUN_SUBDIR:-sweverify_ablation_default_reg}"
PROFILES="${PROFILES:-component_default_reg}"
TEST_MODELS="${TEST_MODELS:-$(earlyeval_sweverify_holdout_models_string)}"

export RUN_SUBDIR PROFILES TEST_MODELS

cd "${SCRIPT_DIR}/.."
bash scripts/run_earlyeval_08_ablation_execute.sh
