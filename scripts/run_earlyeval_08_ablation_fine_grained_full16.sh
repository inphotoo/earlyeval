#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_earlyeval_full16_models.sh"

# Fine-grained process-feature ablations over the paper full-16 SWEVerify folds.
# Variants: no_feedback, no_action, no_thought, process_only.
RUN_SUBDIR="${RUN_SUBDIR:-sweverify_ablation_fine_grained_full16}"
PROFILES="${PROFILES:-fine_grained_process}"
TEST_MODELS="${TEST_MODELS:-$(earlyeval_full16_models_string)}"

export RUN_SUBDIR PROFILES TEST_MODELS

cd "${SCRIPT_DIR}/.."
bash scripts/run_earlyeval_08_ablation_execute.sh
