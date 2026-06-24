#!/usr/bin/env bash
set -euo pipefail

RUN_SUBDIR="${RUN_SUBDIR:-sweverify_ablation_random4}"
SAMPLE_FOLDS="${SAMPLE_FOLDS:-4}"
SAMPLE_SEED="${SAMPLE_SEED:-42}"
PROFILES="${PROFILES:-feature_groups component_with_model_id component_default_reg}"

export RUN_SUBDIR SAMPLE_FOLDS SAMPLE_SEED PROFILES

cd "$(dirname "$0")/.."
bash scripts/run_earlyeval_08_ablation_execute.sh
