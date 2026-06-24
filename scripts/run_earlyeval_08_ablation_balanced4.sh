#!/usr/bin/env bash
set -euo pipefail

# Representative SWEVerify ablation slice over the filtered held-out model set.
# Approximate trajectory success rates: 0.54, 0.60, 0.66, 0.71.
RUN_SUBDIR="${RUN_SUBDIR:-sweverify_ablation_balanced4}"
PROFILES="${PROFILES:-feature_groups component_with_model_id component_default_reg}"
TEST_MODELS="${TEST_MODELS:-20250822_mini-v1.9.1_glm-4.5 20251124_mini-v1.17.0_minimax-m2 20251124_mini-v1.16.0_gpt-5.1-codex 20251211_mini-v1.17.2_gpt-5.2-2025-12-11-high}"

export RUN_SUBDIR PROFILES TEST_MODELS

cd "$(dirname "$0")/.."
bash scripts/run_earlyeval_08_ablation_execute.sh
