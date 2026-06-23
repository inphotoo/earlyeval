#!/usr/bin/env bash
set -euo pipefail

# Fine-grained process-feature ablations on the same representative 4-fold
# SWEVerify slice used by the paper-facing balanced4 ablation.
RUN_SUBDIR="${RUN_SUBDIR:-sweverify_ablation_fine_grained_balanced4}"
PROFILES="${PROFILES:-fine_grained_process}"
TEST_MODELS="${TEST_MODELS:-20250822_mini-v1.9.1_glm-4.5 20251124_mini-v1.17.0_minimax-m2 20251124_mini-v1.16.0_gpt-5.1-codex 20251211_mini-v1.17.2_gpt-5.2-2025-12-11-high}"

export RUN_SUBDIR PROFILES TEST_MODELS

cd "$(dirname "$0")/.."
bash scripts/run_rq_final_08_ablation_execute.sh
