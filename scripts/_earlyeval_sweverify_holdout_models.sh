#!/usr/bin/env bash

# Paper SWE-bench Verified held-out-agent set.
# Excludes the v2.0 Gemini high outlier and other configured low-coverage folds.
EARLYEVAL_SWEVERIFY_HOLDOUT_MODEL_IDS=(
  "20250807_mini-v1.7.0_gpt-5"
  "20250807_mini-v1.7.0_gpt-5-mini"
  "20250807_mini-v1.7.0_gpt-5-nano"
  "20250822_mini-v1.9.1_glm-4.5"
  "20250929_mini-v1.13.3_sonnet-4-5-20250929"
  "20251118_mini-v1.15.0_gemini-3-pro-preview-20251118"
  "20251120_mini-v1.15.0_gpt-5.1-2025-11-13"
  "20251124_mini-v1.16.0_gpt-5.1-codex"
  "20251124_mini-v1.17.0_minimax-m2"
  "20251201_mini-v1.17.1_deepseek-v3.2-reasoner"
  "20251201_mini-v1.17.1_glm-4.6"
  "20251209_mini-v1.17.2_devstral-2512"
  "20251209_mini-v1.17.2_devstral-small-2512"
  "20251210_mini-v1.17.2_kimi-k2-thinking"
  "20251211_mini-v1.17.2_gpt-5.2-2025-12-11"
  "20251211_mini-v1.17.2_gpt-5.2-2025-12-11-high"
)

EARLYEVAL_EXCLUDED_MODEL_IDS=(
  "20251124_mini-v1.16.0_claude-opus-4-5-20251101"
  "20260219_mini-v2.0.0_gpt-5-2-codex"
  "20250726_mini-v1.0.0_gemini-2.5-flash"
  "20260217_mini-v2.0.0_claude-4-6-opus"
  "20260226_mini-v2.0.0_gemini-3-pro-high"
)

earlyeval_sweverify_holdout_models_string() {
  local IFS=" "
  printf "%s" "${EARLYEVAL_SWEVERIFY_HOLDOUT_MODEL_IDS[*]}"
}

earlyeval_excluded_models_string() {
  local IFS=" "
  printf "%s" "${EARLYEVAL_EXCLUDED_MODEL_IDS[*]}"
}
