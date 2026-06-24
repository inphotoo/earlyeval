#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

OUT_DIR="runs/model_holdout_answer_calibrated_full/reports/gold_text_tfidf_dim_sweep"
mkdir -p "${OUT_DIR}"

"${PYTHON_BIN:-python}" \
  gold_text_tfidf_ablation_posthoc.py \
  --run-name model_holdout_answer_calibrated_full \
  --output-subdir gold_text_tfidf_dim_sweep \
  --gold-svd-dims 4 8 16 32 64 full \
  2>&1 | tee "${OUT_DIR}/run.log"
