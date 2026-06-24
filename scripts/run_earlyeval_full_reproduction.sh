#!/usr/bin/env bash
set -euo pipefail

# End-to-end reproduction driver for the EarlyEval experiments.
#
# This script orchestrates the code paths needed to reproduce the experiment
# from raw/input artifacts. It does not ship generated tables or trained models;
# those are written under paper/experiments, outputs, ../data, and ../artifacts.
#
# Required for a true from-raw SWE rebuild:
#   SWE_PARQUET_DIR=/path/to/raw/swe/tool-parquets
#
# Optional switches:
#   BUILD_SWE_SHARED=1       build SWE prefix/FeatureEngineer artifacts first
#   RUN_MAIN=1               run SWE full-16 LightGBM folds
#   RUN_ROBUSTNESS=1         run TerminalBench/Toolathlon leave-one-agent folds
#   RUN_ABLATIONS=1          run SWE full-16 ablations
#   RUN_LR_TFIDF=1           run LR/TF-IDF model comparison
#   RUN_MLP=1                run MLP full-16 comparison
#   RUN_BERT=1               run BERT/CodeBERT full-16 comparison
#   RUN_LLM_LOGIT=1          run local LLM-logit comparison
#   BUILD_TABLES=1           rebuild RQ table outputs from completed artifacts
#
# Defaults are conservative: only preflight + dry-run planning.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-python}"
export PYTHON_BIN
export PYTHONUNBUFFERED=1

echo "[repro] === 0. preflight ==="
"${PYTHON_BIN}" -m earlyeval.cli check preflight --experiment all || {
  echo "[repro] preflight reported missing data/artifacts. This is expected before BUILD_SWE_SHARED=1 on a fresh machine." >&2
}

if [[ "${BUILD_SWE_SHARED:-0}" == "1" ]]; then
  echo "[repro] === 1. build SWE shared prefix/feature artifacts ==="
  bash scripts/run_earlyeval_00_build_swe_shared_artifacts.sh
fi

if [[ "${RUN_MAIN:-0}" == "1" ]]; then
  echo "[repro] === 2. SWE full-16 main LightGBM ==="
  bash scripts/run_earlyeval_03_main_lightgbm_execute.sh
  bash scripts/run_earlyeval_04_summarize_lightgbm_current.sh
  bash scripts/run_earlyeval_05_lightgbm_policy_sweep_valid_acc.sh
  bash scripts/run_earlyeval_12_main_latency_cost.sh
else
  echo "[repro] === 2. SWE main dry-run plan ==="
  bash scripts/run_earlyeval_03_main_lightgbm_dry_run.sh
fi

if [[ "${RUN_ROBUSTNESS:-0}" == "1" ]]; then
  echo "[repro] === 3. TerminalBench/Toolathlon robustness ==="
  bash scripts/run_earlyeval_robustness_loo_answer_features_memory_limited.sh
fi

if [[ "${RUN_ABLATIONS:-0}" == "1" ]]; then
  echo "[repro] === 4. SWE full-16 ablations ==="
  RUN_SUBDIR=sweverify_ablation_feature_groups_full16 \
  PROFILES=feature_groups \
  TEST_MODELS="$(bash -lc 'source scripts/_earlyeval_full16_models.sh; earlyeval_full16_models_string')" \
  bash scripts/run_earlyeval_08_ablation_execute.sh

  RUN_SUBDIR=sweverify_ablation_feature_groups_full16 \
  PROFILES=component_with_model_id \
  TEST_MODELS="$(bash -lc 'source scripts/_earlyeval_full16_models.sh; earlyeval_full16_models_string')" \
  bash scripts/run_earlyeval_08_ablation_execute.sh

  bash scripts/run_earlyeval_08_ablation_default_reg_full16.sh
  bash scripts/run_earlyeval_08_ablation_fine_grained_full16.sh
fi

if [[ "${RUN_LR_TFIDF:-0}" == "1" ]]; then
  echo "[repro] === 5a. LR/TF-IDF architecture comparison ==="
  bash scripts/run_earlyeval_06_model_compare_lr_tfidf.sh
fi

if [[ "${RUN_MLP:-0}" == "1" ]]; then
  echo "[repro] === 5b. MLP architecture comparison ==="
  bash scripts/run_earlyeval_09_direct_mlp_full16.sh
fi

if [[ "${RUN_BERT:-0}" == "1" ]]; then
  echo "[repro] === 5c. BERT/CodeBERT architecture comparison ==="
  bash scripts/run_earlyeval_09_bert_finetune_full16.sh
fi

if [[ "${RUN_LLM_LOGIT:-0}" == "1" ]]; then
  echo "[repro] === 5d. local LLM-logit architecture comparison ==="
  bash scripts/run_earlyeval_09_llm_logit_full16.sh
fi

if [[ "${BUILD_TABLES:-0}" == "1" ]]; then
  echo "[repro] === 6. rebuild paper tables from completed artifacts ==="
  export SWEBENCH_PACKAGE_ROOT="${SWEBENCH_PACKAGE_ROOT:-${ROOT_DIR}}"
  export RQ_TABLES_OUT="${RQ_TABLES_OUT:-${ROOT_DIR}/paper/results/rq_tables_reproduced}"
  "${PYTHON_BIN}" paper_reporting/build_rq_tables_bundle.py
fi

echo "[repro] done"
