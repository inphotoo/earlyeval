#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PACKAGE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${PACKAGE_ROOT}/.." && pwd)"
source "${SCRIPT_DIR}/_rq_final_full16_models.sh"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export TMPDIR="${TMPDIR:-${PACKAGE_ROOT}/tmp}"
mkdir -p "${TMPDIR}"

PYTHON_BIN="${PYTHON_BIN:-/home/ugproj/anaconda3/envs/swebench/bin/python}"
DEVICE="${DEVICE:-cuda}"
THREADS="${THREADS:-16}"
ENCODER_NAME="${ENCODER_NAME:-microsoft/codebert-base}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-8}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-16}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-4}"
EPOCHS="${EPOCHS:-3}"
MAX_TRAIN_ROWS="${MAX_TRAIN_ROWS:-12000}"
CACHE_ROOT="${CACHE_ROOT:-${PACKAGE_ROOT}/paper/experiments/rq_final_lightgbm_17/model_compare/bert_codebert_finetune_full16_cache}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PACKAGE_ROOT}/paper/experiments/rq_final_lightgbm_17/model_compare/bert_codebert_finetune_full16_e3}"
TEST_MODELS="${TEST_MODELS:-$(rq_final_full16_models_string)}"
EXCLUDED_TRAIN_MODELS="${EXCLUDED_TRAIN_MODELS:-$(rq_final_excluded_models_string)}"
VERIFIED_JSONL="${VERIFIED_JSONL:-${REPO_ROOT}/data/swe_verify_500/offical_answer/test.jsonl}"

mkdir -p "${CACHE_ROOT}/folds" "${OUTPUT_ROOT}/folds" "${OUTPUT_ROOT}/logs"

DOWNLOAD_ARGS=()
if [[ "${ALLOW_DOWNLOAD:-0}" == "1" ]]; then
  DOWNLOAD_ARGS+=(--allow-download)
fi

CACHE_OVERWRITE_ARGS=()
if [[ "${CACHE_OVERWRITE:-0}" == "1" ]]; then
  CACHE_OVERWRITE_ARGS+=(--overwrite)
fi

FP16_ARGS=()
if [[ "${FP16:-1}" == "0" ]]; then
  FP16_ARGS+=(--no-fp16)
fi

for model_id in ${TEST_MODELS}; do
  cache_dir="${CACHE_ROOT}/folds/${model_id}"
  fold_dir="${OUTPUT_ROOT}/folds/${model_id}"
  log_path="${OUTPUT_ROOT}/logs/${model_id}.log"
  if [[ -f "${fold_dir}/safe_stop_test_selected.csv" && "${FORCE:-0}" != "1" ]]; then
    echo "[bert] skip existing ${model_id}"
    continue
  fi
  mkdir -p "${cache_dir}" "${fold_dir}" "$(dirname "${log_path}")"
  echo "[bert] start ${model_id}; log=${log_path}"
  (
    cd "${REPO_ROOT}"
    "${PYTHON_BIN}" "${PACKAGE_ROOT}/final3/vendor/architecture_baselines/bert_baselines/build_bert_embedding_cache.py" \
      --verified-jsonl "${VERIFIED_JSONL}" \
      --holdout-models "${model_id}" \
      --exclude-train-models ${EXCLUDED_TRAIN_MODELS} \
      --max-instances 500 \
      --split-strategy per_instance_model \
      --valid-models-per-instance 3 \
      --safe-label-min-step 10 \
      --mask-train-model-id \
      --skip-embeddings \
      --encoder-name "${ENCODER_NAME}" \
      --max-cpu-threads "${THREADS}" \
      --output-dir "${cache_dir}" \
      "${DOWNLOAD_ARGS[@]}" \
      "${CACHE_OVERWRITE_ARGS[@]}"

    "${PYTHON_BIN}" "${PACKAGE_ROOT}/final3/vendor/architecture_baselines/bert_baselines/train_bert_finetune_dual_head.py" \
      --cache-dir "${cache_dir}" \
      --output-dir "${fold_dir}" \
      --encoder-name "${ENCODER_NAME}" \
      --device "${DEVICE}" \
      --batch-size "${TRAIN_BATCH_SIZE}" \
      --eval-batch-size "${EVAL_BATCH_SIZE}" \
      --gradient-accumulation-steps "${GRADIENT_ACCUMULATION_STEPS}" \
      --epochs "${EPOCHS}" \
      --max-train-rows "${MAX_TRAIN_ROWS}" \
      --max-cpu-threads "${THREADS}" \
      --success-thresholds 0.80 0.90 0.95 inf \
      --failure-thresholds 0.80 0.90 0.95 inf \
      --policy-min-steps 0 5 10 15 \
      --consecutive 1 2 3 \
      --score-modes raw calibrated \
      --max-valid-abs-drop-pp 2.0 \
      --min-valid-decision-acc 0.90 \
      "${DOWNLOAD_ARGS[@]}" \
      "${FP16_ARGS[@]}"
  ) >"${log_path}" 2>&1
  echo "[bert] done ${model_id}"
done

echo "[bert] all requested folds finished: ${OUTPUT_ROOT}"
