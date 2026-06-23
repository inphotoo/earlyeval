#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PACKAGE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${PACKAGE_ROOT}/.." && pwd)"
source "${SCRIPT_DIR}/_rq_final_full16_models.sh"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export TMPDIR="${TMPDIR:-${PACKAGE_ROOT}/tmp}"
mkdir -p "${TMPDIR}"

PYTHON_BIN="${PYTHON_BIN:-python}"
DEVICE="${DEVICE:-cuda}"
THREADS="${THREADS:-16}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-Coder-3B-Instruct}"
MODEL_TAG="${MODEL_TAG:-qwen25_coder_3b}"
BATCH_SIZE="${BATCH_SIZE:-4}"
MAX_INPUT_TOKENS="${MAX_INPUT_TOKENS:-4096}"
PROMPT_MODE="${PROMPT_MODE:-dual}"
CACHE_ROOT="${CACHE_ROOT:-${PACKAGE_ROOT}/paper/experiments/rq_final_lightgbm_17/model_compare/llm_logit_${MODEL_TAG}_full16_cache}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PACKAGE_ROOT}/paper/experiments/rq_final_lightgbm_17/model_compare/llm_logit_${MODEL_TAG}_full16}"
TEST_MODELS="${TEST_MODELS:-$(rq_final_full16_models_string)}"
EXCLUDED_TRAIN_MODELS="${EXCLUDED_TRAIN_MODELS:-$(rq_final_excluded_models_string)}"
VERIFIED_JSONL="${VERIFIED_JSONL:-${REPO_ROOT}/data/swe_verify_500/offical_answer/test.jsonl}"

mkdir -p "${CACHE_ROOT}/folds" "${OUTPUT_ROOT}/folds" "${OUTPUT_ROOT}/logs"

DOWNLOAD_ARGS=()
if [[ "${ALLOW_DOWNLOAD:-0}" == "1" ]]; then
  DOWNLOAD_ARGS+=(--allow-download)
fi

OVERWRITE_ARGS=()
if [[ "${OVERWRITE:-0}" == "1" ]]; then
  OVERWRITE_ARGS+=(--overwrite)
fi

for model_id in ${TEST_MODELS}; do
  cache_dir="${CACHE_ROOT}/folds/${model_id}"
  fold_dir="${OUTPUT_ROOT}/folds/${model_id}"
  log_path="${OUTPUT_ROOT}/logs/${model_id}.log"
  if [[ -f "${fold_dir}/safe_stop_test_selected.csv" && "${FORCE:-0}" != "1" ]]; then
    echo "[llm-logit] skip existing ${model_id}"
    continue
  fi
  mkdir -p "${cache_dir}" "${fold_dir}" "$(dirname "${log_path}")"
  echo "[llm-logit] start ${model_id}; log=${log_path}"
  (
    cd "${REPO_ROOT}"
    "${PYTHON_BIN}" "${PACKAGE_ROOT}/final3/vendor/architecture_baselines/bert_baselines/build_bert_embedding_cache.py" \
      --verified-jsonl "${VERIFIED_JSONL}" \
      --holdout-models "${model_id}" \
      --exclude-train-models ${EXCLUDED_TRAIN_MODELS} \
      --max-instances 500 \
      --split-strategy per_instance_model \
      --valid-models-per-instance 3 \
      --smoke-trajectories-per-split 0 \
      --skip-embeddings \
      --max-cpu-threads "${THREADS}" \
      --output-dir "${cache_dir}" \
      "${DOWNLOAD_ARGS[@]}" \
      "${OVERWRITE_ARGS[@]}"

    "${PYTHON_BIN}" "${PACKAGE_ROOT}/final3/vendor/architecture_baselines/llm_logit_baselines/run_local_llm_logits.py" \
      --cache-dir "${cache_dir}" \
      --output-dir "${fold_dir}" \
      --model-name "${MODEL_NAME}" \
      --device "${DEVICE}" \
      --batch-size "${BATCH_SIZE}" \
      --max-input-tokens "${MAX_INPUT_TOKENS}" \
      --prompt-mode "${PROMPT_MODE}" \
      --max-cpu-threads "${THREADS}" \
      "${DOWNLOAD_ARGS[@]}"
  ) >"${log_path}" 2>&1
  echo "[llm-logit] done ${model_id}"
done

echo "[llm-logit] all requested folds finished: ${OUTPUT_ROOT}"
