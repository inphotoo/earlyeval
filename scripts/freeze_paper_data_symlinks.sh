#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/.."

PYTHON_BIN="${PYTHON_BIN:-python}"
OUTPUT_DIR="${OUTPUT_DIR:-paper/checks/freeze_paper_data_symlinks}"
MAX_FILE_MB="${MAX_FILE_MB:-64}"

"${PYTHON_BIN}" -m final3.checks.freeze_paper_data_symlinks \
  --output-dir "${OUTPUT_DIR}" \
  --max-file-mb "${MAX_FILE_MB}" \
  "$@"
