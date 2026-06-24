#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/.."

PYTHON_BIN="${PYTHON_BIN:-python}"
OUTPUT_DIR="${OUTPUT_DIR:-paper/checks/portability_audit}"
HASH_LIMIT_MB="${HASH_LIMIT_MB:-256}"

"${PYTHON_BIN}" -m earlyeval.checks.portability_audit \
  --output-dir "${OUTPUT_DIR}" \
  --hash-limit-mb "${HASH_LIMIT_MB}" \
  "$@"
