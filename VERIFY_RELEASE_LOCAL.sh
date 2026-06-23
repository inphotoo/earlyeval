#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash VERIFY_RELEASE_LOCAL.sh /path/to/SweBench_Organized_Package_final3
#
# This verifies that the code-only GitHub release is aligned with the active
# training/testing source tree. It intentionally ignores generated experiment
# outputs and local-only config files.

if [[ $# -lt 1 ]]; then
  cat >&2 <<'EOF'
Usage:
  bash VERIFY_RELEASE_LOCAL.sh /path/to/SweBench_Organized_Package_final3
EOF
  exit 2
fi

SOURCE_ROOT="$(cd "$1" && pwd)"
RELEASE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "[verify] source=${SOURCE_ROOT}"
echo "[verify] release=${RELEASE_ROOT}"

diff -qr -x '__pycache__' -x '*.pyc' "${SOURCE_ROOT}/final3" "${RELEASE_ROOT}/final3"
diff -qr "${SOURCE_ROOT}/scripts" "${RELEASE_ROOT}/scripts"
diff -qr -x 'paths.yaml' "${SOURCE_ROOT}/configs" "${RELEASE_ROOT}/configs"

cmp -s \
  "${SOURCE_ROOT}/paper/experiments/rq_final_lightgbm_17/build_internal_review_swe16.py" \
  "${RELEASE_ROOT}/paper_reporting/build_internal_review_swe16.py"

cmp -s \
  "${SOURCE_ROOT}/paper/icse_submission_draft/rq_tables_reorg_20260623/build_rq_tables_bundle.py" \
  "${RELEASE_ROOT}/paper_reporting/build_rq_tables_bundle.py"

find "${RELEASE_ROOT}" -type f -name '*.py' -print0 | xargs -0 python -m py_compile
find "${RELEASE_ROOT}/scripts" -type f -name '*.sh' -print0 | xargs -0 -n1 bash -n

echo "[verify] ok: code release matches the current source tree."
