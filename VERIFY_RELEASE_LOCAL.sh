#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash VERIFY_RELEASE_LOCAL.sh
#
# This verifies the code-only release tree after public-name normalization. It
# intentionally checks source code and entrypoints, not generated experiment
# outputs.

RELEASE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${RELEASE_ROOT}"

echo "[verify] release=${RELEASE_ROOT}"

test -d earlyeval
test -f configs/earlyeval.yaml
test -f configs/paths.example.yaml
test -f paper_reporting/build_rq_tables_bundle.py

find earlyeval paper_reporting -type f -name '*.py' -print0 | xargs -0 python -m py_compile
find scripts -type f -name '*.sh' -print0 | xargs -0 -n1 bash -n

python -m earlyeval.cli --help >/dev/null
python -m earlyeval.cli experiment list --registry configs/experiment_registry.yaml >/dev/null

bad_names=(
  "final""3"
  "rq_""final"
  "lightgbm_""17"
  "SweBench_Organized_Package_""final""3"
)
for bad_name in "${bad_names[@]}"; do
  if grep -RInF "${bad_name}" README.md configs earlyeval paper_reporting scripts; then
    echo "[verify] internal release name remains: ${bad_name}" >&2
    exit 1
  fi
done

echo "[verify] ok: code release is self-consistent."
