#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash VERIFY_RELEASE_LOCAL.sh /path/to/SweBench_Organized_Package_final3
#
# This verifies that the GitHub-ready code bundle is byte-for-byte aligned with
# the current training/testing source tree, excluding intentionally omitted
# local-only files and Python bytecode caches.

SOURCE_ROOT="${1:-/data3/djs/SweBench/SweBench_Organized_Package_final3}"
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

for f in \
  rq1_main.csv \
  rq1_threshold_sweep_compact.csv \
  threshold_sweep_all_benchmarks.csv \
  rq2_top10.csv \
  rq2_per_agent_all.csv \
  rq2_summary.csv \
  rq3_ablation_locked095.csv \
  rq3_ablation_locked095_paper.csv \
  token_input_output_summary.csv \
  token_input_output_by_agent.csv \
  main_training_feature_manifest.md \
  main_training_feature_blocks.csv \
  main_training_feature_columns.csv \
  model_price_template.csv \
  tables_latex_draft.tex
do
  cmp -s \
    "${SOURCE_ROOT}/paper/icse_submission_draft/rq_tables_reorg_20260623/${f}" \
    "${RELEASE_ROOT}/results_tables/${f}"
done

find "${RELEASE_ROOT}" -type f -name '*.py' -print0 | xargs -0 python -m py_compile
find "${RELEASE_ROOT}/scripts" -type f -name '*.sh' -print0 | xargs -0 -n1 bash -n

echo "[verify] ok: release bundle matches the current source tree."
