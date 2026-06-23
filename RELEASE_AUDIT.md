# Release Audit

Audit date: 2026-06-23.

This GitHub-ready bundle was checked against the active training/testing tree at:

`/data3/djs/SweBench/SweBench_Organized_Package_final3`

## Checks Passed

- `final3/` matches the active source tree byte-for-byte, excluding only
  `__pycache__/` and `*.pyc`.
- `scripts/` matches the active source tree byte-for-byte.
- `configs/` matches the active source tree except `configs/paths.yaml`, which
  is intentionally omitted because it is local-machine-specific.
- `paper_reporting/build_rq_tables_bundle.py` matches the active paper table
  builder.
- `paper_reporting/build_internal_review_swe16.py` matches the active SWE
  tokenizer/token audit script.
- `results_tables/` matches the current paper-facing outputs copied from
  `paper/icse_submission_draft/rq_tables_reorg_20260623/`.
- All Python files compile with `python -m py_compile`.
- All shell scripts pass `bash -n`.

## Intentional Exclusions

- Large prefix parquet tables.
- Fold prediction parquet files.
- Trained model artifacts and embedding/tokenizer caches.
- Local-only `configs/paths.yaml`.
- Python bytecode caches.
- Large per-trajectory supporting files.

Run `bash VERIFY_RELEASE_LOCAL.sh /path/to/SweBench_Organized_Package_final3`
to repeat the local consistency audit before pushing to GitHub.

