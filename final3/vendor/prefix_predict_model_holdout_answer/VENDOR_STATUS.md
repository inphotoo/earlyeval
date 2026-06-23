# Vendor Status

Last audited: 2026-06-04.

This directory is the active final3-owned copy of the answer-aware training and feature code. The final3 CLI now resolves the dual-head LightGBM trainer here by default:

```text
final3/vendor/prefix_predict_model_holdout_answer/safe_stop_dual_head_retrain.py
```

The legacy source directory remains available for old-run reproduction and artifact lineage:

```text
../SweBench_Organized_Package_final/modules/prefix_predict_model_holdout_answer/
```

## What Belongs Here

- Training code needed by the final3 dual-head LightGBM workflow.
- Answer-aware feature code.
- Posthoc helpers that are still useful for reproducing current table generation or diagnosing current runs.
- Local documentation that explains the feature/training flow.

## What Does Not Belong Here

- Large parquet tables.
- Model binaries, pickles, or run directories.
- Logs, reports, and cache directories.

Large artifacts stay in the shared data or legacy run roots and are tracked through `configs/paths.yaml` and `manifests/artifact_manifest.yaml`.

## Edit Rule

For future paper or training changes, edit this vendored final3 copy first. Treat the old source directory as read-only lineage unless you intentionally need to reproduce an old command exactly.
