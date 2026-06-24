# Vendor Status

Last audited: 2026-06-04.

This directory is the active earlyeval-owned copy of the answer-aware training,
feature, evaluator, and posthoc source code. The earlyeval CLI resolves the
dual-head LightGBM trainer here by default:

```text
earlyeval/vendor/prefix_predict_model_holdout_answer/safe_stop_dual_head_retrain.py
```

Older package paths are artifact lineage only. Runtime code in the GitHub
release should not import modules from an external old package.

## What Belongs Here

- Training code needed by the earlyeval dual-head LightGBM workflow.
- Answer-aware feature code.
- Posthoc helpers that are still useful for reproducing current table generation or diagnosing current runs.
- Local documentation that explains the feature/training flow.

## What Does Not Belong Here

- Large parquet tables.
- Model binaries, pickles, or run directories.
- Logs, reports, and cache directories.

Large artifacts stay outside Git in shared data/artifact directories and are
tracked through path configuration or external artifact manifests.

## Edit Rule

For future paper or training changes, edit this vendored earlyeval copy. Treat any
old source directory as read-only lineage, not as a runtime dependency.
