# Release Audit

Audit date: 2026-06-23.

This is a code-only GitHub release. It is intended to preserve the active
training/testing implementation, not generated experiment outputs.

## Source Alignment

The release is checked against the active local source tree used for the
experiments. Pass that source path explicitly to `VERIFY_RELEASE_LOCAL.sh`.

The following paths are expected to match that source tree:

- `final3/`, excluding Python bytecode caches;
- `scripts/`;
- `configs/`, excluding local-only `configs/paths.yaml`;
- `paper_reporting/build_rq_tables_bundle.py`;
- `paper_reporting/build_internal_review_swe16.py`.

## Included Code Scope

- Main SWE-bench Verified full-16 LightGBM orchestration.
- Vendored answer-aware feature engineering and model-holdout trainer.
- Policy replay, threshold sweeps, latency/cost audits, and reporting helpers.
- SWE full-16 feature/component ablation runners.
- TerminalBench and Toolathlon robustness runners.
- LR/TF-IDF, direct MLP, BERT/CodeBERT, local LLM-logit, and Qwen baseline code.
- Paper table-generation code.

## Intentional Exclusions

The release does not include generated artifacts:

- raw or processed parquet tables;
- FeatureEngineer pickles and trained model files;
- prediction parquet files and policy sweep outputs;
- tokenizer, embedding, and model caches;
- generated CSV/TeX paper tables;
- generated feature manifests.

These files can be rebuilt or distributed separately as data artifacts.

## Verification

Run the local audit before pushing:

```bash
bash VERIFY_RELEASE_LOCAL.sh /path/to/SweBench_Organized_Package_final3
```

The audit performs source-tree comparisons, Python compilation, and shell syntax
checks.
