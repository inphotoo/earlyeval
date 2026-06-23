# EarlyEval Code Release

This is the code-only GitHub-ready bundle for the EarlyEval/SWE-bench Verified,
TerminalBench, and Toolathlon experiments. It intentionally excludes large data,
prediction parquet files, tokenizer caches, and per-trajectory supporting CSVs.

## What Is Included

- `final3/`: Python package for feature construction, LightGBM training,
  policy replay, ablations, robustness evaluation, and reporting helpers.
- `scripts/`: shell entrypoints used to run the final experiments.
- `configs/`: experiment configuration. `paths.yaml` is intentionally omitted;
  start from `configs/paths.example.yaml` for a new machine.
- `paper_reporting/build_rq_tables_bundle.py`: final RQ1/RQ2/RQ3 table builder.
- `paper_reporting/build_internal_review_swe16.py`: SWE tokenizer/ranking/token
  audit used by the internal review tables.
- `results_tables/`: small paper-facing CSV/LaTeX outputs from the latest run.

## What Is Not Included

- Raw or processed prefix parquet tables.
- Trained model/fold prediction parquet files.
- Embedding/tokenizer/model caches.
- The 63MB `supporting/locked095_decisions_all_benchmarks.csv` style files.

Those should be published separately as artifacts if reviewers need exact
post-hoc regeneration.

## Main Entry Points

Run the SWE full-16 LightGBM pipeline:

```bash
bash scripts/run_pure16_full_pipeline.sh
```

Run SWE full-16 ablations:

```bash
bash scripts/run_rq_final_08_ablation_default_reg_full16.sh
bash scripts/run_rq_final_08_ablation_fine_grained_full16.sh

RUN_SUBDIR=sweverify_ablation_feature_groups_full16 \
PROFILES=feature_groups \
TEST_MODELS="$(bash -lc 'source scripts/_rq_final_full16_models.sh; rq_final_full16_models_string')" \
bash scripts/run_rq_final_08_ablation_execute.sh
```

Run architecture comparisons:

```bash
bash scripts/run_rq_final_09_direct_mlp_full16.sh
bash scripts/run_rq_final_09_bert_finetune_full16.sh
```

Run TerminalBench/Toolathlon leave-one-agent robustness:

```bash
bash scripts/run_rich_loo_hard_memory_limited.sh
```

Regenerate the final RQ tables from completed artifacts:

```bash
export SWEBENCH_PACKAGE_ROOT="$(pwd)"
export EARLYEVAL_EXPERIMENT_DIR="/path/to/paper/experiments/rq_final_lightgbm_17"
export EARLYEVAL_PAPER_DATA="/path/to/paper/icse_submission_draft/data"
export RQ_TABLES_OUT="$(pwd)/results_tables_regenerated"

python paper_reporting/build_rq_tables_bundle.py
```

## Current Paper Outputs

The latest small outputs are already copied into `results_tables/`:

- `rq1_main.csv`
- `rq1_threshold_sweep_compact.csv`
- `threshold_sweep_all_benchmarks.csv`
- `rq2_top10.csv`
- `rq2_per_agent_all.csv`
- `rq2_summary.csv`
- `rq3_ablation_locked095_paper.csv`
- `token_input_output_summary.csv`
- `token_input_output_by_agent.csv`
- `tables_latex_draft.tex`

`model_price_template.csv` is included, but Saved$ is not filled because it
requires a model-specific input/output price table.

## Reproducibility Notes

- The locked main operating point is calibrated dual-head `s=f=0.95`,
  `min_step=0`, `consecutive=1`.
- SWE-bench Verified uses the full-16 `lightgbm_main` folds.
- TerminalBench and Toolathlon use the rich leave-one-agent folds.
- Main token savings use a uniform chars/4 estimate across all three
  benchmarks: input/context-call tokens for skipped future calls, and generated
  output tokens for skipped model text.

