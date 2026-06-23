# EarlyEval Code Release

This repository is a code-only release for the EarlyEval experiments on
SWE-bench Verified, TerminalBench, and Toolathlon.

It contains the active training, testing, feature-construction, ablation,
architecture-comparison, policy-replay, and table-generation code. It does not
contain generated paper tables, trained models, prediction files, prefix parquet
tables, tokenizer caches, or other run artifacts.

## Repository Contents

- `final3/`: the main Python package used for experiment orchestration,
  checks, dataset contracts, split handling, policy replay, metrics, reports,
  and final RQ experiment runners.
- `final3/vendor/prefix_predict_model_holdout_answer/`: the vendored
  answer-aware SWE pipeline used by the final runs. This includes the active
  `run_all.py`, `feature_engineer.py`, trainer, evaluator, policy, and post-hoc
  source code. The release uses this vendored copy instead of importing from an
  older external package.
- `final3/vendor/architecture_baselines/`: architecture baselines used for
  model comparison, including direct MLP, BERT/CodeBERT, local LLM-logit, and
  Qwen fine-tuning code.
- `scripts/`: shell entrypoints for preflight checks, SWE full-16 LightGBM
  runs, robustness runs, ablations, LR/TF-IDF, MLP, BERT/CodeBERT, LLM-logit,
  latency/cost audits, and full reproduction orchestration.
- `configs/`: portable experiment configuration and policy presets.
  `configs/paths.yaml` is intentionally not committed because it is
  machine-local; copy `configs/paths.example.yaml` if you want a local override.
- `paper_reporting/`: code that rebuilds the paper-facing RQ tables from
  completed artifacts.
- `VERIFY_RELEASE_LOCAL.sh`: local audit script that checks this code release
  against the active training/testing source tree.

## Not Included

The repository intentionally excludes generated artifacts:

- raw SWE trajectory parquet files;
- processed `prefix_table*.parquet` and `step_table.parquet` files;
- FeatureEngineer pickle files and trained fold models;
- fold prediction parquet files and policy sweep outputs;
- tokenizer, embedding, and model-download caches;
- generated CSV/TeX paper tables and feature manifests.

Those files are outputs or external inputs, not source code. Rebuild them with
the commands below or publish them separately as data artifacts.

## Environment

Install the Python dependencies in a fresh environment:

```bash
python -m pip install -r requirements-github.txt
```

The scripts default to the `python` on `PATH`. To pin a specific interpreter:

```bash
export PYTHON_BIN=/path/to/python
```

Optional local path configuration:

```bash
cp configs/paths.example.yaml configs/paths.yaml
# edit configs/paths.yaml for local data/artifact roots if needed
```

If `configs/paths.yaml` is absent, the code falls back to
`configs/paths.example.yaml`.

## Required Inputs

For a full reproduction, provide the data paths expected by `configs/rq_final.yaml`:

- SWE-bench Verified raw trajectory parquet directory, passed as
  `SWE_PARQUET_DIR` to the SWE shared-artifact builder.
- SWE-bench Verified official answer JSONL, defaulting to
  `../data/swe_verify_500/offical_answer/test.jsonl` unless `VERIFIED_JSONL`
  is set.
- TerminalBench and Toolathlon prefix tables if reproducing robustness runs.
  Their default relative paths are listed in `configs/rq_final.yaml`.
- Optional Hugging Face or local model caches for BERT/CodeBERT, local
  LLM-logit, and Qwen baselines.

## Full Reproduction Driver

The high-level orchestrator is:

```bash
bash scripts/run_earlyeval_full_reproduction.sh
```

By default it runs preflight checks and a dry-run plan. Enable stages with
environment flags:

```bash
BUILD_SWE_SHARED=1 \
RUN_MAIN=1 \
RUN_ROBUSTNESS=1 \
RUN_ABLATIONS=1 \
RUN_LR_TFIDF=1 \
RUN_MLP=1 \
RUN_BERT=1 \
RUN_LLM_LOGIT=1 \
BUILD_TABLES=1 \
SWE_PARQUET_DIR=/path/to/swe/tool-parquets \
bash scripts/run_earlyeval_full_reproduction.sh
```

The driver writes generated artifacts under the configured experiment/data
roots; those outputs are not part of this code-only repository.

## Stage-by-Stage Commands

Build the shared SWE prefix and FeatureEngineer artifacts from raw SWE parquet:

```bash
SWE_PARQUET_DIR=/path/to/swe/tool-parquets \
bash scripts/run_earlyeval_00_build_swe_shared_artifacts.sh
```

Run the SWE-bench Verified full-16 LightGBM main experiment:

```bash
bash scripts/run_rq_final_03_main_lightgbm_execute.sh
bash scripts/run_rq_final_04_summarize_lightgbm_current.sh
bash scripts/run_rq_final_05_lightgbm_policy_sweep_valid_acc.sh
bash scripts/run_rq_final_12_main_latency_cost.sh
```

Run TerminalBench and Toolathlon leave-one-agent robustness:

```bash
bash scripts/run_rich_loo_hard_memory_limited.sh
```

Run the SWE-bench Verified full-16 feature and component ablations:

```bash
source scripts/_rq_final_full16_models.sh

RUN_SUBDIR=sweverify_ablation_feature_groups_full16 \
PROFILES=feature_groups \
TEST_MODELS="$(rq_final_full16_models_string)" \
bash scripts/run_rq_final_08_ablation_execute.sh

RUN_SUBDIR=sweverify_ablation_feature_groups_full16 \
PROFILES=component_with_model_id \
TEST_MODELS="$(rq_final_full16_models_string)" \
bash scripts/run_rq_final_08_ablation_execute.sh

bash scripts/run_rq_final_08_ablation_default_reg_full16.sh
bash scripts/run_rq_final_08_ablation_fine_grained_full16.sh
```

Run architecture comparisons:

```bash
bash scripts/run_rq_final_06_model_compare_lr_tfidf.sh
bash scripts/run_rq_final_09_direct_mlp_full16.sh
bash scripts/run_rq_final_09_bert_finetune_full16.sh
bash scripts/run_rq_final_09_llm_logit_full16.sh
```

Rebuild paper-facing tables from completed artifacts:

```bash
export SWEBENCH_PACKAGE_ROOT="$(pwd)"
export EARLYEVAL_EXPERIMENT_DIR=/path/to/paper/experiments/rq_final_lightgbm_17
export EARLYEVAL_PAPER_DATA=/path/to/paper/icse_submission_draft/data
export RQ_TABLES_OUT=/path/to/output/rq_tables

python paper_reporting/build_rq_tables_bundle.py
```

The table builder is included as code. Its CSV/TeX outputs are generated files
and are intentionally not committed.

## Main Feature Construction

The main SWE-bench Verified model is `I_LightGBM_Dense_AF`, configured in
`configs/rq_final.yaml` and trained through the vendored trainer. The important
source files are:

- `final3/vendor/prefix_predict_model_holdout_answer/feature_engineer.py`:
  dense numeric/boolean/categorical features and TF-IDF/SVD text features.
- `final3/vendor/prefix_predict_model_holdout_answer/feature_dictionary.md`:
  human-readable feature family documentation.
- `final3/vendor/prefix_predict_model_holdout_answer/model_holdout_shadow_valid_retrain.py`:
  leave-one-model training/evaluation backbone used by the final SWE folds.
- `final3/experiments/rq_final.py`: final full-16 LightGBM orchestration.
- `final3/experiments/rq_final_ablation.py`: full-16 ablation orchestration.
- `final3/experiments/lr_tfidf_baselines.py`: LR/TF-IDF comparison features.
- `final3/vendor/architecture_baselines/train_direct_dual_head_mlp.py`:
  direct MLP baseline over the shared feature representation.
- `final3/vendor/architecture_baselines/bert_baselines/`: BERT/CodeBERT
  baseline feature and fine-tuning code.

For the final main run, concrete `model_id` identity is masked from training
features unless a component ablation explicitly enables it.

## Local Release Audit

Before pushing a refreshed code release, run:

```bash
bash VERIFY_RELEASE_LOCAL.sh /path/to/SweBench_Organized_Package_final3
```

The audit checks that `final3/`, `scripts/`, `configs/` except
`configs/paths.yaml`, and the reporting scripts match the active source tree.
It also compiles Python files and syntax-checks shell scripts.
