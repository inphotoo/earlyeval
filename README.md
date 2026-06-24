# EarlyEval Code Release

This repository is a code-only release for the EarlyEval experiments on
SWE-bench Verified, TerminalBench, and Toolathlon.

It contains the active training, testing, feature-construction, ablation,
architecture-comparison, policy-replay, and table-generation code. It does not
contain generated paper tables, trained models, prediction files, prefix parquet
tables, tokenizer caches, or other run artifacts.

## Repository Contents

- `earlyeval/`: the main Python package used for experiment orchestration,
  checks, dataset contracts, split handling, policy replay, metrics, reports,
  and paper experiment runners.
- `earlyeval/vendor/prefix_predict_model_holdout_answer/`: the vendored
  answer-aware SWE pipeline used by the paper runs. This includes the active
  `run_all.py`, `feature_engineer.py`, trainer, evaluator, policy, and post-hoc
  source code. The release uses this vendored copy instead of importing from an
  older external package.
- `earlyeval/vendor/architecture_baselines/`: architecture baselines used for
  model comparison, including direct MLP, BERT/CodeBERT, local LLM-logit, and
  Qwen fine-tuning code.
- `scripts/`: shell entrypoints for SWE-bench Verified LightGBM
  runs, robustness runs, ablations, LR/TF-IDF, MLP, BERT/CodeBERT, LLM-logit,
  latency/cost audits, and full reproduction orchestration.
- `configs/`: portable experiment configuration and policy presets.
  `configs/paths.yaml` is intentionally not committed because it is
  machine-local; copy `configs/paths.example.yaml` if you want a local override.
- `reporting/`: code that rebuilds the paper-facing RQ tables from
  completed artifacts.

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

For a full reproduction, provide the data paths expected by `configs/earlyeval.yaml`:

- SWE-bench Verified raw trajectory parquet directory, passed as
  `SWE_PARQUET_DIR` to the SWE shared-artifact builder.
- SWE-bench Verified official answer JSONL, defaulting to
  `../data/swe_verify_500/offical_answer/test.jsonl` unless `VERIFIED_JSONL`
  is set.
- TerminalBench and Toolathlon prefix tables if reproducing robustness runs.
  Their default relative paths are listed in `configs/earlyeval.yaml`.
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

Run the SWE-bench Verified held-out-agent LightGBM main experiment:

```bash
bash scripts/run_earlyeval_03_main_lightgbm_execute.sh
bash scripts/run_earlyeval_04_summarize_lightgbm_current.sh
bash scripts/run_earlyeval_05_lightgbm_policy_sweep_valid_acc.sh
bash scripts/run_earlyeval_12_main_latency_cost.sh
```

Run TerminalBench and Toolathlon leave-one-agent robustness:

```bash
bash scripts/run_earlyeval_robustness_loo_answer_features_memory_limited.sh
```

Run the SWE-bench Verified held-out-agent feature and component ablations:

```bash
source scripts/_earlyeval_sweverify_holdout_models.sh

RUN_SUBDIR=sweverify_ablation_feature_groups \
PROFILES=feature_groups \
TEST_MODELS="$(earlyeval_sweverify_holdout_models_string)" \
bash scripts/run_earlyeval_08_ablation_execute.sh

RUN_SUBDIR=sweverify_ablation_feature_groups \
PROFILES=component_with_model_id \
TEST_MODELS="$(earlyeval_sweverify_holdout_models_string)" \
bash scripts/run_earlyeval_08_ablation_execute.sh

bash scripts/run_earlyeval_08_ablation_default_reg_sweverify.sh
bash scripts/run_earlyeval_08_ablation_fine_grained_sweverify.sh
```

Run architecture comparisons:

```bash
bash scripts/run_earlyeval_06_model_compare_lr_tfidf.sh
bash scripts/run_earlyeval_09_direct_mlp_sweverify.sh
bash scripts/run_earlyeval_09_bert_finetune_sweverify.sh
bash scripts/run_earlyeval_09_llm_logit_sweverify.sh
```

Rebuild paper-facing tables from completed artifacts:

```bash
export SWEBENCH_PACKAGE_ROOT="$(pwd)"
export EARLYEVAL_EXPERIMENT_DIR=/path/to/paper/experiments/earlyeval_lightgbm
export EARLYEVAL_PAPER_DATA=/path/to/paper/icse_submission_draft/data
export RQ_TABLES_OUT=/path/to/output/rq_tables

python reporting/build_rq_tables.py
```

The table builder is included as code. Its CSV/TeX outputs are generated files
and are intentionally not committed.

## Main Feature Construction

The main SWE-bench Verified model is `I_LightGBM_Dense_AF`, configured in
`configs/earlyeval.yaml` and trained through the vendored trainer. The important
source files are:

- `earlyeval/vendor/prefix_predict_model_holdout_answer/feature_engineer.py`:
  dense numeric/boolean/categorical features and TF-IDF/SVD text features.
- `earlyeval/vendor/prefix_predict_model_holdout_answer/feature_dictionary.md`:
  human-readable feature family documentation.
- `earlyeval/vendor/prefix_predict_model_holdout_answer/model_holdout_shadow_valid_retrain.py`:
  leave-one-model training/evaluation backbone used by the SWE folds.
- `earlyeval/experiments/paper_pipeline.py`: SWE-bench Verified LightGBM orchestration.
- `earlyeval/experiments/sweverify_ablation.py`: SWE-bench Verified ablation orchestration.
- `earlyeval/experiments/lr_tfidf_baselines.py`: LR/TF-IDF comparison features.
- `earlyeval/vendor/architecture_baselines/train_direct_dual_head_mlp.py`:
  direct MLP baseline over the shared feature representation.
- `earlyeval/vendor/architecture_baselines/bert_baselines/`: BERT/CodeBERT
  baseline feature and fine-tuning code.

For the main run, concrete `model_id` identity is masked from training
features unless a component ablation explicitly enables it.
