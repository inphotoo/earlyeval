# Main Training Feature Manifest

This file documents the feature matrix used by the main SWE-bench Verified full-16 LightGBM run.

## Actual Main Run

The actual command index for the completed run is:

```text
paper/experiments/rq_final_lightgbm_17/lightgbm_main/command_index.csv
```

Each fold command uses:

```text
--variants i
--lgbm-preset strong_reg
--safe-label-min-step 10
--policy-min-steps 0
--consecutive 1
--success-thresholds 0.95
--failure-thresholds 0.95
--score-modes calibrated
--mask-train-model-id
```

The `i` variant maps to:

```text
I_LightGBM_Dense_AF = Dense features + Action/Feedback TF-IDF blocks
```

Both dual-head models, `safe_success` and `safe_failure`, use this same feature matrix. They differ only in the training target:

```text
safe_success = final label is resolved and prefix_step_idx >= 10
safe_failure = final label is unresolved and prefix_step_idx >= 10
```

## Matrix Size

The main `I_LightGBM_Dense_AF` matrix has:

```text
Dense columns: 402
Action/feedback/task TF-IDF SVD columns: 320
Total columns: 722
```

The full column list is in:

```text
main_training_feature_columns.csv
```

The block-level summary is in:

```text
main_training_feature_blocks.csv
```

## Included Feature Blocks

| Block | Columns | Main use |
|:--|--:|:--|
| Dense manual numeric | 66 | yes |
| Dense structured gold-answer numeric | 57 | yes |
| Dense manual boolean | 36 | yes |
| Dense structured gold-answer boolean | 20 | yes |
| Last-action taxonomy one-hot | 13 | yes |
| Gold-answer categorical one-hot | 192 | yes |
| Model-id one-hot | 18 | columns exist, but model id is masked |
| Task prompt TF-IDF SVD | 64 | yes |
| Prefix action TF-IDF SVD | 64 | yes |
| Prefix feedback TF-IDF SVD | 64 | yes |
| Last action TF-IDF SVD | 64 | yes |
| Last feedback TF-IDF SVD | 64 | yes |

## Important Exclusions

The shared `FeatureEngineer` also contains other text blocks, but the main `I_LightGBM_Dense_AF` run does not include them:

| Block | Columns | Where it is used |
|:--|--:|:--|
| Prefix thought TF-IDF SVD | 64 | J/fine-grained ablations, not main I |
| Last thought TF-IDF SVD | 64 | J/fine-grained ablations, not main I |
| Prefix assistant-content TF-IDF SVD | 64 | not main I |
| Last assistant-content TF-IDF SVD | 64 | not main I |
| Gold patch raw-text TF-IDF SVD | 64 | not main I |
| Gold test patch raw-text TF-IDF SVD | 64 | not main I |
| Gold fail-to-pass raw-text TF-IDF SVD | 64 | not main I |
| Gold answer summary raw-text TF-IDF SVD | 64 | not main I |

The main run still uses structured gold-answer dense features: patch sizes, file counts, line counts, fail-to-pass counts, keyword counts, repository/difficulty/version categories, and lexical overlap/hit features between trajectory text and gold-answer tokens.

## Model ID Masking

The shared FeatureEngineer was fit with a `model_id` encoder, so `model_id__...` columns appear in the feature list. The main training command passes `--mask-train-model-id`, and the trainer sets `model_id = __MISSING__` for train, validation, and test before transformation. Therefore concrete agent identity columns are not usable by the main model; only the constant `model_id____MISSING__` indicator is active.

## Source Locations

Feature definitions:

```text
final3/vendor/prefix_predict_model_holdout_answer/feature_engineer.py
final3/vendor/prefix_predict_model_holdout_answer/answer_features.py
```

Main run configuration:

```text
configs/rq_final.yaml
```

Main run orchestration and command construction:

```text
scripts/run_rq_final_03_main_lightgbm_execute.sh
final3/experiments/rq_final.py
final3/vendor/prefix_predict_model_holdout_answer/safe_stop_dual_head_retrain.py
```

Per-fold confirmation artifacts:

```text
paper/experiments/rq_final_lightgbm_17/lightgbm_main/folds/*/variant_manifest.csv
paper/experiments/rq_final_lightgbm_17/lightgbm_main/folds/*/feature_importance_I_LightGBM_Dense_AF__safe_success.csv
paper/experiments/rq_final_lightgbm_17/lightgbm_main/folds/*/feature_importance_I_LightGBM_Dense_AF__safe_failure.csv
```
