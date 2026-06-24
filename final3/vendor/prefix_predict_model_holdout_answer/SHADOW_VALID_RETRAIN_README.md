# Shadow-valid retrain quick run

description:

```text
model_holdout_shadow_valid_retrain.py
```

## description

description"description"description:

- test description 3 description heldout models.
- train description test description,description selected instance description.
- valid/calibration description test description shadow copy,description `model_id` description `__MISSING__`.
- description test description,description,description.
- description `prefix_table_filtered.parquet` description `feature_engineer_with_model.pkl`,description prefix/gold join,description fit TF-IDF/SVD.

## description

```bash
cd /path/to/SweBench_Organized_Package_final3

python final3/vendor/prefix_predict_model_holdout_answer/model_holdout_shadow_valid_retrain.py \
  --run-name model_holdout_answer_calibrated_full \
  --output-subdir shadow_valid_retrain_all_non_test \
  --variants default
```

description GPU/LightGBM description,description:

```bash
python final3/vendor/prefix_predict_model_holdout_answer/model_holdout_shadow_valid_retrain.py \
  --run-name model_holdout_answer_calibrated_full \
  --output-subdir shadow_valid_retrain_all_non_test_cpu \
  --variants default \
  --no-gpu-lgbm
```

## description

description"description instance description valid"description:

```bash
python final3/vendor/prefix_predict_model_holdout_answer/model_holdout_shadow_valid_retrain.py \
  --run-name model_holdout_answer_calibrated_full \
  --output-subdir per_instance_traj_valid_retrain \
  --split-strategy per_instance_traj \
  --valid-traj-ratio 0.15 \
  --variants default
```

## description

description:

```text
runs/model_holdout_answer_calibrated_full/reports/shadow_valid_retrain_all_non_test/
```

description:

- `summary.txt`
- `final_step_metrics.csv`
- `prefix_metrics.csv`
- `test_predictions_shadow_valid_retrain.parquet`
- `probability_calibration_summary.csv`
- `step_auc_reports/step_bucket_auc_report.txt`
- `calibration_plots/`
- `model_ranking_report_calibrated/report.txt`

## description

description:valid description train description,description valid description `model_id` description `__MISSING__`,description"description known-task pool description,heldout model test description"description.description;description `--split-strategy per_instance_traj`.
