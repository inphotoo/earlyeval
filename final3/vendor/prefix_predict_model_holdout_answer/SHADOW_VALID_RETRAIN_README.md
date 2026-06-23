# Shadow-valid retrain quick run

这个脚本用于做一个不改旧结果的新验证：

```text
model_holdout_shadow_valid_retrain.py
```

## 默认方案

默认是当前讨论里的“第一个”快速验证口径：

- test 仍然是原来的 3 个 heldout models。
- train 使用所有非 test 模型轨迹，因此每个 selected instance 都可以进入训练。
- valid/calibration 使用同一批非 test 轨迹的 shadow copy，但把 `model_id` 改成 `__MISSING__`。
- 不用 test 做训练、校准、阈值选择。
- 默认复用已有的 `prefix_table_filtered.parquet` 和 `feature_engineer_with_model.pkl`，不重建 prefix/gold join，也不重 fit TF-IDF/SVD。

## 推荐先跑

```bash
cd /path/to/SweBench_Organized_Package_final3

python final3/vendor/prefix_predict_model_holdout_answer/model_holdout_shadow_valid_retrain.py \
  --run-name model_holdout_answer_calibrated_full \
  --output-subdir shadow_valid_retrain_all_non_test \
  --variants default
```

如果 GPU/LightGBM 有问题，改成：

```bash
python final3/vendor/prefix_predict_model_holdout_answer/model_holdout_shadow_valid_retrain.py \
  --run-name model_holdout_answer_calibrated_full \
  --output-subdir shadow_valid_retrain_all_non_test_cpu \
  --variants default \
  --no-gpu-lgbm
```

## 更严格的对照

如果后面要看“每个 instance 内随机留出一部分轨迹做 valid”的版本：

```bash
python final3/vendor/prefix_predict_model_holdout_answer/model_holdout_shadow_valid_retrain.py \
  --run-name model_holdout_answer_calibrated_full \
  --output-subdir per_instance_traj_valid_retrain \
  --split-strategy per_instance_traj \
  --valid-traj-ratio 0.15 \
  --variants default
```

## 输出

默认输出到：

```text
runs/model_holdout_answer_calibrated_full/reports/shadow_valid_retrain_all_non_test/
```

主要文件：

- `summary.txt`
- `final_step_metrics.csv`
- `prefix_metrics.csv`
- `test_predictions_shadow_valid_retrain.parquet`
- `probability_calibration_summary.csv`
- `step_auc_reports/step_bucket_auc_report.txt`
- `calibration_plots/`
- `model_ranking_report_calibrated/report.txt`

## 注意

默认方案是一个快速验证：valid 与 train 轨迹重合，但 valid 的 `model_id` 会被遮掉成 `__MISSING__`，所以它适合回答“如果 known-task pool 都参与训练，heldout model test 还能怎样”的问题。它不是最严格的泛化评估；最严格的对照用 `--split-strategy per_instance_traj`。
