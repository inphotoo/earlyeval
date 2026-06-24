# Recent Model-Holdout Experiments Audit

description:2026-04-28

description `model_holdout_answer_calibrated_full` description,description,description.

description:

```text
MODEL_HOLDOUT_ANSWER_FINAL_CONSOLIDATED_REPORT.md
```

description,description,gold answer TF-IDF,step-bucket AUC,calibrated/raw fine-grid description.

## 1. description

### 1.1 description heldout agent model description valid

description:

- description heldout description test description:
  - `20251124_mini-v1.17.0_minimax-m2`
  - `20251201_mini-v1.17.1_deepseek-v3.2-reasoner`
  - `20251210_mini-v1.17.2_kimi-k2-thinking`
- train description heldout model description:`0`
- valid description heldout model description:`0`
- test description heldout model description:`0`
- test description `model_id` description `__MISSING__`,description `orig_model_id` description,description.

description,description:

> description valid description?

description:**description.**

### 1.2 description instance-holdout

description:

- description **model-holdout**,description **instance-holdout**.
- description SWE-bench description agent model description train/valid.
- heldout test description"description agent model description",description"description".

description I/J description step=0 description:description,description,task prompt description.

## 2. description

description:

```text
runs/model_holdout_answer_calibrated_full
```

### 2.1 prefix table description

description:

```text
runs/model_holdout_answer_calibrated_full/data/prefix_table_filtered.parquet
```

description:

| description | description |
|---|---:|
| description prefix rows | `386380` |
| description trajectories | `9685` |
| description instances | `490` |
| heldout prefix rows | `83169` |
| heldout trajectories | `1458` |

description heldout description:

| heldout model | rows | trajs | instances | prefix-row pos rate |
|---|---:|---:|---:|---:|
| `20251124_mini-v1.17.0_minimax-m2` | `36938` | `488` | `488` | `0.5089` |
| `20251201_mini-v1.17.1_deepseek-v3.2-reasoner` | `22740` | `482` | `482` | `0.5649` |
| `20251210_mini-v1.17.2_kimi-k2-thinking` | `23491` | `488` | `488` | `0.5378` |

### 2.2 split description train/valid/test

description:

```text
runs/model_holdout_answer_calibrated_full/reports/model_holdout_split_summary.csv
runs/model_holdout_answer_calibrated_full/reports/task_answer_ablation_no_task_no_gold/split_reconstruction_summary.json
```

| split | description heldout model | rows | trajs | model description |
|---|---|---:|---:|---:|
| train | description | `259854` | `6417` | `17` |
| valid | description | `42419` | `1130` | `17` |
| test | description | `83169` | `1458` | `3` |

description:

| description | description |
|---|---:|
| train description heldout rows | `0` |
| valid description heldout rows | `0` |
| test description non-heldout rows | `0` |
| `train_models ∩ holdout_models` | description |
| `valid_models ∩ holdout_models` | description |
| `test_models == holdout_models` | description |

### 2.3 description

description:

```text
runs/model_holdout_answer_calibrated_full/reports/test_predictions_all_models.parquet
runs/model_holdout_answer_calibrated_full/reports/task_answer_ablation_posthoc/test_predictions_task_answer_ablation.parquet
runs/model_holdout_answer_calibrated_full/reports/task_answer_ablation_no_task_no_gold/test_predictions_task_answer_ablation.parquet
```

description:

| description | description |
|---|---|
| rows | `83169` |
| trajectories | `1458` |
| instances | `489` |
| `split` | description `test` |
| `model_id` description | description `__MISSING__` |
| `model_id_input_mode` | description `test_missing` |
| `orig_model_id` | description heldout model |
| non-heldout `orig_model_id` rows | `0` |

## 3. description heldout model description

description:

```text
model_holdout_split.py
run_all.py
gold_text_tfidf_ablation_posthoc.py
task_and_answer_ablation_posthoc.py
feature_engineer.py
```

### 3.1 split description

`model_holdout_split.py` description:

```python
trainval = work[~work["model_id"].isin(heldout)]
test = work[work["model_id"].isin(heldout)]
```

description:

- heldout description test.
- description heldout description train/valid.
- description `train_models & test_models`,description.

### 3.2 description train description fit

`run_all.py` description:

```python
fe_with_model.fit(df_train)
fe_no_model.fit(df_train)
```

description:

- dense description train description.
- TF-IDF/SVD description train description fit.
- valid/test description transform,description fit.

### 3.3 valid/test description

`run_all.py` description posthoc description:

```python
df_train["model_id_input_mode"] = "train_seen"
df_valid["model_id"] = "__MISSING__"
df_test["model_id"] = "__MISSING__"
```

description:

- train description.
- valid/test description.
- heldout test description `orig_model_id`,description.

### 3.4 description heldout test label

description:

- description valid description raw probability description label.
- description validation-only sigmoid/Platt calibration.

description:

- description valid description sweep description.
- test description valid description.

description:

- valid description heldout model description.
- description valid description SWE instances description heldout test description,description model-holdout,description instance-holdout.

## 4. description

### 4.1 prefix-row description AUC

description:

```text
runs/model_holdout_answer_calibrated_full/reports/evaluation_report.txt
```

description `N=83169`,description prefix rows description.

| Model | ROC-AUC | PR-AUC | Brier |
|---|---:|---:|---:|
| `H_LightGBM_Dense` | `0.8773` | `0.8762` | `0.1422` |
| `I_LightGBM_Dense_AF` | `0.8839` | `0.8893` | `0.1406` |
| `J_LightGBM_Dense_AF_Thought` | `0.8803` | `0.8837` | `0.1432` |
| `K_LightGBM_Dense_Full` | `0.8623` | `0.8508` | `0.1759` |
| `D_Dense_Full_LR` | `0.7814` | `0.7927` | `0.2015` |
| `G_TfIdf_Full_LR` | `0.7864` | `0.7918` | `0.1933` |

description:

- LightGBM description setting description LR.
- `I/J` description,`K` description full raw text description,description raw text TF-IDF description.

### 4.2 final-step trajectory AUC

description:

```text
runs/model_holdout_answer_calibrated_full/reports/task_answer_ablation_posthoc/summary.txt
runs/model_holdout_answer_calibrated_full/reports/task_answer_ablation_no_task_no_gold/summary.txt
```

description trajectory description,`rows=1458`.

| Model                                    | Acc@0.5 | ROC-AUC | PR-AUC | Brier |
|---|---:|---:|---:|---:|
| `J_LightGBM_Dense_AF_Thought`            | `0.8258` | `0.9007` | `0.9295` | `0.1276` |
| `I_LightGBM_Dense_AF`                    | `0.8251` | `0.9000` | `0.9299` | `0.1282` |
| `Abl_NoTaskSignal_LightGBM`              | `0.8313` | `0.8960` | `0.9264` | `0.1296` |
| `Abl_NoTaskPromptTfidf_LightGBM`         | `0.8354` | `0.8946` | `0.9225` | `0.1289` |
| `Abl_NoGoldAnswer_LightGBM`              | `0.7888` | `0.8271` | `0.8520` | `0.1771` |
| `Abl_NoTaskSignal_NoGoldAnswer_LightGBM` | `0.7469` | `0.8054` | `0.8583` | `0.1778` |

description:

- description task prompt TF-IDF description,AUC description `0.9000/0.9007` description `0.8946`.
- description task signal description,description `0.8960`.
- description gold answer description,description `0.8271`.
- task + gold description,description `0.8054`,description prefix description.

## 5. step-bucket AUC:description"description"

description:

```text
runs/model_holdout_answer_calibrated_full/reports/task_answer_ablation_no_task_no_gold/step_bucket_auc_report.txt
runs/model_holdout_answer_calibrated_full/reports/task_answer_ablation_step_buckets/step_bucket_report_all_task_answer_models.txt
runs/model_holdout_answer_calibrated_full/reports/task_answer_ablation_step_buckets/step_bucket_metrics_all_task_answer_models.csv
```

description calibrated probabilities.ROC-AUC description,description.

| bucket | `J` | `I` | `NoTaskSignal` | `NoTaskPromptTfidf` | `NoGoldAnswer` | `NoTask+NoGold` |
|---|---:|---:|---:|---:|---:|---:|
| `step=0`| `0.8846` | `0.8890` | `0.8508` | `0.8533` | `0.8267` | `0.5000` |
| `1-3`   | `0.8851` | `0.8908` | `0.8603` | `0.8629` | `0.8304` | `0.6647` |
| `4-6`   | `0.8876` | `0.8919` | `0.8700` | `0.8741` | `0.8304` | `0.7464` |
| `7-12`  | `0.8896` | `0.8940` | `0.8773` | `0.8794` | `0.8324` | `0.7711` |
| `13-24` | `0.8938` | `0.8978` | `0.8837` | `0.8828` | `0.8359` | `0.7910` |
| `25+`   | `0.8656` | `0.8693` | `0.8599` | `0.8593` | `0.8018` | `0.7841` |
| `final` | `0.9007` | `0.9000` | `0.8960` | `0.8946` | `0.8271` | `0.8054` |

description:

- `NoTask+NoGold` description `step=0` description,AUC description `0.5000`.
- description AUC description,description prefix description action,feedback,thought,description,description,description,API token description.
- `I/J` description `step=0` description,description task prompt description gold answer description,description"description".
- `NoTaskSignal` description `NoTaskPromptTfidf` description `step=0` description,description task prompt description;gold answer description.
- `NoGoldAnswer` description `step=0` description `0.8267`,description task prompt / repo / difficulty-like static signal description.

## 6. description

description AUC description,description sanity check:

> description SWE instance description heldout description resolved rate,description heldout description.

description:

| description | description |
|---|---:|
| test trajectories | `1458` |
| test instances | `489` |
| prior missing | `0` |
| prior ROC-AUC | `0.9071` |
| prior PR-AUC | `0.9177` |

description:

- description/description.
- description,description.
- description I/J description step=0 AUC description:description.

description:

- description:**description,description agent model description,description.**
- description:**description SWE instance description.**

## 7. Gold raw-text TF-IDF description

### 7.1 `gold_patch_tfidf` description

description:

```text
runs/model_holdout_answer_calibrated_full/reports/gold_text_tfidf_patch_dims/summary.txt
```

final-step calibrated:

| Model | Acc@0.5 | ROC-AUC | Brier |
|---|---:|---:|---:|
| `I_LightGBM_Dense_AF`  | `0.8251` | `0.9000` | `0.1282` |
| `GoldPatchTfidf_Dim8`  | `0.8210` | `0.8992` | `0.1298` |
| `GoldPatchTfidf_Dim16` | `0.8278` | `0.8994` | `0.1297` |
| `GoldPatchTfidf_Dim32` | `0.8278` | `0.8994` | `0.1281` |

description:

- patch raw text TF-IDF description.
- description 8 description 32,AUC description,description `I`.

### 7.2 `gold_test_patch_tfidf` description `gold_fail_to_pass_tfidf`

description:

```text
runs/model_holdout_answer_calibrated_full/reports/gold_text_tfidf_other_dim16/summary.txt
```

final-step calibrated:

| Model | Acc@0.5 | ROC-AUC | Brier |
|---|---:|---:|---:|
| `I_LightGBM_Dense_AF` | `0.8251` | `0.9000` | `0.1282` |
| `GoldTestPatchTfidf_Dim16` | `0.8244` | `0.8996` | `0.1307` |
| `GoldFailToPassTfidf_Dim16` | `0.8320` | `0.9008` | `0.1271` |

description:

- `gold_fail_to_pass_tfidf` description final-step AUC description:`0.9000 -> 0.9008`.
- description prefix-row AUC description `I`:`0.8839 -> 0.8806`.
- description raw text TF-IDF description;description gold answer description.

## 8. I/J description

description LightGBM description gain description:

### 8.1 `I_LightGBM_Dense_AF`

| description | gain description |
|---|---:|
| task prompt TF-IDF/SVD | `51.9%` |
| gold structured answer | `30.5%` |
| model_id one-hot | `6.6%` |
| feedback TF-IDF/SVD | `4.5%` |
| action TF-IDF/SVD | `2.9%` |
| process dense | `2.8%` |

### 8.2 `J_LightGBM_Dense_AF_Thought`

| description | gain description |
|---|---:|
| task prompt TF-IDF/SVD | `48.0%` |
| gold structured answer | `31.9%` |
| model_id one-hot | `5.9%` |
| thought TF-IDF/SVD | `5.2%` |
| feedback TF-IDF/SVD | `3.2%` |
| process dense | `2.9%` |
| action TF-IDF/SVD | `2.5%` |

description:

- I/J description,description.
- `model_id one-hot` description train description,description test description `model_id` description `__MISSING__`,description heldout description.
- description setting description,LightGBM description"description/description -> description"description.

## 9. valid-only description

description:

```text
runs/model_holdout_answer_calibrated_full/reports/threshold_tuning_summary/threshold_tuning_summary.md
runs/model_holdout_answer_calibrated_full/reports/threshold_tuning_summary/rate_preserving_policy_summary.md
runs/model_holdout_answer_calibrated_full/reports/threshold_tuning_summary/rate_preserving_policy_all_grids.csv
runs/model_holdout_answer_calibrated_full/reports/threshold_tuning_summary/policy_selected_thresholds_compact.csv
runs/model_holdout_answer_calibrated_full/reports/threshold_tuning_summary/target_precision_calibrated_grid025_compact.csv
runs/model_holdout_answer_calibrated_full/reports/threshold_tuning_summary/target_precision_raw_grid001_compact.csv
```

description:

1. **description valid description**.
2. description `abs(valid ΔRate)` description,description `0.5pp / 1pp / 2pp`.
3. description rate description,description **valid Save description**description.
4. test description valid description;test description.

description"description rate,description save",description:

```text
runs/model_holdout_answer_calibrated_full/reports/threshold_tuning_summary/rate_preserving_policy_summary.md
runs/model_holdout_answer_calibrated_full/reports/threshold_tuning_summary/rate_preserving_policy_all_grids.csv
```

`target_precision_*` description:description"description success/failure description precision description",description.

### 9.1 description

| Run dir | description | description / description | description |
|---|---|---|---|
| `asymmetric_valid_threshold_tuning_fine` | calibrated `prob_cal__` | `ThrS=0.65..0.95` step `0.025`; `ThrF=0.05..0.45` step `0.025`; policies: `rate_1pp/rate_2pp/prec90` | description valid description,description test |
| `asymmetric_valid_threshold_tuning_fine_calibrated_step001_rate` | calibrated `prob_cal__` | `ThrS=0.65..0.95` step `0.001`; `ThrF=0.05..0.45` step `0.001` | calibrated description;description rate-preserving policy |
| `asymmetric_valid_threshold_tuning_fine_raw` | raw `prob__` | description `0.025` description policies | raw description policy sweep |
| `asymmetric_valid_threshold_tuning_fine_raw_step001` | raw `prob__` | `ThrS=0.65..0.95` step `0.001`; `ThrF=0.05..0.45` step `0.001` | description raw description,description `724206` description valid/test sweep rows |
| `two_end_precision_targets_fine` | calibrated `prob_cal__` | target precision = `0.75/0.80/0.85/0.90`,description `0.025` | valid description/description precision description target,description |
| `two_end_precision_targets_fine_raw_step001` | raw `prob__` | target precision = `0.75/0.80/0.85/0.90`,description `0.001` | description raw description target-precision description |

description:

- `ThrS` description success threshold:`p >= ThrS` description success.
- `ThrF` description failure threshold:`p <= ThrF` description failure.
- `target_precision=0.90` description"overall Acc description 90%",description valid description **success description precision description failure description precision description >= 90%**.
- `Test Acc / Test PrecS / Test PrecF` description heldout test description.
- description `0.025` description"description",description;description raw description calibrated description `0.001` description.

### 9.2 description:calibrated grid 0.001,rate-preserving `rate_1pp`

description:valid description `abs(ΔRate) <= 1pp`,description valid Save.description test description valid description heldout-test description.

| Model | ThrS | ThrF | Valid ΔRate | Valid Save | Valid Acc | Test ΔRate | Test Save | Test Acc | Test PrecS | Test PrecF | FP | FN | N |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `J` | `0.791` | `0.450` | `+0.6%` | `70.5%` | `75.1%` | `-5.4%` | `78.6%` | `84.2%` | `91.3%` | `76.5%` | `55` | `134` | `1200` |
| `I` | `0.801` | `0.450` | `+0.9%` | `61.6%` | `78.0%` | `-5.8%` | `73.0%` | `84.6%` | `92.2%` | `77.1%` | `44` | `129` | `1127` |
| `H` | `0.783` | `0.450` | `+0.8%` | `62.9%` | `75.2%` | `-0.8%` | `76.7%` | `86.5%` | `89.5%` | `82.3%` | `75` | `86` | `1196` |
| `K` | `0.743` | `0.387` | `+0.8%` | `50.2%` | `76.8%` | `-0.1%` | `60.8%` | `89.6%` | `90.9%` | `88.1%` | `46` | `48` | `908` |
| `G` | `0.728` | `0.450` | `+0.7%` | `69.7%` | `65.6%` | `-2.9%` | `77.3%` | `74.3%` | `81.2%` | `64.4%` | `137` | `179` | `1231` |
| `D` | `0.802` | `0.450` | `+1.0%` | `75.0%` | `75.2%` | `-8.2%` | `79.4%` | `73.6%` | `83.2%` | `63.9%` | `107` | `226` | `1262` |

description:

- `H/K` description rate-preserving description:test ΔRate description `-0.8% / -0.1%`,description `76.7% / 60.8%`.
- `I/J` description valid description rate description,description test description,ΔRate description `-5.8% / -5.4%`;description,description failure.
- `D/G` description:description save description,description test rate description,description `D`.

### 9.3 description:raw grid 0.001,rate-preserving `rate_1pp`

| Model | ThrS | ThrF | Valid ΔRate | Valid Save | Valid Acc | Test ΔRate | Test Save | Test Acc | Test PrecS | Test PrecF | FP | FN | N |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `J` | `0.889` | `0.450` | `+0.1%` | `42.4%` | `83.1%` | `-3.1%` | `55.6%` | `89.8%` | `94.8%` | `85.3%` | `21` | `66` | `856` |
| `I` | `0.883` | `0.450` | `+0.7%` | `48.0%` | `83.2%` | `-2.4%` | `59.3%` | `90.9%` | `94.8%` | `86.7%` | `24` | `59` | `908` |
| `H` | `0.851` | `0.450` | `+0.8%` | `62.6%` | `75.1%` | `-0.8%` | `76.5%` | `86.5%` | `89.4%` | `82.2%` | `75` | `86` | `1193` |
| `K` | `0.753` | `0.444` | `+1.0%` | `39.5%` | `74.0%` | `+0.5%` | `51.7%` | `90.3%` | `90.7%` | `89.8%` | `41` | `34` | `775` |
| `G` | `0.650` | `0.411` | `-1.0%` | `75.4%` | `64.6%` | `-3.8%` | `81.7%` | `73.5%` | `80.6%` | `64.1%` | `143` | `199` | `1292` |
| `D` | `0.661` | `0.450` | `+1.0%` | `91.0%` | `71.5%` | `-4.9%` | `93.8%` | `72.2%` | `79.7%` | `62.5%` | `163` | `235` | `1431` |

### 9.4 description:calibrated grid 0.025

description:

```text
runs/model_holdout_answer_calibrated_full/reports/asymmetric_valid_threshold_tuning_fine/report.txt
```

| Model | Policy | ThrS | ThrF | Valid Acc | Valid PrecS | Valid PrecF | Test Acc | Test PrecS | Test PrecF | Test ΔRate | Test Save | FP | FN | N |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `J` | `prec90` | `0.875` | `0.175` | `92.3%` | `94.0%` | `90.7%` | `92.0%` | `95.7%` | `90.0%` | `-1.9%` | `37.6%` | `9` | `37` | `576` |
| `J` | `rate_1pp` | `0.800` | `0.450` | `74.9%` | `78.5%` | `69.8%` | `84.4%` | `92.0%` | `76.5%` | `-5.9%` | `76.3%` | `48` | `134` | `1168` |
| `I` | `prec90` | `0.900` | `0.100` | `92.3%` | `94.0%` | `90.7%` | `93.8%` | `97.1%` | `92.7%` | `-1.2%` | `26.6%` | `3` | `21` | `389` |
| `I` | `rate_1pp` | `0.825` | `0.425` | `79.6%` | `82.7%` | `75.9%` | `86.5%` | `94.7%` | `79.2%` | `-6.0%` | `66.6%` | `25` | `112` | `1015` |
| `H` | `prec90` | `0.900` | `0.225` | `93.0%` | `96.5%` | `91.3%` | `93.1%` | `95.5%` | `91.6%` | `-1.3%` | `34.3%` | `9` | `28` | `533` |
| `H` | `rate_1pp` | `0.800` | `0.450` | `74.9%` | `78.4%` | `70.3%` | `86.6%` | `89.9%` | `82.1%` | `-1.3%` | `74.2%` | `68` | `87` | `1160` |
| `K` | `prec90` | `0.775` | `0.050` | `NA` | `NA` | `NA` | `94.3%` | `94.3%` | `NA` | `+0.1%` | `2.4%` | `2` | `0` | `35` |
| `K` | `rate_1pp` | `0.750` | `0.250` | `80.9%` | `84.0%` | `74.3%` | `91.9%` | `92.0%` | `91.7%` | `+0.5%` | `43.1%` | `31` | `23` | `665` |

description `rate_1pp/rate_2pp` description"description valid rate,description save",description;description test description,description precision/Acc description.

### 9.5 description:Target precision,calibrated grid 0.025

description:valid description `Prec(S) >= target` description `Prec(F) >= target`,description.

| Model | Target | ThrS | ThrF | Valid Acc | Valid PrecS | Valid PrecF | Test Acc | Test PrecS | Test PrecF | Test ΔRate | Test Save | FP | FN | N |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `J` | `0.75` | `0.650` | `0.350` | `75.5%` | `75.3%` | `76.0%` | `84.9%` | `85.7%` | `83.3%` | `+2.9%` | `87.7%` | `123` | `80` | `1340` |
| `J` | `0.80` | `0.850` | `0.300` | `84.6%` | `88.5%` | `80.2%` | `89.4%` | `95.0%` | `85.0%` | `-3.4%` | `53.0%` | `18` | `68` | `809` |
| `J` | `0.85` | `0.850` | `0.225` | `87.9%` | `88.5%` | `87.0%` | `91.4%` | `95.0%` | `88.2%` | `-2.0%` | `48.9%` | `18` | `47` | `756` |
| `J` | `0.90` | `0.875` | `0.175` | `92.3%` | `94.0%` | `90.7%` | `92.0%` | `95.7%` | `90.0%` | `-1.9%` | `37.6%` | `9` | `37` | `576` |
| `I` | `0.75` | `0.650` | `0.450` | `75.9%` | `75.8%` | `76.3%` | `83.3%` | `86.3%` | `78.5%` | `-0.1%` | `91.8%` | `116` | `117` | `1392` |
| `I` | `0.80` | `0.825` | `0.275` | `86.6%` | `88.0%` | `84.9%` | `91.2%` | `94.7%` | `87.3%` | `-2.1%` | `59.3%` | `25` | `55` | `910` |
| `I` | `0.85` | `0.850` | `0.250` | `87.4%` | `89.1%` | `85.8%` | `91.2%` | `95.4%` | `87.6%` | `-2.5%` | `52.5%` | `17` | `53` | `793` |
| `I` | `0.90` | `0.900` | `0.100` | `92.3%` | `94.0%` | `90.7%` | `93.8%` | `97.1%` | `92.7%` | `-1.2%` | `26.6%` | `3` | `21` | `389` |
| `H` | `0.75` | `0.725` | `0.350` | `75.1%` | `75.0%` | `75.5%` | `86.4%` | `86.5%` | `86.2%` | `+3.6%` | `79.4%` | `110` | `57` | `1227` |
| `H` | `0.80` | `0.825` | `0.350` | `80.5%` | `80.2%` | `81.1%` | `88.9%` | `91.0%` | `86.0%` | `-0.4%` | `63.2%` | `52` | `58` | `994` |
| `H` | `0.85` | `0.850` | `0.250` | `86.1%` | `86.3%` | `85.8%` | `92.3%` | `93.5%` | `90.7%` | `-0.1%` | `50.1%` | `30` | `32` | `804` |
| `H` | `0.90` | `0.900` | `0.225` | `93.0%` | `96.5%` | `91.3%` | `93.1%` | `95.5%` | `91.6%` | `-1.3%` | `34.3%` | `9` | `28` | `533` |
| `K` | `0.75` | `0.750` | `0.200` | `85.7%` | `82.6%` | `100.0%` | `92.2%` | `92.0%` | `92.8%` | `+1.4%` | `32.4%` | `31` | `10` | `527` |
| `K` | `0.80` | `0.750` | `0.200` | `85.7%` | `82.6%` | `100.0%` | `92.2%` | `92.0%` | `92.8%` | `+1.4%` | `32.4%` | `31` | `10` | `527` |
| `K` | `0.85` | no valid pair | no valid pair | - | - | - | - | - | - | - | - | - | - | - |
| `K` | `0.90` | no valid pair | no valid pair | - | - | - | - | - | - | - | - | - | - | - |

description D/G/I/J/H/K description `threshold_tuning_summary.md`.

### 9.6 description:description raw grid 0.001 description target-precision description

`two_end_precision_targets_fine_raw_step001` description raw description.description,description:

| Model | Target | ThrS | ThrF | Test Acc | Test PrecS | Test PrecF | Test ΔRate | Test Save | FP | FN | N |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `J` | `0.90` | `0.900` | `0.319` | `91.8%` | `94.7%` | `89.4%` | `-1.6%` | `45.2%` | `17` | `40` | `699` |
| `I` | `0.90` | `0.919` | `0.222` | `93.6%` | `96.4%` | `92.1%` | `-1.2%` | `30.9%` | `6` | `24` | `472` |
| `H` | `0.90` | `0.938` | `0.191` | `92.9%` | `94.8%` | `91.4%` | `-1.0%` | `39.2%` | `14` | `29` | `607` |
| `K` | `0.85` | `0.756` | `0.356` | `92.7%` | `92.0%` | `93.9%` | `+1.1%` | `36.0%` | `29` | `13` | `574` |

description"description"description:

- calibrated description `0.025` description;
- raw description `0.001` description;
- description,description,description calibrated `0.001` target-precision description.

## 10. description

description:

> description model-holdout setting description,description heldout agent model description train/valid;description heldout model_id.description LightGBM description task/answer description prefix description,description agent model description.

description:

> description.

description:

- description instance-holdout.
- description AUC description `0.9071`.
- I/J description step=0 AUC description final AUC,description.

## 11. description

description:

1. description `instance-holdout + model-holdout`:heldout description heldout instances description train/valid.
2. description `NoTaskSignal + NoGoldAnswer + description prefix token`:description,description,repo description,API description.
3. description baseline:description `gold_patch_chars + difficulty + repo + task_prompt_svd`,description.
4. description baseline:description"description"description.
5. description valid-only,description valid/test description.
