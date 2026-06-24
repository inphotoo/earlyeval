# Model-Holdout Answer Experiments - Final Consolidated Report

description:2026-04-28

description `model_holdout_answer_calibrated_full` description,description,gold answer TF-IDF description,step-bucket description,valid-only description.

description:

- description `RECENT_MODEL_HOLDOUT_EXPERIMENTS_AUDIT.md` description.
- description/description.
- description,description instance-holdout description.

## 0. description

description;description:

| description | description | description |
|---|---|---|
| task/gold description | `runs/model_holdout_answer_calibrated_full/reports/task_answer_ablation_posthoc/` | description task prompt,task signal,structured gold answer description |
| no task + no gold description | `runs/model_holdout_answer_calibrated_full/reports/task_answer_ablation_no_task_no_gold/` | description prefix description AUC description |
| description step-bucket AUC | `runs/model_holdout_answer_calibrated_full/reports/task_answer_ablation_step_buckets/` | description prefix description AUC description |
| description action description | `runs/model_holdout_answer_calibrated_full/reports/task_answer_ablation_posthoc/decision_action_analysis/` | description,description action/feedback signal description |
| gold patch description | `runs/model_holdout_answer_calibrated_full/reports/gold_text_tfidf_patch_dims/` | description `gold_patch_tfidf` description 8/16/32 description |
| gold test/fail text description | `runs/model_holdout_answer_calibrated_full/reports/gold_text_tfidf_other_dim16/` | description `gold_test_patch_tfidf` description `gold_fail_to_pass_tfidf` |
| calibrated fine-grid description | `runs/model_holdout_answer_calibrated_full/reports/asymmetric_valid_threshold_tuning_fine_calibrated_step001_rate/` | calibrated description `0.001` description |
| rate-preserving description | `runs/model_holdout_answer_calibrated_full/reports/threshold_tuning_summary/rate_preserving_policy_summary.md` | description"valid rate description + save description"description |

## 1. description

### 1.1 description

- description heldout agent model description train description valid.
- train description heldout model rows = `0`,valid description heldout model rows = `0`,test description non-heldout rows = `0`.
- test description `model_id` description `__MISSING__`,description heldout model description `orig_model_id` description.
- description **model-holdout**,description **instance-holdout**:description SWE-bench instance description agent model description train/valid.

### 1.2 description

- prefix-row description,LightGBM description LR;`I/J/H` description.
- final-step trajectory description,`I/J` ROC-AUC description `0.900` description.
- `NoTaskSignal` description `NoTaskPromptTfidf` description AUC,description task prompt TF-IDF description.
- `NoGoldAnswer` description,description structured gold answer description.
- `NoTask+NoGold` description final AUC `0.8054`,description step=0 AUC description `0.5000`;description prefix description,description.

### 1.3 description

description **valid description rate,description save**:

- `H/K` description:test ΔRate description `-0.8%`,`-0.1%`,description `76.7%`,`60.8%` prefix steps.
- `I/J` description,description test description,rate description `5-6pp`.
- `D/G` description:save description,description test rate description precision description.

### 1.4 description

description:

> description model-holdout setting description,description heldout agent model description,description task/answer description prefix description,description agent model description.

description:

> description SWE-bench description.

description instance-holdout;description ROC-AUC `0.9071`.

## 2. description split description

description:

```text
runs/model_holdout_answer_calibrated_full
```

### 2.1 description prefix table

| description | description |
|---|---:|
| description prefix rows | `386380` |
| description trajectories | `9685` |
| description instances | `490` |
| heldout prefix rows | `83169` |
| heldout trajectories | `1458` |

### 2.2 train/valid/test

| split | description | rows | trajectories | model description |
|---|---|---:|---:|---:|
| train | non-heldout | `259854` | `6417` | `17` |
| valid | non-heldout | `42419` | `1130` | `17` |
| test | heldout only | `83169` | `1458` | `3` |

description heldout agent model:

```text
20251124_mini-v1.17.0_minimax-m2
20251201_mini-v1.17.1_deepseek-v3.2-reasoner
20251210_mini-v1.17.2_kimi-k2-thinking
```

### 2.3 description

| description | rows description | rows |
|---|---|---:|
| `evaluation_report.txt` / `metrics_summary.csv` | description prefix rows | `83169` |
| final-step leaderboard / ablation final metrics | description trajectory description | `1458` |
| threshold valid sweep | valid trajectories | `1130` |
| threshold test application | heldout test trajectories | `1458` |

description `83169` description `1458` description,description.

## 3. description

| Model | description | description |
|---|---|---|
| `D_Dense_Full_LR` | LR | dense + full text SVD |
| `G_TfIdf_Full_LR` | LR | full text TF-IDF/SVD |
| `H_LightGBM_Dense` | LightGBM | dense / structured features |
| `I_LightGBM_Dense_AF` | LightGBM | dense + action/feedback/task text |
| `J_LightGBM_Dense_AF_Thought` | LightGBM | dense + action/feedback/thought/task text |
| `K_LightGBM_Dense_Full` | LightGBM | dense + full raw text blocks |

description:

| Model | description |
|---|---|
| `Abl_NoTaskPromptTfidf_LightGBM` | description `tfidf_task_prompt__svd_*` description `64` description task prompt TF-IDF/SVD description |
| `Abl_NoTaskSignal_LightGBM` | description task prompt TF-IDF/SVD + `task_prompt_chars`,description `65` description task signal description |
| `Abl_NoGoldAnswer_LightGBM` | description structured gold-answer dense features,description `269` description |
| `Abl_NoTaskSignal_NoGoldAnswer_LightGBM` | description task signal + structured gold-answer features,description `334` description |

## 4. description

### 4.1 prefix-row description

description:

```text
runs/model_holdout_answer_calibrated_full/reports/metrics_summary.csv
```

description:`83169` description heldout test prefix rows.

| Model | ROC-AUC | PR-AUC | Brier | N |
| --- | ---: | ---: | ---: | ---: |
| `H_LightGBM_Dense` | `0.8773` | `0.8762` | `0.1422` | `83169` |
| `I_LightGBM_Dense_AF` | `0.8839` | `0.8893` | `0.1406` | `83169` |
| `J_LightGBM_Dense_AF_Thought` | `0.8803` | `0.8837` | `0.1432` | `83169` |
| `K_LightGBM_Dense_Full` | `0.8623` | `0.8508` | `0.1759` | `83169` |
| `D_Dense_Full_LR` | `0.7814` | `0.7927` | `0.2015` | `83169` |
| `G_TfIdf_Full_LR` | `0.7864` | `0.7918` | `0.1933` | `83169` |

description:

- LightGBM description LR.
- `I/J/H` description;`K` description full text description `I/J/H`.
- description raw/full text description,description.

### 4.2 final-step trajectory description

description:

```text
runs/model_holdout_answer_calibrated_full/reports/task_answer_ablation_posthoc/final_step_metrics.csv
runs/model_holdout_answer_calibrated_full/reports/task_answer_ablation_no_task_no_gold/final_step_metrics.csv
```

description:`1458` description heldout test trajectories description.description calibrated probability.

| Model | Acc@0.5 | ROC-AUC | PR-AUC | Brier | Rows |
| --- | ---: | ---: | ---: | ---: | ---: |
| `J_LightGBM_Dense_AF_Thought` | `0.8258` | `0.9007` | `0.9295` | `0.1276` | `1458` |
| `I_LightGBM_Dense_AF` | `0.8251` | `0.9000` | `0.9299` | `0.1282` | `1458` |
| `Abl_NoTaskSignal_LightGBM` | `0.8313` | `0.8960` | `0.9264` | `0.1296` | `1458` |
| `Abl_NoTaskPromptTfidf_LightGBM` | `0.8354` | `0.8946` | `0.9225` | `0.1289` | `1458` |
| `Abl_NoGoldAnswer_LightGBM` | `0.7888` | `0.8271` | `0.8520` | `0.1771` | `1458` |
| `Abl_NoTaskSignal_NoGoldAnswer_LightGBM` | `0.7469` | `0.8054` | `0.8583` | `0.1778` | `1458` |

description:

- description task prompt TF-IDF description task signal,AUC description.
- description structured gold answer,AUC description `0.900` description `0.8271`,description.
- task + gold description,final AUC description `0.8054`,description prefix description.

## 5. step-bucket AUC:description vs description

description:

```text
runs/model_holdout_answer_calibrated_full/reports/task_answer_ablation_step_buckets/step_bucket_metrics_all_task_answer_models.csv
runs/model_holdout_answer_calibrated_full/reports/task_answer_ablation_no_task_no_gold/step_bucket_auc_by_model.csv
```

ROC-AUC description;Platt calibration description,description AUC description raw description.

| bucket | J | I | NoTaskSignal | NoTaskPromptTfidf | NoGoldAnswer | NoTask+NoGold |
|---|---:|---:|---:|---:|---:|---:|
| `step=0` | `0.8846` | `0.8890` | `0.8508` | `0.8533` | `0.8267` | `0.5000` |
| `1-3` | `0.8851` | `0.8908` | `0.8603` | `0.8629` | `0.8304` | `0.6647` |
| `4-6` | `0.8876` | `0.8919` | `0.8700` | `0.8741` | `0.8304` | `0.7464` |
| `7-12` | `0.8896` | `0.8940` | `0.8773` | `0.8794` | `0.8324` | `0.7711` |
| `13-24` | `0.8938` | `0.8978` | `0.8837` | `0.8828` | `0.8359` | `0.7910` |
| `25+` | `0.8656` | `0.8693` | `0.8599` | `0.8593` | `0.8018` | `0.7841` |
| `final` | `0.9007` | `0.9000` | `0.8960` | `0.8946` | `0.8271` | `0.8054` |

description:

- `NoTask+NoGold` description `step=0` description `0.5000`,description.
- description `0.8054`,description action,feedback,thought,description,description.
- `I/J` description `step=0` description,description,task prompt,structured gold answer description.
- `NoGoldAnswer` description `step=0` description `0.8267`,description task prompt / static task signal description.

### 5.1 description action

description:

```text
runs/model_holdout_answer_calibrated_full/reports/task_answer_ablation_posthoc/decision_action_analysis/
```

description:

```text
runs/model_holdout_answer_calibrated_full/reports/task_answer_ablation_posthoc/decision_action_analysis/decision_action_threshold_readable_report.md
runs/model_holdout_answer_calibrated_full/reports/task_answer_ablation_no_task_no_gold/decision_action_analysis/decision_action_threshold_readable_report.md
runs/model_holdout_answer_calibrated_full/reports/test_without_valid_instances_posthoc/decision_action_analysis/combined_decision_action_tradeoff_report.md
```

description"description valid-instance description heldout-test trajectories"description filtered description,description `test_without_valid_instances_posthoc` description;description action description,description,FP/FN,adjusted resolve rate description step description.

description:description trajectory,description predictor,description,description prefix description early-stop description;`p >= threshold` description success,`p <= 1 - threshold` description failure.description `decision_prefix_id` join description prefix table,description `last_action_text`,`last_feedback_text`,action subtype description/traceback/tool-error description.

description:

| Model | Thr | Decision | N | Acc | Avg step | Step0 | Top action subtype | Tests so far |
|---|---:|---|---:|---:|---:|---:|---|---:|
| `I`                 | `0.8` | success | `572` | `0.918` | `0.50` | `0.927` | `none:530; read_search:36` | `0.07` |
| `I`                 | `0.8` | failure | `387` | `0.894` | `2.67` | `0.819` | `none:317; read_search:58` | `0.17` |
| `I`                 | `0.9` | success | `103` | `0.971` | `0.65` | `0.816` | `none:84; read_search:17` | `0.06` |
| `I`                 | `0.9` | failure | `286` | `0.927` | `4.56` | `0.783` | `none:224; read_search:50` | `0.51` |
| `J`                 | `0.8` | success | `598` | `0.920` | `0.72` | `0.906` | `none:542; read_search:43` | `0.08` |
| `J`                 | `0.8` | failure | `386` | `0.883` | `2.80` | `0.785` | `none:303; read_search:73` | `0.13` |
| `J`                 | `0.9` | success | `85`  | `0.976` | `2.75` | `0.388` | `read_search:44; none:33` | `0.07` |
| `J`                 | `0.9` | failure | `285` | `0.937` | `4.06` | `0.670` | `none:191; read_search:81` | `0.26` |
| `NoTaskSignal`      | `0.9` | success | `67`  | `0.955` | `5.79` | `0.209` | `read_search:37; none:14; test:8` | `0.45` |
| `NoTaskSignal`      | `0.9` | failure | `38`  | `1.000` | `32.00`| `0.079` | `read_search:24; run_cli:5; test:4` | `3.45` |
| `NoTaskPromptTfidf` | `0.9` | success | `54`  | `0.963` | `8.91` | `0.000` | `read_search:34; test:10; run_cli:9` | `1.19` |
| `NoTaskPromptTfidf` | `0.9` | failure | `56`  | `1.000` | `19.86`| `0.161` | `read_search:34; none:9; run_cli:6` | `1.50` |
| `NoTask+NoGold`     | `0.7` | success | `966` | `0.747` | `6.45` | `0.000` | `read_search:797; run_cli:82; test:49` | `0.50` |
| `NoTask+NoGold`     | `0.7` | failure | `255` | `0.784` | `14.68`| `0.000` | `read_search:204; run_cli:39; run_python:6` | `0.71` |
| `NoTask+NoGold`     | `0.8` | success | `206` | `0.951` | `13.38`| `0.000` | `read_search:105; test:40; run_python:30` | `1.55` |
| `NoTask+NoGold`     | `0.8` | failure | `180` | `0.861` | `22.37`| `0.000` | `read_search:135; run_cli:32; test:7` | `1.67` |
| `NoTask+NoGold`     | `0.9` | failure | `58`  | `0.948` | `18.55`| `0.000` | `read_search:44; run_cli:8; test:4` | `1.43` |

description:

- `I/J` description `0.8` description `step=0` description,description action description `none`.description early decision description traceback,description/description.
- description `0.9` description,`J` description success description:`step0_share` description `0.906` description `0.388`,top subtype description `none` description `read_search`.description thought description,description success description/description.
- failure description success description,description `I/J` description `none` description `read_search` description;last-step test fail/pass/traceback/tool-error description,description.
- description task signal description task prompt TF-IDF description,description,description action description `read_search`,`run_cli`,`test`.description `NoTaskSignal` description `0.9 failure` description `32` description,`tests_so_far=3.45`,description.
- `NoTask+NoGold` description:`0.7` description step0 description,description agent description action description;`0.8` description `386/1458` description,description `90.9%`,description,description.
- `NoTask+NoGold` description `0.9/0.95` description failure,description success description.description task/gold description,description"description"description,description"description"description.
- description command kind description,description step=0 description `find`,`ls`,`grep_pipeline`,`cat_read`;description `pytest`,test pass/fail,traceback,tool-error description.

## 6. description AUC description

description"description"description,description"description agent model description"description.

description sanity check:

> description SWE instance description heldout description resolved rate,description heldout description.

| description | description |
|---|---:|
| test trajectories | `1458` |
| test instances | `489` |
| prior missing | `0` |
| prior ROC-AUC | `0.9071` |
| prior PR-AUC | `0.9177` |

description AUC description:description agent model description/description.description"description"description"description".

## 7. Gold raw-text TF-IDF description

### 7.1 description

description:

```text
runs/model_holdout_answer_calibrated_full/reports/gold_text_tfidf_patch_dims/final_step_metrics.csv
runs/model_holdout_answer_calibrated_full/reports/gold_text_tfidf_other_dim16/final_step_metrics.csv
```

description:final-step trajectory,calibrated probability.

| Model | Acc@0.5 | ROC-AUC | PR-AUC | Brier |
| --- | ---: | ---: | ---: | ---: |
| `I_LightGBM_Dense_AF` | `0.8251` | `0.9000` | `0.9299` | `0.1282` |
| `O_LightGBM_Dense_AF_GoldPatchTfidf_Dim8` | `0.8210` | `0.8992` | `0.9284` | `0.1298` |
| `O_LightGBM_Dense_AF_GoldPatchTfidf_Dim16` | `0.8278` | `0.8994` | `0.9273` | `0.1297` |
| `O_LightGBM_Dense_AF_GoldPatchTfidf_Dim32` | `0.8278` | `0.8994` | `0.9265` | `0.1281` |
| `P_LightGBM_Dense_AF_GoldTestPatchTfidf_Dim16` | `0.8244` | `0.8996` | `0.9276` | `0.1307` |
| `Q_LightGBM_Dense_AF_GoldFailToPassTfidf_Dim16` | `0.8320` | `0.9008` | `0.9273` | `0.1271` |

### 7.2 description

- `gold_patch_tfidf` description 8 description 32 description,AUC description,description `I`.
- `gold_test_patch_tfidf` description.
- `gold_fail_to_pass_tfidf` description final-step AUC description:`0.9000 -> 0.9008`.
- description,raw gold text TF-IDF description;description structured gold answer description,description patch description,description,hunk description,description,API token description,description,description prefix description/API/test token overlap description.

## 8. description

LightGBM gain description:

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

- `model_id one-hot` description train description,description valid/test description `model_id` description `__MISSING__`,description heldout agent model description.
- task prompt TF-IDF gain description,description;description task prompt description AUC description,description structured gold answer description.
- structured gold answer description:description.

## 9. valid-only description

description:

1. description valid description.
2. description `abs(valid ΔRate)` description.
3. description rate description,description valid Save.
4. test description valid description.

description:

```text
runs/model_holdout_answer_calibrated_full/reports/threshold_tuning_summary/rate_preserving_policy_all_grids.csv
runs/model_holdout_answer_calibrated_full/reports/threshold_tuning_summary/rate_preserving_policy_summary.md
```

description calibrated `0.001` fine-grid,`rate_1pp` policy:valid description `abs(ΔRate) <= 1pp` description save.

| Model | ThrS | ThrF | Valid ΔRate | Valid Save | Valid Acc | Test ΔRate | Test Save | Test Acc | Test PrecS | Test PrecF | FP | FN | N |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `J` | `0.791` | `0.450` | `+0.6%` | `70.5%` | `75.1%` | `-5.4%` | `78.6%` | `84.2%` | `91.3%` | `76.5%` | `55` | `134` | `1200` |
| `I` | `0.801` | `0.450` | `+0.9%` | `61.6%` | `78.0%` | `-5.8%` | `73.0%` | `84.6%` | `92.2%` | `77.1%` | `44` | `129` | `1127` |
| `H` | `0.783` | `0.450` | `+0.8%` | `62.9%` | `75.2%` | `-0.8%` | `76.7%` | `86.5%` | `89.5%` | `82.3%` | `75` | `86` | `1196` |
| `K` | `0.743` | `0.387` | `+0.8%` | `50.2%` | `76.8%` | `-0.1%` | `60.8%` | `89.6%` | `90.9%` | `88.1%` | `46` | `48` | `908` |
| `G` | `0.728` | `0.450` | `+0.7%` | `69.7%` | `65.6%` | `-2.9%` | `77.3%` | `74.3%` | `81.2%` | `64.4%` | `137` | `179` | `1231` |
| `D` | `0.802` | `0.450` | `+1.0%` | `75.0%` | `75.2%` | `-8.2%` | `79.4%` | `73.6%` | `83.2%` | `63.9%` | `107` | `226` | `1262` |

description:

- description rate:description `H` description `K`.
- description save description heldout test rate description:`I/J` description.
- `D/G` description,description.

## 10. description

Platt/sigmoid calibration description valid description,description:

- ROC-AUC / PR-AUC description,description.
- Brier / log-loss description,description.
- description calibrated description raw description:description `rate_1pp` description,raw `I/J` description,precision description save description;calibrated `I/J` save description test rate description.

description:

- description,description AUC description.
- description,description raw description calibrated description,description valid description.

## 11. description

### 11.1 description

description:

- prefix-row:`I/J/H` description AUC description Brier.
- final-step:`I/J` description AUC `0.900` description.
- description:`NoGoldAnswer` description `NoTask+NoGold` description,description structured answer description.

### 11.2 description

description:

- calibrated `0.001` fine-grid,`rate_1pp` policy.
- description `H/K` description,`I/J` description.

### 11.3 description

description"description",description:

- instance-holdout description repo/instance group-holdout;
- description fit TF-IDF/SVD description dense encoders;
- description instance description step=0,prefix bucket,final-step description;
- description no-task/no-gold description,description.

## 12. description

```text
runs/model_holdout_answer_calibrated_full/reports/metrics_summary.csv
runs/model_holdout_answer_calibrated_full/reports/model_holdout_split_summary.csv
runs/model_holdout_answer_calibrated_full/reports/model_ranking_report_like_ref_calibrated/final_step_prefix_model_leaderboard.csv
runs/model_holdout_answer_calibrated_full/reports/task_answer_ablation_posthoc/final_step_metrics.csv
runs/model_holdout_answer_calibrated_full/reports/task_answer_ablation_posthoc/variant_manifest.csv
runs/model_holdout_answer_calibrated_full/reports/task_answer_ablation_posthoc/decision_action_analysis/decision_action_report.md
runs/model_holdout_answer_calibrated_full/reports/task_answer_ablation_posthoc/decision_action_analysis/decision_action_threshold_readable_report.md
runs/model_holdout_answer_calibrated_full/reports/task_answer_ablation_posthoc/decision_action_analysis/decision_action_cases.csv
runs/model_holdout_answer_calibrated_full/reports/task_answer_ablation_no_task_no_gold/decision_action_analysis/decision_action_threshold_readable_report.md
runs/model_holdout_answer_calibrated_full/reports/task_answer_ablation_no_task_no_gold/decision_action_analysis/decision_action_cases.csv
runs/model_holdout_answer_calibrated_full/reports/test_without_valid_instances_posthoc/decision_action_analysis/combined_decision_action_tradeoff_report.md
runs/model_holdout_answer_calibrated_full/reports/test_without_valid_instances_posthoc/decision_action_analysis/combined_decision_action_tradeoff.csv
runs/model_holdout_answer_calibrated_full/reports/task_answer_ablation_no_task_no_gold/final_step_metrics.csv
runs/model_holdout_answer_calibrated_full/reports/task_answer_ablation_no_task_no_gold/variant_manifest.csv
runs/model_holdout_answer_calibrated_full/reports/task_answer_ablation_step_buckets/step_bucket_metrics_all_task_answer_models.csv
runs/model_holdout_answer_calibrated_full/reports/task_answer_ablation_no_task_no_gold/step_bucket_auc_by_model.csv
runs/model_holdout_answer_calibrated_full/reports/gold_text_tfidf_patch_dims/final_step_metrics.csv
runs/model_holdout_answer_calibrated_full/reports/gold_text_tfidf_other_dim16/final_step_metrics.csv
runs/model_holdout_answer_calibrated_full/reports/threshold_tuning_summary/rate_preserving_policy_summary.md
runs/model_holdout_answer_calibrated_full/reports/threshold_tuning_summary/rate_preserving_policy_all_grids.csv
RECENT_MODEL_HOLDOUT_EXPERIMENTS_AUDIT.md
MODEL_HOLDOUT_ANSWER_FEATURES_FLOW.md
```
