# Model-Holdout + Gold Answer Features:description

description:

- final3 description:`final3/vendor/prefix_predict_model_holdout_answer/`
- description:`run_all.py`
- gold answer description:`answer_features.py`
- description:`feature_engineer.py`
- model holdout description:`model_holdout_split.py`
- validation-only description:`probability_calibration.py`
- trajectory description:`model_ranking_report_posthoc.py`
- gold text TF-IDF description:`gold_text_tfidf_ablation_posthoc.py`

## 1. description

description:description SWE-bench description `t` description,description `resolved=True`.

description prefix row:

- description 0 description:description.
- description 1 description:description 1 description action/feedback.
- description 2 description:description 2 description.
- description:description.

description prefix row description:description.

description:

- `prefix-level`:description prefix row description,description full run description `83169` description.
- `trajectory-level final-step`:description prefix row description,description full run description `1458` description.

description"description",description:

`runs/model_holdout_answer_calibrated_full/reports/model_ranking_report_like_ref_calibrated/final_step_prefix_model_leaderboard.csv`

description `evaluation_report.txt` description prefix-level description AUC.

## 2. description:model holdout description

description `--split-by model_holdout`.

description:

1. description verified description `500` description SWE-bench instance.
2. description parquet description `instance_id` description.
3. description 3 description holdout description.
4. description.
5. valid description instance description.
6. test description 3 description holdout description.

description full run description:

- `83169` prefix rows
- `1458` trajectories
- `489` instances
- `3` heldout agent models

description heldout description:

- `20251124_mini-v1.17.0_minimax-m2`
- `20251201_mini-v1.17.1_deepseek-v3.2-reasoner`
- `20251210_mini-v1.17.2_kimi-k2-thinking`

### description

description valid description,description.

test description.

description:description holdout description,description.

## 3. model_id description

description:

- `FeatureEngineer(include_model_id=True)`
- description `model_id` description one-hot.
- description,description"description"description.

valid description test description:

- `model_id` description `__MISSING__`
- `model` description `__MISSING__`
- description `orig_model_id`,description,description.

description heldout test description"description model_id"description.

description:

```text
model_id feature values: ['__MISSING__']
model_id_input_mode: ['test_missing']
```

description"description model_id description",description:

- description:description model_id.
- valid/test:description `__MISSING__`.

description"description,description"description.

## 4. description prefix table

description:

1. description parquet description.
2. `step_builder.py` description messages description step table.
3. `prefix_builder.py` description prefix table.
4. `answer_features.py` description `instance_id` join description verified answer description.
5. `feature_engineer.py` description prefix table description.
6. `trainer.py` description LR / LightGBM.
7. `evaluator.py` description prefix-level description.
8. `model_ranking_report_posthoc.py` description trajectory-level description/description.

prefix table description:

"description,description,description."

## 5. Gold answer description

gold answer description:

`swebench_verified/test.jsonl`

description instance description:

- `repo`
- `difficulty`
- `version`
- `problem_statement`
- `hints_text`
- `patch`
- `test_patch`
- `FAIL_TO_PASS`
- `PASS_TO_PASS`

description gold answer description.description"description",description.

## 6. description

### `patch`

description diff.

description:description.

### `test_patch`

description diff.

description:description.

description SWE-bench description,description.

### `FAIL_TO_PASS`

description.

description buggy description,description patch description.

description:description"description bug description".

description,description.

### `PASS_TO_PASS`

description.

description,description.

description:description"description".

description:description bug description.

## 7. Gold answer description

### 7.1 description

| description | description | description | description |
|---|---:|---|---|
| `gold_has_answer` | description | description instance description join description verified answer.description 1. | description. |
| `gold_repo` | description | description. | one-hot description,description Django / sklearn / matplotlib. |
| `gold_difficulty` | description | description,description `<15_min_fix`. | one-hot description. |
| `gold_version` | description | SWE-bench description. | one-hot description/description. |
| `gold_problem_statement_chars` | description | issue description. | description,description. |
| `gold_hints_chars` | description | hints description. | hints description,description. |
| `gold_has_hints` | description | description hints. | description. |

description:

- `gold_difficulty__<15_min_fix` description `gold_difficulty` one-hot description.
- description `<15_min_fix`,description 1,description 0.
- LightGBM description,description:description.

### 7.2 gold code patch description

description `patch`.

| description | description | description | description |
|---|---:|---|---|
| `gold_patch_chars` | description | description patch description. | description,description. |
| `gold_patch_hunks` | description | patch description `@@ ... @@` description. | description,description. |
| `gold_patch_files_count` | description | description patch description. | description. |
| `gold_patch_added_lines` | description | patch description,description `+++` description. | description. |
| `gold_patch_deleted_lines` | description | patch description,description `---` description. | description. |
| `gold_patch_max_dir_depth` | description | patch description. | description,description. |
| `gold_primary_patch_ext` | description | description,description `.py`. | one-hot description. |
| `gold_primary_patch_dir` | description | description. | one-hot description. |

description:

- `gold_patch_max_dir_depth=4` description `django/db/models/query.py`.
- description"description",description.
- description,description.

### 7.3 gold test patch description

description `test_patch`.

| description | description | description | description |
|---|---:|---|---|
| `gold_has_test_patch` | description | description. | description,description/description. |
| `gold_test_patch_chars` | description | test patch description. | description,description. |
| `gold_test_patch_hunks` | description | test patch description `@@ ... @@` description. | description. |
| `gold_test_files_count` | description | test patch description. | description. |
| `gold_test_added_lines` | description | test patch description. | description. |
| `gold_test_deleted_lines` | description | test patch description. | description. |

description:

- `gold_test_patch_chars` description,description.
- description:"description/description".
- description.

### 7.4 description

| description | description | description | description |
|---|---:|---|---|
| `gold_fail_to_pass_count` | description | `FAIL_TO_PASS` description. | description,description. |
| `gold_pass_to_pass_count` | description | `PASS_TO_PASS` description. | description,description. |
| `gold_fail_to_pass_text` | description | description `FAIL_TO_PASS` description. | description TF-IDF,description token. |

`FAIL_TO_PASS` description"description".

description.description:

```text
tests.test_xxx::test_bug_case
```

description action/feedback/thought description,description TF-IDF description.

### 7.5 patch token / description

| description | description | description | description |
|---|---:|---|---|
| `gold_patch_api_token_count` | description | description patch description `def` / `class` description,description token description. | description/description/description,description. |
| `gold_patch_import_token_count` | description | description patch description import description token description. | import description,description. |
| `gold_patch_exception_keyword_count` | description | patch/test/problem description error,exception,traceback,assert description. | description/description. |
| `gold_patch_test_keyword_count` | description | test,pytest,unittest,assert description. | description. |
| `gold_patch_config_keyword_count` | description | config,setting,option,env,version description. | description/description. |

description"description",description"description".

## 8. Gold answer description prefix description

description,description"description gold answer description"description.

description 6 description:

- `prefix_action_text`:description action.
- `prefix_feedback_text`:description/description.
- `prefix_thought_text`:description thought.
- `last_action_text`:description action.
- `last_feedback_text`:description.
- `last_thought_text`:description thought.

description gold answer token description 3 description:

- file tokens:description patch/test_patch description basename.
- api tokens:description patch description,description,import description,description token.
- test tokens:`FAIL_TO_PASS`,`PASS_TO_PASS`,test patch description token.

description:

```text
gold_{prefix/last}_{action/feedback/thought}_{file/api/test}_{hits/jaccard/hit_any}
```

description:

| description | description |
|---|---|
| `_hits` | description gold token description. |
| `_jaccard` | description / description,description 0 description 1. |
| `_hit_any` | description token. |

description:

| description | description |
|---|---|
| `gold_prefix_action_file_hits` | description,action description. |
| `gold_last_feedback_test_hit_any` | description token. |
| `gold_prefix_thought_api_jaccard` | thought description API token description. |

description:

- description/API/description,description.
- description gold patch description,description.

## 9. Gold raw text TF-IDF description

description gold description,description gold description TF-IDF:

| TF-IDF block | description | description |
|---|---|---|
| `tfidf_gold_patch` | `gold_patch_text` | description patch description. |
| `tfidf_gold_test_patch` | `gold_test_patch_text` | description patch description. |
| `tfidf_gold_fail_to_pass` | `gold_fail_to_pass_text` | description. |
| `tfidf_gold_answer_summary` | `gold_answer_summary_text` | problem/hints/patch files/test files/fail tests/API tokens description. |

TF-IDF description:

1. description.
2. description gold description.
3. description SVD description.
4. description.

description"description".

description:

- description patch description.
- description.
- description.

description full gold raw text description:description,description,LightGBM description/description,description.

## 10. Dense,AF,Thought,Full description

### Dense

Dense description,description:

- description:description,description,description.
- description:read/edit/test/bash description.
- description:description traceback,description,description.
- description/description:description,description edit,description submit.
- thought/content description.
- gold answer description.
- gold answer description prefix description.
- description `include_model_id=True`,description `model_id` one-hot.

description Dense description gold answer description.

### AF

AF description Action + Feedback description TF-IDF:

- `tfidf_task_prompt`
- `tfidf_prefix_action`
- `tfidf_prefix_feedback`
- `tfidf_last_action`
- `tfidf_last_feedback`

### Thought

Thought description:

- `tfidf_prefix_thought`
- `tfidf_last_thought`

### Full

Full description TF-IDF:

- AF
- Thought
- assistant content
- gold raw text TF-IDF

description,`Full` description gold raw text TF-IDF.

## 11. H / I / J / K / D / G description

| description | description | description | description gold description | description gold raw text TF-IDF |
|---|---|---|---:|---:|
| `H_LightGBM_Dense` | LightGBM | Dense | description | description |
| `I_LightGBM_Dense_AF` | LightGBM | Dense + AF | description | description |
| `J_LightGBM_Dense_AF_Thought` | LightGBM | Dense + AF + Thought | description | description |
| `K_LightGBM_Dense_Full` | LightGBM | Dense + Full | description | description |
| `D_Dense_Full_LR` | Logistic Regression | Dense + Full | description | description |
| `G_TfIdf_Full_LR` | Logistic Regression | Full TF-IDF only | description | description |

description:

- `I` description"description".
- `I` description gold answer description.
- `I` description `gold_patch_text/test_patch_text/fail_to_pass_text` description raw text TF-IDF.
- `K` description `I` description raw gold text TF-IDF.

description `I > K` description"description".

description:

> description;description TF-IDF description holdout description.

## 12. description

### description

description:

- `gold_patch_chars`
- `gold_test_patch_chars`
- `gold_patch_hunks`
- `gold_fail_to_pass_count`

description:

1. description.
2. description `-1` description `0`.
3. description `StandardScaler` description.
4. description Dense description.

LightGBM description:

- description.
- description:`gold_patch_chars > description` description.

LR description:

- description.
- description `gold_patch_chars` description,description.

### description

description:

- `gold_has_test_patch`
- `gold_prefix_action_file_hit_any`

description:

- `False -> 0`
- `True -> 1`

description:

- LightGBM description"description patch"description.
- LR description/description.

### description

description:

- `gold_repo`
- `gold_difficulty`
- `gold_version`
- `gold_primary_patch_ext`
- `gold_primary_patch_dir`
- `model_id`

description:

1. description `LabelEncoder`.
2. description one-hot.
3. description `__MISSING__`.

description:

- `gold_difficulty__<15_min_fix`
- `gold_repo__django/django`
- `model_id____MISSING__`

### description

description:

- `prefix_action_text`
- `prefix_feedback_text`
- `gold_patch_text`
- `gold_fail_to_pass_text`

description:

1. `TfidfVectorizer` description.
2. description block description SVD description.
3. description.

description:

- `tfidf_gold_patch__svd_0`
- `tfidf_prefix_action__svd_12`

description,description.

## 13. description AUC description

### `evaluation_report.txt` description AUC

description prefix-level AUC.

description:

`runs/model_holdout_answer_calibrated_full/reports/evaluation_report.txt`

description full run description `Overall Metrics Summary` description `83169` description prefix row description.

description:

> description.description.

description"description",description.

### `model_ranking_report_like_ref_calibrated` description leaderboard AUC

description:

`runs/model_holdout_answer_calibrated_full/reports/model_ranking_report_like_ref_calibrated/final_step_prefix_model_leaderboard.csv`

description AUC description trajectory-level final-step AUC.

description:

1. description `traj_id` description `prefix_step_idx` description.
2. description.
3. description `1458` description ROC-AUC / PR-AUC / Brier / LogLoss.

description"description AUC".

description:

> description AUC.

### description

ranking report description threshold description AUC.

description:

- description `p >= threshold`,description success.
- description `p <= 1 - threshold`,description failure.
- description,description.

description:

- description
- description
- FN
- FP
- description
- description resolve rate
- description

description"description AUC".

## 14. description prefix-level description trajectory-level description

description calibrated full run description,description trajectory-level final-step description:

| description | Acc@0.5 | ROC-AUC | Brier | description |
|---|---:|---:|---:|---|
| `J_LightGBM_Dense_AF_Thought` | 0.8258 | 0.9007 | 0.1276 | final-step AUC description. |
| `I_LightGBM_Dense_AF` | 0.8251 | 0.9000 | 0.1282 | description gold + AF,description. |
| `H_LightGBM_Dense` | 0.8285 | 0.8971 | 0.1251 | Dense-only description,description. |
| `K_LightGBM_Dense_Full` | 0.8285 | 0.8791 | 0.1434 | description gold raw text description AUC description. |
| `G_TfIdf_Full_LR` | 0.7401 | 0.8125 | 0.1768 | description Full TF-IDF,description. |
| `D_Dense_Full_LR` | 0.7298 | 0.8013 | 0.1822 | LR + Full,description LightGBM. |

description:

- description"description",description `final_step_prefix_model_leaderboard.csv`.
- description `I/J/H`,description `K`.
- `K` description raw gold text TF-IDF description.

## 15. Gold text TF-IDF description

description FeatureEngineer,description step/prefix/gold join.

description:

- `runs/model_holdout_answer_calibrated_full/reports/gold_text_tfidf_patch_dims/`
- `runs/model_holdout_answer_calibrated_full/reports/gold_text_tfidf_other_dim16/`

trajectory-level final-step calibrated description:

| description | Acc@0.5 | ROC-AUC | Brier | description |
|---|---:|---:|---:|---|
| `I_LightGBM_Dense_AF` | 0.8251 | 0.9000 | 0.1282 | description:description gold + AF. |
| `GoldPatchTfidf Dim8` | 0.8210 | 0.8992 | 0.1298 | patch raw text description. |
| `GoldPatchTfidf Dim16` | 0.8278 | 0.8994 | 0.1297 | accuracy description,AUC description. |
| `GoldPatchTfidf Dim32` | 0.8278 | 0.8994 | 0.1281 | Brier description I,description AUC description. |
| `GoldTestPatchTfidf Dim16` | 0.8244 | 0.8996 | 0.1307 | test_patch raw text description. |
| `GoldFailToPassTfidf Dim16` | 0.8320 | 0.9008 | 0.1271 | description,description. |
| `K_LightGBM_Dense_Full` | 0.8285 | 0.8791 | 0.1434 | description raw text description. |

description:

1. `FAIL_TO_PASS` description,description.
2. `gold_patch` raw text description.
3. `gold_test_patch` raw text description.
4. description raw gold text description.

description `K` description.

description:

- description:`I_LightGBM_Dense_AF`
- description:`Dense + AF + gold_fail_to_pass_tfidf_dim16`
- description:`K_LightGBM_Dense_Full` description"description raw gold text description"description.

## 16. description row description

description:

- `83169` prefix rows
- `1458` trajectories
- description `57` description prefix row

description row description,description instance description trajectory description,description:

- description.
- description prefix.
- description parquet.
- description trajectory / instance / model_holdout description.

description,description rows.

description:

- prefix rows
- trajectories
- instances
- avg prefixes per trajectory
- heldout models

description `1458 trajectories / 489 instances / 3 heldout models` description.

## 17. description

description:

1. `model_ranking_report_like_ref_calibrated/final_step_prefix_model_leaderboard.csv`
   - description.
   - description.

2. `model_ranking_report_like_ref_calibrated/report.txt`
   - description,description,description,description,description.

3. `model_ranking_report_like_ref_calibrated/final_step_probability_bins.csv`
   - description,description 0.8-0.9 description.

4. `evaluation_report.txt`
   - description prefix-level description,step bucket,description,top features.

5. `gold_text_tfidf_* / summary.txt`
   - description gold raw text TF-IDF description.

## 18. description

description:

> We evaluate two complementary granularities. Prefix-level metrics measure whether the predictor is informative at every intermediate step, while final-step trajectory-level metrics count each trajectory once and better reflect final outcome prediction. In the model-holdout setting, all heldout test model identities are mapped to `__MISSING__`, so the predictor cannot use the true identity of the three test models. Under the trajectory-level final-step metric, structured gold-answer features combined with process action/feedback features perform best. Adding all raw gold-answer TF-IDF blocks hurts generalization, while a small `FAIL_TO_PASS` TF-IDF block gives a minor positive signal.

description:

> description prefix description trajectory description.prefix description;trajectory description final-step description,description.model-holdout description,description model_id description `__MISSING__`,description.description,description gold answer description action/feedback description;description gold raw text TF-IDF description;description `FAIL_TO_PASS` description TF-IDF description.
