# Recent Model-Holdout Experiments Audit

生成时间：2026-04-28

本文档整理 `model_holdout_answer_calibrated_full` 最近一轮实验的核心流程、数据审计、主要结果和解释边界。

综合新版报告见：

```text
MODEL_HOLDOUT_ANSWER_FINAL_CONSOLIDATED_REPORT.md
```

该报告把主实验、补充消融训练、gold answer TF-IDF、step-bucket AUC、calibrated/raw fine-grid 阈值调优都合并到一份文档里。

## 1. 最重要结论

### 1.1 三个 heldout agent model 的轨迹没有进入训练或 valid

严格结论：

- 这三个 heldout 模型的轨迹行只在 test 中：
  - `20251124_mini-v1.17.0_minimax-m2`
  - `20251201_mini-v1.17.1_deepseek-v3.2-reasoner`
  - `20251210_mini-v1.17.2_kimi-k2-thinking`
- train 中 heldout model 行数：`0`
- valid 中 heldout model 行数：`0`
- test 中非 heldout model 行数：`0`
- test 预测表里 `model_id` 输入统一是 `__MISSING__`，真实模型名只保存在 `orig_model_id` 里用于统计，不作为模型输入。

所以，如果问题是：

> 这三个模型自己的轨迹有没有被训练或 valid 用到？

答案是：**没有。**

### 1.2 但这不是 instance-holdout

也必须非常清楚：

- 这是 **model-holdout**，不是 **instance-holdout**。
- 同一道 SWE-bench 题目会通过其他 agent model 出现在 train/valid。
- heldout test 是“新 agent model 跑已知题目池”，不是“完全新题目”。

这解释了为什么 I/J 在 step=0 就很强：模型能利用题目难度、官方答案结构、task prompt 等静态信号。

## 2. 数据审计结果

主运行目录：

```text
runs/model_holdout_answer_calibrated_full
```

### 2.1 prefix table 全量数据

读取：

```text
runs/model_holdout_answer_calibrated_full/data/prefix_table_filtered.parquet
```

审计结果：

| 项目 | 数值 |
|---|---:|
| 全部 prefix rows | `386380` |
| 全部 trajectories | `9685` |
| 全部 instances | `490` |
| heldout prefix rows | `83169` |
| heldout trajectories | `1458` |

三个 heldout 模型在全表中的数量：

| heldout model | rows | trajs | instances | prefix-row pos rate |
|---|---:|---:|---:|---:|
| `20251124_mini-v1.17.0_minimax-m2` | `36938` | `488` | `488` | `0.5089` |
| `20251201_mini-v1.17.1_deepseek-v3.2-reasoner` | `22740` | `482` | `482` | `0.5649` |
| `20251210_mini-v1.17.2_kimi-k2-thinking` | `23491` | `488` | `488` | `0.5378` |

### 2.2 split 后的 train/valid/test

来自：

```text
runs/model_holdout_answer_calibrated_full/reports/model_holdout_split_summary.csv
runs/model_holdout_answer_calibrated_full/reports/task_answer_ablation_no_task_no_gold/split_reconstruction_summary.json
```

| split | 是否 heldout model | rows | trajs | model 数 |
|---|---|---:|---:|---:|
| train | 否 | `259854` | `6417` | `17` |
| valid | 否 | `42419` | `1130` | `17` |
| test | 是 | `83169` | `1458` | `3` |

额外核验：

| 检查项 | 结果 |
|---|---:|
| train 中 heldout rows | `0` |
| valid 中 heldout rows | `0` |
| test 中 non-heldout rows | `0` |
| `train_models ∩ holdout_models` | 空 |
| `valid_models ∩ holdout_models` | 空 |
| `test_models == holdout_models` | 是 |

### 2.3 预测表检查

检查过以下预测表：

```text
runs/model_holdout_answer_calibrated_full/reports/test_predictions_all_models.parquet
runs/model_holdout_answer_calibrated_full/reports/task_answer_ablation_posthoc/test_predictions_task_answer_ablation.parquet
runs/model_holdout_answer_calibrated_full/reports/task_answer_ablation_no_task_no_gold/test_predictions_task_answer_ablation.parquet
```

共同结果：

| 项目 | 结果 |
|---|---|
| rows | `83169` |
| trajectories | `1458` |
| instances | `489` |
| `split` | 只有 `test` |
| `model_id` 输入 | 只有 `__MISSING__` |
| `model_id_input_mode` | 只有 `test_missing` |
| `orig_model_id` | 只有三个 heldout model |
| non-heldout `orig_model_id` rows | `0` |

## 3. 代码链路如何保证没有 heldout model 进入训练

关键代码：

```text
model_holdout_split.py
run_all.py
gold_text_tfidf_ablation_posthoc.py
task_and_answer_ablation_posthoc.py
feature_engineer.py
```

### 3.1 split 逻辑

`model_holdout_split.py` 中核心逻辑：

```python
trainval = work[~work["model_id"].isin(heldout)]
test = work[work["model_id"].isin(heldout)]
```

也就是说：

- heldout 模型的轨迹直接进 test。
- 非 heldout 模型再拆 train/valid。
- 代码还检查 `train_models & test_models`，如果有重叠会报错。

### 3.2 训练特征只在 train 上 fit

`run_all.py` 中：

```python
fe_with_model.fit(df_train)
fe_no_model.fit(df_train)
```

所以：

- dense 编码器只从 train 学。
- TF-IDF/SVD 也只从 train 文本 fit。
- valid/test 只是 transform，不参与 fit。

### 3.3 valid/test 的模型身份被隐藏

`run_all.py` 和 posthoc 脚本都会做：

```python
df_train["model_id_input_mode"] = "train_seen"
df_valid["model_id"] = "__MISSING__"
df_test["model_id"] = "__MISSING__"
```

含义：

- train 可以看到训练模型身份。
- valid/test 都看不到真实模型身份。
- heldout test 的真实模型名只保留在 `orig_model_id`，用于后续分模型统计。

### 3.4 校准和阈值选择没有用 heldout test label

概率校准：

- 使用 valid 上的 raw probability 和 label。
- 方法是 validation-only sigmoid/Platt calibration。

不对称阈值选择：

- 在 valid 上 sweep 阈值。
- test 只是应用 valid 选出的阈值。

注意边界：

- valid 没有 heldout model 轨迹。
- 但 valid 的 SWE instances 会和 heldout test 题目池重叠，因为这是 model-holdout，不是 instance-holdout。

## 4. 主实验结果

### 4.1 prefix-row 总体 AUC

来自：

```text
runs/model_holdout_answer_calibrated_full/reports/evaluation_report.txt
```

这里的 `N=83169`，表示按所有 prefix rows 计算。

| Model | ROC-AUC | PR-AUC | Brier |
|---|---:|---:|---:|
| `H_LightGBM_Dense` | `0.8773` | `0.8762` | `0.1422` |
| `I_LightGBM_Dense_AF` | `0.8839` | `0.8893` | `0.1406` |
| `J_LightGBM_Dense_AF_Thought` | `0.8803` | `0.8837` | `0.1432` |
| `K_LightGBM_Dense_Full` | `0.8623` | `0.8508` | `0.1759` |
| `D_Dense_Full_LR` | `0.7814` | `0.7927` | `0.2015` |
| `G_TfIdf_Full_LR` | `0.7864` | `0.7918` | `0.1933` |

解释：

- LightGBM 在这个 setting 下明显强于 LR。
- `I/J` 最稳，`K` 加了 full raw text 后反而变差，说明 raw text TF-IDF 不是稳定增益。

### 4.2 final-step trajectory AUC

来自：

```text
runs/model_holdout_answer_calibrated_full/reports/task_answer_ablation_posthoc/summary.txt
runs/model_holdout_answer_calibrated_full/reports/task_answer_ablation_no_task_no_gold/summary.txt
```

这里按每条 trajectory 的最后一步计算，`rows=1458`。

| Model                                    | Acc@0.5 | ROC-AUC | PR-AUC | Brier |
|---|---:|---:|---:|---:|
| `J_LightGBM_Dense_AF_Thought`            | `0.8258` | `0.9007` | `0.9295` | `0.1276` |
| `I_LightGBM_Dense_AF`                    | `0.8251` | `0.9000` | `0.9299` | `0.1282` |
| `Abl_NoTaskSignal_LightGBM`              | `0.8313` | `0.8960` | `0.9264` | `0.1296` |
| `Abl_NoTaskPromptTfidf_LightGBM`         | `0.8354` | `0.8946` | `0.9225` | `0.1289` |
| `Abl_NoGoldAnswer_LightGBM`              | `0.7888` | `0.8271` | `0.8520` | `0.1771` |
| `Abl_NoTaskSignal_NoGoldAnswer_LightGBM` | `0.7469` | `0.8054` | `0.8583` | `0.1778` |

关键解释：

- 去掉 task prompt TF-IDF 后，AUC 只从 `0.9000/0.9007` 降到约 `0.8946`。
- 去掉全部 task signal 后，仍有 `0.8960`。
- 去掉 gold answer 结构化特征后，明显降到 `0.8271`。
- task + gold 都去掉后，仍有 `0.8054`，因为还保留了 prefix 过程文本和过程特征。

## 5. step-bucket AUC：为什么看起来“模型无关也很强”

来自：

```text
runs/model_holdout_answer_calibrated_full/reports/task_answer_ablation_no_task_no_gold/step_bucket_auc_report.txt
runs/model_holdout_answer_calibrated_full/reports/task_answer_ablation_step_buckets/step_bucket_report_all_task_answer_models.txt
runs/model_holdout_answer_calibrated_full/reports/task_answer_ablation_step_buckets/step_bucket_metrics_all_task_answer_models.csv
```

下面使用 calibrated probabilities。ROC-AUC 是排序指标，单调校准基本不改变排序。

| bucket | `J` | `I` | `NoTaskSignal` | `NoTaskPromptTfidf` | `NoGoldAnswer` | `NoTask+NoGold` |
|---|---:|---:|---:|---:|---:|---:|
| `step=0`| `0.8846` | `0.8890` | `0.8508` | `0.8533` | `0.8267` | `0.5000` |
| `1-3`   | `0.8851` | `0.8908` | `0.8603` | `0.8629` | `0.8304` | `0.6647` |
| `4-6`   | `0.8876` | `0.8919` | `0.8700` | `0.8741` | `0.8304` | `0.7464` |
| `7-12`  | `0.8896` | `0.8940` | `0.8773` | `0.8794` | `0.8324` | `0.7711` |
| `13-24` | `0.8938` | `0.8978` | `0.8837` | `0.8828` | `0.8359` | `0.7910` |
| `25+`   | `0.8656` | `0.8693` | `0.8599` | `0.8593` | `0.8018` | `0.7841` |
| `final` | `0.9007` | `0.9000` | `0.8960` | `0.8946` | `0.8271` | `0.8054` |

解释：

- `NoTask+NoGold` 在 `step=0` 完全没能力，AUC 是 `0.5000`。
- 后面 AUC 上升，是因为 prefix 中逐渐出现 action、feedback、thought、报错、测试输出、文件名、API token 等过程信息。
- `I/J` 在 `step=0` 就很高，是因为它们还保留 task prompt 和 gold answer 结构化特征，本质上已经能做“题目难度预测”。
- `NoTaskSignal` 和 `NoTaskPromptTfidf` 仍然在 `step=0` 很高，说明单独去掉 task prompt 并不够；gold answer 结构化特征仍然可以强力判断题目难度。
- `NoGoldAnswer` 在 `step=0` 仍有 `0.8267`，说明 task prompt / repo / difficulty-like static signal 也很强。

## 6. 同题其他模型成功率先验

为了判断高 AUC 是不是离谱，做了一个 sanity check：

> 只用同一道 SWE instance 上其他非 heldout 模型的 resolved rate，去预测 heldout 三个模型最终是否成功。

结果：

| 指标 | 数值 |
|---|---:|
| test trajectories | `1458` |
| test instances | `489` |
| prior missing | `0` |
| prior ROC-AUC | `0.9071` |
| prior PR-AUC | `0.9177` |

解释：

- 同一道题对不同模型的成功/失败高度相关。
- 容易题大多数模型都容易过，难题大多数模型都容易挂。
- 因此 I/J 的高 step=0 AUC 并不神秘：它们很大程度是在学习题目难度。

这也说明：

- 当前实验能证明：**对已知题目池，新 agent model 没见过，也可以迁移预测。**
- 当前实验不能证明：**对完全新 SWE instance 也同样强。**

## 7. Gold raw-text TF-IDF 消融结论

### 7.1 `gold_patch_tfidf` 维度变化

来自：

```text
runs/model_holdout_answer_calibrated_full/reports/gold_text_tfidf_patch_dims/summary.txt
```

final-step calibrated：

| Model | Acc@0.5 | ROC-AUC | Brier |
|---|---:|---:|---:|
| `I_LightGBM_Dense_AF`  | `0.8251` | `0.9000` | `0.1282` |
| `GoldPatchTfidf_Dim8`  | `0.8210` | `0.8992` | `0.1298` |
| `GoldPatchTfidf_Dim16` | `0.8278` | `0.8994` | `0.1297` |
| `GoldPatchTfidf_Dim32` | `0.8278` | `0.8994` | `0.1281` |

结论：

- patch raw text TF-IDF 没有带来稳定提升。
- 维度从 8 到 32，AUC 基本不变，而且都略低于 `I`。

### 7.2 `gold_test_patch_tfidf` 和 `gold_fail_to_pass_tfidf`

来自：

```text
runs/model_holdout_answer_calibrated_full/reports/gold_text_tfidf_other_dim16/summary.txt
```

final-step calibrated：

| Model | Acc@0.5 | ROC-AUC | Brier |
|---|---:|---:|---:|
| `I_LightGBM_Dense_AF` | `0.8251` | `0.9000` | `0.1282` |
| `GoldTestPatchTfidf_Dim16` | `0.8244` | `0.8996` | `0.1307` |
| `GoldFailToPassTfidf_Dim16` | `0.8320` | `0.9008` | `0.1271` |

结论：

- `gold_fail_to_pass_tfidf` 有极小 final-step AUC 提升：`0.9000 -> 0.9008`。
- 但 prefix-row AUC 反而低于 `I`：`0.8839 -> 0.8806`。
- 所以 raw text TF-IDF 不是当前主要收益来源；结构化 gold answer 特征更关键。

## 8. I/J 特征重要性解释

对 LightGBM 的 gain 做粗分组：

### 8.1 `I_LightGBM_Dense_AF`

| 特征组 | gain 占比 |
|---|---:|
| task prompt TF-IDF/SVD | `51.9%` |
| gold structured answer | `30.5%` |
| model_id one-hot | `6.6%` |
| feedback TF-IDF/SVD | `4.5%` |
| action TF-IDF/SVD | `2.9%` |
| process dense | `2.8%` |

### 8.2 `J_LightGBM_Dense_AF_Thought`

| 特征组 | gain 占比 |
|---|---:|
| task prompt TF-IDF/SVD | `48.0%` |
| gold structured answer | `31.9%` |
| model_id one-hot | `5.9%` |
| thought TF-IDF/SVD | `5.2%` |
| feedback TF-IDF/SVD | `3.2%` |
| process dense | `2.9%` |
| action TF-IDF/SVD | `2.5%` |

解释：

- I/J 不是主要靠后期轨迹，而是主要靠题目信息和答案结构。
- `model_id one-hot` 在 train 中有重要性，但 test 的 `model_id` 是 `__MISSING__`，所以它不能解释 heldout 三个模型之间的真实身份差异。
- 当前 setting 下，LightGBM 学到的是“题目/答案复杂度 -> 成功概率”的非线性映射。

## 9. valid-only 阈值调优结果

详细整理文件：

```text
runs/model_holdout_answer_calibrated_full/reports/threshold_tuning_summary/threshold_tuning_summary.md
runs/model_holdout_answer_calibrated_full/reports/threshold_tuning_summary/rate_preserving_policy_summary.md
runs/model_holdout_answer_calibrated_full/reports/threshold_tuning_summary/rate_preserving_policy_all_grids.csv
runs/model_holdout_answer_calibrated_full/reports/threshold_tuning_summary/policy_selected_thresholds_compact.csv
runs/model_holdout_answer_calibrated_full/reports/threshold_tuning_summary/target_precision_calibrated_grid025_compact.csv
runs/model_holdout_answer_calibrated_full/reports/threshold_tuning_summary/target_precision_raw_grid001_compact.csv
```

当前主口径已经统一为：

1. **只在 valid 上选阈值**。
2. 先要求 `abs(valid ΔRate)` 在容忍范围内，例如 `0.5pp / 1pp / 2pp`。
3. 在满足 rate 约束的候选里，选择 **valid Save 最大**的阈值对。
4. test 只用于把 valid 选出的阈值应用过去并报告实际结果；test 不参与选阈值。

所以后面如果讨论“尽量不改变 rate，同时最大化 save”，主看：

```text
runs/model_holdout_answer_calibrated_full/reports/threshold_tuning_summary/rate_preserving_policy_summary.md
runs/model_holdout_answer_calibrated_full/reports/threshold_tuning_summary/rate_preserving_policy_all_grids.csv
```

`target_precision_*` 是另一条旁支：它回答“想要 success/failure 两端 precision 达到某个目标时会怎样”，不再作为主选择规则。

### 9.1 跑过哪些阈值实验

| Run dir | 概率列 | 阈值网格 / 选取方式 | 作用 |
|---|---|---|---|
| `asymmetric_valid_threshold_tuning_fine` | calibrated `prob_cal__` | `ThrS=0.65..0.95` step `0.025`; `ThrF=0.05..0.45` step `0.025`; policies: `rate_1pp/rate_2pp/prec90` | 在 valid 上按策略选阈值，再应用到 test |
| `asymmetric_valid_threshold_tuning_fine_calibrated_step001_rate` | calibrated `prob_cal__` | `ThrS=0.65..0.95` step `0.001`; `ThrF=0.05..0.45` step `0.001` | calibrated 概率的细网格全扫；主用于 rate-preserving policy |
| `asymmetric_valid_threshold_tuning_fine_raw` | raw `prob__` | 同样 `0.025` 网格和 policies | raw 概率版 policy sweep |
| `asymmetric_valid_threshold_tuning_fine_raw_step001` | raw `prob__` | `ThrS=0.65..0.95` step `0.001`; `ThrF=0.05..0.45` step `0.001` | 更密的 raw 网格全扫，共 `724206` 条 valid/test sweep rows |
| `two_end_precision_targets_fine` | calibrated `prob_cal__` | target precision = `0.75/0.80/0.85/0.90`，粗网格 `0.025` | valid 上要求成功端/失败端 precision 都达到 target，然后选省步最多的阈值对 |
| `two_end_precision_targets_fine_raw_step001` | raw `prob__` | target precision = `0.75/0.80/0.85/0.90`，细网格 `0.001` | 更密 raw 网格下的 target-precision 结果 |

说明：

- `ThrS` 是 success threshold：`p >= ThrS` 提前判 success。
- `ThrF` 是 failure threshold：`p <= ThrF` 提前判 failure。
- `target_precision=0.90` 不是“overall Acc 必须 90%”，而是 valid 上 **success 端 precision 和 failure 端 precision 都要 >= 90%**。
- `Test Acc / Test PrecS / Test PrecF` 才是 heldout test 上实际表现。
- 如果觉得 `0.025` 网格“不够全”，确实如此；现在 raw 和 calibrated 概率版都已经有 `0.001` 细网格。

### 9.2 主结果：calibrated grid 0.001，rate-preserving `rate_1pp`

选取规则：valid 上 `abs(ΔRate) <= 1pp`，然后最大化 valid Save。下面的 test 列是应用 valid 选出阈值后的 heldout-test 实际结果。

| Model | ThrS | ThrF | Valid ΔRate | Valid Save | Valid Acc | Test ΔRate | Test Save | Test Acc | Test PrecS | Test PrecF | FP | FN | N |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `J` | `0.791` | `0.450` | `+0.6%` | `70.5%` | `75.1%` | `-5.4%` | `78.6%` | `84.2%` | `91.3%` | `76.5%` | `55` | `134` | `1200` |
| `I` | `0.801` | `0.450` | `+0.9%` | `61.6%` | `78.0%` | `-5.8%` | `73.0%` | `84.6%` | `92.2%` | `77.1%` | `44` | `129` | `1127` |
| `H` | `0.783` | `0.450` | `+0.8%` | `62.9%` | `75.2%` | `-0.8%` | `76.7%` | `86.5%` | `89.5%` | `82.3%` | `75` | `86` | `1196` |
| `K` | `0.743` | `0.387` | `+0.8%` | `50.2%` | `76.8%` | `-0.1%` | `60.8%` | `89.6%` | `90.9%` | `88.1%` | `46` | `48` | `908` |
| `G` | `0.728` | `0.450` | `+0.7%` | `69.7%` | `65.6%` | `-2.9%` | `77.3%` | `74.3%` | `81.2%` | `64.4%` | `137` | `179` | `1231` |
| `D` | `0.802` | `0.450` | `+1.0%` | `75.0%` | `75.2%` | `-8.2%` | `79.4%` | `73.6%` | `83.2%` | `63.9%` | `107` | `226` | `1262` |

直观结论：

- `H/K` 是 rate-preserving 下最稳的两个：test ΔRate 约 `-0.8% / -0.1%`，同时还能省 `76.7% / 60.8%`。
- `I/J` 在 valid 上 rate 很贴近，但 test 上会偏保守，ΔRate 约 `-5.8% / -5.4%`；它们省步高，但会多判一些真实成功为 failure。
- `D/G` 仍然不适合作为主早停策略：虽然 save 很高，但 test rate 下滑明显，尤其 `D`。

### 9.3 对照：raw grid 0.001，rate-preserving `rate_1pp`

| Model | ThrS | ThrF | Valid ΔRate | Valid Save | Valid Acc | Test ΔRate | Test Save | Test Acc | Test PrecS | Test PrecF | FP | FN | N |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `J` | `0.889` | `0.450` | `+0.1%` | `42.4%` | `83.1%` | `-3.1%` | `55.6%` | `89.8%` | `94.8%` | `85.3%` | `21` | `66` | `856` |
| `I` | `0.883` | `0.450` | `+0.7%` | `48.0%` | `83.2%` | `-2.4%` | `59.3%` | `90.9%` | `94.8%` | `86.7%` | `24` | `59` | `908` |
| `H` | `0.851` | `0.450` | `+0.8%` | `62.6%` | `75.1%` | `-0.8%` | `76.5%` | `86.5%` | `89.4%` | `82.2%` | `75` | `86` | `1193` |
| `K` | `0.753` | `0.444` | `+1.0%` | `39.5%` | `74.0%` | `+0.5%` | `51.7%` | `90.3%` | `90.7%` | `89.8%` | `41` | `34` | `775` |
| `G` | `0.650` | `0.411` | `-1.0%` | `75.4%` | `64.6%` | `-3.8%` | `81.7%` | `73.5%` | `80.6%` | `64.1%` | `143` | `199` | `1292` |
| `D` | `0.661` | `0.450` | `+1.0%` | `91.0%` | `71.5%` | `-4.9%` | `93.8%` | `72.2%` | `79.7%` | `62.5%` | `163` | `235` | `1431` |

### 9.4 旧对照：calibrated grid 0.025

来自：

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

这里 `rate_1pp/rate_2pp` 是“尽量不改变 valid rate，同时最大化 save”，不是精度目标；因此 test 上可能很省步，但 precision/Acc 不一定达到某个目标。

### 9.5 旁支：Target precision，calibrated grid 0.025

选取规则：valid 上同时满足 `Prec(S) >= target` 和 `Prec(F) >= target`，满足后选择加权节省最多的阈值对。

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

完整 D/G/I/J/H/K 全表见 `threshold_tuning_summary.md`。

### 9.6 旁支：更细 raw grid 0.001 的 target-precision 意义

`two_end_precision_targets_fine_raw_step001` 是更细的 raw 概率网格。它能找到更细的阈值，例如：

| Model | Target | ThrS | ThrF | Test Acc | Test PrecS | Test PrecF | Test ΔRate | Test Save | FP | FN | N |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `J` | `0.90` | `0.900` | `0.319` | `91.8%` | `94.7%` | `89.4%` | `-1.6%` | `45.2%` | `17` | `40` | `699` |
| `I` | `0.90` | `0.919` | `0.222` | `93.6%` | `96.4%` | `92.1%` | `-1.2%` | `30.9%` | `6` | `24` | `472` |
| `H` | `0.90` | `0.938` | `0.191` | `92.9%` | `94.8%` | `91.4%` | `-1.0%` | `39.2%` | `14` | `29` | `607` |
| `K` | `0.85` | `0.756` | `0.356` | `92.7%` | `92.0%` | `93.9%` | `+1.1%` | `36.0%` | `29` | `13` | `574` |

所以“阈值不全”的判断是对的：

- calibrated 版目前主要是 `0.025` 网格；
- raw 版已经有 `0.001` 细网格；
- 如果要最完整、公平比较，下一步可以再跑一个 calibrated `0.001` target-precision 版本。

## 10. 当前最准确的表述

推荐表述：

> 在 model-holdout setting 下，三个 heldout agent model 的轨迹没有进入 train/valid；模型在测试时也看不到真实 heldout model_id。结果说明 LightGBM 可以利用已知题目池上的 task/answer 结构化信号和 prefix 过程信号，迁移预测未见过的 agent model 表现。

不要过度表述为：

> 模型已经证明可以泛化到完全新题目。

因为：

- 当前不是 instance-holdout。
- 同题其他模型成功率先验本身 AUC 已达 `0.9071`。
- I/J 的 step=0 AUC 已接近 final AUC，说明题目难度信号非常强。

## 11. 后续如果要进一步验证

建议按优先级：

1. 做严格 `instance-holdout + model-holdout`：heldout 模型和 heldout instances 都不出现在 train/valid。
2. 做 `NoTaskSignal + NoGoldAnswer + 脱敏 prefix token`：去掉文件名、测试名、repo 名、API 名等题目指纹。
3. 做只用静态题目信号的 baseline：例如 `gold_patch_chars + difficulty + repo + task_prompt_svd`，明确上限。
4. 做只用过程信号的 baseline：确认“轨迹进展”本身能贡献多少。
5. 阈值策略继续用 valid-only，但报告时明确 valid/test 共享题目池这一点。
