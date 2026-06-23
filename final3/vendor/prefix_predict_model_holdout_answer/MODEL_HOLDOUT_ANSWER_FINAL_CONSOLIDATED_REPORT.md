# Model-Holdout Answer Experiments — Final Consolidated Report

生成时间：2026-04-28

本文档把 `model_holdout_answer_calibrated_full` 这一轮完整重跑后的主实验、补充消融训练、gold answer TF-IDF 消融、step-bucket 分析、valid-only 阈值调优和数据审计合并到一份报告里。

这份报告的定位：

- 作为 `RECENT_MODEL_HOLDOUT_EXPERIMENTS_AUDIT.md` 的更高层总结版。
- 把之前报告里没有纳入的补充训练/消融结果补齐。
- 明确哪些结论可以说，哪些因为不是 instance-holdout 不能过度外推。

## 0. 本次新增合并进来的结果

旧报告主要覆盖了主训练和部分初步分析；本报告额外纳入这些已经跑完的数据：

| 新增结果 | 路径 | 作用 |
|---|---|---|
| task/gold 消融训练 | `runs/model_holdout_answer_calibrated_full/reports/task_answer_ablation_posthoc/` | 分析 task prompt、task signal、structured gold answer 的贡献 |
| no task + no gold 联合消融 | `runs/model_holdout_answer_calibrated_full/reports/task_answer_ablation_no_task_no_gold/` | 验证只靠 prefix 过程信号时 AUC 如何变化 |
| 全模型 step-bucket AUC | `runs/model_holdout_answer_calibrated_full/reports/task_answer_ablation_step_buckets/` | 看不同 prefix 阶段的 AUC 变化 |
| 阈值截断最后 action 分析 | `runs/model_holdout_answer_calibrated_full/reports/task_answer_ablation_posthoc/decision_action_analysis/` | 看不同阈值下模型在哪一步、看到什么 action/feedback signal 后直接下结论 |
| gold patch 维度消融 | `runs/model_holdout_answer_calibrated_full/reports/gold_text_tfidf_patch_dims/` | 比较 `gold_patch_tfidf` 的 8/16/32 维 |
| gold test/fail text 消融 | `runs/model_holdout_answer_calibrated_full/reports/gold_text_tfidf_other_dim16/` | 比较 `gold_test_patch_tfidf` 和 `gold_fail_to_pass_tfidf` |
| calibrated fine-grid 阈值 | `runs/model_holdout_answer_calibrated_full/reports/asymmetric_valid_threshold_tuning_fine_calibrated_step001_rate/` | calibrated 概率下 `0.001` 网格阈值全扫 |
| rate-preserving 阈值汇总 | `runs/model_holdout_answer_calibrated_full/reports/threshold_tuning_summary/rate_preserving_policy_summary.md` | 统一用“valid rate 尽量不变 + save 最大”选阈值 |

## 1. 一页结论

### 1.1 严格数据结论

- 三个 heldout agent model 的轨迹没有进入 train 或 valid。
- train 中 heldout model rows = `0`，valid 中 heldout model rows = `0`，test 中 non-heldout rows = `0`。
- test 输入里的 `model_id` 统一被替换成 `__MISSING__`，真实 heldout model 名只在 `orig_model_id` 里用于统计。
- 但是这仍然是 **model-holdout**，不是 **instance-holdout**：同一道 SWE-bench instance 会通过其他 agent model 出现在 train/valid。

### 1.2 主性能结论

- prefix-row 粒度上，LightGBM 明显强于 LR；`I/J/H` 是最强一组。
- final-step trajectory 粒度上，`I/J` ROC-AUC 都在 `0.900` 左右。
- `NoTaskSignal` 和 `NoTaskPromptTfidf` 几乎不伤 AUC，说明 task prompt TF-IDF 不是唯一核心。
- `NoGoldAnswer` 明显下降，说明 structured gold answer 是关键特征来源。
- `NoTask+NoGold` 仍有 final AUC `0.8054`，但 step=0 AUC 是 `0.5000`；它的能力来自后续 prefix 过程信号，而不是静态题目信息。

### 1.3 阈值策略结论

如果目标是 **valid 上尽量不改变 rate，同时最大化 save**：

- `H/K` 最稳：test ΔRate 分别约 `-0.8%`、`-0.1%`，同时能省 `76.7%`、`60.8%` prefix steps。
- `I/J` 省步很强，但 test 上偏保守，rate 会下降约 `5–6pp`。
- `D/G` 不建议作为主早停策略：save 高，但 test rate 和 precision 风险明显。

### 1.4 最安全表述

可以说：

> 在 model-holdout setting 下，模型没有见过三个 heldout agent model 的轨迹，但能利用已知题目池上的 task/answer 结构化信号和 prefix 过程信号，迁移预测新 agent model 的成功概率。

不建议说：

> 已经证明可以泛化到完全新 SWE-bench 题目。

原因是当前不是 instance-holdout；同题其他模型成功率先验本身已经能达到 ROC-AUC `0.9071`。

## 2. 数据和 split 审计

主运行目录：

```text
runs/model_holdout_answer_calibrated_full
```

### 2.1 全量 prefix table

| 项目 | 数值 |
|---|---:|
| 全部 prefix rows | `386380` |
| 全部 trajectories | `9685` |
| 全部 instances | `490` |
| heldout prefix rows | `83169` |
| heldout trajectories | `1458` |

### 2.2 train/valid/test

| split | 模型类型 | rows | trajectories | model 数 |
|---|---|---:|---:|---:|
| train | non-heldout | `259854` | `6417` | `17` |
| valid | non-heldout | `42419` | `1130` | `17` |
| test | heldout only | `83169` | `1458` | `3` |

三个 heldout agent model：

```text
20251124_mini-v1.17.0_minimax-m2
20251201_mini-v1.17.1_deepseek-v3.2-reasoner
20251210_mini-v1.17.2_kimi-k2-thinking
```

### 2.3 重要粒度差异

| 指标位置 | rows 含义 | rows |
|---|---|---:|
| `evaluation_report.txt` / `metrics_summary.csv` | 全部 prefix rows | `83169` |
| final-step leaderboard / ablation final metrics | 每条 trajectory 最后一步 | `1458` |
| threshold valid sweep | valid trajectories | `1130` |
| threshold test application | heldout test trajectories | `1458` |

所以 `83169` 和 `1458` 不是数据不一致，而是统计粒度不同。

## 3. 模型和特征配置简表

| Model | 类型 | 主要输入 |
|---|---|---|
| `D_Dense_Full_LR` | LR | dense + full text SVD |
| `G_TfIdf_Full_LR` | LR | full text TF-IDF/SVD |
| `H_LightGBM_Dense` | LightGBM | dense / structured features |
| `I_LightGBM_Dense_AF` | LightGBM | dense + action/feedback/task text |
| `J_LightGBM_Dense_AF_Thought` | LightGBM | dense + action/feedback/thought/task text |
| `K_LightGBM_Dense_Full` | LightGBM | dense + full raw text blocks |

消融模型：

| Model | 去掉内容 |
|---|---|
| `Abl_NoTaskPromptTfidf_LightGBM` | 去掉 `tfidf_task_prompt__svd_*` 共 `64` 个 task prompt TF-IDF/SVD 特征 |
| `Abl_NoTaskSignal_LightGBM` | 去掉 task prompt TF-IDF/SVD + `task_prompt_chars`，共 `65` 个 task signal 特征 |
| `Abl_NoGoldAnswer_LightGBM` | 去掉 structured gold-answer dense features，共 `269` 个特征 |
| `Abl_NoTaskSignal_NoGoldAnswer_LightGBM` | 同时去掉 task signal + structured gold-answer features，共 `334` 个特征 |

## 4. 主模型结果

### 4.1 prefix-row 总体结果

来源：

```text
runs/model_holdout_answer_calibrated_full/reports/metrics_summary.csv
```

统计粒度：`83169` 个 heldout test prefix rows。

| Model | ROC-AUC | PR-AUC | Brier | N |
| --- | ---: | ---: | ---: | ---: |
| `H_LightGBM_Dense` | `0.8773` | `0.8762` | `0.1422` | `83169` |
| `I_LightGBM_Dense_AF` | `0.8839` | `0.8893` | `0.1406` | `83169` |
| `J_LightGBM_Dense_AF_Thought` | `0.8803` | `0.8837` | `0.1432` | `83169` |
| `K_LightGBM_Dense_Full` | `0.8623` | `0.8508` | `0.1759` | `83169` |
| `D_Dense_Full_LR` | `0.7814` | `0.7927` | `0.2015` | `83169` |
| `G_TfIdf_Full_LR` | `0.7864` | `0.7918` | `0.1933` | `83169` |

解释：

- LightGBM 系列在这个任务上明显强于 LR。
- `I/J/H` 是主力模型；`K` 加 full text 后没有超过 `I/J/H`。
- 这说明 raw/full text 不是越多越好，结构化信号和有限的过程文本更稳定。

### 4.2 final-step trajectory 结果

来源：

```text
runs/model_holdout_answer_calibrated_full/reports/task_answer_ablation_posthoc/final_step_metrics.csv
runs/model_holdout_answer_calibrated_full/reports/task_answer_ablation_no_task_no_gold/final_step_metrics.csv
```

统计粒度：`1458` 条 heldout test trajectories 的最后一步。下表使用 calibrated probability。

| Model | Acc@0.5 | ROC-AUC | PR-AUC | Brier | Rows |
| --- | ---: | ---: | ---: | ---: | ---: |
| `J_LightGBM_Dense_AF_Thought` | `0.8258` | `0.9007` | `0.9295` | `0.1276` | `1458` |
| `I_LightGBM_Dense_AF` | `0.8251` | `0.9000` | `0.9299` | `0.1282` | `1458` |
| `Abl_NoTaskSignal_LightGBM` | `0.8313` | `0.8960` | `0.9264` | `0.1296` | `1458` |
| `Abl_NoTaskPromptTfidf_LightGBM` | `0.8354` | `0.8946` | `0.9225` | `0.1289` | `1458` |
| `Abl_NoGoldAnswer_LightGBM` | `0.7888` | `0.8271` | `0.8520` | `0.1771` | `1458` |
| `Abl_NoTaskSignal_NoGoldAnswer_LightGBM` | `0.7469` | `0.8054` | `0.8583` | `0.1778` | `1458` |

核心解读：

- 去掉 task prompt TF-IDF 或 task signal，AUC 只小幅下降。
- 去掉 structured gold answer，AUC 从约 `0.900` 降到 `0.8271`，影响很大。
- task + gold 都去掉后，final AUC 仍有 `0.8054`，说明 prefix 过程本身包含相当多信息。

## 5. step-bucket AUC：静态信号 vs 过程信号

来源：

```text
runs/model_holdout_answer_calibrated_full/reports/task_answer_ablation_step_buckets/step_bucket_metrics_all_task_answer_models.csv
runs/model_holdout_answer_calibrated_full/reports/task_answer_ablation_no_task_no_gold/step_bucket_auc_by_model.csv
```

ROC-AUC 是排序指标；Platt calibration 是单调变换，因此 AUC 与 raw 排序一致。

| bucket | J | I | NoTaskSignal | NoTaskPromptTfidf | NoGoldAnswer | NoTask+NoGold |
|---|---:|---:|---:|---:|---:|---:|
| `step=0` | `0.8846` | `0.8890` | `0.8508` | `0.8533` | `0.8267` | `0.5000` |
| `1-3` | `0.8851` | `0.8908` | `0.8603` | `0.8629` | `0.8304` | `0.6647` |
| `4-6` | `0.8876` | `0.8919` | `0.8700` | `0.8741` | `0.8304` | `0.7464` |
| `7-12` | `0.8896` | `0.8940` | `0.8773` | `0.8794` | `0.8324` | `0.7711` |
| `13-24` | `0.8938` | `0.8978` | `0.8837` | `0.8828` | `0.8359` | `0.7910` |
| `25+` | `0.8656` | `0.8693` | `0.8599` | `0.8593` | `0.8018` | `0.7841` |
| `final` | `0.9007` | `0.9000` | `0.8960` | `0.8946` | `0.8271` | `0.8054` |

关键解释：

- `NoTask+NoGold` 在 `step=0` 是 `0.5000`，完全没有静态区分能力。
- 它后面逐步升到 `0.8054`，说明 action、feedback、thought、错误信息、测试输出等过程信号确实有用。
- `I/J` 在 `step=0` 已经很强，主要是题目难度、task prompt、structured gold answer 在起作用。
- `NoGoldAnswer` 的 `step=0` 仍有 `0.8267`，说明 task prompt / static task signal 本身也能预测题目难度。

### 5.1 阈值截断时的最后 action

来源：

```text
runs/model_holdout_answer_calibrated_full/reports/task_answer_ablation_posthoc/decision_action_analysis/
```

其中更适合阅读阈值变化的是：

```text
runs/model_holdout_answer_calibrated_full/reports/task_answer_ablation_posthoc/decision_action_analysis/decision_action_threshold_readable_report.md
runs/model_holdout_answer_calibrated_full/reports/task_answer_ablation_no_task_no_gold/decision_action_analysis/decision_action_threshold_readable_report.md
runs/model_holdout_answer_calibrated_full/reports/test_without_valid_instances_posthoc/decision_action_analysis/combined_decision_action_tradeoff_report.md
```

如果采用“去掉 valid-instance 对应 heldout-test trajectories”的 filtered 口径，优先看最后一个 `test_without_valid_instances_posthoc` 报告；它同时合并了 action 信号、早停准确率、FP/FN、adjusted resolve rate 和 step 节省比例。

分析方法：对每条 trajectory、每个 predictor、每个阈值，从 prefix 序列里找第一个会触发 early-stop 的位置；`p >= threshold` 判为 success，`p <= 1 - threshold` 判为 failure。然后把这个 `decision_prefix_id` join 回 prefix table，取该步的 `last_action_text`、`last_feedback_text`、action subtype 和测试/traceback/tool-error 等信号。

关键表：

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

解读：

- `I/J` 主模型在 `0.8` 阈值下大量直接在 `step=0` 做决定，最后 action 是 `none`。这说明许多 early decision 不是因为看到了测试结果或 traceback，而是因为静态题目/答案先验已经足够强。
- 阈值升到 `0.9` 后，`J` 的 success 决策明显后移：`step0_share` 从 `0.906` 降到 `0.388`，top subtype 从 `none` 转到 `read_search`。也就是说加 thought 特征后，极高置信 success 更容易等到模型至少做过定位/搜索。
- failure 决策比 success 更晚一点，但 `I/J` 仍然以 `none` 和早期 `read_search` 为主；last-step test fail/pass/traceback/tool-error 比例很低，说明主模型并不是主要等测试失败后才判失败。
- 去掉 task signal 或 task prompt TF-IDF 后，高阈值决策显著后移，最后 action 更多变成 `read_search`、`run_cli`、`test`。例如 `NoTaskSignal` 的 `0.9 failure` 平均到第 `32` 步，`tests_so_far=3.45`，这更像真正依赖过程信号。
- `NoTask+NoGold` 是最干净的过程信号对照：`0.7` 以上没有任何 step0 截断，全部要等 agent 做过 action 后才触发；`0.8` 时只截断 `386/1458` 条，但准确率到 `90.9%`，说明只靠过程信号也能形成高质量早停，只是覆盖率明显下降。
- `NoTask+NoGold` 在 `0.9/0.95` 基本只剩 failure，没有 success 截断。这说明没有 task/gold 后，模型更容易形成“确定失败”的高置信，而很难形成“确定成功”的高置信。
- 从 command kind 看，非 step=0 的主要触发动作通常是 `find`、`ls`、`grep_pipeline`、`cat_read`；真正的 `pytest`、test pass/fail、traceback、tool-error 在主模型高置信截断里不是主导信号。

## 6. 为什么 AUC 会这么高

这个实验不是“完全新题目”的泛化，而是“新 agent model 跑已知题目池”的泛化。

已有 sanity check：

> 只用同一道 SWE instance 上其他非 heldout 模型的 resolved rate，预测 heldout 三个模型最终是否成功。

| 指标 | 数值 |
|---|---:|
| test trajectories | `1458` |
| test instances | `489` |
| prior missing | `0` |
| prior ROC-AUC | `0.9071` |
| prior PR-AUC | `0.9177` |

所以高 AUC 不离谱：同一道题对不同 agent model 的成功/失败高度相关。模型学到的很大一部分是“题目难度”和“官方答案复杂度”。

## 7. Gold raw-text TF-IDF 消融

### 7.1 结果表

来源：

```text
runs/model_holdout_answer_calibrated_full/reports/gold_text_tfidf_patch_dims/final_step_metrics.csv
runs/model_holdout_answer_calibrated_full/reports/gold_text_tfidf_other_dim16/final_step_metrics.csv
```

统计粒度：final-step trajectory，calibrated probability。

| Model | Acc@0.5 | ROC-AUC | PR-AUC | Brier |
| --- | ---: | ---: | ---: | ---: |
| `I_LightGBM_Dense_AF` | `0.8251` | `0.9000` | `0.9299` | `0.1282` |
| `O_LightGBM_Dense_AF_GoldPatchTfidf_Dim8` | `0.8210` | `0.8992` | `0.9284` | `0.1298` |
| `O_LightGBM_Dense_AF_GoldPatchTfidf_Dim16` | `0.8278` | `0.8994` | `0.9273` | `0.1297` |
| `O_LightGBM_Dense_AF_GoldPatchTfidf_Dim32` | `0.8278` | `0.8994` | `0.9265` | `0.1281` |
| `P_LightGBM_Dense_AF_GoldTestPatchTfidf_Dim16` | `0.8244` | `0.8996` | `0.9276` | `0.1307` |
| `Q_LightGBM_Dense_AF_GoldFailToPassTfidf_Dim16` | `0.8320` | `0.9008` | `0.9273` | `0.1271` |

### 7.2 结论

- `gold_patch_tfidf` 从 8 到 32 维，AUC 基本不变，且没有稳定超过 `I`。
- `gold_test_patch_tfidf` 没有明显收益。
- `gold_fail_to_pass_tfidf` 有极小 final-step AUC 提升：`0.9000 -> 0.9008`。
- 综合看，raw gold text TF-IDF 不是主要收益来源；更重要的是 structured gold answer 特征，例如 patch 规模、文件数、hunk 数、测试数、API token 命中、路径深度、与 prefix 的文件/API/test token overlap 等。

## 8. 特征重要性解释

LightGBM gain 粗分组：

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

解释边界：

- `model_id one-hot` 在 train 中有重要性，但 valid/test 都把 `model_id` 隐藏成 `__MISSING__`，不能解释 heldout agent model 的真实身份差异。
- task prompt TF-IDF gain 很高，不等于它单独决定结果；消融显示去掉 task prompt 后 AUC 只小幅下降，因为 structured gold answer 和其他过程信号可以补偿。
- structured gold answer 的效果更稳：去掉它会造成明显下降。

## 9. valid-only 阈值调优

主选择规则已经统一为：

1. 只在 valid 上选择阈值。
2. 要求 `abs(valid ΔRate)` 在容忍范围内。
3. 在满足 rate 约束的候选里，最大化 valid Save。
4. test 只用于应用 valid 选出的阈值并报告实际结果。

主表来源：

```text
runs/model_holdout_answer_calibrated_full/reports/threshold_tuning_summary/rate_preserving_policy_all_grids.csv
runs/model_holdout_answer_calibrated_full/reports/threshold_tuning_summary/rate_preserving_policy_summary.md
```

下面是 calibrated `0.001` fine-grid、`rate_1pp` policy：valid 上 `abs(ΔRate) <= 1pp` 后最大化 save。

| Model | ThrS | ThrF | Valid ΔRate | Valid Save | Valid Acc | Test ΔRate | Test Save | Test Acc | Test PrecS | Test PrecF | FP | FN | N |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `J` | `0.791` | `0.450` | `+0.6%` | `70.5%` | `75.1%` | `-5.4%` | `78.6%` | `84.2%` | `91.3%` | `76.5%` | `55` | `134` | `1200` |
| `I` | `0.801` | `0.450` | `+0.9%` | `61.6%` | `78.0%` | `-5.8%` | `73.0%` | `84.6%` | `92.2%` | `77.1%` | `44` | `129` | `1127` |
| `H` | `0.783` | `0.450` | `+0.8%` | `62.9%` | `75.2%` | `-0.8%` | `76.7%` | `86.5%` | `89.5%` | `82.3%` | `75` | `86` | `1196` |
| `K` | `0.743` | `0.387` | `+0.8%` | `50.2%` | `76.8%` | `-0.1%` | `60.8%` | `89.6%` | `90.9%` | `88.1%` | `46` | `48` | `908` |
| `G` | `0.728` | `0.450` | `+0.7%` | `69.7%` | `65.6%` | `-2.9%` | `77.3%` | `74.3%` | `81.2%` | `64.4%` | `137` | `179` | `1231` |
| `D` | `0.802` | `0.450` | `+1.0%` | `75.0%` | `75.2%` | `-8.2%` | `79.4%` | `73.6%` | `83.2%` | `63.9%` | `107` | `226` | `1262` |

策略建议：

- 如果优先保 rate：选 `H` 或 `K`。
- 如果优先 save 且能接受 heldout test rate 下降：`I/J` 可作为激进策略。
- `D/G` 只能作为分析对照，不建议作为主策略。

## 10. 校准的作用

Platt/sigmoid calibration 用 valid 数据改变概率刻度，不改变排序能力：

- ROC-AUC / PR-AUC 基本不变，因为排序不变。
- Brier / log-loss 和阈值行为会变，因为概率被重新映射。
- 阈值策略里 calibrated 和 raw 的差异很明显：同一个 `rate_1pp` 目标下，raw `I/J` 更保守、precision 更高但 save 更低；calibrated `I/J` save 更高但 test rate 掉得更多。

因此：

- 如果只看排序，用 AUC 评价即可。
- 如果要做早停决策，必须明确使用 raw 还是 calibrated 概率，并且阈值只能从 valid 选择。

## 11. 最终建议

### 11.1 报告主实验时

推荐主报：

- prefix-row：`I/J/H` 的 AUC 和 Brier。
- final-step：`I/J` 的 AUC `0.900` 左右。
- 消融：`NoGoldAnswer` 和 `NoTask+NoGold` 的下降，证明 structured answer 和过程信号的作用。

### 11.2 报告早停策略时

推荐主报：

- calibrated `0.001` fine-grid，`rate_1pp` policy。
- 结论以 `H/K` 最稳为主，`I/J` 作为更激进省步方案。

### 11.3 下一步实验

如果要回答“完全新题目是否也有效”，必须做：

- instance-holdout 或 repo/instance group-holdout；
- 重新 fit TF-IDF/SVD 和 dense encoders；
- 在新 instance 上评估 step=0、prefix bucket、final-step 三套指标；
- 同时保留 no-task/no-gold 消融，区分题目先验和过程信号。

## 12. 主要引用文件

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
