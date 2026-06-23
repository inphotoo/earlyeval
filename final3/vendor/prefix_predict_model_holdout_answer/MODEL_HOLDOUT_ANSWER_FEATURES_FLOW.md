# Model-Holdout + Gold Answer Features：流程与字段说明

这份文档解释当前目录的完整实验流程：

- final3 代码目录：`final3/vendor/prefix_predict_model_holdout_answer/`
- 主运行：`run_all.py`
- gold answer 特征：`answer_features.py`
- 特征工程：`feature_engineer.py`
- model holdout 划分：`model_holdout_split.py`
- validation-only 概率校准：`probability_calibration.py`
- trajectory 排名报告：`model_ranking_report_posthoc.py`
- gold text TF-IDF 事后消融：`gold_text_tfidf_ablation_posthoc.py`

## 1. 这套实验到底在预测什么

目标是：给定一条 SWE-bench 轨迹跑到第 `t` 步时已有的信息，预测这条轨迹最终是否会 `resolved=True`。

一条完整轨迹会被拆成很多个 prefix row：

- 第 0 步：只看到初始任务。
- 第 1 步：看到第 1 步 action/feedback。
- 第 2 步：看到前 2 步。
- 直到最后一步：看到整条轨迹。

每个 prefix row 的标签都相同：这条完整轨迹最后有没有解出来。

因此当前数据里有两个常用统计口径：

- `prefix-level`：每个 prefix row 都算一次，当前 full run 是 `83169` 行。
- `trajectory-level final-step`：每条轨迹只取最后一个 prefix row 算一次，当前 full run 是 `1458` 条轨迹。

你现在更关心“每条轨迹算一次”，所以主结论应该优先看：

`runs/model_holdout_answer_calibrated_full/reports/model_ranking_report_like_ref_calibrated/final_step_prefix_model_leaderboard.csv`

而不是只看 `evaluation_report.txt` 里的 prefix-level 总 AUC。

## 2. 数据划分：model holdout 是怎么做的

当前实验使用 `--split-by model_holdout`。

核心逻辑：

1. 从官方 verified 集合读取最多 `500` 个 SWE-bench instance。
2. 和轨迹 parquet 里的 `instance_id` 取交集。
3. 自动或手动选择 3 个 holdout 模型作为测试模型。
4. 训练集使用其余模型。
5. valid 从训练模型内部按 instance 分组抽一部分。
6. test 只来自 3 个 holdout 模型。

当前 full run 的测试集是：

- `83169` prefix rows
- `1458` trajectories
- `489` instances
- `3` heldout agent models

三个 heldout 模型是：

- `20251124_mini-v1.17.0_minimax-m2`
- `20251201_mini-v1.17.1_deepseek-v3.2-reasoner`
- `20251210_mini-v1.17.2_kimi-k2-thinking`

### 短轨迹过滤

训练和 valid 里会过滤过短轨迹，减少训练噪声。

test 不过滤过短轨迹。

这点很重要：最终测试保留 holdout 模型的实际轨迹分布，不为了好看指标而删测试样本。

## 3. model_id 是怎么处理的

训练时：

- `FeatureEngineer(include_model_id=True)`
- 训练模型的 `model_id` 会进入 one-hot。
- 也就是说，训练时模型确实可以学习“训练模型身份”的影响。

valid 和 test 时：

- `model_id` 被统一改成 `__MISSING__`
- `model` 也被统一改成 `__MISSING__`
- 原始模型身份保存在 `orig_model_id`，只用于分组统计和报告，不进入模型输入。

所以当前 heldout test 是“看不到真实 model_id”的。

报告里可以验证：

```text
model_id feature values: ['__MISSING__']
model_id_input_mode: ['test_missing']
```

这不是“不用 model_id 训练”，而是：

- 训练：用训练模型的 model_id。
- valid/test：未知模型统一映射到 `__MISSING__`。

这样更接近“新模型没见过，只知道它是未知模型”的设定。

## 4. 从原始轨迹到 prefix table

主流程大致是：

1. 读取 parquet 轨迹。
2. `step_builder.py` 把原始 messages 重建成 step table。
3. `prefix_builder.py` 把每条轨迹展开成 prefix table。
4. `answer_features.py` 按 `instance_id` join 官方 verified answer 信息。
5. `feature_engineer.py` 把 prefix table 转成训练矩阵。
6. `trainer.py` 训练 LR / LightGBM。
7. `evaluator.py` 输出 prefix-level 报告。
8. `model_ranking_report_posthoc.py` 输出 trajectory-level 排名/早停报告。

prefix table 的一行表示：

“某条轨迹跑到某一步时，我们已经看到了什么，以及这条轨迹最终有没有成功。”

## 5. Gold answer 特征从哪里来

gold answer 特征来自：

`swebench_verified/test.jsonl`

每个 instance 里主要使用这些字段：

- `repo`
- `difficulty`
- `version`
- `problem_statement`
- `hints_text`
- `patch`
- `test_patch`
- `FAIL_TO_PASS`
- `PASS_TO_PASS`

这里的 gold answer 是官方答案信息。这个实验是刻意使用“已知最终答案辅助预测”，不是线上真实不可见答案场景。

## 6. 最容易混淆的几个字段

### `patch`

官方正确修复代码的 diff。

简单说：这题最后应该怎么改业务代码。

### `test_patch`

官方答案里新增或修改测试的 diff。

简单说：这题最后用哪些测试来证明修好了。

有些 SWE-bench 题有测试补丁，有些没有。

### `FAIL_TO_PASS`

这是一个测试名列表。

这些测试在原始 buggy 代码上应该失败，在打上正确 patch 后应该通过。

用人话说：这些是“证明 bug 被修好的关键测试”。

模型看到它以后，能知道这题主要要修到哪些测试通过。

### `PASS_TO_PASS`

这也是一个测试名列表。

这些测试在原始代码上应该通过，在修复后也应该继续通过。

用人话说：这些是“不要被修坏的回归测试”。

它告诉模型：修 bug 时不能破坏哪些已有行为。

## 7. Gold answer 字段含义

### 7.1 基础元信息

| 字段 | 类型 | 含义 | 对训练的作用 |
|---|---:|---|---|
| `gold_has_answer` | 数值 | 这个 instance 是否成功 join 到 verified answer。当前正常应为 1。 | 区分是否有答案元数据。 |
| `gold_repo` | 类别 | 题目所在仓库。 | one-hot 后告诉模型仓库差异，例如 Django / sklearn / matplotlib。 |
| `gold_difficulty` | 类别 | 官方难度标签，例如 `<15_min_fix`。 | one-hot 后告诉模型题目难度先验。 |
| `gold_version` | 类别 | SWE-bench 记录的版本字段。 | one-hot 后给模型版本/数据来源先验。 |
| `gold_problem_statement_chars` | 数值 | issue 描述字符数。 | 问题描述越长，可能越复杂。 |
| `gold_hints_chars` | 数值 | hints 文本字符数。 | hints 越多，可能说明题目有更多提示。 |
| `gold_has_hints` | 布尔 | 是否有 hints。 | 告诉模型有没有额外提示。 |

例子：

- `gold_difficulty__<15_min_fix` 是 `gold_difficulty` one-hot 后的一列。
- 如果这题难度是 `<15_min_fix`，这一列就是 1，否则是 0。
- LightGBM 看到这个特征重要，意思是：官方简单题和成功概率有明显关系。

### 7.2 gold code patch 规模

这些字段来自官方 `patch`。

| 字段 | 类型 | 含义 | 对训练的作用 |
|---|---:|---|---|
| `gold_patch_chars` | 数值 | 官方代码 patch 的字符长度。 | 修复越长，通常题越复杂。 |
| `gold_patch_hunks` | 数值 | patch 里 `@@ ... @@` 代码块数量。 | 修改点越分散，通常越复杂。 |
| `gold_patch_files_count` | 数值 | 官方代码 patch 修改了几个文件。 | 多文件修改通常更难。 |
| `gold_patch_added_lines` | 数值 | patch 新增了多少真实代码行，不含 `+++` 文件头。 | 新增代码规模。 |
| `gold_patch_deleted_lines` | 数值 | patch 删除了多少真实代码行，不含 `---` 文件头。 | 删除代码规模。 |
| `gold_patch_max_dir_depth` | 数值 | patch 触达文件路径的最大目录深度。 | 修改位置越深，可能越偏内部实现。 |
| `gold_primary_patch_ext` | 类别 | 第一个被修改文件的后缀，例如 `.py`。 | one-hot 后给模型文件类型信息。 |
| `gold_primary_patch_dir` | 类别 | 第一个被修改文件所在目录。 | one-hot 后给模型模块位置先验。 |

例子：

- `gold_patch_max_dir_depth=4` 可能表示修改文件类似 `django/db/models/query.py`。
- 它不是“第几行”，而是文件路径有多深。
- 路径更深通常说明改的是内部模块，不是顶层文件。

### 7.3 gold test patch 规模

这些字段来自官方 `test_patch`。

| 字段 | 类型 | 含义 | 对训练的作用 |
|---|---:|---|---|
| `gold_has_test_patch` | 布尔 | 官方答案是否包含测试补丁。 | 有测试补丁的题，轨迹里跑测试/提到测试名可能更有用。 |
| `gold_test_patch_chars` | 数值 | test patch 的字符长度。 | 测试改动越长，验证逻辑可能越复杂。 |
| `gold_test_patch_hunks` | 数值 | test patch 里 `@@ ... @@` 测试代码块数量。 | 测试修改点数量。 |
| `gold_test_files_count` | 数值 | test patch 修改了几个测试文件。 | 涉及多个测试文件通常更复杂。 |
| `gold_test_added_lines` | 数值 | test patch 新增了多少测试行。 | 新增测试规模。 |
| `gold_test_deleted_lines` | 数值 | test patch 删除了多少测试行。 | 删除测试规模。 |

例子：

- `gold_test_patch_chars` 大，不代表模型直接知道答案正确与否。
- 它只告诉模型：“官方验证这个问题的测试补丁很长/很短”。
- 这类信息更像题目结构和验证复杂度。

### 7.4 测试列表数量

| 字段 | 类型 | 含义 | 对训练的作用 |
|---|---:|---|---|
| `gold_fail_to_pass_count` | 数值 | `FAIL_TO_PASS` 里有多少个关键失败测试。 | 关键测试越多，修复目标可能越复杂。 |
| `gold_pass_to_pass_count` | 数值 | `PASS_TO_PASS` 里有多少个回归测试。 | 回归保护越多，修复约束越多。 |
| `gold_fail_to_pass_text` | 文本 | 把 `FAIL_TO_PASS` 测试名拼成一段文本。 | 可进入 TF-IDF，让模型识别测试名 token。 |

`FAIL_TO_PASS` 不是“训练标签”。

它是测试名清单。例如里面可能出现：

```text
tests.test_xxx::test_bug_case
```

如果某条轨迹的 action/feedback/thought 里也出现类似测试名，匹配特征和 TF-IDF 特征就可能增强成功概率。

### 7.5 patch token / 关键词统计

| 字段 | 类型 | 含义 | 对训练的作用 |
|---|---:|---|---|
| `gold_patch_api_token_count` | 数值 | 从 patch 里的 `def` / `class` 名、文件名里提取出的 token 数。 | 修复涉及的函数/类/模块越多，可能越复杂。 |
| `gold_patch_import_token_count` | 数值 | 从 patch 的 import 语句中提取的模块 token 数。 | import 改动越多，可能涉及依赖或模块边界。 |
| `gold_patch_exception_keyword_count` | 数值 | patch/test/problem 中 error、exception、traceback、assert 等词出现次数。 | 错误/异常类题目的先验信号。 |
| `gold_patch_test_keyword_count` | 数值 | test、pytest、unittest、assert 等测试相关词出现次数。 | 测试驱动信号强弱。 |
| `gold_patch_config_keyword_count` | 数值 | config、setting、option、env、version 等配置词出现次数。 | 配置/版本类题目的先验信号。 |

这些字段不是直接判断“当前轨迹做对没做对”，而是描述“官方答案长什么样”。

## 8. Gold answer 和当前 prefix 怎么匹配

除了静态字段，代码还做了“当前轨迹文本是否提到 gold answer 相关内容”的匹配。

被匹配的当前轨迹文本有 6 类：

- `prefix_action_text`：到当前步为止所有 action。
- `prefix_feedback_text`：到当前步为止所有工具输出/反馈。
- `prefix_thought_text`：到当前步为止所有 thought。
- `last_action_text`：最后一步 action。
- `last_feedback_text`：最后一步反馈。
- `last_thought_text`：最后一步 thought。

被匹配的 gold answer token 有 3 类：

- file tokens：官方 patch/test_patch 涉及文件名和 basename。
- api tokens：官方 patch 里的函数名、类名、import 模块名、文件名 token。
- test tokens：`FAIL_TO_PASS`、`PASS_TO_PASS`、test patch 文件名里的 token。

字段命名模板：

```text
gold_{prefix/last}_{action/feedback/thought}_{file/api/test}_{hits/jaccard/hit_any}
```

含义：

| 后缀 | 含义 |
|---|---|
| `_hits` | 当前文本和 gold token 有多少个重合词。 |
| `_jaccard` | 重合词数量 / 两边总词集合数量，范围 0 到 1。 |
| `_hit_any` | 是否至少命中一个 token。 |

例子：

| 字段 | 人话解释 |
|---|---|
| `gold_prefix_action_file_hits` | 到当前步为止，action 里提到了多少个官方答案会改到的文件名。 |
| `gold_last_feedback_test_hit_any` | 最后一步反馈里是否出现了官方关键测试名相关 token。 |
| `gold_prefix_thought_api_jaccard` | thought 里提到的词和官方修复 API token 的重合比例。 |

这些字段的作用是：

- 如果轨迹越早开始碰到官方答案相关文件/API/测试名，模型可能更倾向判断最终会成功。
- 如果轨迹一直没有碰到 gold patch 相关位置，模型可能更倾向判断失败。

## 9. Gold raw text TF-IDF 是什么

结构化 gold 字段之外，还有几段原始 gold 文本可以进 TF-IDF：

| TF-IDF block | 原始列 | 含义 |
|---|---|---|
| `tfidf_gold_patch` | `gold_patch_text` | 官方代码 patch 原文。 |
| `tfidf_gold_test_patch` | `gold_test_patch_text` | 官方测试 patch 原文。 |
| `tfidf_gold_fail_to_pass` | `gold_fail_to_pass_text` | 关键失败测试名文本。 |
| `tfidf_gold_answer_summary` | `gold_answer_summary_text` | problem/hints/patch files/test files/fail tests/API tokens 的摘要拼接。 |

TF-IDF 的作用方式：

1. 只在训练集上学习词表。
2. 把每道题的 gold 文本转成词频权重向量。
3. 再用 SVD 压成固定维度。
4. 拼到模型输入矩阵里。

它不是“比较当前轨迹文本和答案文本是否一样”。

它更像告诉模型：

- 这题答案 patch 经常出现哪些词。
- 这题测试名经常出现哪些词。
- 这些词在训练集中通常对应更容易成功还是更容易失败。

这也是为什么 full gold raw text 不一定更好：文本维度多、噪声大，LightGBM 容易学到训练模型/训练题里的偶然词，而不是真正可泛化的规律。

## 10. Dense、AF、Thought、Full 分别是什么

### Dense

Dense 是结构化特征，包括：

- 轨迹进度：当前步数、已经调用多少工具、文本长度等。
- 动作统计：read/edit/test/bash 次数。
- 错误和测试状态：是否出现 traceback、测试失败、测试通过。
- 循环/风险：重复查看、长期没 edit、过早 submit。
- thought/content 长度统计。
- gold answer 结构化字段。
- gold answer 与当前 prefix 的命中特征。
- 如果 `include_model_id=True`，还包括 `model_id` one-hot。

当前实验里的 Dense 默认已经包含 gold answer 的结构化字段。

### AF

AF 是 Action + Feedback 的文本 TF-IDF：

- `tfidf_task_prompt`
- `tfidf_prefix_action`
- `tfidf_prefix_feedback`
- `tfidf_last_action`
- `tfidf_last_feedback`

### Thought

Thought 是额外加入：

- `tfidf_prefix_thought`
- `tfidf_last_thought`

### Full

Full 是所有文本 TF-IDF：

- AF
- Thought
- assistant content
- gold raw text TF-IDF

在当前代码里，`Full` 会包含 gold raw text TF-IDF。

## 11. H / I / J / K / D / G 这些模型到底用了哪些答案特征

| 模型 | 算法 | 输入 | 是否用结构化 gold 字段 | 是否用 gold raw text TF-IDF |
|---|---|---|---:|---:|
| `H_LightGBM_Dense` | LightGBM | Dense | 是 | 否 |
| `I_LightGBM_Dense_AF` | LightGBM | Dense + AF | 是 | 否 |
| `J_LightGBM_Dense_AF_Thought` | LightGBM | Dense + AF + Thought | 是 | 否 |
| `K_LightGBM_Dense_Full` | LightGBM | Dense + Full | 是 | 是 |
| `D_Dense_Full_LR` | Logistic Regression | Dense + Full | 是 | 是 |
| `G_TfIdf_Full_LR` | Logistic Regression | Full TF-IDF only | 否 | 是 |

关键结论：

- `I` 不是“完全不用答案”。
- `I` 用了结构化 gold answer 字段和匹配字段。
- `I` 没用 `gold_patch_text/test_patch_text/fail_to_pass_text` 这些 raw text TF-IDF。
- `K` 才是在 `I` 的基础上进一步加入所有 raw gold text TF-IDF。

所以 `I > K` 的含义不是“答案特征没用”。

更准确地说：

> 结构化答案特征有用；全量原始答案文本 TF-IDF 在当前 holdout 设置下带来了噪声和过拟合。

## 12. 模型训练时这些字段怎么用

### 数值字段

例如：

- `gold_patch_chars`
- `gold_test_patch_chars`
- `gold_patch_hunks`
- `gold_fail_to_pass_count`

处理方式：

1. 转成数字。
2. 缺失填 `-1` 或 `0`。
3. 用 `StandardScaler` 标准化。
4. 放进 Dense 矩阵。

LightGBM 使用方式：

- 自动学习阈值切分。
- 例如它可以学到：`gold_patch_chars > 某个值` 时成功概率变化。

LR 使用方式：

- 学一个线性权重。
- 例如 `gold_patch_chars` 越大，概率整体上升或下降。

### 布尔字段

例如：

- `gold_has_test_patch`
- `gold_prefix_action_file_hit_any`

处理方式：

- `False -> 0`
- `True -> 1`

模型使用方式：

- LightGBM 可以学到“有无测试 patch”与成功概率的非线性关系。
- LR 学一个正/负权重。

### 类别字段

例如：

- `gold_repo`
- `gold_difficulty`
- `gold_version`
- `gold_primary_patch_ext`
- `gold_primary_patch_dir`
- `model_id`

处理方式：

1. 训练集拟合 `LabelEncoder`。
2. 转成 one-hot。
3. 未见类别统一映射到 `__MISSING__`。

例子：

- `gold_difficulty__<15_min_fix`
- `gold_repo__django/django`
- `model_id____MISSING__`

### 文本字段

例如：

- `prefix_action_text`
- `prefix_feedback_text`
- `gold_patch_text`
- `gold_fail_to_pass_text`

处理方式：

1. `TfidfVectorizer` 把文本转成词权重。
2. 每个 block 单独做 SVD 降维。
3. 降维后的列拼进最终矩阵。

例子：

- `tfidf_gold_patch__svd_0`
- `tfidf_prefix_action__svd_12`

这些列不是某个具体词，而是多个词压缩后的综合方向。

## 13. 当前 AUC 到底是什么口径

### `evaluation_report.txt` 里的 AUC

这是 prefix-level AUC。

文件：

`runs/model_holdout_answer_calibrated_full/reports/evaluation_report.txt`

当前 full run 的 `Overall Metrics Summary` 是在 `83169` 个 prefix row 上计算。

含义：

> 每个中间步骤都算一个样本。长轨迹会贡献更多行。

这个指标适合看“模型在过程中的每一步判断能力”，但不适合作为最终轨迹排序的唯一主指标。

### `model_ranking_report_like_ref_calibrated` 里的 leaderboard AUC

文件：

`runs/model_holdout_answer_calibrated_full/reports/model_ranking_report_like_ref_calibrated/final_step_prefix_model_leaderboard.csv`

这里的 AUC 是 trajectory-level final-step AUC。

代码逻辑：

1. 对每个 `traj_id` 只取 `prefix_step_idx` 最大的那一行。
2. 每条轨迹只保留一个概率。
3. 用这 `1458` 条轨迹计算 ROC-AUC / PR-AUC / Brier / LogLoss。

所以它不是“决策后的 AUC”。

它是：

> 每条轨迹最后一步的概率排序 AUC。

### 阈值决策表是什么

ranking report 里每个 threshold 下的表不是 AUC。

它是在模拟早停：

- 如果某一步 `p >= threshold`，提前判定 success。
- 如果某一步 `p <= 1 - threshold`，提前判定 failure。
- 如果一直没触发阈值，保留原始结果。

这些表里的核心指标是：

- 决策数
- 决策准确率
- FN
- FP
- 节省步数比例
- 调整后的 resolve rate
- 排名是否变化

这里没有“决策后的 AUC”。

## 14. 当前 prefix-level 和 trajectory-level 结果差异

当前 calibrated full run 里，主要模型的 trajectory-level final-step 结果如下：

| 模型 | Acc@0.5 | ROC-AUC | Brier | 说明 |
|---|---:|---:|---:|---|
| `J_LightGBM_Dense_AF_Thought` | 0.8258 | 0.9007 | 0.1276 | final-step AUC 最高。 |
| `I_LightGBM_Dense_AF` | 0.8251 | 0.9000 | 0.1282 | 结构化 gold + AF，很稳。 |
| `H_LightGBM_Dense` | 0.8285 | 0.8971 | 0.1251 | Dense-only 也很强，说明结构化信号贡献大。 |
| `K_LightGBM_Dense_Full` | 0.8285 | 0.8791 | 0.1434 | 加全量 gold raw text 后 AUC 下降。 |
| `G_TfIdf_Full_LR` | 0.7401 | 0.8125 | 0.1768 | 只有 Full TF-IDF，泛化较弱。 |
| `D_Dense_Full_LR` | 0.7298 | 0.8013 | 0.1822 | LR + Full，不如 LightGBM。 |

这说明：

- 如果按“每条轨迹算一次”，应该看 `final_step_prefix_model_leaderboard.csv`。
- 当前最稳的主结果是 `I/J/H`，不是 `K`。
- `K` 的 raw gold text TF-IDF 是负优化。

## 15. Gold text TF-IDF 消融结论

事后消融复用了已有中间表和 FeatureEngineer，没有重跑最慢的 step/prefix/gold join。

输出目录：

- `runs/model_holdout_answer_calibrated_full/reports/gold_text_tfidf_patch_dims/`
- `runs/model_holdout_answer_calibrated_full/reports/gold_text_tfidf_other_dim16/`

trajectory-level final-step calibrated 结果：

| 模型 | Acc@0.5 | ROC-AUC | Brier | 结论 |
|---|---:|---:|---:|---|
| `I_LightGBM_Dense_AF` | 0.8251 | 0.9000 | 0.1282 | 基准：结构化 gold + AF。 |
| `GoldPatchTfidf Dim8` | 0.8210 | 0.8992 | 0.1298 | patch raw text 维度低时略差。 |
| `GoldPatchTfidf Dim16` | 0.8278 | 0.8994 | 0.1297 | accuracy 略升，AUC 没升。 |
| `GoldPatchTfidf Dim32` | 0.8278 | 0.8994 | 0.1281 | Brier 接近 I，但 AUC 仍略低。 |
| `GoldTestPatchTfidf Dim16` | 0.8244 | 0.8996 | 0.1307 | test_patch raw text 基本无增益。 |
| `GoldFailToPassTfidf Dim16` | 0.8320 | 0.9008 | 0.1271 | 小幅最好，说明关键测试名有一点有效信号。 |
| `K_LightGBM_Dense_Full` | 0.8285 | 0.8791 | 0.1434 | 全量 raw text 明显负优化。 |

结论：

1. `FAIL_TO_PASS` 测试名文本最值得保留，小幅正收益。
2. `gold_patch` raw text 只能算弱信号。
3. `gold_test_patch` raw text 当前没有明显收益。
4. 把所有 raw gold text 一股脑加进去会明显伤泛化。

建议主线不要用 `K` 作为主结论。

更合理的主线是：

- 主模型：`I_LightGBM_Dense_AF`
- 可选增强：`Dense + AF + gold_fail_to_pass_tfidf_dim16`
- 对照：`K_LightGBM_Dense_Full` 作为“全量 raw gold text 反而过拟合”的消融。

## 16. 为什么 row 数会和旧实验差很多

当前测试集：

- `83169` prefix rows
- `1458` trajectories
- 平均每条轨迹约 `57` 个 prefix row

旧实验如果 row 数少很多，不一定是 instance 或 trajectory 少，而可能是：

- 每条轨迹平均步骤更短。
- 旧流程过滤了更多中间 prefix。
- 使用的数据源不是同一个 parquet。
- 是否按 trajectory / instance / model_holdout 划分不同。

所以比较数据规模时，不应该只看 rows。

至少要同时看：

- prefix rows
- trajectories
- instances
- avg prefixes per trajectory
- heldout models

当前这个实验的核心规模应该按 `1458 trajectories / 489 instances / 3 heldout models` 理解。

## 17. 推荐报告阅读顺序

优先看：

1. `model_ranking_report_like_ref_calibrated/final_step_prefix_model_leaderboard.csv`
   - 每条轨迹最后一步算一次。
   - 最适合作为最终预测能力指标。

2. `model_ranking_report_like_ref_calibrated/report.txt`
   - 看不同阈值下的早停、误杀、误判成功、排名变化、节省步数。

3. `model_ranking_report_like_ref_calibrated/final_step_probability_bins.csv`
   - 看概率分桶是否校准，例如 0.8-0.9 桶真实成功率是否也高。

4. `evaluation_report.txt`
   - 看 prefix-level 过程指标、step bucket、特征贡献、top features。

5. `gold_text_tfidf_* / summary.txt`
   - 看 gold raw text TF-IDF 的消融。

## 18. 当前最适合写进结论的话

可以这样表述：

> We evaluate two complementary granularities. Prefix-level metrics measure whether the predictor is informative at every intermediate step, while final-step trajectory-level metrics count each trajectory once and better reflect final outcome prediction. In the model-holdout setting, all heldout test model identities are mapped to `__MISSING__`, so the predictor cannot use the true identity of the three test models. Under the trajectory-level final-step metric, structured gold-answer features combined with process action/feedback features perform best. Adding all raw gold-answer TF-IDF blocks hurts generalization, while a small `FAIL_TO_PASS` TF-IDF block gives a minor positive signal.

中文版本：

> 当前实验同时报告 prefix 级和 trajectory 级指标。prefix 级指标衡量模型在每个中间步骤是否有判断能力；trajectory 级 final-step 指标每条轨迹只算一次，更适合作为最终结果预测能力。model-holdout 测试时，三个测试模型的真实 model_id 都被映射成 `__MISSING__`，模型看不到测试模型身份。结果上，结构化 gold answer 特征加 action/feedback 过程文本最稳；全量 gold raw text TF-IDF 会带来噪声和过拟合；单独加入 `FAIL_TO_PASS` 测试名 TF-IDF 有小幅正收益。
