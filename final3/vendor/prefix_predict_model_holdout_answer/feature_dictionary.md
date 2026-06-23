# Feature Dictionary

## A 组: Prefix Progress / 元信息

| 特征 | 类型 | 说明 |
|------|------|------|
| prefix_step_idx | int | 当前前缀步数 |
| steps_observed_so_far | int | 等于 prefix_step_idx |
| actions_so_far | int | 当前已发生 action 数 |
| observations_so_far | int | 已聚合 observation 数 |
| tool_messages_so_far | int | 已聚合 tool 消息数 |
| tool_calls_so_far | int | 已调用工具次数 |
| distinct_tools_so_far | int | 已使用工具种类数 |
| prefix_action_chars | int | 前缀 action 文本总长度 |
| prefix_feedback_chars | int | 前缀 feedback 文本总长度 |
| task_prompt_chars | int | 初始任务文本长度 |
| has_any_action | bool | 是否已进入 step>=1 |
| model_id | categorical | 模型 ID（可选） |

## B 组: Last-Step 特征

| 特征 | 类型 | 说明 |
|------|------|------|
| last_step_action_major_type | categorical | 最后一步一级动作类型 |
| last_step_action_primary_subtype | categorical | 最后一步主子类型 |
| last_step_tool_count | int | 最后一步工具调用数 |
| last_step_has_tool_output | bool | 最后一步是否有 tool 输出 |
| last_step_has_observation | bool | 最后一步是否有 observation |
| last_step_action_chars | int | 最后一步 action 文本长度 |
| last_step_feedback_chars | int | 最后一步反馈文本长度 |
| last_step_tool_error_seen | bool | 最后一步是否工具报错 |
| last_step_traceback_seen | bool | 最后一步是否 traceback |
| last_step_test_fail_seen | bool | 最后一步是否测试失败 |
| last_step_test_pass_seen | bool | 最后一步是否测试通过 |
| last_step_fail_count | int/null | 最后一步提取出的 fail 数 |

## C 组: 累计动作计数

| 特征 | 类型 | 说明 |
|------|------|------|
| read_view_so_far | int | 文件查看次数 |
| read_search_so_far | int | 搜索次数 |
| edit_create_so_far | int | create 次数 |
| edit_replace_so_far | int | str_replace 次数 |
| edit_insert_so_far | int | insert 次数 |
| edit_undo_so_far | int | undo_edit 次数 |
| edits_so_far | int | 所有 edit 合计 |
| tests_so_far | int | 测试运行次数 |
| run_python_so_far | int | 非测试 Python 执行次数 |
| run_cli_so_far | int | 普通 CLI 执行次数 |
| git_ops_so_far | int | Git 操作次数 |
| cleanup_so_far | int | 清理次数 |
| submit_so_far | int | 提交次数 |
| bash_calls_so_far | int | bash 调用总数 |
| editor_calls_so_far | int | str_replace_editor 调用总数 |

## D 组: Milestone / 首次发生位置

| 特征 | 类型 | 说明 |
|------|------|------|
| first_edit_step | int/null | 第一次 edit 发生步 |
| first_test_step | int/null | 第一次 test 发生步 |
| first_run_python_step | int/null | 第一次非测试 Python 执行步 |
| first_submit_step | int/null | 第一次 submit 步 |
| first_error_step | int/null | 第一次工具错误步 |
| first_traceback_step | int/null | 第一次 traceback 步 |
| first_read_step | int/null | 第一次 read 步 |

## E 组: Recency / 距离上次事件

| 特征 | 类型 | 说明 |
|------|------|------|
| steps_since_last_edit | int/null | 距离上次 edit 的步数 |
| steps_since_last_test | int/null | 距离上次 test 的步数 |
| steps_since_last_submit | int/null | 距离上次 submit 的步数 |
| steps_since_last_error | int/null | 距离上次错误的步数 |
| steps_since_last_traceback | int/null | 距离上次 traceback 的步数 |
| steps_since_last_read | int/null | 距离上次 read 的步数 |

## F 组: 比例与节奏

| 特征 | 类型 | 说明 |
|------|------|------|
| read_to_edit_ratio | float | (read_view + read_search) / max(edits,1) |
| edit_to_test_ratio | float | edits / max(tests,1) |
| bash_to_editor_ratio | float | bash_calls / max(editor_calls,1) |
| error_per_action_ratio | float | tool_errors / max(actions,1) |
| submit_per_action_ratio | float | submit / max(actions,1) |
| feedback_chars_per_action | float | feedback_chars / max(actions,1) |
| action_chars_per_step | float | action_chars / max(actions,1) |
| distinct_tools_per_step | float | distinct_tools / max(actions,1) |

## G 组: Observation 错误与测试状态

| 特征 | 类型 | 说明 |
|------|------|------|
| traceback_seen | bool | 前缀内任意 traceback |
| tool_error_seen | bool | 前缀内任意工具错误 |
| assertion_error_seen | bool | 出现 AssertionError |
| type_error_seen | bool | 出现 TypeError |
| value_error_seen | bool | 出现 ValueError |
| syntax_error_seen | bool | 出现 SyntaxError |
| import_error_seen | bool | 出现 ImportError |
| file_not_found_seen | bool | 出现 file not found |
| timeout_seen | bool | 出现 timeout |
| permission_error_seen | bool | 出现权限错误 |
| test_fail_seen | bool | 出现测试失败 |
| test_pass_seen | bool | 出现测试通过 |
| all_tests_passed_seen | bool | 出现强通过信号 |
| last_fail_count | int/null | 最近一次测试失败数 |
| best_fail_count_so_far | int/null | 当前最小 fail 数 |
| fail_count_delta_from_prev_test | int/null | fail 数变化 |
| test_improving_seen | bool | 是否出现 fail 数下降 |

## H 组: 循环 / 迷茫 / 风险

| 特征 | 类型 | 说明 |
|------|------|------|
| repeated_same_action_consecutive | bool | 连续两步 action 高度相似 |
| repeated_same_search_consecutive | bool | 连续重复搜索 |
| repeated_same_view_consecutive | bool | 连续重复看同一文件 |
| looping_read_seen | bool | 长时间 read 无 edit |
| long_no_edit_streak | int | 当前连续未 edit 步数 |
| long_read_streak | int | 当前连续 read 步数 |
| edit_failed_seen | bool | 出现编辑失败 |
| submit_without_test_seen | bool | 在未测试前就提交 |
| premature_submit_seen | bool | 过早提交 |
| multi_submit_seen | bool | 多次 submit |
| submit_then_edit_again_seen | bool | submit 后又继续 edit |
| test_after_submit_seen | bool | submit 后才开始测试 |

## I 组: TF-IDF 文本特征

| 特征组 | 文本来源 | 默认配置 |
|--------|----------|----------|
| tfidf_task_prompt | 初始任务描述 | ngram=(1,2), min_df=5, max_features=30000 |
| tfidf_prefix_action | 前缀所有 action 拼接 | 同上 |
| tfidf_prefix_feedback | 前缀所有 feedback 拼接 | 同上 |
| tfidf_last_action | 最后一步 action | 同上 |
| tfidf_last_feedback | 最后一步 feedback | 同上 |
