# Feature Dictionary

## A description: Prefix Progress / description

| description | description | description |
|------|------|------|
| prefix_step_idx | int | description |
| steps_observed_so_far | int | description prefix_step_idx |
| actions_so_far | int | description action description |
| observations_so_far | int | description observation description |
| tool_messages_so_far | int | description tool description |
| tool_calls_so_far | int | description |
| distinct_tools_so_far | int | description |
| prefix_action_chars | int | description action description |
| prefix_feedback_chars | int | description feedback description |
| task_prompt_chars | int | description |
| has_any_action | bool | description step>=1 |
| model_id | categorical | description ID(description) |

## B description: Last-Step description

| description | description | description |
|------|------|------|
| last_step_action_major_type | categorical | description |
| last_step_action_primary_subtype | categorical | description |
| last_step_tool_count | int | description |
| last_step_has_tool_output | bool | description tool description |
| last_step_has_observation | bool | description observation |
| last_step_action_chars | int | description action description |
| last_step_feedback_chars | int | description |
| last_step_tool_error_seen | bool | description |
| last_step_traceback_seen | bool | description traceback |
| last_step_test_fail_seen | bool | description |
| last_step_test_pass_seen | bool | description |
| last_step_fail_count | int/null | description fail description |

## C description: description

| description | description | description |
|------|------|------|
| read_view_so_far | int | description |
| read_search_so_far | int | description |
| edit_create_so_far | int | create description |
| edit_replace_so_far | int | str_replace description |
| edit_insert_so_far | int | insert description |
| edit_undo_so_far | int | undo_edit description |
| edits_so_far | int | description edit description |
| tests_so_far | int | description |
| run_python_so_far | int | description Python description |
| run_cli_so_far | int | description CLI description |
| git_ops_so_far | int | Git description |
| cleanup_so_far | int | description |
| submit_so_far | int | description |
| bash_calls_so_far | int | bash description |
| editor_calls_so_far | int | str_replace_editor description |

## D description: Milestone / description

| description | description | description |
|------|------|------|
| first_edit_step | int/null | description edit description |
| first_test_step | int/null | description test description |
| first_run_python_step | int/null | description Python description |
| first_submit_step | int/null | description submit description |
| first_error_step | int/null | description |
| first_traceback_step | int/null | description traceback description |
| first_read_step | int/null | description read description |

## E description: Recency / description

| description | description | description |
|------|------|------|
| steps_since_last_edit | int/null | description edit description |
| steps_since_last_test | int/null | description test description |
| steps_since_last_submit | int/null | description submit description |
| steps_since_last_error | int/null | description |
| steps_since_last_traceback | int/null | description traceback description |
| steps_since_last_read | int/null | description read description |

## F description: description

| description | description | description |
|------|------|------|
| read_to_edit_ratio | float | (read_view + read_search) / max(edits,1) |
| edit_to_test_ratio | float | edits / max(tests,1) |
| bash_to_editor_ratio | float | bash_calls / max(editor_calls,1) |
| error_per_action_ratio | float | tool_errors / max(actions,1) |
| submit_per_action_ratio | float | submit / max(actions,1) |
| feedback_chars_per_action | float | feedback_chars / max(actions,1) |
| action_chars_per_step | float | action_chars / max(actions,1) |
| distinct_tools_per_step | float | distinct_tools / max(actions,1) |

## G description: Observation description

| description | description | description |
|------|------|------|
| traceback_seen | bool | description traceback |
| tool_error_seen | bool | description |
| assertion_error_seen | bool | description AssertionError |
| type_error_seen | bool | description TypeError |
| value_error_seen | bool | description ValueError |
| syntax_error_seen | bool | description SyntaxError |
| import_error_seen | bool | description ImportError |
| file_not_found_seen | bool | description file not found |
| timeout_seen | bool | description timeout |
| permission_error_seen | bool | description |
| test_fail_seen | bool | description |
| test_pass_seen | bool | description |
| all_tests_passed_seen | bool | description |
| last_fail_count | int/null | description |
| best_fail_count_so_far | int/null | description fail description |
| fail_count_delta_from_prev_test | int/null | fail description |
| test_improving_seen | bool | description fail description |

## H description: description / description / description

| description | description | description |
|------|------|------|
| repeated_same_action_consecutive | bool | description action description |
| repeated_same_search_consecutive | bool | description |
| repeated_same_view_consecutive | bool | description |
| looping_read_seen | bool | description read description edit |
| long_no_edit_streak | int | description edit description |
| long_read_streak | int | description read description |
| edit_failed_seen | bool | description |
| submit_without_test_seen | bool | description |
| premature_submit_seen | bool | description |
| multi_submit_seen | bool | description submit |
| submit_then_edit_again_seen | bool | submit description edit |
| test_after_submit_seen | bool | submit description |

## I description: TF-IDF description

| description | description | description |
|--------|----------|----------|
| tfidf_task_prompt | description | ngram=(1,2), min_df=5, max_features=30000 |
| tfidf_prefix_action | description action description | description |
| tfidf_prefix_feedback | description feedback description | description |
| tfidf_last_action | description action | description |
| tfidf_last_feedback | description feedback | description |
