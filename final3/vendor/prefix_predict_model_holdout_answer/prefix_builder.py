"""
Prefix 表构建器。

将每条轨迹展开为 0..n_steps 个 prefix 样本，
并为每个 prefix 计算全部手工特征（A~H 组 + J 组；I 组为 feature_engineer 中 TF-IDF）。
"""
from __future__ import annotations

import glob
import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

import config
from action_classifier import classify_action
from observation_parser import parse_observation
from step_builder import (
    build_preamble_info,
    _parse_messages,
    _is_action_message,
    _load_and_deduplicate_trajectories,
    _apply_max_trajectories_limit,
)
from utils import get_logger, timer

logger = get_logger("prefix_builder")


def _iter_tool_parquet_files(input_dir: str) -> list[str]:
    p = Path(input_dir)
    if p.is_file() and p.name.startswith("tool"):
        return [str(p)]
    files = sorted(glob.glob(str(p / "tool-*.parquet")))
    if not files:
        files = sorted(glob.glob(str(p / "*.parquet")))
        files = [f for f in files if Path(f).name.startswith("tool")]
    return files


def _similarity(a: str, b: str) -> float:
    """简单的 Jaccard 相似度用于判断 action 是否重复。"""
    if not a or not b:
        return 0.0
    sa, sb = set(a.split()), set(b.split())
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def build_prefix_samples_for_trajectory(
    row: pd.Series,
    step_df_for_traj: Optional[pd.DataFrame] = None,
) -> list[dict]:
    """
    为一条轨迹构建所有 prefix 样本。

    如果提供了 step_df_for_traj，直接使用；否则从 row 重建。
    """
    from step_builder import rebuild_steps_for_trajectory

    traj_id = row.get("traj_id", "")
    instance_id = row.get("instance_id", "")
    resolved = int(bool(row.get("resolved", False)))
    model = row.get("model", "")

    # ── 获取 preamble ──
    preamble = build_preamble_info(row)
    task_prompt_text = preamble["task_prompt_text"]

    # ── 获取 steps ──
    if step_df_for_traj is not None and len(step_df_for_traj) > 0:
        steps = step_df_for_traj.sort_values("step_idx").to_dict("records")
    else:
        steps = rebuild_steps_for_trajectory(row)

    n_steps = len(steps)
    sample_weight = 1.0 / (n_steps + 1)

    prefix_samples = []

    # ── prefix_step_idx = 0（只有 preamble）──
    p0 = _build_prefix_features(
        traj_id=traj_id,
        instance_id=instance_id,
        resolved=resolved,
        model=model,
        task_prompt_text=task_prompt_text,
        steps=[],
        prefix_step_idx=0,
        n_steps_total=n_steps,
        sample_weight=sample_weight,
    )
    prefix_samples.append(p0)

    # ── prefix_step_idx = 1..n_steps ──
    for t in range(1, n_steps + 1):
        p = _build_prefix_features(
            traj_id=traj_id,
            instance_id=instance_id,
            resolved=resolved,
            model=model,
            task_prompt_text=task_prompt_text,
            steps=steps[:t],
            prefix_step_idx=t,
            n_steps_total=n_steps,
            sample_weight=sample_weight,
        )
        prefix_samples.append(p)

    return prefix_samples


def _build_prefix_features(
    traj_id: str,
    instance_id: str,
    resolved: int,
    model: str,
    task_prompt_text: str,
    steps: list[dict],
    prefix_step_idx: int,
    n_steps_total: int,
    sample_weight: float,
) -> dict:
    """为一个 prefix 构建全部特征（A~H 组）。"""

    prefix_id = f"{traj_id}::p{prefix_step_idx}"
    t = prefix_step_idx
    n = len(steps)

    # ══════════════════════════════════════════════
    # A 组: Prefix Progress / 元信息
    # ══════════════════════════════════════════════
    prefix_action_text_parts = []
    prefix_feedback_text_parts = []
    prefix_thought_text_parts = []
    prefix_assistant_content_parts = []
    tool_calls_total = 0
    tool_msg_total = 0
    obs_total = 0
    distinct_tools = set()
    thought_step_count = 0
    content_step_count = 0
    thought_eq_content_count = 0
    thought_action_overlap_vals = []
    content_action_overlap_vals = []

    for s in steps:
        prefix_action_text_parts.append(s.get("action_text", ""))
        prefix_feedback_text_parts.append(s.get("combined_feedback_text", ""))
        prefix_thought_text_parts.append(s.get("thought_text", ""))
        prefix_assistant_content_parts.append(s.get("assistant_content_text", ""))
        tool_calls_total += s.get("tool_count", 0)
        if s.get("has_tool_output"):
            tool_msg_total += 1
        if s.get("has_observation"):
            obs_total += 1
        for tn in (s.get("tool_names") or []):
            distinct_tools.add(tn)
        if s.get("has_thought_text"):
            thought_step_count += 1
        if s.get("has_assistant_content_text"):
            content_step_count += 1
        if s.get("thought_equals_content"):
            thought_eq_content_count += 1
        tao = s.get("thought_action_overlap_ratio", 0.0)
        cao = s.get("content_action_overlap_ratio", 0.0)
        if tao is not None:
            thought_action_overlap_vals.append(tao)
        if cao is not None:
            content_action_overlap_vals.append(cao)

    prefix_action_text = "\n".join(prefix_action_text_parts)
    prefix_feedback_text = "\n".join(prefix_feedback_text_parts)
    prefix_thought_text = "\n".join(prefix_thought_text_parts)
    prefix_assistant_content_text = "\n".join(prefix_assistant_content_parts)

    feat = {
        # 标识
        "prefix_id": prefix_id,
        "traj_id": traj_id,
        "instance_id": instance_id,
        # 关键修复：group_id 必须与 trajectory 对齐，避免同一 instance 下多轨迹混组。
        "group_id": traj_id,
        "resolved": resolved,
        "model": model,
        "prefix_step_idx": t,
        "n_steps_total_for_weighting": n_steps_total,
        "sample_weight": sample_weight,

        # 文本字段（用于后续 TF-IDF）
        "task_prompt_text": task_prompt_text,
        "prefix_action_text": prefix_action_text,
        "prefix_feedback_text": prefix_feedback_text,
        "last_action_text": steps[-1].get("action_text", "") if steps else "",
        "last_feedback_text": steps[-1].get("combined_feedback_text", "") if steps else "",
        # 新增：thought / assistant_content 文本字段
        "prefix_thought_text": prefix_thought_text,
        "last_thought_text": steps[-1].get("thought_text", "") if steps else "",
        "prefix_assistant_content_text": prefix_assistant_content_text,
        "last_assistant_content_text": steps[-1].get("assistant_content_text", "") if steps else "",

        # A 组
        "steps_observed_so_far": n,
        "actions_so_far": n,
        "observations_so_far": obs_total,
        "tool_messages_so_far": tool_msg_total,
        "tool_calls_so_far": tool_calls_total,
        "distinct_tools_so_far": len(distinct_tools),
        "prefix_action_chars": len(prefix_action_text),
        "prefix_feedback_chars": len(prefix_feedback_text),
        "task_prompt_chars": len(task_prompt_text),
        "has_any_action": n > 0,
        "model_id": model,

        # 标签
        "label": resolved,
    }

    # ══════════════════════════════════════════════
    # B 组: Last-Step 特征
    # ══════════════════════════════════════════════
    if steps:
        last = steps[-1]
        feat.update({
            "last_step_action_major_type": last.get("action_major_type", "none"),
            "last_step_action_primary_subtype": last.get("action_primary_subtype", "none"),
            "last_step_subtypes": last.get("action_subtypes", []),
            "last_step_tool_count": last.get("tool_count", 0),
            "last_step_has_tool_output": last.get("has_tool_output", False),
            "last_step_has_observation": last.get("has_observation", False),
            "last_step_action_chars": last.get("action_char_len", 0),
            "last_step_feedback_chars": last.get("feedback_char_len", 0),
            "last_step_tool_error_seen": last.get("tool_error_seen_this_step", False),
            "last_step_traceback_seen": last.get("traceback_seen_this_step", False),
            "last_step_test_fail_seen": last.get("test_fail_seen_this_step", False),
            "last_step_test_pass_seen": last.get("test_pass_seen_this_step", False),
            "last_step_fail_count": last.get("last_fail_count_this_step"),
        })
    else:
        feat.update({
            "last_step_action_major_type": "none",
            "last_step_action_primary_subtype": "none",
            "last_step_subtypes": [],
            "last_step_tool_count": 0,
            "last_step_has_tool_output": False,
            "last_step_has_observation": False,
            "last_step_action_chars": 0,
            "last_step_feedback_chars": 0,
            "last_step_tool_error_seen": False,
            "last_step_traceback_seen": False,
            "last_step_test_fail_seen": False,
            "last_step_test_pass_seen": False,
            "last_step_fail_count": None,
        })

    # ══════════════════════════════════════════════
    # C 组: 累计动作计数
    # ══════════════════════════════════════════════
    subtype_counts = {st: 0 for st in config.ALL_SUBTYPES}
    bash_calls = 0
    editor_calls = 0

    for s in steps:
        primary = s.get("action_primary_subtype", "")
        if primary in subtype_counts:
            subtype_counts[primary] += 1
        action_text = s.get("action_text", "")
        if action_text.strip().startswith("str_replace_editor"):
            editor_calls += 1
        else:
            bash_calls += 1

    feat.update({
        "read_view_so_far": subtype_counts["read_view"],
        "read_search_so_far": subtype_counts["read_search"],
        "edit_create_so_far": subtype_counts["edit_create"],
        "edit_replace_so_far": subtype_counts["edit_replace"],
        "edit_insert_so_far": subtype_counts["edit_insert"],
        "edit_undo_so_far": subtype_counts["edit_undo"],
        "edits_so_far": sum(subtype_counts[k] for k in
                           ["edit_create", "edit_replace", "edit_insert", "edit_undo"]),
        "tests_so_far": subtype_counts["test"],
        "run_python_so_far": subtype_counts["run_python"],
        "run_cli_so_far": subtype_counts["run_cli"],
        "git_ops_so_far": subtype_counts["git"],
        "cleanup_so_far": subtype_counts["cleanup"],
        "submit_so_far": subtype_counts["submit"],
        "bash_calls_so_far": bash_calls,
        "editor_calls_so_far": editor_calls,
    })

    # ══════════════════════════════════════════════
    # D 组: Milestone / 首次发生位置
    # ══════════════════════════════════════════════
    first_edit = first_test = first_run_python = first_submit = None
    first_error = first_traceback = first_read = None

    for s in steps:
        idx = s["step_idx"]
        major = s.get("action_major_type", "")
        primary = s.get("action_primary_subtype", "")

        if major == "edit" and first_edit is None:
            first_edit = idx
        if primary == "test" and first_test is None:
            first_test = idx
        if primary == "run_python" and first_run_python is None:
            first_run_python = idx
        if primary == "submit" and first_submit is None:
            first_submit = idx
        if major == "read" and first_read is None:
            first_read = idx
        if s.get("tool_error_seen_this_step") and first_error is None:
            first_error = idx
        if s.get("traceback_seen_this_step") and first_traceback is None:
            first_traceback = idx

    feat.update({
        "first_edit_step": first_edit,
        "first_test_step": first_test,
        "first_run_python_step": first_run_python,
        "first_submit_step": first_submit,
        "first_error_step": first_error,
        "first_traceback_step": first_traceback,
        "first_read_step": first_read,
        "first_edit_seen": first_edit is not None,
        "first_test_seen": first_test is not None,
        "first_submit_seen": first_submit is not None,
        "first_error_seen": first_error is not None,
        "first_traceback_seen": first_traceback is not None,
    })

    # ══════════════════════════════════════════════
    # E 组: Recency / 距离上次事件
    # ══════════════════════════════════════════════
    last_edit = last_test = last_submit = last_error = last_traceback = last_read = None

    for s in steps:
        idx = s["step_idx"]
        major = s.get("action_major_type", "")
        primary = s.get("action_primary_subtype", "")
        if major == "edit":
            last_edit = idx
        if primary == "test":
            last_test = idx
        if primary == "submit":
            last_submit = idx
        if major == "read":
            last_read = idx
        if s.get("tool_error_seen_this_step"):
            last_error = idx
        if s.get("traceback_seen_this_step"):
            last_traceback = idx

    feat.update({
        "steps_since_last_edit": (t - last_edit) if last_edit is not None else None,
        "steps_since_last_test": (t - last_test) if last_test is not None else None,
        "steps_since_last_submit": (t - last_submit) if last_submit is not None else None,
        "steps_since_last_error": (t - last_error) if last_error is not None else None,
        "steps_since_last_traceback": (t - last_traceback) if last_traceback is not None else None,
        "steps_since_last_read": (t - last_read) if last_read is not None else None,
    })

    # ══════════════════════════════════════════════
    # F 组: 比例与节奏
    # ══════════════════════════════════════════════
    reads = feat["read_view_so_far"] + feat["read_search_so_far"]
    edits = feat["edits_so_far"]
    tests = feat["tests_so_far"]
    actions = max(n, 1)

    feat.update({
        "read_to_edit_ratio": reads / max(edits, 1),
        "edit_to_test_ratio": edits / max(tests, 1),
        "bash_to_editor_ratio": bash_calls / max(editor_calls, 1),
        "error_per_action_ratio": sum(
            1 for s in steps if s.get("tool_error_seen_this_step")
        ) / actions,
        "submit_per_action_ratio": feat["submit_so_far"] / actions,
        "feedback_chars_per_action": feat["prefix_feedback_chars"] / actions,
        "action_chars_per_step": feat["prefix_action_chars"] / actions,
        "distinct_tools_per_step": feat["distinct_tools_so_far"] / actions,
    })

    # ══════════════════════════════════════════════
    # G 组: Observation 错误与测试状态
    # ══════════════════════════════════════════════
    traceback_seen = any(s.get("traceback_seen_this_step") for s in steps)
    tool_error_seen = any(s.get("tool_error_seen_this_step") for s in steps)
    test_fail_seen = any(s.get("test_fail_seen_this_step") for s in steps)
    test_pass_seen = any(s.get("test_pass_seen_this_step") for s in steps)

    # 从所有 feedback 文本提取细化错误
    all_feedback = prefix_feedback_text
    all_sig = parse_observation(all_feedback)

    # fail count 追踪
    fail_counts_history = []
    for s in steps:
        fc = s.get("last_fail_count_this_step")
        if fc is not None:
            fail_counts_history.append(fc)

    best_fail = min(fail_counts_history) if fail_counts_history else None
    last_fail = fail_counts_history[-1] if fail_counts_history else None
    delta = None
    if len(fail_counts_history) >= 2:
        delta = fail_counts_history[-1] - fail_counts_history[-2]
    test_improving = any(
        fail_counts_history[i] < fail_counts_history[i - 1]
        for i in range(1, len(fail_counts_history))
    ) if len(fail_counts_history) >= 2 else False

    feat.update({
        "traceback_seen": traceback_seen,
        "tool_error_seen": tool_error_seen,
        "assertion_error_seen": all_sig.assertion_error,
        "type_error_seen": all_sig.type_error,
        "value_error_seen": all_sig.value_error,
        "syntax_error_seen": all_sig.syntax_error,
        "import_error_seen": all_sig.import_error,
        "file_not_found_seen": all_sig.file_not_found,
        "timeout_seen": all_sig.timeout,
        "permission_error_seen": all_sig.permission_error,
        "test_fail_seen": test_fail_seen,
        "test_pass_seen": test_pass_seen,
        "all_tests_passed_seen": all_sig.all_tests_passed,
        "last_fail_count": last_fail,
        "best_fail_count_so_far": best_fail,
        "fail_count_delta_from_prev_test": delta,
        "test_improving_seen": test_improving,
    })

    # ══════════════════════════════════════════════
    # H 组: 循环 / 迷茫 / 风险特征
    # ══════════════════════════════════════════════
    repeated_action = False
    repeated_search = False
    repeated_view = False
    edit_failed_seen = False
    submit_without_test = False
    premature_submit = False
    multi_submit = False
    submit_then_edit = False
    test_after_submit = False

    # 连续未 edit 和连续 read
    long_no_edit_streak = 0
    long_read_streak = 0
    cur_no_edit = 0
    cur_read = 0
    looping_read = False

    has_submitted = False
    has_tested_before_submit = False

    for i, s in enumerate(steps):
        major = s.get("action_major_type", "")
        primary = s.get("action_primary_subtype", "")

        # 连续相同 action
        if i > 0:
            prev = steps[i - 1]
            sim = _similarity(s.get("action_text", ""), prev.get("action_text", ""))
            if sim > 0.85:
                repeated_action = True
            if primary == "read_search" and prev.get("action_primary_subtype") == "read_search" and sim > 0.85:
                repeated_search = True
            if primary == "read_view" and prev.get("action_primary_subtype") == "read_view" and sim > 0.85:
                repeated_view = True

        # edit failed
        feedback = s.get("combined_feedback_text", "")
        if "pattern not found" in feedback.lower() or \
           "not unique in the file" in feedback.lower() or \
           "did not appear verbatim" in feedback.lower() or \
           "replacement was not performed" in feedback.lower():
            edit_failed_seen = True

        # no-edit streak
        if major == "edit":
            cur_no_edit = 0
        else:
            cur_no_edit += 1
        long_no_edit_streak = max(long_no_edit_streak, cur_no_edit)

        # read streak
        if major == "read":
            cur_read += 1
        else:
            cur_read = 0
        long_read_streak = max(long_read_streak, cur_read)

        if long_read_streak >= 5 and edits == 0:
            looping_read = True

        # submit / test 逻辑
        if primary == "test":
            has_tested_before_submit = True if not has_submitted else has_tested_before_submit
            if has_submitted:
                test_after_submit = True

        if primary == "submit":
            if not has_submitted:
                # 首次 submit
                if not has_tested_before_submit:
                    submit_without_test = True
                if i < 3:  # 前 3 步就提交
                    premature_submit = True
            else:
                multi_submit = True
            has_submitted = True

        if has_submitted and major == "edit":
            submit_then_edit = True

    feat.update({
        "repeated_same_action_consecutive": repeated_action,
        "repeated_same_search_consecutive": repeated_search,
        "repeated_same_view_consecutive": repeated_view,
        "looping_read_seen": looping_read,
        "long_no_edit_streak": long_no_edit_streak,
        "long_read_streak": long_read_streak,
        "edit_failed_seen": edit_failed_seen,
        "submit_without_test_seen": submit_without_test,
        "premature_submit_seen": premature_submit,
        "multi_submit_seen": multi_submit,
        "submit_then_edit_again_seen": submit_then_edit,
        "test_after_submit_seen": test_after_submit,
    })

    # ══════════════════════════════════════════════
    # J 组: Cognitive / Narrative 统计特征
    # ══════════════════════════════════════════════
    actions_denom = max(n, 1)
    avg_thought_overlap = (
        sum(thought_action_overlap_vals) / len(thought_action_overlap_vals)
        if thought_action_overlap_vals else 0.0
    )
    avg_content_overlap = (
        sum(content_action_overlap_vals) / len(content_action_overlap_vals)
        if content_action_overlap_vals else 0.0
    )

    feat.update({
        "thought_steps_so_far": thought_step_count,
        "thought_density": thought_step_count / actions_denom,
        "prefix_thought_chars": len(prefix_thought_text),
        "avg_thought_chars_per_step": len(prefix_thought_text) / actions_denom,
        "last_thought_chars": len(steps[-1].get("thought_text", "")) if steps else 0,
        "assistant_content_steps_so_far": content_step_count,
        "prefix_assistant_content_chars": len(prefix_assistant_content_text),
        "avg_assistant_content_chars_per_step": len(prefix_assistant_content_text) / actions_denom,
        "last_assistant_content_chars": len(steps[-1].get("assistant_content_text", "")) if steps else 0,
        "thought_equals_content_rate": thought_eq_content_count / actions_denom if n > 0 else 0.0,
        "thought_action_overlap_avg": avg_thought_overlap,
        "content_action_overlap_avg": avg_content_overlap,
    })

    return feat


def build_prefix_table(
    input_dir: Optional[str] = None,
    *,
    max_trajectories: Optional[int] = None,
    sample_trajectories_seed: Optional[int] = None,
) -> pd.DataFrame:
    """
    读取 tool parquet 文件，为每条轨迹构建所有 prefix 样本。

    max_trajectories / sample_trajectories_seed:
        与 build_step_table 一致（去重后截断；可选种子打乱再截断）。
    """
    input_dir = input_dir or config.PARQUET_INPUT_DIR
    traj_df = _load_and_deduplicate_trajectories(input_dir=input_dir, dedup_seed=config.SPLIT_SEED)
    traj_df = _apply_max_trajectories_limit(
        traj_df, max_trajectories, sample_trajectories_seed=sample_trajectories_seed
    )
    logger.info(f"Building prefix table from deduplicated trajectories: {len(traj_df)}")

    # 分块写盘目录与配置
    part_dir = config.DATA_DIR / "prefix_parts"
    part_dir.mkdir(parents=True, exist_ok=True)
    chunk_size = max(int(config.PREFIX_CHUNK_SIZE), 1)
    part_paths: list[Path] = []

    buffer: list[dict] = []
    total_trajs = len(traj_df)
    total_prefix_samples = 0

    def flush_buffer():
        nonlocal buffer, part_paths, total_prefix_samples
        if not buffer:
            return
        df_chunk = pd.DataFrame(buffer)
        if df_chunk.empty:
            buffer = []
            return
        part_idx = len(part_paths)
        part_path = part_dir / f"prefix_table.part-{part_idx:04d}.parquet"
        df_chunk.to_parquet(part_path, index=False)
        part_paths.append(part_path)
        total_prefix_samples += len(df_chunk)
        logger.info(
            f"Flushed {len(df_chunk)} prefix samples to {part_path} "
            f"(total so far: {total_prefix_samples})"
        )
        buffer = []

    with timer(logger, "Building prefix samples from deduplicated trajectories"):
        for _, row in traj_df.iterrows():
            try:
                samples = build_prefix_samples_for_trajectory(row)
                buffer.extend(samples)
                # 分块写盘，避免一次性占用过多内存
                if len(buffer) >= chunk_size:
                    flush_buffer()
            except Exception as e:
                logger.warning(f"  Failed for traj {row.get('traj_id')}: {e}")

    # 将剩余 buffer 落盘
    flush_buffer()

    logger.info(f"Total trajectories processed: {total_trajs}")
    logger.info(f"Total prefix samples (streamed): {total_prefix_samples}")

    # 将所有分块重新读入，拼成完整 prefix_df，保持对上层接口不变
    if part_paths:
        dfs = [pd.read_parquet(p) for p in sorted(part_paths)]
        prefix_df = pd.concat(dfs, ignore_index=True)
    else:
        # 极端小数据场景：没有触发 flush，仅在内存中
        prefix_df = pd.DataFrame(buffer)
        total_prefix_samples = len(prefix_df)

    logger.info(f"Prefix table shape: {prefix_df.shape}")

    # 统计
    if len(prefix_df) > 0:
        logger.info(f"Label distribution:\n{prefix_df['label'].value_counts().to_string()}")
        logger.info(f"Prefix step distribution:\n{prefix_df['prefix_step_idx'].describe().to_string()}")

    return prefix_df


if __name__ == "__main__":
    prefix_df = build_prefix_table()
    prefix_df.to_parquet(config.PREFIX_TABLE_PATH, index=False)
    logger.info(f"Saved prefix table to {config.PREFIX_TABLE_PATH}")
