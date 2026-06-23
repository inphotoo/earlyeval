"""
Step 重建器。

按照文档 §3 的规则，将每条轨迹的 messages 列表重建为 step 序列。

Step 边界定义：
- preamble (step_idx=0): 第一条 assistant action 之前的消息
- 正式 step (step_idx>=1): 每遇到一个 role=assistant + message_type=action 就开新 step
  该 step 包含 action message 本身 + 后续的 tool/observation 消息
"""
from __future__ import annotations

import glob
import json
import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

import config
from action_classifier import classify_action
from observation_parser import parse_observation
from utils import get_logger, timer

logger = get_logger("step_builder")


def _iter_tool_parquet_files(input_dir: str) -> list[str]:
    """
    遍历输入目录中的 parquet 文件。

    优先级：
    1. 如果 input_dir 是文件：直接返回
    2. 查找 tool-*.parquet 文件
    3. fallback: 查找所有 *.parquet 文件
    """
    p = Path(input_dir)
    if p.is_file():
        return [str(p)]
    files = sorted(glob.glob(str(p / "tool-*.parquet")))
    if not files:
        # fallback: 尝试所有 parquet
        files = sorted(glob.glob(str(p / "*.parquet")))
    return files


def _build_instance_dedup_key(df: pd.DataFrame) -> pd.Series:
    """
    构建用于全局去重的 key。

    规则：
    - instance_id 非空：使用 instance_id（跨文件全局唯一约束）
    - instance_id 为空：回退到 traj_id，避免把缺失值错误地合并到一条
    """
    if "instance_id" not in df.columns:
        raise KeyError("input data does not contain `instance_id` column")

    inst = df["instance_id"].fillna("").astype(str).str.strip()
    traj = df.get("traj_id", pd.Series(index=df.index, dtype="object")).fillna("").astype(str)
    missing_inst_mask = inst == ""
    fallback = "__MISSING_INSTANCE__::" + traj
    return pd.Series(np.where(missing_inst_mask, fallback, inst), index=df.index)


def _load_and_deduplicate_trajectories(
    input_dir: str,
    dedup_seed: Optional[int] = None,
) -> pd.DataFrame:
    """
    读取所有 tool parquet，并在最前面做全局 instance_id 去重。

    若同一个 instance_id 出现多条轨迹，则使用固定随机种子随机保留 1 条，
    保证可复现，且从源头消除 train/test instance 泄漏风险。
    """
    parquet_files = _iter_tool_parquet_files(input_dir)
    if not parquet_files:
        raise FileNotFoundError(f"No tool-*.parquet files found in {input_dir}")

    logger.info(f"Found {len(parquet_files)} tool parquet files")
    frames = []
    required_cols = {"messages", "traj_id"}
    for pf in parquet_files:
        df_part = pd.read_parquet(pf)
        # 仅接收“原始轨迹 parquet”，跳过 step/prefix 等中间产物 parquet
        if not required_cols.issubset(set(df_part.columns)):
            logger.info(
                f"  {Path(pf).name}: skipped (not a raw trajectory parquet, "
                f"missing columns: {sorted(required_cols - set(df_part.columns))})"
            )
            continue
        logger.info(f"  {Path(pf).name}: {len(df_part)} trajectories")
        frames.append(df_part)

    if not frames:
        raise FileNotFoundError(
            "No valid raw trajectory parquet found. "
            "Expected columns include at least: messages, traj_id."
        )

    raw_df = pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0].reset_index(drop=True)
    logger.info(f"Raw trajectories (before dedup): {len(raw_df)}")
    if len(raw_df) == 0:
        raise ValueError(
            "Raw trajectory dataframe is empty. "
            "Please check conversion output and input data path."
        )

    if os.environ.get("SWE_PREFIX_SKIP_INSTANCE_DEDUP", "").strip().lower() in ("1", "true", "yes"):
        logger.warning(
            "SWE_PREFIX_SKIP_INSTANCE_DEDUP is set: skipping global instance_id deduplication "
            "(multiple trajectories per instance are kept)."
        )
        return raw_df.reset_index(drop=True)

    dedup_seed = config.SPLIT_SEED if dedup_seed is None else dedup_seed
    dedup_key = _build_instance_dedup_key(raw_df)
    raw_df = raw_df.copy()
    raw_df["_instance_dedup_key"] = dedup_key

    dup_key_mask = raw_df["_instance_dedup_key"].duplicated(keep=False)
    n_dup_rows = int(dup_key_mask.sum())
    n_dup_keys = int(raw_df.loc[dup_key_mask, "_instance_dedup_key"].nunique())

    if n_dup_rows > 0:
        logger.warning(
            "Detected duplicated instance_id keys globally: "
            f"{n_dup_keys} keys, {n_dup_rows} rows. "
            f"Applying deterministic random dedup with seed={dedup_seed}."
        )
        rng = np.random.default_rng(dedup_seed)
        raw_df["_dedup_rand"] = rng.random(len(raw_df))
        kept_df = (
            raw_df.sort_values(["_instance_dedup_key", "_dedup_rand"])
            .groupby("_instance_dedup_key", as_index=False, sort=False)
            .head(1)
            .drop(columns=["_dedup_rand"])
            .reset_index(drop=True)
        )
        logger.info(
            f"Trajectories after global instance_id dedup: {len(kept_df)} "
            f"(dropped {len(raw_df) - len(kept_df)})"
        )
    else:
        kept_df = raw_df.reset_index(drop=True)
        logger.info("No duplicated instance_id keys detected; dedup skipped")

    return kept_df.drop(columns=["_instance_dedup_key"])


def _parse_messages(raw) -> list[dict]:
    if isinstance(raw, str):
        return json.loads(raw)
    if isinstance(raw, list):
        return raw
    raise TypeError(f"Unsupported messages type: {type(raw)}")


def _extract_text(content) -> str:
    """从 content 字段提取文本。"""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(item.get("text", ""))
            elif isinstance(item, str):
                parts.append(item)
        return "\n".join(parts)
    if isinstance(content, dict):
        return content.get("text", str(content))
    return str(content) if content else ""


def _is_action_message(msg: dict) -> bool:
    return (
        msg.get("role") == "assistant"
        and msg.get("message_type") == "action"
        and "action" in msg
    )


def _word_overlap(text_a: str, text_b: str) -> float:
    """两段文本的 Jaccard 词汇重合度。"""
    if not text_a or not text_b:
        return 0.0
    sa, sb = set(text_a.lower().split()), set(text_b.lower().split())
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def _text_nearly_equal(a: str, b: str, threshold: float = 0.9) -> bool:
    """判断两段文本是否几乎相同（Jaccard >= threshold）。"""
    if not a.strip() and not b.strip():
        return True
    if not a.strip() or not b.strip():
        return False
    return _word_overlap(a, b) >= threshold


def rebuild_steps_for_trajectory(row: pd.Series) -> list[dict]:
    """
    为一条轨迹重建 step 列表。

    Returns:
        list[dict]  每个 dict 是一个 step 的结构化信息
    """
    traj_id = row.get("traj_id", "")
    instance_id = row.get("instance_id", "")
    resolved = bool(row.get("resolved", False))
    model = row.get("model", "")

    messages = _parse_messages(row["messages"])

    # ── 找到所有 action message 的索引 ──
    action_indices = [i for i, m in enumerate(messages) if _is_action_message(m)]

    if not action_indices:
        # 没有 action，只有 preamble
        return []

    steps = []

    for pos, act_idx in enumerate(action_indices):
        step_idx = pos + 1
        # step 结束位置：下一个 action 之前或 messages 末尾
        if pos + 1 < len(action_indices):
            end_idx = action_indices[pos + 1]
        else:
            end_idx = len(messages)

        act_msg = messages[act_idx]
        following = messages[act_idx + 1: end_idx]

        # ── 提取 action 内容 ──
        thought_text = act_msg.get("thought", "") or ""
        action_text = act_msg.get("action", "") or ""
        content_text = _extract_text(act_msg.get("content"))

        # ── thought / content 低维统计 ──
        has_thought = bool(thought_text.strip())
        has_content = bool(content_text.strip())
        thought_char_len = len(thought_text)
        content_char_len = len(content_text)
        thought_eq_content = _text_nearly_equal(thought_text, content_text)
        thought_action_overlap = _word_overlap(thought_text, action_text)
        content_action_overlap = _word_overlap(content_text, action_text)

        # ── 分类 ──
        major_type, subtypes, primary_subtype = classify_action(action_text)

        # ── tool calls ──
        tool_calls = act_msg.get("tool_calls") or []
        tool_names = []
        tool_args_parts = []
        for tc in tool_calls:
            func = ((tc or {}).get("function") or {})
            tool_names.append(func.get("name", "unknown"))
            args = func.get("arguments", "")
            if isinstance(args, dict):
                args = json.dumps(args, ensure_ascii=False)
            tool_args_parts.append(str(args))

        # ── tool output + observation ──
        tool_output_parts = []
        observation_parts = []
        for fm in following:
            if fm.get("role") == "tool":
                tool_output_parts.append(_extract_text(fm.get("content")))
            if fm.get("message_type") == "observation":
                observation_parts.append(_extract_text(fm.get("content")))

        tool_output_text = "\n".join(tool_output_parts)
        observation_text = "\n".join(observation_parts)
        combined_feedback = (tool_output_text + "\n" + observation_text).strip()

        # ── observation 信号 ──
        sig = parse_observation(combined_feedback)

        step = {
            "traj_id": traj_id,
            "instance_id": instance_id,
            # 与 prefix_table 对齐：按 trajectory 分组，避免 instance 多轨迹混组。
            "group_id": traj_id,
            "resolved": int(resolved),
            "model": model,
            "step_idx": step_idx,
            "message_start_idx": act_idx,
            "message_end_idx": end_idx,
            "thought_text": thought_text,
            "action_text": action_text,
            "assistant_content_text": content_text,
            "tool_names": tool_names,
            "tool_count": len(tool_calls),
            "tool_args_text": "\n".join(tool_args_parts),
            "tool_output_text": tool_output_text,
            "observation_text": observation_text,
            "combined_feedback_text": combined_feedback,
            "action_major_type": major_type,
            "action_subtypes": subtypes,
            "action_primary_subtype": primary_subtype,
            "has_tool_output": len(tool_output_parts) > 0,
            "has_observation": len(observation_parts) > 0,
            "tool_error_seen_this_step": sig.tool_error,
            "traceback_seen_this_step": sig.traceback_seen,
            "test_fail_seen_this_step": sig.test_fail_seen,
            "test_pass_seen_this_step": sig.test_pass_seen,
            "last_fail_count_this_step": sig.fail_count,
            "action_char_len": len(action_text),
            "feedback_char_len": len(combined_feedback),
            # ── 新增：thought / assistant_content 低维统计 ──
            "thought_char_len": thought_char_len,
            "assistant_content_char_len": content_char_len,
            "has_thought_text": has_thought,
            "has_assistant_content_text": has_content,
            "thought_equals_content": thought_eq_content,
            "thought_action_overlap_ratio": thought_action_overlap,
            "content_action_overlap_ratio": content_action_overlap,
        }
        steps.append(step)

    return steps


def _apply_max_trajectories_limit(
    traj_df: pd.DataFrame,
    max_trajectories: Optional[int],
    *,
    sample_trajectories_seed: Optional[int] = None,
) -> pd.DataFrame:
    """
    在去重后的轨迹表上应用条数上限。

    sample_trajectories_seed:
        若提供，先用该种子打乱行顺序再取前 N 条（可复现的随机子集）；
        若为 None，则保持去重后的原有顺序，取前 N 条。
    """
    if max_trajectories is None or max_trajectories <= 0:
        return traj_df
    n0 = len(traj_df)
    work = traj_df
    if sample_trajectories_seed is not None:
        work = work.sample(frac=1.0, random_state=sample_trajectories_seed).reset_index(drop=True)
        logger.info(
            f"sample_trajectories_seed={sample_trajectories_seed}: "
            f"shuffled trajectories before cap (n={n0})"
        )
    out = work.iloc[: int(max_trajectories)].reset_index(drop=True)
    logger.info(f"max_trajectories={max_trajectories}: using {len(out)} / {n0} trajectories")
    return out


def build_step_table(
    input_dir: Optional[str] = None,
    *,
    max_trajectories: Optional[int] = None,
    sample_trajectories_seed: Optional[int] = None,
) -> pd.DataFrame:
    """
    读取所有 tool split parquet，重建 step 表。

    max_trajectories:
        若设置，在去重后仅保留至多 N 条轨迹；顺序默认与去重后一致，或由 sample_trajectories_seed 打乱后再取前 N。

    Returns:
        pd.DataFrame  step_table
    """
    input_dir = input_dir or config.PARQUET_INPUT_DIR
    traj_df = _load_and_deduplicate_trajectories(input_dir=input_dir, dedup_seed=config.SPLIT_SEED)
    traj_df = _apply_max_trajectories_limit(
        traj_df, max_trajectories, sample_trajectories_seed=sample_trajectories_seed
    )
    all_steps = []
    total_no_action = 0

    with timer(logger, "Rebuilding steps from deduplicated trajectories"):
        for _, row in traj_df.iterrows():
            try:
                steps = rebuild_steps_for_trajectory(row)
                if not steps:
                    total_no_action += 1
                all_steps.extend(steps)
            except Exception as e:
                logger.warning(f"  Failed to rebuild steps for traj {row.get('traj_id')}: {e}")

    logger.info(f"Total trajectories: {len(traj_df)}")
    logger.info(f"Trajectories with no action: {total_no_action}")
    logger.info(f"Total steps rebuilt: {len(all_steps)}")

    step_df = pd.DataFrame(all_steps)
    logger.info(f"Step table shape: {step_df.shape}")
    logger.info(f"Step table columns: {list(step_df.columns)}")

    # ── 验证统计 ──
    if len(step_df) > 0:
        n_trajs = step_df["traj_id"].nunique()
        avg_steps = step_df.groupby("traj_id").size().mean()
        logger.info(f"Unique trajectories in step table: {n_trajs}")
        logger.info(f"Average steps per trajectory: {avg_steps:.2f}")

        # 动作分类统计
        action_counts = step_df["action_primary_subtype"].value_counts()
        logger.info(f"Action subtype distribution:\n{action_counts.to_string()}")

        # resolved 分布
        res_counts = step_df.groupby("traj_id")["resolved"].first().value_counts()
        logger.info(f"Resolved distribution (trajectory level):\n{res_counts.to_string()}")

    return step_df


def build_preamble_info(row: pd.Series) -> dict:
    """提取 preamble (step=0) 的信息。"""
    messages = _parse_messages(row["messages"])
    action_indices = [i for i, m in enumerate(messages) if _is_action_message(m)]

    preamble_end = action_indices[0] if action_indices else len(messages)
    preamble_msgs = messages[:preamble_end]

    task_parts = []
    for m in preamble_msgs:
        if m.get("role") in ("user", "system"):
            task_parts.append(_extract_text(m.get("content")))

    return {
        "traj_id": row.get("traj_id", ""),
        "instance_id": row.get("instance_id", ""),
        "group_id": row.get("traj_id", ""),
        "resolved": int(bool(row.get("resolved", False))),
        "model": row.get("model", ""),
        "task_prompt_text": "\n".join(task_parts),
        "n_preamble_messages": len(preamble_msgs),
    }


if __name__ == "__main__":
    step_df = build_step_table()
    step_df.to_parquet(config.STEP_TABLE_PATH, index=False)
    logger.info(f"Saved step table to {config.STEP_TABLE_PATH}")
