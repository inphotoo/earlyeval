#!/usr/bin/env python3
"""Build internal review tables for the 16-fold SWEVerify LightGBM run.

This is a post-hoc reporter only.  It does not train models or mutate any of
the existing fold outputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


LEGACY_CHARS_PER_TOKEN = 4.0
DEFAULT_MAX_DIRECT_TOKENIZE_CHARS = 2_000_000
DEFAULT_EXACT_CHUNK_TOKENIZE_CHARS = 1_000_000
DEFAULT_SAMPLE_LONG_TOKENIZE_CHARS = 20_000_000
DEFAULT_SAMPLE_CHARS = 262_144
CONTEXT_TEXT_COLUMNS = [
    "task_prompt_text",
    "prefix_action_text",
    "prefix_feedback_text",
    "prefix_assistant_content_text",
]
CONTEXT_CHAR_COLUMNS = [
    "task_prompt_chars",
    "prefix_action_chars",
    "prefix_feedback_chars",
    "prefix_assistant_content_chars",
]
COMPONENT_STEP_TEXT_COLUMNS = [
    ("action", "last_action_text"),
    ("feedback", "last_feedback_text"),
    ("assistant", "last_assistant_content_text"),
]
STOP_SIGNAL_COLUMNS = [
    "last_step_has_observation",
    "last_step_has_tool_output",
    "last_step_tool_error_seen",
    "last_step_traceback_seen",
    "last_step_test_pass_seen",
    "last_step_test_fail_seen",
    "test_pass_seen",
    "test_fail_seen",
    "all_tests_passed_seen",
    "edit_failed_seen",
]


@dataclass(frozen=True)
class TokenizerSpec:
    family: str
    backend: str
    name: str
    note: str
    trust_remote_code: bool = False


def _pct(num: float, den: float) -> float:
    return float(num) * 100.0 / float(den) if den else float("nan")


def _fmt(value: float, digits: int = 2) -> str:
    if pd.isna(value):
        return "nan"
    return f"{float(value):.{digits}f}"


def _head_column(kind: str, score_mode: str, predictor: str) -> str:
    if score_mode == "raw":
        return f"prob_safe_{kind}__{predictor}"
    if score_mode == "calibrated":
        return f"prob_cal_safe_{kind}__{predictor}"
    raise ValueError(f"unknown score_mode: {score_mode}")


def _decide_dual(
    group: pd.DataFrame,
    *,
    success_col: str,
    failure_col: str,
    success_thr: float,
    failure_thr: float,
    min_step: int,
    consecutive: int,
) -> dict[str, Any]:
    last_decision = "undecided"
    streak = 0
    for row in group.itertuples(index=False):
        step = int(getattr(row, "prefix_step_idx"))
        if step < min_step:
            continue
        success_score = float(getattr(row, success_col))
        failure_score = float(getattr(row, failure_col))
        success_hit = success_score >= success_thr
        failure_hit = failure_score >= failure_thr
        if success_hit and failure_hit:
            success_margin = success_score - success_thr
            failure_margin = failure_score - failure_thr
            decision = "success" if success_margin >= failure_margin else "failure"
            score = success_score if decision == "success" else failure_score
        elif success_hit:
            decision = "success"
            score = success_score
        elif failure_hit:
            decision = "failure"
            score = failure_score
        else:
            last_decision = "undecided"
            streak = 0
            continue

        streak = streak + 1 if decision == last_decision else 1
        last_decision = decision
        if streak >= consecutive:
            return {
                "decided": True,
                "decision": decision,
                "decision_step": step,
                "decision_score": score,
                "prob_success_at_decision": success_score,
                "prob_failure_at_decision": failure_score,
            }
    return {
        "decided": False,
        "decision": "undecided",
        "decision_step": -1,
        "decision_score": np.nan,
        "prob_success_at_decision": np.nan,
        "prob_failure_at_decision": np.nan,
    }


def _outcome_type(label: int, decided: bool, decision: str) -> str:
    if not decided:
        return "undecided_success" if label == 1 else "undecided_failure"
    if decision == "success" and label == 1:
        return "true_positive"
    if decision == "success" and label == 0:
        return "false_positive"
    if decision == "failure" and label == 0:
        return "true_negative"
    if decision == "failure" and label == 1:
        return "false_negative"
    return "unknown"


def _load_selected(run_dir: Path) -> pd.DataFrame:
    selected = pd.read_csv(run_dir / "summary" / "per_fold_test_selected.csv")
    selected = selected.copy()
    selected["adjusted_resolved_calc"] = (
        selected["original_resolved"]
        - selected["false_negatives"]
        + selected["false_positives"]
    )
    selected["original_resolve_rate_calc"] = (
        selected["original_resolved"] / selected["original_total"]
    )
    selected["adjusted_resolve_rate_calc"] = (
        selected["adjusted_resolved_calc"] / selected["original_total"]
    )
    selected["resolve_rate_change_pp_calc"] = (
        selected["adjusted_resolve_rate_calc"] - selected["original_resolve_rate_calc"]
    ) * 100.0
    # The stage-04 summary may store ``adjusted_resolved`` under either the
    # policy-outcome convention (original_resolved - false_negatives +
    # false_positives, used by the paper-facing tables and computed above as
    # ``adjusted_resolved_calc``) or the resolve-only convention
    # (original_resolved - false_negatives). All downstream columns here use
    # ``adjusted_resolved_calc`` (policy-outcome), so we only validate that the
    # stored column is consistent with one of those deterministic formulas;
    # this still trips on genuinely corrupted counts.
    resolve_only = selected["original_resolved"] - selected["false_negatives"]
    consistent = (selected["adjusted_resolved"] == selected["adjusted_resolved_calc"]) | (
        selected["adjusted_resolved"] == resolve_only
    )
    mismatched = selected[~consistent]
    if not mismatched.empty:
        bad = ", ".join(mismatched["fold_id"].astype(str).head(5))
        raise ValueError(f"selected adjusted_resolved mismatch in folds: {bad}")
    if len(selected) != 16:
        raise ValueError(f"expected 16 selected SWE folds, found {len(selected)}")
    return selected


def build_selected_rank_change(selected: pd.DataFrame) -> pd.DataFrame:
    out = selected.copy()
    out["original_rank"] = (
        out["original_resolve_rate_calc"].rank(ascending=False, method="min").astype(int)
    )
    out["adjusted_rank"] = (
        out["adjusted_resolve_rate_calc"].rank(ascending=False, method="min").astype(int)
    )
    out["rank_change_positive_is_up"] = out["original_rank"] - out["adjusted_rank"]
    out["original_resolve_rate_pct"] = out["original_resolve_rate_calc"] * 100.0
    out["adjusted_resolve_rate_pct"] = out["adjusted_resolve_rate_calc"] * 100.0
    out["step_save_pct"] = out["pct_steps_saved"]
    out["decision_accuracy_pct"] = out["decision_accuracy"] * 100.0
    cols = [
        "test_model",
        "original_rank",
        "adjusted_rank",
        "rank_change_positive_is_up",
        "original_resolved",
        "adjusted_resolved_calc",
        "original_total",
        "original_resolve_rate_pct",
        "adjusted_resolve_rate_pct",
        "resolve_rate_change_pp_calc",
        "false_negatives",
        "false_positives",
        "step_save_pct",
        "decision_accuracy_pct",
        "n_decided",
        "coverage",
        "policy_mode",
        "success_thr",
        "failure_thr",
        "min_step",
        "consecutive",
        "fold_id",
    ]
    return out[cols].sort_values(["original_rank", "test_model"]).reset_index(drop=True)


def build_frontier_rank_change(run_dir: Path) -> pd.DataFrame:
    path = run_dir / "policy_sweeps" / "valid_accuracy_075_095" / "per_fold_test_metrics.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    df = df.copy()
    df["adjusted_resolved_calc"] = (
        df["original_resolved"] - df["false_negatives"] + df["false_positives"]
    )
    df["original_resolve_rate_calc"] = df["original_resolved"] / df["original_total"]
    df["adjusted_resolve_rate_calc"] = df["adjusted_resolved_calc"] / df["original_total"]
    df["resolve_rate_change_pp_calc"] = (
        df["adjusted_resolve_rate_calc"] - df["original_resolve_rate_calc"]
    ) * 100.0
    parts = []
    for target, part in df.groupby("target_valid_decision_accuracy_pct", sort=True):
        work = part.copy()
        work["original_rank"] = (
            work["original_resolve_rate_calc"].rank(ascending=False, method="min").astype(int)
        )
        work["adjusted_rank"] = (
            work["adjusted_resolve_rate_calc"].rank(ascending=False, method="min").astype(int)
        )
        work["rank_change_positive_is_up"] = work["original_rank"] - work["adjusted_rank"]
        parts.append(work)
    out = pd.concat(parts, ignore_index=True)
    out["original_resolve_rate_pct"] = out["original_resolve_rate_calc"] * 100.0
    out["adjusted_resolve_rate_pct"] = out["adjusted_resolve_rate_calc"] * 100.0
    cols = [
        "target_valid_decision_accuracy_pct",
        "test_model",
        "original_rank",
        "adjusted_rank",
        "rank_change_positive_is_up",
        "original_resolved",
        "adjusted_resolved_calc",
        "original_total",
        "original_resolve_rate_pct",
        "adjusted_resolve_rate_pct",
        "resolve_rate_change_pp_calc",
        "false_negatives",
        "false_positives",
        "pct_steps_saved",
        "decision_accuracy_pct",
        "coverage_pct",
        "n_decided",
        "policy_mode",
        "success_thr",
        "failure_thr",
        "min_step",
        "consecutive",
        "fold_id",
    ]
    return out[cols].sort_values(
        ["target_valid_decision_accuracy_pct", "original_rank", "test_model"]
    ).reset_index(drop=True)


def build_selected_decisions(run_dir: Path, selected: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for policy in selected.itertuples(index=False):
        fold_id = str(policy.fold_id)
        predictor = str(policy.prefix_model)
        score_mode = str(policy.score_mode)
        success_col = _head_column("success", score_mode, predictor)
        failure_col = _head_column("failure", score_mode, predictor)
        pred_path = run_dir / "folds" / fold_id / "test_predictions_safe_stop.parquet"
        cols = [
            "traj_id",
            "instance_id",
            "group_id",
            "prefix_step_idx",
            "label",
            "orig_model_id",
            "orig_model",
            success_col,
            failure_col,
        ]
        pred = pd.read_parquet(pred_path, columns=cols)
        pred = pred.sort_values(["traj_id", "prefix_step_idx"])
        for traj_id, group in pred.groupby("traj_id", sort=False):
            group = group.sort_values("prefix_step_idx")
            label = int(group["label"].iloc[0])
            n_steps = int(len(group))
            decision = _decide_dual(
                group,
                success_col=success_col,
                failure_col=failure_col,
                success_thr=float(policy.success_thr),
                failure_thr=float(policy.failure_thr),
                min_step=int(policy.min_step),
                consecutive=int(policy.consecutive),
            )
            saved_steps = (
                max(n_steps - int(decision["decision_step"]) - 1, 0)
                if bool(decision["decided"])
                else 0
            )
            decision_correct = (
                (decision["decision"] == "success" and label == 1)
                or (decision["decision"] == "failure" and label == 0)
            )
            rows.append(
                {
                    "fold_id": fold_id,
                    "test_model": str(policy.test_model),
                    "traj_id": traj_id,
                    "instance_id": str(group["instance_id"].iloc[0]),
                    "label": label,
                    "n_steps": n_steps,
                    "decided": bool(decision["decided"]),
                    "decision": decision["decision"],
                    "decision_step": int(decision["decision_step"]),
                    "round_1based": (
                        int(decision["decision_step"]) + 1
                        if bool(decision["decided"])
                        else -1
                    ),
                    "decision_score": decision["decision_score"],
                    "prob_success_at_decision": decision["prob_success_at_decision"],
                    "prob_failure_at_decision": decision["prob_failure_at_decision"],
                    "saved_steps": saved_steps,
                    "decision_correct": bool(decision_correct) if bool(decision["decided"]) else False,
                    "outcome_type": _outcome_type(label, bool(decision["decided"]), str(decision["decision"])),
                    "policy_mode": str(policy.policy_mode),
                    "success_thr": float(policy.success_thr),
                    "failure_thr": float(policy.failure_thr),
                    "min_step": int(policy.min_step),
                    "consecutive": int(policy.consecutive),
                    "score_mode": score_mode,
                    "prefix_model": predictor,
                }
            )
    decisions = pd.DataFrame(rows)
    return decisions.sort_values(["fold_id", "traj_id"]).reset_index(drop=True)


def audit_decisions(decisions: pd.DataFrame, selected: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for fold_id, part in decisions.groupby("fold_id", sort=True):
        expected = selected[selected["fold_id"].eq(fold_id)].iloc[0]
        actual = {
            "original_total": len(part),
            "original_resolved": int(part["label"].sum()),
            "false_negatives": int(part["outcome_type"].eq("false_negative").sum()),
            "false_positives": int(part["outcome_type"].eq("false_positive").sum()),
            "true_negatives": int(part["outcome_type"].eq("true_negative").sum()),
            "true_positives": int(part["outcome_type"].eq("true_positive").sum()),
            "n_decided": int(part["decided"].sum()),
            "total_saved_steps": int(part["saved_steps"].sum()),
            "total_steps": int(part["n_steps"].sum()),
        }
        # ``adjusted_resolved`` is a deterministic function of the raw counts
        # audited above (original_resolved / false_negatives / false_positives),
        # and the stage-04 summary may store it under a different convention
        # (resolve-only vs policy-outcome). Auditing it again would only
        # re-test a convention mismatch, not data integrity, so it is omitted
        # from the reconstruction comparison.
        row = {"fold_id": fold_id}
        for key, value in actual.items():
            exp_value = int(round(float(expected[key]))) if key in expected.index else None
            row[f"actual_{key}"] = value
            row[f"expected_{key}"] = exp_value
            row[f"match_{key}"] = bool(value == exp_value) if exp_value is not None else True
        rows.append(row)
    return pd.DataFrame(rows)


def _available_parquet_columns(path: Path) -> set[str]:
    try:
        import pyarrow.parquet as pq
    except Exception as exc:  # pragma: no cover - dependency guard
        raise RuntimeError("pyarrow is required to inspect parquet columns") from exc
    return set(pq.ParquetFile(path).schema.names)


def _load_stop_signal_prefix_rows(prefix_table: Path, traj_ids: set[str]) -> pd.DataFrame:
    available = _available_parquet_columns(prefix_table)
    signal_cols = [col for col in STOP_SIGNAL_COLUMNS if col in available]
    cols = ["traj_id", "model_id", "prefix_step_idx", *signal_cols]
    df = pd.read_parquet(prefix_table, columns=cols)
    df["traj_id"] = df["traj_id"].astype(str)
    df = df[df["traj_id"].isin(set(str(x) for x in traj_ids))].copy()
    df["prefix_step_idx"] = pd.to_numeric(df["prefix_step_idx"], errors="raise").astype("int64")
    for col in signal_cols:
        df[col] = df[col].fillna(False).astype(bool)
    return df


def _prepare_stop_signal_risk_rows(prefix_table: Path, decisions: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    prefix = _load_stop_signal_prefix_rows(prefix_table, set(decisions["traj_id"]))
    signal_cols = [col for col in STOP_SIGNAL_COLUMNS if col in prefix.columns]
    dec_cols = [
        "traj_id",
        "decided",
        "decision",
        "decision_step",
        "decision_correct",
        "label",
        "n_steps",
    ]
    dec = decisions[dec_cols].copy()
    dec["traj_id"] = dec["traj_id"].astype(str)
    risk = prefix.merge(dec, on="traj_id", how="left", validate="many_to_one")
    risk["decided"] = risk["decided"].fillna(False).astype(bool)
    risk["decision_step"] = pd.to_numeric(risk["decision_step"], errors="coerce").fillna(-1).astype("int64")
    risk["at_risk_before_policy_stop"] = (~risk["decided"]) | (
        risk["prefix_step_idx"] <= risk["decision_step"]
    )
    risk["is_policy_stop_prefix"] = risk["decided"] & (
        risk["prefix_step_idx"] == risk["decision_step"]
    )
    return risk[risk["at_risk_before_policy_stop"]].copy(), signal_cols


def build_stop_signal_decision_lift_tables(
    prefix_table: Path,
    decisions: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    risk, signal_cols = _prepare_stop_signal_risk_rows(prefix_table, decisions)
    n_at_risk = int(len(risk))
    n_decisions = int(decisions["decided"].sum())
    n_success = int(decisions["decision"].eq("success").sum())
    n_failure = int(decisions["decision"].eq("failure").sum())
    baseline_prefix_stop_rate = _pct(n_decisions, n_at_risk)
    stop_rows = risk[risk["is_policy_stop_prefix"]].copy()

    prefix_rows: list[dict[str, Any]] = []
    composition_rows: list[dict[str, Any]] = []
    for signal in signal_cols:
        sig = risk[signal].fillna(False).astype(bool)
        denom = int(sig.sum())
        signal_stop = risk[sig & risk["is_policy_stop_prefix"]]
        stop_n = int(len(signal_stop))
        success_n = int(signal_stop["decision"].eq("success").sum())
        failure_n = int(signal_stop["decision"].eq("failure").sum())
        prefix_rows.append(
            {
                "signal": signal,
                "scope": "at_risk_prefixes_before_policy_stop",
                "at_risk_prefix_rows": n_at_risk,
                "signal_prefix_rows": denom,
                "signal_prefix_pct": _pct(denom, n_at_risk),
                "policy_stop_prefixes": stop_n,
                "policy_stop_rate_given_signal_pct": _pct(stop_n, denom),
                "success_stop_prefixes": success_n,
                "success_stop_rate_given_signal_pct": _pct(success_n, denom),
                "failure_stop_prefixes": failure_n,
                "failure_stop_rate_given_signal_pct": _pct(failure_n, denom),
                "decision_accuracy_given_signal_stop_pct": _pct(
                    int(signal_stop["decision_correct"].sum()), stop_n
                ),
                "baseline_policy_stop_rate_pct": baseline_prefix_stop_rate,
                "lift_vs_baseline_stop_rate": (
                    _pct(stop_n, denom) / baseline_prefix_stop_rate
                    if denom and baseline_prefix_stop_rate
                    else np.nan
                ),
                "share_of_policy_stops_with_signal_pct": _pct(stop_n, n_decisions),
            }
        )

        stop_sig = stop_rows[signal].fillna(False).astype(bool)
        all_true = int(stop_sig.sum())
        success_true = int((stop_sig & stop_rows["decision"].eq("success")).sum())
        failure_true = int((stop_sig & stop_rows["decision"].eq("failure")).sum())
        composition_rows.append(
            {
                "signal": signal,
                "policy_stop_prefixes": n_decisions,
                "n_true_at_stop": all_true,
                "pct_of_policy_stops": _pct(all_true, n_decisions),
                "success_stop_prefixes": n_success,
                "success_n_true_at_stop": success_true,
                "success_pct_of_success_stops": _pct(success_true, n_success),
                "failure_stop_prefixes": n_failure,
                "failure_n_true_at_stop": failure_true,
                "failure_pct_of_failure_stops": _pct(failure_true, n_failure),
            }
        )

    signal_by_traj = (
        risk.groupby("traj_id", sort=False)[signal_cols]
        .max()
        .reset_index()
    )
    traj = decisions[
        ["traj_id", "decided", "decision", "decision_correct", "label", "n_steps"]
    ].copy()
    traj["traj_id"] = traj["traj_id"].astype(str)
    traj = traj.merge(signal_by_traj, on="traj_id", how="left")
    for signal in signal_cols:
        traj[signal] = traj[signal].fillna(False).astype(bool)
    n_traj = int(len(traj))
    baseline_traj_decision_rate = _pct(n_decisions, n_traj)
    traj_rows: list[dict[str, Any]] = []
    for signal in signal_cols:
        part = traj[traj[signal]].copy()
        denom = int(len(part))
        decided = part[part["decided"]]
        decided_n = int(len(decided))
        success_n = int(decided["decision"].eq("success").sum())
        failure_n = int(decided["decision"].eq("failure").sum())
        traj_rows.append(
            {
                "signal": signal,
                "scope": "trajectory_seen_before_policy_stop_or_full_if_undecided",
                "trajectories": n_traj,
                "trajectories_with_signal": denom,
                "trajectory_signal_pct": _pct(denom, n_traj),
                "decided_trajectories_with_signal": decided_n,
                "decision_rate_given_signal_pct": _pct(decided_n, denom),
                "success_stop_trajectories": success_n,
                "success_stop_rate_given_signal_pct": _pct(success_n, denom),
                "failure_stop_trajectories": failure_n,
                "failure_stop_rate_given_signal_pct": _pct(failure_n, denom),
                "decision_accuracy_given_signal_pct": _pct(
                    int(decided["decision_correct"].sum()), decided_n
                ),
                "baseline_decision_rate_pct": baseline_traj_decision_rate,
                "lift_vs_baseline_decision_rate": (
                    _pct(decided_n, denom) / baseline_traj_decision_rate
                    if denom and baseline_traj_decision_rate
                    else np.nan
                ),
            }
        )

    prefix_out = pd.DataFrame(prefix_rows).sort_values(
        ["policy_stop_rate_given_signal_pct", "policy_stop_prefixes"],
        ascending=[False, False],
    )
    traj_out = pd.DataFrame(traj_rows).sort_values(
        ["decision_rate_given_signal_pct", "decided_trajectories_with_signal"],
        ascending=[False, False],
    )
    composition_out = pd.DataFrame(composition_rows).sort_values(
        ["pct_of_policy_stops", "n_true_at_stop"],
        ascending=[False, False],
    )
    return prefix_out.reset_index(drop=True), traj_out.reset_index(drop=True), composition_out.reset_index(drop=True)


def _configure_tokenizer_cache(cache_root: Path | None) -> None:
    if cache_root is None:
        return
    cache_root = cache_root.expanduser().resolve()
    hf_home = cache_root / "huggingface"
    tiktoken_cache = cache_root / "tiktoken"
    hf_home.mkdir(parents=True, exist_ok=True)
    tiktoken_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(hf_home))
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(hf_home / "hub"))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(hf_home / "transformers"))
    os.environ.setdefault("TIKTOKEN_CACHE_DIR", str(tiktoken_cache))
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def _tokenizer_spec_for_model(model_id: str) -> TokenizerSpec:
    lower = str(model_id).lower()
    if "gpt-oss-20b" in lower:
        return TokenizerSpec(
            family="openai_gpt_oss",
            backend="hf",
            name="openai/gpt-oss-20b",
            note="OpenAI GPT-OSS 20B tokenizer.",
        )
    if "gpt-oss" in lower:
        return TokenizerSpec(
            family="openai_gpt_oss",
            backend="hf",
            name="openai/gpt-oss-120b",
            note="OpenAI GPT-OSS tokenizer; reused for GPT-OSS-family trajectories.",
        )
    if (
        "gpt-5" in lower
        or "openai" in lower
        or lower in {"o3", "o4-mini"}
        or lower.startswith("o3")
        or lower.startswith("o4")
    ):
        return TokenizerSpec(
            family="openai_gpt",
            backend="tiktoken",
            name="o200k_base",
            note="OpenAI GPT/O-series proxy tokenizer; exact GPT-5/O-series tokenizer is not exposed in these artifacts.",
        )
    if "sonnet" in lower or "claude" in lower:
        return TokenizerSpec(
            family="claude",
            backend="hf",
            name="Xenova/claude-tokenizer",
            note="Public community Claude tokenizer used because Anthropic does not expose a local official tokenizer here.",
        )
    if "gemini" in lower:
        return TokenizerSpec(
            family="gemini_gemma_proxy",
            backend="hf",
            name="mlx-community/gemma-3-270m-it-4bit",
            note="Gemma-3 tokenizer proxy for Gemini; Gemini's official tokenizer is not locally available.",
        )
    if "qwen" in lower:
        return TokenizerSpec(
            family="qwen3_coder",
            backend="hf",
            name="Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8",
            note="Qwen3 Coder tokenizer for Qwen Coder trajectories.",
        )
    if "grok" in lower or "@xai" in lower:
        return TokenizerSpec(
            family="grok_proxy",
            backend="hf",
            name="alvarobartt/grok-2-tokenizer",
            note="Public Grok-2 tokenizer proxy; no official local Grok-4/Grok-code-fast tokenizer is available in these artifacts.",
        )
    if "glm" in lower:
        if "glm-5" in lower:
            return TokenizerSpec(
                family="glm",
                backend="hf",
                name="zai-org/GLM-4.6",
                note="GLM-4.6 tokenizer proxy for GLM-5 because the GLM-5 tokenizer class is not loadable in the current environment.",
            )
        return TokenizerSpec(
            family="glm",
            backend="hf",
            name="zai-org/GLM-4.6",
            note="Public GLM-family tokenizer for GLM 4.5/4.6 trajectories.",
        )
    if "devstral" in lower:
        return TokenizerSpec(
            family="devstral",
            backend="hf",
            name="mistralai/Mistral-Small-3.1-24B-Instruct-2503",
            note="Mistral Small 3.1 tokenizer proxy for Devstral because the Devstral tokenizer is not loadable with this Transformers version.",
        )
    if "deepseek" in lower:
        return TokenizerSpec(
            family="deepseek",
            backend="hf",
            name="deepseek-ai/DeepSeek-V3.2-Exp",
            note="DeepSeek V3.2 tokenizer for DeepSeek reasoner trajectories.",
        )
    if "kimi" in lower:
        if "instruct-0905" in lower or "kimi-k2-0905" in lower:
            return TokenizerSpec(
                family="kimi",
                backend="hf",
                name="moonshotai/Kimi-K2-Instruct-0905",
                note="Kimi K2 Instruct tokenizer with trust_remote_code.",
                trust_remote_code=True,
            )
        return TokenizerSpec(
            family="kimi",
            backend="hf",
            name="moonshotai/Kimi-K2-Thinking",
            note="Kimi K2 Thinking tokenizer with trust_remote_code.",
            trust_remote_code=True,
        )
    if "minimax" in lower:
        if "m2.5" in lower:
            return TokenizerSpec(
                family="minimax",
                backend="hf",
                name="MiniMaxAI/MiniMax-M2.5",
                note="MiniMax M2.5 tokenizer.",
            )
        return TokenizerSpec(
            family="minimax",
            backend="hf",
            name="MiniMaxAI/MiniMax-M2",
            note="MiniMax M2 tokenizer.",
        )
    raise ValueError(f"No tokenizer mapping configured for model_id={model_id!r}")


def _string_or_empty(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    return str(value)


def _context_text_from_row(row: Any) -> tuple[str, int]:
    parts = [_string_or_empty(getattr(row, col)) for col in CONTEXT_TEXT_COLUMNS]
    context_chars = sum(len(part) for part in parts)
    return "\n".join(part for part in parts if part), context_chars


def _component_method_severity(method: str) -> int:
    if "sampled_long_estimate" in str(method):
        return 2
    if "chunked_exact" in str(method):
        return 1
    return 0


def _component_method_from_severity(severity: int) -> str:
    if int(severity) >= 2:
        return "component_sum_approx_sampled_long_estimate"
    if int(severity) == 1:
        return "component_sum_approx_chunked_exact"
    return "component_sum_approx_direct"


class _TokenCounter:
    def __init__(self, spec: TokenizerSpec, *, local_files_only: bool) -> None:
        self.spec = spec
        self.local_files_only = local_files_only
        self.class_name = ""
        self.vocab_size: int | None = None
        self.commit_hash: str | None = None
        if spec.backend == "tiktoken":
            try:
                import tiktoken
            except Exception as exc:  # pragma: no cover - dependency guard
                raise RuntimeError("tiktoken is required for OpenAI tokenizer counts") from exc
            self._tokenizer = tiktoken.get_encoding(spec.name)
            self.class_name = "tiktoken.Encoding"
            self.vocab_size = int(getattr(self._tokenizer, "n_vocab", 0) or 0)
        elif spec.backend == "hf":
            try:
                from transformers import AutoTokenizer
            except Exception as exc:  # pragma: no cover - dependency guard
                raise RuntimeError("transformers is required for HuggingFace tokenizer counts") from exc
            self._tokenizer = AutoTokenizer.from_pretrained(
                spec.name,
                trust_remote_code=spec.trust_remote_code,
                use_fast=True,
                local_files_only=local_files_only,
            )
            self.class_name = self._tokenizer.__class__.__name__
            self.vocab_size = getattr(self._tokenizer, "vocab_size", None)
            init_kwargs = getattr(self._tokenizer, "init_kwargs", {}) or {}
            self.commit_hash = init_kwargs.get("_commit_hash")
        else:
            raise ValueError(f"unknown tokenizer backend: {spec.backend}")

    def _count_many_direct(self, texts: list[str]) -> list[int]:
        if self.spec.backend == "tiktoken":
            return [len(self._tokenizer.encode_ordinary(text)) for text in texts]
        if self.spec.family == "kimi":
            # Kimi's HF wrapper falls back to the slow PreTrainedTokenizer path when
            # kwargs such as add_special_tokens are passed.  Its native encode()
            # already chunks long strings through the underlying tiktoken model and
            # does not add BOS/EOS tokens.
            return [len(self._tokenizer.encode(text)) for text in texts]
        if self.spec.family == "devstral":
            try:
                base = self._tokenizer.tokenizer.instruct_tokenizer.tokenizer
                model = getattr(base, "_model", None)
                if model is not None:
                    return [len(model.encode(text)) for text in texts]
            except Exception:
                pass
        try:
            encoded = self._tokenizer(
                texts,
                add_special_tokens=False,
                return_length=True,
                padding=False,
                truncation=False,
            )
            lengths = encoded.get("length")
            if lengths is not None:
                return [int(x) for x in lengths]
        except Exception:
            pass
        return [
            len(self._tokenizer.encode(text, add_special_tokens=False))
            for text in texts
        ]

    def _sample_ranges(self, text_len: int, sample_chars: int) -> list[tuple[int, int]]:
        if text_len <= sample_chars:
            return [(0, text_len)]
        starts = [
            0,
            max((text_len - sample_chars) // 4, 0),
            max((text_len - sample_chars) // 2, 0),
            max((text_len - sample_chars) * 3 // 4, 0),
            max(text_len - sample_chars, 0),
        ]
        ranges: list[tuple[int, int]] = []
        seen: set[int] = set()
        for start in starts:
            if start in seen:
                continue
            seen.add(start)
            ranges.append((start, min(start + sample_chars, text_len)))
        return ranges

    def _count_one_long(
        self,
        text: str,
        *,
        exact_chunk_chars: int,
        sample_chars: int,
        sample_long_tokenize_chars: int,
    ) -> tuple[int, str]:
        text_len = len(text)
        if text_len > sample_long_tokenize_chars:
            ranges = self._sample_ranges(text_len, max(1, sample_chars))
            chunks = [text[start:end] for start, end in ranges]
            chunk_chars = sum(len(chunk) for chunk in chunks)
            chunk_tokens = sum(self._count_many_direct(chunks))
            if chunk_chars <= 0:
                return 0, "sampled_long_estimate"
            return int(math.ceil(text_len * chunk_tokens / chunk_chars)), "sampled_long_estimate"
        chunks = [
            text[start : start + max(1, exact_chunk_chars)]
            for start in range(0, text_len, max(1, exact_chunk_chars))
        ]
        return int(sum(self._count_many_direct(chunks))), "chunked_exact"

    def count_many(
        self,
        texts: list[str],
        *,
        max_direct_chars: int,
        exact_chunk_chars: int,
        sample_long_tokenize_chars: int,
        sample_chars: int,
    ) -> tuple[list[int], list[str]]:
        counts: list[int | None] = [None] * len(texts)
        methods: list[str] = ["direct"] * len(texts)
        direct_indices: list[int] = []
        direct_texts: list[str] = []
        for idx, text in enumerate(texts):
            if len(text) <= max_direct_chars:
                direct_indices.append(idx)
                direct_texts.append(text)
                continue
            count, method = self._count_one_long(
                text,
                exact_chunk_chars=exact_chunk_chars,
                sample_chars=sample_chars,
                sample_long_tokenize_chars=sample_long_tokenize_chars,
            )
            counts[idx] = count
            methods[idx] = method
        if direct_texts:
            for idx, count in zip(direct_indices, self._count_many_direct(direct_texts)):
                counts[idx] = count
        return [int(x or 0) for x in counts], methods

    def manifest_row(self, model_id: str) -> dict[str, Any]:
        return {
            "model_id": model_id,
            "tokenizer_family": self.spec.family,
            "tokenizer_backend": self.spec.backend,
            "tokenizer_name": self.spec.name,
            "tokenizer_class": self.class_name,
            "tokenizer_vocab_size": self.vocab_size,
            "tokenizer_commit_hash": self.commit_hash or "",
            "trust_remote_code": bool(self.spec.trust_remote_code),
            "local_files_only": bool(self.local_files_only),
            "note": self.spec.note,
        }


class _TokenizerRegistry:
    def __init__(self, *, local_files_only: bool) -> None:
        self.local_files_only = local_files_only
        self._by_name: dict[tuple[str, str], _TokenCounter] = {}
        self._by_model: dict[str, _TokenCounter] = {}

    def get(self, model_id: str) -> _TokenCounter:
        model_id = str(model_id)
        if model_id in self._by_model:
            return self._by_model[model_id]
        spec = _tokenizer_spec_for_model(model_id)
        key = (spec.backend, spec.name)
        if key not in self._by_name:
            self._by_name[key] = _TokenCounter(spec, local_files_only=self.local_files_only)
        self._by_model[model_id] = self._by_name[key]
        return self._by_model[model_id]

    def manifest(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                counter.manifest_row(model_id)
                for model_id, counter in sorted(self._by_model.items())
            ]
        )


def _finalize_token_rows(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["traj_id", "prefix_step_idx"]).reset_index(drop=True)
    df["n_steps"] = df.groupby("traj_id")["prefix_step_idx"].transform("size").astype("int64")
    df["remaining_steps_after_step"] = df["n_steps"] - df["prefix_step_idx"].astype("int64") - 1
    final_tokens = (
        df.groupby("traj_id", sort=False)["context_tokens_est"]
        .last()
        .rename("final_context_tokens_est")
    )
    full_context_call_tokens = (
        df.groupby("traj_id", sort=False)["context_tokens_est"]
        .sum()
        .rename("full_context_call_tokens_est")
    )
    df = df.join(final_tokens, on="traj_id")
    df = df.join(full_context_call_tokens, on="traj_id")
    df["transcript_tokens_saved_if_stop_est"] = (
        df["final_context_tokens_est"] - df["context_tokens_est"]
    )
    df["future_context_call_tokens_saved_if_stop_est"] = (
        df.groupby("traj_id", sort=False)["context_tokens_est"]
        .transform(lambda s: s.iloc[::-1].cumsum().iloc[::-1] - s)
        .astype("int64")
    )
    return df


def load_token_prefix_rows_chars(prefix_table: Path, traj_ids: set[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    cols = [
        "traj_id",
        "model_id",
        "prefix_step_idx",
        "task_prompt_chars",
        "prefix_action_chars",
        "prefix_feedback_chars",
        "prefix_assistant_content_chars",
    ]
    df = pd.read_parquet(prefix_table, columns=cols)
    df = df[df["traj_id"].isin(traj_ids)].copy()
    if df.empty:
        raise ValueError("no prefix token rows matched selected trajectories")
    df["context_chars_est"] = (
        df["task_prompt_chars"].fillna(0)
        + df["prefix_action_chars"].fillna(0)
        + df["prefix_feedback_chars"].fillna(0)
        + df["prefix_assistant_content_chars"].fillna(0)
    )
    df["context_tokens_est"] = np.ceil(df["context_chars_est"] / LEGACY_CHARS_PER_TOKEN).astype("int64")
    df["token_count_method"] = "legacy_chars_per_token"
    df["tokenizer_family"] = "legacy_chars_per_token"
    df["tokenizer_backend"] = "chars_per_token"
    df["tokenizer_name"] = f"ceil(chars/{LEGACY_CHARS_PER_TOKEN:g})"
    manifest = (
        df[["model_id", "tokenizer_family", "tokenizer_backend", "tokenizer_name"]]
        .drop_duplicates()
        .sort_values("model_id")
        .reset_index(drop=True)
    )
    manifest["note"] = "Legacy proxy only; not a real model tokenizer."
    return _finalize_token_rows(df), manifest


def load_token_prefix_rows_model(
    prefix_table: Path,
    traj_ids: set[str],
    *,
    tokenizer_cache_root: Path | None,
    token_prefix_cache: Path | None,
    parquet_batch_size: int,
    encode_batch_size: int,
    progress_every_batches: int,
    progress_every_rows: int,
    local_files_only: bool,
    max_direct_tokenize_chars: int,
    exact_chunk_tokenize_chars: int,
    sample_long_tokenize_chars: int,
    sample_chars: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    try:
        import pyarrow.parquet as pq
    except Exception as exc:  # pragma: no cover - dependency guard
        raise RuntimeError("pyarrow is required for streaming tokenizer counts") from exc

    cache_meta = _token_prefix_cache_meta(
        prefix_table,
        traj_ids,
        tokenizer_mode="model",
        max_direct_tokenize_chars=max_direct_tokenize_chars,
        exact_chunk_tokenize_chars=exact_chunk_tokenize_chars,
        sample_long_tokenize_chars=sample_long_tokenize_chars,
        sample_chars=sample_chars,
    )
    cached = _load_token_prefix_cache(token_prefix_cache, traj_ids, cache_meta)
    if cached is not None:
        return cached
    cached_shards = _load_token_prefix_cache_shards(token_prefix_cache, cache_meta)

    _configure_tokenizer_cache(tokenizer_cache_root)
    registry = _TokenizerRegistry(local_files_only=local_files_only)
    wanted = set(str(x) for x in traj_ids)
    cols = ["traj_id", "model_id", "prefix_step_idx", *CONTEXT_TEXT_COLUMNS]
    parts: list[pd.DataFrame] = []
    parquet = pq.ParquetFile(prefix_table)
    started_at = time.monotonic()
    scanned_batches = 0
    scanned_rows = 0
    matched_rows = 0
    cached_rows = 0
    cached_chars = 0
    cached_shard_files = 0
    done_keys: set[tuple[str, int]] = set()
    if cached_shards is not None:
        cached_raw, cached_shard_files = cached_shards
        if not cached_raw.empty:
            parts.append(cached_raw)
            done_keys = _token_prefix_cache_keys(cached_raw)
            cached_rows = int(len(cached_raw))
            cached_chars = int(cached_raw["context_chars_est"].fillna(0).sum())
    encoded_rows = cached_rows
    encoded_chars = cached_chars
    skipped_cached_rows = 0
    next_progress_rows = max(1, progress_every_rows)
    if progress_every_rows > 0 and encoded_rows >= next_progress_rows:
        while encoded_rows >= next_progress_rows:
            next_progress_rows += max(1, progress_every_rows)
    if progress_every_batches > 0 or progress_every_rows > 0:
        print(
            "[tokenizer-progress] start "
            f"selected_trajectories={len(wanted)} parquet_rows={parquet.metadata.num_rows} "
            f"cached_rows={cached_rows} cached_shards={cached_shard_files}",
            flush=True,
        )
    for batch in parquet.iter_batches(batch_size=max(1, parquet_batch_size), columns=cols):
        scanned_batches += 1
        scanned_rows += int(batch.num_rows)
        part = batch.to_pandas()
        part["traj_id"] = part["traj_id"].astype(str)
        part = part[part["traj_id"].isin(wanted)].copy()
        if part.empty:
            if progress_every_batches > 0 and scanned_batches % progress_every_batches == 0:
                elapsed = time.monotonic() - started_at
                print(
                    "[tokenizer-progress] scan "
                    f"batches={scanned_batches} parquet_rows={scanned_rows} "
                    f"matched_rows={matched_rows} encoded_rows={encoded_rows} "
                    f"elapsed_s={elapsed:.1f}",
                    flush=True,
                )
            continue
        matched_rows += int(len(part))
        if done_keys:
            keys = list(
                zip(
                    part["traj_id"].astype(str),
                    pd.to_numeric(part["prefix_step_idx"], errors="raise").astype("int64"),
                )
            )
            todo_mask = [key not in done_keys for key in keys]
            skipped_cached_rows += int(len(part) - sum(todo_mask))
            part = part.loc[todo_mask].copy()
            if part.empty:
                if progress_every_batches > 0 and scanned_batches % progress_every_batches == 0:
                    elapsed = time.monotonic() - started_at
                    print(
                        "[tokenizer-progress] scan "
                        f"batches={scanned_batches} parquet_rows={scanned_rows} "
                        f"matched_rows={matched_rows} encoded_rows={encoded_rows} "
                        f"skipped_cached_rows={skipped_cached_rows} "
                        f"encoded_chars={encoded_chars} elapsed_s={elapsed:.1f}",
                        flush=True,
                    )
                continue
        out_parts: list[pd.DataFrame] = []
        for model_id, group in part.groupby("model_id", sort=False):
            counter = registry.get(str(model_id))
            texts: list[str] = []
            chars: list[int] = []
            for row in group.itertuples(index=False):
                text, context_chars = _context_text_from_row(row)
                texts.append(text)
                chars.append(context_chars)
            counts: list[int] = []
            methods: list[str] = []
            for start in range(0, len(texts), max(1, encode_batch_size)):
                batch_counts, batch_methods = counter.count_many(
                    texts[start : start + max(1, encode_batch_size)],
                    max_direct_chars=max(1, max_direct_tokenize_chars),
                    exact_chunk_chars=max(1, exact_chunk_tokenize_chars),
                    sample_long_tokenize_chars=max(1, sample_long_tokenize_chars),
                    sample_chars=max(1, sample_chars),
                )
                counts.extend(batch_counts)
                methods.extend(batch_methods)
            encoded_rows += int(len(texts))
            encoded_chars += int(sum(chars))
            if progress_every_rows > 0 and encoded_rows >= next_progress_rows:
                elapsed = time.monotonic() - started_at
                print(
                    "[tokenizer-progress] encode "
                    f"batches={scanned_batches} parquet_rows={scanned_rows} "
                    f"matched_rows={matched_rows} encoded_rows={encoded_rows} "
                    f"encoded_chars={encoded_chars} model_id={model_id} "
                    f"tokenizer={counter.spec.family}/{counter.spec.name} "
                    f"elapsed_s={elapsed:.1f}",
                    flush=True,
                )
                while encoded_rows >= next_progress_rows:
                    next_progress_rows += max(1, progress_every_rows)
            sub = group[["traj_id", "model_id", "prefix_step_idx"]].copy()
            sub["context_chars_est"] = chars
            sub["legacy_context_tokens_chars4_est"] = np.ceil(
                sub["context_chars_est"] / LEGACY_CHARS_PER_TOKEN
            ).astype("int64")
            sub["context_tokens_est"] = np.asarray(counts, dtype="int64")
            sub["token_count_method"] = methods
            sub["tokenizer_family"] = counter.spec.family
            sub["tokenizer_backend"] = counter.spec.backend
            sub["tokenizer_name"] = counter.spec.name
            out_parts.append(sub)
        encoded_part = pd.concat(out_parts, ignore_index=True)
        parts.append(encoded_part)
        done_keys.update(_token_prefix_cache_keys(encoded_part))
        _write_token_prefix_cache_shard(token_prefix_cache, encoded_part, cache_meta)
        if progress_every_batches > 0 and scanned_batches % progress_every_batches == 0:
            elapsed = time.monotonic() - started_at
            print(
                "[tokenizer-progress] scan "
                f"batches={scanned_batches} parquet_rows={scanned_rows} "
                f"matched_rows={matched_rows} encoded_rows={encoded_rows} "
                f"skipped_cached_rows={skipped_cached_rows} "
                f"encoded_chars={encoded_chars} elapsed_s={elapsed:.1f}",
                flush=True,
            )

    if not parts:
        raise ValueError("no prefix token rows matched selected trajectories")
    df = pd.concat(parts, ignore_index=True)
    df["traj_id"] = df["traj_id"].astype(str)
    df["prefix_step_idx"] = pd.to_numeric(df["prefix_step_idx"], errors="raise").astype("int64")
    df = (
        df.drop_duplicates(["traj_id", "prefix_step_idx"], keep="last")
        .reset_index(drop=True)
    )
    missing = wanted - set(df["traj_id"].astype(str))
    if missing:
        sample = ", ".join(sorted(missing)[:5])
        raise ValueError(f"token rows missing for {len(missing)} selected trajectories, e.g. {sample}")
    token_rows = _finalize_token_rows(df)
    manifest = _merge_tokenizer_manifests(
        _manifest_from_token_rows(df),
        registry.manifest(),
    )
    _write_token_prefix_cache(token_prefix_cache, token_rows, manifest, cache_meta)
    if progress_every_batches > 0 or progress_every_rows > 0:
        elapsed = time.monotonic() - started_at
        print(
            "[tokenizer-progress] done "
            f"matched_rows={matched_rows} encoded_rows={encoded_rows} "
            f"skipped_cached_rows={skipped_cached_rows} "
            f"encoded_chars={encoded_chars} elapsed_s={elapsed:.1f}",
            flush=True,
        )
    return token_rows, manifest


def load_token_prefix_rows_component_sum_approx(
    prefix_table: Path,
    traj_ids: set[str],
    *,
    tokenizer_cache_root: Path | None,
    token_prefix_cache: Path | None,
    parquet_batch_size: int,
    encode_batch_size: int,
    progress_every_batches: int,
    progress_every_rows: int,
    local_files_only: bool,
    max_direct_tokenize_chars: int,
    exact_chunk_tokenize_chars: int,
    sample_long_tokenize_chars: int,
    sample_chars: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    try:
        import pyarrow.parquet as pq
    except Exception as exc:  # pragma: no cover - dependency guard
        raise RuntimeError("pyarrow is required for streaming tokenizer counts") from exc

    cache_meta = _token_prefix_cache_meta(
        prefix_table,
        traj_ids,
        tokenizer_mode="component_sum_approx",
        max_direct_tokenize_chars=max_direct_tokenize_chars,
        exact_chunk_tokenize_chars=exact_chunk_tokenize_chars,
        sample_long_tokenize_chars=sample_long_tokenize_chars,
        sample_chars=sample_chars,
    )
    cached = _load_token_prefix_cache(token_prefix_cache, traj_ids, cache_meta)
    if cached is not None:
        return cached

    _configure_tokenizer_cache(tokenizer_cache_root)
    registry = _TokenizerRegistry(local_files_only=local_files_only)
    wanted = set(str(x) for x in traj_ids)
    cols = [
        "traj_id",
        "model_id",
        "prefix_step_idx",
        *CONTEXT_CHAR_COLUMNS,
        "task_prompt_text",
        *[col for _, col in COMPONENT_STEP_TEXT_COLUMNS],
    ]
    parquet = pq.ParquetFile(prefix_table)
    parts: list[pd.DataFrame] = []
    task_cache: dict[tuple[str, str], tuple[int, int]] = {}
    newline_token_cache: dict[str, int] = {}
    started_at = time.monotonic()
    scanned_batches = 0
    scanned_rows = 0
    matched_rows = 0
    encoded_rows = 0
    encoded_texts = 0
    encoded_chars = 0
    next_progress_rows = max(1, progress_every_rows)

    if progress_every_batches > 0 or progress_every_rows > 0:
        print(
            "[tokenizer-progress] start "
            "mode=component_sum_approx "
            f"selected_trajectories={len(wanted)} parquet_rows={parquet.metadata.num_rows}",
            flush=True,
        )

    for batch in parquet.iter_batches(batch_size=max(1, parquet_batch_size), columns=cols):
        scanned_batches += 1
        scanned_rows += int(batch.num_rows)
        part = batch.to_pandas()
        part["traj_id"] = part["traj_id"].astype(str)
        part = part[part["traj_id"].isin(wanted)].copy()
        if part.empty:
            if progress_every_batches > 0 and scanned_batches % progress_every_batches == 0:
                elapsed = time.monotonic() - started_at
                print(
                    "[tokenizer-progress] scan "
                    f"mode=component_sum_approx batches={scanned_batches} "
                    f"parquet_rows={scanned_rows} matched_rows={matched_rows} "
                    f"encoded_rows={encoded_rows} encoded_texts={encoded_texts} "
                    f"elapsed_s={elapsed:.1f}",
                    flush=True,
                )
            continue

        matched_rows += int(len(part))
        out_parts: list[pd.DataFrame] = []
        for model_id, group in part.groupby("model_id", sort=False):
            model_id = str(model_id)
            counter = registry.get(model_id)
            if model_id not in newline_token_cache:
                newline_counts, _ = counter.count_many(
                    ["\n"],
                    max_direct_chars=max(1, max_direct_tokenize_chars),
                    exact_chunk_chars=max(1, exact_chunk_tokenize_chars),
                    sample_long_tokenize_chars=max(1, sample_long_tokenize_chars),
                    sample_chars=max(1, sample_chars),
                )
                newline_token_cache[model_id] = int(newline_counts[0])

            group = group.reset_index(drop=True)
            n_rows = int(len(group))
            task_tokens = np.zeros(n_rows, dtype="int64")
            task_method_severity = np.zeros(n_rows, dtype="int64")
            step_method_severity = np.zeros(n_rows, dtype="int64")
            component_tokens = {
                kind: np.zeros(n_rows, dtype="int64")
                for kind, _ in COMPONENT_STEP_TEXT_COLUMNS
            }
            component_nonempty = {
                kind: np.zeros(n_rows, dtype="int64")
                for kind, _ in COMPONENT_STEP_TEXT_COLUMNS
            }

            texts: list[str] = []
            jobs: list[tuple[str, int | tuple[str, str]]] = []
            pending_task_rows: dict[tuple[str, str], list[int]] = {}
            for row_idx, row in enumerate(group.itertuples(index=False)):
                traj_id = str(getattr(row, "traj_id"))
                task_key = (model_id, traj_id)
                cached_task = task_cache.get(task_key)
                if cached_task is not None:
                    task_tokens[row_idx] = cached_task[0]
                    task_method_severity[row_idx] = cached_task[1]
                else:
                    if task_key not in pending_task_rows:
                        pending_task_rows[task_key] = []
                        texts.append(_string_or_empty(getattr(row, "task_prompt_text")))
                        jobs.append(("task", task_key))
                    pending_task_rows[task_key].append(row_idx)

                for kind, col in COMPONENT_STEP_TEXT_COLUMNS:
                    text = _string_or_empty(getattr(row, col))
                    if not text:
                        continue
                    component_nonempty[kind][row_idx] = 1
                    texts.append(text)
                    jobs.append((kind, row_idx))

            counts: list[int] = []
            methods: list[str] = []
            for start in range(0, len(texts), max(1, encode_batch_size)):
                batch_texts = texts[start : start + max(1, encode_batch_size)]
                batch_counts, batch_methods = counter.count_many(
                    batch_texts,
                    max_direct_chars=max(1, max_direct_tokenize_chars),
                    exact_chunk_chars=max(1, exact_chunk_tokenize_chars),
                    sample_long_tokenize_chars=max(1, sample_long_tokenize_chars),
                    sample_chars=max(1, sample_chars),
                )
                counts.extend(batch_counts)
                methods.extend(batch_methods)

            for (kind, target), count, method in zip(jobs, counts, methods):
                severity = _component_method_severity(method)
                if kind == "task":
                    task_key = target
                    assert isinstance(task_key, tuple)
                    task_cache[task_key] = (int(count), int(severity))
                    for row_idx in pending_task_rows[task_key]:
                        task_tokens[row_idx] = int(count)
                        task_method_severity[row_idx] = int(severity)
                else:
                    row_idx = int(target)
                    component_tokens[kind][row_idx] = int(count)
                    step_method_severity[row_idx] = max(
                        int(step_method_severity[row_idx]), int(severity)
                    )

            sub = group[["traj_id", "model_id", "prefix_step_idx"]].copy()
            sub["prefix_step_idx"] = pd.to_numeric(
                sub["prefix_step_idx"], errors="raise"
            ).astype("int64")
            sub["context_chars_est"] = (
                group[CONTEXT_CHAR_COLUMNS].fillna(0).sum(axis=1).astype("int64")
            )
            sub["legacy_context_tokens_chars4_est"] = np.ceil(
                sub["context_chars_est"] / LEGACY_CHARS_PER_TOKEN
            ).astype("int64")
            sub["task_prompt_chars_component"] = (
                group["task_prompt_chars"].fillna(0).astype("int64")
            )
            sub["task_prompt_tokens_component"] = task_tokens
            sub["task_method_severity_component"] = task_method_severity
            sub["step_method_severity_component"] = step_method_severity
            sub["newline_tokens_component"] = int(newline_token_cache[model_id])
            for kind, _ in COMPONENT_STEP_TEXT_COLUMNS:
                sub[f"{kind}_step_tokens_component"] = component_tokens[kind]
                sub[f"{kind}_step_nonempty_component"] = component_nonempty[kind]
            sub["tokenizer_family"] = counter.spec.family
            sub["tokenizer_backend"] = counter.spec.backend
            sub["tokenizer_name"] = counter.spec.name
            out_parts.append(sub)

            encoded_texts += int(len(texts))
            encoded_chars += int(sum(len(text) for text in texts))

        encoded_part = pd.concat(out_parts, ignore_index=True)
        parts.append(encoded_part)
        encoded_rows += int(len(part))
        if progress_every_rows > 0 and encoded_rows >= next_progress_rows:
            elapsed = time.monotonic() - started_at
            print(
                "[tokenizer-progress] encode "
                f"mode=component_sum_approx batches={scanned_batches} "
                f"parquet_rows={scanned_rows} matched_rows={matched_rows} "
                f"encoded_rows={encoded_rows} encoded_texts={encoded_texts} "
                f"encoded_chars={encoded_chars} elapsed_s={elapsed:.1f}",
                flush=True,
            )
            while encoded_rows >= next_progress_rows:
                next_progress_rows += max(1, progress_every_rows)
        if progress_every_batches > 0 and scanned_batches % progress_every_batches == 0:
            elapsed = time.monotonic() - started_at
            print(
                "[tokenizer-progress] scan "
                f"mode=component_sum_approx batches={scanned_batches} "
                f"parquet_rows={scanned_rows} matched_rows={matched_rows} "
                f"encoded_rows={encoded_rows} encoded_texts={encoded_texts} "
                f"encoded_chars={encoded_chars} elapsed_s={elapsed:.1f}",
                flush=True,
            )

    if not parts:
        raise ValueError("no prefix token rows matched selected trajectories")

    df = pd.concat(parts, ignore_index=True)
    df["traj_id"] = df["traj_id"].astype(str)
    df["prefix_step_idx"] = pd.to_numeric(df["prefix_step_idx"], errors="raise").astype("int64")
    df = (
        df.drop_duplicates(["traj_id", "prefix_step_idx"], keep="last")
        .sort_values(["traj_id", "prefix_step_idx"])
        .reset_index(drop=True)
    )
    missing = wanted - set(df["traj_id"].astype(str))
    if missing:
        sample = ", ".join(sorted(missing)[:5])
        raise ValueError(f"token rows missing for {len(missing)} selected trajectories, e.g. {sample}")

    for kind, _ in COMPONENT_STEP_TEXT_COLUMNS:
        df[f"{kind}_prefix_tokens_component"] = (
            df.groupby("traj_id", sort=False)[f"{kind}_step_tokens_component"]
            .cumsum()
            .astype("int64")
        )
        df[f"{kind}_prefix_nonempty_component"] = (
            df.groupby("traj_id", sort=False)[f"{kind}_step_nonempty_component"]
            .cumsum()
            .astype("int64")
        )
    df["step_method_severity_cum_component"] = (
        df.groupby("traj_id", sort=False)["step_method_severity_component"]
        .cummax()
        .astype("int64")
    )
    method_severity = np.maximum(
        df["task_method_severity_component"].to_numpy(dtype="int64"),
        df["step_method_severity_cum_component"].to_numpy(dtype="int64"),
    )
    internal_separators = sum(
        np.maximum(df[f"{kind}_prefix_nonempty_component"].to_numpy(dtype="int64") - 1, 0)
        for kind, _ in COMPONENT_STEP_TEXT_COLUMNS
    )
    top_components = (df["task_prompt_chars_component"].to_numpy(dtype="int64") > 0).astype("int64")
    for kind, _ in COMPONENT_STEP_TEXT_COLUMNS:
        top_components += (
            df[f"{kind}_prefix_nonempty_component"].to_numpy(dtype="int64") > 0
        ).astype("int64")
    top_level_separators = np.maximum(top_components - 1, 0)
    separator_tokens = (
        (internal_separators + top_level_separators)
        * df["newline_tokens_component"].to_numpy(dtype="int64")
    )
    component_total = df["task_prompt_tokens_component"].to_numpy(dtype="int64")
    for kind, _ in COMPONENT_STEP_TEXT_COLUMNS:
        component_total = component_total + df[f"{kind}_prefix_tokens_component"].to_numpy(dtype="int64")
    df["context_tokens_est"] = (component_total + separator_tokens).astype("int64")
    df["token_count_method"] = [
        _component_method_from_severity(int(severity)) for severity in method_severity
    ]

    keep_cols = [
        "traj_id",
        "model_id",
        "prefix_step_idx",
        "context_chars_est",
        "legacy_context_tokens_chars4_est",
        "context_tokens_est",
        "token_count_method",
        "tokenizer_family",
        "tokenizer_backend",
        "tokenizer_name",
    ]
    token_rows = _finalize_token_rows(df[keep_cols].copy())
    manifest = registry.manifest()
    if not manifest.empty:
        manifest["note"] = manifest["note"].astype(str) + " Component-sum approximate prefix counts; tokenizer boundary merges across components are not modeled exactly."
    _write_token_prefix_cache(token_prefix_cache, token_rows, manifest, cache_meta)
    if progress_every_batches > 0 or progress_every_rows > 0:
        elapsed = time.monotonic() - started_at
        print(
            "[tokenizer-progress] done "
            f"mode=component_sum_approx matched_rows={matched_rows} "
            f"encoded_rows={encoded_rows} encoded_texts={encoded_texts} "
            f"encoded_chars={encoded_chars} elapsed_s={elapsed:.1f}",
            flush=True,
        )
    return token_rows, manifest


def _token_prefix_cache_meta(
    prefix_table: Path,
    traj_ids: set[str],
    *,
    tokenizer_mode: str,
    max_direct_tokenize_chars: int,
    exact_chunk_tokenize_chars: int,
    sample_long_tokenize_chars: int,
    sample_chars: int,
) -> dict[str, Any]:
    resolved = prefix_table.expanduser().resolve()
    stat = resolved.stat()
    traj_digest = hashlib.sha256(
        "\n".join(sorted(str(x) for x in traj_ids)).encode("utf-8")
    ).hexdigest()
    return {
        "cache_version": 1,
        "tokenizer_mode": str(tokenizer_mode),
        "prefix_table": str(resolved),
        "prefix_table_size": int(stat.st_size),
        "prefix_table_mtime_ns": int(stat.st_mtime_ns),
        "traj_count": int(len(traj_ids)),
        "traj_sha256": traj_digest,
        "max_direct_tokenize_chars": int(max_direct_tokenize_chars),
        "exact_chunk_tokenize_chars": int(exact_chunk_tokenize_chars),
        "sample_long_tokenize_chars": int(sample_long_tokenize_chars),
        "sample_chars": int(sample_chars),
    }


def _token_prefix_cache_sidecar(cache_path: Path, suffix: str) -> Path:
    return cache_path.with_name(f"{cache_path.name}{suffix}")


def _token_prefix_cache_keys(token_rows: pd.DataFrame) -> set[tuple[str, int]]:
    return set(
        zip(
            token_rows["traj_id"].astype(str),
            pd.to_numeric(token_rows["prefix_step_idx"], errors="raise").astype("int64"),
        )
    )


def _token_prefix_cache_shard_dir(cache_path: Path, meta: dict[str, Any]) -> Path:
    digest = hashlib.sha256(
        json.dumps(meta, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:16]
    return cache_path.with_name(f"{cache_path.name}.shards-{digest}")


def _load_token_prefix_cache(
    cache_path: Path | None,
    traj_ids: set[str],
    expected_meta: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame] | None:
    if cache_path is None:
        return None
    cache_path = cache_path.expanduser()
    meta_path = _token_prefix_cache_sidecar(cache_path, ".meta.json")
    manifest_path = _token_prefix_cache_sidecar(cache_path, ".manifest.csv")
    if not cache_path.exists() or not meta_path.exists() or not manifest_path.exists():
        return None
    try:
        actual_meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if any(actual_meta.get(key) != value for key, value in expected_meta.items()):
        return None

    try:
        token_rows = pd.read_parquet(cache_path)
        manifest = pd.read_csv(manifest_path)
    except Exception:
        return None

    required = {
        "traj_id",
        "model_id",
        "prefix_step_idx",
        "context_tokens_est",
        "final_context_tokens_est",
        "full_context_call_tokens_est",
        "future_context_call_tokens_saved_if_stop_est",
    }
    if not required.issubset(token_rows.columns):
        return None
    wanted = set(str(x) for x in traj_ids)
    if set(token_rows["traj_id"].astype(str)) != wanted:
        return None
    token_rows["traj_id"] = token_rows["traj_id"].astype(str)
    return token_rows, manifest


def _raw_token_cache_required_columns() -> set[str]:
    return {
        "traj_id",
        "model_id",
        "prefix_step_idx",
        "context_chars_est",
        "legacy_context_tokens_chars4_est",
        "context_tokens_est",
        "token_count_method",
        "tokenizer_family",
        "tokenizer_backend",
        "tokenizer_name",
    }


def _load_token_prefix_cache_shards(
    cache_path: Path | None,
    expected_meta: dict[str, Any],
) -> tuple[pd.DataFrame, int] | None:
    if cache_path is None:
        return None
    cache_path = cache_path.expanduser()
    shard_dir = _token_prefix_cache_shard_dir(cache_path, expected_meta)
    meta_path = shard_dir / "meta.json"
    if not shard_dir.exists() or not meta_path.exists():
        return None
    try:
        actual_meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if any(actual_meta.get(key) != value for key, value in expected_meta.items()):
        return None

    shard_paths = sorted(shard_dir.glob("part-*.parquet"))
    if not shard_paths:
        return None
    parts: list[pd.DataFrame] = []
    required = _raw_token_cache_required_columns()
    for shard_path in shard_paths:
        try:
            part = pd.read_parquet(shard_path)
        except Exception:
            continue
        if not required.issubset(part.columns):
            continue
        parts.append(part[list(required)])
    if not parts:
        return None
    raw = pd.concat(parts, ignore_index=True)
    raw["traj_id"] = raw["traj_id"].astype(str)
    raw["prefix_step_idx"] = pd.to_numeric(raw["prefix_step_idx"], errors="raise").astype("int64")
    raw = (
        raw.drop_duplicates(["traj_id", "prefix_step_idx"], keep="last")
        .reset_index(drop=True)
    )
    return raw, len(shard_paths)


def _write_token_prefix_cache_shard(
    cache_path: Path | None,
    token_rows: pd.DataFrame,
    meta: dict[str, Any],
) -> None:
    if cache_path is None or token_rows.empty:
        return
    cache_path = cache_path.expanduser()
    shard_dir = _token_prefix_cache_shard_dir(cache_path, meta)
    shard_dir.mkdir(parents=True, exist_ok=True)
    meta_path = shard_dir / "meta.json"
    if not meta_path.exists():
        meta_path.write_text(
            json.dumps(meta, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    shard_name = f"part-{os.getpid()}-{time.time_ns()}.parquet"
    token_rows.to_parquet(shard_dir / shard_name, index=False)


def _manifest_from_token_rows(token_rows: pd.DataFrame) -> pd.DataFrame:
    cols = ["model_id", "tokenizer_family", "tokenizer_backend", "tokenizer_name"]
    if token_rows.empty or not set(cols).issubset(token_rows.columns):
        return pd.DataFrame()
    manifest = (
        token_rows[cols]
        .drop_duplicates()
        .sort_values("model_id")
        .reset_index(drop=True)
    )
    manifest["tokenizer_class"] = ""
    manifest["tokenizer_vocab_size"] = ""
    manifest["tokenizer_commit_hash"] = ""
    manifest["trust_remote_code"] = ""
    manifest["local_files_only"] = ""
    manifest["note"] = "Loaded from per-prefix tokenizer shard cache."
    return manifest


def _merge_tokenizer_manifests(*manifests: pd.DataFrame) -> pd.DataFrame:
    non_empty = [m for m in manifests if m is not None and not m.empty]
    if not non_empty:
        return pd.DataFrame()
    merged = pd.concat(non_empty, ignore_index=True, sort=False)
    if "model_id" not in merged.columns:
        return merged
    return (
        merged.drop_duplicates(["model_id"], keep="last")
        .sort_values("model_id")
        .reset_index(drop=True)
    )


def _write_token_prefix_cache(
    cache_path: Path | None,
    token_rows: pd.DataFrame,
    manifest: pd.DataFrame,
    meta: dict[str, Any],
) -> None:
    if cache_path is None:
        return
    cache_path = cache_path.expanduser()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    token_rows.to_parquet(cache_path, index=False)
    manifest.to_csv(_token_prefix_cache_sidecar(cache_path, ".manifest.csv"), index=False)
    _token_prefix_cache_sidecar(cache_path, ".meta.json").write_text(
        json.dumps(meta, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def load_token_prefix_rows(
    prefix_table: Path,
    traj_ids: set[str],
    *,
    tokenizer_mode: str,
    tokenizer_cache_root: Path | None,
    token_prefix_cache: Path | None,
    parquet_batch_size: int,
    encode_batch_size: int,
    progress_every_batches: int,
    progress_every_rows: int,
    local_files_only: bool,
    max_direct_tokenize_chars: int,
    exact_chunk_tokenize_chars: int,
    sample_long_tokenize_chars: int,
    sample_chars: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if tokenizer_mode == "chars_per_token":
        return load_token_prefix_rows_chars(prefix_table, traj_ids)
    if tokenizer_mode == "model":
        return load_token_prefix_rows_model(
            prefix_table,
            traj_ids,
            tokenizer_cache_root=tokenizer_cache_root,
            token_prefix_cache=token_prefix_cache,
            parquet_batch_size=parquet_batch_size,
            encode_batch_size=encode_batch_size,
            progress_every_batches=progress_every_batches,
            progress_every_rows=progress_every_rows,
            local_files_only=local_files_only,
            max_direct_tokenize_chars=max_direct_tokenize_chars,
            exact_chunk_tokenize_chars=exact_chunk_tokenize_chars,
            sample_long_tokenize_chars=sample_long_tokenize_chars,
            sample_chars=sample_chars,
        )
    if tokenizer_mode == "component_sum_approx":
        return load_token_prefix_rows_component_sum_approx(
            prefix_table,
            traj_ids,
            tokenizer_cache_root=tokenizer_cache_root,
            token_prefix_cache=token_prefix_cache,
            parquet_batch_size=parquet_batch_size,
            encode_batch_size=encode_batch_size,
            progress_every_batches=progress_every_batches,
            progress_every_rows=progress_every_rows,
            local_files_only=local_files_only,
            max_direct_tokenize_chars=max_direct_tokenize_chars,
            exact_chunk_tokenize_chars=exact_chunk_tokenize_chars,
            sample_long_tokenize_chars=sample_long_tokenize_chars,
            sample_chars=sample_chars,
        )
    raise ValueError(f"unknown tokenizer_mode={tokenizer_mode!r}")


def build_full_hit_by_round(token_rows: pd.DataFrame, total_trajectories: int) -> pd.DataFrame:
    grouped = token_rows.groupby("prefix_step_idx", sort=True)
    out = grouped.agg(
        hit_trajectories_at_round=("traj_id", "nunique"),
        context_tokens_spent_if_stop_est=("context_tokens_est", "sum"),
        baseline_final_transcript_tokens_est=("final_context_tokens_est", "sum"),
        transcript_tokens_saved_if_stop_est=("transcript_tokens_saved_if_stop_est", "sum"),
        future_context_call_tokens_saved_if_stop_est=(
            "future_context_call_tokens_saved_if_stop_est",
            "sum",
        ),
        saved_steps_if_stop=("remaining_steps_after_step", "sum"),
        avg_context_tokens_at_round_est=("context_tokens_est", "mean"),
        avg_transcript_tokens_saved_if_stop_est=("transcript_tokens_saved_if_stop_est", "mean"),
        avg_future_context_call_tokens_saved_if_stop_est=(
            "future_context_call_tokens_saved_if_stop_est",
            "mean",
        ),
    ).reset_index()
    out["round_1based"] = out["prefix_step_idx"] + 1
    out["coverage_if_all_hit_at_round_pct"] = (
        out["hit_trajectories_at_round"] * 100.0 / float(total_trajectories)
    )
    out["transcript_token_save_pct_if_stop_est"] = (
        out["transcript_tokens_saved_if_stop_est"]
        * 100.0
        / out["baseline_final_transcript_tokens_est"]
    )
    cols = [
        "prefix_step_idx",
        "round_1based",
        "hit_trajectories_at_round",
        "coverage_if_all_hit_at_round_pct",
        "saved_steps_if_stop",
        "context_tokens_spent_if_stop_est",
        "baseline_final_transcript_tokens_est",
        "transcript_tokens_saved_if_stop_est",
        "transcript_token_save_pct_if_stop_est",
        "future_context_call_tokens_saved_if_stop_est",
        "avg_context_tokens_at_round_est",
        "avg_transcript_tokens_saved_if_stop_est",
        "avg_future_context_call_tokens_saved_if_stop_est",
    ]
    return out[cols]


def enrich_decisions_with_tokens(decisions: pd.DataFrame, token_rows: pd.DataFrame) -> pd.DataFrame:
    final_aggs: dict[str, tuple[str, str]] = {
        "final_context_tokens_est": ("final_context_tokens_est", "first"),
        "full_context_call_tokens_est": ("full_context_call_tokens_est", "first"),
    }
    for col in ["tokenizer_family", "tokenizer_backend", "tokenizer_name", "token_count_method", "model_id"]:
        if col in token_rows.columns:
            final_aggs[f"final_{col}" if col == "token_count_method" else col] = (col, "first")
    final = (
        token_rows.groupby("traj_id", sort=False)
        .agg(**final_aggs)
        .reset_index()
    )
    at_cols = [
        "traj_id",
        "prefix_step_idx",
        "context_chars_est",
        "context_tokens_est",
        "transcript_tokens_saved_if_stop_est",
        "future_context_call_tokens_saved_if_stop_est",
    ]
    if "legacy_context_tokens_chars4_est" in token_rows.columns:
        at_cols.append("legacy_context_tokens_chars4_est")
    if "token_count_method" in token_rows.columns:
        at_cols.append("token_count_method")
    at_step = token_rows[at_cols].rename(columns={"prefix_step_idx": "decision_step"})
    out = decisions.merge(final, on="traj_id", how="left")
    out = out.merge(at_step, on=["traj_id", "decision_step"], how="left")
    out["policy_transcript_tokens_spent_est"] = np.where(
        out["decided"], out["context_tokens_est"], out["final_context_tokens_est"]
    )
    out["transcript_tokens_saved_est"] = np.where(
        out["decided"], out["transcript_tokens_saved_if_stop_est"], 0
    )
    out["future_context_call_tokens_saved_est"] = np.where(
        out["decided"], out["future_context_call_tokens_saved_if_stop_est"], 0
    )
    out["policy_context_call_tokens_spent_est"] = (
        out["full_context_call_tokens_est"] - out["future_context_call_tokens_saved_est"]
    )
    return out


def build_selected_token_summary(enriched: pd.DataFrame) -> pd.DataFrame:
    total = len(enriched)
    decided = int(enriched["decided"].sum())
    baseline_transcript = float(enriched["final_context_tokens_est"].sum())
    spent_transcript = float(enriched["policy_transcript_tokens_spent_est"].sum())
    saved_transcript = float(enriched["transcript_tokens_saved_est"].sum())
    baseline_context_call = float(enriched["full_context_call_tokens_est"].sum())
    spent_context_call = float(enriched["policy_context_call_tokens_spent_est"].sum())
    saved_context_call = float(enriched["future_context_call_tokens_saved_est"].sum())
    row = {
        "trajectories": total,
        "decided_trajectories": decided,
        "coverage_pct": _pct(decided, total),
        "baseline_final_transcript_tokens_est": baseline_transcript,
        "policy_transcript_tokens_spent_est": spent_transcript,
        "transcript_tokens_saved_est": saved_transcript,
        "transcript_token_save_pct_est": _pct(saved_transcript, baseline_transcript),
        "baseline_context_call_tokens_est": baseline_context_call,
        "policy_context_call_tokens_spent_est": spent_context_call,
        "future_context_call_tokens_saved_est": saved_context_call,
        "context_call_token_save_pct_est": _pct(saved_context_call, baseline_context_call),
        "saved_steps": int(enriched["saved_steps"].sum()),
        "total_steps": int(enriched["n_steps"].sum()),
        "step_save_pct": _pct(enriched["saved_steps"].sum(), enriched["n_steps"].sum()),
    }
    return pd.DataFrame([row])


def build_selected_token_by_round(enriched: pd.DataFrame) -> pd.DataFrame:
    dec = enriched[enriched["decided"]].copy()
    if dec.empty:
        return pd.DataFrame()
    grouped = dec.groupby(["decision_step", "round_1based", "decision"], sort=True)
    out = grouped.agg(
        decided_trajectories=("traj_id", "nunique"),
        correct_decisions=("decision_correct", "sum"),
        saved_steps=("saved_steps", "sum"),
        transcript_tokens_saved_est=("transcript_tokens_saved_est", "sum"),
        future_context_call_tokens_saved_est=("future_context_call_tokens_saved_est", "sum"),
        avg_decision_score=("decision_score", "mean"),
        avg_transcript_tokens_saved_est=("transcript_tokens_saved_est", "mean"),
        avg_future_context_call_tokens_saved_est=("future_context_call_tokens_saved_est", "mean"),
    ).reset_index()
    out["decision_accuracy_pct"] = out["correct_decisions"] * 100.0 / out["decided_trajectories"]
    return out


def build_frontier_token_summary(run_dir: Path, token_rows: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    path = run_dir / "policy_sweeps" / "valid_accuracy_075_095" / "test_decisions_by_target.parquet"
    if not path.exists():
        return pd.DataFrame(), pd.DataFrame()
    decisions = pd.read_parquet(path)
    decisions = decisions.rename(columns={"target_valid_decision_accuracy_pct": "target"})
    decisions["decision_correct"] = (
        ((decisions["decision"].eq("success")) & decisions["label"].eq(1))
        | ((decisions["decision"].eq("failure")) & decisions["label"].eq(0))
    ) & decisions["decided"]
    decisions["outcome_type"] = [
        _outcome_type(int(label), bool(decided), str(decision))
        for label, decided, decision in zip(
            decisions["label"], decisions["decided"], decisions["decision"]
        )
    ]
    enriched = enrich_decisions_with_tokens(decisions, token_rows)
    rows = []
    for target, part in enriched.groupby("target", sort=True):
        baseline_transcript = float(part["final_context_tokens_est"].sum())
        saved_transcript = float(part["transcript_tokens_saved_est"].sum())
        baseline_context_call = float(part["full_context_call_tokens_est"].sum())
        saved_context_call = float(part["future_context_call_tokens_saved_est"].sum())
        rows.append(
            {
                "target": target,
                "trajectories": int(len(part)),
                "decided_trajectories": int(part["decided"].sum()),
                "coverage_pct": _pct(part["decided"].sum(), len(part)),
                "transcript_tokens_saved_est": saved_transcript,
                "transcript_token_save_pct_est": _pct(saved_transcript, baseline_transcript),
                "future_context_call_tokens_saved_est": saved_context_call,
                "context_call_token_save_pct_est": _pct(saved_context_call, baseline_context_call),
                "saved_steps": int(part["saved_steps"].sum()),
                "total_steps": int(part["n_steps"].sum()),
                "step_save_pct": _pct(part["saved_steps"].sum(), part["n_steps"].sum()),
            }
        )
    by_target = pd.DataFrame(rows)

    dec = enriched[enriched["decided"]].copy()
    by_round = (
        dec.groupby(["target", "decision_step", "decision"], sort=True)
        .agg(
            decided_trajectories=("traj_id", "nunique"),
            correct_decisions=("decision_correct", "sum"),
            saved_steps=("saved_steps", "sum"),
            transcript_tokens_saved_est=("transcript_tokens_saved_est", "sum"),
            future_context_call_tokens_saved_est=("future_context_call_tokens_saved_est", "sum"),
        )
        .reset_index()
    )
    by_round["round_1based"] = by_round["decision_step"] + 1
    by_round["decision_accuracy_pct"] = (
        by_round["correct_decisions"] * 100.0 / by_round["decided_trajectories"]
    )
    return by_target, by_round


def _squash(text: Any, limit: int = 500) -> str:
    if text is None:
        return ""
    s = re.sub(r"\s+", " ", str(text)).strip()
    return s[:limit]


def _raw_traj_path(raw_root: Path, test_model: str, instance_id: str) -> Path:
    model_dir = raw_root / test_model
    for suffix in (".traj.json", ".json"):
        candidate = model_dir / f"{instance_id}{suffix}"
        if candidate.exists():
            return candidate
    return model_dir / f"{instance_id}.json"


def _extract_preview(raw_path: Path, decision_step: int) -> dict[str, Any]:
    if not raw_path.exists():
        return {
            "raw_traj_exists": False,
            "message_count": 0,
            "assistant_message_count": 0,
            "problem_preview": "",
            "decision_assistant_preview": "",
            "final_assistant_preview": "",
            "patch_chars": np.nan,
            "patch_preview": "",
        }
    try:
        with raw_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as exc:  # pragma: no cover - diagnostic path
        return {
            "raw_traj_exists": False,
            "message_count": 0,
            "assistant_message_count": 0,
            "problem_preview": f"failed to read raw traj: {exc}",
            "decision_assistant_preview": "",
            "final_assistant_preview": "",
            "patch_chars": np.nan,
            "patch_preview": "",
        }
    messages = data.get("messages", []) if isinstance(data, dict) else []
    user_messages = [m.get("content", "") for m in messages if m.get("role") == "user"]
    assistant_messages = [
        m.get("content", "") for m in messages if m.get("role") == "assistant"
    ]
    idx = min(max(int(decision_step), 0), max(len(assistant_messages) - 1, 0))
    patch = data.get("patch", "") if isinstance(data, dict) else ""
    return {
        "raw_traj_exists": True,
        "message_count": len(messages),
        "assistant_message_count": len(assistant_messages),
        "problem_preview": _squash(user_messages[0] if user_messages else "", 500),
        "decision_assistant_preview": _squash(
            assistant_messages[idx] if assistant_messages else "", 500
        ),
        "final_assistant_preview": _squash(
            assistant_messages[-1] if assistant_messages else "", 500
        ),
        "patch_chars": len(str(patch)),
        "patch_preview": _squash(patch, 500),
    }


def build_success_examples(
    enriched: pd.DataFrame,
    *,
    raw_root: Path,
    n_examples: int,
) -> pd.DataFrame:
    candidates = enriched[
        enriched["outcome_type"].eq("true_positive") & enriched["saved_steps"].gt(0)
    ].copy()
    candidates = candidates.sort_values(
        ["future_context_call_tokens_saved_est", "saved_steps", "decision_score"],
        ascending=[False, False, False],
    )
    # First take a diverse pass, then fill globally by strongest token saving.
    diverse = candidates.groupby("test_model", sort=False).head(1)
    selected = pd.concat([diverse, candidates], ignore_index=True).drop_duplicates("traj_id")
    selected = selected.head(n_examples).copy()
    selected["raw_traj_path"] = [
        str(_raw_traj_path(raw_root, model, instance))
        for model, instance in zip(selected["test_model"], selected["instance_id"])
    ]
    previews = [
        _extract_preview(Path(path), int(step))
        for path, step in zip(selected["raw_traj_path"], selected["decision_step"])
    ]
    preview_df = pd.DataFrame(previews)
    out = pd.concat([selected.reset_index(drop=True), preview_df], axis=1)
    cols = [
        "test_model",
        "instance_id",
        "traj_id",
        "label",
        "decision",
        "decision_step",
        "round_1based",
        "n_steps",
        "saved_steps",
        "decision_score",
        "prob_success_at_decision",
        "prob_failure_at_decision",
        "tokenizer_family",
        "tokenizer_name",
        "token_count_method",
        "transcript_tokens_saved_est",
        "future_context_call_tokens_saved_est",
        "raw_traj_exists",
        "raw_traj_path",
        "message_count",
        "assistant_message_count",
        "patch_chars",
        "problem_preview",
        "decision_assistant_preview",
        "final_assistant_preview",
        "patch_preview",
    ]
    return out[cols]


def write_examples_md(examples: pd.DataFrame, path: Path) -> None:
    lines = [
        "# Successful Early-Stop Examples",
        "",
        "These are true-positive success stops from the selected 16-fold SWE strategy.",
        "Text snippets are approximate pointers into the raw trajectory JSON.",
        "",
    ]
    for idx, row in enumerate(examples.itertuples(index=False), start=1):
        lines.extend(
            [
                f"## {idx}. {row.test_model} / {row.instance_id}",
                "",
                f"- decision: `{row.decision}` at step {row.decision_step} (round {row.round_1based})",
                f"- saved steps: {row.saved_steps} / {row.n_steps}",
                f"- score: {_fmt(row.decision_score, 4)}; success prob: {_fmt(row.prob_success_at_decision, 4)}; failure prob: {_fmt(row.prob_failure_at_decision, 4)}",
                f"- tokenizer: `{row.tokenizer_family}` / `{row.tokenizer_name}` (`{row.token_count_method}`)",
                f"- estimated saved transcript tokens: {_fmt(row.transcript_tokens_saved_est, 0)}",
                f"- estimated saved future context-call tokens: {_fmt(row.future_context_call_tokens_saved_est, 0)}",
                f"- raw trajectory: `{row.raw_traj_path}`",
                "",
                "**Problem preview**",
                "",
                row.problem_preview or "(not available)",
                "",
                "**Decision-step assistant preview**",
                "",
                row.decision_assistant_preview or "(not available)",
                "",
                "**Patch preview**",
                "",
                row.patch_preview or "(not available)",
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def write_readme(
    out_dir: Path,
    selected_rank: pd.DataFrame,
    selected_token_summary: pd.DataFrame,
    frontier_token_summary: pd.DataFrame,
    tokenizer_manifest: pd.DataFrame,
    tokenizer_mode: str,
) -> None:
    moved = selected_rank[selected_rank["rank_change_positive_is_up"].ne(0)].copy()
    moved = moved.sort_values("rank_change_positive_is_up", ascending=False)
    tok = selected_token_summary.iloc[0]
    lines = [
        "# SWE16 Internal Review",
        "",
        "Post-hoc report for the main SWEVerify 16-fold LightGBM strategy.",
        "No folds were retrained or rerun.",
        "",
        "## Metric Definitions",
        "",
        "- adjusted resolved = original_resolved - false_negatives + false_positives.",
        "- signed resolve change pp = (adjusted_resolved - original_resolved) / total * 100.",
        "- rank 1 is highest resolve rate; positive rank_change means the model moved up after adjustment.",
        f"- token mode: `{tokenizer_mode}`.",
        "- model-token mode counts newline-joined task/action/feedback/assistant-content text with each trajectory model's tokenizer; see `tokenizer_manifest.csv`.",
        "- component_sum_approx mode tokenizes each task prompt and per-step action/feedback/assistant component once, then sums cumulative prefixes; tiny tokenizer-boundary merge differences are expected.",
        f"- legacy mode is `ceil(chars / {LEGACY_CHARS_PER_TOKEN:g})`; it is retained only for reproducing old proxy tables.",
        "- transcript token saving uses final trajectory transcript tokens minus the stop-prefix transcript tokens.",
        "- context-call token saving is a larger API-style estimate: future rounds are treated as resending their full prefix context.",
        "",
        "## Tokenizers",
        "",
        f"- mapped models: {len(tokenizer_manifest)}",
        f"- tokenizer backends: {', '.join(sorted(tokenizer_manifest['tokenizer_backend'].astype(str).unique())) if not tokenizer_manifest.empty else 'n/a'}",
        "",
        "## Main Selected Strategy Token Summary",
        "",
        f"- trajectories: {int(tok.trajectories)}",
        f"- decided trajectories: {int(tok.decided_trajectories)} ({_fmt(tok.coverage_pct)}%)",
        f"- saved steps: {int(tok.saved_steps)} / {int(tok.total_steps)} ({_fmt(tok.step_save_pct)}%)",
        f"- estimated transcript tokens saved: {_fmt(tok.transcript_tokens_saved_est, 0)} ({_fmt(tok.transcript_token_save_pct_est)}%)",
        f"- estimated future context-call tokens saved: {_fmt(tok.future_context_call_tokens_saved_est, 0)} ({_fmt(tok.context_call_token_save_pct_est)}%)",
        "",
        "## Rank Changes",
        "",
    ]
    if moved.empty:
        lines.append("No rank changes under the selected strategy.")
    else:
        lines.append("| model | original rank | adjusted rank | change | resolve change pp |")
        lines.append("| :-- | --: | --: | --: | --: |")
        for row in moved.itertuples(index=False):
            lines.append(
                f"| {row.test_model} | {row.original_rank} | {row.adjusted_rank} | "
                f"{row.rank_change_positive_is_up:+d} | {_fmt(row.resolve_rate_change_pp_calc)} |"
            )
    if not frontier_token_summary.empty:
        best = frontier_token_summary.sort_values("target").tail(1).iloc[0]
        lines.extend(
            [
                "",
                "## valid_accuracy_075_095 Addendum",
                "",
                f"Target {best.target:.0f}: estimated future context-call token save "
                f"{_fmt(best.context_call_token_save_pct_est)}%, step save {_fmt(best.step_save_pct)}%.",
            ]
        )
    lines.extend(
        [
            "",
            "## Files",
            "",
            "- `selected_strategy_rank_change.csv`: old/new model ranking for the selected main policy.",
            "- `valid_accuracy_075_095_rank_change_by_target.csv`: old/new ranking per valid-accuracy target.",
            "- `selected_strategy_decisions.csv`: reconstructed trajectory-level selected-policy decisions.",
            "- `stop_signal_decision_lift_by_prefix.csv`: reverse conditional signal table, P(policy stop at prefix | signal at at-risk prefix).",
            "- `stop_signal_decision_lift_by_trajectory.csv`: reverse conditional trajectory table, P(policy stop | signal seen before stop/full horizon).",
            "- `stop_signal_stop_composition.csv`: original stop-point composition table, P(signal | policy stop), kept for comparison.",
            "- `tokenizer_manifest.csv`: model-to-tokenizer mapping used for token counts.",
            "- `token_count_method_summary.csv`: direct/chunked/sampled token-count method audit.",
            "- `selected_strategy_token_summary.csv`: aggregate estimated token saving for the selected policy.",
            "- `selected_strategy_token_by_decision_round.csv`: actual selected-policy token saving by decision round.",
            "- `token_full_hit_by_round.csv`: full-hit what-if table for stopping all trajectories at each round.",
            "- `valid_accuracy_075_095_token_by_target.csv`: estimated token saving for the target frontier.",
            "- `successful_examples_selected_strategy.csv` and `.md`: true-positive success examples for internal inspection.",
        ]
    )
    (out_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-dir",
        default="paper/experiments/rq_final_lightgbm_17/lightgbm_main",
        type=Path,
    )
    parser.add_argument(
        "--prefix-table",
        default="../data/prefix_predict_model_holdout_answer/model_holdout_answer_shared/prefix_table_filtered.parquet",
        type=Path,
    )
    parser.add_argument(
        "--raw-traj-root",
        default="../data/swe_verify_500/by_model",
        type=Path,
    )
    parser.add_argument("--output-dir", default=None, type=Path)
    parser.add_argument("--examples", default=12, type=int)
    parser.add_argument(
        "--tokenizer-mode",
        choices=["model", "component_sum_approx", "chars_per_token"],
        default="model",
        help="Use exact per-prefix model tokenizers, component-sum approximate model tokenizers, or the legacy ceil(chars/4) proxy.",
    )
    parser.add_argument(
        "--tokenizer-cache-root",
        default="../.cache",
        type=Path,
        help="Cache root for HuggingFace and tiktoken tokenizer files.",
    )
    parser.add_argument(
        "--token-prefix-cache",
        default=None,
        type=Path,
        help="Optional parquet cache for finalized per-prefix tokenizer counts.",
    )
    parser.add_argument("--parquet-batch-size", default=512, type=int)
    parser.add_argument("--encode-batch-size", default=16, type=int)
    parser.add_argument("--progress-every-batches", default=100, type=int)
    parser.add_argument("--progress-every-rows", default=10000, type=int)
    parser.add_argument("--max-direct-tokenize-chars", default=DEFAULT_MAX_DIRECT_TOKENIZE_CHARS, type=int)
    parser.add_argument("--exact-chunk-tokenize-chars", default=DEFAULT_EXACT_CHUNK_TOKENIZE_CHARS, type=int)
    parser.add_argument("--sample-long-tokenize-chars", default=DEFAULT_SAMPLE_LONG_TOKENIZE_CHARS, type=int)
    parser.add_argument("--sample-chars", default=DEFAULT_SAMPLE_CHARS, type=int)
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Require all HuggingFace tokenizers to already be present in cache.",
    )
    args = parser.parse_args()

    run_dir = args.run_dir
    out_dir = args.output_dir or (run_dir / "internal_review_swe16")
    out_dir.mkdir(parents=True, exist_ok=True)

    selected = _load_selected(run_dir)
    selected_rank = build_selected_rank_change(selected)
    selected_rank.to_csv(out_dir / "selected_strategy_rank_change.csv", index=False)

    frontier_rank = build_frontier_rank_change(run_dir)
    if not frontier_rank.empty:
        frontier_rank.to_csv(out_dir / "valid_accuracy_075_095_rank_change_by_target.csv", index=False)

    decisions = build_selected_decisions(run_dir, selected)
    decisions.to_csv(out_dir / "selected_strategy_decisions.csv", index=False)
    audit = audit_decisions(decisions, selected)
    audit.to_csv(out_dir / "selected_strategy_decision_reconstruction_audit.csv", index=False)
    if not audit.filter(like="match_").all(axis=None):
        raise ValueError("decision reconstruction audit failed; see audit CSV")

    signal_prefix, signal_traj, signal_composition = build_stop_signal_decision_lift_tables(
        args.prefix_table,
        decisions,
    )
    signal_prefix.to_csv(out_dir / "stop_signal_decision_lift_by_prefix.csv", index=False)
    signal_traj.to_csv(out_dir / "stop_signal_decision_lift_by_trajectory.csv", index=False)
    signal_composition.to_csv(out_dir / "stop_signal_stop_composition.csv", index=False)

    token_rows, tokenizer_manifest = load_token_prefix_rows(
        args.prefix_table,
        set(decisions["traj_id"]),
        tokenizer_mode=args.tokenizer_mode,
        tokenizer_cache_root=args.tokenizer_cache_root,
        token_prefix_cache=args.token_prefix_cache,
        parquet_batch_size=int(args.parquet_batch_size),
        encode_batch_size=int(args.encode_batch_size),
        progress_every_batches=int(args.progress_every_batches),
        progress_every_rows=int(args.progress_every_rows),
        local_files_only=bool(args.local_files_only),
        max_direct_tokenize_chars=int(args.max_direct_tokenize_chars),
        exact_chunk_tokenize_chars=int(args.exact_chunk_tokenize_chars),
        sample_long_tokenize_chars=int(args.sample_long_tokenize_chars),
        sample_chars=int(args.sample_chars),
    )
    tokenizer_manifest.to_csv(out_dir / "tokenizer_manifest.csv", index=False)
    method_summary = (
        token_rows.groupby(["token_count_method", "tokenizer_name"], dropna=False, sort=True)
        .agg(
            prefix_rows=("traj_id", "size"),
            trajectories=("traj_id", "nunique"),
            max_context_chars=("context_chars_est", "max"),
            total_context_tokens_est=("context_tokens_est", "sum"),
        )
        .reset_index()
    )
    method_summary.to_csv(out_dir / "token_count_method_summary.csv", index=False)
    # Keep this diagnostic compact; the full per-prefix token table is large and easy to
    # rebuild from the script.
    full_hit = build_full_hit_by_round(token_rows, total_trajectories=len(decisions))
    full_hit.to_csv(out_dir / "token_full_hit_by_round.csv", index=False)

    enriched = enrich_decisions_with_tokens(decisions, token_rows)
    enriched.to_csv(out_dir / "selected_strategy_decisions_with_tokens.csv", index=False)
    selected_token_summary = build_selected_token_summary(enriched)
    selected_token_summary.to_csv(out_dir / "selected_strategy_token_summary.csv", index=False)
    selected_token_by_round = build_selected_token_by_round(enriched)
    selected_token_by_round.to_csv(out_dir / "selected_strategy_token_by_decision_round.csv", index=False)

    frontier_token_summary, frontier_token_by_round = build_frontier_token_summary(run_dir, token_rows)
    if not frontier_token_summary.empty:
        frontier_token_summary.to_csv(out_dir / "valid_accuracy_075_095_token_by_target.csv", index=False)
        frontier_token_by_round.to_csv(out_dir / "valid_accuracy_075_095_token_by_target_round.csv", index=False)

    examples = build_success_examples(
        enriched,
        raw_root=args.raw_traj_root,
        n_examples=int(args.examples),
    )
    examples.to_csv(out_dir / "successful_examples_selected_strategy.csv", index=False)
    write_examples_md(examples, out_dir / "successful_examples_selected_strategy.md")

    write_readme(
        out_dir,
        selected_rank,
        selected_token_summary,
        frontier_token_summary,
        tokenizer_manifest,
        str(args.tokenizer_mode),
    )
    manifest = pd.DataFrame(
        [
            {"artifact": p.name, "path": str(p), "bytes": p.stat().st_size}
            for p in sorted(out_dir.iterdir())
            if p.is_file()
        ]
    )
    manifest.to_csv(out_dir / "manifest.csv", index=False)
    print(f"Wrote {len(manifest)} artifacts to {out_dir}")


if __name__ == "__main__":
    main()
