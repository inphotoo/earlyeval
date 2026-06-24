#!/usr/bin/env python3
"""Train/evaluate dual-head safe-stop LightGBM predictors.

This is an independent experiment.  It does not change the final-success
predictor.  For each requested LightGBM feature variant it trains two heads:

* safe_success: final label is success and prefix_step_idx >= safe_label_min_step
* safe_failure: final label is failure and prefix_step_idx >= safe_label_min_step

The two heads create an explicit abstain/continue region.  Threshold policies
are selected on validation only, then applied once to heldout test.
"""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import gc
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".mplconfig"))
THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "BLIS_NUM_THREADS",
)
for _thread_env_name in THREAD_ENV_VARS:
    os.environ.setdefault(_thread_env_name, os.environ.get("SWE_MAX_CPU_THREADS", "24"))
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from scipy import sparse

import config
from feature_engineer import FeatureEngineer, TFIDF_ACTION_FEEDBACK, TFIDF_THOUGHT
from gold_text_tfidf_ablation_posthoc import (
    _fit_lgbm_with_cpu_fallback,
    _load_prefix_table,
    _repair_unpickled_tfidf_for_local_sklearn,
    _set_run_dirs,
    _write_feature_importance,
)
from model_holdout_shadow_valid_retrain import (
    _apply_lgbm_preset,
    _build_split,
    _json_default,
    _make_column_mask,
    _prediction_frame,
    _required_columns,
    _safe_name,
    _selected_specs,
    _set_cpu_thread_limits,
    _transform_tfidf_subset_streaming,
)
from probability_calibration import calibration_summary_row, fit_sigmoid_calibrator
from trainer import save_model
from utils import get_logger, rebind_all_file_loggers, timer


LOGGER = get_logger("safe_stop_dual_head_retrain")


@contextlib.contextmanager
def _ram_peak_lock(path: Path | None):
    """Serialize the RAM-heavy load+fit phase across parallel fold subprocesses.

    When `path` is None this is a no-op. Otherwise we open the file (creating
    it if missing) and grab an exclusive `fcntl.flock`; concurrent processes
    that ask for the same lock file block here until the current holder
    exits. The lock is released even on exceptions.
    """

    if path is None:
        yield
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = open(path, "a+", encoding="utf-8")
    try:
        LOGGER.info("Waiting for RAM-peak lock: %s", path)
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        LOGGER.info("Acquired RAM-peak lock: %s", path)
        yield
    finally:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        except Exception:
            pass
        handle.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", default="model_holdout_answer_calibrated_full")
    parser.add_argument("--prefix-table", type=Path, default=None)
    parser.add_argument(
        "--verified-jsonl",
        type=Path,
        default=PROJECT_ROOT.parents[2] / "swebench_verified" / "test.jsonl",
    )
    parser.add_argument("--holdout-models", default="auto_mid3")
    parser.add_argument(
        "--exclude-train-models",
        nargs="*",
        default=None,
        help=(
            "Optional list of model_id values to drop from the prefix table "
            "before splitting. Use this to keep audited or low-coverage "
            "models out of train/valid in addition to the held-out test "
            "models. The held-out test models are excluded automatically by "
            "--holdout-models; this flag handles the other config-level "
            "exclusions."
        ),
    )
    parser.add_argument("--max-instances", type=int, default=500)
    parser.add_argument(
        "--split-strategy",
        choices=("per_instance_model", "per_instance_traj"),
        default="per_instance_model",
    )
    parser.add_argument("--valid-models-per-instance", type=int, default=3)
    parser.add_argument("--valid-traj-ratio", type=float, default=0.15)
    parser.add_argument("--valid-per-instance", type=int, default=0)
    parser.add_argument("--seed", type=int, default=config.SPLIT_SEED)
    parser.add_argument("--output-subdir", default="safe_stop_dual_head_retrain")
    parser.add_argument("--variants", nargs="+", default=["default"])
    parser.add_argument("--safe-label-min-step", type=int, default=10)
    parser.add_argument("--policy-min-steps", nargs="+", type=int, default=[0, 5, 10, 15])
    parser.add_argument("--consecutive", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument(
        "--success-thresholds",
        nargs="+",
        type=float,
        default=[0.50, 0.60, 0.70, 0.80, 0.90],
    )
    parser.add_argument(
        "--failure-thresholds",
        nargs="+",
        type=float,
        default=[0.50, 0.60, 0.70, 0.80, 0.90],
    )
    parser.add_argument("--score-modes", nargs="+", choices=("raw", "calibrated"), default=["raw", "calibrated"])
    parser.add_argument("--max-valid-abs-drop-pp", type=float, default=2.0)
    parser.add_argument("--min-valid-decision-acc", type=float, default=0.90)
    parser.add_argument("--fallback-min-save-pct", type=float, default=5.0)
    parser.add_argument("--lgbm-preset", choices=("default", "strong_reg"), default="default")
    parser.add_argument("--mask-train-model-id", action="store_true")
    parser.add_argument(
        "--feature-engineer-path",
        type=Path,
        default=None,
        help=(
            "Optional path to a pre-fit FeatureEngineer pickle. Defaults to the "
            "global feature_engineer_with_model.pkl that was fit on the full "
            "prefix table. Ignored when --fit-feature-engineer-on-train is set."
        ),
    )
    parser.add_argument(
        "--fit-feature-engineer-on-train",
        action="store_true",
        help=(
            "Fit a fresh FeatureEngineer on this fold's train split only. "
            "This guarantees that no test_model trajectories contribute to "
            "the dense scaler statistics, label encoders, TF-IDF vocabulary, "
            "or SVD basis. The fitted engineer is saved to "
            "<output_dir>/models/feature_engineer_fold_local.pkl."
        ),
    )
    parser.add_argument(
        "--ram-peak-lock-path",
        type=Path,
        default=None,
        help=(
            "Optional path to a lock file used to serialize the RAM-heavy "
            "phase (loading the prefix table with text columns + fitting "
            "the FeatureEngineer) across parallel fold subprocesses. Only "
            "one fold can hold this lock at a time; other folds wait until "
            "the holder finishes the fit and drops text columns. The lock "
            "file itself is empty; pass the same path to every parallel "
            "fold of the same run."
        ),
    )
    parser.add_argument("--no-gpu-lgbm", action="store_true")
    parser.add_argument("--low-memory", action="store_true")
    parser.add_argument("--eager-load-text-columns", action="store_true")
    parser.add_argument("--text-batch-size", type=int, default=4096)
    parser.add_argument(
        "--max-cpu-threads",
        type=int,
        default=int(os.environ.get("SWE_MAX_CPU_THREADS", "24")),
    )
    parser.add_argument(
        "--smoke-trajectories-per-split",
        type=int,
        default=0,
        help="Debug only: sample this many whole trajectories per train/valid/test split before training.",
    )
    return parser.parse_args()


def _safe_targets(frame: pd.DataFrame, min_step: int) -> tuple[np.ndarray, np.ndarray]:
    labels = frame["label"].to_numpy(dtype=int)
    steps = frame["prefix_step_idx"].to_numpy(dtype=int)
    eligible = steps >= int(min_step)
    y_success = ((labels == 1) & eligible).astype(int)
    y_failure = ((labels == 0) & eligible).astype(int)
    return y_success, y_failure


def _build_matrices(
    *,
    prefix_path: Path,
    feature_engineer: FeatureEngineer,
    df_train: pd.DataFrame,
    df_valid: pd.DataFrame,
    df_test: pd.DataFrame,
    needs_thought: bool,
    eager_load_text_columns: bool,
    text_batch_size: int,
) -> tuple[dict[str, sparse.csr_matrix], list[str], list[str]]:
    tfidf_af_cols = list(TFIDF_ACTION_FEEDBACK.keys())
    tfidf_thought_cols = list(TFIDF_THOUGHT.keys())
    tfidf_af_thought_cols = tfidf_af_cols + tfidf_thought_cols
    with timer(LOGGER, "Build Dense / AF / Thought matrices"):
        X_train_dense = sparse.csr_matrix(feature_engineer.transform_dense(df_train))
        X_valid_dense = sparse.csr_matrix(feature_engineer.transform_dense(df_valid))
        X_test_dense = sparse.csr_matrix(feature_engineer.transform_dense(df_test))
        if eager_load_text_columns:
            X_train_af = feature_engineer.transform_tfidf_subset(df_train, tfidf_af_cols)
            X_valid_af = feature_engineer.transform_tfidf_subset(df_valid, tfidf_af_cols)
            X_test_af = feature_engineer.transform_tfidf_subset(df_test, tfidf_af_cols)
            if needs_thought:
                X_train_thought = feature_engineer.transform_tfidf_subset(df_train, tfidf_thought_cols)
                X_valid_thought = feature_engineer.transform_tfidf_subset(df_valid, tfidf_thought_cols)
                X_test_thought = feature_engineer.transform_tfidf_subset(df_test, tfidf_thought_cols)
            else:
                X_train_thought = sparse.csr_matrix((len(df_train), 0))
                X_valid_thought = sparse.csr_matrix((len(df_valid), 0))
                X_test_thought = sparse.csr_matrix((len(df_test), 0))
        else:
            X_train_af, X_valid_af, X_test_af = _transform_tfidf_subset_streaming(
                prefix_table_path=prefix_path,
                feature_engineer=feature_engineer,
                df_train=df_train,
                df_valid=df_valid,
                df_test=df_test,
                column_names=tfidf_af_cols,
                batch_size=text_batch_size,
            )
            if needs_thought:
                X_train_thought, X_valid_thought, X_test_thought = _transform_tfidf_subset_streaming(
                    prefix_table_path=prefix_path,
                    feature_engineer=feature_engineer,
                    df_train=df_train,
                    df_valid=df_valid,
                    df_test=df_test,
                    column_names=tfidf_thought_cols,
                    batch_size=text_batch_size,
                )
            else:
                X_train_thought = sparse.csr_matrix((len(df_train), 0))
                X_valid_thought = sparse.csr_matrix((len(df_valid), 0))
                X_test_thought = sparse.csr_matrix((len(df_test), 0))

    matrices = {
        "train_dense": X_train_dense,
        "valid_dense": X_valid_dense,
        "test_dense": X_test_dense,
        "train_af": X_train_af,
        "valid_af": X_valid_af,
        "test_af": X_test_af,
        "train_thought": X_train_thought,
        "valid_thought": X_valid_thought,
        "test_thought": X_test_thought,
    }
    names_af = (
        list(feature_engineer.dense_feature_names)
        + feature_engineer.get_tfidf_feature_names_for_columns(tfidf_af_cols)
    )
    names_j = (
        list(feature_engineer.dense_feature_names)
        + feature_engineer.get_tfidf_feature_names_for_columns(tfidf_af_thought_cols)
    )
    return matrices, names_af, names_j


def _matrices_for_spec(
    *,
    spec: dict[str, Any],
    matrices: dict[str, sparse.csr_matrix],
    names_af: list[str],
    names_j: list[str],
) -> tuple[sparse.csr_matrix, sparse.csr_matrix, sparse.csr_matrix, list[str], list[str]]:
    if spec["base"] == "af":
        X_train = sparse.hstack([matrices["train_dense"], matrices["train_af"]], format="csr")
        X_valid = sparse.hstack([matrices["valid_dense"], matrices["valid_af"]], format="csr")
        X_test = sparse.hstack([matrices["test_dense"], matrices["test_af"]], format="csr")
        return X_train, X_valid, X_test, list(names_af), []

    if spec["base"] != "af_thought":
        raise ValueError(f"Unknown base matrix: {spec['base']}")

    X_train = sparse.hstack(
        [matrices["train_dense"], matrices["train_af"], matrices["train_thought"]],
        format="csr",
    )
    X_valid = sparse.hstack(
        [matrices["valid_dense"], matrices["valid_af"], matrices["valid_thought"]],
        format="csr",
    )
    X_test = sparse.hstack(
        [matrices["test_dense"], matrices["test_af"], matrices["test_thought"]],
        format="csr",
    )
    feature_names = list(names_j)
    removed_names: list[str] = []
    remove_fn = spec.get("remove_fn")
    if remove_fn is not None:
        keep_cols, removed_names = _make_column_mask(feature_names, remove_fn)
        X_train = X_train[:, keep_cols].tocsr()
        X_valid = X_valid[:, keep_cols].tocsr()
        X_test = X_test[:, keep_cols].tocsr()
        feature_names = [feature_names[idx] for idx in keep_cols]
    return X_train, X_valid, X_test, feature_names, removed_names


def _head_column(prefix: str, score_mode: str, predictor: str) -> str:
    if score_mode == "raw":
        return f"prob_safe_{prefix}__{predictor}"
    if score_mode == "calibrated":
        return f"prob_cal_safe_{prefix}__{predictor}"
    raise ValueError(score_mode)


def _originals(df: pd.DataFrame) -> dict[str, dict[str, Any]]:
    final_idx = df.groupby("traj_id", sort=False)["prefix_step_idx"].idxmax()
    final_df = df.loc[final_idx]
    out: dict[str, dict[str, Any]] = {}
    for agent_model, part in final_df.groupby("orig_model_id", sort=True):
        total = int(len(part))
        resolved = int(part["label"].sum())
        out[str(agent_model)] = {
            "total": total,
            "resolved": resolved,
            "resolve_rate": resolved / total if total else 0.0,
        }
    return out


def _records(df: pd.DataFrame, success_col: str, failure_col: str) -> list[dict[str, Any]]:
    needed = ["traj_id", "orig_model_id", "label", "prefix_step_idx", success_col, failure_col]
    work = df[needed].copy()
    records: list[dict[str, Any]] = []
    for _, group in work.groupby("traj_id", sort=False):
        group = group.sort_values("prefix_step_idx")
        records.append(
            {
                "agent_model": str(group["orig_model_id"].iloc[0]),
                "label": int(group["label"].iloc[0]),
                "n_steps": int(len(group)),
                "steps": group["prefix_step_idx"].to_numpy(dtype=np.int32),
                "success": group[success_col].to_numpy(dtype=np.float64),
                "failure": group[failure_col].to_numpy(dtype=np.float64),
            }
        )
    return records


def _policy_grid(
    *,
    success_thresholds: list[float],
    failure_thresholds: list[float],
    min_steps: list[int],
    consecutive_values: list[int],
) -> list[dict[str, Any]]:
    policies: list[dict[str, Any]] = []
    for success_thr in success_thresholds:
        for min_step in min_steps:
            for consecutive in consecutive_values:
                policies.append(
                    {
                        "policy_mode": "success_only",
                        "success_thr": float(success_thr),
                        "failure_thr": math.inf,
                        "min_step": int(min_step),
                        "consecutive": int(consecutive),
                    }
                )
    for failure_thr in failure_thresholds:
        for min_step in min_steps:
            for consecutive in consecutive_values:
                policies.append(
                    {
                        "policy_mode": "failure_only",
                        "success_thr": math.inf,
                        "failure_thr": float(failure_thr),
                        "min_step": int(min_step),
                        "consecutive": int(consecutive),
                    }
                )
    for success_thr in success_thresholds:
        for failure_thr in failure_thresholds:
            for min_step in min_steps:
                for consecutive in consecutive_values:
                    policies.append(
                        {
                            "policy_mode": "dual",
                            "success_thr": float(success_thr),
                            "failure_thr": float(failure_thr),
                            "min_step": int(min_step),
                            "consecutive": int(consecutive),
                        }
                    )
    return policies


def _decide_dual(
    record: dict[str, Any],
    *,
    success_thr: float,
    failure_thr: float,
    min_step: int,
    consecutive: int,
) -> tuple[bool, str, int, float]:
    last_decision = "undecided"
    streak = 0
    for step_value, success_score, failure_score in zip(
        record["steps"],
        record["success"],
        record["failure"],
    ):
        step = int(step_value)
        if step < min_step:
            continue
        success_hit = float(success_score) >= success_thr
        failure_hit = float(failure_score) >= failure_thr
        if success_hit and failure_hit:
            success_margin = float(success_score) - success_thr
            failure_margin = float(failure_score) - failure_thr
            decision = "success" if success_margin >= failure_margin else "failure"
            score = float(success_score if decision == "success" else failure_score)
        elif success_hit:
            decision = "success"
            score = float(success_score)
        elif failure_hit:
            decision = "failure"
            score = float(failure_score)
        else:
            last_decision = "undecided"
            streak = 0
            continue

        streak = streak + 1 if decision == last_decision else 1
        last_decision = decision
        if streak >= consecutive:
            return True, decision, step, score
    return False, "undecided", -1, float("nan")


def _empty_counts() -> dict[str, int]:
    return {
        "decided_failure": 0,
        "decided_success": 0,
        "undecided": 0,
        "false_negatives": 0,
        "true_negatives": 0,
        "false_positives": 0,
        "true_positives": 0,
        "total_saved_steps": 0,
        "total_steps": 0,
    }


def _summarize(counts: dict[str, int], original: dict[str, Any]) -> dict[str, Any]:
    tp = int(counts["true_positives"])
    tn = int(counts["true_negatives"])
    fp = int(counts["false_positives"])
    fn = int(counts["false_negatives"])
    decided_success = int(counts["decided_success"])
    decided_failure = int(counts["decided_failure"])
    n_decided = decided_success + decided_failure
    undecided_resolved = int(original["resolved"]) - tp - fn
    adjusted_resolved = tp + undecided_resolved
    adjusted_rate = adjusted_resolved / int(original["total"]) if original["total"] else 0.0
    total_steps = int(counts["total_steps"])
    return {
        "original_total": int(original["total"]),
        "original_resolved": int(original["resolved"]),
        "original_resolve_rate": float(original["resolve_rate"]),
        "decided_failure": decided_failure,
        "decided_success": decided_success,
        "undecided": int(counts["undecided"]),
        "false_negatives": fn,
        "true_negatives": tn,
        "false_positives": fp,
        "true_positives": tp,
        "n_decided": n_decided,
        "coverage": n_decided / int(original["total"]) if original["total"] else float("nan"),
        "decision_accuracy": (tp + tn) / n_decided if n_decided else float("nan"),
        "precision_success": tp / decided_success if decided_success else float("nan"),
        "precision_failure": tn / decided_failure if decided_failure else float("nan"),
        "adjusted_resolved": int(adjusted_resolved),
        "adjusted_resolve_rate": float(adjusted_rate),
        "resolve_rate_drop": float(original["resolve_rate"] - adjusted_rate),
        "pct_steps_saved": (
            float(counts["total_saved_steps"]) * 100.0 / float(total_steps)
            if total_steps
            else float("nan")
        ),
        "total_saved_steps": int(counts["total_saved_steps"]),
        "total_steps": total_steps,
    }


def _evaluate_policies(
    df: pd.DataFrame,
    *,
    run_label: str,
    predictors: list[str],
    score_modes: list[str],
    policies: list[dict[str, Any]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    originals = _originals(df)
    aggregate_rows: list[dict[str, Any]] = []
    per_agent_rows: list[dict[str, Any]] = []
    for predictor in predictors:
        for score_mode in score_modes:
            success_col = _head_column("success", score_mode, predictor)
            failure_col = _head_column("failure", score_mode, predictor)
            if success_col not in df.columns or failure_col not in df.columns:
                continue
            records = _records(df, success_col, failure_col)
            for policy in policies:
                per_agent = {agent_model: _empty_counts() for agent_model in originals}
                for record in records:
                    agent_model = record["agent_model"]
                    label = int(record["label"])
                    n_steps = int(record["n_steps"])
                    per_agent[agent_model]["total_steps"] += n_steps
                    decided, decision, decision_step, _ = _decide_dual(
                        record,
                        success_thr=float(policy["success_thr"]),
                        failure_thr=float(policy["failure_thr"]),
                        min_step=int(policy["min_step"]),
                        consecutive=int(policy["consecutive"]),
                    )
                    if not decided:
                        per_agent[agent_model]["undecided"] += 1
                        continue
                    per_agent[agent_model]["total_saved_steps"] += max(n_steps - decision_step - 1, 0)
                    if decision == "failure":
                        per_agent[agent_model]["decided_failure"] += 1
                        if label == 1:
                            per_agent[agent_model]["false_negatives"] += 1
                        else:
                            per_agent[agent_model]["true_negatives"] += 1
                    else:
                        per_agent[agent_model]["decided_success"] += 1
                        if label == 0:
                            per_agent[agent_model]["false_positives"] += 1
                        else:
                            per_agent[agent_model]["true_positives"] += 1

                policy_meta = {
                    "run": run_label,
                    "score_mode": score_mode,
                    "prefix_model": predictor,
                    **policy,
                }
                total_counts = _empty_counts()
                total_original = {"total": 0, "resolved": 0, "resolve_rate": 0.0}
                for agent_model, counts in per_agent.items():
                    summary = _summarize(counts, originals[agent_model])
                    per_agent_rows.append({**policy_meta, "agent_model": agent_model, **summary})
                    for key, value in counts.items():
                        total_counts[key] += int(value)
                    total_original["total"] += int(originals[agent_model]["total"])
                    total_original["resolved"] += int(originals[agent_model]["resolved"])
                total_original["resolve_rate"] = (
                    total_original["resolved"] / total_original["total"]
                    if total_original["total"]
                    else 0.0
                )
                aggregate_rows.append({**policy_meta, **_summarize(total_counts, total_original)})
    return pd.DataFrame(aggregate_rows), pd.DataFrame(per_agent_rows)


def _policy_key(row: pd.Series) -> dict[str, Any]:
    return {
        "policy_mode": row["policy_mode"],
        "success_thr": float(row["success_thr"]),
        "failure_thr": float(row["failure_thr"]),
        "min_step": int(row["min_step"]),
        "consecutive": int(row["consecutive"]),
    }


def _select_policies(
    valid_aggregate: pd.DataFrame,
    *,
    max_valid_abs_drop_pp: float,
    min_valid_decision_acc: float,
    fallback_min_save_pct: float,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for group_key, part in valid_aggregate.groupby(["run", "score_mode", "prefix_model"], sort=False):
        work = part.copy()
        work["drop_pp"] = work["resolve_rate_drop"] * 100.0
        work["abs_drop_pp"] = work["drop_pp"].abs()
        work["decision_accuracy_for_filter"] = work["decision_accuracy"].fillna(-1.0)
        work["pct_steps_saved_for_sort"] = work["pct_steps_saved"].fillna(0.0)
        strict = work[
            (work["abs_drop_pp"] <= max_valid_abs_drop_pp)
            & (work["decision_accuracy_for_filter"] >= min_valid_decision_acc)
            & (work["pct_steps_saved_for_sort"] > 0.0)
        ].copy()
        if not strict.empty:
            chosen = strict.sort_values(
                ["pct_steps_saved_for_sort", "abs_drop_pp", "decision_accuracy_for_filter"],
                ascending=[False, True, False],
            ).iloc[0]
            status = "valid_constraints_pass"
        else:
            fallback = work[work["pct_steps_saved_for_sort"] >= fallback_min_save_pct].copy()
            if fallback.empty:
                fallback = work
            chosen = fallback.sort_values(
                ["abs_drop_pp", "pct_steps_saved_for_sort", "decision_accuracy_for_filter"],
                ascending=[True, False, False],
            ).iloc[0]
            status = "fallback_min_abs_valid_drop"
        row = chosen.to_dict()
        row["policy_id"] = f"{group_key[0]}__{group_key[1]}__{_safe_name(group_key[2])}"
        row["selection_status"] = status
        row["valid_abs_drop_pp"] = float(chosen["abs_drop_pp"])
        rows.append(row)
    return pd.DataFrame(rows)


def _evaluate_selected(
    df: pd.DataFrame,
    *,
    run_label: str,
    selected: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for _, row in selected.iterrows():
        aggregate, _ = _evaluate_policies(
            df,
            run_label=run_label,
            predictors=[str(row["prefix_model"])],
            score_modes=[str(row["score_mode"])],
            policies=[_policy_key(row)],
        )
        aggregate["policy_id"] = row["policy_id"]
        rows.append(aggregate)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _fmt(value: Any, digits: int = 1) -> str:
    try:
        value = float(value)
    except Exception:
        return "-"
    if math.isnan(value) or math.isinf(value):
        return "-"
    return f"{value:.{digits}f}"


def _write_report(output_dir: Path, selected: pd.DataFrame, test_selected: pd.DataFrame) -> None:
    lines = [
        "# Safe-Stop Dual-Head Report",
        "",
        "Policy is selected on validation only; heldout test is used only for locked-policy evaluation.",
        "",
        "| Model | Score | Status | Mode | S_thr | F_thr | Min | K | Valid Save | Valid Drop pp | Valid Acc | Test Save | Test Drop pp | Test Acc | Test FN | Test FP |",
        "|:--|:--|:--|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|",
    ]
    if selected.empty:
        lines.append("| - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - |")
    else:
        merged = selected.merge(
            test_selected,
            on=["policy_id", "run", "score_mode", "prefix_model"],
            how="left",
            suffixes=("_valid", "_test"),
        )
        for _, row in merged.sort_values(["prefix_model", "score_mode"]).iterrows():
            lines.append(
                "| "
                f"{row['prefix_model']} | {row['score_mode']} | {row['selection_status']} | "
                f"{row['policy_mode_valid']} | {_fmt(row['success_thr_valid'], 2)} | "
                f"{_fmt(row['failure_thr_valid'], 2)} | {int(row['min_step_valid'])} | "
                f"{int(row['consecutive_valid'])} | {_fmt(row['pct_steps_saved_valid'])}% | "
                f"{_fmt(float(row['resolve_rate_drop_valid']) * 100.0)} | "
                f"{_fmt(float(row['decision_accuracy_valid']) * 100.0)}% | "
                f"{_fmt(row.get('pct_steps_saved_test', math.nan))}% | "
                f"{_fmt(float(row.get('resolve_rate_drop_test', math.nan)) * 100.0)} | "
                f"{_fmt(float(row.get('decision_accuracy_test', math.nan)) * 100.0)}% | "
                f"{int(row.get('false_negatives_test', 0)) if not pd.isna(row.get('false_negatives_test', math.nan)) else '-'} | "
                f"{int(row.get('false_positives_test', 0)) if not pd.isna(row.get('false_positives_test', math.nan)) else '-'} |"
            )
    lines.extend(
        [
            "",
            "## Output Files",
            "",
            "- `valid_predictions_safe_stop.parquet`",
            "- `test_predictions_safe_stop.parquet`",
            "- `safe_stop_valid_policy_grid.csv`",
            "- `safe_stop_valid_policy_per_agent.csv`",
            "- `safe_stop_selected_policies.csv`",
            "- `safe_stop_test_selected.csv`",
        ]
    )
    (output_dir / "safe_stop_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    max_cpu_threads = _set_cpu_thread_limits(args.max_cpu_threads)
    run_root = config.RUNTIME_ROOT / "runs" / args.run_name
    _set_run_dirs(run_root)
    rebind_all_file_loggers()
    config.LGBM_PARAMS["num_threads"] = max_cpu_threads
    preset_updates = _apply_lgbm_preset(args.lgbm_preset)
    if args.no_gpu_lgbm:
        config.LGBM_PARAMS["device"] = "cpu"
        config.LGBM_PARAMS.pop("gpu_device_id", None)
    LOGGER.info("CPU thread cap: %s; lgbm_preset=%s", max_cpu_threads, args.lgbm_preset)

    prefix_path = args.prefix_table or config.PREFIX_TABLE_FILTERED_PATH
    output_dir = config.REPORT_DIR / args.output_subdir
    output_model_dir = output_dir / "models"
    output_model_dir.mkdir(parents=True, exist_ok=True)

    fit_feature_engineer_on_train = bool(args.fit_feature_engineer_on_train)
    if fit_feature_engineer_on_train:
        # Force eager text loading so the train split has the text columns
        # available for the in-fold FeatureEngineer.fit call. Without this we
        # would have to re-read text columns from disk after splitting.
        if not args.eager_load_text_columns:
            LOGGER.info(
                "fit_feature_engineer_on_train=True forces eager text loading."
            )
            args.eager_load_text_columns = True
        # Use a placeholder engineer just to compute required column names; it
        # is discarded before fitting.
        loaded_feature_engineer: FeatureEngineer | None = None
        required_columns = _required_columns(
            FeatureEngineer(include_model_id=True, tfidf_level="with_thought"),
            include_text=True,
        )
    else:
        fe_path = args.feature_engineer_path or (config.MODEL_DIR / "feature_engineer_with_model.pkl")
        loaded_feature_engineer = FeatureEngineer.load(fe_path)
        _repair_unpickled_tfidf_for_local_sklearn(loaded_feature_engineer)
        required_columns = _required_columns(
            loaded_feature_engineer, include_text=args.eager_load_text_columns
        )
    specs = _selected_specs(args.variants)
    predictors = [spec["predictor"] for spec in specs]
    needs_thought = any(spec["base"] == "af_thought" for spec in specs)

    with _ram_peak_lock(args.ram_peak_lock_path):
        with timer(LOGGER, "Load cached prefix table and build split"):
            prefix_df = _load_prefix_table(prefix_path, required_columns)
            excluded_train_models = sorted({str(item) for item in (args.exclude_train_models or []) if str(item)})
            if excluded_train_models:
                before_models = int(prefix_df["model_id"].nunique())
                before_rows = int(len(prefix_df))
                mask = ~prefix_df["model_id"].astype(str).isin(set(excluded_train_models))
                prefix_df = prefix_df.loc[mask].copy()
                LOGGER.info(
                    "Dropped %d configured-excluded model(s) before split: kept %d/%d models, %d/%d rows.",
                    len(excluded_train_models),
                    int(prefix_df["model_id"].nunique()),
                    before_models,
                    int(len(prefix_df)),
                    before_rows,
                )
            df_train, df_valid, df_test, split_meta, split_summary = _build_split(
                prefix_df,
                verified_jsonl=args.verified_jsonl,
                holdout_models=args.holdout_models,
                max_instances=args.max_instances,
                split_strategy=args.split_strategy,
                valid_traj_ratio=args.valid_traj_ratio,
                valid_per_instance=args.valid_per_instance,
                valid_models_per_instance=args.valid_models_per_instance,
                shadow_valid_max_trajectories=0,
                seed=args.seed,
                smoke_trajectories_per_split=args.smoke_trajectories_per_split,
                mask_train_model_id=args.mask_train_model_id,
            )
            split_meta["safe_label_min_step"] = int(args.safe_label_min_step)
            split_meta["lgbm_preset"] = args.lgbm_preset
            split_meta["lgbm_preset_updates"] = preset_updates
            split_meta["lgbm_params_used"] = dict(config.LGBM_PARAMS)
            split_meta["excluded_train_models"] = excluded_train_models
            del prefix_df
            gc.collect()

        if fit_feature_engineer_on_train:
            # Strict no-leak: fit the FeatureEngineer on this fold's train split
            # only. Test trajectories never enter the scaler, the label encoders,
            # the TF-IDF vocabulary, or the SVD basis.
            with timer(LOGGER, "Fit fold-local FeatureEngineer on train"):
                feature_engineer = FeatureEngineer(include_model_id=True, tfidf_level="with_thought")
                feature_engineer.fit(df_train)
                fe_local_path = output_model_dir / "feature_engineer_fold_local.pkl"
                feature_engineer.save(fe_local_path)
                LOGGER.info("Saved fold-local FeatureEngineer to %s", fe_local_path)
            split_meta["feature_engineer_fit_on_train"] = True
            split_meta["feature_engineer_fold_local_path"] = str(fe_local_path)
            split_meta["feature_engineer_source"] = "fit_on_train"
            # Free the eagerly-loaded text columns now that fit is done. Matrix
            # building will switch to the streaming path which re-reads only the
            # rows it needs from disk in batches. This keeps peak RAM per fold
            # at "no-text" level (~hundreds of MB) instead of "all-text" level
            # (multiple GB), which matters when several folds run in parallel.
            text_columns_to_drop = sorted(
                {
                    feature_engineer.active_text_columns[name]
                    for name in (list(TFIDF_ACTION_FEEDBACK.keys()) + list(TFIDF_THOUGHT.keys()))
                    if name in feature_engineer.active_text_columns
                }
            )
            for frame_to_trim in (df_train, df_valid, df_test):
                present = [col for col in text_columns_to_drop if col in frame_to_trim.columns]
                if present:
                    frame_to_trim.drop(columns=present, inplace=True)
            if text_columns_to_drop:
                LOGGER.info(
                    "Dropped %d text column(s) from in-memory train/valid/test after fit; "
                    "matrix building will stream them from disk.",
                    len(text_columns_to_drop),
                )
                args.eager_load_text_columns = False
                gc.collect()
        else:
            feature_engineer = loaded_feature_engineer
            split_meta["feature_engineer_fit_on_train"] = False
            split_meta["feature_engineer_source"] = "shared_pkl"
            split_meta["feature_engineer_pkl_path"] = str(
                args.feature_engineer_path or (config.MODEL_DIR / "feature_engineer_with_model.pkl")
            )

    # ── RAM-peak lock released here. Matrix building, training, calibration,
    # and policy evaluation all run with text dropped from the in-memory
    # frames; they are RAM-cheap and safe to run in parallel across folds.
    y_success_train, y_failure_train = _safe_targets(df_train, args.safe_label_min_step)
    y_success_valid, y_failure_valid = _safe_targets(df_valid, args.safe_label_min_step)
    y_success_test, y_failure_test = _safe_targets(df_test, args.safe_label_min_step)
    w_train = df_train["sample_weight"].to_numpy(dtype=np.float32)
    w_valid = df_valid["sample_weight"].to_numpy(dtype=np.float32)

    matrices, names_af, names_j = _build_matrices(
        prefix_path=prefix_path,
        feature_engineer=feature_engineer,
        df_train=df_train,
        df_valid=df_valid,
        df_test=df_test,
        needs_thought=needs_thought,
        eager_load_text_columns=args.eager_load_text_columns,
        text_batch_size=args.text_batch_size,
    )

    valid_pred = _prediction_frame(df_valid)
    test_pred = _prediction_frame(df_test)
    del df_train, df_valid, df_test
    gc.collect()

    calibration_rows: list[dict[str, Any]] = []
    variant_rows: list[dict[str, Any]] = []
    for spec in specs:
        public_name = spec["predictor"]
        X_train, X_valid, X_test, feature_names, removed_names = _matrices_for_spec(
            spec=spec,
            matrices=matrices,
            names_af=names_af,
            names_j=names_j,
        )
        for head_name, y_train, y_valid, y_test, column_prefix in (
            ("safe_success", y_success_train, y_success_valid, y_success_test, "success"),
            ("safe_failure", y_failure_train, y_failure_valid, y_failure_test, "failure"),
        ):
            model_name = f"{public_name}__{head_name}"
            with timer(LOGGER, f"Train {model_name}"):
                model = _fit_lgbm_with_cpu_fallback(
                    X_train=X_train,
                    y_train=y_train,
                    w_train=w_train,
                    X_valid=X_valid,
                    y_valid=y_valid,
                    w_valid=w_valid,
                    feature_names=feature_names,
                    model_name=model_name,
                )
                save_model(model, output_model_dir / f"{_safe_name(model_name)}.lgb")
                valid_raw = np.asarray(model.predict(X_valid), dtype=np.float64)
                test_raw = np.asarray(model.predict(X_test), dtype=np.float64)
                calibrator = fit_sigmoid_calibrator(valid_raw, y_valid, sample_weight=w_valid)
                valid_cal = calibrator.predict(valid_raw)
                test_cal = calibrator.predict(test_raw)
                save_model(calibrator, output_model_dir / f"calibrator_{_safe_name(model_name)}.pkl")
                calibration_rows.append(
                    {
                        "head": head_name,
                        **calibration_summary_row(
                            model_name=model_name,
                            calibrator=calibrator,
                            y_valid=y_valid,
                            raw_prob_valid=valid_raw,
                            y_test=y_test,
                            raw_prob_test=test_raw,
                        ),
                    }
                )
                valid_pred[_head_column(column_prefix, "raw", public_name)] = valid_raw.astype(np.float32)
                valid_pred[_head_column(column_prefix, "calibrated", public_name)] = valid_cal.astype(np.float32)
                test_pred[_head_column(column_prefix, "raw", public_name)] = test_raw.astype(np.float32)
                test_pred[_head_column(column_prefix, "calibrated", public_name)] = test_cal.astype(np.float32)
                _write_feature_importance(
                    model,
                    feature_names,
                    output_dir / f"feature_importance_{_safe_name(model_name)}.csv",
                )

        variant_rows.append(
            {
                "predictor": public_name,
                "base_matrix": spec["base"],
                "description": spec["description"],
                "removed_feature_count": int(len(removed_names)),
                "kept_feature_count": int(len(feature_names)),
                "removed_feature_examples": "; ".join(removed_names[:30]),
            }
        )
        if args.low_memory or spec.get("remove_fn") is not None:
            del X_train, X_valid, X_test
            gc.collect()

    output_dir.mkdir(parents=True, exist_ok=True)
    valid_pred.to_parquet(output_dir / "valid_predictions_safe_stop.parquet", index=False)
    test_pred.to_parquet(output_dir / "test_predictions_safe_stop.parquet", index=False)
    split_summary.to_csv(output_dir / "split_summary.csv", index=False)
    (output_dir / "split_metadata.json").write_text(
        json.dumps(split_meta, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )
    pd.DataFrame(variant_rows).to_csv(output_dir / "variant_manifest.csv", index=False)
    pd.DataFrame(calibration_rows).to_csv(output_dir / "safe_stop_calibration_summary.csv", index=False)

    policies = _policy_grid(
        success_thresholds=args.success_thresholds,
        failure_thresholds=args.failure_thresholds,
        min_steps=args.policy_min_steps,
        consecutive_values=args.consecutive,
    )
    valid_grid, valid_per_agent = _evaluate_policies(
        valid_pred,
        run_label=args.output_subdir,
        predictors=predictors,
        score_modes=args.score_modes,
        policies=policies,
    )
    selected = _select_policies(
        valid_grid,
        max_valid_abs_drop_pp=args.max_valid_abs_drop_pp,
        min_valid_decision_acc=args.min_valid_decision_acc,
        fallback_min_save_pct=args.fallback_min_save_pct,
    )
    test_selected = _evaluate_selected(test_pred, run_label=args.output_subdir, selected=selected)

    valid_grid.to_csv(output_dir / "safe_stop_valid_policy_grid.csv", index=False)
    valid_per_agent.to_csv(output_dir / "safe_stop_valid_policy_per_agent.csv", index=False)
    selected.to_csv(output_dir / "safe_stop_selected_policies.csv", index=False)
    test_selected.to_csv(output_dir / "safe_stop_test_selected.csv", index=False)
    _write_report(output_dir, selected, test_selected)
    print(f"Saved safe-stop dual-head results: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
