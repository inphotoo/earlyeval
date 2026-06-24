#!/usr/bin/env python3
"""Shared utilities for staged BERT/CodeBERT safe-stop baselines.

These baselines are additive experiments: they reuse the completed
``prefix_predict_model_holdout_answer`` artifacts and the existing safe-stop
policy evaluator, but write all new artifacts under ``final2/results``.
"""

from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from torch import nn

def _package_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "configs" / "earlyeval.yaml").exists():
            return parent
    raise RuntimeError("Could not locate earlyeval root")


PACKAGE_ROOT = _package_root()
REPO_ROOT = PACKAGE_ROOT.parent
PROJECT_ROOT = PACKAGE_ROOT / "earlyeval" / "vendor" / "prefix_predict_model_holdout_answer"
sys.path.insert(0, str(PROJECT_ROOT))

import config
from feature_engineer import FeatureEngineer
from gold_text_tfidf_ablation_posthoc import (
    _load_prefix_table,
    _repair_unpickled_tfidf_for_local_sklearn,
    _set_run_dirs,
)
from model_holdout_shadow_valid_retrain import (
    _build_split,
    _json_default,
    _prediction_frame,
    _required_columns,
    _set_cpu_thread_limits,
)
from probability_calibration import calibration_summary_row, fit_sigmoid_calibrator
from safe_stop_dual_head_retrain import (
    _evaluate_policies,
    _evaluate_selected,
    _head_column,
    _policy_grid,
    _safe_targets,
    _select_policies,
    _write_report,
)
from trainer import save_model
from utils import rebind_all_file_loggers


DEFAULT_ENCODER = "microsoft/codebert-base"
DEFAULT_RUN_NAME = "model_holdout_answer_calibrated_full"
DEFAULT_VERIFIED_JSONL = REPO_ROOT / "data" / "swe_verify_500" / "offical_answer" / "test.jsonl"
DEFAULT_RESULTS_ROOT = PACKAGE_ROOT / "paper" / "experiments" / "earlyeval_architecture_smoke"
SHARED_ANSWER_DATA_ROOT = REPO_ROOT / "data" / "prefix_predict_model_holdout_answer" / "model_holdout_answer_shared"
DEFAULT_PREFIX_TABLE = SHARED_ANSWER_DATA_ROOT / "prefix_table_filtered.parquet"

METADATA_COLUMNS = [
    "prefix_id",
    "traj_id",
    "group_id",
    "instance_id",
    "prefix_step_idx",
    "n_steps_total_for_weighting",
    "sample_weight",
    "label",
    "model_id",
    "model",
]

TEXT_COLUMNS = [
    "task_prompt_text",
    "gold_answer_summary_text",
    "last_action_text",
    "last_feedback_text",
    "last_thought_text",
    "prefix_action_text",
    "prefix_feedback_text",
    "prefix_thought_text",
]

TEXT_CACHE_COLUMNS = [
    "prefix_id",
    "traj_id",
    "instance_id",
    "group_id",
    "prefix_step_idx",
    "n_steps_total_for_weighting",
    "sample_weight",
    "label",
    "split",
    "model_id",
    "model",
    "orig_model_id",
    "orig_model",
    "model_id_input_mode",
    "row_idx",
    "safe_success_label",
    "safe_failure_label",
    "text_a",
    "text_b",
    "text_a_chars",
    "text_b_chars",
]

def completed_run_root(run_name: str) -> Path:
    run_root = PROJECT_ROOT / "runs" / run_name
    _set_run_dirs(run_root)
    rebind_all_file_loggers()
    return run_root


def resolve_prefix_path(run_name: str, prefix_table: Path | None) -> Path:
    completed_run_root(run_name)
    return prefix_table or DEFAULT_PREFIX_TABLE


def load_feature_engineer(run_name: str) -> FeatureEngineer:
    completed_run_root(run_name)
    feature_engineer = FeatureEngineer.load(config.MODEL_DIR / "feature_engineer_with_model.pkl")
    _repair_unpickled_tfidf_for_local_sklearn(feature_engineer)
    return feature_engineer


def load_split_frames(
    *,
    run_name: str,
    prefix_table: Path | None,
    verified_jsonl: Path,
    holdout_models: str,
    max_instances: int,
    split_strategy: str,
    valid_traj_ratio: float,
    valid_per_instance: int,
    valid_models_per_instance: int,
    smoke_trajectories_per_split: int,
    seed: int,
    mask_train_model_id: bool,
    include_dense_columns: bool,
    exclude_train_models: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any], pd.DataFrame, FeatureEngineer | None, Path]:
    prefix_path = resolve_prefix_path(run_name, prefix_table)
    feature_engineer = None
    if include_dense_columns:
        feature_engineer = load_feature_engineer(run_name)
        columns = _required_columns(feature_engineer, include_text=False)
    else:
        columns = METADATA_COLUMNS
    prefix_df = _load_prefix_table(prefix_path, columns)
    excluded_train_models = sorted({str(item) for item in (exclude_train_models or []) if str(item)})
    if excluded_train_models:
        prefix_df = prefix_df.loc[
            ~prefix_df["model_id"].astype(str).isin(set(excluded_train_models))
        ].copy()
    df_train, df_valid, df_test, split_meta, split_summary = _build_split(
        prefix_df,
        verified_jsonl=verified_jsonl,
        holdout_models=holdout_models,
        max_instances=max_instances,
        split_strategy=split_strategy,
        valid_traj_ratio=valid_traj_ratio,
        valid_per_instance=valid_per_instance,
        valid_models_per_instance=valid_models_per_instance,
        shadow_valid_max_trajectories=0,
        seed=seed,
        smoke_trajectories_per_split=smoke_trajectories_per_split,
        mask_train_model_id=mask_train_model_id,
    )
    split_meta["learner"] = "bert_baseline"
    split_meta["excluded_train_models"] = excluded_train_models
    return df_train, df_valid, df_test, split_meta, split_summary, feature_engineer, prefix_path


def add_safe_labels(frame: pd.DataFrame, safe_label_min_step: int) -> pd.DataFrame:
    success, failure = _safe_targets(frame, safe_label_min_step)
    out = frame.copy()
    out["safe_success_label"] = success.astype(np.int8)
    out["safe_failure_label"] = failure.astype(np.int8)
    return out


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    return str(value)


def _head_text(value: Any, max_chars: int) -> str:
    text = _clean_text(value)
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[:max_chars]


def _tail_text(value: Any, max_chars: int) -> str:
    text = _clean_text(value)
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[-max_chars:]


def _join_sections(sections: Iterable[tuple[str, str]]) -> str:
    parts = []
    for title, text in sections:
        text = _clean_text(text).strip()
        if text:
            parts.append(f"[{title}]\n{text}")
    return "\n\n".join(parts)


def build_bert_text_pair(
    row: pd.Series,
    *,
    task_chars: int,
    gold_chars: int,
    last_action_chars: int,
    last_feedback_chars: int,
    last_thought_chars: int,
    prefix_action_tail_chars: int,
    prefix_feedback_tail_chars: int,
    prefix_thought_tail_chars: int,
) -> tuple[str, str]:
    text_a = _join_sections(
        [
            ("TASK", _head_text(row.get("task_prompt_text", ""), task_chars)),
            ("GOLD ANSWER SUMMARY", _head_text(row.get("gold_answer_summary_text", ""), gold_chars)),
            ("LAST ACTION", _head_text(row.get("last_action_text", ""), last_action_chars)),
            ("LAST FEEDBACK", _head_text(row.get("last_feedback_text", ""), last_feedback_chars)),
            ("LAST THOUGHT", _head_text(row.get("last_thought_text", ""), last_thought_chars)),
        ]
    )
    text_b = _join_sections(
        [
            ("PREFIX ACTION TAIL", _tail_text(row.get("prefix_action_text", ""), prefix_action_tail_chars)),
            ("PREFIX FEEDBACK TAIL", _tail_text(row.get("prefix_feedback_text", ""), prefix_feedback_tail_chars)),
            ("PREFIX THOUGHT TAIL", _tail_text(row.get("prefix_thought_text", ""), prefix_thought_tail_chars)),
        ]
    )
    return text_a, text_b


def resolve_device(device_arg: str) -> torch.device:
    requested = str(device_arg).strip().lower()
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not visible in this Python environment. "
            "Run from a GPU-enabled conda environment or set DEVICE=cpu for a small debug run."
        )
    return torch.device(requested)


def load_hf_encoder(
    *,
    encoder_name: str,
    device: torch.device,
    local_files_only: bool,
    fp16: bool,
):
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        encoder_name,
        local_files_only=local_files_only,
        use_fast=True,
    )
    model = AutoModel.from_pretrained(
        encoder_name,
        local_files_only=local_files_only,
    )
    model.eval()
    model.to(device)
    if fp16 and device.type == "cuda":
        model.half()
    return tokenizer, model


def mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)
    pooled = (last_hidden_state * mask).sum(dim=1)
    denom = mask.sum(dim=1).clamp(min=1.0)
    return pooled / denom


def pool_encoder_output(outputs: Any, attention_mask: torch.Tensor, pooling: str) -> torch.Tensor:
    if pooling == "cls":
        return outputs.last_hidden_state[:, 0]
    if pooling == "mean":
        return mean_pool(outputs.last_hidden_state, attention_mask)
    raise ValueError(f"Unknown pooling mode: {pooling}")


class DualHeadMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list[int], dropout: float):
        super().__init__()
        layers: list[nn.Module] = []
        current = int(input_dim)
        for hidden in hidden_dims:
            hidden = int(hidden)
            if hidden <= 0:
                continue
            layers.extend([nn.Linear(current, hidden), nn.ReLU(), nn.Dropout(float(dropout))])
            current = hidden
        self.trunk = nn.Sequential(*layers) if layers else nn.Identity()
        self.success_head = nn.Linear(current, 1)
        self.failure_head = nn.Linear(current, 1)

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.trunk(features)
        return self.success_head(hidden).squeeze(-1), self.failure_head(hidden).squeeze(-1)


class BertDualHeadClassifier(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        *,
        encoder_dim: int,
        hidden_dims: list[int],
        dropout: float,
        pooling: str,
    ):
        super().__init__()
        self.encoder = encoder
        self.pooling = pooling
        self.head = DualHeadMLP(encoder_dim, hidden_dims, dropout)

    def forward(self, **batch: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        outputs = self.encoder(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            token_type_ids=batch.get("token_type_ids"),
            return_dict=True,
        )
        pooled = pool_encoder_output(outputs, batch["attention_mask"], self.pooling)
        return self.head(pooled)


def weighted_dual_loss(
    success_logits: torch.Tensor,
    failure_logits: torch.Tensor,
    success_targets: torch.Tensor,
    failure_targets: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    loss_fn = nn.BCEWithLogitsLoss(reduction="none")
    success_loss = loss_fn(success_logits, success_targets.float())
    failure_loss = loss_fn(failure_logits, failure_targets.float())
    weights = weights.float().clamp(min=0.0)
    denom = weights.sum().clamp(min=1.0)
    return ((success_loss + failure_loss) * weights).sum() / denom


def binary_metric_row(model_name: str, split: str, y: np.ndarray, p: np.ndarray) -> dict[str, Any]:
    row: dict[str, Any] = {
        "model": model_name,
        "split": split,
        "rows": int(len(y)),
        "pos_rate": float(np.mean(y)),
        "mean_prob": float(np.mean(p)),
    }
    try:
        row["roc_auc"] = float(roc_auc_score(y, p))
    except Exception:
        row["roc_auc"] = float("nan")
    try:
        row["pr_auc"] = float(average_precision_score(y, p))
    except Exception:
        row["pr_auc"] = float("nan")
    try:
        row["brier"] = float(brier_score_loss(y, p))
    except Exception:
        row["brier"] = float("nan")
    try:
        row["log_loss"] = float(log_loss(y, np.clip(p, 1e-6, 1 - 1e-6), labels=[0, 1]))
    except Exception:
        row["log_loss"] = float("nan")
    return row


def safe_stop_outputs(
    *,
    output_dir: Path,
    run_label: str,
    predictor_name: str,
    valid_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    valid_success_raw: np.ndarray,
    valid_failure_raw: np.ndarray,
    test_success_raw: np.ndarray,
    test_failure_raw: np.ndarray,
    score_modes: list[str],
    success_thresholds: list[float],
    failure_thresholds: list[float],
    policy_min_steps: list[int],
    consecutive: list[int],
    max_valid_abs_drop_pp: float,
    min_valid_decision_acc: float,
    fallback_min_save_pct: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    output_dir.mkdir(parents=True, exist_ok=True)
    models_dir = output_dir / "models"
    models_dir.mkdir(exist_ok=True)

    y_success_valid = valid_frame["safe_success_label"].to_numpy(dtype=np.int8)
    y_failure_valid = valid_frame["safe_failure_label"].to_numpy(dtype=np.int8)
    y_success_test = test_frame["safe_success_label"].to_numpy(dtype=np.int8)
    y_failure_test = test_frame["safe_failure_label"].to_numpy(dtype=np.int8)
    w_valid = valid_frame["sample_weight"].to_numpy(dtype=np.float32)

    calibration_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    calibrated: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for head_name, y_valid, y_test, valid_raw, test_raw in (
        ("safe_success", y_success_valid, y_success_test, valid_success_raw, test_success_raw),
        ("safe_failure", y_failure_valid, y_failure_test, valid_failure_raw, test_failure_raw),
    ):
        calibrator = fit_sigmoid_calibrator(valid_raw, y_valid, sample_weight=w_valid)
        valid_cal = calibrator.predict(valid_raw)
        test_cal = calibrator.predict(test_raw)
        save_model(calibrator, models_dir / f"calibrator_{predictor_name}__{head_name}.pkl")
        calibrated[head_name] = (valid_cal, test_cal)
        calibration_rows.append(
            {
                "head": head_name,
                **calibration_summary_row(
                    model_name=f"{predictor_name}__{head_name}",
                    calibrator=calibrator,
                    y_valid=y_valid,
                    raw_prob_valid=valid_raw,
                    y_test=y_test,
                    raw_prob_test=test_raw,
                ),
            }
        )
        metric_rows.extend(
            [
                binary_metric_row(f"{predictor_name}__{head_name}", "valid_raw", y_valid, valid_raw),
                binary_metric_row(f"{predictor_name}__{head_name}", "valid_calibrated", y_valid, valid_cal),
                binary_metric_row(f"{predictor_name}__{head_name}", "test_raw", y_test, test_raw),
                binary_metric_row(f"{predictor_name}__{head_name}", "test_calibrated", y_test, test_cal),
            ]
        )

    valid_pred = _prediction_frame(valid_frame)
    test_pred = _prediction_frame(test_frame)
    valid_pred[_head_column("success", "raw", predictor_name)] = valid_success_raw.astype(np.float32)
    valid_pred[_head_column("failure", "raw", predictor_name)] = valid_failure_raw.astype(np.float32)
    test_pred[_head_column("success", "raw", predictor_name)] = test_success_raw.astype(np.float32)
    test_pred[_head_column("failure", "raw", predictor_name)] = test_failure_raw.astype(np.float32)
    valid_pred[_head_column("success", "calibrated", predictor_name)] = calibrated["safe_success"][0].astype(np.float32)
    valid_pred[_head_column("failure", "calibrated", predictor_name)] = calibrated["safe_failure"][0].astype(np.float32)
    test_pred[_head_column("success", "calibrated", predictor_name)] = calibrated["safe_success"][1].astype(np.float32)
    test_pred[_head_column("failure", "calibrated", predictor_name)] = calibrated["safe_failure"][1].astype(np.float32)

    valid_pred.to_parquet(output_dir / "valid_predictions_safe_stop.parquet", index=False)
    test_pred.to_parquet(output_dir / "test_predictions_safe_stop.parquet", index=False)
    pd.DataFrame(calibration_rows).to_csv(output_dir / "safe_stop_calibration_summary.csv", index=False)
    pd.DataFrame(metric_rows).to_csv(output_dir / "head_metrics.csv", index=False)

    policies = _policy_grid(
        success_thresholds=success_thresholds,
        failure_thresholds=failure_thresholds,
        min_steps=policy_min_steps,
        consecutive_values=consecutive,
    )
    valid_grid, valid_per_agent = _evaluate_policies(
        valid_pred,
        run_label=run_label,
        predictors=[predictor_name],
        score_modes=score_modes,
        policies=policies,
    )
    selected = _select_policies(
        valid_grid,
        max_valid_abs_drop_pp=max_valid_abs_drop_pp,
        min_valid_decision_acc=min_valid_decision_acc,
        fallback_min_save_pct=fallback_min_save_pct,
    )
    test_selected = _evaluate_selected(test_pred, run_label=run_label, selected=selected)

    valid_grid.to_csv(output_dir / "safe_stop_valid_policy_grid.csv", index=False)
    valid_per_agent.to_csv(output_dir / "safe_stop_valid_policy_per_agent.csv", index=False)
    selected.to_csv(output_dir / "safe_stop_selected_policies.csv", index=False)
    test_selected.to_csv(output_dir / "safe_stop_test_selected.csv", index=False)
    _write_report(output_dir, selected, test_selected)
    return selected, test_selected


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )


def normalize_weights(weights: np.ndarray) -> np.ndarray:
    weights = np.asarray(weights, dtype=np.float32)
    weights = np.where(np.isfinite(weights) & (weights > 0), weights, 1.0).astype(np.float32)
    mean = float(weights.mean()) if len(weights) else 1.0
    if mean > 0:
        weights = weights / mean
    return weights


def sample_train_indices(
    weights: np.ndarray,
    max_rows: int,
    seed: int,
) -> np.ndarray:
    n_rows = int(len(weights))
    if max_rows <= 0 or max_rows >= n_rows:
        return np.arange(n_rows, dtype=np.int64)
    weights = normalize_weights(weights).astype(np.float64)
    probs = weights / float(weights.sum()) if float(weights.sum()) > 0 else None
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n_rows, size=int(max_rows), replace=False, p=probs)).astype(np.int64)


def set_threads(max_cpu_threads: int) -> int:
    return _set_cpu_thread_limits(max_cpu_threads)
