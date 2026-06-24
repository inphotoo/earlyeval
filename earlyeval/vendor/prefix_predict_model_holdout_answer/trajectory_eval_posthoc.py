#!/usr/bin/env python3
"""Post-hoc trajectory-level evaluation for model-holdout prefix predictions.

This script does not retrain models and does not rebuild splits.  It only reads
the existing ``test_predictions_all_models.parquet`` produced by ``run_all.py``
and aggregates prefix-level probabilities into trajectory-level reports.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)


DEFAULT_RUN_NAME = "model_holdout_answer_full"
DEFAULT_THRESHOLDS = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]


def _default_predictions_path() -> Path:
    here = Path(__file__).resolve().parent
    return here / "runs" / DEFAULT_RUN_NAME / "reports" / "test_predictions_all_models.parquet"


def _default_output_dir(predictions_path: Path) -> Path:
    return predictions_path.parent / "trajectory_eval"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--predictions",
        type=Path,
        default=_default_predictions_path(),
        help="Path to test_predictions_all_models.parquet from the existing run.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for trajectory-level reports. Defaults to predictions parent / trajectory_eval.",
    )
    parser.add_argument(
        "--thresholds",
        nargs="+",
        type=float,
        default=DEFAULT_THRESHOLDS,
        help="Symmetric confidence thresholds. success if p>=thr; failure if p<=1-thr.",
    )
    parser.add_argument(
        "--predictors",
        nargs="+",
        default=None,
        help="Predictor names without prob__ prefix. Defaults to all probability columns.",
    )
    parser.add_argument("--consecutive-k", type=int, default=1)
    parser.add_argument("--delay-steps", type=int, default=0)
    parser.add_argument(
        "--strict-missing-model-id",
        dest="strict_missing_model_id",
        action="store_true",
        default=True,
        help="Fail if test model_id inputs are not all __MISSING__.",
    )
    parser.add_argument(
        "--allow-known-model-id",
        dest="strict_missing_model_id",
        action="store_false",
        help="Do not fail if model_id inputs are known. Not recommended for this experiment.",
    )
    return parser.parse_args()


def _safe_float(value: Any) -> float:
    try:
        if value is None:
            return float("nan")
        out = float(value)
        return out
    except Exception:
        return float("nan")


def _safe_div(num: float, den: float) -> float:
    return float(num) / float(den) if den else 0.0


def _metric_or_nan(fn: Any, y_true: np.ndarray, y_score: np.ndarray) -> float:
    try:
        return float(fn(y_true, y_score))
    except Exception:
        return float("nan")


def _classification_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    y_pred = (y_prob >= 0.5).astype(int)
    out: dict[str, float] = {
        "n_trajectories": int(len(y_true)),
        "n_positive": int(np.sum(y_true == 1)),
        "n_negative": int(np.sum(y_true == 0)),
        "positive_rate": _safe_div(float(np.sum(y_true == 1)), float(len(y_true))),
        "mean_prob": float(np.mean(y_prob)) if len(y_prob) else float("nan"),
        "accuracy_at_0_5": _metric_or_nan(accuracy_score, y_true, y_pred),
        "balanced_accuracy_at_0_5": _metric_or_nan(balanced_accuracy_score, y_true, y_pred),
        "precision_at_0_5": _metric_or_nan(
            lambda yt, yp: precision_score(yt, yp, zero_division=0), y_true, y_pred
        ),
        "recall_at_0_5": _metric_or_nan(
            lambda yt, yp: recall_score(yt, yp, zero_division=0), y_true, y_pred
        ),
        "f1_at_0_5": _metric_or_nan(lambda yt, yp: f1_score(yt, yp, zero_division=0), y_true, y_pred),
        "roc_auc": _metric_or_nan(roc_auc_score, y_true, y_prob),
        "pr_auc": _metric_or_nan(average_precision_score, y_true, y_prob),
        "brier": _metric_or_nan(brier_score_loss, y_true, y_prob),
        "log_loss": _metric_or_nan(
            lambda yt, yp: log_loss(yt, np.clip(yp, 1e-6, 1 - 1e-6), labels=[0, 1]), y_true, y_prob
        ),
    }
    return out


def _rank_corr(x: pd.Series, y: pd.Series, method: str) -> float:
    try:
        value = x.corr(y, method=method)
        return float(value) if value is not None else float("nan")
    except Exception:
        return float("nan")


def _jsonable(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        if math.isnan(float(value)):
            return None
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def _load_predictions(path: Path, predictors: list[str] | None) -> tuple[pd.DataFrame, list[str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Predictions parquet not found: {path}")
    df = pd.read_parquet(path)
    if df.empty:
        raise RuntimeError(f"Predictions parquet is empty: {path}")

    required = {
        "traj_id",
        "instance_id",
        "label",
        "prefix_step_idx",
        "n_steps_total_for_weighting",
        "model_id",
        "orig_model_id",
        "model_id_input_mode",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise RuntimeError(f"Predictions table missing columns: {missing}")

    prob_cols = [c for c in df.columns if c.startswith("prob__")]
    if predictors:
        wanted = [f"prob__{name}" for name in predictors]
        missing_prob = sorted(set(wanted) - set(prob_cols))
        if missing_prob:
            raise RuntimeError(f"Requested predictors missing from predictions table: {missing_prob}")
        prob_cols = wanted
    if not prob_cols:
        raise RuntimeError("No prob__ columns found in predictions table")

    df = df.copy()
    df["label"] = df["label"].astype(int)
    df["prefix_step_idx"] = df["prefix_step_idx"].astype(int)
    df["n_steps_total_for_weighting"] = df["n_steps_total_for_weighting"].astype(int)
    df = df.sort_values(["traj_id", "prefix_step_idx"]).reset_index(drop=True)
    return df, [c.removeprefix("prob__") for c in prob_cols]


def _validate_test_model_id(df: pd.DataFrame, strict: bool) -> dict[str, Any]:
    model_id_values = sorted(str(v) for v in df["model_id"].dropna().unique())
    input_modes = sorted(str(v) for v in df["model_id_input_mode"].dropna().unique())
    heldout_models = sorted(str(v) for v in df["orig_model_id"].dropna().unique())
    ok = model_id_values == ["__MISSING__"] and input_modes == ["test_missing"]
    if strict and not ok:
        raise RuntimeError(
            "Expected test model features to be unknown: "
            f"model_id={model_id_values}, model_id_input_mode={input_modes}"
        )
    return {
        "test_model_id_values": model_id_values,
        "test_model_id_input_modes": input_modes,
        "heldout_orig_model_ids": heldout_models,
        "n_heldout_orig_models": len(heldout_models),
        "test_model_id_unknown": bool(ok),
    }


def _final_step_rows(df: pd.DataFrame) -> pd.DataFrame:
    idx = df.groupby("traj_id", sort=False)["prefix_step_idx"].idxmax()
    final_df = df.loc[idx].copy().sort_values(["orig_model_id", "instance_id", "traj_id"])
    return final_df.reset_index(drop=True)


def build_final_step_reports(
    final_df: pd.DataFrame,
    predictors: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    by_model_rows: list[dict[str, Any]] = []
    bin_rows: list[dict[str, Any]] = []
    y_true_all = final_df["label"].to_numpy(dtype=int)

    bins = np.linspace(0.0, 1.0, 11)
    labels = [f"{bins[i]:.1f}-{bins[i + 1]:.1f}" for i in range(len(bins) - 1)]

    for predictor in predictors:
        prob_col = f"prob__{predictor}"
        y_prob_all = final_df[prob_col].to_numpy(dtype=float)
        row = {"predictor": predictor, **_classification_metrics(y_true_all, y_prob_all)}
        row["n_instances"] = int(final_df["instance_id"].nunique())
        row["n_heldout_models"] = int(final_df["orig_model_id"].nunique())
        row["model_id_input_mode"] = ",".join(sorted(final_df["model_id_input_mode"].astype(str).unique()))
        row["model_id_feature_value"] = ",".join(sorted(final_df["model_id"].astype(str).unique()))
        rows.append(row)

        for model_id, group in final_df.groupby("orig_model_id", sort=True):
            y_true = group["label"].to_numpy(dtype=int)
            y_prob = group[prob_col].to_numpy(dtype=float)
            model_row = {
                "predictor": predictor,
                "orig_model_id": model_id,
                "n_instances": int(group["instance_id"].nunique()),
                **_classification_metrics(y_true, y_prob),
            }
            by_model_rows.append(model_row)

        tmp = final_df[["traj_id", "instance_id", "orig_model_id", "label", prob_col]].copy()
        tmp["prob_bin"] = pd.cut(
            tmp[prob_col].astype(float),
            bins=bins,
            labels=labels,
            include_lowest=True,
            right=False,
        ).astype(str)
        tmp.loc[tmp[prob_col] >= 1.0, "prob_bin"] = labels[-1]
        tmp["pred_label_at_0_5"] = (tmp[prob_col] >= 0.5).astype(int)
        tmp["correct_at_0_5"] = (tmp["pred_label_at_0_5"] == tmp["label"]).astype(int)
        for bin_name, group in tmp.groupby("prob_bin", sort=True):
            if bin_name == "nan" or group.empty:
                continue
            bin_rows.append(
                {
                    "predictor": predictor,
                    "prob_bin": bin_name,
                    "n_trajectories": int(len(group)),
                    "n_instances": int(group["instance_id"].nunique()),
                    "actual_positive_rate": float(group["label"].mean()),
                    "predicted_positive_rate_at_0_5": float(group["pred_label_at_0_5"].mean()),
                    "accuracy_at_0_5": float(group["correct_at_0_5"].mean()),
                    "mean_prob": float(group[prob_col].mean()),
                    "min_prob": float(group[prob_col].min()),
                    "max_prob": float(group[prob_col].max()),
                }
            )

    return pd.DataFrame(rows), pd.DataFrame(by_model_rows), pd.DataFrame(bin_rows)


@dataclass
class Decision:
    decided: bool
    decision: str
    decision_step: int
    decision_prob: float
    correct: bool


def _decide_trajectory(
    steps: np.ndarray,
    probs: np.ndarray,
    label: int,
    threshold: float,
    consecutive_k: int,
    delay_steps: int,
) -> Decision:
    low = 1.0 - threshold
    success_streak = 0
    failure_streak = 0
    for step, prob in zip(steps, probs):
        step_int = int(step)
        prob_float = float(prob)
        if step_int < delay_steps:
            success_streak = 0
            failure_streak = 0
            continue
        if prob_float >= threshold:
            success_streak += 1
            failure_streak = 0
        elif prob_float <= low:
            failure_streak += 1
            success_streak = 0
        else:
            success_streak = 0
            failure_streak = 0
        if success_streak >= consecutive_k:
            return Decision(True, "success", step_int, prob_float, label == 1)
        if failure_streak >= consecutive_k:
            return Decision(True, "failure", step_int, prob_float, label == 0)
    return Decision(False, "undecided", -1, float("nan"), False)


def _threshold_metrics(decision_df: pd.DataFrame) -> dict[str, Any]:
    n_total = int(len(decision_df))
    decided = decision_df[decision_df["decided"]].copy()
    n_decided = int(len(decided))
    n_success_decided = int((decided["decision"] == "success").sum()) if n_decided else 0
    n_failure_decided = int((decided["decision"] == "failure").sum()) if n_decided else 0
    n_correct = int(decided["correct"].sum()) if n_decided else 0
    n_success_correct = int(((decided["decision"] == "success") & (decided["correct"])).sum()) if n_decided else 0
    n_failure_correct = int(((decided["decision"] == "failure") & (decided["correct"])).sum()) if n_decided else 0
    resolved_total = int((decision_df["label"] == 1).sum())
    unresolved_total = int((decision_df["label"] == 0).sum())
    saved_steps = (
        (decided["n_steps_total"] - decided["decision_step"]).clip(lower=0).sum() if n_decided else 0
    )
    total_steps = int(decision_df["n_steps_total"].sum())
    return {
        "n_total": n_total,
        "n_decided": n_decided,
        "n_undecided": int(n_total - n_decided),
        "n_correct_decided": n_correct,
        "decided_ratio": _safe_div(n_decided, n_total),
        "accuracy_decided": _safe_div(n_correct, n_decided) if n_decided else float("nan"),
        "accuracy_all_undecided_wrong": _safe_div(n_correct, n_total),
        "n_success_decided": n_success_decided,
        "n_failure_decided": n_failure_decided,
        "success_precision": _safe_div(n_success_correct, n_success_decided)
        if n_success_decided
        else float("nan"),
        "failure_precision": _safe_div(n_failure_correct, n_failure_decided)
        if n_failure_decided
        else float("nan"),
        "resolved_recall_as_success": _safe_div(n_success_correct, resolved_total),
        "unresolved_recall_as_failure": _safe_div(n_failure_correct, unresolved_total),
        "n_success_wrong": int(n_success_decided - n_success_correct),
        "n_failure_wrong": int(n_failure_decided - n_failure_correct),
        "avg_decision_step": float(decided["decision_step"].mean()) if n_decided else float("nan"),
        "median_decision_step": float(decided["decision_step"].median()) if n_decided else float("nan"),
        "avg_decision_fraction": float((decided["decision_step"] / decided["n_steps_total"]).mean())
        if n_decided
        else float("nan"),
        "saved_steps": int(saved_steps),
        "total_steps": total_steps,
        "saved_ratio_all": _safe_div(float(saved_steps), float(total_steps)),
    }


def build_threshold_reports(
    df: pd.DataFrame,
    predictors: list[str],
    thresholds: Iterable[float],
    consecutive_k: int,
    delay_steps: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    thresholds = sorted(set(round(float(t), 4) for t in thresholds))
    trajectory_groups = []
    for traj_id, group in df.groupby("traj_id", sort=False):
        first = group.iloc[0]
        trajectory_groups.append(
            {
                "traj_id": traj_id,
                "instance_id": first["instance_id"],
                "orig_model_id": first["orig_model_id"],
                "label": int(first["label"]),
                "n_steps_total": int(first["n_steps_total_for_weighting"]),
                "steps": group["prefix_step_idx"].to_numpy(dtype=int),
                "group": group,
            }
        )

    rows: list[dict[str, Any]] = []
    by_model_rows: list[dict[str, Any]] = []
    case_rows: list[dict[str, Any]] = []

    for predictor in predictors:
        prob_col = f"prob__{predictor}"
        for threshold in thresholds:
            decisions: list[dict[str, Any]] = []
            for item in trajectory_groups:
                probs = item["group"][prob_col].to_numpy(dtype=float)
                dec = _decide_trajectory(
                    item["steps"],
                    probs,
                    item["label"],
                    threshold,
                    consecutive_k=consecutive_k,
                    delay_steps=delay_steps,
                )
                row = {
                    "predictor": predictor,
                    "threshold": threshold,
                    "traj_id": item["traj_id"],
                    "instance_id": item["instance_id"],
                    "orig_model_id": item["orig_model_id"],
                    "label": item["label"],
                    "n_steps_total": item["n_steps_total"],
                    "decided": dec.decided,
                    "decision": dec.decision,
                    "decision_step": dec.decision_step,
                    "decision_prob": dec.decision_prob,
                    "correct": dec.correct,
                }
                decisions.append(row)
                case_rows.append(row)

            decision_df = pd.DataFrame(decisions)
            rows.append(
                {
                    "predictor": predictor,
                    "threshold": threshold,
                    "consecutive_k": consecutive_k,
                    "delay_steps": delay_steps,
                    **_threshold_metrics(decision_df),
                }
            )
            for model_id, group in decision_df.groupby("orig_model_id", sort=True):
                by_model_rows.append(
                    {
                        "predictor": predictor,
                        "threshold": threshold,
                        "orig_model_id": model_id,
                        "consecutive_k": consecutive_k,
                        "delay_steps": delay_steps,
                        **_threshold_metrics(group),
                    }
                )

    cases_df = pd.DataFrame(case_rows)
    return pd.DataFrame(rows), pd.DataFrame(by_model_rows), cases_df


def build_rank_reports(final_df: pd.DataFrame, predictors: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    trajectory_rows: list[dict[str, Any]] = []
    bucket_rows: list[dict[str, Any]] = []
    model_rank_rows: list[dict[str, Any]] = []
    model_rank_summary_rows: list[dict[str, Any]] = []
    global_rate = float(final_df["label"].mean())

    for predictor in predictors:
        prob_col = f"prob__{predictor}"
        tmp = final_df[["traj_id", "instance_id", "orig_model_id", "label", prob_col]].copy()
        tmp = tmp.sort_values(prob_col, ascending=False).reset_index(drop=True)
        n = len(tmp)
        tmp["rank_position"] = np.arange(1, n + 1)
        tmp["rank_percentile"] = tmp["rank_position"] / n
        tmp["rank_decile"] = np.minimum(((tmp["rank_position"] - 1) * 10 // max(n, 1)) + 1, 10).astype(int)
        tmp["rank_bucket"] = np.where(
            tmp["rank_percentile"] <= 1 / 3,
            "top",
            np.where(tmp["rank_percentile"] <= 2 / 3, "middle", "bottom"),
        )

        trajectory_rows.append(
            {
                "predictor": predictor,
                "n_trajectories": n,
                "spearman_prob_vs_label": _rank_corr(tmp[prob_col], tmp["label"], "spearman"),
                "kendall_prob_vs_label": _rank_corr(tmp[prob_col], tmp["label"], "kendall"),
                "top_10pct_resolved_rate": float(tmp[tmp["rank_decile"] == 1]["label"].mean()),
                "bottom_10pct_unresolved_rate": float(1.0 - tmp[tmp["rank_decile"] == 10]["label"].mean()),
                "top_25pct_resolved_rate": float(tmp.iloc[: max(1, math.ceil(n * 0.25))]["label"].mean()),
                "bottom_25pct_unresolved_rate": float(1.0 - tmp.iloc[-max(1, math.ceil(n * 0.25)) :]["label"].mean()),
                "global_resolved_rate": global_rate,
            }
        )

        for bucket_name, group in tmp.groupby("rank_bucket", sort=False):
            bucket_rows.append(
                {
                    "predictor": predictor,
                    "bucket_type": "thirds",
                    "bucket": bucket_name,
                    "n_trajectories": int(len(group)),
                    "actual_resolved_rate": float(group["label"].mean()),
                    "lift_vs_global": float(group["label"].mean() - global_rate),
                    "mean_prob": float(group[prob_col].mean()),
                    "min_rank_position": int(group["rank_position"].min()),
                    "max_rank_position": int(group["rank_position"].max()),
                }
            )
        for decile, group in tmp.groupby("rank_decile", sort=True):
            bucket_rows.append(
                {
                    "predictor": predictor,
                    "bucket_type": "decile",
                    "bucket": f"decile_{int(decile):02d}",
                    "n_trajectories": int(len(group)),
                    "actual_resolved_rate": float(group["label"].mean()),
                    "lift_vs_global": float(group["label"].mean() - global_rate),
                    "mean_prob": float(group[prob_col].mean()),
                    "min_rank_position": int(group["rank_position"].min()),
                    "max_rank_position": int(group["rank_position"].max()),
                }
            )

        per_model = (
            tmp.groupby("orig_model_id", sort=True)
            .agg(
                n_trajectories=("traj_id", "size"),
                n_instances=("instance_id", "nunique"),
                true_resolve_rate=("label", "mean"),
                predicted_mean_prob=(prob_col, "mean"),
                predicted_median_prob=(prob_col, "median"),
            )
            .reset_index()
        )
        per_model["true_rank"] = per_model["true_resolve_rate"].rank(method="min", ascending=False).astype(int)
        per_model["predicted_rank"] = per_model["predicted_mean_prob"].rank(method="min", ascending=False).astype(int)
        per_model["rank_delta_pred_minus_true"] = per_model["predicted_rank"] - per_model["true_rank"]
        per_model.insert(0, "predictor", predictor)
        model_rank_rows.extend(per_model.to_dict("records"))
        model_rank_summary_rows.append(
            {
                "predictor": predictor,
                "n_heldout_models": int(per_model["orig_model_id"].nunique()),
                "spearman_model_rank": _rank_corr(
                    per_model["predicted_mean_prob"], per_model["true_resolve_rate"], "spearman"
                ),
                "kendall_model_rank": _rank_corr(
                    per_model["predicted_mean_prob"], per_model["true_resolve_rate"], "kendall"
                ),
                "mean_abs_rank_delta": float(per_model["rank_delta_pred_minus_true"].abs().mean()),
                "max_abs_rank_delta": int(per_model["rank_delta_pred_minus_true"].abs().max()),
            }
        )

    return (
        pd.DataFrame(trajectory_rows),
        pd.DataFrame(bucket_rows),
        pd.DataFrame(model_rank_rows),
        pd.DataFrame(model_rank_summary_rows),
    )


def write_report(
    output_dir: Path,
    metadata: dict[str, Any],
    final_metrics: pd.DataFrame,
    threshold_sweep: pd.DataFrame,
    model_rank_summary: pd.DataFrame,
) -> None:
    best_acc = final_metrics.sort_values(["accuracy_at_0_5", "roc_auc"], ascending=False).head(5)
    best_auc = final_metrics.sort_values(["roc_auc", "accuracy_at_0_5"], ascending=False).head(5)
    best_threshold = threshold_sweep.sort_values(
        ["accuracy_decided", "decided_ratio", "saved_ratio_all"], ascending=False
    ).head(10)
    best_model_rank = model_rank_summary.sort_values(
        ["spearman_model_rank", "mean_abs_rank_delta"], ascending=[False, True]
    ).head(10)

    lines: list[str] = []
    lines.append("# Trajectory-Level Posthoc Evaluation")
    lines.append("")
    lines.append("This report is computed from the existing test prediction parquet only.")
    lines.append("No split rebuilding, retraining, or test-data filtering is performed.")
    lines.append("")
    lines.append("## Test Model-ID Mode")
    lines.append(f"- `model_id` feature values: `{metadata['test_model_id_values']}`")
    lines.append(f"- `model_id_input_mode`: `{metadata['test_model_id_input_modes']}`")
    lines.append(f"- heldout `orig_model_id` count: `{metadata['n_heldout_orig_models']}`")
    lines.append(f"- heldout `orig_model_id`: `{metadata['heldout_orig_model_ids']}`")
    lines.append("")
    lines.append("## Best Final-Step Accuracy")
    lines.append("```")
    lines.append(best_acc.to_string(index=False))
    lines.append("```")
    lines.append("")
    lines.append("## Best Final-Step ROC-AUC")
    lines.append("```")
    lines.append(best_auc.to_string(index=False))
    lines.append("```")
    lines.append("")
    lines.append("## Best Threshold Decisions")
    lines.append("```")
    lines.append(best_threshold.to_string(index=False))
    lines.append("```")
    lines.append("")
    lines.append("## Best Heldout-Model Relative Ranking")
    lines.append("```")
    lines.append(best_model_rank.to_string(index=False))
    lines.append("```")
    lines.append("")
    lines.append("## Output Files")
    for name in [
        "trajectory_final_step_metrics.csv",
        "trajectory_final_step_by_model.csv",
        "trajectory_final_step_probability_bins.csv",
        "trajectory_threshold_sweep.csv",
        "trajectory_threshold_sweep_by_model.csv",
        "trajectory_threshold_cases.parquet",
        "trajectory_rank_correlation.csv",
        "trajectory_rank_buckets.csv",
        "model_relative_ranking.csv",
        "model_relative_ranking_summary.csv",
        "trajectory_eval_metadata.json",
    ]:
        lines.append(f"- `{name}`")
    (output_dir / "trajectory_eval_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    predictions_path = args.predictions.resolve()
    output_dir = args.output_dir.resolve() if args.output_dir else _default_output_dir(predictions_path).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    df, predictors = _load_predictions(predictions_path, args.predictors)
    metadata = _validate_test_model_id(df, strict=args.strict_missing_model_id)
    metadata.update(
        {
            "predictions_path": str(predictions_path),
            "output_dir": str(output_dir),
            "n_prefix_rows": int(len(df)),
            "n_trajectories": int(df["traj_id"].nunique()),
            "n_instances": int(df["instance_id"].nunique()),
            "predictors": predictors,
            "thresholds": sorted(set(round(float(t), 4) for t in args.thresholds)),
            "consecutive_k": int(args.consecutive_k),
            "delay_steps": int(args.delay_steps),
        }
    )

    final_df = _final_step_rows(df)
    final_metrics, final_by_model, final_bins = build_final_step_reports(final_df, predictors)
    threshold_sweep, threshold_by_model, threshold_cases = build_threshold_reports(
        df,
        predictors,
        args.thresholds,
        consecutive_k=args.consecutive_k,
        delay_steps=args.delay_steps,
    )
    rank_corr, rank_buckets, model_rank, model_rank_summary = build_rank_reports(final_df, predictors)

    final_metrics.to_csv(output_dir / "trajectory_final_step_metrics.csv", index=False)
    final_by_model.to_csv(output_dir / "trajectory_final_step_by_model.csv", index=False)
    final_bins.to_csv(output_dir / "trajectory_final_step_probability_bins.csv", index=False)
    threshold_sweep.to_csv(output_dir / "trajectory_threshold_sweep.csv", index=False)
    threshold_by_model.to_csv(output_dir / "trajectory_threshold_sweep_by_model.csv", index=False)
    threshold_cases.to_parquet(output_dir / "trajectory_threshold_cases.parquet", index=False)
    rank_corr.to_csv(output_dir / "trajectory_rank_correlation.csv", index=False)
    rank_buckets.to_csv(output_dir / "trajectory_rank_buckets.csv", index=False)
    model_rank.to_csv(output_dir / "model_relative_ranking.csv", index=False)
    model_rank_summary.to_csv(output_dir / "model_relative_ranking_summary.csv", index=False)

    with (output_dir / "trajectory_eval_metadata.json").open("w", encoding="utf-8") as fp:
        json.dump(metadata, fp, ensure_ascii=False, indent=2, default=_jsonable)
    write_report(output_dir, metadata, final_metrics, threshold_sweep, model_rank_summary)

    print(f"[trajectory_eval] predictions: {predictions_path}")
    print(f"[trajectory_eval] output_dir: {output_dir}")
    print(
        "[trajectory_eval] rows="
        f"{metadata['n_prefix_rows']} trajectories={metadata['n_trajectories']} "
        f"instances={metadata['n_instances']} heldout_models={metadata['n_heldout_orig_models']}"
    )
    print(f"[trajectory_eval] test model_id values: {metadata['test_model_id_values']}")
    print(f"[trajectory_eval] wrote trajectory_eval_report.md and CSV/parquet reports")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
