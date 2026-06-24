#!/usr/bin/env python3
"""Reference-style model ranking report for the heldout-model test predictions.

This is a post-hoc reporter: it only reads the existing
``test_predictions_all_models.parquet`` from a completed run.  It does not
retrain, rebuild splits, or change the test set.
"""

from __future__ import annotations

import argparse
import math
import os
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

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
    os.environ.setdefault(
        _thread_env_name,
        os.environ.get("SWE_MAX_CPU_THREADS", "24"),
    )

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)


DEFAULT_RUN_NAME = "model_holdout_answer_full"
DEFAULT_THRESHOLDS = [0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
DEFAULT_MIN_MAIN_PREFIX_MODELS = 4
DEFAULT_MIN_OVERALL_PREFIX_MODELS = 1


def _default_predictions_path() -> Path:
    return PROJECT_ROOT / "runs" / DEFAULT_RUN_NAME / "reports" / "test_predictions_all_models.parquet"


def _default_output_dir(predictions_path: Path) -> Path:
    return predictions_path.parent / "model_ranking_report_like_ref"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, default=_default_predictions_path())
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--thresholds", nargs="+", type=float, default=DEFAULT_THRESHOLDS)
    parser.add_argument(
        "--prefix-models",
        nargs="+",
        default=["auto_good"],
        help=(
            "Prefix predictors without probability-column prefix. Use auto_good to select "
            "top main predictors plus the best overall diagnostic predictor."
        ),
    )
    parser.add_argument("--top-main", type=int, default=DEFAULT_MIN_MAIN_PREFIX_MODELS)
    parser.add_argument("--top-overall-extra", type=int, default=DEFAULT_MIN_OVERALL_PREFIX_MODELS)
    parser.add_argument("--plot-thresholds", nargs="+", type=float, default=[0.60, 0.75, 0.85, 0.90])
    parser.add_argument(
        "--score-mode",
        choices=["raw", "calibrated"],
        default="raw",
        help="Use raw prob__* columns or validation-calibrated prob_cal__* columns.",
    )
    return parser.parse_args()


def _pct(value: float | None) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "nan%"
    return f"{value:.1%}"


def _display_width(text: object) -> int:
    width = 0
    for char in str(text):
        width += 2 if unicodedata.east_asian_width(char) in {"F", "W"} else 1
    return width


def _fit_display(text: object, width: int) -> str:
    out = ""
    used = 0
    for char in str(text):
        char_width = 2 if unicodedata.east_asian_width(char) in {"F", "W"} else 1
        if used + char_width > width:
            break
        out += char
        used += char_width
    return out


def _cell(text: object, width: int, align: str = "left") -> str:
    text_str = _fit_display(text, width)
    pad = max(width - _display_width(text_str), 0)
    if align == "right":
        return " " * pad + text_str
    if align == "center":
        left = pad // 2
        right = pad - left
        return " " * left + text_str + " " * right
    return text_str + " " * pad


def _safe_div(num: float, den: float) -> float:
    return float(num) / float(den) if den else 0.0


def _safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name)


def _short_agent_name(name: str) -> str:
    name = re.sub(r"^20\d{6}_mini-v[0-9.]+_", "", name)
    name = name.replace("2025-12-11", "20251211")
    name = name.replace("2025-11-13", "20251113")
    return name[:42]


def _metric_or_nan(fn: Any, y_true: np.ndarray, y_score: np.ndarray) -> float:
    try:
        return float(fn(y_true, y_score))
    except Exception:
        return float("nan")


def _prob_prefix(score_mode: str) -> str:
    return "prob_cal__" if score_mode == "calibrated" else "prob__"


def _prob_col(predictor: str, score_mode: str) -> str:
    return f"{_prob_prefix(score_mode)}{predictor}"


def _load_predictions(path: Path, score_mode: str) -> tuple[pd.DataFrame, list[str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Predictions parquet not found: {path}")
    df = pd.read_parquet(path)
    required = {
        "traj_id",
        "instance_id",
        "label",
        "prefix_step_idx",
        "model_id",
        "orig_model_id",
        "model_id_input_mode",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise RuntimeError(f"Predictions table missing required columns: {missing}")
    prefix = _prob_prefix(score_mode)
    prob_cols = [c for c in df.columns if c.startswith(prefix)]
    if not prob_cols:
        raise RuntimeError(f"No {prefix} columns found in predictions table")

    df = df.copy()
    df["label"] = df["label"].astype(int)
    df["prefix_step_idx"] = df["prefix_step_idx"].astype(int)
    df = df.sort_values(["traj_id", "prefix_step_idx"]).reset_index(drop=True)
    return df, [c[len(prefix):] for c in prob_cols]


def _validate_missing_model_id(df: pd.DataFrame) -> dict[str, Any]:
    model_id_values = sorted(str(v) for v in df["model_id"].dropna().unique())
    input_modes = sorted(str(v) for v in df["model_id_input_mode"].dropna().unique())
    heldout = sorted(str(v) for v in df["orig_model_id"].dropna().unique())
    ok = model_id_values == ["__MISSING__"] and input_modes == ["test_missing"]
    if not ok:
        raise RuntimeError(
            "This report expects heldout test model_id inputs to be unknown: "
            f"model_id={model_id_values}, model_id_input_mode={input_modes}"
        )
    return {
        "model_id_values": model_id_values,
        "model_id_input_modes": input_modes,
        "heldout_models": heldout,
    }


def _final_step_df(df: pd.DataFrame) -> pd.DataFrame:
    idx = df.groupby("traj_id", sort=False)["prefix_step_idx"].idxmax()
    return df.loc[idx].copy().reset_index(drop=True)


def compute_final_step_leaderboard(df: pd.DataFrame, predictors: list[str], score_mode: str) -> pd.DataFrame:
    final_df = _final_step_df(df)
    y_true = final_df["label"].to_numpy(dtype=int)
    rows: list[dict[str, Any]] = []
    for predictor in predictors:
        prob_col = _prob_col(predictor, score_mode)
        y_prob = final_df[prob_col].to_numpy(dtype=float)
        y_pred = (y_prob >= 0.5).astype(int)
        rows.append(
            {
                "prefix_model": predictor,
                "is_ablation": predictor.startswith("Abl_"),
                "n_trajectories": int(len(final_df)),
                "n_instances": int(final_df["instance_id"].nunique()),
                "positive_rate": float(y_true.mean()),
                "accuracy_at_0_5": _metric_or_nan(accuracy_score, y_true, y_pred),
                "roc_auc": _metric_or_nan(roc_auc_score, y_true, y_prob),
                "pr_auc": _metric_or_nan(average_precision_score, y_true, y_prob),
                "brier": _metric_or_nan(brier_score_loss, y_true, y_prob),
                "log_loss": _metric_or_nan(
                    lambda yt, yp: log_loss(yt, np.clip(yp, 1e-6, 1 - 1e-6), labels=[0, 1]),
                    y_true,
                    y_prob,
                ),
                "mean_prob": float(y_prob.mean()),
            }
        )
    out = pd.DataFrame(rows)
    return out.sort_values(["roc_auc", "accuracy_at_0_5"], ascending=False).reset_index(drop=True)


def select_prefix_models(
    requested: list[str],
    leaderboard: pd.DataFrame,
    available: list[str],
    top_main: int,
    top_overall_extra: int,
) -> list[str]:
    if requested and requested != ["auto_good"]:
        missing = sorted(set(requested) - set(available))
        if missing:
            raise RuntimeError(f"Requested prefix models not in predictions: {missing}")
        return list(dict.fromkeys(requested))

    selected: list[str] = []
    main = leaderboard[~leaderboard["is_ablation"]].head(top_main)
    selected.extend(main["prefix_model"].tolist())
    overall = leaderboard.head(max(top_main + top_overall_extra, top_main))
    for name in overall["prefix_model"].tolist():
        if name not in selected:
            selected.append(name)
        if len([x for x in selected if x.startswith("Abl_")]) >= top_overall_extra:
            break
    for must_keep in [
        "D_Dense_Full_LR",
        "G_TfIdf_Full_LR",
        "I_LightGBM_Dense_AF",
        "J_LightGBM_Dense_AF_Thought",
    ]:
        if must_keep in available and must_keep not in selected:
            selected.append(must_keep)
    return selected


@dataclass
class Decision:
    decided: bool
    decision: str
    decision_step: int
    decision_prob: float
    correct: bool


def decide_step(group: pd.DataFrame, prob_col: str, threshold: float) -> Decision:
    low = 1.0 - threshold
    label = int(group["label"].iloc[0])
    for _, row in group.sort_values("prefix_step_idx").iterrows():
        step = int(row["prefix_step_idx"])
        prob = float(row[prob_col])
        if prob >= threshold:
            return Decision(True, "success", step, prob, label == 1)
        if prob <= low:
            return Decision(True, "failure", step, prob, label == 0)
    return Decision(False, "undecided", -1, float("nan"), False)


def original_stats(df: pd.DataFrame) -> dict[str, dict[str, Any]]:
    final_df = _final_step_df(df)
    out: dict[str, dict[str, Any]] = {}
    for model, group in final_df.groupby("orig_model_id", sort=True):
        total = int(len(group))
        resolved = int(group["label"].sum())
        out[str(model)] = {
            "total": total,
            "resolved": resolved,
            "resolve_rate": resolved / total if total else 0.0,
        }
    return out


def compute_adjusted_rates(
    df: pd.DataFrame,
    prefix_models: list[str],
    thresholds: Iterable[float],
    originals: dict[str, dict[str, Any]],
    good_prefix_models: set[str],
    score_mode: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    trajectory_groups = [(traj_id, group) for traj_id, group in df.groupby("traj_id", sort=False)]
    thresholds = sorted(set(round(float(t), 4) for t in thresholds))

    for prefix_model in prefix_models:
        prob_col = _prob_col(prefix_model, score_mode)
        for threshold in thresholds:
            per_agent: dict[str, dict[str, Any]] = {}
            for model in originals:
                per_agent[model] = {
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

            for _, group in trajectory_groups:
                agent_model = str(group["orig_model_id"].iloc[0])
                if agent_model not in per_agent:
                    continue
                label = int(group["label"].iloc[0])
                n_steps = int(len(group))
                per_agent[agent_model]["total_steps"] += n_steps
                decision = decide_step(group, prob_col, threshold)
                if not decision.decided:
                    per_agent[agent_model]["undecided"] += 1
                    continue
                per_agent[agent_model]["total_saved_steps"] += max(n_steps - decision.decision_step - 1, 0)
                if decision.decision == "failure":
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

            for agent_model, counts in per_agent.items():
                orig = originals[agent_model]
                true_positives = int(counts["true_positives"])
                true_negatives = int(counts["true_negatives"])
                false_positives = int(counts["false_positives"])
                false_negatives = int(counts["false_negatives"])
                decided_success = int(counts["decided_success"])
                decided_failure = int(counts["decided_failure"])
                n_decided = decided_success + decided_failure
                undecided_resolved = orig["resolved"] - true_positives - false_negatives
                adjusted_resolved = true_positives + undecided_resolved
                adjusted_rate = adjusted_resolved / orig["total"] if orig["total"] else 0.0
                total_steps = int(counts["total_steps"])
                saved_steps = int(counts["total_saved_steps"])
                decision_acc = (true_positives + true_negatives) / n_decided if n_decided else float("nan")
                precision_success = true_positives / decided_success if decided_success else float("nan")
                precision_failure = true_negatives / decided_failure if decided_failure else float("nan")
                rows.append(
                    {
                        "agent_model": agent_model,
                        "prefix_model": prefix_model,
                        "good_prefix_model": prefix_model in good_prefix_models,
                        "threshold": threshold,
                        "original_total": int(orig["total"]),
                        "original_resolved": int(orig["resolved"]),
                        "original_resolve_rate": round(float(orig["resolve_rate"]), 6),
                        "decided_failure": decided_failure,
                        "decided_success": decided_success,
                        "undecided": int(counts["undecided"]),
                        "false_negatives": false_negatives,
                        "true_negatives": true_negatives,
                        "false_positives": false_positives,
                        "true_positives": true_positives,
                        "n_decided": n_decided,
                        "decision_accuracy": round(decision_acc, 6) if not math.isnan(decision_acc) else np.nan,
                        "precision_success": round(precision_success, 6)
                        if not math.isnan(precision_success)
                        else np.nan,
                        "precision_failure": round(precision_failure, 6)
                        if not math.isnan(precision_failure)
                        else np.nan,
                        "adjusted_resolved": int(adjusted_resolved),
                        "adjusted_resolve_rate": round(float(adjusted_rate), 6),
                        "resolve_rate_drop": round(float(orig["resolve_rate"] - adjusted_rate), 6),
                        "pct_steps_saved": round(_safe_div(saved_steps * 100.0, total_steps), 6),
                        "total_saved_steps": saved_steps,
                        "total_steps": total_steps,
                    }
                )
    return pd.DataFrame(rows)


def compute_ranking_preservation(all_rates: pd.DataFrame, prefix_models: list[str], thresholds: Iterable[float]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    thresholds = sorted(set(round(float(t), 4) for t in thresholds))
    for prefix_model in prefix_models:
        for threshold in thresholds:
            sub = all_rates[
                (all_rates["prefix_model"] == prefix_model) & (all_rates["threshold"] == threshold)
            ].copy()
            if len(sub) < 2:
                rows.append(
                    {
                        "prefix_model": prefix_model,
                        "threshold": threshold,
                        "n_agent_models": int(len(sub)),
                        "kendall_tau": np.nan,
                        "kendall_p": np.nan,
                        "spearman_rho": np.nan,
                        "spearman_p": np.nan,
                        "max_rank_change": np.nan,
                    }
                )
                continue
            orig_ranks = sub["original_resolve_rate"].rank(ascending=False, method="min")
            adj_ranks = sub["adjusted_resolve_rate"].rank(ascending=False, method="min")
            max_rank_change = int((orig_ranks - adj_ranks).abs().max())
            tau, tau_p = sp_stats.kendalltau(orig_ranks, adj_ranks)
            rho, rho_p = sp_stats.spearmanr(orig_ranks, adj_ranks)
            rows.append(
                {
                    "prefix_model": prefix_model,
                    "threshold": threshold,
                    "n_agent_models": int(len(sub)),
                    "kendall_tau": round(float(tau), 6) if not math.isnan(float(tau)) else np.nan,
                    "kendall_p": round(float(tau_p), 6) if not math.isnan(float(tau_p)) else np.nan,
                    "spearman_rho": round(float(rho), 6) if not math.isnan(float(rho)) else np.nan,
                    "spearman_p": round(float(rho_p), 6) if not math.isnan(float(rho_p)) else np.nan,
                    "max_rank_change": max_rank_change,
                }
            )
    return pd.DataFrame(rows)


def _ranked_sub(sub: pd.DataFrame) -> pd.DataFrame:
    sub = sub.copy().sort_values("original_resolve_rate", ascending=False)
    sub["orig_rank"] = sub["original_resolve_rate"].rank(ascending=False, method="min").astype(int)
    sub["adj_rank"] = sub["adjusted_resolve_rate"].rank(ascending=False, method="min").astype(int)
    sub["rank_delta"] = sub["orig_rank"] - sub["adj_rank"]
    return sub


def generate_report(
    all_rates: pd.DataFrame,
    ranking_df: pd.DataFrame,
    originals: dict[str, dict[str, Any]],
    leaderboard: pd.DataFrame,
    prefix_models: list[str],
    thresholds: list[float],
    metadata: dict[str, Any],
    output_dir: Path,
) -> str:
    good_set = set(prefix_models)
    lines: list[str] = []
    lines.append("=" * 98)
    lines.append('Public-release English note.')
    lines.append("=" * 98)
    lines.append("")
    lines.append('Public-release English note.')
    lines.append(
        'Public-release English note.'
        f"{metadata['n_trajectories']} trajectories, {metadata['n_instances']} instances"
    )
    lines.append(f"  heldout agent models: {len(metadata['heldout_models'])}")
    lines.append(
        f"  score mode: {metadata.get('score_mode', 'raw')} "
        f"({metadata.get('score_column_prefix', 'prob__')} columns)"
    )
    lines.append("  model_id feature values: ['__MISSING__']; model_id_input_mode: ['test_missing']")
    lines.append("")
    lines.append('Public-release English note.')
    lines.append('Public-release English note.')
    lines.append('Public-release English note.')
    lines.append('Public-release English note.')
    lines.append('Public-release English note.')
    lines.append('Public-release English note.')
    lines.append("")

    lines.append('Public-release English note.')
    lines.append("-" * 98)
    selected = leaderboard[leaderboard["prefix_model"].isin(prefix_models)].copy()
    selected["plot_marker"] = "★"
    lines.append(
        'Public-release English note.'
        f"{_cell('Acc@0.5', 8, 'right')} {_cell('ROC-AUC', 8, 'right')} "
        f"{_cell('PR-AUC', 8, 'right')} {_cell('Brier', 8, 'right')}"
    )
    lines.append(f"  {'-'*4} {'-'*36} {'-'*5} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
    for _, row in selected.iterrows():
        lines.append(
            f"  {_cell('★', 4)} {_cell(row['prefix_model'], 36)} "
            f"{_cell(str(bool(row['is_ablation'])), 5, 'right')} "
            f"{row['accuracy_at_0_5']:>8.3f} {row['roc_auc']:>8.3f} "
            f"{row['pr_auc']:>8.3f} {row['brier']:>8.3f}"
        )
    lines.append("")

    lines.append('Public-release English note.')
    lines.append("-" * 98)
    orig_sorted = sorted(originals.items(), key=lambda x: x[1]["resolve_rate"], reverse=True)
    best_agent = orig_sorted[0][0] if orig_sorted else ""
    lines.append(
        'Public-release English note.'
        'Public-release English note.'
    )
    lines.append(f"  {'-'*4}  {'-'*62} {'-'*5} {'-'*8} {'-'*8}")
    for rank, (name, stats) in enumerate(orig_sorted, 1):
        mark = "★ " if name == best_agent else "  "
        lines.append(
            f"  {rank:>4d}  {_cell(mark + name, 62)} {stats['total']:>5d} "
            f"{stats['resolved']:>8d} {stats['resolve_rate']:>7.1%}"
        )
    lines.append('Public-release English note.')

    section_num = 2
    for prefix_model in prefix_models:
        marker = "★ " if prefix_model in good_set else ""
        lines.append("")
        lines.append('Public-release English note.')
        lines.append("-" * 98)
        section_num += 1
        for threshold in thresholds:
            sub = all_rates[
                (all_rates["prefix_model"] == prefix_model) & (all_rates["threshold"] == threshold)
            ].copy()
            if sub.empty:
                continue
            sub = _ranked_sub(sub)
            agg_decided = int(sub["n_decided"].sum())
            agg_correct = int((sub["true_positives"] + sub["true_negatives"]).sum())
            agg_total = int(sub["original_total"].sum())
            agg_acc = agg_correct / agg_decided if agg_decided else None
            agg_fn = int(sub["false_negatives"].sum())
            agg_fp = int(sub["false_positives"].sum())

            lines.append(f"\n  Threshold = {threshold:.2f}")
            if agg_decided:
                lines.append(
                    'Public-release English note.'
                    'Public-release English note.'
                )
            else:
                lines.append('Public-release English note.')
            lines.append('Public-release English note.')
            lines.append(
                'Public-release English note.'
                'Public-release English note.'
                'Public-release English note.'
                f"{_cell('FN', 3, 'right')} {_cell('FP', 3, 'right')} "
                'Public-release English note.'
            )
            lines.append(
                f"  {'-'*6} {'-'*6} {'-'*3}  {'-'*42} "
                f"{'-'*7} {'-'*7} {'-'*6} {'-'*3} {'-'*3} "
                f"{'-'*6} {'-'*6} {'-'*6}"
            )
            for _, row in sub.iterrows():
                delta = int(row["rank_delta"])
                delta_str = f"{delta:+d}" if delta else " ="
                da_str = _pct(float(row["decision_accuracy"])) if not pd.isna(row["decision_accuracy"]) else "nan%"
                name = _short_agent_name(str(row["agent_model"]))
                lines.append(
                    f"  {int(row['orig_rank']):>6d} {int(row['adj_rank']):>6d} {delta_str:>3s}  "
                    f"{_cell(name, 42)} {row['original_resolve_rate']:>6.1%} "
                    f"{row['adjusted_resolve_rate']:>6.1%} {row['resolve_rate_drop']:>5.1%} "
                    f"{int(row['false_negatives']):>3d} {int(row['false_positives']):>3d} "
                    f"{int(row['n_decided']):>6d} {da_str:>6s} {row['pct_steps_saved']:>5.1f}%"
                )

    lines.append("")
    lines.append('Public-release English note.')
    lines.append("-" * 98)
    lines.append(
        'Public-release English note.'
        f"{_cell('Kendall τ', 10, 'right')} {_cell('Spearman ρ', 11, 'right')} "
        'Public-release English note.'
    )
    lines.append(f"  {'-'*36} {'-'*6} {'-'*6} {'-'*10} {'-'*11} {'-'*12}")
    for _, row in ranking_df.iterrows():
        tau = row["kendall_tau"]
        rho = row["spearman_rho"]
        max_change = row["max_rank_change"]
        lines.append(
            f"  {_cell(row['prefix_model'], 36)} {row['threshold']:>6.2f} "
            f"{int(row['n_agent_models']):>6d} "
            f"{tau:>10.4f} {rho:>11.4f} {int(max_change):>12d}"
        )
    section_num += 1

    lines.append("")
    lines.append('Public-release English note.')
    lines.append("-" * 98)
    for prefix_model in prefix_models:
        pm_data = all_rates[all_rates["prefix_model"] == prefix_model]
        if pm_data.empty:
            continue
        lines.append(f"\n  Prefix Model: ★ {prefix_model}")
        for threshold in thresholds:
            sub = pm_data[pm_data["threshold"] == threshold]
            if sub.empty:
                continue
            total_decided = int(sub["n_decided"].sum())
            total_wrong = int(sub["false_negatives"].sum() + sub["false_positives"].sum())
            decision_acc = (total_decided - total_wrong) / total_decided if total_decided else None
            weighted_saved = _safe_div(float(sub["total_saved_steps"].sum()) * 100.0, float(sub["total_steps"].sum()))
            avg_drop = float(sub["resolve_rate_drop"].mean())
            max_abs_rank_change = int(
                ranking_df[
                    (ranking_df["prefix_model"] == prefix_model) & (ranking_df["threshold"] == threshold)
                ]["max_rank_change"].iloc[0]
            )
            lines.append(
                'Public-release English note.'
                'Public-release English note.'
                'Public-release English note.'
            )
    section_num += 1

    lines.append("")
    lines.append('Public-release English note.')
    lines.append("-" * 98)
    top_final = leaderboard.head(12)
    lines.append(
        f"  {_cell('Prefix Model', 36)} {_cell('Abl?', 5, 'right')} "
        f"{_cell('Acc@0.5', 8, 'right')} {_cell('ROC-AUC', 8, 'right')} "
        f"{_cell('PR-AUC', 8, 'right')} {_cell('Brier', 8, 'right')} {_cell('MeanP', 8, 'right')}"
    )
    lines.append(f"  {'-'*36} {'-'*5} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
    for _, row in top_final.iterrows():
        mark = "★" if row["prefix_model"] in good_set else " "
        lines.append(
            f"  {_cell(mark + row['prefix_model'], 36)} {str(bool(row['is_ablation'])):>5s} "
            f"{row['accuracy_at_0_5']:>8.3f} {row['roc_auc']:>8.3f} "
            f"{row['pr_auc']:>8.3f} {row['brier']:>8.3f} {row['mean_prob']:>8.3f}"
        )
    lines.append("")
    lines.append('Public-release English note.')
    lines.append('Public-release English note.')

    lines.append("")
    lines.append('Public-release English note.')
    lines.append("-" * 98)
    for name in [
        "report.txt",
        "agent_model_adjusted_rates.csv",
        "ranking_preservation.csv",
        "final_step_prefix_model_leaderboard.csv",
        "final_step_probability_bins.csv",
        "prefix_model_final_step_leaderboard.png",
        "resolve_rate_comparison_<prefix_model>.png",
        "rate_drop_vs_threshold_<prefix_model>.png",
        "steps_saved_vs_threshold_<prefix_model>.png",
        "tradeoff_<prefix_model>.png",
    ]:
        lines.append(f"  - {name}")
    lines.append('Public-release English note.')
    return "\n".join(lines) + "\n"


def make_probability_bins(df: pd.DataFrame, prefix_models: list[str], score_mode: str) -> pd.DataFrame:
    final_df = _final_step_df(df)
    bins = np.linspace(0.0, 1.0, 11)
    labels = [f"{bins[i]:.1f}-{bins[i + 1]:.1f}" for i in range(len(bins) - 1)]
    rows: list[dict[str, Any]] = []
    for prefix_model in prefix_models:
        prob_col = _prob_col(prefix_model, score_mode)
        tmp = final_df[["traj_id", "instance_id", "orig_model_id", "label", prob_col]].copy()
        tmp["prob_bin"] = pd.cut(
            tmp[prob_col].astype(float),
            bins=bins,
            labels=labels,
            include_lowest=True,
            right=False,
        ).astype(str)
        tmp.loc[tmp[prob_col] >= 1.0, "prob_bin"] = labels[-1]
        tmp["pred"] = (tmp[prob_col] >= 0.5).astype(int)
        tmp["correct"] = (tmp["pred"] == tmp["label"]).astype(int)
        for prob_bin, group in tmp.groupby("prob_bin", sort=True):
            if prob_bin == "nan" or group.empty:
                continue
            rows.append(
                {
                    "prefix_model": prefix_model,
                    "prob_bin": prob_bin,
                    "n_trajectories": int(len(group)),
                    "actual_resolve_rate": float(group["label"].mean()),
                    "accuracy_at_0_5": float(group["correct"].mean()),
                    "mean_prob": float(group[prob_col].mean()),
                }
            )
    return pd.DataFrame(rows)


def make_plots(
    all_rates: pd.DataFrame,
    leaderboard: pd.DataFrame,
    prefix_models: list[str],
    plot_thresholds: list[float],
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    selected_leaderboard = leaderboard.head(14).copy()
    selected_leaderboard["label"] = selected_leaderboard["prefix_model"].map(
        lambda x: ("★ " if x in prefix_models else "  ") + x
    )
    fig, ax = plt.subplots(figsize=(11, 7))
    y = np.arange(len(selected_leaderboard))
    colors = ["tab:orange" if name in prefix_models else "tab:blue" for name in selected_leaderboard["prefix_model"]]
    ax.barh(y - 0.18, selected_leaderboard["roc_auc"], height=0.35, color=colors, alpha=0.85, label="ROC-AUC")
    ax.barh(
        y + 0.18,
        selected_leaderboard["accuracy_at_0_5"],
        height=0.35,
        color="tab:green",
        alpha=0.55,
        label="Accuracy@0.5",
    )
    ax.set_yticks(y)
    ax.set_yticklabels(selected_leaderboard["label"], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(0.55, 0.95)
    ax.set_xlabel("Trajectory final-step score")
    ax.set_title("Final-step trajectory leaderboard (★ selected for detailed ranking report)")
    ax.grid(axis="x", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "prefix_model_final_step_leaderboard.png", dpi=160)
    plt.close(fig)

    for prefix_model in prefix_models:
        pm_data = all_rates[all_rates["prefix_model"] == prefix_model].copy()
        if pm_data.empty:
            continue
        agents = (
            pm_data[["agent_model", "original_resolve_rate"]]
            .drop_duplicates()
            .sort_values("original_resolve_rate", ascending=False)["agent_model"]
            .tolist()
        )
        thresholds = sorted(pm_data["threshold"].unique())
        chosen_thresholds = [t for t in plot_thresholds if t in thresholds]
        if not chosen_thresholds:
            chosen_thresholds = thresholds[: min(4, len(thresholds))]

        fig, axes = plt.subplots(1, len(chosen_thresholds), figsize=(5 * len(chosen_thresholds), 4.2), sharey=True)
        if len(chosen_thresholds) == 1:
            axes = [axes]
        for ax, threshold in zip(axes, chosen_thresholds):
            sub = _ranked_sub(pm_data[pm_data["threshold"] == threshold]).sort_values(
                "original_resolve_rate", ascending=True
            )
            y = np.arange(len(sub))
            labels = [_short_agent_name(x) for x in sub["agent_model"]]
            ax.barh(y, sub["original_resolve_rate"] * 100, alpha=0.50, label="Original", color="steelblue")
            ax.barh(y, sub["adjusted_resolve_rate"] * 100, alpha=0.70, label="After decision", color="coral")
            ax.set_yticks(y)
            ax.set_yticklabels(labels, fontsize=8)
            ax.set_xlabel("Resolve Rate (%)")
            ax.set_title(f"thr={threshold:.2f}")
            ax.grid(axis="x", alpha=0.25)
            if ax is axes[0]:
                ax.legend(fontsize=8)
        fig.suptitle(f"Resolve-rate comparison - ★ {prefix_model}", fontsize=12)
        fig.tight_layout()
        fig.savefig(output_dir / f"resolve_rate_comparison_{_safe_name(prefix_model)}.png", dpi=160)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(9, 5.5))
        for agent in agents:
            sub = pm_data[pm_data["agent_model"] == agent].sort_values("threshold")
            label = ("★ " if agent == agents[0] else "") + _short_agent_name(agent)
            ax.plot(
                sub["threshold"].to_numpy(dtype=float),
                (sub["resolve_rate_drop"] * 100).to_numpy(dtype=float),
                "o-",
                label=label,
                markersize=5,
            )
        ax.axhline(0, color="black", linewidth=0.8, alpha=0.5)
        ax.set_xlabel("Threshold")
        ax.set_ylabel("Original - adjusted resolve rate (pp)")
        ax.set_title(f"Rate change vs threshold - ★ {prefix_model}")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(output_dir / f"rate_drop_vs_threshold_{_safe_name(prefix_model)}.png", dpi=160)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(9, 5.5))
        for agent in agents:
            sub = pm_data[pm_data["agent_model"] == agent].sort_values("threshold")
            label = ("★ " if agent == agents[0] else "") + _short_agent_name(agent)
            ax.plot(
                sub["threshold"].to_numpy(dtype=float),
                sub["pct_steps_saved"].to_numpy(dtype=float),
                "o-",
                label=label,
                markersize=5,
            )
        ax.set_xlabel("Threshold")
        ax.set_ylabel("Prefix steps saved (%)")
        ax.set_title(f"Compute savings vs threshold - ★ {prefix_model}")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(output_dir / f"steps_saved_vs_threshold_{_safe_name(prefix_model)}.png", dpi=160)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(9, 6))
        for agent in agents:
            sub = pm_data[pm_data["agent_model"] == agent].sort_values("threshold")
            label = ("★ " if agent == agents[0] else "") + _short_agent_name(agent)
            ax.plot(
                sub["pct_steps_saved"].to_numpy(dtype=float),
                (sub["resolve_rate_drop"] * 100).to_numpy(dtype=float),
                "o-",
                label=label,
                markersize=5,
            )
            for _, row in sub.iterrows():
                ax.annotate(
                    f"{row['threshold']:.2f}",
                    (row["pct_steps_saved"], row["resolve_rate_drop"] * 100),
                    fontsize=6,
                    alpha=0.7,
                )
        ax.axhline(0, color="black", linewidth=0.8, alpha=0.5)
        ax.set_xlabel("Prefix steps saved (%)")
        ax.set_ylabel("Original - adjusted resolve rate (pp)")
        ax.set_title(f"Savings vs rate change - ★ {prefix_model}")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(output_dir / f"tradeoff_{_safe_name(prefix_model)}.png", dpi=160)
        plt.close(fig)


def main() -> int:
    args = parse_args()
    predictions_path = args.predictions.resolve()
    output_dir = args.output_dir.resolve() if args.output_dir else _default_output_dir(predictions_path).resolve()
    if args.output_dir is None and args.score_mode == "calibrated":
        output_dir = output_dir.with_name(output_dir.name + "_calibrated")
    output_dir.mkdir(parents=True, exist_ok=True)

    df, available_predictors = _load_predictions(predictions_path, args.score_mode)
    id_metadata = _validate_missing_model_id(df)
    originals = original_stats(df)
    leaderboard = compute_final_step_leaderboard(df, available_predictors, args.score_mode)
    prefix_models = select_prefix_models(
        args.prefix_models,
        leaderboard,
        available_predictors,
        top_main=args.top_main,
        top_overall_extra=args.top_overall_extra,
    )
    thresholds = sorted(set(round(float(t), 4) for t in args.thresholds))
    all_rates = compute_adjusted_rates(df, prefix_models, thresholds, originals, set(prefix_models), args.score_mode)
    ranking_df = compute_ranking_preservation(all_rates, prefix_models, thresholds)
    prob_bins = make_probability_bins(df, prefix_models, args.score_mode)

    all_rates.to_csv(output_dir / "agent_model_adjusted_rates.csv", index=False)
    ranking_df.to_csv(output_dir / "ranking_preservation.csv", index=False)
    leaderboard.to_csv(output_dir / "final_step_prefix_model_leaderboard.csv", index=False)
    prob_bins.to_csv(output_dir / "final_step_probability_bins.csv", index=False)

    metadata = {
        **id_metadata,
        "predictions_path": str(predictions_path),
        "output_dir": str(output_dir),
        "n_prefix_rows": int(len(df)),
        "n_trajectories": int(df["traj_id"].nunique()),
        "n_instances": int(df["instance_id"].nunique()),
        "score_mode": args.score_mode,
        "score_column_prefix": _prob_prefix(args.score_mode),
    }
    report = generate_report(
        all_rates=all_rates,
        ranking_df=ranking_df,
        originals=originals,
        leaderboard=leaderboard,
        prefix_models=prefix_models,
        thresholds=thresholds,
        metadata=metadata,
        output_dir=output_dir,
    )
    (output_dir / "report.txt").write_text(report, encoding="utf-8")
    make_plots(all_rates, leaderboard, prefix_models, sorted(set(args.plot_thresholds)), output_dir)

    print(f"[ranking_report] predictions: {predictions_path}")
    print(f"[ranking_report] output_dir: {output_dir}")
    print(f"[ranking_report] score_mode: {args.score_mode}")
    print(f"[ranking_report] selected prefix models: {', '.join(prefix_models)}")
    print(
        f"[ranking_report] test rows={len(df)} trajectories={df['traj_id'].nunique()} "
        f"instances={df['instance_id'].nunique()} heldout_models={len(originals)}"
    )
    print("[ranking_report] wrote report.txt, CSV summaries, and PNG plots")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
