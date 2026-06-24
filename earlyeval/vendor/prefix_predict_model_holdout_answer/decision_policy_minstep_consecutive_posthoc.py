#!/usr/bin/env python3
"""Post-hoc early decision policy sweep with min-step and consecutive-hit gates."""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".mplconfig"))
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

import config
from gold_text_tfidf_ablation_posthoc import _set_run_dirs


DEFAULT_REPORT_SUBDIRS = [
    "per_instance_model_valid3_retrain",
    "per_instance_model_valid3_top3_retrain",
    "per_instance_model_valid3_bottom3_retrain",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", default="model_holdout_answer_calibrated_full")
    parser.add_argument("--report-subdirs", nargs="+", default=DEFAULT_REPORT_SUBDIRS)
    parser.add_argument(
        "--score-modes",
        nargs="+",
        default=["raw", "prefix_calibrated", "trajectory_calibrated"],
        choices=["raw", "prefix_calibrated", "trajectory_calibrated"],
    )
    parser.add_argument("--thresholds", nargs="+", type=float, default=[0.70, 0.80, 0.85, 0.90, 0.95])
    parser.add_argument("--min-steps", nargs="+", type=int, default=[0, 1, 3, 5, 10, 15])
    parser.add_argument("--consecutive", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--prefix-models", nargs="+", default=None)
    parser.add_argument(
        "--output-name",
        default="decision_policy_minstep_consecutive",
        help="Output directory name under reports/<first report subdir>/../",
    )
    return parser.parse_args()


def _prediction_path(report_dir: Path, score_mode: str) -> Path:
    if score_mode == "trajectory_calibrated":
        return report_dir / "trajectory_calibrated_posthoc" / "test_predictions_trajectory_calibrated.parquet"
    return report_dir / "test_predictions_shadow_valid_retrain.parquet"


def _prob_col(prefix_model: str, score_mode: str) -> str:
    if score_mode == "raw":
        return f"prob__{prefix_model}"
    if score_mode == "prefix_calibrated":
        return f"prob_cal__{prefix_model}"
    if score_mode == "trajectory_calibrated":
        return f"prob_traj_cal__{prefix_model}"
    raise ValueError(score_mode)


def _available_prefix_models(df: pd.DataFrame, score_mode: str) -> list[str]:
    if score_mode == "raw":
        prefix = "prob__"
    elif score_mode == "prefix_calibrated":
        prefix = "prob_cal__"
    else:
        prefix = "prob_traj_cal__"
    return sorted(col.removeprefix(prefix) for col in df.columns if col.startswith(prefix))


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


def _trajectory_records(df: pd.DataFrame, prob_cols: list[str]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    needed = ["traj_id", "orig_model_id", "label", "prefix_step_idx", *prob_cols]
    work = df[needed].copy()
    for _, group in work.groupby("traj_id", sort=False):
        group = group.sort_values("prefix_step_idx")
        records.append(
            {
                "agent_model": str(group["orig_model_id"].iloc[0]),
                "label": int(group["label"].iloc[0]),
                "n_steps": int(len(group)),
                "steps": group["prefix_step_idx"].to_numpy(dtype=np.int32),
                "probs": {
                    prob_col: group[prob_col].to_numpy(dtype=np.float64)
                    for prob_col in prob_cols
                },
            }
        )
    return records


def _decide_arrays(
    steps: np.ndarray,
    probs: np.ndarray,
    *,
    threshold: float,
    min_step: int,
    consecutive: int,
) -> tuple[bool, str, int, float]:
    low = 1.0 - threshold
    success_streak = 0
    failure_streak = 0
    for step_value, prob_value in zip(steps, probs):
        step = int(step_value)
        if step < min_step:
            continue
        prob = float(prob_value)
        if prob >= threshold:
            success_streak += 1
            failure_streak = 0
            if success_streak >= consecutive:
                return True, "success", step, prob
        elif prob <= low:
            failure_streak += 1
            success_streak = 0
            if failure_streak >= consecutive:
                return True, "failure", step, prob
        else:
            success_streak = 0
            failure_streak = 0
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


def _safe_div(num: float, den: float) -> float:
    return num / den if den else float("nan")


def _summarize_counts(counts: dict[str, int], original: dict[str, Any]) -> dict[str, Any]:
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
    decision_accuracy = (tp + tn) / n_decided if n_decided else float("nan")
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
        "decision_accuracy": decision_accuracy,
        "precision_success": tp / decided_success if decided_success else float("nan"),
        "precision_failure": tn / decided_failure if decided_failure else float("nan"),
        "adjusted_resolved": int(adjusted_resolved),
        "adjusted_resolve_rate": float(adjusted_rate),
        "resolve_rate_drop": float(original["resolve_rate"] - adjusted_rate),
        "pct_steps_saved": _safe_div(float(counts["total_saved_steps"]) * 100.0, float(counts["total_steps"])),
        "total_saved_steps": int(counts["total_saved_steps"]),
        "total_steps": int(counts["total_steps"]),
    }


def evaluate_policy_grid(
    df: pd.DataFrame,
    *,
    run_label: str,
    score_mode: str,
    prefix_models: list[str],
    thresholds: list[float],
    min_steps: list[int],
    consecutive_values: list[int],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    originals = _originals(df)
    prob_cols = [
        _prob_col(prefix_model, score_mode)
        for prefix_model in prefix_models
        if _prob_col(prefix_model, score_mode) in df.columns
    ]
    records = _trajectory_records(df, prob_cols)
    aggregate_rows: list[dict[str, Any]] = []
    per_agent_rows: list[dict[str, Any]] = []

    for prefix_model in prefix_models:
        prob_col = _prob_col(prefix_model, score_mode)
        if prob_col not in df.columns:
            continue
        for threshold in thresholds:
            threshold = round(float(threshold), 4)
            for min_step in min_steps:
                for consecutive in consecutive_values:
                    per_agent = {agent_model: _empty_counts() for agent_model in originals}
                    for record in records:
                        agent_model = record["agent_model"]
                        if agent_model not in per_agent:
                            continue
                        label = int(record["label"])
                        n_steps = int(record["n_steps"])
                        per_agent[agent_model]["total_steps"] += n_steps
                        decided, decision, decision_step, _ = _decide_arrays(
                            record["steps"],
                            record["probs"][prob_col],
                            threshold=threshold,
                            min_step=int(min_step),
                            consecutive=int(consecutive),
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

                    for agent_model, counts in per_agent.items():
                        summary = _summarize_counts(counts, originals[agent_model])
                        per_agent_rows.append(
                            {
                                "run": run_label,
                                "score_mode": score_mode,
                                "prefix_model": prefix_model,
                                "threshold": threshold,
                                "min_step": int(min_step),
                                "consecutive": int(consecutive),
                                "agent_model": agent_model,
                                **summary,
                            }
                        )

                    total_counts = _empty_counts()
                    total_original = {"total": 0, "resolved": 0, "resolve_rate": 0.0}
                    for agent_model, counts in per_agent.items():
                        for key, value in counts.items():
                            total_counts[key] += int(value)
                        total_original["total"] += int(originals[agent_model]["total"])
                        total_original["resolved"] += int(originals[agent_model]["resolved"])
                    total_original["resolve_rate"] = (
                        total_original["resolved"] / total_original["total"]
                        if total_original["total"]
                        else 0.0
                    )
                    summary = _summarize_counts(total_counts, total_original)
                    aggregate_rows.append(
                        {
                            "run": run_label,
                            "score_mode": score_mode,
                            "prefix_model": prefix_model,
                            "threshold": threshold,
                            "min_step": int(min_step),
                            "consecutive": int(consecutive),
                            **summary,
                        }
                    )

    return pd.DataFrame(aggregate_rows), pd.DataFrame(per_agent_rows)


def _fmt_pct(value: float) -> str:
    if value is None or math.isnan(value):
        return "-"
    return f"{value * 100:.1f}%"


def write_report(output_dir: Path, aggregate: pd.DataFrame) -> None:
    lines: list[str] = []
    lines.append("# Min-Step and Consecutive Early-Decision Sweep")
    lines.append("")
    lines.append("Policy: skip prefixes with `prefix_step_idx < min_step`; then require `consecutive` same-side hits.")
    lines.append("Success hit: `p >= threshold`; failure hit: `p <= 1-threshold`.")
    lines.append("")
    focus = aggregate[
        aggregate["prefix_model"].isin(
            [
                "I_LightGBM_Dense_AF",
                "J_LightGBM_Dense_AF_Thought",
                "Abl_NoTaskSignal_LightGBM",
                "Abl_NoTaskPromptTfidf_LightGBM",
            ]
        )
    ].copy()
    if focus.empty:
        focus = aggregate.copy()

    for run in sorted(focus["run"].unique()):
        lines.append(f"## {run}")
        for score_mode in sorted(focus["score_mode"].unique()):
            part = focus[(focus["run"] == run) & (focus["score_mode"] == score_mode)].copy()
            if part.empty:
                continue
            lines.append(f"### {score_mode}")
            chosen = part[
                (part["threshold"].isin([0.8, 0.85, 0.9]))
                & (part["min_step"].isin([0, 3, 5, 10]))
                & (part["consecutive"].isin([1, 2, 3]))
            ].copy()
            chosen["abs_drop"] = chosen["resolve_rate_drop"].abs()
            chosen = chosen.sort_values(
                ["prefix_model", "threshold", "abs_drop", "consecutive", "min_step"],
                ascending=[True, True, True, True, True],
            )
            for prefix_model in chosen["prefix_model"].unique():
                sub = chosen[chosen["prefix_model"] == prefix_model].head(8)
                lines.append(f"#### {prefix_model}")
                lines.append("| Thr | MinStep | Consecutive | Coverage | Acc | Save% | Orig | Adj | Drop | FN | FP |")
                lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
                for _, row in sub.iterrows():
                    lines.append(
                        "| "
                        f"{row['threshold']:.2f} | {int(row['min_step'])} | {int(row['consecutive'])} | "
                        f"{_fmt_pct(row['coverage'])} | {_fmt_pct(row['decision_accuracy'])} | "
                        f"{row['pct_steps_saved']:.1f}% | {_fmt_pct(row['original_resolve_rate'])} | "
                        f"{_fmt_pct(row['adjusted_resolve_rate'])} | {_fmt_pct(row['resolve_rate_drop'])} | "
                        f"{int(row['false_negatives'])} | {int(row['false_positives'])} |"
                    )
                lines.append("")
    lines.append("## Output Files")
    lines.append("- `policy_sweep_aggregate.csv`")
    lines.append("- `policy_sweep_per_agent.csv`")
    (output_dir / "policy_sweep_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    run_root = PROJECT_ROOT / "runs" / args.run_name
    _set_run_dirs(run_root)
    report_root = config.REPORT_DIR
    output_dir = report_root / args.output_name
    output_dir.mkdir(parents=True, exist_ok=True)

    all_aggregate: list[pd.DataFrame] = []
    all_per_agent: list[pd.DataFrame] = []
    for report_subdir in args.report_subdirs:
        report_dir = report_root / report_subdir
        for score_mode in args.score_modes:
            pred_path = _prediction_path(report_dir, score_mode)
            if not pred_path.is_file():
                print(f"[skip] missing predictions: {pred_path}", file=sys.stderr)
                continue
            df = pd.read_parquet(pred_path)
            available = _available_prefix_models(df, score_mode)
            prefix_models = args.prefix_models or available
            missing = [model for model in prefix_models if model not in available]
            if missing:
                print(f"[warn] {report_subdir}/{score_mode}: missing models {missing}", file=sys.stderr)
            prefix_models = [model for model in prefix_models if model in available]
            aggregate, per_agent = evaluate_policy_grid(
                df,
                run_label=report_subdir,
                score_mode=score_mode,
                prefix_models=prefix_models,
                thresholds=args.thresholds,
                min_steps=args.min_steps,
                consecutive_values=args.consecutive,
            )
            all_aggregate.append(aggregate)
            all_per_agent.append(per_agent)

    aggregate_df = pd.concat(all_aggregate, ignore_index=True) if all_aggregate else pd.DataFrame()
    per_agent_df = pd.concat(all_per_agent, ignore_index=True) if all_per_agent else pd.DataFrame()
    aggregate_df.to_csv(output_dir / "policy_sweep_aggregate.csv", index=False)
    per_agent_df.to_csv(output_dir / "policy_sweep_per_agent.csv", index=False)
    if not aggregate_df.empty:
        write_report(output_dir, aggregate_df)
    print(f"Saved policy sweep: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
