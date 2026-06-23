#!/usr/bin/env python3
"""Post-hoc process-signal rescue policies.

This script does not retrain.  It tests policies intended to reduce reliance on
static/agent prior:

* asymmetric success/failure thresholds
* failure-only and success-only policies
* optional probability movement gates relative to the first prefix
* delayed start and consecutive-hit requirements
"""

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

DEFAULT_PREFIX_MODELS = [
    "I_LightGBM_Dense_AF",
    "J_LightGBM_Dense_AF_Thought",
    "Abl_NoTaskSignal_LightGBM",
    "Abl_NoTaskSignal_NoGoldAnswer_LightGBM",
]

RUN_LABELS = {
    "per_instance_model_valid3_retrain": "mid3",
    "per_instance_model_valid3_top3_retrain": "top3",
    "per_instance_model_valid3_bottom3_retrain": "bottom3",
}

SCORE_LABELS = {
    "raw": "raw",
    "prefix_calibrated": "prefix-cal",
    "trajectory_calibrated": "traj-cal",
}

MODEL_LABELS = {
    "I_LightGBM_Dense_AF": "I",
    "J_LightGBM_Dense_AF_Thought": "J",
    "Abl_NoTaskSignal_LightGBM": "NoTask",
    "Abl_NoTaskSignal_NoGoldAnswer_LightGBM": "NoTask+NoGold",
}


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
    parser.add_argument("--prefix-models", nargs="+", default=DEFAULT_PREFIX_MODELS)
    parser.add_argument("--min-steps", nargs="+", type=int, default=[5, 10, 15])
    parser.add_argument("--consecutive", nargs="+", type=int, default=[2, 3])
    parser.add_argument("--delta-thresholds", nargs="+", type=float, default=[0.0, 0.05, 0.10])
    parser.add_argument("--output-name", default="process_signal_policy_rescue")
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


def _records(df: pd.DataFrame, prob_cols: list[str]) -> list[dict[str, Any]]:
    needed = ["traj_id", "orig_model_id", "label", "prefix_step_idx", *prob_cols]
    work = df[needed].copy()
    records: list[dict[str, Any]] = []
    for _, group in work.groupby("traj_id", sort=False):
        group = group.sort_values("prefix_step_idx")
        probs = {
            prob_col: group[prob_col].to_numpy(dtype=np.float64)
            for prob_col in prob_cols
        }
        records.append(
            {
                "agent_model": str(group["orig_model_id"].iloc[0]),
                "label": int(group["label"].iloc[0]),
                "n_steps": int(len(group)),
                "steps": group["prefix_step_idx"].to_numpy(dtype=np.int32),
                "probs": probs,
                "p0": {prob_col: float(values[0]) for prob_col, values in probs.items()},
            }
        )
    return records


def _policy_grid(
    *,
    min_steps: list[int],
    consecutive_values: list[int],
    delta_thresholds: list[float],
) -> list[dict[str, Any]]:
    base_specs: list[dict[str, Any]] = []
    for failure_thr in [0.05, 0.10, 0.15, 0.20, 0.25]:
        base_specs.append(
            {
                "policy_mode": "failure_only",
                "success_thr": np.inf,
                "failure_thr": failure_thr,
            }
        )
    for success_thr in [0.85, 0.90, 0.95, 0.97]:
        base_specs.append(
            {
                "policy_mode": "success_only",
                "success_thr": success_thr,
                "failure_thr": -np.inf,
            }
        )
    for success_thr in [0.90, 0.95, 0.97]:
        for failure_thr in [0.10, 0.15, 0.20]:
            base_specs.append(
                {
                    "policy_mode": "asymmetric",
                    "success_thr": success_thr,
                    "failure_thr": failure_thr,
                }
            )

    policies: list[dict[str, Any]] = []
    for spec in base_specs:
        for min_step in min_steps:
            for consecutive in consecutive_values:
                for delta in delta_thresholds:
                    if spec["policy_mode"] == "success_only":
                        delta_down = 0.0
                        delta_up = delta
                    elif spec["policy_mode"] == "failure_only":
                        delta_down = delta
                        delta_up = 0.0
                    else:
                        delta_down = delta
                        delta_up = delta
                    policies.append(
                        {
                            **spec,
                            "min_step": int(min_step),
                            "consecutive": int(consecutive),
                            "delta_up": float(delta_up),
                            "delta_down": float(delta_down),
                        }
                    )
    return policies


def _decide(
    steps: np.ndarray,
    probs: np.ndarray,
    *,
    p0: float,
    success_thr: float,
    failure_thr: float,
    min_step: int,
    consecutive: int,
    delta_up: float,
    delta_down: float,
) -> tuple[bool, str, int, float]:
    success_streak = 0
    failure_streak = 0
    for step_value, prob_value in zip(steps, probs):
        step = int(step_value)
        if step < min_step:
            continue
        prob = float(prob_value)
        success_hit = prob >= success_thr and (prob - p0) >= delta_up
        failure_hit = prob <= failure_thr and (p0 - prob) >= delta_down
        if success_hit:
            success_streak += 1
            failure_streak = 0
            if success_streak >= consecutive:
                return True, "success", step, prob
        elif failure_hit:
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
        "pct_steps_saved": _safe_div(float(counts["total_saved_steps"]) * 100.0, float(counts["total_steps"])),
        "total_saved_steps": int(counts["total_saved_steps"]),
        "total_steps": int(counts["total_steps"]),
    }


def evaluate(
    df: pd.DataFrame,
    *,
    run_label: str,
    score_mode: str,
    prefix_models: list[str],
    policies: list[dict[str, Any]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    prob_cols = [
        _prob_col(prefix_model, score_mode)
        for prefix_model in prefix_models
        if _prob_col(prefix_model, score_mode) in df.columns
    ]
    records = _records(df, prob_cols)
    originals = _originals(df)
    aggregate_rows: list[dict[str, Any]] = []
    per_agent_rows: list[dict[str, Any]] = []

    for prefix_model in prefix_models:
        prob_col = _prob_col(prefix_model, score_mode)
        if prob_col not in df.columns:
            continue
        for policy in policies:
            per_agent = {agent_model: _empty_counts() for agent_model in originals}
            for record in records:
                agent_model = record["agent_model"]
                if agent_model not in per_agent:
                    continue
                label = int(record["label"])
                n_steps = int(record["n_steps"])
                per_agent[agent_model]["total_steps"] += n_steps
                decided, decision, decision_step, _ = _decide(
                    record["steps"],
                    record["probs"][prob_col],
                    p0=record["p0"][prob_col],
                    success_thr=float(policy["success_thr"]),
                    failure_thr=float(policy["failure_thr"]),
                    min_step=int(policy["min_step"]),
                    consecutive=int(policy["consecutive"]),
                    delta_up=float(policy["delta_up"]),
                    delta_down=float(policy["delta_down"]),
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
                "prefix_model": prefix_model,
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


def _prediction_path(report_dir: Path, score_mode: str) -> Path:
    if score_mode == "trajectory_calibrated":
        return report_dir / "trajectory_calibrated_posthoc" / "test_predictions_trajectory_calibrated.parquet"
    return report_dir / "test_predictions_shadow_valid_retrain.parquet"


def _shorten(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["run_short"] = out["run"].map(RUN_LABELS).fillna(out["run"])
    out["score_short"] = out["score_mode"].map(SCORE_LABELS).fillna(out["score_mode"])
    out["model_short"] = out["prefix_model"].map(MODEL_LABELS).fillna(out["prefix_model"])
    out["drop_pp"] = out["resolve_rate_drop"] * 100.0
    out["coverage_pct"] = out["coverage"] * 100.0
    out["decision_acc_pct"] = out["decision_accuracy"] * 100.0
    out["policy_label"] = (
        out["policy_mode"].astype(str)
        + "_s"
        + out["success_thr"].replace(np.inf, np.nan).round(2).astype(str)
        + "_f"
        + out["failure_thr"].replace(-np.inf, np.nan).round(2).astype(str)
        + "_m"
        + out["min_step"].astype(str)
        + "_k"
        + out["consecutive"].astype(str)
        + "_d"
        + out[["delta_up", "delta_down"]].max(axis=1).round(2).astype(str)
    )
    return out


def _fmt(value: float, digits: int = 1) -> str:
    if value is None or math.isnan(float(value)):
        return "-"
    return f"{float(value):.{digits}f}"


def write_report(output_dir: Path, aggregate: pd.DataFrame) -> None:
    lines: list[str] = []
    lines.append("# Process-Signal Rescue Policy Report")
    lines.append("")
    lines.append("这些都是 post-hoc 策略实验，没有重训。")
    lines.append("")
    lines.append("## Best Policies with Save >= 15% and |Drop| <= 2pp")
    candidate = aggregate[
        (aggregate["pct_steps_saved"] >= 15.0)
        & (aggregate["drop_pp"].abs() <= 2.0)
        & (aggregate["model_short"].isin(["I", "J"]))
    ].copy()
    if candidate.empty:
        lines.append("No policy met the filter.")
    else:
        candidate = candidate.sort_values(
            ["run_short", "model_short", "score_short", "pct_steps_saved"],
            ascending=[True, True, True, False],
        )
        candidate = candidate.groupby(["run_short", "model_short", "score_short"], as_index=False).head(5)
        lines.append("| Run | Model | Score | Mode | S_thr | F_thr | Min | K | Delta | Coverage | Acc | Save% | Drop pp | FN | FP |")
        lines.append("|:--|:--|:--|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|")
        for _, row in candidate.iterrows():
            lines.append(
                "| "
                f"{row['run_short']} | {row['model_short']} | {row['score_short']} | "
                f"{row['policy_mode']} | {_fmt(row['success_thr'], 2)} | {_fmt(row['failure_thr'], 2)} | "
                f"{int(row['min_step'])} | {int(row['consecutive'])} | "
                f"{_fmt(max(row['delta_up'], row['delta_down']), 2)} | "
                f"{_fmt(row['coverage_pct'])}% | {_fmt(row['decision_acc_pct'])}% | "
                f"{_fmt(row['pct_steps_saved'])}% | {_fmt(row['drop_pp'])} | "
                f"{int(row['false_negatives'])} | {int(row['false_positives'])} |"
            )
    lines.append("")

    lines.append("## Failure-Only Policies")
    failure = aggregate[
        (aggregate["policy_mode"] == "failure_only")
        & (aggregate["model_short"].isin(["I", "J"]))
    ].copy()
    failure = failure.sort_values(
        ["run_short", "model_short", "score_short", "failure_thr", "pct_steps_saved"],
        ascending=[True, True, True, True, False],
    )
    failure = failure.groupby(["run_short", "model_short", "score_short", "failure_thr"], as_index=False).head(1)
    lines.append("| Run | Model | Score | F_thr | Min | K | Delta | Coverage | Acc | Save% | Drop pp | FN | FP |")
    lines.append("|:--|:--|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|")
    for _, row in failure.iterrows():
        lines.append(
            "| "
            f"{row['run_short']} | {row['model_short']} | {row['score_short']} | "
            f"{_fmt(row['failure_thr'], 2)} | {int(row['min_step'])} | {int(row['consecutive'])} | "
            f"{_fmt(row['delta_down'], 2)} | {_fmt(row['coverage_pct'])}% | "
            f"{_fmt(row['decision_acc_pct'])}% | {_fmt(row['pct_steps_saved'])}% | "
            f"{_fmt(row['drop_pp'])} | {int(row['false_negatives'])} | {int(row['false_positives'])} |"
        )
    lines.append("")

    lines.append("## Output Files")
    lines.append("- `process_rescue_aggregate.csv`")
    lines.append("- `process_rescue_per_agent.csv`")
    lines.append("- `process_rescue_best_save_ge15_absdrop_le2.csv`")
    lines.append("- `process_rescue_failure_only_best.csv`")
    (output_dir / "process_rescue_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    run_root = PROJECT_ROOT / "runs" / args.run_name
    _set_run_dirs(run_root)
    output_dir = config.REPORT_DIR / args.output_name
    output_dir.mkdir(parents=True, exist_ok=True)
    policies = _policy_grid(
        min_steps=args.min_steps,
        consecutive_values=args.consecutive,
        delta_thresholds=args.delta_thresholds,
    )

    all_aggregate: list[pd.DataFrame] = []
    all_per_agent: list[pd.DataFrame] = []
    for report_subdir in args.report_subdirs:
        report_dir = config.REPORT_DIR / report_subdir
        for score_mode in args.score_modes:
            pred_path = _prediction_path(report_dir, score_mode)
            if not pred_path.is_file():
                print(f"[skip] missing predictions: {pred_path}", file=sys.stderr)
                continue
            df = pd.read_parquet(pred_path)
            aggregate, per_agent = evaluate(
                df,
                run_label=report_subdir,
                score_mode=score_mode,
                prefix_models=args.prefix_models,
                policies=policies,
            )
            all_aggregate.append(aggregate)
            all_per_agent.append(per_agent)

    aggregate_df = _shorten(pd.concat(all_aggregate, ignore_index=True)) if all_aggregate else pd.DataFrame()
    per_agent_df = _shorten(pd.concat(all_per_agent, ignore_index=True)) if all_per_agent else pd.DataFrame()
    aggregate_df.to_csv(output_dir / "process_rescue_aggregate.csv", index=False)
    per_agent_df.to_csv(output_dir / "process_rescue_per_agent.csv", index=False)
    if not aggregate_df.empty:
        best = aggregate_df[
            (aggregate_df["pct_steps_saved"] >= 15.0)
            & (aggregate_df["drop_pp"].abs() <= 2.0)
        ].sort_values(["run_short", "model_short", "score_short", "pct_steps_saved"], ascending=[True, True, True, False])
        best.to_csv(output_dir / "process_rescue_best_save_ge15_absdrop_le2.csv", index=False)
        failure = aggregate_df[aggregate_df["policy_mode"] == "failure_only"].sort_values(
            ["run_short", "model_short", "score_short", "failure_thr", "pct_steps_saved"],
            ascending=[True, True, True, True, False],
        )
        failure.groupby(["run_short", "model_short", "score_short", "failure_thr"], as_index=False).head(1).to_csv(
            output_dir / "process_rescue_failure_only_best.csv",
            index=False,
        )
        write_report(output_dir, aggregate_df)
    print(f"Saved process rescue results: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
