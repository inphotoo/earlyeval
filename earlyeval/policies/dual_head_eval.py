from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd


def _head_column(head: str, score_mode: str, predictor: str) -> str:
    if score_mode == "raw":
        return f"prob_safe_{head}__{predictor}"
    if score_mode == "calibrated":
        return f"prob_cal_safe_{head}__{predictor}"
    raise ValueError(f"Unsupported score mode: {score_mode}")


def _originals(df: pd.DataFrame) -> dict[str, dict[str, Any]]:
    model_col = "orig_model_id" if "orig_model_id" in df.columns else "model_id"
    final_idx = df.groupby("traj_id", sort=False)["prefix_step_idx"].idxmax()
    final_df = df.loc[final_idx]
    out: dict[str, dict[str, Any]] = {}
    for agent_model, part in final_df.groupby(model_col, sort=True):
        total = int(len(part))
        resolved = int(part["label"].sum())
        out[str(agent_model)] = {
            "total": total,
            "resolved": resolved,
            "resolve_rate": resolved / total if total else 0.0,
        }
    return out


def _records(df: pd.DataFrame, success_col: str, failure_col: str) -> list[dict[str, Any]]:
    model_col = "orig_model_id" if "orig_model_id" in df.columns else "model_id"
    needed = ["traj_id", model_col, "label", "prefix_step_idx", success_col, failure_col]
    work = df[needed].copy()
    records: list[dict[str, Any]] = []
    for _, group in work.groupby("traj_id", sort=False):
        group = group.sort_values("prefix_step_idx")
        records.append(
            {
                "agent_model": str(group[model_col].iloc[0]),
                "label": int(group["label"].iloc[0]),
                "n_steps": int(len(group)),
                "steps": group["prefix_step_idx"].to_numpy(dtype=np.int32),
                "success": group[success_col].to_numpy(dtype=np.float64),
                "failure": group[failure_col].to_numpy(dtype=np.float64),
            }
        )
    return records


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
    adjusted_resolved = tp + fp + undecided_resolved
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
            float(counts["total_saved_steps"]) * 100.0 / float(total_steps) if total_steps else float("nan")
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
                    saved = int((record["steps"] > int(decision_step)).sum())
                    per_agent[agent_model]["total_saved_steps"] += saved
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
                    total_original["resolved"] / total_original["total"] if total_original["total"] else 0.0
                )
                aggregate_rows.append({**policy_meta, **_summarize(total_counts, total_original)})
    return pd.DataFrame(aggregate_rows), pd.DataFrame(per_agent_rows)
