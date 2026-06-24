from __future__ import annotations

import math
from dataclasses import asdict
from typing import Any

import numpy as np

from final3.core.contracts import PolicySpec


def head_column(head: str, score_mode: str, predictor: str) -> str:
    """Return the probability column name for a dual-head predictor."""

    if score_mode == "raw":
        return f"prob_safe_{head}__{predictor}"
    if score_mode == "calibrated":
        return f"prob_cal_safe_{head}__{predictor}"
    raise ValueError(f"Unsupported score mode: {score_mode}")


def decide_dual(
    steps: np.ndarray,
    success_scores: np.ndarray,
    failure_scores: np.ndarray,
    policy: PolicySpec,
) -> tuple[bool, str, int, float]:
    """Scan one trajectory prefix sequence and return the first safe-stop decision."""

    last_decision = "undecided"
    streak = 0
    for step_value, success_score, failure_score in zip(steps, success_scores, failure_scores):
        step = int(step_value)
        if step < int(policy.min_step):
            continue
        success_hit = float(success_score) >= float(policy.success_thr)
        failure_hit = float(failure_score) >= float(policy.failure_thr)
        if success_hit and failure_hit:
            # When both heads fire, use the larger threshold margin to choose
            # the decision direction.
            success_margin = float(success_score) - float(policy.success_thr)
            failure_margin = float(failure_score) - float(policy.failure_thr)
            decision = "success" if success_margin >= failure_margin else "failure"
            score = float(success_score if decision == "success" else failure_score)
        elif success_hit:
            decision = "success"
            score = float(success_score)
        elif failure_hit:
            decision = "failure"
            score = float(failure_score)
        else:
            # Any non-hit prefix breaks the consecutive-hit streak.
            last_decision = "undecided"
            streak = 0
            continue

        # The consecutive gate avoids stopping on a single probability spike.
        streak = streak + 1 if decision == last_decision else 1
        last_decision = decision
        if streak >= int(policy.consecutive):
            return True, decision, step, score
    return False, "undecided", -1, float("nan")


def _originals(frame) -> dict[str, dict[str, Any]]:
    """Compute full-run resolved baselines for each agent/model."""

    final_idx = frame.groupby("traj_id", sort=False)["prefix_step_idx"].idxmax()
    final_df = frame.loc[final_idx].copy()
    model_col = "orig_model_id" if "orig_model_id" in final_df.columns else "model_id"
    out: dict[str, dict[str, Any]] = {}
    for agent_model, part in final_df.groupby(model_col, sort=True):
        total = int(len(part))
        resolved = int(part["label"].astype(int).sum())
        out[str(agent_model)] = {
            "total": total,
            "resolved": resolved,
            "resolve_rate": resolved / total if total else 0.0,
        }
    return out


def _empty_counts() -> dict[str, int]:
    """Create counters for decisions, confusion-matrix cells, and saved steps."""

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


def _summarize_counts(counts: dict[str, int], original: dict[str, Any]) -> dict[str, Any]:
    """Convert raw policy counters into reportable rates and totals."""

    tp = int(counts["true_positives"])
    tn = int(counts["true_negatives"])
    fp = int(counts["false_positives"])
    fn = int(counts["false_negatives"])
    decided_success = int(counts["decided_success"])
    decided_failure = int(counts["decided_failure"])
    n_decided = decided_success + decided_failure
    undecided_resolved = int(original["resolved"]) - tp - fn
    adjusted_resolved = tp + fp + undecided_resolved
    total = int(original["total"])
    adjusted_rate = adjusted_resolved / total if total else 0.0
    total_steps = int(counts["total_steps"])
    original_rate = float(original["resolve_rate"])
    return {
        "original_total": total,
        "original_resolved": int(original["resolved"]),
        "original_resolve_rate": original_rate,
        "decided_failure": decided_failure,
        "decided_success": decided_success,
        "undecided": int(counts["undecided"]),
        "false_negatives": fn,
        "true_negatives": tn,
        "false_positives": fp,
        "true_positives": tp,
        "n_decided": n_decided,
        "coverage_pct": 100.0 * n_decided / total if total else float("nan"),
        "decision_accuracy_pct": 100.0 * (tp + tn) / n_decided if n_decided else float("nan"),
        "precision_success_pct": 100.0 * tp / decided_success if decided_success else float("nan"),
        "precision_failure_pct": 100.0 * tn / decided_failure if decided_failure else float("nan"),
        "adjusted_resolved": int(adjusted_resolved),
        "adjusted_resolve_rate": float(adjusted_rate),
        "resolve_rate_drop_pp": 100.0 * (original_rate - adjusted_rate),
        "resolve_rate_change_pp": 100.0 * (adjusted_rate - original_rate),
        "pct_steps_saved": 100.0 * float(counts["total_saved_steps"]) / float(total_steps) if total_steps else float("nan"),
        "total_saved_steps": int(counts["total_saved_steps"]),
        "total_steps": total_steps,
    }


def apply_policy(frame, policy: PolicySpec):
    """Apply a safe-stop policy to a prefix prediction table."""

    import pandas as pd

    success_col = head_column("success", policy.score_mode, policy.predictor)
    failure_col = head_column("failure", policy.score_mode, policy.predictor)
    required = ["traj_id", "label", "prefix_step_idx", success_col, failure_col]
    missing = [col for col in required if col not in frame.columns]
    if missing:
        raise ValueError(f"Prediction table is missing required columns: {missing}")

    model_col = "orig_model_id" if "orig_model_id" in frame.columns else "model_id"
    if model_col not in frame.columns:
        frame = frame.copy()
        frame["model_id"] = "__UNKNOWN__"
        model_col = "model_id"

    decision_rows: list[dict[str, Any]] = []
    per_agent_counts = {agent: _empty_counts() for agent in sorted(frame[model_col].astype(str).unique())}
    for traj_id, group in frame.groupby("traj_id", sort=False):
        # Make each trajectory decision independently after explicit sorting.
        group = group.sort_values("prefix_step_idx")
        agent = str(group[model_col].iloc[0])
        label = int(group["label"].iloc[0])
        steps = group["prefix_step_idx"].to_numpy(dtype=np.int32)
        n_steps = int(len(group))
        decided, decision, decision_step, decision_score = decide_dual(
            steps,
            group[success_col].to_numpy(dtype=np.float64),
            group[failure_col].to_numpy(dtype=np.float64),
            policy,
        )
        counts = per_agent_counts.setdefault(agent, _empty_counts())
        counts["total_steps"] += n_steps
        saved_steps = 0
        if not decided:
            counts["undecided"] += 1
        else:
            # Saved steps are prefixes strictly after the decision step. Counting
            # rows avoids assuming that prefix_step_idx is contiguous.
            saved_steps = int((steps > int(decision_step)).sum())
            counts["total_saved_steps"] += saved_steps
            if decision == "failure":
                counts["decided_failure"] += 1
                if label == 1:
                    counts["false_negatives"] += 1
                else:
                    counts["true_negatives"] += 1
            else:
                counts["decided_success"] += 1
                if label == 0:
                    counts["false_positives"] += 1
                else:
                    counts["true_positives"] += 1
        decision_rows.append(
            {
                "traj_id": traj_id,
                "agent_model": agent,
                "label": label,
                "n_steps": n_steps,
                "decided": bool(decided),
                "decision": decision,
                "decision_step": int(decision_step),
                "decision_score": decision_score,
                "saved_steps": int(saved_steps),
            }
        )

    originals = _originals(frame)
    total_counts = _empty_counts()
    per_agent_rows = []
    for agent, counts in per_agent_counts.items():
        # Smoke tables may lack a full baseline for every agent; use an empty
        # baseline in that case so diagnostics can still run.
        original = originals.get(agent, {"total": 0, "resolved": 0, "resolve_rate": 0.0})
        per_agent_rows.append({"agent_model": agent, **_summarize_counts(counts, original)})
        for key, value in counts.items():
            total_counts[key] += int(value)

    total_original = {
        "total": sum(item["total"] for item in originals.values()),
        "resolved": sum(item["resolved"] for item in originals.values()),
    }
    total_original["resolve_rate"] = (
        total_original["resolved"] / total_original["total"] if total_original["total"] else 0.0
    )
    summary = {
        "policy_name": policy.name,
        **asdict(policy),
        **_summarize_counts(total_counts, total_original),
    }
    if math.isinf(float(summary["failure_thr"])):
        summary["failure_thr"] = "inf"
    if math.isinf(float(summary["success_thr"])):
        summary["success_thr"] = "inf"

    return pd.DataFrame(decision_rows), pd.DataFrame([summary]), pd.DataFrame(per_agent_rows)
