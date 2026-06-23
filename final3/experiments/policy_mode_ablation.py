from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from final3.core.io import ensure_dir, write_json, write_table
from final3.experiments.rq_final import (
    _aggregate_selected_policy,
    _default_output_dir,
    _eligible_lightgbm_folds,
    load_rq_final_config,
)


def _probability(value: str) -> float:
    if str(value).lower() == "inf":
        return float("inf")
    return float(value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Re-evaluate final LightGBM predictions under success-only, failure-only, and dual-head policies."
    )
    parser.add_argument("--config", type=Path, default=Path("configs/rq_final.yaml"))
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--test-models", nargs="*", default=None)
    parser.add_argument("--predictors", nargs="+", default=["I_LightGBM_Dense_AF"])
    parser.add_argument("--score-modes", nargs="+", choices=("raw", "calibrated"), default=["raw", "calibrated"])
    parser.add_argument(
        "--policy-modes",
        nargs="+",
        choices=("success_only", "failure_only", "dual"),
        default=["success_only", "failure_only", "dual"],
    )
    parser.add_argument("--success-thresholds", nargs="+", type=_probability, default=[0.80, 0.90, 0.95])
    parser.add_argument("--failure-thresholds", nargs="+", type=_probability, default=[0.80, 0.90, 0.95])
    parser.add_argument("--policy-min-steps", nargs="+", type=int, default=[0, 5, 10])
    parser.add_argument("--consecutive", nargs="+", type=int, default=[1, 2])
    parser.add_argument("--reference-threshold", type=float, default=0.95)
    parser.add_argument("--reference-score-mode", choices=("raw", "calibrated"), default="calibrated")
    parser.add_argument("--allow-missing", action="store_true")
    return parser.parse_args()


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


def _decide(
    record: dict[str, Any],
    *,
    policy_mode: str,
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
        if policy_mode == "success_only":
            decision = "success" if success_hit else "undecided"
            score = float(success_score)
        elif policy_mode == "failure_only":
            decision = "failure" if failure_hit else "undecided"
            score = float(failure_score)
        elif policy_mode == "dual":
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
                decision = "undecided"
                score = float("nan")
        else:
            raise ValueError(f"Unsupported policy mode: {policy_mode}")

        if decision == "undecided":
            last_decision = "undecided"
            streak = 0
            continue
        streak = streak + 1 if decision == last_decision else 1
        last_decision = decision
        if streak >= consecutive:
            return True, decision, step, score
    return False, "undecided", -1, float("nan")


def _policy_grid(args: argparse.Namespace) -> list[dict[str, Any]]:
    policies: list[dict[str, Any]] = []
    for policy_mode in args.policy_modes:
        if policy_mode == "success_only":
            thresholds = [(success_thr, float("inf")) for success_thr in args.success_thresholds]
        elif policy_mode == "failure_only":
            thresholds = [(float("inf"), failure_thr) for failure_thr in args.failure_thresholds]
        else:
            thresholds = [
                (success_thr, failure_thr)
                for success_thr in args.success_thresholds
                for failure_thr in args.failure_thresholds
            ]
        for success_thr, failure_thr in thresholds:
            for min_step in args.policy_min_steps:
                for consecutive in args.consecutive:
                    policies.append(
                        {
                            "policy_mode": policy_mode,
                            "success_thr": float(success_thr),
                            "failure_thr": float(failure_thr),
                            "min_step": int(min_step),
                            "consecutive": int(consecutive),
                        }
                    )
    return policies


def _evaluate_frame(
    df: pd.DataFrame,
    *,
    run_label: str,
    fold_id: str,
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
                    decided, decision, decision_step, _ = _decide(record, **policy)
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
                    "fold_id": fold_id,
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


def _fmt_thr(value: float) -> str:
    if math.isinf(float(value)):
        return "inf"
    return f"{float(value):.2f}"


def _aggregate_grid(per_fold: pd.DataFrame) -> pd.DataFrame:
    if per_fold.empty:
        return pd.DataFrame()
    keys = ["policy_mode", "prefix_model", "score_mode", "success_thr", "failure_thr", "min_step", "consecutive"]
    rows: list[dict[str, Any]] = []
    for values, part in per_fold.groupby(keys, sort=True):
        row = dict(zip(keys, values))
        row.update(_aggregate_selected_policy(part))
        rows.append(row)
    out = pd.DataFrame(rows)
    if not out.empty:
        out["success_thr_label"] = out["success_thr"].map(_fmt_thr)
        out["failure_thr_label"] = out["failure_thr"].map(_fmt_thr)
    return out


def _reference_rows(grid: pd.DataFrame, args: argparse.Namespace) -> dict[str, pd.DataFrame]:
    if grid.empty:
        empty = pd.DataFrame()
        return {
            "policy_mode": empty,
            "score_mode": empty,
            "min_step_k": empty,
        }
    thr = float(args.reference_threshold)
    score_mode = str(args.reference_score_mode)
    base = grid[grid["prefix_model"].eq("I_LightGBM_Dense_AF")].copy()
    policy_mode_ref = base[
        base["score_mode"].eq(score_mode)
        & base["min_step"].eq(0)
        & base["consecutive"].eq(1)
        & (
            (base["policy_mode"].eq("success_only") & base["success_thr"].eq(thr) & np.isinf(base["failure_thr"]))
            | (base["policy_mode"].eq("failure_only") & np.isinf(base["success_thr"]) & base["failure_thr"].eq(thr))
            | (base["policy_mode"].eq("dual") & base["success_thr"].eq(thr) & base["failure_thr"].eq(thr))
        )
    ].copy()
    score_mode_ref = base[
        base["policy_mode"].eq("dual")
        & base["success_thr"].eq(thr)
        & base["failure_thr"].eq(thr)
        & base["min_step"].eq(0)
        & base["consecutive"].eq(1)
    ].copy()
    min_step_k_ref = base[
        base["policy_mode"].eq("dual")
        & base["score_mode"].eq(score_mode)
        & base["success_thr"].eq(thr)
        & base["failure_thr"].eq(thr)
    ].copy()
    return {
        "policy_mode": policy_mode_ref,
        "score_mode": score_mode_ref,
        "min_step_k": min_step_k_ref,
    }


def run_policy_mode_ablation(args: argparse.Namespace) -> dict[str, Any]:
    cfg = load_rq_final_config(args.config)
    experiment_root = _default_output_dir(cfg, "rq_final_lightgbm_17")
    run_dir = args.run_dir or (experiment_root / "lightgbm_main")
    output_dir = ensure_dir(args.output_dir or (experiment_root / "policy_ablation" / "sweverify_policy_mode_full16"))
    folds = [row for row in _eligible_lightgbm_folds(cfg, dataset="sweverify") if row["eligible"]]
    if args.test_models:
        wanted = set(str(item) for item in args.test_models)
        folds = [row for row in folds if str(row["test_model"]) in wanted or str(row["fold_id"]) in wanted]

    policies = _policy_grid(args)
    fold_rows: list[pd.DataFrame] = []
    agent_rows: list[pd.DataFrame] = []
    missing_columns = ["fold_id", "test_model", "path", "reason"]
    missing: list[dict[str, str]] = []
    for fold in folds:
        fold_id = str(fold["fold_id"])
        pred_path = run_dir / "folds" / fold_id / "test_predictions_safe_stop.parquet"
        if not pred_path.exists():
            missing.append({"fold_id": fold_id, "test_model": str(fold["test_model"]), "path": str(pred_path)})
            continue
        df = pd.read_parquet(pred_path)
        aggregate, per_agent = _evaluate_frame(
            df,
            run_label=run_dir.name,
            fold_id=fold_id,
            predictors=[str(item) for item in args.predictors],
            score_modes=[str(item) for item in args.score_modes],
            policies=policies,
        )
        if aggregate.empty:
            missing.append(
                {
                    "fold_id": fold_id,
                    "test_model": str(fold["test_model"]),
                    "path": str(pred_path),
                    "reason": "missing predictor probability columns",
                }
            )
            continue
        aggregate.insert(1, "test_model", str(fold["test_model"]))
        per_agent.insert(1, "test_model", str(fold["test_model"]))
        fold_rows.append(aggregate)
        agent_rows.append(per_agent)

    if missing and not args.allow_missing:
        write_table(pd.DataFrame(missing, columns=missing_columns), output_dir / "missing_predictions.csv")
        raise FileNotFoundError(
            f"Missing or unusable predictions for {len(missing)} fold(s); "
            f"see {output_dir / 'missing_predictions.csv'}"
        )

    per_fold = pd.concat(fold_rows, ignore_index=True) if fold_rows else pd.DataFrame()
    per_agent = pd.concat(agent_rows, ignore_index=True) if agent_rows else pd.DataFrame()
    aggregate = _aggregate_grid(per_fold)
    refs = _reference_rows(aggregate, args)

    write_table(per_fold, output_dir / "policy_grid_per_fold.csv")
    write_table(per_agent, output_dir / "policy_grid_per_agent.csv")
    write_table(aggregate, output_dir / "policy_grid_aggregate.csv")
    write_table(refs["policy_mode"], output_dir / "policy_mode_main_points.csv")
    write_table(refs["score_mode"], output_dir / "score_mode_main_points.csv")
    write_table(refs["min_step_k"], output_dir / "min_step_k_main_points.csv")
    write_table(pd.DataFrame(missing, columns=missing_columns), output_dir / "missing_predictions.csv")

    lines = [
        "# Policy Mode Ablation",
        "",
        f"- source: `{run_dir}`",
        f"- output: `{output_dir}`",
        f"- completed folds: `{per_fold['fold_id'].nunique() if not per_fold.empty else 0}`",
        f"- missing folds: `{len(missing)}`",
        "",
        "## Tables",
        "",
        "- `policy_grid_aggregate.csv`: full success-only / failure-only / dual grid.",
        "- `policy_mode_main_points.csv`: calibrated 0.95, min_step=0, k=1 comparison.",
        "- `score_mode_main_points.csv`: raw vs calibrated at dual 0.95, min_step=0, k=1.",
        "- `min_step_k_main_points.csv`: min_step and consecutive-k sensitivity at calibrated dual 0.95.",
    ]
    (output_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")
    payload = {
        "ok": True,
        "run_dir": str(run_dir),
        "output_dir": str(output_dir),
        "folds_requested": len(folds),
        "folds_completed": int(per_fold["fold_id"].nunique()) if not per_fold.empty else 0,
        "missing": missing,
        "predictors": [str(item) for item in args.predictors],
        "score_modes": [str(item) for item in args.score_modes],
        "policy_modes": [str(item) for item in args.policy_modes],
    }
    write_json(output_dir / "policy_mode_ablation_manifest.json", payload)
    return payload


def main() -> int:
    args = parse_args()
    payload = run_policy_mode_ablation(args)
    print(payload["output_dir"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
