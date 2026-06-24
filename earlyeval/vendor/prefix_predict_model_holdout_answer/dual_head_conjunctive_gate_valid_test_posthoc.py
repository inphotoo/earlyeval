#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import re
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
REPORTS_DIR = PROJECT_ROOT / "runs" / "model_holdout_answer_calibrated_full" / "reports"
BASE_VISUAL_DIR = REPORTS_DIR / "safe_stop_dual_head_visual_summary"
OUT_DIR = BASE_VISUAL_DIR / "problem_diagnosis" / "dual_head_conjunctive_gate_posthoc"

RUN_RE = re.compile(r"per_instance_model_valid3_(top3|mid3|bottom3)(?:_.+?)?_safe_stop_dual_head_retrain$")
SCORE_MODES = ["raw", "calibrated"]
THRESHOLDS = [0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
MIN_STEPS = [0, 5, 10, 15, 20]
CONSECUTIVE = [1, 2, 3]


def auc_rank(y, score) -> float:
    data = pd.DataFrame({"y": y, "score": score}).dropna()
    if data.empty:
        return float("nan")
    y_arr = data["y"].astype(int).to_numpy()
    n_pos = int((y_arr == 1).sum())
    n_neg = int((y_arr == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = data["score"].rank(method="average").to_numpy()
    sum_pos = float(ranks[y_arr == 1].sum())
    return (sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def fmt(x: float) -> str:
    return "-" if pd.isna(x) else f"{x:.3f}"


def pct(x: float) -> str:
    return "-" if pd.isna(x) else f"{x * 100:.1f}%"


def infer_prefix_model(columns: list[str]) -> str:
    for column in columns:
        if column.startswith("prob_safe_success__"):
            return column.removeprefix("prob_safe_success__")
        if column.startswith("prob_cal_safe_success__"):
            return column.removeprefix("prob_cal_safe_success__")
    raise ValueError("Could not infer prefix model from prediction columns.")


def score_columns(prefix_model: str, score_mode: str) -> tuple[str, str]:
    if score_mode == "raw":
        return (
            f"prob_safe_success__{prefix_model}",
            f"prob_safe_failure__{prefix_model}",
        )
    if score_mode == "calibrated":
        return (
            f"prob_cal_safe_success__{prefix_model}",
            f"prob_cal_safe_failure__{prefix_model}",
        )
    raise ValueError(score_mode)


def parse_run_dir(path: Path) -> tuple[str, str, str]:
    prefix = "per_instance_model_valid3_"
    suffix = "_safe_stop_dual_head_retrain"
    if not path.name.startswith(prefix) or not path.name.endswith(suffix):
        raise ValueError(f"Unexpected run dir name: {path.name}")
    body = path.name[len(prefix) : -len(suffix)]
    parts = body.split("_", 1)
    split = parts[0]
    if split not in {"top3", "mid3", "bottom3"}:
        raise ValueError(f"Unexpected run dir name: {path.name}")
    rest = parts[1] if len(parts) > 1 else ""
    if rest in {"i", "j"}:
        return split, rest.upper(), "baseline"
    if rest.startswith("i_") or rest.startswith("j_"):
        return split, rest[0].upper(), rest[2:]
    return split, "IJ", rest or "baseline"


def load_records(df: pd.DataFrame, success_col: str, failure_col: str) -> list[dict]:
    needed = ["traj_id", "orig_model_id", "label", "prefix_step_idx", success_col, failure_col]
    work = df[needed].copy()
    records: list[dict] = []
    for traj_id, group in work.groupby("traj_id", sort=False):
        group = group.sort_values("prefix_step_idx")
        records.append(
            {
                "traj_id": str(traj_id),
                "agent_model": str(group["orig_model_id"].iloc[0]),
                "label": int(group["label"].iloc[0]),
                "n_steps": int(len(group)),
                "steps": group["prefix_step_idx"].to_numpy(dtype=np.int32),
                "success": group[success_col].to_numpy(dtype=np.float64),
                "failure": group[failure_col].to_numpy(dtype=np.float64),
            }
        )
    return records


def original_summary(records: list[dict]) -> dict[str, float]:
    total = int(len(records))
    resolved = int(sum(int(rec["label"]) for rec in records))
    return {
        "original_total": total,
        "original_resolved": resolved,
        "original_resolve_rate": resolved / total if total else float("nan"),
    }


def empty_counts() -> dict[str, int]:
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


def decide_current(
    record: dict,
    *,
    success_thr: float,
    failure_thr: float,
    min_step: int,
    consecutive: int,
) -> tuple[bool, str, int, float]:
    last_decision = "undecided"
    streak = 0
    for step_value, success_score, failure_score in zip(record["steps"], record["success"], record["failure"]):
        step = int(step_value)
        if step < min_step:
            continue
        success_hit = float(success_score) >= success_thr
        failure_hit = float(failure_score) >= failure_thr
        if success_hit and failure_hit:
            success_margin = float(success_score) - success_thr
            failure_margin = float(failure_score) - failure_thr
            decision = "success" if success_margin >= failure_margin else "failure"
        elif success_hit:
            decision = "success"
        elif failure_hit:
            decision = "failure"
        else:
            last_decision = "undecided"
            streak = 0
            continue
        streak = streak + 1 if decision == last_decision else 1
        last_decision = decision
        if streak >= consecutive:
            score = float(success_score - failure_score)
            return True, decision, step, score
    return False, "undecided", -1, float("nan")


def decide_conjunctive(
    record: dict,
    *,
    threshold: float,
    min_step: int,
    consecutive: int,
) -> tuple[bool, str, int, float]:
    low = 1.0 - threshold
    last_decision = "undecided"
    streak = 0
    for step_value, success_score, failure_score in zip(record["steps"], record["success"], record["failure"]):
        step = int(step_value)
        if step < min_step:
            continue
        success_hit = float(success_score) >= threshold and float(failure_score) <= low
        failure_hit = float(failure_score) >= threshold and float(success_score) <= low
        if success_hit:
            decision = "success"
        elif failure_hit:
            decision = "failure"
        else:
            last_decision = "undecided"
            streak = 0
            continue
        streak = streak + 1 if decision == last_decision else 1
        last_decision = decision
        if streak >= consecutive:
            score = float(success_score - failure_score)
            return True, decision, step, score
    return False, "undecided", -1, float("nan")


def evaluate_policy(
    records: list[dict],
    policy: dict,
    *,
    mode: str,
) -> dict[str, float]:
    original = original_summary(records)
    counts = empty_counts()
    decided_scores = []
    decided_labels = []
    for record in records:
        counts["total_steps"] += int(record["n_steps"])
        if mode == "current":
            decided, decision, decision_step, score = decide_current(
                record,
                success_thr=float(policy["success_thr"]),
                failure_thr=float(policy["failure_thr"]),
                min_step=int(policy["min_step"]),
                consecutive=int(policy["consecutive"]),
            )
        elif mode == "conj":
            decided, decision, decision_step, score = decide_conjunctive(
                record,
                threshold=float(policy["threshold"]),
                min_step=int(policy["min_step"]),
                consecutive=int(policy["consecutive"]),
            )
        else:
            raise ValueError(mode)

        if not decided:
            counts["undecided"] += 1
            continue

        counts["total_saved_steps"] += max(int(record["n_steps"]) - decision_step - 1, 0)
        decided_scores.append(float(score))
        decided_labels.append(int(record["label"]))
        if decision == "failure":
            counts["decided_failure"] += 1
            if record["label"] == 1:
                counts["false_negatives"] += 1
            else:
                counts["true_negatives"] += 1
        else:
            counts["decided_success"] += 1
            if record["label"] == 0:
                counts["false_positives"] += 1
            else:
                counts["true_positives"] += 1

    tp = counts["true_positives"]
    tn = counts["true_negatives"]
    fp = counts["false_positives"]
    fn = counts["false_negatives"]
    decided_success = counts["decided_success"]
    decided_failure = counts["decided_failure"]
    n_decided = decided_success + decided_failure
    undecided_resolved = original["original_resolved"] - tp - fn
    adjusted_resolved = tp + undecided_resolved
    adjusted_rate = adjusted_resolved / original["original_total"] if original["original_total"] else float("nan")
    labels_arr = np.asarray(decided_labels, dtype=np.int32)
    scores_arr = np.asarray(decided_scores, dtype=np.float64)
    margin_auc = auc_rank(labels_arr, scores_arr) if n_decided and len(np.unique(labels_arr)) > 1 else float("nan")
    return {
        **original,
        "decided_failure": decided_failure,
        "decided_success": decided_success,
        "undecided": counts["undecided"],
        "false_negatives": fn,
        "true_negatives": tn,
        "false_positives": fp,
        "true_positives": tp,
        "n_decided": n_decided,
        "coverage": n_decided / original["original_total"] if original["original_total"] else float("nan"),
        "decision_accuracy": (tp + tn) / n_decided if n_decided else float("nan"),
        "precision_success": tp / decided_success if decided_success else float("nan"),
        "precision_failure": tn / decided_failure if decided_failure else float("nan"),
        "adjusted_resolved": adjusted_resolved,
        "adjusted_resolve_rate": adjusted_rate,
        "resolve_rate_drop": original["original_resolve_rate"] - adjusted_rate,
        "pct_steps_saved": counts["total_saved_steps"] * 100.0 / counts["total_steps"] if counts["total_steps"] else float("nan"),
        "total_saved_steps": counts["total_saved_steps"],
        "total_steps": counts["total_steps"],
        "margin_auc": margin_auc,
        "abs_drop_pp": abs((original["original_resolve_rate"] - adjusted_rate) * 100.0),
        "drop_pp": (original["original_resolve_rate"] - adjusted_rate) * 100.0,
        "decision_accuracy_for_filter": (tp + tn) / n_decided if n_decided else float("nan"),
        "pct_steps_saved_for_sort": counts["total_saved_steps"] * 100.0 / counts["total_steps"] if counts["total_steps"] else float("nan"),
    }


def select_policy(valid_grid: pd.DataFrame) -> pd.Series:
    work = valid_grid.copy()
    strict = work[
        (work["abs_drop_pp"] <= 2.0)
        & (work["decision_accuracy_for_filter"] >= 0.90)
        & (work["pct_steps_saved_for_sort"] > 0.0)
    ].copy()
    if not strict.empty:
        chosen = strict.sort_values(
            ["pct_steps_saved_for_sort", "abs_drop_pp", "decision_accuracy_for_filter"],
            ascending=[False, True, False],
        ).iloc[0]
        chosen = chosen.copy()
        chosen["selection_status"] = "valid_constraints_pass"
        return chosen
    fallback = work[work["pct_steps_saved_for_sort"] >= 5.0].copy()
    if fallback.empty:
        fallback = work
    chosen = fallback.sort_values(
        ["abs_drop_pp", "pct_steps_saved_for_sort", "decision_accuracy_for_filter"],
        ascending=[True, False, False],
    ).iloc[0]
    chosen = chosen.copy()
    chosen["selection_status"] = "fallback_min_abs_valid_drop"
    return chosen


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    run_dirs = sorted(
        path
        for path in REPORTS_DIR.glob("per_instance_model_valid3_*_safe_stop_dual_head_retrain")
        if RUN_RE.match(path.name)
    )
    baseline_rows = []
    conj_valid_rows = []
    conj_test_rows = []
    comparison_rows = []
    sweep_rows = []

    for run_dir in run_dirs:
        split, variant, strategy = parse_run_dir(run_dir)
        baseline_path = run_dir / "safe_stop_selected_policies.csv"
        if not baseline_path.exists():
            continue
        valid_path = run_dir / "valid_predictions_safe_stop.parquet"
        test_path = run_dir / "test_predictions_safe_stop.parquet"
        if not valid_path.exists() or not test_path.exists():
            continue

        valid_df = pd.read_parquet(valid_path)
        test_df = pd.read_parquet(test_path)

        baseline_selected = pd.read_csv(baseline_path)
        if baseline_selected.empty:
            continue
        # Baseline current dual-head policies from existing report, re-evaluated to add margin AUC.
        for _, row in baseline_selected.iterrows():
            score_mode = str(row["score_mode"])
            prefix_model = str(row["prefix_model"])
            succ_col, fail_col = score_columns(prefix_model, score_mode)
            if succ_col not in valid_df.columns or fail_col not in valid_df.columns:
                continue
            valid_records = load_records(valid_df, succ_col, fail_col)
            test_records = load_records(test_df, succ_col, fail_col)
            policy = {
                "policy_mode": str(row["policy_mode"]),
                "success_thr": float(row["success_thr"]),
                "failure_thr": float(row["failure_thr"]),
                "min_step": int(row["min_step"]),
                "consecutive": int(row["consecutive"]),
            }
            baseline_valid = evaluate_policy(valid_records, policy, mode="current")
            baseline_test = evaluate_policy(test_records, policy, mode="current")
            baseline_rows.append(
                {
                    "split": split,
                    "variant": variant,
                    "strategy": strategy,
                    "run": run_dir.name,
                    "score_mode": score_mode,
                    "prefix_model": prefix_model,
                    **policy,
                    "valid_abs_drop_pp": baseline_valid["abs_drop_pp"],
                    "valid_decision_accuracy": baseline_valid["decision_accuracy"],
                    "valid_precision_success": baseline_valid["precision_success"],
                    "valid_precision_failure": baseline_valid["precision_failure"],
                    "valid_margin_auc": baseline_valid["margin_auc"],
                    "valid_pct_steps_saved": baseline_valid["pct_steps_saved"],
                    "valid_drop_pp": baseline_valid["drop_pp"],
                    "test_abs_drop_pp": baseline_test["abs_drop_pp"],
                    "test_decision_accuracy": baseline_test["decision_accuracy"],
                    "test_precision_success": baseline_test["precision_success"],
                    "test_precision_failure": baseline_test["precision_failure"],
                    "test_margin_auc": baseline_test["margin_auc"],
                    "test_pct_steps_saved": baseline_test["pct_steps_saved"],
                    "test_drop_pp": baseline_test["drop_pp"],
                    "test_coverage": baseline_test["coverage"],
                    "test_adjusted_resolve_rate": baseline_test["adjusted_resolve_rate"],
                    "test_original_resolve_rate": baseline_test["original_resolve_rate"],
                    "test_total_saved_steps": baseline_test["total_saved_steps"],
                    "test_total_steps": baseline_test["total_steps"],
                }
            )

        # Conjunctive gate sweep on valid.
        for prefix_model in sorted(baseline_selected["prefix_model"].astype(str).unique()):
            for score_mode in SCORE_MODES:
                succ_col, fail_col = score_columns(prefix_model, score_mode)
                if succ_col not in valid_df.columns or fail_col not in valid_df.columns:
                    continue
                valid_records = load_records(valid_df, succ_col, fail_col)
                test_records = load_records(test_df, succ_col, fail_col)
                total_traj = len(valid_records)

                for threshold in THRESHOLDS:
                    for min_step in MIN_STEPS:
                        for consecutive in CONSECUTIVE:
                            policy = {
                                "policy_mode": "conjunctive",
                                "threshold": float(threshold),
                                "min_step": int(min_step),
                                "consecutive": int(consecutive),
                            }
                            valid_metrics = evaluate_policy(valid_records, policy, mode="conj")
                            valid_metrics.update(
                                {
                                    "split": split,
                                    "variant": variant,
                                    "strategy": strategy,
                                    "run": run_dir.name,
                                    "score_mode": score_mode,
                                    "prefix_model": prefix_model,
                                    **policy,
                                    "selection_status": None,
                                }
                            )
                            valid_metrics["n_all_subset"] = total_traj
                            valid_metrics["valid_abs_drop_pp"] = valid_metrics["abs_drop_pp"]
                            valid_metrics["valid_decision_accuracy"] = valid_metrics["decision_accuracy"]
                            valid_metrics["valid_precision_success"] = valid_metrics["precision_success"]
                            valid_metrics["valid_precision_failure"] = valid_metrics["precision_failure"]
                            valid_metrics["valid_margin_auc"] = valid_metrics["margin_auc"]
                            valid_metrics["valid_pct_steps_saved"] = valid_metrics["pct_steps_saved"]
                            valid_metrics["valid_drop_pp"] = valid_metrics["drop_pp"]
                            sweep_rows.append(valid_metrics)

                valid_grid = pd.DataFrame(
                    [
                        r
                        for r in sweep_rows
                        if r["run"] == run_dir.name
                        and r["score_mode"] == score_mode
                        and r["prefix_model"] == prefix_model
                        and r["policy_mode"] == "conjunctive"
                    ]
                )
                chosen = select_policy(valid_grid)
                chosen_row = chosen.to_dict()
                chosen_policy = {
                    "policy_mode": chosen_row["policy_mode"],
                    "threshold": float(chosen_row["threshold"]),
                    "min_step": int(chosen_row["min_step"]),
                    "consecutive": int(chosen_row["consecutive"]),
                }
                test_metrics = evaluate_policy(test_records, chosen_policy, mode="conj")
                conj_valid_rows.append(chosen_row)
                conj_test_rows.append(
                    {
                        "split": split,
                        "variant": variant,
                        "strategy": strategy,
                        "run": run_dir.name,
                        "score_mode": score_mode,
                        "prefix_model": prefix_model,
                        **chosen_policy,
                        "selection_status": chosen_row["selection_status"],
                        "valid_abs_drop_pp": chosen_row["valid_abs_drop_pp"],
                        "valid_decision_accuracy": chosen_row["valid_decision_accuracy"],
                        "valid_precision_success": chosen_row["valid_precision_success"],
                        "valid_precision_failure": chosen_row["valid_precision_failure"],
                        "valid_margin_auc": chosen_row["valid_margin_auc"],
                        "valid_pct_steps_saved": chosen_row["valid_pct_steps_saved"],
                        "valid_drop_pp": chosen_row["valid_drop_pp"],
                        "test_abs_drop_pp": test_metrics["abs_drop_pp"],
                        "test_decision_accuracy": test_metrics["decision_accuracy"],
                        "test_precision_success": test_metrics["precision_success"],
                        "test_precision_failure": test_metrics["precision_failure"],
                        "test_margin_auc": test_metrics["margin_auc"],
                        "test_pct_steps_saved": test_metrics["pct_steps_saved"],
                        "test_drop_pp": test_metrics["drop_pp"],
                        "test_coverage": test_metrics["coverage"],
                        "test_adjusted_resolve_rate": test_metrics["adjusted_resolve_rate"],
                        "test_original_resolve_rate": test_metrics["original_resolve_rate"],
                        "test_total_saved_steps": test_metrics["total_saved_steps"],
                        "test_total_steps": test_metrics["total_steps"],
                    }
                )

    baseline = pd.DataFrame(baseline_rows)
    conj_valid = pd.DataFrame(conj_valid_rows)
    conj_test = pd.DataFrame(conj_test_rows)
    sweep = pd.DataFrame(sweep_rows)

    baseline.to_csv(OUT_DIR / "current_dual_baseline_recomputed.csv", index=False)
    conj_valid.to_csv(OUT_DIR / "conjunctive_valid_selected.csv", index=False)
    conj_test.to_csv(OUT_DIR / "conjunctive_test_selected.csv", index=False)
    sweep.to_csv(OUT_DIR / "conjunctive_valid_sweep.csv", index=False)

    # Comparison table.
    merged = baseline.merge(
        conj_test,
        on=["split", "variant", "strategy", "run", "score_mode", "prefix_model"],
        suffixes=("_current", "_conj"),
    )
    merged["delta_test_margin_auc"] = merged["test_margin_auc_conj"] - merged["test_margin_auc_current"]
    merged["delta_test_decision_accuracy"] = merged["test_decision_accuracy_conj"] - merged["test_decision_accuracy_current"]
    merged["delta_test_pct_steps_saved"] = merged["test_pct_steps_saved_conj"] - merged["test_pct_steps_saved_current"]
    merged["delta_test_drop_pp"] = merged["test_drop_pp_conj"] - merged["test_drop_pp_current"]
    merged["delta_test_abs_drop_pp"] = merged["test_abs_drop_pp_conj"] - merged["test_abs_drop_pp_current"]
    merged["delta_test_coverage"] = merged["test_coverage_conj"] - merged["test_coverage_current"]

    # Summaries.
    improvement_counts = {
        "better_drop": int((merged["delta_test_drop_pp"] < 0).sum()),
        "better_acc": int((merged["delta_test_decision_accuracy"] > 0).sum()),
        "better_save": int((merged["delta_test_pct_steps_saved"] > 0).sum()),
        "better_margin_auc": int((merged["delta_test_margin_auc"] > 0).sum()),
    }

    lines = [
        "# Dual-Head Conjunctive Gate Report",
        "",
        'Public-release English note.',
        'Public-release English note.',
        "",
        'Public-release English note.',
        "",
        'Public-release English note.',
        "",
        "## Baseline vs Conjunctive Selected",
        "",
        "| Split | Variant | Strategy | Score | Baseline Policy | Conj Policy | Baseline Test Acc | Conj Test Acc | ΔAcc | Baseline Test Save | Conj Test Save | ΔSave | Baseline Test Drop pp | Conj Test Drop pp | ΔDrop pp | Baseline Margin AUC | Conj Margin AUC | ΔAUC |",
        "|:--|:--|:--|:--|:--|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|",
    ]
    for _, row in merged.sort_values(["split", "variant", "strategy", "score_mode"]).iterrows():
        baseline_policy = f"{row['policy_mode_current']} s{row['success_thr']:.2f}/f{row['failure_thr']:.2f} min{int(row['min_step_current'])} k{int(row['consecutive_current'])}"
        conj_policy = f"{row['policy_mode_conj']} thr{row['threshold']:.2f} min{int(row['min_step_conj'])} k{int(row['consecutive_conj'])}"
        lines.append(
            f"| {row['split']} | {row['variant']} | {row['strategy']} | {row['score_mode']} | {baseline_policy} | {conj_policy} | "
            f"{pct(row['test_decision_accuracy_current'])} | {pct(row['test_decision_accuracy_conj'])} | "
            f"{row['delta_test_decision_accuracy']*100.0:+.1f}pp | {row['test_pct_steps_saved_current']:.1f}% | {row['test_pct_steps_saved_conj']:.1f}% | "
            f"{row['delta_test_pct_steps_saved']:+.1f}pp | {row['test_drop_pp_current']:+.2f} | {row['test_drop_pp_conj']:+.2f} | "
            f"{row['delta_test_drop_pp']:+.2f} | {fmt(row['test_margin_auc_current'])} | {fmt(row['test_margin_auc_conj'])} | {row['delta_test_margin_auc']:+.3f} |"
        )
    lines += [
        "",
        "## Conjunctive Sweep Candidates",
        "",
        'Public-release English note.',
        "",
        "| Split | Variant | Strategy | Score | Thr | Min | K | Status | Valid Acc | Valid Save | Valid Drop pp | Valid AUC | Test Acc | Test Save | Test Drop pp | Test AUC |",
        "|:--|:--|:--|:--|--:|--:|--:|:--|--:|--:|--:|--:|--:|--:|--:|--:|",
    ]
    for _, row in conj_valid.sort_values(["split", "variant", "strategy", "score_mode"]).iterrows():
        test_row = conj_test[
            (conj_test["run"] == row["run"]) & (conj_test["score_mode"] == row["score_mode"])
        ].iloc[0]
        lines.append(
            f"| {row['split']} | {row['variant']} | {row['strategy']} | {row['score_mode']} | {row['threshold']:.2f} | {int(row['min_step'])} | {int(row['consecutive'])} | {row['selection_status']} | "
            f"{pct(row['valid_decision_accuracy'])} | {row['valid_pct_steps_saved']:.1f}% | {row['valid_drop_pp']:+.2f} | {fmt(row['valid_margin_auc'])} | "
            f"{pct(test_row['test_decision_accuracy'])} | {test_row['test_pct_steps_saved']:.1f}% | {test_row['test_drop_pp']:+.2f} | {fmt(test_row['test_margin_auc'])} |"
        )
    lines += [
        "",
        "## Files",
        "",
        "- `current_dual_baseline_recomputed.csv`",
        "- `conjunctive_valid_sweep.csv`",
        "- `conjunctive_valid_selected.csv`",
        "- `conjunctive_test_selected.csv`",
    ]
    (OUT_DIR / "dual_head_conjunctive_gate_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(OUT_DIR / "dual_head_conjunctive_gate_report.md")


if __name__ == "__main__":
    main()
