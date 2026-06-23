#!/usr/bin/env python3
"""Analyze which last actions/signals trigger threshold decisions.

This is a post-hoc report for the task/answer ablation predictions.  It joins
the prediction table back to the cached prefix table, finds the first prefix
that would trigger a symmetric early-stop decision at each threshold, and
summarizes the action type plus feedback signals visible at that decision step.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_RUN_DIR = PROJECT_ROOT / "runs" / "model_holdout_answer_calibrated_full"
DEFAULT_REPORT_DIR = DEFAULT_RUN_DIR / "reports" / "task_answer_ablation_posthoc"
DEFAULT_PREDICTIONS = DEFAULT_REPORT_DIR / "test_predictions_task_answer_ablation.parquet"
DEFAULT_PREFIX_TABLE = DEFAULT_RUN_DIR / "data" / "prefix_table_filtered.parquet"
DEFAULT_OUTPUT_DIR = DEFAULT_REPORT_DIR / "decision_action_analysis"
DEFAULT_THRESHOLDS = [0.50, 0.60, 0.70, 0.80, 0.90, 0.95]

SIGNAL_COLUMNS = [
    "last_step_tool_error_seen",
    "last_step_traceback_seen",
    "last_step_test_fail_seen",
    "last_step_test_pass_seen",
    "tool_error_seen",
    "traceback_seen",
    "test_fail_seen",
    "test_pass_seen",
    "all_tests_passed_seen",
    "first_edit_seen",
    "first_test_seen",
    "first_submit_seen",
]

NUMERIC_SIGNAL_COLUMNS = [
    "edits_so_far",
    "tests_so_far",
    "submit_so_far",
    "steps_since_last_edit",
    "steps_since_last_test",
    "last_fail_count",
    "best_fail_count_so_far",
    "fail_count_delta_from_prev_test",
]

PREFIX_COLUMNS = [
    "prefix_id",
    "traj_id",
    "prefix_step_idx",
    "last_action_text",
    "last_feedback_text",
    "last_step_action_major_type",
    "last_step_action_primary_subtype",
    *SIGNAL_COLUMNS,
    *NUMERIC_SIGNAL_COLUMNS,
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--prefix-table", type=Path, default=DEFAULT_PREFIX_TABLE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--thresholds", nargs="+", type=float, default=DEFAULT_THRESHOLDS)
    parser.add_argument(
        "--score-modes",
        nargs="+",
        choices=("raw", "calibrated"),
        default=["calibrated"],
        help="Use raw prob__ columns, calibrated prob_cal__ columns, or both.",
    )
    parser.add_argument(
        "--predictors",
        nargs="+",
        default=None,
        help="Predictor names without prob__/prob_cal__ prefix. Defaults to all in the predictions table.",
    )
    parser.add_argument("--consecutive-k", type=int, default=1)
    parser.add_argument("--delay-steps", type=int, default=0)
    parser.add_argument("--max-action-chars", type=int, default=360)
    parser.add_argument("--max-feedback-chars", type=int, default=240)
    return parser.parse_args()


def _clean_snippet(text: Any, max_chars: int) -> str:
    if text is None or (isinstance(text, float) and np.isnan(text)):
        return ""
    out = re.sub(r"\s+", " ", str(text)).strip()
    if len(out) > max_chars:
        return out[: max_chars - 3] + "..."
    return out


def _command_kind(text: Any) -> str:
    if text is None or (isinstance(text, float) and np.isnan(text)):
        return "none"
    text = str(text).strip()
    if not text:
        return "none"
    low = text.lower()
    if "cat <<" in low and ">" in low:
        return "write_file_shell"
    if low.startswith("cat >"):
        return "write_file_shell"
    if "sed -i" in low:
        return "sed_edit"
    if "python - <<" in low or "python3 - <<" in low:
        return "python_heredoc"
    if low.startswith("find "):
        return "find"
    if low == "ls" or low.startswith("ls ") or low.startswith("ls\n"):
        return "ls"
    if low.startswith("grep ") or " grep " in low:
        return "grep_pipeline"
    if low.startswith("cat "):
        return "cat_read"
    if low.startswith("sed "):
        return "sed_read"
    if low.startswith("pytest") or " pytest" in low or "python -m pytest" in low:
        return "pytest"
    if low.startswith("python ") or low.startswith("python3 "):
        return "python"
    if low.startswith("git "):
        return "git"
    if low.startswith("true"):
        return "format_error_true"
    if low.startswith("submit"):
        return "submit"
    return low.split()[0][:40]


def _available_predictors(pred_df: pd.DataFrame, score_modes: list[str]) -> list[str]:
    prefixes = []
    if "raw" in score_modes:
        prefixes.append("prob__")
    if "calibrated" in score_modes:
        prefixes.append("prob_cal__")
    names: set[str] = set()
    for col in pred_df.columns:
        for prefix in prefixes:
            if col.startswith(prefix):
                names.add(col[len(prefix):])
    return sorted(names)


def _decide_first(
    group: pd.DataFrame,
    prob_col: str,
    threshold: float,
    consecutive_k: int,
    delay_steps: int,
) -> dict[str, Any]:
    low = 1.0 - threshold
    success_streak = 0
    failure_streak = 0
    for _, row in group.sort_values("prefix_step_idx").iterrows():
        step = int(row["prefix_step_idx"])
        if step < delay_steps:
            success_streak = 0
            failure_streak = 0
            continue
        prob = float(row[prob_col])
        if prob >= threshold:
            success_streak += 1
            failure_streak = 0
        elif prob <= low:
            failure_streak += 1
            success_streak = 0
        else:
            success_streak = 0
            failure_streak = 0
        if success_streak >= consecutive_k:
            return {
                "decided": True,
                "decision": "success",
                "decision_step": step,
                "decision_prob": prob,
                "decision_prefix_id": row["prefix_id"],
            }
        if failure_streak >= consecutive_k:
            return {
                "decided": True,
                "decision": "failure",
                "decision_step": step,
                "decision_prob": prob,
                "decision_prefix_id": row["prefix_id"],
            }
    return {
        "decided": False,
        "decision": "undecided",
        "decision_step": -1,
        "decision_prob": np.nan,
        "decision_prefix_id": "",
    }


def build_decision_cases(
    pred_df: pd.DataFrame,
    predictors: list[str],
    score_modes: list[str],
    thresholds: list[float],
    consecutive_k: int,
    delay_steps: int,
) -> pd.DataFrame:
    trajectory_groups = []
    for traj_id, group in pred_df.groupby("traj_id", sort=False):
        first = group.iloc[0]
        trajectory_groups.append(
            {
                "traj_id": traj_id,
                "instance_id": first["instance_id"],
                "orig_model_id": first.get("orig_model_id", ""),
                "label": int(first["label"]),
                "n_steps_total": int(first["n_steps_total_for_weighting"]),
                "group": group.sort_values("prefix_step_idx"),
            }
        )

    rows: list[dict[str, Any]] = []
    for predictor in predictors:
        for score_mode in score_modes:
            prob_col = f"prob__{predictor}" if score_mode == "raw" else f"prob_cal__{predictor}"
            if prob_col not in pred_df.columns:
                continue
            for threshold in thresholds:
                threshold = float(threshold)
                for item in trajectory_groups:
                    dec = _decide_first(
                        item["group"],
                        prob_col,
                        threshold,
                        consecutive_k=consecutive_k,
                        delay_steps=delay_steps,
                    )
                    correct = (
                        (dec["decision"] == "success" and item["label"] == 1)
                        or (dec["decision"] == "failure" and item["label"] == 0)
                    )
                    saved_steps = (
                        max(item["n_steps_total"] - int(dec["decision_step"]), 0)
                        if dec["decided"]
                        else 0
                    )
                    rows.append(
                        {
                            "predictor": predictor,
                            "score_mode": score_mode,
                            "threshold": threshold,
                            "low_threshold": 1.0 - threshold,
                            "traj_id": item["traj_id"],
                            "instance_id": item["instance_id"],
                            "orig_model_id": item["orig_model_id"],
                            "label": item["label"],
                            "n_steps_total": item["n_steps_total"],
                            **dec,
                            "correct": bool(correct) if dec["decided"] else False,
                            "saved_steps": int(saved_steps),
                        }
                    )
    return pd.DataFrame(rows)


def _top_values(series: pd.Series, k: int = 5) -> str:
    if series.empty:
        return ""
    counts = series.fillna("none").astype(str).value_counts().head(k)
    return "; ".join(f"{idx}:{int(val)}" for idx, val in counts.items())


def _mean_bool(series: pd.Series) -> float:
    if series.empty:
        return float("nan")
    return float(series.fillna(False).astype(bool).mean())


def build_summary(cases: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    keys = ["predictor", "score_mode", "threshold", "decision"]
    totals = cases.groupby(["predictor", "score_mode", "threshold"], dropna=False).size().to_dict()
    for key, group in cases.groupby(keys, dropna=False, sort=True):
        predictor, score_mode, threshold, decision = key
        decided = group[group["decided"]].copy()
        denom = int(totals.get((predictor, score_mode, threshold), len(group)))
        rows.append(
            {
                "predictor": predictor,
                "score_mode": score_mode,
                "threshold": threshold,
                "decision": decision,
                "n_total": int(denom),
                "n_rows": int(len(group)),
                "n_decided": int(len(decided)),
                "share_of_all": float(len(group) / denom) if denom else 0.0,
                "accuracy_decided": float(decided["correct"].mean()) if len(decided) else np.nan,
                "avg_decision_step": float(decided["decision_step"].mean()) if len(decided) else np.nan,
                "median_decision_step": float(decided["decision_step"].median()) if len(decided) else np.nan,
                "step0_share": float((decided["decision_step"] == 0).mean()) if len(decided) else np.nan,
                "avg_saved_steps": float(decided["saved_steps"].mean()) if len(decided) else np.nan,
                "top_action_major": _top_values(decided["last_step_action_major_type"]) if len(decided) else "",
                "top_action_subtype": _top_values(decided["last_step_action_primary_subtype"]) if len(decided) else "",
                **{
                    f"rate_{col}": _mean_bool(decided[col]) if len(decided) and col in decided else np.nan
                    for col in SIGNAL_COLUMNS
                },
                **{
                    f"mean_{col}": float(pd.to_numeric(decided[col], errors="coerce").mean())
                    if len(decided) and col in decided
                    else np.nan
                    for col in NUMERIC_SIGNAL_COLUMNS
                },
            }
        )
    return pd.DataFrame(rows)


def build_action_summary(cases: pd.DataFrame) -> pd.DataFrame:
    decided = cases[cases["decided"]].copy()
    if decided.empty:
        return pd.DataFrame()
    keys = [
        "predictor",
        "score_mode",
        "threshold",
        "decision",
        "last_step_action_major_type",
        "last_step_action_primary_subtype",
    ]
    grouped = decided.groupby(keys, dropna=False, sort=True)
    rows = []
    for key, group in grouped:
        rows.append(
            {
                "predictor": key[0],
                "score_mode": key[1],
                "threshold": key[2],
                "decision": key[3],
                "action_major": key[4],
                "action_subtype": key[5],
                "n": int(len(group)),
                "accuracy": float(group["correct"].mean()),
                "avg_decision_step": float(group["decision_step"].mean()),
                "step0_share": float((group["decision_step"] == 0).mean()),
                "rate_last_test_fail": _mean_bool(group["last_step_test_fail_seen"]),
                "rate_last_test_pass": _mean_bool(group["last_step_test_pass_seen"]),
                "rate_last_traceback": _mean_bool(group["last_step_traceback_seen"]),
                "rate_last_tool_error": _mean_bool(group["last_step_tool_error_seen"]),
                "mean_tests_so_far": float(pd.to_numeric(group["tests_so_far"], errors="coerce").mean()),
                "mean_edits_so_far": float(pd.to_numeric(group["edits_so_far"], errors="coerce").mean()),
            }
        )
    out = pd.DataFrame(rows)
    return out.sort_values(["predictor", "score_mode", "threshold", "decision", "n"], ascending=[True, True, True, True, False])


def build_by_agent_summary(cases: pd.DataFrame) -> pd.DataFrame:
    decided = cases[cases["decided"]].copy()
    if decided.empty:
        return pd.DataFrame()
    rows = []
    for key, group in decided.groupby(["predictor", "score_mode", "threshold", "orig_model_id", "decision"], sort=True):
        rows.append(
            {
                "predictor": key[0],
                "score_mode": key[1],
                "threshold": key[2],
                "orig_model_id": key[3],
                "decision": key[4],
                "n": int(len(group)),
                "accuracy": float(group["correct"].mean()),
                "avg_decision_step": float(group["decision_step"].mean()),
                "step0_share": float((group["decision_step"] == 0).mean()),
                "top_action_subtype": _top_values(group["last_step_action_primary_subtype"]),
                "rate_last_test_fail": _mean_bool(group["last_step_test_fail_seen"]),
                "rate_last_test_pass": _mean_bool(group["last_step_test_pass_seen"]),
                "rate_last_traceback": _mean_bool(group["last_step_traceback_seen"]),
            }
        )
    return pd.DataFrame(rows)


def build_command_summary(cases: pd.DataFrame) -> pd.DataFrame:
    decided = cases[cases["decided"]].copy()
    if decided.empty:
        return pd.DataFrame()
    rows = []
    for key, group in decided.groupby(
        ["predictor", "score_mode", "threshold", "decision", "last_action_kind"],
        dropna=False,
        sort=True,
    ):
        examples = group["last_action_snippet"].dropna().astype(str)
        examples = examples[examples.str.len() > 0]
        rows.append(
            {
                "predictor": key[0],
                "score_mode": key[1],
                "threshold": key[2],
                "decision": key[3],
                "last_action_kind": key[4],
                "n": int(len(group)),
                "accuracy": float(group["correct"].mean()),
                "avg_decision_step": float(group["decision_step"].mean()),
                "step0_share": float((group["decision_step"] == 0).mean()),
                "rate_last_test_fail": _mean_bool(group["last_step_test_fail_seen"]),
                "rate_last_test_pass": _mean_bool(group["last_step_test_pass_seen"]),
                "rate_last_traceback": _mean_bool(group["last_step_traceback_seen"]),
                "rate_last_tool_error": _mean_bool(group["last_step_tool_error_seen"]),
                "mean_tests_so_far": float(pd.to_numeric(group["tests_so_far"], errors="coerce").mean()),
                "mean_edits_so_far": float(pd.to_numeric(group["edits_so_far"], errors="coerce").mean()),
                "example_action": examples.iloc[0] if len(examples) else "",
            }
        )
    out = pd.DataFrame(rows)
    return out.sort_values(
        ["predictor", "score_mode", "threshold", "decision", "n"],
        ascending=[True, True, True, True, False],
    )


def write_markdown_report(
    output_path: Path,
    cases: pd.DataFrame,
    summary: pd.DataFrame,
    action_summary: pd.DataFrame,
    command_summary: pd.DataFrame,
    predictors: list[str],
    thresholds: list[float],
    score_modes: list[str],
) -> None:
    lines: list[str] = []
    lines.append("# Decision-Step Last Action Analysis")
    lines.append("")
    lines.append("This report analyzes the first prefix step that triggers an early-stop decision.")
    lines.append("")
    lines.append("Decision rule: success if `p >= threshold`; failure if `p <= 1 - threshold`; otherwise continue.")
    lines.append("")
    lines.append(f"- Predictors: `{', '.join(predictors)}`")
    lines.append(f"- Score modes: `{', '.join(score_modes)}`")
    lines.append(f"- Thresholds: `{', '.join(f'{t:.2f}' for t in thresholds)}`")
    lines.append(f"- Trajectories per predictor/mode/threshold: `{cases['traj_id'].nunique()}`")
    lines.append("")

    calibrated = summary[summary["score_mode"] == "calibrated"].copy()
    if not calibrated.empty:
        focus = calibrated[
            calibrated["threshold"].isin([0.70, 0.80, 0.90, 0.95])
            & calibrated["decision"].isin(["success", "failure"])
        ].copy()
        keep_cols = [
            "predictor",
            "threshold",
            "decision",
            "n_rows",
            "accuracy_decided",
            "avg_decision_step",
            "step0_share",
            "top_action_subtype",
            "rate_last_step_test_fail_seen",
            "rate_last_step_test_pass_seen",
            "rate_last_step_traceback_seen",
            "rate_last_step_tool_error_seen",
            "mean_tests_so_far",
            "mean_edits_so_far",
        ]
        focus = focus[[c for c in keep_cols if c in focus.columns]]
        lines.append("## Calibrated Main Summary")
        lines.append("")
        lines.append(focus.to_markdown(index=False, floatfmt=".3f"))
        lines.append("")

    if not action_summary.empty:
        action_focus = action_summary[
            (action_summary["score_mode"] == "calibrated")
            & (action_summary["threshold"].isin([0.80, 0.90]))
        ].copy()
        action_focus = action_focus.groupby(
            ["predictor", "threshold", "decision"], group_keys=False
        ).head(5)
        keep_cols = [
            "predictor",
            "threshold",
            "decision",
            "action_major",
            "action_subtype",
            "n",
            "accuracy",
            "avg_decision_step",
            "step0_share",
            "rate_last_test_fail",
            "rate_last_test_pass",
            "rate_last_traceback",
            "rate_last_tool_error",
            "mean_tests_so_far",
            "mean_edits_so_far",
        ]
        lines.append("## Top Action Types At Decision")
        lines.append("")
        lines.append(action_focus[[c for c in keep_cols if c in action_focus.columns]].to_markdown(index=False, floatfmt=".3f"))
        lines.append("")

    if not command_summary.empty:
        command_focus = command_summary[
            (command_summary["score_mode"] == "calibrated")
            & (command_summary["threshold"].isin([0.80, 0.90]))
        ].copy()
        command_focus = command_focus.groupby(
            ["predictor", "threshold", "decision"], group_keys=False
        ).head(7)
        keep_cols = [
            "predictor",
            "threshold",
            "decision",
            "last_action_kind",
            "n",
            "accuracy",
            "avg_decision_step",
            "step0_share",
            "rate_last_test_fail",
            "rate_last_test_pass",
            "rate_last_tool_error",
            "example_action",
        ]
        lines.append("## Top Command Kinds At Decision")
        lines.append("")
        lines.append(command_focus[[c for c in keep_cols if c in command_focus.columns]].to_markdown(index=False, floatfmt=".3f"))
        lines.append("")

    lines.append("## Output Files")
    lines.append("")
    lines.append("- `decision_action_cases.csv`: one row per trajectory/predictor/mode/threshold decision candidate.")
    lines.append("- `decision_action_summary.csv`: aggregate signal rates by predictor, score mode, threshold, and decision side.")
    lines.append("- `decision_action_by_action_type.csv`: aggregate rows by last action major/subtype.")
    lines.append("- `decision_action_by_command_kind.csv`: aggregate rows by normalized shell command kind.")
    lines.append("- `decision_action_by_agent_model.csv`: aggregate rows by heldout agent model.")
    lines.append("")
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    pred_df = pd.read_parquet(args.predictions)
    predictors = args.predictors or _available_predictors(pred_df, args.score_modes)
    missing = []
    for predictor in predictors:
        for score_mode in args.score_modes:
            col = f"prob__{predictor}" if score_mode == "raw" else f"prob_cal__{predictor}"
            if col not in pred_df.columns:
                missing.append(col)
    if missing:
        raise RuntimeError(f"Missing probability columns: {missing}")

    prefix_cols = [c for c in PREFIX_COLUMNS]
    prefix_df = pd.read_parquet(args.prefix_table, columns=prefix_cols)
    prefix_df = prefix_df.drop_duplicates("prefix_id", keep="first")
    prefix_df = prefix_df.drop(columns=["traj_id", "prefix_step_idx"], errors="ignore")

    cases = build_decision_cases(
        pred_df=pred_df,
        predictors=predictors,
        score_modes=args.score_modes,
        thresholds=sorted(set(args.thresholds)),
        consecutive_k=args.consecutive_k,
        delay_steps=args.delay_steps,
    )
    cases = cases.merge(prefix_df, left_on="decision_prefix_id", right_on="prefix_id", how="left")
    cases["last_action_snippet"] = cases["last_action_text"].map(lambda x: _clean_snippet(x, args.max_action_chars))
    cases["last_feedback_snippet"] = cases["last_feedback_text"].map(lambda x: _clean_snippet(x, args.max_feedback_chars))
    cases["last_action_kind"] = cases["last_action_snippet"].map(_command_kind)
    cases.drop(columns=["last_action_text", "last_feedback_text"], inplace=True)

    cases_path = args.output_dir / "decision_action_cases.csv"
    summary_path = args.output_dir / "decision_action_summary.csv"
    action_summary_path = args.output_dir / "decision_action_by_action_type.csv"
    command_summary_path = args.output_dir / "decision_action_by_command_kind.csv"
    by_agent_path = args.output_dir / "decision_action_by_agent_model.csv"
    report_path = args.output_dir / "decision_action_report.md"

    cases.to_csv(cases_path, index=False)
    summary = build_summary(cases)
    action_summary = build_action_summary(cases)
    command_summary = build_command_summary(cases)
    by_agent_summary = build_by_agent_summary(cases)
    summary.to_csv(summary_path, index=False)
    action_summary.to_csv(action_summary_path, index=False)
    command_summary.to_csv(command_summary_path, index=False)
    by_agent_summary.to_csv(by_agent_path, index=False)
    write_markdown_report(
        report_path,
        cases=cases,
        summary=summary,
        action_summary=action_summary,
        command_summary=command_summary,
        predictors=predictors,
        thresholds=sorted(set(args.thresholds)),
        score_modes=args.score_modes,
    )

    print(f"Wrote {cases_path}")
    print(f"Wrote {summary_path}")
    print(f"Wrote {action_summary_path}")
    print(f"Wrote {command_summary_path}")
    print(f"Wrote {by_agent_path}")
    print(f"Wrote {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
