#!/usr/bin/env python3
"""Build a two-end precision-target report from existing asymmetric sweeps.

Selection rule:
  On valid, for each prefix model and target precision t, select the
  (success_threshold, failure_threshold) pair with:

    precision_success >= t and precision_failure >= t

  among those pairs, maximize pct_steps_saved.  Then apply the selected pair to
  heldout test and report test-side rate change / savings.

This script reuses outputs from ``asymmetric_threshold_valid_tuning_posthoc.py``.
It does not recompute features, predictions, or calibration.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_RUN_NAME = "model_holdout_answer_calibrated_full"
DEFAULT_MODELS = [
    "J_LightGBM_Dense_AF_Thought",
    "I_LightGBM_Dense_AF",
    "H_LightGBM_Dense",
    "K_LightGBM_Dense_Full",
    "G_TfIdf_Full_LR",
    "D_Dense_Full_LR",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", default=DEFAULT_RUN_NAME)
    parser.add_argument("--input-subdir", default="asymmetric_valid_threshold_tuning")
    parser.add_argument("--output-subdir", default="two_end_precision_targets")
    parser.add_argument("--targets", nargs="+", type=float, default=[0.75, 0.80, 0.85, 0.90])
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    return parser.parse_args()


def _run_reports_dir(run_name: str) -> Path:
    return PROJECT_ROOT / "runs" / run_name / "reports"


def _fmt_pct(value: Any, digits: int = 1) -> str:
    try:
        if pd.isna(value):
            return "nan%"
        return f"{100.0 * float(value):.{digits}f}%"
    except Exception:
        return "nan%"


def _select_valid_thresholds(valid_sweep: pd.DataFrame, models: list[str], targets: list[float]) -> pd.DataFrame:
    rows = []
    for model_name in models:
        group = valid_sweep[valid_sweep["prefix_model"] == model_name].copy()
        if group.empty:
            continue
        for target in targets:
            candidates = group[
                (group["precision_success"] >= target)
                & (group["precision_failure"] >= target)
                & (group["decided_success"] > 0)
                & (group["decided_failure"] > 0)
            ].copy()
            if candidates.empty:
                rows.append(
                    {
                        "prefix_model": model_name,
                        "target_precision": float(target),
                        "has_valid_candidate": False,
                    }
                )
                continue
            best = candidates.sort_values(
                ["pct_steps_saved", "decision_accuracy", "decision_rate"],
                ascending=[False, False, False],
            ).iloc[0].to_dict()
            best["target_precision"] = float(target)
            best["has_valid_candidate"] = True
            rows.append(best)
    return pd.DataFrame(rows)


def _apply_to_test(test_sweep: pd.DataFrame, selected: pd.DataFrame) -> pd.DataFrame:
    rows = []
    indexed = test_sweep.set_index(["prefix_model", "success_threshold", "failure_threshold"])
    for _, row in selected.iterrows():
        out = row.to_dict()
        if not bool(row.get("has_valid_candidate", False)):
            out["split"] = "test"
            rows.append(out)
            continue
        key = (
            row["prefix_model"],
            float(row["success_threshold"]),
            float(row["failure_threshold"]),
        )
        if key not in indexed.index:
            out["split"] = "test_missing_threshold_pair"
            rows.append(out)
            continue
        test_row = indexed.loc[key]
        if isinstance(test_row, pd.DataFrame):
            test_row = test_row.iloc[0]
        for col, value in test_row.to_dict().items():
            out[f"test_{col}"] = value
        out["test_prefix_model"] = row["prefix_model"]
        out["test_success_threshold"] = float(row["success_threshold"])
        out["test_failure_threshold"] = float(row["failure_threshold"])
        rows.append(out)
    return pd.DataFrame(rows)


def _original_ranks(predictions: pd.DataFrame) -> dict[str, int]:
    final_idx = predictions.groupby("traj_id")["prefix_step_idx"].idxmax()
    final = predictions.loc[final_idx].copy()
    orig = (
        final.groupby("orig_model_id")
        .agg(total=("traj_id", "nunique"), resolved=("label", "sum"))
        .reset_index()
    )
    orig["rate"] = orig["resolved"] / orig["total"]
    orig = orig.sort_values("rate", ascending=False).reset_index(drop=True)
    return {str(model): idx + 1 for idx, model in enumerate(orig["orig_model_id"])}


def _per_agent_rank_change(
    predictions: pd.DataFrame,
    prob_col: str,
    success_threshold: float,
    failure_threshold: float,
    orig_rank: dict[str, int],
) -> int:
    rows = []
    for agent_model, agent_frame in predictions.groupby("orig_model_id", sort=False):
        total = 0
        resolved = 0
        fp = 0
        fn = 0
        for _, traj in agent_frame.sort_values(["traj_id", "prefix_step_idx"]).groupby("traj_id", sort=False):
            label = int(traj["label"].iloc[0])
            total += 1
            resolved += label
            decision = None
            for _, row in traj.iterrows():
                prob = float(row[prob_col])
                if prob >= success_threshold:
                    decision = "success"
                    break
                if prob <= failure_threshold:
                    decision = "failure"
                    break
            if decision == "success" and label == 0:
                fp += 1
            elif decision == "failure" and label == 1:
                fn += 1
        adjusted = resolved + fp - fn
        rows.append(
            {
                "agent_model": str(agent_model),
                "adjusted_rate": adjusted / total if total else np.nan,
            }
        )
    adjusted_df = pd.DataFrame(rows).sort_values("adjusted_rate", ascending=False).reset_index(drop=True)
    new_rank = {str(model): idx + 1 for idx, model in enumerate(adjusted_df["agent_model"])}
    if not new_rank:
        return 0
    return int(max(abs(new_rank[m] - orig_rank.get(m, new_rank[m])) for m in new_rank))


def _add_rank_changes(results: pd.DataFrame, predictions: pd.DataFrame) -> pd.DataFrame:
    out = results.copy()
    orig_rank = _original_ranks(predictions)
    rank_changes = []
    for _, row in out.iterrows():
        if not bool(row.get("has_valid_candidate", False)):
            rank_changes.append(np.nan)
            continue
        prob_col = f"prob_cal__{row['prefix_model']}"
        if prob_col not in predictions.columns:
            rank_changes.append(np.nan)
            continue
        rank_changes.append(
            _per_agent_rank_change(
                predictions,
                prob_col,
                float(row["success_threshold"]),
                float(row["failure_threshold"]),
                orig_rank,
            )
        )
    out["test_max_rank_change"] = rank_changes
    return out


def _write_report(results: pd.DataFrame, output_dir: Path, models: list[str], targets: list[float]) -> str:
    lines = []
    lines.append("=" * 108)
    lines.append("  Two-end precision target report - thresholds selected on valid, applied to heldout test")
    lines.append("=" * 108)
    lines.append("")
    lines.append('Public-release English note.')
    lines.append('Public-release English note.')
    lines.append('Public-release English note.')
    lines.append("")
    for model_name in models:
        sub = results[results["prefix_model"] == model_name].copy()
        if sub.empty:
            continue
        lines.append(f"  Prefix Model: ★ {model_name}")
        for target in targets:
            row = sub[np.isclose(sub["target_precision"], target)]
            if row.empty:
                continue
            r = row.iloc[0]
            if not bool(r.get("has_valid_candidate", False)):
                lines.append('Public-release English note.')
                continue
            lines.append(
                f"    target_precision={target:.2f}, ThrS={float(r['success_threshold']):.2f}, "
                f"ThrF={float(r['failure_threshold']):.2f}: "
                'Public-release English note.'
                'Public-release English note.'
                'Public-release English note.'
                'Public-release English note.'
                'Public-release English note.'
                'Public-release English note.'
                'Public-release English note.'
                f"FP={int(r['test_false_positives'])}, FN={int(r['test_false_negatives'])}"
            )
        lines.append("")
    text = "\n".join(lines)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "report.txt").write_text(text + "\n", encoding="utf-8")
    return text


def main() -> int:
    args = parse_args()
    reports_dir = _run_reports_dir(args.run_name)
    input_dir = reports_dir / args.input_subdir
    output_dir = reports_dir / args.output_subdir
    valid_sweep = pd.read_csv(input_dir / "valid_sweep.csv")
    test_sweep = pd.read_csv(input_dir / "test_sweep.csv")
    predictions = pd.read_parquet(reports_dir / "test_predictions_all_models.parquet")
    selected = _select_valid_thresholds(valid_sweep, args.models, args.targets)
    results = _apply_to_test(test_sweep, selected)
    results = _add_rank_changes(results, predictions)
    output_dir.mkdir(parents=True, exist_ok=True)
    selected.to_csv(output_dir / "valid_precision_target_selected_thresholds.csv", index=False)
    results.to_csv(output_dir / "test_precision_target_results.csv", index=False)
    text = _write_report(results, output_dir, args.models, args.targets)
    print(text)
    print(f"\n[two_end_precision_target_report] wrote {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

