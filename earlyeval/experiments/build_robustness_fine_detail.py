from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from earlyeval.experiments.build_reporting_detail import (
    ROOT,
    TARGETS,
    _actual_change_pp,
    _format_thr,
    _markdown_table,
    _normalize_summary_columns,
    _policy_from_row,
)
from earlyeval.experiments.build_robustness_detail import (
    DATASETS,
    RUNS,
    _attach_mean_abs_from_parts,
    _plot_frontier,
    _plot_per_model_heatmap,
    _write_csv,
    _write_summary_readme,
)
from earlyeval.experiments.fast_cartesian_policy_sweep import evaluate_cartesian_grid
from earlyeval.policies.safe_stop import apply_policy


FINE_THRESHOLDS = [round(x / 100.0, 2) for x in range(75, 96)]


def _select_fine_dual_policy_for_target(valid_grid: pd.DataFrame, target: float) -> tuple[pd.Series, str]:
    work = _normalize_summary_columns(valid_grid)
    work["decision_accuracy_fraction"] = pd.to_numeric(work["decision_accuracy_pct"], errors="coerce").fillna(-100.0) / 100.0
    work["step_save_for_sort"] = pd.to_numeric(work["step_save_pct"], errors="coerce").fillna(0.0)
    work["valid_abs_change_pp"] = _actual_change_pp(work).abs()
    work["success_thr_for_sort"] = pd.to_numeric(work["success_thr"], errors="coerce")
    work["failure_thr_for_sort"] = pd.to_numeric(work["failure_thr"], errors="coerce")
    work["threshold_gap_for_sort"] = (work["success_thr_for_sort"] - work["failure_thr_for_sort"]).abs()

    strict = work[(work["decision_accuracy_fraction"] >= target) & (work["step_save_for_sort"] > 0.0)].copy()
    if not strict.empty:
        chosen = strict.sort_values(
            [
                "step_save_for_sort",
                "valid_abs_change_pp",
                "decision_accuracy_fraction",
                "threshold_gap_for_sort",
                "success_thr_for_sort",
                "failure_thr_for_sort",
            ],
            ascending=[False, True, False, True, False, False],
        ).iloc[0]
        return chosen, "valid_accuracy_pass"

    fallback = work[work["step_save_for_sort"] > 0.0].copy()
    if fallback.empty:
        fallback = work.copy()
    chosen = fallback.sort_values(
        [
            "decision_accuracy_fraction",
            "valid_abs_change_pp",
            "step_save_for_sort",
            "threshold_gap_for_sort",
            "success_thr_for_sort",
            "failure_thr_for_sort",
        ],
        ascending=[False, True, False, True, False, False],
    ).iloc[0]
    return chosen, "fallback_highest_valid_accuracy"


def _candidate_grid(frame: pd.DataFrame, *, predictor: str) -> pd.DataFrame:
    grid = evaluate_cartesian_grid(
        frame,
        predictor=predictor,
        score_mode="calibrated",
        success_thresholds=FINE_THRESHOLDS,
        failure_thresholds=FINE_THRESHOLDS,
    )
    grid["policy_mode"] = "dual"
    grid["policy_id"] = grid["policy_name"]
    grid["step_save_pct"] = grid["pct_steps_saved"]
    return _normalize_summary_columns(grid)


def _fine_frontier_for_dataset(dataset_dir: Path, dataset: str, *, predictor: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    valid_frame = pd.read_parquet(dataset_dir / "valid_predictions_safe_stop.parquet")
    test_frame = pd.read_parquet(dataset_dir / "test_predictions_safe_stop.parquet")
    valid_grid = _candidate_grid(valid_frame, predictor=predictor)
    selected_rows: list[dict[str, Any]] = []
    aggregate_rows: list[dict[str, Any]] = []
    per_model_rows: list[pd.DataFrame] = []
    cache: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}
    for target in TARGETS:
        selected, status = _select_fine_dual_policy_for_target(valid_grid, target)
        policy = _policy_from_row(selected)
        if policy.name not in cache:
            _, summary, per_agent = apply_policy(test_frame, policy)
            cache[policy.name] = (_normalize_summary_columns(summary), _normalize_summary_columns(per_agent))
        summary, per_agent = cache[policy.name]
        selected_norm = _normalize_summary_columns(pd.DataFrame([selected])).iloc[0]
        selected_rows.append(
            {
                "dataset": dataset,
                "target_valid_decision_accuracy": target,
                "target_valid_decision_accuracy_pct": target * 100.0,
                "selection_status": status,
                "selected_policy_name": policy.name,
                "selected_score_mode": policy.score_mode,
                "selected_predictor": policy.predictor,
                "selected_policy_mode": policy.policy_mode,
                "selected_success_thr": _format_thr(policy.success_thr),
                "selected_failure_thr": _format_thr(policy.failure_thr),
                "selected_valid_save_pct": float(selected_norm["step_save_pct"]),
                "selected_valid_decision_accuracy_pct": float(selected_norm["decision_accuracy_pct"]),
                "selected_valid_resolve_change_pp": float(selected_norm["resolve_rate_change_pp"]),
                "selected_valid_decided_success": int(selected_norm.get("decided_success", 0)),
                "selected_valid_decided_failure": int(selected_norm.get("decided_failure", 0)),
                "selected_valid_false_negatives": int(selected_norm.get("false_negatives", 0)),
                "selected_valid_false_positives": int(selected_norm.get("false_positives", 0)),
            }
        )
        row = summary.iloc[0].to_dict()
        row.update(
            {
                "dataset": dataset,
                "target_valid_decision_accuracy": target,
                "target_valid_decision_accuracy_pct": target * 100.0,
                "selection_status": status,
                "selected_policy_name": policy.name,
                "selected_policy_mode": policy.policy_mode,
                "selected_success_thr": _format_thr(policy.success_thr),
                "selected_failure_thr": _format_thr(policy.failure_thr),
            }
        )
        aggregate_rows.append(row)
        agent = per_agent.copy()
        agent.insert(0, "dataset", dataset)
        agent.insert(1, "target_valid_decision_accuracy", target)
        agent.insert(2, "target_valid_decision_accuracy_pct", target * 100.0)
        agent.insert(3, "selected_policy_name", policy.name)
        agent.insert(4, "selected_policy_mode", policy.policy_mode)
        agent.insert(5, "selected_success_thr", _format_thr(policy.success_thr))
        agent.insert(6, "selected_failure_thr", _format_thr(policy.failure_thr))
        per_model_rows.append(agent)
    return (
        valid_grid,
        pd.DataFrame(selected_rows),
        _normalize_summary_columns(pd.DataFrame(aggregate_rows)),
        _normalize_summary_columns(pd.concat(per_model_rows, ignore_index=True)),
    )


def _display_cols(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["target"] = out["target_valid_decision_accuracy_pct"].astype(float)
    return out[
        [
            "dataset",
            "target",
            "step_save_pct",
            "decision_accuracy_pct",
            "coverage_pct",
            "resolve_rate_change_pp",
            "mean_abs_resolve_rate_change_pp",
            "false_negatives",
            "false_positives",
            "decided_success",
            "decided_failure",
            "selected_policy_mode",
            "selected_success_thr",
            "selected_failure_thr",
        ]
    ]


def _write_readme(run: dict[str, Any], out_dir: Path, frontier: pd.DataFrame) -> None:
    lines = [
        "# Robustness Fine Valid-Accuracy Detail",
        "",
        f"- run_dir: `{run['run_dir']}`",
        f"- feature_preset: `{run['feature_preset']}`",
        f"- predictor: `{run['predictor']}`",
        "- candidate thresholds: `0.75` to `0.95` in `0.01` steps.",
        "- candidate mode: `dual` only.",
        "- selection: choose the highest valid step saving policy meeting each target valid decision accuracy, then apply it unchanged to test.",
        "- tie-break: when validation metrics are exactly tied, prefer success/failure thresholds that are closer together, then the more conservative higher thresholds.",
        "",
        "Resolve-rate shifts use:",
        "",
        "`adjusted_resolved = original_resolved - false_negatives + false_positives`",
        "",
        "## Fine Dual-Only Valid-Accuracy Frontier",
        "",
    ]
    for dataset in DATASETS:
        subset = _display_cols(frontier[frontier["dataset"] == dataset].copy())
        lines.extend([f"### {dataset}", ""])
        lines.extend(_markdown_table(subset, list(subset.columns)))
        lines.append("")
    lines.extend(
        [
            "## Files",
            "",
            "- `fine_valid_policy_candidate_grid.csv`",
            "- `valid_accuracy_075_095_selected_policies.csv`",
            "- `valid_accuracy_075_095_frontier.csv`",
            "- `valid_accuracy_075_095_per_model_test_metrics.csv`",
            "- `valid_accuracy_075_095_frontier.png`",
            "- `valid_accuracy_075_095_per_model_toolathlon_heatmap.png`",
            "- `valid_accuracy_075_095_per_model_terminalbench_heatmap.png`",
            "",
        ]
    )
    (out_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def build_run(run: dict[str, Any]) -> dict[str, Any]:
    run_dir = Path(run["run_dir"])
    out_dir = run_dir / "valid_accuracy_075_095_fine_detail"
    out_dir.mkdir(parents=True, exist_ok=True)
    for stale_name in [
        "selected_test_metrics.csv",
        "selected_test_per_model_metrics.csv",
        "selected_per_model_toolathlon.png",
        "selected_per_model_terminalbench.png",
    ]:
        (out_dir / stale_name).unlink(missing_ok=True)
    candidate_rows = []
    selected_policy_rows = []
    frontier_rows = []
    per_model_rows = []
    for dataset in DATASETS:
        dataset_dir = run_dir / dataset
        candidates, selected_policy, frontier, per_model = _fine_frontier_for_dataset(
            dataset_dir,
            dataset,
            predictor=str(run["predictor"]),
        )
        candidates.insert(0, "dataset", dataset)
        candidate_rows.append(candidates)
        selected_policy_rows.append(selected_policy)
        frontier_rows.append(frontier)
        per_model_rows.append(per_model)
    candidate_grid = pd.concat(candidate_rows, ignore_index=True)
    selected_policy = pd.concat(selected_policy_rows, ignore_index=True)
    frontier = pd.concat(frontier_rows, ignore_index=True)
    per_model = _normalize_summary_columns(pd.concat(per_model_rows, ignore_index=True))
    per_model["mean_abs_resolve_rate_change_pp"] = per_model["resolve_rate_change_pp"].abs()
    frontier = _attach_mean_abs_from_parts(
        frontier,
        per_model,
        ["dataset", "target_valid_decision_accuracy", "target_valid_decision_accuracy_pct"],
    )

    _write_csv(candidate_grid, out_dir / "fine_valid_policy_candidate_grid.csv")
    _write_csv(selected_policy, out_dir / "valid_accuracy_075_095_selected_policies.csv")
    _write_csv(frontier, out_dir / "valid_accuracy_075_095_frontier.csv")
    _write_csv(per_model, out_dir / "valid_accuracy_075_095_per_model_test_metrics.csv")
    _plot_frontier(frontier, out_dir / "valid_accuracy_075_095_frontier.png", f"{run['feature_preset']} Fine Valid-Accuracy Frontier")
    for dataset in DATASETS:
        _plot_per_model_heatmap(
            per_model,
            dataset,
            out_dir / f"valid_accuracy_075_095_per_model_{dataset}_heatmap.png",
            f"{run['feature_preset']} {dataset} Fine Per-Model Actual Shift",
        )
    _write_readme(run, out_dir, frontier)
    return {
        "run_label": run["run_label"],
        "detail_dir": str(out_dir),
        "candidate_rows": int(len(candidate_grid)),
        "frontier_rows": int(len(frontier)),
        "per_model_frontier_rows": int(len(per_model)),
    }


def main() -> int:
    rows = [build_run(run) for run in RUNS]
    pd.DataFrame(rows).to_csv(ROOT / "robustness_fine_detail_manifest.csv", index=False)
    print(pd.DataFrame(rows).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
