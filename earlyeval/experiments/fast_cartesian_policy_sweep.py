from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from earlyeval.core.io import ensure_dir, write_table
from earlyeval.experiments.paper_pipeline import (
    _aggregate_policy_sweep_by_target,
    _default_output_dir,
    _excluded_models_from_config,
    _float_sequence,
    _format_prob,
    _markdown_table,
    _select_policy_for_valid_target,
    _write_lightgbm_policy_sweep_plots,
    load_earlyeval_config,
)
from earlyeval.policies.safe_stop import head_column


def _policy_name(
    *,
    score_mode: str,
    predictor: str,
    success_thr: float,
    failure_thr: float,
    min_step: int,
    consecutive: int,
) -> str:
    return (
        f"{score_mode}__{predictor}__dual__"
        f"s{_format_prob(success_thr)}__f{_format_prob(failure_thr)}__"
        f"min{min_step}__k{consecutive}"
    )


def _empty_counts(shape: tuple[int, int]) -> dict[str, np.ndarray]:
    return {
        "decided_failure": np.zeros(shape, dtype=np.int64),
        "decided_success": np.zeros(shape, dtype=np.int64),
        "undecided": np.zeros(shape, dtype=np.int64),
        "false_negatives": np.zeros(shape, dtype=np.int64),
        "true_negatives": np.zeros(shape, dtype=np.int64),
        "false_positives": np.zeros(shape, dtype=np.int64),
        "true_positives": np.zeros(shape, dtype=np.int64),
        "total_saved_steps": np.zeros(shape, dtype=np.int64),
    }


def evaluate_cartesian_grid(
    frame: pd.DataFrame,
    *,
    predictor: str,
    score_mode: str,
    success_thresholds: list[float],
    failure_thresholds: list[float],
    min_step: int = 0,
    consecutive: int = 1,
    conflict_mode: str = "margin",
    opposite_max: float | None = None,
) -> pd.DataFrame:
    if consecutive != 1:
        raise ValueError("fast cartesian evaluator currently supports consecutive=1 only")
    if min_step != 0:
        raise ValueError("fast cartesian evaluator currently supports min_step=0 only")
    if conflict_mode not in {"margin", "abstain"}:
        raise ValueError(f"Unsupported conflict_mode: {conflict_mode}")
    opposite_max_value = None if opposite_max is None else float(opposite_max)

    success_col = head_column("success", score_mode, predictor)
    failure_col = head_column("failure", score_mode, predictor)
    required = ["traj_id", "label", "prefix_step_idx", success_col, failure_col]
    missing = [col for col in required if col not in frame.columns]
    if missing:
        raise ValueError(f"Prediction table is missing required columns: {missing}")

    s_thr = np.asarray(success_thresholds, dtype=np.float64)
    f_thr = np.asarray(failure_thresholds, dtype=np.float64)
    shape = (len(s_thr), len(f_thr))
    counts = _empty_counts(shape)

    original_total = 0
    original_resolved = 0
    total_steps = 0

    for _, group in frame.groupby("traj_id", sort=False):
        group = group.sort_values("prefix_step_idx")
        label = int(group["label"].iloc[0])
        steps = group["prefix_step_idx"].to_numpy(dtype=np.int32)
        success_scores = group[success_col].to_numpy(dtype=np.float64)
        failure_scores = group[failure_col].to_numpy(dtype=np.float64)
        n_steps = int(len(group))
        saved_by_index = np.asarray([(steps > step).sum() for step in steps], dtype=np.int64)

        original_total += 1
        original_resolved += int(label == 1)
        total_steps += n_steps

        decided = np.zeros(shape, dtype=bool)
        decided_success = np.zeros(shape, dtype=bool)

        for idx, (success_score, failure_score) in enumerate(zip(success_scores, failure_scores)):
            if bool(decided.all()):
                break
            success_hit = success_score >= s_thr[:, None]
            failure_hit = failure_score >= f_thr[None, :]
            if opposite_max_value is not None:
                success_hit = success_hit & (failure_score < opposite_max_value)
                failure_hit = failure_hit & (success_score < opposite_max_value)
            if conflict_mode == "abstain":
                new_decision = (~decided) & (success_hit ^ failure_hit)
            else:
                new_decision = (~decided) & (success_hit | failure_hit)
            if not bool(new_decision.any()):
                continue

            if conflict_mode == "margin":
                success_margin = success_score - s_thr[:, None]
                failure_margin = failure_score - f_thr[None, :]
                choose_success = success_hit & (~failure_hit | (success_margin >= failure_margin))
            else:
                choose_success = success_hit & ~failure_hit
            new_success = new_decision & choose_success

            decided[new_decision] = True
            decided_success[new_success] = True
            counts["total_saved_steps"][new_decision] += int(saved_by_index[idx])

        decided_failure = decided & ~decided_success
        undecided = ~decided
        counts["decided_success"] += decided_success.astype(np.int64)
        counts["decided_failure"] += decided_failure.astype(np.int64)
        counts["undecided"] += undecided.astype(np.int64)
        if label == 1:
            counts["true_positives"] += decided_success.astype(np.int64)
            counts["false_negatives"] += decided_failure.astype(np.int64)
        else:
            counts["false_positives"] += decided_success.astype(np.int64)
            counts["true_negatives"] += decided_failure.astype(np.int64)

    rows: list[dict[str, Any]] = []
    original_rate = original_resolved / original_total if original_total else 0.0
    for i, success_thr in enumerate(success_thresholds):
        for j, failure_thr in enumerate(failure_thresholds):
            tp = int(counts["true_positives"][i, j])
            tn = int(counts["true_negatives"][i, j])
            fp = int(counts["false_positives"][i, j])
            fn = int(counts["false_negatives"][i, j])
            decided_success_count = int(counts["decided_success"][i, j])
            decided_failure_count = int(counts["decided_failure"][i, j])
            n_decided = decided_success_count + decided_failure_count
            adjusted_resolved = int(original_resolved - fn + fp)
            adjusted_rate = adjusted_resolved / original_total if original_total else 0.0
            name = _policy_name(
                score_mode=score_mode,
                predictor=predictor,
                success_thr=float(success_thr),
                failure_thr=float(failure_thr),
                min_step=min_step,
                consecutive=consecutive,
            )
            rows.append(
                {
                    "policy_name": name,
                    "name": name,
                    "predictor": predictor,
                    "score_mode": score_mode,
                    "policy_mode": "dual",
                    "success_thr": float(success_thr),
                    "failure_thr": float(failure_thr),
                    "min_step": int(min_step),
                    "consecutive": int(consecutive),
                    "original_total": int(original_total),
                    "original_resolved": int(original_resolved),
                    "original_resolve_rate": float(original_rate),
                    "decided_failure": decided_failure_count,
                    "decided_success": decided_success_count,
                    "undecided": int(counts["undecided"][i, j]),
                    "false_negatives": fn,
                    "true_negatives": tn,
                    "false_positives": fp,
                    "true_positives": tp,
                    "n_decided": int(n_decided),
                    "coverage_pct": 100.0 * n_decided / original_total if original_total else float("nan"),
                    "decision_accuracy_pct": 100.0 * (tp + tn) / n_decided if n_decided else float("nan"),
                    "precision_success_pct": 100.0 * tp / decided_success_count if decided_success_count else float("nan"),
                    "precision_failure_pct": 100.0 * tn / decided_failure_count if decided_failure_count else float("nan"),
                    "adjusted_resolved": adjusted_resolved,
                    "adjusted_resolve_rate": float(adjusted_rate),
                    "resolve_rate_drop_pp": 100.0 * (original_rate - adjusted_rate),
                    "resolve_rate_change_pp": 100.0 * (adjusted_rate - original_rate),
                    "pct_steps_saved": 100.0 * int(counts["total_saved_steps"][i, j]) / total_steps if total_steps else float("nan"),
                    "total_saved_steps": int(counts["total_saved_steps"][i, j]),
                    "total_steps": int(total_steps),
                }
            )
    return pd.DataFrame(rows)


def _add_selection_metadata(row: dict[str, Any], selected: dict[str, Any], *, fold_id: str, target: float) -> dict[str, Any]:
    out = dict(row)
    out["fold_id"] = fold_id
    out["test_model"] = fold_id
    out["target_valid_decision_accuracy"] = float(target)
    out["target_valid_decision_accuracy_pct"] = float(target) * 100.0
    out["selected_valid_policy_id"] = str(selected["policy_id"])
    out["selected_valid_decision_accuracy_pct"] = float(selected["decision_accuracy_pct"])
    out["selected_valid_step_save_pct"] = float(selected["pct_steps_saved"])
    out["selected_valid_resolve_rate_change_pp"] = float(selected["resolve_rate_change_pp"])
    out["selection_status"] = str(selected["selection_status"])
    return out


def _selected_threshold_summary(selected_frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for target, part in selected_frame.groupby("target_valid_decision_accuracy_pct", sort=True):
        rows.append(
            {
                "target_valid_decision_accuracy_pct": float(target),
                "folds": int(len(part)),
                "success_thr_min": float(part["success_thr"].min()),
                "success_thr_median": float(part["success_thr"].median()),
                "success_thr_max": float(part["success_thr"].max()),
                "failure_thr_min": float(part["failure_thr"].min()),
                "failure_thr_median": float(part["failure_thr"].median()),
                "failure_thr_max": float(part["failure_thr"].max()),
                "same_threshold_folds": int((part["success_thr"].round(10) == part["failure_thr"].round(10)).sum()),
                "valid_decision_accuracy_pct_mean": float(part["decision_accuracy_pct"].mean()),
                "valid_step_save_pct_mean": float(part["pct_steps_saved"].mean()),
            }
        )
    return pd.DataFrame(rows)


def _comparison_to_symmetric(sweep_dir: Path, aggregate_test: pd.DataFrame) -> pd.DataFrame:
    symmetric_path = sweep_dir.parent / "valid_accuracy_075_095" / "aggregate_test_metrics.csv"
    if not symmetric_path.exists():
        return pd.DataFrame()
    symmetric = pd.read_csv(symmetric_path)
    compare = aggregate_test.merge(
        symmetric,
        on="target_valid_decision_accuracy_pct",
        suffixes=("_cartesian", "_symmetric"),
        how="inner",
    )
    if compare.empty:
        return compare
    out = pd.DataFrame(
        {
            "target_valid_decision_accuracy_pct": compare["target_valid_decision_accuracy_pct"],
            "cartesian_step_save_pct": compare["step_save_pct_cartesian"],
            "symmetric_step_save_pct": compare["step_save_pct_symmetric"],
            "step_save_delta_pct": compare["step_save_pct_cartesian"] - compare["step_save_pct_symmetric"],
            "cartesian_decision_accuracy_pct": compare["decision_accuracy_pct_cartesian"],
            "symmetric_decision_accuracy_pct": compare["decision_accuracy_pct_symmetric"],
            "decision_accuracy_delta_pct": compare["decision_accuracy_pct_cartesian"]
            - compare["decision_accuracy_pct_symmetric"],
            "cartesian_resolve_change_pp": compare["resolve_rate_change_pp_cartesian"],
            "symmetric_resolve_change_pp": compare["resolve_rate_change_pp_symmetric"],
            "resolve_change_delta_pp": compare["resolve_rate_change_pp_cartesian"]
            - compare["resolve_rate_change_pp_symmetric"],
            "cartesian_mean_abs_resolve_change_pp": compare["mean_abs_resolve_rate_change_pp_cartesian"],
            "symmetric_mean_abs_resolve_change_pp": compare["mean_abs_resolve_rate_change_pp_symmetric"],
            "mean_abs_resolve_change_delta_pp": compare["mean_abs_resolve_rate_change_pp_cartesian"]
            - compare["mean_abs_resolve_rate_change_pp_symmetric"],
        }
    )
    return out


def _write_readme(
    sweep_dir: Path,
    *,
    run_dir: Path,
    aggregate_test: pd.DataFrame,
    selected_summary: pd.DataFrame,
    comparison: pd.DataFrame,
    completed_count: int,
    skipped_count: int,
    grid_size: int,
    elapsed_sec: float,
) -> None:
    lines = [
        "# LightGBM Cartesian Dual-Threshold Valid-Accuracy Sweep",
        "",
        f"- run_dir: `{run_dir}`",
        f"- sweep_dir: `{sweep_dir}`",
        f"- completed folds used: `{completed_count}`",
        f"- completed folds skipped by config: `{skipped_count}`",
        "- target valid decision accuracy: `0.75` to `0.95`",
        "- policy mode: `dual`",
        "- threshold mode: `cartesian` (`success_thr` and `failure_thr` are selected independently)",
        f"- valid candidate policies per fold: `{grid_size}`",
        "- valid resolve-drop guard: `disabled`",
        "- Selection uses valid metrics only; the selected policy is then applied unchanged to test.",
        f"- elapsed seconds: `{elapsed_sec:.1f}`",
        "",
        "## Outputs",
        "",
        "- `valid_policy_candidate_grid.csv`: all cartesian candidate policies evaluated on valid.",
        "- `per_fold_selected_policies.csv`: valid-selected policy for each fold and target accuracy.",
        "- `selected_threshold_summary.csv`: selected success/failure threshold ranges by target.",
        "- `per_fold_test_metrics.csv`: held-out test metrics after applying each valid-selected policy.",
        "- `aggregate_test_metrics.csv`: count-weighted test aggregate by valid accuracy target.",
        "- `cartesian_vs_symmetric_test_metrics.csv`: comparison with the symmetric sweep when available.",
        "",
        "## Aggregate Test Frontier",
        "",
    ]
    display_rows = []
    for row in aggregate_test.to_dict("records"):
        display_rows.append(
            {
                "valid_acc_target": f"{float(row['target_valid_decision_accuracy_pct']):.0f}",
                "test_save_pct": f"{float(row['step_save_pct']):.2f}",
                "test_acc_pct": f"{float(row['decision_accuracy_pct']):.2f}",
                "test_coverage_pct": f"{float(row['coverage_pct']):.2f}",
                "resolve_change_pp": f"{float(row['resolve_rate_change_pp']):+.2f}",
                "mean_abs_resolve_change_pp": f"{float(row['mean_abs_resolve_rate_change_pp']):.2f}",
                "fn": int(row["false_negatives"]),
                "fp": int(row["false_positives"]),
                "decided": int(row["decided_trajectories"]),
            }
        )
    lines.extend(
        _markdown_table(
            display_rows,
            [
                "valid_acc_target",
                "test_save_pct",
                "test_acc_pct",
                "test_coverage_pct",
                "resolve_change_pp",
                "mean_abs_resolve_change_pp",
                "fn",
                "fp",
                "decided",
            ],
        )
    )
    lines.extend(["", "## Selected Threshold Summary", ""])
    threshold_rows = []
    for row in selected_summary.to_dict("records"):
        threshold_rows.append(
            {
                "valid_acc_target": f"{float(row['target_valid_decision_accuracy_pct']):.0f}",
                "success_thr": (
                    f"{float(row['success_thr_min']):.2f}/"
                    f"{float(row['success_thr_median']):.2f}/"
                    f"{float(row['success_thr_max']):.2f}"
                ),
                "failure_thr": (
                    f"{float(row['failure_thr_min']):.2f}/"
                    f"{float(row['failure_thr_median']):.2f}/"
                    f"{float(row['failure_thr_max']):.2f}"
                ),
                "same_thr_folds": int(row["same_threshold_folds"]),
                "mean_valid_save_pct": f"{float(row['valid_step_save_pct_mean']):.2f}",
            }
        )
    lines.extend(
        _markdown_table(
            threshold_rows,
            ["valid_acc_target", "success_thr", "failure_thr", "same_thr_folds", "mean_valid_save_pct"],
        )
    )
    if not comparison.empty:
        lines.extend(["", "## Delta vs Symmetric Sweep", ""])
        compare_rows = []
        for row in comparison.to_dict("records"):
            compare_rows.append(
                {
                    "valid_acc_target": f"{float(row['target_valid_decision_accuracy_pct']):.0f}",
                    "save_delta_pct": f"{float(row['step_save_delta_pct']):+.2f}",
                    "acc_delta_pct": f"{float(row['decision_accuracy_delta_pct']):+.2f}",
                    "resolve_delta_pp": f"{float(row['resolve_change_delta_pp']):+.2f}",
                    "mean_abs_delta_pp": f"{float(row['mean_abs_resolve_change_delta_pp']):+.2f}",
                }
            )
        lines.extend(
            _markdown_table(
                compare_rows,
                ["valid_acc_target", "save_delta_pct", "acc_delta_pct", "resolve_delta_pp", "mean_abs_delta_pp"],
            )
        )
    lines.extend(
        [
            "",
            "## Plots",
            "",
            "- `aggregate_test_frontier.png`",
            "- `aggregate_test_resolve_change.png`",
            "- `per_fold_test_step_saving.png`",
            "- `per_fold_test_decision_accuracy.png`",
            "- `per_fold_test_resolve_change.png`",
        ]
    )
    (sweep_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_fast_cartesian_sweep(
    *,
    config: Path,
    output_dir: Path | None,
    sweep_name: str,
    fold_limit: int | None = None,
) -> dict[str, Any]:
    started = time.time()
    cfg = load_earlyeval_config(config)
    out = ensure_dir(output_dir or _default_output_dir(cfg, cfg.run_id))
    run_dir = out / "lightgbm_main"
    sweep_dir = ensure_dir(run_dir / "policy_sweeps" / sweep_name)

    main_cfg = cfg.payload.get("main_model") or {}
    sweep_cfg = cfg.payload.get("policy_sweep") or {}
    predictor_values = sweep_cfg.get("prefix_models") or sweep_cfg.get("predictors") or ["I_LightGBM_Dense_AF"]
    predictor = str(predictor_values[0] if isinstance(predictor_values, list) else predictor_values)
    score_mode_values = sweep_cfg.get("score_modes", main_cfg.get("score_modes", [main_cfg.get("score_mode", "calibrated")]))
    score_mode = str(score_mode_values[0] if isinstance(score_mode_values, list) else score_mode_values)
    probability_thresholds = _float_sequence(
        sweep_cfg.get("candidate_probability_thresholds"),
        {"start": 0.30, "stop": 0.99, "step": 0.01},
    )
    success_thresholds = _float_sequence(sweep_cfg.get("candidate_success_thresholds"), probability_thresholds)
    failure_thresholds = _float_sequence(sweep_cfg.get("candidate_failure_thresholds"), probability_thresholds)
    targets = _float_sequence(
        sweep_cfg.get("target_valid_decision_accuracy"),
        {"start": 0.75, "stop": 0.95, "step": 0.01},
    )
    min_step_values = [int(value) for value in sweep_cfg.get("policy_min_steps", [0])]
    consecutive_values = [int(value) for value in sweep_cfg.get("consecutive", [1])]
    if min_step_values != [0] or consecutive_values != [1]:
        raise ValueError("fast cartesian sweep expects policy_min_steps=[0] and consecutive=[1]")
    max_valid_abs_drop_pp_raw = sweep_cfg.get("max_valid_abs_drop_pp", None)
    max_valid_abs_drop_pp = None if max_valid_abs_drop_pp_raw is None else float(max_valid_abs_drop_pp_raw)
    fallback_min_save_pct = float(sweep_cfg.get("fallback_min_save_pct", 0.0))

    excluded_models = _excluded_models_from_config(cfg)
    all_completed = sorted(path.parent for path in (run_dir / "folds").glob("*/_SUCCESS"))
    skipped_completed = [fold_dir for fold_dir in all_completed if fold_dir.name in excluded_models]
    completed = [fold_dir for fold_dir in all_completed if fold_dir.name not in excluded_models]
    if fold_limit is not None:
        completed = completed[: int(fold_limit)]
    if not completed:
        raise FileNotFoundError(f"No completed LightGBM folds found under {run_dir / 'folds'}")

    valid_grid_rows: list[dict[str, Any]] = []
    selected_rows: list[dict[str, Any]] = []
    valid_summary_rows: list[dict[str, Any]] = []
    test_summary_rows: list[dict[str, Any]] = []

    for fold_index, fold_dir in enumerate(completed, start=1):
        fold_started = time.time()
        fold_id = fold_dir.name
        print(f"[fast-cartesian-sweep] fold {fold_index}/{len(completed)}: {fold_id}", flush=True)
        valid_frame = pd.read_parquet(fold_dir / "valid_predictions_safe_stop.parquet")
        test_frame = pd.read_parquet(fold_dir / "test_predictions_safe_stop.parquet")
        fold_valid_grid = evaluate_cartesian_grid(
            valid_frame,
            predictor=predictor,
            score_mode=score_mode,
            success_thresholds=success_thresholds,
            failure_thresholds=failure_thresholds,
        )
        fold_valid_grid["fold_id"] = fold_id
        fold_valid_grid["test_model"] = fold_id
        fold_valid_grid["policy_id"] = fold_valid_grid["policy_name"]
        fold_valid_grid["valid_abs_drop_pp"] = fold_valid_grid["resolve_rate_drop_pp"].astype(float).abs()
        fold_valid_grid["decision_accuracy_fraction"] = fold_valid_grid["decision_accuracy_pct"].astype(float) / 100.0
        valid_grid_rows.extend(fold_valid_grid.to_dict("records"))

        selected_for_fold: list[dict[str, Any]] = []
        for target in targets:
            selected = _select_policy_for_valid_target(
                fold_valid_grid,
                target,
                max_valid_abs_drop_pp=max_valid_abs_drop_pp,
                fallback_min_save_pct=fallback_min_save_pct,
            )
            selected_for_fold.append(selected)
            selected_rows.append(selected)
            valid_summary_rows.append(_add_selection_metadata(selected, selected, fold_id=fold_id, target=target))

        selected_frame = pd.DataFrame(selected_for_fold)
        unique_success = sorted({round(float(value), 6) for value in selected_frame["success_thr"]})
        unique_failure = sorted({round(float(value), 6) for value in selected_frame["failure_thr"]})
        fold_test_grid = evaluate_cartesian_grid(
            test_frame,
            predictor=predictor,
            score_mode=score_mode,
            success_thresholds=unique_success,
            failure_thresholds=unique_failure,
        )
        keyed_test = {
            (round(float(row["success_thr"]), 6), round(float(row["failure_thr"]), 6)): row
            for row in fold_test_grid.to_dict("records")
        }
        for selected in selected_for_fold:
            target = float(selected["target_valid_decision_accuracy"])
            key = (round(float(selected["success_thr"]), 6), round(float(selected["failure_thr"]), 6))
            test_row = keyed_test[key]
            test_summary_rows.append(_add_selection_metadata(test_row, selected, fold_id=fold_id, target=target))
        print(f"[fast-cartesian-sweep] fold done in {time.time() - fold_started:.1f}s", flush=True)

    valid_grid = pd.DataFrame(valid_grid_rows)
    selected_frame = pd.DataFrame(selected_rows)
    valid_summary = pd.DataFrame(valid_summary_rows)
    test_summary = pd.DataFrame(test_summary_rows)
    aggregate_valid = _aggregate_policy_sweep_by_target(valid_summary)
    aggregate_test = _aggregate_policy_sweep_by_target(test_summary)
    selected_summary = _selected_threshold_summary(selected_frame)
    comparison = _comparison_to_symmetric(sweep_dir, aggregate_test)

    write_table(valid_grid, sweep_dir / "valid_policy_candidate_grid.csv")
    write_table(selected_frame, sweep_dir / "per_fold_selected_policies.csv")
    write_table(valid_summary, sweep_dir / "per_fold_valid_metrics.csv")
    write_table(test_summary, sweep_dir / "per_fold_test_metrics.csv")
    write_table(aggregate_valid, sweep_dir / "aggregate_valid_metrics.csv")
    write_table(aggregate_test, sweep_dir / "aggregate_test_metrics.csv")
    write_table(selected_summary, sweep_dir / "selected_threshold_summary.csv")
    if not comparison.empty:
        write_table(comparison, sweep_dir / "cartesian_vs_symmetric_test_metrics.csv")
    plots = _write_lightgbm_policy_sweep_plots(sweep_dir, aggregate_test, test_summary)

    elapsed = time.time() - started
    manifest = {
        "sweep_dir": str(sweep_dir),
        "run_dir": str(run_dir),
        "completed_folds": len(completed),
        "skipped_completed_folds": len(skipped_completed),
        "predictor": predictor,
        "score_mode": score_mode,
        "threshold_mode": "cartesian",
        "success_threshold_count": len(success_thresholds),
        "failure_threshold_count": len(failure_thresholds),
        "candidate_policies_per_fold": len(success_thresholds) * len(failure_thresholds),
        "target_valid_decision_accuracy": targets,
        "elapsed_sec": elapsed,
        "plots": plots,
    }
    (sweep_dir / "sweep_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    _write_readme(
        sweep_dir,
        run_dir=run_dir,
        aggregate_test=aggregate_test,
        selected_summary=selected_summary,
        comparison=comparison,
        completed_count=len(completed),
        skipped_count=len(skipped_completed),
        grid_size=len(success_thresholds) * len(failure_thresholds),
        elapsed_sec=elapsed,
    )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a fast cartesian dual-threshold valid-accuracy sweep.")
    parser.add_argument("--config", type=Path, default=Path("configs/earlyeval.yaml"))
    parser.add_argument("--output-dir", type=Path, default=Path("paper/experiments/earlyeval_lightgbm"))
    parser.add_argument("--sweep-name", default="valid_accuracy_075_095_cartesian")
    parser.add_argument("--fold-limit", type=int, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = run_fast_cartesian_sweep(
        config=args.config,
        output_dir=args.output_dir,
        sweep_name=args.sweep_name,
        fold_limit=args.fold_limit,
    )
    print(json.dumps(manifest, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
