from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import pandas as pd

from earlyeval.core.io import ensure_dir, write_table
from earlyeval.experiments.fast_cartesian_policy_sweep import evaluate_cartesian_grid
from earlyeval.experiments.paper_pipeline import (
    _aggregate_policy_sweep_by_target,
    _default_output_dir,
    _excluded_models_from_config,
    _float_sequence,
    _markdown_table,
    load_earlyeval_config,
)


def _prepare_grid(frame: pd.DataFrame, *, fold_id: str, strategy_name: str, target: float | None = None) -> pd.DataFrame:
    out = frame.copy()
    out["fold_id"] = fold_id
    out["test_model"] = fold_id
    out["policy_id"] = out["policy_name"]
    if target is not None:
        out["target_valid_decision_accuracy"] = float(target)
        out["target_valid_decision_accuracy_pct"] = float(target) * 100.0
    out["selection_strategy"] = strategy_name
    return out


def _success_precision_pass(frame: pd.DataFrame, guard: float | None) -> pd.Series:
    if guard is None:
        return pd.Series(True, index=frame.index)
    precision = pd.to_numeric(frame["precision_success_pct"], errors="coerce")
    decided_success = pd.to_numeric(frame["decided_success"], errors="coerce").fillna(0)
    return (decided_success <= 0) | precision.ge(float(guard))


def _select_policy(
    valid_grid: pd.DataFrame,
    *,
    target_accuracy: float,
    success_precision_guard: float | None,
    fallback_min_save_pct: float,
) -> dict[str, Any]:
    work = valid_grid.copy()
    work["valid_abs_change_pp"] = work["resolve_rate_change_pp"].astype(float).abs()
    work["decision_accuracy_fraction"] = work["decision_accuracy_pct"].fillna(-1.0).astype(float) / 100.0
    work["pct_steps_saved_for_sort"] = work["pct_steps_saved"].fillna(0.0).astype(float)

    strict_mask = (
        work["decision_accuracy_fraction"].ge(float(target_accuracy))
        & work["pct_steps_saved_for_sort"].gt(0.0)
        & _success_precision_pass(work, success_precision_guard)
    )
    strict = work[strict_mask].copy()
    if not strict.empty:
        chosen = strict.sort_values(
            ["pct_steps_saved_for_sort", "valid_abs_change_pp", "decision_accuracy_fraction"],
            ascending=[False, True, False],
        ).iloc[0]
        status = "valid_accuracy_success_precision_pass" if success_precision_guard is not None else "valid_accuracy_pass"
    else:
        fallback = work[
            work["decision_accuracy_fraction"].ge(float(target_accuracy))
            & work["pct_steps_saved_for_sort"].ge(float(fallback_min_save_pct))
        ].copy()
        if fallback.empty:
            fallback = work[work["pct_steps_saved_for_sort"].gt(0.0)].copy()
        if fallback.empty:
            fallback = work.copy()
        chosen = fallback.sort_values(
            ["decision_accuracy_fraction", "valid_abs_change_pp", "pct_steps_saved_for_sort"],
            ascending=[False, True, False],
        ).iloc[0]
        status = "fallback_without_success_precision_guard"
    row = chosen.to_dict()
    row["target_valid_decision_accuracy"] = float(target_accuracy)
    row["target_valid_decision_accuracy_pct"] = float(target_accuracy) * 100.0
    row["selection_status"] = status
    return row


def _attach_selection(row: dict[str, Any], selected: dict[str, Any], *, fold_id: str, target: float, strategy_name: str) -> dict[str, Any]:
    out = dict(row)
    out["fold_id"] = fold_id
    out["test_model"] = fold_id
    out["target_valid_decision_accuracy"] = float(target)
    out["target_valid_decision_accuracy_pct"] = float(target) * 100.0
    out["selected_valid_policy_id"] = str(selected["policy_id"])
    out["selected_valid_decision_accuracy_pct"] = float(selected["decision_accuracy_pct"])
    out["selected_valid_step_save_pct"] = float(selected["pct_steps_saved"])
    out["selected_valid_resolve_rate_change_pp"] = float(selected["resolve_rate_change_pp"])
    out["selected_valid_precision_success_pct"] = selected.get("precision_success_pct")
    out["selection_status"] = str(selected["selection_status"])
    out["selection_strategy"] = strategy_name
    return out


def _aggregate_by_strategy_and_target(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (strategy, target), part in frame.groupby(["selection_strategy", "target_valid_decision_accuracy"], sort=True):
        row = _aggregate_policy_sweep_by_target(part).iloc[0].to_dict()
        row["selection_strategy"] = str(strategy)
        row["target_valid_decision_accuracy"] = float(target)
        row["target_valid_decision_accuracy_pct"] = float(target) * 100.0
        rows.append(row)
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows).sort_values(["selection_strategy", "target_valid_decision_accuracy"])
    leading = ["selection_strategy", "target_valid_decision_accuracy", "target_valid_decision_accuracy_pct"]
    return out[leading + [col for col in out.columns if col not in leading]]


def _write_readme(out_dir: Path, aggregate_test: pd.DataFrame, manifest: dict[str, Any]) -> None:
    lines = [
        "# Policy Selection Rule Diagnostic",
        "",
        "This diagnostic evaluates valid-only selection rules across all completed folds.",
        "",
        f"- completed folds used: `{manifest['completed_folds']}`",
        f"- skipped completed folds: `{manifest['skipped_completed_folds']}`",
        f"- target range: `{manifest['target_valid_decision_accuracy'][0]}` to `{manifest['target_valid_decision_accuracy'][-1]}`",
        "- candidate grid: cartesian dual thresholds, 0.30..0.99 for both success and failure.",
        "",
        "## Strategies",
        "",
        "- `margin_current`: original conflict handling; when both heads hit, choose the larger margin.",
        "- `margin_success_precision_98`: original conflict handling plus `valid_precision_success_pct >= 98` during selection.",
        "- `margin_success_precision_100`: original conflict handling plus `valid_precision_success_pct == 100` during selection.",
        "- `abstain_current`: if both heads hit at a prefix, treat the prefix as uncertain, skip it, and continue.",
        "- `opposite_lt_0p5`: a head can stop only when its own score reaches threshold and the opposite head is below 0.5.",
        "",
        "## Target 95 Summary",
        "",
    ]
    target95 = aggregate_test[aggregate_test["target_valid_decision_accuracy_pct"].round(6).eq(95.0)]
    rows = []
    for row in target95.to_dict("records"):
        rows.append(
            {
                "strategy": row["selection_strategy"],
                "save_pct": f"{float(row['step_save_pct']):.2f}",
                "acc_pct": f"{float(row['decision_accuracy_pct']):.2f}",
                "change_pp": f"{float(row['resolve_rate_change_pp']):+.2f}",
                "mean_abs_pp": f"{float(row['mean_abs_resolve_rate_change_pp']):.2f}",
                "fn": int(row["false_negatives"]),
                "fp": int(row["false_positives"]),
            }
        )
    lines.extend(_markdown_table(rows, ["strategy", "save_pct", "acc_pct", "change_pp", "mean_abs_pp", "fn", "fp"]))
    lines.extend(["", "## Best Tradeoffs By Strategy", ""])
    best_rows = []
    for strategy, part in aggregate_test.groupby("selection_strategy", sort=True):
        candidates = part[part["decision_accuracy_pct"].ge(90.0)].copy()
        if candidates.empty:
            candidates = part.copy()
        row = candidates.sort_values(
            ["mean_abs_resolve_rate_change_pp", "step_save_pct"],
            ascending=[True, False],
        ).iloc[0]
        best_rows.append(
            {
                "strategy": strategy,
                "target": f"{float(row['target_valid_decision_accuracy_pct']):.0f}",
                "save_pct": f"{float(row['step_save_pct']):.2f}",
                "acc_pct": f"{float(row['decision_accuracy_pct']):.2f}",
                "change_pp": f"{float(row['resolve_rate_change_pp']):+.2f}",
                "mean_abs_pp": f"{float(row['mean_abs_resolve_rate_change_pp']):.2f}",
            }
        )
    lines.extend(_markdown_table(best_rows, ["strategy", "target", "save_pct", "acc_pct", "change_pp", "mean_abs_pp"]))
    (out_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_selection_rule_diagnostic(
    *,
    config: Path,
    output_dir: Path | None,
    out_subdir: str,
    fold_limit: int | None = None,
) -> dict[str, Any]:
    started = time.time()
    cfg = load_earlyeval_config(config)
    root = ensure_dir(output_dir or _default_output_dir(cfg, cfg.run_id))
    run_dir = root / "lightgbm_main"
    out_dir = ensure_dir(run_dir / "policy_sweeps" / out_subdir)

    sweep_cfg = cfg.payload.get("policy_sweep") or {}
    predictor_values = sweep_cfg.get("prefix_models") or sweep_cfg.get("predictors") or ["I_LightGBM_Dense_AF"]
    predictor = str(predictor_values[0] if isinstance(predictor_values, list) else predictor_values)
    score_modes = sweep_cfg.get("score_modes", ["calibrated"])
    score_mode = str(score_modes[0] if isinstance(score_modes, list) else score_modes)
    thresholds = _float_sequence(
        sweep_cfg.get("candidate_probability_thresholds"),
        {"start": 0.30, "stop": 0.99, "step": 0.01},
    )
    targets = _float_sequence(
        sweep_cfg.get("target_valid_decision_accuracy"),
        {"start": 0.75, "stop": 0.95, "step": 0.01},
    )
    fallback_min_save_pct = float(sweep_cfg.get("fallback_min_save_pct", 0.0))

    strategies = [
        {"name": "margin_current", "conflict_mode": "margin", "success_precision_guard": None},
        {"name": "margin_success_precision_98", "conflict_mode": "margin", "success_precision_guard": 98.0},
        {"name": "margin_success_precision_100", "conflict_mode": "margin", "success_precision_guard": 100.0},
        {"name": "abstain_current", "conflict_mode": "abstain", "success_precision_guard": None},
        {"name": "opposite_lt_0p5", "conflict_mode": "abstain", "success_precision_guard": None, "opposite_max": 0.5},
    ]

    excluded_models = _excluded_models_from_config(cfg)
    all_completed = sorted(path.parent for path in (run_dir / "folds").glob("*/_SUCCESS"))
    skipped_completed = [fold_dir for fold_dir in all_completed if fold_dir.name in excluded_models]
    completed = [fold_dir for fold_dir in all_completed if fold_dir.name not in excluded_models]
    if fold_limit is not None:
        completed = completed[: int(fold_limit)]

    selected_rows: list[dict[str, Any]] = []
    valid_summary_rows: list[dict[str, Any]] = []
    test_summary_rows: list[dict[str, Any]] = []

    for fold_index, fold_dir in enumerate(completed, start=1):
        fold_id = fold_dir.name
        print(f"[selection-rule-diagnostic] fold {fold_index}/{len(completed)}: {fold_id}", flush=True)
        valid_frame = pd.read_parquet(fold_dir / "valid_predictions_safe_stop.parquet")
        test_frame = pd.read_parquet(fold_dir / "test_predictions_safe_stop.parquet")
        test_grid_cache: dict[str, pd.DataFrame] = {}
        for strategy in strategies:
            valid_grid = evaluate_cartesian_grid(
                valid_frame,
                predictor=predictor,
                score_mode=score_mode,
                success_thresholds=thresholds,
                failure_thresholds=thresholds,
                conflict_mode=strategy["conflict_mode"],
                opposite_max=strategy.get("opposite_max"),
            )
            valid_grid = _prepare_grid(valid_grid, fold_id=fold_id, strategy_name=strategy["name"])
            chosen_for_strategy: list[dict[str, Any]] = []
            for target in targets:
                selected = _select_policy(
                    valid_grid,
                    target_accuracy=target,
                    success_precision_guard=strategy["success_precision_guard"],
                    fallback_min_save_pct=fallback_min_save_pct,
                )
                selected["selection_strategy"] = strategy["name"]
                selected_rows.append(selected)
                valid_summary_rows.append(
                    _attach_selection(selected, selected, fold_id=fold_id, target=target, strategy_name=strategy["name"])
                )
                chosen_for_strategy.append(selected)

            unique_success = sorted({round(float(row["success_thr"]), 6) for row in chosen_for_strategy})
            unique_failure = sorted({round(float(row["failure_thr"]), 6) for row in chosen_for_strategy})
            cache_key = f"{strategy['conflict_mode']}::{unique_success}::{unique_failure}"
            if cache_key not in test_grid_cache:
                test_grid_cache[cache_key] = evaluate_cartesian_grid(
                    test_frame,
                    predictor=predictor,
                    score_mode=score_mode,
                    success_thresholds=unique_success,
                    failure_thresholds=unique_failure,
                    conflict_mode=strategy["conflict_mode"],
                    opposite_max=strategy.get("opposite_max"),
                )
            test_lookup = {
                (round(float(row["success_thr"]), 6), round(float(row["failure_thr"]), 6)): row
                for row in test_grid_cache[cache_key].to_dict("records")
            }
            for selected in chosen_for_strategy:
                target = float(selected["target_valid_decision_accuracy"])
                key = (round(float(selected["success_thr"]), 6), round(float(selected["failure_thr"]), 6))
                test_summary_rows.append(
                    _attach_selection(test_lookup[key], selected, fold_id=fold_id, target=target, strategy_name=strategy["name"])
                )

    selected_frame = pd.DataFrame(selected_rows)
    valid_summary = pd.DataFrame(valid_summary_rows)
    test_summary = pd.DataFrame(test_summary_rows)
    aggregate_valid = _aggregate_by_strategy_and_target(valid_summary)
    aggregate_test = _aggregate_by_strategy_and_target(test_summary)

    write_table(selected_frame, out_dir / "per_fold_selected_policies.csv")
    write_table(valid_summary, out_dir / "per_fold_valid_metrics.csv")
    write_table(test_summary, out_dir / "per_fold_test_metrics.csv")
    write_table(aggregate_valid, out_dir / "aggregate_valid_metrics.csv")
    write_table(aggregate_test, out_dir / "aggregate_test_metrics.csv")

    manifest = {
        "output_dir": str(out_dir),
        "completed_folds": len(completed),
        "skipped_completed_folds": len(skipped_completed),
        "threshold_count": len(thresholds),
        "candidate_policies_per_fold": len(thresholds) * len(thresholds),
        "target_valid_decision_accuracy": targets,
        "strategies": strategies,
        "elapsed_sec": time.time() - started,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    _write_readme(out_dir, aggregate_test, manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate valid-only policy selection rules across folds.")
    parser.add_argument("--config", type=Path, default=Path("configs/earlyeval.yaml"))
    parser.add_argument("--output-dir", type=Path, default=Path("paper/experiments/earlyeval_lightgbm"))
    parser.add_argument("--out-subdir", default="selection_rule_diagnostic")
    parser.add_argument("--fold-limit", type=int, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = run_selection_rule_diagnostic(
        config=args.config,
        output_dir=args.output_dir,
        out_subdir=args.out_subdir,
        fold_limit=args.fold_limit,
    )
    print(json.dumps(manifest, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
