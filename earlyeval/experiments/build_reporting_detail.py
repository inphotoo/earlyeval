from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from earlyeval.core.contracts import PolicySpec
from earlyeval.policies.safe_stop import apply_policy


ROOT = Path("paper/experiments/earlyeval_lightgbm")
OUT = ROOT / "reporting_detail"
TARGETS = [round(x / 100.0, 2) for x in range(75, 96)]


def _ensure_out() -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    return OUT


def _float(value: Any) -> float:
    if isinstance(value, str) and value.lower() == "inf":
        return float("inf")
    if pd.isna(value):
        return float("nan")
    return float(value)


def _format_thr(value: Any) -> str:
    numeric = _float(value)
    if math.isinf(numeric):
        return "inf"
    return f"{numeric:.2f}"


def _pct_from_legacy(frame: pd.DataFrame, col: str, pct_col: str) -> pd.Series:
    if pct_col in frame.columns:
        return pd.to_numeric(frame[pct_col], errors="coerce")
    values = pd.to_numeric(frame[col], errors="coerce")
    return values * 100.0


def _actual_change_pp(frame: pd.DataFrame) -> pd.Series:
    total = pd.to_numeric(frame["original_total"], errors="coerce").replace(0, np.nan)
    fp = pd.to_numeric(frame["false_positives"], errors="coerce").fillna(0)
    fn = pd.to_numeric(frame["false_negatives"], errors="coerce").fillna(0)
    return (fp - fn) * 100.0 / total


def _normalize_summary_columns(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    if "coverage_pct" not in out.columns and "coverage" in out.columns:
        out["coverage_pct"] = _pct_from_legacy(out, "coverage", "coverage_pct")
    if "decision_accuracy_pct" not in out.columns and "decision_accuracy" in out.columns:
        out["decision_accuracy_pct"] = _pct_from_legacy(out, "decision_accuracy", "decision_accuracy_pct")
    if "step_save_pct" not in out.columns and "pct_steps_saved" in out.columns:
        out["step_save_pct"] = pd.to_numeric(out["pct_steps_saved"], errors="coerce")
    if "resolve_rate_change_pp" not in out.columns and {"false_positives", "false_negatives", "original_total"}.issubset(out.columns):
        out["resolve_rate_change_pp"] = _actual_change_pp(out)
    elif {"false_positives", "false_negatives", "original_total"}.issubset(out.columns):
        out["resolve_rate_change_pp"] = _actual_change_pp(out)
    if "original_resolve_rate_pct" not in out.columns and "original_resolve_rate" in out.columns:
        out["original_resolve_rate_pct"] = pd.to_numeric(out["original_resolve_rate"], errors="coerce") * 100.0
    if "adjusted_resolve_rate_pct" not in out.columns and "adjusted_resolve_rate" in out.columns:
        out["adjusted_resolve_rate_pct"] = pd.to_numeric(out["adjusted_resolve_rate"], errors="coerce") * 100.0
    return out


def _policy_from_row(row: pd.Series) -> PolicySpec:
    predictor = str(row.get("predictor", row.get("prefix_model")))
    success_thr = _float(row["success_thr"])
    failure_thr = _float(row["failure_thr"])
    name = "__".join(
        [
            str(row.get("score_mode", "calibrated")),
            predictor,
            str(row.get("policy_mode", "dual")),
            f"s{_format_thr(success_thr)}",
            f"f{_format_thr(failure_thr)}",
            f"min{int(row.get('min_step', 0))}",
            f"k{int(row.get('consecutive', 1))}",
        ]
    )
    return PolicySpec(
        name=name,
        predictor=predictor,
        score_mode=str(row.get("score_mode", "calibrated")),
        policy_mode=str(row.get("policy_mode", "dual")),
        success_thr=success_thr,
        failure_thr=failure_thr,
        min_step=int(row.get("min_step", 0)),
        consecutive=int(row.get("consecutive", 1)),
    )


def _select_policy_for_target(valid_grid: pd.DataFrame, target: float) -> tuple[pd.Series, str]:
    work = _normalize_summary_columns(valid_grid)
    work["decision_accuracy_fraction"] = pd.to_numeric(work["decision_accuracy_pct"], errors="coerce").fillna(-100.0) / 100.0
    work["step_save_for_sort"] = pd.to_numeric(work["step_save_pct"], errors="coerce").fillna(0.0)
    work["valid_abs_change_pp"] = _actual_change_pp(work).abs()
    strict = work[(work["decision_accuracy_fraction"] >= target) & (work["step_save_for_sort"] > 0.0)].copy()
    if not strict.empty:
        chosen = strict.sort_values(
            ["step_save_for_sort", "valid_abs_change_pp", "decision_accuracy_fraction"],
            ascending=[False, True, False],
        ).iloc[0]
        return chosen, "valid_accuracy_pass"
    fallback = work[work["step_save_for_sort"] > 0.0].copy()
    if fallback.empty:
        fallback = work.copy()
    chosen = fallback.sort_values(
        ["decision_accuracy_fraction", "valid_abs_change_pp", "step_save_for_sort"],
        ascending=[False, True, False],
    ).iloc[0]
    return chosen, "fallback_highest_valid_accuracy"


def _summary_with_meta(summary: pd.DataFrame, **meta: Any) -> dict[str, Any]:
    row = summary.iloc[0].to_dict()
    for key, value in meta.items():
        row[key] = value
    return row


def _evaluate_selected_frontier(
    valid_grid: pd.DataFrame,
    test_frame: pd.DataFrame,
    *,
    meta: dict[str, Any],
    predictor_filter: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    grid = valid_grid.copy()
    if predictor_filter is not None:
        grid = grid[grid["prefix_model"].astype(str) == predictor_filter].copy()
    selected_rows: list[dict[str, Any]] = []
    test_rows: list[dict[str, Any]] = []
    cache: dict[str, pd.DataFrame] = {}
    for target in TARGETS:
        selected, status = _select_policy_for_target(grid, target)
        policy = _policy_from_row(selected)
        key = policy.name
        if key not in cache:
            _, summary, _ = apply_policy(test_frame, policy)
            cache[key] = summary
        valid_norm = _normalize_summary_columns(pd.DataFrame([selected])).iloc[0]
        selected_rows.append(
            {
                **meta,
                "target_valid_decision_accuracy": target,
                "target_valid_decision_accuracy_pct": target * 100.0,
                "selection_status": status,
                "selected_policy_name": policy.name,
                "selected_score_mode": policy.score_mode,
                "selected_predictor": policy.predictor,
                "selected_policy_mode": policy.policy_mode,
                "selected_success_thr": _format_thr(policy.success_thr),
                "selected_failure_thr": _format_thr(policy.failure_thr),
                "selected_valid_save_pct": float(valid_norm["step_save_pct"]),
                "selected_valid_decision_accuracy_pct": float(valid_norm["decision_accuracy_pct"]),
                "selected_valid_resolve_change_pp": float(valid_norm["resolve_rate_change_pp"]),
            }
        )
        test_rows.append(
            _summary_with_meta(
                cache[key],
                **meta,
                target_valid_decision_accuracy=target,
                target_valid_decision_accuracy_pct=target * 100.0,
                selection_status=status,
                selected_policy_name=policy.name,
                selected_score_mode=policy.score_mode,
                selected_predictor=policy.predictor,
                selected_policy_mode=policy.policy_mode,
                selected_success_thr=_format_thr(policy.success_thr),
                selected_failure_thr=_format_thr(policy.failure_thr),
            )
        )
    return pd.DataFrame(selected_rows), _normalize_summary_columns(pd.DataFrame(test_rows))


def _aggregate_rows(frame: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if frame.empty:
        return pd.DataFrame()
    norm = _normalize_summary_columns(frame)
    groups = [((), norm)] if not group_cols else norm.groupby(group_cols, sort=True, dropna=False)
    for keys, part in groups:
        if not isinstance(keys, tuple):
            keys = (keys,)
        total = float(pd.to_numeric(part["original_total"], errors="coerce").sum())
        total_steps = float(pd.to_numeric(part["total_steps"], errors="coerce").sum())
        saved_steps = float(pd.to_numeric(part["total_saved_steps"], errors="coerce").sum())
        decided = float(pd.to_numeric(part["n_decided"], errors="coerce").sum())
        original_resolved = float(pd.to_numeric(part["original_resolved"], errors="coerce").sum())
        adjusted_resolved = float(pd.to_numeric(part["adjusted_resolved"], errors="coerce").sum())
        fp = float(pd.to_numeric(part["false_positives"], errors="coerce").fillna(0).sum())
        fn = float(pd.to_numeric(part["false_negatives"], errors="coerce").fillna(0).sum())
        tp = float(pd.to_numeric(part["true_positives"], errors="coerce").fillna(0).sum())
        tn = float(pd.to_numeric(part["true_negatives"], errors="coerce").fillna(0).sum())
        decided_success = (
            float(pd.to_numeric(part["decided_success"], errors="coerce").fillna(0).sum())
            if "decided_success" in part.columns
            else tp + fp
        )
        decided_failure = (
            float(pd.to_numeric(part["decided_failure"], errors="coerce").fillna(0).sum())
            if "decided_failure" in part.columns
            else tn + fn
        )
        fold_change = _actual_change_pp(part)
        totals = pd.to_numeric(part["original_total"], errors="coerce").fillna(0)
        row = {col: key for col, key in zip(group_cols, keys)}
        row.update(
            {
                "rows": int(len(part)),
                "trajectories": int(total),
                "original_resolved": int(original_resolved),
                "adjusted_resolved": int(adjusted_resolved),
                "false_negatives": int(fn),
                "false_positives": int(fp),
                "true_negatives": int(tn),
                "true_positives": int(tp),
                "decided_success": int(decided_success),
                "decided_failure": int(decided_failure),
                "original_resolve_rate_pct": original_resolved * 100.0 / total if total else 0.0,
                "adjusted_resolve_rate_pct": adjusted_resolved * 100.0 / total if total else 0.0,
                "resolve_rate_change_pp": (fp - fn) * 100.0 / total if total else 0.0,
                "mean_abs_resolve_rate_change_pp": float((fold_change.abs() * totals).sum() / total) if total else 0.0,
                "decided_trajectories": int(decided),
                "coverage_pct": decided * 100.0 / total if total else 0.0,
                "decision_accuracy_pct": (tp + tn) * 100.0 / decided if decided else 0.0,
                "saved_steps": int(saved_steps),
                "total_steps": int(total_steps),
                "step_save_pct": saved_steps * 100.0 / total_steps if total_steps else 0.0,
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def _write_csv(frame: pd.DataFrame, name: str) -> Path:
    path = OUT / name
    frame.to_csv(path, index=False)
    return path


def _plot_frontier(frame: pd.DataFrame, group_col: str, title: str, out_name: str) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 1, figsize=(10, 11), sharex=True)
    for label, part in frame.sort_values("target_valid_decision_accuracy_pct").groupby(group_col, sort=True):
        x = part["target_valid_decision_accuracy_pct"].astype(float)
        axes[0].plot(x, part["step_save_pct"].astype(float), marker="o", label=str(label))
        axes[0].plot(x, part["decision_accuracy_pct"].astype(float), marker="s", linestyle="--", alpha=0.75)
        axes[1].plot(x, part["resolve_rate_change_pp"].astype(float), marker="o", label=str(label))
        axes[2].plot(x, part["mean_abs_resolve_rate_change_pp"].astype(float), marker="o", label=str(label))
    axes[0].set_ylabel("Save / Acc (%)")
    axes[0].set_title(title)
    axes[0].grid(alpha=0.25)
    axes[1].axhline(0.0, color="#555555", linewidth=1)
    axes[1].set_ylabel("Actual shift (pp)")
    axes[1].grid(alpha=0.25)
    axes[2].set_ylabel("Mean abs shift (pp)")
    axes[2].set_xlabel("Target valid decision accuracy (%)")
    axes[2].grid(alpha=0.25)
    axes[0].legend(fontsize=7, loc="best")
    fig.tight_layout()
    path = OUT / out_name
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _plot_selected_bars(frame: pd.DataFrame, label_col: str, title: str, out_name: str) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = frame.copy()
    labels = data[label_col].astype(str).tolist()
    x = np.arange(len(labels))
    fig, axes = plt.subplots(3, 1, figsize=(max(8, len(labels) * 1.2), 10), sharex=True)
    axes[0].bar(x, data["step_save_pct"].astype(float), color="#4c78a8")
    axes[0].set_ylabel("Step save (%)")
    axes[0].set_title(title)
    axes[1].bar(x, data["decision_accuracy_pct"].astype(float), color="#59a14f")
    axes[1].set_ylabel("Decision acc (%)")
    colors = ["#e15759" if v < 0 else "#f28e2b" for v in data["resolve_rate_change_pp"].astype(float)]
    axes[2].bar(x, data["resolve_rate_change_pp"].astype(float), color=colors)
    axes[2].axhline(0.0, color="#555555", linewidth=1)
    axes[2].set_ylabel("Actual shift (pp)")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
    for ax in axes:
        ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    path = OUT / out_name
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _plot_main_heatmap(per_fold: pd.DataFrame) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    pivot = per_fold.pivot(
        index="test_model",
        columns="target_valid_decision_accuracy_pct",
        values="resolve_rate_change_pp",
    )
    pivot = pivot.assign(_sort=pivot[95.0]).sort_values("_sort", ascending=False).drop(columns="_sort")
    max_abs = max(abs(float(pivot.min().min())), abs(float(pivot.max().max())), 0.1)
    fig, ax = plt.subplots(figsize=(14, max(7, 0.36 * len(pivot) + 1.5)))
    im = ax.imshow(pivot.values, aspect="auto", cmap="RdBu_r", norm=mcolors.TwoSlopeNorm(vmin=-max_abs, vcenter=0, vmax=max_abs))
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f"{c:.0f}" for c in pivot.columns], rotation=45, ha="right")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=7)
    ax.set_xlabel("Target valid decision accuracy (%)")
    ax.set_ylabel("Held-out SWEVerify model")
    ax.set_title("Main LightGBM Per-Model Actual Resolve Shift")
    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Actual shift (pp)")
    fig.tight_layout()
    path = OUT / "main_sweverify_valid_accuracy_per_model_heatmap.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _main_artifacts() -> dict[str, Any]:
    sweep = ROOT / "lightgbm_main" / "policy_sweeps" / "valid_accuracy_075_095"
    aggregate = pd.read_csv(sweep / "aggregate_test_metrics.csv")
    per_fold = pd.read_csv(sweep / "per_fold_test_metrics.csv")
    selected = pd.read_csv(ROOT / "lightgbm_main" / "summary" / "per_fold_test_selected.csv")
    selected = _normalize_summary_columns(selected)
    selected_agg = _aggregate_rows(selected, [])
    if selected_agg.empty:
        selected_agg = pd.DataFrame([{}])
    _write_csv(aggregate, "main_sweverify_valid_accuracy_frontier.csv")
    _write_csv(per_fold, "main_sweverify_valid_accuracy_per_model.csv")
    _write_csv(selected, "main_sweverify_selected_per_model.csv")
    if not selected_agg.empty:
        _write_csv(selected_agg, "main_sweverify_selected_aggregate.csv")
    _plot_frontier(aggregate.assign(series="main"), "series", "Main SWEVerify Valid-Accuracy Frontier", "main_sweverify_valid_accuracy_frontier.png")
    _plot_main_heatmap(per_fold)
    selected_plot = selected.assign(model=selected["test_model"])
    _plot_selected_bars(selected_plot, "model", "Main SWEVerify Selected Operating Point By Model", "main_sweverify_selected_per_model.png")
    return {"aggregate_rows": int(len(aggregate)), "per_model_rows": int(len(per_fold))}


def _robustness_artifacts() -> dict[str, Any]:
    specs = [
        ("process", ROOT / "robustness_15pct_model_holdout_no_length"),
        ("rich_af_gold", ROOT / "robustness_15pct_model_holdout_rich_af_gold_no_length"),
    ]
    selected_test_rows = []
    selected_valid_rows = []
    frontier_valid_rows = []
    frontier_test_rows = []
    frontier_component_rows = []
    for run_label, run_dir in specs:
        selected_path = run_dir / "summary" / "selected_and_test_metrics.csv"
        if selected_path.exists():
            selected = pd.read_csv(selected_path)
            selected["robustness_run"] = run_label
            selected_test_rows.append(selected[selected["summary_split"] == "test_locked"].copy())
            selected_valid_rows.append(selected[selected["summary_split"] == "valid_selected"].copy())
        fine_dir = run_dir / "valid_accuracy_075_095_fine_detail"
        fine_selected = fine_dir / "valid_accuracy_075_095_selected_policies.csv"
        fine_frontier = fine_dir / "valid_accuracy_075_095_frontier.csv"
        fine_per_model = fine_dir / "valid_accuracy_075_095_per_model_test_metrics.csv"
        if fine_selected.exists() and fine_frontier.exists():
            valid_frontier = pd.read_csv(fine_selected)
            valid_frontier.insert(0, "robustness_run", run_label)
            test_frontier = pd.read_csv(fine_frontier)
            test_frontier.insert(0, "robustness_run", run_label)
            frontier_valid_rows.append(valid_frontier)
            frontier_test_rows.append(test_frontier)
            if fine_per_model.exists():
                test_components = pd.read_csv(fine_per_model)
                test_components.insert(0, "robustness_run", run_label)
                frontier_component_rows.append(test_components)
            else:
                frontier_component_rows.append(test_frontier)
            continue
        for dataset in ["toolathlon", "terminalbench"]:
            dataset_dir = run_dir / dataset
            valid_grid = pd.read_csv(dataset_dir / "safe_stop_valid_policy_grid.csv")
            test_frame = pd.read_parquet(dataset_dir / "test_predictions_safe_stop.parquet")
            valid_frontier, test_frontier = _evaluate_selected_frontier(
                valid_grid,
                test_frame,
                meta={"robustness_run": run_label, "dataset": dataset},
            )
            frontier_valid_rows.append(valid_frontier)
            frontier_test_rows.append(test_frontier)
            frontier_component_rows.append(test_frontier)
    selected_test = _normalize_summary_columns(pd.concat(selected_test_rows, ignore_index=True))
    selected_test["series"] = selected_test["robustness_run"] + "/" + selected_test["dataset"]
    selected_valid = _normalize_summary_columns(pd.concat(selected_valid_rows, ignore_index=True))
    frontier_valid = pd.concat(frontier_valid_rows, ignore_index=True)
    frontier_test = pd.concat(frontier_test_rows, ignore_index=True)
    frontier_components = pd.concat(frontier_component_rows, ignore_index=True)
    frontier_agg = _aggregate_rows(frontier_components, ["robustness_run", "dataset", "target_valid_decision_accuracy", "target_valid_decision_accuracy_pct"])
    frontier_agg["series"] = frontier_agg["robustness_run"] + "/" + frontier_agg["dataset"]
    _write_csv(selected_test, "robustness_selected_test_metrics.csv")
    _write_csv(selected_valid, "robustness_selected_valid_metrics.csv")
    _write_csv(frontier_valid, "robustness_valid_accuracy_selected_policies.csv")
    _write_csv(frontier_test, "robustness_valid_accuracy_per_target_test_metrics.csv")
    _write_csv(frontier_components, "robustness_valid_accuracy_per_model_test_metrics.csv")
    _write_csv(frontier_agg, "robustness_valid_accuracy_frontier.csv")
    _plot_selected_bars(selected_test, "series", "Robustness Selected Test Operating Points", "robustness_selected_test_metrics.png")
    _plot_frontier(
        frontier_agg[frontier_agg["robustness_run"] == "process"].copy(),
        "dataset",
        "Robustness Process Dual-Only Valid-Accuracy Frontier",
        "robustness_process_valid_accuracy_frontier.png",
    )
    _plot_frontier(
        frontier_agg[frontier_agg["robustness_run"] == "rich_af_gold"].copy(),
        "dataset",
        "Robustness Rich AF+Gold Dual-Only Valid-Accuracy Frontier",
        "robustness_rich_af_gold_valid_accuracy_frontier.png",
    )
    return {"selected_rows": int(len(selected_test)), "frontier_rows": int(len(frontier_agg))}


def _lr_tfidf_artifacts() -> dict[str, Any]:
    run_dirs = [ROOT / "model_compare" / "lr_tfidf_a", ROOT / "model_compare" / "lr_tfidf_b"]
    selected_rows = []
    frontier_valid_rows = []
    frontier_test_rows = []
    for run_dir in run_dirs:
        summary_path = run_dir / "summary" / "per_fold_test_selected.csv"
        if summary_path.exists():
            selected_rows.append(pd.read_csv(summary_path))
        for fold_dir in sorted((run_dir / "folds").iterdir()):
            if not fold_dir.is_dir():
                continue
            grid_path = fold_dir / "safe_stop_valid_policy_grid.csv"
            test_path = fold_dir / "test_predictions_safe_stop.parquet"
            if not grid_path.exists() or not test_path.exists():
                continue
            valid_grid = pd.read_csv(grid_path)
            test_frame = pd.read_parquet(test_path)
            for predictor in sorted(valid_grid["prefix_model"].astype(str).unique()):
                valid_frontier, test_frontier = _evaluate_selected_frontier(
                    valid_grid,
                    test_frame,
                    meta={"run_part": run_dir.name, "fold_id": fold_dir.name, "test_model": fold_dir.name, "predictor": predictor},
                    predictor_filter=predictor,
                )
                frontier_valid_rows.append(valid_frontier)
                frontier_test_rows.append(test_frontier)
    selected = _normalize_summary_columns(pd.concat(selected_rows, ignore_index=True))
    selected_agg = _aggregate_rows(selected, ["prefix_model"])
    selected_agg = selected_agg.rename(columns={"prefix_model": "predictor"})
    selected_per_model = selected.copy()
    selected_per_model["predictor"] = selected_per_model["prefix_model"].astype(str)
    frontier_valid = pd.concat(frontier_valid_rows, ignore_index=True)
    frontier_test = pd.concat(frontier_test_rows, ignore_index=True)
    frontier_agg = _aggregate_rows(frontier_test, ["predictor", "target_valid_decision_accuracy", "target_valid_decision_accuracy_pct"])
    _write_csv(selected_agg, "lr_tfidf_selected_full16_aggregate.csv")
    _write_csv(selected_per_model, "lr_tfidf_selected_per_model.csv")
    _write_csv(frontier_valid, "lr_tfidf_valid_accuracy_selected_policies.csv")
    _write_csv(frontier_test, "lr_tfidf_valid_accuracy_per_fold_test_metrics.csv")
    _write_csv(frontier_agg, "lr_tfidf_valid_accuracy_frontier.csv")
    _plot_selected_bars(selected_agg, "predictor", "LR / TF-IDF Selected Full16 Aggregates", "lr_tfidf_selected_full16_aggregate.png")
    _plot_frontier(frontier_agg, "predictor", "LR / TF-IDF Valid-Accuracy Frontiers", "lr_tfidf_valid_accuracy_frontier.png")
    return {"selected_rows": int(len(selected)), "frontier_rows": int(len(frontier_agg))}


def _sweverify_split_rows_from_main_folds() -> pd.DataFrame:
    """Rebuild the sweverify split-audit rows from the live main run folds.

    The standalone ``earlyeval_all_dataset_split_check`` artifact uses a uniform
    instance-disjoint geometry over the full (pre-exclusion) model pool, which no
    longer matches the paper's main run. The main run uses the leave-one-test-
    model shadow-validation split over the 16 retained agents, where train and
    validation deliberately share instances but never a ``(instance, model)``
    pair or trajectory. We read each fold's ``split_metadata.json`` so the split
    audit reflects the geometry the paper actually reports.
    """
    import json

    folds_dir = ROOT / "lightgbm_main" / "folds"
    rows: list[dict[str, Any]] = []
    for meta_path in sorted(folds_dir.glob("*/split_metadata.json")):
        meta = json.loads(meta_path.read_text())
        test_models = list(meta.get("test_models", []))
        train_models = list(meta.get("train_models", []))
        valid_models = list(meta.get("valid_models", []))
        # Shadow validation overlaps train and validation at the instance level
        # (the predictor sees the same instances under different agents), so the
        # instance-disjoint flag is False by construction; leakage safety is
        # instead guaranteed by zero (instance, model)/trajectory overlap.
        train_instances = int(meta.get("train_instances", 0))
        valid_instances = int(meta.get("valid_instances", 0))
        rows.append(
            {
                "dataset": "sweverify",
                "fold_id": meta_path.parent.name,
                "test_model": test_models[0] if test_models else "",
                "manifest": str(meta_path),
                "train_trajectories": int(meta.get("train_trajectories", 0)),
                "valid_trajectories": int(meta.get("valid_trajectories", 0)),
                "test_trajectories": int(meta.get("test_trajectories", 0)),
                "train_instances": train_instances,
                "valid_instances": valid_instances,
                "test_instances": int(meta.get("test_instances", 0)),
                "train_models": len(train_models),
                "valid_models": len(valid_models),
                "test_models": len(test_models),
                "test_model_absent_from_train_valid": (
                    all(m not in train_models and m not in valid_models for m in test_models)
                ),
                # Shadow validation shares the full instance pool between train and
                # validation, so instances are never disjoint; the leakage guard is
                # the zero (instance, model)/trajectory overlap recorded by the run.
                "train_valid_instance_disjoint": False,
            }
        )
    return pd.DataFrame(rows)


def _split_audit_artifacts() -> dict[str, Any]:
    split_index = pd.read_csv("paper/experiments/earlyeval_all_dataset_split_check/splits/split_index.csv")
    sweverify_rows = _sweverify_split_rows_from_main_folds()
    if not sweverify_rows.empty:
        split_index = pd.concat(
            [split_index[split_index["dataset"] != "sweverify"], sweverify_rows],
            ignore_index=True,
        )
    split_counts = (
        split_index.groupby("dataset", sort=True)
        .agg(
            folds=("fold_id", "nunique"),
            train_trajectories_mean=("train_trajectories", "mean"),
            valid_trajectories_mean=("valid_trajectories", "mean"),
            test_trajectories_mean=("test_trajectories", "mean"),
            all_test_model_absent=("test_model_absent_from_train_valid", "all"),
            all_train_valid_instance_disjoint=("train_valid_instance_disjoint", "all"),
        )
        .reset_index()
    )
    prefix_audit = pd.read_csv("paper/experiments/earlyeval_smoke/checks/prefix_audit_summary.csv")
    _write_csv(split_index, "split_check_index.csv")
    _write_csv(split_counts, "split_check_counts.csv")
    _write_csv(prefix_audit, "prefix_audit_summary.csv")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].bar(split_counts["dataset"], split_counts["folds"], color="#4c78a8")
    axes[0].set_ylabel("folds")
    axes[0].set_title("All-Dataset Split Check")
    axes[1].bar(prefix_audit["dataset"], prefix_audit["trajectories"], color="#59a14f")
    axes[1].set_ylabel("trajectories")
    axes[1].set_title("Prefix Audit Trajectories")
    for ax in axes:
        ax.grid(axis="y", alpha=0.25)
        ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    path = OUT / "split_prefix_audit_summary.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return {"split_rows": int(len(split_index)), "prefix_rows": int(len(prefix_audit))}


def _markdown_table(frame: pd.DataFrame, columns: list[str], max_rows: int | None = None) -> list[str]:
    rows = frame[columns].copy()
    if max_rows is not None:
        rows = rows.head(max_rows)
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join([":--"] * len(columns)) + " |"]
    for row in rows.to_dict("records"):
        values = []
        for col in columns:
            value = row[col]
            if isinstance(value, float):
                values.append(f"{value:.2f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return lines


def _write_readme() -> None:
    main = pd.read_csv(OUT / "main_sweverify_valid_accuracy_frontier.csv")
    robustness = pd.read_csv(OUT / "robustness_selected_test_metrics.csv")
    lr = pd.read_csv(OUT / "lr_tfidf_selected_full16_aggregate.csv")
    split_counts = pd.read_csv(OUT / "split_check_counts.csv")
    prefix = pd.read_csv(OUT / "prefix_audit_summary.csv")
    lines = [
        "# Reporting Detail Artifacts",
        "",
        "All resolve-rate shifts use actual adjusted solved count:",
        "",
        "`adjusted_resolved = original_resolved - false_negatives + false_positives`",
        "",
        "`resolve_rate_change_pp = (false_positives - false_negatives) / total * 100`",
        "",
        "## Main SWEVerify LightGBM",
        "",
        "- Frontier source: `lightgbm_main/policy_sweeps/valid_accuracy_075_095`.",
        "- Selected operating point source: `lightgbm_main/summary`.",
        "",
    ]
    compact_main = main[
        [
            "target_valid_decision_accuracy_pct",
            "step_save_pct",
            "decision_accuracy_pct",
            "coverage_pct",
            "resolve_rate_change_pp",
            "mean_abs_resolve_rate_change_pp",
            "false_negatives",
            "false_positives",
        ]
    ].rename(columns={"target_valid_decision_accuracy_pct": "target"})
    lines.extend(_markdown_table(compact_main, list(compact_main.columns)))
    lines.extend(
        [
            "",
            "Plots: `main_sweverify_valid_accuracy_frontier.png`, `main_sweverify_valid_accuracy_per_model_heatmap.png`, `main_sweverify_selected_per_model.png`.",
            "",
            "## Robustness",
            "",
        ]
    )
    rob_cols = [
        "series",
        "original_total",
        "step_save_pct",
        "decision_accuracy_pct",
        "coverage_pct",
        "resolve_rate_change_pp",
        "false_negatives",
        "false_positives",
    ]
    lines.extend(_markdown_table(robustness[rob_cols], rob_cols))
    lines.extend(
        [
            "",
            "Plots: `robustness_selected_test_metrics.png`, `robustness_process_valid_accuracy_frontier.png`, `robustness_rich_af_gold_valid_accuracy_frontier.png` (dual-only fine grid).",
            "",
            "## LR / TF-IDF Baselines",
            "",
        ]
    )
    lr_cols = [
        "predictor",
        "trajectories",
        "step_save_pct",
        "decision_accuracy_pct",
        "coverage_pct",
        "resolve_rate_change_pp",
        "mean_abs_resolve_rate_change_pp",
        "false_negatives",
        "false_positives",
    ]
    lines.extend(_markdown_table(lr[lr_cols], lr_cols))
    lines.extend(
        [
            "",
            "Plots: `lr_tfidf_selected_full16_aggregate.png`, `lr_tfidf_valid_accuracy_frontier.png`.",
            "",
            "## Split / Prefix Audit",
            "",
        ]
    )
    split_cols = [
        "dataset",
        "folds",
        "train_trajectories_mean",
        "valid_trajectories_mean",
        "test_trajectories_mean",
        "all_test_model_absent",
        "all_train_valid_instance_disjoint",
    ]
    lines.extend(_markdown_table(split_counts[split_cols], split_cols))
    lines.append("")
    prefix_cols = ["dataset", "ok", "rows", "trajectories", "models", "has_step0", "p99_trajectory_steps"]
    lines.extend(_markdown_table(prefix[prefix_cols], prefix_cols))
    lines.extend(
        [
            "",
            "Plot: `split_prefix_audit_summary.png`.",
            "",
            "## Files",
            "",
        ]
    )
    for path in sorted(OUT.iterdir()):
        if path.name != "README.md":
            lines.append(f"- `{path.name}`")
    lines.append("")
    (OUT / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    _ensure_out()
    manifest = {
        "main": _main_artifacts(),
        "robustness": _robustness_artifacts(),
        "lr_tfidf": _lr_tfidf_artifacts(),
        "split_audit": _split_audit_artifacts(),
    }
    pd.DataFrame(
        [{"section": key, **value} for key, value in manifest.items()]
    ).to_csv(OUT / "manifest.csv", index=False)
    _write_readme()
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
