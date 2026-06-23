from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from final3.experiments.build_reporting_detail import (
    ROOT,
    TARGETS,
    _actual_change_pp,
    _aggregate_rows,
    _format_thr,
    _markdown_table,
    _normalize_summary_columns,
    _policy_from_row,
    _select_policy_for_target,
)
from final3.policies.safe_stop import apply_policy


RUNS = [
    {
        "run_label": "process",
        "run_dir": ROOT / "robustness_15pct_model_holdout_no_length",
        "predictor": "Robust_LightGBM_Process",
        "feature_preset": "process",
    },
    {
        "run_label": "rich_af_gold",
        "run_dir": ROOT / "robustness_15pct_model_holdout_rich_af_gold_no_length",
        "predictor": "Robust_LightGBM_Dense_AF_Gold",
        "feature_preset": "rich_af_gold",
    },
]
DATASETS = ["toolathlon", "terminalbench"]


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def _summary_row(summary: pd.DataFrame, **meta: Any) -> dict[str, Any]:
    row = summary.iloc[0].to_dict()
    row.update(meta)
    return row


def _per_agent_rows(per_agent: pd.DataFrame, **meta: Any) -> pd.DataFrame:
    out = _normalize_summary_columns(per_agent.copy())
    for key, value in reversed(list(meta.items())):
        out.insert(0, key, value)
    return out


def _attach_mean_abs_from_parts(frame: pd.DataFrame, parts: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    out = frame.copy()
    if out.empty or parts.empty:
        return out
    work = _normalize_summary_columns(parts)
    work["_abs_resolve_change_pp"] = _actual_change_pp(work).abs()
    work["_mean_abs_weight"] = pd.to_numeric(work["original_total"], errors="coerce").fillna(0.0)
    rows: list[dict[str, Any]] = []
    groups = work.groupby(group_cols, sort=False, dropna=False)
    for keys, part in groups:
        if not isinstance(keys, tuple):
            keys = (keys,)
        total = float(part["_mean_abs_weight"].sum())
        if total:
            mean_abs = float((part["_abs_resolve_change_pp"] * part["_mean_abs_weight"]).sum() / total)
        else:
            mean_abs = float(part["_abs_resolve_change_pp"].mean())
        row = {col: key for col, key in zip(group_cols, keys)}
        row["mean_abs_resolve_rate_change_pp"] = mean_abs
        rows.append(row)
    mean_abs_frame = pd.DataFrame(rows)
    if mean_abs_frame.empty:
        return out
    out = out.drop(columns=["mean_abs_resolve_rate_change_pp"], errors="ignore")
    return out.merge(mean_abs_frame, on=group_cols, how="left")


def _selected_test_for_dataset(dataset_dir: Path, dataset: str) -> tuple[dict[str, Any], pd.DataFrame]:
    selected = pd.read_csv(dataset_dir / "safe_stop_selected_policies.csv")
    test_frame = pd.read_parquet(dataset_dir / "test_predictions_safe_stop.parquet")
    policy = _policy_from_row(selected.iloc[0])
    _, summary, per_agent = apply_policy(test_frame, policy)
    selected_norm = _normalize_summary_columns(pd.DataFrame([selected.iloc[0]])).iloc[0]
    aggregate = _normalize_summary_columns(summary)
    row = _summary_row(
        aggregate,
        dataset=dataset,
        test_models=int(test_frame["orig_model_id"].nunique() if "orig_model_id" in test_frame.columns else test_frame["model_id"].nunique()),
        selected_policy_name=policy.name,
        selected_score_mode=policy.score_mode,
        selected_predictor=policy.predictor,
        selected_policy_mode=policy.policy_mode,
        selected_success_thr=_format_thr(policy.success_thr),
        selected_failure_thr=_format_thr(policy.failure_thr),
        selected_valid_save_pct=float(selected_norm["step_save_pct"]),
        selected_valid_decision_accuracy_pct=float(selected_norm["decision_accuracy_pct"]),
        selected_valid_resolve_change_pp=float(selected_norm["resolve_rate_change_pp"]),
    )
    per_agent_out = _per_agent_rows(
        per_agent,
        dataset=dataset,
        selected_policy_name=policy.name,
        selected_policy_mode=policy.policy_mode,
        selected_success_thr=_format_thr(policy.success_thr),
        selected_failure_thr=_format_thr(policy.failure_thr),
    )
    return row, per_agent_out


def _frontier_for_dataset(dataset_dir: Path, dataset: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    valid_grid = pd.read_csv(dataset_dir / "safe_stop_valid_policy_grid.csv")
    test_frame = pd.read_parquet(dataset_dir / "test_predictions_safe_stop.parquet")
    selected_rows: list[dict[str, Any]] = []
    aggregate_rows: list[dict[str, Any]] = []
    per_agent_rows: list[pd.DataFrame] = []
    cache: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}
    for target in TARGETS:
        selected, status = _select_policy_for_target(valid_grid, target)
        policy = _policy_from_row(selected)
        if policy.name not in cache:
            _, summary, per_agent = apply_policy(test_frame, policy)
            cache[policy.name] = (_normalize_summary_columns(summary), _normalize_summary_columns(per_agent))
        summary, per_agent = cache[policy.name]
        valid_norm = _normalize_summary_columns(pd.DataFrame([selected])).iloc[0]
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
                "selected_valid_save_pct": float(valid_norm["step_save_pct"]),
                "selected_valid_decision_accuracy_pct": float(valid_norm["decision_accuracy_pct"]),
                "selected_valid_resolve_change_pp": float(valid_norm["resolve_rate_change_pp"]),
            }
        )
        aggregate_rows.append(
            _summary_row(
                summary,
                dataset=dataset,
                target_valid_decision_accuracy=target,
                target_valid_decision_accuracy_pct=target * 100.0,
                selection_status=status,
                selected_policy_name=policy.name,
                selected_policy_mode=policy.policy_mode,
                selected_success_thr=_format_thr(policy.success_thr),
                selected_failure_thr=_format_thr(policy.failure_thr),
            )
        )
        per_agent_rows.append(
            _per_agent_rows(
                per_agent,
                dataset=dataset,
                target_valid_decision_accuracy=target,
                target_valid_decision_accuracy_pct=target * 100.0,
                selected_policy_name=policy.name,
                selected_policy_mode=policy.policy_mode,
                selected_success_thr=_format_thr(policy.success_thr),
                selected_failure_thr=_format_thr(policy.failure_thr),
            )
        )
    return (
        pd.DataFrame(selected_rows),
        _normalize_summary_columns(pd.DataFrame(aggregate_rows)),
        pd.concat(per_agent_rows, ignore_index=True),
    )


def _plot_frontier(frame: pd.DataFrame, out: Path, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 1, figsize=(10, 11), sharex=True)
    for dataset, part in frame.sort_values("target_valid_decision_accuracy_pct").groupby("dataset", sort=True):
        x = part["target_valid_decision_accuracy_pct"].astype(float)
        axes[0].plot(x, part["step_save_pct"].astype(float), marker="o", label=f"{dataset} save")
        axes[0].plot(x, part["decision_accuracy_pct"].astype(float), marker="s", linestyle="--", label=f"{dataset} acc")
        axes[1].plot(x, part["resolve_rate_change_pp"].astype(float), marker="o", label=str(dataset))
        axes[2].plot(x, part["mean_abs_resolve_rate_change_pp"].astype(float), marker="o", label=str(dataset))
    axes[0].set_ylabel("Save / Acc (%)")
    axes[0].set_title(title)
    axes[1].set_ylabel("Actual shift (pp)")
    axes[1].axhline(0.0, color="#555555", linewidth=1)
    axes[2].set_ylabel("Mean abs shift (pp)")
    axes[2].set_xlabel("Target valid decision accuracy (%)")
    for ax in axes:
        ax.grid(alpha=0.25)
        ax.legend(fontsize=7, loc="best")
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def _plot_per_model_heatmap(frame: pd.DataFrame, dataset: str, out: Path, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.colors as mcolors
    import matplotlib.pyplot as plt

    part = frame[frame["dataset"] == dataset].copy()
    pivot = part.pivot(
        index="agent_model",
        columns="target_valid_decision_accuracy_pct",
        values="resolve_rate_change_pp",
    )
    pivot = pivot.assign(_sort=pivot[95.0]).sort_values("_sort", ascending=False).drop(columns="_sort")
    max_abs = max(abs(float(pivot.min().min())), abs(float(pivot.max().max())), 0.1)
    fig, ax = plt.subplots(figsize=(14, max(5.5, 0.34 * len(pivot) + 1.5)))
    im = ax.imshow(
        pivot.values,
        aspect="auto",
        cmap="RdBu_r",
        norm=mcolors.TwoSlopeNorm(vmin=-max_abs, vcenter=0.0, vmax=max_abs),
    )
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f"{c:.0f}" for c in pivot.columns], rotation=45, ha="right")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=7)
    ax.set_xlabel("Target valid decision accuracy (%)")
    ax.set_ylabel("Held-out model")
    ax.set_title(title)
    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Actual shift (pp)")
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def _plot_selected_per_model(frame: pd.DataFrame, dataset: str, out: Path, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = frame[frame["dataset"] == dataset].sort_values("resolve_rate_change_pp", ascending=False).copy()
    labels = data["agent_model"].astype(str).tolist()
    x = np.arange(len(labels))
    fig, axes = plt.subplots(3, 1, figsize=(max(8, len(labels) * 1.1), 9), sharex=True)
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
    axes[2].set_xticklabels(labels, rotation=30, ha="right", fontsize=7)
    for ax in axes:
        ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def _display_cols(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["target"] = out["target_valid_decision_accuracy_pct"].astype(float)
    cols = [
        "dataset",
        "target",
        "step_save_pct",
        "decision_accuracy_pct",
        "coverage_pct",
        "resolve_rate_change_pp",
        "mean_abs_resolve_rate_change_pp",
        "false_negatives",
        "false_positives",
        "selected_policy_mode",
        "selected_success_thr",
        "selected_failure_thr",
    ]
    return out[cols]


def _write_detail_readme(run: dict[str, Any], out_dir: Path, selected_test: pd.DataFrame, selected_per_model: pd.DataFrame, frontier: pd.DataFrame) -> None:
    selected_display = selected_test[
        [
            "dataset",
            "test_models",
            "original_total",
            "n_decided",
            "coverage_pct",
            "decision_accuracy_pct",
            "step_save_pct",
            "resolve_rate_change_pp",
            "mean_abs_resolve_rate_change_pp",
            "false_negatives",
            "false_positives",
            "selected_policy_mode",
            "selected_success_thr",
            "selected_failure_thr",
        ]
    ].copy()
    per_model_display = selected_per_model[
        [
            "dataset",
            "agent_model",
            "original_total",
            "step_save_pct",
            "decision_accuracy_pct",
            "resolve_rate_change_pp",
            "false_negatives",
            "false_positives",
        ]
    ].copy()
    lines = [
        "# Robustness Valid-Accuracy Detail",
        "",
        f"- run_dir: `{run['run_dir']}`",
        f"- feature_preset: `{run['feature_preset']}`",
        f"- predictor: `{run['predictor']}`",
        "- selection: choose the highest valid step saving policy meeting each target valid decision accuracy, then apply it unchanged to test.",
        "",
        "Resolve-rate shifts use the actual adjusted solved count:",
        "",
        "`adjusted_resolved = original_resolved - false_negatives + false_positives`",
        "",
        "## Selected Operating Point",
        "",
    ]
    lines.extend(_markdown_table(selected_display, list(selected_display.columns)))
    lines.extend(["", "## Selected Operating Point By Held-Out Model", ""])
    lines.extend(_markdown_table(per_model_display, list(per_model_display.columns)))
    lines.extend(["", "## Valid-Accuracy Frontier", ""])
    for dataset in DATASETS:
        subset = _display_cols(frontier[frontier["dataset"] == dataset].copy())
        lines.extend([f"### {dataset}", ""])
        lines.extend(_markdown_table(subset, list(subset.columns)))
        lines.append("")
    lines.extend(
        [
            "## Files",
            "",
            "- `selected_test_metrics.csv`",
            "- `selected_test_per_model_metrics.csv`",
            "- `valid_accuracy_075_095_selected_policies.csv`",
            "- `valid_accuracy_075_095_frontier.csv`",
            "- `valid_accuracy_075_095_per_model_test_metrics.csv`",
            "- `valid_accuracy_075_095_frontier.png`",
            "- `selected_per_model_toolathlon.png`",
            "- `selected_per_model_terminalbench.png`",
            "- `valid_accuracy_075_095_per_model_toolathlon_heatmap.png`",
            "- `valid_accuracy_075_095_per_model_terminalbench_heatmap.png`",
            "",
        ]
    )
    (out_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def _write_summary_readme(run: dict[str, Any], selected_test: pd.DataFrame) -> None:
    run_dir = Path(run["run_dir"])
    table = selected_test[
        [
            "dataset",
            "test_models",
            "original_total",
            "n_decided",
            "coverage_pct",
            "decision_accuracy_pct",
            "step_save_pct",
            "resolve_rate_change_pp",
            "mean_abs_resolve_rate_change_pp",
        ]
    ].rename(
        columns={
            "original_total": "n",
            "n_decided": "decided",
            "decision_accuracy_pct": "acc_pct",
            "step_save_pct": "save_pct",
        }
    )
    lines = [
        "# Robustness 15% Model-Holdout",
        "",
        f"- run_dir: `{run_dir}`",
        f"- predictor: `{run['predictor']}`",
        f"- feature_preset: `{run['feature_preset']}`",
        "- `process`: process/numeric/action only",
        "- `rich_af_gold`: process dense + answer/gold dense + action-feedback TF-IDF/SVD",
        "- coarse valid-accuracy artifacts: `../valid_accuracy_075_095_detail/README.md`",
        "- fine valid-accuracy artifacts: `../valid_accuracy_075_095_fine_detail/README.md`",
        "",
    ]
    lines.extend(_markdown_table(table, list(table.columns)))
    lines.append("")
    (run_dir / "summary" / "README.md").write_text("\n".join(lines), encoding="utf-8")


def build_run(run: dict[str, Any]) -> dict[str, Any]:
    run_dir = Path(run["run_dir"])
    out_dir = run_dir / "valid_accuracy_075_095_detail"
    out_dir.mkdir(parents=True, exist_ok=True)
    selected_rows: list[dict[str, Any]] = []
    selected_per_model_rows: list[pd.DataFrame] = []
    selected_policy_rows: list[pd.DataFrame] = []
    frontier_rows: list[pd.DataFrame] = []
    frontier_per_model_rows: list[pd.DataFrame] = []
    for dataset in DATASETS:
        dataset_dir = run_dir / dataset
        selected_row, selected_per_model = _selected_test_for_dataset(dataset_dir, dataset)
        selected_rows.append(selected_row)
        selected_per_model_rows.append(selected_per_model)
        selected_policy, frontier, per_model = _frontier_for_dataset(dataset_dir, dataset)
        selected_policy_rows.append(selected_policy)
        frontier_rows.append(frontier)
        frontier_per_model_rows.append(per_model)
    selected_test = _normalize_summary_columns(pd.DataFrame(selected_rows))
    selected_per_model = pd.concat(selected_per_model_rows, ignore_index=True)
    selected_per_model["mean_abs_resolve_rate_change_pp"] = selected_per_model["resolve_rate_change_pp"].abs()
    selected_test = _attach_mean_abs_from_parts(selected_test, selected_per_model, ["dataset"])
    selected_policy = pd.concat(selected_policy_rows, ignore_index=True)
    frontier = pd.concat(frontier_rows, ignore_index=True)
    per_model = pd.concat(frontier_per_model_rows, ignore_index=True)
    per_model["mean_abs_resolve_rate_change_pp"] = per_model["resolve_rate_change_pp"].abs()
    frontier = _attach_mean_abs_from_parts(
        frontier,
        per_model,
        ["dataset", "target_valid_decision_accuracy", "target_valid_decision_accuracy_pct"],
    )
    _write_csv(selected_test, out_dir / "selected_test_metrics.csv")
    _write_csv(selected_per_model, out_dir / "selected_test_per_model_metrics.csv")
    _write_csv(selected_policy, out_dir / "valid_accuracy_075_095_selected_policies.csv")
    _write_csv(frontier, out_dir / "valid_accuracy_075_095_frontier.csv")
    _write_csv(per_model, out_dir / "valid_accuracy_075_095_per_model_test_metrics.csv")
    _plot_frontier(frontier, out_dir / "valid_accuracy_075_095_frontier.png", f"{run['feature_preset']} Robustness Valid-Accuracy Frontier")
    for dataset in DATASETS:
        _plot_selected_per_model(
            selected_per_model,
            dataset,
            out_dir / f"selected_per_model_{dataset}.png",
            f"{run['feature_preset']} {dataset} Selected Point By Held-Out Model",
        )
        _plot_per_model_heatmap(
            per_model,
            dataset,
            out_dir / f"valid_accuracy_075_095_per_model_{dataset}_heatmap.png",
            f"{run['feature_preset']} {dataset} Per-Model Actual Shift",
        )
    _write_detail_readme(run, out_dir, selected_test, selected_per_model, frontier)
    _write_summary_readme(run, selected_test)
    return {
        "run_label": run["run_label"],
        "detail_dir": str(out_dir),
        "selected_rows": len(selected_test),
        "frontier_rows": len(frontier),
        "per_model_frontier_rows": len(per_model),
    }


def main() -> int:
    rows = [build_run(run) for run in RUNS]
    pd.DataFrame(rows).to_csv(ROOT / "robustness_detail_manifest.csv", index=False)
    print(pd.DataFrame(rows).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
