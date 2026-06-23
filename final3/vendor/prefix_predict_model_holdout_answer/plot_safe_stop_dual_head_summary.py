#!/usr/bin/env python3
"""Generate plots and readable tables for safe-stop dual-head experiments."""

from __future__ import annotations

import argparse
import math
import os
import re
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".mplconfig"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D


RUN_RE = re.compile(r"per_instance_model_valid3_(?P<split>top3|bottom3|mid3)_(?P<variant>i|j)_safe_stop_dual_head_retrain")
SPLIT_ORDER = ["top3", "mid3", "bottom3"]
SPLIT_COLORS = {"top3": "#1f77b4", "mid3": "#2ca02c", "bottom3": "#d62728"}
MODE_MARKERS = {"calibrated": "o", "raw": "^"}


def parse_args() -> argparse.Namespace:
    module_dir = Path(__file__).resolve().parent
    default_root = module_dir / "runs" / "model_holdout_answer_calibrated_full" / "reports"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports-root", type=Path, default=default_root)
    parser.add_argument("--output-dir", type=Path, default=default_root / "safe_stop_dual_head_visual_summary")
    parser.add_argument("--max-abs-drop-pp", type=float, default=2.0)
    parser.add_argument("--min-acc-pct", type=float, default=90.0)
    return parser.parse_args()


def _fmt(value: float, digits: int = 1) -> str:
    if value is None:
        return "-"
    value = float(value)
    if math.isnan(value):
        return "-"
    return f"{value:.{digits}f}"


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def _variant_label(value: str) -> str:
    return value.upper()


def _model_label(value: str) -> str:
    if value == "I_LightGBM_Dense_AF":
        return "I"
    if value == "J_LightGBM_Dense_AF_Thought":
        return "J"
    return value


def find_runs(reports_root: Path) -> list[dict[str, object]]:
    runs: list[dict[str, object]] = []
    for path in sorted(reports_root.glob("per_instance_model_valid3_*_*_safe_stop_dual_head_retrain")):
        match = RUN_RE.fullmatch(path.name)
        if not match:
            continue
        required = [
            path / "safe_stop_selected_policies.csv",
            path / "safe_stop_test_selected.csv",
            path / "safe_stop_valid_policy_grid.csv",
            path / "safe_stop_calibration_summary.csv",
        ]
        if not all(item.exists() for item in required):
            continue
        runs.append({"path": path, "split": match.group("split"), "variant": match.group("variant")})
    runs.sort(key=lambda item: (SPLIT_ORDER.index(str(item["split"])), str(item["variant"])))
    return runs


def load_data(runs: list[dict[str, object]]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    selected_rows: list[dict[str, object]] = []
    grid_frames: list[pd.DataFrame] = []
    calibration_frames: list[pd.DataFrame] = []

    for run in runs:
        path = Path(run["path"])
        split = str(run["split"])
        variant = str(run["variant"])
        valid = pd.read_csv(path / "safe_stop_selected_policies.csv")
        test = pd.read_csv(path / "safe_stop_test_selected.csv")
        calibration = pd.read_csv(path / "safe_stop_calibration_summary.csv")
        grid = pd.read_csv(path / "safe_stop_valid_policy_grid.csv")

        for frame in (valid, test, calibration, grid):
            frame.insert(0, "variant", variant)
            frame.insert(0, "split", split)
            frame.insert(0, "run_dir", path.name)

        calibration_frames.append(calibration)
        grid_frames.append(grid)

        for _, valid_row in valid.iterrows():
            mask = test["score_mode"].eq(valid_row["score_mode"])
            for key in ["prefix_model", "policy_mode", "success_thr", "failure_thr", "min_step", "consecutive"]:
                mask &= test[key].eq(valid_row[key])
            if not mask.any():
                continue
            test_row = test[mask].iloc[0]
            selected_rows.append(
                {
                    "run_dir": path.name,
                    "split": split,
                    "variant": variant,
                    "model": valid_row["prefix_model"],
                    "model_short": _model_label(str(valid_row["prefix_model"])),
                    "score_mode": valid_row["score_mode"],
                    "policy_mode": valid_row["policy_mode"],
                    "success_thr": valid_row["success_thr"],
                    "failure_thr": valid_row["failure_thr"],
                    "min_step": int(valid_row["min_step"]),
                    "consecutive": int(valid_row["consecutive"]),
                    "policy": f"s{valid_row['success_thr']:g}/f{valid_row['failure_thr']:g}/min{int(valid_row['min_step'])}/k{int(valid_row['consecutive'])}",
                    "valid_coverage_pct": float(valid_row["coverage"]) * 100.0,
                    "valid_acc_pct": float(valid_row["decision_accuracy"]) * 100.0,
                    "valid_save_pct": float(valid_row["pct_steps_saved"]),
                    "valid_drop_pp": float(valid_row["resolve_rate_drop"]) * 100.0,
                    "valid_precision_success_pct": float(valid_row["precision_success"]) * 100.0,
                    "valid_precision_failure_pct": float(valid_row["precision_failure"]) * 100.0,
                    "test_coverage_pct": float(test_row["coverage"]) * 100.0,
                    "test_acc_pct": float(test_row["decision_accuracy"]) * 100.0,
                    "test_save_pct": float(test_row["pct_steps_saved"]),
                    "test_drop_pp": float(test_row["resolve_rate_drop"]) * 100.0,
                    "test_precision_success_pct": float(test_row["precision_success"]) * 100.0,
                    "test_precision_failure_pct": float(test_row["precision_failure"]) * 100.0,
                    "test_original_rate_pct": float(test_row["original_resolve_rate"]) * 100.0,
                    "test_adjusted_rate_pct": float(test_row["adjusted_resolve_rate"]) * 100.0,
                    "test_fn": int(test_row["false_negatives"]),
                    "test_fp": int(test_row["false_positives"]),
                    "test_n_decided": int(test_row["n_decided"]),
                }
            )

    selected = pd.DataFrame(selected_rows)
    grid_all = pd.concat(grid_frames, ignore_index=True) if grid_frames else pd.DataFrame()
    calibration_all = pd.concat(calibration_frames, ignore_index=True) if calibration_frames else pd.DataFrame()

    if not grid_all.empty:
        grid_all["coverage_pct"] = grid_all["coverage"] * 100.0
        grid_all["decision_accuracy_pct"] = grid_all["decision_accuracy"] * 100.0
        grid_all["resolve_rate_drop_pp"] = grid_all["resolve_rate_drop"] * 100.0
        grid_all["abs_drop_pp"] = grid_all["resolve_rate_drop_pp"].abs()
        grid_all["model_short"] = grid_all["prefix_model"].map(_model_label)
    return selected, grid_all, calibration_all


def write_tables(
    selected: pd.DataFrame,
    grid: pd.DataFrame,
    calibration: pd.DataFrame,
    output_dir: Path,
    *,
    max_abs_drop_pp: float,
    min_acc_pct: float,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}

    selected_path = output_dir / "selected_policy_valid_test.csv"
    selected.sort_values(["split", "variant", "score_mode"]).to_csv(selected_path, index=False)
    paths["selected"] = selected_path

    calibration_path = output_dir / "calibration_summary_all.csv"
    calibration.to_csv(calibration_path, index=False)
    paths["calibration"] = calibration_path

    grid_path = output_dir / "valid_policy_grid_all.csv"
    grid.to_csv(grid_path, index=False)
    paths["grid"] = grid_path

    if not grid.empty:
        dual = grid[grid["policy_mode"].eq("dual")].copy()
        passing = dual[
            dual["abs_drop_pp"].le(max_abs_drop_pp)
            & dual["decision_accuracy_pct"].ge(min_acc_pct)
            & dual["pct_steps_saved"].ge(20.0)
        ].copy()
        if not passing.empty:
            best = passing.sort_values(
                ["split", "variant", "score_mode", "pct_steps_saved", "abs_drop_pp"],
                ascending=[True, True, True, False, True],
            ).groupby(["split", "variant", "score_mode"], as_index=False).head(10)
        else:
            best = passing
        best_path = output_dir / "best_valid_dual_candidates.csv"
        best.to_csv(best_path, index=False)
        paths["best"] = best_path
    return paths


def plot_selected_tradeoff(selected: pd.DataFrame, output_dir: Path) -> Path:
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9.5, 6.2))
    for _, row in selected.iterrows():
        color = SPLIT_COLORS.get(row["split"], "gray")
        marker = MODE_MARKERS.get(row["score_mode"], "o")
        ax.scatter(
            row["test_save_pct"],
            row["test_drop_pp"],
            s=95,
            color=color,
            marker=marker,
            edgecolor="white",
            linewidth=0.8,
            alpha=0.9,
        )
        ax.text(
            row["test_save_pct"] + 0.45,
            row["test_drop_pp"] + 0.05,
            f"{row['split']}-{_variant_label(row['variant'])}-{row['score_mode'][0]}",
            fontsize=8,
        )
    ax.axhline(0, color="black", linewidth=0.9)
    ax.axhspan(-2, 2, color="gray", alpha=0.10, label="±2pp band")
    ax.grid(True, alpha=0.25)
    ax.set_xlabel("Test Saved Steps (%)")
    ax.set_ylabel("Test Resolve-Rate Drop (pp)")
    ax.set_title("Safe-Stop Dual-Head Locked Policies: Save% vs Drop")
    handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=color, label=split, markersize=8)
        for split, color in SPLIT_COLORS.items()
    ] + [
        Line2D([0], [0], marker=marker, color="gray", linestyle="None", label=mode, markersize=8)
        for mode, marker in MODE_MARKERS.items()
    ]
    ax.legend(handles=handles, loc="best")
    path = plot_dir / "selected_tradeoff_save_vs_drop.png"
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return path


def plot_selected_bars(selected: pd.DataFrame, output_dir: Path) -> Path:
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    data = selected.sort_values(["split", "variant", "score_mode"]).copy()
    data["label"] = data["split"] + "-" + data["variant"].str.upper() + "-" + data["score_mode"].str.replace("calibrated", "cal", regex=False)
    metrics = [
        ("test_save_pct", "Save %"),
        ("test_drop_pp", "Drop pp"),
        ("test_acc_pct", "Decision Acc %"),
        ("test_coverage_pct", "Coverage %"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(14, 8.2))
    for ax, (col, title) in zip(axes.ravel(), metrics):
        colors = [SPLIT_COLORS.get(split, "gray") for split in data["split"]]
        ax.bar(np.arange(len(data)), data[col], color=colors, alpha=0.82)
        if col == "test_drop_pp":
            ax.axhline(0, color="black", linewidth=0.8)
            ax.axhspan(-2, 2, color="gray", alpha=0.10)
        ax.set_title(title)
        ax.grid(True, axis="y", alpha=0.25)
        ax.set_xticks(np.arange(len(data)))
        ax.set_xticklabels(data["label"], rotation=55, ha="right", fontsize=8)
    fig.suptitle("Safe-Stop Dual-Head Selected Policy Test Metrics", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    path = plot_dir / "selected_policy_metric_bars.png"
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return path


def plot_valid_test_gap(selected: pd.DataFrame, output_dir: Path) -> Path:
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.2, 6.6))
    min_value = min(selected["valid_drop_pp"].min(), selected["test_drop_pp"].min(), -5)
    max_value = max(selected["valid_drop_pp"].max(), selected["test_drop_pp"].max(), 5)
    ax.plot([min_value, max_value], [min_value, max_value], color="black", linestyle="--", linewidth=0.9, label="test = valid")
    ax.axhline(0, color="black", linewidth=0.6)
    ax.axvline(0, color="black", linewidth=0.6)
    ax.axhspan(-2, 2, color="gray", alpha=0.08)
    ax.axvspan(-2, 2, color="gray", alpha=0.08)
    for _, row in selected.iterrows():
        color = SPLIT_COLORS.get(row["split"], "gray")
        marker = MODE_MARKERS.get(row["score_mode"], "o")
        ax.scatter(row["valid_drop_pp"], row["test_drop_pp"], s=90, color=color, marker=marker, edgecolor="white")
        ax.text(row["valid_drop_pp"] + 0.12, row["test_drop_pp"] + 0.12, f"{row['split']}-{_variant_label(row['variant'])}-{row['score_mode'][0]}", fontsize=8)
    ax.set_xlabel("Valid Drop pp")
    ax.set_ylabel("Test Drop pp")
    ax.set_title("Safe-Stop Valid-Selected Policy Transfer")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    path = plot_dir / "valid_to_test_drop_gap.png"
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return path


def plot_fn_fp(selected: pd.DataFrame, output_dir: Path) -> Path:
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    data = selected.sort_values(["split", "variant", "score_mode"]).copy()
    labels = data["split"] + "-" + data["variant"].str.upper() + "-" + data["score_mode"].str.replace("calibrated", "cal", regex=False)
    x = np.arange(len(data))
    width = 0.38
    fig, ax = plt.subplots(figsize=(13.5, 5.3))
    ax.bar(x - width / 2, data["test_fn"], width=width, label="FN: killed success", color="#d62728", alpha=0.82)
    ax.bar(x + width / 2, data["test_fp"], width=width, label="FP: false success", color="#1f77b4", alpha=0.82)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=50, ha="right", fontsize=8)
    ax.set_ylabel("Count")
    ax.set_title("Safe-Stop Test Errors: FN vs FP")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    path = plot_dir / "selected_policy_fn_fp_counts.png"
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return path


def plot_calibration(calibration: pd.DataFrame, output_dir: Path) -> Path:
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    summary = calibration.groupby("head", as_index=False).agg(
        valid_raw_brier=("valid_raw_brier", "mean"),
        valid_cal_brier=("valid_cal_brier", "mean"),
        test_raw_brier=("test_raw_brier", "mean"),
        test_cal_brier=("test_cal_brier", "mean"),
        valid_raw_logloss=("valid_raw_logloss", "mean"),
        valid_cal_logloss=("valid_cal_logloss", "mean"),
        test_raw_logloss=("test_raw_logloss", "mean"),
        test_cal_logloss=("test_cal_logloss", "mean"),
    )
    heads = summary["head"].tolist()
    x = np.arange(len(heads))
    width = 0.2
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    specs = [
        ("Brier", "brier", axes[0]),
        ("Log Loss", "logloss", axes[1]),
    ]
    for title, metric, ax in specs:
        ax.bar(x - 1.5 * width, summary[f"valid_raw_{metric}"], width, label="valid raw", color="#1f77b4", alpha=0.75)
        ax.bar(x - 0.5 * width, summary[f"valid_cal_{metric}"], width, label="valid cal", color="#aec7e8", alpha=0.95)
        ax.bar(x + 0.5 * width, summary[f"test_raw_{metric}"], width, label="test raw", color="#ff7f0e", alpha=0.75)
        ax.bar(x + 1.5 * width, summary[f"test_cal_{metric}"], width, label="test cal", color="#ffbb78", alpha=0.95)
        ax.set_xticks(x)
        ax.set_xticklabels(heads)
        ax.set_title(title)
        ax.grid(True, axis="y", alpha=0.25)
    axes[0].legend(loc="best", fontsize=8)
    fig.suptitle("Dual-Head Calibration: valid improves, test may shift", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    path = plot_dir / "calibration_valid_test_brier_logloss.png"
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return path


def _best_grid_for_heatmap(part: pd.DataFrame, *, max_abs_drop_pp: float, min_acc_pct: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    dual = part[part["policy_mode"].eq("dual") & np.isfinite(part["success_thr"]) & np.isfinite(part["failure_thr"])].copy()
    dual = dual[
        dual["abs_drop_pp"].le(max_abs_drop_pp)
        & dual["decision_accuracy_pct"].ge(min_acc_pct)
    ].copy()
    if dual.empty:
        return pd.DataFrame(), pd.DataFrame()
    dual = dual.sort_values(["success_thr", "failure_thr", "pct_steps_saved", "abs_drop_pp"], ascending=[True, True, False, True])
    best = dual.groupby(["success_thr", "failure_thr"], as_index=False).head(1)
    save = best.pivot_table(index="failure_thr", columns="success_thr", values="pct_steps_saved", aggfunc="mean").sort_index(ascending=False)
    drop = best.pivot_table(index="failure_thr", columns="success_thr", values="resolve_rate_drop_pp", aggfunc="mean").reindex(index=save.index, columns=save.columns)
    return save, drop


def plot_valid_grid_heatmaps(
    grid: pd.DataFrame,
    output_dir: Path,
    *,
    max_abs_drop_pp: float,
    min_acc_pct: float,
) -> list[Path]:
    plot_dir = output_dir / "plots" / "valid_grid_heatmaps"
    plot_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for (split, variant, score_mode), part in grid.groupby(["split", "variant", "score_mode"], sort=False):
        save, drop = _best_grid_for_heatmap(part, max_abs_drop_pp=max_abs_drop_pp, min_acc_pct=min_acc_pct)
        if save.empty:
            continue
        fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), sharey=True)
        for ax, values, title, cmap, norm, label in [
            (axes[0], save, "Best Valid Save% under constraints", "viridis", None, "Save %"),
            (
                axes[1],
                drop,
                "Drop pp of same selected cells",
                "coolwarm",
                TwoSlopeNorm(vmin=-max_abs_drop_pp, vcenter=0.0, vmax=max_abs_drop_pp),
                "Drop pp",
            ),
        ]:
            arr = values.to_numpy(dtype=float)
            im = ax.imshow(arr, aspect="auto", cmap=cmap, norm=norm)
            ax.set_title(title)
            ax.set_xticks(np.arange(len(values.columns)))
            ax.set_xticklabels([f"{col:.1f}" for col in values.columns])
            ax.set_yticks(np.arange(len(values.index)))
            ax.set_yticklabels([f"{idx:.1f}" for idx in values.index])
            ax.set_xlabel("success_thr")
            ax.set_ylabel("failure_thr")
            for y in range(arr.shape[0]):
                for x in range(arr.shape[1]):
                    val = arr[y, x]
                    if np.isnan(val):
                        continue
                    ax.text(x, y, f"{val:.1f}", ha="center", va="center", fontsize=8, color="white" if abs(val) > 35 else "black")
            fig.colorbar(im, ax=ax, shrink=0.82, label=label)
        fig.suptitle(
            f"Valid Dual Policy Grid — {split} / {_variant_label(variant)} / {score_mode}\n"
            f"constraints: |drop|≤{max_abs_drop_pp:g}pp, acc≥{min_acc_pct:g}%; best over min_step/k",
            y=1.02,
        )
        fig.tight_layout()
        path = plot_dir / f"valid_dual_grid_{_safe_name(split)}_{_safe_name(variant)}_{_safe_name(score_mode)}.png"
        fig.savefig(path, dpi=170, bbox_inches="tight")
        plt.close(fig)
        paths.append(path)
    return paths


def write_markdown(
    selected: pd.DataFrame,
    output_dir: Path,
    table_paths: dict[str, Path],
    plot_paths: list[Path],
    *,
    max_abs_drop_pp: float,
    min_acc_pct: float,
) -> Path:
    lines: list[str] = []
    lines.append("# Safe-Stop Dual-Head Visual Summary")
    lines.append("")
    lines.append("这些图表是 posthoc 汇总：只读取已经生成的 `safe_stop_*.csv`，不重新训练。")
    lines.append("Drop 口径：`original_resolve_rate - adjusted_resolve_rate`；正数表示 early stop 后 resolve rate 下降，负数表示被 FP-success 抬高。")
    lines.append("")
    lines.append("## Selected Policies")
    lines.append("")
    lines.append("| Split | Variant | Score | Policy | Test Save | Test Drop pp | Test Acc | Test Coverage | FN | FP |")
    lines.append("|:--|:--|:--|:--|--:|--:|--:|--:|--:|--:|")
    for _, row in selected.sort_values(["split", "variant", "score_mode"]).iterrows():
        lines.append(
            "| "
            f"{row['split']} | {_variant_label(row['variant'])} | {row['score_mode']} | `{row['policy']}` | "
            f"{_fmt(row['test_save_pct'])}% | {_fmt(row['test_drop_pp'])} | "
            f"{_fmt(row['test_acc_pct'])}% | {_fmt(row['test_coverage_pct'])}% | "
            f"{int(row['test_fn'])} | {int(row['test_fp'])} |"
        )
    lines.append("")
    lines.append("## How To Read")
    lines.append("")
    lines.append("- `selected_tradeoff_save_vs_drop.png`：越靠右越省，越靠近 0 越不伤 resolve rate。")
    lines.append("- `valid_to_test_drop_gap.png`：点越靠近虚线，valid 选出来的策略越能迁移到 test。")
    lines.append("- `selected_policy_fn_fp_counts.png`：top3 的主要风险是 FN，bottom3 的主要风险是 FP。")
    lines.append("- `valid_grid_heatmaps/`：只看 valid；约束为 `|drop|≤{:.1f}pp, acc≥{:.1f}%`，每格取最省步数的 min_step/k。".format(max_abs_drop_pp, min_acc_pct))
    lines.append("")
    lines.append("## Files")
    lines.append("")
    for name, path in table_paths.items():
        lines.append(f"- `{path.relative_to(output_dir)}`")
    for path in plot_paths:
        lines.append(f"- `{path.relative_to(output_dir)}`")
    summary_path = output_dir / "safe_stop_visual_summary.md"
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary_path


def main() -> int:
    args = parse_args()
    runs = find_runs(args.reports_root)
    if not runs:
        raise SystemExit(f"No safe-stop dual-head runs found under {args.reports_root}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    selected, grid, calibration = load_data(runs)
    table_paths = write_tables(
        selected,
        grid,
        calibration,
        args.output_dir,
        max_abs_drop_pp=args.max_abs_drop_pp,
        min_acc_pct=args.min_acc_pct,
    )
    plot_paths: list[Path] = []
    plot_paths.append(plot_selected_tradeoff(selected, args.output_dir))
    plot_paths.append(plot_selected_bars(selected, args.output_dir))
    plot_paths.append(plot_valid_test_gap(selected, args.output_dir))
    plot_paths.append(plot_fn_fp(selected, args.output_dir))
    plot_paths.append(plot_calibration(calibration, args.output_dir))
    plot_paths.extend(
        plot_valid_grid_heatmaps(
            grid,
            args.output_dir,
            max_abs_drop_pp=args.max_abs_drop_pp,
            min_acc_pct=args.min_acc_pct,
        )
    )
    summary_path = write_markdown(
        selected,
        args.output_dir,
        table_paths,
        plot_paths,
        max_abs_drop_pp=args.max_abs_drop_pp,
        min_acc_pct=args.min_acc_pct,
    )
    print(f"Saved safe-stop dual-head visual summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
