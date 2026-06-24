#!/usr/bin/env python3
"""Generate readable tables and plots for min-step/consecutive policy sweeps."""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".mplconfig"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D


RUN_LABELS = {
    "per_instance_model_valid3_retrain": "mid3",
    "per_instance_model_valid3_top3_retrain": "top3",
    "per_instance_model_valid3_bottom3_retrain": "bottom3",
}

SCORE_LABELS = {
    "raw": "raw",
    "prefix_calibrated": "prefix-cal",
    "trajectory_calibrated": "traj-cal",
}

MODEL_LABELS = {
    "I_LightGBM_Dense_AF": "I",
    "J_LightGBM_Dense_AF_Thought": "J",
    "Abl_NoTaskSignal_LightGBM": "NoTask",
    "Abl_NoTaskSignal_NoGoldAnswer_LightGBM": "NoTask+NoGold",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    default_root = (
        Path(__file__).resolve().parent
        / "runs"
        / "model_holdout_answer_calibrated_full"
        / "reports"
        / "decision_policy_minstep_consecutive_focus"
    )
    parser.add_argument("--input", type=Path, default=default_root / "policy_sweep_aggregate.csv")
    parser.add_argument("--output-dir", type=Path, default=default_root / "visual_summary")
    return parser.parse_args()


def _safe_name(name: str) -> str:
    return (
        name.replace("+", "plus")
        .replace("/", "_")
        .replace(" ", "_")
        .replace(":", "_")
    )


def _prepare(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["run_short"] = out["run"].map(RUN_LABELS).fillna(out["run"])
    out["score_short"] = out["score_mode"].map(SCORE_LABELS).fillna(out["score_mode"])
    out["model_short"] = out["prefix_model"].map(MODEL_LABELS).fillna(out["prefix_model"])
    for col in ["resolve_rate_drop", "pct_steps_saved", "coverage", "decision_accuracy"]:
        out[f"{col}_pp"] = out[col] * 100.0 if col != "pct_steps_saved" else out[col]
    out["abs_drop_pp"] = out["resolve_rate_drop_pp"].abs()
    out["policy"] = "min" + out["min_step"].astype(str) + "_k" + out["consecutive"].astype(str)
    return out


def _fmt(value: float, digits: int = 1) -> str:
    if value is None or math.isnan(float(value)):
        return "-"
    return f"{float(value):.{digits}f}"


def write_tables(df: pd.DataFrame, output_dir: Path) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    readable_cols = [
        "run_short",
        "score_short",
        "model_short",
        "threshold",
        "min_step",
        "consecutive",
        "coverage_pp",
        "decision_accuracy_pp",
        "pct_steps_saved",
        "resolve_rate_drop_pp",
        "abs_drop_pp",
        "false_negatives",
        "false_positives",
        "adjusted_resolve_rate",
        "original_resolve_rate",
    ]
    readable = df[readable_cols].sort_values(
        ["run_short", "score_short", "model_short", "threshold", "min_step", "consecutive"]
    )
    readable_path = output_dir / "policy_sweep_readable_percent.csv"
    readable.to_csv(readable_path, index=False)

    baseline = df[(df["min_step"] == 0) & (df["consecutive"] == 1)].copy()
    gated = df[(df["min_step"] == 10) & (df["consecutive"] == 3)].copy()
    key = ["run", "score_mode", "prefix_model", "threshold"]
    compare = baseline.merge(
        gated,
        on=key,
        suffixes=("_baseline", "_min10_k3"),
    )
    rows = []
    for _, row in compare.iterrows():
        rows.append(
            {
                "run": RUN_LABELS.get(row["run"], row["run"]),
                "score": SCORE_LABELS.get(row["score_mode"], row["score_mode"]),
                "model": MODEL_LABELS.get(row["prefix_model"], row["prefix_model"]),
                "threshold": row["threshold"],
                "drop_baseline_pp": row["resolve_rate_drop_pp_baseline"],
                "drop_min10_k3_pp": row["resolve_rate_drop_pp_min10_k3"],
                "drop_abs_improvement_pp": row["abs_drop_pp_baseline"] - row["abs_drop_pp_min10_k3"],
                "save_baseline_pct": row["pct_steps_saved_baseline"],
                "save_min10_k3_pct": row["pct_steps_saved_min10_k3"],
                "coverage_baseline_pct": row["coverage_pp_baseline"],
                "coverage_min10_k3_pct": row["coverage_pp_min10_k3"],
                "acc_baseline_pct": row["decision_accuracy_pp_baseline"],
                "acc_min10_k3_pct": row["decision_accuracy_pp_min10_k3"],
                "fn_baseline": row["false_negatives_baseline"],
                "fn_min10_k3": row["false_negatives_min10_k3"],
                "fp_baseline": row["false_positives_baseline"],
                "fp_min10_k3": row["false_positives_min10_k3"],
            }
        )
    compare_df = pd.DataFrame(rows).sort_values(["run", "score", "model", "threshold"])
    compare_path = output_dir / "baseline_vs_min10_k3.csv"
    compare_df.to_csv(compare_path, index=False)

    best = df[df["pct_steps_saved"] >= 20].copy()
    best = best.sort_values(
        ["run_short", "score_short", "model_short", "threshold", "abs_drop_pp", "pct_steps_saved"],
        ascending=[True, True, True, True, True, False],
    )
    best = best.groupby(["run_short", "score_short", "model_short", "threshold"], as_index=False).head(1)
    best_path = output_dir / "best_absdrop_with_save_ge20.csv"
    best[
        [
            "run_short",
            "score_short",
            "model_short",
            "threshold",
            "min_step",
            "consecutive",
            "coverage_pp",
            "decision_accuracy_pp",
            "pct_steps_saved",
            "resolve_rate_drop_pp",
            "false_negatives",
            "false_positives",
        ]
    ].to_csv(best_path, index=False)
    return {
        "readable": readable_path,
        "compare": compare_path,
        "best": best_path,
    }


def plot_tradeoff(df: pd.DataFrame, output_dir: Path) -> list[Path]:
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    colors = {0.80: "#1f77b4", 0.85: "#ff7f0e", 0.90: "#2ca02c"}
    markers = {1: "o", 2: "s", 3: "^"}
    runs = ["mid3", "top3", "bottom3"]
    scores = ["raw", "prefix-cal", "traj-cal"]

    for model_short, model_df in df.groupby("model_short", sort=False):
        fig, axes = plt.subplots(len(runs), len(scores), figsize=(15, 11), sharex=True, sharey=True)
        for row_idx, run in enumerate(runs):
            for col_idx, score in enumerate(scores):
                ax = axes[row_idx, col_idx]
                part = model_df[(model_df["run_short"] == run) & (model_df["score_short"] == score)]
                for _, item in part.iterrows():
                    threshold = round(float(item["threshold"]), 2)
                    consecutive = int(item["consecutive"])
                    min_step = int(item["min_step"])
                    ax.scatter(
                        item["pct_steps_saved"],
                        item["resolve_rate_drop_pp"],
                        s=35 + min_step * 4,
                        marker=markers.get(consecutive, "o"),
                        color=colors.get(threshold, "gray"),
                        alpha=0.72,
                        edgecolor="white",
                        linewidth=0.5,
                    )
                ax.axhline(0, color="black", linewidth=0.8)
                ax.axhspan(-2, 2, color="gray", alpha=0.08)
                ax.grid(True, alpha=0.25)
                if row_idx == 0:
                    ax.set_title(score)
                if col_idx == 0:
                    ax.set_ylabel(f"{run}\nDrop pp")
                if row_idx == len(runs) - 1:
                    ax.set_xlabel("Save %")
        threshold_handles = [
            Line2D([0], [0], marker="o", color="w", markerfacecolor=color, label=f"thr={thr:.2f}", markersize=8)
            for thr, color in colors.items()
        ]
        k_handles = [
            Line2D([0], [0], marker=marker, color="gray", linestyle="None", label=f"k={k}", markersize=8)
            for k, marker in markers.items()
        ]
        fig.legend(handles=threshold_handles + k_handles, loc="lower center", ncol=6)
        fig.suptitle(
            f"Tradeoff: Save% vs Resolve-Rate Drop - {model_short}\n"
            "negative Drop = adjusted resolve rate is higher than original; point size grows with min_step",
            y=0.98,
        )
        fig.tight_layout(rect=[0, 0.05, 1, 0.94])
        path = plot_dir / f"tradeoff_save_vs_drop_{_safe_name(model_short)}.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        paths.append(path)
    return paths


def plot_policy_lines(df: pd.DataFrame, output_dir: Path) -> list[Path]:
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    runs = ["mid3", "top3", "bottom3"]
    scores = ["raw", "prefix-cal", "traj-cal"]
    policies = {
        "baseline min0/k1": (0, 1, "#d62728"),
        "gated min10/k3": (10, 3, "#1f77b4"),
    }
    for model_short, model_df in df.groupby("model_short", sort=False):
        fig, axes = plt.subplots(len(runs), len(scores), figsize=(15, 10), sharex=True, sharey=True)
        for row_idx, run in enumerate(runs):
            for col_idx, score in enumerate(scores):
                ax = axes[row_idx, col_idx]
                for label, (min_step, consecutive, color) in policies.items():
                    part = model_df[
                        (model_df["run_short"] == run)
                        & (model_df["score_short"] == score)
                        & (model_df["min_step"] == min_step)
                        & (model_df["consecutive"] == consecutive)
                    ].sort_values("threshold")
                    if part.empty:
                        continue
                    ax.plot(
                        part["threshold"],
                        part["resolve_rate_drop_pp"],
                        marker="o",
                        color=color,
                        label=label,
                    )
                ax.axhline(0, color="black", linewidth=0.8)
                ax.axhspan(-2, 2, color="gray", alpha=0.08)
                ax.grid(True, alpha=0.25)
                if row_idx == 0:
                    ax.set_title(score)
                if col_idx == 0:
                    ax.set_ylabel(f"{run}\nDrop pp")
                if row_idx == len(runs) - 1:
                    ax.set_xlabel("Threshold")
        fig.legend(loc="lower center", ncol=2)
        fig.suptitle(f"Drop change by threshold - {model_short}", y=0.98)
        fig.tight_layout(rect=[0, 0.05, 1, 0.94])
        path = plot_dir / f"drop_lines_baseline_vs_min10k3_{_safe_name(model_short)}.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        paths.append(path)
    return paths


def plot_heatmaps(df: pd.DataFrame, output_dir: Path) -> list[Path]:
    plot_dir = output_dir / "plots" / "drop_heatmaps"
    plot_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    max_abs = float(np.nanmax(np.abs(df["resolve_rate_drop_pp"])))
    max_abs = max(5.0, math.ceil(max_abs))
    norm = TwoSlopeNorm(vmin=-max_abs, vcenter=0.0, vmax=max_abs)
    for (run, score, model_short), part in df.groupby(["run_short", "score_short", "model_short"], sort=False):
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.9), sharey=True)
        for idx, consecutive in enumerate([1, 2, 3]):
            ax = axes[idx]
            sub = part[part["consecutive"] == consecutive]
            pivot = sub.pivot_table(
                index="threshold",
                columns="min_step",
                values="resolve_rate_drop_pp",
                aggfunc="mean",
            ).sort_index(ascending=False)
            values = pivot.to_numpy()
            im = ax.imshow(values, cmap="coolwarm", norm=norm, aspect="auto")
            ax.set_title(f"k={consecutive}")
            ax.set_xticks(np.arange(len(pivot.columns)))
            ax.set_xticklabels([str(int(x)) for x in pivot.columns])
            ax.set_yticks(np.arange(len(pivot.index)))
            ax.set_yticklabels([f"{x:.2f}" for x in pivot.index])
            ax.set_xlabel("min_step")
            if idx == 0:
                ax.set_ylabel("threshold")
            for y in range(values.shape[0]):
                for x in range(values.shape[1]):
                    val = values[y, x]
                    if np.isnan(val):
                        continue
                    ax.text(x, y, f"{val:.1f}", ha="center", va="center", fontsize=8)
        fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.82, label="Drop pp")
        fig.suptitle(f"Drop heatmap - {run} / {score} / {model_short}")
        fig.tight_layout(rect=[0, 0, 0.92, 0.9])
        path = plot_dir / f"drop_heatmap_{_safe_name(run)}_{_safe_name(score)}_{_safe_name(model_short)}.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        paths.append(path)
    return paths


def write_markdown(df: pd.DataFrame, output_dir: Path, plot_paths: list[Path], table_paths: dict[str, Path]) -> None:
    lines: list[str] = []
    lines.append("# Policy Sweep Visual Summary")
    lines.append("")
    lines.append('Public-release English note.')
    lines.append('Public-release English note.')
    lines.append("")
    lines.append("## Files")
    for name, path in table_paths.items():
        lines.append(f"- `{path.relative_to(output_dir)}`")
    lines.append('Public-release English note.')
    lines.append('Public-release English note.')
    lines.append('Public-release English note.')
    lines.append("")

    lines.append("## Recommended Quick Read")
    focus = df[
        (df["prefix_model"] == "J_LightGBM_Dense_AF_Thought")
        & (df["threshold"].isin([0.8, 0.85, 0.9]))
        & (((df["min_step"] == 0) & (df["consecutive"] == 1)) | ((df["min_step"] == 10) & (df["consecutive"] == 3)))
    ].copy()
    focus["policy"] = np.where(focus["min_step"].eq(0), "baseline", "min10/k3")
    for run in ["mid3", "top3", "bottom3"]:
        lines.append(f"### J model - {run}")
        sub = focus[focus["run_short"] == run].sort_values(["score_short", "threshold", "policy"])
        lines.append("| Score | Thr | Policy | Coverage | Acc | Save% | Drop pp | FN | FP |")
        lines.append("|:--|--:|:--|--:|--:|--:|--:|--:|--:|")
        for _, row in sub.iterrows():
            lines.append(
                "| "
                f"{row['score_short']} | {row['threshold']:.2f} | {row['policy']} | "
                f"{_fmt(row['coverage_pp'])}% | {_fmt(row['decision_accuracy_pp'])}% | "
                f"{_fmt(row['pct_steps_saved'])}% | {_fmt(row['resolve_rate_drop_pp'])} | "
                f"{int(row['false_negatives'])} | {int(row['false_positives'])} |"
            )
        lines.append("")

    lines.append("## Plot Index")
    for path in plot_paths:
        lines.append(f"- `{path.relative_to(output_dir)}`")
    (output_dir / "visual_policy_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    df = pd.read_csv(args.input)
    prepared = _prepare(df)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    table_paths = write_tables(prepared, args.output_dir)
    plot_paths = []
    plot_paths.extend(plot_tradeoff(prepared, args.output_dir))
    plot_paths.extend(plot_policy_lines(prepared, args.output_dir))
    plot_paths.extend(plot_heatmaps(prepared, args.output_dir))
    write_markdown(prepared, args.output_dir, plot_paths, table_paths)
    print(f"Saved visual summary: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
