#!/usr/bin/env python3
"""Plot per-agent threshold curves for dual-head conjunctive gates across splits."""

from __future__ import annotations

import os
import re
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".mplconfig"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

import dual_head_conjunctive_gate_valid_test_posthoc as dh


REPORTS_DIR = dh.REPORTS_DIR
BASE_DIR = (
    REPORTS_DIR
    / "safe_stop_dual_head_visual_summary"
    / "problem_diagnosis"
    / "dual_head_conjunctive_gate_posthoc"
)
OUT_DIR = BASE_DIR / "threshold_agent_curves_all_splits"
THRESHOLDS = dh.THRESHOLDS
SPLITS = ["top3", "mid3", "bottom3"]


def short_agent_name(value: str) -> str:
    patterns = [
        ("gpt-5.2-codex", "gpt-5.2-codex"),
        ("gpt-5-2-codex", "gpt-5.2-codex"),
        ("gpt-5-codex", "gpt-5-codex"),
        ("gpt-5-nano", "gpt-5-nano"),
        ("claude", "claude"),
        ("gemini", "gemini"),
        ("minimax-m2", "minimax-m2"),
        ("deepseek-v3.2", "deepseek-v3.2"),
        ("glm-4.5", "glm-4.5"),
        ("devstral", "devstral-2512"),
        ("qwen", "qwen"),
        ("kimi", "kimi"),
    ]
    lower = value.lower()
    for needle, label in patterns:
        if needle in lower:
            return label
    cleaned = re.sub(r"^\\d+_mini-v[\\d.]+_", "", value)
    return cleaned[-36:]


def make_rows() -> pd.DataFrame:
    selected = pd.read_csv(BASE_DIR / "conjunctive_valid_selected.csv")
    selected = selected[selected["split"].isin(SPLITS)].copy()

    rows: list[dict[str, object]] = []
    for _, selected_row in selected.iterrows():
        run_dir = REPORTS_DIR / str(selected_row["run"])
        test_path = run_dir / "test_predictions_safe_stop.parquet"
        test_df = pd.read_parquet(test_path)

        prefix_model = str(selected_row["prefix_model"])
        success_col, failure_col = dh.score_columns(prefix_model, str(selected_row["score_mode"]))
        records = dh.load_records(test_df, success_col, failure_col)

        by_agent: dict[str, list[dict]] = {}
        for record in records:
            by_agent.setdefault(str(record["agent_model"]), []).append(record)

        for threshold in THRESHOLDS:
            policy = {
                "threshold": float(threshold),
                "min_step": int(selected_row["min_step"]),
                "consecutive": int(selected_row["consecutive"]),
            }
            for agent_model, agent_records in sorted(by_agent.items()):
                metrics = dh.evaluate_policy(agent_records, policy, mode="conj")
                rows.append(
                    {
                        "split": str(selected_row["split"]),
                        "variant": str(selected_row["variant"]),
                        "strategy": str(selected_row.get("strategy", "baseline")),
                        "score_mode": str(selected_row["score_mode"]),
                        "prefix_model": prefix_model,
                        "agent_model": agent_model,
                        "agent_short": short_agent_name(agent_model),
                        "threshold": float(threshold),
                        "min_step": int(selected_row["min_step"]),
                        "consecutive": int(selected_row["consecutive"]),
                        "original_resolve_rate_pct": metrics["original_resolve_rate"] * 100.0,
                        "adjusted_resolve_rate_pct": metrics["adjusted_resolve_rate"] * 100.0,
                        "resolve_delta_pp": (metrics["adjusted_resolve_rate"] - metrics["original_resolve_rate"]) * 100.0,
                        "drop_pp": metrics["drop_pp"],
                        "decision_accuracy_pct": metrics["decision_accuracy"] * 100.0,
                        "coverage_pct": metrics["coverage"] * 100.0,
                        "pct_steps_saved": metrics["pct_steps_saved"],
                        "margin_auc": metrics["margin_auc"],
                        "false_positives": int(metrics["false_positives"]),
                        "false_negatives": int(metrics["false_negatives"]),
                        "n_decided": int(metrics["n_decided"]),
                        "original_total": int(metrics["original_total"]),
                    }
                )
    return pd.DataFrame(rows)


def plot_group(group: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(9.5, 10.5), sharex=True)
    metrics = [
        ("decision_accuracy_pct", "Decision accuracy (%)", None),
        ("resolve_delta_pp", "Resolve-rate delta: adjusted - original (pp)", 0.0),
        ("pct_steps_saved", "Steps saved (%)", None),
    ]
    for axis, (column, ylabel, hline) in zip(axes, metrics):
        for agent_short, agent_group in group.groupby("agent_short", sort=True):
            agent_group = agent_group.sort_values("threshold")
            axis.plot(
                agent_group["threshold"],
                agent_group[column],
                marker="o",
                linewidth=2,
                label=agent_short,
            )
        if hline is not None:
            axis.axhline(hline, color="#444444", linestyle="--", linewidth=1)
        axis.grid(True, alpha=0.28)
        axis.set_ylabel(ylabel)
    title_row = group.iloc[0]
    axes[0].set_title(
        f"{title_row['split']} {title_row['variant']} {title_row['score_mode']} "
        f"{title_row['strategy']} "
        f"(fixed min={int(title_row['min_step'])}, k={int(title_row['consecutive'])})"
    )
    axes[-1].set_xlabel("Conjunctive threshold")
    axes[0].legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def append_snapshot(lines: list[str], curves: pd.DataFrame, split: str, threshold: float) -> None:
    subset = curves[curves["split"].eq(split) & curves["threshold"].eq(threshold)].copy()
    subset = subset.sort_values(["variant", "score_mode", "agent_short"])
    lines += [
        "",
        f"## {split} Threshold {threshold:.2f}",
        "",
        "| Variant | Strategy | Score | Agent | Acc | Resolve Δ | Save | Coverage | FP | FN |",
        "|:--|:--|:--|:--|--:|--:|--:|--:|--:|--:|",
    ]
    for _, row in subset.iterrows():
        lines.append(
            f"| {row['variant']} | {row['strategy']} | {row['score_mode']} | {row['agent_short']} | "
            f"{row['decision_accuracy_pct']:.1f}% | {row['resolve_delta_pp']:+.2f}pp | "
            f"{row['pct_steps_saved']:.1f}% | {row['coverage_pct']:.1f}% | "
            f"{int(row['false_positives'])} | {int(row['false_negatives'])} |"
        )


def write_report(curves: pd.DataFrame, image_paths: list[Path]) -> Path:
    work = curves.copy()
    work["abs_resolve_delta_pp"] = work["resolve_delta_pp"].abs()
    best_rows = []
    for (split, variant, strategy, score_mode, agent_short), group in work.groupby(
        ["split", "variant", "strategy", "score_mode", "agent_short"]
    ):
        candidate = group[group["coverage_pct"].gt(0)].copy()
        if candidate.empty:
            candidate = group.copy()
        best = candidate.sort_values(
            ["abs_resolve_delta_pp", "decision_accuracy_pct", "pct_steps_saved"],
            ascending=[True, False, False],
        ).iloc[0]
        best_rows.append(best)
    best_table = pd.DataFrame(best_rows).sort_values(["split", "variant", "strategy", "score_mode", "agent_short"])

    lines = [
        "# Dual-Head Threshold Agent Curves Across Splits",
        "",
        'Public-release English note.',
        "",
        'Public-release English note.',
        "",
        "## Images",
        "",
    ]
    for path in image_paths:
        lines.append(f"- `{path.name}`")
    lines.append("")
    for path in image_paths:
        lines += [f"### {path.stem}", "", f"![{path.stem}]({path.name})", ""]

    for split in SPLITS:
        append_snapshot(lines, curves, split, 0.95)

    lines += [
        "",
        "## Closest-to-Zero Resolve Delta",
        "",
        "| Split | Variant | Strategy | Score | Agent | Thr | Acc | Resolve Δ | Save | Coverage | FP | FN |",
        "|:--|:--|:--|:--|:--|--:|--:|--:|--:|--:|--:|--:|",
    ]
    for _, row in best_table.iterrows():
        lines.append(
            f"| {row['split']} | {row['variant']} | {row['strategy']} | {row['score_mode']} | {row['agent_short']} | "
            f"{row['threshold']:.2f} | {row['decision_accuracy_pct']:.1f}% | "
            f"{row['resolve_delta_pp']:+.2f}pp | {row['pct_steps_saved']:.1f}% | "
            f"{row['coverage_pct']:.1f}% | {int(row['false_positives'])} | {int(row['false_negatives'])} |"
        )

    lines += [
        "",
        "## Files",
        "",
        "- `threshold_agent_curves_all_splits.csv`",
    ]
    report_path = OUT_DIR / "threshold_agent_curves_all_splits_report.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    curves = make_rows()
    curves.to_csv(OUT_DIR / "threshold_agent_curves_all_splits.csv", index=False)

    image_paths: list[Path] = []
    for (split, variant, strategy, score_mode), group in curves.groupby(["split", "variant", "strategy", "score_mode"], sort=True):
        strategy_tag = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(strategy)).strip("_")
        output_path = OUT_DIR / f"{split}_{str(variant).lower()}_{strategy_tag}_{score_mode}_threshold_agent_curves.png"
        plot_group(group, output_path)
        image_paths.append(output_path)

    report_path = write_report(curves, image_paths)
    print(report_path)


if __name__ == "__main__":
    main()
