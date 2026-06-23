#!/usr/bin/env python3
"""Plot bottom3 per-agent threshold curves for dual-head conjunctive gates."""

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
OUT_DIR = BASE_DIR / "bottom3_threshold_agent_curves"
THRESHOLDS = dh.THRESHOLDS


def short_agent_name(value: str) -> str:
    if "gpt-5-nano" in value:
        return "gpt-5-nano"
    if "glm-4.5" in value:
        return "glm-4.5"
    if "devstral" in value:
        return "devstral-2512"
    return re.sub(r"^\\d+_mini-v[\\d.]+_", "", value)


def make_rows() -> pd.DataFrame:
    selected = pd.read_csv(BASE_DIR / "conjunctive_valid_selected.csv")
    selected = selected[selected["split"].eq("bottom3")].copy()

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
                        "split": "bottom3",
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
        f"bottom3 {title_row['variant']} {title_row['score_mode']} "
        f"{title_row['strategy']} "
        f"(fixed min={int(title_row['min_step'])}, k={int(title_row['consecutive'])})"
    )
    axes[-1].set_xlabel("Conjunctive threshold")
    axes[0].legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def write_report(curves: pd.DataFrame, image_paths: list[Path]) -> Path:
    lines = [
        "# Bottom3 Threshold Agent Curves",
        "",
        "口径：dual-head conjunctive gate；每个 `variant + score_mode` 固定 validation 选出的 `min_step/k`，只扫 threshold。",
        "",
        "`resolve_delta_pp = adjusted_resolve_rate - original_resolve_rate`；越接近 0 越说明离线估计下对真实 resolve rate 的扰动越小。",
        "",
        "## Images",
        "",
    ]
    for path in image_paths:
        lines.append(f"- `{path.name}`")
    lines.append("")
    for path in image_paths:
        lines += [f"### {path.stem}", "", f"![{path.stem}]({path.name})", ""]

    best_rows = []
    work = curves.copy()
    work["abs_resolve_delta_pp"] = work["resolve_delta_pp"].abs()
    for (variant, strategy, score_mode, agent_short), group in work.groupby(["variant", "strategy", "score_mode", "agent_short"]):
        best = group.sort_values(
            ["abs_resolve_delta_pp", "decision_accuracy_pct", "pct_steps_saved"],
            ascending=[True, False, False],
        ).iloc[0]
        best_rows.append(best)
    best_table = pd.DataFrame(best_rows).sort_values(["variant", "strategy", "score_mode", "agent_short"])

    lines += [
        "",
        "## Closest-to-Zero Resolve Delta",
        "",
        "| Variant | Strategy | Score | Agent | Thr | Acc | Resolve Δ | Save | Coverage | FP | FN |",
        "|:--|:--|:--|:--|--:|--:|--:|--:|--:|--:|--:|",
    ]
    for _, row in best_table.iterrows():
        lines.append(
            f"| {row['variant']} | {row['strategy']} | {row['score_mode']} | {row['agent_short']} | {row['threshold']:.2f} | "
            f"{row['decision_accuracy_pct']:.1f}% | {row['resolve_delta_pp']:+.2f}pp | "
            f"{row['pct_steps_saved']:.1f}% | {row['coverage_pct']:.1f}% | "
            f"{int(row['false_positives'])} | {int(row['false_negatives'])} |"
        )

    lines += [
        "",
        "## Snapshot: threshold 0.85 / 0.90",
        "",
        "| Variant | Strategy | Score | Agent | Thr | Acc | Resolve Δ | Save | Coverage | FP | FN |",
        "|:--|:--|:--|:--|--:|--:|--:|--:|--:|--:|--:|",
    ]
    snapshot = curves[curves["threshold"].isin([0.85, 0.90])].copy()
    snapshot = snapshot.sort_values(["variant", "score_mode", "agent_short", "threshold"])
    for _, row in snapshot.iterrows():
        lines.append(
            f"| {row['variant']} | {row['strategy']} | {row['score_mode']} | {row['agent_short']} | {row['threshold']:.2f} | "
            f"{row['decision_accuracy_pct']:.1f}% | {row['resolve_delta_pp']:+.2f}pp | "
            f"{row['pct_steps_saved']:.1f}% | {row['coverage_pct']:.1f}% | "
            f"{int(row['false_positives'])} | {int(row['false_negatives'])} |"
        )
    lines += [
        "",
        "## Files",
        "",
        "- `bottom3_threshold_agent_curves.csv`",
    ]
    report_path = OUT_DIR / "bottom3_threshold_agent_curves_report.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    curves = make_rows()
    csv_path = OUT_DIR / "bottom3_threshold_agent_curves.csv"
    curves.to_csv(csv_path, index=False)

    image_paths: list[Path] = []
    for (variant, strategy, score_mode), group in curves.groupby(["variant", "strategy", "score_mode"], sort=True):
        strategy_tag = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(strategy)).strip("_")
        output_path = OUT_DIR / f"bottom3_{str(variant).lower()}_{strategy_tag}_{score_mode}_threshold_agent_curves.png"
        plot_group(group, output_path)
        image_paths.append(output_path)

    report_path = write_report(curves, image_paths)
    print(report_path)


if __name__ == "__main__":
    main()
