#!/usr/bin/env python3
"""Summarize valid-only rate-preserving threshold sweeps.

This script does not train models and does not recompute predictions.  It reads
previously generated ``valid_sweep.csv`` / ``test_sweep.csv`` files, selects
thresholds on validation only, then reports the corresponding test metrics.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_RUN_NAME = "model_holdout_answer_calibrated_full"
DEFAULT_SOURCES = [
    "calibrated_grid025=asymmetric_valid_threshold_tuning_fine",
    "calibrated_grid001=asymmetric_valid_threshold_tuning_fine_calibrated_step001_rate",
    "raw_grid025=asymmetric_valid_threshold_tuning_fine_raw",
    "raw_grid001=asymmetric_valid_threshold_tuning_fine_raw_step001",
]
DEFAULT_TOLERANCES = [0.005, 0.01, 0.02]
MODEL_SHORT = {
    "J_LightGBM_Dense_AF_Thought": "J",
    "I_LightGBM_Dense_AF": "I",
    "H_LightGBM_Dense": "H",
    "K_LightGBM_Dense_Full": "K",
    "G_TfIdf_Full_LR": "G",
    "D_Dense_Full_LR": "D",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", default=DEFAULT_RUN_NAME)
    parser.add_argument("--output-subdir", default="threshold_tuning_summary")
    parser.add_argument(
        "--sources",
        nargs="+",
        default=DEFAULT_SOURCES,
        help="Entries of the form label=report_subdir. Missing subdirs are skipped.",
    )
    parser.add_argument(
        "--rate-tolerances",
        nargs="+",
        type=float,
        default=DEFAULT_TOLERANCES,
        help="Absolute valid rate-delta tolerances, e.g. 0.01 means 1pp.",
    )
    return parser.parse_args()


def _run_dir(run_name: str) -> Path:
    return PROJECT_ROOT / "runs" / run_name


def _parse_source(value: str) -> tuple[str, str]:
    if "=" not in value:
        return value, value
    label, subdir = value.split("=", 1)
    return label.strip(), subdir.strip()


def _policy_name(tolerance: float) -> str:
    pp = tolerance * 100.0
    if abs(pp - round(pp)) < 1e-9:
        return f"rate_{int(round(pp))}pp"
    return f"rate_{str(pp).replace('.', 'p')}pp"


def _read_sweep(source_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame] | None:
    valid_path = source_dir / "valid_sweep.csv"
    test_path = source_dir / "test_sweep.csv"
    if not valid_path.is_file() or not test_path.is_file():
        return None
    valid = pd.read_csv(valid_path)
    test = pd.read_csv(test_path)
    return valid, test


def _select_one(valid_group: pd.DataFrame, tolerance: float) -> tuple[pd.Series, bool]:
    candidates = valid_group[valid_group["rate_delta"].abs() <= tolerance].copy()
    fallback = False
    if candidates.empty:
        candidates = valid_group.copy()
        candidates["_abs_rate_delta"] = candidates["rate_delta"].abs()
        sort_cols = ["_abs_rate_delta", "pct_steps_saved", "decision_accuracy", "decision_rate"]
        ascending = [True, False, False, False]
        fallback = True
    else:
        candidates["_abs_rate_delta"] = candidates["rate_delta"].abs()
        sort_cols = ["pct_steps_saved", "decision_accuracy", "decision_rate", "_abs_rate_delta"]
        ascending = [False, False, False, True]
    return candidates.sort_values(sort_cols, ascending=ascending).iloc[0], fallback


def _test_match(test: pd.DataFrame, row: pd.Series) -> pd.Series:
    mask = (
        (test["prefix_model"] == row["prefix_model"])
        & np.isclose(test["success_threshold"], float(row["success_threshold"]), atol=1e-9)
        & np.isclose(test["failure_threshold"], float(row["failure_threshold"]), atol=1e-9)
    )
    matches = test.loc[mask]
    if matches.empty:
        raise RuntimeError(
            "Missing test match for "
            f"{row['prefix_model']} ThrS={row['success_threshold']} ThrF={row['failure_threshold']}"
        )
    return matches.iloc[0]


def _get(row: pd.Series, name: str) -> Any:
    return row[name] if name in row.index else np.nan


def _select_source(label: str, source_dir: Path, tolerances: list[float]) -> pd.DataFrame:
    sweeps = _read_sweep(source_dir)
    if sweeps is None:
        return pd.DataFrame()
    valid, test = sweeps
    rows = []
    for prefix_model, group in valid.groupby("prefix_model", sort=False):
        for tolerance in tolerances:
            valid_row, fallback = _select_one(group, tolerance)
            test_row = _test_match(test, valid_row)
            rows.append(
                {
                    "source": label,
                    "source_dir": source_dir.name,
                    "prefix_model": prefix_model,
                    "model": MODEL_SHORT.get(prefix_model, prefix_model),
                    "policy": _policy_name(tolerance),
                    "valid_abs_rate_delta_limit": tolerance,
                    "selection_fallback": bool(fallback),
                    "success_threshold": float(valid_row["success_threshold"]),
                    "failure_threshold": float(valid_row["failure_threshold"]),
                    "valid_rate_delta": float(valid_row["rate_delta"]),
                    "valid_pct_steps_saved": float(valid_row["pct_steps_saved"]),
                    "valid_decision_accuracy": float(valid_row["decision_accuracy"]),
                    "valid_precision_success": float(_get(valid_row, "precision_success")),
                    "valid_precision_failure": float(_get(valid_row, "precision_failure")),
                    "valid_n_decided": int(valid_row["n_decided"]),
                    "valid_total": int(valid_row["total"]),
                    "test_rate_delta": float(test_row["rate_delta"]),
                    "test_pct_steps_saved": float(test_row["pct_steps_saved"]),
                    "test_decision_accuracy": float(test_row["decision_accuracy"]),
                    "test_precision_success": float(_get(test_row, "precision_success")),
                    "test_precision_failure": float(_get(test_row, "precision_failure")),
                    "test_false_positives": int(test_row["false_positives"]),
                    "test_false_negatives": int(test_row["false_negatives"]),
                    "test_n_decided": int(test_row["n_decided"]),
                    "test_total": int(test_row["total"]),
                }
            )
    return pd.DataFrame(rows)


def _fmt_pct(value: Any, digits: int = 1, signed: bool = False) -> str:
    try:
        if pd.isna(value):
            return "NA"
        prefix = "+" if signed and float(value) > 0 else ""
        return f"{prefix}{100.0 * float(value):.{digits}f}%"
    except Exception:
        return "NA"


def _fmt_num(value: Any, digits: int = 3) -> str:
    try:
        if pd.isna(value):
            return "NA"
        return f"{float(value):.{digits}f}"
    except Exception:
        return "NA"


def _write_markdown(output_dir: Path, selected: pd.DataFrame, metadata: dict[str, Any]) -> None:
    lines = [
        "# Rate-Preserving Threshold Summary",
        "",
        "Selection rule: choose thresholds on valid only; require `abs(valid ΔRate)` within the policy tolerance, then maximize `valid Save`.",
        "If no pair satisfies the tolerance, choose the closest valid ΔRate pair and mark `fallback=true`.",
        "Test rows are only the result of applying the valid-selected thresholds to heldout test.",
        "",
        "## Generated Sources",
        "",
        "| Source | Directory | Rows |",
        "|---|---|---:|",
    ]
    for item in metadata["sources"]:
        lines.append(
            f"| `{item['label']}` | `{item['subdir']}` | `{item['selected_rows']}` |"
        )
    lines.extend(
        [
            "",
            "## Main Policy Tables",
            "",
        ]
    )
    if selected.empty:
        lines.append("No completed sweep sources found.")
    else:
        source_order = {name: idx for idx, name in enumerate(selected["source"].drop_duplicates())}
        model_order = {"J": 0, "I": 1, "H": 2, "K": 3, "G": 4, "D": 5}
        selected = selected.assign(
            _source_order=selected["source"].map(source_order),
            _model_order=selected["model"].map(model_order).fillna(99),
        )
        for source, source_df in selected.sort_values(["_source_order", "_model_order", "policy"]).groupby("source", sort=False):
            lines.extend(
                [
                    f"### {source}",
                    "",
                    "| Model | Policy | ThrS | ThrF | Valid ΔRate | Valid Save | Valid Acc | Valid PrecS | Valid PrecF | Test ΔRate | Test Save | Test Acc | Test PrecS | Test PrecF | FP | FN | N | Fallback |",
                    "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
                ]
            )
            for _, row in source_df.iterrows():
                lines.append(
                    f"| `{row['model']}` | `{row['policy']}` | "
                    f"`{row['success_threshold']:.3f}` | `{row['failure_threshold']:.3f}` | "
                    f"`{_fmt_pct(row['valid_rate_delta'], signed=True)}` | "
                    f"`{_fmt_pct(row['valid_pct_steps_saved'])}` | "
                    f"`{_fmt_pct(row['valid_decision_accuracy'])}` | "
                    f"`{_fmt_pct(row['valid_precision_success'])}` | "
                    f"`{_fmt_pct(row['valid_precision_failure'])}` | "
                    f"`{_fmt_pct(row['test_rate_delta'], signed=True)}` | "
                    f"`{_fmt_pct(row['test_pct_steps_saved'])}` | "
                    f"`{_fmt_pct(row['test_decision_accuracy'])}` | "
                    f"`{_fmt_pct(row['test_precision_success'])}` | "
                    f"`{_fmt_pct(row['test_precision_failure'])}` | "
                    f"`{int(row['test_false_positives'])}` | "
                    f"`{int(row['test_false_negatives'])}` | "
                    f"`{int(row['test_n_decided'])}` | "
                    f"`{str(bool(row['selection_fallback'])).lower()}` |"
                )
            lines.append("")
    lines.extend(
        [
            "## Column Meaning",
            "",
            "- `Valid ΔRate`: validation adjusted resolve rate minus original resolve rate after early decisions.",
            "- `Valid Save`: validation saved prefix steps divided by all prefix steps; this is maximized after the rate constraint.",
            "- `Test ΔRate` / `Test Save`: heldout-test actual result after applying the valid-selected thresholds.",
            "- `PrecS`: precision among early success decisions; `PrecF`: precision among early failure decisions.",
            "- `N`: number of heldout-test trajectories receiving an early success/failure decision.",
        ]
    )
    (output_dir / "rate_preserving_policy_summary.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def main() -> int:
    args = parse_args()
    run_dir = _run_dir(args.run_name)
    reports_dir = run_dir / "reports"
    output_dir = reports_dir / args.output_subdir
    output_dir.mkdir(parents=True, exist_ok=True)

    frames = []
    source_meta = []
    for raw_source in args.sources:
        label, subdir = _parse_source(raw_source)
        source_dir = reports_dir / subdir
        selected = _select_source(label, source_dir, args.rate_tolerances)
        if not selected.empty:
            frames.append(selected)
        source_meta.append(
            {
                "label": label,
                "subdir": subdir,
                "exists": bool((source_dir / "valid_sweep.csv").is_file() and (source_dir / "test_sweep.csv").is_file()),
                "selected_rows": int(len(selected)),
            }
        )

    all_selected = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    all_selected.to_csv(output_dir / "rate_preserving_policy_all_grids.csv", index=False)
    metadata = {
        "run_name": args.run_name,
        "rate_tolerances": args.rate_tolerances,
        "sources": source_meta,
        "note": "Thresholds are selected on valid only; test metrics are reporting only.",
    }
    (output_dir / "rate_preserving_policy_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    _write_markdown(output_dir, all_selected, metadata)
    print(f"[rate_preserving_threshold_summary] wrote {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
