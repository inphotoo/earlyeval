#!/usr/bin/env python3
"""Fast fine-grid asymmetric threshold tuning on validation data.

This is a faster companion to ``asymmetric_threshold_valid_tuning_posthoc.py``.
It keeps the same data split and model artifacts, but evaluates dense threshold
grids by precomputing each trajectory's first success/failure hit per threshold.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from asymmetric_threshold_valid_tuning_posthoc import (
    DEFAULT_MODELS,
    DEFAULT_RUN_NAME,
    FeatureEngineer,
    _build_valid_matrices,
    _extract_test_probabilities,
    _grid_values,
    _load_valid_predictions,
    _prepare_valid_frame,
    _repair_unpickled_tfidf_for_local_sklearn,
    _run_dir,
)
from utils import get_logger


LOGGER = get_logger("fine_threshold_valid_tuning")
INF_STEP = np.int64(1_000_000_000)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", default=DEFAULT_RUN_NAME)
    parser.add_argument("--verified-jsonl", default="../../../swebench_verified/test.jsonl")
    parser.add_argument("--holdout-models", default="auto_mid3")
    parser.add_argument("--max-instances", type=int, default=500)
    parser.add_argument("--prefix-table", default=None)
    parser.add_argument("--predictions", default=None)
    parser.add_argument("--output-subdir", default="asymmetric_valid_threshold_tuning_fine_raw_step001")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument(
        "--score-mode",
        choices=("calibrated", "raw"),
        default="raw",
        help="Use validation-calibrated probabilities or raw model probabilities.",
    )
    parser.add_argument("--success-thresholds", nargs="+", type=float, default=[0.70, 0.75, 0.80, 0.85, 0.90, 0.95])
    parser.add_argument("--success-threshold-grid", nargs=3, type=float, metavar=("START", "STOP", "STEP"), default=[0.65, 0.95, 0.001])
    parser.add_argument("--failure-thresholds", nargs="+", type=float, default=[0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40])
    parser.add_argument("--failure-threshold-grid", nargs=3, type=float, metavar=("START", "STOP", "STEP"), default=[0.05, 0.45, 0.001])
    return parser.parse_args()


def _trajectory_first_hit_tables(
    frame: pd.DataFrame,
    prob: np.ndarray,
    success_thresholds: np.ndarray,
    failure_thresholds: np.ndarray,
) -> dict[str, np.ndarray]:
    temp = frame[[
        "traj_id",
        "instance_id",
        "orig_model_id",
        "prefix_step_idx",
        "label",
    ]].copy()
    temp["_prob"] = np.asarray(prob, dtype=np.float64)
    temp = temp.sort_values(["traj_id", "prefix_step_idx"])

    labels: list[int] = []
    n_prefix_rows: list[int] = []
    success_steps: list[np.ndarray] = []
    failure_steps: list[np.ndarray] = []

    for _, group in temp.groupby("traj_id", sort=False):
        probs = group["_prob"].to_numpy(dtype=np.float64)
        steps = group["prefix_step_idx"].to_numpy(dtype=np.int64)
        labels.append(int(group["label"].iloc[0]))
        n_prefix_rows.append(int(len(group)))

        success_hit = probs[:, None] >= success_thresholds[None, :]
        success_has = success_hit.any(axis=0)
        success_first_pos = success_hit.argmax(axis=0)
        success_first = np.full(len(success_thresholds), INF_STEP, dtype=np.int64)
        success_first[success_has] = steps[success_first_pos[success_has]]
        success_steps.append(success_first)

        failure_hit = probs[:, None] <= failure_thresholds[None, :]
        failure_has = failure_hit.any(axis=0)
        failure_first_pos = failure_hit.argmax(axis=0)
        failure_first = np.full(len(failure_thresholds), INF_STEP, dtype=np.int64)
        failure_first[failure_has] = steps[failure_first_pos[failure_has]]
        failure_steps.append(failure_first)

    return {
        "labels": np.asarray(labels, dtype=np.int64),
        "n_prefix_rows": np.asarray(n_prefix_rows, dtype=np.int64),
        "success_steps": np.vstack(success_steps).astype(np.int64, copy=False),
        "failure_steps": np.vstack(failure_steps).astype(np.int64, copy=False),
    }


def _safe_div(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    out = np.full_like(num, np.nan, dtype=np.float64)
    mask = den != 0
    out[mask] = num[mask] / den[mask]
    return out


def _json_safe(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, pd.DataFrame):
        return {
            "type": "DataFrame",
            "shape": [int(value.shape[0]), int(value.shape[1])],
            "columns": [str(col) for col in value.columns],
        }
    return str(value)


def _fast_sweep_one_model(
    first_hits: dict[str, np.ndarray],
    success_thresholds: np.ndarray,
    failure_thresholds: np.ndarray,
    model_name: str,
    split: str,
) -> pd.DataFrame:
    labels = first_hits["labels"]
    negatives = 1 - labels
    n_prefix_rows = first_hits["n_prefix_rows"]
    success_steps = first_hits["success_steps"]
    failure_steps = first_hits["failure_steps"]

    total = int(len(labels))
    resolved = int(labels.sum())
    total_steps = int(n_prefix_rows.sum())
    rows: list[pd.DataFrame] = []

    for success_idx, success_threshold in enumerate(success_thresholds):
        success_step = success_steps[:, success_idx]
        success_mask = success_step[:, None] < failure_steps
        failure_mask = failure_steps < success_step[:, None]
        valid_pair = failure_thresholds < success_threshold
        if not valid_pair.any():
            continue

        decided_success = success_mask.sum(axis=0).astype(np.int64)
        decided_failure = failure_mask.sum(axis=0).astype(np.int64)
        true_positives = (success_mask * labels[:, None]).sum(axis=0).astype(np.int64)
        false_positives = (success_mask * negatives[:, None]).sum(axis=0).astype(np.int64)
        true_negatives = (failure_mask * negatives[:, None]).sum(axis=0).astype(np.int64)
        false_negatives = (failure_mask * labels[:, None]).sum(axis=0).astype(np.int64)
        n_decided = decided_success + decided_failure
        undecided = total - n_decided

        success_saved = n_prefix_rows - success_step - 1
        failure_saved = n_prefix_rows[:, None] - failure_steps - 1
        total_saved_steps = (
            (success_mask * success_saved[:, None]).sum(axis=0)
            + (failure_mask * failure_saved).sum(axis=0)
        ).astype(np.int64)

        adjusted_resolved = resolved - false_negatives
        block = pd.DataFrame(
            {
                "total": total,
                "resolved": resolved,
                "decided_success": decided_success,
                "decided_failure": decided_failure,
                "undecided": undecided,
                "true_positives": true_positives,
                "true_negatives": true_negatives,
                "false_positives": false_positives,
                "false_negatives": false_negatives,
                "total_steps": total_steps,
                "total_saved_steps": total_saved_steps,
                "success_threshold": float(success_threshold),
                "failure_threshold": failure_thresholds,
                "n_decided": n_decided,
                "decision_rate": n_decided / total,
                "decision_accuracy": _safe_div(true_positives + true_negatives, n_decided),
                "precision_success": _safe_div(true_positives, decided_success),
                "precision_failure": _safe_div(true_negatives, decided_failure),
                "original_resolve_rate": resolved / total,
                "adjusted_resolve_rate": adjusted_resolved / total,
                "rate_delta": (adjusted_resolved - resolved) / total,
                "pct_steps_saved": total_saved_steps / total_steps,
                "split": split,
                "prefix_model": model_name,
            }
        )
        rows.append(block.loc[valid_pair].copy())

    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _fast_sweep(
    frame: pd.DataFrame,
    probabilities: dict[str, np.ndarray],
    success_thresholds: list[float],
    failure_thresholds: list[float],
    split: str,
) -> pd.DataFrame:
    success = np.asarray(success_thresholds, dtype=np.float64)
    failure = np.asarray(failure_thresholds, dtype=np.float64)
    frames = []
    for model_name, prob in probabilities.items():
        LOGGER.info("Fast sweep %s %s: %d success x %d failure thresholds", split, model_name, len(success), len(failure))
        first_hits = _trajectory_first_hit_tables(frame, prob, success, failure)
        frames.append(_fast_sweep_one_model(first_hits, success, failure, model_name, split))
    return pd.concat(frames, ignore_index=True)


def main() -> int:
    args = parse_args()
    run_dir = _run_dir(args.run_name)
    prefix_table = (
        Path(args.prefix_table)
        if args.prefix_table
        else run_dir / "data" / "prefix_table_filtered.parquet"
    )
    predictions_path = (
        Path(args.predictions)
        if args.predictions
        else run_dir / "reports" / "test_predictions_all_models.parquet"
    )
    verified_jsonl = Path(args.verified_jsonl)
    if not verified_jsonl.is_absolute():
        verified_jsonl = (Path(__file__).resolve().parent / verified_jsonl).resolve()
    output_dir = run_dir / "reports" / args.output_subdir
    output_dir.mkdir(parents=True, exist_ok=True)

    success_thresholds = _grid_values(args.success_threshold_grid, args.success_thresholds)
    failure_thresholds = _grid_values(args.failure_threshold_grid, args.failure_thresholds)

    LOGGER.info("Loading FeatureEngineer")
    feature_engineer = FeatureEngineer.load(run_dir / "models" / "feature_engineer_with_model.pkl")
    _repair_unpickled_tfidf_for_local_sklearn(feature_engineer)

    valid, split_meta = _prepare_valid_frame(
        prefix_table=prefix_table,
        feature_engineer=feature_engineer,
        verified_jsonl=verified_jsonl,
        holdout_models=args.holdout_models,
        max_instances=args.max_instances,
    )
    matrices = _build_valid_matrices(feature_engineer, valid)
    valid_prob = _load_valid_predictions(args.models, run_dir, matrices, args.score_mode)

    LOGGER.info("Loading heldout test predictions: %s", predictions_path)
    test_predictions = pd.read_parquet(predictions_path)
    test_predictions = test_predictions.sort_values(["traj_id", "prefix_step_idx"]).reset_index(drop=True)
    test_prob = _extract_test_probabilities(test_predictions, args.models, args.score_mode)

    valid_sweep = _fast_sweep(valid, valid_prob, success_thresholds, failure_thresholds, "valid")
    test_sweep = _fast_sweep(test_predictions, test_prob, success_thresholds, failure_thresholds, "test")

    valid_sweep.to_csv(output_dir / "valid_sweep.csv", index=False)
    test_sweep.to_csv(output_dir / "test_sweep.csv", index=False)
    metadata: dict[str, Any] = {
        "run_name": args.run_name,
        "models": args.models,
        "score_mode": args.score_mode,
        "success_thresholds": success_thresholds,
        "failure_thresholds": failure_thresholds,
        "valid_sweep_rows": int(len(valid_sweep)),
        "test_sweep_rows": int(len(test_sweep)),
        "split_meta": _json_safe(split_meta),
        "note": "Fast fine-grid sweep. Thresholds are evaluated on valid/test; selection should use valid only.",
    }
    (output_dir / "threshold_grid_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (output_dir / "report.txt").write_text(
        "\n".join(
            [
                "Fast fine-grid asymmetric threshold sweep",
                f"run_name={args.run_name}",
                f"score_mode={args.score_mode}",
                f"success_thresholds={success_thresholds[0]:.3f}..{success_thresholds[-1]:.3f} n={len(success_thresholds)}",
                f"failure_thresholds={failure_thresholds[0]:.3f}..{failure_thresholds[-1]:.3f} n={len(failure_thresholds)}",
                f"valid_sweep_rows={len(valid_sweep)}",
                f"test_sweep_rows={len(test_sweep)}",
                "Use two_end_precision_target_report_posthoc.py to select thresholds on valid and apply to test.",
            ]
        ),
        encoding="utf-8",
    )
    print(f"[fine_threshold_valid_tuning] wrote {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
