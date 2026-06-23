#!/usr/bin/env python3
"""Post-hoc asymmetric two-sided threshold tuning on validation data.

This script does not retrain any prefix predictor.  It reuses:

* the completed run's ``prefix_table_filtered.parquet``
* the fitted ``feature_engineer_with_model.pkl``
* saved model files and saved Platt calibrators
* saved heldout-test ``test_predictions_all_models.parquet``

It rebuilds validation predictions only, tunes asymmetric thresholds on valid,
then applies the chosen thresholds to the heldout test predictions.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".mplconfig"))
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse

import config
from feature_engineer import (
    BOOL_FEATURES,
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    FeatureEngineer,
)
from model_holdout_split import select_model_holdout_split
from trainer import load_model
from utils import get_logger


LOGGER = get_logger("asymmetric_threshold_valid_tuning")

DEFAULT_RUN_NAME = "model_holdout_answer_calibrated_full"
DEFAULT_MODELS = [
    "I_LightGBM_Dense_AF",
    "J_LightGBM_Dense_AF_Thought",
    "H_LightGBM_Dense",
    "K_LightGBM_Dense_Full",
    "D_Dense_Full_LR",
    "G_TfIdf_Full_LR",
]

MODEL_ARTIFACTS = {
    "D_Dense_Full_LR": "baseline_dense_full_lr.pkl",
    "G_TfIdf_Full_LR": "baseline_tfidf_full_lr.pkl",
    "H_LightGBM_Dense": "baseline_lgbm_dense.lgb",
    "I_LightGBM_Dense_AF": "baseline_lgbm_dense_af.lgb",
    "J_LightGBM_Dense_AF_Thought": "baseline_lgbm_dense_af_thought.lgb",
    "K_LightGBM_Dense_Full": "baseline_lgbm_dense_full.lgb",
}

MODEL_FEATURE_SET = {
    "D_Dense_Full_LR": "dense_full",
    "G_TfIdf_Full_LR": "tfidf_full",
    "H_LightGBM_Dense": "dense",
    "I_LightGBM_Dense_AF": "dense_af",
    "J_LightGBM_Dense_AF_Thought": "dense_af_thought",
    "K_LightGBM_Dense_Full": "dense_full",
}

TFIDF_AF_COLS = [
    "tfidf_task_prompt",
    "tfidf_prefix_action",
    "tfidf_prefix_feedback",
    "tfidf_last_action",
    "tfidf_last_feedback",
]
TFIDF_THOUGHT_COLS = ["tfidf_prefix_thought", "tfidf_last_thought"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", default=DEFAULT_RUN_NAME)
    parser.add_argument("--verified-jsonl", default="../../../swebench_verified/test.jsonl")
    parser.add_argument("--holdout-models", default="auto_mid3")
    parser.add_argument("--max-instances", type=int, default=500)
    parser.add_argument("--prefix-table", default=None)
    parser.add_argument("--predictions", default=None)
    parser.add_argument("--output-subdir", default="asymmetric_valid_threshold_tuning")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument(
        "--score-mode",
        choices=("calibrated", "raw"),
        default="calibrated",
        help="Use validation-calibrated probabilities or raw model probabilities for threshold tuning.",
    )
    parser.add_argument(
        "--success-thresholds",
        nargs="+",
        type=float,
        default=[0.70, 0.75, 0.80, 0.85, 0.90, 0.95],
    )
    parser.add_argument(
        "--success-threshold-grid",
        nargs=3,
        type=float,
        metavar=("START", "STOP", "STEP"),
        default=None,
        help="Override --success-thresholds with an inclusive numeric grid.",
    )
    parser.add_argument(
        "--failure-thresholds",
        nargs="+",
        type=float,
        default=[0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40],
    )
    parser.add_argument(
        "--failure-threshold-grid",
        nargs=3,
        type=float,
        metavar=("START", "STOP", "STEP"),
        default=None,
        help="Override --failure-thresholds with an inclusive numeric grid.",
    )
    parser.add_argument(
        "--selection-policies",
        nargs="+",
        default=["rate_1pp", "rate_2pp", "prec90"],
        choices=["rate_1pp", "rate_2pp", "prec90"],
    )
    parser.add_argument("--skip-plots", action="store_true")
    return parser.parse_args()


def _grid_values(grid: list[float] | None, explicit: list[float]) -> list[float]:
    if grid is None:
        return sorted(set(round(float(x), 6) for x in explicit))
    start, stop, step = [float(x) for x in grid]
    if step <= 0:
        raise ValueError(f"Grid step must be positive, got {step}")
    values = []
    cur = start
    # small epsilon keeps decimal endpoints such as 0.95 included.
    while cur <= stop + step * 1e-6:
        values.append(round(float(cur), 6))
        cur += step
    return sorted(set(values))


def _run_dir(run_name: str) -> Path:
    return PROJECT_ROOT / "runs" / run_name


def _safe_name(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in name)


def _repair_unpickled_tfidf_for_local_sklearn(feature_engineer: FeatureEngineer) -> None:
    repaired = 0
    for vectorizer in feature_engineer.tfidf_vectorizers.values():
        transformer = getattr(vectorizer, "_tfidf", None)
        if transformer is None:
            continue
        if "_idf_diag" in getattr(transformer, "__dict__", {}):
            continue
        raw_idf = getattr(transformer, "__dict__", {}).get("idf_")
        if raw_idf is None:
            continue
        try:
            transformer.idf_ = raw_idf
        except Exception:
            raw_idf = np.asarray(raw_idf, dtype=np.float64)
            transformer._idf_diag = sparse.diags(
                raw_idf,
                offsets=0,
                shape=(len(raw_idf), len(raw_idf)),
                format="csr",
                dtype=np.float64,
            )
        repaired += 1
    if repaired:
        LOGGER.info("Repaired %s TF-IDF transformer(s).", repaired)


def _required_columns(feature_engineer: FeatureEngineer) -> list[str]:
    metadata_columns = [
        "prefix_id",
        "traj_id",
        "group_id",
        "instance_id",
        "prefix_step_idx",
        "n_steps_total_for_weighting",
        "sample_weight",
        "label",
        "model_id",
        "model",
    ]
    dense_columns = (
        list(NUMERIC_FEATURES)
        + list(BOOL_FEATURES)
        + list(CATEGORICAL_FEATURES)
        + ["model_id"]
    )
    text_columns = list(feature_engineer.active_text_columns.values())
    return list(dict.fromkeys(metadata_columns + dense_columns + text_columns))


def _available_columns(path: Path) -> set[str]:
    import pyarrow.parquet as pq

    return set(pq.ParquetFile(path).schema_arrow.names)


def _read_metadata(prefix_table: Path) -> pd.DataFrame:
    cols = [
        "prefix_id",
        "traj_id",
        "instance_id",
        "prefix_step_idx",
        "n_steps_total_for_weighting",
        "sample_weight",
        "label",
        "model_id",
        "model",
    ]
    existing = [c for c in cols if c in _available_columns(prefix_table)]
    LOGGER.info("Loading metadata columns from %s", prefix_table)
    return pd.read_parquet(prefix_table, columns=existing)


def _read_valid_rows(
    prefix_table: Path,
    columns: list[str],
    valid_instances: list[str],
    valid_models: list[str],
) -> pd.DataFrame:
    missing = sorted(set(columns) - _available_columns(prefix_table))
    if missing:
        raise RuntimeError(f"Prefix table missing required columns: {missing}")
    filters = [
        ("instance_id", "in", list(valid_instances)),
        ("model_id", "in", list(valid_models)),
    ]
    LOGGER.info(
        "Loading valid feature rows with predicate pushdown: %s instances, %s models",
        len(valid_instances),
        len(valid_models),
    )
    return pd.read_parquet(prefix_table, columns=columns, filters=filters)


def _prepare_valid_frame(
    prefix_table: Path,
    feature_engineer: FeatureEngineer,
    verified_jsonl: Path,
    holdout_models: str,
    max_instances: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    metadata = _read_metadata(prefix_table)
    _train_idx, valid_idx, _test_idx, split_meta = select_model_holdout_split(
        metadata,
        verified_jsonl=verified_jsonl,
        holdout_models=holdout_models,
        max_instances=max_instances,
    )
    valid_meta = metadata.loc[valid_idx].copy()
    valid_instances = sorted(valid_meta["instance_id"].astype(str).unique())
    valid_models = sorted(valid_meta["model_id"].astype(str).unique())
    columns = _required_columns(feature_engineer)
    valid = _read_valid_rows(prefix_table, columns, valid_instances, valid_models)

    min_steps = int(config.MIN_TRAJECTORY_STEPS)
    short_valid = set(
        valid.groupby("traj_id")["n_steps_total_for_weighting"].first()
        .loc[lambda values: values < min_steps]
        .index
    )
    if short_valid:
        valid = valid[~valid["traj_id"].isin(short_valid)].copy()

    valid["orig_model_id"] = valid["model_id"].astype(str)
    valid["orig_model"] = valid.get("model", valid["model_id"]).astype(str)
    valid["split"] = "valid"
    valid["model_id"] = "__MISSING__"
    if "model" in valid.columns:
        valid["model"] = "__MISSING__"
    valid["model_id_input_mode"] = "valid_missing"
    valid = valid.sort_values(["traj_id", "prefix_step_idx"]).reset_index(drop=True)
    split_meta.update(
        {
            "valid_rows_loaded": int(len(valid)),
            "valid_trajectories_loaded": int(valid["traj_id"].nunique()),
            "valid_instances": int(valid["instance_id"].nunique()),
            "valid_models": valid_models,
        }
    )
    return valid, split_meta


def _transform_tfidf_blocks(
    feature_engineer: FeatureEngineer,
    frame: pd.DataFrame,
    names: list[str],
) -> dict[str, sparse.csr_matrix]:
    blocks: dict[str, sparse.csr_matrix] = {}
    for name in names:
        if name not in feature_engineer.active_text_columns:
            continue
        LOGGER.info("Transforming valid TF-IDF block: %s", name)
        blocks[name] = feature_engineer.transform_tfidf_subset(frame, [name])
    return blocks


def _stack_blocks(
    blocks: dict[str, sparse.csr_matrix],
    names: list[str],
    n_rows: int,
) -> sparse.csr_matrix:
    parts = [blocks[name] for name in names if name in blocks]
    if not parts:
        return sparse.csr_matrix((n_rows, 0))
    return sparse.hstack(parts, format="csr")


def _build_valid_matrices(
    feature_engineer: FeatureEngineer,
    valid: pd.DataFrame,
) -> dict[str, Any]:
    LOGGER.info("Building valid dense matrix")
    dense = feature_engineer.transform_dense(valid)
    dense_sp = sparse.csr_matrix(dense)

    active = list(feature_engineer.active_text_columns.keys())
    blocks = _transform_tfidf_blocks(feature_engineer, valid, active)
    af_cols = [c for c in TFIDF_AF_COLS if c in active]
    thought_cols = [c for c in TFIDF_THOUGHT_COLS if c in active]
    af = _stack_blocks(blocks, af_cols, len(valid))
    af_thought = _stack_blocks(blocks, af_cols + thought_cols, len(valid))
    full = _stack_blocks(blocks, active, len(valid))
    return {
        "dense": dense,
        "dense_af": sparse.hstack([dense_sp, af], format="csr"),
        "dense_af_thought": sparse.hstack([dense_sp, af_thought], format="csr"),
        "dense_full": sparse.hstack([dense_sp, full], format="csr"),
        "tfidf_full": full,
    }


def _predict_model(model: Any, matrix: Any) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return np.asarray(model.predict_proba(matrix)[:, 1], dtype=np.float64)
    return np.asarray(model.predict(matrix), dtype=np.float64).ravel()


def _load_valid_predictions(
    models: list[str],
    run_dir: Path,
    matrices: dict[str, Any],
    score_mode: str,
) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for model_name in models:
        artifact = MODEL_ARTIFACTS.get(model_name)
        feature_set = MODEL_FEATURE_SET.get(model_name)
        if not artifact or not feature_set:
            raise RuntimeError(f"Unsupported model for this posthoc script: {model_name}")
        model_path = run_dir / "models" / artifact
        if not model_path.is_file():
            raise FileNotFoundError(model_path)
        LOGGER.info("Loading model: %s", model_name)
        model = load_model(model_path)
        raw = _predict_model(model, matrices[feature_set])
        if score_mode == "calibrated":
            calibrator_path = run_dir / "models" / f"calibrator_{_safe_name(model_name)}.pkl"
            if not calibrator_path.is_file():
                raise FileNotFoundError(calibrator_path)
            calibrator = load_model(calibrator_path)
            out[model_name] = calibrator.predict(raw)
        else:
            out[model_name] = raw
        LOGGER.info(
            "%s valid %s mean=%.4f min=%.4f max=%.4f",
            model_name,
            score_mode,
            float(out[model_name].mean()),
            float(out[model_name].min()),
            float(out[model_name].max()),
        )
    return out


def _trajectory_arrays(frame: pd.DataFrame, prob: np.ndarray) -> list[dict[str, Any]]:
    temp = frame[[
        "traj_id",
        "instance_id",
        "orig_model_id",
        "prefix_step_idx",
        "label",
    ]].copy()
    temp["_prob"] = np.asarray(prob, dtype=np.float64)
    temp = temp.sort_values(["traj_id", "prefix_step_idx"])
    groups: list[dict[str, Any]] = []
    for traj_id, group in temp.groupby("traj_id", sort=False):
        groups.append(
            {
                "traj_id": traj_id,
                "instance_id": str(group["instance_id"].iloc[0]),
                "agent_model": str(group["orig_model_id"].iloc[0]),
                "label": int(group["label"].iloc[0]),
                "steps": group["prefix_step_idx"].to_numpy(dtype=np.int64),
                "prob": group["_prob"].to_numpy(dtype=np.float64),
                "n_prefix_rows": int(len(group)),
            }
        )
    return groups


def _eval_threshold_pair(
    groups: list[dict[str, Any]],
    success_threshold: float,
    failure_threshold: float,
) -> dict[str, Any]:
    if failure_threshold >= success_threshold:
        return {}
    counts = {
        "total": 0,
        "resolved": 0,
        "decided_success": 0,
        "decided_failure": 0,
        "undecided": 0,
        "true_positives": 0,
        "true_negatives": 0,
        "false_positives": 0,
        "false_negatives": 0,
        "total_steps": 0,
        "total_saved_steps": 0,
    }
    for group in groups:
        label = int(group["label"])
        counts["total"] += 1
        counts["resolved"] += label
        counts["total_steps"] += int(group["n_prefix_rows"])
        decision = None
        decision_step = -1
        decision_prob = np.nan
        for step, prob in zip(group["steps"], group["prob"]):
            p = float(prob)
            if p >= success_threshold:
                decision = "success"
                decision_step = int(step)
                decision_prob = p
                break
            if p <= failure_threshold:
                decision = "failure"
                decision_step = int(step)
                decision_prob = p
                break
        if decision is None:
            counts["undecided"] += 1
            continue
        counts["total_saved_steps"] += max(int(group["n_prefix_rows"]) - decision_step - 1, 0)
        if decision == "success":
            counts["decided_success"] += 1
            if label == 1:
                counts["true_positives"] += 1
            else:
                counts["false_positives"] += 1
        else:
            counts["decided_failure"] += 1
            if label == 0:
                counts["true_negatives"] += 1
            else:
                counts["false_negatives"] += 1
    total = counts["total"]
    n_decided = counts["decided_success"] + counts["decided_failure"]
    adjusted_resolved = counts["resolved"] - counts["false_negatives"]
    precision_success = (
        counts["true_positives"] / counts["decided_success"]
        if counts["decided_success"]
        else np.nan
    )
    precision_failure = (
        counts["true_negatives"] / counts["decided_failure"]
        if counts["decided_failure"]
        else np.nan
    )
    return {
        **counts,
        "success_threshold": float(success_threshold),
        "failure_threshold": float(failure_threshold),
        "n_decided": int(n_decided),
        "decision_rate": n_decided / total if total else np.nan,
        "decision_accuracy": (
            (counts["true_positives"] + counts["true_negatives"]) / n_decided
            if n_decided
            else np.nan
        ),
        "precision_success": precision_success,
        "precision_failure": precision_failure,
        "original_resolve_rate": counts["resolved"] / total if total else np.nan,
        "adjusted_resolve_rate": adjusted_resolved / total if total else np.nan,
        "rate_delta": (adjusted_resolved - counts["resolved"]) / total if total else np.nan,
        "pct_steps_saved": (
            counts["total_saved_steps"] / counts["total_steps"]
            if counts["total_steps"]
            else np.nan
        ),
    }


def _sweep(
    frame: pd.DataFrame,
    probabilities: dict[str, np.ndarray],
    success_thresholds: list[float],
    failure_thresholds: list[float],
    split: str,
) -> pd.DataFrame:
    rows = []
    for model_name, prob in probabilities.items():
        groups = _trajectory_arrays(frame, prob)
        for success_threshold in success_thresholds:
            for failure_threshold in failure_thresholds:
                metrics = _eval_threshold_pair(groups, success_threshold, failure_threshold)
                if not metrics:
                    continue
                metrics["split"] = split
                metrics["prefix_model"] = model_name
                rows.append(metrics)
    return pd.DataFrame(rows)


def _select_thresholds(valid_sweep: pd.DataFrame, policies: list[str]) -> pd.DataFrame:
    rows = []
    for model_name, group in valid_sweep.groupby("prefix_model", sort=False):
        for policy in policies:
            if policy == "rate_1pp":
                candidates = group[group["rate_delta"].abs() <= 0.01].copy()
                label = "valid_abs_rate_delta<=1pp_max_save"
            elif policy == "rate_2pp":
                candidates = group[group["rate_delta"].abs() <= 0.02].copy()
                label = "valid_abs_rate_delta<=2pp_max_save"
            elif policy == "prec90":
                candidates = group[
                    (group["precision_success"] >= 0.90)
                    & (group["precision_failure"] >= 0.90)
                ].copy()
                label = "valid_prec_success_failure>=90_max_save"
            else:
                continue

            if candidates.empty:
                candidates = group.copy()
                candidates["_fallback_abs_rate_delta"] = candidates["rate_delta"].abs()
                best = candidates.sort_values(
                    ["_fallback_abs_rate_delta", "pct_steps_saved"],
                    ascending=[True, False],
                ).iloc[0]
                fallback = True
            else:
                best = candidates.sort_values(
                    ["pct_steps_saved", "decision_accuracy", "decision_rate"],
                    ascending=[False, False, False],
                ).iloc[0]
                fallback = False
            row = best.to_dict()
            row["selection_policy"] = policy
            row["selection_label"] = label
            row["selection_fallback"] = bool(fallback)
            rows.append(row)
    return pd.DataFrame(rows)


def _apply_selected_to_test(test_sweep: pd.DataFrame, selected: pd.DataFrame) -> pd.DataFrame:
    rows = []
    keyed = test_sweep.set_index(["prefix_model", "success_threshold", "failure_threshold"])
    for _, row in selected.iterrows():
        key = (
            row["prefix_model"],
            float(row["success_threshold"]),
            float(row["failure_threshold"]),
        )
        if key not in keyed.index:
            continue
        out = keyed.loc[key].to_dict()
        if isinstance(out, dict) and "split" in out:
            pass
        else:
            out = keyed.loc[key].iloc[0].to_dict()
        out["prefix_model"] = row["prefix_model"]
        out["success_threshold"] = float(row["success_threshold"])
        out["failure_threshold"] = float(row["failure_threshold"])
        out["selection_policy"] = row["selection_policy"]
        out["selection_label"] = row["selection_label"]
        out["valid_rate_delta"] = float(row["rate_delta"])
        out["valid_pct_steps_saved"] = float(row["pct_steps_saved"])
        out["valid_decision_accuracy"] = float(row["decision_accuracy"])
        out["valid_precision_success"] = float(row["precision_success"])
        out["valid_precision_failure"] = float(row["precision_failure"])
        rows.append(out)
    return pd.DataFrame(rows)


def _extract_test_probabilities(
    predictions: pd.DataFrame,
    models: list[str],
    score_mode: str,
) -> dict[str, np.ndarray]:
    out = {}
    prefix = "prob_cal__" if score_mode == "calibrated" else "prob__"
    for model_name in models:
        col = f"{prefix}{model_name}"
        if col not in predictions.columns:
            raise RuntimeError(f"Missing test column: {col}")
        out[model_name] = predictions[col].to_numpy(dtype=np.float64)
    return out


def _make_heatmaps(test_sweep: pd.DataFrame, output_dir: Path) -> None:
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    for model_name, group in test_sweep.groupby("prefix_model", sort=False):
        for metric, title, cmap in [
            ("rate_delta", "Test adjusted rate delta", "coolwarm"),
            ("pct_steps_saved", "Test prefix steps saved", "viridis"),
            ("decision_accuracy", "Test decision accuracy", "magma"),
        ]:
            pivot = group.pivot_table(
                index="failure_threshold",
                columns="success_threshold",
                values=metric,
                aggfunc="mean",
            ).sort_index(ascending=False)
            fig, ax = plt.subplots(figsize=(9, 5.5))
            im = ax.imshow(pivot.to_numpy(dtype=float), aspect="auto", cmap=cmap)
            ax.set_xticks(np.arange(len(pivot.columns)))
            ax.set_xticklabels([f"{x:.2f}" for x in pivot.columns], rotation=45)
            ax.set_yticks(np.arange(len(pivot.index)))
            ax.set_yticklabels([f"{x:.2f}" for x in pivot.index])
            ax.set_xlabel("success_threshold")
            ax.set_ylabel("failure_threshold")
            ax.set_title(f"{title} — {model_name}")
            cbar = fig.colorbar(im, ax=ax)
            cbar.ax.set_ylabel(metric)
            fig.tight_layout()
            fig.savefig(plot_dir / f"heatmap_{metric}_{_safe_name(model_name)}.png", dpi=160)
            plt.close(fig)


def _fmt_pct(value: Any, digits: int = 1) -> str:
    try:
        if pd.isna(value):
            return "nan"
        return f"{100.0 * float(value):.{digits}f}%"
    except Exception:
        return "nan"


def _fmt_float(value: Any, digits: int = 3) -> str:
    try:
        if pd.isna(value):
            return "nan"
        return f"{float(value):.{digits}f}"
    except Exception:
        return "nan"


def _write_report(
    output_dir: Path,
    valid_sweep: pd.DataFrame,
    test_sweep: pd.DataFrame,
    selected: pd.DataFrame,
    test_selected: pd.DataFrame,
    split_meta: dict[str, Any],
) -> None:
    lines = []
    lines.append("=" * 110)
    lines.append("  Validation-tuned asymmetric two-sided threshold report")
    lines.append("=" * 110)
    lines.append("")
    lines.append("含义：概率仍使用已有 validation-only Platt 校准列；本报告在 valid 上校准两个决策阈值。")
    lines.append("规则：p >= success_threshold 判 success；p <= failure_threshold 判 failure；中间区间继续跑。")
    lines.append("test 没有参与阈值选择，只用于应用 valid 选出的阈值。")
    lines.append("")
    lines.append(
        f"Valid rows/trajs/instances: {split_meta.get('valid_rows_loaded')} / "
        f"{split_meta.get('valid_trajectories_loaded')} / {split_meta.get('valid_instances')}"
    )
    lines.append(
        "Heldout test models: "
        + ", ".join(str(x) for x in split_meta.get("holdout_models", []))
    )
    lines.append("")

    lines.append("1. Valid 选阈值后应用到 Test 的结果")
    lines.append("-" * 110)
    header = (
        f"{'Model':34s} {'Policy':10s} {'ThrS':>5s} {'ThrF':>5s} "
        f"{'VΔRate':>8s} {'VSave':>8s} {'TΔRate':>8s} {'TSave':>8s} "
        f"{'TAcc':>7s} {'TPS':>7s} {'TPF':>7s} {'TFP':>5s} {'TFN':>5s}"
    )
    lines.append(header)
    lines.append("-" * len(header))
    ordered = test_selected.sort_values(["prefix_model", "selection_policy"])
    for _, row in ordered.iterrows():
        lines.append(
            f"{str(row['prefix_model'])[:34]:34s} {str(row['selection_policy'])[:10]:10s} "
            f"{float(row['success_threshold']):5.2f} {float(row['failure_threshold']):5.2f} "
            f"{_fmt_pct(row['valid_rate_delta']):>8s} {_fmt_pct(row['valid_pct_steps_saved']):>8s} "
            f"{_fmt_pct(row['rate_delta']):>8s} {_fmt_pct(row['pct_steps_saved']):>8s} "
            f"{_fmt_pct(row['decision_accuracy']):>7s} {_fmt_pct(row['precision_success']):>7s} "
            f"{_fmt_pct(row['precision_failure']):>7s} {int(row['false_positives']):5d} {int(row['false_negatives']):5d}"
        )
    lines.append("")

    lines.append("2. 每个模型在 Test 上 rate_1pp policy 的摘要")
    lines.append("-" * 110)
    rate_1pp = test_selected[test_selected["selection_policy"] == "rate_1pp"].copy()
    for _, row in rate_1pp.sort_values("pct_steps_saved", ascending=False).iterrows():
        lines.append(
            f"- {row['prefix_model']}: ThrS={row['success_threshold']:.2f}, "
            f"ThrF={row['failure_threshold']:.2f}, "
            f"Test ΔRate={_fmt_pct(row['rate_delta'])}, "
            f"Save={_fmt_pct(row['pct_steps_saved'])}, "
            f"Acc={_fmt_pct(row['decision_accuracy'])}, "
            f"FP={int(row['false_positives'])}, FN={int(row['false_negatives'])}"
        )
    lines.append("")

    lines.append("3. 输出文件")
    lines.append("-" * 110)
    for name in [
        "valid_sweep.csv",
        "test_sweep.csv",
        "valid_selected_thresholds.csv",
        "test_selected_results.csv",
        "report.txt",
        "plots/heatmap_<metric>_<model>.png",
    ]:
        lines.append(f"- {name}")
    lines.append("")
    lines.append("说明：ΔRate = adjusted_resolve_rate - original_resolve_rate；正数表示早停后把 rate 抬高。")
    lines.append("      Save = total_saved_steps / total_prefix_rows；越高表示越早停、越省。")
    lines.append("")
    output_dir.joinpath("report.txt").write_text("\n".join(lines), encoding="utf-8")


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
        verified_jsonl = (PROJECT_ROOT / verified_jsonl).resolve()
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

    valid_sweep = _sweep(
        valid,
        valid_prob,
        success_thresholds=success_thresholds,
        failure_thresholds=failure_thresholds,
        split="valid",
    )
    test_sweep = _sweep(
        test_predictions,
        test_prob,
        success_thresholds=success_thresholds,
        failure_thresholds=failure_thresholds,
        split="test",
    )
    selected = _select_thresholds(valid_sweep, args.selection_policies)
    test_selected = _apply_selected_to_test(test_sweep, selected)

    valid_sweep.to_csv(output_dir / "valid_sweep.csv", index=False)
    test_sweep.to_csv(output_dir / "test_sweep.csv", index=False)
    selected.to_csv(output_dir / "valid_selected_thresholds.csv", index=False)
    test_selected.to_csv(output_dir / "test_selected_results.csv", index=False)
    metadata = {
        "run_name": args.run_name,
        "models": args.models,
        "score_mode": args.score_mode,
        "success_thresholds": success_thresholds,
        "failure_thresholds": failure_thresholds,
        "selection_policies": args.selection_policies,
        "valid_sweep_rows": int(len(valid_sweep)),
        "test_sweep_rows": int(len(test_sweep)),
        "note": "Thresholds selected on valid; test labels are not used for selection.",
    }
    (output_dir / "threshold_grid_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    if not args.skip_plots:
        _make_heatmaps(test_sweep, output_dir)
    _write_report(output_dir, valid_sweep, test_sweep, selected, test_selected, split_meta)
    print(f"[asymmetric_threshold_valid_tuning] wrote {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
