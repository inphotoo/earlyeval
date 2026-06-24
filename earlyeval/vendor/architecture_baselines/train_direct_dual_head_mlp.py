#!/usr/bin/env python3
"""Train direct dual-head safe-stop MLP models on the original feature matrix.

This is the direct neural counterpart to the current LightGBM safe-stop line:

    raw prefix features -> safe_success / safe_failure heads -> valid-selected gate

It is intentionally not a second-stage meta-gate over existing probabilities.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import inspect
import json
import os
import sys
from pathlib import Path
from typing import Any

def _package_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "configs" / "earlyeval.yaml").exists():
            return parent
    raise RuntimeError("Could not locate earlyeval root")


PACKAGE_ROOT = _package_root()
REPO_ROOT = PACKAGE_ROOT.parent
PROJECT_ROOT = PACKAGE_ROOT / "earlyeval" / "vendor" / "prefix_predict_model_holdout_answer"
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MaxAbsScaler, StandardScaler

import config
from feature_engineer import FeatureEngineer, TFIDF_ACTION_FEEDBACK, TFIDF_THOUGHT
from gold_text_tfidf_ablation_posthoc import (
    _load_prefix_table,
    _repair_unpickled_tfidf_for_local_sklearn,
    _set_run_dirs,
)
from model_holdout_shadow_valid_retrain import (
    _build_split,
    _json_default,
    _prediction_frame,
    _required_columns,
    _selected_specs,
    _set_cpu_thread_limits,
    _transform_tfidf_subset_streaming,
)
from model_holdout_split import load_verified_instance_ids
from probability_calibration import calibration_summary_row, fit_sigmoid_calibrator
from safe_stop_dual_head_retrain import (
    _build_matrices,
    _evaluate_policies,
    _evaluate_selected,
    _head_column,
    _matrices_for_spec,
    _policy_grid,
    _safe_targets,
    _select_policies,
    _write_report,
)
from trainer import save_model
from utils import get_logger, rebind_all_file_loggers, timer


LOGGER = get_logger("direct_dual_head_mlp")
DEFAULT_OUTPUT = (
    PACKAGE_ROOT
    / "paper"
    / "experiments"
    / "earlyeval_architecture_smoke"
    / "direct_dual_head_mlp"
)
SHARED_ANSWER_DATA_ROOT = REPO_ROOT / "data" / "prefix_predict_model_holdout_answer" / "model_holdout_answer_shared"
DEFAULT_PREFIX_TABLE = SHARED_ANSWER_DATA_ROOT / "prefix_table_filtered.parquet"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", default="model_holdout_answer_calibrated_full")
    parser.add_argument("--prefix-table", type=Path, default=None)
    parser.add_argument(
        "--verified-jsonl",
        type=Path,
        default=REPO_ROOT / "data" / "swe_verify_500" / "offical_answer" / "test.jsonl",
    )
    parser.add_argument("--holdout-models", default="auto_mid3")
    parser.add_argument(
        "--exclude-train-models",
        nargs="*",
        default=None,
        help="Drop configured-excluded model ids before building train/valid/test splits.",
    )
    parser.add_argument("--max-instances", type=int, default=500)
    parser.add_argument("--split-strategy", choices=("per_instance_model", "per_instance_traj"), default="per_instance_model")
    parser.add_argument("--valid-models-per-instance", type=int, default=3)
    parser.add_argument("--valid-traj-ratio", type=float, default=0.15)
    parser.add_argument("--valid-per-instance", type=int, default=0)
    parser.add_argument("--seed", type=int, default=config.SPLIT_SEED)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--variants", nargs="+", default=["i"])
    parser.add_argument(
        "--feature-set",
        choices=("dense", "variants"),
        default="variants",
        help="Use dense-only for fast smoke, or --variants matrices for Dense+AF/J.",
    )
    parser.add_argument("--safe-label-min-step", type=int, default=10)
    parser.add_argument("--policy-min-steps", nargs="+", type=int, default=[0, 5, 10, 15])
    parser.add_argument("--consecutive", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--success-thresholds", nargs="+", type=float, default=[0.80, 0.90, 0.95])
    parser.add_argument("--failure-thresholds", nargs="+", type=float, default=[0.80, 0.90, 0.95])
    parser.add_argument("--score-modes", nargs="+", choices=("raw", "calibrated"), default=["raw", "calibrated"])
    parser.add_argument("--max-valid-abs-drop-pp", type=float, default=2.0)
    parser.add_argument("--min-valid-decision-acc", type=float, default=0.90)
    parser.add_argument("--fallback-min-save-pct", type=float, default=5.0)
    parser.add_argument("--mask-train-model-id", action="store_true", default=True)
    parser.add_argument("--keep-train-model-id", dest="mask_train_model_id", action="store_false")
    parser.add_argument("--low-memory", action="store_true")
    parser.add_argument("--eager-load-text-columns", action="store_true")
    parser.add_argument("--text-batch-size", type=int, default=4096)
    parser.add_argument(
        "--cache-matrices",
        action="store_true",
        help=(
            "Save/load reusable Dense/AF/Thought sparse matrices. "
            "Default cache directory is <output-dir>/matrix_cache."
        ),
    )
    parser.add_argument(
        "--matrix-cache-dir",
        type=Path,
        default=None,
        help="Optional explicit matrix cache directory.",
    )
    parser.add_argument(
        "--rebuild-matrix-cache",
        action="store_true",
        help="Ignore any existing matrix cache and overwrite it after rebuilding.",
    )
    parser.add_argument("--max-cpu-threads", type=int, default=int(os.environ.get("SWE_MAX_CPU_THREADS", "24")))
    parser.add_argument("--hidden", nargs="+", type=int, default=[64])
    parser.add_argument("--alpha", type=float, default=1e-3)
    parser.add_argument("--learning-rate-init", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--max-iter", type=int, default=30)
    parser.add_argument("--validation-fraction", type=float, default=0.10)
    parser.add_argument("--n-iter-no-change", type=int, default=5)
    parser.add_argument("--scaler", choices=("maxabs", "standard", "none"), default="maxabs")
    parser.add_argument(
        "--max-train-rows-per-head",
        type=int,
        default=160_000,
        help="Weighted row subsample for fast smoke runs; <=0 uses all rows.",
    )
    parser.add_argument(
        "--no-balanced-row-sample",
        dest="balanced_row_sample",
        action="store_false",
        help="Disable class-balanced weighted row subsampling.",
    )
    parser.set_defaults(balanced_row_sample=True)
    parser.add_argument(
        "--sample-weight-mode",
        choices=("none", "fit", "weighted_resample", "auto"),
        default="none",
        help=(
            "How to use sample_weight in MLP fitting. `fit` requires a sklearn "
            "MLPClassifier that supports sample_weight; `weighted_resample` draws "
            "training rows with replacement according to sample_weight; `auto` uses "
            "fit when available and otherwise falls back to weighted_resample."
        ),
    )
    parser.add_argument(
        "--weighted-resample-size-per-head",
        type=int,
        default=0,
        help=(
            "Rows drawn with replacement for sample-weight-mode=weighted_resample. "
            "<=0 uses the selected training row count."
        ),
    )
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def _mlp_predictor_name(lightgbm_name: str) -> str:
    return lightgbm_name.replace("LightGBM", "DirectMLP")


def _sample_training_indices(
    y: np.ndarray,
    sample_weight: np.ndarray,
    *,
    max_rows: int,
    balanced: bool,
    seed: int,
) -> np.ndarray:
    n_rows = int(len(y))
    if max_rows <= 0 or max_rows >= n_rows:
        return np.arange(n_rows, dtype=np.int64)
    weights = np.asarray(sample_weight, dtype=np.float64).copy()
    weights = np.where(np.isfinite(weights) & (weights > 0), weights, 1.0)
    if balanced:
        y = np.asarray(y, dtype=int)
        for cls in (0, 1):
            mask = y == cls
            class_sum = float(weights[mask].sum())
            if class_sum > 0:
                weights[mask] *= 0.5 / class_sum
    total = float(weights.sum())
    if total <= 0:
        probabilities = None
    else:
        probabilities = weights / total
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n_rows, size=int(max_rows), replace=False, p=probabilities)).astype(np.int64)


def _mlp_fit_supports_sample_weight() -> bool:
    return "sample_weight" in inspect.signature(MLPClassifier.fit).parameters


def _clean_sample_weight(sample_weight: np.ndarray) -> np.ndarray:
    weights = np.asarray(sample_weight, dtype=np.float64).copy()
    return np.where(np.isfinite(weights) & (weights > 0), weights, 1.0)


def _weighted_resample_indices(
    y: np.ndarray,
    sample_weight: np.ndarray,
    *,
    n_draws: int,
    balanced: bool,
    seed: int,
) -> np.ndarray:
    weights = _clean_sample_weight(sample_weight)
    if balanced:
        y = np.asarray(y, dtype=int)
        for cls in (0, 1):
            mask = y == cls
            class_sum = float(weights[mask].sum())
            if class_sum > 0:
                weights[mask] *= 0.5 / class_sum
    total = float(weights.sum())
    probabilities = None if total <= 0 else weights / total
    rng = np.random.default_rng(seed)
    return rng.choice(len(y), size=int(n_draws), replace=True, p=probabilities).astype(np.int64)


def _fit_estimator_with_sample_weight(
    model: Pipeline,
    X: sparse.csr_matrix,
    y: np.ndarray,
    sample_weight: np.ndarray,
    *,
    args: argparse.Namespace,
    seed: int,
) -> dict[str, Any]:
    requested_mode = str(args.sample_weight_mode)
    supports_fit_weight = _mlp_fit_supports_sample_weight()
    if requested_mode == "none":
        model.fit(X, y)
        return {
            "requested_sample_weight_mode": requested_mode,
            "effective_sample_weight_mode": "none",
            "fit_rows": int(X.shape[0]),
            "supports_fit_sample_weight": bool(supports_fit_weight),
        }
    if requested_mode == "fit" and not supports_fit_weight:
        raise RuntimeError(
            "This sklearn MLPClassifier does not support fit(..., sample_weight=...). "
            "Use --sample-weight-mode weighted_resample, or upgrade sklearn to a version "
            "whose MLPClassifier.fit signature includes sample_weight."
        )
    if requested_mode == "fit" or (requested_mode == "auto" and supports_fit_weight):
        model.fit(X, y, mlp__sample_weight=_clean_sample_weight(sample_weight))
        return {
            "requested_sample_weight_mode": requested_mode,
            "effective_sample_weight_mode": "fit",
            "fit_rows": int(X.shape[0]),
            "supports_fit_sample_weight": bool(supports_fit_weight),
        }

    n_draws = (
        int(args.weighted_resample_size_per_head)
        if int(args.weighted_resample_size_per_head) > 0
        else int(X.shape[0])
    )
    resample_idx = _weighted_resample_indices(
        y,
        sample_weight,
        n_draws=n_draws,
        balanced=bool(args.balanced_row_sample),
        seed=int(seed),
    )
    model.fit(X[resample_idx], y[resample_idx])
    return {
        "requested_sample_weight_mode": requested_mode,
        "effective_sample_weight_mode": "weighted_resample",
        "fit_rows": int(n_draws),
        "supports_fit_sample_weight": bool(supports_fit_weight),
    }


def _make_estimator(args: argparse.Namespace, seed: int):
    steps: list[tuple[str, Any]] = []
    if args.scaler == "maxabs":
        steps.append(("scale", MaxAbsScaler(copy=False)))
    elif args.scaler == "standard":
        steps.append(("scale", StandardScaler(with_mean=False, copy=False)))
    steps.append(
        (
            "mlp",
            MLPClassifier(
                hidden_layer_sizes=tuple(int(x) for x in args.hidden),
                activation="relu",
                solver="adam",
                alpha=float(args.alpha),
                batch_size=int(args.batch_size),
                learning_rate_init=float(args.learning_rate_init),
                max_iter=int(args.max_iter),
                early_stopping=True,
                validation_fraction=float(args.validation_fraction),
                n_iter_no_change=int(args.n_iter_no_change),
                random_state=int(seed),
                verbose=bool(args.verbose),
            ),
        )
    )
    return Pipeline(steps)


def _safe_predict_proba(model: Any, matrix: sparse.csr_matrix) -> np.ndarray:
    probabilities = model.predict_proba(matrix)
    if probabilities.shape[1] == 1:
        return np.zeros(matrix.shape[0], dtype=np.float64)
    return probabilities[:, 1].astype(np.float64)


def _combine_blocks(
    blocks: dict[str, dict[str, sparse.spmatrix]],
    names: list[str],
    split: str,
    n_rows: int,
) -> sparse.csr_matrix:
    parts = [blocks[name][split] for name in names if name in blocks]
    if not parts:
        return sparse.csr_matrix((n_rows, 0))
    return sparse.hstack(parts, format="csr")


def _assemble_streamed_block(
    *,
    block_name: str,
    split_name: str,
    n_rows: int,
    row_parts: list[np.ndarray],
    matrix_parts: list[sparse.spmatrix],
) -> sparse.csr_matrix:
    if not matrix_parts:
        raise RuntimeError(f"No rows matched split={split_name} for TF-IDF block {block_name}.")
    rows = np.concatenate(row_parts)
    matrix = sparse.vstack(matrix_parts, format="csr")
    order = np.argsort(rows)
    rows = rows[order]
    matrix = matrix[order]
    expected = np.arange(n_rows, dtype=np.int64)
    if len(rows) != n_rows or not np.array_equal(rows, expected):
        raise RuntimeError(
            f"One-pass TF-IDF row alignment failed for split={split_name}, block={block_name}: "
            f"matched={len(rows)} expected={n_rows}"
        )
    return matrix


def _transform_tfidf_blocks_one_pass(
    *,
    prefix_table_path: Path,
    feature_engineer: FeatureEngineer,
    df_train: pd.DataFrame,
    df_valid: pd.DataFrame,
    df_test: pd.DataFrame,
    column_names: list[str],
    batch_size: int,
) -> dict[str, dict[str, sparse.csr_matrix]]:
    import pyarrow.parquet as parquet

    active_names = [
        name
        for name in column_names
        if name in feature_engineer.tfidf_vectorizers
        and name in feature_engineer.active_text_columns
    ]
    text_columns = sorted({feature_engineer.active_text_columns[name] for name in active_names})
    if not active_names:
        return {}

    split_frames = {"train": df_train, "valid": df_valid, "test": df_test}
    split_positions = {
        split_name: pd.Series(
            np.arange(len(frame), dtype=np.int64),
            index=frame["prefix_id"].astype(str),
        )
        for split_name, frame in split_frames.items()
    }
    row_parts: dict[str, dict[str, list[np.ndarray]]] = {
        name: {split_name: [] for split_name in split_frames}
        for name in active_names
    }
    matrix_parts: dict[str, dict[str, list[sparse.spmatrix]]] = {
        name: {split_name: [] for split_name in split_frames}
        for name in active_names
    }

    LOGGER.info(
        "One-pass streaming TF-IDF blocks: %s from columns %s",
        ", ".join(active_names),
        ", ".join(text_columns),
    )
    parquet_file = parquet.ParquetFile(prefix_table_path)
    for batch in parquet_file.iter_batches(
        batch_size=batch_size,
        columns=["prefix_id", *text_columns],
    ):
        batch_df = batch.to_pandas()
        batch_prefix_ids = batch_df["prefix_id"].astype(str)
        split_matches: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for split_name, positions in split_positions.items():
            output_rows = positions.reindex(batch_prefix_ids).to_numpy()
            keep_mask = ~pd.isna(output_rows)
            if keep_mask.any():
                split_matches[split_name] = (keep_mask, output_rows[keep_mask].astype(np.int64))
        if not split_matches:
            del batch_df
            continue

        for name in active_names:
            text_column = feature_engineer.active_text_columns[name]
            vectorizer = feature_engineer.tfidf_vectorizers[name]
            reducer = feature_engineer.tfidf_reducers.get(name)
            for split_name, (keep_mask, output_rows) in split_matches.items():
                texts = batch_df.loc[keep_mask, text_column].fillna("")
                X_tfidf = vectorizer.transform(texts)
                del texts
                if reducer is not None:
                    X_tfidf = sparse.csr_matrix(reducer.transform(X_tfidf).astype(np.float32))
                else:
                    X_tfidf = X_tfidf.tocsr()
                row_parts[name][split_name].append(output_rows)
                matrix_parts[name][split_name].append(X_tfidf)
        del batch_df

    blocks: dict[str, dict[str, sparse.csr_matrix]] = {}
    for name in active_names:
        blocks[name] = {}
        for split_name, frame in split_frames.items():
            blocks[name][split_name] = _assemble_streamed_block(
                block_name=name,
                split_name=split_name,
                n_rows=len(frame),
                row_parts=row_parts[name][split_name],
                matrix_parts=matrix_parts[name][split_name],
            )
    return blocks


def _transform_tfidf_blocks_one_pass_global(
    *,
    prefix_table_path: Path,
    feature_engineer: FeatureEngineer,
    frame: pd.DataFrame,
    column_names: list[str],
    batch_size: int,
) -> dict[str, sparse.csr_matrix]:
    import pyarrow.parquet as parquet

    active_names = [
        name
        for name in column_names
        if name in feature_engineer.tfidf_vectorizers
        and name in feature_engineer.active_text_columns
    ]
    text_columns = sorted({feature_engineer.active_text_columns[name] for name in active_names})
    if not active_names:
        return {}

    prefix_ids = frame["prefix_id"].astype(str)
    if prefix_ids.duplicated().any():
        raise RuntimeError("Global matrix cache requires unique prefix_id values.")
    positions = pd.Series(np.arange(len(frame), dtype=np.int64), index=prefix_ids)
    row_parts: dict[str, list[np.ndarray]] = {name: [] for name in active_names}
    matrix_parts: dict[str, list[sparse.spmatrix]] = {name: [] for name in active_names}

    LOGGER.info(
        "One-pass global TF-IDF blocks: %s from columns %s",
        ", ".join(active_names),
        ", ".join(text_columns),
    )
    parquet_file = parquet.ParquetFile(prefix_table_path)
    for batch in parquet_file.iter_batches(
        batch_size=batch_size,
        columns=["prefix_id", *text_columns],
    ):
        batch_df = batch.to_pandas()
        batch_prefix_ids = batch_df["prefix_id"].astype(str)
        output_rows = positions.reindex(batch_prefix_ids).to_numpy()
        keep_mask = ~pd.isna(output_rows)
        if not keep_mask.any():
            del batch_df
            continue
        output_rows = output_rows[keep_mask].astype(np.int64)

        for name in active_names:
            text_column = feature_engineer.active_text_columns[name]
            vectorizer = feature_engineer.tfidf_vectorizers[name]
            reducer = feature_engineer.tfidf_reducers.get(name)
            texts = batch_df.loc[keep_mask, text_column].fillna("")
            X_tfidf = vectorizer.transform(texts)
            del texts
            if reducer is not None:
                X_tfidf = sparse.csr_matrix(reducer.transform(X_tfidf).astype(np.float32))
            else:
                X_tfidf = X_tfidf.tocsr()
            row_parts[name].append(output_rows)
            matrix_parts[name].append(X_tfidf)
        del batch_df

    return {
        name: {
            "global": _assemble_streamed_block(
                block_name=name,
                split_name="global",
                n_rows=len(frame),
                row_parts=row_parts[name],
                matrix_parts=matrix_parts[name],
            )
        }
        for name in active_names
    }


def _build_global_variant_matrices_fast(
    *,
    prefix_path: Path,
    feature_engineer: FeatureEngineer,
    frame: pd.DataFrame,
    needs_thought: bool,
    eager_load_text_columns: bool,
    text_batch_size: int,
) -> dict[str, sparse.csr_matrix]:
    tfidf_af_cols = list(TFIDF_ACTION_FEEDBACK.keys())
    tfidf_thought_cols = list(TFIDF_THOUGHT.keys())
    tfidf_af_thought_cols = tfidf_af_cols + tfidf_thought_cols
    with timer(LOGGER, "Build global Dense / AF / Thought matrices"):
        X_dense = sparse.csr_matrix(feature_engineer.transform_dense(frame))
        if eager_load_text_columns:
            X_af = feature_engineer.transform_tfidf_subset(frame, tfidf_af_cols)
            if needs_thought:
                X_thought = feature_engineer.transform_tfidf_subset(frame, tfidf_thought_cols)
            else:
                X_thought = sparse.csr_matrix((len(frame), 0))
        else:
            requested = tfidf_af_thought_cols if needs_thought else tfidf_af_cols
            blocks = _transform_tfidf_blocks_one_pass_global(
                prefix_table_path=prefix_path,
                feature_engineer=feature_engineer,
                frame=frame,
                column_names=requested,
                batch_size=text_batch_size,
            )
            X_af = _combine_blocks(blocks, tfidf_af_cols, "global", len(frame))
            if needs_thought:
                X_thought = _combine_blocks(blocks, tfidf_thought_cols, "global", len(frame))
            else:
                X_thought = sparse.csr_matrix((len(frame), 0))

    return {
        "train_dense": X_dense,
        "train_af": X_af,
        "train_thought": X_thought,
    }


def _build_variant_matrices_fast(
    *,
    prefix_path: Path,
    feature_engineer: FeatureEngineer,
    df_train: pd.DataFrame,
    df_valid: pd.DataFrame,
    df_test: pd.DataFrame,
    needs_thought: bool,
    eager_load_text_columns: bool,
    text_batch_size: int,
) -> tuple[dict[str, sparse.csr_matrix], list[str], list[str]]:
    tfidf_af_cols = list(TFIDF_ACTION_FEEDBACK.keys())
    tfidf_thought_cols = list(TFIDF_THOUGHT.keys())
    tfidf_af_thought_cols = tfidf_af_cols + tfidf_thought_cols
    with timer(LOGGER, "Build Dense / AF / Thought matrices"):
        X_train_dense = sparse.csr_matrix(feature_engineer.transform_dense(df_train))
        X_valid_dense = sparse.csr_matrix(feature_engineer.transform_dense(df_valid))
        X_test_dense = sparse.csr_matrix(feature_engineer.transform_dense(df_test))
        if eager_load_text_columns:
            X_train_af = feature_engineer.transform_tfidf_subset(df_train, tfidf_af_cols)
            X_valid_af = feature_engineer.transform_tfidf_subset(df_valid, tfidf_af_cols)
            X_test_af = feature_engineer.transform_tfidf_subset(df_test, tfidf_af_cols)
            if needs_thought:
                X_train_thought = feature_engineer.transform_tfidf_subset(df_train, tfidf_thought_cols)
                X_valid_thought = feature_engineer.transform_tfidf_subset(df_valid, tfidf_thought_cols)
                X_test_thought = feature_engineer.transform_tfidf_subset(df_test, tfidf_thought_cols)
            else:
                X_train_thought = sparse.csr_matrix((len(df_train), 0))
                X_valid_thought = sparse.csr_matrix((len(df_valid), 0))
                X_test_thought = sparse.csr_matrix((len(df_test), 0))
        else:
            requested = tfidf_af_thought_cols if needs_thought else tfidf_af_cols
            blocks = _transform_tfidf_blocks_one_pass(
                prefix_table_path=prefix_path,
                feature_engineer=feature_engineer,
                df_train=df_train,
                df_valid=df_valid,
                df_test=df_test,
                column_names=requested,
                batch_size=text_batch_size,
            )
            X_train_af = _combine_blocks(blocks, tfidf_af_cols, "train", len(df_train))
            X_valid_af = _combine_blocks(blocks, tfidf_af_cols, "valid", len(df_valid))
            X_test_af = _combine_blocks(blocks, tfidf_af_cols, "test", len(df_test))
            if needs_thought:
                X_train_thought = _combine_blocks(blocks, tfidf_thought_cols, "train", len(df_train))
                X_valid_thought = _combine_blocks(blocks, tfidf_thought_cols, "valid", len(df_valid))
                X_test_thought = _combine_blocks(blocks, tfidf_thought_cols, "test", len(df_test))
            else:
                X_train_thought = sparse.csr_matrix((len(df_train), 0))
                X_valid_thought = sparse.csr_matrix((len(df_valid), 0))
                X_test_thought = sparse.csr_matrix((len(df_test), 0))

    matrices = {
        "train_dense": X_train_dense,
        "valid_dense": X_valid_dense,
        "test_dense": X_test_dense,
        "train_af": X_train_af,
        "valid_af": X_valid_af,
        "test_af": X_test_af,
        "train_thought": X_train_thought,
        "valid_thought": X_valid_thought,
        "test_thought": X_test_thought,
    }
    names_af = (
        list(feature_engineer.dense_feature_names)
        + feature_engineer.get_tfidf_feature_names_for_columns(tfidf_af_cols)
    )
    names_j = (
        list(feature_engineer.dense_feature_names)
        + feature_engineer.get_tfidf_feature_names_for_columns(tfidf_af_thought_cols)
    )
    return matrices, names_af, names_j


MATRIX_CACHE_FILENAMES = {
    "train_dense": "train_dense.npz",
    "valid_dense": "valid_dense.npz",
    "test_dense": "test_dense.npz",
    "train_af": "train_af.npz",
    "valid_af": "valid_af.npz",
    "test_af": "test_af.npz",
    "train_thought": "train_thought.npz",
    "valid_thought": "valid_thought.npz",
    "test_thought": "test_thought.npz",
}


def _path_signature(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def _split_signature(frame: pd.DataFrame) -> dict[str, Any]:
    prefix_ids = frame["prefix_id"].astype(str).to_numpy()
    digest = hashlib.sha256()
    for prefix_id in prefix_ids:
        digest.update(prefix_id.encode("utf-8"))
        digest.update(b"\0")
    return {
        "rows": int(len(prefix_ids)),
        "prefix_id_sha256": digest.hexdigest(),
    }


def _matrix_cache_required_keys(feature_set: str, needs_thought: bool) -> list[str]:
    keys = ["train_dense", "valid_dense", "test_dense"]
    if feature_set != "dense":
        keys.extend(["train_af", "valid_af", "test_af"])
        if needs_thought:
            keys.extend(["train_thought", "valid_thought", "test_thought"])
    return keys


def _global_matrix_cache_required_keys(feature_set: str, needs_thought: bool) -> list[str]:
    keys = ["train_dense"]
    if feature_set != "dense":
        keys.append("train_af")
        if needs_thought:
            keys.append("train_thought")
    return keys


def _global_matrix_cache_dir(output_dir: Path) -> Path:
    if output_dir.parent.name == "folds":
        return output_dir.parent.parent / "global_matrix_cache"
    return output_dir / "global_matrix_cache"


def _global_feature_frame(
    prefix_df: pd.DataFrame,
    *,
    verified_jsonl: Path,
    max_instances: int,
    mask_train_model_id: bool,
) -> pd.DataFrame:
    verified_ids = load_verified_instance_ids(verified_jsonl)
    available_ids = set(prefix_df["instance_id"].astype(str).unique())
    selected_instances = [item for item in verified_ids if item in available_ids]
    if max_instances and max_instances > 0:
        selected_instances = selected_instances[:max_instances]
    if not selected_instances:
        raise ValueError("No overlap between verified_jsonl and prefix_df instance_id.")
    frame = prefix_df[prefix_df["instance_id"].astype(str).isin(selected_instances)].copy()
    if mask_train_model_id:
        frame["model_id"] = "__MISSING__"
        if "model" in frame.columns:
            frame["model"] = "__MISSING__"
    return frame.reset_index(drop=True)


def _matrix_cache_metadata(
    *,
    feature_set: str,
    variants: list[str],
    needs_thought: bool,
    mask_train_model_id: bool,
    prefix_path: Path,
    feature_engineer_path: Path,
    split_meta: dict[str, Any],
    df_train: pd.DataFrame,
    df_valid: pd.DataFrame,
    df_test: pd.DataFrame,
    required_keys: list[str],
) -> dict[str, Any]:
    metadata = {
        "cache_version": 1,
        "feature_set": feature_set,
        "variants": list(variants),
        "needs_thought": bool(needs_thought),
        "prefix_table": str(prefix_path.resolve()),
        "prefix_table_stat": _path_signature(prefix_path),
        "feature_engineer_path": str(feature_engineer_path.resolve()),
        "feature_engineer_stat": _path_signature(feature_engineer_path),
        "split_context": {
            "holdout_models": split_meta.get("holdout_models"),
            "split_strategy": split_meta.get("split_strategy"),
            "mask_train_model_id": bool(mask_train_model_id),
        },
        "split_signature": {
            "train": _split_signature(df_train),
            "valid": _split_signature(df_valid),
            "test": _split_signature(df_test),
        },
        "matrix_keys": list(required_keys),
    }
    return json.loads(json.dumps(metadata, ensure_ascii=False, default=_json_default))


def _global_matrix_cache_metadata(
    *,
    feature_set: str,
    variants: list[str],
    needs_thought: bool,
    mask_train_model_id: bool,
    prefix_path: Path,
    verified_jsonl: Path,
    max_instances: int,
    feature_engineer_path: Path,
    global_frame: pd.DataFrame,
    required_keys: list[str],
) -> dict[str, Any]:
    metadata = {
        "cache_version": 1,
        "cache_scope": "global_prefix_rows",
        "feature_set": feature_set,
        "variants": list(variants),
        "needs_thought": bool(needs_thought),
        "mask_train_model_id": bool(mask_train_model_id),
        "max_instances": int(max_instances),
        "verified_jsonl": str(verified_jsonl.resolve()),
        "verified_jsonl_stat": _path_signature(verified_jsonl),
        "prefix_table": str(prefix_path.resolve()),
        "prefix_table_stat": _path_signature(prefix_path),
        "feature_engineer_path": str(feature_engineer_path.resolve()),
        "feature_engineer_stat": _path_signature(feature_engineer_path),
        "global_signature": _split_signature(global_frame),
        "matrix_keys": list(required_keys),
    }
    return json.loads(json.dumps(metadata, ensure_ascii=False, default=_json_default))


def _matrix_cache_complete(cache_dir: Path, expected_metadata: dict[str, Any], required_keys: list[str]) -> bool:
    metadata_path = cache_dir / "metadata.json"
    if not metadata_path.is_file():
        return False
    try:
        cached_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    if cached_metadata != expected_metadata:
        return False
    return all((cache_dir / MATRIX_CACHE_FILENAMES[key]).is_file() for key in required_keys)


def _save_sparse_atomic(matrix: sparse.spmatrix, path: Path) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("wb") as handle:
        sparse.save_npz(handle, matrix.tocsr(), compressed=True)
    tmp_path.replace(path)


def _save_matrix_cache(
    *,
    cache_dir: Path,
    matrices: dict[str, sparse.spmatrix],
    metadata: dict[str, Any],
    required_keys: list[str],
) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    for key in required_keys:
        filename = MATRIX_CACHE_FILENAMES[key]
        LOGGER.info("Saving matrix cache: %s", cache_dir / filename)
        _save_sparse_atomic(matrices[key], cache_dir / filename)
    (cache_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )
    LOGGER.info("Saved matrix cache metadata: %s", cache_dir / "metadata.json")


def _load_matrix_cache(cache_dir: Path, required_keys: list[str]) -> dict[str, sparse.csr_matrix]:
    LOGGER.info("Loading matrix cache from %s", cache_dir)
    return {
        key: sparse.load_npz(cache_dir / MATRIX_CACHE_FILENAMES[key]).tocsr()
        for key in required_keys
    }


def _row_indices_for_global_cache(global_frame: pd.DataFrame, frame: pd.DataFrame) -> np.ndarray:
    positions = pd.Series(
        np.arange(len(global_frame), dtype=np.int64),
        index=global_frame["prefix_id"].astype(str),
    )
    indices = positions.reindex(frame["prefix_id"].astype(str)).to_numpy()
    if pd.isna(indices).any():
        missing = frame.loc[pd.isna(indices), "prefix_id"].astype(str).head(5).tolist()
        raise RuntimeError(f"Global matrix cache missing split prefix_id values: {missing}")
    return indices.astype(np.int64)


def _slice_global_matrices(
    *,
    global_matrices: dict[str, sparse.csr_matrix],
    global_frame: pd.DataFrame,
    df_train: pd.DataFrame,
    df_valid: pd.DataFrame,
    df_test: pd.DataFrame,
    feature_set: str,
    needs_thought: bool,
) -> dict[str, sparse.csr_matrix]:
    split_frames = {"train": df_train, "valid": df_valid, "test": df_test}
    split_indices = {
        split_name: _row_indices_for_global_cache(global_frame, frame)
        for split_name, frame in split_frames.items()
    }
    matrices: dict[str, sparse.csr_matrix] = {}
    for split_name, frame in split_frames.items():
        idx = split_indices[split_name]
        matrices[f"{split_name}_dense"] = global_matrices["train_dense"][idx].tocsr()
        if feature_set != "dense":
            matrices[f"{split_name}_af"] = global_matrices["train_af"][idx].tocsr()
            if needs_thought:
                matrices[f"{split_name}_thought"] = global_matrices["train_thought"][idx].tocsr()
            else:
                matrices[f"{split_name}_thought"] = sparse.csr_matrix((len(frame), 0))
    return matrices


def _binary_metric_row(model_name: str, split: str, y: np.ndarray, p: np.ndarray) -> dict[str, Any]:
    row: dict[str, Any] = {
        "model": model_name,
        "split": split,
        "rows": int(len(y)),
        "pos_rate": float(np.mean(y)),
        "mean_prob": float(np.mean(p)),
    }
    try:
        row["roc_auc"] = float(roc_auc_score(y, p))
    except Exception:
        row["roc_auc"] = float("nan")
    try:
        row["pr_auc"] = float(average_precision_score(y, p))
    except Exception:
        row["pr_auc"] = float("nan")
    try:
        row["brier"] = float(brier_score_loss(y, p))
    except Exception:
        row["brier"] = float("nan")
    try:
        row["log_loss"] = float(log_loss(y, np.clip(p, 1e-6, 1 - 1e-6), labels=[0, 1]))
    except Exception:
        row["log_loss"] = float("nan")
    return row


def main() -> int:
    args = parse_args()
    max_cpu_threads = _set_cpu_thread_limits(args.max_cpu_threads)
    run_root = PROJECT_ROOT / "runs" / args.run_name
    _set_run_dirs(run_root)
    rebind_all_file_loggers()
    LOGGER.info("CPU thread cap: %s; direct MLP output=%s", max_cpu_threads, args.output_dir)

    output_dir = args.output_dir
    output_model_dir = output_dir / "models"
    output_model_dir.mkdir(parents=True, exist_ok=True)

    prefix_path = args.prefix_table or DEFAULT_PREFIX_TABLE
    feature_engineer_path = config.MODEL_DIR / "feature_engineer_with_model.pkl"
    feature_engineer = FeatureEngineer.load(feature_engineer_path)
    _repair_unpickled_tfidf_for_local_sklearn(feature_engineer)
    required_columns = _required_columns(feature_engineer, include_text=args.eager_load_text_columns)
    if args.feature_set == "dense":
        specs = [
            {
                "predictor": "Dense_DirectMLP",
                "base": "dense",
                "description": "Dense features only",
            }
        ]
        needs_thought = False
    else:
        specs = _selected_specs(args.variants)
        needs_thought = any(spec["base"] == "af_thought" for spec in specs)

    with timer(LOGGER, "Load cached prefix table and build split"):
        prefix_df = _load_prefix_table(prefix_path, required_columns)
        excluded_train_models = sorted({str(item) for item in (args.exclude_train_models or []) if str(item)})
        if excluded_train_models:
            before_models = int(prefix_df["model_id"].nunique())
            before_rows = int(len(prefix_df))
            prefix_df = prefix_df.loc[
                ~prefix_df["model_id"].astype(str).isin(set(excluded_train_models))
            ].copy()
            LOGGER.info(
                "Dropped %d configured-excluded model(s) before split: kept %d/%d models, %d/%d rows.",
                len(excluded_train_models),
                int(prefix_df["model_id"].nunique()),
                before_models,
                int(len(prefix_df)),
                before_rows,
            )
        global_feature_df = _global_feature_frame(
            prefix_df,
            verified_jsonl=args.verified_jsonl,
            max_instances=args.max_instances,
            mask_train_model_id=bool(args.mask_train_model_id),
        )
        df_train, df_valid, df_test, split_meta, split_summary = _build_split(
            prefix_df,
            verified_jsonl=args.verified_jsonl,
            holdout_models=args.holdout_models,
            max_instances=args.max_instances,
            split_strategy=args.split_strategy,
            valid_traj_ratio=args.valid_traj_ratio,
            valid_per_instance=args.valid_per_instance,
            valid_models_per_instance=args.valid_models_per_instance,
            shadow_valid_max_trajectories=0,
            seed=args.seed,
            smoke_trajectories_per_split=0,
            mask_train_model_id=args.mask_train_model_id,
        )
        split_meta["learner"] = "direct_mlp"
        split_meta["safe_label_min_step"] = int(args.safe_label_min_step)
        split_meta["excluded_train_models"] = excluded_train_models
        split_meta["mlp_config"] = {
            "hidden": args.hidden,
            "alpha": args.alpha,
            "learning_rate_init": args.learning_rate_init,
            "batch_size": args.batch_size,
            "max_iter": args.max_iter,
            "scaler": args.scaler,
            "max_train_rows_per_head": args.max_train_rows_per_head,
            "balanced_row_sample": args.balanced_row_sample,
            "sample_weight_mode": args.sample_weight_mode,
            "weighted_resample_size_per_head": args.weighted_resample_size_per_head,
            "sklearn_mlp_fit_supports_sample_weight": _mlp_fit_supports_sample_weight(),
        }
        del prefix_df
        gc.collect()

    dense_feature_names = list(feature_engineer.dense_feature_names)
    tfidf_af_cols = list(TFIDF_ACTION_FEEDBACK.keys())
    tfidf_thought_cols = list(TFIDF_THOUGHT.keys())
    tfidf_af_thought_cols = tfidf_af_cols + tfidf_thought_cols
    if args.feature_set == "dense":
        names_af = dense_feature_names
        names_j = dense_feature_names
    else:
        names_af = dense_feature_names + feature_engineer.get_tfidf_feature_names_for_columns(tfidf_af_cols)
        names_j = dense_feature_names + feature_engineer.get_tfidf_feature_names_for_columns(tfidf_af_thought_cols)
    matrix_cache_dir = args.matrix_cache_dir or (output_dir / "matrix_cache")
    required_keys = _matrix_cache_required_keys(args.feature_set, needs_thought)
    use_global_matrix_cache = bool(args.cache_matrices and args.mask_train_model_id)
    global_matrix_cache_dir = _global_matrix_cache_dir(output_dir)
    global_required_keys = _global_matrix_cache_required_keys(args.feature_set, needs_thought)
    global_matrix_cache_metadata = _global_matrix_cache_metadata(
        feature_set=args.feature_set,
        variants=list(args.variants),
        needs_thought=needs_thought,
        mask_train_model_id=bool(args.mask_train_model_id),
        prefix_path=prefix_path,
        verified_jsonl=args.verified_jsonl,
        max_instances=args.max_instances,
        feature_engineer_path=feature_engineer_path,
        global_frame=global_feature_df,
        required_keys=global_required_keys,
    )
    matrix_cache_metadata = _matrix_cache_metadata(
        feature_set=args.feature_set,
        variants=list(args.variants),
        needs_thought=needs_thought,
        mask_train_model_id=bool(args.mask_train_model_id),
        prefix_path=prefix_path,
        feature_engineer_path=feature_engineer_path,
        split_meta=split_meta,
        df_train=df_train,
        df_valid=df_valid,
        df_test=df_test,
        required_keys=required_keys,
    )

    y_success_train, y_failure_train = _safe_targets(df_train, args.safe_label_min_step)
    y_success_valid, y_failure_valid = _safe_targets(df_valid, args.safe_label_min_step)
    y_success_test, y_failure_test = _safe_targets(df_test, args.safe_label_min_step)
    w_train = df_train["sample_weight"].to_numpy(dtype=np.float32)
    w_valid = df_valid["sample_weight"].to_numpy(dtype=np.float32)

    if use_global_matrix_cache:
        if not args.rebuild_matrix_cache and _matrix_cache_complete(
            global_matrix_cache_dir,
            global_matrix_cache_metadata,
            global_required_keys,
        ):
            LOGGER.info("Global matrix cache hit: %s", global_matrix_cache_dir)
            global_matrices = _load_matrix_cache(global_matrix_cache_dir, global_required_keys)
        else:
            LOGGER.info(
                "Global matrix cache miss or rebuild requested: %s",
                global_matrix_cache_dir,
            )
            if args.feature_set == "dense":
                with timer(LOGGER, "Build global Dense matrices"):
                    global_matrices = {
                        "train_dense": sparse.csr_matrix(feature_engineer.transform_dense(global_feature_df)),
                    }
            else:
                global_matrices = _build_global_variant_matrices_fast(
                    prefix_path=prefix_path,
                    feature_engineer=feature_engineer,
                    frame=global_feature_df,
                    needs_thought=needs_thought,
                    eager_load_text_columns=args.eager_load_text_columns,
                    text_batch_size=args.text_batch_size,
                )
            _save_matrix_cache(
                cache_dir=global_matrix_cache_dir,
                matrices=global_matrices,
                metadata=global_matrix_cache_metadata,
                required_keys=global_required_keys,
            )
        matrices = _slice_global_matrices(
            global_matrices=global_matrices,
            global_frame=global_feature_df,
            df_train=df_train,
            df_valid=df_valid,
            df_test=df_test,
            feature_set=args.feature_set,
            needs_thought=needs_thought,
        )
        del global_matrices
    elif args.cache_matrices and not args.rebuild_matrix_cache and _matrix_cache_complete(
        matrix_cache_dir,
        matrix_cache_metadata,
        required_keys,
    ):
        LOGGER.info("Matrix cache hit: %s", matrix_cache_dir)
        matrices = _load_matrix_cache(matrix_cache_dir, required_keys)
    else:
        if args.cache_matrices:
            LOGGER.info(
                "Matrix cache miss or rebuild requested: %s",
                matrix_cache_dir,
            )
        if args.feature_set == "dense":
            with timer(LOGGER, "Build Dense matrices"):
                matrices = {
                    "train_dense": sparse.csr_matrix(feature_engineer.transform_dense(df_train)),
                    "valid_dense": sparse.csr_matrix(feature_engineer.transform_dense(df_valid)),
                    "test_dense": sparse.csr_matrix(feature_engineer.transform_dense(df_test)),
                }
        else:
            matrices, _, _ = _build_variant_matrices_fast(
                prefix_path=prefix_path,
                feature_engineer=feature_engineer,
                df_train=df_train,
                df_valid=df_valid,
                df_test=df_test,
                needs_thought=needs_thought,
                eager_load_text_columns=args.eager_load_text_columns,
                text_batch_size=args.text_batch_size,
            )
        if args.cache_matrices:
            _save_matrix_cache(
                cache_dir=matrix_cache_dir,
                matrices=matrices,
                metadata=matrix_cache_metadata,
                required_keys=required_keys,
            )
    del global_feature_df

    valid_pred = _prediction_frame(df_valid)
    test_pred = _prediction_frame(df_test)
    del df_train, df_valid, df_test
    gc.collect()

    calibration_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    fit_rows: list[dict[str, Any]] = []
    variant_rows: list[dict[str, Any]] = []
    predictors: list[str] = []
    for spec in specs:
        public_name = (
            spec["predictor"]
            if spec["base"] == "dense"
            else _mlp_predictor_name(spec["predictor"])
        )
        if spec["base"] == "dense":
            X_train = matrices["train_dense"]
            X_valid = matrices["valid_dense"]
            X_test = matrices["test_dense"]
            feature_names = list(feature_engineer.dense_feature_names)
            removed_names: list[str] = []
        else:
            X_train, X_valid, X_test, feature_names, removed_names = _matrices_for_spec(
                spec=spec,
                matrices=matrices,
                names_af=names_af,
                names_j=names_j,
            )
        predictors.append(public_name)
        for head_name, y_train, y_valid, y_test, column_prefix, seed_offset in (
            ("safe_success", y_success_train, y_success_valid, y_success_test, "success", 0),
            ("safe_failure", y_failure_train, y_failure_valid, y_failure_test, "failure", 11),
        ):
            model_name = f"{public_name}__{head_name}"
            indices = _sample_training_indices(
                y_train,
                w_train,
                max_rows=int(args.max_train_rows_per_head),
                balanced=bool(args.balanced_row_sample),
                seed=int(args.seed + seed_offset),
            )
            with timer(LOGGER, f"Train {model_name} on {len(indices)} rows"):
                model = _make_estimator(args, seed=args.seed + seed_offset)
                fit_info = _fit_estimator_with_sample_weight(
                    model,
                    X_train[indices],
                    y_train[indices],
                    w_train[indices],
                    args=args,
                    seed=int(args.seed + seed_offset + 101),
                )
                save_model(model, output_model_dir / f"{model_name}.pkl")
                valid_raw = _safe_predict_proba(model, X_valid)
                test_raw = _safe_predict_proba(model, X_test)
                calibrator = fit_sigmoid_calibrator(valid_raw, y_valid, sample_weight=w_valid)
                valid_cal = calibrator.predict(valid_raw)
                test_cal = calibrator.predict(test_raw)
                save_model(calibrator, output_model_dir / f"calibrator_{model_name}.pkl")
            fit_rows.append(
                {
                    "model": model_name,
                    "selected_rows": int(len(indices)),
                    "positive_rate_selected_rows": float(np.mean(y_train[indices])),
                    "sample_weight_sum_selected_rows": float(np.sum(w_train[indices])),
                    **fit_info,
                }
            )
            metric_rows.extend(
                [
                    _binary_metric_row(model_name, "valid_raw", y_valid, valid_raw),
                    _binary_metric_row(model_name, "valid_calibrated", y_valid, valid_cal),
                    _binary_metric_row(model_name, "test_raw", y_test, test_raw),
                    _binary_metric_row(model_name, "test_calibrated", y_test, test_cal),
                ]
            )
            calibration_rows.append(
                {
                    "head": head_name,
                    **calibration_summary_row(
                        model_name=model_name,
                        calibrator=calibrator,
                        y_valid=y_valid,
                        raw_prob_valid=valid_raw,
                        y_test=y_test,
                        raw_prob_test=test_raw,
                    ),
                }
            )
            valid_pred[_head_column(column_prefix, "raw", public_name)] = valid_raw.astype(np.float32)
            valid_pred[_head_column(column_prefix, "calibrated", public_name)] = valid_cal.astype(np.float32)
            test_pred[_head_column(column_prefix, "raw", public_name)] = test_raw.astype(np.float32)
            test_pred[_head_column(column_prefix, "calibrated", public_name)] = test_cal.astype(np.float32)

        variant_rows.append(
            {
                "predictor": public_name,
                "source_variant": spec["predictor"],
                "base_matrix": spec["base"],
                "description": spec["description"],
                "removed_feature_count": int(len(removed_names)),
                "kept_feature_count": int(len(feature_names)),
                "removed_feature_examples": "; ".join(removed_names[:30]),
            }
        )
        if args.low_memory:
            del X_train, X_valid, X_test
            gc.collect()

    output_dir.mkdir(parents=True, exist_ok=True)
    valid_pred.to_parquet(output_dir / "valid_predictions_safe_stop.parquet", index=False)
    test_pred.to_parquet(output_dir / "test_predictions_safe_stop.parquet", index=False)
    split_summary.to_csv(output_dir / "split_summary.csv", index=False)
    (output_dir / "split_metadata.json").write_text(
        json.dumps(split_meta, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )
    pd.DataFrame(variant_rows).to_csv(output_dir / "variant_manifest.csv", index=False)
    pd.DataFrame(calibration_rows).to_csv(output_dir / "safe_stop_calibration_summary.csv", index=False)
    pd.DataFrame(metric_rows).to_csv(output_dir / "head_metrics.csv", index=False)
    pd.DataFrame(fit_rows).to_csv(output_dir / "direct_mlp_fit_summary.csv", index=False)

    policies = _policy_grid(
        success_thresholds=args.success_thresholds,
        failure_thresholds=args.failure_thresholds,
        min_steps=args.policy_min_steps,
        consecutive_values=args.consecutive,
    )
    valid_grid, valid_per_agent = _evaluate_policies(
        valid_pred,
        run_label=output_dir.name,
        predictors=predictors,
        score_modes=args.score_modes,
        policies=policies,
    )
    selected = _select_policies(
        valid_grid,
        max_valid_abs_drop_pp=args.max_valid_abs_drop_pp,
        min_valid_decision_acc=args.min_valid_decision_acc,
        fallback_min_save_pct=args.fallback_min_save_pct,
    )
    test_selected = _evaluate_selected(test_pred, run_label=output_dir.name, selected=selected)

    valid_grid.to_csv(output_dir / "safe_stop_valid_policy_grid.csv", index=False)
    valid_per_agent.to_csv(output_dir / "safe_stop_valid_policy_per_agent.csv", index=False)
    selected.to_csv(output_dir / "safe_stop_selected_policies.csv", index=False)
    test_selected.to_csv(output_dir / "safe_stop_test_selected.csv", index=False)
    _write_report(output_dir, selected, test_selected)
    print(output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
