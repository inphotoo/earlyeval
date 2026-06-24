#!/usr/bin/env python3
"""Post-hoc gold raw-text TF-IDF ablations on the completed model-holdout run.

This script intentionally reuses expensive artifacts from an existing run:

* ``data/prefix_table_filtered.parquet``
* ``models/feature_engineer_with_model.pkl``
* ``reports/test_predictions_all_models.parquet`` for baseline comparison

It does not rebuild step tables, prefix tables, gold-answer joins, or refit
TF-IDF/SVD.  It only reconstructs the same model-holdout split, transforms the
already-fitted TF-IDF blocks, and trains three additional LightGBM variants:

* Dense + AF + gold patch TF-IDF
* Dense + AF + gold test patch TF-IDF
* Dense + AF + FAIL_TO_PASS TF-IDF
"""

from __future__ import annotations

import argparse
import gc
import math
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".mplconfig"))

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)

sys.path.insert(0, str(PROJECT_ROOT))

import config
from feature_engineer import (
    BOOL_FEATURES,
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    TFIDF_ACTION_FEEDBACK,
    FeatureEngineer,
)
from model_holdout_split import select_model_holdout_split
from probability_calibration import calibration_summary_row, fit_sigmoid_calibrator
from trainer import save_model, train_lightgbm
from utils import get_logger, rebind_all_file_loggers, timer


LOGGER = get_logger("gold_text_tfidf_ablation")

BASELINE_PREDICTORS = [
    "I_LightGBM_Dense_AF",
    "K_LightGBM_Dense_Full",
]

GOLD_TEXT_VARIANTS = [
    (
        "O_LightGBM_Dense_AF_GoldPatchTfidf",
        "tfidf_gold_patch",
        "Dense + AF + gold_patch_tfidf",
    ),
    (
        "P_LightGBM_Dense_AF_GoldTestPatchTfidf",
        "tfidf_gold_test_patch",
        "Dense + AF + gold_test_patch_tfidf",
    ),
    (
        "Q_LightGBM_Dense_AF_GoldFailToPassTfidf",
        "tfidf_gold_fail_to_pass",
        "Dense + AF + gold_fail_to_pass_tfidf",
    ),
]

GOLD_BLOCK_ALIASES = {
    "patch": "tfidf_gold_patch",
    "gold_patch": "tfidf_gold_patch",
    "test": "tfidf_gold_test_patch",
    "test_patch": "tfidf_gold_test_patch",
    "gold_test_patch": "tfidf_gold_test_patch",
    "fail": "tfidf_gold_fail_to_pass",
    "fail_to_pass": "tfidf_gold_fail_to_pass",
    "gold_fail_to_pass": "tfidf_gold_fail_to_pass",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-name",
        default="model_holdout_answer_calibrated_full",
        help="Completed run to reuse. Default: model_holdout_answer_calibrated_full",
    )
    parser.add_argument(
        "--prefix-table",
        type=Path,
        default=None,
        help="Optional explicit prefix_table_filtered.parquet path.",
    )
    parser.add_argument(
        "--verified-jsonl",
        type=Path,
        default=PROJECT_ROOT.parents[2] / "swebench_verified" / "test.jsonl",
        help="Same verified JSONL used by the completed model-holdout run.",
    )
    parser.add_argument("--holdout-models", default="auto_mid3")
    parser.add_argument("--max-instances", type=int, default=500)
    parser.add_argument(
        "--output-subdir",
        default="gold_text_tfidf_ablation",
        help="Subdirectory under reports/ for ablation outputs.",
    )
    parser.add_argument(
        "--smoke-trajectories-per-split",
        type=int,
        default=0,
        help="Debug only: sample whole trajectories per split before training.",
    )
    parser.add_argument(
        "--no-gpu-lgbm",
        action="store_true",
        help="Force CPU LightGBM for this post-hoc run.",
    )
    parser.add_argument(
        "--skip-ranking-reports",
        action="store_true",
        help="Only write predictions/metric CSVs; skip ref-style ranking reports.",
    )
    parser.add_argument(
        "--gold-svd-dim",
        type=int,
        default=None,
        help=(
            "Use only the first N dimensions from each gold raw-text SVD block. "
            "This reuses the fitted FeatureEngineer; it does not refit TF-IDF/SVD. "
            "Unset means use the full saved block dimension."
        ),
    )
    parser.add_argument(
        "--gold-svd-dims",
        nargs="+",
        default=None,
        help=(
            "Run a dimension sweep in one pass. Values may be integers or full/all. "
            "Example: --gold-svd-dims 4 8 16 32 64 full"
        ),
    )
    parser.add_argument(
        "--gold-blocks",
        nargs="+",
        default=["all"],
        help=(
            "Which gold raw-text blocks to run: all, patch, test_patch, fail_to_pass. "
            "Example: --gold-blocks patch test_patch"
        ),
    )
    parser.add_argument(
        "--prediction-baselines",
        nargs="+",
        default=BASELINE_PREDICTORS,
        help="Existing predictors copied from the completed prediction table.",
    )
    return parser.parse_args()


def _safe_artifact_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name)


def _parse_dim_specs(
    *,
    single_dim: int | None,
    dim_values: list[str] | None,
) -> tuple[list[tuple[str, int | None]], bool]:
    """Return ``[(display_label, dim_or_none)]`` and whether this is a sweep."""
    if dim_values:
        specs: list[tuple[str, int | None]] = []
        seen: set[str] = set()
        for raw in dim_values:
            value = str(raw).strip().lower()
            if value in {"full", "all", "none"}:
                key = "full"
                spec = ("Full", None)
            else:
                dim = int(value)
                if dim <= 0:
                    raise ValueError(f"gold SVD dimensions must be positive: {raw}")
                key = str(dim)
                spec = (f"Dim{dim}", dim)
            if key not in seen:
                specs.append(spec)
                seen.add(key)
        return specs, True
    if single_dim is None:
        return [("Full", None)], False
    if single_dim <= 0:
        raise ValueError("--gold-svd-dim must be positive when provided")
    return [(f"Dim{single_dim}", single_dim)], False


def _select_gold_variants(gold_blocks: list[str]) -> list[tuple[str, str, str]]:
    requested = [str(item).strip().lower() for item in gold_blocks if str(item).strip()]
    if not requested or "all" in requested:
        return list(GOLD_TEXT_VARIANTS)

    wanted_blocks: list[str] = []
    for item in requested:
        if item not in GOLD_BLOCK_ALIASES:
            raise ValueError(
                f"Unknown --gold-blocks value: {item}. "
                "Use all, patch, test_patch, fail_to_pass."
            )
        block = GOLD_BLOCK_ALIASES[item]
        if block not in wanted_blocks:
            wanted_blocks.append(block)
    return [variant for variant in GOLD_TEXT_VARIANTS if variant[1] in set(wanted_blocks)]


def _set_run_dirs(run_root: Path) -> None:
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
    config.DATA_DIR = run_root / "data"
    config.MODEL_DIR = run_root / "models"
    config.REPORT_DIR = run_root / "reports"
    config.LOG_DIR = run_root / "logs"
    config.STEP_TABLE_PATH = config.DATA_DIR / "step_table.parquet"
    config.PREFIX_TABLE_PATH = config.DATA_DIR / "prefix_table.parquet"
    config.PREFIX_TABLE_FILTERED_PATH = config.DATA_DIR / "prefix_table_filtered.parquet"
    for directory in [config.DATA_DIR, config.MODEL_DIR, config.REPORT_DIR, config.LOG_DIR]:
        directory.mkdir(parents=True, exist_ok=True)
    rebind_all_file_loggers()


def _required_columns(feature_engineer: FeatureEngineer) -> list[str]:
    text_blocks = list(TFIDF_ACTION_FEEDBACK.keys()) + [
        variant[1] for variant in GOLD_TEXT_VARIANTS
    ]
    text_columns = [
        feature_engineer.active_text_columns[name]
        for name in text_blocks
        if name in feature_engineer.active_text_columns
    ]
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
    return list(dict.fromkeys(metadata_columns + dense_columns + text_columns))


def _repair_unpickled_tfidf_for_local_sklearn(feature_engineer: FeatureEngineer) -> None:
    """Repair sklearn 1.6 pickles when loaded under older sklearn versions.

    The completed run's FeatureEngineer was saved with sklearn 1.6.x.  Some
    local shells still have sklearn 1.1.x, where ``TfidfTransformer`` expects
    ``_idf_diag`` instead of the directly pickled ``idf_`` array.  Re-assigning
    through the property rebuilds the expected sparse diagonal without refitting.
    """
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
        LOGGER.info("Repaired %s TF-IDF transformer(s) for local sklearn compatibility.", repaired)


def _load_prefix_table(path: Path, columns: list[str]) -> pd.DataFrame:
    import pyarrow.parquet as parquet

    available = set(parquet.ParquetFile(path).schema_arrow.names)
    missing = sorted(set(columns) - available)
    if missing:
        raise RuntimeError(f"Prefix table is missing required columns: {missing}")
    LOGGER.info("Loading reusable prefix table columns from %s", path)
    return pd.read_parquet(path, columns=columns)


def _sample_trajectories(frame: pd.DataFrame, max_trajectories: int, seed: int) -> pd.DataFrame:
    if max_trajectories <= 0:
        return frame
    traj_ids = frame["traj_id"].drop_duplicates().to_numpy()
    if len(traj_ids) <= max_trajectories:
        return frame
    rng = np.random.default_rng(seed)
    selected = set(rng.choice(traj_ids, size=max_trajectories, replace=False).tolist())
    return frame[frame["traj_id"].isin(selected)].copy()


def _reconstruct_model_holdout_split(
    prefix_df: pd.DataFrame,
    *,
    verified_jsonl: Path,
    holdout_models: str,
    max_instances: int,
    smoke_trajectories_per_split: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    train_idx, valid_idx, test_idx, split_meta = select_model_holdout_split(
        prefix_df,
        verified_jsonl=verified_jsonl,
        holdout_models=holdout_models,
        max_instances=max_instances,
    )
    df_train = prefix_df.loc[train_idx].copy()
    df_valid = prefix_df.loc[valid_idx].copy()
    df_test = prefix_df.loc[test_idx].copy()

    min_steps = int(config.MIN_TRAJECTORY_STEPS)
    short_train = set(
        df_train.groupby("traj_id")["n_steps_total_for_weighting"].first()
        .loc[lambda values: values < min_steps]
        .index
    )
    short_valid = set(
        df_valid.groupby("traj_id")["n_steps_total_for_weighting"].first()
        .loc[lambda values: values < min_steps]
        .index
    )
    df_train = df_train[~df_train["traj_id"].isin(short_train)].copy()
    df_valid = df_valid[~df_valid["traj_id"].isin(short_valid)].copy()

    if smoke_trajectories_per_split > 0:
        df_train = _sample_trajectories(df_train, smoke_trajectories_per_split, 1001)
        df_valid = _sample_trajectories(df_valid, smoke_trajectories_per_split, 1002)
        df_test = _sample_trajectories(df_test, smoke_trajectories_per_split, 1003)
        split_meta["smoke_trajectories_per_split"] = smoke_trajectories_per_split

    for frame in (df_train, df_valid, df_test):
        frame["orig_model_id"] = frame["model_id"].astype(str)
        frame["orig_model"] = frame.get("model", frame["model_id"]).astype(str)

    df_train["split"] = "train"
    df_valid["split"] = "valid"
    df_test["split"] = "test"
    df_train["model_id_input_mode"] = "train_seen"

    for frame, mode in ((df_valid, "valid_missing"), (df_test, "test_missing")):
        frame["model_id"] = "__MISSING__"
        if "model" in frame.columns:
            frame["model"] = "__MISSING__"
        frame["model_id_input_mode"] = mode

    split_meta.update(
        {
            "train_rows": int(len(df_train)),
            "valid_rows": int(len(df_valid)),
            "test_rows": int(len(df_test)),
            "train_trajectories": int(df_train["traj_id"].nunique()),
            "valid_trajectories": int(df_valid["traj_id"].nunique()),
            "test_trajectories": int(df_test["traj_id"].nunique()),
            "train_pos_rate": float(df_train["label"].mean()),
            "valid_pos_rate": float(df_valid["label"].mean()),
            "test_pos_rate": float(df_test["label"].mean()),
            "model_id_feature_mode": "train_seen_valid_missing_test_missing",
        }
    )
    return df_train, df_valid, df_test, split_meta


def _fit_lgbm_with_cpu_fallback(
    *,
    X_train,
    y_train: np.ndarray,
    w_train: np.ndarray,
    X_valid,
    y_valid: np.ndarray,
    w_valid: np.ndarray,
    feature_names: list[str],
    model_name: str,
):
    original_params = dict(config.LGBM_PARAMS)
    try:
        return train_lightgbm(
            X_train=X_train,
            y_train=y_train,
            w_train=w_train,
            X_valid=X_valid,
            y_valid=y_valid,
            w_valid=w_valid,
            feature_names=feature_names,
            model_name=model_name,
        )
    except Exception as exc:
        LOGGER.error("[%s] LightGBM failed with current params: %s", model_name, exc)
        LOGGER.info("[%s] Retrying with CPU LightGBM.", model_name)
        config.LGBM_PARAMS["device"] = "cpu"
        config.LGBM_PARAMS.pop("gpu_device_id", None)
        try:
            return train_lightgbm(
                X_train=X_train,
                y_train=y_train,
                w_train=w_train,
                X_valid=X_valid,
                y_valid=y_valid,
                w_valid=w_valid,
                feature_names=feature_names,
                model_name=f"{model_name}_cpu",
            )
        finally:
            config.LGBM_PARAMS.clear()
            config.LGBM_PARAMS.update(original_params)


def _metric_or_nan(metric_fn, labels: np.ndarray, probabilities: np.ndarray) -> float:
    try:
        return float(metric_fn(labels, probabilities))
    except Exception:
        return float("nan")


def _binary_metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    probabilities = np.asarray(probabilities, dtype=np.float64)
    predictions = (probabilities >= 0.5).astype(int)
    return {
        "rows": float(len(labels)),
        "pos_rate": float(np.mean(labels)),
        "mean_prob": float(np.mean(probabilities)),
        "accuracy_at_0_5": _metric_or_nan(accuracy_score, labels, predictions),
        "roc_auc": _metric_or_nan(roc_auc_score, labels, probabilities),
        "pr_auc": _metric_or_nan(average_precision_score, labels, probabilities),
        "brier": _metric_or_nan(brier_score_loss, labels, probabilities),
        "log_loss": _metric_or_nan(
            lambda truth, probs: log_loss(
                truth,
                np.clip(probs, 1e-6, 1.0 - 1e-6),
                labels=[0, 1],
            ),
            labels,
            probabilities,
        ),
    }


def _collect_metrics(pred_df: pd.DataFrame, predictors: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    prefix_rows: list[dict[str, Any]] = []
    final_rows: list[dict[str, Any]] = []
    final_idx = pred_df.groupby("traj_id", sort=False)["prefix_step_idx"].idxmax()
    final_df = pred_df.loc[final_idx].copy()

    for predictor in predictors:
        for score_mode, column_prefix in [("raw", "prob__"), ("calibrated", "prob_cal__")]:
            column = f"{column_prefix}{predictor}"
            if column not in pred_df.columns:
                continue
            prefix_metrics = _binary_metrics(
                pred_df["label"].to_numpy(dtype=int),
                pred_df[column].to_numpy(dtype=float),
            )
            prefix_rows.append(
                {
                    "scope": "prefix",
                    "score_mode": score_mode,
                    "predictor": predictor,
                    **prefix_metrics,
                }
            )
            final_metrics = _binary_metrics(
                final_df["label"].to_numpy(dtype=int),
                final_df[column].to_numpy(dtype=float),
            )
            final_rows.append(
                {
                    "scope": "final_step_trajectory",
                    "score_mode": score_mode,
                    "predictor": predictor,
                    **final_metrics,
                }
            )
    return pd.DataFrame(prefix_rows), pd.DataFrame(final_rows)


def _write_feature_importance(
    model,
    feature_names: list[str],
    output_path: Path,
    *,
    top_n: int = 80,
) -> None:
    try:
        gains = model.feature_importance(importance_type="gain")
        splits = model.feature_importance(importance_type="split")
    except Exception as exc:
        LOGGER.warning("Feature importance skipped for %s: %s", output_path.name, exc)
        return
    rows = pd.DataFrame(
        {
            "feature": feature_names,
            "gain": gains,
            "split": splits,
        }
    ).sort_values(["gain", "split"], ascending=False)
    rows.head(top_n).to_csv(output_path, index=False)


def _model_name_for_dim(base_name: str, dim_label: str, sweep_mode: bool) -> str:
    if not sweep_mode:
        return base_name
    return f"{base_name}_{dim_label}"


def _slice_block_dim(
    matrix: sparse.csr_matrix,
    feature_names: list[str],
    requested_dim: int | None,
) -> tuple[sparse.csr_matrix, list[str]]:
    if requested_dim is None or requested_dim <= 0:
        return matrix, feature_names
    keep_dim = min(int(requested_dim), int(matrix.shape[1]))
    return matrix[:, :keep_dim].tocsr(), feature_names[:keep_dim]


def _merge_with_existing_predictions(
    *,
    existing_path: Path,
    df_test: pd.DataFrame,
    new_prediction_df: pd.DataFrame,
    baseline_predictors: list[str],
    output_path: Path,
) -> pd.DataFrame:
    metadata_columns = [
        "prefix_id",
        "traj_id",
        "instance_id",
        "group_id",
        "prefix_step_idx",
        "n_steps_total_for_weighting",
        "sample_weight",
        "label",
        "split",
        "model_id",
        "model",
        "orig_model_id",
        "orig_model",
        "model_id_input_mode",
    ]

    if existing_path.is_file() and len(df_test) == 83169:
        existing = pd.read_parquet(existing_path)
        keep_columns = [column for column in metadata_columns if column in existing.columns]
        for predictor in baseline_predictors:
            for column_prefix in ("prob__", "prob_cal__"):
                column = f"{column_prefix}{predictor}"
                if column in existing.columns:
                    keep_columns.append(column)
        pred_df = existing[keep_columns].copy()
        key_columns = ["traj_id", "prefix_step_idx"]
        pred_df = pred_df.merge(
            new_prediction_df,
            on=key_columns,
            how="left",
            validate="one_to_one",
        )
        new_columns = [
            column for column in new_prediction_df.columns if column not in key_columns
        ]
        if pred_df[new_columns].isna().any().any():
            raise RuntimeError("Existing predictions did not align with new test predictions.")
    else:
        pred_df = df_test[[column for column in metadata_columns if column in df_test.columns]].copy()
        key_columns = ["traj_id", "prefix_step_idx"]
        pred_df = pred_df.merge(
            new_prediction_df,
            on=key_columns,
            how="left",
            validate="one_to_one",
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pred_df.to_parquet(output_path, index=False)
    LOGGER.info("Saved ablation predictions: %s", output_path)
    return pred_df


def _write_text_summary(
    output_path: Path,
    *,
    split_meta: dict[str, Any],
    prefix_metrics: pd.DataFrame,
    final_metrics: pd.DataFrame,
    predictors: list[str],
) -> None:
    def format_float(value: Any, digits: int = 4) -> str:
        try:
            numeric = float(value)
            if math.isnan(numeric):
                return "nan"
            return f"{numeric:.{digits}f}"
        except Exception:
            return "nan"

    lines = [
        "Gold raw-text TF-IDF ablation",
        "=" * 36,
        "",
        "Reused artifacts:",
        "  - prefix_table_filtered.parquet",
        "  - feature_engineer_with_model.pkl",
        "  - existing test_predictions_all_models.parquet for I/K comparison",
        "",
        "Not rerun:",
        "  - step table building",
        "  - prefix table building",
        "  - gold-answer feature join",
        "  - TF-IDF/SVD fitting",
        "",
        "Split:",
        f"  train rows/trajs: {split_meta['train_rows']} / {split_meta['train_trajectories']}",
        f"  valid rows/trajs: {split_meta['valid_rows']} / {split_meta['valid_trajectories']}",
        f"  test  rows/trajs: {split_meta['test_rows']} / {split_meta['test_trajectories']}",
        f"  valid/test model_id input: __MISSING__",
        f"  holdout models: {', '.join(split_meta.get('holdout_models', []))}",
        "",
        "Final-step trajectory metrics:",
    ]
    if split_meta.get("gold_svd_dims"):
        insert_at = lines.index("Split:")
        lines[insert_at:insert_at] = [
            f"Gold raw-text SVD dimensions used per block: {', '.join(map(str, split_meta['gold_svd_dims']))}",
            "",
        ]
    elif split_meta.get("gold_svd_dim") is not None:
        insert_at = lines.index("Split:")
        lines[insert_at:insert_at] = [
            f"Gold raw-text SVD dimensions used per block: {split_meta['gold_svd_dim']}",
            "",
        ]

    final_view = final_metrics[
        (final_metrics["score_mode"] == "calibrated")
        & (final_metrics["predictor"].isin(predictors))
    ].copy()
    if final_view.empty:
        final_view = final_metrics[final_metrics["predictor"].isin(predictors)].copy()
    final_view = final_view.sort_values("roc_auc", ascending=False)
    lines.append(
        "  "
        + f"{'predictor':48s} {'mode':10s} {'acc@0.5':>8s} {'auc':>8s} "
        + f"{'brier':>8s} {'meanP':>8s}"
    )
    for _, row in final_view.iterrows():
        lines.append(
            "  "
            + f"{str(row['predictor'])[:48]:48s} {str(row['score_mode'])[:10]:10s} "
            + f"{format_float(row['accuracy_at_0_5']):>8s} "
            + f"{format_float(row['roc_auc']):>8s} "
            + f"{format_float(row['brier']):>8s} "
            + f"{format_float(row['mean_prob']):>8s}"
        )

    lines.append("")
    lines.append("Prefix-row metrics:")
    prefix_view = prefix_metrics[
        (prefix_metrics["score_mode"] == "calibrated")
        & (prefix_metrics["predictor"].isin(predictors))
    ].copy()
    if prefix_view.empty:
        prefix_view = prefix_metrics[prefix_metrics["predictor"].isin(predictors)].copy()
    prefix_view = prefix_view.sort_values("roc_auc", ascending=False)
    lines.append(
        "  "
        + f"{'predictor':48s} {'mode':10s} {'acc@0.5':>8s} {'auc':>8s} "
        + f"{'brier':>8s} {'meanP':>8s}"
    )
    for _, row in prefix_view.iterrows():
        lines.append(
            "  "
            + f"{str(row['predictor'])[:48]:48s} {str(row['score_mode'])[:10]:10s} "
            + f"{format_float(row['accuracy_at_0_5']):>8s} "
            + f"{format_float(row['roc_auc']):>8s} "
            + f"{format_float(row['brier']):>8s} "
            + f"{format_float(row['mean_prob']):>8s}"
        )

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    LOGGER.info("Saved text summary: %s", output_path)


def _run_ranking_reports(
    *,
    predictions_path: Path,
    output_dir: Path,
    predictors: list[str],
) -> None:
    report_script = PROJECT_ROOT / "model_ranking_report_posthoc.py"
    for score_mode in ("raw", "calibrated"):
        mode_output = output_dir / f"model_ranking_report_{score_mode}"
        command = [
            sys.executable,
            str(report_script),
            "--predictions",
            str(predictions_path),
            "--output-dir",
            str(mode_output),
            "--score-mode",
            score_mode,
            "--prefix-models",
            *predictors,
        ]
        LOGGER.info("Running ranking report: %s", " ".join(command))
        subprocess.run(command, check=True)


def _write_dimension_effect_tables(
    *,
    output_dir: Path,
    variant_manifest: pd.DataFrame,
    prefix_metrics: pd.DataFrame,
    final_metrics: pd.DataFrame,
) -> None:
    if variant_manifest.empty:
        return
    keep_cols = [
        "predictor",
        "gold_block",
        "gold_block_label",
        "dim_label",
        "dim",
        "description",
    ]
    manifest = variant_manifest[keep_cols].copy()
    prefix_out = prefix_metrics.merge(manifest, on="predictor", how="inner")
    final_out = final_metrics.merge(manifest, on="predictor", how="inner")
    sort_cols = ["gold_block", "score_mode", "dim"]
    prefix_out.sort_values(sort_cols, na_position="last").to_csv(
        output_dir / "prefix_dimension_effect.csv",
        index=False,
    )
    final_out.sort_values(sort_cols, na_position="last").to_csv(
        output_dir / "final_step_dimension_effect.csv",
        index=False,
    )


def main() -> int:
    args = parse_args()
    dim_specs, sweep_mode = _parse_dim_specs(
        single_dim=args.gold_svd_dim,
        dim_values=args.gold_svd_dims,
    )
    selected_gold_variants = _select_gold_variants(args.gold_blocks)
    run_root = config.RUNTIME_ROOT / "runs" / args.run_name
    _set_run_dirs(run_root)

    if args.no_gpu_lgbm:
        config.LGBM_PARAMS["device"] = "cpu"
        config.LGBM_PARAMS.pop("gpu_device_id", None)

    prefix_path = args.prefix_table or config.PREFIX_TABLE_FILTERED_PATH
    feature_engineer_path = config.MODEL_DIR / "feature_engineer_with_model.pkl"
    existing_predictions_path = config.REPORT_DIR / "test_predictions_all_models.parquet"
    output_dir = config.REPORT_DIR / args.output_subdir
    output_model_dir = output_dir / "models"
    output_model_dir.mkdir(parents=True, exist_ok=True)

    if not prefix_path.is_file():
        raise FileNotFoundError(f"Missing reusable prefix table: {prefix_path}")
    if not feature_engineer_path.is_file():
        raise FileNotFoundError(f"Missing reusable FeatureEngineer: {feature_engineer_path}")

    feature_engineer = FeatureEngineer.load(feature_engineer_path)
    _repair_unpickled_tfidf_for_local_sklearn(feature_engineer)
    required_columns = _required_columns(feature_engineer)

    with timer(LOGGER, "Load prefix table and reconstruct split"):
        prefix_df = _load_prefix_table(prefix_path, required_columns)
        df_train, df_valid, df_test, split_meta = _reconstruct_model_holdout_split(
            prefix_df,
            verified_jsonl=args.verified_jsonl,
            holdout_models=args.holdout_models,
            max_instances=args.max_instances,
            smoke_trajectories_per_split=args.smoke_trajectories_per_split,
        )
        del prefix_df
        gc.collect()

    y_train = df_train["label"].to_numpy(dtype=int)
    y_valid = df_valid["label"].to_numpy(dtype=int)
    y_test = df_test["label"].to_numpy(dtype=int)
    w_train = df_train["sample_weight"].to_numpy(dtype=np.float32)
    w_valid = df_valid["sample_weight"].to_numpy(dtype=np.float32)

    LOGGER.info(
        "Split rows: train=%s valid=%s test=%s; pos rates train=%.4f valid=%.4f test=%.4f",
        len(df_train),
        len(df_valid),
        len(df_test),
        float(np.mean(y_train)),
        float(np.mean(y_valid)),
        float(np.mean(y_test)),
    )

    tfidf_af_cols = list(TFIDF_ACTION_FEEDBACK.keys())
    new_predictions = pd.DataFrame(
        {
            "traj_id": df_test["traj_id"].to_numpy(),
            "prefix_step_idx": df_test["prefix_step_idx"].to_numpy(),
        }
    )
    calibration_rows: list[dict[str, Any]] = []

    with timer(LOGGER, "Build reusable Dense + AF base matrices"):
        X_train_dense = sparse.csr_matrix(feature_engineer.transform_dense(df_train))
        X_valid_dense = sparse.csr_matrix(feature_engineer.transform_dense(df_valid))
        X_test_dense = sparse.csr_matrix(feature_engineer.transform_dense(df_test))
        X_train_af = feature_engineer.transform_tfidf_subset(df_train, tfidf_af_cols)
        X_valid_af = feature_engineer.transform_tfidf_subset(df_valid, tfidf_af_cols)
        X_test_af = feature_engineer.transform_tfidf_subset(df_test, tfidf_af_cols)
        X_train_base = sparse.hstack([X_train_dense, X_train_af], format="csr")
        X_valid_base = sparse.hstack([X_valid_dense, X_valid_af], format="csr")
        X_test_base = sparse.hstack([X_test_dense, X_test_af], format="csr")
        base_feature_names = (
            feature_engineer.dense_feature_names
            + feature_engineer.get_tfidf_feature_names_for_columns(tfidf_af_cols)
        )
        LOGGER.info("Dense + AF base shape: train=%s valid=%s test=%s", X_train_base.shape, X_valid_base.shape, X_test_base.shape)

    variant_manifest_rows: list[dict[str, Any]] = []

    for base_public_name, gold_block, description in selected_gold_variants:
        if gold_block not in feature_engineer.active_text_columns:
            raise RuntimeError(f"{gold_block} is not in the saved FeatureEngineer.")
        with timer(LOGGER, f"Transform gold block once: {gold_block}"):
            X_train_gold = feature_engineer.transform_tfidf_subset(df_train, [gold_block])
            X_valid_gold = feature_engineer.transform_tfidf_subset(df_valid, [gold_block])
            X_test_gold = feature_engineer.transform_tfidf_subset(df_test, [gold_block])
            full_gold_feature_names = feature_engineer.get_tfidf_feature_names_for_columns([gold_block])

        for dim_label, dim_value in dim_specs:
            public_name = _model_name_for_dim(base_public_name, dim_label, sweep_mode)
            with timer(LOGGER, f"Train {public_name}: {description}, {dim_label}"):
                train_gold_slice, gold_feature_names = _slice_block_dim(
                    X_train_gold,
                    full_gold_feature_names,
                    dim_value,
                )
                valid_gold_slice, _ = _slice_block_dim(
                    X_valid_gold,
                    full_gold_feature_names,
                    dim_value,
                )
                test_gold_slice, _ = _slice_block_dim(
                    X_test_gold,
                    full_gold_feature_names,
                    dim_value,
                )
                X_train_variant = sparse.hstack([X_train_base, train_gold_slice], format="csr")
                X_valid_variant = sparse.hstack([X_valid_base, valid_gold_slice], format="csr")
                X_test_variant = sparse.hstack([X_test_base, test_gold_slice], format="csr")
                feature_names = base_feature_names + gold_feature_names
                if len(feature_names) != X_train_variant.shape[1]:
                    raise RuntimeError(
                        f"{public_name}: feature name length {len(feature_names)} "
                        f"!= matrix columns {X_train_variant.shape[1]}"
                    )

                model = _fit_lgbm_with_cpu_fallback(
                    X_train=X_train_variant,
                    y_train=y_train,
                    w_train=w_train,
                    X_valid=X_valid_variant,
                    y_valid=y_valid,
                    w_valid=w_valid,
                    feature_names=feature_names,
                    model_name=public_name,
                )
                model_path = output_model_dir / f"{_safe_artifact_name(public_name)}.lgb"
                save_model(model, model_path)

                valid_raw = np.asarray(model.predict(X_valid_variant), dtype=np.float64)
                test_raw = np.asarray(model.predict(X_test_variant), dtype=np.float64)
                calibrator = fit_sigmoid_calibrator(valid_raw, y_valid, sample_weight=w_valid)
                test_cal = calibrator.predict(test_raw)
                save_model(
                    calibrator,
                    output_model_dir / f"calibrator_{_safe_artifact_name(public_name)}.pkl",
                )
                calibration_rows.append(
                    calibration_summary_row(
                        model_name=public_name,
                        calibrator=calibrator,
                        y_valid=y_valid,
                        raw_prob_valid=valid_raw,
                        y_test=y_test,
                        raw_prob_test=test_raw,
                    )
                )

                new_predictions[f"prob__{public_name}"] = test_raw.astype(np.float32)
                new_predictions[f"prob_cal__{public_name}"] = test_cal.astype(np.float32)
                _write_feature_importance(
                    model,
                    feature_names,
                    output_dir / f"feature_importance_{_safe_artifact_name(public_name)}.csv",
                )
                variant_manifest_rows.append(
                    {
                        "predictor": public_name,
                        "base_predictor": base_public_name,
                        "gold_block": gold_block,
                        "gold_block_label": description,
                        "dim_label": dim_label,
                        "dim": int(dim_value) if dim_value is not None else np.nan,
                        "description": f"{description}, {dim_label}",
                    }
                )

                del train_gold_slice, valid_gold_slice, test_gold_slice
                del X_train_variant, X_valid_variant, X_test_variant
                gc.collect()

        del X_train_gold, X_valid_gold, X_test_gold
        gc.collect()

    prediction_output_path = output_dir / "test_predictions_gold_text_tfidf_ablation.parquet"
    pred_df = _merge_with_existing_predictions(
        existing_path=existing_predictions_path,
        df_test=df_test,
        new_prediction_df=new_predictions,
        baseline_predictors=args.prediction_baselines,
        output_path=prediction_output_path,
    )

    variant_manifest = pd.DataFrame(variant_manifest_rows)
    variant_manifest.to_csv(output_dir / "variant_manifest.csv", index=False)
    all_predictors = list(args.prediction_baselines) + variant_manifest["predictor"].tolist()
    prefix_metrics, final_metrics = _collect_metrics(pred_df, all_predictors)
    prefix_metrics.to_csv(output_dir / "prefix_metrics.csv", index=False)
    final_metrics.to_csv(output_dir / "final_step_metrics.csv", index=False)
    _write_dimension_effect_tables(
        output_dir=output_dir,
        variant_manifest=variant_manifest,
        prefix_metrics=prefix_metrics,
        final_metrics=final_metrics,
    )
    pd.DataFrame(calibration_rows).to_csv(output_dir / "probability_calibration_summary.csv", index=False)
    split_meta["gold_svd_dim"] = args.gold_svd_dim
    split_meta["gold_svd_dims"] = [label for label, _ in dim_specs] if sweep_mode else None
    pd.Series(split_meta).to_json(output_dir / "split_reconstruction_summary.json", force_ascii=False, indent=2)
    _write_text_summary(
        output_dir / "summary.txt",
        split_meta=split_meta,
        prefix_metrics=prefix_metrics,
        final_metrics=final_metrics,
        predictors=all_predictors,
    )

    if not args.skip_ranking_reports:
        _run_ranking_reports(
            predictions_path=prediction_output_path,
            output_dir=output_dir,
            predictors=all_predictors,
        )

    LOGGER.info("Gold text TF-IDF ablation complete: %s", output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
