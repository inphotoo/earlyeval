#!/usr/bin/env python3
"""Retrain selected LightGBM models with a known-task shadow validation split.

This is a lightweight follow-up to the completed answer-aware model-holdout run.
It intentionally reuses expensive artifacts:

* ``runs/<run-name>/data/prefix_table_filtered.parquet``
* ``runs/<run-name>/models/feature_engineer_with_model.pkl``

It does not rebuild step tables, prefix tables, gold-answer joins, or refit
TF-IDF/SVD by default.  The default split strategy keeps the original heldout
test models unchanged, randomly holds out a few non-test agent models inside
each instance for validation/calibration, and trains on the remaining non-test
models.  The older full-shadow validation mode remains available only when
explicitly requested with ``--split-strategy all_non_test_shadow_valid``.
"""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import gc
import json
import math
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable

PROJECT_ROOT = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".mplconfig"))
THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "BLIS_NUM_THREADS",
)
for _thread_env_name in THREAD_ENV_VARS:
    os.environ.setdefault(_thread_env_name, "24")

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
from evaluator import plot_calibration
from feature_engineer import (
    BOOL_FEATURES,
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    TFIDF_ACTION_FEEDBACK,
    TFIDF_THOUGHT,
    FeatureEngineer,
)
from gold_text_tfidf_ablation_posthoc import (
    _fit_lgbm_with_cpu_fallback,
    _load_prefix_table,
    _repair_unpickled_tfidf_for_local_sklearn,
    _set_run_dirs,
    _write_feature_importance,
)
from model_holdout_split import choose_auto_holdout_models, load_verified_instance_ids
from probability_calibration import calibration_summary_row, fit_sigmoid_calibrator
from trainer import save_model
from utils import get_logger, rebind_all_file_loggers, timer


LOGGER = get_logger("shadow_valid_retrain")

DEFAULT_PREDICTORS = [
    "I_LightGBM_Dense_AF",
    "J_LightGBM_Dense_AF_Thought",
    "Abl_NoTaskSignal_LightGBM",
    "Abl_NoTaskPromptTfidf_LightGBM",
    "Abl_NoGoldAnswer_LightGBM",
    "Abl_NoTaskSignal_NoGoldAnswer_LightGBM",
    "Abl_NoFeedback_LightGBM",
    "Abl_NoAction_LightGBM",
    "Abl_NoThought_LightGBM",
    "Abl_ProcessOnly_LightGBM",
]


def _cpu_thread_count(max_cpu_threads: int | None) -> int:
    try:
        return max(1, int(max_cpu_threads or 1))
    except Exception:
        return 24


def _set_cpu_thread_limits(max_cpu_threads: int | None) -> int:
    threads = _cpu_thread_count(max_cpu_threads)
    for env_name in THREAD_ENV_VARS:
        os.environ[env_name] = str(threads)
    os.environ["SWE_MAX_CPU_THREADS"] = str(threads)
    return threads


def _thread_limited_env(max_cpu_threads: int | None) -> dict[str, str]:
    threads = _cpu_thread_count(max_cpu_threads)
    env = os.environ.copy()
    for env_name in THREAD_ENV_VARS:
        env[env_name] = str(threads)
    env["SWE_MAX_CPU_THREADS"] = str(threads)
    return env


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-name",
        default="model_holdout_answer_calibrated_full",
        help="Completed run whose cached prefix table / FeatureEngineer are reused.",
    )
    parser.add_argument("--prefix-table", type=Path, default=None)
    parser.add_argument(
        "--verified-jsonl",
        type=Path,
        default=PROJECT_ROOT.parents[2] / "swebench_verified" / "test.jsonl",
        help="Same verified JSONL used by the completed model-holdout run.",
    )
    parser.add_argument("--holdout-models", default="auto_mid3")
    parser.add_argument(
        "--max-cpu-threads",
        type=int,
        default=int(os.environ.get("SWE_MAX_CPU_THREADS", "24")),
        help="Maximum CPU threads for BLAS/OpenMP/LightGBM and posthoc subprocesses.",
    )
    parser.add_argument("--max-instances", type=int, default=500)
    parser.add_argument(
        "--split-strategy",
        choices=("all_non_test_shadow_valid", "per_instance_traj", "per_instance_model"),
        default="per_instance_model",
        help=(
            "per_instance_model: hold out random non-test agent models inside each "
            "instance for validation. per_instance_traj: hold out random non-test "
            "trajectories inside each instance. all_non_test_shadow_valid: train on all "
            "non-test rows and calibrate on a model_id-masked shadow copy."
        ),
    )
    parser.add_argument("--valid-traj-ratio", type=float, default=0.15)
    parser.add_argument(
        "--valid-per-instance",
        type=int,
        default=0,
        help="For per_instance_traj only: fixed validation trajectories per instance; 0 uses ratio.",
    )
    parser.add_argument(
        "--valid-models-per-instance",
        type=int,
        default=3,
        help=(
            "For per_instance_model only: randomly hold out up to this many non-test "
            "model_ids per instance for validation, always leaving at least one train model."
        ),
    )
    parser.add_argument(
        "--shadow-valid-max-trajectories",
        type=int,
        default=0,
        help=(
            "For all_non_test_shadow_valid only: train on all non-test trajectories, "
            "but sample at most this many trajectories for shadow validation/calibration. "
            "0 keeps the full shadow-valid set."
        ),
    )
    parser.add_argument("--seed", type=int, default=config.SPLIT_SEED)
    parser.add_argument(
        "--mask-train-model-id",
        action="store_true",
        help=(
            "Set train model_id/model to __MISSING__ as well as valid/test. "
            "This removes agent-model identity as a train-time shortcut without "
            "changing the saved FeatureEngineer."
        ),
    )
    parser.add_argument(
        "--lgbm-preset",
        choices=("default", "strong_reg"),
        default="default",
        help="Optional LightGBM parameter preset for independent regularization experiments.",
    )
    parser.add_argument(
        "--save-valid-predictions",
        action="store_true",
        help=(
            "Also write valid_predictions_shadow_valid_retrain.parquet. This makes "
            "validation-only threshold/policy selection cheap and avoids re-predicting valid prefixes later."
        ),
    )
    parser.add_argument(
        "--output-subdir",
        default="per_instance_model_valid3_retrain",
        help="New report subdirectory under the reused run's reports/ directory.",
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        default=["default"],
        help=(
            "Models to run: default/all, I, J, no_task_signal, no_task_tfidf, "
            "no_gold_answer, no_task_signal_no_gold_answer, no_feedback, "
            "no_action, no_thought, process_only."
        ),
    )
    parser.add_argument(
        "--smoke-trajectories-per-split",
        type=int,
        default=0,
        help="Debug only: sample whole trajectories per split before training.",
    )
    parser.add_argument(
        "--refit-feature-engineer",
        action="store_true",
        help=(
            "Refit FeatureEngineer on this new train split and save it only under the new "
            "output folder. Default is to reuse the completed run's fitted object."
        ),
    )
    parser.add_argument("--no-gpu-lgbm", action="store_true", help="Force CPU LightGBM.")
    parser.add_argument(
        "--low-memory",
        action="store_true",
        help=(
            "Avoid holding AF and AF+Thought base matrices at the same time. "
            "This rebuilds per-model hstack matrices, trading a little time for lower RAM."
        ),
    )
    parser.add_argument(
        "--cache-matrices",
        action="store_true",
        help=(
            "Save/load reusable Dense/AF/Thought sparse matrices under the output directory. "
            "Useful for resuming if training fails after the expensive transform step."
        ),
    )
    parser.add_argument(
        "--eager-load-text-columns",
        action="store_true",
        help=(
            "Load all TF-IDF text columns into the split DataFrames before transforming. "
            "Default streams one text column at a time to keep peak RAM lower."
        ),
    )
    parser.add_argument(
        "--text-batch-size",
        type=int,
        default=4096,
        help="Rows per parquet batch when streaming TF-IDF text columns.",
    )
    parser.add_argument(
        "--matrix-cache-dir",
        type=Path,
        default=None,
        help="Optional explicit matrix cache directory. Default: <output_dir>/matrix_cache.",
    )
    parser.add_argument("--skip-ranking-reports", action="store_true")
    parser.add_argument("--skip-diagnostics", action="store_true")
    return parser.parse_args()


def _safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name)


def _json_default(obj: Any) -> Any:
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def _require_lightgbm_available() -> None:
    try:
        import lightgbm  # noqa: F401
    except Exception as exc:
        raise RuntimeError(
            "lightgbm is not importable in this Python environment. "
            "Activate the environment used for the original SWE-bench run first "
            "(for example: `conda activate swebench`), then verify with "
            "`python -c \"import lightgbm, sys; print(sys.executable, lightgbm.__version__)\"`. "
            f"Current Python: {sys.executable}"
        ) from exc


def _apply_lgbm_preset(preset: str) -> dict[str, Any]:
    if preset == "default":
        return {}
    if preset != "strong_reg":
        raise ValueError(f"Unknown LightGBM preset: {preset}")
    updates: dict[str, Any] = {
        "learning_rate": 0.03,
        "num_leaves": 31,
        "max_depth": 6,
        "min_child_samples": 200,
        "subsample": 0.75,
        "bagging_freq": 1,
        "colsample_bytree": 0.70,
        "reg_alpha": 0.5,
        "reg_lambda": 10.0,
        "min_gain_to_split": 0.01,
    }
    config.LGBM_PARAMS.update(updates)
    return updates


def _required_columns(feature_engineer: FeatureEngineer, *, include_text: bool = True) -> list[str]:
    tfidf_cols = list(TFIDF_ACTION_FEEDBACK.keys()) + list(TFIDF_THOUGHT.keys())
    text_columns = []
    if include_text:
        text_columns = [
            feature_engineer.active_text_columns[name]
            for name in tfidf_cols
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


def _model_stats(work: pd.DataFrame) -> pd.DataFrame:
    stats = work.groupby("model_id").agg(
        trajs=("traj_id", "nunique"),
        instances=("instance_id", "nunique"),
        prefixes=("label", "size"),
        success=("label", "mean"),
    ).sort_values("success")
    traj_success = (
        work[["model_id", "traj_id", "label"]]
        .drop_duplicates("traj_id")
        .groupby("model_id")["label"]
        .mean()
    )
    stats["traj_success"] = traj_success
    return stats


def _resolve_holdout_models(work: pd.DataFrame, holdout_models: str) -> tuple[list[str], pd.DataFrame]:
    stats = _model_stats(work)
    heldout = choose_auto_holdout_models(stats, holdout_models)
    missing = [model for model in heldout if model not in stats.index]
    if missing:
        raise ValueError(f"Requested holdout models not found: {missing}")
    return heldout, stats


def _drop_short_trajectories(frame: pd.DataFrame) -> pd.DataFrame:
    min_steps = int(config.MIN_TRAJECTORY_STEPS)
    short = set(
        frame.groupby("traj_id")["n_steps_total_for_weighting"]
        .first()
        .loc[lambda values: values < min_steps]
        .index
    )
    if not short:
        return frame.copy()
    return frame[~frame["traj_id"].isin(short)].copy()


def _sample_trajectories(frame: pd.DataFrame, max_trajectories: int, seed: int) -> pd.DataFrame:
    if max_trajectories <= 0:
        return frame
    traj_ids = frame["traj_id"].drop_duplicates().to_numpy()
    if len(traj_ids) <= max_trajectories:
        return frame
    rng = np.random.default_rng(seed)
    selected = set(rng.choice(traj_ids, size=max_trajectories, replace=False).tolist())
    return frame[frame["traj_id"].isin(selected)].copy()


def _select_valid_traj_ids_per_instance(
    trainval: pd.DataFrame,
    *,
    ratio: float,
    fixed_per_instance: int,
    seed: int,
) -> set[str]:
    traj_meta = (
        trainval[["instance_id", "traj_id", "model_id", "label"]]
        .drop_duplicates("traj_id")
        .copy()
    )
    rng = np.random.default_rng(seed + 911)
    valid_ids: set[str] = set()
    for instance_id, part in traj_meta.groupby("instance_id", sort=False):
        traj_ids = part["traj_id"].astype(str).to_numpy()
        n_traj = len(traj_ids)
        if n_traj <= 1:
            continue
        if fixed_per_instance > 0:
            n_valid = min(fixed_per_instance, n_traj - 1)
        else:
            n_valid = int(round(n_traj * ratio))
            n_valid = max(1, min(n_valid, n_traj - 1))
        chosen = rng.choice(traj_ids, size=n_valid, replace=False).tolist()
        valid_ids.update(map(str, chosen))
    return valid_ids


def _select_valid_model_pairs_per_instance(
    trainval: pd.DataFrame,
    *,
    models_per_instance: int,
    seed: int,
) -> set[tuple[str, str]]:
    if models_per_instance <= 0:
        raise ValueError("--valid-models-per-instance must be > 0 for per_instance_model.")

    model_meta = (
        trainval[["instance_id", "model_id"]]
        .drop_duplicates()
        .assign(
            instance_id=lambda frame: frame["instance_id"].astype(str),
            model_id=lambda frame: frame["model_id"].astype(str),
        )
    )
    rng = np.random.default_rng(seed + 3571)
    valid_pairs: set[tuple[str, str]] = set()
    for instance_id, part in model_meta.groupby("instance_id", sort=False):
        model_ids = part["model_id"].to_numpy()
        n_models = len(model_ids)
        if n_models <= 1:
            continue
        n_valid = min(models_per_instance, n_models - 1)
        chosen = rng.choice(model_ids, size=n_valid, replace=False).tolist()
        valid_pairs.update((str(instance_id), str(model_id)) for model_id in chosen)
    return valid_pairs


def _attach_metadata_and_mask_model_id(
    df_train: pd.DataFrame,
    df_valid: pd.DataFrame,
    df_test: pd.DataFrame,
    *,
    mask_train_model_id: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    for frame in (df_train, df_valid, df_test):
        frame["orig_model_id"] = frame["model_id"].astype(str)
        frame["orig_model"] = frame.get("model", frame["model_id"]).astype(str)
    df_train["split"] = "train"
    df_valid["split"] = "valid"
    df_test["split"] = "test"
    if mask_train_model_id:
        df_train["model_id"] = "__MISSING__"
        if "model" in df_train.columns:
            df_train["model"] = "__MISSING__"
        df_train["model_id_input_mode"] = "train_missing"
    else:
        df_train["model_id_input_mode"] = "train_seen"
    for frame, mode in ((df_valid, "valid_missing"), (df_test, "test_missing")):
        frame["model_id"] = "__MISSING__"
        if "model" in frame.columns:
            frame["model"] = "__MISSING__"
        frame["model_id_input_mode"] = mode
    return df_train, df_valid, df_test


def _build_split(
    prefix_df: pd.DataFrame,
    *,
    verified_jsonl: Path,
    holdout_models: str,
    max_instances: int,
    split_strategy: str,
    valid_traj_ratio: float,
    valid_per_instance: int,
    valid_models_per_instance: int,
    shadow_valid_max_trajectories: int,
    seed: int,
    smoke_trajectories_per_split: int,
    mask_train_model_id: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any], pd.DataFrame]:
    verified_ids = load_verified_instance_ids(verified_jsonl)
    available_ids = set(prefix_df["instance_id"].astype(str).unique())
    selected_instances = [item for item in verified_ids if item in available_ids]
    if max_instances and max_instances > 0:
        selected_instances = selected_instances[:max_instances]
    if not selected_instances:
        raise ValueError("No overlap between verified_jsonl and prefix_df instance_id.")

    work = prefix_df[prefix_df["instance_id"].astype(str).isin(selected_instances)].copy()
    heldout, stats = _resolve_holdout_models(work, holdout_models)
    trainval = work[~work["model_id"].isin(heldout)].copy()
    df_test = work[work["model_id"].isin(heldout)].copy()
    if trainval.empty or df_test.empty:
        raise ValueError(f"Empty trainval/test split for heldout={heldout}")

    trainval = _drop_short_trajectories(trainval)
    valid_model_pairs: set[tuple[str, str]] = set()
    if split_strategy == "all_non_test_shadow_valid":
        df_train = trainval.copy()
        df_valid = trainval.copy()
        if shadow_valid_max_trajectories > 0:
            df_valid = _sample_trajectories(
                df_valid,
                max_trajectories=shadow_valid_max_trajectories,
                seed=seed + 2117,
            )
        valid_traj_ids: set[str] = set(df_valid["traj_id"].astype(str).unique())
        train_traj_ids: set[str] = set(df_train["traj_id"].astype(str).unique())
    elif split_strategy == "per_instance_traj":
        valid_traj_ids = _select_valid_traj_ids_per_instance(
            trainval,
            ratio=valid_traj_ratio,
            fixed_per_instance=valid_per_instance,
            seed=seed,
        )
        df_valid = trainval[trainval["traj_id"].astype(str).isin(valid_traj_ids)].copy()
        df_train = trainval[~trainval["traj_id"].astype(str).isin(valid_traj_ids)].copy()
        train_traj_ids = set(df_train["traj_id"].astype(str).unique())
        if df_valid.empty or df_train.empty:
            raise ValueError("per_instance_traj produced empty train or valid split.")
    elif split_strategy == "per_instance_model":
        valid_model_pairs = _select_valid_model_pairs_per_instance(
            trainval,
            models_per_instance=valid_models_per_instance,
            seed=seed,
        )
        if not valid_model_pairs:
            raise ValueError("per_instance_model selected zero validation model pairs.")
        row_model_pairs = pd.MultiIndex.from_frame(
            trainval[["instance_id", "model_id"]].astype(str)
        )
        valid_mask = row_model_pairs.isin(valid_model_pairs)
        df_valid = trainval[valid_mask].copy()
        df_train = trainval[~valid_mask].copy()
        valid_traj_ids = set(df_valid["traj_id"].astype(str).unique())
        train_traj_ids = set(df_train["traj_id"].astype(str).unique())
        if df_valid.empty or df_train.empty:
            raise ValueError("per_instance_model produced empty train or valid split.")
    else:
        raise ValueError(f"Unknown split strategy: {split_strategy}")

    if smoke_trajectories_per_split > 0:
        if split_strategy == "all_non_test_shadow_valid":
            df_train = _sample_trajectories(df_train, smoke_trajectories_per_split, seed + 1001)
            chosen = set(df_train["traj_id"].astype(str).unique())
            df_valid = df_valid[df_valid["traj_id"].astype(str).isin(chosen)].copy()
        else:
            df_train = _sample_trajectories(df_train, smoke_trajectories_per_split, seed + 1001)
            df_valid = _sample_trajectories(df_valid, smoke_trajectories_per_split, seed + 1002)
        df_test = _sample_trajectories(df_test, smoke_trajectories_per_split, seed + 1003)

    df_train, df_valid, df_test = _attach_metadata_and_mask_model_id(
        df_train,
        df_valid,
        df_test,
        mask_train_model_id=mask_train_model_id,
    )

    split_summary = []
    for split_name, frame in (("train", df_train), ("valid", df_valid), ("test", df_test)):
        group_col = "orig_model_id" if "orig_model_id" in frame.columns else "model_id"
        for model_id, part in frame.groupby(group_col):
            split_summary.append(
                {
                    "split": split_name,
                    "orig_model_id": model_id,
                    "instances": int(part["instance_id"].nunique()),
                    "trajectories": int(part["traj_id"].nunique()),
                    "prefixes": int(len(part)),
                    "label_rate": float(part["label"].mean()),
                    "is_heldout_test_model": str(model_id) in set(heldout),
                }
            )

    meta = {
        "mode": "model_holdout_shadow_valid_retrain",
        "split_strategy": split_strategy,
        "verified_jsonl": str(verified_jsonl),
        "verified_instances_total": int(len(verified_ids)),
        "selected_instances": int(len(selected_instances)),
        "holdout_models": heldout,
        "train_models": sorted(df_train["orig_model_id"].astype(str).unique().tolist()),
        "valid_models": sorted(df_valid["orig_model_id"].astype(str).unique().tolist()),
        "test_models": sorted(df_test["orig_model_id"].astype(str).unique().tolist()),
        "valid_traj_ratio": float(valid_traj_ratio),
        "valid_per_instance": int(valid_per_instance),
        "valid_models_per_instance": int(valid_models_per_instance),
        "shadow_valid_max_trajectories": int(shadow_valid_max_trajectories),
        "seed": int(seed),
        "train_rows": int(len(df_train)),
        "valid_rows": int(len(df_valid)),
        "test_rows": int(len(df_test)),
        "train_trajectories": int(df_train["traj_id"].nunique()),
        "valid_trajectories": int(df_valid["traj_id"].nunique()),
        "test_trajectories": int(df_test["traj_id"].nunique()),
        "train_instances": int(df_train["instance_id"].nunique()),
        "valid_instances": int(df_valid["instance_id"].nunique()),
        "test_instances": int(df_test["instance_id"].nunique()),
        "train_pos_rate": float(df_train["label"].mean()),
        "valid_pos_rate": float(df_valid["label"].mean()),
        "test_pos_rate": float(df_test["label"].mean()),
        "model_id_feature_mode": "train_seen_valid_missing_test_missing",
        "mask_train_model_id": bool(mask_train_model_id),
        "valid_trajectory_overlap_with_train": int(
            len(set(df_valid["traj_id"].astype(str)) & set(df_train["traj_id"].astype(str)))
        ),
        "note": (
            "Default strategy uses all non-test trajectories for training and a "
            "model_id-masked shadow copy for validation/calibration; test heldout "
            "models are never used for fitting, calibration, or threshold selection."
        ),
    }
    if mask_train_model_id:
        meta["model_id_feature_mode"] = "train_missing_valid_missing_test_missing"
        meta["note"] = (
            "Train/valid/test model_id inputs are all masked to __MISSING__; "
            "orig_model_id is retained only as metadata for grouped evaluation."
        )
    meta["model_stats"] = stats.reset_index().to_dict(orient="records")
    if split_strategy == "per_instance_traj":
        meta["valid_trajectory_overlap_with_train"] = 0
        meta["valid_trajectories_selected"] = int(len(valid_traj_ids))
        meta["train_trajectories_selected"] = int(len(train_traj_ids))
    if split_strategy == "per_instance_model":
        meta["valid_trajectory_overlap_with_train"] = 0
        meta["valid_trajectories_selected"] = int(len(valid_traj_ids))
        meta["train_trajectories_selected"] = int(len(train_traj_ids))
        meta["valid_model_pairs_selected"] = int(len(valid_model_pairs))
        meta["note"] = (
            "Per-instance model strategy randomly holds out non-test model_ids inside "
            "each selected instance for validation; test heldout models are never used "
            "for fitting, calibration, or threshold selection."
        )
        if mask_train_model_id:
            meta["note"] += " Train/valid/test model_id inputs are all masked to __MISSING__."
        meta["valid_model_pair_examples"] = [
            {"instance_id": instance_id, "model_id": model_id}
            for instance_id, model_id in sorted(valid_model_pairs)[:30]
        ]
    if smoke_trajectories_per_split > 0:
        meta["smoke_trajectories_per_split"] = int(smoke_trajectories_per_split)
    return df_train, df_valid, df_test, meta, pd.DataFrame(split_summary)


def _is_task_tfidf_feature(feature_name: str) -> bool:
    return feature_name.startswith("tfidf_task_prompt__")


def _is_task_dense_feature(feature_name: str) -> bool:
    return feature_name == "task_prompt_chars"


def _is_gold_answer_feature(feature_name: str) -> bool:
    return feature_name.startswith("gold_")


def _is_action_tfidf_feature(feature_name: str) -> bool:
    return feature_name.startswith("tfidf_prefix_action__") or feature_name.startswith("tfidf_last_action__")


def _is_feedback_tfidf_feature(feature_name: str) -> bool:
    return feature_name.startswith("tfidf_prefix_feedback__") or feature_name.startswith("tfidf_last_feedback__")


def _is_thought_tfidf_feature(feature_name: str) -> bool:
    return feature_name.startswith("tfidf_prefix_thought__") or feature_name.startswith("tfidf_last_thought__")


_ACTIVITY_COUNT_FEATURES = {
    "prefix_step_idx",
    "steps_observed_so_far",
    "actions_so_far",
    "observations_so_far",
    "tool_messages_so_far",
    "tool_calls_so_far",
    "distinct_tools_so_far",
    "prefix_action_chars",
    "prefix_feedback_chars",
    "task_prompt_chars",
    "read_view_so_far",
    "read_search_so_far",
    "edit_create_so_far",
    "edit_replace_so_far",
    "edit_insert_so_far",
    "edit_undo_so_far",
    "edits_so_far",
    "tests_so_far",
    "run_python_so_far",
    "run_cli_so_far",
    "git_ops_so_far",
    "cleanup_so_far",
    "submit_so_far",
    "bash_calls_so_far",
    "editor_calls_so_far",
    "has_any_action",
}

_LAST_STEP_FEATURES = {
    "last_step_tool_count",
    "last_step_action_chars",
    "last_step_feedback_chars",
    "last_step_has_tool_output",
    "last_step_has_observation",
    "last_step_tool_error_seen",
    "last_step_traceback_seen",
    "last_step_test_fail_seen",
    "last_step_test_pass_seen",
}

_EVENT_TIMING_FEATURES = {
    "first_edit_step",
    "first_test_step",
    "first_run_python_step",
    "first_submit_step",
    "first_error_step",
    "first_traceback_step",
    "first_read_step",
    "first_edit_seen",
    "first_test_seen",
    "first_submit_seen",
    "first_error_seen",
    "first_traceback_seen",
    "steps_since_last_edit",
    "steps_since_last_test",
    "steps_since_last_submit",
    "steps_since_last_error",
    "steps_since_last_traceback",
    "steps_since_last_read",
}

_WORKING_PATTERN_FEATURES = {
    "read_to_edit_ratio",
    "edit_to_test_ratio",
    "bash_to_editor_ratio",
    "error_per_action_ratio",
    "submit_per_action_ratio",
    "feedback_chars_per_action",
    "action_chars_per_step",
    "distinct_tools_per_step",
    "long_no_edit_streak",
    "long_read_streak",
    "thought_steps_so_far",
    "thought_density",
    "prefix_thought_chars",
    "avg_thought_chars_per_step",
    "last_thought_chars",
    "assistant_content_steps_so_far",
    "prefix_assistant_content_chars",
    "avg_assistant_content_chars_per_step",
    "last_assistant_content_chars",
    "thought_equals_content_rate",
    "thought_action_overlap_avg",
    "content_action_overlap_avg",
    "repeated_same_action_consecutive",
    "repeated_same_search_consecutive",
    "repeated_same_view_consecutive",
    "looping_read_seen",
    "edit_failed_seen",
    "submit_without_test_seen",
    "premature_submit_seen",
    "multi_submit_seen",
    "submit_then_edit_again_seen",
    "test_after_submit_seen",
}

_ERROR_TEST_STATUS_FEATURES = {
    "last_fail_count",
    "best_fail_count_so_far",
    "fail_count_delta_from_prev_test",
    "traceback_seen",
    "tool_error_seen",
    "assertion_error_seen",
    "type_error_seen",
    "value_error_seen",
    "syntax_error_seen",
    "import_error_seen",
    "file_not_found_seen",
    "timeout_seen",
    "permission_error_seen",
    "test_fail_seen",
    "test_pass_seen",
    "all_tests_passed_seen",
    "test_improving_seen",
}


def _is_activity_count_feature(feature_name: str) -> bool:
    return feature_name in _ACTIVITY_COUNT_FEATURES


def _is_last_step_feature(feature_name: str) -> bool:
    return feature_name in _LAST_STEP_FEATURES or feature_name.startswith(
        "last_step_action_major_type__"
    ) or feature_name.startswith("last_step_action_primary_subtype__")


def _is_event_timing_feature(feature_name: str) -> bool:
    return feature_name in _EVENT_TIMING_FEATURES


def _is_working_pattern_feature(feature_name: str) -> bool:
    return feature_name in _WORKING_PATTERN_FEATURES


def _is_error_test_status_feature(feature_name: str) -> bool:
    return feature_name in _ERROR_TEST_STATUS_FEATURES


def _is_behavioral_family_feature(feature_name: str) -> bool:
    return (
        _is_activity_count_feature(feature_name)
        or _is_last_step_feature(feature_name)
        or _is_event_timing_feature(feature_name)
        or _is_working_pattern_feature(feature_name)
        or _is_error_test_status_feature(feature_name)
    )


def _is_textual_family_feature(feature_name: str) -> bool:
    return (
        _is_task_tfidf_feature(feature_name)
        or _is_action_tfidf_feature(feature_name)
        or _is_feedback_tfidf_feature(feature_name)
    )


def _is_reference_family_feature(feature_name: str) -> bool:
    return _is_gold_answer_feature(feature_name)


def _is_prefix_gold_overlap_feature(feature_name: str) -> bool:
    if not feature_name.startswith("gold_"):
        return False
    overlap_scopes = (
        "gold_prefix_action_",
        "gold_prefix_feedback_",
        "gold_prefix_thought_",
        "gold_last_action_",
        "gold_last_feedback_",
        "gold_last_thought_",
    )
    return feature_name.startswith(overlap_scopes)


def _is_gold_descriptor_feature(feature_name: str) -> bool:
    return _is_gold_answer_feature(feature_name) and not _is_prefix_gold_overlap_feature(feature_name)


def _make_column_mask(
    feature_names: list[str],
    should_remove: Callable[[str], bool],
) -> tuple[list[int], list[str]]:
    keep_cols = [idx for idx, name in enumerate(feature_names) if not should_remove(name)]
    removed = [name for name in feature_names if should_remove(name)]
    if not removed:
        raise RuntimeError("Ablation removed zero columns; check feature-name rules.")
    return keep_cols, removed


def _selected_specs(requested: list[str]) -> list[dict[str, Any]]:
    aliases = {
        "default": "default",
        "all": "default",
        "i": "I",
        "dense_af": "I",
        "j": "J",
        "dense_af_thought": "J",
        "no_task": "no_task_signal",
        "no_task_signal": "no_task_signal",
        "no_task_tfidf": "no_task_tfidf",
        "no_task_prompt": "no_task_tfidf",
        "no_task_prompt_tfidf": "no_task_tfidf",
        "no_gold": "no_gold_answer",
        "no_gold_answer": "no_gold_answer",
        "no_answer": "no_gold_answer",
        "no_task_no_gold": "no_task_signal_no_gold_answer",
        "no_task_signal_no_gold": "no_task_signal_no_gold_answer",
        "no_task_signal_no_gold_answer": "no_task_signal_no_gold_answer",
        "no_feedback": "no_feedback",
        "no_action": "no_action",
        "no_thought": "no_thought",
        "process_only": "process_only",
        "drop_family_behavioral": "drop_family_behavioral",
        "drop_family_textual": "drop_family_textual",
        "drop_family_reference": "drop_family_reference",
        "drop_group_activity_counts": "drop_group_activity_counts",
        "drop_group_last_step": "drop_group_last_step",
        "drop_group_event_timing": "drop_group_event_timing",
        "drop_group_working_pattern": "drop_group_working_pattern",
        "drop_group_error_test_status": "drop_group_error_test_status",
        "drop_group_task_prompt": "drop_group_task_prompt",
        "drop_group_action_text": "drop_group_action_text",
        "drop_group_feedback_text": "drop_group_feedback_text",
        "drop_group_gold_descriptors": "drop_group_gold_descriptors",
        "drop_group_prefix_gold_overlap": "drop_group_prefix_gold_overlap",
    }
    specs = {
        "I": {
            "predictor": "I_LightGBM_Dense_AF",
            "base": "af",
            "description": "Dense + AF",
        },
        "J": {
            "predictor": "J_LightGBM_Dense_AF_Thought",
            "base": "af_thought",
            "description": "Dense + AF + Thought",
        },
        "no_task_signal": {
            "predictor": "Abl_NoTaskSignal_LightGBM",
            "base": "af_thought",
            "description": "Dense + AF + Thought, remove task_prompt TF-IDF and task_prompt_chars",
            "remove_fn": lambda name: _is_task_tfidf_feature(name) or _is_task_dense_feature(name),
        },
        "no_task_tfidf": {
            "predictor": "Abl_NoTaskPromptTfidf_LightGBM",
            "base": "af_thought",
            "description": "Dense + AF + Thought, remove task_prompt TF-IDF block only",
            "remove_fn": _is_task_tfidf_feature,
        },
        "no_gold_answer": {
            "predictor": "Abl_NoGoldAnswer_LightGBM",
            "base": "af_thought",
            "description": "Dense + AF + Thought, remove structured gold-answer dense features",
            "remove_fn": _is_gold_answer_feature,
        },
        "no_task_signal_no_gold_answer": {
            "predictor": "Abl_NoTaskSignal_NoGoldAnswer_LightGBM",
            "base": "af_thought",
            "description": "Dense + AF + Thought, remove task signal and structured gold-answer features",
            "remove_fn": lambda name: (
                _is_task_tfidf_feature(name)
                or _is_task_dense_feature(name)
                or _is_gold_answer_feature(name)
            ),
        },
        "no_feedback": {
            "predictor": "Abl_NoFeedback_LightGBM",
            "base": "af_thought",
            "description": "Dense + AF + Thought, remove feedback TF-IDF blocks",
            "remove_fn": _is_feedback_tfidf_feature,
        },
        "no_action": {
            "predictor": "Abl_NoAction_LightGBM",
            "base": "af_thought",
            "description": "Dense + AF + Thought, remove action TF-IDF blocks",
            "remove_fn": _is_action_tfidf_feature,
        },
        "no_thought": {
            "predictor": "Abl_NoThought_LightGBM",
            "base": "af_thought",
            "description": "Dense + AF + Thought, remove thought TF-IDF blocks",
            "remove_fn": _is_thought_tfidf_feature,
        },
        "process_only": {
            "predictor": "Abl_ProcessOnly_LightGBM",
            "base": "af_thought",
            "description": "Dense + process text only; remove task prompt TF-IDF, model_id one-hot, and gold-answer features",
            "remove_fn": lambda name: (
                _is_task_tfidf_feature(name)
                or _is_task_dense_feature(name)
                or name.startswith("model_id__")
                or _is_gold_answer_feature(name)
            ),
        },
        "drop_family_behavioral": {
            "predictor": "Tbl_NoBehavioralFamily_LightGBM",
            "base": "af",
            "description": "Main Dense+AF base; remove all behavioral feature-table groups",
            "remove_fn": _is_behavioral_family_feature,
        },
        "drop_family_textual": {
            "predictor": "Tbl_NoTextualFamily_LightGBM",
            "base": "af",
            "description": "Main Dense+AF base; remove task, action, and feedback TF-IDF/SVD blocks",
            "remove_fn": _is_textual_family_feature,
        },
        "drop_family_reference": {
            "predictor": "Tbl_NoReferenceFamily_LightGBM",
            "base": "af",
            "description": "Main Dense+AF base; remove all structured reference-solution features",
            "remove_fn": _is_reference_family_feature,
        },
        "drop_group_activity_counts": {
            "predictor": "Tbl_NoActivityCounts_LightGBM",
            "base": "af",
            "description": "Main Dense+AF base; remove activity-count features",
            "remove_fn": _is_activity_count_feature,
        },
        "drop_group_last_step": {
            "predictor": "Tbl_NoLastStep_LightGBM",
            "base": "af",
            "description": "Main Dense+AF base; remove last-step features",
            "remove_fn": _is_last_step_feature,
        },
        "drop_group_event_timing": {
            "predictor": "Tbl_NoEventTiming_LightGBM",
            "base": "af",
            "description": "Main Dense+AF base; remove event-timing features",
            "remove_fn": _is_event_timing_feature,
        },
        "drop_group_working_pattern": {
            "predictor": "Tbl_NoWorkingPattern_LightGBM",
            "base": "af",
            "description": "Main Dense+AF base; remove working-pattern features",
            "remove_fn": _is_working_pattern_feature,
        },
        "drop_group_error_test_status": {
            "predictor": "Tbl_NoErrorTestStatus_LightGBM",
            "base": "af",
            "description": "Main Dense+AF base; remove error and test-status features",
            "remove_fn": _is_error_test_status_feature,
        },
        "drop_group_task_prompt": {
            "predictor": "Tbl_NoTaskPrompt_LightGBM",
            "base": "af",
            "description": "Main Dense+AF base; remove task-prompt TF-IDF/SVD block",
            "remove_fn": _is_task_tfidf_feature,
        },
        "drop_group_action_text": {
            "predictor": "Tbl_NoActionText_LightGBM",
            "base": "af",
            "description": "Main Dense+AF base; remove prefix-action and last-action TF-IDF/SVD blocks",
            "remove_fn": _is_action_tfidf_feature,
        },
        "drop_group_feedback_text": {
            "predictor": "Tbl_NoFeedbackText_LightGBM",
            "base": "af",
            "description": "Main Dense+AF base; remove prefix-feedback and last-feedback TF-IDF/SVD blocks",
            "remove_fn": _is_feedback_tfidf_feature,
        },
        "drop_group_gold_descriptors": {
            "predictor": "Tbl_NoGoldDescriptors_LightGBM",
            "base": "af",
            "description": "Main Dense+AF base; remove reference-solution descriptor features",
            "remove_fn": _is_gold_descriptor_feature,
        },
        "drop_group_prefix_gold_overlap": {
            "predictor": "Tbl_NoPrefixGoldOverlap_LightGBM",
            "base": "af",
            "description": "Main Dense+AF base; remove prefix-reference overlap and hit features",
            "remove_fn": _is_prefix_gold_overlap_feature,
        },
    }
    raw = [str(item).strip().lower() for item in requested if str(item).strip()]
    if not raw or "default" in raw or "all" in raw:
        keys = ["I", "J", "no_task_signal", "no_task_tfidf", "no_gold_answer", "no_task_signal_no_gold_answer"]
    else:
        keys = []
        for item in raw:
            if item not in aliases:
                raise ValueError(f"Unknown --variants value: {item}")
            key = aliases[item]
            if key == "default":
                for default_key in [
                    "I",
                    "J",
                    "no_task_signal",
                    "no_task_tfidf",
                    "no_gold_answer",
                    "no_task_signal_no_gold_answer",
                ]:
                    if default_key not in keys:
                        keys.append(default_key)
            elif key not in keys:
                keys.append(key)
    return [specs[key] for key in keys]


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
        for score_mode, column_prefix in (("raw", "prob__"), ("calibrated", "prob_cal__")):
            column = f"{column_prefix}{predictor}"
            if column not in pred_df.columns:
                continue
            prefix_rows.append(
                {
                    "scope": "prefix",
                    "score_mode": score_mode,
                    "predictor": predictor,
                    **_binary_metrics(
                        pred_df["label"].to_numpy(dtype=int),
                        pred_df[column].to_numpy(dtype=float),
                    ),
                }
            )
            final_rows.append(
                {
                    "scope": "final_step_trajectory",
                    "score_mode": score_mode,
                    "predictor": predictor,
                    **_binary_metrics(
                        final_df["label"].to_numpy(dtype=int),
                        final_df[column].to_numpy(dtype=float),
                    ),
                }
            )
    return pd.DataFrame(prefix_rows), pd.DataFrame(final_rows)


def _prediction_frame(df_test: pd.DataFrame) -> pd.DataFrame:
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
    return df_test[[column for column in metadata_columns if column in df_test.columns]].copy()


def _write_summary(
    output_path: Path,
    *,
    split_meta: dict[str, Any],
    prefix_metrics: pd.DataFrame,
    final_metrics: pd.DataFrame,
    variant_manifest: pd.DataFrame,
    predictors: list[str],
    refit_feature_engineer: bool,
) -> None:
    def fmt(value: Any, digits: int = 4) -> str:
        try:
            numeric = float(value)
            if math.isnan(numeric):
                return "nan"
            return f"{numeric:.{digits}f}"
        except Exception:
            return "nan"

    lines = [
        "Model-holdout shadow-valid retrain",
        "=" * 38,
        "",
        'Public-release English note.',
        f"  - split_strategy: {split_meta['split_strategy']}",
        "  - test: original three heldout models only; never used for training/calibration",
        f"  - model_id feature mode: {split_meta.get('model_id_feature_mode', 'unknown')}",
        f"  - LightGBM preset: {split_meta.get('lgbm_preset', 'default')}",
        f"  - FeatureEngineer: {'refit on new train' if refit_feature_engineer else 'reused from completed run'}",
        "",
        'Public-release English note.',
        "  - reused prefix_table_filtered.parquet",
        "  - reused gold-answer enriched columns in that table",
        "  - did not rebuild step table / prefix table / gold joins",
        "  - did not refit TF-IDF/SVD unless --refit-feature-engineer is set",
        "",
        "Split:",
        f"  train rows/trajs/instances: {split_meta['train_rows']} / {split_meta['train_trajectories']} / {split_meta['train_instances']}",
        f"  valid rows/trajs/instances: {split_meta['valid_rows']} / {split_meta['valid_trajectories']} / {split_meta['valid_instances']}",
        f"  test  rows/trajs/instances: {split_meta['test_rows']} / {split_meta['test_trajectories']} / {split_meta['test_instances']}",
        f"  train/valid trajectory overlap: {split_meta['valid_trajectory_overlap_with_train']}",
        f"  holdout models: {', '.join(split_meta.get('holdout_models', []))}",
        "",
        "Variants:",
    ]
    for _, row in variant_manifest.iterrows():
        removed = int(row.get("removed_feature_count", 0))
        removed_msg = f"; removed {removed} columns" if removed else ""
        lines.append(f"  - {row['predictor']}: {row['description']}{removed_msg}")

    def append_table(title: str, metrics: pd.DataFrame) -> None:
        lines.extend(["", title + ":"])
        view = metrics[
            (metrics["score_mode"] == "calibrated")
            & (metrics["predictor"].isin(predictors))
        ].copy()
        if view.empty:
            view = metrics[metrics["predictor"].isin(predictors)].copy()
        view = view.sort_values("roc_auc", ascending=False)
        lines.append(
            "  "
            + f"{'predictor':48s} {'mode':10s} {'acc@0.5':>8s} {'auc':>8s} "
            + f"{'pr_auc':>8s} {'brier':>8s} {'meanP':>8s}"
        )
        for _, row in view.iterrows():
            lines.append(
                "  "
                + f"{str(row['predictor'])[:48]:48s} {str(row['score_mode'])[:10]:10s} "
                + f"{fmt(row['accuracy_at_0_5']):>8s} "
                + f"{fmt(row['roc_auc']):>8s} "
                + f"{fmt(row['pr_auc']):>8s} "
                + f"{fmt(row['brier']):>8s} "
                + f"{fmt(row['mean_prob']):>8s}"
            )

    append_table("Final-step trajectory metrics", final_metrics)
    append_table("Prefix-row metrics", prefix_metrics)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    LOGGER.info("Saved summary: %s", output_path)


def _run_ranking_reports(
    *,
    predictions_path: Path,
    output_dir: Path,
    predictors: list[str],
    max_cpu_threads: int,
) -> None:
    report_script = PROJECT_ROOT / "model_ranking_report_posthoc.py"
    subprocess_env = _thread_limited_env(max_cpu_threads)
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
        subprocess.run(command, check=True, env=subprocess_env)


def _bucket_label(step: int) -> str:
    for bucket_name, lo, hi in config.STEP_BUCKETS:
        if lo <= step <= hi:
            return bucket_name
    return "other"


def _step_auc_rows(pred_df: pd.DataFrame, predictors: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    bucket_rows: list[dict[str, Any]] = []
    exact_rows: list[dict[str, Any]] = []
    work = pred_df.copy()
    work["step_bucket"] = work["prefix_step_idx"].astype(int).map(_bucket_label)
    for predictor in predictors:
        for score_mode, column_prefix in (("raw", "prob__"), ("calibrated", "prob_cal__")):
            column = f"{column_prefix}{predictor}"
            if column not in work.columns:
                continue
            for bucket, part in work.groupby("step_bucket", sort=False):
                labels = part["label"].to_numpy(dtype=int)
                probs = part[column].to_numpy(dtype=float)
                row = {
                    "predictor": predictor,
                    "score_mode": score_mode,
                    "bucket": bucket,
                    "n_rows": int(len(part)),
                    "n_pos": int(labels.sum()),
                    "n_neg": int(len(labels) - labels.sum()),
                }
                row.update(_binary_metrics(labels, probs))
                bucket_rows.append(row)
            for step, part in work.groupby("prefix_step_idx", sort=True):
                labels = part["label"].to_numpy(dtype=int)
                probs = part[column].to_numpy(dtype=float)
                if len(part) < 10:
                    continue
                row = {
                    "predictor": predictor,
                    "score_mode": score_mode,
                    "prefix_step_idx": int(step),
                    "n_rows": int(len(part)),
                    "n_pos": int(labels.sum()),
                    "n_neg": int(len(labels) - labels.sum()),
                }
                row.update(_binary_metrics(labels, probs))
                exact_rows.append(row)
    return pd.DataFrame(bucket_rows), pd.DataFrame(exact_rows)


def _write_step_auc_reports(output_dir: Path, pred_df: pd.DataFrame, predictors: list[str]) -> None:
    report_dir = output_dir / "step_auc_reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    bucket_df, exact_df = _step_auc_rows(pred_df, predictors)
    bucket_df.to_csv(report_dir / "step_bucket_auc.csv", index=False)
    exact_df.to_csv(report_dir / "exact_step_auc.csv", index=False)

    lines = ["Step-bucket AUC report", "=" * 22, ""]
    view = bucket_df[bucket_df["score_mode"] == "calibrated"].copy()
    if view.empty:
        view = bucket_df.copy()
    for predictor in predictors:
        part = view[view["predictor"] == predictor]
        if part.empty:
            continue
        lines.extend([predictor, "-" * len(predictor)])
        lines.append(f"{'bucket':14s} {'n':>8s} {'pos':>8s} {'auc':>8s} {'pr_auc':>8s} {'brier':>8s}")
        for _, row in part.iterrows():
            lines.append(
                f"{str(row['bucket'])[:14]:14s} "
                f"{int(row['n_rows']):8d} {int(row['n_pos']):8d} "
                f"{float(row['roc_auc']):8.4f} {float(row['pr_auc']):8.4f} {float(row['brier']):8.4f}"
            )
        lines.append("")
    (report_dir / "step_bucket_auc_report.txt").write_text("\n".join(lines), encoding="utf-8")

    lines = ["Exact-step AUC report", "=" * 21, ""]
    exact_view = exact_df[exact_df["score_mode"] == "calibrated"].copy()
    if exact_view.empty:
        exact_view = exact_df.copy()
    for predictor in predictors:
        part = exact_view[exact_view["predictor"] == predictor]
        if part.empty:
            continue
        lines.extend([predictor, "-" * len(predictor)])
        lines.append(f"{'step':>5s} {'n':>8s} {'pos':>8s} {'auc':>8s} {'pr_auc':>8s} {'brier':>8s}")
        for _, row in part.sort_values("prefix_step_idx").iterrows():
            lines.append(
                f"{int(row['prefix_step_idx']):5d} "
                f"{int(row['n_rows']):8d} {int(row['n_pos']):8d} "
                f"{float(row['roc_auc']):8.4f} {float(row['pr_auc']):8.4f} {float(row['brier']):8.4f}"
            )
        lines.append("")
    (report_dir / "exact_step_auc_report.txt").write_text("\n".join(lines), encoding="utf-8")
    LOGGER.info("Saved step AUC reports: %s", report_dir)


def _write_calibration_plots(output_dir: Path, pred_df: pd.DataFrame, predictors: list[str]) -> None:
    plot_dir = output_dir / "calibration_plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    final_idx = pred_df.groupby("traj_id", sort=False)["prefix_step_idx"].idxmax()
    final_df = pred_df.loc[final_idx].copy()
    y_true = final_df["label"].to_numpy(dtype=int)
    for predictor in predictors:
        for score_mode, column_prefix in (("raw", "prob__"), ("calibrated", "prob_cal__")):
            column = f"{column_prefix}{predictor}"
            if column not in final_df.columns:
                continue
            try:
                plot_calibration(
                    y_true,
                    final_df[column].to_numpy(dtype=float),
                    f"{predictor} ({score_mode})",
                    plot_dir / f"calibration_{score_mode}_{_safe_name(predictor)}.png",
                )
            except Exception as exc:
                LOGGER.warning("Calibration plot failed for %s/%s: %s", predictor, score_mode, exc)
    LOGGER.info("Saved calibration plots: %s", plot_dir)


MATRIX_CACHE_FILENAMES = {
    "X_train_dense": "X_train_dense.npz",
    "X_valid_dense": "X_valid_dense.npz",
    "X_test_dense": "X_test_dense.npz",
    "X_train_af": "X_train_af.npz",
    "X_valid_af": "X_valid_af.npz",
    "X_test_af": "X_test_af.npz",
    "X_train_thought": "X_train_thought.npz",
    "X_valid_thought": "X_valid_thought.npz",
    "X_test_thought": "X_test_thought.npz",
}


def _matrix_cache_complete(cache_dir: Path) -> bool:
    return (cache_dir / "metadata.json").is_file() and all(
        (cache_dir / filename).is_file()
        for filename in MATRIX_CACHE_FILENAMES.values()
    )


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
) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    for key, filename in MATRIX_CACHE_FILENAMES.items():
        LOGGER.info("Saving matrix cache: %s", cache_dir / filename)
        _save_sparse_atomic(matrices[key], cache_dir / filename)
    (cache_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )
    LOGGER.info("Saved matrix cache metadata: %s", cache_dir / "metadata.json")


def _load_matrix_cache(cache_dir: Path) -> dict[str, sparse.csr_matrix]:
    LOGGER.info("Loading matrix cache from %s", cache_dir)
    return {
        key: sparse.load_npz(cache_dir / filename).tocsr()
        for key, filename in MATRIX_CACHE_FILENAMES.items()
    }


GLOBAL_ROW_MATRIX_CACHE_FILENAMES = {
    "prefix_ids": "prefix_ids.npy",
    "X_dense": "X_dense.npz",
    "X_af": "X_af.npz",
    "X_thought": "X_thought.npz",
}


@contextlib.contextmanager
def _file_lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _save_numpy_atomic(array: np.ndarray, path: Path) -> None:
    array = np.asarray(array)
    if array.dtype == object:
        array = array.astype(str)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("wb") as handle:
        np.save(handle, array, allow_pickle=False)
    tmp_path.replace(path)


def _global_row_matrix_cache_complete(cache_dir: Path, *, include_thought: bool) -> bool:
    required = [
        "metadata.json",
        GLOBAL_ROW_MATRIX_CACHE_FILENAMES["prefix_ids"],
        GLOBAL_ROW_MATRIX_CACHE_FILENAMES["X_dense"],
        GLOBAL_ROW_MATRIX_CACHE_FILENAMES["X_af"],
    ]
    if include_thought:
        required.append(GLOBAL_ROW_MATRIX_CACHE_FILENAMES["X_thought"])
    return all((cache_dir / filename).is_file() for filename in required)


def _masked_model_id_frame(frame: pd.DataFrame) -> pd.DataFrame:
    masked = frame.copy()
    if "model_id" in masked.columns:
        masked["model_id"] = "__MISSING__"
    if "model" in masked.columns:
        masked["model"] = "__MISSING__"
    return masked


def _transform_tfidf_rows_streaming(
    *,
    prefix_table_path: Path,
    feature_engineer: FeatureEngineer,
    row_frame: pd.DataFrame,
    column_names: list[str],
    batch_size: int,
) -> sparse.csr_matrix:
    import pyarrow.parquet as parquet

    if row_frame["prefix_id"].astype(str).duplicated().any():
        raise RuntimeError("prefix_id must be unique for global row matrix cache.")
    positions = pd.Series(
        np.arange(len(row_frame), dtype=np.int64),
        index=row_frame["prefix_id"].astype(str),
    )
    parts: list[sparse.spmatrix] = []

    for name in column_names:
        if name not in feature_engineer.tfidf_vectorizers:
            continue
        text_column = feature_engineer.active_text_columns[name]
        LOGGER.info("Streaming global TF-IDF block %s from column %s", name, text_column)
        vectorizer = feature_engineer.tfidf_vectorizers[name]
        reducer = feature_engineer.tfidf_reducers.get(name)
        parquet_file = parquet.ParquetFile(prefix_table_path)
        chunk_rows: list[np.ndarray] = []
        chunk_matrices: list[sparse.spmatrix] = []

        for batch in parquet_file.iter_batches(
            batch_size=batch_size,
            columns=["prefix_id", text_column],
        ):
            batch_df = batch.to_pandas()
            batch_prefix_ids = batch_df["prefix_id"].astype(str)
            output_rows = positions.reindex(batch_prefix_ids).to_numpy()
            keep_mask = ~pd.isna(output_rows)
            if not keep_mask.any():
                del batch_df
                continue
            texts = batch_df.loc[keep_mask, text_column].fillna("")
            X_tfidf = vectorizer.transform(texts)
            del texts
            if reducer is not None:
                X_tfidf = sparse.csr_matrix(reducer.transform(X_tfidf).astype(np.float32))
            else:
                X_tfidf = X_tfidf.tocsr()
            chunk_rows.append(output_rows[keep_mask].astype(np.int64))
            chunk_matrices.append(X_tfidf)
            del batch_df

        if not chunk_matrices:
            raise RuntimeError(f"No rows matched global TF-IDF block {name}.")
        rows = np.concatenate(chunk_rows)
        matrix = sparse.vstack(chunk_matrices, format="csr")
        order = np.argsort(rows)
        rows = rows[order]
        matrix = matrix[order]
        expected = np.arange(len(row_frame), dtype=np.int64)
        if len(rows) != len(row_frame) or not np.array_equal(rows, expected):
            raise RuntimeError(
                f"Global TF-IDF row alignment failed for block={name}: "
                f"matched={len(rows)} expected={len(row_frame)}"
            )
        parts.append(matrix)
        del parquet_file, chunk_rows, chunk_matrices, rows, matrix
        gc.collect()

    if not parts:
        return sparse.csr_matrix((len(row_frame), 0))
    return sparse.hstack(parts, format="csr")


def build_or_load_global_row_matrix_cache(
    *,
    cache_dir: Path,
    prefix_table_path: Path,
    feature_engineer: FeatureEngineer,
    required_columns: list[str],
    include_thought: bool,
    text_batch_size: int,
    mask_model_id_inputs: bool,
) -> dict[str, Any]:
    if not mask_model_id_inputs:
        raise ValueError(
            "Global row matrix cache currently requires masked model_id inputs; "
            "use --mask-train-model-id to match the main setting."
        )
    cache_dir.mkdir(parents=True, exist_ok=True)
    lock_path = cache_dir / "build.lock"
    with _file_lock(lock_path):
        if not _global_row_matrix_cache_complete(cache_dir, include_thought=include_thought):
            LOGGER.info("Building global row matrix cache under %s", cache_dir)
            row_df = _load_prefix_table(prefix_table_path, required_columns)
            row_df = row_df.reset_index(drop=True)
            prefix_ids = row_df["prefix_id"].astype(str).to_numpy()
            if len(prefix_ids) != len(set(prefix_ids.tolist())):
                raise RuntimeError("prefix_id must be unique in the prefix table.")

            dense_df = _masked_model_id_frame(row_df) if mask_model_id_inputs else row_df
            X_dense = sparse.csr_matrix(feature_engineer.transform_dense(dense_df))
            X_af = _transform_tfidf_rows_streaming(
                prefix_table_path=prefix_table_path,
                feature_engineer=feature_engineer,
                row_frame=row_df[["prefix_id"]],
                column_names=list(TFIDF_ACTION_FEEDBACK.keys()),
                batch_size=text_batch_size,
            )
            if include_thought:
                X_thought = _transform_tfidf_rows_streaming(
                    prefix_table_path=prefix_table_path,
                    feature_engineer=feature_engineer,
                    row_frame=row_df[["prefix_id"]],
                    column_names=list(TFIDF_THOUGHT.keys()),
                    batch_size=text_batch_size,
                )
            else:
                X_thought = None

            _save_numpy_atomic(prefix_ids, cache_dir / GLOBAL_ROW_MATRIX_CACHE_FILENAMES["prefix_ids"])
            _save_sparse_atomic(X_dense, cache_dir / GLOBAL_ROW_MATRIX_CACHE_FILENAMES["X_dense"])
            _save_sparse_atomic(X_af, cache_dir / GLOBAL_ROW_MATRIX_CACHE_FILENAMES["X_af"])
            if include_thought and X_thought is not None:
                _save_sparse_atomic(X_thought, cache_dir / GLOBAL_ROW_MATRIX_CACHE_FILENAMES["X_thought"])
            metadata = {
                "prefix_table_path": str(prefix_table_path),
                "rows": int(len(prefix_ids)),
                "dense_shape": list(X_dense.shape),
                "af_shape": list(X_af.shape),
                "thought_shape": list(X_thought.shape) if X_thought is not None else [int(len(prefix_ids)), 0],
                "include_thought": bool(include_thought),
                "model_id_input_mode": "all_missing",
                "text_batch_size": int(text_batch_size),
            }
            (cache_dir / "metadata.json").write_text(
                json.dumps(metadata, ensure_ascii=False, indent=2, default=_json_default),
                encoding="utf-8",
            )
            LOGGER.info("Saved global row matrix cache metadata: %s", cache_dir / "metadata.json")
            del row_df, dense_df, prefix_ids, X_dense, X_af, X_thought
            gc.collect()
        else:
            LOGGER.info("Global row matrix cache is complete: %s", cache_dir)

    prefix_ids = np.load(cache_dir / GLOBAL_ROW_MATRIX_CACHE_FILENAMES["prefix_ids"], allow_pickle=False)
    cache: dict[str, Any] = {
        "prefix_ids": prefix_ids.astype(str),
        "X_dense": sparse.load_npz(cache_dir / GLOBAL_ROW_MATRIX_CACHE_FILENAMES["X_dense"]).tocsr(),
        "X_af": sparse.load_npz(cache_dir / GLOBAL_ROW_MATRIX_CACHE_FILENAMES["X_af"]).tocsr(),
    }
    if include_thought:
        cache["X_thought"] = sparse.load_npz(cache_dir / GLOBAL_ROW_MATRIX_CACHE_FILENAMES["X_thought"]).tocsr()
    else:
        cache["X_thought"] = sparse.csr_matrix((len(prefix_ids), 0))
    return cache


def slice_global_row_matrix_cache(
    *,
    cache: dict[str, Any],
    df_train: pd.DataFrame,
    df_valid: pd.DataFrame,
    df_test: pd.DataFrame,
    include_thought: bool,
) -> dict[str, sparse.csr_matrix]:
    positions = pd.Series(
        np.arange(len(cache["prefix_ids"]), dtype=np.int64),
        index=pd.Index(cache["prefix_ids"].astype(str)),
    )

    def row_positions(frame: pd.DataFrame, split_name: str) -> np.ndarray:
        values = positions.reindex(frame["prefix_id"].astype(str)).to_numpy()
        if pd.isna(values).any():
            missing = int(pd.isna(values).sum())
            raise RuntimeError(f"Global row cache missing {missing} prefix_id value(s) for split={split_name}.")
        return values.astype(np.int64)

    train_pos = row_positions(df_train, "train")
    valid_pos = row_positions(df_valid, "valid")
    test_pos = row_positions(df_test, "test")
    empty_train = sparse.csr_matrix((len(df_train), 0))
    empty_valid = sparse.csr_matrix((len(df_valid), 0))
    empty_test = sparse.csr_matrix((len(df_test), 0))
    X_thought = cache.get("X_thought")

    return {
        "train_dense": cache["X_dense"][train_pos].tocsr(),
        "valid_dense": cache["X_dense"][valid_pos].tocsr(),
        "test_dense": cache["X_dense"][test_pos].tocsr(),
        "train_af": cache["X_af"][train_pos].tocsr(),
        "valid_af": cache["X_af"][valid_pos].tocsr(),
        "test_af": cache["X_af"][test_pos].tocsr(),
        "train_thought": X_thought[train_pos].tocsr() if include_thought and X_thought is not None else empty_train,
        "valid_thought": X_thought[valid_pos].tocsr() if include_thought and X_thought is not None else empty_valid,
        "test_thought": X_thought[test_pos].tocsr() if include_thought and X_thought is not None else empty_test,
    }


def _transform_tfidf_subset_streaming(
    *,
    prefix_table_path: Path,
    feature_engineer: FeatureEngineer,
    df_train: pd.DataFrame,
    df_valid: pd.DataFrame,
    df_test: pd.DataFrame,
    column_names: list[str],
    batch_size: int,
) -> tuple[sparse.csr_matrix, sparse.csr_matrix, sparse.csr_matrix]:
    import pyarrow.parquet as parquet

    train_parts: list[sparse.spmatrix] = []
    valid_parts: list[sparse.spmatrix] = []
    test_parts: list[sparse.spmatrix] = []
    split_frames = {
        "train": df_train,
        "valid": df_valid,
        "test": df_test,
    }
    split_positions = {
        split_name: pd.Series(
            np.arange(len(frame), dtype=np.int64),
            index=frame["prefix_id"].astype(str),
        )
        for split_name, frame in split_frames.items()
    }

    for name in column_names:
        if name not in feature_engineer.tfidf_vectorizers:
            continue
        text_column = feature_engineer.active_text_columns[name]
        LOGGER.info("Streaming TF-IDF block %s from column %s", name, text_column)
        vectorizer = feature_engineer.tfidf_vectorizers[name]
        reducer = feature_engineer.tfidf_reducers.get(name)
        parquet_file = parquet.ParquetFile(prefix_table_path)

        chunk_rows: dict[str, list[np.ndarray]] = {split_name: [] for split_name in split_frames}
        chunk_matrices: dict[str, list[sparse.spmatrix]] = {split_name: [] for split_name in split_frames}

        for batch in parquet_file.iter_batches(
            batch_size=batch_size,
            columns=["prefix_id", text_column],
        ):
            batch_df = batch.to_pandas()
            batch_prefix_ids = batch_df["prefix_id"].astype(str)
            for split_name, positions in split_positions.items():
                output_rows = positions.reindex(batch_prefix_ids).to_numpy()
                keep_mask = ~pd.isna(output_rows)
                if not keep_mask.any():
                    continue
                texts = batch_df.loc[keep_mask, text_column].fillna("")
                X_tfidf = vectorizer.transform(texts)
                del texts
                if reducer is not None:
                    X_tfidf = sparse.csr_matrix(
                        reducer.transform(X_tfidf).astype(np.float32)
                    )
                else:
                    X_tfidf = X_tfidf.tocsr()
                chunk_rows[split_name].append(output_rows[keep_mask].astype(np.int64))
                chunk_matrices[split_name].append(X_tfidf)
            del batch_df

        block_parts: dict[str, sparse.spmatrix] = {}
        for split_name, frame in split_frames.items():
            if not chunk_matrices[split_name]:
                raise RuntimeError(f"No rows matched split={split_name} for TF-IDF block {name}.")
            rows = np.concatenate(chunk_rows[split_name])
            matrix = sparse.vstack(chunk_matrices[split_name], format="csr")
            order = np.argsort(rows)
            rows = rows[order]
            matrix = matrix[order]
            expected = np.arange(len(frame), dtype=np.int64)
            if len(rows) != len(frame) or not np.array_equal(rows, expected):
                raise RuntimeError(
                    f"Streaming TF-IDF row alignment failed for split={split_name}, block={name}: "
                    f"matched={len(rows)} expected={len(frame)}"
                )
            block_parts[split_name] = matrix

        train_parts.append(block_parts["train"])
        valid_parts.append(block_parts["valid"])
        test_parts.append(block_parts["test"])
        del block_parts, chunk_rows, chunk_matrices, parquet_file
        gc.collect()

    def combine(parts: list[sparse.spmatrix], n_rows: int) -> sparse.csr_matrix:
        if not parts:
            return sparse.csr_matrix((n_rows, 0))
        return sparse.hstack(parts, format="csr")

    return (
        combine(train_parts, len(df_train)),
        combine(valid_parts, len(df_valid)),
        combine(test_parts, len(df_test)),
    )


def main() -> int:
    args = parse_args()
    max_cpu_threads = _set_cpu_thread_limits(args.max_cpu_threads)
    _require_lightgbm_available()
    specs = _selected_specs(args.variants)
    run_root = config.RUNTIME_ROOT / "runs" / args.run_name
    _set_run_dirs(run_root)
    rebind_all_file_loggers()
    config.LGBM_PARAMS["num_threads"] = max_cpu_threads
    LOGGER.info("CPU thread cap: %s", max_cpu_threads)
    lgbm_preset_updates = _apply_lgbm_preset(args.lgbm_preset)
    if lgbm_preset_updates:
        LOGGER.info("LightGBM preset %s updates: %s", args.lgbm_preset, lgbm_preset_updates)

    if args.no_gpu_lgbm:
        config.LGBM_PARAMS["device"] = "cpu"
        config.LGBM_PARAMS.pop("gpu_device_id", None)

    prefix_path = args.prefix_table or config.PREFIX_TABLE_FILTERED_PATH
    output_dir = config.REPORT_DIR / args.output_subdir
    output_model_dir = output_dir / "models"
    output_model_dir.mkdir(parents=True, exist_ok=True)
    matrix_cache_dir = args.matrix_cache_dir or (output_dir / "matrix_cache")
    if not prefix_path.is_file():
        raise FileNotFoundError(prefix_path)

    source_feature_engineer_path = config.MODEL_DIR / "feature_engineer_with_model.pkl"
    if not source_feature_engineer_path.is_file():
        raise FileNotFoundError(source_feature_engineer_path)

    source_feature_engineer = FeatureEngineer.load(source_feature_engineer_path)
    _repair_unpickled_tfidf_for_local_sklearn(source_feature_engineer)
    required_columns = _required_columns(
        source_feature_engineer,
        include_text=args.eager_load_text_columns,
    )

    with timer(LOGGER, "Load cached prefix table and build new split"):
        prefix_df = _load_prefix_table(prefix_path, required_columns)
        df_train, df_valid, df_test, split_meta, split_summary = _build_split(
            prefix_df,
            verified_jsonl=args.verified_jsonl,
            holdout_models=args.holdout_models,
            max_instances=args.max_instances,
            split_strategy=args.split_strategy,
            valid_traj_ratio=args.valid_traj_ratio,
            valid_per_instance=args.valid_per_instance,
            valid_models_per_instance=args.valid_models_per_instance,
            shadow_valid_max_trajectories=args.shadow_valid_max_trajectories,
            seed=args.seed,
            smoke_trajectories_per_split=args.smoke_trajectories_per_split,
            mask_train_model_id=args.mask_train_model_id,
        )
        split_meta["lgbm_preset"] = args.lgbm_preset
        split_meta["lgbm_preset_updates"] = lgbm_preset_updates
        split_meta["lgbm_params_used"] = dict(config.LGBM_PARAMS)
        del prefix_df
        gc.collect()

    if args.refit_feature_engineer:
        with timer(LOGGER, "Refit FeatureEngineer on new train split"):
            feature_engineer = FeatureEngineer(include_model_id=True, tfidf_level="with_gold_answer")
            feature_engineer.fit(df_train)
            feature_engineer.save(output_model_dir / "feature_engineer_with_model_refit.pkl")
    else:
        feature_engineer = source_feature_engineer

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
    tfidf_thought_cols = list(TFIDF_THOUGHT.keys())
    tfidf_af_thought_cols = tfidf_af_cols + tfidf_thought_cols
    needs_thought = any(spec["base"] == "af_thought" for spec in specs)

    with timer(LOGGER, "Build/load reusable Dense / AF / Thought matrices"):
        if args.cache_matrices and _matrix_cache_complete(matrix_cache_dir):
            cached = _load_matrix_cache(matrix_cache_dir)
            X_train_dense = cached["X_train_dense"]
            X_valid_dense = cached["X_valid_dense"]
            X_test_dense = cached["X_test_dense"]
            X_train_af = cached["X_train_af"]
            X_valid_af = cached["X_valid_af"]
            X_test_af = cached["X_test_af"]
            X_train_thought = cached["X_train_thought"]
            X_valid_thought = cached["X_valid_thought"]
            X_test_thought = cached["X_test_thought"]
            del cached
        else:
            X_train_dense = sparse.csr_matrix(feature_engineer.transform_dense(df_train))
            X_valid_dense = sparse.csr_matrix(feature_engineer.transform_dense(df_valid))
            X_test_dense = sparse.csr_matrix(feature_engineer.transform_dense(df_test))
            if args.eager_load_text_columns:
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
                X_train_af, X_valid_af, X_test_af = _transform_tfidf_subset_streaming(
                    prefix_table_path=prefix_path,
                    feature_engineer=feature_engineer,
                    df_train=df_train,
                    df_valid=df_valid,
                    df_test=df_test,
                    column_names=tfidf_af_cols,
                    batch_size=args.text_batch_size,
                )
                if needs_thought:
                    X_train_thought, X_valid_thought, X_test_thought = _transform_tfidf_subset_streaming(
                        prefix_table_path=prefix_path,
                        feature_engineer=feature_engineer,
                        df_train=df_train,
                        df_valid=df_valid,
                        df_test=df_test,
                        column_names=tfidf_thought_cols,
                        batch_size=args.text_batch_size,
                    )
                else:
                    X_train_thought = sparse.csr_matrix((len(df_train), 0))
                    X_valid_thought = sparse.csr_matrix((len(df_valid), 0))
                    X_test_thought = sparse.csr_matrix((len(df_test), 0))

            if args.cache_matrices:
                _save_matrix_cache(
                    cache_dir=matrix_cache_dir,
                    matrices={
                        "X_train_dense": X_train_dense,
                        "X_valid_dense": X_valid_dense,
                        "X_test_dense": X_test_dense,
                        "X_train_af": X_train_af,
                        "X_valid_af": X_valid_af,
                        "X_test_af": X_test_af,
                        "X_train_thought": X_train_thought,
                        "X_valid_thought": X_valid_thought,
                        "X_test_thought": X_test_thought,
                    },
                    metadata={
                        "run_name": args.run_name,
                        "output_subdir": args.output_subdir,
                        "split_strategy": args.split_strategy,
                        "max_instances": args.max_instances,
                        "holdout_models": split_meta.get("holdout_models", []),
                        "train_rows": len(df_train),
                        "valid_rows": len(df_valid),
                        "test_rows": len(df_test),
                        "tfidf_af_cols": tfidf_af_cols,
                        "tfidf_thought_cols": tfidf_thought_cols,
                        "refit_feature_engineer": bool(args.refit_feature_engineer),
                    },
                )

        names_af_base = (
            list(feature_engineer.dense_feature_names)
            + feature_engineer.get_tfidf_feature_names_for_columns(tfidf_af_cols)
        )
        names_j_base = (
            list(feature_engineer.dense_feature_names)
            + feature_engineer.get_tfidf_feature_names_for_columns(tfidf_af_thought_cols)
        )

        if args.low_memory:
            X_train_af_base = X_valid_af_base = X_test_af_base = None
            X_train_j_base = X_valid_j_base = X_test_j_base = None
            LOGGER.info(
                "Low-memory block shapes: dense train/valid/test=%s/%s/%s; "
                "AF=%s/%s/%s; Thought=%s/%s/%s",
                X_train_dense.shape,
                X_valid_dense.shape,
                X_test_dense.shape,
                X_train_af.shape,
                X_valid_af.shape,
                X_test_af.shape,
                X_train_thought.shape,
                X_valid_thought.shape,
                X_test_thought.shape,
            )
        else:
            X_train_af_base = sparse.hstack([X_train_dense, X_train_af], format="csr")
            X_valid_af_base = sparse.hstack([X_valid_dense, X_valid_af], format="csr")
            X_test_af_base = sparse.hstack([X_test_dense, X_test_af], format="csr")
            X_train_j_base = sparse.hstack([X_train_dense, X_train_af, X_train_thought], format="csr")
            X_valid_j_base = sparse.hstack([X_valid_dense, X_valid_af, X_valid_thought], format="csr")
            X_test_j_base = sparse.hstack([X_test_dense, X_test_af, X_test_thought], format="csr")
            LOGGER.info(
                "Matrix shapes: AF train/valid/test=%s/%s/%s; AF+Thought=%s/%s/%s",
                X_train_af_base.shape,
                X_valid_af_base.shape,
                X_test_af_base.shape,
                X_train_j_base.shape,
                X_valid_j_base.shape,
                X_test_j_base.shape,
            )

    pred_df = _prediction_frame(df_test)
    valid_pred_df = _prediction_frame(df_valid) if args.save_valid_predictions else None
    del df_train, df_valid, df_test
    gc.collect()
    calibration_rows: list[dict[str, Any]] = []
    variant_rows: list[dict[str, Any]] = []
    predictors: list[str] = []

    for spec in specs:
        public_name = spec["predictor"]
        if spec["base"] == "af":
            if args.low_memory:
                X_train_model = sparse.hstack([X_train_dense, X_train_af], format="csr")
                X_valid_model = sparse.hstack([X_valid_dense, X_valid_af], format="csr")
                X_test_model = sparse.hstack([X_test_dense, X_test_af], format="csr")
            else:
                X_train_model, X_valid_model, X_test_model = X_train_af_base, X_valid_af_base, X_test_af_base
            feature_names = list(names_af_base)
            removed_names: list[str] = []
            remove_fn = spec.get("remove_fn")
            if remove_fn is not None:
                keep_cols, removed_names = _make_column_mask(feature_names, remove_fn)
                X_train_model = X_train_model[:, keep_cols].tocsr()
                X_valid_model = X_valid_model[:, keep_cols].tocsr()
                X_test_model = X_test_model[:, keep_cols].tocsr()
                feature_names = [feature_names[idx] for idx in keep_cols]
        elif spec["base"] == "af_thought":
            if args.low_memory:
                X_train_model = sparse.hstack([X_train_dense, X_train_af, X_train_thought], format="csr")
                X_valid_model = sparse.hstack([X_valid_dense, X_valid_af, X_valid_thought], format="csr")
                X_test_model = sparse.hstack([X_test_dense, X_test_af, X_test_thought], format="csr")
            else:
                X_train_model, X_valid_model, X_test_model = X_train_j_base, X_valid_j_base, X_test_j_base
            feature_names = list(names_j_base)
            removed_names = []
            remove_fn = spec.get("remove_fn")
            if remove_fn is not None:
                keep_cols, removed_names = _make_column_mask(feature_names, remove_fn)
                X_train_model = X_train_model[:, keep_cols].tocsr()
                X_valid_model = X_valid_model[:, keep_cols].tocsr()
                X_test_model = X_test_model[:, keep_cols].tocsr()
                feature_names = [feature_names[idx] for idx in keep_cols]
        else:
            raise ValueError(f"Unknown base matrix: {spec['base']}")

        if len(feature_names) != X_train_model.shape[1]:
            raise RuntimeError(f"{public_name}: feature names != matrix columns")

        with timer(LOGGER, f"Train {public_name}"):
            model = _fit_lgbm_with_cpu_fallback(
                X_train=X_train_model,
                y_train=y_train,
                w_train=w_train,
                X_valid=X_valid_model,
                y_valid=y_valid,
                w_valid=w_valid,
                feature_names=feature_names,
                model_name=public_name,
            )
            save_model(model, output_model_dir / f"{_safe_name(public_name)}.lgb")
            valid_raw = np.asarray(model.predict(X_valid_model), dtype=np.float64)
            test_raw = np.asarray(model.predict(X_test_model), dtype=np.float64)
            calibrator = fit_sigmoid_calibrator(valid_raw, y_valid, sample_weight=w_valid)
            test_cal = calibrator.predict(test_raw)
            save_model(calibrator, output_model_dir / f"calibrator_{_safe_name(public_name)}.pkl")
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
            pred_df[f"prob__{public_name}"] = test_raw.astype(np.float32)
            pred_df[f"prob_cal__{public_name}"] = test_cal.astype(np.float32)
            if valid_pred_df is not None:
                valid_cal = calibrator.predict(valid_raw)
                valid_pred_df[f"prob__{public_name}"] = valid_raw.astype(np.float32)
                valid_pred_df[f"prob_cal__{public_name}"] = valid_cal.astype(np.float32)
            _write_feature_importance(
                model,
                feature_names,
                output_dir / f"feature_importance_{_safe_name(public_name)}.csv",
            )
            predictors.append(public_name)
            variant_rows.append(
                {
                    "predictor": public_name,
                    "base_matrix": spec["base"],
                    "description": spec["description"],
                    "removed_feature_count": int(len(removed_names)),
                    "kept_feature_count": int(len(feature_names)),
                    "removed_feature_examples": "; ".join(removed_names[:30]),
                }
            )

        if args.low_memory or spec.get("remove_fn") is not None:
            del X_train_model, X_valid_model, X_test_model
            gc.collect()

    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = output_dir / "test_predictions_shadow_valid_retrain.parquet"
    pred_df.to_parquet(prediction_path, index=False)
    if valid_pred_df is not None:
        valid_prediction_path = output_dir / "valid_predictions_shadow_valid_retrain.parquet"
        valid_pred_df.to_parquet(valid_prediction_path, index=False)
        LOGGER.info("Saved valid predictions: %s", valid_prediction_path)
    split_summary.to_csv(output_dir / "split_summary.csv", index=False)
    (output_dir / "split_metadata.json").write_text(
        json.dumps(split_meta, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )

    variant_manifest = pd.DataFrame(variant_rows)
    variant_manifest.to_csv(output_dir / "variant_manifest.csv", index=False)
    pd.DataFrame(calibration_rows).to_csv(output_dir / "probability_calibration_summary.csv", index=False)
    prefix_metrics, final_metrics = _collect_metrics(pred_df, predictors)
    prefix_metrics.to_csv(output_dir / "prefix_metrics.csv", index=False)
    final_metrics.to_csv(output_dir / "final_step_metrics.csv", index=False)
    _write_summary(
        output_dir / "summary.txt",
        split_meta=split_meta,
        prefix_metrics=prefix_metrics,
        final_metrics=final_metrics,
        variant_manifest=variant_manifest,
        predictors=predictors,
        refit_feature_engineer=args.refit_feature_engineer,
    )

    if not args.skip_diagnostics:
        _write_step_auc_reports(output_dir, pred_df, predictors)
        _write_calibration_plots(output_dir, pred_df, predictors)

    if not args.skip_ranking_reports:
        _run_ranking_reports(
            predictions_path=prediction_path,
            output_dir=output_dir,
            predictors=predictors,
            max_cpu_threads=max_cpu_threads,
        )

    LOGGER.info("Shadow-valid retrain complete: %s", output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
