from __future__ import annotations

import argparse
import gc
import json
import math
import pickle
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

from earlyeval.core.io import ensure_dir, write_json, write_table
from earlyeval.experiments.paper_pipeline import (
    _default_output_dir,
    _markdown_table,
    _answer_module_root,
    _resolve_project_path,
    _stable_seed,
    load_earlyeval_config,
)


PREDICTORS = {
    "process": "Robust_LightGBM_Process",
    "rich_af_gold": "Robust_LightGBM_Dense_AF_Gold",
}
COMMON_NUMERIC_FEATURES = [
    "prefix_step_idx",
    "task_prompt_chars",
    "prefix_action_chars",
    "prefix_feedback_chars",
    "prefix_thought_chars",
    "prefix_assistant_content_chars",
    "last_step_action_chars",
    "last_step_feedback_chars",
    "tool_calls_so_far",
    "distinct_tools_so_far",
    "actions_so_far",
    "observations_so_far",
]
METADATA_NUMERIC_COLUMNS = [
    "n_steps_total_for_weighting",
]
COMMON_CATEGORICAL_FEATURES = [
    "last_step_action_major_type",
    "last_step_action_primary_subtype",
]
ACTION_FEEDBACK_TEXT_COLUMNS = [
    "task_prompt_text",
    "prefix_action_text",
    "prefix_feedback_text",
    "last_action_text",
    "last_feedback_text",
]


def _legacy_module_root() -> Path:
    return _answer_module_root()


def _import_legacy() -> dict[str, Any]:
    root = _legacy_module_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from answer_features import (  # type: ignore
        ANSWER_BOOL_FEATURES,
        ANSWER_CATEGORICAL_FEATURES,
        ANSWER_NUMERIC_FEATURES,
    )
    from probability_calibration import calibration_summary_row, fit_sigmoid_calibrator  # type: ignore
    from safe_stop_dual_head_retrain import (  # type: ignore
        _evaluate_policies,
        _evaluate_selected,
        _head_column,
        _policy_grid,
        _safe_targets,
        _select_policies,
        _write_report,
    )

    return {
        "calibration_summary_row": calibration_summary_row,
        "fit_sigmoid_calibrator": fit_sigmoid_calibrator,
        "_evaluate_policies": _evaluate_policies,
        "_evaluate_selected": _evaluate_selected,
        "_head_column": _head_column,
        "_policy_grid": _policy_grid,
        "_safe_targets": _safe_targets,
        "_select_policies": _select_policies,
        "_write_report": _write_report,
        "ANSWER_BOOL_FEATURES": ANSWER_BOOL_FEATURES,
        "ANSWER_CATEGORICAL_FEATURES": ANSWER_CATEGORICAL_FEATURES,
        "ANSWER_NUMERIC_FEATURES": ANSWER_NUMERIC_FEATURES,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run simple 15%-model-holdout robustness baselines on Toolathlon/TerminalBench."
    )
    parser.add_argument("--config", type=Path, default=Path("configs/earlyeval.yaml"))
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--run-subdir", default="robustness_15pct_model_holdout")
    parser.add_argument("--datasets", nargs="+", default=["toolathlon", "terminalbench"])
    parser.add_argument("--feature-preset", choices=("process", "rich_af_gold"), default="process")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--split-mode", choices=("random_ratio", "leave_one_model"), default="random_ratio")
    parser.add_argument(
        "--only-test-models",
        nargs="+",
        default=None,
        help="Optional explicit test model ids for leave_one_model mode.",
    )
    parser.add_argument("--test-model-ratio", type=float, default=0.15)
    parser.add_argument("--valid-ratio", type=float, default=None)
    parser.add_argument("--max-cpu-threads", type=int, default=2)
    parser.add_argument("--safe-label-min-step", type=int, default=10)
    parser.add_argument("--policy-min-steps", nargs="+", type=int, default=[0])
    parser.add_argument("--consecutive", nargs="+", type=int, default=[1])
    parser.add_argument("--success-thresholds", nargs="+", type=float, default=[0.75, 0.80, 0.85, 0.90, 0.95])
    parser.add_argument("--failure-thresholds", nargs="+", type=float, default=[0.75, 0.80, 0.85, 0.90, 0.95])
    parser.add_argument("--score-modes", nargs="+", choices=("raw", "calibrated"), default=["calibrated"])
    parser.add_argument("--max-valid-abs-drop-pp", type=float, default=5.0)
    parser.add_argument("--min-valid-decision-acc", type=float, default=0.85)
    parser.add_argument("--fallback-min-save-pct", type=float, default=0.0)
    parser.add_argument("--num-boost-round", type=int, default=250)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--num-leaves", type=int, default=31)
    parser.add_argument("--tfidf-max-features", type=int, default=30000)
    parser.add_argument("--tfidf-min-df", type=int, default=5)
    parser.add_argument("--tfidf-svd-dim", type=int, default=64)
    parser.add_argument("--tfidf-ngram-max", type=int, default=2)
    parser.add_argument("--parquet-batch-size", type=int, default=16384)
    parser.add_argument("--smoke-trajectories-per-split", type=int, default=0)
    parser.add_argument("--max-train-rows", type=int, default=0)
    parser.add_argument("--max-valid-rows", type=int, default=0)
    parser.add_argument("--max-test-rows", type=int, default=0)
    parser.add_argument("--no-save-models", action="store_true")
    return parser.parse_args()


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_") or "model"


def _read_dataset_frame(
    path: Path,
    spec: dict[str, Any],
    *,
    feature_preset: str,
    parquet_batch_size: int,
) -> pd.DataFrame:
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(path)
    present = set(pf.schema_arrow.names)
    base_columns = [
        spec.get("traj_col", "traj_id"),
        spec.get("instance_col", "instance_id"),
        spec.get("model_col", "model_id"),
        spec.get("step_col", "prefix_step_idx"),
        spec.get("label_col", "label"),
        "prefix_id",
        "group_id",
        "sample_weight",
        *METADATA_NUMERIC_COLUMNS,
        *COMMON_NUMERIC_FEATURES,
        *COMMON_CATEGORICAL_FEATURES,
    ]
    if feature_preset == "rich_af_gold":
        base_columns.extend([col for col in ACTION_FEEDBACK_TEXT_COLUMNS if col in present])
        base_columns.extend([col for col in present if col.startswith("gold_") and not col.endswith("_text")])
    columns = [str(col) for col in dict.fromkeys(base_columns) if str(col) in present]
    if feature_preset == "rich_af_gold" and parquet_batch_size > 0:
        chunks = [
            batch.to_pandas()
            for batch in pf.iter_batches(batch_size=int(parquet_batch_size), columns=columns)
        ]
        frame = pd.concat(chunks, ignore_index=True, copy=False) if chunks else pd.DataFrame(columns=columns)
        del chunks
        gc.collect()
    else:
        frame = pq.read_table(path, columns=columns).to_pandas()
    rename = {
        str(spec.get("traj_col", "traj_id")): "traj_id",
        str(spec.get("instance_col", "instance_id")): "instance_id",
        str(spec.get("model_col", "model_id")): "model_id",
        str(spec.get("step_col", "prefix_step_idx")): "prefix_step_idx",
        str(spec.get("label_col", "label")): "label",
    }
    frame = frame.rename(columns=rename)
    if "prefix_id" not in frame:
        frame["prefix_id"] = frame["traj_id"].astype(str) + "::p" + frame["prefix_step_idx"].astype(str)
    if "group_id" not in frame:
        frame["group_id"] = frame["traj_id"].astype(str)
    if "sample_weight" not in frame:
        denom = pd.to_numeric(frame["n_steps_total_for_weighting"], errors="coerce").fillna(1.0).clip(lower=1.0)
        frame["sample_weight"] = (1.0 / denom).astype(np.float32)
    return frame


def _read_plan_metadata_frame(path: Path, spec: dict[str, Any]) -> pd.DataFrame:
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(path)
    present = set(pf.schema_arrow.names)
    raw_columns = [
        str(spec.get("traj_col", "traj_id")),
        str(spec.get("instance_col", "instance_id")),
        str(spec.get("model_col", "model_id")),
        str(spec.get("label_col", "label")),
        "n_steps_total_for_weighting",
    ]
    columns = [col for col in dict.fromkeys(raw_columns) if col in present]
    frame = pq.read_table(path, columns=columns).to_pandas()
    frame = frame.rename(
        columns={
            str(spec.get("traj_col", "traj_id")): "traj_id",
            str(spec.get("instance_col", "instance_id")): "instance_id",
            str(spec.get("model_col", "model_id")): "model_id",
            str(spec.get("label_col", "label")): "label",
        }
    )
    if "n_steps_total_for_weighting" not in frame:
        frame["n_steps_total_for_weighting"] = 1
    return frame


def _list_dataset_models(path: Path, spec: dict[str, Any]) -> list[str]:
    import pyarrow.parquet as pq

    model_col = str(spec.get("model_col", "model_id"))
    traj_col = str(spec.get("traj_col", "traj_id"))
    table = pq.read_table(path, columns=[model_col, traj_col])
    frame = table.to_pandas()
    frame = frame.rename(columns={model_col: "model_id", traj_col: "traj_id"})
    counts = frame[["model_id", "traj_id"]].drop_duplicates("traj_id").groupby("model_id")["traj_id"].count()
    return counts.sort_values(ascending=False).index.astype(str).tolist()


def _drop_short_train_valid(frame: pd.DataFrame, *, min_steps: int) -> pd.DataFrame:
    if min_steps <= 0:
        return frame.copy()
    traj_steps = frame.groupby("traj_id", sort=False)["n_steps_total_for_weighting"].first()
    short = set(traj_steps.loc[lambda values: values < min_steps].index.astype(str))
    if not short:
        return frame.copy()
    return frame[~frame["traj_id"].astype(str).isin(short)].copy()


def _sample_trajectories(frame: pd.DataFrame, *, max_trajectories: int, seed: int) -> pd.DataFrame:
    if max_trajectories <= 0:
        return frame
    traj_ids = frame["traj_id"].drop_duplicates().astype(str).to_numpy()
    if len(traj_ids) <= max_trajectories:
        return frame
    rng = np.random.default_rng(seed)
    selected = set(rng.choice(traj_ids, size=max_trajectories, replace=False).tolist())
    return frame[frame["traj_id"].astype(str).isin(selected)].copy()


def _sample_rows(frame: pd.DataFrame, *, max_rows: int, seed: int) -> pd.DataFrame:
    if max_rows <= 0 or len(frame) <= max_rows:
        return frame
    return frame.sample(n=max_rows, random_state=seed).sort_values(["traj_id", "prefix_step_idx"]).copy()


def _model_holdout_plan(
    frame: pd.DataFrame,
    *,
    dataset: str,
    cfg: Any,
    test_model_ratio: float,
    test_models_override: list[str] | None,
    valid_ratio: float,
    min_steps: int,
    smoke_trajectories_per_split: int,
) -> dict[str, Any]:
    traj = frame[["traj_id", "instance_id", "model_id", "label"]].drop_duplicates("traj_id").copy()
    model_counts = traj.groupby("model_id")["traj_id"].count().sort_values(ascending=False)
    models = model_counts.index.astype(str).tolist()
    if len(models) < 2:
        raise ValueError(f"{dataset}: need at least two models for model-holdout split.")
    if test_models_override:
        requested = [str(item) for item in test_models_override]
        missing = sorted(set(requested) - set(models))
        if missing:
            raise ValueError(f"{dataset}: requested test model(s) not present: {missing}")
        if len(set(requested)) >= len(models):
            raise ValueError(f"{dataset}: leave at least one non-test model for train/valid.")
        test_models = sorted(set(requested))
        strategy = "leave_one_model_known_task" if len(test_models) == 1 else "explicit_model_holdout_known_task"
    else:
        n_test = max(1, int(round(len(models) * float(test_model_ratio))))
        n_test = min(n_test, len(models) - 1)
        rng = np.random.default_rng(_stable_seed(cfg.seed, dataset, "robustness_15pct"))
        test_models = sorted(rng.choice(models, size=n_test, replace=False).tolist())
        strategy = "random_15pct_model_holdout_known_task"

    test_mask = frame["model_id"].astype(str).isin(test_models)
    trainval = frame[~test_mask].copy()
    trainval = _drop_short_train_valid(trainval, min_steps=min_steps)
    short_ids = set(frame.loc[~test_mask, "traj_id"].astype(str)) - set(trainval["traj_id"].astype(str))
    instances = sorted(trainval["instance_id"].astype(str).dropna().unique().tolist())
    rng_valid = np.random.default_rng(_stable_seed(cfg.seed, dataset, "valid_instances", ",".join(test_models)))
    rng_valid.shuffle(instances)
    valid_n = max(1, int(round(len(instances) * float(valid_ratio)))) if instances else 0
    valid_instances = set(instances[:valid_n])

    train_smoke: set[str] | None = None
    valid_smoke: set[str] | None = None
    test_smoke: set[str] | None = None
    if smoke_trajectories_per_split > 0:
        base_seed = _stable_seed(cfg.seed, dataset, "smoke")
        train_meta = trainval[~trainval["instance_id"].astype(str).isin(valid_instances)]
        valid_meta = trainval[trainval["instance_id"].astype(str).isin(valid_instances)]
        test_meta = frame[test_mask]
        train_smoke = set(_sample_trajectories(train_meta, max_trajectories=smoke_trajectories_per_split, seed=base_seed + 1)["traj_id"].astype(str))
        valid_smoke = set(_sample_trajectories(valid_meta, max_trajectories=smoke_trajectories_per_split, seed=base_seed + 2)["traj_id"].astype(str))
        test_smoke = set(_sample_trajectories(test_meta, max_trajectories=smoke_trajectories_per_split, seed=base_seed + 3)["traj_id"].astype(str))

    return {
        "strategy": strategy,
        "test_models": test_models,
        "valid_instances": valid_instances,
        "short_trainvalid_traj_ids": short_ids,
        "train_smoke_traj_ids": train_smoke,
        "valid_smoke_traj_ids": valid_smoke,
        "test_smoke_traj_ids": test_smoke,
    }


def _split_mask_by_plan(frame: pd.DataFrame, plan: dict[str, Any], split_name: str) -> pd.Series:
    model_values = frame["model_id"].astype(str)
    traj_values = frame["traj_id"].astype(str)
    instance_values = frame["instance_id"].astype(str)
    test_mask = model_values.isin(set(plan["test_models"]))
    short_mask = traj_values.isin(set(plan["short_trainvalid_traj_ids"]))
    trainvalid_mask = ~test_mask & ~short_mask
    valid_mask = trainvalid_mask & instance_values.isin(set(plan["valid_instances"]))
    train_mask = trainvalid_mask & ~valid_mask
    if plan.get("train_smoke_traj_ids") is not None:
        train_mask &= traj_values.isin(set(plan["train_smoke_traj_ids"]))
        valid_mask &= traj_values.isin(set(plan["valid_smoke_traj_ids"]))
        test_mask &= traj_values.isin(set(plan["test_smoke_traj_ids"]))
    if split_name == "train":
        return train_mask
    if split_name == "valid":
        return valid_mask
    if split_name == "test":
        return test_mask
    raise ValueError(f"Unknown split_name: {split_name}")


def _split_frame_by_plan(frame: pd.DataFrame, plan: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_mask = _split_mask_by_plan(frame, plan, "train")
    valid_mask = _split_mask_by_plan(frame, plan, "valid")
    test_mask = _split_mask_by_plan(frame, plan, "test")
    return frame[train_mask].copy(), frame[valid_mask].copy(), frame[test_mask].copy()


def _annotate_split_frame(split_frame: pd.DataFrame, split_name: str) -> pd.DataFrame:
    split_frame["split"] = split_name
    split_frame["orig_model_id"] = split_frame["model_id"].astype(str)
    split_frame["orig_model"] = split_frame["model_id"].astype(str)
    split_frame["model_id"] = "__MISSING__"
    split_frame["model"] = "__MISSING__"
    split_frame["model_id_input_mode"] = f"{split_name}_missing"
    return split_frame


def _split_info(split_frame: pd.DataFrame) -> dict[str, Any]:
    return {
        "rows": int(len(split_frame)),
        "trajectories": int(split_frame["traj_id"].nunique()),
        "instances": int(split_frame["instance_id"].nunique()),
        "models": sorted(split_frame["orig_model_id"].astype(str).unique().tolist()),
        "instance_ids": set(split_frame["instance_id"].astype(str).dropna().tolist()),
    }


def _split_summary_rows(
    split_frame: pd.DataFrame,
    *,
    dataset: str,
    split_name: str,
    plan: dict[str, Any],
) -> list[dict[str, Any]]:
    rows = []
    for model_id, part in split_frame.groupby("orig_model_id", sort=True):
        rows.append(
            {
                "dataset": dataset,
                "split": split_name,
                "orig_model_id": model_id,
                "instances": int(part["instance_id"].nunique()),
                "trajectories": int(part["traj_id"].nunique()),
                "prefixes": int(len(part)),
                "label_rate": float(part["label"].mean()),
                "is_heldout_test_model": str(model_id) in set(plan["test_models"]),
            }
        )
    return rows


def _split_metadata_from_infos(
    *,
    dataset: str,
    cfg: Any,
    plan: dict[str, Any],
    infos: dict[str, dict[str, Any]],
    test_model_ratio: float,
    valid_ratio: float,
    min_steps: int,
    smoke_trajectories_per_split: int,
) -> dict[str, Any]:
    train_info = infos["train"]
    valid_info = infos["valid"]
    test_info = infos["test"]
    train_models = set(train_info["models"])
    valid_models = set(valid_info["models"])
    test_models = set(plan["test_models"])
    return {
        "dataset": dataset,
        "strategy": str(plan["strategy"]),
        "seed": int(cfg.seed),
        "test_model_ratio": float(test_model_ratio),
        "valid_ratio": float(valid_ratio),
        "test_models": list(plan["test_models"]),
        "train_models": list(train_info["models"]),
        "valid_models": list(valid_info["models"]),
        "train_rows": int(train_info["rows"]),
        "valid_rows": int(valid_info["rows"]),
        "test_rows": int(test_info["rows"]),
        "train_trajectories": int(train_info["trajectories"]),
        "valid_trajectories": int(valid_info["trajectories"]),
        "test_trajectories": int(test_info["trajectories"]),
        "train_instances": int(train_info["instances"]),
        "valid_instances": int(valid_info["instances"]),
        "test_instances": int(test_info["instances"]),
        "test_model_absent_from_train_valid": bool(
            test_models.isdisjoint(train_models) and test_models.isdisjoint(valid_models)
        ),
        "train_valid_instance_disjoint": bool(
            set(train_info["instance_ids"]).isdisjoint(set(valid_info["instance_ids"]))
        ),
        "test_kept_unfiltered": True,
        "train_valid_short_filter_min_steps": int(min_steps),
        "smoke_trajectories_per_split": int(smoke_trajectories_per_split),
    }


def _mask_and_summarize_split(
    df_train: pd.DataFrame,
    df_valid: pd.DataFrame,
    df_test: pd.DataFrame,
    *,
    dataset: str,
    cfg: Any,
    plan: dict[str, Any],
    test_model_ratio: float,
    valid_ratio: float,
    min_steps: int,
    smoke_trajectories_per_split: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any], pd.DataFrame]:
    if df_train.empty or df_valid.empty or df_test.empty:
        raise ValueError(f"{dataset}: empty split train={len(df_train)} valid={len(df_valid)} test={len(df_test)}")
    for split_name, split_frame in (("train", df_train), ("valid", df_valid), ("test", df_test)):
        _annotate_split_frame(split_frame, split_name)

    summary_rows = []
    infos = {}
    for split_name, split_frame in (("train", df_train), ("valid", df_valid), ("test", df_test)):
        infos[split_name] = _split_info(split_frame)
        summary_rows.extend(_split_summary_rows(split_frame, dataset=dataset, split_name=split_name, plan=plan))
    split_meta = _split_metadata_from_infos(
        dataset=dataset,
        cfg=cfg,
        plan=plan,
        infos=infos,
        test_model_ratio=test_model_ratio,
        valid_ratio=valid_ratio,
        min_steps=min_steps,
        smoke_trajectories_per_split=smoke_trajectories_per_split,
    )
    return df_train, df_valid, df_test, split_meta, pd.DataFrame(summary_rows)


def _split_by_model_holdout(
    frame: pd.DataFrame,
    *,
    dataset: str,
    cfg: Any,
    test_model_ratio: float,
    test_models_override: list[str] | None,
    valid_ratio: float,
    min_steps: int,
    smoke_trajectories_per_split: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any], pd.DataFrame]:
    plan = _model_holdout_plan(
        frame,
        dataset=dataset,
        cfg=cfg,
        test_model_ratio=test_model_ratio,
        test_models_override=test_models_override,
        valid_ratio=valid_ratio,
        min_steps=min_steps,
        smoke_trajectories_per_split=smoke_trajectories_per_split,
    )
    df_train, df_valid, df_test = _split_frame_by_plan(frame, plan)
    return _mask_and_summarize_split(
        df_train,
        df_valid,
        df_test,
        dataset=dataset,
        cfg=cfg,
        plan=plan,
        test_model_ratio=test_model_ratio,
        valid_ratio=valid_ratio,
        min_steps=min_steps,
        smoke_trajectories_per_split=smoke_trajectories_per_split,
    )


def _model_holdout_plan_from_path(
    path: Path,
    spec: dict[str, Any],
    *,
    dataset: str,
    cfg: Any,
    test_model_ratio: float,
    test_models_override: list[str] | None,
    valid_ratio: float,
    min_steps: int,
    smoke_trajectories_per_split: int,
) -> dict[str, Any]:
    meta_frame = _read_plan_metadata_frame(path, spec)
    plan = _model_holdout_plan(
        meta_frame,
        dataset=dataset,
        cfg=cfg,
        test_model_ratio=test_model_ratio,
        test_models_override=test_models_override,
        valid_ratio=valid_ratio,
        min_steps=min_steps,
        smoke_trajectories_per_split=smoke_trajectories_per_split,
    )
    del meta_frame
    gc.collect()
    return plan


def _read_one_split_frame_streamed(
    path: Path,
    spec: dict[str, Any],
    *,
    feature_preset: str,
    plan: dict[str, Any],
    split_name: str,
    parquet_batch_size: int,
) -> pd.DataFrame:
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(path)
    present = set(pf.schema_arrow.names)
    base_columns = [
        spec.get("traj_col", "traj_id"),
        spec.get("instance_col", "instance_id"),
        spec.get("model_col", "model_id"),
        spec.get("step_col", "prefix_step_idx"),
        spec.get("label_col", "label"),
        "prefix_id",
        "group_id",
        "sample_weight",
        *METADATA_NUMERIC_COLUMNS,
        *COMMON_NUMERIC_FEATURES,
        *COMMON_CATEGORICAL_FEATURES,
    ]
    if feature_preset == "rich_af_gold":
        base_columns.extend([col for col in ACTION_FEEDBACK_TEXT_COLUMNS if col in present])
        base_columns.extend([col for col in present if col.startswith("gold_") and not col.endswith("_text")])
    columns = [str(col) for col in dict.fromkeys(base_columns) if str(col) in present]
    rename = {
        str(spec.get("traj_col", "traj_id")): "traj_id",
        str(spec.get("instance_col", "instance_id")): "instance_id",
        str(spec.get("model_col", "model_id")): "model_id",
        str(spec.get("step_col", "prefix_step_idx")): "prefix_step_idx",
        str(spec.get("label_col", "label")): "label",
    }

    chunks: list[pd.DataFrame] = []
    batch_size = max(1, int(parquet_batch_size))
    for batch in pf.iter_batches(batch_size=batch_size, columns=columns):
        chunk = batch.to_pandas().rename(columns=rename)
        if "prefix_id" not in chunk:
            chunk["prefix_id"] = chunk["traj_id"].astype(str) + "::p" + chunk["prefix_step_idx"].astype(str)
        if "group_id" not in chunk:
            chunk["group_id"] = chunk["traj_id"].astype(str)
        if "sample_weight" not in chunk:
            if "n_steps_total_for_weighting" in chunk:
                denom = pd.to_numeric(chunk["n_steps_total_for_weighting"], errors="coerce").fillna(1.0).clip(lower=1.0)
            else:
                denom = pd.Series(1.0, index=chunk.index)
            chunk["sample_weight"] = (1.0 / denom).astype(np.float32)
        mask = _split_mask_by_plan(chunk, plan, split_name)
        if bool(mask.any()):
            chunks.append(chunk.loc[mask].copy())
        del chunk, mask
        gc.collect()

    renamed_columns = [rename.get(col, col) for col in columns]
    frame = pd.concat(chunks, ignore_index=True, copy=False) if chunks else pd.DataFrame(columns=renamed_columns)
    del chunks
    gc.collect()
    return _annotate_split_frame(frame, split_name)


def _read_split_frames_streamed(
    path: Path,
    spec: dict[str, Any],
    *,
    dataset: str,
    cfg: Any,
    feature_preset: str,
    test_model_ratio: float,
    test_models_override: list[str] | None,
    valid_ratio: float,
    min_steps: int,
    smoke_trajectories_per_split: int,
    parquet_batch_size: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any], pd.DataFrame]:
    meta_frame = _read_dataset_frame(
        path,
        spec,
        feature_preset="process",
        parquet_batch_size=parquet_batch_size,
    )
    plan = _model_holdout_plan(
        meta_frame,
        dataset=dataset,
        cfg=cfg,
        test_model_ratio=test_model_ratio,
        test_models_override=test_models_override,
        valid_ratio=valid_ratio,
        min_steps=min_steps,
        smoke_trajectories_per_split=smoke_trajectories_per_split,
    )
    del meta_frame
    gc.collect()

    import pyarrow.parquet as pq

    pf = pq.ParquetFile(path)
    present = set(pf.schema_arrow.names)
    base_columns = [
        spec.get("traj_col", "traj_id"),
        spec.get("instance_col", "instance_id"),
        spec.get("model_col", "model_id"),
        spec.get("step_col", "prefix_step_idx"),
        spec.get("label_col", "label"),
        "prefix_id",
        "group_id",
        "sample_weight",
        *METADATA_NUMERIC_COLUMNS,
        *COMMON_NUMERIC_FEATURES,
        *COMMON_CATEGORICAL_FEATURES,
    ]
    if feature_preset == "rich_af_gold":
        base_columns.extend([col for col in ACTION_FEEDBACK_TEXT_COLUMNS if col in present])
        base_columns.extend([col for col in present if col.startswith("gold_") and not col.endswith("_text")])
    columns = [str(col) for col in dict.fromkeys(base_columns) if str(col) in present]
    rename = {
        str(spec.get("traj_col", "traj_id")): "traj_id",
        str(spec.get("instance_col", "instance_id")): "instance_id",
        str(spec.get("model_col", "model_id")): "model_id",
        str(spec.get("step_col", "prefix_step_idx")): "prefix_step_idx",
        str(spec.get("label_col", "label")): "label",
    }

    train_chunks: list[pd.DataFrame] = []
    valid_chunks: list[pd.DataFrame] = []
    test_chunks: list[pd.DataFrame] = []
    for batch in pf.iter_batches(batch_size=max(1, int(parquet_batch_size)), columns=columns):
        chunk = batch.to_pandas().rename(columns=rename)
        if "prefix_id" not in chunk:
            chunk["prefix_id"] = chunk["traj_id"].astype(str) + "::p" + chunk["prefix_step_idx"].astype(str)
        if "group_id" not in chunk:
            chunk["group_id"] = chunk["traj_id"].astype(str)
        if "sample_weight" not in chunk:
            denom = pd.to_numeric(chunk["n_steps_total_for_weighting"], errors="coerce").fillna(1.0).clip(lower=1.0)
            chunk["sample_weight"] = (1.0 / denom).astype(np.float32)
        train_part, valid_part, test_part = _split_frame_by_plan(chunk, plan)
        if not train_part.empty:
            train_chunks.append(train_part)
        if not valid_part.empty:
            valid_chunks.append(valid_part)
        if not test_part.empty:
            test_chunks.append(test_part)
        del chunk, train_part, valid_part, test_part
        gc.collect()

    df_train = pd.concat(train_chunks, ignore_index=True) if train_chunks else pd.DataFrame(columns=columns)
    del train_chunks
    gc.collect()
    df_valid = pd.concat(valid_chunks, ignore_index=True) if valid_chunks else pd.DataFrame(columns=columns)
    del valid_chunks
    gc.collect()
    df_test = pd.concat(test_chunks, ignore_index=True) if test_chunks else pd.DataFrame(columns=columns)
    del test_chunks
    gc.collect()
    return _mask_and_summarize_split(
        df_train,
        df_valid,
        df_test,
        dataset=dataset,
        cfg=cfg,
        plan=plan,
        test_model_ratio=test_model_ratio,
        valid_ratio=valid_ratio,
        min_steps=min_steps,
        smoke_trajectories_per_split=smoke_trajectories_per_split,
    )


def _feature_columns(
    frame: pd.DataFrame,
    *,
    legacy: dict[str, Any],
    feature_preset: str,
) -> tuple[list[str], list[str], list[str]]:
    numeric = [col for col in COMMON_NUMERIC_FEATURES if col in frame.columns]
    categorical = [col for col in COMMON_CATEGORICAL_FEATURES if col in frame.columns]
    text_columns: list[str] = []
    if feature_preset == "rich_af_gold":
        numeric.extend([col for col in legacy["ANSWER_NUMERIC_FEATURES"] if col in frame.columns])
        numeric.extend([col for col in legacy["ANSWER_BOOL_FEATURES"] if col in frame.columns])
        categorical.extend([col for col in legacy["ANSWER_CATEGORICAL_FEATURES"] if col in frame.columns])
        text_columns = [col for col in ACTION_FEEDBACK_TEXT_COLUMNS if col in frame.columns]
    return list(dict.fromkeys(numeric)), list(dict.fromkeys(categorical)), text_columns


def _fit_feature_schema(
    train: pd.DataFrame,
    numeric: list[str],
    categorical: list[str],
    text_columns: list[str],
    *,
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, Any]]:
    categories = {}
    for col in categorical:
        values = train[col].fillna("__NULL__").astype(str)
        categories[col] = sorted(values.unique().tolist())
    text_blocks = {}
    text_schema_rows = []
    for col in text_columns:
        vectorizer = TfidfVectorizer(
            ngram_range=(1, int(args.tfidf_ngram_max)),
            min_df=int(args.tfidf_min_df),
            max_features=int(args.tfidf_max_features),
            sublinear_tf=True,
            dtype=np.float32,
        )
        texts = train[col].fillna("").astype(str)
        try:
            X_text = vectorizer.fit_transform(texts)
        except ValueError:
            text_schema_rows.append({"column": col, "vocabulary_size": 0, "svd_components": 0, "enabled": False})
            continue
        reducer = None
        n_features = int(X_text.shape[1])
        svd_components = 0
        max_components = min(int(args.tfidf_svd_dim), n_features - 1, len(train) - 1)
        if max_components >= 2:
            reducer = TruncatedSVD(n_components=max_components, random_state=42)
            reducer.fit(X_text)
            svd_components = int(max_components)
        text_blocks[col] = {"vectorizer": vectorizer, "reducer": reducer}
        text_schema_rows.append(
            {
                "column": col,
                "vocabulary_size": n_features,
                "svd_components": svd_components,
                "enabled": True,
            }
        )
        del X_text
        gc.collect()
    schema = {
        "feature_preset": args.feature_preset,
        "numeric": numeric,
        "categorical": categorical,
        "categories": categories,
        "text_blocks": text_schema_rows,
        "tfidf": {
            "ngram_range": [1, int(args.tfidf_ngram_max)],
            "min_df": int(args.tfidf_min_df),
            "max_features": int(args.tfidf_max_features),
            "svd_dim": int(args.tfidf_svd_dim),
        },
    }
    return schema, {"text_blocks": text_blocks}


def _build_features(frame: pd.DataFrame, schema: dict[str, Any], transformers: dict[str, Any]) -> sparse.csr_matrix:
    numeric = list(schema["numeric"])
    categorical = list(schema["categorical"])
    parts: list[sparse.spmatrix] = []
    if numeric:
        numeric_frame = frame[numeric].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
        numeric_frame = numeric_frame.fillna(0.0).astype(np.float32)
        parts.append(sparse.csr_matrix(numeric_frame.to_numpy(dtype=np.float32)))
    for col in categorical:
        values = frame[col].fillna("__NULL__").astype(str)
        categories = list(schema["categories"][col])
        encoded = pd.get_dummies(values, prefix=col, dtype=np.float32)
        expected = [f"{col}_{value}" for value in categories]
        encoded = encoded.reindex(columns=expected, fill_value=0.0)
        parts.append(sparse.csr_matrix(encoded.to_numpy(dtype=np.float32)))
    for col, block in (transformers.get("text_blocks") or {}).items():
        texts = frame[col].fillna("").astype(str)
        X_text = block["vectorizer"].transform(texts)
        reducer = block.get("reducer")
        if reducer is not None:
            X_text = sparse.csr_matrix(reducer.transform(X_text).astype(np.float32))
        else:
            X_text = X_text.tocsr().astype(np.float32)
        parts.append(X_text)
    if not parts:
        raise ValueError("No process features are available.")
    return sparse.hstack(parts, format="csr").astype(np.float32)


class ConstantProbabilityModel:
    def __init__(self, probability: float):
        self.probability = float(probability)

    def predict_proba(self, matrix: Any) -> np.ndarray:
        n_rows = int(matrix.shape[0]) if hasattr(matrix, "shape") else len(matrix)
        prob = np.full(n_rows, self.probability, dtype=np.float64)
        return np.column_stack([1.0 - prob, prob])


def _fit_lgbm(
    X_train: Any,
    y_train: np.ndarray,
    *,
    sample_weight: np.ndarray,
    args: argparse.Namespace,
    seed: int,
) -> tuple[Any, dict[str, Any]]:
    classes = np.unique(y_train)
    if len(classes) < 2:
        model = ConstantProbabilityModel(float(classes[0]))
        return model, {"constant": True, "constant_probability": float(classes[0]), "best_iteration": 0}
    import lightgbm as lgb

    model = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=int(args.num_boost_round),
        learning_rate=float(args.learning_rate),
        num_leaves=int(args.num_leaves),
        subsample=0.9,
        colsample_bytree=0.9,
        reg_alpha=0.1,
        reg_lambda=1.0,
        min_child_samples=50,
        random_state=int(seed),
        n_jobs=int(args.max_cpu_threads),
        verbosity=-1,
    )
    model.fit(X_train, y_train, sample_weight=sample_weight)
    return model, {"constant": False, "best_iteration": int(getattr(model, "best_iteration_", 0) or 0)}


def _positive_probability(model: Any, matrix: Any) -> np.ndarray:
    proba = model.predict_proba(matrix)
    if proba.ndim != 2 or proba.shape[1] < 2:
        return np.asarray(proba).ravel().astype(np.float64)
    return np.asarray(proba[:, 1], dtype=np.float64)


def _prediction_frame(frame: pd.DataFrame) -> pd.DataFrame:
    columns = [
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
    return frame[[col for col in columns if col in frame.columns]].copy()


def _save_model(path: Path, model: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(model, handle)


def _run_dataset(
    *,
    cfg: Any,
    args: argparse.Namespace,
    dataset: str,
    dataset_dir: Path,
    test_models: list[str] | None = None,
) -> dict[str, Any]:
    legacy = _import_legacy()
    predictor = PREDICTORS[args.feature_preset]
    datasets = cfg.payload.get("datasets") or {}
    if dataset not in datasets:
        raise KeyError(f"Unknown dataset: {dataset}")
    spec = datasets[dataset] or {}
    path = _resolve_project_path(spec["prefix_table"])
    if not path.exists():
        raise FileNotFoundError(path)
    valid_ratio = float(args.valid_ratio if args.valid_ratio is not None else (cfg.payload.get("split") or {}).get("valid_ratio", 0.15))
    min_steps = int(((cfg.payload.get("split") or {}).get("train_valid_filter") or {}).get("min_steps", 2))
    split_plan: dict[str, Any] | None = None
    split_infos: dict[str, dict[str, Any]] | None = None
    split_summary_rows: list[dict[str, Any]] | None = None
    if args.feature_preset == "rich_af_gold":
        split_plan = _model_holdout_plan_from_path(
            path,
            spec,
            dataset=dataset,
            cfg=cfg,
            test_model_ratio=float(args.test_model_ratio),
            test_models_override=test_models,
            valid_ratio=valid_ratio,
            min_steps=min_steps,
            smoke_trajectories_per_split=int(args.smoke_trajectories_per_split),
        )
        df_train = _read_one_split_frame_streamed(
            path,
            spec,
            feature_preset=args.feature_preset,
            plan=split_plan,
            split_name="train",
            parquet_batch_size=int(args.parquet_batch_size),
        )
        if df_train.empty:
            raise ValueError(f"{dataset}: empty split train={len(df_train)}")
        split_infos = {"train": _split_info(df_train)}
        split_summary_rows = _split_summary_rows(df_train, dataset=dataset, split_name="train", plan=split_plan)
        df_train = _sample_rows(df_train, max_rows=int(args.max_train_rows), seed=_stable_seed(cfg.seed, dataset, "train_rows"))
        sample_for_columns = df_train.head(1).copy()
        split_meta = None
        split_summary = None
    else:
        frame = _read_dataset_frame(
            path,
            spec,
            feature_preset=args.feature_preset,
            parquet_batch_size=int(args.parquet_batch_size),
        )
        df_train, df_valid, df_test, split_meta, split_summary = _split_by_model_holdout(
            frame,
            dataset=dataset,
            cfg=cfg,
            test_model_ratio=float(args.test_model_ratio),
            test_models_override=test_models,
            valid_ratio=valid_ratio,
            min_steps=min_steps,
            smoke_trajectories_per_split=int(args.smoke_trajectories_per_split),
        )
        del frame
        gc.collect()
        df_train = _sample_rows(df_train, max_rows=int(args.max_train_rows), seed=_stable_seed(cfg.seed, dataset, "train_rows"))
        df_valid = _sample_rows(df_valid, max_rows=int(args.max_valid_rows), seed=_stable_seed(cfg.seed, dataset, "valid_rows"))
        df_test = _sample_rows(df_test, max_rows=int(args.max_test_rows), seed=_stable_seed(cfg.seed, dataset, "test_rows"))
        sample_for_columns = pd.concat([df_train.head(1), df_valid.head(1), df_test.head(1)], ignore_index=True)
    numeric, categorical, text_columns = _feature_columns(
        sample_for_columns,
        legacy=legacy,
        feature_preset=args.feature_preset,
    )
    schema, transformers = _fit_feature_schema(
        df_train,
        numeric,
        categorical,
        text_columns,
        args=args,
    )
    y_success_train, y_failure_train = legacy["_safe_targets"](df_train, args.safe_label_min_step)
    w_train = df_train["sample_weight"].to_numpy(dtype=np.float32)

    fit_rows = []
    cal_rows = []
    trained_heads = []
    models_dir = ensure_dir(dataset_dir / "models") if not args.no_save_models else None
    X_train = _build_features(df_train, schema, transformers)
    train_rows = int(X_train.shape[0])
    feature_count = int(X_train.shape[1])
    del df_train
    gc.collect()
    for head_name, y_train, column_prefix, seed_offset in (
        ("safe_success", y_success_train, "success", 0),
        ("safe_failure", y_failure_train, "failure", 97),
    ):
        model_name = f"{predictor}__{head_name}"
        model, model_meta = _fit_lgbm(
            X_train,
            y_train,
            sample_weight=w_train,
            args=args,
            seed=int(cfg.seed) + seed_offset,
        )
        if models_dir is not None:
            _save_model(models_dir / f"{model_name}.pkl", model)
        fit_rows.append(
            {
                "dataset": dataset,
                "model_name": model_name,
                "predictor": predictor,
                "feature_preset": args.feature_preset,
                "head": head_name,
                "train_rows": train_rows,
                "features": feature_count,
                "positive_rate": float(np.mean(y_train)),
                **model_meta,
            }
        )
        trained_heads.append(
            {
                "head_name": head_name,
                "model_name": model_name,
                "model": model,
                "column_prefix": column_prefix,
            }
        )
    del X_train, y_success_train, y_failure_train, w_train
    gc.collect()

    if args.feature_preset == "rich_af_gold":
        assert split_plan is not None and split_infos is not None and split_summary_rows is not None
        df_valid = _read_one_split_frame_streamed(
            path,
            spec,
            feature_preset=args.feature_preset,
            plan=split_plan,
            split_name="valid",
            parquet_batch_size=int(args.parquet_batch_size),
        )
        if df_valid.empty:
            raise ValueError(f"{dataset}: empty split valid={len(df_valid)}")
        split_infos["valid"] = _split_info(df_valid)
        split_summary_rows.extend(_split_summary_rows(df_valid, dataset=dataset, split_name="valid", plan=split_plan))
        df_valid = _sample_rows(df_valid, max_rows=int(args.max_valid_rows), seed=_stable_seed(cfg.seed, dataset, "valid_rows"))
    y_success_valid, y_failure_valid = legacy["_safe_targets"](df_valid, args.safe_label_min_step)
    w_valid = df_valid["sample_weight"].to_numpy(dtype=np.float32)
    valid_pred = _prediction_frame(df_valid)
    X_valid = _build_features(df_valid, schema, transformers)
    del df_valid
    gc.collect()
    valid_outputs = {}
    valid_targets = {
        "safe_success": y_success_valid,
        "safe_failure": y_failure_valid,
    }
    for head in trained_heads:
        y_valid = valid_targets[str(head["head_name"])]
        valid_raw = _positive_probability(head["model"], X_valid)
        calibrator = legacy["fit_sigmoid_calibrator"](valid_raw, y_valid, sample_weight=w_valid)
        valid_cal = calibrator.predict(valid_raw)
        column_prefix = str(head["column_prefix"])
        valid_pred[legacy["_head_column"](column_prefix, "raw", predictor)] = valid_raw.astype(np.float32)
        valid_pred[legacy["_head_column"](column_prefix, "calibrated", predictor)] = valid_cal.astype(np.float32)
        valid_outputs[str(head["head_name"])] = {
            "calibrator": calibrator,
            "y_valid": y_valid,
            "valid_raw": valid_raw,
        }
    del X_valid, y_success_valid, y_failure_valid, w_valid
    gc.collect()

    if args.feature_preset == "rich_af_gold":
        assert split_plan is not None and split_infos is not None and split_summary_rows is not None
        df_test = _read_one_split_frame_streamed(
            path,
            spec,
            feature_preset=args.feature_preset,
            plan=split_plan,
            split_name="test",
            parquet_batch_size=int(args.parquet_batch_size),
        )
        if df_test.empty:
            raise ValueError(f"{dataset}: empty split test={len(df_test)}")
        split_infos["test"] = _split_info(df_test)
        split_summary_rows.extend(_split_summary_rows(df_test, dataset=dataset, split_name="test", plan=split_plan))
        split_meta = _split_metadata_from_infos(
            dataset=dataset,
            cfg=cfg,
            plan=split_plan,
            infos=split_infos,
            test_model_ratio=float(args.test_model_ratio),
            valid_ratio=valid_ratio,
            min_steps=min_steps,
            smoke_trajectories_per_split=int(args.smoke_trajectories_per_split),
        )
        split_summary = pd.DataFrame(split_summary_rows)
        df_test = _sample_rows(df_test, max_rows=int(args.max_test_rows), seed=_stable_seed(cfg.seed, dataset, "test_rows"))
    y_success_test, y_failure_test = legacy["_safe_targets"](df_test, args.safe_label_min_step)
    test_pred = _prediction_frame(df_test)
    X_test = _build_features(df_test, schema, transformers)
    del df_test
    gc.collect()
    test_targets = {
        "safe_success": y_success_test,
        "safe_failure": y_failure_test,
    }
    for head in trained_heads:
        head_name = str(head["head_name"])
        column_prefix = str(head["column_prefix"])
        valid_output = valid_outputs[head_name]
        y_test = test_targets[head_name]
        test_raw = _positive_probability(head["model"], X_test)
        test_cal = valid_output["calibrator"].predict(test_raw)
        test_pred[legacy["_head_column"](column_prefix, "raw", predictor)] = test_raw.astype(np.float32)
        test_pred[legacy["_head_column"](column_prefix, "calibrated", predictor)] = test_cal.astype(np.float32)
        if models_dir is not None:
            _save_model(models_dir / f"calibrator_{head['model_name']}.pkl", valid_output["calibrator"])
        cal_rows.append(
            {
                "dataset": dataset,
                "head": head_name,
                **legacy["calibration_summary_row"](
                    model_name=str(head["model_name"]),
                    calibrator=valid_output["calibrator"],
                    y_valid=valid_output["y_valid"],
                    raw_prob_valid=valid_output["valid_raw"],
                    y_test=y_test,
                    raw_prob_test=test_raw,
                ),
            }
        )

    dataset_dir.mkdir(parents=True, exist_ok=True)
    valid_pred.to_parquet(dataset_dir / "valid_predictions_safe_stop.parquet", index=False)
    test_pred.to_parquet(dataset_dir / "test_predictions_safe_stop.parquet", index=False)
    split_summary.to_csv(dataset_dir / "split_summary.csv", index=False)
    write_json(dataset_dir / "split_metadata.json", split_meta)
    write_json(dataset_dir / "feature_schema.json", {**schema, "feature_count": feature_count})
    if models_dir is not None:
        _save_model(models_dir / "feature_transformers.pkl", transformers)
    pd.DataFrame(fit_rows).to_csv(dataset_dir / "model_fit_summary.csv", index=False)
    pd.DataFrame(cal_rows).to_csv(dataset_dir / "safe_stop_calibration_summary.csv", index=False)

    policies = legacy["_policy_grid"](
        success_thresholds=args.success_thresholds,
        failure_thresholds=args.failure_thresholds,
        min_steps=args.policy_min_steps,
        consecutive_values=args.consecutive,
    )
    valid_grid, valid_per_agent = legacy["_evaluate_policies"](
        valid_pred,
        run_label=str(dataset_dir),
        predictors=[predictor],
        score_modes=args.score_modes,
        policies=policies,
    )
    selected = legacy["_select_policies"](
        valid_grid,
        max_valid_abs_drop_pp=float(args.max_valid_abs_drop_pp),
        min_valid_decision_acc=float(args.min_valid_decision_acc),
        fallback_min_save_pct=float(args.fallback_min_save_pct),
    )
    test_selected = legacy["_evaluate_selected"](test_pred, run_label=str(dataset_dir), selected=selected)
    valid_grid.to_csv(dataset_dir / "safe_stop_valid_policy_grid.csv", index=False)
    valid_per_agent.to_csv(dataset_dir / "safe_stop_valid_policy_per_agent.csv", index=False)
    selected.to_csv(dataset_dir / "safe_stop_selected_policies.csv", index=False)
    test_selected.to_csv(dataset_dir / "safe_stop_test_selected.csv", index=False)
    legacy["_write_report"](dataset_dir, selected, test_selected)
    (dataset_dir / "_SUCCESS").write_text("robustness model-holdout completed\n", encoding="utf-8")

    del X_test, y_success_test, y_failure_test, trained_heads, valid_outputs, transformers
    gc.collect()
    return {
        "dataset": dataset,
        "output_dir": str(dataset_dir),
        "test_models": split_meta["test_models"],
        "test_model": split_meta["test_models"][0] if len(split_meta["test_models"]) == 1 else ",".join(split_meta["test_models"]),
        "train_rows": split_meta["train_rows"],
        "valid_rows": split_meta["valid_rows"],
        "test_rows": split_meta["test_rows"],
        "selected_rows": int(len(selected)),
        "predictor": predictor,
    }


def _write_summary(
    run_dir: Path,
    dataset_results: list[dict[str, Any]],
    *,
    predictor: str,
    feature_preset: str,
) -> None:
    summary_dir = ensure_dir(run_dir / "summary")
    rows = []
    for result in dataset_results:
        dataset_dir = Path(result["output_dir"])
        test_path = dataset_dir / "safe_stop_test_selected.csv"
        selected_path = dataset_dir / "safe_stop_selected_policies.csv"
        if not test_path.exists() or not selected_path.exists():
            continue
        test = pd.read_csv(test_path)
        selected = pd.read_csv(selected_path)
        for frame, split in ((selected, "valid_selected"), (test, "test_locked")):
            copy = frame.copy()
            copy.insert(0, "dataset", result["dataset"])
            copy.insert(1, "test_model", result.get("test_model", ""))
            copy.insert(2, "summary_split", split)
            rows.append(copy)
    combined = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    if not combined.empty:
        write_table(combined, summary_dir / "selected_and_test_metrics.csv")
    table_rows = []
    for result in dataset_results:
        test_path = Path(result["output_dir"]) / "safe_stop_test_selected.csv"
        if not test_path.exists():
            continue
        test = pd.read_csv(test_path)
        if test.empty:
            continue
        row = test.iloc[0].to_dict()
        table_rows.append(
            {
                "dataset": result["dataset"],
                "test_model": result.get("test_model", ""),
                "test_models": len(result["test_models"]),
                "n": int(row["original_total"]),
                "decided": int(row["n_decided"]),
                "coverage_pct": f"{float(row['coverage']) * 100.0:.2f}",
                "acc_pct": f"{float(row['decision_accuracy']) * 100.0:.2f}" if not pd.isna(row["decision_accuracy"]) else "nan",
                "save_pct": f"{float(row['pct_steps_saved']):.2f}",
                "resolve_change_pp": f"{-float(row['resolve_rate_drop']) * 100.0:+.2f}",
            }
        )
    lines = [
        "# Robustness 15% Model-Holdout",
        "",
        f"- run_dir: `{run_dir}`",
        f"- predictor: `{predictor}`",
        f"- feature_preset: `{feature_preset}`",
        "- `process`: process/numeric/action only",
        "- `rich_af_gold`: process dense + answer/gold dense + action-feedback TF-IDF/SVD",
        "",
        *_markdown_table(
            table_rows,
            ["dataset", "test_model", "test_models", "n", "decided", "coverage_pct", "acc_pct", "save_pct", "resolve_change_pp"],
        ),
        "",
    ]
    (summary_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    cfg = load_earlyeval_config(args.config)
    predictor = PREDICTORS[args.feature_preset]
    out = ensure_dir(args.output_dir or _default_output_dir(cfg, "earlyeval_lightgbm"))
    run_dir = ensure_dir(out / args.run_subdir)
    command_rows = []
    datasets_cfg = cfg.payload.get("datasets") or {}
    for dataset in args.datasets:
        if args.split_mode == "leave_one_model":
            spec = datasets_cfg[dataset] or {}
            path = _resolve_project_path(spec["prefix_table"])
            model_ids = _list_dataset_models(path, spec)
            if args.only_test_models:
                requested = [str(item) for item in args.only_test_models]
                model_ids = [model_id for model_id in model_ids if model_id in set(requested)]
                missing = sorted(set(requested) - set(model_ids))
                if missing:
                    raise ValueError(f"{dataset}: requested --only-test-models not present: {missing}")
            for model_id in model_ids:
                dataset_dir = run_dir / dataset / _safe_name(model_id)
                status = "skipped_existing" if (dataset_dir / "_SUCCESS").exists() and not args.force else "pending"
                command_rows.append(
                    {
                        "dataset": dataset,
                        "test_model": model_id,
                        "test_models": model_id,
                        "output_dir": str(dataset_dir),
                        "status": status,
                        "feature_preset": args.feature_preset,
                        "split_mode": args.split_mode,
                        "test_model_ratio": float(args.test_model_ratio),
                        "smoke_trajectories_per_split": int(args.smoke_trajectories_per_split),
                        "max_train_rows": int(args.max_train_rows),
                        "max_valid_rows": int(args.max_valid_rows),
                        "max_test_rows": int(args.max_test_rows),
                    }
                )
        else:
            dataset_dir = run_dir / dataset
            status = "skipped_existing" if (dataset_dir / "_SUCCESS").exists() and not args.force else "pending"
            command_rows.append(
                {
                    "dataset": dataset,
                    "test_model": "",
                    "test_models": "",
                    "output_dir": str(dataset_dir),
                    "status": status,
                    "feature_preset": args.feature_preset,
                    "split_mode": args.split_mode,
                    "test_model_ratio": float(args.test_model_ratio),
                    "smoke_trajectories_per_split": int(args.smoke_trajectories_per_split),
                    "max_train_rows": int(args.max_train_rows),
                    "max_valid_rows": int(args.max_valid_rows),
                    "max_test_rows": int(args.max_test_rows),
                }
            )
    write_table(pd.DataFrame(command_rows), run_dir / "command_index.csv")
    write_json(
        run_dir / "run_manifest.json",
        {
            "ok": True,
            "execute": bool(args.execute),
            "datasets": list(args.datasets),
            "run_dir": str(run_dir),
            "predictor": predictor,
            "feature_preset": args.feature_preset,
            "split_mode": args.split_mode,
            "test_model_ratio": float(args.test_model_ratio),
            "features": "process-only or rich dense+answer/gold+action-feedback TF-IDF depending on feature_preset",
        },
    )
    if not args.execute:
        print(json.dumps({"ok": True, "execute": False, "run_dir": str(run_dir), "datasets": args.datasets}, indent=2))
        return 0

    completed = 0
    skipped = 0
    failed = []
    results = []
    for row in command_rows:
        dataset = row["dataset"]
        dataset_dir = Path(row["output_dir"])
        if (dataset_dir / "_SUCCESS").exists() and not args.force:
            skipped += 1
            results.append({"dataset": dataset, "output_dir": str(dataset_dir), "test_models": [], "predictor": predictor})
            continue
        test_models = [str(row["test_model"])] if args.split_mode == "leave_one_model" else None
        print(f"[robustness] dataset={dataset} test_model={row.get('test_model', '')}", flush=True)
        try:
            result = _run_dataset(
                cfg=cfg,
                args=args,
                dataset=dataset,
                dataset_dir=dataset_dir,
                test_models=test_models,
            )
            completed += 1
            results.append(result)
        except Exception as exc:
            failed.append({"dataset": dataset, "error": repr(exc)})
            break
    _write_summary(run_dir, results, predictor=predictor, feature_preset=args.feature_preset)
    summary = {
        "ok": not failed,
        "completed": completed,
        "skipped": skipped,
        "failed": failed,
        "run_dir": str(run_dir),
    }
    write_json(run_dir / "execution_summary.json", summary)
    print(json.dumps(summary, indent=2))
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
