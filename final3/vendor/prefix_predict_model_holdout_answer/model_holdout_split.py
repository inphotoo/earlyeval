"""Model-holdout split helpers for prefix prediction."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

import config
from utils import get_logger

logger = get_logger("model_holdout_split")

DEFAULT_EXCLUDED_EXTREME_MODEL = "20251124_mini-v1.16.0_claude-opus-4-5-20251101"
DEFAULT_HOLDOUT_MODELS = [
    "20251124_mini-v1.17.0_minimax-m2",
    "20251201_mini-v1.17.1_deepseek-v3.2-reasoner",
    "20251210_mini-v1.17.2_kimi-k2-thinking",
]


def load_verified_instance_ids(path: str | Path) -> list[str]:
    out = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            obj = json.loads(line)
            instance_id = str(obj.get("instance_id", "")).strip()
            if instance_id:
                out.append(instance_id)
    return out


def _choose_auto_mid3(model_stats: pd.DataFrame) -> list[str]:
    eligible = model_stats[
        (model_stats["trajs"] >= 450)
        & (model_stats.index != DEFAULT_EXCLUDED_EXTREME_MODEL)
    ].sort_values("success")
    preferred = [m for m in DEFAULT_HOLDOUT_MODELS if m in eligible.index]
    if len(preferred) == 3:
        return preferred
    preferred_any_coverage = [m for m in DEFAULT_HOLDOUT_MODELS if m in model_stats.index]
    if len(preferred_any_coverage) == 3:
        logger.warning(
            "auto_mid3: no full-coverage eligible model set; using default mid holdouts "
            "present in this small/smoke dataset."
        )
        return preferred_any_coverage
    if len(eligible) < 4:
        raise ValueError(
            "Not enough eligible models for auto_mid3. "
            f"eligible={eligible.index.tolist()}"
        )
    mid = len(eligible) // 2
    start = max(0, mid - 1)
    return eligible.index[start:start + 3].tolist()


def _choose_auto_extreme3(model_stats: pd.DataFrame, *, highest: bool) -> list[str]:
    score_col = "traj_success" if "traj_success" in model_stats.columns else "success"
    eligible = model_stats[model_stats["trajs"] >= 450].sort_values(score_col)
    if len(eligible) < 3:
        label = "auto_high_success3" if highest else "auto_low_success3"
        raise ValueError(
            f"Not enough eligible models for {label}. "
            f"eligible={eligible.index.tolist()}"
        )
    selected = eligible.tail(3) if highest else eligible.head(3)
    return selected.index.tolist()


def choose_auto_holdout_models(model_stats: pd.DataFrame, holdout_models: str) -> list[str]:
    if holdout_models == "auto_mid3":
        return _choose_auto_mid3(model_stats)
    if holdout_models == "auto_high_success3":
        return _choose_auto_extreme3(model_stats, highest=True)
    if holdout_models == "auto_low_success3":
        return _choose_auto_extreme3(model_stats, highest=False)
    return [x.strip() for x in holdout_models.split(",") if x.strip()]


def select_model_holdout_split(
    prefix_df: pd.DataFrame,
    *,
    verified_jsonl: str | Path,
    holdout_models: str = "auto_mid3",
    max_instances: int = 500,
    valid_ratio: float | None = None,
    seed: int | None = None,
) -> tuple[pd.Index, pd.Index, pd.Index, dict]:
    if "model_id" not in prefix_df.columns:
        raise ValueError("model_holdout split requires model_id column")
    if "instance_id" not in prefix_df.columns:
        raise ValueError("model_holdout split requires instance_id column")

    valid_ratio = config.VALID_RATIO if valid_ratio is None else valid_ratio
    seed = config.SPLIT_SEED if seed is None else seed
    verified_ids = load_verified_instance_ids(verified_jsonl)
    available_ids = set(prefix_df["instance_id"].astype(str).unique())
    selected_instances = [x for x in verified_ids if x in available_ids]
    if max_instances and max_instances > 0:
        selected_instances = selected_instances[:max_instances]
    if not selected_instances:
        raise ValueError("No overlap between verified_jsonl and prefix_df instance_id")

    work = prefix_df[prefix_df["instance_id"].astype(str).isin(selected_instances)].copy()
    model_stats = work.groupby("model_id").agg(
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
    model_stats["traj_success"] = traj_success

    heldout = choose_auto_holdout_models(model_stats, holdout_models)
    missing = [m for m in heldout if m not in model_stats.index]
    if missing:
        raise ValueError(f"Requested holdout models not found after filtering: {missing}")

    trainval = work[~work["model_id"].isin(heldout)]
    test = work[work["model_id"].isin(heldout)]
    if trainval.empty or test.empty:
        raise ValueError(f"Empty trainval/test in model_holdout split. heldout={heldout}")

    groups = trainval["instance_id"].values
    gss = GroupShuffleSplit(n_splits=1, test_size=valid_ratio, random_state=seed + 17)
    train_pos, valid_pos = next(gss.split(trainval, groups=groups))
    train_idx = trainval.index[train_pos]
    valid_idx = trainval.index[valid_pos]
    test_idx = test.index

    train_models = set(prefix_df.loc[train_idx, "model_id"].astype(str).unique())
    test_models = set(prefix_df.loc[test_idx, "model_id"].astype(str).unique())
    overlap = sorted(train_models & test_models)
    if overlap:
        raise AssertionError(f"Model leakage in holdout split: {overlap}")

    meta = {
        "mode": "model_holdout",
        "verified_jsonl": str(verified_jsonl),
        "verified_instances_total": len(verified_ids),
        "selected_instances": len(selected_instances),
        "holdout_models": heldout,
        "train_models": sorted(train_models),
        "valid_models": sorted(prefix_df.loc[valid_idx, "model_id"].astype(str).unique()),
        "test_models": sorted(test_models),
        "model_stats": model_stats.reset_index(),
    }
    logger.info(
        "Model-holdout split: selected_instances=%s train_models=%s holdout=%s",
        len(selected_instances), len(train_models), heldout,
    )
    return train_idx, valid_idx, test_idx, meta


def write_model_holdout_summary(prefix_df: pd.DataFrame, split_meta: dict, output_path: str | Path) -> pd.DataFrame:
    rows = []
    for split_name in ["train", "valid", "test"]:
        sub = prefix_df[prefix_df["split"] == split_name]
        if sub.empty:
            continue
        for model_id, part in sub.groupby("model_id"):
            rows.append({
                "split": split_name,
                "model_id": model_id,
                "instances": part["instance_id"].nunique(),
                "trajectories": part["traj_id"].nunique() if "traj_id" in part else np.nan,
                "prefixes": len(part),
                "label_rate": float(part["label"].mean()) if "label" in part else np.nan,
                "is_holdout": model_id in set(split_meta.get("holdout_models", [])),
            })
    summary = pd.DataFrame(rows).sort_values(["split", "is_holdout", "label_rate", "model_id"])
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_path, index=False)
    logger.info(f"Saved model holdout split summary: {output_path}")
    return summary
