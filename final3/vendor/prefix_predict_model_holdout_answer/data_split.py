'Public-release English note.'
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

import config
from utils import get_logger

logger = get_logger("data_split")


def split_by_group(
    df: pd.DataFrame,
    group_col: str = "group_id",
    train_ratio: float = None,
    valid_ratio: float = None,
    test_ratio: float = None,
    seed: int = None,
    *,
    check_instance_leak: bool = True,
) -> tuple[pd.Index, pd.Index, pd.Index]:
    'Public-release English note.'
    train_ratio = train_ratio or config.TRAIN_RATIO
    valid_ratio = valid_ratio or config.VALID_RATIO
    test_ratio = test_ratio or config.TEST_RATIO
    seed = seed or config.SPLIT_SEED

    groups = df[group_col].values
    n_unique_groups = df[group_col].nunique()

    # Public-release English note.
    if n_unique_groups < 2:
        logger.warning(
            'Public-release English note.'
            'Public-release English note.'
        )
        all_idx = df.index.copy()
        return all_idx, all_idx, all_idx

    # Public-release English note.
    test_size = test_ratio
    gss1 = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    trainval_idx, test_idx = next(gss1.split(df, groups=groups))

    # Public-release English note.
    trainval_df = df.iloc[trainval_idx]
    trainval_groups = trainval_df[group_col].values
    n_tv_rows = len(trainval_df)
    n_tv_groups = trainval_df[group_col].nunique()

    # Public-release English note.
    # Public-release English note.
    valid_size_adj = valid_ratio / (train_ratio + valid_ratio)
    use_fallback_tv = n_tv_groups < 2 or n_tv_rows < 2

    if not use_fallback_tv:
        gss2 = GroupShuffleSplit(n_splits=1, test_size=valid_size_adj, random_state=seed + 1)
        try:
            train_sub_idx, valid_sub_idx = next(
                gss2.split(trainval_df, groups=trainval_groups)
            )
        except ValueError as e:
            logger.warning(
                f"Second GroupShuffleSplit failed ({e}); "
                f"trainval rows={n_tv_rows}, groups={n_tv_groups}. Using mini-data fallback."
            )
            use_fallback_tv = True

    if use_fallback_tv:
        logger.warning(
            f"Train/valid split fallback: trainval has {n_tv_rows} rows, {n_tv_groups} {group_col} groups. "
            'Public-release English note.'
        )
        train_sub_idx = np.arange(n_tv_rows, dtype=int)
        valid_sub_idx = train_sub_idx.copy()

    train_final = df.index[trainval_idx[train_sub_idx]]
    valid_final = df.index[trainval_idx[valid_sub_idx]]
    test_final = df.index[test_idx]

    # Public-release English note.
    if use_fallback_tv:
        logger.warning(
            "Skipping group/instance leakage checks (train/valid overlap allowed in mini-data fallback)."
        )
    else:
        _verify_no_group_leak(df, train_final, valid_final, test_final, group_col)
        if check_instance_leak:
            _verify_no_instance_leak(df, train_final, valid_final, test_final)
        else:
            logger.warning(
                "check_instance_leak=False: skipping instance_id cross-split check "
                "(intended for multi-trajectory-per-instance + split by trajectory)."
            )

    logger.info(f"Split sizes - Train: {len(train_final)}, Valid: {len(valid_final)}, Test: {len(test_final)}")
    logger.info(f"Split ratios - Train: {len(train_final)/len(df):.3f}, "
                f"Valid: {len(valid_final)/len(df):.3f}, "
                f"Test: {len(test_final)/len(df):.3f}")

    # Public-release English note.
    for name, idx in [("Train", train_final), ("Valid", valid_final), ("Test", test_final)]:
        sub = df.loc[idx]
        n_groups = sub[group_col].nunique()
        n_trajs = sub["traj_id"].nunique() if "traj_id" in sub.columns else "N/A"
        pos_rate = sub["label"].mean() if "label" in sub.columns else "N/A"
        logger.info(f"  {name}: {n_groups} groups, {n_trajs} trajs, pos_rate={pos_rate:.4f}" if isinstance(pos_rate, float) else f"  {name}: {n_groups} groups, {n_trajs} trajs")

    return train_final, valid_final, test_final


def _verify_no_group_leak(df, train_idx, valid_idx, test_idx, group_col):
    'Public-release English note.'
    if (
        len(train_idx) == len(valid_idx) == len(test_idx)
        and set(train_idx) == set(valid_idx) == set(test_idx)
    ):
        logger.warning("Group leakage check skipped (smoke mode: identical splits).")
        return

    train_groups = set(df.loc[train_idx, group_col].unique())
    valid_groups = set(df.loc[valid_idx, group_col].unique())
    test_groups = set(df.loc[test_idx, group_col].unique())

    tv = train_groups & valid_groups
    tt = train_groups & test_groups
    vt = valid_groups & test_groups

    if tv or tt or vt:
        msg = f"Group leakage detected! train∩valid={len(tv)}, train∩test={len(tt)}, valid∩test={len(vt)}"
        logger.error(msg)
        raise ValueError(msg)

    logger.info("Group leakage check PASSED - no overlap between splits")


def _verify_no_instance_leak(df, train_idx, valid_idx, test_idx):
    'Public-release English note.'
    if "instance_id" not in df.columns:
        logger.warning("instance_id column not found; skip instance leakage check")
        return

    if (
        len(train_idx) == len(valid_idx) == len(test_idx)
        and set(train_idx) == set(valid_idx) == set(test_idx)
    ):
        logger.warning("Instance leakage check skipped (smoke mode: identical splits).")
        return

    train_inst = set(df.loc[train_idx, "instance_id"].astype(str))
    valid_inst = set(df.loc[valid_idx, "instance_id"].astype(str))
    test_inst = set(df.loc[test_idx, "instance_id"].astype(str))

    tv = train_inst & valid_inst
    tt = train_inst & test_inst
    vt = valid_inst & test_inst
    if tv or tt or vt:
        msg = (
            "Instance leakage detected! "
            f"train∩valid={len(tv)}, train∩test={len(tt)}, valid∩test={len(vt)}"
        )
        logger.error(msg)
        raise ValueError(msg)

    logger.info("Instance leakage check PASSED - no overlap between splits")
