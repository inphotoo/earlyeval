from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LeakageReport:
    """Leakage audit result for split and model-holdout checks."""

    ok: bool
    overlapping_instances: list[str]
    overlapping_models: list[str]


def validate_disjoint_groups(frame, split_col: str = "split", group_col: str = "instance_id") -> list[str]:
    """Return group ids that appear in more than one split."""

    if split_col not in frame.columns or group_col not in frame.columns:
        raise ValueError(f"Missing required columns: {split_col}, {group_col}")
    counts = frame.groupby(group_col)[split_col].nunique(dropna=True)
    return sorted(str(idx) for idx, value in counts.items() if int(value) > 1)


def validate_model_holdout(
    frame,
    split_col: str = "split",
    model_col: str = "orig_model_id",
    *,
    allow_known_task_overlap: bool = False,
) -> LeakageReport:
    """Check that held-out models do not appear in train or validation splits."""

    if split_col not in frame.columns or model_col not in frame.columns:
        raise ValueError(f"Missing required columns: {split_col}, {model_col}")
    train_models = set(frame.loc[frame[split_col].eq("train"), model_col].astype(str))
    test_models = set(frame.loc[frame[split_col].eq("test"), model_col].astype(str))
    valid_models = set(frame.loc[frame[split_col].eq("valid"), model_col].astype(str))
    overlap = sorted((train_models | valid_models) & test_models)
    instance_overlap = (
        validate_disjoint_groups(frame, split_col=split_col, group_col="instance_id")
        if "instance_id" in frame.columns
        else []
    )
    instance_overlap_breaks_ok = bool(instance_overlap) and not allow_known_task_overlap
    return LeakageReport(
        ok=not overlap and not instance_overlap_breaks_ok,
        overlapping_instances=instance_overlap,
        overlapping_models=overlap,
    )
