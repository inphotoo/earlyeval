from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LeakageReport:
    """split 泄漏检查结果。

    `ok=False` 表示当前 split 不适合做主实验结论。`overlapping_instances`
    和 `overlapping_models` 分别记录 instance-level 与 model-holdout 层面的
    交叉项，便于报告里直接定位问题。
    """

    ok: bool
    overlapping_instances: list[str]
    overlapping_models: list[str]


def validate_disjoint_groups(frame, split_col: str = "split", group_col: str = "instance_id") -> list[str]:
    """检查同一 group 是否跨多个 split。

    主实验默认要求同一 SWE instance 不能同时出现在 train/valid/test 中。
    这个函数也可以用于其它 group，例如按 `traj_id` 或自定义 task family
    检查。返回的是违规 group id 列表；空列表表示通过。
    """

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
    """检查 heldout test model 是否泄漏到 train/valid。

    model-holdout 实验的核心约束是：测试 agent/model 在训练和验证中完全
    不可见。默认情况下函数同时检查 instance 级别的 overlap，因为对严格
    holdout 来说，model 泄漏和 instance 泄漏都会让结果虚高。

    `allow_known_task_overlap=True` 适用于 final3 的
    `leave_one_test_model_known_task` 设定：测试 model 仍然必须不出现在
    train/valid，但 instance 在 train/valid/test 之间允许重叠（known
    task 重复评估），所以这种重叠不算泄漏，不影响 `ok` 字段。返回值仍然
    会列出 instance overlap，便于报告里展示重叠规模。
    """

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
