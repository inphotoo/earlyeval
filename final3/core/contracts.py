from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class TrajectoryRecord:
    """统一的轨迹输入契约。

    `final3` 后续会接入 SWE-bench、TerminalBench、Toolathlon 或其它
    benchmark。不同来源的原始 JSON 字段差异很大，所以进入核心流水线前
    先归一化成这个最小结构。

    字段说明：
    - `benchmark`: 数据来源名称，用于报告和 adapter 分流。
    - `instance_id`: 题目/任务 id；做 instance-level split 时以它防泄漏。
    - `traj_id`: 单条轨迹的唯一 id；同一 instance 可以有多模型、多 trial。
    - `model_id`: 产生该轨迹的 agent/model 名称。训练时可能被 mask 成
      `__MISSING__`，但原始值要保留在上游 manifest 或 `metadata` 中。
    - `resolved`: 最终二分类标签，表示完整轨迹是否解决问题。
    - `messages`: 已归一化的消息/动作序列，后续会构建 step/prefix。
    - `patch`: 可选最终 patch 或答案文本。不是所有 benchmark 都有。
    - `metadata`: 不进入核心模型契约的附加信息，例如 trial_id、cost、tokens。
    """

    benchmark: str
    instance_id: str
    traj_id: str
    model_id: str
    resolved: bool
    messages: list[dict[str, Any]]
    patch: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PrefixRecord:
    """单个 prefix 样本的最小契约。

    一条轨迹会展开成多个 prefix：`p0, p1, ..., pN`。每个 prefix 表示
    agent 执行到某一步时，模型可见的信息快照。当前主策略最终使用的是
    prefix 级概率，再聚合成轨迹级 safe-stop 决策。

    `label` 是完整轨迹的最终 resolved 标签，不是当前 prefix 的局部标签。
    `sample_weight` 通常设为 `1/(n_steps+1)`，避免长轨迹因为 prefix 更多而
    在训练中占过大权重。
    """

    prefix_id: str
    traj_id: str
    instance_id: str
    model_id: str
    prefix_step_idx: int
    label: int
    sample_weight: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PolicySpec:
    """safe-stop 策略的完整可执行描述。

    当前主策略是双头策略：一个 head 预测 safe-success，另一个 head 预测
    safe-failure。策略在每个 prefix 上检查：

    - success 分数是否达到 `success_thr`
    - failure 分数是否达到 `failure_thr`
    - 当前步数是否不小于 `min_step`
    - 同一方向是否连续命中 `consecutive` 次

    这样可以把“预测最终会成功”和“预测已经可以安全判失败”分开建模，
    避免用单一 final-success 概率强行兼顾两个方向。
    """

    name: str
    predictor: str
    score_mode: str
    policy_mode: str
    success_thr: float
    failure_thr: float
    min_step: int = 0
    consecutive: int = 1


@dataclass(frozen=True)
class ExperimentRun:
    """一次 final3 入口调用的轻量运行记录。

    这个 dataclass 不负责保存大产物，只描述一次运行的身份、模式、配置和
    输出目录。后续如果加入 run registry，可以直接把它序列化到 metadata。
    """

    run_id: str
    mode: str
    config_path: Path | None
    output_dir: Path
    metadata: dict[str, Any] = field(default_factory=dict)
