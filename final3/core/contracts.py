from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class TrajectoryRecord:
    """Normalized trajectory-level input record used across benchmarks."""

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
    """Single prefix example derived from a full trajectory."""

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
    """Executable dual-head safe-stop policy configuration."""

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
    """Lightweight metadata record for one final3 experiment entrypoint call."""

    run_id: str
    mode: str
    config_path: Path | None
    output_dir: Path
    metadata: dict[str, Any] = field(default_factory=dict)
