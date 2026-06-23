from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from final3.core.legacy_paths import answer_module_root, require_path, shared_prefix_table_filtered_path
from final3.core.paths import load_paths


@dataclass(frozen=True)
class DualHeadRunSpec:
    run_name: str = "model_holdout_answer_calibrated_full"
    prefix_table: Path | None = None
    verified_jsonl: Path | None = None
    feature_engineer_path: Path | None = None
    holdout_models: str = "auto_mid3"
    max_instances: int = 500
    output_subdir: str = "final3_current_safe_stop_dual_head"
    variants: tuple[str, ...] = ("i",)
    lgbm_preset: str = "strong_reg"
    safe_label_min_step: int = 10
    policy_min_steps: tuple[int, ...] = (0,)
    consecutive: tuple[int, ...] = (1,)
    success_thresholds: tuple[float, ...] = (0.95,)
    failure_thresholds: tuple[float, ...] = (0.95,)
    score_modes: tuple[str, ...] = ("calibrated",)
    max_cpu_threads: int = 8
    low_memory: bool = True
    mask_train_model_id: bool = True


def build_dual_head_command(spec: DualHeadRunSpec) -> list[str]:
    script = require_path(answer_module_root() / "safe_stop_dual_head_retrain.py", "dual-head trainer")
    prefix_table = require_path(
        spec.prefix_table or shared_prefix_table_filtered_path(),
        "shared answer-aware prefix table",
    )
    paths = load_paths()
    verified_jsonl = require_path(
        spec.verified_jsonl or paths.data_root / "swe_verify_500" / "offical_answer" / "test.jsonl",
        "SWE-bench verified jsonl",
    )
    feature_engineer = require_path(
        spec.feature_engineer_path or paths.feature_engineer_with_model,
        "shared feature engineer pickle",
    )
    python_executable = paths.python_executable or Path(sys.executable)
    command = [
        str(python_executable),
        str(script),
        "--run-name",
        spec.run_name,
        "--prefix-table",
        str(prefix_table),
        "--verified-jsonl",
        str(verified_jsonl),
        "--feature-engineer-path",
        str(feature_engineer),
        "--holdout-models",
        spec.holdout_models,
        "--max-instances",
        str(spec.max_instances),
        "--split-strategy",
        "per_instance_model",
        "--output-subdir",
        spec.output_subdir,
        "--variants",
        *spec.variants,
        "--lgbm-preset",
        spec.lgbm_preset,
        "--safe-label-min-step",
        str(spec.safe_label_min_step),
        "--policy-min-steps",
        *(str(v) for v in spec.policy_min_steps),
        "--consecutive",
        *(str(v) for v in spec.consecutive),
        "--success-thresholds",
        *(str(v) for v in spec.success_thresholds),
        "--failure-thresholds",
        *(str(v) for v in spec.failure_thresholds),
        "--score-modes",
        *spec.score_modes,
        "--max-cpu-threads",
        str(spec.max_cpu_threads),
    ]
    if spec.low_memory:
        command.append("--low-memory")
    if spec.mask_train_model_id:
        command.append("--mask-train-model-id")
    return command


def run_dual_head(spec: DualHeadRunSpec, *, execute: bool = False) -> dict[str, object]:
    cmd = build_dual_head_command(spec)
    payload: dict[str, object] = {
        "execute": bool(execute),
        "command": cmd,
        "note": "Dry-run by default. Use --execute only after confirming this heavy training job is intended.",
    }
    if execute:
        subprocess.run(cmd, check=True)
        payload["completed"] = True
    return payload
