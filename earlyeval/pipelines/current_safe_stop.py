from __future__ import annotations

from pathlib import Path

from earlyeval.policies.apply import apply_policy_to_file


def package_root() -> Path:
    return Path(__file__).resolve().parents[2]


def run_current_safe_stop(
    *,
    mode: str,
    predictions: str | Path | None = None,
    output_dir: str | Path | None = None,
    preset: str = "current_safe_stop",
) -> dict[str, str]:
    if mode not in {"smoke", "main", "full"}:
        raise ValueError(f"Unsupported mode: {mode}")

    if mode == "full":
        raise RuntimeError(
            "full mode is intentionally opt-in but not wired to heavy experiments in earlyeval v1. "
            "Use dedicated experiment entries after reviewing configs/experiment_registry.yaml."
        )

    if mode == "smoke":
        predictions = predictions or package_root() / "examples" / "smoke_predictions.csv"
        output_dir = output_dir or package_root() / "outputs" / "current_safe_stop_smoke"
    else:
        if predictions is None:
            raise ValueError("--predictions is required for --mode main")
        output_dir = output_dir or package_root() / "outputs" / "current_safe_stop_main"

    return apply_policy_to_file(
        predictions=predictions,
        output_dir=output_dir,
        preset=preset,
    )
