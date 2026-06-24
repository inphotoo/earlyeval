from __future__ import annotations

from pathlib import Path

from earlyeval.core.io import ensure_dir, read_table, write_json, write_table
from earlyeval.policies.presets import load_policy_preset
from earlyeval.policies.safe_stop import apply_policy


def apply_policy_to_file(
    *,
    predictions: str | Path,
    output_dir: str | Path,
    preset: str = "current_safe_stop",
    preset_config: str | Path | None = None,
) -> dict[str, str]:
    output = ensure_dir(output_dir)
    policy = load_policy_preset(preset, preset_config)
    frame = read_table(predictions)
    decisions, summary, per_agent = apply_policy(frame, policy)
    paths = {
        "decisions": str(write_table(decisions, output / "policy_decisions.csv")),
        "summary": str(write_table(summary, output / "policy_summary.csv")),
        "per_agent": str(write_table(per_agent, output / "policy_per_agent.csv")),
        "metadata": str(
            write_json(
                output / "run_metadata.json",
                {
                    "preset": preset,
                    "preset_config": str(preset_config) if preset_config else None,
                    "predictions": str(predictions),
                    "output_dir": str(output),
                },
            )
        ),
    }
    return paths
