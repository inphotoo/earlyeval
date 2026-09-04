"""Rebuild `examples/smoke_predictions.csv` from public SWE-bench Verified results.

Task ids, agent names, ground-truth `resolved` labels, and trajectory lengths are
read from the official SWE-bench experiments repository:

    https://github.com/SWE-bench/experiments
    evaluation/verified/<run_id>/per_instance_details.json

The two probability columns cannot come from that source: they are the output of a
trained EarlyEval predictor, which is not part of this code-only release. They are
generated here as monotone curves consistent with each run's real label, purely so
the smoke test exercises every branch of the safe-stop decision rule. Do not read
them as measurements.

Usage:
    python examples/build_fixture.py [--out examples/smoke_predictions.csv]
"""

from __future__ import annotations

import argparse
import csv
import json
import urllib.request
from pathlib import Path

BASE = "https://raw.githubusercontent.com/SWE-bench/experiments/main/evaluation/verified"

# Real leaderboard runs of the mini-SWE-agent scaffold, one per model family.
RUNS = {
    "mini-swe-agent__claude-opus-4.5": "20251124_mini-v1.16.0_claude-opus-4-5-20251101",
    "mini-swe-agent__gpt-5.2-high": "20251211_mini-v1.17.2_gpt-5.2-2025-12-11-high",
    "mini-swe-agent__glm-4.6": "20251201_mini-v1.17.1_glm-4.6",
}

PREDICTOR = "I_LightGBM_Dense_AF"
SUCC = f"prob_cal_safe_success__{PREDICTOR}"
FAIL = f"prob_cal_safe_failure__{PREDICTOR}"
THRESHOLD = 0.95

MIN_STEPS, MAX_STEPS = 14, 40


def fetch(run_id: str) -> dict[str, dict]:
    url = f"{BASE}/{run_id}/per_instance_details.json"
    with urllib.request.urlopen(url) as response:
        return json.loads(response.read().decode("utf-8"))


def pick(details: dict[str, dict], offset: int) -> list[tuple[str, bool, int, str]]:
    """Choose one success-stop, one failure-stop, and one undecided run.

    `offset` walks each agent to a different slice of the task list so the
    fixture spans several repositories instead of repeating the same instance.
    """
    usable = sorted(
        (name, bool(row["resolved"]), int(row["api_calls"]))
        for name, row in details.items()
        if row.get("api_calls") and MIN_STEPS <= int(row["api_calls"]) <= MAX_STEPS
    )
    resolved = [item for item in usable if item[1]]
    failed = [item for item in usable if not item[1]]
    return [
        (*resolved[offset % len(resolved)], "success"),
        (*failed[offset % len(failed)], "failure"),
        (*resolved[(offset + len(resolved) // 2) % len(resolved)], "undecided"),
    ]


def curve(step: int, n_steps: int, kind: str) -> tuple[float, float]:
    """Illustrative calibrated probabilities for one prefix. Not measurements."""
    frac = step / n_steps
    if kind == "undecided":
        return round(0.28 + 0.40 * frac, 4), round(0.36 - 0.16 * frac, 4)

    cross = max(4, round(n_steps * 0.5))  # halt near the midpoint of the run
    if step < cross:
        rising = round(0.22 + (THRESHOLD - 0.22) * (step / cross) * 0.93, 4)
    else:
        rising = round(min(0.988, THRESHOLD + 0.038 * ((step - cross) / cross)), 4)
    falling = round(max(0.012, 0.42 - 0.37 * frac), 4)
    return (rising, falling) if kind == "success" else (falling, rising)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("examples/smoke_predictions.csv"))
    args = parser.parse_args()

    rows = []
    for offset, (agent, run_id) in enumerate(RUNS.items()):
        details = fetch(run_id)
        for instance_id, resolved, n_steps, kind in pick(details, offset * 7):
            for step in range(n_steps + 1):
                p_success, p_failure = curve(step, n_steps, kind)
                rows.append(
                    {
                        "traj_id": f"{agent}::{instance_id}",
                        "instance_id": instance_id,
                        "model_id": agent,
                        "label": int(resolved),
                        "prefix_step_idx": step,
                        "n_steps_total": n_steps,
                        SUCC: p_success,
                        FAIL: p_failure,
                    }
                )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="\n") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    trajectories = len({row["traj_id"] for row in rows})
    print(f"wrote {args.out}  rows={len(rows)}  trajectories={trajectories}")


if __name__ == "__main__":
    main()
