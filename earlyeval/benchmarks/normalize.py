from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from earlyeval.core.io import ensure_dir, write_table


def _load_records(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    payload = json.loads(text)
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict) and isinstance(payload.get("records"), list):
        return payload["records"]
    if isinstance(payload, dict):
        return [payload]
    raise ValueError(f"Unsupported JSON payload in {path}")


def _messages_from_terminalbench(row: dict[str, Any]) -> list[dict[str, Any]]:
    raw_steps = row.get("steps", [])
    if isinstance(raw_steps, str):
        try:
            raw_steps = json.loads(raw_steps)
        except json.JSONDecodeError:
            raw_steps = []
    messages: list[dict[str, Any]] = []
    if not isinstance(raw_steps, list):
        return messages
    for item in raw_steps:
        if not isinstance(item, dict):
            continue
        src = item.get("src") or item.get("role") or item.get("type")
        role = "assistant" if src == "agent" else src
        messages.append(
            {
                "role": role,
                "content": item.get("msg") or item.get("content") or "",
                "tools": item.get("tools") or [],
                "observation": item.get("obs") or item.get("observation") or "",
            }
        )
    return messages


def normalize_record(row: dict[str, Any], benchmark: str, row_idx: int) -> dict[str, Any]:
    if benchmark == "terminalbench":
        task_name = str(row.get("task_name") or row.get("instance_id") or f"terminalbench_{row_idx:06d}")
        trial = str(row.get("trial_id") or row.get("trial_name") or row_idx)
        return {
            "benchmark": "terminalbench",
            "instance_id": task_name,
            "traj_id": str(row.get("traj_id") or f"terminalbench::{row_idx:06d}::{trial}"),
            "model_id": str(row.get("model") or row.get("agent") or "__MISSING_MODEL__"),
            "resolved": bool(row.get("reward") == 1 or row.get("resolved") is True),
            "messages": _messages_from_terminalbench(row),
            "patch": str(row.get("patch") or ""),
        }
    return {
        "benchmark": benchmark,
        "instance_id": str(row.get("instance_id") or row.get("task_id") or f"{benchmark}_{row_idx:06d}"),
        "traj_id": str(row.get("traj_id") or row.get("trajectory_id") or f"{benchmark}::{row_idx:06d}"),
        "model_id": str(row.get("model_id") or row.get("model") or "__MISSING_MODEL__"),
        "resolved": bool(row.get("resolved") or row.get("success") or row.get("passed")),
        "messages": row.get("messages") or row.get("trajectory") or [],
        "patch": str(row.get("patch") or ""),
    }


def normalize_file(*, benchmark: str, input_path: str | Path, output_dir: str | Path) -> dict[str, str]:
    import pandas as pd

    output = ensure_dir(output_dir)
    raw = _load_records(Path(input_path))
    rows = [normalize_record(row, benchmark, idx) for idx, row in enumerate(raw)]
    frame = pd.DataFrame(rows)
    audit = pd.DataFrame(
        [
            {
                "trajectories": int(len(frame)),
                "instances": int(frame["instance_id"].nunique()) if not frame.empty else 0,
                "models": int(frame["model_id"].nunique()) if not frame.empty else 0,
                "resolved_rate": float(frame["resolved"].mean()) if not frame.empty else 0.0,
                "empty_messages": int((frame["messages"].map(len) == 0).sum()) if not frame.empty else 0,
            }
        ]
    )
    return {
        "normalized": str(write_table(frame, output / "normalized_trajectories.jsonl")),
        "audit": str(write_table(audit, output / "quality_audit.csv")),
    }
