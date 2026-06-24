from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from earlyeval.core.paths import package_root


def load_experiment_registry(path: str | Path | None = None) -> dict[str, Any]:
    cfg_path = Path(path) if path else package_root() / "configs" / "experiment_registry.yaml"
    if not cfg_path.is_absolute():
        cfg_path = package_root() / cfg_path
    payload = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    return payload


def list_experiments(path: str | Path | None = None) -> list[dict[str, Any]]:
    registry = load_experiment_registry(path)
    rows = []
    for name, cfg in (registry.get("experiments", {}) or {}).items():
        item = {"name": name}
        if isinstance(cfg, dict):
            item.update(cfg)
        rows.append(item)
    return rows
