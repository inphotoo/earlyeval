from __future__ import annotations

from pathlib import Path
from typing import Any

from earlyeval.core.contracts import PolicySpec


CURRENT_SAFE_STOP = PolicySpec(
    name="current_safe_stop",
    predictor="I_LightGBM_Dense_AF",
    score_mode="calibrated",
    policy_mode="dual",
    success_thr=0.95,
    failure_thr=0.95,
    min_step=0,
    consecutive=1,
)


def default_config_path() -> Path:
    return Path(__file__).resolve().parents[2] / "configs" / "policy_presets.yaml"


def _load_yaml(path: Path) -> dict[str, Any]:
    import yaml

    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Policy preset file must be a mapping: {path}")
    return payload


def load_policy_preset(name: str, config_path: str | Path | None = None) -> PolicySpec:
    if name == CURRENT_SAFE_STOP.name and config_path is None:
        path = default_config_path()
        if not path.exists():
            return CURRENT_SAFE_STOP
    else:
        path = Path(config_path) if config_path is not None else default_config_path()

    if not path.exists():
        if name == CURRENT_SAFE_STOP.name:
            return CURRENT_SAFE_STOP
        raise FileNotFoundError(f"Policy preset config not found: {path}")

    payload = _load_yaml(path)
    raw = (payload.get("presets") or {}).get(name)
    if not raw:
        if name == CURRENT_SAFE_STOP.name:
            return CURRENT_SAFE_STOP
        raise KeyError(f"Unknown policy preset: {name}")

    return PolicySpec(
        name=name,
        predictor=str(raw["predictor"]),
        score_mode=str(raw.get("score_mode", "calibrated")),
        policy_mode=str(raw.get("policy_mode", "dual")),
        success_thr=float(raw["success_thr"]),
        failure_thr=float(raw["failure_thr"]),
        min_step=int(raw.get("min_step", 0)),
        consecutive=int(raw.get("consecutive", 1)),
    )
