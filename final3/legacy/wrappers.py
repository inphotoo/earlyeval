from __future__ import annotations


MIGRATION_MAP = {
    "safe_stop_dual_head_retrain.py": "python -m final3.cli train dual-head [--execute]",
    "valid_policy_tuning_posthoc.py": "python -m final3.cli policy apply --preset current_safe_stop --predictions ...",
}


def explain_legacy_entry(name: str) -> str:
    target = MIGRATION_MAP.get(name)
    if target is None:
        known = ", ".join(sorted(MIGRATION_MAP))
        return f"No final3 wrapper is registered for {name}. Known entries: {known}"
    return f"{name} is replaced by:\n\n  {target}\n"
