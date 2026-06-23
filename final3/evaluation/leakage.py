from __future__ import annotations

from final3.core.splits import validate_disjoint_groups, validate_model_holdout


def audit_split_frame(frame) -> dict[str, object]:
    instance_overlap = validate_disjoint_groups(frame) if "instance_id" in frame.columns else []
    model_report = (
        validate_model_holdout(frame)
        if "orig_model_id" in frame.columns and "split" in frame.columns
        else None
    )
    return {
        "ok": not instance_overlap and (model_report.ok if model_report else True),
        "overlapping_instances": instance_overlap,
        "overlapping_models": model_report.overlapping_models if model_report else [],
    }
