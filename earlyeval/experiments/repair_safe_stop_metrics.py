from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def default_root() -> Path:
    return Path(__file__).resolve().parents[2] / "paper" / "experiments" / "earlyeval_lightgbm"


def _as_number(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _safe_rate(num: pd.Series, den: pd.Series) -> pd.Series:
    den = den.replace({0: np.nan})
    return num / den


def _changed(before: pd.Series, after: pd.Series) -> pd.Series:
    left = _as_number(before)
    right = _as_number(after)
    return pd.Series(~np.isclose(left, right, rtol=1e-10, atol=1e-12, equal_nan=True), index=before.index)


def _assign_numeric(df: pd.DataFrame, col: str, values: pd.Series, changed_mask: pd.Series) -> pd.Series:
    if col not in df.columns:
        return changed_mask
    changed_mask = changed_mask | _changed(df[col], values)
    if values.notna().all() and np.all(np.isclose(values, np.round(values), rtol=0.0, atol=1e-9)):
        df[col] = np.round(values).astype(int)
    else:
        df[col] = values
    return changed_mask


def _repair_metric_scheme(
    df: pd.DataFrame,
    *,
    total_col: str,
    resolved_col: str,
    false_negative_col: str,
) -> tuple[bool, int]:
    required = {total_col, resolved_col, false_negative_col, "adjusted_resolved"}
    if not required.issubset(df.columns):
        return False, 0

    total = _as_number(df[total_col])
    resolved = _as_number(df[resolved_col])
    false_negatives = _as_number(df[false_negative_col]).fillna(0)
    false_positives = (
        _as_number(df["false_positives"]).fillna(0)
        if "false_positives" in df.columns
        else 0
    )
    adjusted_resolved = resolved - false_negatives + false_positives
    original_rate = (
        _as_number(df["original_resolve_rate"])
        if "original_resolve_rate" in df.columns
        else _safe_rate(resolved, total)
    )
    original_rate = original_rate.where(original_rate.notna(), _safe_rate(resolved, total))
    adjusted_rate = _safe_rate(adjusted_resolved, total)
    drop = original_rate - adjusted_rate
    drop_pp = drop * 100.0
    change_pp = -drop_pp

    changed_mask = pd.Series(False, index=df.index)
    changed_mask = _assign_numeric(df, "adjusted_resolved", adjusted_resolved, changed_mask)
    changed_mask = _assign_numeric(df, "adjusted_resolve_rate", adjusted_rate, changed_mask)
    changed_mask = _assign_numeric(df, "adjusted_resolve_rate_pct", adjusted_rate * 100.0, changed_mask)
    changed_mask = _assign_numeric(df, "resolve_rate_drop", drop, changed_mask)
    changed_mask = _assign_numeric(df, "resolve_rate_drop_pp", drop_pp, changed_mask)
    changed_mask = _assign_numeric(df, "resolve_rate_change_pp", change_pp, changed_mask)
    changed_mask = _assign_numeric(df, "drop_pp", drop_pp, changed_mask)
    changed_mask = _assign_numeric(df, "abs_drop_pp", drop_pp.abs(), changed_mask)
    changed_mask = _assign_numeric(df, "valid_abs_drop_pp", drop_pp.abs(), changed_mask)
    changed_mask = _assign_numeric(df, "test_resolve_rate_change_pp", change_pp, changed_mask)
    changed_mask = _assign_numeric(df, "valid_resolve_rate_change_pp", change_pp, changed_mask)
    changed_mask = _assign_numeric(df, "rate_delta", adjusted_rate - original_rate, changed_mask)
    return bool(changed_mask.any()), int(changed_mask.sum())


def repair_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    repaired = df.copy()
    total_changed = 0
    schemes = [
        ("original_total", "original_resolved", "false_negatives"),
        ("total", "resolved", "false_negatives"),
    ]
    for total_col, resolved_col, false_negative_col in schemes:
        changed, rows = _repair_metric_scheme(
            repaired,
            total_col=total_col,
            resolved_col=resolved_col,
            false_negative_col=false_negative_col,
        )
        if changed:
            total_changed += rows
            break
    return repaired, total_changed


def repair_csv(
    path: Path,
    *,
    dry_run: bool,
    backup_suffix: str | None = None,
) -> dict[str, Any] | None:
    try:
        frame = pd.read_csv(path)
    except Exception as exc:
        return {"path": str(path), "status": "read_error", "error": str(exc)}

    repaired, rows_changed = repair_frame(frame)
    if rows_changed == 0:
        return None
    backup_path: Path | None = None
    if not dry_run:
        if backup_suffix:
            backup_path = path.with_suffix(path.suffix + backup_suffix)
            if backup_path.exists():
                # Preserve the original snapshot; never overwrite an existing
                # backup so re-running the repair multiple times does not
                # silently destroy the first untouched copy.
                pass
            else:
                backup_path.write_bytes(path.read_bytes())
        repaired.to_csv(path, index=False)
    payload = {
        "path": str(path),
        "status": "dry_run" if dry_run else "updated",
        "rows_changed": rows_changed,
    }
    if backup_path is not None:
        payload["backup"] = str(backup_path)
    return payload


def _csv_frame_changed(path: Path, frame: pd.DataFrame) -> bool:
    if not path.exists():
        return True
    try:
        existing = pd.read_csv(path)
    except Exception:
        return True
    if list(existing.columns) != list(frame.columns) or len(existing) != len(frame):
        return True
    for col in frame.columns:
        left_numeric = pd.to_numeric(existing[col], errors="coerce")
        right_numeric = pd.to_numeric(frame[col], errors="coerce")
        numeric_mask = left_numeric.notna() | right_numeric.notna()
        if numeric_mask.any():
            same_numeric = np.isclose(
                left_numeric[numeric_mask],
                right_numeric[numeric_mask],
                rtol=1e-10,
                atol=1e-12,
                equal_nan=True,
            )
            if not bool(np.all(same_numeric)):
                return True
        text_mask = ~numeric_mask
        if text_mask.any():
            left_text = existing.loc[text_mask, col].astype("string").fillna("<NA>")
            right_text = frame.loc[text_mask, col].astype("string").fillna("<NA>")
            if not bool((left_text == right_text).all()):
                return True
    return False


def _aggregate_selected_policy(frame: pd.DataFrame) -> dict[str, Any]:
    if frame.empty:
        return {}
    total = float(frame["original_total"].sum())
    total_steps = float(frame["total_steps"].sum())
    saved_steps = float(frame["total_saved_steps"].sum())
    decided = float(frame["n_decided"].sum())
    original_resolved = float(frame["original_resolved"].sum())
    adjusted_resolved = float(frame["adjusted_resolved"].sum())
    false_negatives = float(frame["false_negatives"].sum()) if "false_negatives" in frame.columns else 0.0
    true_negatives = float(frame["true_negatives"].sum()) if "true_negatives" in frame.columns else 0.0
    false_positives = float(frame["false_positives"].sum()) if "false_positives" in frame.columns else 0.0
    true_positives = float(frame["true_positives"].sum()) if "true_positives" in frame.columns else 0.0
    true_decisions = true_negatives + true_positives
    fold_totals = frame["original_total"].astype(float)
    fold_resolve_change_pp = (
        (frame["adjusted_resolved"].astype(float) - frame["original_resolved"].astype(float))
        * 100.0
        / fold_totals.replace(0.0, float("nan"))
    )
    mean_abs_resolve_rate_change_pp = (
        float((fold_resolve_change_pp.abs() * fold_totals).sum() / total) if total else 0.0
    )
    return {
        "folds": int(frame["fold_id"].nunique()) if "fold_id" in frame.columns else int(len(frame)),
        "trajectories": int(total),
        "original_resolved": int(original_resolved),
        "adjusted_resolved": int(adjusted_resolved),
        "false_negatives": int(false_negatives),
        "false_positives": int(false_positives),
        "true_negatives": int(true_negatives),
        "true_positives": int(true_positives),
        "original_resolve_rate_pct": original_resolved * 100.0 / total if total else 0.0,
        "adjusted_resolve_rate_pct": adjusted_resolved * 100.0 / total if total else 0.0,
        "resolve_rate_change_pp": (adjusted_resolved - original_resolved) * 100.0 / total if total else 0.0,
        "mean_abs_resolve_rate_change_pp": mean_abs_resolve_rate_change_pp,
        "decided_trajectories": int(decided),
        "coverage_pct": decided * 100.0 / total if total else 0.0,
        "decision_accuracy_pct": true_decisions * 100.0 / decided if decided else 0.0,
        "saved_steps": int(saved_steps),
        "total_steps": int(total_steps),
        "step_save_pct": saved_steps * 100.0 / total_steps if total_steps else 0.0,
    }


def _aggregate_by_target(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty or "target_valid_decision_accuracy" not in frame.columns:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for target, part in frame.groupby("target_valid_decision_accuracy", sort=True):
        row = _aggregate_selected_policy(part)
        row["target_valid_decision_accuracy"] = float(target)
        row["target_valid_decision_accuracy_pct"] = float(target) * 100.0
        rows.append(row)
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows).sort_values("target_valid_decision_accuracy")
    leading = ["target_valid_decision_accuracy", "target_valid_decision_accuracy_pct"]
    return out[leading + [col for col in out.columns if col not in leading]]


def refresh_csv_aggregates(root: Path, *, dry_run: bool) -> list[dict[str, Any]]:
    updates: list[dict[str, Any]] = []
    for per_fold in root.rglob("per_fold_test_selected.csv"):
        frame, _ = repair_frame(pd.read_csv(per_fold))
        aggregate = _aggregate_selected_policy(frame)
        if not aggregate:
            continue
        out = per_fold.parent / "aggregate_test_summary.csv"
        aggregate_frame = pd.DataFrame([aggregate])
        if not _csv_frame_changed(out, aggregate_frame):
            continue
        if not dry_run:
            aggregate_frame.to_csv(out, index=False)
        updates.append({"path": str(out), "status": "dry_run" if dry_run else "updated", "source": str(per_fold)})

    for split in ("test", "valid"):
        for per_fold in root.rglob(f"per_fold_{split}_metrics.csv"):
            frame, _ = repair_frame(pd.read_csv(per_fold))
            aggregate = _aggregate_by_target(frame)
            if aggregate.empty:
                continue
            out = per_fold.parent / f"aggregate_{split}_metrics.csv"
            if not _csv_frame_changed(out, aggregate):
                continue
            if not dry_run:
                aggregate.to_csv(out, index=False)
            updates.append({"path": str(out), "status": "dry_run" if dry_run else "updated", "source": str(per_fold)})
    return updates


def repair_tree(
    root: Path,
    *,
    dry_run: bool,
    backup_suffix: str | None = None,
) -> list[dict[str, Any]]:
    updates: list[dict[str, Any]] = []
    for csv_path in sorted(root.rglob("*.csv")):
        result = repair_csv(csv_path, dry_run=dry_run, backup_suffix=backup_suffix)
        if result:
            updates.append(result)
    updates.extend(refresh_csv_aggregates(root, dry_run=dry_run))
    return updates


def refresh_lightgbm_summary(*, config: Path, output_dir: Path, dry_run: bool) -> dict[str, Any] | None:
    if dry_run:
        return {"status": "dry_run", "config": str(config), "output_dir": str(output_dir)}
    from earlyeval.experiments.paper_pipeline import summarize_lightgbm_main

    return summarize_lightgbm_main(config=config, output_dir=output_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Repair safe-stop resolve-rate metrics using adjusted_resolved = original_resolved - false_negatives + false_positives."
    )
    parser.add_argument("--root", type=Path, action="append", default=None, help="Artifact root to scan recursively.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--backup-suffix",
        default=None,
        help=(
            "When set (e.g. '.bak'), keep an unmodified snapshot of every CSV "
            "we rewrite at <path><suffix>. Existing backup files are never "
            "overwritten, so it is safe to re-run the repair multiple times."
        ),
    )
    parser.add_argument("--refresh-lightgbm-summary", action="store_true")
    parser.add_argument("--config", type=Path, default=Path("configs/earlyeval.yaml"))
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--audit-json", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    roots = args.root or [default_root()]
    updates: list[dict[str, Any]] = []
    for root in roots:
        updates.extend(
            repair_tree(root, dry_run=bool(args.dry_run), backup_suffix=args.backup_suffix)
        )

    summary_result = None
    if args.refresh_lightgbm_summary:
        output_dir = args.output_dir or roots[0]
        summary_result = refresh_lightgbm_summary(config=args.config, output_dir=output_dir, dry_run=bool(args.dry_run))

    payload = {
        "roots": [str(path) for path in roots],
        "dry_run": bool(args.dry_run),
        "updated_files": len([item for item in updates if item.get("status") != "read_error"]),
        "updates": updates,
        "lightgbm_summary": summary_result,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default))
    audit_path = args.audit_json
    if audit_path and not args.dry_run:
        audit_path.parent.mkdir(parents=True, exist_ok=True)
        audit_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")
    return 0


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        numeric = float(value)
        return None if math.isnan(numeric) else numeric
    if isinstance(value, float) and math.isnan(value):
        return None
    return str(value)


if __name__ == "__main__":
    raise SystemExit(main())
