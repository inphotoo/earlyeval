from __future__ import annotations

import hashlib
import math
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from earlyeval.core.io import ensure_dir, write_json, write_table
from earlyeval.core.paths import load_paths, package_root


@dataclass(frozen=True)
class EarlyEvalConfig:
    path: Path
    payload: dict[str, Any]

    @property
    def seed(self) -> int:
        return int((self.payload.get("run") or {}).get("seed", 42))

    @property
    def run_id(self) -> str:
        smoke = self.payload.get("smoke") or {}
        run = self.payload.get("run") or {}
        return str(smoke.get("run_id") or run.get("default_run_id") or "earlyeval_smoke")


def _resolve_project_path(value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return package_root() / path


def load_earlyeval_config(config: str | Path = "configs/earlyeval.yaml") -> EarlyEvalConfig:
    cfg_path = _resolve_project_path(config)
    payload = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    return EarlyEvalConfig(path=cfg_path, payload=payload)


def _default_output_dir(cfg: EarlyEvalConfig, run_id: str | None = None) -> Path:
    run = cfg.payload.get("run") or {}
    root = _resolve_project_path(run.get("output_root", "paper/experiments"))
    return root / (run_id or cfg.run_id)


def _dataset_items(cfg: EarlyEvalConfig, names: list[str] | None = None) -> list[tuple[str, dict[str, Any]]]:
    datasets = cfg.payload.get("datasets") or {}
    selected = names or list(datasets)
    missing = [name for name in selected if name not in datasets]
    if missing:
        raise KeyError(f"Unknown earlyeval dataset(s): {', '.join(missing)}")
    return [(name, datasets[name] or {}) for name in selected]


def _read_parquet_columns(path: Path, columns: list[str]):
    import pyarrow.parquet as pq

    present = set(pq.ParquetFile(path).schema_arrow.names)
    keep = [col for col in columns if col in present]
    if not keep:
        raise ValueError(f"No requested columns are present in {path}")
    return pq.read_table(path, columns=keep).to_pandas()


def _stable_seed(seed: int, *parts: object) -> int:
    digest = hashlib.sha256("::".join([str(seed), *(str(part) for part in parts)]).encode("utf-8")).hexdigest()
    return int(digest[:12], 16)


def _markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> list[str]:
    if not rows:
        return []
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(":--" for _ in columns) + " |",
    ]
    for row in rows:
        values = [str(row.get(col, "")).replace("\n", " ") for col in columns]
        lines.append("| " + " | ".join(values) + " |")
    return lines


def build_execution_plan(config: str | Path = "configs/earlyeval.yaml", output_dir: str | Path | None = None) -> dict[str, Any]:
    cfg = load_earlyeval_config(config)
    out = ensure_dir(output_dir or _default_output_dir(cfg))
    plan = {
        "ok": True,
        "config": str(cfg.path),
        "output_dir": str(out),
        "run_id": out.name,
        "execution": "serial",
        "heavy_training_executed": False,
        "stages": [
            {"id": "00_preflight", "command": "python -m earlyeval.cli check preflight --experiment all"},
            {"id": "01_prefix_audit", "command": "python -m earlyeval.cli experiment paper-suite --stage audit-prefix"},
            {"id": "02_make_splits", "command": "python -m earlyeval.cli experiment paper-suite --stage make-splits"},
            {"id": "03_main_lightgbm", "status": "planned_heavy", "dataset_scope": "sweverify, toolathlon, terminalbench"},
            {"id": "04_lightgbm_valid_accuracy_sweep", "command": "python -m earlyeval.cli experiment paper-suite --stage lightgbm-policy-sweep"},
            {"id": "05_sweverify_ablation", "status": "planned_heavy", "dataset_scope": "sweverify only"},
            {"id": "06_model_compare", "status": "planned_heavy", "dataset_scope": "sweverify primary"},
            {"id": "07_latency", "status": "planned", "scope": "predictor methods on sweverify"},
            {"id": "08_paper_refresh", "status": "planned", "output": "paper/results/earlyeval_<run_id>"},
        ],
        "datasets": {
            name: {
                "role": spec.get("role"),
                "prefix_table": str(_resolve_project_path(spec["prefix_table"])),
                "split_enabled": bool(spec.get("split_enabled", True)),
                "ablations_enabled": bool(spec.get("ablations_enabled", False)),
                "answer_features_available": bool(spec.get("answer_features_available", False)),
            }
            for name, spec in _dataset_items(cfg)
        },
        "split": cfg.payload.get("split") or {},
        "ablation": cfg.payload.get("ablation") or {},
        "latency": cfg.payload.get("latency") or {},
    }
    write_json(out / "manifests" / "execution_plan.json", plan)
    return plan


def audit_prefix_tables(
    config: str | Path = "configs/earlyeval.yaml",
    output_dir: str | Path | None = None,
    datasets: list[str] | None = None,
) -> dict[str, Any]:
    import pandas as pd
    import pyarrow.parquet as pq

    cfg = load_earlyeval_config(config)
    out = ensure_dir(output_dir or _default_output_dir(cfg))
    checks_dir = ensure_dir(out / "checks")
    audit_cfg = cfg.payload.get("prefix_audit") or {}
    numeric_checks = list(audit_cfg.get("numeric_feature_checks") or [])
    action_cols = list(audit_cfg.get("action_columns") or [])
    summary_rows: list[dict[str, Any]] = []
    feature_rows: list[dict[str, Any]] = []
    action_rows: list[dict[str, Any]] = []
    schema_rows: list[dict[str, Any]] = []

    for name, spec in _dataset_items(cfg, datasets):
        path = _resolve_project_path(spec["prefix_table"])
        if not path.exists():
            summary_rows.append({"dataset": name, "ok": False, "path": str(path), "error": "missing prefix parquet"})
            continue

        pf = pq.ParquetFile(path)
        cols = pf.schema_arrow.names
        colset = set(cols)
        required = list(spec.get("required_columns") or [])
        missing = [col for col in required if col not in colset]
        step_col = str(spec.get("step_col", "prefix_step_idx"))
        traj_col = str(spec.get("traj_col", "traj_id"))
        model_col = str(spec.get("model_col", "model_id"))
        instance_col = str(spec.get("instance_col", "instance_id"))
        label_col = str(spec.get("label_col", "label"))
        light_cols = [
            step_col,
            traj_col,
            model_col,
            instance_col,
            label_col,
            "n_steps_total_for_weighting",
            *numeric_checks,
            *action_cols,
        ]
        frame = _read_parquet_columns(path, list(dict.fromkeys(light_cols)))
        traj_step_max = frame.groupby(traj_col, sort=False)[step_col].max()
        labels = frame.drop_duplicates(traj_col).groupby(label_col)[traj_col].count().to_dict() if label_col in frame else {}
        summary_rows.append(
            {
                "dataset": name,
                "ok": not missing,
                "role": spec.get("role"),
                "path": str(path),
                "rows": int(pf.metadata.num_rows),
                "row_groups": int(pf.metadata.num_row_groups),
                "columns": int(len(cols)),
                "missing_required_columns": ";".join(missing),
                "trajectories": int(frame[traj_col].nunique()),
                "instances": int(frame[instance_col].nunique()) if instance_col in frame else None,
                "models": int(frame[model_col].nunique()) if model_col in frame else None,
                "label_0_trajectories": int(labels.get(0, labels.get("0", 0))),
                "label_1_trajectories": int(labels.get(1, labels.get("1", 0))),
                "has_step0": bool((frame[step_col] == 0).any()),
                "step0_rows": int((frame[step_col] == 0).sum()),
                "min_step": int(frame[step_col].min()),
                "max_step": int(frame[step_col].max()),
                "zero_step_only_trajectories": int((traj_step_max == 0).sum()),
                "one_step_or_less_trajectories": int((traj_step_max <= 1).sum()),
                "p99_trajectory_steps": float(traj_step_max.quantile(0.99)),
                "ablations_enabled": bool(spec.get("ablations_enabled", False)),
                "answer_features_available": bool(spec.get("answer_features_available", False)),
            }
        )
        for col in required:
            schema_rows.append({"dataset": name, "column": col, "present": col in colset, "required": True})
        for col in numeric_checks:
            if col not in frame:
                feature_rows.append({"dataset": name, "column": col, "present": False})
                continue
            series = frame[col]
            numeric = pd.to_numeric(series, errors="coerce")
            non_null = int(series.notna().sum())
            non_zero = int((numeric.fillna(0) != 0).sum())
            feature_rows.append(
                {
                    "dataset": name,
                    "column": col,
                    "present": True,
                    "non_null_rows": non_null,
                    "non_null_pct": non_null * 100.0 / len(frame) if len(frame) else 0.0,
                    "non_zero_rows": non_zero,
                    "non_zero_pct": non_zero * 100.0 / len(frame) if len(frame) else 0.0,
                }
            )
        for col in action_cols:
            if col not in frame:
                action_rows.append({"dataset": name, "column": col, "value": "__MISSING_COLUMN__", "rows": 0})
                continue
            counts = frame[col].fillna("__NULL__").astype(str).value_counts(dropna=False).head(50)
            for value, count in counts.items():
                action_rows.append({"dataset": name, "column": col, "value": value, "rows": int(count)})

    summary = pd.DataFrame(summary_rows)
    features = pd.DataFrame(feature_rows)
    actions = pd.DataFrame(action_rows)
    schema = pd.DataFrame(schema_rows)
    write_table(summary, checks_dir / "prefix_audit_summary.csv")
    write_table(features, checks_dir / "prefix_audit_feature_nonempty.csv")
    write_table(actions, checks_dir / "prefix_audit_action_values.csv")
    write_table(schema, checks_dir / "prefix_audit_schema.csv")
    lines = [
        "# EarlyEval Prefix Audit",
        "",
        f"- config: `{cfg.path}`",
        f"- output: `{checks_dir}`",
        f"- datasets: `{', '.join(summary['dataset'].astype(str)) if not summary.empty else ''}`",
        f"- ok: `{bool(summary['ok'].all()) if not summary.empty and 'ok' in summary else False}`",
        "",
        "This audit uses parquet metadata and selected lightweight columns. It does not load full text columns.",
        "",
    ]
    if not summary.empty:
        display_cols = [
            "dataset",
            "ok",
            "rows",
            "trajectories",
            "models",
            "has_step0",
            "zero_step_only_trajectories",
            "one_step_or_less_trajectories",
            "p99_trajectory_steps",
        ]
        lines.extend(_markdown_table(summary.to_dict("records"), display_cols))
        lines.append("")
    (checks_dir / "prefix_audit_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {
        "ok": bool(summary["ok"].all()) if not summary.empty and "ok" in summary else False,
        "output_dir": str(checks_dir),
        "summary": str(checks_dir / "prefix_audit_summary.csv"),
        "features": str(checks_dir / "prefix_audit_feature_nonempty.csv"),
        "actions": str(checks_dir / "prefix_audit_action_values.csv"),
        "schema": str(checks_dir / "prefix_audit_schema.csv"),
    }


def make_split_manifests(
    config: str | Path = "configs/earlyeval.yaml",
    output_dir: str | Path | None = None,
    datasets: list[str] | None = None,
    max_folds: int | None = None,
) -> dict[str, Any]:
    """Materialize per-fold split manifests under `paper/experiments/<run_id>/splits`.

    The manifest is an *approximate, paper-facing description* of the split
    used by downstream training; it is not the actual split fed to the
    trainer.

    The strategy and seed are aligned with the legacy trainer:

    - test = all trajectories from the held-out `test_model`;
    - inside the remaining models, `valid_models_per_instance` distinct
      `(instance_id, model_id)` pairs per instance go to valid, everything
      else stays in train.

    Filters that this manifest applies on the prefix table:

    - excluded_train_models from `configs/earlyeval.yaml` (matches what
      `_lightgbm_command --exclude-train-models` does to the trainer).

    Filters the manifest does *not* apply (the trainer does, so the manifest
    counts are upper bounds rather than exact training counts):

    - the `verified_jsonl` instance allow-list (SWEVerify only);
    - `--max-instances` cap (typically 500 for SWEVerify);
    - `_drop_short_trajectories` (legacy `MIN_TRAJECTORY_STEPS = 5`).

    Because the trainer applies these reductions before the random pair
    selection, the *specific* `(instance, model)` pairs the trainer routes
    to valid may differ from the ones in this manifest even when seeds match.
    The shadow-validation property still holds: trajectories and
    `(instance, model)` pairs are disjoint between train and valid;
    instances overlap by design.
    """

    import numpy as np
    import pandas as pd

    cfg = load_earlyeval_config(config)
    out = ensure_dir(output_dir or _default_output_dir(cfg))
    split_dir = ensure_dir(out / "splits")
    split_cfg = cfg.payload.get("split") or {}
    valid_models_per_instance = int(split_cfg.get("valid_models_per_instance", 3))
    valid_ratio = float(split_cfg.get("valid_ratio", 0.15))
    min_trajs = _min_trajectories_per_model(cfg)
    exclude_models = _excluded_models_from_config(cfg)
    index_rows: list[dict[str, Any]] = []
    selected_datasets = datasets
    if selected_datasets is None:
        selected_datasets = [name for name, spec in _dataset_items(cfg) if bool(spec.get("split_enabled", True))]

    for dataset, spec in _dataset_items(cfg, selected_datasets):
        if not bool(spec.get("split_enabled", True)):
            continue
        path = _resolve_project_path(spec["prefix_table"])
        traj_col = str(spec.get("traj_col", "traj_id"))
        instance_col = str(spec.get("instance_col", "instance_id"))
        model_col = str(spec.get("model_col", "model_id"))
        label_col = str(spec.get("label_col", "label"))
        frame = _read_parquet_columns(path, [traj_col, instance_col, model_col, label_col])
        traj = frame.drop_duplicates(traj_col).copy()
        traj[instance_col] = traj[instance_col].astype(str)
        traj[model_col] = traj[model_col].astype(str)
        model_counts = traj.groupby(model_col)[traj_col].count().sort_values(ascending=False)
        eligible = [str(model) for model, count in model_counts.items() if int(count) >= min_trajs and str(model) not in exclude_models]
        if max_folds is not None:
            eligible = eligible[: int(max_folds)]
        dataset_dir = ensure_dir(split_dir / dataset)
        for test_model in eligible:
            test_mask = traj[model_col].eq(test_model)
            remaining_pool = traj.loc[~test_mask].copy()
            # Apply the configured exclude_train_models filter so the manifest
            # train-pool size is honest about the rows the trainer also drops.
            additional_excluded = sorted(exclude_models - {test_model})
            if additional_excluded:
                remaining = remaining_pool.loc[~remaining_pool[model_col].isin(set(additional_excluded))].copy()
            else:
                remaining = remaining_pool
            test = traj.loc[test_mask]

            # Mirror legacy `_select_valid_model_pairs_per_instance`: for each
            # instance in the remaining pool, pick `valid_models_per_instance`
            # distinct model_ids and route those (instance, model) pairs to
            # valid. Singleton-model instances stay entirely in train.
            rng = np.random.default_rng(int(cfg.seed) + 3571)
            unique_pairs = remaining[[instance_col, model_col]].drop_duplicates()
            valid_pairs: set[tuple[str, str]] = set()
            for instance_id, part in unique_pairs.groupby(instance_col, sort=False):
                model_ids = part[model_col].to_numpy()
                n_models = len(model_ids)
                if n_models <= 1:
                    continue
                n_valid = min(int(valid_models_per_instance), int(n_models) - 1)
                chosen = rng.choice(model_ids, size=n_valid, replace=False).tolist()
                valid_pairs.update((str(instance_id), str(model)) for model in chosen)

            pair_index = pd.MultiIndex.from_frame(remaining[[instance_col, model_col]])
            valid_mask = pair_index.isin(valid_pairs)
            valid = remaining.loc[valid_mask]
            train = remaining.loc[~valid_mask]

            train_models = set(train[model_col])
            valid_models = set(valid[model_col])
            train_instances = set(train[instance_col])
            valid_instances = set(valid[instance_col])
            shared_instances = int(len(train_instances & valid_instances))
            train_pairs = set(map(tuple, train[[instance_col, model_col]].to_numpy()))
            valid_pairs_actual = set(map(tuple, valid[[instance_col, model_col]].to_numpy()))
            train_traj_ids = set(train[traj_col].astype(str))
            valid_traj_ids = set(valid[traj_col].astype(str))

            fold_id = test_model.replace("/", "_").replace(" ", "_")
            manifest = {
                "dataset": dataset,
                "fold_id": fold_id,
                "test_model": test_model,
                "strategy": str(split_cfg.get("strategy", "leave_one_test_model_known_task")),
                "trainval_split_strategy": "per_instance_model",
                "seed": int(cfg.seed),
                "valid_models_per_instance": int(valid_models_per_instance),
                "valid_ratio_planning_only": valid_ratio,
                "known_task_instance_overlap_allowed": True,
                "manifest_is_approximate": True,
                "filters_applied_in_manifest": ["excluded_train_models"],
                "filters_skipped_in_manifest": [
                    "verified_jsonl_instance_allow_list",
                    "max_instances_cap",
                    "_drop_short_trajectories_min_5",
                ],
                "additional_excluded_models": list(additional_excluded),
                "test_model_absent_from_train_valid": bool(test_model not in train_models | valid_models),
                "train_valid_traj_disjoint": bool(train_traj_ids.isdisjoint(valid_traj_ids)),
                "train_valid_instance_pair_disjoint": bool(train_pairs.isdisjoint(valid_pairs_actual)),
                "train_valid_shared_instance_count": shared_instances,
                "train_valid_shared_instance_pct": (
                    100.0 * shared_instances / len(train_instances | valid_instances)
                    if (train_instances | valid_instances)
                    else 0.0
                ),
                "test_kept_unfiltered": True,
                "train_valid_short_filter_allowed": bool(
                    (split_cfg.get("train_valid_filter") or {}).get("allow_short_trajectory_filter", True)
                ),
                "do_not_filter_longest_percent": bool(split_cfg.get("do_not_filter_longest_percent", True)),
                "counts": {
                    "train_trajectories": int(train[traj_col].nunique()),
                    "valid_trajectories": int(valid[traj_col].nunique()),
                    "test_trajectories": int(test[traj_col].nunique()),
                    "train_instances": int(len(train_instances)),
                    "valid_instances": int(len(valid_instances)),
                    "test_instances": int(test[instance_col].nunique()),
                    "train_models": int(len(train_models)),
                    "valid_models": int(len(valid_models)),
                    "test_models": int(test[model_col].nunique()),
                    "valid_instance_model_pairs": int(len(valid_pairs_actual)),
                    "train_instance_model_pairs": int(len(train_pairs)),
                },
                "notes": [
                    "trainval_split_strategy=per_instance_model: validation contains valid_models_per_instance (instance, model) pairs per instance; the rest of (instance, model) pairs go to train.",
                    "Trajectories and (instance, model) pairs are disjoint between train and valid.",
                    "Instances overlap between train and valid by design (shadow validation).",
                    "Test trajectories come exclusively from the held-out test_model; test instances may overlap with train/valid instances under the known-task setting.",
                ],
            }
            manifest_path = dataset_dir / fold_id / "split_manifest.json"
            write_json(manifest_path, manifest)
            index_rows.append(
                {
                    "dataset": dataset,
                    "fold_id": fold_id,
                    "test_model": test_model,
                    "manifest": str(manifest_path),
                    **manifest["counts"],
                    "test_model_absent_from_train_valid": manifest["test_model_absent_from_train_valid"],
                    "train_valid_traj_disjoint": manifest["train_valid_traj_disjoint"],
                    "train_valid_instance_pair_disjoint": manifest["train_valid_instance_pair_disjoint"],
                    "train_valid_shared_instance_count": manifest["train_valid_shared_instance_count"],
                    "train_valid_shared_instance_pct": manifest["train_valid_shared_instance_pct"],
                }
            )
    index = pd.DataFrame(index_rows)
    write_table(index, split_dir / "split_index.csv")
    write_json(split_dir / "split_summary.json", {"ok": True, "folds": int(len(index)), "index": str(split_dir / "split_index.csv")})
    return {"ok": True, "folds": int(len(index)), "output_dir": str(split_dir), "index": str(split_dir / "split_index.csv")}


def _safe_fold_name(name: str) -> str:
    keep = []
    for char in name:
        keep.append(char if char.isalnum() or char in {"-", "_", "."} else "_")
    return "".join(keep).strip("_") or "fold"


def _answer_module_root() -> Path:
    return load_paths().vendor_answer_module_root


def _shared_feature_engineer_path() -> Path:
    return load_paths().feature_engineer_with_model


def _trainer_path(cfg: EarlyEvalConfig) -> Path:
    return _answer_module_root() / "safe_stop_dual_head_retrain.py"


def _legacy_trainer_path(cfg: EarlyEvalConfig) -> Path:
    """Backward-compatible alias for older earlyeval experiment helpers."""

    return _trainer_path(cfg)


def _excluded_models_from_config(cfg: EarlyEvalConfig) -> set[str]:
    split_cfg = cfg.payload.get("split") or {}
    excluded = set(str(v) for v in split_cfg.get("exclude_models", []) or [])
    excluded.update(str(v) for v in (split_cfg.get("excluded_model_reasons", {}) or {}).keys())
    return excluded


# Default for `min_trajectories_per_model` when yaml omits it. Both
# `_eligible_lightgbm_folds` (used by training) and `make_split_manifests`
# (used by paper documentation) must agree on this number, otherwise the
# eligible fold list silently differs between training and the manifest.
_DEFAULT_MIN_TRAJECTORIES_PER_MODEL = 100


def _min_trajectories_per_model(cfg: EarlyEvalConfig) -> int:
    split_cfg = cfg.payload.get("split") or {}
    return int(split_cfg.get("min_trajectories_per_model", _DEFAULT_MIN_TRAJECTORIES_PER_MODEL))


# Default for `safe_label_min_step` when neither the yaml `main_model`
# section nor the CLI provide one. Mirrors the legacy trainer default.
_DEFAULT_SAFE_LABEL_MIN_STEP = 10


def _safe_label_min_step(cfg: EarlyEvalConfig) -> int:
    main = cfg.payload.get("main_model") or {}
    return int(main.get("safe_label_min_step", _DEFAULT_SAFE_LABEL_MIN_STEP))


def _fit_feature_engineer_on_train(cfg: EarlyEvalConfig) -> bool:
    """Whether to fit a fresh FeatureEngineer per fold (strict no-leak) or
    reuse the global pre-fit pickle (fast, transductive).

    Default is False (shared pkl) because per-fold fit adds 90-150 min of
    single-threaded work per fold and serializes that phase across parallel
    folds via the RAM-peak lock, making the main run 5-10x slower for a
    leakage that is mild and identical across LightGBM main and every
    baseline. Set to True only when verifying paper-headline numbers under
    strict per-fold fit.
    """

    main = cfg.payload.get("main_model") or {}
    return bool(main.get("feature_engineer_fit_on_train", False))


def _eligible_lightgbm_folds(cfg: EarlyEvalConfig, dataset: str = "sweverify") -> list[dict[str, Any]]:
    import pyarrow.parquet as pq

    datasets = cfg.payload.get("datasets") or {}
    spec = datasets[dataset]
    split_cfg = cfg.payload.get("split") or {}
    min_trajs = _min_trajectories_per_model(cfg)
    excluded = _excluded_models_from_config(cfg)
    path = _resolve_project_path(spec["prefix_table"])
    model_col = str(spec.get("model_col", "model_id"))
    traj_col = str(spec.get("traj_col", "traj_id"))
    step_col = str(spec.get("step_col", "prefix_step_idx"))
    frame = pq.read_table(path, columns=[traj_col, model_col, step_col]).to_pandas()
    traj = frame.groupby([model_col, traj_col], sort=False)[step_col].max().reset_index()
    counts = traj.groupby(model_col)[traj_col].count().sort_values(ascending=False)
    max_steps = traj.groupby(model_col)[step_col].max()
    rows = []
    for model, count in counts.items():
        model = str(model)
        reason = ""
        eligible = True
        if model in excluded:
            eligible = False
            reason = str((split_cfg.get("excluded_model_reasons", {}) or {}).get(model, "configured exclusion"))
        elif int(count) < min_trajs:
            eligible = False
            reason = f"below min_trajectories_per_model={min_trajs}"
        rows.append(
            {
                "test_model": model,
                "fold_id": _safe_fold_name(model),
                "trajectories": int(count),
                "max_prefix_step": int(max_steps.loc[model]),
                "eligible": eligible,
                "exclusion_reason": reason,
            }
        )
    return rows


def _ram_peak_lock_path(run_dir: Path) -> Path:
    """Shared RAM-peak lock path for every fold under a single run.

    The legacy trainer takes `--ram-peak-lock-path` and grabs an exclusive
    fcntl lock around the prefix-table load + FeatureEngineer.fit phase.
    Pointing every parallel fold of a given run at the same lock file
    guarantees that only one fold holds text columns in RAM at a time even
    when MAX_PARALLEL_FOLDS > 1.
    """

    return run_dir / "ram_peak.lock"


def _python_executable(cfg: EarlyEvalConfig) -> Path:
    paths = load_paths()
    value = paths.python_executable or Path(str((cfg.payload.get("runtime") or {}).get("python_executable") or sys.executable))
    return value if value.is_absolute() else _resolve_project_path(value)


def _lightgbm_command(cfg: EarlyEvalConfig, *, test_model: str, fold_output_dir: Path) -> list[str]:
    dataset = (cfg.payload.get("datasets") or {})["sweverify"]
    main = cfg.payload.get("main_model") or {}
    resources = cfg.payload.get("resources") or {}
    excluded_train_models = sorted(_excluded_models_from_config(cfg) - {test_model})
    command = [
        str(_python_executable(cfg)),
        str(_legacy_trainer_path(cfg)),
        "--run-name",
        str(main.get("run_name", "model_holdout_answer_calibrated_full")),
        "--prefix-table",
        str(_resolve_project_path(dataset["prefix_table"])),
        "--verified-jsonl",
        str(_resolve_project_path(dataset["verified_jsonl"])),
        "--feature-engineer-path",
        str(_shared_feature_engineer_path()),
        "--holdout-models",
        test_model,
        "--max-instances",
        "500",
        "--split-strategy",
        "per_instance_model",
        "--seed",
        str(cfg.seed),
        "--output-subdir",
        str(fold_output_dir.resolve()),
        "--variants",
        *[str(v) for v in main.get("variants", ["i"])],
        "--lgbm-preset",
        str(main.get("lgbm_preset", "strong_reg")),
        "--safe-label-min-step",
        str(_safe_label_min_step(cfg)),
        "--policy-min-steps",
        *[str(v) for v in main.get("policy_min_steps", [0])],
        "--consecutive",
        *[str(v) for v in main.get("consecutive", [1])],
        "--success-thresholds",
        *[str(v) for v in main.get("success_thresholds", [0.95])],
        "--failure-thresholds",
        *[str(v) for v in main.get("failure_thresholds", [0.95])],
        "--score-modes",
        *[str(v) for v in main.get("score_modes", [main.get("score_mode", "calibrated")])],
        "--max-cpu-threads",
        str(
            int(
                os.environ.get(
                    "EARLYEVAL_LGBM_THREADS_PER_FOLD",
                    str(
                        resources.get(
                            "lightgbm_threads_per_fold",
                            resources.get(
                                "lightgbm_num_threads",
                                resources.get("max_cpu_threads", 8),
                            ),
                        )
                    ),
                )
            )
        ),
        "--low-memory",
    ]
    if bool(main.get("mask_train_model_id", True)):
        command.append("--mask-train-model-id")
    if excluded_train_models:
        command.append("--exclude-train-models")
        command.extend(excluded_train_models)
    if _fit_feature_engineer_on_train(cfg):
        # Strict no-leak path: fit a fresh FeatureEngineer per fold and
        # serialize the high-RAM phase across parallel folds via a shared
        # lock. Only enabled when explicitly requested in the yaml because
        # it adds 90-150 min of single-threaded fit work per fold.
        command.append("--fit-feature-engineer-on-train")
        command.extend(
            [
                "--ram-peak-lock-path",
                str((fold_output_dir.parent.parent / "ram_peak.lock").resolve()),
            ]
        )
    return command


def _run_fold_subprocess(row: dict[str, Any], command: list[str]) -> tuple[dict[str, Any], int]:
    """Run a single fold subprocess and write its log + success marker.

    Returns the original row metadata and the process return code so callers
    can decide whether to mark it completed or failed without sharing mutable
    state across worker threads.
    """

    fold_dir = Path(row["output_dir"])
    log_path = Path(row["log"])
    fold_dir.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        log.write("COMMAND:\n")
        log.write(" ".join(command) + "\n\n")
        log.flush()
        proc = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
    if proc.returncode == 0:
        (fold_dir / "_SUCCESS").write_text("completed\n", encoding="utf-8")
    return row, int(proc.returncode)


def run_lightgbm_main(
    config: str | Path = "configs/earlyeval.yaml",
    output_dir: str | Path | None = None,
    *,
    execute: bool = False,
    force: bool = False,
    fold_limit: int | None = None,
    max_parallel_folds: int = 1,
) -> dict[str, Any]:
    import pandas as pd

    cfg = load_earlyeval_config(config)
    out = ensure_dir(output_dir or _default_output_dir(cfg, cfg.run_id))
    run_dir = ensure_dir(out / "lightgbm_main")
    logs_dir = ensure_dir(run_dir / "logs")
    all_folds = _eligible_lightgbm_folds(cfg, dataset="sweverify")
    eligible = [row for row in all_folds if row["eligible"]]
    if fold_limit is not None:
        eligible = eligible[: int(fold_limit)]
    rows = []
    commands: list[list[str]] = []
    for order, fold in enumerate(eligible, start=1):
        fold_dir = run_dir / "folds" / fold["fold_id"]
        command = _lightgbm_command(cfg, test_model=fold["test_model"], fold_output_dir=fold_dir)
        commands.append(command)
        marker = fold_dir / "safe_stop_test_selected.csv"
        status = "pending"
        if marker.exists() and not force:
            status = "skipped_existing"
        rows.append(
            {
                "order": order,
                "fold_id": fold["fold_id"],
                "test_model": fold["test_model"],
                "trajectories": fold["trajectories"],
                "output_dir": str(fold_dir),
                "log": str(logs_dir / f"{fold['fold_id']}.log"),
                "status": status,
                "command": " ".join(command),
            }
        )
    command_index = pd.DataFrame(rows)
    write_table(pd.DataFrame(all_folds), run_dir / "fold_eligibility.csv")
    write_table(command_index, run_dir / "command_index.csv")
    parallelism = max(1, int(max_parallel_folds))
    write_json(
        run_dir / "run_manifest.json",
        {
            "config": str(cfg.path),
            "output_dir": str(run_dir),
            "eligible_folds": len(eligible),
            "execute": bool(execute),
            "force": bool(force),
            "fold_limit": fold_limit,
            "max_parallel_folds": parallelism,
            "heavy_training": True,
            "serial": parallelism == 1,
        },
    )
    if not execute:
        return {
            "ok": True,
            "execute": False,
            "eligible_folds": len(eligible),
            "output_dir": str(run_dir),
            "command_index": str(run_dir / "command_index.csv"),
            "fold_eligibility": str(run_dir / "fold_eligibility.csv"),
            "max_parallel_folds": parallelism,
            "note": "Dry-run only. Add --execute to run LightGBM folds.",
        }

    skipped = 0
    pending: list[tuple[dict[str, Any], list[str]]] = []
    for row, command in zip(rows, commands):
        marker = Path(row["output_dir"]) / "safe_stop_test_selected.csv"
        if marker.exists() and not force:
            skipped += 1
            continue
        pending.append((row, command))

    completed = 0
    failed: list[dict[str, Any]] = []

    def _record_result(row: dict[str, Any], rc: int) -> None:
        nonlocal completed
        if rc == 0:
            completed += 1
            return
        failed.append(
            {
                "fold_id": row["fold_id"],
                "test_model": row["test_model"],
                "returncode": rc,
                "log": str(row["log"]),
            }
        )
        print(
            f"[lightgbm-main] fold {row['fold_id']} failed (rc={rc}); "
            f"continuing with the next fold. See log: {row['log']}",
            flush=True,
        )

    if parallelism <= 1:
        for row, command in pending:
            row_back, rc = _run_fold_subprocess(row, command)
            _record_result(row_back, rc)
    else:
        from concurrent.futures import ThreadPoolExecutor, as_completed

        print(
            f"[lightgbm-main] running {len(pending)} fold(s) with parallelism={parallelism}.",
            flush=True,
        )
        with ThreadPoolExecutor(max_workers=parallelism) as executor:
            future_to_fold = {
                executor.submit(_run_fold_subprocess, row, command): row["fold_id"]
                for row, command in pending
            }
            for future in as_completed(future_to_fold):
                row_back, rc = future.result()
                _record_result(row_back, rc)
    summary = {
        "ok": not failed,
        "execute": True,
        "completed": completed,
        "skipped": skipped,
        "failed": failed,
        "max_parallel_folds": parallelism,
        "output_dir": str(run_dir),
        "command_index": str(run_dir / "command_index.csv"),
    }
    write_json(run_dir / "execution_summary.json", summary)
    return summary


def _read_completed_fold_tables(run_dir: Path, *, excluded_models: set[str] | None = None) -> dict[str, Any]:
    import pandas as pd

    folds_dir = run_dir / "folds"
    excluded_models = excluded_models or set()
    all_completed = sorted(path.parent for path in folds_dir.glob("*/_SUCCESS"))
    completed = [fold_dir for fold_dir in all_completed if fold_dir.name not in excluded_models]
    skipped_completed = [fold_dir for fold_dir in all_completed if fold_dir.name in excluded_models]
    test_rows = []
    valid_rows = []
    calibration_rows = []
    for fold_dir in completed:
        fold_id = fold_dir.name
        test_path = fold_dir / "safe_stop_test_selected.csv"
        valid_path = fold_dir / "safe_stop_selected_policies.csv"
        calibration_path = fold_dir / "safe_stop_calibration_summary.csv"
        if test_path.exists():
            frame = pd.read_csv(test_path)
            frame.insert(0, "fold_id", fold_id)
            frame.insert(1, "test_model", fold_id)
            test_rows.append(frame)
        if valid_path.exists():
            frame = pd.read_csv(valid_path)
            frame.insert(0, "fold_id", fold_id)
            frame.insert(1, "test_model", fold_id)
            valid_rows.append(frame)
        if calibration_path.exists():
            frame = pd.read_csv(calibration_path)
            frame.insert(0, "fold_id", fold_id)
            frame.insert(1, "test_model", fold_id)
            calibration_rows.append(frame)
    return {
        "completed_folds": completed,
        "skipped_completed_folds": skipped_completed,
        "test": pd.concat(test_rows, ignore_index=True) if test_rows else pd.DataFrame(),
        "valid": pd.concat(valid_rows, ignore_index=True) if valid_rows else pd.DataFrame(),
        "calibration": pd.concat(calibration_rows, ignore_index=True) if calibration_rows else pd.DataFrame(),
    }


def _aggregate_selected_policy(frame) -> dict[str, Any]:
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
        "folds": int(frame["fold_id"].nunique()),
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


def _write_lightgbm_summary_plots(summary_dir: Path, test_frame) -> list[str]:
    if test_frame.empty:
        return []
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return []

    plot_paths: list[str] = []
    plot_frame = test_frame.copy()
    plot_frame["step_save_pct_plot"] = plot_frame["pct_steps_saved"].astype(float)
    plot_frame["decision_accuracy_pct_plot"] = plot_frame["decision_accuracy"].astype(float) * 100.0
    plot_frame["resolve_rate_change_pp_plot"] = -plot_frame["resolve_rate_drop"].astype(float) * 100.0
    plot_frame = plot_frame.sort_values("step_save_pct_plot", ascending=True)

    specs = [
        ("step_save_pct_plot", "Step Saving (%)", "per_model_step_saving.png"),
        ("decision_accuracy_pct_plot", "Decision Accuracy (%)", "per_model_decision_accuracy.png"),
        ("resolve_rate_change_pp_plot", "Resolve-Rate Change (pp)", "per_model_resolve_change.png"),
    ]
    for column, title, filename in specs:
        fig, ax = plt.subplots(figsize=(9, max(4, 0.36 * len(plot_frame))))
        ax.barh(plot_frame["fold_id"].astype(str), plot_frame[column].astype(float), color="#4c78a8")
        ax.set_xlabel(title)
        ax.set_ylabel("Held-out test model")
        ax.set_title(title + " by Fold")
        ax.grid(axis="x", alpha=0.25)
        fig.tight_layout()
        out = summary_dir / filename
        fig.savefig(out, dpi=160)
        plt.close(fig)
        plot_paths.append(str(out))
    return plot_paths


def _float_sequence(raw: Any, default: Any) -> list[float]:
    raw = default if raw is None else raw
    if isinstance(raw, dict):
        start = float(raw["start"])
        stop = float(raw["stop"])
        step = float(raw.get("step", 0.01))
        if step <= 0:
            raise ValueError(f"Range step must be positive: {raw}")
        values = []
        current = start
        while current <= stop + (step / 10.0):
            values.append(round(float(current), 6))
            current += step
        return sorted(set(values))
    return sorted({round(float(value), 6) for value in raw})


def _format_prob(value: float) -> str:
    if math.isinf(float(value)):
        return "inf"
    return f"{float(value):.3f}".rstrip("0").rstrip(".")


def _target_slug(value: float) -> str:
    return f"{int(round(float(value) * 100)):03d}"


def _infer_prefix_models(frame, score_modes: list[str]) -> list[str]:
    models: set[str] = set()
    for score_mode in score_modes:
        success_prefix = "prob_cal_safe_success__" if score_mode == "calibrated" else "prob_safe_success__"
        failure_prefix = "prob_cal_safe_failure__" if score_mode == "calibrated" else "prob_safe_failure__"
        for column in frame.columns:
            if column.startswith(success_prefix):
                name = column[len(success_prefix) :]
                if f"{failure_prefix}{name}" in frame.columns:
                    models.add(name)
    if not models:
        raise ValueError("Could not infer any prefix model probability columns from the prediction table.")
    return sorted(models)


def _build_policy_sweep_candidates(sweep_cfg: dict[str, Any], main_cfg: dict[str, Any], prefix_models: list[str]):
    from earlyeval.core.contracts import PolicySpec

    probability_thresholds = _float_sequence(
        sweep_cfg.get("candidate_probability_thresholds"),
        {"start": 0.50, "stop": 0.99, "step": 0.01},
    )
    success_thresholds = _float_sequence(sweep_cfg.get("candidate_success_thresholds"), probability_thresholds)
    failure_thresholds = _float_sequence(sweep_cfg.get("candidate_failure_thresholds"), probability_thresholds)
    policy_modes = [str(value) for value in sweep_cfg.get("policy_modes", ["dual"])]
    min_steps = [int(value) for value in sweep_cfg.get("policy_min_steps", main_cfg.get("policy_min_steps", [0]))]
    consecutive_values = [int(value) for value in sweep_cfg.get("consecutive", main_cfg.get("consecutive", [1]))]
    score_modes = [str(value) for value in sweep_cfg.get("score_modes", main_cfg.get("score_modes", [main_cfg.get("score_mode", "calibrated")]))]
    dual_threshold_mode = str(sweep_cfg.get("dual_threshold_mode", "symmetric"))

    candidates = []
    for prefix_model in prefix_models:
        for score_mode in score_modes:
            for min_step in min_steps:
                for consecutive in consecutive_values:
                    if "dual" in policy_modes:
                        if dual_threshold_mode == "cartesian":
                            pairs = [(success_thr, failure_thr) for success_thr in success_thresholds for failure_thr in failure_thresholds]
                        elif dual_threshold_mode == "symmetric":
                            pairs = [(thr, thr) for thr in probability_thresholds]
                        else:
                            raise ValueError(f"Unsupported dual_threshold_mode: {dual_threshold_mode}")
                        for success_thr, failure_thr in pairs:
                            name = (
                                f"{score_mode}__{prefix_model}__dual__"
                                f"s{_format_prob(success_thr)}__f{_format_prob(failure_thr)}__"
                                f"min{min_step}__k{consecutive}"
                            )
                            candidates.append(
                                PolicySpec(
                                    name=name,
                                    predictor=prefix_model,
                                    score_mode=score_mode,
                                    policy_mode="dual",
                                    success_thr=float(success_thr),
                                    failure_thr=float(failure_thr),
                                    min_step=min_step,
                                    consecutive=consecutive,
                                )
                            )
                    if "success_only" in policy_modes:
                        for success_thr in success_thresholds:
                            name = (
                                f"{score_mode}__{prefix_model}__success_only__"
                                f"s{_format_prob(success_thr)}__finf__min{min_step}__k{consecutive}"
                            )
                            candidates.append(
                                PolicySpec(
                                    name=name,
                                    predictor=prefix_model,
                                    score_mode=score_mode,
                                    policy_mode="success_only",
                                    success_thr=float(success_thr),
                                    failure_thr=float("inf"),
                                    min_step=min_step,
                                    consecutive=consecutive,
                                )
                            )
                    if "failure_only" in policy_modes:
                        for failure_thr in failure_thresholds:
                            name = (
                                f"{score_mode}__{prefix_model}__failure_only__"
                                f"sinf__f{_format_prob(failure_thr)}__min{min_step}__k{consecutive}"
                            )
                            candidates.append(
                                PolicySpec(
                                    name=name,
                                    predictor=prefix_model,
                                    score_mode=score_mode,
                                    policy_mode="failure_only",
                                    success_thr=float("inf"),
                                    failure_thr=float(failure_thr),
                                    min_step=min_step,
                                    consecutive=consecutive,
                                )
                            )
    return candidates


def _policy_from_row(row: dict[str, Any]):
    from earlyeval.core.contracts import PolicySpec

    return PolicySpec(
        name=str(row["policy_name"]),
        predictor=str(row["predictor"]),
        score_mode=str(row["score_mode"]),
        policy_mode=str(row["policy_mode"]),
        success_thr=float(row["success_thr"]),
        failure_thr=float(row["failure_thr"]),
        min_step=int(row["min_step"]),
        consecutive=int(row["consecutive"]),
    )


def _evaluate_policy_candidates(frame, policies, *, fold_id: str, test_model: str):
    from earlyeval.policies.safe_stop import apply_policy

    rows = []
    for policy in policies:
        _, summary, _ = apply_policy(frame, policy)
        row = summary.iloc[0].to_dict()
        row["fold_id"] = fold_id
        row["test_model"] = test_model
        row["policy_id"] = policy.name
        row["valid_abs_drop_pp"] = abs(float(row.get("resolve_rate_drop_pp", 0.0)))
        row["decision_accuracy_fraction"] = float(row.get("decision_accuracy_pct", 0.0)) / 100.0
        rows.append(row)
    return rows


def _select_policy_for_valid_target(valid_grid, target_accuracy: float, *, max_valid_abs_drop_pp: float | None, fallback_min_save_pct: float) -> dict[str, Any]:
    work = valid_grid.copy()
    work["valid_abs_drop_pp"] = work["resolve_rate_drop_pp"].astype(float).abs()
    work["decision_accuracy_fraction"] = work["decision_accuracy_pct"].fillna(-1.0).astype(float) / 100.0
    work["pct_steps_saved_for_sort"] = work["pct_steps_saved"].fillna(0.0).astype(float)
    strict_mask = (work["decision_accuracy_fraction"] >= float(target_accuracy)) & (work["pct_steps_saved_for_sort"] > 0.0)
    if max_valid_abs_drop_pp is not None:
        strict_mask = strict_mask & (work["valid_abs_drop_pp"] <= float(max_valid_abs_drop_pp))
    strict = work[strict_mask].copy()
    if not strict.empty:
        chosen = strict.sort_values(
            ["pct_steps_saved_for_sort", "valid_abs_drop_pp", "decision_accuracy_fraction"],
            ascending=[False, True, False],
        ).iloc[0]
        status = "valid_accuracy_and_drop_pass" if max_valid_abs_drop_pp is not None else "valid_accuracy_pass"
    else:
        fallback = work[
            (work["decision_accuracy_fraction"] >= float(target_accuracy))
            & (work["pct_steps_saved_for_sort"] >= float(fallback_min_save_pct))
        ].copy()
        if not fallback.empty:
            chosen = fallback.sort_values(
                ["pct_steps_saved_for_sort", "valid_abs_drop_pp", "decision_accuracy_fraction"],
                ascending=[False, True, False],
            ).iloc[0]
            status = "fallback_valid_accuracy_only"
        else:
            fallback = work[work["pct_steps_saved_for_sort"] > 0.0].copy()
            if fallback.empty:
                fallback = work.copy()
            chosen = fallback.sort_values(
                ["decision_accuracy_fraction", "valid_abs_drop_pp", "pct_steps_saved_for_sort"],
                ascending=[False, True, False],
            ).iloc[0]
            status = "fallback_highest_valid_accuracy"
    row = chosen.to_dict()
    row["target_valid_decision_accuracy"] = float(target_accuracy)
    row["target_valid_decision_accuracy_pct"] = float(target_accuracy) * 100.0
    row["selection_status"] = status
    return row


def _aggregate_policy_sweep_by_target(frame):
    import pandas as pd

    rows = []
    if frame.empty:
        return pd.DataFrame()
    for target, part in frame.groupby("target_valid_decision_accuracy", sort=True):
        aggregate = _aggregate_selected_policy(part)
        aggregate["target_valid_decision_accuracy"] = float(target)
        aggregate["target_valid_decision_accuracy_pct"] = float(target) * 100.0
        rows.append(aggregate)
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows).sort_values("target_valid_decision_accuracy")
    leading = ["target_valid_decision_accuracy", "target_valid_decision_accuracy_pct"]
    return out[leading + [col for col in out.columns if col not in leading]]


def _write_lightgbm_policy_sweep_plots(sweep_dir: Path, aggregate_frame, per_fold_frame) -> list[str]:
    if aggregate_frame.empty:
        return []
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return []

    plot_paths: list[str] = []
    x = aggregate_frame["target_valid_decision_accuracy_pct"].astype(float)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(x, aggregate_frame["step_save_pct"].astype(float), marker="o", label="Test step saving")
    ax.plot(x, aggregate_frame["decision_accuracy_pct"].astype(float), marker="o", label="Test decision accuracy")
    ax.set_xlabel("Target valid decision accuracy (%)")
    ax.set_ylabel("Percent")
    ax.set_title("Test Frontier Selected On Valid")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    out = sweep_dir / "aggregate_test_frontier.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    plot_paths.append(str(out))

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(x, aggregate_frame["resolve_rate_change_pp"].astype(float), marker="o", color="#f58518")
    ax.axhline(0.0, color="#555555", linewidth=1)
    ax.set_xlabel("Target valid decision accuracy (%)")
    ax.set_ylabel("Resolve-rate change (pp)")
    ax.set_title("Test Resolve-Rate Change Selected On Valid")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    out = sweep_dir / "aggregate_test_resolve_change.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    plot_paths.append(str(out))

    if not per_fold_frame.empty:
        for metric, label, filename in [
            ("pct_steps_saved", "Test step saving (%)", "per_fold_test_step_saving.png"),
            ("decision_accuracy_pct", "Test decision accuracy (%)", "per_fold_test_decision_accuracy.png"),
            ("resolve_rate_change_pp", "Test resolve-rate change (pp)", "per_fold_test_resolve_change.png"),
        ]:
            fig, ax = plt.subplots(figsize=(9, 5))
            for fold_id, part in per_fold_frame.sort_values("target_valid_decision_accuracy").groupby("fold_id", sort=True):
                ax.plot(
                    part["target_valid_decision_accuracy_pct"].astype(float),
                    part[metric].astype(float),
                    marker="o",
                    linewidth=1.2,
                    alpha=0.75,
                    label=str(fold_id),
                )
            ax.set_xlabel("Target valid decision accuracy (%)")
            ax.set_ylabel(label)
            ax.set_title(label + " By Held-Out Model")
            ax.grid(alpha=0.25)
            if metric == "resolve_rate_change_pp":
                ax.axhline(0.0, color="#555555", linewidth=1)
            ax.legend(fontsize=7, loc="best")
            fig.tight_layout()
            out = sweep_dir / filename
            fig.savefig(out, dpi=160)
            plt.close(fig)
            plot_paths.append(str(out))
    return plot_paths


def run_lightgbm_policy_sweep(
    config: str | Path = "configs/earlyeval.yaml",
    output_dir: str | Path | None = None,
    *,
    fold_limit: int | None = None,
) -> dict[str, Any]:
    import pandas as pd
    from earlyeval.policies.safe_stop import apply_policy

    cfg = load_earlyeval_config(config)
    out = ensure_dir(output_dir or _default_output_dir(cfg, cfg.run_id))
    run_dir = out / "lightgbm_main"
    if not run_dir.exists():
        raise FileNotFoundError(f"LightGBM run directory does not exist: {run_dir}")
    excluded_models = _excluded_models_from_config(cfg)
    all_completed = sorted(path.parent for path in (run_dir / "folds").glob("*/_SUCCESS"))
    skipped_completed = [fold_dir for fold_dir in all_completed if fold_dir.name in excluded_models]
    completed = [fold_dir for fold_dir in all_completed if fold_dir.name not in excluded_models]
    if fold_limit is not None:
        completed = completed[: int(fold_limit)]
    if not completed:
        raise FileNotFoundError(f"No completed LightGBM folds found under {run_dir / 'folds'}")

    main_cfg = cfg.payload.get("main_model") or {}
    sweep_cfg = cfg.payload.get("policy_sweep") or {}
    targets = _float_sequence(
        sweep_cfg.get("target_valid_decision_accuracy"),
        {"start": 0.75, "stop": 0.95, "step": 0.01},
    )
    if not targets:
        raise ValueError("policy_sweep.target_valid_decision_accuracy is empty")
    first_slug = _target_slug(targets[0])
    last_slug = _target_slug(targets[-1])
    sweep_name = str(sweep_cfg.get("output_subdir") or f"valid_accuracy_{first_slug}_{last_slug}")
    sweep_dir = ensure_dir(run_dir / "policy_sweeps" / sweep_name)
    max_valid_abs_drop_pp_raw = sweep_cfg.get("max_valid_abs_drop_pp", None)
    max_valid_abs_drop_pp = None if max_valid_abs_drop_pp_raw is None else float(max_valid_abs_drop_pp_raw)
    fallback_min_save_pct = float(sweep_cfg.get("fallback_min_save_pct", 0.0))
    write_decisions = bool(sweep_cfg.get("write_decisions", True))

    valid_grid_rows: list[dict[str, Any]] = []
    selected_rows: list[dict[str, Any]] = []
    valid_summary_rows: list[dict[str, Any]] = []
    test_summary_rows: list[dict[str, Any]] = []
    valid_decision_rows = []
    test_decision_rows = []

    for fold_index, fold_dir in enumerate(completed, start=1):
        fold_id = fold_dir.name
        print(f"[lightgbm-policy-sweep] fold {fold_index}/{len(completed)}: {fold_id}", flush=True)
        valid_path = fold_dir / "valid_predictions_safe_stop.parquet"
        test_path = fold_dir / "test_predictions_safe_stop.parquet"
        if not valid_path.exists() or not test_path.exists():
            raise FileNotFoundError(f"Fold is missing raw prediction parquet files: {fold_dir}")
        valid_frame = pd.read_parquet(valid_path)
        test_frame = pd.read_parquet(test_path)
        configured_models = sweep_cfg.get("prefix_models") or sweep_cfg.get("predictors")
        score_modes = [str(value) for value in sweep_cfg.get("score_modes", main_cfg.get("score_modes", [main_cfg.get("score_mode", "calibrated")]))]
        if isinstance(configured_models, str):
            prefix_models = [configured_models]
        elif configured_models:
            prefix_models = [str(value) for value in configured_models]
        else:
            prefix_models = _infer_prefix_models(valid_frame, score_modes)
        policies = _build_policy_sweep_candidates(sweep_cfg, main_cfg, prefix_models)
        fold_valid_grid = pd.DataFrame(
            _evaluate_policy_candidates(valid_frame, policies, fold_id=fold_id, test_model=fold_id)
        )
        valid_grid_rows.extend(fold_valid_grid.to_dict("records"))

        for target in targets:
            selected = _select_policy_for_valid_target(
                fold_valid_grid,
                target,
                max_valid_abs_drop_pp=max_valid_abs_drop_pp,
                fallback_min_save_pct=fallback_min_save_pct,
            )
            selected_rows.append(selected)
            policy = _policy_from_row(selected)
            valid_decisions, valid_summary, _ = apply_policy(valid_frame, policy)
            test_decisions, test_summary, _ = apply_policy(test_frame, policy)
            for summary_frame, sink in [(valid_summary, valid_summary_rows), (test_summary, test_summary_rows)]:
                row = summary_frame.iloc[0].to_dict()
                row["fold_id"] = fold_id
                row["test_model"] = fold_id
                row["target_valid_decision_accuracy"] = float(target)
                row["target_valid_decision_accuracy_pct"] = float(target) * 100.0
                row["selected_valid_policy_id"] = str(selected["policy_id"])
                row["selected_valid_decision_accuracy_pct"] = float(selected["decision_accuracy_pct"])
                row["selected_valid_step_save_pct"] = float(selected["pct_steps_saved"])
                row["selected_valid_resolve_rate_change_pp"] = float(selected["resolve_rate_change_pp"])
                row["selection_status"] = str(selected["selection_status"])
                sink.append(row)
            if write_decisions:
                decision_meta = {
                    "fold_id": fold_id,
                    "test_model": fold_id,
                    "target_valid_decision_accuracy": float(target),
                    "target_valid_decision_accuracy_pct": float(target) * 100.0,
                    "selected_valid_policy_id": str(selected["policy_id"]),
                    "selection_status": str(selected["selection_status"]),
                }
                valid_decisions = valid_decisions.assign(split="valid", **decision_meta)
                test_decisions = test_decisions.assign(split="test", **decision_meta)
                valid_decision_rows.append(valid_decisions)
                test_decision_rows.append(test_decisions)

    valid_grid = pd.DataFrame(valid_grid_rows)
    selected_frame = pd.DataFrame(selected_rows)
    valid_summary = pd.DataFrame(valid_summary_rows)
    test_summary = pd.DataFrame(test_summary_rows)
    aggregate_valid = _aggregate_policy_sweep_by_target(valid_summary)
    aggregate_test = _aggregate_policy_sweep_by_target(test_summary)

    write_table(valid_grid, sweep_dir / "valid_policy_candidate_grid.csv")
    write_table(selected_frame, sweep_dir / "per_fold_selected_policies.csv")
    write_table(valid_summary, sweep_dir / "per_fold_valid_metrics.csv")
    write_table(test_summary, sweep_dir / "per_fold_test_metrics.csv")
    write_table(aggregate_valid, sweep_dir / "aggregate_valid_metrics.csv")
    write_table(aggregate_test, sweep_dir / "aggregate_test_metrics.csv")
    if write_decisions:
        if valid_decision_rows:
            write_table(pd.concat(valid_decision_rows, ignore_index=True), sweep_dir / "valid_decisions_by_target.parquet")
        if test_decision_rows:
            write_table(pd.concat(test_decision_rows, ignore_index=True), sweep_dir / "test_decisions_by_target.parquet")

    plots = _write_lightgbm_policy_sweep_plots(sweep_dir, aggregate_test, test_summary)
    display_rows = []
    if not aggregate_test.empty:
        for row in aggregate_test.to_dict("records"):
            display_rows.append(
                {
                    "valid_acc_target": f"{float(row['target_valid_decision_accuracy_pct']):.0f}",
                    "test_save_pct": f"{float(row['step_save_pct']):.2f}",
                    "test_acc_pct": f"{float(row['decision_accuracy_pct']):.2f}",
                    "test_coverage_pct": f"{float(row['coverage_pct']):.2f}",
                    "original_resolve_pct": f"{float(row['original_resolve_rate_pct']):.2f}",
                    "adjusted_resolve_pct": f"{float(row['adjusted_resolve_rate_pct']):.2f}",
                    "actual_resolve_change_pp": f"{float(row['resolve_rate_change_pp']):+.2f}",
                    "mean_abs_actual_change_pp": f"{float(row['mean_abs_resolve_rate_change_pp']):.2f}",
                    "false_negatives": int(row["false_negatives"]),
                    "false_positives": int(row["false_positives"]),
                    "decided": int(row["decided_trajectories"]),
                }
            )
    lines = [
        "# LightGBM Valid-Accuracy Policy Sweep",
        "",
        f"- run_dir: `{run_dir}`",
        f"- sweep_dir: `{sweep_dir}`",
        f"- completed folds used: `{len(completed)}`",
        f"- completed folds skipped by config: `{len(skipped_completed)}`",
        f"- target valid decision accuracy: `{_format_prob(targets[0])}` to `{_format_prob(targets[-1])}`",
        f"- valid resolve-drop guard: `{'disabled' if max_valid_abs_drop_pp is None else str(max_valid_abs_drop_pp) + 'pp'}`",
        "- Selection uses valid metrics only; the selected policy is then applied unchanged to test.",
        "",
        "## Outputs",
        "",
        "- `valid_policy_candidate_grid.csv`: all candidate policies evaluated on valid.",
        "- `per_fold_selected_policies.csv`: valid-selected policy for each fold and target accuracy.",
        "- `per_fold_test_metrics.csv`: held-out test metrics after applying each valid-selected policy.",
        "- `aggregate_test_metrics.csv`: count-weighted test aggregate by valid accuracy target.",
        "- `test_decisions_by_target.parquet`: per-trajectory test decisions for downstream analysis.",
        "",
    ]
    if display_rows:
        lines.extend(["## Aggregate Test Frontier", ""])
        lines.extend(
            _markdown_table(
                display_rows,
                [
                    "valid_acc_target",
                    "test_save_pct",
                    "test_acc_pct",
                    "test_coverage_pct",
                    "original_resolve_pct",
                    "adjusted_resolve_pct",
                    "actual_resolve_change_pp",
                    "mean_abs_actual_change_pp",
                    "false_negatives",
                    "false_positives",
                    "decided",
                ],
            )
        )
        lines.append("")
    if plots:
        lines.extend(["## Plots", ""])
        for path in plots:
            lines.append(f"- `{Path(path).name}`")
        lines.append("")
    (sweep_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (sweep_dir / "_SUCCESS").write_text("lightgbm policy sweep completed\n", encoding="utf-8")
    payload = {
        "ok": True,
        "completed_folds": len(completed),
        "skipped_completed_folds": [fold_dir.name for fold_dir in skipped_completed],
        "targets": targets,
        "sweep_dir": str(sweep_dir),
        "candidate_policies": int(len(valid_grid) / len(completed)) if completed else 0,
        "plots": plots,
    }
    write_json(sweep_dir / "sweep_manifest.json", payload)
    print(f"[lightgbm-policy-sweep] wrote {sweep_dir}", flush=True)
    return payload


def summarize_lightgbm_main(
    config: str | Path = "configs/earlyeval.yaml",
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    import pandas as pd

    cfg = load_earlyeval_config(config)
    out = ensure_dir(output_dir or _default_output_dir(cfg, cfg.run_id))
    run_dir = out / "lightgbm_main"
    if not run_dir.exists():
        raise FileNotFoundError(f"LightGBM run directory does not exist: {run_dir}")
    summary_dir = ensure_dir(run_dir / "summary")
    excluded_models = _excluded_models_from_config(cfg)
    tables = _read_completed_fold_tables(run_dir, excluded_models=excluded_models)
    test_frame = tables["test"]
    valid_frame = tables["valid"]
    calibration = tables["calibration"]
    completed = tables["completed_folds"]
    skipped_completed = tables["skipped_completed_folds"]
    eligible = [row for row in _eligible_lightgbm_folds(cfg, dataset="sweverify") if row["eligible"]]

    if not test_frame.empty:
        test_frame["test_resolve_rate_change_pp"] = -test_frame["resolve_rate_drop"].astype(float) * 100.0
        test_frame["test_decision_accuracy_pct"] = test_frame["decision_accuracy"].astype(float) * 100.0
        test_frame["test_coverage_pct"] = test_frame["coverage"].astype(float) * 100.0
        write_table(test_frame, summary_dir / "per_fold_test_selected.csv")
    if not valid_frame.empty:
        valid_frame["valid_resolve_rate_change_pp"] = -valid_frame["resolve_rate_drop"].astype(float) * 100.0
        valid_frame["valid_decision_accuracy_pct"] = valid_frame["decision_accuracy"].astype(float) * 100.0
        valid_frame["valid_coverage_pct"] = valid_frame["coverage"].astype(float) * 100.0
        write_table(valid_frame, summary_dir / "per_fold_valid_selected.csv")
    if not calibration.empty:
        write_table(calibration, summary_dir / "per_fold_calibration.csv")

    aggregate = _aggregate_selected_policy(test_frame)
    aggregate_frame = pd.DataFrame([aggregate]) if aggregate else pd.DataFrame()
    if not aggregate_frame.empty:
        write_table(aggregate_frame, summary_dir / "aggregate_test_summary.csv")
    plots = _write_lightgbm_summary_plots(summary_dir, test_frame)

    display_rows = []
    if not test_frame.empty:
        cols = [
            "fold_id",
            "original_total",
            "n_decided",
            "test_coverage_pct",
            "test_decision_accuracy_pct",
            "pct_steps_saved",
            "test_resolve_rate_change_pp",
        ]
        for row in test_frame[cols].sort_values("fold_id").to_dict("records"):
            display_rows.append(
                {
                    "fold": row["fold_id"],
                    "n": int(row["original_total"]),
                    "decided": int(row["n_decided"]),
                    "coverage_pct": f"{float(row['test_coverage_pct']):.2f}",
                    "decision_acc_pct": f"{float(row['test_decision_accuracy_pct']):.2f}",
                    "step_save_pct": f"{float(row['pct_steps_saved']):.2f}",
                    "resolve_change_pp": f"{float(row['test_resolve_rate_change_pp']):+.2f}",
                }
            )
    lines = [
        "# LightGBM Main Current Summary",
        "",
        f"- run_dir: `{run_dir}`",
        f"- completed folds: `{len(completed)} / {len(eligible)}`",
        f"- completed folds skipped by config: `{len(skipped_completed)}`",
        f"- this report only uses folds with `_SUCCESS`.",
        "",
    ]
    if aggregate:
        lines.extend(
            [
                "## Aggregate Completed Folds",
                "",
                f"- trajectories: `{aggregate['trajectories']}`",
                f"- step saving: `{aggregate['step_save_pct']:.2f}%`",
                f"- decision accuracy: `{aggregate['decision_accuracy_pct']:.2f}%`",
                f"- coverage: `{aggregate['coverage_pct']:.2f}%`",
                f"- resolve-rate change: `{aggregate['resolve_rate_change_pp']:+.2f}pp`",
                f"- mean absolute resolve-rate change: `{aggregate['mean_abs_resolve_rate_change_pp']:.2f}pp`",
                "",
            ]
        )
    if display_rows:
        lines.extend(["## Per-Fold Test Results", ""])
        lines.extend(
            _markdown_table(
                display_rows,
                ["fold", "n", "decided", "coverage_pct", "decision_acc_pct", "step_save_pct", "resolve_change_pp"],
            )
        )
        lines.append("")
    if plots:
        lines.extend(["## Plots", ""])
        for path in plots:
            lines.append(f"- `{Path(path).name}`")
        lines.append("")
    (summary_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    payload = {
        "ok": True,
        "completed_folds": len(completed),
        "skipped_completed_folds": [fold_dir.name for fold_dir in skipped_completed],
        "eligible_folds": len(eligible),
        "summary_dir": str(summary_dir),
        "aggregate": aggregate,
        "plots": plots,
    }
    write_json(summary_dir / "summary_manifest.json", payload)
    return payload


def run_smoke(config: str | Path = "configs/earlyeval.yaml", output_dir: str | Path | None = None) -> dict[str, Any]:
    cfg = load_earlyeval_config(config)
    out = ensure_dir(output_dir or _default_output_dir(cfg, cfg.run_id))
    smoke = cfg.payload.get("smoke") or {}
    plan = build_execution_plan(config=config, output_dir=out)
    audit = audit_prefix_tables(config=config, output_dir=out, datasets=list(smoke.get("audit_datasets") or []))
    splits = make_split_manifests(
        config=config,
        output_dir=out,
        datasets=list(smoke.get("split_datasets") or []),
        max_folds=int(smoke.get("max_folds_per_dataset", 2)),
    )
    success = out / "_SUCCESS"
    success.write_text("earlyeval smoke completed\n", encoding="utf-8")
    return {
        "ok": bool(audit.get("ok")) and bool(splits.get("ok")),
        "output_dir": str(out),
        "plan": plan["config"],
        "audit": audit,
        "splits": splits,
        "success_marker": str(success),
        "heavy_training_executed": False,
    }


def run_earlyeval_stage(
    *,
    stage: str,
    config: str | Path = "configs/earlyeval.yaml",
    output_dir: str | Path | None = None,
    datasets: list[str] | None = None,
    max_folds: int | None = None,
    execute: bool = False,
    force: bool = False,
    max_parallel_folds: int = 1,
) -> dict[str, Any]:
    if stage == "plan":
        return build_execution_plan(config=config, output_dir=output_dir)
    if stage == "audit-prefix":
        return audit_prefix_tables(config=config, output_dir=output_dir, datasets=datasets)
    if stage == "make-splits":
        return make_split_manifests(config=config, output_dir=output_dir, datasets=datasets, max_folds=max_folds)
    if stage == "smoke":
        return run_smoke(config=config, output_dir=output_dir)
    if stage == "lightgbm-main":
        return run_lightgbm_main(
            config=config,
            output_dir=output_dir,
            execute=execute,
            force=force,
            fold_limit=max_folds,
            max_parallel_folds=max_parallel_folds,
        )
    if stage == "lightgbm-summary":
        return summarize_lightgbm_main(config=config, output_dir=output_dir)
    if stage == "lightgbm-policy-sweep":
        return run_lightgbm_policy_sweep(config=config, output_dir=output_dir, fold_limit=max_folds)
    raise ValueError(f"Unsupported earlyeval stage: {stage}")
