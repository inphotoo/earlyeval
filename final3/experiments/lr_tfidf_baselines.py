from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression

from final3.core.io import ensure_dir, write_json, write_table
from final3.experiments.rq_final import (
    _default_output_dir,
    _eligible_lightgbm_folds,
    _excluded_models_from_config,
    _fit_feature_engineer_on_train,
    _markdown_table,
    _resolve_project_path,
    _safe_label_min_step,
    _answer_module_root,
    _shared_feature_engineer_path,
    load_rq_final_config,
)


def _drop_excluded_train_models(prefix_df, *, excluded: set[str], logger=print):
    """Filter prefix_df rows whose model_id is in `excluded`.

    The held-out test_model should be excluded from `excluded` upstream;
    this function only removes the *additional* models the config blocks
    from train/valid (low coverage, audited outliers, etc.).
    """

    if not excluded:
        return prefix_df
    before_models = int(prefix_df["model_id"].nunique())
    before_rows = int(len(prefix_df))
    mask = ~prefix_df["model_id"].astype(str).isin(set(excluded))
    filtered = prefix_df.loc[mask].copy()
    logger(
        f"[lr-tfidf] excluded {len(excluded)} configured model(s) before split: "
        f"kept {int(filtered['model_id'].nunique())}/{before_models} models, "
        f"{int(len(filtered))}/{before_rows} rows."
    )
    return filtered


def _legacy_module_root() -> Path:
    return _answer_module_root()


def _import_legacy():
    root = _legacy_module_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    import config  # type: ignore
    from feature_engineer import FeatureEngineer, TFIDF_ACTION_FEEDBACK, TFIDF_THOUGHT  # type: ignore
    from gold_text_tfidf_ablation_posthoc import _repair_unpickled_tfidf_for_local_sklearn, _set_run_dirs  # type: ignore
    from model_holdout_shadow_valid_retrain import (  # type: ignore
        _build_split,
        _json_default,
        _load_prefix_table,
        _prediction_frame,
        _required_columns,
        _safe_name,
        _set_cpu_thread_limits,
        _transform_tfidf_subset_streaming,
    )
    from probability_calibration import calibration_summary_row, fit_sigmoid_calibrator  # type: ignore
    from safe_stop_dual_head_retrain import (  # type: ignore
        _evaluate_policies,
        _evaluate_selected,
        _head_column,
        _policy_grid,
        _safe_targets,
        _select_policies,
        _write_report,
    )
    from trainer import save_model  # type: ignore

    return {
        "config": config,
        "FeatureEngineer": FeatureEngineer,
        "TFIDF_ACTION_FEEDBACK": TFIDF_ACTION_FEEDBACK,
        "TFIDF_THOUGHT": TFIDF_THOUGHT,
        "_build_split": _build_split,
        "_json_default": _json_default,
        "_load_prefix_table": _load_prefix_table,
        "_prediction_frame": _prediction_frame,
        "_repair_unpickled_tfidf_for_local_sklearn": _repair_unpickled_tfidf_for_local_sklearn,
        "_required_columns": _required_columns,
        "_safe_name": _safe_name,
        "_set_cpu_thread_limits": _set_cpu_thread_limits,
        "_set_run_dirs": _set_run_dirs,
        "_transform_tfidf_subset_streaming": _transform_tfidf_subset_streaming,
        "calibration_summary_row": calibration_summary_row,
        "fit_sigmoid_calibrator": fit_sigmoid_calibrator,
        "_evaluate_policies": _evaluate_policies,
        "_evaluate_selected": _evaluate_selected,
        "_head_column": _head_column,
        "_policy_grid": _policy_grid,
        "_safe_targets": _safe_targets,
        "_select_policies": _select_policies,
        "_write_report": _write_report,
        "save_model": save_model,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run final3 LR / TF-IDF dual-head safe-stop baselines.")
    parser.add_argument("--config", type=Path, default=Path("configs/rq_final.yaml"))
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--execute", action="store_true", help="Actually run folds. Without this, write a dry-run plan.")
    parser.add_argument("--force", action="store_true", help="Re-run folds even when _SUCCESS exists.")
    parser.add_argument("--max-folds", type=int, default=None)
    parser.add_argument("--test-models", nargs="*", default=None)
    parser.add_argument(
        "--variants",
        nargs="+",
        default=["dense_af", "tfidf_af"],
        choices=("dense", "dense_af", "tfidf_af", "dense_af_thought", "tfidf_af_thought"),
    )
    parser.add_argument("--run-subdir", default="lr_tfidf_baselines")
    parser.add_argument("--max-instances", type=int, default=500)
    parser.add_argument(
        "--smoke-trajectories-per-split",
        type=int,
        default=0,
        help="Sample this many trajectories per train/valid/test split after split construction. Full run keeps 0.",
    )
    parser.add_argument(
        "--safe-label-min-step",
        type=int,
        default=None,
        help=(
            "Override the safe-stop label min-step. When omitted, falls back "
            "to configs/rq_final.yaml main_model.safe_label_min_step."
        ),
    )
    parser.add_argument("--policy-min-steps", nargs="+", type=int, default=[0])
    parser.add_argument("--consecutive", nargs="+", type=int, default=[1])
    parser.add_argument("--success-thresholds", nargs="+", type=float, default=[0.80, 0.90, 0.95])
    parser.add_argument("--failure-thresholds", nargs="+", type=float, default=[0.80, 0.90, 0.95])
    parser.add_argument("--score-modes", nargs="+", choices=("raw", "calibrated"), default=["calibrated"])
    parser.add_argument("--max-valid-abs-drop-pp", type=float, default=2.0)
    parser.add_argument("--min-valid-decision-acc", type=float, default=0.90)
    parser.add_argument("--fallback-min-save-pct", type=float, default=0.0)
    parser.add_argument("--max-cpu-threads", type=int, default=int(os.environ.get("SWE_MAX_CPU_THREADS", "4")))
    parser.add_argument("--text-batch-size", type=int, default=4096)
    parser.add_argument("--solver", default="saga", choices=("saga", "liblinear"))
    parser.add_argument("--max-iter", type=int, default=80)
    parser.add_argument("--c", type=float, default=1.0)
    parser.add_argument("--class-weight", default="balanced", choices=("balanced", "none"))
    parser.add_argument(
        "--fit-feature-engineer-on-train",
        action="store_true",
        help=(
            "Override the yaml main_model.feature_engineer_fit_on_train flag "
            "and force every fold to fit a fresh FeatureEngineer on its own "
            "train split instead of reusing the shared pre-fit pkl. Strict "
            "no-leak but ~5-10x slower."
        ),
    )
    return parser.parse_args()


def _variant_specs(requested: list[str]) -> list[dict[str, Any]]:
    specs = {
        "dense": {
            "predictor": "LR_Dense",
            "description": "LogisticRegression over dense engineered features only",
            "blocks": ("dense",),
        },
        "dense_af": {
            "predictor": "LR_Dense_AF",
            "description": "LogisticRegression over dense + action/feedback TF-IDF-SVD features",
            "blocks": ("dense", "af"),
        },
        "tfidf_af": {
            "predictor": "TFIDF_LR_AF",
            "description": "LogisticRegression over action/feedback TF-IDF-SVD features only",
            "blocks": ("af",),
        },
        "dense_af_thought": {
            "predictor": "LR_Dense_AF_Thought",
            "description": "LogisticRegression over dense + action/feedback + thought TF-IDF-SVD features",
            "blocks": ("dense", "af", "thought"),
        },
        "tfidf_af_thought": {
            "predictor": "TFIDF_LR_AF_Thought",
            "description": "LogisticRegression over action/feedback + thought TF-IDF-SVD features only",
            "blocks": ("af", "thought"),
        },
    }
    out = []
    seen = set()
    for item in requested:
        if item in seen:
            continue
        seen.add(item)
        out.append(specs[item])
    return out


def _combine_blocks(
    matrices: dict[str, sparse.csr_matrix],
    *,
    split: str,
    blocks: tuple[str, ...],
) -> sparse.csr_matrix:
    parts = [matrices[f"{split}_{block}"] for block in blocks]
    if len(parts) == 1:
        return parts[0].tocsr()
    return sparse.hstack(parts, format="csr")


def _feature_names(
    feature_engineer: Any,
    *,
    blocks: tuple[str, ...],
    tfidf_af_cols: list[str],
    tfidf_thought_cols: list[str],
) -> list[str]:
    names: list[str] = []
    if "dense" in blocks:
        names.extend(list(feature_engineer.dense_feature_names))
    if "af" in blocks:
        names.extend(feature_engineer.get_tfidf_feature_names_for_columns(tfidf_af_cols))
    if "thought" in blocks:
        names.extend(feature_engineer.get_tfidf_feature_names_for_columns(tfidf_thought_cols))
    return names


def _fit_lr(
    X: sparse.csr_matrix,
    y: np.ndarray,
    *,
    sample_weight: np.ndarray,
    solver: str,
    max_iter: int,
    c_value: float,
    class_weight: str | None,
    seed: int,
) -> tuple[LogisticRegression, bool]:
    model = LogisticRegression(
        C=float(c_value),
        class_weight=class_weight,
        max_iter=int(max_iter),
        random_state=int(seed),
        solver=solver,
        n_jobs=1,
    )
    converged = True
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        model.fit(X, y, sample_weight=sample_weight)
    if any(issubclass(item.category, ConvergenceWarning) for item in caught):
        converged = False
    return model, converged


def _positive_probability(model: LogisticRegression, matrix: sparse.csr_matrix) -> np.ndarray:
    probabilities = model.predict_proba(matrix)
    classes = list(model.classes_)
    if 1 not in classes:
        return np.zeros(matrix.shape[0], dtype=np.float64)
    return np.asarray(probabilities[:, classes.index(1)], dtype=np.float64)


def _run_fold(
    *,
    cfg: Any,
    args: argparse.Namespace,
    fold: dict[str, Any],
    fold_dir: Path,
) -> dict[str, Any]:
    legacy = _import_legacy()
    config = legacy["config"]
    config_payload = cfg.payload
    dataset = (config_payload.get("datasets") or {})["sweverify"]
    run_root = fold_dir / "_legacy_runtime"
    legacy["_set_cpu_thread_limits"](args.max_cpu_threads)
    legacy["_set_run_dirs"](run_root)

    prefix_path = _resolve_project_path(dataset["prefix_table"])
    verified_jsonl = _resolve_project_path(dataset["verified_jsonl"])
    # Decide up front whether to (a) reuse the global pre-fit pkl (default,
    # fast, transductive feature engineering shared with LightGBM main) or
    # (b) fit a fresh FeatureEngineer per fold (strict no-leak, opt-in).
    fit_on_train = bool(getattr(args, "fit_feature_engineer_on_train", False)) or _fit_feature_engineer_on_train(cfg)
    if fit_on_train:
        # Per-fold fit needs the text columns at fit time, so build the
        # column list from a placeholder engineer first.
        placeholder_engineer = legacy["FeatureEngineer"](
            include_model_id=True, tfidf_level="with_thought"
        )
        required_columns = legacy["_required_columns"](placeholder_engineer, include_text=True)
    else:
        # Reuse the shared pkl that was fit once on the full prefix table.
        shared_feature_engineer_path = _shared_feature_engineer_path()
        feature_engineer = legacy["FeatureEngineer"].load(shared_feature_engineer_path)
        legacy["_repair_unpickled_tfidf_for_local_sklearn"](feature_engineer)
        required_columns = legacy["_required_columns"](feature_engineer, include_text=False)
    specs = _variant_specs(args.variants)
    needs_af = any("af" in spec["blocks"] for spec in specs)
    needs_thought = any("thought" in spec["blocks"] for spec in specs)

    prefix_df = legacy["_load_prefix_table"](prefix_path, required_columns)
    excluded_train_models = _excluded_models_from_config(cfg) - {str(fold["test_model"])}
    prefix_df = _drop_excluded_train_models(prefix_df, excluded=excluded_train_models)
    df_train, df_valid, df_test, split_meta, split_summary = legacy["_build_split"](
        prefix_df,
        verified_jsonl=verified_jsonl,
        holdout_models=str(fold["test_model"]),
        max_instances=int(args.max_instances),
        split_strategy="per_instance_model",
        valid_traj_ratio=float((config_payload.get("split") or {}).get("valid_ratio", 0.15)),
        valid_per_instance=0,
        valid_models_per_instance=int(
            (config_payload.get("split") or {}).get("valid_models_per_instance", 3)
        ),
        shadow_valid_max_trajectories=0,
        seed=cfg.seed,
        smoke_trajectories_per_split=int(args.smoke_trajectories_per_split),
        mask_train_model_id=True,
    )
    del prefix_df
    gc.collect()
    if fit_on_train:
        feature_engineer = legacy["FeatureEngineer"](
            include_model_id=True, tfidf_level="with_thought"
        )
        feature_engineer.fit(df_train)
        fold_dir.mkdir(parents=True, exist_ok=True)
        fe_local_path = fold_dir / "models" / "feature_engineer_fold_local.pkl"
        fe_local_path.parent.mkdir(parents=True, exist_ok=True)
        feature_engineer.save(fe_local_path)
        split_meta["feature_engineer_fit_on_train"] = True
        split_meta["feature_engineer_source"] = "fit_on_train"
        split_meta["feature_engineer_fold_local_path"] = str(fe_local_path)
    else:
        split_meta["feature_engineer_fit_on_train"] = False
        split_meta["feature_engineer_source"] = "shared_pkl"
        split_meta["feature_engineer_pkl_path"] = str(_shared_feature_engineer_path())
    split_meta["excluded_train_models"] = sorted(excluded_train_models)

    safe_label_min_step = (
        int(args.safe_label_min_step)
        if args.safe_label_min_step is not None
        else _safe_label_min_step(cfg)
    )
    split_meta["safe_label_min_step"] = int(safe_label_min_step)
    split_meta["baseline_family"] = "lr_tfidf"
    split_meta["variants"] = list(args.variants)
    split_meta["solver"] = args.solver
    split_meta["max_iter"] = int(args.max_iter)
    split_meta["class_weight"] = args.class_weight

    y_success_train, y_failure_train = legacy["_safe_targets"](df_train, safe_label_min_step)
    y_success_valid, y_failure_valid = legacy["_safe_targets"](df_valid, safe_label_min_step)
    y_success_test, y_failure_test = legacy["_safe_targets"](df_test, safe_label_min_step)
    w_train = df_train["sample_weight"].to_numpy(dtype=np.float32)
    w_valid = df_valid["sample_weight"].to_numpy(dtype=np.float32)

    tfidf_af_cols = list(legacy["TFIDF_ACTION_FEEDBACK"].keys())
    tfidf_thought_cols = list(legacy["TFIDF_THOUGHT"].keys())
    X_train_dense = sparse.csr_matrix(feature_engineer.transform_dense(df_train))
    X_valid_dense = sparse.csr_matrix(feature_engineer.transform_dense(df_valid))
    X_test_dense = sparse.csr_matrix(feature_engineer.transform_dense(df_test))
    if needs_af:
        X_train_af, X_valid_af, X_test_af = legacy["_transform_tfidf_subset_streaming"](
            prefix_table_path=prefix_path,
            feature_engineer=feature_engineer,
            df_train=df_train,
            df_valid=df_valid,
            df_test=df_test,
            column_names=tfidf_af_cols,
            batch_size=args.text_batch_size,
        )
    else:
        X_train_af = sparse.csr_matrix((len(df_train), 0))
        X_valid_af = sparse.csr_matrix((len(df_valid), 0))
        X_test_af = sparse.csr_matrix((len(df_test), 0))
    if needs_thought:
        X_train_thought, X_valid_thought, X_test_thought = legacy["_transform_tfidf_subset_streaming"](
            prefix_table_path=prefix_path,
            feature_engineer=feature_engineer,
            df_train=df_train,
            df_valid=df_valid,
            df_test=df_test,
            column_names=tfidf_thought_cols,
            batch_size=args.text_batch_size,
        )
    else:
        X_train_thought = sparse.csr_matrix((len(df_train), 0))
        X_valid_thought = sparse.csr_matrix((len(df_valid), 0))
        X_test_thought = sparse.csr_matrix((len(df_test), 0))
    matrices = {
        "train_dense": X_train_dense,
        "valid_dense": X_valid_dense,
        "test_dense": X_test_dense,
        "train_af": X_train_af,
        "valid_af": X_valid_af,
        "test_af": X_test_af,
        "train_thought": X_train_thought,
        "valid_thought": X_valid_thought,
        "test_thought": X_test_thought,
    }

    valid_pred = legacy["_prediction_frame"](df_valid)
    test_pred = legacy["_prediction_frame"](df_test)
    del df_train, df_valid, df_test
    gc.collect()

    class_weight = None if args.class_weight == "none" else args.class_weight
    calibration_rows: list[dict[str, Any]] = []
    fit_rows: list[dict[str, Any]] = []
    variant_rows: list[dict[str, Any]] = []
    predictors: list[str] = []
    models_dir = ensure_dir(fold_dir / "models")
    seed = int(cfg.seed)
    for spec in specs:
        predictor = str(spec["predictor"])
        predictors.append(predictor)
        blocks = tuple(spec["blocks"])
        X_train = _combine_blocks(matrices, split="train", blocks=blocks)
        X_valid = _combine_blocks(matrices, split="valid", blocks=blocks)
        X_test = _combine_blocks(matrices, split="test", blocks=blocks)
        names = _feature_names(
            feature_engineer,
            blocks=blocks,
            tfidf_af_cols=tfidf_af_cols,
            tfidf_thought_cols=tfidf_thought_cols,
        )
        variant_rows.append(
            {
                "predictor": predictor,
                "description": spec["description"],
                "blocks": "+".join(blocks),
                "feature_count": int(X_train.shape[1]),
            }
        )
        for head_name, y_train, y_valid, y_test, column_prefix, seed_offset in (
            ("safe_success", y_success_train, y_success_valid, y_success_test, "success", 0),
            ("safe_failure", y_failure_train, y_failure_valid, y_failure_test, "failure", 97),
        ):
            model_name = f"{predictor}__{head_name}"
            model, converged = _fit_lr(
                X_train,
                y_train,
                sample_weight=w_train,
                solver=args.solver,
                max_iter=args.max_iter,
                c_value=args.c,
                class_weight=class_weight,
                seed=seed + seed_offset,
            )
            legacy["save_model"](model, models_dir / f"{legacy['_safe_name'](model_name)}.pkl")
            valid_raw = _positive_probability(model, X_valid)
            test_raw = _positive_probability(model, X_test)
            calibrator = legacy["fit_sigmoid_calibrator"](valid_raw, y_valid, sample_weight=w_valid)
            valid_cal = calibrator.predict(valid_raw)
            test_cal = calibrator.predict(test_raw)
            legacy["save_model"](calibrator, models_dir / f"calibrator_{legacy['_safe_name'](model_name)}.pkl")
            valid_pred[legacy["_head_column"](column_prefix, "raw", predictor)] = valid_raw.astype(np.float32)
            valid_pred[legacy["_head_column"](column_prefix, "calibrated", predictor)] = valid_cal.astype(np.float32)
            test_pred[legacy["_head_column"](column_prefix, "raw", predictor)] = test_raw.astype(np.float32)
            test_pred[legacy["_head_column"](column_prefix, "calibrated", predictor)] = test_cal.astype(np.float32)
            calibration_rows.append(
                {
                    "head": head_name,
                    **legacy["calibration_summary_row"](
                        model_name=model_name,
                        calibrator=calibrator,
                        y_valid=y_valid,
                        raw_prob_valid=valid_raw,
                        y_test=y_test,
                        raw_prob_test=test_raw,
                    ),
                }
            )
            fit_rows.append(
                {
                    "model_name": model_name,
                    "predictor": predictor,
                    "head": head_name,
                    "solver": args.solver,
                    "max_iter": int(args.max_iter),
                    "converged": bool(converged),
                    "n_iter_max": int(np.max(getattr(model, "n_iter_", [0]))),
                    "train_rows": int(X_train.shape[0]),
                    "features": int(X_train.shape[1]),
                    "positive_rate": float(np.mean(y_train)),
                }
            )
        del X_train, X_valid, X_test
        gc.collect()

    fold_dir.mkdir(parents=True, exist_ok=True)
    valid_pred.to_parquet(fold_dir / "valid_predictions_safe_stop.parquet", index=False)
    test_pred.to_parquet(fold_dir / "test_predictions_safe_stop.parquet", index=False)
    split_summary.to_csv(fold_dir / "split_summary.csv", index=False)
    (fold_dir / "split_metadata.json").write_text(
        json.dumps(split_meta, ensure_ascii=False, indent=2, default=legacy["_json_default"]),
        encoding="utf-8",
    )
    pd.DataFrame(variant_rows).to_csv(fold_dir / "variant_manifest.csv", index=False)
    pd.DataFrame(calibration_rows).to_csv(fold_dir / "safe_stop_calibration_summary.csv", index=False)
    pd.DataFrame(fit_rows).to_csv(fold_dir / "lr_fit_summary.csv", index=False)

    policies = legacy["_policy_grid"](
        success_thresholds=args.success_thresholds,
        failure_thresholds=args.failure_thresholds,
        min_steps=args.policy_min_steps,
        consecutive_values=args.consecutive,
    )
    valid_grid, valid_per_agent = legacy["_evaluate_policies"](
        valid_pred,
        run_label=str(fold_dir),
        predictors=predictors,
        score_modes=args.score_modes,
        policies=policies,
    )
    selected = legacy["_select_policies"](
        valid_grid,
        max_valid_abs_drop_pp=args.max_valid_abs_drop_pp,
        min_valid_decision_acc=args.min_valid_decision_acc,
        fallback_min_save_pct=args.fallback_min_save_pct,
    )
    test_selected = legacy["_evaluate_selected"](test_pred, run_label=str(fold_dir), selected=selected)
    valid_grid.to_csv(fold_dir / "safe_stop_valid_policy_grid.csv", index=False)
    valid_per_agent.to_csv(fold_dir / "safe_stop_valid_policy_per_agent.csv", index=False)
    selected.to_csv(fold_dir / "safe_stop_selected_policies.csv", index=False)
    test_selected.to_csv(fold_dir / "safe_stop_test_selected.csv", index=False)
    legacy["_write_report"](fold_dir, selected, test_selected)
    (fold_dir / "_SUCCESS").write_text("lr/tfidf baseline completed\n", encoding="utf-8")
    return {
        "fold_id": fold["fold_id"],
        "test_model": fold["test_model"],
        "output_dir": str(fold_dir),
        "predictors": predictors,
        "selected_rows": int(len(selected)),
    }


def _aggregate_completed(run_dir: Path) -> pd.DataFrame:
    rows = []
    for fold_dir in sorted((run_dir / "folds").glob("*/_SUCCESS")):
        test_path = fold_dir.parent / "safe_stop_test_selected.csv"
        if not test_path.exists():
            continue
        frame = pd.read_csv(test_path)
        frame.insert(0, "fold_id", fold_dir.parent.name)
        frame.insert(1, "test_model", fold_dir.parent.name)
        rows.append(frame)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def _write_summary(run_dir: Path) -> None:
    summary_dir = ensure_dir(run_dir / "summary")
    frame = _aggregate_completed(run_dir)
    if frame.empty:
        return
    write_table(frame, summary_dir / "per_fold_test_selected.csv")
    display = frame.copy()
    display["test_decision_accuracy_pct"] = display["decision_accuracy"].astype(float) * 100.0
    display["test_resolve_rate_change_pp"] = -display["resolve_rate_drop"].astype(float) * 100.0
    rows = []
    for row in display.sort_values(["prefix_model", "fold_id"]).to_dict("records"):
        rows.append(
            {
                "predictor": row["prefix_model"],
                "fold": row["fold_id"],
                "n": int(row["original_total"]),
                "decided": int(row["n_decided"]),
                "acc_pct": f"{float(row['test_decision_accuracy_pct']):.2f}",
                "save_pct": f"{float(row['pct_steps_saved']):.2f}",
                "resolve_change_pp": f"{float(row['test_resolve_rate_change_pp']):+.2f}",
            }
        )
    lines = [
        "# LR / TF-IDF Model-Comparison Baselines",
        "",
        f"- run_dir: `{run_dir}`",
        f"- completed folds: `{frame['fold_id'].nunique()}`",
        "",
        "## Per-Fold Selected Policies",
        "",
        *_markdown_table(rows, ["predictor", "fold", "n", "decided", "acc_pct", "save_pct", "resolve_change_pp"]),
        "",
    ]
    (summary_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    cfg = load_rq_final_config(args.config)
    out = ensure_dir(args.output_dir or _default_output_dir(cfg, "rq_final_lightgbm_17"))
    run_dir = ensure_dir(out / "model_compare" / args.run_subdir)
    logs_dir = ensure_dir(run_dir / "logs")
    excluded = _excluded_models_from_config(cfg)
    folds = [row for row in _eligible_lightgbm_folds(cfg, dataset="sweverify") if row["eligible"] and row["test_model"] not in excluded]
    if args.test_models:
        wanted = set(args.test_models)
        folds = [row for row in folds if row["test_model"] in wanted or row["fold_id"] in wanted]
    if args.max_folds is not None:
        folds = folds[: int(args.max_folds)]

    command_rows = []
    for order, fold in enumerate(folds, start=1):
        fold_dir = run_dir / "folds" / str(fold["fold_id"])
        marker = fold_dir / "_SUCCESS"
        command_rows.append(
            {
                "order": order,
                "fold_id": fold["fold_id"],
                "test_model": fold["test_model"],
                "trajectories": fold["trajectories"],
                "output_dir": str(fold_dir),
                "log": str(logs_dir / f"{fold['fold_id']}.log"),
                "status": "skipped_existing" if marker.exists() and not args.force else "pending",
                "variants": " ".join(args.variants),
                "max_instances": int(args.max_instances),
                "smoke_trajectories_per_split": int(args.smoke_trajectories_per_split),
            }
        )
    write_table(pd.DataFrame(command_rows), run_dir / "command_index.csv")
    write_json(
        run_dir / "run_manifest.json",
        {
            "ok": True,
            "execute": bool(args.execute),
            "config": str(cfg.path),
            "run_dir": str(run_dir),
            "folds": len(folds),
            "variants": args.variants,
            "max_instances": int(args.max_instances),
            "smoke_trajectories_per_split": int(args.smoke_trajectories_per_split),
            "max_cpu_threads": int(args.max_cpu_threads),
            "note": "CPU LR/TF-IDF dual-head baselines; no GPU is used.",
        },
    )
    if not args.execute:
        print(json.dumps({"ok": True, "execute": False, "folds": len(folds), "run_dir": str(run_dir)}, indent=2))
        return 0

    completed = 0
    skipped = 0
    failed: list[dict[str, Any]] = []
    # LR/TF-IDF baseline runs in-process and mutates legacy module globals
    # (run dirs, thread caps), so concurrent folds would race. Stay serial
    # and just keep going across single-fold failures instead of breaking.
    for row, fold in zip(command_rows, folds):
        fold_dir = Path(row["output_dir"])
        if (fold_dir / "_SUCCESS").exists() and not args.force:
            skipped += 1
            continue
        print(f"[lr-tfidf] fold {row['order']}/{len(folds)}: {fold['fold_id']}", flush=True)
        log_path = Path(row["log"])
        try:
            result = _run_fold(cfg=cfg, args=args, fold=fold, fold_dir=fold_dir)
            completed += 1
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
        except Exception as exc:
            failed.append({"fold_id": fold["fold_id"], "error": repr(exc), "log": str(log_path)})
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(f"FAILED: {exc!r}\n")
            print(
                f"[lr-tfidf] fold {fold['fold_id']} failed; continuing with next fold. Log: {log_path}",
                flush=True,
            )
            continue
    _write_summary(run_dir)
    summary = {
        "ok": not failed,
        "completed": completed,
        "skipped": skipped,
        "failed": failed,
        "run_dir": str(run_dir),
    }
    write_json(run_dir / "execution_summary.json", summary)
    print(json.dumps(summary, indent=2))
    if failed:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
