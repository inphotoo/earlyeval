#!/usr/bin/env python3
"""Post-hoc task-prompt and gold-answer ablations for model-holdout runs.

This intentionally reuses the completed run's expensive artifacts:

* data/prefix_table_filtered.parquet
* models/feature_engineer_with_model.pkl
* reports/test_predictions_all_models.parquet for baseline comparison

It does not rebuild step tables, prefix tables, gold-answer joins, or refit
TF-IDF/SVD.  It reconstructs the same model-holdout split, rebuilds the
Dense + AF + Thought matrix with the fitted FeatureEngineer, then trains a
small number of LightGBM ablations.
"""

from __future__ import annotations

import argparse
import gc
import math
import os
import re
import sys
from pathlib import Path
from typing import Any, Callable

PROJECT_ROOT = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".mplconfig"))

import numpy as np
import pandas as pd
from scipy import sparse

sys.path.insert(0, str(PROJECT_ROOT))

import config
from feature_engineer import (
    BOOL_FEATURES,
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    TFIDF_ACTION_FEEDBACK,
    TFIDF_THOUGHT,
    FeatureEngineer,
)
from gold_text_tfidf_ablation_posthoc import (
    _collect_metrics,
    _fit_lgbm_with_cpu_fallback,
    _load_prefix_table,
    _merge_with_existing_predictions,
    _repair_unpickled_tfidf_for_local_sklearn,
    _reconstruct_model_holdout_split,
    _run_ranking_reports,
    _set_run_dirs,
    _write_feature_importance,
)
from probability_calibration import calibration_summary_row, fit_sigmoid_calibrator
from trainer import save_model
from utils import get_logger, rebind_all_file_loggers, timer


LOGGER = get_logger("task_answer_ablation")

BASELINE_PREDICTORS = [
    "I_LightGBM_Dense_AF",
    "J_LightGBM_Dense_AF_Thought",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-name",
        default="model_holdout_answer_calibrated_full",
        help="Completed run to reuse. Default: model_holdout_answer_calibrated_full",
    )
    parser.add_argument(
        "--prefix-table",
        type=Path,
        default=None,
        help="Optional explicit prefix_table_filtered.parquet path.",
    )
    parser.add_argument(
        "--verified-jsonl",
        type=Path,
        default=PROJECT_ROOT.parents[2] / "swebench_verified" / "test.jsonl",
        help="Same verified JSONL used by the completed model-holdout run.",
    )
    parser.add_argument("--holdout-models", default="auto_mid3")
    parser.add_argument("--max-instances", type=int, default=500)
    parser.add_argument(
        "--output-subdir",
        default="task_answer_ablation_posthoc",
        help="Subdirectory under reports/ for ablation outputs.",
    )
    parser.add_argument(
        "--smoke-trajectories-per-split",
        type=int,
        default=0,
        help="Debug only: sample whole trajectories per split before training.",
    )
    parser.add_argument(
        "--no-gpu-lgbm",
        action="store_true",
        help="Force CPU LightGBM for this post-hoc run.",
    )
    parser.add_argument(
        "--skip-ranking-reports",
        action="store_true",
        help="Only write predictions/metric CSVs; skip ref-style ranking reports.",
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        default=["all"],
        help=(
            "Ablations to run: all, no_task_tfidf, no_task_signal, no_gold_answer, "
            "no_task_signal_no_gold_answer."
        ),
    )
    parser.add_argument(
        "--prediction-baselines",
        nargs="+",
        default=BASELINE_PREDICTORS,
        help="Existing predictors copied from the completed prediction table.",
    )
    return parser.parse_args()


def _safe_artifact_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name)


def _required_columns(feature_engineer: FeatureEngineer) -> list[str]:
    tfidf_cols = list(TFIDF_ACTION_FEEDBACK.keys()) + list(TFIDF_THOUGHT.keys())
    text_columns = [
        feature_engineer.active_text_columns[name]
        for name in tfidf_cols
        if name in feature_engineer.active_text_columns
    ]
    metadata_columns = [
        "prefix_id",
        "traj_id",
        "group_id",
        "instance_id",
        "prefix_step_idx",
        "n_steps_total_for_weighting",
        "sample_weight",
        "label",
        "model_id",
        "model",
    ]
    dense_columns = (
        list(NUMERIC_FEATURES)
        + list(BOOL_FEATURES)
        + list(CATEGORICAL_FEATURES)
        + ["model_id"]
    )
    return list(dict.fromkeys(metadata_columns + dense_columns + text_columns))


def _is_task_tfidf_feature(feature_name: str) -> bool:
    return feature_name.startswith("tfidf_task_prompt__")


def _is_task_dense_feature(feature_name: str) -> bool:
    return feature_name == "task_prompt_chars"


def _is_gold_answer_feature(feature_name: str) -> bool:
    return feature_name.startswith("gold_")


def _make_column_mask(
    feature_names: list[str],
    should_remove: Callable[[str], bool],
) -> tuple[list[int], list[str]]:
    keep_cols = [idx for idx, name in enumerate(feature_names) if not should_remove(name)]
    removed_names = [name for name in feature_names if should_remove(name)]
    if not removed_names:
        raise RuntimeError("Ablation removed zero columns; check feature-name rules.")
    return keep_cols, removed_names


def _selected_variants(requested: list[str]) -> list[dict[str, Any]]:
    aliases = {
        "no_task": "no_task_tfidf",
        "no_task_prompt": "no_task_tfidf",
        "no_task_prompt_tfidf": "no_task_tfidf",
        "no_task_tfidf": "no_task_tfidf",
        "no_task_signal": "no_task_signal",
        "no_gold": "no_gold_answer",
        "no_answer": "no_gold_answer",
        "no_gold_answer": "no_gold_answer",
        "no_task_no_gold": "no_task_signal_no_gold_answer",
        "no_task_signal_no_gold": "no_task_signal_no_gold_answer",
        "no_task_signal_no_gold_answer": "no_task_signal_no_gold_answer",
    }
    specs = {
        "no_task_tfidf": {
            "predictor": "Abl_NoTaskPromptTfidf_LightGBM",
            "description": "Dense + AF + Thought, remove only task_prompt TF-IDF block",
            "remove_fn": _is_task_tfidf_feature,
        },
        "no_task_signal": {
            "predictor": "Abl_NoTaskSignal_LightGBM",
            "description": "Dense + AF + Thought, remove task_prompt TF-IDF and task_prompt_chars",
            "remove_fn": lambda name: _is_task_tfidf_feature(name) or _is_task_dense_feature(name),
        },
        "no_gold_answer": {
            "predictor": "Abl_NoGoldAnswer_LightGBM",
            "description": "Dense + AF + Thought, remove all structured gold-answer dense features",
            "remove_fn": _is_gold_answer_feature,
        },
        "no_task_signal_no_gold_answer": {
            "predictor": "Abl_NoTaskSignal_NoGoldAnswer_LightGBM",
            "description": "Dense + AF + Thought, remove task signal and structured gold-answer features",
            "remove_fn": lambda name: (
                _is_task_tfidf_feature(name)
                or _is_task_dense_feature(name)
                or _is_gold_answer_feature(name)
            ),
        },
    }
    raw_values = [str(item).strip().lower() for item in requested if str(item).strip()]
    if not raw_values or "all" in raw_values:
        keys = ["no_task_tfidf", "no_task_signal", "no_gold_answer"]
    else:
        keys = []
        for value in raw_values:
            if value not in aliases:
                raise ValueError(
                    f"Unknown --variants value: {value}. "
                    "Use all, no_task_tfidf, no_task_signal, no_gold_answer, "
                    "no_task_signal_no_gold_answer."
                )
            key = aliases[value]
            if key not in keys:
                keys.append(key)
    return [specs[key] for key in keys]


def _write_text_summary(
    output_path: Path,
    *,
    split_meta: dict[str, Any],
    prefix_metrics: pd.DataFrame,
    final_metrics: pd.DataFrame,
    variant_manifest: pd.DataFrame,
    predictors: list[str],
) -> None:
    def format_float(value: Any, digits: int = 4) -> str:
        try:
            numeric = float(value)
            if math.isnan(numeric):
                return "nan"
            return f"{numeric:.{digits}f}"
        except Exception:
            return "nan"

    lines = [
        "Task-prompt and gold-answer ablation",
        "=" * 42,
        "",
        "Reused artifacts:",
        "  - prefix_table_filtered.parquet",
        "  - feature_engineer_with_model.pkl",
        "  - existing test_predictions_all_models.parquet for I/J comparison",
        "",
        "Not rerun:",
        "  - step table building",
        "  - prefix table building",
        "  - gold-answer feature join",
        "  - TF-IDF/SVD fitting",
        "",
        "Base matrix:",
        "  - Dense + AF + Thought",
        "  - gold raw-text TF-IDF blocks are not included in this base",
        "",
        "Split:",
        f"  train rows/trajs: {split_meta['train_rows']} / {split_meta['train_trajectories']}",
        f"  valid rows/trajs: {split_meta['valid_rows']} / {split_meta['valid_trajectories']}",
        f"  test  rows/trajs: {split_meta['test_rows']} / {split_meta['test_trajectories']}",
        "  valid/test model_id input: __MISSING__",
        f"  holdout models: {', '.join(split_meta.get('holdout_models', []))}",
        "",
        "Variants:",
    ]
    for _, row in variant_manifest.iterrows():
        lines.append(
            f"  - {row['predictor']}: removed {int(row['removed_feature_count'])} columns; "
            f"{row['description']}"
        )

    def append_metric_table(title: str, metrics: pd.DataFrame) -> None:
        lines.extend(["", title + ":"])
        view = metrics[
            (metrics["score_mode"] == "calibrated")
            & (metrics["predictor"].isin(predictors))
        ].copy()
        if view.empty:
            view = metrics[metrics["predictor"].isin(predictors)].copy()
        view = view.sort_values("roc_auc", ascending=False)
        lines.append(
            "  "
            + f"{'predictor':48s} {'mode':10s} {'acc@0.5':>8s} {'auc':>8s} "
            + f"{'pr_auc':>8s} {'brier':>8s} {'meanP':>8s}"
        )
        for _, row in view.iterrows():
            lines.append(
                "  "
                + f"{str(row['predictor'])[:48]:48s} {str(row['score_mode'])[:10]:10s} "
                + f"{format_float(row['accuracy_at_0_5']):>8s} "
                + f"{format_float(row['roc_auc']):>8s} "
                + f"{format_float(row['pr_auc']):>8s} "
                + f"{format_float(row['brier']):>8s} "
                + f"{format_float(row['mean_prob']):>8s}"
            )

    append_metric_table("Final-step trajectory metrics", final_metrics)
    append_metric_table("Prefix-row metrics", prefix_metrics)

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    LOGGER.info("Saved text summary: %s", output_path)


def main() -> int:
    args = parse_args()
    variants = _selected_variants(args.variants)
    run_root = PROJECT_ROOT / "runs" / args.run_name
    _set_run_dirs(run_root)
    rebind_all_file_loggers()

    if args.no_gpu_lgbm:
        config.LGBM_PARAMS["device"] = "cpu"
        config.LGBM_PARAMS.pop("gpu_device_id", None)

    prefix_path = args.prefix_table or config.PREFIX_TABLE_FILTERED_PATH
    feature_engineer_path = config.MODEL_DIR / "feature_engineer_with_model.pkl"
    existing_predictions_path = config.REPORT_DIR / "test_predictions_all_models.parquet"
    output_dir = config.REPORT_DIR / args.output_subdir
    output_model_dir = output_dir / "models"
    output_model_dir.mkdir(parents=True, exist_ok=True)

    if not prefix_path.is_file():
        raise FileNotFoundError(f"Missing reusable prefix table: {prefix_path}")
    if not feature_engineer_path.is_file():
        raise FileNotFoundError(f"Missing reusable FeatureEngineer: {feature_engineer_path}")

    feature_engineer = FeatureEngineer.load(feature_engineer_path)
    _repair_unpickled_tfidf_for_local_sklearn(feature_engineer)
    required_columns = _required_columns(feature_engineer)

    with timer(LOGGER, "Load prefix table and reconstruct split"):
        prefix_df = _load_prefix_table(prefix_path, required_columns)
        df_train, df_valid, df_test, split_meta = _reconstruct_model_holdout_split(
            prefix_df,
            verified_jsonl=args.verified_jsonl,
            holdout_models=args.holdout_models,
            max_instances=args.max_instances,
            smoke_trajectories_per_split=args.smoke_trajectories_per_split,
        )
        del prefix_df
        gc.collect()

    y_train = df_train["label"].to_numpy(dtype=int)
    y_valid = df_valid["label"].to_numpy(dtype=int)
    y_test = df_test["label"].to_numpy(dtype=int)
    w_train = df_train["sample_weight"].to_numpy(dtype=np.float32)
    w_valid = df_valid["sample_weight"].to_numpy(dtype=np.float32)

    LOGGER.info(
        "Split rows: train=%s valid=%s test=%s; pos rates train=%.4f valid=%.4f test=%.4f",
        len(df_train),
        len(df_valid),
        len(df_test),
        float(np.mean(y_train)),
        float(np.mean(y_valid)),
        float(np.mean(y_test)),
    )

    tfidf_cols = list(TFIDF_ACTION_FEEDBACK.keys()) + list(TFIDF_THOUGHT.keys())
    new_predictions = pd.DataFrame(
        {
            "traj_id": df_test["traj_id"].to_numpy(),
            "prefix_step_idx": df_test["prefix_step_idx"].to_numpy(),
        }
    )
    calibration_rows: list[dict[str, Any]] = []
    variant_manifest_rows: list[dict[str, Any]] = []

    with timer(LOGGER, "Build reusable Dense + AF + Thought base matrices"):
        X_train_dense = sparse.csr_matrix(feature_engineer.transform_dense(df_train))
        X_valid_dense = sparse.csr_matrix(feature_engineer.transform_dense(df_valid))
        X_test_dense = sparse.csr_matrix(feature_engineer.transform_dense(df_test))
        X_train_tfidf = feature_engineer.transform_tfidf_subset(df_train, tfidf_cols)
        X_valid_tfidf = feature_engineer.transform_tfidf_subset(df_valid, tfidf_cols)
        X_test_tfidf = feature_engineer.transform_tfidf_subset(df_test, tfidf_cols)
        X_train_base = sparse.hstack([X_train_dense, X_train_tfidf], format="csr")
        X_valid_base = sparse.hstack([X_valid_dense, X_valid_tfidf], format="csr")
        X_test_base = sparse.hstack([X_test_dense, X_test_tfidf], format="csr")
        base_feature_names = (
            list(feature_engineer.dense_feature_names)
            + feature_engineer.get_tfidf_feature_names_for_columns(tfidf_cols)
        )
        if len(base_feature_names) != X_train_base.shape[1]:
            raise RuntimeError(
                f"Base feature names {len(base_feature_names)} != columns {X_train_base.shape[1]}"
            )
        LOGGER.info(
            "Dense + AF + Thought base shape: train=%s valid=%s test=%s",
            X_train_base.shape,
            X_valid_base.shape,
            X_test_base.shape,
        )

    for spec in variants:
        public_name = spec["predictor"]
        keep_cols, removed_names = _make_column_mask(base_feature_names, spec["remove_fn"])
        feature_names = [base_feature_names[idx] for idx in keep_cols]

        with timer(LOGGER, f"Train {public_name}"):
            X_train_variant = X_train_base[:, keep_cols].tocsr()
            X_valid_variant = X_valid_base[:, keep_cols].tocsr()
            X_test_variant = X_test_base[:, keep_cols].tocsr()
            model = _fit_lgbm_with_cpu_fallback(
                X_train=X_train_variant,
                y_train=y_train,
                w_train=w_train,
                X_valid=X_valid_variant,
                y_valid=y_valid,
                w_valid=w_valid,
                feature_names=feature_names,
                model_name=public_name,
            )
            model_path = output_model_dir / f"{_safe_artifact_name(public_name)}.lgb"
            save_model(model, model_path)

            valid_raw = np.asarray(model.predict(X_valid_variant), dtype=np.float64)
            test_raw = np.asarray(model.predict(X_test_variant), dtype=np.float64)
            calibrator = fit_sigmoid_calibrator(valid_raw, y_valid, sample_weight=w_valid)
            test_cal = calibrator.predict(test_raw)
            save_model(
                calibrator,
                output_model_dir / f"calibrator_{_safe_artifact_name(public_name)}.pkl",
            )
            calibration_rows.append(
                calibration_summary_row(
                    model_name=public_name,
                    calibrator=calibrator,
                    y_valid=y_valid,
                    raw_prob_valid=valid_raw,
                    y_test=y_test,
                    raw_prob_test=test_raw,
                )
            )
            new_predictions[f"prob__{public_name}"] = test_raw.astype(np.float32)
            new_predictions[f"prob_cal__{public_name}"] = test_cal.astype(np.float32)
            _write_feature_importance(
                model,
                feature_names,
                output_dir / f"feature_importance_{_safe_artifact_name(public_name)}.csv",
            )
            variant_manifest_rows.append(
                {
                    "predictor": public_name,
                    "description": spec["description"],
                    "base_matrix": "Dense + AF + Thought",
                    "removed_feature_count": len(removed_names),
                    "kept_feature_count": len(feature_names),
                    "removed_feature_examples": "; ".join(removed_names[:30]),
                }
            )

            del X_train_variant, X_valid_variant, X_test_variant
            gc.collect()

    prediction_output_path = output_dir / "test_predictions_task_answer_ablation.parquet"
    pred_df = _merge_with_existing_predictions(
        existing_path=existing_predictions_path,
        df_test=df_test,
        new_prediction_df=new_predictions,
        baseline_predictors=args.prediction_baselines,
        output_path=prediction_output_path,
    )

    variant_manifest = pd.DataFrame(variant_manifest_rows)
    variant_manifest.to_csv(output_dir / "variant_manifest.csv", index=False)
    all_predictors = list(args.prediction_baselines) + variant_manifest["predictor"].tolist()
    prefix_metrics, final_metrics = _collect_metrics(pred_df, all_predictors)
    prefix_metrics.to_csv(output_dir / "prefix_metrics.csv", index=False)
    final_metrics.to_csv(output_dir / "final_step_metrics.csv", index=False)
    pd.DataFrame(calibration_rows).to_csv(output_dir / "probability_calibration_summary.csv", index=False)
    pd.Series(split_meta).to_json(
        output_dir / "split_reconstruction_summary.json",
        force_ascii=False,
        indent=2,
    )
    _write_text_summary(
        output_dir / "summary.txt",
        split_meta=split_meta,
        prefix_metrics=prefix_metrics,
        final_metrics=final_metrics,
        variant_manifest=variant_manifest,
        predictors=all_predictors,
    )

    if not args.skip_ranking_reports:
        _run_ranking_reports(
            predictions_path=prediction_output_path,
            output_dir=output_dir,
            predictors=all_predictors,
        )

    LOGGER.info("Task/answer ablation complete: %s", output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
