#!/usr/bin/env python3
"""Post-hoc trajectory-level calibration and step metrics.

This reuses a completed ``model_holdout_shadow_valid_retrain.py`` output.
It does not retrain LightGBM models.  It:

1. Reconstructs the same train/valid/test split.
2. Predicts only the validation final prefix with saved LightGBM models.
3. Fits one sigmoid calibrator per predictor on validation trajectories.
4. Applies that trajectory-level calibrator to existing test raw probabilities.
5. Writes exact-step and step-bucket AUC/accuracy tables.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".mplconfig"))
THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "BLIS_NUM_THREADS",
)
for _thread_env_name in THREAD_ENV_VARS:
    os.environ.setdefault(
        _thread_env_name,
        os.environ.get("SWE_MAX_CPU_THREADS", "24"),
    )
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)

import config
from feature_engineer import FeatureEngineer, TFIDF_ACTION_FEEDBACK, TFIDF_THOUGHT
from gold_text_tfidf_ablation_posthoc import (
    _load_prefix_table,
    _repair_unpickled_tfidf_for_local_sklearn,
    _set_run_dirs,
)
from model_holdout_shadow_valid_retrain import (
    _build_split,
    _make_column_mask,
    _required_columns,
    _safe_name,
    _selected_specs,
    _transform_tfidf_subset_streaming,
)
from probability_calibration import fit_sigmoid_calibrator
from utils import get_logger, rebind_all_file_loggers, timer


LOGGER = get_logger("trajectory_calibration_step_report")


def _cpu_thread_count(max_cpu_threads: int | None) -> int:
    try:
        return max(1, int(max_cpu_threads or 1))
    except Exception:
        return 24


def _set_cpu_thread_limits(max_cpu_threads: int | None) -> int:
    threads = _cpu_thread_count(max_cpu_threads)
    for env_name in THREAD_ENV_VARS:
        os.environ[env_name] = str(threads)
    os.environ["SWE_MAX_CPU_THREADS"] = str(threads)
    return threads


def _thread_limited_env(max_cpu_threads: int | None) -> dict[str, str]:
    threads = _cpu_thread_count(max_cpu_threads)
    env = os.environ.copy()
    for env_name in THREAD_ENV_VARS:
        env[env_name] = str(threads)
    env["SWE_MAX_CPU_THREADS"] = str(threads)
    return env


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", default="model_holdout_answer_calibrated_full")
    parser.add_argument("--report-subdir", default="per_instance_model_valid3_retrain")
    parser.add_argument("--prefix-table", type=Path, default=None)
    parser.add_argument(
        "--verified-jsonl",
        type=Path,
        default=PROJECT_ROOT.parents[2] / "swebench_verified" / "test.jsonl",
    )
    parser.add_argument("--holdout-models", default="auto_mid3")
    parser.add_argument("--max-instances", type=int, default=500)
    parser.add_argument("--valid-models-per-instance", type=int, default=3)
    parser.add_argument("--seed", type=int, default=config.SPLIT_SEED)
    parser.add_argument("--text-batch-size", type=int, default=4096)
    parser.add_argument(
        "--max-cpu-threads",
        type=int,
        default=int(os.environ.get("SWE_MAX_CPU_THREADS", "24")),
        help="Maximum CPU threads for BLAS/OpenMP/LightGBM prediction and ranking subprocesses.",
    )
    parser.add_argument("--variants", nargs="+", default=["default"])
    parser.add_argument("--skip-ranking-report", action="store_true")
    return parser.parse_args()


def _safe_metric(metric_fn, y_true: np.ndarray, y_prob: np.ndarray) -> float:
    try:
        return float(metric_fn(y_true, y_prob))
    except Exception:
        return float("nan")


def _binary_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=np.float64)
    y_pred = (y_prob >= 0.5).astype(int)
    return {
        "actual_rate": float(np.mean(y_true)),
        "mean_prob": float(np.mean(y_prob)),
        "bias_prob_minus_actual": float(np.mean(y_prob) - np.mean(y_true)),
        "accuracy_at_0_5": float(accuracy_score(y_true, y_pred)),
        "roc_auc": _safe_metric(roc_auc_score, y_true, y_prob),
        "pr_auc": _safe_metric(average_precision_score, y_true, y_prob),
        "brier": _safe_metric(brier_score_loss, y_true, y_prob),
        "log_loss": _safe_metric(
            lambda truth, prob: log_loss(
                truth,
                np.clip(prob, 1e-6, 1.0 - 1e-6),
                labels=[0, 1],
            ),
            y_true,
            y_prob,
        ),
    }


def _bucket_label(step: int) -> str:
    for bucket_name, lo, hi in config.STEP_BUCKETS:
        if lo <= step <= hi:
            return bucket_name
    return "other"


def _build_valid_final_matrices(
    *,
    prefix_path: Path,
    feature_engineer: FeatureEngineer,
    df_valid_final: pd.DataFrame,
    text_batch_size: int,
) -> dict[str, sparse.csr_matrix]:
    tfidf_af_cols = list(TFIDF_ACTION_FEEDBACK.keys())
    tfidf_thought_cols = list(TFIDF_THOUGHT.keys())
    with timer(LOGGER, "Build valid-final Dense / AF / Thought matrices"):
        X_dense = sparse.csr_matrix(feature_engineer.transform_dense(df_valid_final))
        X_af, _, _ = _transform_tfidf_subset_streaming(
            prefix_table_path=prefix_path,
            feature_engineer=feature_engineer,
            df_train=df_valid_final,
            df_valid=df_valid_final,
            df_test=df_valid_final,
            column_names=tfidf_af_cols,
            batch_size=text_batch_size,
        )
        X_thought, _, _ = _transform_tfidf_subset_streaming(
            prefix_table_path=prefix_path,
            feature_engineer=feature_engineer,
            df_train=df_valid_final,
            df_valid=df_valid_final,
            df_test=df_valid_final,
            column_names=tfidf_thought_cols,
            batch_size=text_batch_size,
        )
    return {
        "dense": X_dense,
        "af": X_af,
        "thought": X_thought,
    }


def _feature_names(feature_engineer: FeatureEngineer) -> tuple[list[str], list[str]]:
    tfidf_af_cols = list(TFIDF_ACTION_FEEDBACK.keys())
    tfidf_thought_cols = list(TFIDF_THOUGHT.keys())
    names_af = (
        list(feature_engineer.dense_feature_names)
        + feature_engineer.get_tfidf_feature_names_for_columns(tfidf_af_cols)
    )
    names_j = (
        list(feature_engineer.dense_feature_names)
        + feature_engineer.get_tfidf_feature_names_for_columns(tfidf_af_cols + tfidf_thought_cols)
    )
    return names_af, names_j


def _predict_valid_final_raw(
    *,
    output_dir: Path,
    specs: list[dict[str, Any]],
    matrices: dict[str, sparse.csr_matrix],
    feature_engineer: FeatureEngineer,
) -> dict[str, np.ndarray]:
    import lightgbm as lgb

    names_af, names_j = _feature_names(feature_engineer)
    raw_predictions: dict[str, np.ndarray] = {}
    model_dir = output_dir / "models"

    for spec in specs:
        predictor = spec["predictor"]
        model_path = model_dir / f"{_safe_name(predictor)}.lgb"
        if not model_path.is_file():
            LOGGER.warning("Skipping missing model: %s", model_path)
            continue
        if spec["base"] == "af":
            X_model = sparse.hstack([matrices["dense"], matrices["af"]], format="csr")
            feature_names = list(names_af)
        elif spec["base"] == "af_thought":
            X_model = sparse.hstack(
                [matrices["dense"], matrices["af"], matrices["thought"]],
                format="csr",
            )
            feature_names = list(names_j)
            remove_fn = spec.get("remove_fn")
            if remove_fn is not None:
                keep_cols, _ = _make_column_mask(feature_names, remove_fn)
                X_model = X_model[:, keep_cols].tocsr()
        else:
            raise ValueError(f"Unknown base matrix: {spec['base']}")

        LOGGER.info("Predicting valid-final raw probabilities: %s", predictor)
        booster = lgb.Booster(model_file=str(model_path))
        raw_predictions[predictor] = np.asarray(booster.predict(X_model), dtype=np.float64)
        del booster, X_model
        gc.collect()

    return raw_predictions


def _write_probability_reports(
    *,
    output_dir: Path,
    pred_df: pd.DataFrame,
    valid_final: pd.DataFrame,
    valid_raw: dict[str, np.ndarray],
    predictors: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    final_idx = pred_df.groupby("traj_id", sort=False)["prefix_step_idx"].idxmax()
    test_final = pred_df.loc[final_idx].copy()
    y_valid = valid_final["label"].to_numpy(dtype=int)
    y_test = test_final["label"].to_numpy(dtype=int)

    calibration_rows: list[dict[str, Any]] = []
    final_metric_rows: list[dict[str, Any]] = []
    calibrated_pred = pred_df.copy()

    for predictor in predictors:
        raw_col = f"prob__{predictor}"
        old_cal_col = f"prob_cal__{predictor}"
        if raw_col not in pred_df.columns or predictor not in valid_raw:
            continue
        calibrator = fit_sigmoid_calibrator(valid_raw[predictor], y_valid, sample_weight=None)
        test_raw_all = pred_df[raw_col].to_numpy(dtype=np.float64)
        test_traj_cal_all = calibrator.predict(test_raw_all)
        if old_cal_col in calibrated_pred.columns:
            calibrated_pred[f"prob_prefix_cal__{predictor}"] = calibrated_pred[old_cal_col].astype(np.float32)
        calibrated_pred[f"prob_traj_cal__{predictor}"] = test_traj_cal_all.astype(np.float32)
        calibrated_pred[old_cal_col] = test_traj_cal_all.astype(np.float32)

        test_final_raw = test_final[raw_col].to_numpy(dtype=np.float64)
        test_final_traj_cal = calibrator.predict(test_final_raw)
        if old_cal_col in test_final.columns:
            test_final_prefix_cal = test_final[old_cal_col].to_numpy(dtype=np.float64)
        else:
            test_final_prefix_cal = np.full(test_final_raw.shape, np.nan)
        valid_traj_cal = calibrator.predict(valid_raw[predictor])

        coef = None
        intercept = None
        if calibrator.estimator is not None:
            coef = float(calibrator.estimator.coef_.ravel()[0])
            intercept = float(calibrator.estimator.intercept_.ravel()[0])
        calibration_rows.append(
            {
                "predictor": predictor,
                "method": "sigmoid_platt_on_valid_final_trajectory_logits",
                "coef": coef,
                "intercept": intercept,
                "valid_actual_rate": float(np.mean(y_valid)),
                "valid_raw_mean": float(np.mean(valid_raw[predictor])),
                "valid_traj_cal_mean": float(np.mean(valid_traj_cal)),
                "test_actual_rate": float(np.mean(y_test)),
                "test_raw_final_mean": float(np.mean(test_final_raw)),
                "test_prefix_cal_final_mean": float(np.nanmean(test_final_prefix_cal)),
                "test_traj_cal_final_mean": float(np.mean(test_final_traj_cal)),
                "test_raw_bias": float(np.mean(test_final_raw) - np.mean(y_test)),
                "test_prefix_cal_bias": float(np.nanmean(test_final_prefix_cal) - np.mean(y_test)),
                "test_traj_cal_bias": float(np.mean(test_final_traj_cal) - np.mean(y_test)),
            }
        )

        for score_mode, probabilities in (
            ("raw", test_final_raw),
            ("prefix_calibrated", test_final_prefix_cal),
            ("trajectory_calibrated", test_final_traj_cal),
        ):
            if np.isnan(probabilities).all():
                continue
            final_metric_rows.append(
                {
                    "scope": "final_step_trajectory",
                    "score_mode": score_mode,
                    "predictor": predictor,
                    "n_trajectories": int(len(test_final)),
                    **_binary_metrics(y_test, probabilities),
                }
            )

    pred_out = output_dir / "test_predictions_trajectory_calibrated.parquet"
    calibrated_pred.to_parquet(pred_out, index=False)
    calibration_df = pd.DataFrame(calibration_rows)
    final_metrics_df = pd.DataFrame(final_metric_rows)
    calibration_df.to_csv(output_dir / "trajectory_calibration_summary.csv", index=False)
    final_metrics_df.to_csv(output_dir / "trajectory_final_metrics.csv", index=False)
    LOGGER.info("Saved trajectory-calibrated predictions: %s", pred_out)
    return calibrated_pred, calibration_df, final_metrics_df


def _step_metrics(pred_df: pd.DataFrame, predictors: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    score_columns = []
    for predictor in predictors:
        candidates = [
            ("raw", f"prob__{predictor}"),
            ("prefix_calibrated", f"prob_prefix_cal__{predictor}"),
            ("trajectory_calibrated", f"prob_traj_cal__{predictor}"),
        ]
        for score_mode, column in candidates:
            if column in pred_df.columns:
                score_columns.append((predictor, score_mode, column))

    exact_rows: list[dict[str, Any]] = []
    bucket_rows: list[dict[str, Any]] = []
    work = pred_df.copy()
    work["step_bucket"] = work["prefix_step_idx"].astype(int).map(_bucket_label)

    for predictor, score_mode, column in score_columns:
        for step, part in work.groupby("prefix_step_idx", sort=True):
            y_true = part["label"].to_numpy(dtype=int)
            y_prob = part[column].to_numpy(dtype=np.float64)
            row = {
                "predictor": predictor,
                "score_mode": score_mode,
                "prefix_step_idx": int(step),
                "n_rows": int(len(part)),
                "n_pos": int(y_true.sum()),
                "n_neg": int(len(y_true) - y_true.sum()),
            }
            row.update(_binary_metrics(y_true, y_prob))
            exact_rows.append(row)

        for bucket, part in work.groupby("step_bucket", sort=False):
            y_true = part["label"].to_numpy(dtype=int)
            y_prob = part[column].to_numpy(dtype=np.float64)
            row = {
                "predictor": predictor,
                "score_mode": score_mode,
                "step_bucket": bucket,
                "n_rows": int(len(part)),
                "n_pos": int(y_true.sum()),
                "n_neg": int(len(y_true) - y_true.sum()),
            }
            row.update(_binary_metrics(y_true, y_prob))
            bucket_rows.append(row)

    return pd.DataFrame(exact_rows), pd.DataFrame(bucket_rows)


def _write_markdown_report(
    *,
    output_dir: Path,
    calibration_df: pd.DataFrame,
    final_metrics_df: pd.DataFrame,
    bucket_metrics_df: pd.DataFrame,
) -> None:
    def fmt(value: Any, digits: int = 4) -> str:
        try:
            value = float(value)
            if math.isnan(value):
                return "-"
            return f"{value:.{digits}f}"
        except Exception:
            return "-"

    lines = [
        "# Trajectory Calibration and Step Metrics",
        "",
        "## Final-step trajectory calibration",
        "",
        "| Predictor | Mode | Actual | MeanProb | Bias | Acc@0.5 | AUC | Brier |",
        "|:--|:--|--:|--:|--:|--:|--:|--:|",
    ]
    view = final_metrics_df.sort_values(["score_mode", "roc_auc"], ascending=[True, False])
    for _, row in view.iterrows():
        lines.append(
            f"| {row['predictor']} | {row['score_mode']} | {fmt(row['actual_rate'])} | "
            f"{fmt(row['mean_prob'])} | {fmt(row['bias_prob_minus_actual'])} | "
            f"{fmt(row['accuracy_at_0_5'])} | {fmt(row['roc_auc'])} | {fmt(row['brier'])} |"
        )

    lines.extend(
        [
            "",
            "## Calibration shift",
            "",
            "| Predictor | Valid actual | Valid raw | Valid traj-cal | Test actual | Test raw | Test prefix-cal | Test traj-cal |",
            "|:--|--:|--:|--:|--:|--:|--:|--:|",
        ]
    )
    for _, row in calibration_df.iterrows():
        lines.append(
            f"| {row['predictor']} | {fmt(row['valid_actual_rate'])} | {fmt(row['valid_raw_mean'])} | "
            f"{fmt(row['valid_traj_cal_mean'])} | {fmt(row['test_actual_rate'])} | "
            f"{fmt(row['test_raw_final_mean'])} | {fmt(row['test_prefix_cal_final_mean'])} | "
            f"{fmt(row['test_traj_cal_final_mean'])} |"
        )

    lines.extend(
        [
            "",
            "## Step-bucket AUC / accuracy",
            "",
            "| Predictor | Mode | Bucket | N | PosRate | MeanProb | Acc@0.5 | AUC | Brier |",
            "|:--|:--|:--|--:|--:|--:|--:|--:|--:|",
        ]
    )
    preferred_modes = ["raw", "trajectory_calibrated"]
    bucket_view = bucket_metrics_df[bucket_metrics_df["score_mode"].isin(preferred_modes)].copy()
    for _, row in bucket_view.iterrows():
        lines.append(
            f"| {row['predictor']} | {row['score_mode']} | {row['step_bucket']} | "
            f"{int(row['n_rows'])} | {fmt(row['actual_rate'])} | {fmt(row['mean_prob'])} | "
            f"{fmt(row['accuracy_at_0_5'])} | {fmt(row['roc_auc'])} | {fmt(row['brier'])} |"
        )

    (output_dir / "trajectory_calibration_step_report.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def _run_ranking_report(
    output_dir: Path,
    predictions_path: Path,
    predictors: list[str],
    max_cpu_threads: int,
) -> None:
    report_script = PROJECT_ROOT / "model_ranking_report_posthoc.py"
    ranking_dir = output_dir / "model_ranking_report_trajectory_calibrated"
    command = [
        sys.executable,
        str(report_script),
        "--predictions",
        str(predictions_path),
        "--output-dir",
        str(ranking_dir),
        "--score-mode",
        "calibrated",
        "--prefix-models",
        *predictors,
    ]
    LOGGER.info("Running trajectory-calibrated ranking report: %s", " ".join(command))
    subprocess.run(command, check=True, env=_thread_limited_env(max_cpu_threads))


def main() -> int:
    args = parse_args()
    max_cpu_threads = _set_cpu_thread_limits(args.max_cpu_threads)
    run_root = PROJECT_ROOT / "runs" / args.run_name
    _set_run_dirs(run_root)
    rebind_all_file_loggers()
    config.LGBM_PARAMS["num_threads"] = max_cpu_threads
    LOGGER.info("CPU thread cap: %s", max_cpu_threads)

    source_output_dir = config.REPORT_DIR / args.report_subdir
    output_dir = source_output_dir / "trajectory_calibrated_posthoc"
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix_path = args.prefix_table or config.PREFIX_TABLE_FILTERED_PATH
    prediction_path = source_output_dir / "test_predictions_shadow_valid_retrain.parquet"
    feature_engineer_path = config.MODEL_DIR / "feature_engineer_with_model.pkl"

    specs = _selected_specs(args.variants)
    predictors = [spec["predictor"] for spec in specs]

    feature_engineer = FeatureEngineer.load(feature_engineer_path)
    _repair_unpickled_tfidf_for_local_sklearn(feature_engineer)
    required_columns = _required_columns(feature_engineer, include_text=False)

    with timer(LOGGER, "Load metadata/dense prefix table and reconstruct split"):
        prefix_df = _load_prefix_table(prefix_path, required_columns)
        _, df_valid, _, split_meta, _ = _build_split(
            prefix_df,
            verified_jsonl=args.verified_jsonl,
            holdout_models=args.holdout_models,
            max_instances=args.max_instances,
            split_strategy="per_instance_model",
            valid_traj_ratio=0.15,
            valid_per_instance=0,
            valid_models_per_instance=args.valid_models_per_instance,
            shadow_valid_max_trajectories=0,
            seed=args.seed,
            smoke_trajectories_per_split=0,
        )
        del prefix_df
        gc.collect()

    valid_final_idx = df_valid.groupby("traj_id", sort=False)["prefix_step_idx"].idxmax()
    df_valid_final = df_valid.loc[valid_final_idx].copy()
    LOGGER.info(
        "Valid-final trajectories: %s; label rate=%.4f",
        len(df_valid_final),
        float(df_valid_final["label"].mean()),
    )

    matrices = _build_valid_final_matrices(
        prefix_path=prefix_path,
        feature_engineer=feature_engineer,
        df_valid_final=df_valid_final,
        text_batch_size=args.text_batch_size,
    )
    valid_raw = _predict_valid_final_raw(
        output_dir=source_output_dir,
        specs=specs,
        matrices=matrices,
        feature_engineer=feature_engineer,
    )
    del matrices
    gc.collect()

    pred_df = pd.read_parquet(prediction_path)
    calibrated_pred, calibration_df, final_metrics_df = _write_probability_reports(
        output_dir=output_dir,
        pred_df=pred_df,
        valid_final=df_valid_final,
        valid_raw=valid_raw,
        predictors=predictors,
    )
    exact_step_df, bucket_step_df = _step_metrics(calibrated_pred, predictors)
    exact_step_df.to_csv(output_dir / "exact_step_auc_accuracy.csv", index=False)
    bucket_step_df.to_csv(output_dir / "step_bucket_auc_accuracy.csv", index=False)
    _write_markdown_report(
        output_dir=output_dir,
        calibration_df=calibration_df,
        final_metrics_df=final_metrics_df,
        bucket_metrics_df=bucket_step_df,
    )
    (output_dir / "split_metadata_used.json").write_text(
        json.dumps(split_meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    if not args.skip_ranking_report:
        _run_ranking_report(
            output_dir,
            output_dir / "test_predictions_trajectory_calibrated.parquet",
            predictors,
            max_cpu_threads,
        )

    LOGGER.info("Trajectory calibration posthoc complete: %s", output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
