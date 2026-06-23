#!/usr/bin/env python3
"""Validation-only early-stop policy tuning.

This script is deliberately post-hoc and lightweight:

1. Reconstruct the same per-instance-model validation split for completed runs.
2. Predict all validation prefixes with already-trained LightGBM models.
3. Sweep early-stop policies on validation only.
4. Select one policy per run / predictor / score mode using validation metrics.
5. Apply the locked selected policy to the existing heldout-test predictions.

No LightGBM retraining is performed and validation prefix predictions are not
saved by default.
"""

from __future__ import annotations

import argparse
import gc
import math
import os
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
    os.environ.setdefault(_thread_env_name, os.environ.get("SWE_MAX_CPU_THREADS", "24"))
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from scipy import sparse

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
)
from process_signal_policy_rescue_posthoc import (
    DEFAULT_REPORT_SUBDIRS,
    MODEL_LABELS,
    RUN_LABELS,
    SCORE_LABELS,
    _policy_grid,
    _prediction_path,
    _shorten,
    evaluate,
)
from trainer import load_model
from utils import get_logger, rebind_all_file_loggers, timer


LOGGER = get_logger("valid_policy_tuning_posthoc")

DEFAULT_HOLDOUT_BY_REPORT = {
    "per_instance_model_valid3_retrain": "auto_mid3",
    "per_instance_model_valid3_top3_retrain": "auto_top3",
    "per_instance_model_valid3_bottom3_retrain": "auto_bottom3",
}


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", default="model_holdout_answer_calibrated_full")
    parser.add_argument("--report-subdirs", nargs="+", default=DEFAULT_REPORT_SUBDIRS)
    parser.add_argument(
        "--holdout-models",
        default=None,
        help="Override holdout model selector for every report subdir. Default uses subdir mapping.",
    )
    parser.add_argument("--prefix-table", type=Path, default=None)
    parser.add_argument(
        "--verified-jsonl",
        type=Path,
        default=PROJECT_ROOT.parents[2] / "swebench_verified" / "test.jsonl",
    )
    parser.add_argument("--max-instances", type=int, default=500)
    parser.add_argument("--valid-models-per-instance", type=int, default=3)
    parser.add_argument("--seed", type=int, default=config.SPLIT_SEED)
    parser.add_argument("--variants", nargs="+", default=["I", "J"])
    parser.add_argument(
        "--score-modes",
        nargs="+",
        default=["raw", "prefix_calibrated"],
        choices=["raw", "prefix_calibrated"],
    )
    parser.add_argument("--min-steps", nargs="+", type=int, default=[10, 15])
    parser.add_argument("--consecutive", nargs="+", type=int, default=[2, 3])
    parser.add_argument("--delta-thresholds", nargs="+", type=float, default=[0.0, 0.05])
    parser.add_argument("--text-batch-size", type=int, default=16384)
    parser.add_argument("--output-name", default="valid_policy_tuning")
    parser.add_argument("--max-valid-abs-drop-pp", type=float, default=2.0)
    parser.add_argument("--min-valid-decision-acc", type=float, default=0.90)
    parser.add_argument("--fallback-min-save-pct", type=float, default=5.0)
    parser.add_argument("--save-valid-predictions", action="store_true")
    parser.add_argument(
        "--max-cpu-threads",
        type=int,
        default=int(os.environ.get("SWE_MAX_CPU_THREADS", "24")),
    )
    return parser.parse_args()


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


def _transform_tfidf_single_streaming(
    *,
    prefix_table_path: Path,
    feature_engineer: FeatureEngineer,
    frame: pd.DataFrame,
    column_names: list[str],
    batch_size: int,
) -> sparse.csr_matrix:
    import pyarrow.parquet as parquet

    active_blocks: list[tuple[str, str, Any, Any]] = []
    for name in column_names:
        if name in feature_engineer.tfidf_vectorizers:
            active_blocks.append(
                (
                    name,
                    feature_engineer.active_text_columns[name],
                    feature_engineer.tfidf_vectorizers[name],
                    feature_engineer.tfidf_reducers.get(name),
                )
            )
    if not active_blocks:
        return sparse.csr_matrix((len(frame), 0))

    LOGGER.info(
        "Streaming %s validation TF-IDF block(s) in one parquet pass: %s",
        len(active_blocks),
        ", ".join(name for name, _, _, _ in active_blocks),
    )
    positions = pd.Series(
        np.arange(len(frame), dtype=np.int64),
        index=frame["prefix_id"].astype(str),
    )
    unique_text_columns = list(dict.fromkeys(text_column for _, text_column, _, _ in active_blocks))
    chunk_rows: dict[str, list[np.ndarray]] = {name: [] for name, _, _, _ in active_blocks}
    chunk_matrices: dict[str, list[sparse.spmatrix]] = {name: [] for name, _, _, _ in active_blocks}
    parquet_file = parquet.ParquetFile(prefix_table_path)

    matched_rows = 0
    for batch_idx, batch in enumerate(parquet_file.iter_batches(
        batch_size=batch_size,
        columns=["prefix_id", *unique_text_columns],
    ), start=1):
        batch_df = batch.to_pandas()
        batch_prefix_ids = batch_df["prefix_id"].astype(str)
        output_rows = positions.reindex(batch_prefix_ids).to_numpy()
        keep_mask = ~pd.isna(output_rows)
        if keep_mask.any():
            kept_rows = output_rows[keep_mask].astype(np.int64)
            matched_rows += int(len(kept_rows))
            for name, text_column, vectorizer, reducer in active_blocks:
                texts = batch_df.loc[keep_mask, text_column].fillna("")
                X_tfidf = vectorizer.transform(texts)
                del texts
                if reducer is not None:
                    X_tfidf = sparse.csr_matrix(
                        reducer.transform(X_tfidf).astype(np.float32)
                    )
                else:
                    X_tfidf = X_tfidf.tocsr()
                chunk_rows[name].append(kept_rows)
                chunk_matrices[name].append(X_tfidf)
        del batch_df
        if batch_idx % 100 == 0:
            LOGGER.info(
                "Scanned %s parquet batches for validation TF-IDF; matched rows so far=%s/%s",
                batch_idx,
                matched_rows,
                len(frame),
            )

    parts: list[sparse.spmatrix] = []
    for name, _, _, _ in active_blocks:
        if not chunk_matrices[name]:
            raise RuntimeError(f"No rows matched validation TF-IDF block {name}.")
        rows = np.concatenate(chunk_rows[name])
        matrix = sparse.vstack(chunk_matrices[name], format="csr")
        order = np.argsort(rows)
        rows = rows[order]
        matrix = matrix[order]
        expected = np.arange(len(frame), dtype=np.int64)
        if len(rows) != len(frame) or not np.array_equal(rows, expected):
            raise RuntimeError(
                f"Validation TF-IDF row alignment failed for block={name}: "
                f"matched={len(rows)} expected={len(frame)}"
            )
        parts.append(matrix)
        del rows, matrix
        gc.collect()
    del chunk_rows, chunk_matrices, parquet_file
    return sparse.hstack(parts, format="csr")


def _build_valid_matrices(
    *,
    prefix_path: Path,
    feature_engineer: FeatureEngineer,
    df_valid: pd.DataFrame,
    specs: list[dict[str, Any]],
    text_batch_size: int,
) -> dict[str, sparse.csr_matrix]:
    tfidf_af_cols = list(TFIDF_ACTION_FEEDBACK.keys())
    tfidf_thought_cols = list(TFIDF_THOUGHT.keys())
    needs_thought = any(spec["base"] == "af_thought" for spec in specs)
    with timer(LOGGER, "Build validation Dense / AF / Thought matrices"):
        X_dense = sparse.csr_matrix(feature_engineer.transform_dense(df_valid))
        X_af = _transform_tfidf_single_streaming(
            prefix_table_path=prefix_path,
            feature_engineer=feature_engineer,
            frame=df_valid,
            column_names=tfidf_af_cols,
            batch_size=text_batch_size,
        )
        if needs_thought:
            X_thought = _transform_tfidf_single_streaming(
                prefix_table_path=prefix_path,
                feature_engineer=feature_engineer,
                frame=df_valid,
                column_names=tfidf_thought_cols,
                batch_size=text_batch_size,
            )
        else:
            X_thought = sparse.csr_matrix((len(df_valid), 0))
    return {"dense": X_dense, "af": X_af, "thought": X_thought}


def _predict_valid_prefixes(
    *,
    report_dir: Path,
    specs: list[dict[str, Any]],
    matrices: dict[str, sparse.csr_matrix],
    feature_engineer: FeatureEngineer,
    df_valid: pd.DataFrame,
) -> pd.DataFrame:
    names_af, names_j = _feature_names(feature_engineer)
    pred_df = df_valid[
        [
            "prefix_id",
            "traj_id",
            "instance_id",
            "group_id",
            "prefix_step_idx",
            "n_steps_total_for_weighting",
            "sample_weight",
            "label",
            "split",
            "model_id",
            "model",
            "orig_model_id",
            "orig_model",
            "model_id_input_mode",
        ]
    ].copy()
    model_dir = report_dir / "models"

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

        LOGGER.info("Predicting validation prefixes: %s", predictor)
        booster = load_model(model_path)
        raw = np.asarray(booster.predict(X_model), dtype=np.float64)
        pred_df[f"prob__{predictor}"] = raw.astype(np.float32)

        calibrator_path = model_dir / f"calibrator_{_safe_name(predictor)}.pkl"
        if calibrator_path.is_file():
            calibrator = load_model(calibrator_path)
            pred_df[f"prob_cal__{predictor}"] = calibrator.predict(raw).astype(np.float32)
        else:
            LOGGER.warning("Missing calibrator: %s", calibrator_path)

        del booster, X_model, raw
        gc.collect()

    return pred_df


def _resolve_holdout_for_report(report_subdir: str, override: str | None) -> str:
    if override:
        return override
    return DEFAULT_HOLDOUT_BY_REPORT.get(report_subdir, "auto_mid3")


def _has_required_valid_prediction_columns(
    prediction_path: Path,
    *,
    predictors: list[str],
    score_modes: list[str],
) -> bool:
    if not prediction_path.is_file():
        return False
    try:
        import pyarrow.parquet as parquet

        columns = set(parquet.ParquetFile(prediction_path).schema_arrow.names)
    except Exception:
        try:
            columns = set(pd.read_parquet(prediction_path).columns)
        except Exception:
            return False
    required = {
        "traj_id",
        "orig_model_id",
        "label",
        "prefix_step_idx",
    }
    for predictor in predictors:
        if "raw" in score_modes:
            required.add(f"prob__{predictor}")
        if "prefix_calibrated" in score_modes:
            required.add(f"prob_cal__{predictor}")
    return required.issubset(columns)


def _policy_key(row: pd.Series) -> dict[str, Any]:
    return {
        "policy_mode": row["policy_mode"],
        "success_thr": float(row["success_thr"]),
        "failure_thr": float(row["failure_thr"]),
        "min_step": int(row["min_step"]),
        "consecutive": int(row["consecutive"]),
        "delta_up": float(row["delta_up"]),
        "delta_down": float(row["delta_down"]),
    }


def _select_policies(
    valid_aggregate: pd.DataFrame,
    *,
    max_valid_abs_drop_pp: float,
    min_valid_decision_acc: float,
    fallback_min_save_pct: float,
) -> pd.DataFrame:
    selected_rows: list[dict[str, Any]] = []
    if valid_aggregate.empty:
        return pd.DataFrame()

    for group_key, part in valid_aggregate.groupby(["run", "score_mode", "prefix_model"], sort=False):
        work = part.copy()
        work["drop_pp"] = work["resolve_rate_drop"] * 100.0
        work["abs_drop_pp"] = work["drop_pp"].abs()
        work["decision_accuracy_for_filter"] = work["decision_accuracy"].fillna(-1.0)
        work["pct_steps_saved_for_sort"] = work["pct_steps_saved"].fillna(0.0)

        strict = work[
            (work["abs_drop_pp"] <= max_valid_abs_drop_pp)
            & (work["decision_accuracy_for_filter"] >= min_valid_decision_acc)
            & (work["pct_steps_saved_for_sort"] > 0.0)
        ].copy()
        if not strict.empty:
            chosen = strict.sort_values(
                [
                    "pct_steps_saved_for_sort",
                    "abs_drop_pp",
                    "decision_accuracy_for_filter",
                    "coverage",
                ],
                ascending=[False, True, False, False],
            ).iloc[0]
            status = "valid_constraints_pass"
        else:
            fallback = work[work["pct_steps_saved_for_sort"] >= fallback_min_save_pct].copy()
            if fallback.empty:
                fallback = work
            chosen = fallback.sort_values(
                [
                    "abs_drop_pp",
                    "pct_steps_saved_for_sort",
                    "decision_accuracy_for_filter",
                    "coverage",
                ],
                ascending=[True, False, False, False],
            ).iloc[0]
            status = "fallback_min_abs_valid_drop"

        policy_id = (
            f"{group_key[0]}__{group_key[1]}__{_safe_name(group_key[2])}"
        )
        row = chosen.to_dict()
        row["policy_id"] = policy_id
        row["selection_status"] = status
        row["valid_abs_drop_pp"] = float(chosen["abs_drop_pp"])
        selected_rows.append(row)

    return pd.DataFrame(selected_rows)


def _evaluate_selected_on_test(
    *,
    run_name: str,
    report_subdir: str,
    selected: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if selected.empty:
        return pd.DataFrame(), pd.DataFrame()
    report_dir = config.REPORT_DIR / report_subdir
    aggregate_rows: list[pd.DataFrame] = []
    per_agent_rows: list[pd.DataFrame] = []
    loaded: dict[str, pd.DataFrame] = {}

    for _, row in selected.iterrows():
        score_mode = str(row["score_mode"])
        if score_mode not in loaded:
            pred_path = _prediction_path(report_dir, score_mode)
            if not pred_path.is_file():
                LOGGER.warning("Skipping missing test predictions: %s", pred_path)
                continue
            loaded[score_mode] = pd.read_parquet(pred_path)
        policy = _policy_key(row)
        aggregate, per_agent = evaluate(
            loaded[score_mode],
            run_label=report_subdir,
            score_mode=score_mode,
            prefix_models=[str(row["prefix_model"])],
            policies=[policy],
        )
        aggregate["policy_id"] = row["policy_id"]
        per_agent["policy_id"] = row["policy_id"]
        aggregate_rows.append(aggregate)
        per_agent_rows.append(per_agent)

    del loaded
    gc.collect()
    aggregate_df = pd.concat(aggregate_rows, ignore_index=True) if aggregate_rows else pd.DataFrame()
    per_agent_df = pd.concat(per_agent_rows, ignore_index=True) if per_agent_rows else pd.DataFrame()
    return aggregate_df, per_agent_df


def _prefix_columns(df: pd.DataFrame, prefix: str, skip: set[str]) -> pd.DataFrame:
    out = df.copy()
    rename = {col: f"{prefix}{col}" for col in out.columns if col not in skip}
    return out.rename(columns=rename)


def _fmt(value: Any, digits: int = 1) -> str:
    try:
        value = float(value)
    except Exception:
        return "-"
    if math.isnan(value) or math.isinf(value):
        return "-"
    return f"{value:.{digits}f}"


def _write_report(
    *,
    output_dir: Path,
    selected: pd.DataFrame,
    test_selected: pd.DataFrame,
    max_valid_abs_drop_pp: float,
    min_valid_decision_acc: float,
) -> None:
    selected_short = _shorten(selected) if not selected.empty else selected
    test_short = _shorten(test_selected) if not test_selected.empty else test_selected
    join_cols = ["policy_id", "run", "score_mode", "prefix_model"]
    valid_view = _prefix_columns(
        selected_short,
        "valid_",
        skip=set(join_cols + ["run_short", "score_short", "model_short", "policy_label"]),
    )
    test_view = _prefix_columns(
        test_short,
        "test_",
        skip=set(join_cols + ["run_short", "score_short", "model_short", "policy_label"]),
    )
    merged = valid_view.merge(
        test_view,
        on=join_cols,
        how="left",
        suffixes=("", "_test"),
    )

    lines: list[str] = [
        "# Valid-Only Policy Tuning",
        "",
        "策略选择只看 validation：test 只用于最后一次 locked-policy 评估。",
        "",
        (
            f"Selection rule: maximize valid Save% subject to "
            f"`|valid Drop| <= {max_valid_abs_drop_pp:.1f}pp` and "
            f"`valid decision Acc >= {min_valid_decision_acc * 100:.1f}%`; "
            "if no policy passes, use fallback with minimum valid |Drop|."
        ),
        "",
        "## Selected Policies and Test Result",
        "",
        "| Run | Model | Score | Status | Mode | S_thr | F_thr | Min | K | Delta | Valid Save | Valid Drop pp | Valid Acc | Test Save | Test Drop pp | Test Acc | Test FN | Test FP |",
        "|:--|:--|:--|:--|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|",
    ]

    if merged.empty:
        lines.append("| - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - |")
    else:
        merged["run_short"] = merged["run"].map(RUN_LABELS).fillna(merged["run"])
        merged["score_short"] = merged["score_mode"].map(SCORE_LABELS).fillna(merged["score_mode"])
        merged["model_short"] = merged["prefix_model"].map(MODEL_LABELS).fillna(merged["prefix_model"])
        merged = merged.sort_values(["run_short", "model_short", "score_short"])
        for _, row in merged.iterrows():
            delta = max(float(row["valid_delta_up"]), float(row["valid_delta_down"]))
            valid_drop_pp = float(row["valid_resolve_rate_drop"]) * 100.0
            test_drop_pp = float(row["test_resolve_rate_drop"]) * 100.0 if not pd.isna(row.get("test_resolve_rate_drop")) else math.nan
            lines.append(
                "| "
                f"{row['run_short']} | {row['model_short']} | {row['score_short']} | "
                f"{row['valid_selection_status']} | {row['valid_policy_mode']} | "
                f"{_fmt(row['valid_success_thr'], 2)} | {_fmt(row['valid_failure_thr'], 2)} | "
                f"{int(row['valid_min_step'])} | {int(row['valid_consecutive'])} | "
                f"{_fmt(delta, 2)} | {_fmt(row['valid_pct_steps_saved'])}% | "
                f"{_fmt(valid_drop_pp)} | {_fmt(float(row['valid_decision_accuracy']) * 100.0)}% | "
                f"{_fmt(row.get('test_pct_steps_saved', math.nan))}% | {_fmt(test_drop_pp)} | "
                f"{_fmt(float(row.get('test_decision_accuracy', math.nan)) * 100.0)}% | "
                f"{int(row.get('test_false_negatives', 0)) if not pd.isna(row.get('test_false_negatives', math.nan)) else '-'} | "
                f"{int(row.get('test_false_positives', 0)) if not pd.isna(row.get('test_false_positives', math.nan)) else '-'} |"
            )

    lines.extend(
        [
            "",
            "## Output Files",
            "",
            "- `valid_policy_grid.csv`: all validation-swept policies.",
            "- `valid_policy_per_agent.csv`: validation metrics per validation model.",
            "- `selected_policies.csv`: policies selected from validation only.",
            "- `test_results_for_selected_policies.csv`: locked selected policies on heldout test.",
            "- `test_per_agent_for_selected_policies.csv`: heldout-test per-agent breakdown.",
        ]
    )
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    max_cpu_threads = _set_cpu_thread_limits(args.max_cpu_threads)
    run_root = PROJECT_ROOT / "runs" / args.run_name
    _set_run_dirs(run_root)
    rebind_all_file_loggers()
    config.LGBM_PARAMS["num_threads"] = max_cpu_threads
    LOGGER.info("CPU thread cap: %s", max_cpu_threads)

    output_dir = config.REPORT_DIR / args.output_name
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix_path = args.prefix_table or config.PREFIX_TABLE_FILTERED_PATH
    feature_engineer_path = config.MODEL_DIR / "feature_engineer_with_model.pkl"
    specs = _selected_specs(args.variants)
    predictors = [spec["predictor"] for spec in specs]
    policies = _policy_grid(
        min_steps=args.min_steps,
        consecutive_values=args.consecutive,
        delta_thresholds=args.delta_thresholds,
    )

    feature_engineer = FeatureEngineer.load(feature_engineer_path)
    _repair_unpickled_tfidf_for_local_sklearn(feature_engineer)
    required_columns = _required_columns(feature_engineer, include_text=False)

    all_valid_aggregate: list[pd.DataFrame] = []
    all_valid_per_agent: list[pd.DataFrame] = []
    all_selected: list[pd.DataFrame] = []
    all_test_aggregate: list[pd.DataFrame] = []
    all_test_per_agent: list[pd.DataFrame] = []

    for report_subdir in args.report_subdirs:
        report_dir = config.REPORT_DIR / report_subdir
        holdout_models = _resolve_holdout_for_report(report_subdir, args.holdout_models)
        LOGGER.info("Processing %s with holdout_models=%s", report_subdir, holdout_models)

        saved_valid_path = report_dir / "valid_predictions_shadow_valid_retrain.parquet"
        if _has_required_valid_prediction_columns(
            saved_valid_path,
            predictors=predictors,
            score_modes=args.score_modes,
        ):
            LOGGER.info("Using saved valid predictions: %s", saved_valid_path)
            valid_pred = pd.read_parquet(saved_valid_path)
            split_meta = {
                "valid_trajectories": int(valid_pred["traj_id"].nunique()),
                "valid_rows": int(len(valid_pred)),
            }
        else:
            with timer(LOGGER, f"Load metadata/dense prefix table and reconstruct valid split for {report_subdir}"):
                prefix_df = _load_prefix_table(prefix_path, required_columns)
                _, df_valid, _, split_meta, _ = _build_split(
                    prefix_df,
                    verified_jsonl=args.verified_jsonl,
                    holdout_models=holdout_models,
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
            LOGGER.info(
                "%s valid: %s trajectories, %s prefixes, label_rate=%.4f",
                report_subdir,
                df_valid["traj_id"].nunique(),
                len(df_valid),
                float(df_valid["label"].mean()),
            )

            matrices = _build_valid_matrices(
                prefix_path=prefix_path,
                feature_engineer=feature_engineer,
                df_valid=df_valid,
                specs=specs,
                text_batch_size=args.text_batch_size,
            )
            valid_pred = _predict_valid_prefixes(
                report_dir=report_dir,
                specs=specs,
                matrices=matrices,
                feature_engineer=feature_engineer,
                df_valid=df_valid,
            )
            if args.save_valid_predictions:
                valid_pred.to_parquet(output_dir / f"valid_predictions_{_safe_name(report_subdir)}.parquet", index=False)

            del matrices, df_valid
            gc.collect()

        run_valid_aggregate: list[pd.DataFrame] = []
        run_valid_per_agent: list[pd.DataFrame] = []
        for score_mode in args.score_modes:
            aggregate, per_agent = evaluate(
                valid_pred,
                run_label=report_subdir,
                score_mode=score_mode,
                prefix_models=predictors,
                policies=policies,
            )
            run_valid_aggregate.append(aggregate)
            run_valid_per_agent.append(per_agent)

        valid_aggregate = pd.concat(run_valid_aggregate, ignore_index=True)
        valid_per_agent = pd.concat(run_valid_per_agent, ignore_index=True)
        valid_aggregate["split_holdout_models"] = holdout_models
        valid_aggregate["valid_trajectories"] = int(split_meta["valid_trajectories"])
        valid_aggregate["valid_prefixes"] = int(split_meta["valid_rows"])
        valid_per_agent["split_holdout_models"] = holdout_models
        all_valid_aggregate.append(valid_aggregate)
        all_valid_per_agent.append(valid_per_agent)

        selected = _select_policies(
            valid_aggregate,
            max_valid_abs_drop_pp=args.max_valid_abs_drop_pp,
            min_valid_decision_acc=args.min_valid_decision_acc,
            fallback_min_save_pct=args.fallback_min_save_pct,
        )
        if not selected.empty:
            selected["split_holdout_models"] = holdout_models
            all_selected.append(selected)
            test_aggregate, test_per_agent = _evaluate_selected_on_test(
                run_name=args.run_name,
                report_subdir=report_subdir,
                selected=selected,
            )
            all_test_aggregate.append(test_aggregate)
            all_test_per_agent.append(test_per_agent)

        del valid_pred
        gc.collect()

    valid_aggregate_df = pd.concat(all_valid_aggregate, ignore_index=True) if all_valid_aggregate else pd.DataFrame()
    valid_per_agent_df = pd.concat(all_valid_per_agent, ignore_index=True) if all_valid_per_agent else pd.DataFrame()
    selected_df = pd.concat(all_selected, ignore_index=True) if all_selected else pd.DataFrame()
    test_aggregate_df = pd.concat(all_test_aggregate, ignore_index=True) if all_test_aggregate else pd.DataFrame()
    test_per_agent_df = pd.concat(all_test_per_agent, ignore_index=True) if all_test_per_agent else pd.DataFrame()

    if not valid_aggregate_df.empty:
        _shorten(valid_aggregate_df).to_csv(output_dir / "valid_policy_grid.csv", index=False)
    else:
        valid_aggregate_df.to_csv(output_dir / "valid_policy_grid.csv", index=False)
    if not valid_per_agent_df.empty:
        _shorten(valid_per_agent_df).to_csv(output_dir / "valid_policy_per_agent.csv", index=False)
    else:
        valid_per_agent_df.to_csv(output_dir / "valid_policy_per_agent.csv", index=False)
    if not selected_df.empty:
        _shorten(selected_df).to_csv(output_dir / "selected_policies.csv", index=False)
    else:
        selected_df.to_csv(output_dir / "selected_policies.csv", index=False)
    if not test_aggregate_df.empty:
        _shorten(test_aggregate_df).to_csv(output_dir / "test_results_for_selected_policies.csv", index=False)
    else:
        test_aggregate_df.to_csv(output_dir / "test_results_for_selected_policies.csv", index=False)
    if not test_per_agent_df.empty:
        _shorten(test_per_agent_df).to_csv(output_dir / "test_per_agent_for_selected_policies.csv", index=False)
    else:
        test_per_agent_df.to_csv(output_dir / "test_per_agent_for_selected_policies.csv", index=False)

    _write_report(
        output_dir=output_dir,
        selected=selected_df,
        test_selected=test_aggregate_df,
        max_valid_abs_drop_pp=args.max_valid_abs_drop_pp,
        min_valid_decision_acc=args.min_valid_decision_acc,
    )
    print(f"Saved valid-only policy tuning results: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
