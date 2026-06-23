#!/usr/bin/env python3
from __future__ import annotations
"""
SWE-smith Prefix Success Prediction — 主流程脚本

完整流水线：
  1. 读取 tool split parquet → 重建 step_table
  2. 构建 prefix_table（含全部手工特征 A~H）
  3. 按 instance_id（默认）或 traj group_id 切分 train/valid/test
  4. 特征化（dense + TF-IDF）
  5. 训练四个 baseline 模型
  6. 评估 + ablation + 生成报告

用法:
  export SWE_PARQUET_DIR=/path/to/parquet_dir
  python run_all.py

  # 或直接指定路径（bash 多模型/多轨迹数据务必按题目切分，避免同一 instance 泄漏到多个 split）
  python run_all.py --data-dir /path/to/parquet_dir --split-by instance

  # 去重后随机抽 N 条轨迹做快速试验（可复现）
  python run_all.py --run-name smoke --max-trajectories 2000 --sample-trajectories-seed 42 --split-by instance

  # 跳过某些阶段
  python run_all.py --skip-step-table   # 如果已有 step_table.parquet
  python run_all.py --skip-prefix-table # 如果已有 prefix_table.parquet
"""
import argparse
import os
import re
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import precision_recall_curve, roc_curve

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# 确保项目根目录在 path 中
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

import config
from answer_features import enrich_prefix_with_answer_features
from utils import get_logger, timer, save_json, rebind_all_file_loggers
from step_builder import build_step_table, _iter_tool_parquet_files
from prefix_builder import build_prefix_table
from feature_engineer import (
    FeatureEngineer, NUMERIC_FEATURES, BOOL_FEATURES, CATEGORICAL_FEATURES,
)
from data_split import split_by_group
from model_holdout_split import select_model_holdout_split, write_model_holdout_summary
from trainer import train_logistic_regression, train_lightgbm, save_model
from probability_calibration import fit_sigmoid_calibrator, calibration_summary_row
from evaluator import (
    compute_metrics, compute_bucketed_metrics,
    compute_threshold_decision_table,
    compute_trajectory_level_savings,
    compute_trajectory_savings_at_precision_levels,
    compute_feature_group_contribution,
    per_step_accuracy_dgkn,
    plot_calibration, plot_roc_pr, plot_metrics_by_step,
    plot_feature_importance_lr, plot_feature_importance_lgbm,
    generate_full_report,
)

logger = get_logger("run_all")


def _validate_prefix_group_integrity(prefix_df: pd.DataFrame):
    """
    强校验：
    1) (group_id, prefix_step_idx) 必须唯一
    2) 每个 group_id 只能映射到一个 traj_id
    """
    dup_mask = prefix_df.duplicated(subset=["group_id", "prefix_step_idx"], keep=False)
    dup_n = int(dup_mask.sum())
    if dup_n > 0:
        dup_rows = prefix_df.loc[dup_mask, ["group_id", "traj_id", "prefix_step_idx", "label"]]
        examples = dup_rows.sort_values(["group_id", "prefix_step_idx"]).head(10).to_dict("records")
        raise ValueError(
            "Prefix group integrity check failed: duplicated (group_id, prefix_step_idx) detected. "
            f"duplicate_rows={dup_n}, examples={examples}"
        )

    group_traj = prefix_df.groupby("group_id")["traj_id"].nunique()
    multi_traj_groups = int((group_traj > 1).sum())
    if multi_traj_groups > 0:
        raise ValueError(
            "Prefix group integrity check failed: one group_id maps to multiple trajectories. "
            f"multi_traj_groups={multi_traj_groups}"
        )


def parse_args():
    ap = argparse.ArgumentParser(description="SWE-smith Prefix Success Prediction Pipeline")
    ap.add_argument("--data-dir", type=str, default=None,
                    help="Path to directory containing tool-*.parquet files")
    ap.add_argument("--single-parquet", type=str, default=None,
                    help="Use a single tool-*.parquet file for quick validation")
    ap.add_argument("--quick-verify", action="store_true",
                    help="Quick validation mode: auto pick one parquet, skip LightGBM and ablation")
    ap.add_argument("--skip-step-table", action="store_true",
                    help="Skip step table building (use existing)")
    ap.add_argument("--skip-prefix-table", action="store_true",
                    help="Skip prefix table building (use existing)")
    ap.add_argument("--skip-lgbm", action="store_true",
                    help="Skip LightGBM training")
    ap.add_argument("--skip-ablation", action="store_true",
                    help="Skip ablation experiments")
    ap.add_argument("--no-gpu-lgbm", action="store_true",
                    help="Use CPU for LightGBM instead of GPU")
    ap.add_argument("--run-name", type=str, default=None,
                    help="Run name for output isolation. Creates runs/<name>/{data,models,reports,logs}")
    ap.add_argument(
        "--max-trajectories",
        type=int,
        default=None,
        metavar="N",
        help="After instance dedup, only use at most N trajectories (faster smoke test). "
        "Default order is dedup order; use --sample-trajectories-seed for reproducible random subset. "
        "Combine with --quick-verify for minimal runtime.",
    )
    ap.add_argument(
        "--sample-trajectories-seed",
        type=int,
        default=None,
        metavar="SEED",
        help="With --max-trajectories, shuffle deduplicated trajectories with this seed before taking the first N.",
    )
    ap.add_argument(
        "--split-by",
        type=str,
        choices=("trajectory", "instance", "model_holdout"),
        default="instance",
        help="instance（默认）: 按 instance_id 划分，同一 SWE 题目所有轨迹只在 train/valid/test 之一（bash 多模型必选）。"
        "trajectory: 按 traj（group_id）划分，允许同一题目多条轨迹分到不同集合（易泄漏，仅用于对照实验）。"
        "model_holdout: 按模型留出测试，训练模型身份 one-hot，测试模型身份映射到 __MISSING__。",
    )
    ap.add_argument(
        "--verified-jsonl",
        type=str,
        default=str(PROJECT_ROOT.parents[2] / "swebench_verified" / "test.jsonl"),
        help="SWE-bench verified test.jsonl used for model_holdout filtering and gold answer features.",
    )
    ap.add_argument(
        "--holdout-models",
        type=str,
        default="auto_mid3",
        help="For --split-by model_holdout: auto_mid3 or comma-separated model ids.",
    )
    ap.add_argument(
        "--max-instances",
        type=int,
        default=500,
        help="For --split-by model_holdout: max verified instances to keep after parquet overlap.",
    )
    ap.add_argument(
        "--disable-answer-features",
        action="store_true",
        help="Do not join gold-answer features. Intended only for debugging.",
    )
    ap.add_argument(
        "--reuse-answer-enriched",
        action="store_true",
        help="If data/prefix_table_answer_enriched.parquet exists, load it instead of recomputing gold-answer enrichment.",
    )
    ap.add_argument(
        "--reuse-feature-engineers",
        action="store_true",
        help="If saved FeatureEngineer pickles exist, load them instead of refitting TF-IDF/SVD.",
    )
    ap.add_argument(
        "--disable-prob-calibration",
        action="store_true",
        help="Disable validation-only probability calibration columns prob_cal__*. Raw prob__* columns are always kept.",
    )
    return ap.parse_args()


def _save_test_artifacts(
    df_test: pd.DataFrame,
    y_test: np.ndarray,
    all_predictions: dict[str, np.ndarray],
    save_dir: Path,
    *,
    calibrated_predictions: dict[str, np.ndarray] | None = None,
    scores_for_rank_curves: dict[str, np.ndarray] | None = None,
):
    """
    保存测试集逐样本预测与曲线点数据，便于复用和二次分析。

    scores_for_rank_curves:
        若提供，ROC/PR 曲线 CSV 用该分数排序（与 eval 中 LR 的 decision_function 一致）；
        未给出的模型仍用 predict_proba。
    """
    save_dir.mkdir(parents=True, exist_ok=True)
    curve_dir = save_dir / "curve_data"
    curve_dir.mkdir(parents=True, exist_ok=True)

    preferred_cols = [
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
    base_cols = [c for c in preferred_cols if c in df_test.columns]
    pred_df = df_test[base_cols].copy() if base_cols else pd.DataFrame(index=df_test.index)

    for model_name, y_prob in all_predictions.items():
        pred_df[f"prob__{model_name}"] = np.asarray(y_prob, dtype=np.float32)

        y_curve = (
            np.asarray(scores_for_rank_curves[model_name], dtype=np.float64)
            if scores_for_rank_curves is not None and model_name in scores_for_rank_curves
            else y_prob
        )
        fpr, tpr, roc_thr = roc_curve(y_test, y_curve)
        roc_df = pd.DataFrame({
            "fpr": fpr,
            "tpr": tpr,
            "threshold": roc_thr,
            "model": model_name,
        })
        roc_df.to_csv(curve_dir / f"roc_curve_points_{model_name}.csv", index=False)

        precision, recall, pr_thr = precision_recall_curve(y_test, y_curve)
        pr_df = pd.DataFrame({
            "recall": recall,
            "precision": precision,
            "model": model_name,
        })
        if len(pr_thr) > 0:
            pr_df["threshold"] = np.append(pr_thr, np.nan)
        else:
            pr_df["threshold"] = np.nan
        pr_df.to_csv(curve_dir / f"pr_curve_points_{model_name}.csv", index=False)

    if calibrated_predictions:
        for model_name, y_prob in calibrated_predictions.items():
            pred_df[f"prob_cal__{model_name}"] = np.asarray(y_prob, dtype=np.float32)

    pred_csv = save_dir / "test_predictions_all_models.csv"
    pred_parquet = save_dir / "test_predictions_all_models.parquet"
    pred_df.to_csv(pred_csv, index=False)
    pred_df.to_parquet(pred_parquet, index=False)
    logger.info(f"Saved test predictions: {pred_csv}")
    logger.info(f"Saved test predictions: {pred_parquet}")

    # 多模型合并曲线点（便于一次性读取）
    all_roc_rows = []
    all_pr_rows = []
    for model_name, y_prob in all_predictions.items():
        y_curve = (
            np.asarray(scores_for_rank_curves[model_name], dtype=np.float64)
            if scores_for_rank_curves is not None and model_name in scores_for_rank_curves
            else y_prob
        )
        fpr, tpr, roc_thr = roc_curve(y_test, y_curve)
        all_roc_rows.append(pd.DataFrame({
            "model": model_name,
            "fpr": fpr,
            "tpr": tpr,
            "threshold": roc_thr,
        }))

        precision, recall, pr_thr = precision_recall_curve(y_test, y_curve)
        pr_part = pd.DataFrame({
            "model": model_name,
            "recall": recall,
            "precision": precision,
        })
        pr_part["threshold"] = np.append(pr_thr, np.nan) if len(pr_thr) > 0 else np.nan
        all_pr_rows.append(pr_part)

    if all_roc_rows:
        pd.concat(all_roc_rows, ignore_index=True).to_csv(
            save_dir / "roc_curve_points_all_models.csv", index=False
        )
    if all_pr_rows:
        pd.concat(all_pr_rows, ignore_index=True).to_csv(
            save_dir / "pr_curve_points_all_models.csv", index=False
        )

    try:
        acc_df = per_step_accuracy_dgkn(pred_df, y_test)
        acc_path = save_dir / "per_step_accuracy_DGKN.csv"
        acc_df.to_csv(acc_path, index=False)
        logger.info(
            f"Saved per-step D/G/K/N table (mean_prob__* + accuracy__* @ 0.5): {acc_path}"
        )
    except Exception as e:
        logger.warning(f"per_step_accuracy_DGKN skipped: {e}")


def _save_metrics_by_heldout_model(
    df_test: pd.DataFrame,
    y_test: np.ndarray,
    all_predictions: dict[str, np.ndarray],
    save_path: Path,
):
    """Save per-original-model metrics for model-holdout evaluation."""
    group_col = "orig_model_id" if "orig_model_id" in df_test.columns else "model_id"
    rows = []
    for heldout_model, idx in df_test.groupby(group_col, sort=True).groups.items():
        positions = df_test.index.get_indexer(idx)
        if len(positions) == 0:
            continue
        for model_name, y_prob in all_predictions.items():
            metrics = compute_metrics(y_test[positions], np.asarray(y_prob)[positions])
            rows.append({
                "heldout_model": heldout_model,
                "predictor": model_name,
                "prefixes": int(len(positions)),
                "instances": int(df_test.loc[idx, "instance_id"].nunique()),
                "trajectories": int(df_test.loc[idx, "traj_id"].nunique()) if "traj_id" in df_test else 0,
                "roc_auc": metrics.get("roc_auc"),
                "pr_auc": metrics.get("pr_auc"),
                "log_loss": metrics.get("log_loss"),
                "brier_score": metrics.get("brier_score"),
                "pos_rate": metrics.get("pos_rate"),
            })
    out = pd.DataFrame(rows)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(save_path, index=False)
    logger.info(f"Saved metrics by heldout model: {save_path}")


def _transform_tfidf_blocks_once(
    fe: FeatureEngineer,
    df: pd.DataFrame,
    column_names: list[str],
    split_name: str,
) -> dict[str, sparse.csr_matrix]:
    """Transform each TF-IDF block once so AF/Thought/Full subsets can reuse it."""
    blocks: dict[str, sparse.csr_matrix] = {}
    for name in column_names:
        X_block = fe.transform_tfidf_subset(df, [name])
        blocks[name] = X_block
        logger.info(f"TF-IDF block {split_name}/{name} shape: {X_block.shape}")
    return blocks


def _stack_tfidf_blocks(
    blocks: dict[str, sparse.csr_matrix],
    column_names: list[str],
    n_rows: int,
) -> sparse.csr_matrix:
    parts = [blocks[name] for name in column_names if name in blocks]
    if not parts:
        return sparse.csr_matrix((n_rows, 0))
    return sparse.hstack(parts, format="csr")


def _train_lightgbm_with_fallback(
    *,
    X_train,
    y_train,
    w_train,
    X_valid,
    y_valid,
    w_valid,
    X_test,
    feature_names,
    model_name: str,
    model_path: Path,
):
    """
    训练 LightGBM 并在 GPU 失败时自动回退 CPU，返回 (booster, y_test_prob)。
    """
    original_params = dict(config.LGBM_PARAMS)
    try:
        booster = train_lightgbm(
            X_train=X_train,
            y_train=y_train,
            w_train=w_train,
            X_valid=X_valid,
            y_valid=y_valid,
            w_valid=w_valid,
            feature_names=feature_names,
            model_name=model_name,
        )
        y_prob = booster.predict(X_test)
        save_model(booster, model_path)
        return booster, y_prob
    except Exception as e:
        logger.error(f"[{model_name}] LightGBM training failed: {e}")
        logger.info(f"[{model_name}] Trying LightGBM with CPU...")
        config.LGBM_PARAMS["device"] = "cpu"
        if "gpu_device_id" in config.LGBM_PARAMS:
            del config.LGBM_PARAMS["gpu_device_id"]
        try:
            booster = train_lightgbm(
                X_train=X_train,
                y_train=y_train,
                w_train=w_train,
                X_valid=X_valid,
                y_valid=y_valid,
                w_valid=w_valid,
                feature_names=feature_names,
                model_name=model_name,
            )
            y_prob = booster.predict(X_test)
            save_model(booster, model_path)
            return booster, y_prob
        finally:
            config.LGBM_PARAMS.clear()
            config.LGBM_PARAMS.update(original_params)


def _run_ablation_lgbm(
    args,
    all_models: dict,
    all_predictions: dict,
    *,
    public_name: str,
    train_slug: str,
    X_train,
    X_valid,
    X_test,
    feature_names: list[str],
    y_train: np.ndarray,
    y_valid: np.ndarray,
    w_train: np.ndarray,
    w_valid: np.ndarray,
    register_prediction=None,
):
    """与每项 LR 消融同特征矩阵的 LightGBM；--skip-lgbm 时跳过。"""
    if args.skip_lgbm:
        return
    n_feat = int(X_train.shape[1])
    if len(feature_names) != n_feat:
        logger.warning(
            f"[{public_name}] feature_names len {len(feature_names)} != n_features {n_feat}; skip LGBM"
        )
        return
    model_path = config.MODEL_DIR / f"ablation_lgbm_{train_slug}.lgb"
    try:
        booster, y_prob = _train_lightgbm_with_fallback(
            X_train=X_train,
            y_train=y_train,
            w_train=w_train,
            X_valid=X_valid,
            y_valid=y_valid,
            w_valid=w_valid,
            X_test=X_test,
            feature_names=feature_names,
            model_name=f"abl_lgbm_{train_slug}",
            model_path=model_path,
        )
        all_models[public_name] = booster
        if register_prediction is None:
            all_predictions[public_name] = y_prob
        else:
            register_prediction(public_name, booster.predict(X_valid), y_prob)
    except Exception as e:
        logger.error(f"[{public_name}] Ablation LightGBM failed: {e}")


def main():
    args = parse_args()

    # ── 输出目录隔离 ──
    if args.run_name:
        run_root = config.PROJECT_ROOT / "runs" / args.run_name
        config.DATA_DIR = run_root / "data"
        config.MODEL_DIR = run_root / "models"
        config.REPORT_DIR = run_root / "reports"
        config.LOG_DIR = run_root / "logs"
        config.STEP_TABLE_PATH = config.DATA_DIR / "step_table.parquet"
        config.PREFIX_TABLE_PATH = config.DATA_DIR / "prefix_table.parquet"
        config.PREFIX_TABLE_FILTERED_PATH = config.DATA_DIR / "prefix_table_filtered.parquet"
        for d in [config.DATA_DIR, config.MODEL_DIR, config.REPORT_DIR, config.LOG_DIR]:
            d.mkdir(parents=True, exist_ok=True)
        # logger 在模块导入时已初始化，这里需要重绑 file handler 到新的 LOG_DIR
        rebind_all_file_loggers()

    if args.data_dir:
        config.PARQUET_INPUT_DIR = args.data_dir

    # 快速验证模式：只跑一个 parquet + 跳过重计算阶段
    if args.quick_verify:
        source = args.single_parquet or config.PARQUET_INPUT_DIR
        files = _iter_tool_parquet_files(source)
        if not files:
            raise FileNotFoundError(f"No tool-*.parquet files found in {source}")
        args.single_parquet = files[0]
        args.skip_lgbm = True
        args.skip_ablation = True

    if args.single_parquet:
        single = Path(args.single_parquet)
        if not single.exists():
            raise FileNotFoundError(f"single parquet not found: {single}")
        config.PARQUET_INPUT_DIR = str(single)

    if args.no_gpu_lgbm:
        config.LGBM_PARAMS["device"] = "cpu"
        if "gpu_device_id" in config.LGBM_PARAMS:
            del config.LGBM_PARAMS["gpu_device_id"]

    if args.split_by == "model_holdout":
        os.environ.setdefault("SWE_PREFIX_SKIP_INSTANCE_DEDUP", "1")
        logger.info(
            "model_holdout mode: keeping multiple model trajectories per instance "
            "(SWE_PREFIX_SKIP_INSTANCE_DEDUP=1)."
        )

    if args.sample_trajectories_seed is not None and (
        args.max_trajectories is None or args.max_trajectories <= 0
    ):
        logger.warning(
            "--sample-trajectories-seed has no effect without a positive --max-trajectories"
        )

    logger.info("=" * 80)
    logger.info("SWE-smith Prefix Success Prediction Pipeline")
    logger.info("=" * 80)
    logger.info(f"Parquet input dir: {config.PARQUET_INPUT_DIR}")
    if args.quick_verify:
        logger.info("Quick verify mode enabled (single parquet + skip lgbm/ablation)")
    logger.info(f"Data output dir:   {config.DATA_DIR}")
    logger.info(f"Model output dir:  {config.MODEL_DIR}")
    logger.info(f"Report output dir: {config.REPORT_DIR}")

    # ══════════════════════════════════════════════════════════
    # Phase 1: Step Table
    # ══════════════════════════════════════════════════════════
    if not args.skip_step_table:
        with timer(logger, "Phase 1: Building step table"):
            step_df = build_step_table(
                config.PARQUET_INPUT_DIR,
                max_trajectories=args.max_trajectories,
                sample_trajectories_seed=args.sample_trajectories_seed,
            )
            step_df.to_parquet(config.STEP_TABLE_PATH, index=False)
            logger.info(f"Step table saved: {config.STEP_TABLE_PATH} ({len(step_df)} rows)")
    else:
        logger.info("Phase 1: Skipped (using existing step table)")

    # ══════════════════════════════════════════════════════════
    # Phase 2: Prefix Table
    # ══════════════════════════════════════════════════════════
    if not args.skip_prefix_table:
        with timer(logger, "Phase 2: Building prefix table"):
            prefix_df = build_prefix_table(
                config.PARQUET_INPUT_DIR,
                max_trajectories=args.max_trajectories,
                sample_trajectories_seed=args.sample_trajectories_seed,
            )
            prefix_df.to_parquet(config.PREFIX_TABLE_PATH, index=False)
            logger.info(f"Prefix table saved: {config.PREFIX_TABLE_PATH} ({len(prefix_df)} rows)")
    else:
        logger.info("Phase 2: Loading existing prefix table")
        prefix_df = pd.read_parquet(config.PREFIX_TABLE_PATH)
        logger.info(f"Loaded prefix table: {len(prefix_df)} rows")

    # 关键数据一致性检查，防止 group 级别数据泄漏。
    _validate_prefix_group_integrity(prefix_df)

    answer_summary = {}
    if not args.disable_answer_features:
        with timer(logger, "Phase 2.2: Gold answer feature enrichment"):
            answer_path = config.DATA_DIR / "prefix_table_answer_enriched.parquet"
            answer_summary_path = config.REPORT_DIR / "gold_answer_enrichment_summary.json"
            if args.reuse_answer_enriched and answer_path.exists():
                prefix_df = pd.read_parquet(answer_path)
                if answer_summary_path.exists():
                    import json
                    with open(answer_summary_path, "r", encoding="utf-8") as f:
                        answer_summary = json.load(f)
                logger.info(f"Reused answer-enriched prefix table: {answer_path}")
            else:
                prefix_df, answer_summary = enrich_prefix_with_answer_features(
                    prefix_df,
                    args.verified_jsonl,
                )
                prefix_df.to_parquet(answer_path, index=False)
                save_json(answer_summary, answer_summary_path)
                logger.info(f"Answer-enriched prefix table saved: {answer_path}")

    # ══════════════════════════════════════════════════════════
    # Phase 2.5: 过滤短轨迹
    # ══════════════════════════════════════════════════════════
    min_steps = config.MIN_TRAJECTORY_STEPS
    if args.split_by == "model_holdout":
        logger.info(
            f"Model-holdout mode: deferring < {min_steps} step filtering until after split; "
            "test split will remain unfiltered."
        )
    else:
        logger.info(f"Filtering trajectories with < {min_steps} steps...")
        traj_step_counts = prefix_df.groupby("traj_id")["n_steps_total_for_weighting"].first()
        short_trajs = set(traj_step_counts[traj_step_counts < min_steps].index)
        n_before = prefix_df["traj_id"].nunique()
        prefix_df = prefix_df[~prefix_df["traj_id"].isin(short_trajs)].reset_index(drop=True)
        n_after = prefix_df["traj_id"].nunique()
        logger.info(
            f"Filtered: {n_before} -> {n_after} trajectories "
            f"(removed {n_before - n_after} with < {min_steps} steps), "
            f"{len(prefix_df)} prefix samples remaining"
        )

    # 保存过滤后的数据集
    prefix_df.to_parquet(config.PREFIX_TABLE_FILTERED_PATH, index=False)
    logger.info(f"Filtered prefix table saved: {config.PREFIX_TABLE_FILTERED_PATH}")

    # 统计过滤后轨迹的步数分布
    filtered_step_counts = prefix_df.groupby("traj_id")["n_steps_total_for_weighting"].first()
    if len(filtered_step_counts):
        logger.info(
            f"Filtered trajectory step distribution: "
            f"mean={filtered_step_counts.mean():.2f}, "
            f"median={filtered_step_counts.median():.1f}, "
            f"min={filtered_step_counts.min()}, "
            f"max={filtered_step_counts.max()}"
        )

    # ══════════════════════════════════════════════════════════
    # Phase 3: Data Split
    # ══════════════════════════════════════════════════════════
    with timer(logger, "Phase 3: Splitting data"):
        # group_id 重复性检查
        group_traj = prefix_df.groupby("group_id")["traj_id"].nunique()
        multi_traj_groups = (group_traj > 1).sum()
        logger.info(f"Groups with multiple trajectories: {multi_traj_groups} / {len(group_traj)}")

        split_meta = {"mode": args.split_by}
        if args.split_by == "model_holdout":
            logger.info("Split mode: by model holdout (test models unseen during training)")
            train_idx, valid_idx, test_idx, split_meta = select_model_holdout_split(
                prefix_df,
                verified_jsonl=args.verified_jsonl,
                holdout_models=args.holdout_models,
                max_instances=args.max_instances,
            )
        elif args.split_by == "instance":
            if "instance_id" not in prefix_df.columns:
                raise ValueError("--split-by instance requires instance_id column in prefix table")
            split_col = "instance_id"
            check_inst = True
            logger.info("Split mode: by instance_id (same SWE instance never spans train/valid/test)")
            train_idx, valid_idx, test_idx = split_by_group(
                prefix_df,
                group_col=split_col,
                check_instance_leak=check_inst,
            )
        else:
            split_col = "group_id"
            check_inst = False
            logger.info(
                "Split mode: by trajectory (group_id); instance_id may repeat across splits "
                "(check_instance_leak disabled)"
            )
            train_idx, valid_idx, test_idx = split_by_group(
                prefix_df,
                group_col=split_col,
                check_instance_leak=check_inst,
            )

        prefix_df["split"] = "none"
        prefix_df.loc[train_idx, "split"] = "train"
        prefix_df.loc[valid_idx, "split"] = "valid"
        prefix_df.loc[test_idx, "split"] = "test"

        df_train = prefix_df.loc[train_idx].copy()
        df_valid = prefix_df.loc[valid_idx].copy()
        df_test = prefix_df.loc[test_idx].copy()

        if args.split_by == "model_holdout":
            short_train = set(
                df_train.groupby("traj_id")["n_steps_total_for_weighting"].first()
                .loc[lambda s: s < min_steps]
                .index
            )
            short_valid = set(
                df_valid.groupby("traj_id")["n_steps_total_for_weighting"].first()
                .loc[lambda s: s < min_steps]
                .index
            )
            before_train, before_valid, before_test = len(df_train), len(df_valid), len(df_test)
            df_train = df_train[~df_train["traj_id"].isin(short_train)].copy()
            df_valid = df_valid[~df_valid["traj_id"].isin(short_valid)].copy()
            logger.info(
                "Model-holdout short-trajectory filtering: "
                f"train {before_train}->{len(df_train)}, valid {before_valid}->{len(df_valid)}, "
                f"test kept unfiltered at {before_test}."
            )
            prefix_df.loc[:, "split"] = "none"
            prefix_df.loc[df_train.index, "split"] = "train"
            prefix_df.loc[df_valid.index, "split"] = "valid"
            prefix_df.loc[df_test.index, "split"] = "test"
            write_model_holdout_summary(
                prefix_df,
                split_meta,
                config.REPORT_DIR / "model_holdout_split_summary.csv",
            )

            for frame in (df_train, df_valid, df_test):
                if "orig_model_id" not in frame.columns:
                    frame["orig_model_id"] = frame["model_id"].astype(str)
                if "orig_model" not in frame.columns:
                    frame["orig_model"] = frame.get("model", frame["model_id"]).astype(str)
            df_train["model_id_input_mode"] = "train_seen"
            for frame, mode in ((df_valid, "valid_missing"), (df_test, "test_missing")):
                frame["model_id"] = "__MISSING__"
                if "model" in frame.columns:
                    frame["model"] = "__MISSING__"
                frame["model_id_input_mode"] = mode
            implementation_model_meta = {
                "model_id_feature_mode": "train_seen_valid_missing_test_missing",
                "holdout_models": split_meta.get("holdout_models", []),
                "train_models": split_meta.get("train_models", []),
            }
        else:
            implementation_model_meta = {"model_id_feature_mode": "normal"}

        y_train = df_train["label"].values.astype(int)
        y_valid = df_valid["label"].values.astype(int)
        y_test = df_test["label"].values.astype(int)
        w_train = df_train["sample_weight"].values.astype(np.float32)
        w_valid = df_valid["sample_weight"].values.astype(np.float32)
        step_test = df_test["prefix_step_idx"].values
        n_steps_total_test = df_test["n_steps_total_for_weighting"].values

        logger.info(f"Train: {len(df_train)} samples, pos_rate={y_train.mean():.4f}")
        logger.info(f"Valid: {len(df_valid)} samples, pos_rate={y_valid.mean():.4f}")
        logger.info(f"Test:  {len(df_test)} samples, pos_rate={y_test.mean():.4f}")

    # ══════════════════════════════════════════════════════════
    # Phase 4: Feature Engineering
    # ══════════════════════════════════════════════════════════
    with timer(logger, "Phase 4: Feature engineering"):
        # 版本 A：包含 model_id
        fe_with_model_path = config.MODEL_DIR / "feature_engineer_with_model.pkl"
        if args.reuse_feature_engineers and fe_with_model_path.exists():
            fe_with_model = FeatureEngineer.load(fe_with_model_path)
            logger.info(f"Reused FeatureEngineer with model_id: {fe_with_model_path}")
        else:
            fe_with_model = FeatureEngineer(
                include_model_id=True,
                tfidf_level="with_gold_answer" if not args.disable_answer_features else "with_thought",
            )
            fe_with_model.fit(df_train)
            fe_with_model.save(fe_with_model_path)

        # 版本 B：不含 model_id
        fe_no_model_path = config.MODEL_DIR / "feature_engineer_no_model.pkl"
        if args.reuse_feature_engineers and fe_no_model_path.exists():
            fe_no_model = FeatureEngineer.load(fe_no_model_path)
            logger.info(f"Reused FeatureEngineer without model_id: {fe_no_model_path}")
        else:
            fe_no_model = FeatureEngineer(
                include_model_id=False,
                tfidf_level="with_gold_answer" if not args.disable_answer_features else "with_thought",
            )
            fe_no_model.fit(df_train)
            fe_no_model.save(fe_no_model_path)

        # ── 构建 Dense 特征矩阵 ──
        logger.info("Building dense feature matrices...")
        X_train_dense = fe_with_model.transform_dense(df_train)
        X_valid_dense = fe_with_model.transform_dense(df_valid)
        X_test_dense = fe_with_model.transform_dense(df_test)
        logger.info(f"Dense shape: {X_train_dense.shape}")

        # ── 构建分层 TF-IDF 特征矩阵 ──
        logger.info("Building hierarchical TF-IDF feature matrices...")
        
        # TF-IDF 1: Action + Feedback (基础版)
        tfidf_af_cols = ["tfidf_task_prompt", "tfidf_prefix_action", "tfidf_prefix_feedback",
                         "tfidf_last_action", "tfidf_last_feedback"]
        tfidf_all_cols = list(fe_with_model.active_text_columns.keys())
        train_tfidf_blocks = _transform_tfidf_blocks_once(fe_with_model, df_train, tfidf_all_cols, "train")
        valid_tfidf_blocks = _transform_tfidf_blocks_once(fe_with_model, df_valid, tfidf_all_cols, "valid")
        test_tfidf_blocks = _transform_tfidf_blocks_once(fe_with_model, df_test, tfidf_all_cols, "test")
        X_train_tfidf_af = _stack_tfidf_blocks(train_tfidf_blocks, tfidf_af_cols, len(df_train))
        X_valid_tfidf_af = _stack_tfidf_blocks(valid_tfidf_blocks, tfidf_af_cols, len(df_valid))
        X_test_tfidf_af = _stack_tfidf_blocks(test_tfidf_blocks, tfidf_af_cols, len(df_test))
        logger.info(f"TF-IDF AF shape: {X_train_tfidf_af.shape}")
        
        # TF-IDF 2: AF + Thought (进阶版)
        tfidf_af_thought_cols = tfidf_af_cols + ["tfidf_prefix_thought", "tfidf_last_thought"]
        X_train_tfidf_af_thought = _stack_tfidf_blocks(train_tfidf_blocks, tfidf_af_thought_cols, len(df_train))
        X_valid_tfidf_af_thought = _stack_tfidf_blocks(valid_tfidf_blocks, tfidf_af_thought_cols, len(df_valid))
        X_test_tfidf_af_thought = _stack_tfidf_blocks(test_tfidf_blocks, tfidf_af_thought_cols, len(df_test))
        logger.info(f"TF-IDF AF+Thought shape: {X_train_tfidf_af_thought.shape}")
        
        # TF-IDF 3: Full (AF + Thought + Assistant Content)
        X_train_tfidf_full = _stack_tfidf_blocks(train_tfidf_blocks, tfidf_all_cols, len(df_train))
        X_valid_tfidf_full = _stack_tfidf_blocks(valid_tfidf_blocks, tfidf_all_cols, len(df_valid))
        X_test_tfidf_full = _stack_tfidf_blocks(test_tfidf_blocks, tfidf_all_cols, len(df_test))
        logger.info(f"TF-IDF Full shape: {X_train_tfidf_full.shape}")

        # ── 构建 Dense + 分层 TF-IDF 组合矩阵 ──
        logger.info("Building combined Dense + TF-IDF matrices...")
        X_train_dense_sp = sparse.csr_matrix(X_train_dense)
        X_valid_dense_sp = sparse.csr_matrix(X_valid_dense)
        X_test_dense_sp = sparse.csr_matrix(X_test_dense)
        
        # Baseline B: Dense + AF
        X_train_dense_af = sparse.hstack([X_train_dense_sp, X_train_tfidf_af], format="csr")
        X_valid_dense_af = sparse.hstack([X_valid_dense_sp, X_valid_tfidf_af], format="csr")
        X_test_dense_af = sparse.hstack([X_test_dense_sp, X_test_tfidf_af], format="csr")
        logger.info(f"Dense + AF shape: {X_train_dense_af.shape}")
        feat_names_dense_af = (
            fe_with_model.dense_feature_names
            + fe_with_model.get_tfidf_feature_names_for_columns(tfidf_af_cols)
        )
        
        # Baseline C: Dense + AF + Thought
        X_train_dense_af_thought = sparse.hstack([X_train_dense_sp, X_train_tfidf_af_thought], format="csr")
        X_valid_dense_af_thought = sparse.hstack([X_valid_dense_sp, X_valid_tfidf_af_thought], format="csr")
        X_test_dense_af_thought = sparse.hstack([X_test_dense_sp, X_test_tfidf_af_thought], format="csr")
        logger.info(f"Dense + AF + Thought shape: {X_train_dense_af_thought.shape}")
        feat_names_dense_af_thought = (
            fe_with_model.dense_feature_names
            + fe_with_model.get_tfidf_feature_names_for_columns(tfidf_af_thought_cols)
        )
        
        # Baseline D: Dense + Full (主模型)
        X_train_all = sparse.hstack([X_train_dense_sp, X_train_tfidf_full], format="csr")
        X_valid_all = sparse.hstack([X_valid_dense_sp, X_valid_tfidf_full], format="csr")
        X_test_all = sparse.hstack([X_test_dense_sp, X_test_tfidf_full], format="csr")
        logger.info(f"Dense + Full shape: {X_train_all.shape}")
        feat_names_dense_full = fe_with_model.get_all_feature_names()
        feat_names_tfidf_af = fe_with_model.get_tfidf_feature_names_for_columns(tfidf_af_cols)
        feat_names_tfidf_af_thought = fe_with_model.get_tfidf_feature_names_for_columns(tfidf_af_thought_cols)
        feat_names_tfidf_full = fe_with_model.get_tfidf_feature_names_for_columns(
            list(fe_with_model.active_text_columns.keys())
        )

        # ── 不含 model_id 版本 (用于 Ablation 9) ──
        logger.info("Building no-model_id matrices...")
        X_train_dense_nomodel = fe_no_model.transform_dense(df_train)
        X_valid_dense_nomodel = fe_no_model.transform_dense(df_valid)
        X_test_dense_nomodel = fe_no_model.transform_dense(df_test)
        X_train_dense_nomodel_sp = sparse.csr_matrix(X_train_dense_nomodel)
        X_valid_dense_nomodel_sp = sparse.csr_matrix(X_valid_dense_nomodel)
        X_test_dense_nomodel_sp = sparse.csr_matrix(X_test_dense_nomodel)
        logger.info(f"Dense(no-model_id) shape: {X_train_dense_nomodel.shape}")

    # ══════════════════════════════════════════════════════════
    # Phase 5: Baseline 训练 (7 个模型)
    # ══════════════════════════════════════════════════════════
    all_models = {}
    all_predictions = {}
    calibrated_predictions = {}
    calibration_rows = []
    implementation_checks = {}

    def _safe_artifact_name(name: str) -> str:
        return re.sub(r"[^A-Za-z0-9_.-]+", "_", name)

    def _register_prediction(
        model_name: str,
        valid_prob: np.ndarray,
        test_prob: np.ndarray,
    ) -> None:
        test_prob = np.asarray(test_prob, dtype=np.float64).ravel()
        valid_prob = np.asarray(valid_prob, dtype=np.float64).ravel()
        all_predictions[model_name] = test_prob
        if args.disable_prob_calibration:
            return
        try:
            calibrator = fit_sigmoid_calibrator(
                raw_prob_valid=valid_prob,
                y_valid=y_valid,
                sample_weight=w_valid,
            )
            cal_test_prob = calibrator.predict(test_prob)
            calibrated_predictions[model_name] = cal_test_prob
            calibration_rows.append(
                calibration_summary_row(
                    model_name=model_name,
                    calibrator=calibrator,
                    y_valid=y_valid,
                    raw_prob_valid=valid_prob,
                    y_test=y_test,
                    raw_prob_test=test_prob,
                )
            )
            save_model(
                calibrator,
                config.MODEL_DIR / f"calibrator_{_safe_artifact_name(model_name)}.pkl",
            )
        except Exception as exc:
            logger.error(f"[{model_name}] probability calibration failed: {exc}")

    # ── Baseline A: Dense Only LR ──
    with timer(logger, "Phase 5A: Dense Only LR"):
        lr_dense = train_logistic_regression(
            X_train_dense, y_train, w_train,
            X_valid_dense, y_valid,
            model_name="dense_lr",
        )
        save_model(lr_dense, config.MODEL_DIR / "baseline_dense_lr.pkl")
        all_models["A_Dense_LR"] = lr_dense
        _register_prediction(
            "A_Dense_LR",
            lr_dense.predict_proba(X_valid_dense)[:, 1],
            lr_dense.predict_proba(X_test_dense)[:, 1],
        )

    # ── Baseline B: Dense + AF (Action + Feedback) LR ──
    with timer(logger, "Phase 5B: Dense + AF LR"):
        X_train_dense_af_sp = sparse.csr_matrix(X_train_dense_af)
        X_valid_dense_af_sp = sparse.csr_matrix(X_valid_dense_af)
        X_test_dense_af_sp = sparse.csr_matrix(X_test_dense_af)

        lr_dense_af = train_logistic_regression(
            X_train_dense_af_sp, y_train, w_train,
            X_valid_dense_af_sp, y_valid,
            model_name="dense_af_lr",
        )
        save_model(lr_dense_af, config.MODEL_DIR / "baseline_dense_af_lr.pkl")
        all_models["B_Dense_AF_LR"] = lr_dense_af
        _register_prediction(
            "B_Dense_AF_LR",
            lr_dense_af.predict_proba(X_valid_dense_af_sp)[:, 1],
            lr_dense_af.predict_proba(X_test_dense_af_sp)[:, 1],
        )

    # ── Baseline C: Dense + AF + Thought LR ──
    with timer(logger, "Phase 5C: Dense + AF + Thought LR"):
        X_train_dense_af_thought_sp = sparse.csr_matrix(X_train_dense_af_thought)
        X_valid_dense_af_thought_sp = sparse.csr_matrix(X_valid_dense_af_thought)
        X_test_dense_af_thought_sp = sparse.csr_matrix(X_test_dense_af_thought)

        lr_dense_af_thought = train_logistic_regression(
            X_train_dense_af_thought_sp, y_train, w_train,
            X_valid_dense_af_thought_sp, y_valid,
            model_name="dense_af_thought_lr",
        )
        save_model(lr_dense_af_thought, config.MODEL_DIR / "baseline_dense_af_thought_lr.pkl")
        all_models["C_Dense_AF_Thought_LR"] = lr_dense_af_thought
        _register_prediction(
            "C_Dense_AF_Thought_LR",
            lr_dense_af_thought.predict_proba(X_valid_dense_af_thought_sp)[:, 1],
            lr_dense_af_thought.predict_proba(X_test_dense_af_thought_sp)[:, 1],
        )

    # ── Baseline D: Dense + Full (AF+Thought+AC) LR (主模型) ──
    with timer(logger, "Phase 5D: Dense + Full LR (main model)"):
        X_train_all_sp = sparse.csr_matrix(X_train_all)
        X_valid_all_sp = sparse.csr_matrix(X_valid_all)
        X_test_all_sp = sparse.csr_matrix(X_test_all)

        lr_all = train_logistic_regression(
            X_train_all_sp, y_train, w_train,
            X_valid_all_sp, y_valid,
            model_name="dense_full_lr",
        )
        save_model(lr_all, config.MODEL_DIR / "baseline_dense_full_lr.pkl")
        all_models["D_Dense_Full_LR"] = lr_all
        _register_prediction(
            "D_Dense_Full_LR",
            lr_all.predict_proba(X_valid_all_sp)[:, 1],
            lr_all.predict_proba(X_test_all_sp)[:, 1],
        )

    # ── Baseline E: TF-IDF AF Only LR ──
    with timer(logger, "Phase 5E: TF-IDF AF Only LR"):
        X_train_tfidf_af_sp = sparse.csr_matrix(X_train_tfidf_af)
        X_valid_tfidf_af_sp = sparse.csr_matrix(X_valid_tfidf_af)
        X_test_tfidf_af_sp = sparse.csr_matrix(X_test_tfidf_af)

        lr_tfidf_af = train_logistic_regression(
            X_train_tfidf_af_sp, y_train, w_train,
            X_valid_tfidf_af_sp, y_valid,
            model_name="tfidf_af_lr",
        )
        save_model(lr_tfidf_af, config.MODEL_DIR / "baseline_tfidf_af_lr.pkl")
        all_models["E_TfIdf_AF_LR"] = lr_tfidf_af
        _register_prediction(
            "E_TfIdf_AF_LR",
            lr_tfidf_af.predict_proba(X_valid_tfidf_af_sp)[:, 1],
            lr_tfidf_af.predict_proba(X_test_tfidf_af_sp)[:, 1],
        )

    # ── Baseline F: TF-IDF AF + Thought LR ──
    with timer(logger, "Phase 5F: TF-IDF AF + Thought LR"):
        X_train_tfidf_af_thought_sp = sparse.csr_matrix(X_train_tfidf_af_thought)
        X_valid_tfidf_af_thought_sp = sparse.csr_matrix(X_valid_tfidf_af_thought)
        X_test_tfidf_af_thought_sp = sparse.csr_matrix(X_test_tfidf_af_thought)

        lr_tfidf_af_thought = train_logistic_regression(
            X_train_tfidf_af_thought_sp, y_train, w_train,
            X_valid_tfidf_af_thought_sp, y_valid,
            model_name="tfidf_af_thought_lr",
        )
        save_model(lr_tfidf_af_thought, config.MODEL_DIR / "baseline_tfidf_af_thought_lr.pkl")
        all_models["F_TfIdf_AF_Thought_LR"] = lr_tfidf_af_thought
        _register_prediction(
            "F_TfIdf_AF_Thought_LR",
            lr_tfidf_af_thought.predict_proba(X_valid_tfidf_af_thought_sp)[:, 1],
            lr_tfidf_af_thought.predict_proba(X_test_tfidf_af_thought_sp)[:, 1],
        )

    # ── Baseline G: TF-IDF Full LR ──
    with timer(logger, "Phase 5G: TF-IDF Full LR"):
        X_train_tfidf_full_sp = sparse.csr_matrix(X_train_tfidf_full)
        X_valid_tfidf_full_sp = sparse.csr_matrix(X_valid_tfidf_full)
        X_test_tfidf_full_sp = sparse.csr_matrix(X_test_tfidf_full)

        lr_tfidf_full = train_logistic_regression(
            X_train_tfidf_full_sp, y_train, w_train,
            X_valid_tfidf_full_sp, y_valid,
            model_name="tfidf_full_lr",
        )
        save_model(lr_tfidf_full, config.MODEL_DIR / "baseline_tfidf_full_lr.pkl")
        all_models["G_TfIdf_Full_LR"] = lr_tfidf_full
        _register_prediction(
            "G_TfIdf_Full_LR",
            lr_tfidf_full.predict_proba(X_valid_tfidf_full_sp)[:, 1],
            lr_tfidf_full.predict_proba(X_test_tfidf_full_sp)[:, 1],
        )

    # ── Baseline H~N: LightGBM 系列（非线性） ──
    if not args.skip_lgbm:
        with timer(logger, "Phase 5H~N: LightGBM variants"):
            lgbm_specs = [
                (
                    "H_LightGBM_Dense",
                    "lgbm_dense",
                    config.MODEL_DIR / "baseline_lgbm_dense.lgb",
                    X_train_dense,
                    X_valid_dense,
                    X_test_dense,
                    fe_with_model.dense_feature_names,
                ),
                (
                    "I_LightGBM_Dense_AF",
                    "lgbm_dense_af",
                    config.MODEL_DIR / "baseline_lgbm_dense_af.lgb",
                    X_train_dense_af,
                    X_valid_dense_af,
                    X_test_dense_af,
                    feat_names_dense_af,
                ),
                (
                    "J_LightGBM_Dense_AF_Thought",
                    "lgbm_dense_af_thought",
                    config.MODEL_DIR / "baseline_lgbm_dense_af_thought.lgb",
                    X_train_dense_af_thought,
                    X_valid_dense_af_thought,
                    X_test_dense_af_thought,
                    feat_names_dense_af_thought,
                ),
                (
                    "K_LightGBM_Dense_Full",
                    "lgbm_dense_full",
                    config.MODEL_DIR / "baseline_lgbm_dense_full.lgb",
                    X_train_all,
                    X_valid_all,
                    X_test_all,
                    feat_names_dense_full,
                ),
                (
                    "L_LightGBM_TfIdf_AF",
                    "lgbm_tfidf_af",
                    config.MODEL_DIR / "baseline_lgbm_tfidf_af.lgb",
                    X_train_tfidf_af,
                    X_valid_tfidf_af,
                    X_test_tfidf_af,
                    feat_names_tfidf_af,
                ),
                (
                    "M_LightGBM_TfIdf_AF_Thought",
                    "lgbm_tfidf_af_thought",
                    config.MODEL_DIR / "baseline_lgbm_tfidf_af_thought.lgb",
                    X_train_tfidf_af_thought,
                    X_valid_tfidf_af_thought,
                    X_test_tfidf_af_thought,
                    feat_names_tfidf_af_thought,
                ),
                (
                    "N_LightGBM_TfIdf_Full",
                    "lgbm_tfidf_full",
                    config.MODEL_DIR / "baseline_lgbm_tfidf_full.lgb",
                    X_train_tfidf_full,
                    X_valid_tfidf_full,
                    X_test_tfidf_full,
                    feat_names_tfidf_full,
                ),
            ]

            for public_name, train_name, save_path, xtr, xva, xte, feat_names in lgbm_specs:
                try:
                    model, y_prob = _train_lightgbm_with_fallback(
                        X_train=xtr,
                        y_train=y_train,
                        w_train=w_train,
                        X_valid=xva,
                        y_valid=y_valid,
                        w_valid=w_valid,
                        X_test=xte,
                        feature_names=feat_names,
                        model_name=train_name,
                        model_path=save_path,
                    )
                    all_models[public_name] = model
                    _register_prediction(public_name, model.predict(xva), y_prob)
                except Exception as e:
                    logger.error(f"[{train_name}] GPU+CPU fallback both failed: {e}")
    else:
        logger.info("Phase 5H~N: Skipped LightGBM")

    # ══════════════════════════════════════════════════════════
    # Phase 6: Ablation — 系统的特征组消融实验
    # ══════════════════════════════════════════════════════════
    # 切片消融 LR 的列名（与 coef_ 维度一致，供 Phase 7 特征重要性 / 特征组）
    ablation_feature_name_overrides: dict[str, list[str]] = {}
    if not args.skip_ablation:
        with timer(logger, "Phase 6: Ablation experiments"):
            # 预计算：各 TF-IDF 组的列索引范围
            dense_dim = X_train_dense.shape[1]
            tfidf_offsets = {}
            offset = dense_dim
            for tname in ["tfidf_task_prompt", "tfidf_prefix_action", "tfidf_prefix_feedback",
                          "tfidf_last_action", "tfidf_last_feedback",
                          "tfidf_prefix_thought", "tfidf_last_thought",
                          "tfidf_prefix_assistant_content", "tfidf_last_assistant_content"]:
                vec = fe_with_model.tfidf_vectorizers.get(tname)
                if vec:
                    reducer = fe_with_model.tfidf_reducers.get(tname)
                    n = reducer.n_components if reducer is not None else len(vec.vocabulary_)
                    tfidf_offsets[tname] = (offset, offset + n)
                    offset += n
            total_dim = offset
            fn_full = fe_with_model.get_all_feature_names()
            if len(fn_full) != total_dim:
                logger.warning(
                    f"get_all_feature_names()={len(fn_full)} vs total_dim={total_dim} "
                    "(ablation LGBM 列名可能对不齐)"
                )

            # ── Ablation 1: Dense only (无任何 TF-IDF) ──
            logger.info("Ablation 1: Dense only (A~H+J groups, no TF-IDF)")
            X_train_dense_only = X_train_dense_sp
            X_valid_dense_only = X_valid_dense
            X_test_dense_only = X_test_dense
            lr_dense_only = train_logistic_regression(
                X_train_dense, y_train, w_train,
                X_valid_dense_only, y_valid,
                model_name="dense_only",
            )
            save_model(lr_dense_only, config.MODEL_DIR / "ablation_dense_only.pkl")
            all_models["Abl_DenseOnly_LR"] = lr_dense_only
            _register_prediction(
                "Abl_DenseOnly_LR",
                lr_dense_only.predict_proba(X_valid_dense_only)[:, 1],
                lr_dense_only.predict_proba(X_test_dense_only)[:, 1],
            )
            _run_ablation_lgbm(
                args, all_models, all_predictions,
                public_name="Abl_DenseOnly_LightGBM",
                train_slug="dense_only",
                X_train=X_train_dense,
                X_valid=X_valid_dense,
                X_test=X_test_dense,
                feature_names=fe_with_model.dense_feature_names,
                y_train=y_train,
                y_valid=y_valid,
                w_train=w_train,
                w_valid=w_valid,
                register_prediction=_register_prediction,
            )

            # ── Ablation 2: Dense + action + feedback (无 thought/content) ──
            logger.info("Ablation 2: Dense + action + feedback (no thought, no assistant_content)")
            remove_cols = set()
            for tname in ["tfidf_prefix_thought", "tfidf_last_thought",
                          "tfidf_prefix_assistant_content", "tfidf_last_assistant_content"]:
                if tname in tfidf_offsets:
                    start, end = tfidf_offsets[tname]
                    remove_cols.update(range(start, end))
            keep_cols = [i for i in range(total_dim) if i not in remove_cols]
            X_train_no_thought_content = X_train_all[:, keep_cols]
            X_valid_no_thought_content = X_valid_all[:, keep_cols]
            X_test_no_thought_content = X_test_all[:, keep_cols]
            lr_no_thought_content = train_logistic_regression(
                X_train_no_thought_content, y_train, w_train,
                X_valid_no_thought_content, y_valid,
                model_name="dense_action_feedback",
            )
            save_model(lr_no_thought_content, config.MODEL_DIR / "ablation_dense_action_feedback.pkl")
            all_models["Abl_NoThoughtContent_LR"] = lr_no_thought_content
            _register_prediction(
                "Abl_NoThoughtContent_LR",
                lr_no_thought_content.predict_proba(X_valid_no_thought_content)[:, 1],
                lr_no_thought_content.predict_proba(X_test_no_thought_content)[:, 1],
            )
            ablation_feature_name_overrides["Abl_NoThoughtContent_LR"] = [fn_full[i] for i in keep_cols]
            _run_ablation_lgbm(
                args, all_models, all_predictions,
                public_name="Abl_NoThoughtContent_LightGBM",
                train_slug="no_thought_content",
                X_train=X_train_no_thought_content,
                X_valid=X_valid_no_thought_content,
                X_test=X_test_no_thought_content,
                feature_names=[fn_full[i] for i in keep_cols],
                y_train=y_train,
                y_valid=y_valid,
                w_train=w_train,
                w_valid=w_valid,
                register_prediction=_register_prediction,
            )

            # ── Ablation 3: Dense + action + feedback + thought (无 assistant_content) ──
            logger.info("Ablation 3: Dense + action + feedback + thought (no assistant_content)")
            remove_cols = set()
            for tname in ["tfidf_prefix_assistant_content", "tfidf_last_assistant_content"]:
                if tname in tfidf_offsets:
                    start, end = tfidf_offsets[tname]
                    remove_cols.update(range(start, end))
            keep_cols = [i for i in range(total_dim) if i not in remove_cols]
            X_train_no_assistant_content = X_train_all[:, keep_cols]
            X_valid_no_assistant_content = X_valid_all[:, keep_cols]
            X_test_no_assistant_content = X_test_all[:, keep_cols]
            lr_no_assistant_content = train_logistic_regression(
                X_train_no_assistant_content, y_train, w_train,
                X_valid_no_assistant_content, y_valid,
                model_name="dense_action_feedback_thought",
            )
            save_model(lr_no_assistant_content, config.MODEL_DIR / "ablation_dense_action_feedback_thought.pkl")
            all_models["Abl_NoAssistantContent_LR"] = lr_no_assistant_content
            _register_prediction(
                "Abl_NoAssistantContent_LR",
                lr_no_assistant_content.predict_proba(X_valid_no_assistant_content)[:, 1],
                lr_no_assistant_content.predict_proba(X_test_no_assistant_content)[:, 1],
            )
            ablation_feature_name_overrides["Abl_NoAssistantContent_LR"] = [fn_full[i] for i in keep_cols]
            _run_ablation_lgbm(
                args, all_models, all_predictions,
                public_name="Abl_NoAssistantContent_LightGBM",
                train_slug="no_assistant_content",
                X_train=X_train_no_assistant_content,
                X_valid=X_valid_no_assistant_content,
                X_test=X_test_no_assistant_content,
                feature_names=[fn_full[i] for i in keep_cols],
                y_train=y_train,
                y_valid=y_valid,
                w_train=w_train,
                w_valid=w_valid,
                register_prediction=_register_prediction,
            )

            # ── Ablation 4: Dense + AF + Thought (即 Baseline C，已训练) ──
            all_models["Abl_Base_LR"] = all_models["C_Dense_AF_Thought_LR"]
            all_predictions["Abl_Base_LR"] = all_predictions["C_Dense_AF_Thought_LR"]
            if "C_Dense_AF_Thought_LR" in calibrated_predictions:
                calibrated_predictions["Abl_Base_LR"] = calibrated_predictions["C_Dense_AF_Thought_LR"]
            logger.info("Ablation 4: Using Baseline C (Dense + AF + Thought) as reference")
            j_lgbm = all_models.get("J_LightGBM_Dense_AF_Thought")
            if j_lgbm is not None:
                all_models["Abl_Base_LightGBM"] = j_lgbm
                all_predictions["Abl_Base_LightGBM"] = all_predictions["J_LightGBM_Dense_AF_Thought"]
                if "J_LightGBM_Dense_AF_Thought" in calibrated_predictions:
                    calibrated_predictions["Abl_Base_LightGBM"] = calibrated_predictions["J_LightGBM_Dense_AF_Thought"]

            # ── Ablation 5: 去掉 task prompt (从 Dense + AF + Thought 基底出发) ──
            logger.info("Ablation 5: Dense + AF + Thought without task prompt")
            dense_dim_base = X_train_dense.shape[1]
            af_thought_offsets = {}
            offset = dense_dim_base
            for tname in ["tfidf_task_prompt", "tfidf_prefix_action", "tfidf_prefix_feedback",
                          "tfidf_last_action", "tfidf_last_feedback",
                          "tfidf_prefix_thought", "tfidf_last_thought"]:
                vec = fe_with_model.tfidf_vectorizers.get(tname)
                if vec:
                    reducer = fe_with_model.tfidf_reducers.get(tname)
                    n = reducer.n_components if reducer is not None else len(vec.vocabulary_)
                    af_thought_offsets[tname] = (offset, offset + n)
                    offset += n

            if "tfidf_task_prompt" in af_thought_offsets:
                start, end = af_thought_offsets["tfidf_task_prompt"]
                col_mask = list(range(start)) + list(range(end, offset))
                X_train_no_task = X_train_dense_af_thought[:, col_mask]
                X_valid_no_task = X_valid_dense_af_thought[:, col_mask]
                X_test_no_task = X_test_dense_af_thought[:, col_mask]
                lr_no_task = train_logistic_regression(
                    X_train_no_task, y_train, w_train,
                    X_valid_no_task, y_valid,
                    model_name="dense_af_thought_no_task",
                )
                save_model(lr_no_task, config.MODEL_DIR / "ablation_dense_af_thought_no_task.pkl")
                all_models["Abl_NoTaskPrompt_LR"] = lr_no_task
                _register_prediction(
                    "Abl_NoTaskPrompt_LR",
                    lr_no_task.predict_proba(X_valid_no_task)[:, 1],
                    lr_no_task.predict_proba(X_test_no_task)[:, 1],
                )
                _abl5_names = [feat_names_dense_af_thought[i] for i in col_mask]
                ablation_feature_name_overrides["Abl_NoTaskPrompt_LR"] = _abl5_names
                _run_ablation_lgbm(
                    args, all_models, all_predictions,
                    public_name="Abl_NoTaskPrompt_LightGBM",
                    train_slug="no_task_prompt",
                    X_train=X_train_no_task,
                    X_valid=X_valid_no_task,
                    X_test=X_test_no_task,
                    feature_names=_abl5_names,
                    y_train=y_train,
                    y_valid=y_valid,
                    w_train=w_train,
                    w_valid=w_valid,
                    register_prediction=_register_prediction,
                )
            else:
                logger.warning("tfidf_task_prompt not found, skipping Ablation 5")

            # ── Ablation 6: 去掉 feedback (从 Dense + AF + Thought 基底出发) ──
            logger.info("Ablation 6: Dense + AF + Thought without feedback")
            remove_cols = set()
            for tname in ["tfidf_prefix_feedback", "tfidf_last_feedback"]:
                if tname in af_thought_offsets:
                    start, end = af_thought_offsets[tname]
                    remove_cols.update(range(start, end))
            keep_cols = [i for i in range(offset) if i not in remove_cols]
            X_train_no_fb = X_train_dense_af_thought[:, keep_cols]
            X_valid_no_fb = X_valid_dense_af_thought[:, keep_cols]
            X_test_no_fb = X_test_dense_af_thought[:, keep_cols]
            lr_no_fb = train_logistic_regression(
                X_train_no_fb, y_train, w_train,
                X_valid_no_fb, y_valid,
                model_name="dense_af_thought_no_feedback",
            )
            save_model(lr_no_fb, config.MODEL_DIR / "ablation_dense_af_thought_no_feedback.pkl")
            all_models["Abl_NoFeedback_LR"] = lr_no_fb
            _register_prediction(
                "Abl_NoFeedback_LR",
                lr_no_fb.predict_proba(X_valid_no_fb)[:, 1],
                lr_no_fb.predict_proba(X_test_no_fb)[:, 1],
            )
            ablation_feature_name_overrides["Abl_NoFeedback_LR"] = [
                feat_names_dense_af_thought[i] for i in keep_cols
            ]
            _run_ablation_lgbm(
                args, all_models, all_predictions,
                public_name="Abl_NoFeedback_LightGBM",
                train_slug="no_feedback",
                X_train=X_train_no_fb,
                X_valid=X_valid_no_fb,
                X_test=X_test_no_fb,
                feature_names=[feat_names_dense_af_thought[i] for i in keep_cols],
                y_train=y_train,
                y_valid=y_valid,
                w_train=w_train,
                w_valid=w_valid,
                register_prediction=_register_prediction,
            )

            # ── Ablation 7: 去掉 action (从 Dense + AF + Thought 基底出发) ──
            logger.info("Ablation 7: Dense + AF + Thought without action")
            remove_cols = set()
            for tname in ["tfidf_prefix_action", "tfidf_last_action"]:
                if tname in af_thought_offsets:
                    start, end = af_thought_offsets[tname]
                    remove_cols.update(range(start, end))
            keep_cols = [i for i in range(offset) if i not in remove_cols]
            X_train_no_action = X_train_dense_af_thought[:, keep_cols]
            X_valid_no_action = X_valid_dense_af_thought[:, keep_cols]
            X_test_no_action = X_test_dense_af_thought[:, keep_cols]
            lr_no_action = train_logistic_regression(
                X_train_no_action, y_train, w_train,
                X_valid_no_action, y_valid,
                model_name="dense_af_thought_no_action",
            )
            save_model(lr_no_action, config.MODEL_DIR / "ablation_dense_af_thought_no_action.pkl")
            all_models["Abl_NoAction_LR"] = lr_no_action
            _register_prediction(
                "Abl_NoAction_LR",
                lr_no_action.predict_proba(X_valid_no_action)[:, 1],
                lr_no_action.predict_proba(X_test_no_action)[:, 1],
            )
            ablation_feature_name_overrides["Abl_NoAction_LR"] = [
                feat_names_dense_af_thought[i] for i in keep_cols
            ]
            _run_ablation_lgbm(
                args, all_models, all_predictions,
                public_name="Abl_NoAction_LightGBM",
                train_slug="no_action",
                X_train=X_train_no_action,
                X_valid=X_valid_no_action,
                X_test=X_test_no_action,
                feature_names=[feat_names_dense_af_thought[i] for i in keep_cols],
                y_train=y_train,
                y_valid=y_valid,
                w_train=w_train,
                w_valid=w_valid,
                register_prediction=_register_prediction,
            )

            # ── Ablation 8: 去掉 thought (从 Dense + AF + Thought 基底出发) ──
            logger.info("Ablation 8: Dense + AF + Thought without thought")
            remove_cols = set()
            for tname in ["tfidf_prefix_thought", "tfidf_last_thought"]:
                if tname in af_thought_offsets:
                    start, end = af_thought_offsets[tname]
                    remove_cols.update(range(start, end))
            keep_cols = [i for i in range(offset) if i not in remove_cols]
            X_train_no_thought = X_train_dense_af_thought[:, keep_cols]
            X_valid_no_thought = X_valid_dense_af_thought[:, keep_cols]
            X_test_no_thought = X_test_dense_af_thought[:, keep_cols]
            lr_no_thought = train_logistic_regression(
                X_train_no_thought, y_train, w_train,
                X_valid_no_thought, y_valid,
                model_name="dense_af_thought_no_thought",
            )
            save_model(lr_no_thought, config.MODEL_DIR / "ablation_dense_af_thought_no_thought.pkl")
            all_models["Abl_NoThought_LR"] = lr_no_thought
            _register_prediction(
                "Abl_NoThought_LR",
                lr_no_thought.predict_proba(X_valid_no_thought)[:, 1],
                lr_no_thought.predict_proba(X_test_no_thought)[:, 1],
            )
            ablation_feature_name_overrides["Abl_NoThought_LR"] = [
                feat_names_dense_af_thought[i] for i in keep_cols
            ]
            _run_ablation_lgbm(
                args, all_models, all_predictions,
                public_name="Abl_NoThought_LightGBM",
                train_slug="no_thought",
                X_train=X_train_no_thought,
                X_valid=X_valid_no_thought,
                X_test=X_test_no_thought,
                feature_names=[feat_names_dense_af_thought[i] for i in keep_cols],
                y_train=y_train,
                y_valid=y_valid,
                w_train=w_train,
                w_valid=w_valid,
                register_prediction=_register_prediction,
            )

            # ── Ablation 9: 去掉 model_id (从 Dense + AF + Thought 基底出发) ──
            logger.info("Ablation 9: Dense + AF + Thought without model_id")
            X_train_base_nomodel = sparse.hstack(
                [X_train_dense_nomodel_sp, X_train_tfidf_af_thought], format="csr"
            )
            X_valid_base_nomodel = sparse.hstack(
                [X_valid_dense_nomodel_sp, X_valid_tfidf_af_thought], format="csr"
            )
            X_test_base_nomodel = sparse.hstack(
                [X_test_dense_nomodel_sp, X_test_tfidf_af_thought], format="csr"
            )
            lr_no_model = train_logistic_regression(
                X_train_base_nomodel, y_train, w_train,
                X_valid_base_nomodel, y_valid,
                model_name="dense_af_thought_no_model",
            )
            save_model(lr_no_model, config.MODEL_DIR / "ablation_dense_af_thought_no_model.pkl")
            all_models["Abl_NoModel_LR"] = lr_no_model
            _register_prediction(
                "Abl_NoModel_LR",
                lr_no_model.predict_proba(X_valid_base_nomodel)[:, 1],
                lr_no_model.predict_proba(X_test_base_nomodel)[:, 1],
            )
            _abl9_names = (
                fe_no_model.dense_feature_names
                + fe_no_model.get_tfidf_feature_names_for_columns(tfidf_af_thought_cols)
            )
            _run_ablation_lgbm(
                args, all_models, all_predictions,
                public_name="Abl_NoModel_LightGBM",
                train_slug="no_model",
                X_train=X_train_base_nomodel,
                X_valid=X_valid_base_nomodel,
                X_test=X_test_base_nomodel,
                feature_names=_abl9_names,
                y_train=y_train,
                y_valid=y_valid,
                w_train=w_train,
                w_valid=w_valid,
                register_prediction=_register_prediction,
            )

            # ── Ablation 10: Process-only（去 task prompt + 去 model_id）──
            logger.info("Ablation 10: Process-only (no task prompt, no model_id)")
            process_cols = [
                "tfidf_prefix_action", "tfidf_prefix_feedback", "tfidf_prefix_thought",
                "tfidf_last_action", "tfidf_last_feedback", "tfidf_last_thought",
            ]
            X_train_tfidf_process = fe_no_model.transform_tfidf_subset(df_train, process_cols)
            X_valid_tfidf_process = fe_no_model.transform_tfidf_subset(df_valid, process_cols)
            X_test_tfidf_process = fe_no_model.transform_tfidf_subset(df_test, process_cols)
            X_train_process_only = sparse.hstack(
                [X_train_dense_nomodel_sp, X_train_tfidf_process], format="csr"
            )
            X_valid_process_only = sparse.hstack(
                [X_valid_dense_nomodel_sp, X_valid_tfidf_process], format="csr"
            )
            X_test_process_only = sparse.hstack(
                [X_test_dense_nomodel_sp, X_test_tfidf_process], format="csr"
            )
            lr_process_only = train_logistic_regression(
                X_train_process_only, y_train, w_train,
                X_valid_process_only, y_valid,
                model_name="process_only_no_task_no_model",
            )
            save_model(lr_process_only, config.MODEL_DIR / "ablation_process_only_no_task_no_model.pkl")
            all_models["Abl_ProcessOnly_LR"] = lr_process_only
            _register_prediction(
                "Abl_ProcessOnly_LR",
                lr_process_only.predict_proba(X_valid_process_only)[:, 1],
                lr_process_only.predict_proba(X_test_process_only)[:, 1],
            )
            _abl10_process_cols = [
                "tfidf_prefix_action", "tfidf_prefix_feedback", "tfidf_prefix_thought",
                "tfidf_last_action", "tfidf_last_feedback", "tfidf_last_thought",
            ]
            _abl10_names = (
                fe_no_model.dense_feature_names
                + fe_no_model.get_tfidf_feature_names_for_columns(_abl10_process_cols)
            )
            _run_ablation_lgbm(
                args, all_models, all_predictions,
                public_name="Abl_ProcessOnly_LightGBM",
                train_slug="process_only",
                X_train=X_train_process_only,
                X_valid=X_valid_process_only,
                X_test=X_test_process_only,
                feature_names=_abl10_names,
                y_train=y_train,
                y_valid=y_valid,
                w_train=w_train,
                w_valid=w_valid,
                register_prediction=_register_prediction,
            )

    # Mixed-model 实现检查（写入报告）
    implementation_checks = {
        **implementation_model_meta,
        "gold_answer_features_enabled": not args.disable_answer_features,
        "probability_calibration_enabled": not args.disable_prob_calibration,
        "probability_calibration_method": "sigmoid_platt_on_valid_logits",
        "probability_calibration_valid_model_id_mode": (
            "valid_missing" if args.split_by == "model_holdout" else "normal"
        ),
        "gold_answer_enrichment_summary": answer_summary,
        "dense_standardization_applied": bool(getattr(config, "DENSE_STANDARDIZE", False)),
        "dense_af_train_shape": tuple(int(v) for v in X_train_dense_af.shape),
        "dense_af_thought_train_shape": tuple(int(v) for v in X_train_dense_af_thought.shape),
        "dense_full_train_shape": tuple(int(v) for v in X_train_all.shape),
        "dense_train_cols": int(X_train_dense_sp.shape[1]),
        "tfidf_af_cols": int(X_train_tfidf_af.shape[1]),
        "tfidf_af_thought_cols": int(X_train_tfidf_af_thought.shape[1]),
        "tfidf_full_cols": int(X_train_tfidf_full.shape[1]),
        "dense_af_nnz": int(X_train_dense_af.nnz),
        "dense_af_thought_nnz": int(X_train_dense_af_thought.nnz),
        "dense_full_nnz": int(X_train_all.nnz),
        "dense_af_tfidf_match": int(X_train_dense_af.shape[1]) == int(X_train_dense_sp.shape[1] + X_train_tfidf_af.shape[1]),
        "dense_af_thought_tfidf_match": int(X_train_dense_af_thought.shape[1]) == int(X_train_dense_sp.shape[1] + X_train_tfidf_af_thought.shape[1]),
        "dense_full_tfidf_match": int(X_train_all.shape[1]) == int(X_train_dense_sp.shape[1] + X_train_tfidf_full.shape[1]),
    }
    for name in ("C_Dense_AF_Thought_LR", "D_Dense_Full_LR"):
        model = all_models.get(name)
        if model is None:
            continue
        n_iter = getattr(model, "n_iter_", None)
        if n_iter is not None:
            try:
                implementation_checks[f"{name}_n_iter_max"] = int(np.max(np.asarray(n_iter)))
            except Exception:
                pass
    if calibration_rows:
        cal_path = config.REPORT_DIR / "probability_calibration_summary.csv"
        pd.DataFrame(calibration_rows).to_csv(cal_path, index=False)
        logger.info(f"Saved probability calibration summary: {cal_path}")
    if args.skip_ablation:
        logger.info("Phase 6: Skipped ablation")

    # ══════════════════════════════════════════════════════════
    # Phase 7: Evaluation
    # ══════════════════════════════════════════════════════════
    with timer(logger, "Phase 7: Evaluation"):
        _save_test_artifacts(
            df_test=df_test,
            y_test=y_test,
            all_predictions=all_predictions,
            save_dir=config.REPORT_DIR,
            calibrated_predictions=calibrated_predictions,
        )
        if args.split_by == "model_holdout":
            _save_metrics_by_heldout_model(
                df_test=df_test,
                y_test=y_test,
                all_predictions=all_predictions,
                save_path=config.REPORT_DIR / "metrics_by_heldout_model.csv",
            )

        all_results = {}

        def _fmt_metric_line(v, nd: int = 4) -> str:
            if v is None:
                return "N/A"
            try:
                fv = float(v)
                if np.isnan(fv):
                    return "N/A"
                return f"{fv:.{nd}f}"
            except (TypeError, ValueError):
                return "N/A"

        for model_name, y_prob in all_predictions.items():
            logger.info(f"Evaluating {model_name}...")

            # 核心指标
            metrics = compute_metrics(y_test, y_prob)
            logger.info(
                f"  {model_name}: AUC={_fmt_metric_line(metrics.get('roc_auc'))}, "
                f"PR-AUC={_fmt_metric_line(metrics.get('pr_auc'))}, "
                f"LogLoss={_fmt_metric_line(metrics.get('log_loss'))}, "
                f"Brier={_fmt_metric_line(metrics.get('brier_score'))}"
            )

            # 分桶指标
            bucketed = compute_bucketed_metrics(y_test, y_prob, step_test)
            threshold_table = compute_threshold_decision_table(
                y_true=y_test,
                y_prob=y_prob,
                step_indices=step_test,
                n_steps_total=n_steps_total_test,
            )

            # 轨迹级别的步骤节省统计（新方法）
            trajectory_savings = compute_trajectory_level_savings(
                df_test=df_test,
                y_prob=y_prob,
            )

            # 按精确度水平：分别取满足 Prec(S)≥x / Prec(F)≥x 的最小 thr（见 evaluator 返回结构）
            precision_level_savings = compute_trajectory_savings_at_precision_levels(
                df_test=df_test,
                y_prob=y_prob,
            )

            # 特征重要性
            fi = []
            model_obj = all_models.get(model_name)
            if model_obj is not None:
                if hasattr(model_obj, "coef_"):
                    # 根据模型类型选择合适的特征名来源，避免使用过时的模型命名
                    if model_name in ablation_feature_name_overrides:
                        feat_names = ablation_feature_name_overrides[model_name]
                    elif model_name in ("A_Dense_LR", "Abl_DenseOnly_LR"):
                        # 纯 Dense 模型
                        feat_names = fe_with_model.dense_feature_names
                    elif model_name in ("Abl_NoModel_LR", "Abl_ProcessOnly_LR"):
                        # 不含 model_id 的 dense+text 模型
                        if model_name == "Abl_ProcessOnly_LR":
                            process_cols = [
                                "tfidf_prefix_action", "tfidf_prefix_feedback", "tfidf_prefix_thought",
                                "tfidf_last_action", "tfidf_last_feedback", "tfidf_last_thought",
                            ]
                            feat_names = fe_no_model.dense_feature_names + fe_no_model.get_tfidf_feature_names_for_columns(process_cols)
                        else:
                            feat_names = fe_no_model.dense_feature_names + fe_no_model.get_tfidf_feature_names_for_columns(tfidf_af_thought_cols)
                    elif model_name in (
                        "E_TfIdf_AF_LR",
                        "F_TfIdf_AF_Thought_LR",
                        "G_TfIdf_Full_LR",
                    ):
                        # 纯 TF-IDF 模型：按各自子集获取特征名（含分组降维后列）。
                        if model_name == "E_TfIdf_AF_LR":
                            cols = tfidf_af_cols
                        elif model_name == "F_TfIdf_AF_Thought_LR":
                            cols = tfidf_af_thought_cols
                        else:
                            cols = list(fe_with_model.active_text_columns.keys())
                        feat_names = fe_with_model.get_tfidf_feature_names_for_columns(cols)
                    else:
                        # Dense + TF-IDF 组合模型：使用完整特征名列表
                        feat_names = fe_with_model.get_all_feature_names()

                    fi = plot_feature_importance_lr(
                        model_obj,
                        feat_names,
                        top_k=40,
                        save_path=config.REPORT_DIR
                        / f"feature_importance_{model_name}.png",
                    )
                elif hasattr(model_obj, "feature_importance"):
                    fi = plot_feature_importance_lgbm(
                        model_obj, top_k=40,
                        save_path=config.REPORT_DIR / f"feature_importance_{model_name}.png",
                    )

            # 特征组贡献度分析
            fg_contribution = {}
            if model_obj is not None:
                is_lgbm = hasattr(model_obj, "feature_importance")
                if hasattr(model_obj, "coef_") or is_lgbm:
                    if is_lgbm:
                        try:
                            fg_feat_names = model_obj.feature_name()
                        except Exception:
                            fg_feat_names = []
                    elif model_name in ablation_feature_name_overrides:
                        fg_feat_names = ablation_feature_name_overrides[model_name]
                    elif model_name in ("A_Dense_LR", "Abl_DenseOnly_LR"):
                        fg_feat_names = fe_with_model.dense_feature_names
                    elif model_name in ("Abl_NoModel_LR", "Abl_ProcessOnly_LR"):
                        fg_feat_names = fe_no_model.dense_feature_names
                    else:
                        fg_feat_names = fe_with_model.get_all_feature_names()
                    fg_contribution = compute_feature_group_contribution(
                        model_obj, fg_feat_names, is_lgbm=is_lgbm
                    )

            all_results[model_name] = {
                "metrics": metrics,
                "bucketed": bucketed,
                "threshold_table": threshold_table,
                "trajectory_savings": trajectory_savings,
                "precision_level_savings": precision_level_savings,
                "feature_importance": fi,
                "feature_group_contribution": fg_contribution,
            }

            # 绘图（仅主要模型）
            # 与当前 Baseline 命名保持一致：主 Dense、主 Dense+Full、主 TF-IDF Full、LightGBM
            _plot_abl_lgbm = (
                model_name.startswith("Abl_")
                and model_name.endswith("_LightGBM")
            )
            if model_name in (
                "A_Dense_LR",
                "D_Dense_Full_LR",
                "G_TfIdf_Full_LR",
                "H_LightGBM_Dense",
                "I_LightGBM_Dense_AF",
                "K_LightGBM_Dense_Full",
                "N_LightGBM_TfIdf_Full",
            ) or _plot_abl_lgbm:
                y_logit_panel = None
                if model_name.endswith("_LR"):
                    yp = np.asarray(y_prob, dtype=np.float64)
                    eps = 1e-12
                    c = np.clip(yp, eps, 1.0 - eps)
                    y_logit_panel = np.log(c / (1.0 - c))
                plot_calibration(
                    y_test,
                    y_prob,
                    model_name,
                    config.REPORT_DIR / f"calibration_{model_name}.png",
                    y_logit=y_logit_panel,
                )
                plot_roc_pr(
                    y_test,
                    y_prob,
                    model_name,
                    config.REPORT_DIR / f"roc_pr_{model_name}.png",
                )
                plot_metrics_by_step(
                    bucketed,
                    model_name,
                    config.REPORT_DIR / f"step_metrics_{model_name}.png",
                )

        # 全模型 ROC 对比图
        _plot_multi_roc(y_test, all_predictions, config.REPORT_DIR / "roc_comparison_all.png")
        _plot_multi_pr(y_test, all_predictions, config.REPORT_DIR / "pr_comparison_all.png")

        # 生成文本报告
        report_text = generate_full_report(
            all_results,
            config.REPORT_DIR,
            report_metadata={"implementation_checks": implementation_checks},
        )
        logger.info("\n" + report_text)

    logger.info("=" * 80)
    logger.info("Pipeline completed successfully!")
    logger.info(f"Reports: {config.REPORT_DIR}")
    logger.info(f"Models:  {config.MODEL_DIR}")
    logger.info(f"Data:    {config.DATA_DIR}")
    logger.info("=" * 80)


def _plot_multi_roc(y_true, predictions_dict, save_path):
    """多模型 ROC 对比图。"""
    from sklearn.metrics import roc_auc_score
    fig, ax = plt.subplots(figsize=(8, 6))
    for name, y_prob in predictions_dict.items():
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc = roc_auc_score(y_true, y_prob)
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.4f})")
    ax.plot([0, 1], [0, 1], "k--")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve Comparison")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def _plot_multi_pr(y_true, predictions_dict, save_path):
    """多模型 PR 对比图。"""
    from sklearn.metrics import average_precision_score
    fig, ax = plt.subplots(figsize=(8, 6))
    for name, y_prob in predictions_dict.items():
        prec, rec, _ = precision_recall_curve(y_true, y_prob)
        ap = average_precision_score(y_true, y_prob)
        ax.plot(rec, prec, label=f"{name} (AP={ap:.4f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("PR Curve Comparison")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    main()
