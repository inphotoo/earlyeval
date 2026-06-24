#!/usr/bin/env python3
"""Compare prefix predictor AUC against other-model task-prior baselines.

Baselines are computed at trajectory level for the same heldout target rows:
- heldout_other_*: other heldout models in the same split on the same instance.
- train_other_*: all non-holdout models from the full prefix table on the same instance.
- all_other_*: all available non-target models on the same instance.

For top3, gpt-5-2-codex is excluded from both evaluation targets and other-model pools by default.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
RUN_NAME = "model_holdout_answer_calibrated_full"
REPORTS_DIR = PROJECT_ROOT / "runs" / RUN_NAME / "reports"
DATA_DIR = PROJECT_ROOT / "runs" / RUN_NAME / "data"
PREFIX_TABLE = DATA_DIR / "prefix_table_filtered.parquet"
OUT_DIR = REPORTS_DIR / "safe_stop_dual_head_visual_summary" / "problem_diagnosis" / "other_model_prior_auc"

PREFIX_MODELS = {
    "I": "I_LightGBM_Dense_AF",
    "J": "J_LightGBM_Dense_AF_Thought",
}
SPLITS = ["top3", "mid3", "bottom3"]
STRATEGIES = ["strong_reg", "no_model_id"]
VARIANTS = ["I", "J"]
SCORE_MODES = ["raw", "prefix_calibrated"]
TOP3_EXCLUDE_SUBSTR = "gpt-5-2-codex"


def auc_rank(y: Iterable[float], score: Iterable[float]) -> float:
    data = pd.DataFrame({"y": y, "score": score}).dropna()
    if data.empty:
        return float("nan")
    y_arr = data["y"].astype(int).to_numpy()
    n_pos = int((y_arr == 1).sum())
    n_neg = int((y_arr == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = data["score"].rank(method="average").to_numpy()
    sum_pos = float(ranks[y_arr == 1].sum())
    return (sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def prob_col(prefix_model: str, score_mode: str) -> str:
    if score_mode == "raw":
        return f"prob__{prefix_model}"
    if score_mode == "prefix_calibrated":
        return f"prob_cal__{prefix_model}"
    raise ValueError(score_mode)


def run_dir(split: str, variant: str, strategy: str) -> Path:
    return REPORTS_DIR / f"per_instance_model_valid3_{split}_{variant.lower()}_{strategy}_retrain"


def load_full_final_labels() -> pd.DataFrame:
    cols = ["traj_id", "instance_id", "model", "model_id", "label", "prefix_step_idx"]
    df = pd.read_parquet(PREFIX_TABLE, columns=cols)
    idx = df.groupby("traj_id", sort=False)["prefix_step_idx"].idxmax()
    final = df.loc[idx, ["traj_id", "instance_id", "model_id", "label"]].copy()
    final = final.rename(columns={"model_id": "model"})
    final["label"] = final["label"].astype(int)
    return final.drop_duplicates(["instance_id", "model"], keep="last")


def pool_stats(labels: list[int]) -> dict[str, float]:
    if not labels:
        return {
            "n_other": 0,
            "all_correct": float("nan"),
            "any_correct": float("nan"),
            "mean_success": float("nan"),
            "none_correct": float("nan"),
        }
    arr = np.asarray(labels, dtype=float)
    return {
        "n_other": int(len(arr)),
        "all_correct": float(arr.min()),
        "any_correct": float(arr.max()),
        "mean_success": float(arr.mean()),
        "none_correct": float(1.0 - arr.max()),
    }


def add_prior_scores(
    target: pd.DataFrame,
    full_final: pd.DataFrame,
    *,
    holdout_models: set[str],
    exclude_models: set[str],
) -> pd.DataFrame:
    full_by_instance = {
        instance: dict(zip(part["model"].astype(str), part["label"].astype(int)))
        for instance, part in full_final.groupby("instance_id", sort=False)
    }
    heldout_by_instance = {
        instance: dict(zip(part["orig_model_id"].astype(str), part["label"].astype(int)))
        for instance, part in target.groupby("instance_id", sort=False)
    }
    rows = []
    for row in target.itertuples(index=False):
        instance = str(row.instance_id)
        model = str(row.orig_model_id)
        row_dict = row._asdict()

        pools: dict[str, list[int]] = {}
        held = heldout_by_instance.get(instance, {})
        pools["heldout_other"] = [v for m, v in held.items() if m != model and m not in exclude_models]

        full = full_by_instance.get(instance, {})
        pools["train_other"] = [
            v for m, v in full.items()
            if m not in holdout_models and m not in exclude_models
        ]
        pools["all_other"] = [
            v for m, v in full.items()
            if m != model and m not in exclude_models
        ]

        for prefix, labels in pools.items():
            stats = pool_stats(labels)
            for key, value in stats.items():
                row_dict[f"{prefix}_{key}"] = value
        rows.append(row_dict)
    return pd.DataFrame(rows)


def final_rows_from_predictions(df: pd.DataFrame) -> pd.DataFrame:
    idx = df.groupby("traj_id", sort=False)["prefix_step_idx"].idxmax()
    return df.loc[idx].copy()


def auc_for_step(df: pd.DataFrame, pcol: str, step: str | int) -> tuple[float, int]:
    if step == "all_prefix":
        part = df[["label", pcol]].copy()
    elif step == "last":
        part = final_rows_from_predictions(df)[["label", pcol]].copy()
    else:
        part = df.loc[df["prefix_step_idx"].eq(int(step)), ["label", pcol]].copy()
    return auc_rank(part["label"], part[pcol]), int(len(part))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=str(OUT_DIR))
    parser.add_argument("--include-top3-gpt52codex", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    full_final = load_full_final_labels()
    rows = []
    prior_records = []

    for split in SPLITS:
        for variant in VARIANTS:
            prefix_model = PREFIX_MODELS[variant]
            for strategy in STRATEGIES:
                rd = run_dir(split, variant, strategy)
                pred_path = rd / "test_predictions_shadow_valid_retrain.parquet"
                meta_path = rd / "split_metadata.json"
                if not pred_path.exists() or not meta_path.exists():
                    continue
                meta = json.loads(meta_path.read_text())
                holdout_models = set(map(str, meta.get("holdout_models", [])))
                exclude_models = set()
                if split == "top3" and not args.include_top3_gpt52codex:
                    exclude_models = {m for m in holdout_models if TOP3_EXCLUDE_SUBSTR in m}

                df = pd.read_parquet(pred_path)
                if exclude_models:
                    df = df.loc[~df["orig_model_id"].astype(str).isin(exclude_models)].copy()
                final_target_base = final_rows_from_predictions(df)
                final_target_base = final_target_base[["traj_id", "instance_id", "orig_model_id", "label"]].copy()
                final_target_base["label"] = final_target_base["label"].astype(int)
                with_priors = add_prior_scores(
                    final_target_base,
                    full_final,
                    holdout_models=holdout_models,
                    exclude_models=exclude_models,
                )
                prior_records.append(
                    with_priors.assign(split=split, variant=variant, strategy=strategy)
                )

                for score_mode in SCORE_MODES:
                    pcol = prob_col(prefix_model, score_mode)
                    if pcol not in df.columns:
                        continue
                    eval_df = df[["traj_id", "instance_id", "orig_model_id", "label", "prefix_step_idx", pcol]].copy()
                    final_eval = final_rows_from_predictions(eval_df)
                    merged = final_eval[["traj_id", "label", pcol]].merge(
                        with_priors.drop(columns=["label"]), on="traj_id", how="left"
                    )
                    row = {
                        "split": split,
                        "variant": variant,
                        "strategy": strategy,
                        "score_mode": score_mode,
                        "prefix_model": prefix_model,
                        "n_traj": int(len(final_eval)),
                        "pos_rate": float(final_eval["label"].mean()),
                        "excluded_top3_gpt52codex": bool(exclude_models),
                    }
                    for step in [0, 5, 10, "last", "all_prefix"]:
                        auc, n = auc_for_step(eval_df, pcol, step)
                        row[f"model_auc_{step}"] = auc
                        row[f"model_n_{step}"] = n
                    for pool in ["heldout_other", "train_other", "all_other"]:
                        for signal in ["all_correct", "mean_success", "any_correct", "none_correct"]:
                            col = f"{pool}_{signal}"
                            row[f"auc_{col}"] = auc_rank(merged["label"], merged[col])
                        row[f"mean_n_{pool}"] = float(merged[f"{pool}_n_other"].mean())
                        row[f"min_n_{pool}"] = float(merged[f"{pool}_n_other"].min())
                    rows.append(row)

    out = pd.DataFrame(rows)
    out.to_csv(output_dir / "other_model_prior_auc_by_run.csv", index=False)
    pd.concat(prior_records, ignore_index=True).to_csv(output_dir / "other_model_prior_scores_by_traj.csv", index=False)

    # Compact summary averaged over I/J variants for each split/strategy/score mode.
    auc_cols = [c for c in out.columns if c.startswith("model_auc_") or c.startswith("auc_")]
    summary = (
        out.groupby(["split", "strategy", "score_mode"], as_index=False)
        .agg({**{c: "mean" for c in auc_cols}, "n_traj": "mean", "pos_rate": "mean"})
    )
    for baseline in [
        "auc_train_other_all_correct",
        "auc_train_other_mean_success",
        "auc_heldout_other_all_correct",
        "auc_heldout_other_mean_success",
        "auc_all_other_all_correct",
        "auc_all_other_mean_success",
    ]:
        summary[f"adv_last_vs_{baseline}"] = summary["model_auc_last"] - summary[baseline]
        summary[f"adv_step0_vs_{baseline}"] = summary["model_auc_0"] - summary[baseline]
    summary.to_csv(output_dir / "other_model_prior_auc_summary.csv", index=False)

    # Markdown report.
    def fmt(x: float) -> str:
        return "-" if pd.isna(x) else f"{x:.3f}"

    lines = []
    lines += [
        "# Other-Model Task-Prior AUC Baseline", "",
        'Public-release English note.', "",
        'Public-release English note.', 
        'Public-release English note.',
        'Public-release English note.',
        'Public-release English note.',
        'Public-release English note.',
        'Public-release English note.', "",
        "## Main Summary", "",
        "| Split | Strategy | Score | N | PosRate | Model AUC step0 | Model AUC last | Train all-correct | Train mean-success | Heldout all-correct | Heldout mean-success | All-other all-correct | All-other mean-success | Last Adv vs TrainMean |", 
        "|:--|:--|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|",
    ]
    for _, r in summary.sort_values(["split", "strategy", "score_mode"]).iterrows():
        lines.append(
            f"| {r['split']} | {r['strategy']} | {r['score_mode']} | {r['n_traj']:.0f} | {r['pos_rate']*100:.1f}% | "
            f"{fmt(r['model_auc_0'])} | {fmt(r['model_auc_last'])} | "
            f"{fmt(r['auc_train_other_all_correct'])} | {fmt(r['auc_train_other_mean_success'])} | "
            f"{fmt(r['auc_heldout_other_all_correct'])} | {fmt(r['auc_heldout_other_mean_success'])} | "
            f"{fmt(r['auc_all_other_all_correct'])} | {fmt(r['auc_all_other_mean_success'])} | "
            f"{fmt(r['adv_last_vs_auc_train_other_mean_success'])} |"
        )

    lines += ["", "## Averaged by Strategy", ""]
    overall = (
        summary.groupby(["strategy", "score_mode"], as_index=False)
        [["model_auc_0", "model_auc_last", "auc_train_other_all_correct", "auc_train_other_mean_success", "auc_all_other_mean_success", "adv_last_vs_auc_train_other_mean_success"]]
        .mean()
    )
    lines += [
        "| Strategy | Score | Model step0 | Model last | Train all-correct | Train mean-success | All-other mean-success | Last Adv vs TrainMean |",
        "|:--|:--|--:|--:|--:|--:|--:|--:|",
    ]
    for _, r in overall.sort_values(["strategy", "score_mode"]).iterrows():
        lines.append(
            f"| {r['strategy']} | {r['score_mode']} | {fmt(r['model_auc_0'])} | {fmt(r['model_auc_last'])} | "
            f"{fmt(r['auc_train_other_all_correct'])} | {fmt(r['auc_train_other_mean_success'])} | "
            f"{fmt(r['auc_all_other_mean_success'])} | {fmt(r['adv_last_vs_auc_train_other_mean_success'])} |"
        )

    lines += [
        "", "## Files", "",
        'Public-release English note.',
        'Public-release English note.',
        'Public-release English note.',
    ]
    (output_dir / "other_model_prior_auc_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(output_dir / "other_model_prior_auc_report.md")


if __name__ == "__main__":
    main()
