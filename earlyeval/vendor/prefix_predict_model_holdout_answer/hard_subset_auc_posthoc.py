#!/usr/bin/env python3
"""Hard-subset AUC analysis against other-model task prior.

Uses train_other_mean_success as task-difficulty prior and evaluates whether prefix
predictors add ranking power inside ambiguous prior bands.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
RUN_NAME = "model_holdout_answer_calibrated_full"
REPORTS_DIR = PROJECT_ROOT / "runs" / RUN_NAME / "reports"
BASE_PRIOR_DIR = REPORTS_DIR / "safe_stop_dual_head_visual_summary" / "problem_diagnosis" / "other_model_prior_auc"
OUT_DIR = BASE_PRIOR_DIR / "hard_subset_auc"

PREFIX_MODELS = {
    "I": "I_LightGBM_Dense_AF",
    "J": "J_LightGBM_Dense_AF_Thought",
}
SPLITS = ["top3", "mid3", "bottom3"]
STRATEGIES = ["strong_reg", "no_model_id"]
VARIANTS = ["I", "J"]
SCORE_MODES = ["raw", "prefix_calibrated"]
STEPS: list[int | str] = [0, 5, 10, "last"]
SUBSETS = [
    ("very_hard_prior_0_0.3", 0.0, 0.3),
    ("ambiguous_prior_0.3_0.7", 0.3, 0.7),
    ("strict_ambiguous_prior_0.4_0.6", 0.4, 0.6),
    ("broad_ambiguous_prior_0.2_0.8", 0.2, 0.8),
    ("easy_prior_0.7_1.0", 0.7, 1.0),
]


def auc_rank(y, score) -> float:
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


def final_rows(df: pd.DataFrame) -> pd.DataFrame:
    idx = df.groupby("traj_id", sort=False)["prefix_step_idx"].idxmax()
    return df.loc[idx].copy()


def step_rows(df: pd.DataFrame, step: int | str) -> pd.DataFrame:
    if step == "last":
        return final_rows(df)
    return df.loc[df["prefix_step_idx"].eq(int(step))].copy()


def step_name(step: int | str) -> str:
    return f"step{step}" if isinstance(step, int) else str(step)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prior-scores", default=str(BASE_PRIOR_DIR / "other_model_prior_scores_by_traj.csv"))
    parser.add_argument("--output-dir", default=str(OUT_DIR))
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prior_scores = pd.read_csv(args.prior_scores)

    rows = []
    for split in SPLITS:
        for variant in VARIANTS:
            prefix_model = PREFIX_MODELS[variant]
            for strategy in STRATEGIES:
                run_prior = prior_scores[
                    (prior_scores["split"].eq(split))
                    & (prior_scores["variant"].eq(variant))
                    & (prior_scores["strategy"].eq(strategy))
                ].copy()
                if run_prior.empty:
                    continue
                run_prior = run_prior.drop_duplicates("traj_id")
                prior_cols = [
                    "traj_id", "train_other_mean_success", "train_other_all_correct",
                    "heldout_other_mean_success", "all_other_mean_success",
                ]
                run_prior = run_prior[prior_cols]

                rd = run_dir(split, variant, strategy)
                pred_path = rd / "test_predictions_shadow_valid_retrain.parquet"
                if not pred_path.exists():
                    continue
                df = pd.read_parquet(pred_path)
                df = df[df["traj_id"].isin(set(run_prior["traj_id"]))].copy()

                for score_mode in SCORE_MODES:
                    pcol = prob_col(prefix_model, score_mode)
                    if pcol not in df.columns:
                        continue
                    for subset_name, lo, hi in SUBSETS:
                        subset_prior = run_prior[
                            run_prior["train_other_mean_success"].between(lo, hi, inclusive="both")
                        ].copy()
                        subset_ids = set(subset_prior["traj_id"])
                        for step in STEPS:
                            part = step_rows(df, step)
                            part = part[part["traj_id"].isin(subset_ids)][
                                ["traj_id", "instance_id", "orig_model_id", "label", pcol]
                            ].merge(subset_prior, on="traj_id", how="left")
                            n = int(len(part))
                            pos = int(part["label"].sum()) if n else 0
                            model_auc = auc_rank(part["label"], part[pcol])
                            train_mean_auc = auc_rank(part["label"], part["train_other_mean_success"])
                            train_all_auc = auc_rank(part["label"], part["train_other_all_correct"])
                            heldout_mean_auc = auc_rank(part["label"], part["heldout_other_mean_success"])
                            all_mean_auc = auc_rank(part["label"], part["all_other_mean_success"])
                            rows.append({
                                "split": split,
                                "variant": variant,
                                "strategy": strategy,
                                "score_mode": score_mode,
                                "prefix_model": prefix_model,
                                "subset": subset_name,
                                "prior_lo": lo,
                                "prior_hi": hi,
                                "step": step_name(step),
                                "n": n,
                                "pos": pos,
                                "pos_rate": pos / n if n else np.nan,
                                "model_auc": model_auc,
                                "train_mean_success_auc": train_mean_auc,
                                "train_all_correct_auc": train_all_auc,
                                "heldout_mean_success_auc": heldout_mean_auc,
                                "all_mean_success_auc": all_mean_auc,
                                "gain_vs_train_mean": model_auc - train_mean_auc,
                                "gain_vs_all_mean": model_auc - all_mean_auc,
                            })

    detail = pd.DataFrame(rows)
    detail.to_csv(output_dir / "hard_subset_auc_by_run_step.csv", index=False)

    summary = (
        detail.groupby(["subset", "split", "strategy", "score_mode", "step"], as_index=False)
        .agg(
            n=("n", "mean"),
            pos_rate=("pos_rate", "mean"),
            model_auc=("model_auc", "mean"),
            train_mean_success_auc=("train_mean_success_auc", "mean"),
            train_all_correct_auc=("train_all_correct_auc", "mean"),
            heldout_mean_success_auc=("heldout_mean_success_auc", "mean"),
            all_mean_success_auc=("all_mean_success_auc", "mean"),
            gain_vs_train_mean=("gain_vs_train_mean", "mean"),
            gain_vs_all_mean=("gain_vs_all_mean", "mean"),
        )
    )
    summary.to_csv(output_dir / "hard_subset_auc_summary.csv", index=False)

    compact = (
        summary.groupby(["subset", "split", "step"], as_index=False)
        .agg(
            n=("n", "mean"),
            pos_rate=("pos_rate", "mean"),
            model_auc=("model_auc", "mean"),
            train_mean_success_auc=("train_mean_success_auc", "mean"),
            train_all_correct_auc=("train_all_correct_auc", "mean"),
            heldout_mean_success_auc=("heldout_mean_success_auc", "mean"),
            all_mean_success_auc=("all_mean_success_auc", "mean"),
            gain_vs_train_mean=("gain_vs_train_mean", "mean"),
            gain_vs_all_mean=("gain_vs_all_mean", "mean"),
        )
    )
    compact.to_csv(output_dir / "hard_subset_auc_compact.csv", index=False)

    def fmt(x: float) -> str:
        return "-" if pd.isna(x) else f"{x:.3f}"
    def fmt_pct(x: float) -> str:
        return "-" if pd.isna(x) else f"{x*100:.1f}%"

    lines = [
        "# Hard-Subset AUC over Task Prior", "",
        'Public-release English note.', "",
        'Public-release English note.',
        'Public-release English note.',
        'Public-release English note.', "",
    ]

    for subset_name, _, _ in SUBSETS:
        sub = compact[compact["subset"].eq(subset_name)].copy()
        if sub.empty:
            continue
        lines += [f"## {subset_name}", "", "| Split | Step | N | PosRate | Model AUC | TrainMean Prior AUC | Gain | HeldoutMean AUC |", "|:--|:--|--:|--:|--:|--:|--:|--:|"]
        for _, r in sub.sort_values(["split", "step"]).iterrows():
            lines.append(
                f"| {r['split']} | {r['step']} | {r['n']:.0f} | {fmt_pct(r['pos_rate'])} | "
                f"{fmt(r['model_auc'])} | {fmt(r['train_mean_success_auc'])} | "
                f"{fmt(r['gain_vs_train_mean'])} | {fmt(r['heldout_mean_success_auc'])} |"
            )
        lines.append("")

    # Quick takeaways from ambiguous subsets.
    lines += ["## Quick Takeaways", ""]
    for subset_name in ["ambiguous_prior_0.3_0.7", "strict_ambiguous_prior_0.4_0.6"]:
        sub = compact[(compact["subset"].eq(subset_name)) & (compact["step"].eq("last"))]
        if sub.empty:
            continue
        lines.append(f"### {subset_name} / last")
        for _, r in sub.sort_values("split").iterrows():
            lines.append(
                f"- `{r['split']}`: model AUC `{fmt(r['model_auc'])}`, prior AUC `{fmt(r['train_mean_success_auc'])}`, gain `{fmt(r['gain_vs_train_mean'])}`, N≈`{r['n']:.0f}`."
            )
        lines.append("")

    lines += [
        "## Files", "",
        'Public-release English note.',
        'Public-release English note.',
        'Public-release English note.',
    ]
    (output_dir / "hard_subset_auc_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(output_dir / "hard_subset_auc_report.md")


if __name__ == "__main__":
    main()
