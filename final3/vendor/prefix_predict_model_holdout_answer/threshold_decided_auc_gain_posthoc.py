#!/usr/bin/env python3
"""AUC gain on threshold-decided early-stop subsets.

For each symmetric threshold policy, compute AUC on only trajectories that would
be early-stopped. This is different from full trajectory-level AUC at a fixed
step: it is a selected-subset diagnostic for early decisions.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
from process_signal_policy_rescue_posthoc import _records, _decide, _prob_col

RUN_NAME = "model_holdout_answer_calibrated_full"
REPORTS_DIR = PROJECT_ROOT / "runs" / RUN_NAME / "reports"
BASE_PRIOR_DIR = REPORTS_DIR / "safe_stop_dual_head_visual_summary" / "problem_diagnosis" / "other_model_prior_auc"
OUT_DIR = BASE_PRIOR_DIR / "threshold_decided_auc_gain"

PREFIX_MODELS = {
    "I": "I_LightGBM_Dense_AF",
    "J": "J_LightGBM_Dense_AF_Thought",
}
SPLITS = ["top3", "mid3", "bottom3"]
STRATEGIES = ["strong_reg", "no_model_id"]
VARIANTS = ["I", "J"]
SCORE_MODES = ["raw", "prefix_calibrated"]
THRESHOLDS = [0.80, 0.85, 0.90, 0.95, 0.97]
SUBSETS = [
    ("all", 0.0, 1.0),
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


def run_dir(split: str, variant: str, strategy: str) -> Path:
    return REPORTS_DIR / f"per_instance_model_valid3_{split}_{variant.lower()}_{strategy}_retrain"


def build_decisions(df: pd.DataFrame, prob_col: str, threshold: float) -> pd.DataFrame:
    records = _records(df, [prob_col])
    rows = []
    failure_thr = 1.0 - threshold
    for record in records:
        decided, decision, decision_step, decision_prob = _decide(
            record["steps"],
            record["probs"][prob_col],
            p0=record["p0"][prob_col],
            success_thr=threshold,
            failure_thr=failure_thr,
            min_step=10,
            consecutive=2,
            delta_up=0.0,
            delta_down=0.0,
        )
        n_steps = int(record["n_steps"])
        rows.append(
            {
                "traj_id": str(record.get("traj_id", "")),
                "agent_model": record["agent_model"],
                "label": int(record["label"]),
                "n_steps": n_steps,
                "decided": bool(decided),
                "decision": decision,
                "decision_step": int(decision_step),
                "decision_prob": float(decision_prob) if decided else np.nan,
                "saved_steps": max(n_steps - int(decision_step) - 1, 0) if decided else 0,
            }
        )
    # _records currently does not include traj_id, so recover by grouping order.
    traj_ids = [str(k) for k, _ in df.groupby("traj_id", sort=False)]
    for row, traj_id in zip(rows, traj_ids):
        row["traj_id"] = traj_id
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prior-scores", default=str(BASE_PRIOR_DIR / "other_model_prior_scores_by_traj.csv"))
    parser.add_argument("--output-dir", default=str(OUT_DIR))
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prior_scores = pd.read_csv(args.prior_scores)

    rows = []
    decision_rows = []
    for split in SPLITS:
        for variant in VARIANTS:
            prefix_model = PREFIX_MODELS[variant]
            for strategy in STRATEGIES:
                run_prior = prior_scores[
                    (prior_scores["split"].eq(split))
                    & (prior_scores["variant"].eq(variant))
                    & (prior_scores["strategy"].eq(strategy))
                ].drop_duplicates("traj_id")
                if run_prior.empty:
                    continue
                prior_cols = [
                    "traj_id", "train_other_mean_success", "train_other_all_correct",
                    "heldout_other_mean_success", "all_other_mean_success",
                ]
                run_prior = run_prior[prior_cols]
                valid_traj_ids = set(run_prior["traj_id"])

                rd = run_dir(split, variant, strategy)
                pred_path = rd / "test_predictions_shadow_valid_retrain.parquet"
                if not pred_path.exists():
                    continue
                df = pd.read_parquet(pred_path)
                df = df[df["traj_id"].isin(valid_traj_ids)].copy()

                for score_mode in SCORE_MODES:
                    prob_col = _prob_col(prefix_model, score_mode)
                    if prob_col not in df.columns:
                        continue
                    for threshold in THRESHOLDS:
                        decisions = build_decisions(df, prob_col, threshold)
                        decisions = decisions.merge(run_prior, on="traj_id", how="left")
                        decisions["split"] = split
                        decisions["variant"] = variant
                        decisions["strategy"] = strategy
                        decisions["score_mode"] = score_mode
                        decisions["threshold"] = threshold
                        decision_rows.append(decisions)

                        total_n = len(decisions)
                        for subset_name, lo, hi in SUBSETS:
                            part_all = decisions[
                                decisions["train_other_mean_success"].between(lo, hi, inclusive="both")
                            ].copy()
                            part = part_all[part_all["decided"]].copy()
                            n_all = int(len(part_all))
                            n_dec = int(len(part))
                            pos = int(part["label"].sum()) if n_dec else 0
                            model_auc = auc_rank(part["label"], part["decision_prob"])
                            train_mean_auc = auc_rank(part["label"], part["train_other_mean_success"])
                            all_mean_auc = auc_rank(part["label"], part["all_other_mean_success"])
                            heldout_mean_auc = auc_rank(part["label"], part["heldout_other_mean_success"])
                            acc = np.nan
                            if n_dec:
                                pred_label = part["decision"].map({"success": 1, "failure": 0}).astype(int)
                                acc = float((pred_label.to_numpy() == part["label"].to_numpy()).mean())
                            rows.append(
                                {
                                    "split": split,
                                    "variant": variant,
                                    "strategy": strategy,
                                    "score_mode": score_mode,
                                    "threshold": threshold,
                                    "failure_thr": 1.0 - threshold,
                                    "subset": subset_name,
                                    "n_all_subset": n_all,
                                    "n_decided": n_dec,
                                    "coverage_within_subset": n_dec / n_all if n_all else np.nan,
                                    "coverage_over_run": n_dec / total_n if total_n else np.nan,
                                    "pos_rate_decided": pos / n_dec if n_dec else np.nan,
                                    "decision_acc": acc,
                                    "mean_decision_step": float(part["decision_step"].mean()) if n_dec else np.nan,
                                    "save_pct_decided_den_run_steps": float(part["saved_steps"].sum() / decisions["n_steps"].sum() * 100.0) if len(decisions) else np.nan,
                                    "model_auc_decided": model_auc,
                                    "train_mean_success_auc_decided": train_mean_auc,
                                    "all_mean_success_auc_decided": all_mean_auc,
                                    "heldout_mean_success_auc_decided": heldout_mean_auc,
                                    "gain_vs_train_mean_decided": model_auc - train_mean_auc,
                                    "gain_vs_all_mean_decided": model_auc - all_mean_auc,
                                }
                            )

    detail = pd.DataFrame(rows)
    detail.to_csv(output_dir / "threshold_decided_auc_gain_by_run.csv", index=False)
    if decision_rows:
        pd.concat(decision_rows, ignore_index=True).to_csv(output_dir / "threshold_decisions_with_prior_scores.csv", index=False)

    summary = (
        detail.groupby(["subset", "split", "strategy", "score_mode", "threshold"], as_index=False)
        .agg(
            n_all_subset=("n_all_subset", "mean"),
            n_decided=("n_decided", "mean"),
            coverage_within_subset=("coverage_within_subset", "mean"),
            decision_acc=("decision_acc", "mean"),
            mean_decision_step=("mean_decision_step", "mean"),
            save_pct_decided_den_run_steps=("save_pct_decided_den_run_steps", "mean"),
            model_auc_decided=("model_auc_decided", "mean"),
            train_mean_success_auc_decided=("train_mean_success_auc_decided", "mean"),
            all_mean_success_auc_decided=("all_mean_success_auc_decided", "mean"),
            heldout_mean_success_auc_decided=("heldout_mean_success_auc_decided", "mean"),
            gain_vs_train_mean_decided=("gain_vs_train_mean_decided", "mean"),
            gain_vs_all_mean_decided=("gain_vs_all_mean_decided", "mean"),
        )
    )
    summary.to_csv(output_dir / "threshold_decided_auc_gain_summary.csv", index=False)

    compact = (
        summary.groupby(["subset", "split", "threshold"], as_index=False)
        .agg(
            n_all_subset=("n_all_subset", "mean"),
            n_decided=("n_decided", "mean"),
            coverage_within_subset=("coverage_within_subset", "mean"),
            decision_acc=("decision_acc", "mean"),
            mean_decision_step=("mean_decision_step", "mean"),
            save_pct_decided_den_run_steps=("save_pct_decided_den_run_steps", "mean"),
            model_auc_decided=("model_auc_decided", "mean"),
            train_mean_success_auc_decided=("train_mean_success_auc_decided", "mean"),
            heldout_mean_success_auc_decided=("heldout_mean_success_auc_decided", "mean"),
            gain_vs_train_mean_decided=("gain_vs_train_mean_decided", "mean"),
        )
    )
    compact.to_csv(output_dir / "threshold_decided_auc_gain_compact.csv", index=False)

    def fmt(x: float) -> str:
        return "-" if pd.isna(x) else f"{x:.3f}"
    def pct(x: float) -> str:
        return "-" if pd.isna(x) else f"{x*100:.1f}%"

    lines = [
        "# Threshold-Decided AUC Gain Report", "",
        "口径：对每个双端阈值 `p>=thr / p<=1-thr`，固定 `min_step=10,k=2`，只在实际触发 early-stop 的 trajectory 上计算 AUC。", "",
        "注意：这是 selected/decided subset AUC，不等同于全量 trajectory fixed-step AUC；高阈值会选择更容易/更高置信的样本，因此需要同时看 coverage。", "",
    ]
    for subset_name in ["all", "ambiguous_prior_0.3_0.7", "strict_ambiguous_prior_0.4_0.6"]:
        sub = compact[compact["subset"].eq(subset_name)].copy()
        if sub.empty:
            continue
        lines += [f"## {subset_name}", "", "| Split | Thr | Ndec | CovSubset | Acc | AvgStep | Model AUC | Prior AUC | Gain |", "|:--|--:|--:|--:|--:|--:|--:|--:|--:|"]
        for _, r in sub.sort_values(["split", "threshold"]).iterrows():
            lines.append(
                f"| {r['split']} | {r['threshold']:.2f} | {r['n_decided']:.0f} | {pct(r['coverage_within_subset'])} | "
                f"{pct(r['decision_acc'])} | {r['mean_decision_step']:.1f} | {fmt(r['model_auc_decided'])} | "
                f"{fmt(r['train_mean_success_auc_decided'])} | {fmt(r['gain_vs_train_mean_decided'])} |"
            )
        lines.append("")

    lines += ["## Quick Takeaways", ""]
    for subset_name in ["ambiguous_prior_0.3_0.7", "strict_ambiguous_prior_0.4_0.6"]:
        sub = compact[(compact["subset"].eq(subset_name)) & (compact["threshold"].eq(0.90))]
        if sub.empty:
            continue
        lines.append(f"### {subset_name} / threshold 0.90")
        for _, r in sub.sort_values("split").iterrows():
            lines.append(
                f"- `{r['split']}`: decided N≈`{r['n_decided']:.0f}`, coverage `{pct(r['coverage_within_subset'])}`, model AUC `{fmt(r['model_auc_decided'])}`, prior AUC `{fmt(r['train_mean_success_auc_decided'])}`, gain `{fmt(r['gain_vs_train_mean_decided'])}`."
            )
        lines.append("")

    lines += [
        "## Files", "",
        "- `threshold_decided_auc_gain_by_run.csv`：逐 split / variant / strategy / score / threshold 明细。",
        "- `threshold_decided_auc_gain_summary.csv`：按 strategy/score 汇总。",
        "- `threshold_decided_auc_gain_compact.csv`：跨 strategy/score 平均后的核心表。",
        "- `threshold_decisions_with_prior_scores.csv`：每条 trajectory 的 threshold decision 与 prior 分数。",
    ]
    (output_dir / "threshold_decided_auc_gain_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(output_dir / "threshold_decided_auc_gain_report.md")


if __name__ == "__main__":
    main()
