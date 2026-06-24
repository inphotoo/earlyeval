#!/usr/bin/env python3
"""Policy sweep for clean-top threshold-decided AUC gain.

Question: can policy knobs (threshold, min_step, consecutive k, policy mode) make
clean-top decided-subset model AUC exceed the other-model task-prior baseline?
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
OUT_DIR = BASE_PRIOR_DIR / "top3_policy_auc_gain_sweep"

PREFIX_MODELS = {
    "I": "I_LightGBM_Dense_AF",
    "J": "J_LightGBM_Dense_AF_Thought",
}
STRATEGIES = ["strong_reg", "no_model_id"]
VARIANTS = ["I", "J"]
SCORE_MODES = ["raw", "prefix_calibrated"]
THRESHOLDS = [0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.97, 0.99]
MIN_STEPS = [0, 5, 10, 15, 20, 30]
CONSECUTIVE = [1, 2, 3, 4]
POLICY_MODES = ["symmetric", "success_only", "failure_only"]
TOP3_EXCLUDE = "gpt-5-2-codex"


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


def run_dir(variant: str, strategy: str) -> Path:
    return REPORTS_DIR / f"per_instance_model_valid3_top3_{variant.lower()}_{strategy}_retrain"


def decision_frame(df: pd.DataFrame, prob_col: str, *, policy_mode: str, threshold: float, min_step: int, consecutive: int) -> pd.DataFrame:
    records = _records(df, [prob_col])
    traj_ids = [str(k) for k, _ in df.groupby("traj_id", sort=False)]
    rows = []
    if policy_mode == "symmetric":
        success_thr = threshold
        failure_thr = 1.0 - threshold
    elif policy_mode == "success_only":
        success_thr = threshold
        failure_thr = -float("inf")
    elif policy_mode == "failure_only":
        success_thr = float("inf")
        failure_thr = 1.0 - threshold
    else:
        raise ValueError(policy_mode)

    for traj_id, record in zip(traj_ids, records):
        decided, decision, decision_step, decision_prob = _decide(
            record["steps"],
            record["probs"][prob_col],
            p0=record["p0"][prob_col],
            success_thr=success_thr,
            failure_thr=failure_thr,
            min_step=min_step,
            consecutive=consecutive,
            delta_up=0.0,
            delta_down=0.0,
        )
        n_steps = int(record["n_steps"])
        rows.append(
            {
                "traj_id": traj_id,
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
    return pd.DataFrame(rows)


def summarize_decisions(decisions: pd.DataFrame, *, subset_name: str, lo: float, hi: float) -> dict[str, float | int | str]:
    universe = decisions[decisions["train_other_mean_success"].between(lo, hi, inclusive="both")].copy()
    decided = universe[universe["decided"]].copy()
    n_all = int(len(universe))
    n_dec = int(len(decided))
    pred = decided["decision"].map({"success": 1, "failure": 0}) if n_dec else pd.Series(dtype=float)
    model_auc = auc_rank(decided["label"], decided["decision_prob"])
    prior_auc = auc_rank(decided["label"], decided["train_other_mean_success"])
    heldout_auc = auc_rank(decided["label"], decided["heldout_other_mean_success"])
    all_auc = auc_rank(decided["label"], decided["all_other_mean_success"])
    return {
        "subset": subset_name,
        "prior_lo": lo,
        "prior_hi": hi,
        "n_all_subset": n_all,
        "n_decided": n_dec,
        "coverage_within_subset": n_dec / n_all if n_all else np.nan,
        "pos_rate_decided": float(decided["label"].mean()) if n_dec else np.nan,
        "decision_acc": float((pred.to_numpy() == decided["label"].to_numpy()).mean()) if n_dec else np.nan,
        "mean_decision_step": float(decided["decision_step"].mean()) if n_dec else np.nan,
        "save_pct_run_denominator": float(decided["saved_steps"].sum() / decisions["n_steps"].sum() * 100.0) if len(decisions) else np.nan,
        "model_auc_decided": model_auc,
        "train_mean_success_auc_decided": prior_auc,
        "heldout_mean_success_auc_decided": heldout_auc,
        "all_mean_success_auc_decided": all_auc,
        "gain_vs_train_mean_decided": model_auc - prior_auc,
        "gain_vs_all_mean_decided": model_auc - all_auc,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prior-scores", default=str(BASE_PRIOR_DIR / "other_model_prior_scores_by_traj.csv"))
    parser.add_argument("--output-dir", default=str(OUT_DIR))
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    prior = pd.read_csv(args.prior_scores)
    prior = prior[(prior["split"].eq("top3")) & (~prior["orig_model_id"].astype(str).str.contains(TOP3_EXCLUDE, regex=False))]
    prior = prior.drop_duplicates(["traj_id"])[
        ["traj_id", "orig_model_id", "train_other_mean_success", "heldout_other_mean_success", "all_other_mean_success"]
    ]
    valid_ids = set(prior["traj_id"])
    subsets = [("all", 0.0, 1.0), ("ambiguous_0.3_0.7", 0.3, 0.7), ("not_easy_0_0.8", 0.0, 0.8), ("easy_0.7_1", 0.7, 1.0)]

    rows = []
    for variant, prefix_model in PREFIX_MODELS.items():
        for strategy in STRATEGIES:
            pred_path = run_dir(variant, strategy) / "test_predictions_shadow_valid_retrain.parquet"
            if not pred_path.exists():
                continue
            df = pd.read_parquet(pred_path)
            df = df[df["traj_id"].isin(valid_ids)].copy()
            for score_mode in SCORE_MODES:
                prob_col = _prob_col(prefix_model, score_mode)
                if prob_col not in df.columns:
                    continue
                for policy_mode in POLICY_MODES:
                    for threshold in THRESHOLDS:
                        for min_step in MIN_STEPS:
                            for consecutive in CONSECUTIVE:
                                # Avoid impossible high-k with too-small selected sample? still compute.
                                decisions = decision_frame(
                                    df,
                                    prob_col,
                                    policy_mode=policy_mode,
                                    threshold=threshold,
                                    min_step=min_step,
                                    consecutive=consecutive,
                                ).merge(prior.drop(columns=["orig_model_id"]), on="traj_id", how="left")
                                for subset_name, lo, hi in subsets:
                                    row = summarize_decisions(decisions, subset_name=subset_name, lo=lo, hi=hi)
                                    row.update(
                                        {
                                            "variant": variant,
                                            "strategy": strategy,
                                            "score_mode": score_mode,
                                            "prefix_model": prefix_model,
                                            "policy_mode": policy_mode,
                                            "threshold": threshold,
                                            "failure_thr": 1.0 - threshold,
                                            "min_step": min_step,
                                            "consecutive": consecutive,
                                        }
                                    )
                                    rows.append(row)

    detail = pd.DataFrame(rows)
    detail.to_csv(output_dir / "top3_policy_auc_gain_sweep_by_run.csv", index=False)

    summary = (
        detail.groupby(["subset", "policy_mode", "threshold", "min_step", "consecutive", "strategy", "score_mode"], as_index=False)
        .agg(
            n_all_subset=("n_all_subset", "mean"),
            n_decided=("n_decided", "mean"),
            coverage_within_subset=("coverage_within_subset", "mean"),
            decision_acc=("decision_acc", "mean"),
            mean_decision_step=("mean_decision_step", "mean"),
            save_pct_run_denominator=("save_pct_run_denominator", "mean"),
            model_auc_decided=("model_auc_decided", "mean"),
            train_mean_success_auc_decided=("train_mean_success_auc_decided", "mean"),
            gain_vs_train_mean_decided=("gain_vs_train_mean_decided", "mean"),
        )
    )
    summary.to_csv(output_dir / "top3_policy_auc_gain_sweep_summary.csv", index=False)

    compact = (
        summary.groupby(["subset", "policy_mode", "threshold", "min_step", "consecutive"], as_index=False)
        .agg(
            n_decided=("n_decided", "mean"),
            coverage_within_subset=("coverage_within_subset", "mean"),
            decision_acc=("decision_acc", "mean"),
            mean_decision_step=("mean_decision_step", "mean"),
            save_pct_run_denominator=("save_pct_run_denominator", "mean"),
            model_auc_decided=("model_auc_decided", "mean"),
            train_mean_success_auc_decided=("train_mean_success_auc_decided", "mean"),
            gain_vs_train_mean_decided=("gain_vs_train_mean_decided", "mean"),
        )
    )
    compact.to_csv(output_dir / "top3_policy_auc_gain_sweep_compact.csv", index=False)

    # Candidate tables under minimum support constraints.
    candidate_sets = []
    for min_n, min_cov in [(30, 0.05), (50, 0.10), (100, 0.20)]:
        cand = compact[
            (compact["n_decided"] >= min_n)
            & (compact["coverage_within_subset"] >= min_cov)
            & compact["model_auc_decided"].notna()
        ].copy()
        cand["min_n"] = min_n
        cand["min_cov"] = min_cov
        cand = cand.sort_values(["subset", "gain_vs_train_mean_decided"], ascending=[True, False])
        candidate_sets.append(cand.groupby("subset", as_index=False).head(10))
    candidates = pd.concat(candidate_sets, ignore_index=True) if candidate_sets else pd.DataFrame()
    candidates.to_csv(output_dir / "top3_policy_auc_gain_positive_candidates.csv", index=False)

    def fmt(x: float) -> str:
        return "-" if pd.isna(x) else f"{x:.3f}"
    def pct(x: float) -> str:
        return "-" if pd.isna(x) else f"{x*100:.1f}%"

    lines = [
        "# Clean-Top Policy AUC Gain Sweep", "",
        'Public-release English note.', "",
        'Public-release English note.', "",
        "## Best Candidates with Support Constraints", "",
    ]
    for min_n, min_cov in [(30, 0.05), (50, 0.10), (100, 0.20)]:
        lines += [f"### min N decided >= {min_n}, coverage >= {min_cov:.0%}", "", "| Subset | Policy | Thr | MinStep | k | Ndec | Cov | Acc | Model AUC | Prior AUC | Gain |", "|:--|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|"]
        cand = compact[
            (compact["n_decided"] >= min_n)
            & (compact["coverage_within_subset"] >= min_cov)
            & compact["model_auc_decided"].notna()
        ].sort_values("gain_vs_train_mean_decided", ascending=False).head(20)
        for _, r in cand.iterrows():
            lines.append(
                f"| {r['subset']} | {r['policy_mode']} | {r['threshold']:.2f} | {int(r['min_step'])} | {int(r['consecutive'])} | "
                f"{r['n_decided']:.0f} | {pct(r['coverage_within_subset'])} | {pct(r['decision_acc'])} | "
                f"{fmt(r['model_auc_decided'])} | {fmt(r['train_mean_success_auc_decided'])} | {fmt(r['gain_vs_train_mean_decided'])} |"
            )
        lines.append("")

    # Best by subset regardless but requiring n>=30.
    lines += ["## Best by Subset, min N decided >= 30", "", "| Subset | Policy | Thr | MinStep | k | Ndec | Cov | Model AUC | Prior AUC | Gain |", "|:--|:--|--:|--:|--:|--:|--:|--:|--:|--:|"]
    cand = compact[(compact["n_decided"] >= 30) & compact["model_auc_decided"].notna()].copy()
    for _, r in cand.sort_values(["subset", "gain_vs_train_mean_decided"], ascending=[True, False]).groupby("subset", as_index=False).head(5).iterrows():
        lines.append(
            f"| {r['subset']} | {r['policy_mode']} | {r['threshold']:.2f} | {int(r['min_step'])} | {int(r['consecutive'])} | "
            f"{r['n_decided']:.0f} | {pct(r['coverage_within_subset'])} | {fmt(r['model_auc_decided'])} | "
            f"{fmt(r['train_mean_success_auc_decided'])} | {fmt(r['gain_vs_train_mean_decided'])} |"
        )

    lines += [
        "", "## Files", "",
        "- `top3_policy_auc_gain_sweep_by_run.csv`",
        "- `top3_policy_auc_gain_sweep_summary.csv`",
        "- `top3_policy_auc_gain_sweep_compact.csv`",
        "- `top3_policy_auc_gain_positive_candidates.csv`",
    ]
    (output_dir / "top3_policy_auc_gain_sweep_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(output_dir / "top3_policy_auc_gain_sweep_report.md")


if __name__ == "__main__":
    main()
