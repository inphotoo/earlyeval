#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
from process_signal_policy_rescue_posthoc import _decide, _prob_col

RUN_NAME = "model_holdout_answer_calibrated_full"
REPORTS_DIR = PROJECT_ROOT / "runs" / RUN_NAME / "reports"
BASE_PRIOR_DIR = REPORTS_DIR / "safe_stop_dual_head_visual_summary" / "problem_diagnosis" / "other_model_prior_auc"
OUT_DIR = BASE_PRIOR_DIR / "top3_policy_auc_gain_sweep"

PREFIX_MODELS = {"I": "I_LightGBM_Dense_AF", "J": "J_LightGBM_Dense_AF_Thought"}
STRATEGIES = ["strong_reg", "no_model_id"]
VARIANTS = ["I", "J"]
SCORE_MODES = ["raw", "prefix_calibrated"]
THRESHOLDS = [0.75, 0.80, 0.85, 0.90, 0.95, 0.97]
MIN_STEPS = [0, 5, 10, 15, 20]
CONSECUTIVE = [1, 2, 3]
POLICY_MODES = ["symmetric", "success_only", "failure_only"]
TOP3_EXCLUDE = "gpt-5-2-codex"
SUBSETS = [("all", 0.0, 1.0), ("ambiguous_0.3_0.7", 0.3, 0.7), ("not_easy_0_0.8", 0.0, 0.8), ("easy_0.7_1", 0.7, 1.0)]


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


def make_records(df: pd.DataFrame, prob_col: str) -> list[dict]:
    records = []
    needed = ["traj_id", "orig_model_id", "label", "prefix_step_idx", prob_col]
    for traj_id, g in df[needed].groupby("traj_id", sort=False):
        g = g.sort_values("prefix_step_idx")
        probs = g[prob_col].to_numpy(dtype=np.float64)
        records.append({
            "traj_id": str(traj_id),
            "agent_model": str(g["orig_model_id"].iloc[0]),
            "label": int(g["label"].iloc[0]),
            "n_steps": int(len(g)),
            "steps": g["prefix_step_idx"].to_numpy(dtype=np.int32),
            "probs": probs,
            "p0": float(probs[0]),
        })
    return records


def eval_policy(records: list[dict], prior_map: dict[str, dict], *, policy_mode: str, threshold: float, min_step: int, consecutive: int, subset_name: str, lo: float, hi: float) -> dict:
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

    rows = []
    n_all_subset = 0
    total_steps_all = 0
    for rec in records:
        prior = prior_map.get(rec["traj_id"])
        if prior is None:
            continue
        train_prior = prior["train_other_mean_success"]
        if not (lo <= train_prior <= hi):
            continue
        n_all_subset += 1
        total_steps_all += rec["n_steps"]
        decided, decision, decision_step, decision_prob = _decide(
            rec["steps"], rec["probs"], p0=rec["p0"],
            success_thr=success_thr, failure_thr=failure_thr,
            min_step=min_step, consecutive=consecutive,
            delta_up=0.0, delta_down=0.0,
        )
        if not decided:
            continue
        rows.append({
            "label": rec["label"],
            "decision": decision,
            "decision_prob": decision_prob,
            "train_prior": train_prior,
            "heldout_prior": prior["heldout_other_mean_success"],
            "all_prior": prior["all_other_mean_success"],
            "decision_step": decision_step,
            "saved_steps": max(rec["n_steps"] - decision_step - 1, 0),
        })
    dec = pd.DataFrame(rows)
    n_dec = int(len(dec))
    if n_dec:
        pred = dec["decision"].map({"success": 1, "failure": 0}).astype(int).to_numpy()
        labels = dec["label"].astype(int).to_numpy()
        model_auc = auc_rank(labels, dec["decision_prob"])
        prior_auc = auc_rank(labels, dec["train_prior"])
        all_auc = auc_rank(labels, dec["all_prior"])
        heldout_auc = auc_rank(labels, dec["heldout_prior"])
        saved_steps = float(dec["saved_steps"].sum())
        n_pos_decided = int((labels == 1).sum())
        n_neg_decided = int((labels == 0).sum())
    else:
        model_auc = prior_auc = all_auc = heldout_auc = np.nan
        pred = labels = np.array([])
        saved_steps = 0.0
        n_pos_decided = 0
        n_neg_decided = 0
    return {
        "subset": subset_name,
        "n_all_subset": n_all_subset,
        "n_decided": n_dec,
        "n_pos_decided": n_pos_decided,
        "n_neg_decided": n_neg_decided,
        "coverage_within_subset": n_dec / n_all_subset if n_all_subset else np.nan,
        "decision_acc": float((pred == labels).mean()) if n_dec else np.nan,
        "pos_rate_decided": float(labels.mean()) if n_dec else np.nan,
        "mean_decision_step": float(dec["decision_step"].mean()) if n_dec else np.nan,
        "save_pct_subset_denominator": float(saved_steps / total_steps_all * 100.0) if total_steps_all else np.nan,
        "model_auc_decided": model_auc,
        "train_mean_success_auc_decided": prior_auc,
        "all_mean_success_auc_decided": all_auc,
        "heldout_mean_success_auc_decided": heldout_auc,
        "gain_vs_train_mean_decided": model_auc - prior_auc,
        "gain_vs_all_mean_decided": model_auc - all_auc,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=str(OUT_DIR))
    args = parser.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    prior = pd.read_csv(BASE_PRIOR_DIR / "other_model_prior_scores_by_traj.csv")
    prior = prior[(prior["split"].eq("top3")) & (~prior["orig_model_id"].astype(str).str.contains(TOP3_EXCLUDE, regex=False))]
    prior = prior.drop_duplicates("traj_id")
    prior_map = prior.set_index("traj_id")[["train_other_mean_success", "heldout_other_mean_success", "all_other_mean_success"]].to_dict("index")
    valid_ids = set(prior_map)

    rows = []
    for variant, prefix_model in PREFIX_MODELS.items():
        for strategy in STRATEGIES:
            p = run_dir(variant, strategy) / "test_predictions_shadow_valid_retrain.parquet"
            if not p.exists():
                continue
            df = pd.read_parquet(p)
            df = df[df["traj_id"].isin(valid_ids)].copy()
            for score_mode in SCORE_MODES:
                prob_col = _prob_col(prefix_model, score_mode)
                if prob_col not in df.columns:
                    continue
                records = make_records(df, prob_col)
                for policy_mode in POLICY_MODES:
                    for threshold in THRESHOLDS:
                        for min_step in MIN_STEPS:
                            for consecutive in CONSECUTIVE:
                                for subset_name, lo, hi in SUBSETS:
                                    row = eval_policy(
                                        records, prior_map,
                                        policy_mode=policy_mode, threshold=threshold,
                                        min_step=min_step, consecutive=consecutive,
                                        subset_name=subset_name, lo=lo, hi=hi,
                                    )
                                    row.update({
                                        "variant": variant,
                                        "strategy": strategy,
                                        "score_mode": score_mode,
                                        "policy_mode": policy_mode,
                                        "threshold": threshold,
                                        "failure_thr": 1.0 - threshold,
                                        "min_step": min_step,
                                        "consecutive": consecutive,
                                    })
                                    rows.append(row)
    detail = pd.DataFrame(rows)
    detail.to_csv(out / "top3_policy_auc_gain_sweep_by_run.csv", index=False)
    compact = detail.groupby(["subset", "policy_mode", "threshold", "min_step", "consecutive"], as_index=False).agg(
        n_all_subset=("n_all_subset", "mean"),
        n_decided=("n_decided", "mean"),
        n_pos_decided=("n_pos_decided", "mean"),
        n_neg_decided=("n_neg_decided", "mean"),
        coverage_within_subset=("coverage_within_subset", "mean"),
        decision_acc=("decision_acc", "mean"),
        pos_rate_decided=("pos_rate_decided", "mean"),
        mean_decision_step=("mean_decision_step", "mean"),
        save_pct_subset_denominator=("save_pct_subset_denominator", "mean"),
        model_auc_decided=("model_auc_decided", "mean"),
        train_mean_success_auc_decided=("train_mean_success_auc_decided", "mean"),
        all_mean_success_auc_decided=("all_mean_success_auc_decided", "mean"),
        heldout_mean_success_auc_decided=("heldout_mean_success_auc_decided", "mean"),
        gain_vs_train_mean_decided=("gain_vs_train_mean_decided", "mean"),
        gain_vs_all_mean_decided=("gain_vs_all_mean_decided", "mean"),
    )
    compact.to_csv(out / "top3_policy_auc_gain_sweep_compact.csv", index=False)

    candidates = compact[compact["model_auc_decided"].notna()].copy()
    candidates["usable_30_5pct"] = (candidates["n_decided"] >= 30) & (candidates["coverage_within_subset"] >= 0.05)
    candidates["usable_50_10pct"] = (candidates["n_decided"] >= 50) & (candidates["coverage_within_subset"] >= 0.10)
    candidates["balanced_20_20"] = (candidates["n_pos_decided"] >= 20) & (candidates["n_neg_decided"] >= 20)
    candidates.sort_values(["gain_vs_train_mean_decided", "n_decided"], ascending=[False, False]).to_csv(out / "top3_policy_auc_gain_candidates_sorted.csv", index=False)

    def fmt(x): return "-" if pd.isna(x) else f"{x:.3f}"
    def pct(x): return "-" if pd.isna(x) else f"{x*100:.1f}%"
    lines = ["# Clean-Top Policy AUC Gain Sweep", "", 'Public-release English note.', ""]
    for title, filt in [
        ("All candidates, min N>=30 and coverage>=5%", (candidates.n_decided >= 30) & (candidates.coverage_within_subset >= 0.05)),
        ("Stricter candidates, min N>=50 and coverage>=10%", (candidates.n_decided >= 50) & (candidates.coverage_within_subset >= 0.10)),
        ("Balanced candidates, min pos>=20 and neg>=20", (candidates.n_pos_decided >= 20) & (candidates.n_neg_decided >= 20)),
        ("Any positive gain, no support filter", candidates.gain_vs_train_mean_decided > 0),
    ]:
        lines += [f"## {title}", "", "| Subset | Policy | Thr | MinStep | k | Ndec | Pos | Neg | Cov | Acc | Model AUC | Prior AUC | Gain |", "|:--|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|"]
        sub = candidates[filt].sort_values("gain_vs_train_mean_decided", ascending=False).head(25)
        for _, r in sub.iterrows():
            lines.append(f"| {r['subset']} | {r['policy_mode']} | {r['threshold']:.2f} | {int(r['min_step'])} | {int(r['consecutive'])} | {r['n_decided']:.0f} | {r['n_pos_decided']:.0f} | {r['n_neg_decided']:.0f} | {pct(r['coverage_within_subset'])} | {pct(r['decision_acc'])} | {fmt(r['model_auc_decided'])} | {fmt(r['train_mean_success_auc_decided'])} | {fmt(r['gain_vs_train_mean_decided'])} |")
        if sub.empty:
            lines.append("| - | - | - | - | - | - | - | - | - | - | - | - | - |")
        lines.append("")
    lines += ["## Files", "", "- `top3_policy_auc_gain_sweep_by_run.csv`", "- `top3_policy_auc_gain_sweep_compact.csv`", "- `top3_policy_auc_gain_candidates_sorted.csv`"]
    (out / "top3_policy_auc_gain_sweep_report.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    print(out / "top3_policy_auc_gain_sweep_report.md")

if __name__ == "__main__":
    main()
