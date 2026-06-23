#!/usr/bin/env python3
from __future__ import annotations

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
OUT_DIR = BASE_PRIOR_DIR / "hard_subset_policy_auc_rescue_all_splits"

SPLITS = ["top3", "mid3", "bottom3"]
PREFIX_MODELS = {"I": "I_LightGBM_Dense_AF", "J": "J_LightGBM_Dense_AF_Thought"}
STRATEGIES = ["strong_reg", "no_model_id"]
SCORE_MODES = ["raw", "prefix_calibrated"]
POLICY_MODES = ["symmetric", "success_only", "failure_only"]
THRESHOLDS = [0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
MIN_STEPS = [0, 5, 10, 15, 20]
CONSECUTIVE = [1, 2, 3]
HARD_BANDS = [
    ("amb_0.2_0.8", 0.20, 0.80),
    ("amb_0.25_0.75", 0.25, 0.75),
    ("amb_0.3_0.7", 0.30, 0.70),
    ("amb_0.35_0.65", 0.35, 0.65),
    ("amb_0.4_0.6", 0.40, 0.60),
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


def make_records(df: pd.DataFrame, prob_col: str) -> list[dict]:
    records = []
    needed = ["traj_id", "orig_model_id", "label", "prefix_step_idx", prob_col]
    for traj_id, group in df[needed].groupby("traj_id", sort=False):
        group = group.sort_values("prefix_step_idx")
        probs = group[prob_col].to_numpy(dtype=np.float64)
        records.append(
            {
                "traj_id": str(traj_id),
                "agent_model": str(group["orig_model_id"].iloc[0]),
                "label": int(group["label"].iloc[0]),
                "n_steps": int(len(group)),
                "steps": group["prefix_step_idx"].to_numpy(dtype=np.int32),
                "probs": probs,
                "p0": float(probs[0]),
            }
        )
    return records


def decide_records(
    records: list[dict],
    prior_map: dict[str, float],
    *,
    low: float,
    high: float,
    policy_mode: str,
    threshold: float,
    min_step: int,
    consecutive: int,
) -> tuple[pd.DataFrame, int, int]:
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
        train_prior = prior_map.get(rec["traj_id"])
        if train_prior is None or not (low <= train_prior <= high):
            continue
        n_all_subset += 1
        total_steps_all += int(rec["n_steps"])
        decided, decision, decision_step, decision_prob = _decide(
            rec["steps"],
            rec["probs"],
            p0=rec["p0"],
            success_thr=success_thr,
            failure_thr=failure_thr,
            min_step=min_step,
            consecutive=consecutive,
            delta_up=0.0,
            delta_down=0.0,
        )
        if not decided:
            continue
        rows.append(
            {
                "traj_id": rec["traj_id"],
                "agent_model": rec["agent_model"],
                "label": rec["label"],
                "decision": decision,
                "decision_prob": decision_prob,
                "decision_step": decision_step,
                "saved_steps": max(rec["n_steps"] - decision_step - 1, 0),
                "train_prior": train_prior,
            }
        )
    return pd.DataFrame(rows), n_all_subset, total_steps_all


def summarize_decisions(decisions: pd.DataFrame, n_all_subset: int, total_steps_all: int) -> dict:
    n_decided = int(len(decisions))
    if n_decided:
        labels = decisions["label"].astype(int)
        pred = decisions["decision"].map({"success": 1, "failure": 0}).astype(int)
        model_auc = auc_rank(labels, decisions["decision_prob"])
        prior_auc = auc_rank(labels, decisions["train_prior"])
        n_pos = int(labels.sum())
        n_neg = int(n_decided - n_pos)
        saved_steps = float(decisions["saved_steps"].sum())
    else:
        model_auc = prior_auc = float("nan")
        n_pos = n_neg = 0
        saved_steps = 0.0
        labels = pred = pd.Series(dtype=int)
    return {
        "n_all_subset": n_all_subset,
        "n_decided": n_decided,
        "n_pos_decided": n_pos,
        "n_neg_decided": n_neg,
        "coverage": n_decided / n_all_subset if n_all_subset else float("nan"),
        "decision_acc": float((pred == labels).mean()) if n_decided else float("nan"),
        "save_pct": saved_steps / total_steps_all * 100.0 if total_steps_all else float("nan"),
        "model_auc": model_auc,
        "prior_auc": prior_auc,
        "gain": model_auc - prior_auc,
    }


def fmt(value: float) -> str:
    return "-" if pd.isna(value) else f"{value:.3f}"


def pct(value: float) -> str:
    return "-" if pd.isna(value) else f"{value * 100:.1f}%"


def write_table(lines: list[str], title: str, frame: pd.DataFrame, limit: int = 30) -> None:
    lines += [
        f"## {title}",
        "",
        "| Split | Variant | Strategy | Score | Band | Policy | Thr | MinStep | k | N | Pos | Neg | Cov | Acc | Save | Model AUC | Prior AUC | Gain |",
        "|:--|:--|:--|:--|:--|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|",
    ]
    if frame.empty:
        lines.append("| - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - |")
        lines.append("")
        return
    for _, row in frame.head(limit).iterrows():
        lines.append(
            f"| {row['split']} | {row['variant']} | {row['strategy']} | {row['score_mode']} | {row['band']} | {row['policy_mode']} | "
            f"{row['threshold']:.2f} | {int(row['min_step'])} | {int(row['consecutive'])} | "
            f"{int(row['n_decided'])} | {int(row['n_pos_decided'])} | {int(row['n_neg_decided'])} | "
            f"{pct(row['coverage'])} | {pct(row['decision_acc'])} | {row['save_pct']:.1f}% | "
            f"{fmt(row['model_auc'])} | {fmt(row['prior_auc'])} | {fmt(row['gain'])} |"
        )
    lines.append("")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    prior_all = pd.read_csv(BASE_PRIOR_DIR / "other_model_prior_scores_by_traj.csv")
    rows = []
    agent_rows = []
    for split in SPLITS:
        split_prior = prior_all[prior_all["split"].eq(split)].drop_duplicates("traj_id")
        prior_map = split_prior.set_index("traj_id")["train_other_mean_success"].astype(float).to_dict()
        valid_ids = set(prior_map)
        for variant, prefix_model in PREFIX_MODELS.items():
            for strategy in STRATEGIES:
                pred_path = run_dir(split, variant, strategy) / "test_predictions_shadow_valid_retrain.parquet"
                if not pred_path.exists():
                    continue
                df = pd.read_parquet(pred_path)
                df = df[df["traj_id"].isin(valid_ids)].copy()
                for score_mode in SCORE_MODES:
                    prob_col = _prob_col(prefix_model, score_mode)
                    if prob_col not in df.columns:
                        continue
                    records = make_records(df, prob_col)
                    for band_name, low, high in HARD_BANDS:
                        for policy_mode in POLICY_MODES:
                            for threshold in THRESHOLDS:
                                for min_step in MIN_STEPS:
                                    for consecutive in CONSECUTIVE:
                                        decisions, n_all_subset, total_steps_all = decide_records(
                                            records,
                                            prior_map,
                                            low=low,
                                            high=high,
                                            policy_mode=policy_mode,
                                            threshold=threshold,
                                            min_step=min_step,
                                            consecutive=consecutive,
                                        )
                                        base = {
                                            "split": split,
                                            "variant": variant,
                                            "strategy": strategy,
                                            "score_mode": score_mode,
                                            "band": band_name,
                                            "low": low,
                                            "high": high,
                                            "policy_mode": policy_mode,
                                            "threshold": threshold,
                                            "failure_thr": 1.0 - threshold,
                                            "min_step": min_step,
                                            "consecutive": consecutive,
                                        }
                                        rows.append({**base, **summarize_decisions(decisions, n_all_subset, total_steps_all)})
                                        if decisions.empty:
                                            continue
                                        for agent_model, group in decisions.groupby("agent_model"):
                                            agent_rows.append(
                                                {
                                                    **base,
                                                    "agent_model": agent_model,
                                                    **summarize_decisions(group, n_all_subset, total_steps_all),
                                                }
                                            )

    detail = pd.DataFrame(rows)
    agents = pd.DataFrame(agent_rows)
    detail.to_csv(OUT_DIR / "hard_subset_policy_auc_rescue_by_run.csv", index=False)
    agents.to_csv(OUT_DIR / "hard_subset_policy_auc_rescue_by_agent.csv", index=False)

    detail["support_10_10"] = (detail["n_pos_decided"] >= 10) & (detail["n_neg_decided"] >= 10)
    detail["practical_symmetric"] = (
        detail["support_10_10"]
        & detail["policy_mode"].eq("symmetric")
        & detail["coverage"].ge(0.10)
        & detail["decision_acc"].ge(0.70)
        & detail["gain"].gt(0)
        & detail["model_auc"].notna()
    )
    detail.sort_values(["gain", "coverage", "n_decided"], ascending=[False, False, False]).to_csv(
        OUT_DIR / "hard_subset_policy_auc_candidates_sorted.csv", index=False
    )
    practical = detail[detail["practical_symmetric"]].sort_values(["gain", "coverage"], ascending=[False, False])
    practical.to_csv(OUT_DIR / "hard_subset_policy_auc_practical_symmetric_candidates.csv", index=False)

    best_by_split_band = (
        practical.sort_values(["gain", "coverage"], ascending=[False, False])
        .groupby(["split", "band"], as_index=False)
        .head(5)
    )
    best_by_split_band.to_csv(OUT_DIR / "hard_subset_policy_auc_best_practical_by_split_band.csv", index=False)

    stable_rows = []
    if not agents.empty:
        key_cols = [
            "split",
            "variant",
            "strategy",
            "score_mode",
            "band",
            "policy_mode",
            "threshold",
            "min_step",
            "consecutive",
        ]
        per_agent = agents[
            (agents["n_pos_decided"] >= 3)
            & (agents["n_neg_decided"] >= 3)
            & agents["model_auc"].notna()
        ].copy()
        expected_agents = prior_all.groupby("split")["orig_model_id"].nunique().to_dict()
        for key, group in per_agent.groupby(key_cols, dropna=False):
            split = key[0]
            if len(group) < expected_agents.get(split, 0):
                continue
            stable_rows.append(
                dict(
                    zip(key_cols, key),
                    min_agent_gain=float(group["gain"].min()),
                    mean_agent_gain=float(group["gain"].mean()),
                    min_agent_auc=float(group["model_auc"].min()),
                    mean_agent_auc=float(group["model_auc"].mean()),
                    total_n=int(group["n_decided"].sum()),
                    min_agent_pos=int(group["n_pos_decided"].min()),
                    min_agent_neg=int(group["n_neg_decided"].min()),
                    min_agent_acc=float(group["decision_acc"].min()),
                    mean_agent_acc=float(group["decision_acc"].mean()),
                )
            )
    stable = pd.DataFrame(stable_rows)
    if not stable.empty:
        stable = stable.sort_values(["min_agent_gain", "mean_agent_gain"], ascending=[False, False])
        stable.to_csv(OUT_DIR / "hard_subset_policy_auc_stable_all_agents.csv", index=False)

    lines = [
        "# Hard-Subset Policy AUC Rescue Across Splits",
        "",
        "同一套 hard/ambiguous prior band，在 `top3 / mid3 / bottom3` 上扫 `threshold / min_step / consecutive k / policy_mode`。",
        "",
        "Baseline 是 `train_other_mean_success` 的 AUC；这里看 threshold-decided subset 上 model score 的 AUC gain。",
        "",
        "筛选口径：`practical_symmetric = symmetric + Pos>=10 + Neg>=10 + Coverage>=10% + Acc>=70% + Gain>0`。",
        "",
    ]

    write_table(lines, "Top practical symmetric candidates", practical, limit=40)
    for split in SPLITS:
        write_table(lines, f"{split}: best practical candidates by band", best_by_split_band[best_by_split_band["split"].eq(split)], limit=30)

    lines += [
        "## Practical candidate counts",
        "",
        "| Split | Count | Best Gain | Best AUC | Best Prior AUC | Best Coverage |",
        "|:--|--:|--:|--:|--:|--:|",
    ]
    for split in SPLITS:
        sub = practical[practical["split"].eq(split)]
        if sub.empty:
            lines.append(f"| {split} | 0 | - | - | - | - |")
        else:
            best = sub.iloc[0]
            lines.append(
                f"| {split} | {len(sub)} | {fmt(best['gain'])} | {fmt(best['model_auc'])} | {fmt(best['prior_auc'])} | {pct(best['coverage'])} |"
            )
    lines.append("")

    if stable.empty:
        lines += ["## Stable across all agents in split", "", "没有满足每个 agent 都有足够正负例的稳定表。", ""]
    else:
        stable_pos = stable[stable["min_agent_gain"].gt(0)].head(40)
        lines += [
            "## Stable across all agents in split",
            "",
            "| Split | Variant | Strategy | Score | Band | Policy | Thr | MinStep | k | Total N | Min Pos | Min Neg | Min Acc | Min AUC | Min Gain | Mean Gain |",
            "|:--|:--|:--|:--|:--|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|",
        ]
        if stable_pos.empty:
            lines.append("| - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - |")
        else:
            for _, row in stable_pos.iterrows():
                lines.append(
                    f"| {row['split']} | {row['variant']} | {row['strategy']} | {row['score_mode']} | {row['band']} | {row['policy_mode']} | "
                    f"{row['threshold']:.2f} | {int(row['min_step'])} | {int(row['consecutive'])} | {int(row['total_n'])} | "
                    f"{int(row['min_agent_pos'])} | {int(row['min_agent_neg'])} | {pct(row['min_agent_acc'])} | "
                    f"{fmt(row['min_agent_auc'])} | {fmt(row['min_agent_gain'])} | {fmt(row['mean_agent_gain'])} |"
                )
        lines.append("")

    lines += [
        "## Files",
        "",
        "- `hard_subset_policy_auc_rescue_by_run.csv`",
        "- `hard_subset_policy_auc_rescue_by_agent.csv`",
        "- `hard_subset_policy_auc_candidates_sorted.csv`",
        "- `hard_subset_policy_auc_practical_symmetric_candidates.csv`",
        "- `hard_subset_policy_auc_best_practical_by_split_band.csv`",
        "- `hard_subset_policy_auc_stable_all_agents.csv`",
    ]
    (OUT_DIR / "hard_subset_policy_auc_rescue_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(OUT_DIR / "hard_subset_policy_auc_rescue_report.md")


if __name__ == "__main__":
    main()
