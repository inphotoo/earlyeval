from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_PREDICTOR = "Robust_LightGBM_Dense_AF_Gold"
DEFAULT_THRESHOLDS = (0.75, 0.80, 0.85, 0.90, 0.95, 0.97)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build fixed-threshold and within-model ranking summaries for TerminalBench cross-agent folds."
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--dataset-subdir", default="terminalbench_harness_core16")
    parser.add_argument("--predictor", default=DEFAULT_PREDICTOR)
    parser.add_argument("--thresholds", nargs="+", type=float, default=list(DEFAULT_THRESHOLDS))
    parser.add_argument("--score-mode", choices=("calibrated", "raw"), default="calibrated")
    return parser.parse_args()


def _head_column(prefix: str, score_mode: str, predictor: str) -> str:
    if score_mode == "calibrated":
        return f"prob_cal_safe_{prefix}__{predictor}"
    return f"prob_safe_{prefix}__{predictor}"


def _split_unit(unit: str) -> tuple[str, str]:
    clean = unit.removeprefix("tb__")
    if "__agent__" not in clean:
        return clean, ""
    model, agent = clean.split("__agent__", 1)
    return model, agent


def _empty_counts() -> dict[str, int]:
    return {
        "decided_failure": 0,
        "decided_success": 0,
        "undecided": 0,
        "false_negatives": 0,
        "true_negatives": 0,
        "false_positives": 0,
        "true_positives": 0,
        "total_saved_steps": 0,
        "total_steps": 0,
    }


def _summarize(counts: dict[str, int], total: int, resolved: int) -> dict[str, Any]:
    true_pos = int(counts["true_positives"])
    true_neg = int(counts["true_negatives"])
    false_pos = int(counts["false_positives"])
    false_neg = int(counts["false_negatives"])
    decided_success = int(counts["decided_success"])
    decided_failure = int(counts["decided_failure"])
    decided = decided_success + decided_failure
    original_rate = resolved / total if total else 0.0
    adjusted_resolved = resolved - false_neg + false_pos
    adjusted_rate = adjusted_resolved / total if total else 0.0
    total_steps = int(counts["total_steps"])
    return {
        "original_total": int(total),
        "original_resolved": int(resolved),
        "original_resolve_rate": float(original_rate),
        "decided_failure": decided_failure,
        "decided_success": decided_success,
        "undecided": int(counts["undecided"]),
        "false_negatives": false_neg,
        "true_negatives": true_neg,
        "false_positives": false_pos,
        "true_positives": true_pos,
        "n_decided": int(decided),
        "coverage": decided / total if total else float("nan"),
        "decision_accuracy": (true_pos + true_neg) / decided if decided else float("nan"),
        "precision_success": true_pos / decided_success if decided_success else float("nan"),
        "precision_failure": true_neg / decided_failure if decided_failure else float("nan"),
        "adjusted_resolved": int(adjusted_resolved),
        "adjusted_resolve_rate": float(adjusted_rate),
        "resolve_rate_drop": float(original_rate - adjusted_rate),
        "pct_steps_saved": (counts["total_saved_steps"] * 100.0 / total_steps) if total_steps else float("nan"),
        "total_saved_steps": int(counts["total_saved_steps"]),
        "total_steps": total_steps,
    }


def _decide(group: pd.DataFrame, success_col: str, failure_col: str, threshold: float) -> tuple[bool, str, int]:
    ordered = group.sort_values("prefix_step_idx")
    for _, row in ordered.iterrows():
        success_score = float(row[success_col])
        failure_score = float(row[failure_col])
        success_hit = success_score >= threshold
        failure_hit = failure_score >= threshold
        if success_hit and failure_hit:
            decision = "success" if success_score - threshold >= failure_score - threshold else "failure"
        elif success_hit:
            decision = "success"
        elif failure_hit:
            decision = "failure"
        else:
            continue
        return True, decision, int(row["prefix_step_idx"])
    return False, "undecided", -1


def _evaluate_unit(predictions: pd.DataFrame, threshold: float, success_col: str, failure_col: str) -> dict[str, Any]:
    counts = _empty_counts()
    total = 0
    resolved = 0
    for _, group in predictions.groupby("traj_id", sort=False):
        total += 1
        label = int(group["label"].iloc[0])
        resolved += label
        n_steps = int(len(group))
        counts["total_steps"] += n_steps
        decided, decision, decision_step = _decide(group, success_col, failure_col, threshold)
        if not decided:
            counts["undecided"] += 1
            continue
        counts["total_saved_steps"] += max(n_steps - decision_step - 1, 0)
        if decision == "failure":
            counts["decided_failure"] += 1
            if label:
                counts["false_negatives"] += 1
            else:
                counts["true_negatives"] += 1
        else:
            counts["decided_success"] += 1
            if label:
                counts["true_positives"] += 1
            else:
                counts["false_positives"] += 1
    return {**counts, **_summarize(counts, total, resolved)}


def _aggregate(per_unit: pd.DataFrame) -> pd.DataFrame:
    rows = []
    count_cols = [
        "decided_failure",
        "decided_success",
        "undecided",
        "false_negatives",
        "true_negatives",
        "false_positives",
        "true_positives",
        "total_saved_steps",
        "total_steps",
    ]
    for threshold, part in per_unit.groupby("threshold", sort=True):
        counts = {col: int(part[col].sum()) for col in count_cols}
        total = int(part["original_total"].sum())
        resolved = int(part["original_resolved"].sum())
        summary = _summarize(counts, total, resolved)
        rows.append(
            {
                "threshold": float(threshold),
                "n_units": int(part["unit"].nunique()),
                **summary,
                "delta_resolve_pp": -float(summary["resolve_rate_drop"]) * 100.0,
                "coverage_pct": float(summary["coverage"]) * 100.0,
                "decision_accuracy_pct": float(summary["decision_accuracy"]) * 100.0,
                "precision_success_pct": float(summary["precision_success"]) * 100.0,
                "precision_failure_pct": float(summary["precision_failure"]) * 100.0,
                "step_save_pct": float(summary["pct_steps_saved"]),
            }
        )
    ordered_cols = [
        "threshold",
        "n_units",
        "original_total",
        "original_resolved",
        "original_resolve_rate",
        "adjusted_resolved",
        "adjusted_resolve_rate",
        "delta_resolve_pp",
        "coverage_pct",
        "decision_accuracy_pct",
        "precision_success_pct",
        "precision_failure_pct",
        "step_save_pct",
        "total_saved_steps",
        "total_steps",
        "false_negatives",
        "true_negatives",
        "false_positives",
        "true_positives",
    ]
    return pd.DataFrame(rows)[ordered_cols]


def _pair_summary(part: pd.DataFrame) -> dict[str, Any]:
    preserved = 0
    reversed_count = 0
    comparable = 0
    values = part[["original_resolve_rate", "adjusted_resolve_rate"]].to_numpy(dtype=float)
    for i in range(len(values)):
        for j in range(i + 1, len(values)):
            full_order = np.sign(values[i, 0] - values[j, 0])
            early_order = np.sign(values[i, 1] - values[j, 1])
            if full_order == 0 or early_order == 0:
                continue
            comparable += 1
            if full_order == early_order:
                preserved += 1
            else:
                reversed_count += 1
    return {
        "pair_order_preserved": int(preserved),
        "pair_order_reversed": int(reversed_count),
        "pair_preserve_rate": preserved / comparable if comparable else float("nan"),
    }


def _ranking(per_unit_at_threshold: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    summary_rows = []
    for model, part in per_unit_at_threshold.groupby("model", sort=True):
        ranked = part.copy()
        ranked["full_rank"] = ranked["original_resolve_rate"].rank(method="first", ascending=False).astype(int)
        ranked["early_rank"] = ranked["adjusted_resolve_rate"].rank(method="first", ascending=False).astype(int)
        ranked["rank_shift_up_positive"] = ranked["full_rank"] - ranked["early_rank"]
        ranked = ranked.sort_values(["full_rank", "agent"]).copy()
        rows.append(ranked)
        full = ranked["full_rank"].to_numpy(dtype=float)
        early = ranked["early_rank"].to_numpy(dtype=float)
        rho = float(pd.Series(full).corr(pd.Series(early), method="spearman")) if len(ranked) > 1 else float("nan")
        pairs = _pair_summary(ranked)
        summary_rows.append(
            {
                "model": model,
                "n_agents": int(len(ranked)),
                "spearman_rho": rho,
                **pairs,
                "top_agent_same": bool(ranked.loc[ranked["full_rank"].idxmin(), "agent"] == ranked.loc[ranked["early_rank"].idxmin(), "agent"]),
                "exact_all_ranks_same": bool((ranked["full_rank"] == ranked["early_rank"]).all()),
                "max_abs_rank_shift": int(ranked["rank_shift_up_positive"].abs().max()),
                "mean_abs_rank_shift": float(ranked["rank_shift_up_positive"].abs().mean()),
            }
        )
    ranking = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    ranking_cols = [
        "model",
        "agent",
        "unit",
        "original_total",
        "original_resolve_rate",
        "adjusted_resolve_rate",
        "resolve_rate_drop",
        "full_rank",
        "early_rank",
        "rank_shift_up_positive",
        "coverage",
        "decision_accuracy",
        "pct_steps_saved",
        "decided_success",
        "decided_failure",
        "n_decided",
    ]
    return ranking[ranking_cols], pd.DataFrame(summary_rows)


def _read_predictions(unit_dir: Path) -> pd.DataFrame:
    path = unit_dir / "test_predictions_safe_stop.parquet"
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_parquet(path)


def build_summary(run_dir: Path, output_dir: Path, dataset_subdir: str, predictor: str, thresholds: list[float], score_mode: str) -> None:
    success_col = _head_column("success", score_mode, predictor)
    failure_col = _head_column("failure", score_mode, predictor)
    rows = []
    unit_root = run_dir / dataset_subdir
    for unit_dir in sorted(path for path in unit_root.iterdir() if path.is_dir()):
        predictions = _read_predictions(unit_dir)
        missing = [col for col in (success_col, failure_col) if col not in predictions.columns]
        if missing:
            raise KeyError(f"{unit_dir}: missing prediction columns {missing}")
        unit = unit_dir.name
        model, agent = _split_unit(unit)
        for threshold in thresholds:
            summary = _evaluate_unit(predictions, float(threshold), success_col, failure_col)
            rows.append(
                {
                    "unit": unit,
                    "model": model,
                    "agent": agent,
                    "score_mode": score_mode,
                    "prefix_model": predictor,
                    "policy_mode": "dual",
                    "threshold": float(threshold),
                    "success_thr": float(threshold),
                    "failure_thr": float(threshold),
                    "min_step": 0,
                    "consecutive": 1,
                    **summary,
                }
            )
    per_unit = pd.DataFrame(rows)
    global_summary = _aggregate(per_unit)
    output_dir.mkdir(parents=True, exist_ok=True)
    per_unit.to_csv(output_dir / "terminalbench_fixed_symmetric_thresholds_per_unit.csv", index=False)
    global_summary.to_csv(output_dir / "terminalbench_fixed_symmetric_thresholds_global_summary.csv", index=False)
    for threshold in thresholds:
        threshold_key = f"{int(round(float(threshold) * 100)):03d}"
        current = per_unit[per_unit["threshold"] == float(threshold)].copy()
        ranking, ranking_summary = _ranking(current)
        ranking.to_csv(output_dir / f"terminalbench_fixed_{threshold_key}_within_model_agent_ranking.csv", index=False)
        ranking_summary.to_csv(output_dir / f"terminalbench_fixed_{threshold_key}_within_model_agent_ranking_summary.csv", index=False)

    lines = [
        "# TerminalBench Cross-Agent Fixed-Threshold Summaries",
        "",
        f"- run_dir: `{run_dir}`",
        f"- dataset_subdir: `{dataset_subdir}`",
        f"- predictor: `{predictor}`",
        f"- score_mode: `{score_mode}`",
        "- policy: calibrated dual-head, symmetric success/failure thresholds, min_step=0, consecutive=1",
        "",
        "## Global Results",
        "",
        global_summary[
            ["threshold", "coverage_pct", "decision_accuracy_pct", "step_save_pct", "delta_resolve_pp"]
        ].to_markdown(index=False),
        "",
    ]
    (output_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir or (args.run_dir / "summary" / "fixed_thresholds_main_aligned")
    build_summary(
        run_dir=args.run_dir,
        output_dir=output_dir,
        dataset_subdir=args.dataset_subdir,
        predictor=args.predictor,
        thresholds=[float(value) for value in args.thresholds],
        score_mode=args.score_mode,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
