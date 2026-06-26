from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_PREDICTOR = "Robust_LightGBM_Dense_AF_Gold"
DEFAULT_THRESHOLDS = (0.75, 0.80, 0.85, 0.90, 0.95, 0.97)
TOKEN_METHOD = "model_tokenizer_component_sum_context_call_plus_generated_output"

TOKEN_PREFIX_COLUMNS = [
    "traj_id",
    "prefix_step_idx",
    "baseline_input_tokens_est",
    "baseline_external_input_tokens_est",
    "baseline_output_tokens_est",
    "baseline_transcript_total_tokens_est",
    "external_input_tokens_cum",
    "generated_output_tokens_cum",
    "transcript_total_tokens_cum",
    "future_context_call_input_tokens_saved_if_stop_est",
    "token_count_method",
    "tokenizer_family",
    "tokenizer_backend",
    "tokenizer_name",
]

TOKEN_COUNT_COLUMNS = [
    "baseline_input_tokens_est",
    "baseline_external_input_tokens_est",
    "baseline_output_tokens_est",
    "baseline_transcript_total_tokens_est",
    "baseline_total_api_tokens_est",
    "saved_input_tokens_est",
    "saved_external_input_tokens_est",
    "saved_output_tokens_est",
    "saved_transcript_total_tokens_est",
    "saved_total_api_tokens_est",
]

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
    parser.add_argument(
        "--token-prefix-cache",
        type=Path,
        default=None,
        help="Optional per-prefix tokenizer-count parquet produced by the paper token accounting pipeline.",
    )
    parser.add_argument(
        "--token-id-mode",
        choices=("exact", "strip-terminalbench-unit-suffix"),
        default="strip-terminalbench-unit-suffix",
        help="How prediction trajectory IDs map to token-cache trajectory IDs.",
    )
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


def _empty_counts() -> dict[str, float]:
    counts: dict[str, float] = {
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
    counts.update({col: 0.0 for col in TOKEN_COUNT_COLUMNS})
    return counts


def _pct(num: float, den: float) -> float:
    return float(num) * 100.0 / float(den) if den else float("nan")


def _summarize(counts: dict[str, float], total: int, resolved: int) -> dict[str, Any]:
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
    out = {
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
    for col in TOKEN_COUNT_COLUMNS:
        out[col] = float(counts.get(col, 0.0))
    out["input_token_save_pct_est"] = _pct(out["saved_input_tokens_est"], out["baseline_input_tokens_est"])
    out["output_token_save_pct_est"] = _pct(out["saved_output_tokens_est"], out["baseline_output_tokens_est"])
    out["total_token_save_pct_est"] = _pct(out["saved_total_api_tokens_est"], out["baseline_total_api_tokens_est"])
    out["external_input_token_save_pct_est"] = _pct(
        out["saved_external_input_tokens_est"], out["baseline_external_input_tokens_est"]
    )
    out["transcript_total_token_save_pct_est"] = _pct(
        out["saved_transcript_total_tokens_est"], out["baseline_transcript_total_tokens_est"]
    )
    return out


def _threshold_hit(score: float, threshold: float) -> bool:
    return math.isfinite(threshold) and score >= threshold


def _decide(
    group: pd.DataFrame,
    success_col: str,
    failure_col: str,
    success_threshold: float,
    failure_threshold: float,
    min_step: int = 0,
    consecutive: int = 1,
) -> tuple[bool, str, int, pd.Series | None]:
    ordered = group.sort_values("prefix_step_idx")
    success_run = 0
    failure_run = 0
    for _, row in ordered.iterrows():
        step = int(row["prefix_step_idx"])
        if step < min_step:
            continue
        success_score = float(row[success_col])
        failure_score = float(row[failure_col])
        success_hit = _threshold_hit(success_score, success_threshold)
        failure_hit = _threshold_hit(failure_score, failure_threshold)
        success_run = success_run + 1 if success_hit else 0
        failure_run = failure_run + 1 if failure_hit else 0
        success_ready = success_run >= consecutive
        failure_ready = failure_run >= consecutive
        if success_ready and failure_ready:
            success_margin = success_score - success_threshold
            failure_margin = failure_score - failure_threshold
            decision = "success" if success_margin >= failure_margin else "failure"
        elif success_ready:
            decision = "success"
        elif failure_ready:
            decision = "failure"
        else:
            continue
        return True, decision, step, row
    return False, "undecided", -1, None


def _add_token_counts(counts: dict[str, float], ordered: pd.DataFrame, decision_row: pd.Series | None, decided: bool) -> None:
    required = set(TOKEN_PREFIX_COLUMNS[2:10])
    if not required.issubset(ordered.columns):
        return
    last = ordered.iloc[-1]
    baseline_input = float(last["baseline_input_tokens_est"])
    baseline_external_input = float(last["baseline_external_input_tokens_est"])
    baseline_output = float(last["baseline_output_tokens_est"])
    baseline_transcript = float(last["baseline_transcript_total_tokens_est"])
    counts["baseline_input_tokens_est"] += baseline_input
    counts["baseline_external_input_tokens_est"] += baseline_external_input
    counts["baseline_output_tokens_est"] += baseline_output
    counts["baseline_transcript_total_tokens_est"] += baseline_transcript
    counts["baseline_total_api_tokens_est"] += baseline_input + baseline_output
    if not decided or decision_row is None:
        return
    saved_input = float(decision_row.get("future_context_call_input_tokens_saved_if_stop_est", 0.0) or 0.0)
    saved_external_input = baseline_external_input - float(
        decision_row.get("external_input_tokens_cum", baseline_external_input) or 0.0
    )
    saved_output = baseline_output - float(decision_row.get("generated_output_tokens_cum", baseline_output) or 0.0)
    saved_transcript = baseline_transcript - float(
        decision_row.get("transcript_total_tokens_cum", baseline_transcript) or 0.0
    )
    saved_input = max(saved_input, 0.0)
    saved_external_input = max(saved_external_input, 0.0)
    saved_output = max(saved_output, 0.0)
    saved_transcript = max(saved_transcript, 0.0)
    counts["saved_input_tokens_est"] += saved_input
    counts["saved_external_input_tokens_est"] += saved_external_input
    counts["saved_output_tokens_est"] += saved_output
    counts["saved_transcript_total_tokens_est"] += saved_transcript
    counts["saved_total_api_tokens_est"] += saved_input + saved_output


def _evaluate_unit(
    predictions: pd.DataFrame,
    success_threshold: float,
    failure_threshold: float,
    success_col: str,
    failure_col: str,
    min_step: int = 0,
    consecutive: int = 1,
) -> dict[str, Any]:
    counts = _empty_counts()
    total = 0
    resolved = 0
    for _, group in predictions.groupby("traj_id", sort=False):
        ordered = group.sort_values("prefix_step_idx")
        total += 1
        label = int(ordered["label"].iloc[0])
        resolved += label
        n_steps = int(len(ordered))
        counts["total_steps"] += n_steps
        decided, decision, decision_step, decision_row = _decide(
            ordered,
            success_col,
            failure_col,
            success_threshold,
            failure_threshold,
            min_step=min_step,
            consecutive=consecutive,
        )
        _add_token_counts(counts, ordered, decision_row, decided)
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
        *TOKEN_COUNT_COLUMNS,
    ]
    for threshold, part in per_unit.groupby("threshold", sort=True):
        counts = {col: float(part[col].sum()) for col in count_cols}
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
                "input_token_save_pct_est": float(summary["input_token_save_pct_est"]),
                "output_token_save_pct_est": float(summary["output_token_save_pct_est"]),
                "total_token_save_pct_est": float(summary["total_token_save_pct_est"]),
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
        "baseline_input_tokens_est",
        "saved_input_tokens_est",
        "input_token_save_pct_est",
        "baseline_output_tokens_est",
        "saved_output_tokens_est",
        "output_token_save_pct_est",
        "baseline_total_api_tokens_est",
        "saved_total_api_tokens_est",
        "total_token_save_pct_est",
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
        "input_token_save_pct_est",
        "output_token_save_pct_est",
        "total_token_save_pct_est",
        "decided_success",
        "decided_failure",
        "n_decided",
    ]
    ranking_cols = [col for col in ranking_cols if col in ranking.columns]
    return ranking[ranking_cols], pd.DataFrame(summary_rows)


def _aggregate_all(per_unit: pd.DataFrame) -> pd.DataFrame:
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
        *TOKEN_COUNT_COLUMNS,
    ]
    counts = {col: float(per_unit[col].sum()) for col in count_cols}
    total = int(per_unit["original_total"].sum())
    resolved = int(per_unit["original_resolved"].sum())
    summary = _summarize(counts, total, resolved)
    return pd.DataFrame(
        [
            {
                "n_units": int(per_unit["unit"].nunique()),
                **summary,
                "delta_resolve_pp": -float(summary["resolve_rate_drop"]) * 100.0,
                "coverage_pct": float(summary["coverage"]) * 100.0,
                "decision_accuracy_pct": float(summary["decision_accuracy"]) * 100.0,
                "precision_success_pct": float(summary["precision_success"]) * 100.0,
                "precision_failure_pct": float(summary["precision_failure"]) * 100.0,
                "step_save_pct": float(summary["pct_steps_saved"]),
            }
        ]
    )


def _agent_name_aggregate(per_unit: pd.DataFrame) -> pd.DataFrame:
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
        *TOKEN_COUNT_COLUMNS,
    ]
    rows: list[dict[str, Any]] = []
    for (threshold, agent), part in per_unit.groupby(["threshold", "agent"], sort=True):
        counts = {col: float(part[col].sum()) for col in count_cols}
        total = int(part["original_total"].sum())
        resolved = int(part["original_resolved"].sum())
        summary = _summarize(counts, total, resolved)
        rows.append(
            {
                "threshold": float(threshold),
                "agent": str(agent),
                "n_units": int(part["unit"].nunique()),
                "total_trajectories": total,
                "full_weighted_resolve_rate": float(summary["original_resolve_rate"]) * 100.0,
                "early_weighted_resolve_rate": float(summary["adjusted_resolve_rate"]) * 100.0,
                "delta_resolve_pp": -float(summary["resolve_rate_drop"]) * 100.0,
                "coverage_pct": float(summary["coverage"]) * 100.0,
                "step_save_pct": float(summary["pct_steps_saved"]),
                "input_token_save_pct_est": float(summary["input_token_save_pct_est"]),
                "output_token_save_pct_est": float(summary["output_token_save_pct_est"]),
                "total_token_save_pct_est": float(summary["total_token_save_pct_est"]),
                "baseline_input_tokens_est": float(summary["baseline_input_tokens_est"]),
                "saved_input_tokens_est": float(summary["saved_input_tokens_est"]),
                "baseline_output_tokens_est": float(summary["baseline_output_tokens_est"]),
                "saved_output_tokens_est": float(summary["saved_output_tokens_est"]),
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["full_rank"] = out.groupby("threshold")["full_weighted_resolve_rate"].rank(method="first", ascending=False).astype(int)
    out["early_rank"] = out.groupby("threshold")["early_weighted_resolve_rate"].rank(method="first", ascending=False).astype(int)
    out["rank_shift_up_positive"] = out["full_rank"] - out["early_rank"]
    return out.sort_values(["threshold", "full_rank", "agent"]).reset_index(drop=True)


def _read_predictions(unit_dir: Path) -> pd.DataFrame:
    path = unit_dir / "test_predictions_safe_stop.parquet"
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_parquet(path)


def _token_traj_id(prediction_traj_id: str, mode: str) -> str:
    value = str(prediction_traj_id)
    if mode == "strip-terminalbench-unit-suffix" and "__tb__" in value:
        return value.split("__tb__", 1)[0]
    return value


def _load_token_prefix_cache(path: Path | None, wanted_token_ids: set[str]) -> pd.DataFrame | None:
    if path is None:
        return None
    if not path.exists():
        raise FileNotFoundError(path)
    token_rows = pd.read_parquet(path, columns=TOKEN_PREFIX_COLUMNS)
    token_rows["token_traj_id"] = token_rows["traj_id"].astype(str)
    token_rows = token_rows[token_rows["token_traj_id"].isin(wanted_token_ids)].copy()
    missing = wanted_token_ids - set(token_rows["token_traj_id"].astype(str).unique())
    if missing:
        sample = ", ".join(sorted(missing)[:5])
        raise ValueError(f"Token prefix cache is missing {len(missing)} trajectories, e.g. {sample}")
    token_rows["prefix_step_idx"] = pd.to_numeric(token_rows["prefix_step_idx"], errors="raise").astype("int64")
    keep_cols = ["token_traj_id", "prefix_step_idx", *TOKEN_PREFIX_COLUMNS[2:]]
    return token_rows[keep_cols].drop_duplicates(["token_traj_id", "prefix_step_idx"], keep="last")


def _attach_token_rows(predictions: pd.DataFrame, token_rows: pd.DataFrame | None, token_id_mode: str) -> pd.DataFrame:
    if token_rows is None:
        return predictions
    out = predictions.copy()
    out["token_traj_id"] = out["traj_id"].map(lambda value: _token_traj_id(value, token_id_mode))
    out["prefix_step_idx"] = pd.to_numeric(out["prefix_step_idx"], errors="raise").astype("int64")
    out = out.merge(token_rows, on=["token_traj_id", "prefix_step_idx"], how="left")
    missing = out[out["baseline_input_tokens_est"].isna()][["traj_id", "prefix_step_idx"]].drop_duplicates()
    if not missing.empty:
        sample = missing.head(5).to_dict(orient="records")
        raise ValueError(f"Missing token rows for {len(missing)} prediction prefixes, e.g. {sample}")
    return out


def build_summary(
    run_dir: Path,
    output_dir: Path,
    dataset_subdir: str,
    predictor: str,
    thresholds: list[float],
    score_mode: str,
    token_prefix_cache: Path | None = None,
    token_id_mode: str = "strip-terminalbench-unit-suffix",
) -> None:
    success_col = _head_column("success", score_mode, predictor)
    failure_col = _head_column("failure", score_mode, predictor)
    unit_root = run_dir / dataset_subdir
    unit_payloads: list[tuple[Path, str, str, str, pd.DataFrame]] = []
    wanted_token_ids: set[str] = set()
    for unit_dir in sorted(path for path in unit_root.iterdir() if path.is_dir()):
        predictions = _read_predictions(unit_dir)
        missing = [col for col in (success_col, failure_col) if col not in predictions.columns]
        if missing:
            raise KeyError(f"{unit_dir}: missing prediction columns {missing}")
        unit = unit_dir.name
        model, agent = _split_unit(unit)
        wanted_token_ids.update(_token_traj_id(value, token_id_mode) for value in predictions["traj_id"].astype(str).unique())
        unit_payloads.append((unit_dir, unit, model, agent, predictions))

    token_rows = _load_token_prefix_cache(token_prefix_cache, wanted_token_ids)
    rows = []
    selected_rows = []
    for unit_dir, unit, model, agent, predictions_raw in unit_payloads:
        predictions = _attach_token_rows(predictions_raw, token_rows, token_id_mode)
        for threshold in thresholds:
            summary = _evaluate_unit(predictions, float(threshold), float(threshold), success_col, failure_col)
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
        selected_path = unit_dir / "safe_stop_test_selected.csv"
        if selected_path.exists():
            selected = pd.read_csv(selected_path)
            if not selected.empty:
                selected_row = selected.iloc[0]
                min_step = int(selected_row.get("min_step", 0))
                consecutive = int(selected_row.get("consecutive", 1))
                summary = _evaluate_unit(
                    predictions,
                    float(selected_row["success_thr"]),
                    float(selected_row["failure_thr"]),
                    success_col,
                    failure_col,
                    min_step=min_step,
                    consecutive=consecutive,
                )
                selected_rows.append(
                    {
                        "unit": unit,
                        "model": model,
                        "agent": agent,
                        "score_mode": score_mode,
                        "prefix_model": predictor,
                        "policy_mode": str(selected_row.get("policy_mode", "dual")),
                        "success_thr": float(selected_row["success_thr"]),
                        "failure_thr": float(selected_row["failure_thr"]),
                        "min_step": min_step,
                        "consecutive": consecutive,
                        **summary,
                        "policy_id": f"{dataset_subdir}/{unit}__{score_mode}__{predictor}",
                    }
                )
    per_unit = pd.DataFrame(rows)
    global_summary = _aggregate(per_unit)
    agent_aggregate = _agent_name_aggregate(per_unit)
    output_dir.mkdir(parents=True, exist_ok=True)
    per_unit.to_csv(output_dir / "terminalbench_fixed_threshold_per_unit.csv", index=False)
    global_summary.to_csv(output_dir / "terminalbench_fixed_threshold_global_summary.csv", index=False)
    agent_aggregate.to_csv(output_dir / "terminalbench_fixed_threshold_agent_name_aggregate.csv", index=False)
    for threshold in thresholds:
        threshold_key = f"{int(round(float(threshold) * 100)):03d}"
        current = per_unit[per_unit["threshold"] == float(threshold)].copy()
        ranking, ranking_summary = _ranking(current)
        ranking.to_csv(output_dir / f"terminalbench_fixed{threshold_key}_within_model_agent_ranking.csv", index=False)
        ranking_summary.to_csv(output_dir / f"terminalbench_fixed{threshold_key}_ranking_summary.csv", index=False)

    selected_per_unit = pd.DataFrame(selected_rows)
    selected_global = pd.DataFrame()
    if not selected_per_unit.empty:
        selected_global = _aggregate_all(selected_per_unit)
        selected_per_unit.to_csv(output_dir / "terminalbench_selected_policy_test_per_unit.csv", index=False)
        selected_global.to_csv(output_dir / "terminalbench_selected_policy_global_summary.csv", index=False)
        selected_ranking, selected_ranking_summary = _ranking(selected_per_unit)
        selected_ranking.to_csv(output_dir / "terminalbench_selected_policy_within_model_agent_ranking.csv", index=False)
        selected_ranking_summary.to_csv(output_dir / "terminalbench_selected_policy_ranking_summary.csv", index=False)

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
            [
                "threshold",
                "coverage_pct",
                "decision_accuracy_pct",
                "step_save_pct",
                "input_token_save_pct_est",
                "output_token_save_pct_est",
                "total_token_save_pct_est",
                "delta_resolve_pp",
            ]
        ].to_markdown(index=False),
        "",
        f"- token method: `{TOKEN_METHOD}`" if token_rows is not None else "- token method: not computed",
    ]
    if not selected_global.empty:
        lines.extend(
            [
                "",
                "## Selected-Policy Aggregate",
                "",
                selected_global[
                    [
                        "coverage_pct",
                        "decision_accuracy_pct",
                        "step_save_pct",
                        "input_token_save_pct_est",
                        "output_token_save_pct_est",
                        "total_token_save_pct_est",
                        "delta_resolve_pp",
                    ]
                ].to_markdown(index=False),
                "",
            ]
        )
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
        token_prefix_cache=args.token_prefix_cache,
        token_id_mode=args.token_id_mode,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
