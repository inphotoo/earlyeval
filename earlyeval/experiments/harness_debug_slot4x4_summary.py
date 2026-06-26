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

METADATA_COLUMNS = [
    "traj_id",
    "prefix_step_idx",
    "harness_pair_id",
    "harness_model_group",
    "harness_agent_slot",
    "harness_agent_raw",
    "harness_original_model_id",
]

SETTING_SPECS = {
    "leave_model": {
        "run_subdir": "terminalbench_slot4x4_leave_model",
        "heldout_col": "harness_model_group",
        "rank_item_col": "harness_agent_slot",
        "rank_item_name": "agent_slot",
    },
    "leave_agent": {
        "run_subdir": "terminalbench_slot4x4_leave_agent",
        "heldout_col": "harness_agent_slot",
        "rank_item_col": "harness_model_group",
        "rank_item_name": "model_group",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build compact summaries for the TerminalBench 4x4 harness-debug exclusion experiment."
    )
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset-subdir", default="terminalbench_slot4x4")
    parser.add_argument("--predictor", default=DEFAULT_PREDICTOR)
    parser.add_argument("--thresholds", nargs="+", type=float, default=list(DEFAULT_THRESHOLDS))
    parser.add_argument("--score-mode", choices=("calibrated", "raw"), default="calibrated")
    parser.add_argument("--prefix-table", type=Path, required=True)
    parser.add_argument("--token-prefix-cache", type=Path, default=None)
    parser.add_argument("--allow-incomplete", action="store_true")
    return parser.parse_args()


def _head_column(prefix: str, score_mode: str, predictor: str) -> str:
    if score_mode == "calibrated":
        return f"prob_cal_safe_{prefix}__{predictor}"
    return f"prob_safe_{prefix}__{predictor}"


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


def _ratio_pct(num: float, den: float) -> float:
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
    out["input_token_save_pct_est"] = _ratio_pct(out["saved_input_tokens_est"], out["baseline_input_tokens_est"])
    out["output_token_save_pct_est"] = _ratio_pct(out["saved_output_tokens_est"], out["baseline_output_tokens_est"])
    out["total_token_save_pct_est"] = _ratio_pct(out["saved_total_api_tokens_est"], out["baseline_total_api_tokens_est"])
    out["external_input_token_save_pct_est"] = _ratio_pct(
        out["saved_external_input_tokens_est"], out["baseline_external_input_tokens_est"]
    )
    out["transcript_total_token_save_pct_est"] = _ratio_pct(
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
    counts["saved_input_tokens_est"] += max(saved_input, 0.0)
    counts["saved_external_input_tokens_est"] += max(saved_external_input, 0.0)
    counts["saved_output_tokens_est"] += max(saved_output, 0.0)
    counts["saved_transcript_total_tokens_est"] += max(saved_transcript, 0.0)
    counts["saved_total_api_tokens_est"] += max(saved_input, 0.0) + max(saved_output, 0.0)


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
        delta_resolve_pp = -float(summary["resolve_rate_drop"]) * 100.0
        mean_abs_delta = float("nan")
        weighted_mean_abs_delta = float("nan")
        if "delta_resolve_pp" in part.columns:
            abs_delta = part["delta_resolve_pp"].astype(float).abs()
            mean_abs_delta = float(abs_delta.mean())
            weights = part["original_total"].astype(float)
            weighted_mean_abs_delta = float(np.average(abs_delta, weights=weights)) if float(weights.sum()) else float("nan")
        rows.append(
            {
                "threshold": float(threshold),
                "n_units": int(part["unit"].nunique()),
                **summary,
                "delta_resolve_pp": delta_resolve_pp,
                "abs_delta_resolve_pp": abs(delta_resolve_pp),
                "mean_abs_delta_resolve_pp": mean_abs_delta,
                "weighted_mean_abs_delta_resolve_pp": weighted_mean_abs_delta,
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
    return pd.DataFrame(rows)


def _strip_slot_suffix(traj_id: str) -> str:
    value = str(traj_id)
    if "__tbslot__" in value:
        return value.split("__tbslot__", 1)[0]
    if "__tb__" in value:
        return value.split("__tb__", 1)[0]
    return value


def _load_predictions(fold_dir: Path) -> pd.DataFrame:
    path = fold_dir / "test_predictions_safe_stop.parquet"
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_parquet(path)


def _load_metadata(prefix_table: Path) -> pd.DataFrame:
    if not prefix_table.exists():
        raise FileNotFoundError(prefix_table)
    metadata = pd.read_parquet(prefix_table, columns=METADATA_COLUMNS)
    metadata["prefix_step_idx"] = pd.to_numeric(metadata["prefix_step_idx"], errors="raise").astype("int64")
    return metadata.drop_duplicates(["traj_id", "prefix_step_idx"], keep="last")


def _attach_metadata(predictions: pd.DataFrame, metadata: pd.DataFrame) -> pd.DataFrame:
    if set(METADATA_COLUMNS[2:]).issubset(predictions.columns):
        return predictions
    out = predictions.copy()
    out["prefix_step_idx"] = pd.to_numeric(out["prefix_step_idx"], errors="raise").astype("int64")
    out = out.merge(metadata, on=["traj_id", "prefix_step_idx"], how="left")
    missing = out[out["harness_pair_id"].isna()][["traj_id", "prefix_step_idx"]].drop_duplicates()
    if not missing.empty:
        sample = missing.head(5).to_dict(orient="records")
        raise ValueError(f"Missing slot4x4 metadata for {len(missing)} prediction prefixes, e.g. {sample}")
    return out


def _load_token_rows(path: Path | None, wanted_ids: set[str]) -> pd.DataFrame | None:
    if path is None:
        return None
    if not path.exists():
        raise FileNotFoundError(path)
    token_rows = pd.read_parquet(path, columns=TOKEN_PREFIX_COLUMNS)
    token_rows["token_traj_id"] = token_rows["traj_id"].astype(str)
    token_rows = token_rows[token_rows["token_traj_id"].isin(wanted_ids)].copy()
    missing = wanted_ids - set(token_rows["token_traj_id"].astype(str).unique())
    if missing:
        sample = ", ".join(sorted(missing)[:5])
        raise ValueError(f"Token prefix cache is missing {len(missing)} trajectories, e.g. {sample}")
    token_rows["prefix_step_idx"] = pd.to_numeric(token_rows["prefix_step_idx"], errors="raise").astype("int64")
    keep = ["token_traj_id", "prefix_step_idx", *TOKEN_PREFIX_COLUMNS[2:]]
    return token_rows[keep].drop_duplicates(["token_traj_id", "prefix_step_idx"], keep="last")


def _attach_token_rows(predictions: pd.DataFrame, token_rows: pd.DataFrame | None) -> pd.DataFrame:
    if token_rows is None:
        return predictions
    out = predictions.copy()
    out["token_traj_id"] = out["traj_id"].map(_strip_slot_suffix)
    out["prefix_step_idx"] = pd.to_numeric(out["prefix_step_idx"], errors="raise").astype("int64")
    out = out.merge(token_rows, on=["token_traj_id", "prefix_step_idx"], how="left")
    missing = out[out["baseline_input_tokens_est"].isna()][["traj_id", "prefix_step_idx"]].drop_duplicates()
    if not missing.empty:
        sample = missing.head(5).to_dict(orient="records")
        raise ValueError(f"Missing token rows for {len(missing)} prediction prefixes, e.g. {sample}")
    return out


def _single_value(frame: pd.DataFrame, col: str) -> str:
    values = sorted(str(value) for value in frame[col].dropna().unique())
    if len(values) != 1:
        raise ValueError(f"{col} should have exactly one value, got {values[:8]}")
    return values[0]


def _pct(value: float) -> float:
    return float(value) * 100.0 if math.isfinite(float(value)) else float("nan")


def _collect_fold_payloads(
    experiment_root: Path,
    dataset_subdir: str,
    success_col: str,
    failure_col: str,
    metadata: pd.DataFrame,
    allow_incomplete: bool,
) -> tuple[list[dict[str, Any]], set[str]]:
    payloads: list[dict[str, Any]] = []
    wanted_token_ids: set[str] = set()
    for setting, spec in SETTING_SPECS.items():
        unit_root = experiment_root / spec["run_subdir"] / dataset_subdir
        if not unit_root.exists():
            raise FileNotFoundError(unit_root)
        fold_dirs = sorted(path for path in unit_root.iterdir() if path.is_dir())
        for fold_dir in fold_dirs:
            if not (fold_dir / "_SUCCESS").exists() and not allow_incomplete:
                raise RuntimeError(f"Incomplete fold: {fold_dir}")
            predictions = _attach_metadata(_load_predictions(fold_dir), metadata)
            missing = [col for col in (success_col, failure_col) if col not in predictions.columns]
            if missing:
                raise KeyError(f"{fold_dir}: missing prediction columns {missing}")
            needed = [
                "harness_pair_id",
                "harness_model_group",
                "harness_agent_slot",
                "harness_agent_raw",
                "harness_original_model_id",
            ]
            missing_meta = [col for col in needed if col not in predictions.columns]
            if missing_meta:
                raise KeyError(f"{fold_dir}: missing metadata columns {missing_meta}")
            wanted_token_ids.update(_strip_slot_suffix(value) for value in predictions["traj_id"].astype(str).unique())
            payloads.append(
                {
                    "setting": setting,
                    "fold_dir": fold_dir,
                    "heldout_unit": fold_dir.name,
                    "predictions": predictions,
                }
            )
    return payloads, wanted_token_ids


def _build_per_pair(
    payloads: list[dict[str, Any]],
    token_rows: pd.DataFrame | None,
    thresholds: list[float],
    success_col: str,
    failure_col: str,
    score_mode: str,
    predictor: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for payload in payloads:
        setting = str(payload["setting"])
        spec = SETTING_SPECS[setting]
        fold_dir = Path(payload["fold_dir"])
        heldout_unit = str(payload["heldout_unit"])
        predictions = _attach_token_rows(payload["predictions"], token_rows)
        for pair_id, pair in predictions.groupby("harness_pair_id", sort=True):
            model_group = _single_value(pair, "harness_model_group")
            agent_slot = _single_value(pair, "harness_agent_slot")
            agent_raw = _single_value(pair, "harness_agent_raw")
            original_model_id = _single_value(pair, "harness_original_model_id")
            expected_heldout = _single_value(pair, str(spec["heldout_col"]))
            if expected_heldout != heldout_unit:
                raise ValueError(f"{fold_dir}: heldout mismatch {heldout_unit=} {expected_heldout=}")
            for threshold in thresholds:
                summary = _evaluate_unit(pair, float(threshold), float(threshold), success_col, failure_col)
                delta_resolve_pp = -float(summary["resolve_rate_drop"]) * 100.0
                rows.append(
                    {
                        "setting": setting,
                        "fold_dir": str(fold_dir),
                        "heldout_unit": heldout_unit,
                        "rank_item_type": str(spec["rank_item_name"]),
                        "rank_item": _single_value(pair, str(spec["rank_item_col"])),
                        "unit": str(pair_id),
                        "pair_id": str(pair_id),
                        "model_group": model_group,
                        "agent_slot": agent_slot,
                        "agent_raw": agent_raw,
                        "original_model_id": original_model_id,
                        "score_mode": score_mode,
                        "prefix_model": predictor,
                        "policy_mode": "dual",
                        "threshold": float(threshold),
                        "success_thr": float(threshold),
                        "failure_thr": float(threshold),
                        "min_step": 0,
                        "consecutive": 1,
                        **summary,
                        "full_resolve_rate_pct": _pct(summary["original_resolve_rate"]),
                        "early_resolve_rate_pct": _pct(summary["adjusted_resolve_rate"]),
                        "delta_resolve_pp": delta_resolve_pp,
                        "abs_delta_resolve_pp": abs(delta_resolve_pp),
                        "coverage_pct": _pct(summary["coverage"]),
                        "decision_accuracy_pct": _pct(summary["decision_accuracy"]),
                        "step_save_pct": float(summary["pct_steps_saved"]),
                    }
                )
    return pd.DataFrame(rows)


def _ranking_pair_summary(part: pd.DataFrame) -> dict[str, Any]:
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
        "pair_order_comparable": int(comparable),
        "pair_order_preserved": int(preserved),
        "pair_order_reversed": int(reversed_count),
        "pair_preserve_rate": preserved / comparable if comparable else float("nan"),
    }


def _build_rankings(per_pair: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rank_rows: list[pd.DataFrame] = []
    summary_rows: list[dict[str, Any]] = []
    for (setting, threshold, heldout_unit), part in per_pair.groupby(["setting", "threshold", "heldout_unit"], sort=True):
        ranked = part.copy()
        ranked["full_rank"] = ranked["original_resolve_rate"].rank(method="first", ascending=False).astype(int)
        ranked["early_rank"] = ranked["adjusted_resolve_rate"].rank(method="first", ascending=False).astype(int)
        ranked["rank_shift_up_positive"] = ranked["full_rank"] - ranked["early_rank"]
        ranked = ranked.sort_values(["full_rank", "rank_item"]).copy()
        rank_rows.append(ranked)
        full = ranked["full_rank"].to_numpy(dtype=float)
        early = ranked["early_rank"].to_numpy(dtype=float)
        rho = float(pd.Series(full).corr(pd.Series(early), method="spearman")) if len(ranked) > 1 else float("nan")
        pairs = _ranking_pair_summary(ranked)
        full_top = ranked.loc[ranked["full_rank"].idxmin(), "rank_item"]
        early_top = ranked.loc[ranked["early_rank"].idxmin(), "rank_item"]
        summary_rows.append(
            {
                "setting": setting,
                "threshold": float(threshold),
                "heldout_unit": heldout_unit,
                "rank_item_type": ranked["rank_item_type"].iloc[0],
                "n_ranked": int(len(ranked)),
                "spearman_rho": rho,
                **pairs,
                "top_item_same": bool(full_top == early_top),
                "exact_all_ranks_same": bool((ranked["full_rank"] == ranked["early_rank"]).all()),
                "max_abs_rank_shift": int(ranked["rank_shift_up_positive"].abs().max()),
                "mean_abs_rank_shift": float(ranked["rank_shift_up_positive"].abs().mean()),
            }
        )
    ranking = pd.concat(rank_rows, ignore_index=True) if rank_rows else pd.DataFrame()
    keep = [
        "setting",
        "threshold",
        "heldout_unit",
        "rank_item_type",
        "rank_item",
        "pair_id",
        "model_group",
        "agent_slot",
        "agent_raw",
        "original_model_id",
        "original_total",
        "full_resolve_rate_pct",
        "early_resolve_rate_pct",
        "delta_resolve_pp",
        "abs_delta_resolve_pp",
        "full_rank",
        "early_rank",
        "rank_shift_up_positive",
        "coverage_pct",
        "decision_accuracy_pct",
        "step_save_pct",
        "input_token_save_pct_est",
        "output_token_save_pct_est",
        "total_token_save_pct_est",
        "n_decided",
        "decided_success",
        "decided_failure",
    ]
    keep = [col for col in keep if col in ranking.columns]
    return ranking[keep], pd.DataFrame(summary_rows)


def _aggregate_entity_ranking(per_pair: pd.DataFrame) -> pd.DataFrame:
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
    for (setting, threshold, rank_item), part in per_pair.groupby(["setting", "threshold", "rank_item"], sort=True):
        counts = {col: float(part[col].sum()) for col in count_cols}
        total = int(part["original_total"].sum())
        resolved = int(part["original_resolved"].sum())
        summary = _summarize(counts, total, resolved)
        delta_resolve_pp = -float(summary["resolve_rate_drop"]) * 100.0
        rows.append(
            {
                "setting": setting,
                "threshold": float(threshold),
                "rank_item_type": part["rank_item_type"].iloc[0],
                "rank_item": rank_item,
                "n_pairs": int(part["pair_id"].nunique()),
                "total_trajectories": total,
                "full_weighted_resolve_rate_pct": _pct(summary["original_resolve_rate"]),
                "early_weighted_resolve_rate_pct": _pct(summary["adjusted_resolve_rate"]),
                "delta_resolve_pp": delta_resolve_pp,
                "abs_delta_resolve_pp": abs(delta_resolve_pp),
                "coverage_pct": _pct(summary["coverage"]),
                "decision_accuracy_pct": _pct(summary["decision_accuracy"]),
                "step_save_pct": float(summary["pct_steps_saved"]),
                "input_token_save_pct_est": float(summary["input_token_save_pct_est"]),
                "output_token_save_pct_est": float(summary["output_token_save_pct_est"]),
                "total_token_save_pct_est": float(summary["total_token_save_pct_est"]),
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["full_rank"] = out.groupby(["setting", "threshold"])["full_weighted_resolve_rate_pct"].rank(
        method="first", ascending=False
    ).astype(int)
    out["early_rank"] = out.groupby(["setting", "threshold"])["early_weighted_resolve_rate_pct"].rank(
        method="first", ascending=False
    ).astype(int)
    out["rank_shift_up_positive"] = out["full_rank"] - out["early_rank"]
    return out.sort_values(["setting", "threshold", "full_rank", "rank_item"]).reset_index(drop=True)


def _global_by_setting(per_pair: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for setting, part in per_pair.groupby("setting", sort=True):
        current = _aggregate(part)
        current.insert(0, "setting", setting)
        frames.append(current)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _write_readme(
    output_dir: Path,
    experiment_root: Path,
    thresholds: list[float],
    global_summary: pd.DataFrame,
    ranking_summary: pd.DataFrame,
    token_rows: pd.DataFrame | None,
) -> None:
    lines = [
        "# TerminalBench 4x4 Harness-Debug Exclusion",
        "",
        "This folder is a compact summary for the TerminalBench diagnostic experiment with four base models and four agent slots.",
        "",
        "## Design",
        "",
        "- `leave_model`: train on 3 model groups x 4 agent slots; test the held-out model group on its 4 agent slots.",
        "- `leave_agent`: train on 4 model groups x 3 agent slots; test the held-out agent slot on its 4 model groups.",
        "- Total training runs: 8.",
        "- Total evaluated pair entries: 32, counting 16 model-agent pairs under `leave_model` plus the same 16 pairs under `leave_agent`.",
        "- Policy: calibrated dual-head, symmetric success/failure threshold, `min_step=0`, `consecutive=1`.",
        "",
        "## Inputs",
        "",
        f"- experiment_root: `{experiment_root}`",
        f"- thresholds: `{', '.join(str(value) for value in thresholds)}`",
        f"- token method: `{TOKEN_METHOD}`" if token_rows is not None else "- token method: not computed",
        "",
        "## Files",
        "",
        "- `slot4x4_fixed_threshold_per_pair.csv`: one row per setting, threshold, and model-agent pair.",
        "- `slot4x4_fixed_threshold_global_by_setting.csv`: global metrics by setting and threshold.",
        "- `slot4x4_fixed_threshold_rankings.csv`: within-heldout ranking rows.",
        "- `slot4x4_fixed_threshold_ranking_summary.csv`: rank preservation summary per held-out group.",
        "- `slot4x4_fixed_threshold_entity_aggregate_ranking.csv`: aggregate agent-slot ranking for `leave_model` and aggregate model ranking for `leave_agent`.",
        "- `abs_delta_resolve_pp` is the absolute value of signed `delta_resolve_pp`; global `mean_abs_delta_resolve_pp` averages this over model-agent pairs.",
        "",
        "## Global Metrics",
        "",
    ]
    display_cols = [
        "setting",
        "threshold",
        "coverage_pct",
        "decision_accuracy_pct",
        "step_save_pct",
        "input_token_save_pct_est",
        "output_token_save_pct_est",
        "total_token_save_pct_est",
        "delta_resolve_pp",
        "abs_delta_resolve_pp",
        "mean_abs_delta_resolve_pp",
        "weighted_mean_abs_delta_resolve_pp",
    ]
    present = [col for col in display_cols if col in global_summary.columns]
    lines.append(global_summary[present].to_markdown(index=False))
    lines.extend(["", "## Rank Preservation", ""])
    rank_cols = [
        "setting",
        "threshold",
        "heldout_unit",
        "rank_item_type",
        "spearman_rho",
        "pair_preserve_rate",
        "top_item_same",
        "exact_all_ranks_same",
        "max_abs_rank_shift",
    ]
    present_rank = [col for col in rank_cols if col in ranking_summary.columns]
    lines.append(ranking_summary[present_rank].to_markdown(index=False))
    lines.append("")
    output_dir.joinpath("README.md").write_text("\n".join(lines), encoding="utf-8")


def build_summary(
    experiment_root: Path,
    output_dir: Path,
    dataset_subdir: str,
    predictor: str,
    thresholds: list[float],
    score_mode: str,
    prefix_table: Path,
    token_prefix_cache: Path | None,
    allow_incomplete: bool,
) -> None:
    success_col = _head_column("success", score_mode, predictor)
    failure_col = _head_column("failure", score_mode, predictor)
    metadata = _load_metadata(prefix_table)
    payloads, wanted_token_ids = _collect_fold_payloads(
        experiment_root=experiment_root,
        dataset_subdir=dataset_subdir,
        success_col=success_col,
        failure_col=failure_col,
        metadata=metadata,
        allow_incomplete=allow_incomplete,
    )
    token_rows = _load_token_rows(token_prefix_cache, wanted_token_ids)
    per_pair = _build_per_pair(
        payloads=payloads,
        token_rows=token_rows,
        thresholds=thresholds,
        success_col=success_col,
        failure_col=failure_col,
        score_mode=score_mode,
        predictor=predictor,
    )
    global_summary = _global_by_setting(per_pair)
    rankings, ranking_summary = _build_rankings(per_pair)
    entity_ranking = _aggregate_entity_ranking(per_pair)

    output_dir.mkdir(parents=True, exist_ok=True)
    per_pair.to_csv(output_dir / "slot4x4_fixed_threshold_per_pair.csv", index=False)
    global_summary.to_csv(output_dir / "slot4x4_fixed_threshold_global_by_setting.csv", index=False)
    rankings.to_csv(output_dir / "slot4x4_fixed_threshold_rankings.csv", index=False)
    ranking_summary.to_csv(output_dir / "slot4x4_fixed_threshold_ranking_summary.csv", index=False)
    entity_ranking.to_csv(output_dir / "slot4x4_fixed_threshold_entity_aggregate_ranking.csv", index=False)

    _write_readme(
        output_dir=output_dir,
        experiment_root=experiment_root,
        thresholds=thresholds,
        global_summary=global_summary,
        ranking_summary=ranking_summary,
        token_rows=token_rows,
    )


def main() -> int:
    args = parse_args()
    build_summary(
        experiment_root=args.experiment_root,
        output_dir=args.output_dir,
        dataset_subdir=args.dataset_subdir,
        predictor=args.predictor,
        thresholds=[float(value) for value in args.thresholds],
        score_mode=args.score_mode,
        prefix_table=args.prefix_table,
        token_prefix_cache=args.token_prefix_cache,
        allow_incomplete=bool(args.allow_incomplete),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
