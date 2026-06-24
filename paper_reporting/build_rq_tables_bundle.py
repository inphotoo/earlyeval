#!/usr/bin/env python3
"""Build paper-ready RQ1/RQ2/RQ3 reporting tables.

This is a post-hoc reporter. It reads completed safe-stop prediction artifacts,
replays fixed-threshold policies on held-out folds, and writes a compact bundle
of CSV/LaTeX/README outputs. It does not train models or mutate experiment
directories.
"""

from __future__ import annotations

import json
import math
import os
import re
import hashlib
import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _default_package_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "configs" / "rq_final.yaml").exists():
            return parent
    return Path.cwd()


ROOT = Path(os.environ.get("SWEBENCH_PACKAGE_ROOT", str(_default_package_root()))).resolve()
PAPER_DATA = Path(
    os.environ.get(
        "EARLYEVAL_PAPER_DATA",
        str(ROOT / "paper/icse_submission_draft/data"),
    )
).resolve()
EXP = Path(
    os.environ.get(
        "EARLYEVAL_EXPERIMENT_DIR",
        str(ROOT / "paper/experiments/rq_final_lightgbm_17"),
    )
).resolve()
OUT = Path(
    os.environ.get(
        "RQ_TABLES_OUT",
        str(ROOT / "paper/icse_submission_draft/rq_tables_reorg_20260623"),
    )
).resolve()
SUPPORTING = OUT / "supporting"

THRESHOLDS = [0.75, 0.80, 0.85, 0.90, 0.95, 0.97]
LEGACY_CHAR_TOKEN_RATIO = 4.0
TOKEN_METHOD = "model_tokenizer_component_sum_context_call_plus_generated_output"
TOKENIZER_CACHE_ROOT = Path(
    os.environ.get("EARLYEVAL_TOKENIZER_CACHE_ROOT", str(ROOT / "outputs/tokenizer_cache"))
).resolve()
TOKEN_PREFIX_CACHE_DIR = Path(
    os.environ.get("EARLYEVAL_TOKEN_PREFIX_CACHE_DIR", str(EXP / "token_prefix_cache"))
).resolve()
TOKENIZER_LOCAL_FILES_ONLY = os.environ.get("EARLYEVAL_TOKENIZER_LOCAL_FILES_ONLY", "0").strip() in {
    "1",
    "true",
    "TRUE",
    "yes",
}
TOKEN_PARQUET_BATCH_SIZE = int(os.environ.get("EARLYEVAL_TOKEN_PARQUET_BATCH_SIZE", "8192"))
TOKEN_ENCODE_BATCH_SIZE = int(os.environ.get("EARLYEVAL_TOKEN_ENCODE_BATCH_SIZE", "64"))
TOKEN_PROGRESS_EVERY_ROWS = int(os.environ.get("EARLYEVAL_TOKEN_PROGRESS_EVERY_ROWS", "50000"))
TOKEN_PROGRESS_EVERY_BATCHES = int(os.environ.get("EARLYEVAL_TOKEN_PROGRESS_EVERY_BATCHES", "25"))


@dataclass(frozen=True)
class BenchConfig:
    benchmark: str
    dataset_key: str
    n_test_agents: int
    run_root: Path
    fold_glob: str
    predictor: str
    score_mode: str = "calibrated"
    preset: str = "locked095"
    prefix_table: Path | None = None


BENCHES = [
    BenchConfig(
        benchmark="SWE-bench Verified",
        dataset_key="sweverify",
        n_test_agents=16,
        run_root=EXP / "lightgbm_main/folds",
        fold_glob="*",
        predictor="I_LightGBM_Dense_AF",
        prefix_table=ROOT / "paper/data/raw/0049_dataset_sweverify_prefix_table.parquet",
    ),
    BenchConfig(
        benchmark="TerminalBench",
        dataset_key="terminalbench",
        n_test_agents=33,
        run_root=EXP
        / "robustness_loo_model_holdout_rich_af_gold_memory_limited/terminalbench",
        fold_glob="*",
        predictor="Robust_LightGBM_Dense_AF_Gold",
        preset="rich_af_gold_locked095",
        prefix_table=ROOT / "paper/data/raw/0050_dataset_terminalbench_prefix_table.parquet",
    ),
    BenchConfig(
        benchmark="Toolathlon",
        dataset_key="toolathlon",
        n_test_agents=22,
        run_root=EXP
        / "robustness_loo_model_holdout_rich_af_gold_memory_limited/toolathlon",
        fold_glob="*",
        predictor="Robust_LightGBM_Dense_AF_Gold",
        preset="rich_af_gold_locked095",
        prefix_table=ROOT / "paper/data/raw/0051_dataset_toolathlon_prefix_table.parquet",
    ),
]


def _pct(num: float, den: float) -> float:
    return float(num) * 100.0 / float(den) if den else float("nan")


def _round_df(df: pd.DataFrame, digits: int = 4) -> pd.DataFrame:
    out = df.copy()
    for col in out.select_dtypes(include=["float", "float64", "float32"]).columns:
        out[col] = out[col].round(digits)
    return out


def _score_cols(score_mode: str, predictor: str) -> tuple[str, str]:
    if score_mode == "calibrated":
        return (
            f"prob_cal_safe_success__{predictor}",
            f"prob_cal_safe_failure__{predictor}",
        )
    if score_mode == "raw":
        return (f"prob_safe_success__{predictor}", f"prob_safe_failure__{predictor}")
    raise ValueError(score_mode)


def _agent_from_fold_dir(path: Path) -> str:
    selected = path / "safe_stop_test_selected.csv"
    if selected.exists():
        df = pd.read_csv(selected, nrows=1)
        if "run" in df.columns and len(df):
            run = str(df["run"].iloc[0])
            if run:
                return Path(run).name
    return path.name


def _display_agent(raw: str) -> str:
    text = str(raw)
    mapping = {
        "20251118_mini-v1.15.0_gemini-3-pro-preview-20251118": "Gemini-3-Pro",
        "20251211_mini-v1.17.2_gpt-5.2-2025-12-11-high": "GPT-5.2-High",
        "20250929_mini-v1.13.3_sonnet-4-5-20250929": "Claude-Sonnet-4.5",
        "20251211_mini-v1.17.2_gpt-5.2-2025-12-11": "GPT-5.2",
        "20251124_mini-v1.16.0_gpt-5.1-codex": "GPT-5.1-Codex",
        "20251120_mini-v1.15.0_gpt-5.1-2025-11-13": "GPT-5.1",
        "20250807_mini-v1.7.0_gpt-5": "GPT-5",
        "20251210_mini-v1.17.2_kimi-k2-thinking": "Kimi-K2-Thinking",
        "20251201_mini-v1.17.1_deepseek-v3.2-reasoner": "DeepSeek-V3.2",
        "20251124_mini-v1.17.0_minimax-m2": "MiniMax-M2",
        "20250822_mini-v1.9.1_glm-4.5": "GLM-4.5",
        "20250807_mini-v1.7.0_gpt-5-mini": "GPT-5-Mini",
        "20250807_mini-v1.7.0_gpt-5-nano": "GPT-5-Nano",
        "20251209_mini-v1.17.2_devstral-2512": "Devstral-Large",
        "20251209_mini-v1.17.2_devstral-small-2512": "Devstral-Small",
        "20251201_mini-v1.17.1_glm-4.6": "GLM-4.6",
    }
    if text in mapping:
        return mapping[text]
    if "@" in text:
        left, provider = text.split("@", 1)
        return f"{left} ({provider})"
    return text.replace("_", "/")


def _decide(
    steps: np.ndarray,
    success: np.ndarray,
    failure: np.ndarray,
    *,
    success_thr: float,
    failure_thr: float,
    min_step: int = 0,
    consecutive: int = 1,
) -> tuple[bool, str, int, float, float, float]:
    last_decision = "undecided"
    streak = 0
    for step_value, success_score, failure_score in zip(steps, success, failure):
        step = int(step_value)
        if step < min_step:
            continue
        success_score = float(success_score)
        failure_score = float(failure_score)
        success_hit = success_score >= success_thr
        failure_hit = failure_score >= failure_thr
        if success_hit and failure_hit:
            success_margin = success_score - success_thr
            failure_margin = failure_score - failure_thr
            decision = "success" if success_margin >= failure_margin else "failure"
            score = success_score if decision == "success" else failure_score
        elif success_hit:
            decision = "success"
            score = success_score
        elif failure_hit:
            decision = "failure"
            score = failure_score
        else:
            last_decision = "undecided"
            streak = 0
            continue
        streak = streak + 1 if decision == last_decision else 1
        last_decision = decision
        if streak >= consecutive:
            return True, decision, step, score, success_score, failure_score
    return False, "undecided", -1, float("nan"), float("nan"), float("nan")


def _load_prediction_records(config: BenchConfig) -> list[dict[str, Any]]:
    success_col, failure_col = _score_cols(config.score_mode, config.predictor)
    cols = [
        "traj_id",
        "orig_model_id",
        "orig_model",
        "label",
        "prefix_step_idx",
        success_col,
        failure_col,
    ]
    records: list[dict[str, Any]] = []
    fold_dirs = [p for p in sorted(config.run_root.glob(config.fold_glob)) if p.is_dir()]
    for fold_dir in fold_dirs:
        pred_path = fold_dir / "test_predictions_safe_stop.parquet"
        if not pred_path.exists():
            continue
        df = pd.read_parquet(pred_path, columns=cols)
        agent = _agent_from_fold_dir(fold_dir)
        for traj_id, group in df.groupby("traj_id", sort=False):
            group = group.sort_values("prefix_step_idx")
            agent_model = str(group["orig_model_id"].iloc[0])
            if agent_model in {"", "nan", "__MISSING__"}:
                agent_model = str(group["orig_model"].iloc[0])
            records.append(
                {
                    "benchmark": config.benchmark,
                    "dataset_key": config.dataset_key,
                    "preset": config.preset,
                    "fold_id": fold_dir.name,
                    "agent": agent_model,
                    "agent_dir": agent,
                    "agent_display": _display_agent(agent_model),
                    "traj_id": str(traj_id),
                    "label": int(group["label"].iloc[0]),
                    "n_steps": int(len(group)),
                    "steps": group["prefix_step_idx"].to_numpy(dtype=np.int32),
                    "success": group[success_col].to_numpy(dtype=np.float64),
                    "failure": group[failure_col].to_numpy(dtype=np.float64),
                }
            )
    if not records:
        raise RuntimeError(f"no prediction records loaded for {config.benchmark}")
    return records


def _evaluate_records(
    records: list[dict[str, Any]],
    *,
    threshold: float,
    policy_mode: str = "dual",
) -> tuple[dict[str, Any], pd.DataFrame]:
    if policy_mode == "dual":
        success_thr = threshold
        failure_thr = threshold
    elif policy_mode == "success_only":
        success_thr = threshold
        failure_thr = math.inf
    elif policy_mode == "failure_only":
        success_thr = math.inf
        failure_thr = threshold
    else:
        raise ValueError(policy_mode)

    per_agent: dict[str, dict[str, Any]] = {}
    decisions: list[dict[str, Any]] = []
    for record in records:
        agent = record["agent"]
        stats = per_agent.setdefault(
            agent,
            {
                "benchmark": record["benchmark"],
                "dataset_key": record["dataset_key"],
                "preset": record["preset"],
                "agent": agent,
                "agent_display": record["agent_display"],
                "original_total": 0,
                "original_resolved": 0,
                "false_negatives": 0,
                "true_negatives": 0,
                "false_positives": 0,
                "true_positives": 0,
                "decided_failure": 0,
                "decided_success": 0,
                "undecided": 0,
                "total_saved_steps": 0,
                "total_steps": 0,
            },
        )
        label = int(record["label"])
        n_steps = int(record["n_steps"])
        decided, decision, decision_step, score, ps, pf = _decide(
            record["steps"],
            record["success"],
            record["failure"],
            success_thr=success_thr,
            failure_thr=failure_thr,
        )
        stats["original_total"] += 1
        stats["original_resolved"] += label
        stats["total_steps"] += n_steps
        if not decided:
            stats["undecided"] += 1
            outcome = "undecided_success" if label else "undecided_failure"
            saved_steps = 0
        else:
            saved_steps = max(n_steps - decision_step - 1, 0)
            stats["total_saved_steps"] += saved_steps
            if decision == "success":
                stats["decided_success"] += 1
                if label:
                    stats["true_positives"] += 1
                    outcome = "true_positive"
                else:
                    stats["false_positives"] += 1
                    outcome = "false_positive"
            else:
                stats["decided_failure"] += 1
                if label:
                    stats["false_negatives"] += 1
                    outcome = "false_negative"
                else:
                    stats["true_negatives"] += 1
                    outcome = "true_negative"
        decisions.append(
            {
                "benchmark": record["benchmark"],
                "dataset_key": record["dataset_key"],
                "preset": record["preset"],
                "threshold": threshold,
                "policy_mode": policy_mode,
                "agent": agent,
                "agent_display": record["agent_display"],
                "traj_id": record["traj_id"],
                "label": label,
                "n_steps": n_steps,
                "decided": bool(decided),
                "decision": decision,
                "decision_step": int(decision_step),
                "round_1based": int(decision_step + 1) if decided else -1,
                "decision_score": score,
                "prob_success_at_decision": ps,
                "prob_failure_at_decision": pf,
                "saved_steps": int(saved_steps),
                "outcome_type": outcome,
            }
        )

    rows = [_summarize_agent(stats, threshold, policy_mode) for stats in per_agent.values()]
    per_agent_df = pd.DataFrame(rows)
    aggregate = _summarize_agent(
        _combine_stats(per_agent.values()), threshold, policy_mode, aggregate=True
    )
    return aggregate, pd.DataFrame(decisions), per_agent_df


def _combine_stats(stats_iter: Any) -> dict[str, Any]:
    combined: dict[str, Any] = {
        "benchmark": "",
        "dataset_key": "",
        "preset": "",
        "agent": "ALL",
        "agent_display": "All",
        "original_total": 0,
        "original_resolved": 0,
        "false_negatives": 0,
        "true_negatives": 0,
        "false_positives": 0,
        "true_positives": 0,
        "decided_failure": 0,
        "decided_success": 0,
        "undecided": 0,
        "total_saved_steps": 0,
        "total_steps": 0,
    }
    first = True
    for stats in stats_iter:
        if first:
            combined["benchmark"] = stats["benchmark"]
            combined["dataset_key"] = stats["dataset_key"]
            combined["preset"] = stats["preset"]
            first = False
        for key in [
            "original_total",
            "original_resolved",
            "false_negatives",
            "true_negatives",
            "false_positives",
            "true_positives",
            "decided_failure",
            "decided_success",
            "undecided",
            "total_saved_steps",
            "total_steps",
        ]:
            combined[key] += int(stats[key])
    return combined


def _summarize_agent(
    stats: dict[str, Any], threshold: float, policy_mode: str, aggregate: bool = False
) -> dict[str, Any]:
    tp = int(stats["true_positives"])
    fp = int(stats["false_positives"])
    fn = int(stats["false_negatives"])
    tn = int(stats["true_negatives"])
    decided_success = int(stats["decided_success"])
    decided_failure = int(stats["decided_failure"])
    n_decided = decided_success + decided_failure
    original_total = int(stats["original_total"])
    original_resolved = int(stats["original_resolved"])
    adjusted_resolved = original_resolved - fn + fp
    precision = tp / decided_success if decided_success else float("nan")
    recall = tp / (tp + fn) if (tp + fn) else float("nan")
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision == precision and recall == recall and (precision + recall)
        else float("nan")
    )
    row = {
        "benchmark": stats["benchmark"],
        "dataset_key": stats["dataset_key"],
        "preset": stats["preset"],
        "threshold": float(threshold),
        "policy_mode": policy_mode,
        "agent": stats["agent"],
        "agent_display": stats["agent_display"],
        "original_total": original_total,
        "original_resolved": original_resolved,
        "original_resolve_rate_pct": _pct(original_resolved, original_total),
        "adjusted_resolved": adjusted_resolved,
        "adjusted_resolve_rate_pct": _pct(adjusted_resolved, original_total),
        "resolve_change_pp": _pct(adjusted_resolved - original_resolved, original_total),
        "false_negatives": fn,
        "true_negatives": tn,
        "false_positives": fp,
        "true_positives": tp,
        "decided_failure": decided_failure,
        "decided_success": decided_success,
        "undecided": int(stats["undecided"]),
        "n_decided": n_decided,
        "coverage_pct": _pct(n_decided, original_total),
        "decision_accuracy_pct": _pct(tp + tn, n_decided),
        "precision_resolved": precision * 100.0 if precision == precision else float("nan"),
        "recall_resolved": recall * 100.0 if recall == recall else float("nan"),
        "f1_resolved": f1 * 100.0 if f1 == f1 else float("nan"),
        "total_saved_steps": int(stats["total_saved_steps"]),
        "total_steps": int(stats["total_steps"]),
        "global_step_save_pct": _pct(stats["total_saved_steps"], stats["total_steps"]),
    }
    if aggregate:
        row["agent"] = "ALL"
        row["agent_display"] = "All"
    return row


def _rank_change(per_agent: pd.DataFrame) -> pd.DataFrame:
    out = per_agent.copy()
    out["original_rank"] = (
        out["original_resolve_rate_pct"].rank(ascending=False, method="min").astype(int)
    )
    out["adjusted_rank"] = (
        out["adjusted_resolve_rate_pct"].rank(ascending=False, method="min").astype(int)
    )
    out["rank_change_positive_is_up"] = out["original_rank"] - out["adjusted_rank"]
    out = out.sort_values(["original_rank", "agent_display"]).reset_index(drop=True)
    return out


def _spearman_from_ranks(ranked: pd.DataFrame) -> float:
    if len(ranked) < 2:
        return float("nan")
    return float(ranked["original_rank"].corr(ranked["adjusted_rank"], method="spearman"))


def _build_decisions_all() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, pd.DataFrame]]:
    agg_rows: list[dict[str, Any]] = []
    per_agent_rows: list[pd.DataFrame] = []
    decisions_all: list[pd.DataFrame] = []
    ranked_by_bench: dict[str, pd.DataFrame] = {}
    for config in BENCHES:
        records = _load_prediction_records(config)
        for thr in THRESHOLDS:
            aggregate, decisions, per_agent = _evaluate_records(records, threshold=thr)
            agg_rows.append(aggregate)
            per_agent_rows.append(per_agent)
            decisions_all.append(decisions)
            if abs(thr - 0.95) < 1e-9:
                ranked_by_bench[config.benchmark] = _rank_change(per_agent)
    return (
        pd.DataFrame(agg_rows),
        pd.concat(per_agent_rows, ignore_index=True),
        pd.concat(decisions_all, ignore_index=True),
        ranked_by_bench,
    )


def _load_tokenizer_helpers() -> Any:
    candidates = [
        Path(__file__).with_name("build_internal_review_swe16.py"),
        ROOT / "paper/experiments/rq_final_lightgbm_17/build_internal_review_swe16.py",
    ]
    for path in candidates:
        if not path.exists():
            continue
        spec = importlib.util.spec_from_file_location("earlyeval_tokenizer_helpers", path)
        if spec is None or spec.loader is None:
            continue
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module
    raise FileNotFoundError("Could not find build_internal_review_swe16.py for tokenizer helpers.")


def _bool_env(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y"}


def _token_prefix_cache_path(dataset_key: str) -> Path:
    return TOKEN_PREFIX_CACHE_DIR / f"{dataset_key}_model_tokenizer_source_split_prefix_tokens.parquet"


def _token_prefix_cache_sidecar(cache_path: Path, suffix: str) -> Path:
    return cache_path.with_name(f"{cache_path.name}{suffix}")


def _source_token_cache_meta(prefix_table: Path, traj_ids: set[str]) -> dict[str, Any]:
    resolved = prefix_table.expanduser().resolve()
    stat = resolved.stat()
    traj_digest = hashlib.sha256(
        "\n".join(sorted(str(x) for x in traj_ids)).encode("utf-8")
    ).hexdigest()
    return {
        "cache_version": 3,
        "token_method": TOKEN_METHOD,
        "prefix_table": str(resolved),
        "prefix_table_size": int(stat.st_size),
        "prefix_table_mtime_ns": int(stat.st_mtime_ns),
        "traj_count": int(len(traj_ids)),
        "traj_sha256": traj_digest,
        "tokenizer_local_files_only": bool(TOKENIZER_LOCAL_FILES_ONLY),
    }


def _required_source_token_columns() -> set[str]:
    return {
        "traj_id",
        "model_id",
        "prefix_step_idx",
        "context_call_input_tokens_est",
        "future_context_call_input_tokens_saved_if_stop_est",
        "baseline_input_tokens_est",
        "external_input_tokens_cum",
        "generated_output_tokens_cum",
        "transcript_total_tokens_cum",
        "baseline_external_input_tokens_est",
        "baseline_output_tokens_est",
        "baseline_transcript_total_tokens_est",
        "token_count_method",
        "tokenizer_family",
        "tokenizer_backend",
        "tokenizer_name",
    }


def _load_source_token_cache(
    cache_path: Path,
    expected_meta: dict[str, Any],
    traj_ids: set[str],
) -> tuple[pd.DataFrame, pd.DataFrame] | None:
    meta_path = _token_prefix_cache_sidecar(cache_path, ".meta.json")
    manifest_path = _token_prefix_cache_sidecar(cache_path, ".manifest.csv")
    if not cache_path.exists() or not meta_path.exists() or not manifest_path.exists():
        return None
    try:
        actual_meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if any(actual_meta.get(key) != value for key, value in expected_meta.items()):
        return None
    try:
        token_rows = pd.read_parquet(cache_path)
        manifest = pd.read_csv(manifest_path)
    except Exception:
        return None
    if not _required_source_token_columns().issubset(token_rows.columns):
        return None
    wanted = set(str(x) for x in traj_ids)
    if set(token_rows["traj_id"].astype(str)) != wanted:
        return None
    token_rows["traj_id"] = token_rows["traj_id"].astype(str)
    return token_rows, manifest


def _write_source_token_cache(
    cache_path: Path,
    token_rows: pd.DataFrame,
    manifest: pd.DataFrame,
    meta: dict[str, Any],
) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    token_rows.to_parquet(cache_path, index=False)
    manifest.to_csv(_token_prefix_cache_sidecar(cache_path, ".manifest.csv"), index=False)
    _token_prefix_cache_sidecar(cache_path, ".meta.json").write_text(
        json.dumps(meta, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _safe_string(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    return str(value)


def _add_prefix_source_token_features(raw: pd.DataFrame, helpers: Any) -> pd.DataFrame:
    df = raw.sort_values(["traj_id", "prefix_step_idx"]).reset_index(drop=True)
    for kind in ["action", "feedback", "thought", "assistant"]:
        df[f"{kind}_prefix_tokens_component"] = (
            df.groupby("traj_id", sort=False)[f"{kind}_step_tokens_component"]
            .cumsum()
            .astype("int64")
        )
        df[f"{kind}_prefix_nonempty_component"] = (
            df.groupby("traj_id", sort=False)[f"{kind}_step_nonempty_component"]
            .cumsum()
            .astype("int64")
        )
    df["step_method_severity_cum_component"] = (
        df.groupby("traj_id", sort=False)["step_method_severity_component"]
        .cummax()
        .astype("int64")
    )
    method_severity = np.maximum(
        df["task_method_severity_component"].to_numpy(dtype="int64"),
        df["step_method_severity_cum_component"].to_numpy(dtype="int64"),
    )

    context_components = ["action", "feedback", "assistant"]
    internal_separators = sum(
        np.maximum(df[f"{kind}_prefix_nonempty_component"].to_numpy(dtype="int64") - 1, 0)
        for kind in context_components
    )
    top_components = (df["task_prompt_chars_component"].to_numpy(dtype="int64") > 0).astype("int64")
    for kind in context_components:
        top_components += (
            df[f"{kind}_prefix_nonempty_component"].to_numpy(dtype="int64") > 0
        ).astype("int64")
    top_level_separators = np.maximum(top_components - 1, 0)
    separator_tokens = (
        (internal_separators + top_level_separators)
        * df["newline_tokens_component"].to_numpy(dtype="int64")
    )

    context_total = df["task_prompt_tokens_component"].to_numpy(dtype="int64")
    for kind in context_components:
        context_total = context_total + df[f"{kind}_prefix_tokens_component"].to_numpy(dtype="int64")
    df["context_call_input_tokens_est"] = (context_total + separator_tokens).astype("int64")

    df["external_input_tokens_cum"] = (
        df["task_prompt_tokens_component"].to_numpy(dtype="int64")
        + df["feedback_prefix_tokens_component"].to_numpy(dtype="int64")
    )
    df["generated_output_tokens_cum"] = (
        df["action_prefix_tokens_component"].to_numpy(dtype="int64")
        + df["thought_prefix_tokens_component"].to_numpy(dtype="int64")
        + df["assistant_prefix_tokens_component"].to_numpy(dtype="int64")
    )
    df["transcript_total_tokens_cum"] = (
        df["external_input_tokens_cum"] + df["generated_output_tokens_cum"]
    )
    df["token_count_method"] = [
        helpers._component_method_from_severity(int(severity)) for severity in method_severity
    ]

    df["future_context_call_input_tokens_saved_if_stop_est"] = (
        df.groupby("traj_id", sort=False)["context_call_input_tokens_est"]
        .transform(lambda s: s.iloc[::-1].cumsum().iloc[::-1] - s)
        .astype("int64")
    )
    baseline_input = (
        df.groupby("traj_id", sort=False)["context_call_input_tokens_est"]
        .sum()
        .rename("baseline_input_tokens_est")
    )
    final = (
        df.groupby("traj_id", sort=False)
        .tail(1)[
            [
                "traj_id",
                "external_input_tokens_cum",
                "generated_output_tokens_cum",
                "transcript_total_tokens_cum",
            ]
        ]
        .rename(
            columns={
                "external_input_tokens_cum": "baseline_external_input_tokens_est",
                "generated_output_tokens_cum": "baseline_output_tokens_est",
                "transcript_total_tokens_cum": "baseline_transcript_total_tokens_est",
            }
        )
    )
    df = df.join(baseline_input, on="traj_id").merge(final, on="traj_id", how="left")
    keep_cols = [
        "traj_id",
        "model_id",
        "prefix_step_idx",
        "context_chars_est",
        "legacy_context_tokens_chars4_est",
        "context_call_input_tokens_est",
        "future_context_call_input_tokens_saved_if_stop_est",
        "baseline_input_tokens_est",
        "external_input_tokens_cum",
        "generated_output_tokens_cum",
        "transcript_total_tokens_cum",
        "baseline_external_input_tokens_est",
        "baseline_output_tokens_est",
        "baseline_transcript_total_tokens_est",
        "token_count_method",
        "tokenizer_family",
        "tokenizer_backend",
        "tokenizer_name",
    ]
    return df[keep_cols].copy()


def _build_source_token_rows(
    prefix_table: Path,
    traj_ids: set[str],
    *,
    dataset_key: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    try:
        import pyarrow.parquet as pq
    except Exception as exc:  # pragma: no cover - dependency guard
        raise RuntimeError("pyarrow is required for streaming tokenizer counts") from exc

    helpers = _load_tokenizer_helpers()
    helpers._configure_tokenizer_cache(TOKENIZER_CACHE_ROOT)
    registry = helpers._TokenizerRegistry(local_files_only=TOKENIZER_LOCAL_FILES_ONLY)
    wanted = set(str(x) for x in traj_ids)
    cols = [
        "traj_id",
        "model_id",
        "prefix_step_idx",
        "task_prompt_chars",
        "prefix_action_chars",
        "prefix_feedback_chars",
        "prefix_thought_chars",
        "prefix_assistant_content_chars",
        "task_prompt_text",
        "last_action_text",
        "last_feedback_text",
        "last_thought_text",
        "last_assistant_content_text",
    ]
    parquet = pq.ParquetFile(prefix_table)
    parts: list[pd.DataFrame] = []
    task_cache: dict[tuple[str, str], tuple[int, int]] = {}
    newline_token_cache: dict[str, int] = {}
    scanned_rows = 0
    matched_rows = 0
    encoded_rows = 0
    next_progress_rows = max(1, TOKEN_PROGRESS_EVERY_ROWS)

    print(
        "[tokenizer-progress] start "
        f"dataset={dataset_key} selected_trajectories={len(wanted)} "
        f"parquet_rows={parquet.metadata.num_rows}",
        flush=True,
    )
    for batch_idx, batch in enumerate(
        parquet.iter_batches(batch_size=max(1, TOKEN_PARQUET_BATCH_SIZE), columns=cols),
        start=1,
    ):
        scanned_rows += int(batch.num_rows)
        part = batch.to_pandas()
        part["traj_id"] = part["traj_id"].astype(str)
        part = part[part["traj_id"].isin(wanted)].copy()
        if part.empty:
            if TOKEN_PROGRESS_EVERY_BATCHES > 0 and batch_idx % TOKEN_PROGRESS_EVERY_BATCHES == 0:
                print(
                    "[tokenizer-progress] scan "
                    f"dataset={dataset_key} batches={batch_idx} parquet_rows={scanned_rows} "
                    f"matched_rows={matched_rows} encoded_rows={encoded_rows}",
                    flush=True,
                )
            continue
        matched_rows += int(len(part))
        out_parts: list[pd.DataFrame] = []
        for model_id, group in part.groupby("model_id", sort=False):
            model_id = str(model_id)
            counter = registry.get(model_id)
            if model_id not in newline_token_cache:
                newline_counts, _ = counter.count_many(
                    ["\n"],
                    max_direct_chars=helpers.DEFAULT_MAX_DIRECT_TOKENIZE_CHARS,
                    exact_chunk_chars=helpers.DEFAULT_EXACT_CHUNK_TOKENIZE_CHARS,
                    sample_long_tokenize_chars=helpers.DEFAULT_SAMPLE_LONG_TOKENIZE_CHARS,
                    sample_chars=helpers.DEFAULT_SAMPLE_CHARS,
                )
                newline_token_cache[model_id] = int(newline_counts[0])

            group = group.reset_index(drop=True)
            n_rows = int(len(group))
            task_tokens = np.zeros(n_rows, dtype="int64")
            task_method_severity = np.zeros(n_rows, dtype="int64")
            step_method_severity = np.zeros(n_rows, dtype="int64")
            component_tokens = {
                kind: np.zeros(n_rows, dtype="int64")
                for kind in ["action", "feedback", "thought", "assistant"]
            }
            component_nonempty = {
                kind: np.zeros(n_rows, dtype="int64")
                for kind in ["action", "feedback", "thought", "assistant"]
            }

            texts: list[str] = []
            jobs: list[tuple[str, int | tuple[str, str]]] = []
            pending_task_rows: dict[tuple[str, str], list[int]] = {}
            for row_idx, row in enumerate(group.itertuples(index=False)):
                traj_id = str(getattr(row, "traj_id"))
                task_key = (model_id, traj_id)
                cached_task = task_cache.get(task_key)
                if cached_task is not None:
                    task_tokens[row_idx] = cached_task[0]
                    task_method_severity[row_idx] = cached_task[1]
                else:
                    if task_key not in pending_task_rows:
                        pending_task_rows[task_key] = []
                        texts.append(_safe_string(getattr(row, "task_prompt_text")))
                        jobs.append(("task", task_key))
                    pending_task_rows[task_key].append(row_idx)

                for kind, col in [
                    ("action", "last_action_text"),
                    ("feedback", "last_feedback_text"),
                    ("thought", "last_thought_text"),
                    ("assistant", "last_assistant_content_text"),
                ]:
                    text = _safe_string(getattr(row, col))
                    if not text:
                        continue
                    component_nonempty[kind][row_idx] = 1
                    texts.append(text)
                    jobs.append((kind, row_idx))

            counts: list[int] = []
            methods: list[str] = []
            for start in range(0, len(texts), max(1, TOKEN_ENCODE_BATCH_SIZE)):
                batch_counts, batch_methods = counter.count_many(
                    texts[start : start + max(1, TOKEN_ENCODE_BATCH_SIZE)],
                    max_direct_chars=helpers.DEFAULT_MAX_DIRECT_TOKENIZE_CHARS,
                    exact_chunk_chars=helpers.DEFAULT_EXACT_CHUNK_TOKENIZE_CHARS,
                    sample_long_tokenize_chars=helpers.DEFAULT_SAMPLE_LONG_TOKENIZE_CHARS,
                    sample_chars=helpers.DEFAULT_SAMPLE_CHARS,
                )
                counts.extend(batch_counts)
                methods.extend(batch_methods)

            for (kind, target), count, method in zip(jobs, counts, methods):
                severity = helpers._component_method_severity(method)
                if kind == "task":
                    task_key = target
                    assert isinstance(task_key, tuple)
                    task_cache[task_key] = (int(count), int(severity))
                    for row_idx in pending_task_rows[task_key]:
                        task_tokens[row_idx] = int(count)
                        task_method_severity[row_idx] = int(severity)
                else:
                    row_idx = int(target)
                    component_tokens[kind][row_idx] = int(count)
                    step_method_severity[row_idx] = max(
                        int(step_method_severity[row_idx]), int(severity)
                    )

            sub = group[["traj_id", "model_id", "prefix_step_idx"]].copy()
            sub["prefix_step_idx"] = pd.to_numeric(
                sub["prefix_step_idx"], errors="raise"
            ).astype("int64")
            for col in [
                "task_prompt_chars",
                "prefix_action_chars",
                "prefix_feedback_chars",
                "prefix_thought_chars",
                "prefix_assistant_content_chars",
            ]:
                sub[col] = pd.to_numeric(group[col], errors="coerce").fillna(0).astype("int64")
            sub["context_chars_est"] = (
                sub["task_prompt_chars"]
                + sub["prefix_action_chars"]
                + sub["prefix_feedback_chars"]
                + sub["prefix_assistant_content_chars"]
            )
            sub["legacy_context_tokens_chars4_est"] = np.ceil(
                sub["context_chars_est"] / LEGACY_CHAR_TOKEN_RATIO
            ).astype("int64")
            sub["task_prompt_chars_component"] = sub["task_prompt_chars"]
            sub["task_prompt_tokens_component"] = task_tokens
            sub["task_method_severity_component"] = task_method_severity
            sub["step_method_severity_component"] = step_method_severity
            sub["newline_tokens_component"] = int(newline_token_cache[model_id])
            for kind in ["action", "feedback", "thought", "assistant"]:
                sub[f"{kind}_step_tokens_component"] = component_tokens[kind]
                sub[f"{kind}_step_nonempty_component"] = component_nonempty[kind]
            sub["tokenizer_family"] = counter.spec.family
            sub["tokenizer_backend"] = counter.spec.backend
            sub["tokenizer_name"] = counter.spec.name
            out_parts.append(sub)

        encoded_part = pd.concat(out_parts, ignore_index=True)
        parts.append(encoded_part)
        encoded_rows += int(len(part))
        if TOKEN_PROGRESS_EVERY_ROWS > 0 and encoded_rows >= next_progress_rows:
            print(
                "[tokenizer-progress] encode "
                f"dataset={dataset_key} batches={batch_idx} parquet_rows={scanned_rows} "
                f"matched_rows={matched_rows} encoded_rows={encoded_rows}",
                flush=True,
            )
            while encoded_rows >= next_progress_rows:
                next_progress_rows += max(1, TOKEN_PROGRESS_EVERY_ROWS)
        if TOKEN_PROGRESS_EVERY_BATCHES > 0 and batch_idx % TOKEN_PROGRESS_EVERY_BATCHES == 0:
            print(
                "[tokenizer-progress] scan "
                f"dataset={dataset_key} batches={batch_idx} parquet_rows={scanned_rows} "
                f"matched_rows={matched_rows} encoded_rows={encoded_rows}",
                flush=True,
            )

    if not parts:
        raise ValueError(f"No token rows matched selected trajectories for {dataset_key}.")
    raw = pd.concat(parts, ignore_index=True)
    raw["traj_id"] = raw["traj_id"].astype(str)
    raw["prefix_step_idx"] = pd.to_numeric(raw["prefix_step_idx"], errors="raise").astype("int64")
    raw = raw.drop_duplicates(["traj_id", "prefix_step_idx"], keep="last")
    missing = wanted - set(raw["traj_id"].astype(str))
    if missing:
        sample = ", ".join(sorted(missing)[:5])
        raise ValueError(f"{dataset_key}: missing token rows for {len(missing)} trajectories, e.g. {sample}")
    token_rows = _add_prefix_source_token_features(raw, helpers)
    manifest = registry.manifest()
    if not manifest.empty:
        manifest.insert(0, "dataset_key", dataset_key)
        manifest["note"] = (
            manifest["note"].astype(str)
            + " Component-sum prefix/source counts; tokenizer boundary merges across components are not modeled exactly."
        )
    print(
        "[tokenizer-progress] done "
        f"dataset={dataset_key} matched_rows={matched_rows} encoded_rows={encoded_rows}",
        flush=True,
    )
    return token_rows, manifest


def _load_or_build_source_token_rows(
    prefix_table: Path,
    traj_ids: set[str],
    *,
    dataset_key: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    cache_path = _token_prefix_cache_path(dataset_key)
    meta = _source_token_cache_meta(prefix_table, traj_ids)
    cached = _load_source_token_cache(cache_path, meta, traj_ids)
    if cached is not None:
        return cached
    token_rows, manifest = _build_source_token_rows(prefix_table, traj_ids, dataset_key=dataset_key)
    _write_source_token_cache(cache_path, token_rows, manifest, meta)
    return token_rows, manifest


def _component_token_savings(
    prefix_table: Path,
    decisions: pd.DataFrame,
    *,
    dataset_key: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if prefix_table is None or not prefix_table.exists():
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    wanted = decisions.loc[decisions["dataset_key"] == dataset_key, "traj_id"].astype(str)
    wanted_set = set(wanted)
    if not wanted_set:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    token_rows, manifest = _load_or_build_source_token_rows(
        prefix_table,
        wanted_set,
        dataset_key=dataset_key,
    )
    if token_rows.empty:
        return pd.DataFrame(), pd.DataFrame(), manifest, pd.DataFrame()

    token_rows = token_rows.copy()
    token_rows["dataset_key"] = dataset_key
    token_rows["traj_id"] = token_rows["traj_id"].astype(str)
    token_rows["prefix_step_idx"] = pd.to_numeric(
        token_rows["prefix_step_idx"], errors="raise"
    ).astype("int64")

    baseline_cols = [
        "traj_id",
        "baseline_input_tokens_est",
        "baseline_external_input_tokens_est",
        "baseline_output_tokens_est",
        "baseline_transcript_total_tokens_est",
    ]
    baseline = token_rows[baseline_cols].drop_duplicates("traj_id", keep="last")
    spend = token_rows.rename(columns={"prefix_step_idx": "decision_step_for_join"})[
        [
            "traj_id",
            "decision_step_for_join",
            "external_input_tokens_cum",
            "generated_output_tokens_cum",
            "transcript_total_tokens_cum",
            "future_context_call_input_tokens_saved_if_stop_est",
        ]
    ]
    dec = decisions[decisions["dataset_key"] == dataset_key].copy()
    dec["traj_id"] = dec["traj_id"].astype(str)
    dec["decision_step_for_join"] = (
        pd.to_numeric(dec["decision_step"], errors="coerce").fillna(-1).clip(lower=0).astype("int64")
    )
    dec = dec.merge(baseline, on="traj_id", how="left").merge(
        spend,
        on=["traj_id", "decision_step_for_join"],
        how="left",
    )
    missing = dec[dec["baseline_input_tokens_est"].isna()]["traj_id"].drop_duplicates()
    if not missing.empty:
        sample = ", ".join(missing.astype(str).head(5))
        raise ValueError(
            f"{dataset_key}: token rows missing for {len(missing)} decision trajectories, e.g. {sample}"
        )

    decided = dec["decided"].fillna(False).astype(bool)
    dec["external_input_tokens_cum"] = np.where(
        decided,
        dec["external_input_tokens_cum"],
        dec["baseline_external_input_tokens_est"],
    )
    dec["generated_output_tokens_cum"] = np.where(
        decided,
        dec["generated_output_tokens_cum"],
        dec["baseline_output_tokens_est"],
    )
    dec["transcript_total_tokens_cum"] = np.where(
        decided,
        dec["transcript_total_tokens_cum"],
        dec["baseline_transcript_total_tokens_est"],
    )
    dec["saved_input_tokens_est"] = np.where(
        decided,
        dec["future_context_call_input_tokens_saved_if_stop_est"].fillna(0),
        0.0,
    )
    dec["saved_external_input_tokens_est"] = np.where(
        decided,
        dec["baseline_external_input_tokens_est"] - dec["external_input_tokens_cum"],
        0.0,
    )
    dec["saved_output_tokens_est"] = np.where(
        decided,
        dec["baseline_output_tokens_est"] - dec["generated_output_tokens_cum"],
        0.0,
    )
    dec["saved_transcript_total_tokens_est"] = np.where(
        decided,
        dec["baseline_transcript_total_tokens_est"] - dec["transcript_total_tokens_cum"],
        0.0,
    )
    for col in [
        "saved_input_tokens_est",
        "saved_external_input_tokens_est",
        "saved_output_tokens_est",
        "saved_transcript_total_tokens_est",
    ]:
        dec[col] = dec[col].clip(lower=0)
    dec["saved_total_api_tokens_est"] = (
        dec["saved_input_tokens_est"] + dec["saved_output_tokens_est"]
    )
    dec["baseline_total_api_tokens_est"] = (
        dec["baseline_input_tokens_est"] + dec["baseline_output_tokens_est"]
    )
    by_agent = (
        dec.groupby(
            ["benchmark", "dataset_key", "threshold", "policy_mode", "agent", "agent_display"],
            as_index=False,
        )
        .agg(
            baseline_input_tokens_est=("baseline_input_tokens_est", "sum"),
            baseline_output_tokens_est=("baseline_output_tokens_est", "sum"),
            baseline_total_api_tokens_est=("baseline_total_api_tokens_est", "sum"),
            saved_input_tokens_est=("saved_input_tokens_est", "sum"),
            saved_output_tokens_est=("saved_output_tokens_est", "sum"),
            saved_total_api_tokens_est=("saved_total_api_tokens_est", "sum"),
            baseline_external_input_tokens_est=("baseline_external_input_tokens_est", "sum"),
            saved_external_input_tokens_est=("saved_external_input_tokens_est", "sum"),
            baseline_transcript_total_tokens_est=("baseline_transcript_total_tokens_est", "sum"),
            saved_transcript_total_tokens_est=("saved_transcript_total_tokens_est", "sum"),
            trajectories=("traj_id", "nunique"),
            decided_trajectories=("decided", "sum"),
        )
    )
    summary = (
        by_agent.groupby(["benchmark", "dataset_key", "threshold", "policy_mode"], as_index=False)
        .agg(
            baseline_input_tokens_est=("baseline_input_tokens_est", "sum"),
            baseline_output_tokens_est=("baseline_output_tokens_est", "sum"),
            baseline_total_api_tokens_est=("baseline_total_api_tokens_est", "sum"),
            saved_input_tokens_est=("saved_input_tokens_est", "sum"),
            saved_output_tokens_est=("saved_output_tokens_est", "sum"),
            saved_total_api_tokens_est=("saved_total_api_tokens_est", "sum"),
            baseline_external_input_tokens_est=("baseline_external_input_tokens_est", "sum"),
            saved_external_input_tokens_est=("saved_external_input_tokens_est", "sum"),
            baseline_transcript_total_tokens_est=("baseline_transcript_total_tokens_est", "sum"),
            saved_transcript_total_tokens_est=("saved_transcript_total_tokens_est", "sum"),
            trajectories=("trajectories", "sum"),
            decided_trajectories=("decided_trajectories", "sum"),
        )
    )
    summary["token_method"] = TOKEN_METHOD
    summary["input_token_save_pct_est"] = (
        summary["saved_input_tokens_est"] * 100.0 / summary["baseline_input_tokens_est"]
    )
    summary["output_token_save_pct_est"] = (
        summary["saved_output_tokens_est"] * 100.0 / summary["baseline_output_tokens_est"]
    )
    summary["total_token_save_pct_est"] = (
        summary["saved_total_api_tokens_est"] * 100.0 / summary["baseline_total_api_tokens_est"]
    )
    summary["external_input_token_save_pct_est"] = (
        summary["saved_external_input_tokens_est"]
        * 100.0
        / summary["baseline_external_input_tokens_est"]
    )
    summary["transcript_total_token_save_pct_est"] = (
        summary["saved_transcript_total_tokens_est"]
        * 100.0
        / summary["baseline_transcript_total_tokens_est"]
    )
    by_agent["input_token_save_pct_est"] = (
        by_agent["saved_input_tokens_est"] * 100.0 / by_agent["baseline_input_tokens_est"]
    )
    by_agent["output_token_save_pct_est"] = (
        by_agent["saved_output_tokens_est"] * 100.0 / by_agent["baseline_output_tokens_est"]
    )
    by_agent["total_token_save_pct_est"] = (
        by_agent["saved_total_api_tokens_est"] * 100.0 / by_agent["baseline_total_api_tokens_est"]
    )
    by_agent["external_input_token_save_pct_est"] = (
        by_agent["saved_external_input_tokens_est"]
        * 100.0
        / by_agent["baseline_external_input_tokens_est"]
    )
    by_agent["transcript_total_token_save_pct_est"] = (
        by_agent["saved_transcript_total_tokens_est"]
        * 100.0
        / by_agent["baseline_transcript_total_tokens_est"]
    )
    by_agent["token_method"] = TOKEN_METHOD

    method_summary = (
        token_rows.groupby(
            [
                "dataset_key",
                "model_id",
                "token_count_method",
                "tokenizer_family",
                "tokenizer_backend",
                "tokenizer_name",
            ],
            dropna=False,
            as_index=False,
        )
        .agg(
            counted_prefixes=("traj_id", "size"),
            trajectories=("traj_id", "nunique"),
            context_call_input_tokens_est=("context_call_input_tokens_est", "sum"),
        )
        .sort_values(["dataset_key", "model_id", "token_count_method"])
        .reset_index(drop=True)
    )
    method_summary["token_method"] = TOKEN_METHOD
    method_summary["prefix_table"] = str(prefix_table)
    return summary, by_agent, manifest, method_summary


def _build_token_tables(
    decisions_095: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summaries = []
    agents = []
    manifests = []
    method_summaries = []
    for config in BENCHES:
        summary, by_agent, manifest, method_summary = _component_token_savings(
            config.prefix_table, decisions_095, dataset_key=config.dataset_key
        )
        if not summary.empty:
            summaries.append(summary)
            agents.append(by_agent)
        if not manifest.empty:
            manifests.append(manifest)
        if not method_summary.empty:
            method_summaries.append(method_summary)
    return (
        pd.concat(summaries, ignore_index=True) if summaries else pd.DataFrame(),
        pd.concat(agents, ignore_index=True) if agents else pd.DataFrame(),
        pd.concat(manifests, ignore_index=True) if manifests else pd.DataFrame(),
        pd.concat(method_summaries, ignore_index=True) if method_summaries else pd.DataFrame(),
    )


def _build_rq1(
    threshold_agg: pd.DataFrame, token_summary: pd.DataFrame
) -> pd.DataFrame:
    main = threshold_agg[np.isclose(threshold_agg["threshold"], 0.95)].copy()
    if "input_token_save_pct_est" not in main.columns:
        token_main = token_summary[np.isclose(token_summary["threshold"], 0.95)].copy()
        main = main.merge(
            token_main[
                [
                    "dataset_key",
                    "threshold",
                    "input_token_save_pct_est",
                    "output_token_save_pct_est",
                    "saved_input_tokens_est",
                    "saved_output_tokens_est",
                    "baseline_input_tokens_est",
                    "baseline_output_tokens_est",
                    "token_method",
                ]
            ],
            on=["dataset_key", "threshold"],
            how="left",
        )
    agent_counts = {b.dataset_key: b.n_test_agents for b in BENCHES}
    main["n_test_agents"] = main["dataset_key"].map(agent_counts)
    cols = [
        "benchmark",
        "n_test_agents",
        "precision_resolved",
        "recall_resolved",
        "f1_resolved",
        "global_step_save_pct",
        "input_token_save_pct_est",
        "output_token_save_pct_est",
        "coverage_pct",
        "decision_accuracy_pct",
        "resolve_change_pp",
        "saved_input_tokens_est",
        "saved_output_tokens_est",
        "baseline_input_tokens_est",
        "baseline_output_tokens_est",
        "token_method",
    ]
    bench_order = {b.benchmark: idx for idx, b in enumerate(BENCHES)}
    out = main[cols].replace([np.inf, -np.inf], np.nan).copy()
    out["_bench_order"] = out["benchmark"].map(bench_order)
    return out.sort_values("_bench_order").drop(columns="_bench_order")


def _attach_token_summary(threshold_agg: pd.DataFrame, token_summary: pd.DataFrame) -> pd.DataFrame:
    token_cols = [
        "dataset_key",
        "threshold",
        "policy_mode",
        "input_token_save_pct_est",
        "output_token_save_pct_est",
        "total_token_save_pct_est",
        "saved_input_tokens_est",
        "saved_output_tokens_est",
        "saved_total_api_tokens_est",
        "baseline_input_tokens_est",
        "baseline_output_tokens_est",
        "baseline_total_api_tokens_est",
        "saved_external_input_tokens_est",
        "baseline_external_input_tokens_est",
        "external_input_token_save_pct_est",
        "saved_transcript_total_tokens_est",
        "baseline_transcript_total_tokens_est",
        "transcript_total_token_save_pct_est",
        "token_method",
    ]
    return threshold_agg.merge(token_summary[token_cols], on=["dataset_key", "threshold", "policy_mode"], how="left")


def _attach_pass1_shift_summary(
    threshold_agg: pd.DataFrame,
    threshold_per_agent: pd.DataFrame,
) -> pd.DataFrame:
    group_cols = ["benchmark", "dataset_key", "preset", "threshold", "policy_mode"]
    per_agent = threshold_per_agent.copy()
    per_agent["resolve_change_pp"] = pd.to_numeric(
        per_agent["resolve_change_pp"], errors="coerce"
    )
    shift = (
        per_agent.groupby(group_cols, as_index=False, dropna=False)
        .agg(
            mean_delta_p1_pp=("resolve_change_pp", "mean"),
            mean_abs_delta_p1_pp=("resolve_change_pp", lambda s: float(s.abs().mean())),
            max_abs_delta_p1_pp=("resolve_change_pp", lambda s: float(s.abs().max())),
        )
    )
    out = threshold_agg.merge(shift, on=group_cols, how="left")
    out["signed_aggregate_delta_p1_pp"] = out["resolve_change_pp"]
    return out


def _build_threshold_compact(threshold_agg: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "benchmark",
        "n_test_agents",
        "threshold",
        "precision_resolved",
        "recall_resolved",
        "f1_resolved",
        "coverage_pct",
        "decision_accuracy_pct",
        "global_step_save_pct",
        "input_token_save_pct_est",
        "output_token_save_pct_est",
        "external_input_token_save_pct_est",
        "resolve_change_pp",
        "signed_aggregate_delta_p1_pp",
        "mean_abs_delta_p1_pp",
        "mean_delta_p1_pp",
        "max_abs_delta_p1_pp",
    ]
    agent_counts = {b.dataset_key: b.n_test_agents for b in BENCHES}
    bench_order = {b.benchmark: idx for idx, b in enumerate(BENCHES)}
    out = threshold_agg.copy()
    out["n_test_agents"] = out["dataset_key"].map(agent_counts)
    out = out[cols].replace([np.inf, -np.inf], np.nan)
    out["_bench_order"] = out["benchmark"].map(bench_order)
    return out.sort_values(["_bench_order", "threshold"]).drop(columns="_bench_order")


def _build_threshold_pass1_shift(
    threshold_agg: pd.DataFrame,
    threshold_per_agent: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_cols = [
        "benchmark",
        "dataset_key",
        "threshold",
        "policy_mode",
        "original_resolve_rate_pct",
        "adjusted_resolve_rate_pct",
        "resolve_change_pp",
        "signed_aggregate_delta_p1_pp",
        "mean_abs_delta_p1_pp",
        "mean_delta_p1_pp",
        "max_abs_delta_p1_pp",
        "coverage_pct",
        "decision_accuracy_pct",
        "global_step_save_pct",
        "input_token_save_pct_est",
        "output_token_save_pct_est",
    ]
    bench_order = {b.benchmark: idx for idx, b in enumerate(BENCHES)}
    summary = threshold_agg[[c for c in summary_cols if c in threshold_agg.columns]].copy()
    summary["_bench_order"] = summary["benchmark"].map(bench_order)
    summary = summary.sort_values(["_bench_order", "threshold"]).drop(columns="_bench_order")

    detail_cols = [
        "benchmark",
        "dataset_key",
        "threshold",
        "policy_mode",
        "agent",
        "agent_display",
        "original_resolve_rate_pct",
        "adjusted_resolve_rate_pct",
        "resolve_change_pp",
        "original_total",
        "coverage_pct",
        "decision_accuracy_pct",
        "global_step_save_pct",
    ]
    detail = threshold_per_agent[[c for c in detail_cols if c in threshold_per_agent.columns]].copy()
    detail["_bench_order"] = detail["benchmark"].map(bench_order)
    detail = detail.sort_values(["_bench_order", "threshold", "agent_display"]).drop(
        columns="_bench_order"
    )
    return summary, detail


def _prefix_table_counts(config: BenchConfig, decisions_all: pd.DataFrame) -> dict[str, Any]:
    if config.prefix_table is None or not config.prefix_table.exists():
        return {
            "benchmark": config.benchmark,
            "dataset_key": config.dataset_key,
            "decomposed_prefixes": 0,
            "collected_trajectories": 0,
            "collected_agents": 0,
            "collected_instances": 0,
            "evaluated_trajectories_locked095": 0,
            "evaluated_agents_locked095": 0,
        }
    try:
        import pyarrow.parquet as pq
    except Exception as exc:  # pragma: no cover - dependency guard
        raise RuntimeError("pyarrow is required to count prefix parquet files") from exc

    parquet = pq.ParquetFile(config.prefix_table)
    available = set(parquet.schema.names)
    cols = [col for col in ["traj_id", "model_id", "instance_id"] if col in available]
    traj_ids: set[str] = set()
    model_ids: set[str] = set()
    instance_ids: set[str] = set()
    for batch in parquet.iter_batches(batch_size=max(1, TOKEN_PARQUET_BATCH_SIZE), columns=cols):
        part = batch.to_pandas()
        if "traj_id" in part.columns:
            traj_ids.update(part["traj_id"].dropna().astype(str).unique())
        if "model_id" in part.columns:
            model_ids.update(part["model_id"].dropna().astype(str).unique())
        if "instance_id" in part.columns:
            instance_ids.update(part["instance_id"].dropna().astype(str).unique())

    locked = decisions_all[
        (decisions_all["dataset_key"].eq(config.dataset_key))
        & np.isclose(decisions_all["threshold"], 0.95)
    ]
    return {
        "benchmark": config.benchmark,
        "dataset_key": config.dataset_key,
        "decomposed_prefixes": int(parquet.metadata.num_rows),
        "collected_trajectories": int(len(traj_ids)),
        "collected_agents": int(len(model_ids)),
        "collected_instances": int(len(instance_ids)),
        "evaluated_trajectories_locked095": int(locked["traj_id"].astype(str).nunique()),
        "evaluated_agents_locked095": int(locked["agent"].astype(str).nunique()),
    }


def _build_trajectory_prefix_counts(decisions_all: pd.DataFrame) -> pd.DataFrame:
    rows = [_prefix_table_counts(config, decisions_all) for config in BENCHES]
    out = pd.DataFrame(rows)
    total = {
        "benchmark": "All benchmarks",
        "dataset_key": "ALL",
        "decomposed_prefixes": int(out["decomposed_prefixes"].sum()),
        "collected_trajectories": int(out["collected_trajectories"].sum()),
        "collected_agents": int(out["collected_agents"].sum()),
        "collected_instances": int(out["collected_instances"].sum()),
        "evaluated_trajectories_locked095": int(out["evaluated_trajectories_locked095"].sum()),
        "evaluated_agents_locked095": int(out["evaluated_agents_locked095"].sum()),
    }
    return pd.concat([out, pd.DataFrame([total])], ignore_index=True)


def _build_rq2(
    ranked_by_bench: dict[str, pd.DataFrame], token_agent: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    all_rows = []
    summary_rows = []
    token_agent_095 = token_agent[np.isclose(token_agent["threshold"], 0.95)].copy()
    for benchmark, ranked in ranked_by_bench.items():
        merged = ranked.merge(
            token_agent_095[
                [
                    "dataset_key",
                    "agent",
                    "saved_input_tokens_est",
                    "saved_output_tokens_est",
                    "baseline_input_tokens_est",
                    "baseline_output_tokens_est",
                    "input_token_save_pct_est",
                    "output_token_save_pct_est",
                ]
            ],
            on=["dataset_key", "agent"],
            how="left",
        )
        all_rows.append(merged)
        rho = _spearman_from_ranks(merged)
        signed_aggregate = _pct(
            merged["adjusted_resolved"].sum() - merged["original_resolved"].sum(),
            merged["original_total"].sum(),
        )
        summary_rows.append(
            {
                "benchmark": benchmark,
                "dataset_key": merged["dataset_key"].iloc[0],
                "n_agents": int(len(merged)),
                "signed_aggregate_delta_p1_pp": signed_aggregate,
                "spearman_rho_full_vs_earlyeval_rank": rho,
                "mean_abs_delta_p1_pp": float(merged["resolve_change_pp"].abs().mean()),
                "sum_saved_input_tokens_est": float(
                    merged["saved_input_tokens_est"].fillna(0).sum()
                ),
                "sum_saved_output_tokens_est": float(
                    merged["saved_output_tokens_est"].fillna(0).sum()
                ),
            }
        )
    per_agent_all = pd.concat(all_rows, ignore_index=True)
    top10 = (
        per_agent_all.sort_values(["benchmark", "original_rank", "agent_display"])
        .groupby("benchmark", as_index=False, group_keys=False)
        .head(10)
        .copy()
    )
    return top10, per_agent_all, pd.DataFrame(summary_rows)


def _summarize_ablation_rows(rows: list[pd.DataFrame]) -> pd.DataFrame:
    df = pd.concat(rows, ignore_index=True)
    group_cols = ["profile", "predictor", "score_mode", "policy_mode", "success_thr", "failure_thr", "min_step", "consecutive"]
    out_rows = []
    for keys, group in df.groupby(group_cols, dropna=False):
        rec = dict(zip(group_cols, keys))
        total = int(group["original_total"].sum())
        original_resolved = int(group["original_resolved"].sum())
        fn = int(group["false_negatives"].sum())
        fp = int(group["false_positives"].sum())
        tp = int(group["true_positives"].sum())
        tn = int(group["true_negatives"].sum())
        n_decided = int(group["n_decided"].sum())
        total_saved_steps = int(group["total_saved_steps"].sum())
        total_steps = int(group["total_steps"].sum())
        adjusted = original_resolved - fn + fp
        rec.update(
            {
                "n_folds": int(len(group)),
                "n_trajectories": total,
                "original_resolved": original_resolved,
                "adjusted_resolved": adjusted,
                "false_negatives": fn,
                "false_positives": fp,
                "true_negatives": tn,
                "true_positives": tp,
                "decided_trajectories": n_decided,
                "coverage_pct": _pct(n_decided, total),
                "decision_accuracy_pct": _pct(tp + tn, n_decided),
                "global_step_save_pct": _pct(total_saved_steps, total_steps),
                "resolve_change_pp": _pct(adjusted - original_resolved, total),
                "mean_abs_resolve_change_pp": float(group["resolve_change_pp"].abs().mean())
                if "resolve_change_pp" in group
                else float("nan"),
            }
        )
        out_rows.append(rec)
    return pd.DataFrame(out_rows)


def _build_feature_group_locked095() -> tuple[pd.DataFrame, pd.DataFrame]:
    base = EXP / "ablations/sweverify/sweverify_ablation_feature_groups_full16"
    configs = [
        (
            "feature_groups",
            base / "feature_groups/folds",
            [
                "I_LightGBM_Dense_AF",
                "J_LightGBM_Dense_AF_Thought",
                "Abl_NoTaskPromptTfidf_LightGBM",
                "Abl_NoTaskSignal_LightGBM",
                "Abl_NoGoldAnswer_LightGBM",
                "Abl_NoTaskSignal_NoGoldAnswer_LightGBM",
            ],
        ),
        (
            "component_with_model_id",
            base / "component_with_model_id/folds",
            ["I_LightGBM_Dense_AF"],
        ),
    ]
    aggregate_rows = []
    per_fold_rows = []
    for profile, fold_root, predictors in configs:
        for predictor in predictors:
            for fold_dir in sorted(fold_root.glob("*")):
                pred_path = fold_dir / "test_predictions_safe_stop.parquet"
                if not pred_path.exists():
                    continue
                config = BenchConfig(
                    benchmark="SWE-bench Verified",
                    dataset_key="sweverify",
                    n_test_agents=16,
                    run_root=fold_root,
                    fold_glob=fold_dir.name,
                    predictor=predictor,
                    preset=profile,
                )
                records = _load_prediction_records(config)
                agg, _, _ = _evaluate_records(records, threshold=0.95)
                row = dict(agg)
                row.update(
                    {
                        "profile": profile,
                        "predictor": predictor,
                        "score_mode": "calibrated",
                        "policy_mode": "dual",
                        "success_thr": 0.95,
                        "failure_thr": 0.95,
                        "min_step": 0,
                        "consecutive": 1,
                        "fold_id": fold_dir.name,
                    }
                )
                per_fold_rows.append(pd.DataFrame([row]))
    if per_fold_rows:
        per_fold = pd.concat(per_fold_rows, ignore_index=True)
        aggregate = _summarize_ablation_rows([per_fold])
    else:
        per_fold = pd.DataFrame()
        aggregate = pd.DataFrame()
    return aggregate, per_fold


def _build_rq3() -> pd.DataFrame:
    fg_locked, fg_per_fold = _build_feature_group_locked095()
    if not fg_locked.empty:
        fg_locked.to_csv(SUPPORTING / "rq3_feature_groups_full16_locked095_per_fold_aggregate.csv", index=False)
        fg_per_fold.to_csv(SUPPORTING / "rq3_feature_groups_full16_locked095_per_fold.csv", index=False)
        fg_locked.to_csv(PAPER_DATA / "table_ablation_feature_groups_full16_locked095.csv", index=False)
    pieces = []
    for name in [
        "table_ablation_default_reg_full16_locked095.csv",
        "table_ablation_fine_grained_full16_locked095.csv",
    ]:
        p = PAPER_DATA / name
        if p.exists():
            df = pd.read_csv(p)
            pieces.append(df)
    if not fg_locked.empty:
        pieces.append(fg_locked)
    if pieces:
        out = pd.concat(pieces, ignore_index=True, sort=False)
    else:
        out = pd.DataFrame()
    if not out.empty:
        order = [
            "profile",
            "predictor",
            "score_mode",
            "policy_mode",
            "success_thr",
            "failure_thr",
            "n_folds",
            "n_trajectories",
            "coverage_pct",
            "decision_accuracy_pct",
            "global_step_save_pct",
            "resolve_change_pp",
            "mean_abs_resolve_change_pp",
        ]
        cols = [c for c in order if c in out.columns] + [c for c in out.columns if c not in order]
        out = out[cols]
    return out


def _build_rq3_paper(rq3: pd.DataFrame) -> pd.DataFrame:
    if rq3.empty:
        return rq3
    labels = {
        ("feature_groups", "I_LightGBM_Dense_AF"): "Full (feature groups)",
        ("feature_groups", "Abl_NoTaskPromptTfidf_LightGBM"): "- Task-prompt TF-IDF",
        ("feature_groups", "Abl_NoTaskSignal_LightGBM"): "- Task-signal",
        ("feature_groups", "Abl_NoGoldAnswer_LightGBM"): "- Gold-answer",
        ("feature_groups", "Abl_NoTaskSignal_NoGoldAnswer_LightGBM"): "- Task-signal & Gold-answer",
        ("component_with_model_id", "I_LightGBM_Dense_AF"): "+ Model-id",
        ("feature_groups", "J_LightGBM_Dense_AF_Thought"): "+ Thought",
        ("fine_grained_process", "Abl_NoFeedback_LightGBM"): "- Feedback",
        ("fine_grained_process", "Abl_NoAction_LightGBM"): "- Action",
        ("fine_grained_process", "Abl_NoThought_LightGBM"): "- Thought",
        ("fine_grained_process", "Abl_ProcessOnly_LightGBM"): "Process-only",
        ("component_default_reg", "I_LightGBM_Dense_AF"): "Full (default-reg prior)",
    }
    order = {key: idx for idx, key in enumerate(labels)}
    out = rq3.copy()
    out["variant"] = [
        labels.get((str(row.profile), str(row.predictor)), str(row.predictor))
        for row in out.itertuples(index=False)
    ]
    out["_order"] = [
        order.get((str(row.profile), str(row.predictor)), len(order))
        for row in out.itertuples(index=False)
    ]
    cols = [
        "variant",
        "profile",
        "predictor",
        "coverage_pct",
        "decision_accuracy_pct",
        "global_step_save_pct",
        "resolve_change_pp",
        "mean_abs_resolve_change_pp",
        "n_folds",
        "n_trajectories",
    ]
    return out.sort_values("_order")[[c for c in cols if c in out.columns]]


def _price_template(per_agent_all: pd.DataFrame) -> pd.DataFrame:
    rows = (
        per_agent_all[["benchmark", "dataset_key", "agent", "agent_display"]]
        .drop_duplicates()
        .sort_values(["benchmark", "agent_display"])
        .reset_index(drop=True)
    )
    rows["input_usd_per_1m_tokens"] = np.nan
    rows["output_usd_per_1m_tokens"] = np.nan
    rows["price_source"] = ""
    return rows


def _format_signed(x: float, digits: int = 2) -> str:
    if pd.isna(x):
        return "??"
    return f"{float(x):+.{digits}f}"


def _format_rank_delta(x: Any) -> str:
    if pd.isna(x):
        return "??"
    x = int(x)
    if x == 0:
        return "0"
    if x > 0:
        return f"\\rankup{{{x}}}"
    return f"\\rankdn{{{abs(x)}}}"


def _write_latex(
    rq1: pd.DataFrame,
    threshold_compact: pd.DataFrame,
    rq2_top10: pd.DataFrame,
    rq2_summary: pd.DataFrame,
    rq3: pd.DataFrame,
) -> None:
    def fmt_pct(value: float) -> str:
        if pd.isna(value):
            return "--"
        return f"{float(value):.2f}"

    mean_abs_note = " / ".join(
        f"{row.benchmark}: {float(row.mean_abs_delta_p1_pp):.2f} pp"
        for row in rq2_summary.itertuples(index=False)
    )
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{RQ1: EarlyEval's decision quality and resource savings on each benchmark at the fixed calibrated operating point ($s=f=0.95$). Context-token savings are API-style input/context-call savings for skipped future calls; output-token savings are skipped model-generated trajectory tokens. Token counts use model-specific local tokenizers when available and documented close-family proxy tokenizers otherwise.}",
        r"\label{tab:rq1}",
        r"\small",
        r"\setlength{\tabcolsep}{4.5pt}",
        r"\begin{tabular}{l c c c c c c c}",
        r"\toprule",
        r"\textbf{Benchmark} & \textbf{n\_test\_agents} & \textbf{precision} & \textbf{recall} & \textbf{F1} & \textbf{\%saved\_steps} & \textbf{\%saved\_context\_tokens} & \textbf{\%saved\_output\_tokens} \\",
        r"\midrule",
    ]
    order = ["SWE-bench Verified", "TerminalBench", "Toolathlon"]
    rq1_ordered = rq1.set_index("benchmark").loc[order].reset_index()
    for row in rq1_ordered.itertuples(index=False):
        lines.append(
            f"{row.benchmark} & {int(row.n_test_agents)} & "
            f"{fmt_pct(row.precision_resolved)} & {fmt_pct(row.recall_resolved)} & {fmt_pct(row.f1_resolved)} & "
            f"{fmt_pct(row.global_step_save_pct)} & {fmt_pct(row.input_token_save_pct_est)} & "
            f"{fmt_pct(row.output_token_save_pct_est)} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""]

    lines += [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{RQ1 threshold sweep: overall decision quality, savings, and Pass@1 shifts across fixed symmetric calibrated thresholds. $\Delta$P@1 is the signed aggregate resolve-rate change (EarlyEval minus full run, pp); mean $|\Delta|$P@1 is the mean absolute per-agent resolve-rate change.}",
        r"\label{tab:rq1_threshold_sweep}",
        r"\small",
        r"\setlength{\tabcolsep}{4.5pt}",
        r"\begin{tabular}{l c c c c c c c}",
        r"\toprule",
        r"Benchmark & Thr & Acc & \%Steps & \%In & \%Out & $\Delta$P@1 & mean $|\Delta|$P@1 \\",
        r"\midrule",
    ]
    for benchmark in order:
        sub = threshold_compact[threshold_compact["benchmark"].eq(benchmark)].copy()
        if sub.empty:
            continue
        first = True
        for row in sub.sort_values("threshold").itertuples(index=False):
            bench_cell = benchmark if first else ""
            first = False
            lines.append(
                f"{bench_cell} & {float(row.threshold):.2f} & "
                f"{fmt_pct(row.decision_accuracy_pct)} & {fmt_pct(row.global_step_save_pct)} & "
                f"{fmt_pct(row.input_token_save_pct_est)} & {fmt_pct(row.output_token_save_pct_est)} & "
                f"${_format_signed(row.signed_aggregate_delta_p1_pp)}$ & "
                f"{fmt_pct(row.mean_abs_delta_p1_pp)} \\\\"
            )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""]

    bench_order = ["SWE-bench Verified", "TerminalBench", "Toolathlon"]
    top_by_bench = {
        b: rq2_top10[rq2_top10["benchmark"] == b].reset_index(drop=True) for b in bench_order
    }
    summary_by_bench = {
        row.benchmark: row for row in rq2_summary.itertuples(index=False)
    }
    lines += [
        r"\begin{table*}[t]",
        r"\centering",
        rf"\caption{{RQ2: Per-agent faithfulness and savings under EarlyEval ($s=f=0.95$, calibrated). Each block lists its top-10 agents by full-run Pass@1; summary rows report all-agent signed aggregate and Spearman rank correlation. Mean absolute $\Delta$P@1: {mean_abs_note}. Saved\$ is left blank until model-specific input/output prices are filled in \texttt{{model_price_template.csv}}.}}",
        r"\label{tab:rq2}",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{4pt}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{l c c r @{\hskip 12pt} l c c r @{\hskip 12pt} l c c r}",
        r"\toprule",
        r"\multicolumn{4}{c}{\textbf{SWE-bench Verified}} & \multicolumn{4}{c}{\textbf{TerminalBench}} & \multicolumn{4}{c}{\textbf{Toolathlon}} \\",
        r"\cmidrule(lr){1-4}\cmidrule(lr){5-8}\cmidrule(lr){9-12}",
        r"Agent & $\Delta$P@1 & $\Delta$Rk & Saved\$ & Agent & $\Delta$P@1 & $\Delta$Rk & Saved\$ & Agent & $\Delta$P@1 & $\Delta$Rk & Saved\$ \\",
        r"\midrule",
    ]
    for idx in range(10):
        cells = []
        for b in bench_order:
            sub = top_by_bench[b]
            if idx < len(sub):
                row = sub.iloc[idx]
                cells.extend(
                    [
                        str(row["agent_display"]),
                        f"${_format_signed(row['resolve_change_pp'])}$",
                        _format_rank_delta(row["rank_change_positive_is_up"]),
                        "??",
                    ]
                )
            else:
                cells.extend(["", "", "", ""])
        lines.append(" & ".join(cells) + r" \\")
    lines.append(r"\cmidrule(lr){1-4}\cmidrule(lr){5-8}\cmidrule(lr){9-12}")
    cells = []
    for b in bench_order:
        summ = summary_by_bench[b]
        n_agents = int(summ.n_agents)
        # Use the true all-agent aggregate from rq1 for the table value when possible.
        cells.extend(
            [
                f"All {n_agents}",
                f"${_format_signed(float(summ.signed_aggregate_delta_p1_pp))}$",
                f"$\\rho\\,{float(summ.spearman_rho_full_vs_earlyeval_rank):.3f}$",
                "??",
            ]
        )
    lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}%", r"}", r"\end{table*}", ""]

    if not rq2_summary.empty:
        lines += [
            r"% Notes:",
            r"% - Saved\$ can be filled after model_price_template.csv has input/output prices.",
            r"% - Toolathlon has no positive/resolved stops at s=f=0.95, so resolved precision/F1 are undefined in RQ1.",
            "",
        ]

    if not rq3.empty:
        def rq3_label(profile: str, predictor: str) -> str:
            mapping = {
                ("feature_groups", "I_LightGBM_Dense_AF"): "Full (feature groups)",
                ("feature_groups", "J_LightGBM_Dense_AF_Thought"): "+ Thought",
                ("feature_groups", "Abl_NoTaskPromptTfidf_LightGBM"): "- Task-prompt TF-IDF",
                ("feature_groups", "Abl_NoTaskSignal_LightGBM"): "- Task-signal",
                ("feature_groups", "Abl_NoGoldAnswer_LightGBM"): "- Gold-answer",
                ("feature_groups", "Abl_NoTaskSignal_NoGoldAnswer_LightGBM"): "- Task-signal \\& Gold-answer",
                ("component_with_model_id", "I_LightGBM_Dense_AF"): "+ Model-id",
                ("component_default_reg", "I_LightGBM_Dense_AF"): "Full (default-reg prior)",
                ("fine_grained_process", "Abl_NoFeedback_LightGBM"): "- Feedback",
                ("fine_grained_process", "Abl_NoAction_LightGBM"): "- Action",
                ("fine_grained_process", "Abl_NoThought_LightGBM"): "- Thought",
                ("fine_grained_process", "Abl_ProcessOnly_LightGBM"): "Process-only",
            }
            return mapping.get((str(profile), str(predictor)), str(predictor).replace("_", "\\_"))

        order = [
            ("feature_groups", "I_LightGBM_Dense_AF"),
            ("feature_groups", "Abl_NoTaskPromptTfidf_LightGBM"),
            ("feature_groups", "Abl_NoTaskSignal_LightGBM"),
            ("feature_groups", "Abl_NoGoldAnswer_LightGBM"),
            ("feature_groups", "Abl_NoTaskSignal_NoGoldAnswer_LightGBM"),
            ("component_with_model_id", "I_LightGBM_Dense_AF"),
            ("feature_groups", "J_LightGBM_Dense_AF_Thought"),
            ("fine_grained_process", "Abl_NoFeedback_LightGBM"),
            ("fine_grained_process", "Abl_NoAction_LightGBM"),
            ("fine_grained_process", "Abl_NoThought_LightGBM"),
            ("fine_grained_process", "Abl_ProcessOnly_LightGBM"),
            ("component_default_reg", "I_LightGBM_Dense_AF"),
        ]
        keyed = {
            (str(row.profile), str(row.predictor)): row for row in rq3.itertuples(index=False)
        }
        lines += [
            r"\begin{table}[t]",
            r"\centering",
            r"\caption{RQ3: SWE-bench Verified full-16 ablations replayed at the locked calibrated operating point ($s=f=0.95$). Feature-group rows are the newly completed full-16 runs; default/fine-grained rows are retained as supporting locked-point comparisons.}",
            r"\label{tab:rq3_ablation}",
            r"\small",
            r"\setlength{\tabcolsep}{4pt}",
            r"\begin{tabular}{l c c c c}",
            r"\toprule",
            r"Variant & Coverage & Accuracy & Saved Steps & $\Delta$P@1 \\",
            r"\midrule",
        ]
        for key in order:
            row = keyed.get(key)
            if row is None:
                continue
            lines.append(
                f"{rq3_label(row.profile, row.predictor)} & "
                f"{fmt_pct(row.coverage_pct)} & {fmt_pct(row.decision_accuracy_pct)} & "
                f"{fmt_pct(row.global_step_save_pct)} & ${_format_signed(row.resolve_change_pp)}$ \\\\"
            )
        lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]

    (OUT / "tables_latex_draft.tex").write_text("\n".join(lines), encoding="utf-8")


def _write_readme(
    rq1: pd.DataFrame,
    threshold_agg: pd.DataFrame,
    rq2_summary: pd.DataFrame,
    rq3: pd.DataFrame,
    trajectory_prefix_counts: pd.DataFrame,
) -> None:
    txt = f"""# RQ Tables Reorganization Bundle

Generated on 2026-06-23.

This folder collects the paper-facing RQ1/RQ2/RQ3 tables requested for the
current draft. All quality/ranking numbers are post-hoc replays on the
completed held-out prediction parquet files; no models are trained here.

## Main choices

- Operating point for RQ1/RQ2/RQ3 main rows: calibrated dual-head
  `s=f=0.95`, `min_step=0`, `consecutive=1`.
- SWE-bench Verified uses the full-16 `lightgbm_main` held-out folds.
- TerminalBench and Toolathlon use the leave-one-agent `rich_af_gold`
  robustness folds, matching the stronger/current RQ2 paper discussion.
- `threshold_sweep_all_benchmarks.csv` uses the same held-out fold set for
  each benchmark and replays fixed symmetric thresholds.
- Main token split uses
  `{TOKEN_METHOD}`: saved input tokens are API-style context-call tokens for
  skipped future model calls; saved output tokens are skipped model-generated
  trajectory tokens. Counts use model-specific local tokenizers when available
  and documented close-family proxy tokenizers otherwise.
- The source-split columns (`saved_external_input_tokens_est`,
  `saved_output_tokens_est`) also show how much of the truncated unique
  trajectory came from external/task/tool feedback vs model-generated text.
- Saved dollar values are intentionally not filled until a model price table is
  provided. Fill `model_price_template.csv` with input/output USD per million
  tokens, then use `rq2_per_agent_all.csv` saved token columns to compute cost.

## Files

- `rq1_main.csv`: RQ1 table values, including precision/recall/F1 for resolved
  stops and global saved steps/input/output token ratios.
- `threshold_sweep_all_benchmarks.csv`: full fixed-threshold tradeoff for all
  three benchmarks.
- `rq1_threshold_sweep_compact.csv`: paper-friendly threshold sweep with the
  main quality/saving columns, including signed aggregate Delta P@1 and
  mean absolute per-agent Delta P@1 across every threshold.
- `rq1_threshold_pass1_shift_summary.csv`: threshold-level Pass@1 shift summary
  for all three benchmarks.
- `supporting/rq1_threshold_pass1_shift_by_agent.csv`: per-agent Pass@1 shift
  detail used to compute the threshold-level mean absolute Delta P@1.
- `rq2_top10.csv`: top-10 agents per benchmark by full-run Pass@1, with
  delta P@1 and rank change.
- `rq2_per_agent_all.csv`: all per-agent rows for SWE/TB/Toolathlon.
- `rq2_summary.csv`: all-agent Spearman rho and mean absolute delta P@1.
- `rq3_ablation_locked095.csv`: locked `s=f=0.95` RQ3 ablations, combining
  existing default/fine-grained locked tables and newly replayed feature-group
  full-16 rows.
- `rq3_ablation_locked095_paper.csv`: cleaner paper-facing RQ3 ablation table
  with only the main reporting columns.
- `token_input_output_summary.csv`: aggregate input/context-call and output
  token savings.
- `token_input_output_by_agent.csv`: per-agent input/context-call and output
  token savings.
- `token_source_split_summary.csv` and `token_source_split_by_agent.csv`:
  backward-compatible copies with the same columns.
- `trajectory_prefix_counts.csv`: collected trajectory and decomposed prefix
  counts for SWE-bench Verified, TerminalBench, and Toolathlon.
- `model_price_template.csv`: price table to fill for Saved$.
- `tables_latex_draft.tex`: LaTeX draft for the RQ1/RQ2/RQ3 tables, including
  the threshold-level Pass@1 shift table.
- `supporting/tokenizer_manifest_all_benchmarks.csv`: exact tokenizer/proxy
  mapping used for each model id.
- `supporting/token_count_method_summary_all_benchmarks.csv`: count method
  summary, including direct/chunked/sampled long-string handling.
- `supporting/`: per-fold RQ3 replay, tokenizer audit files, and supporting
  checks.

## Quick headline snapshot

RQ1 rows:

```text
{_round_df(rq1, 2).to_string(index=False)}
```

Threshold sweep rows:

```text
{_round_df(threshold_agg[['benchmark','threshold','coverage_pct','decision_accuracy_pct','global_step_save_pct','input_token_save_pct_est','output_token_save_pct_est','resolve_change_pp','mean_abs_delta_p1_pp']], 2).to_string(index=False)}
```

RQ2 summary:

```text
{_round_df(rq2_summary, 4).to_string(index=False)}
```

Collected trajectory/prefix counts:

```text
{_round_df(trajectory_prefix_counts, 0).to_string(index=False)}
```

RQ3 locked 0.95 rows:

```text
{_round_df(rq3[['profile','predictor','coverage_pct','decision_accuracy_pct','global_step_save_pct','resolve_change_pp']] if not rq3.empty else rq3, 2).to_string(index=False)}
```

## Regenerate

```bash
cd "$SWEBENCH_PACKAGE_ROOT"
python paper/icse_submission_draft/rq_tables_reorg_20260623/build_rq_tables_bundle.py
```
"""
    # Keep README.md as the hand-maintained usage guide. The generated snapshot
    # is still useful for quick numeric diffs after reruns.
    (OUT / "README.generated.md").write_text(txt, encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    SUPPORTING.mkdir(parents=True, exist_ok=True)

    threshold_agg, threshold_per_agent, decisions_095, ranked_by_bench = _build_decisions_all()
    token_summary, token_agent, tokenizer_manifest, token_method_summary = _build_token_tables(
        decisions_095
    )
    threshold_agg = _attach_token_summary(threshold_agg, token_summary)
    threshold_agg = _attach_pass1_shift_summary(threshold_agg, threshold_per_agent)
    threshold_compact = _build_threshold_compact(threshold_agg)
    threshold_pass1_summary, threshold_pass1_by_agent = _build_threshold_pass1_shift(
        threshold_agg,
        threshold_per_agent,
    )
    trajectory_prefix_counts = _build_trajectory_prefix_counts(decisions_095)
    rq1 = _build_rq1(threshold_agg, token_summary)
    rq2_top10, rq2_per_agent_all, rq2_summary = _build_rq2(ranked_by_bench, token_agent)
    rq3 = _build_rq3()
    rq3_paper = _build_rq3_paper(rq3)

    threshold_agg = _round_df(threshold_agg)
    threshold_compact = _round_df(threshold_compact)
    threshold_pass1_summary = _round_df(threshold_pass1_summary)
    threshold_pass1_by_agent = _round_df(threshold_pass1_by_agent)
    threshold_per_agent = _round_df(threshold_per_agent)
    decisions_095 = _round_df(decisions_095)
    token_summary = _round_df(token_summary)
    token_agent = _round_df(token_agent)
    tokenizer_manifest = _round_df(tokenizer_manifest)
    token_method_summary = _round_df(token_method_summary)
    trajectory_prefix_counts = _round_df(trajectory_prefix_counts)
    rq1 = _round_df(rq1)
    rq2_top10 = _round_df(rq2_top10)
    rq2_per_agent_all = _round_df(rq2_per_agent_all)
    rq2_summary = _round_df(rq2_summary)
    rq3 = _round_df(rq3)
    rq3_paper = _round_df(rq3_paper)

    threshold_agg.to_csv(OUT / "threshold_sweep_all_benchmarks.csv", index=False)
    threshold_compact.to_csv(OUT / "rq1_threshold_sweep_compact.csv", index=False)
    threshold_pass1_summary.to_csv(OUT / "rq1_threshold_pass1_shift_summary.csv", index=False)
    threshold_pass1_by_agent.to_csv(
        SUPPORTING / "rq1_threshold_pass1_shift_by_agent.csv",
        index=False,
    )
    threshold_per_agent.to_csv(SUPPORTING / "threshold_sweep_per_agent_all_benchmarks.csv", index=False)
    decisions_095.to_csv(SUPPORTING / "locked095_decisions_all_benchmarks.csv", index=False)
    token_summary.to_csv(OUT / "token_input_output_summary.csv", index=False)
    token_agent.to_csv(OUT / "token_input_output_by_agent.csv", index=False)
    token_summary.to_csv(OUT / "token_source_split_summary.csv", index=False)
    token_agent.to_csv(OUT / "token_source_split_by_agent.csv", index=False)
    tokenizer_manifest.to_csv(SUPPORTING / "tokenizer_manifest_all_benchmarks.csv", index=False)
    token_method_summary.to_csv(
        SUPPORTING / "token_count_method_summary_all_benchmarks.csv",
        index=False,
    )
    trajectory_prefix_counts.to_csv(OUT / "trajectory_prefix_counts.csv", index=False)
    trajectory_prefix_counts.to_csv(
        SUPPORTING / "trajectory_prefix_counts.csv",
        index=False,
    )
    rq1.to_csv(OUT / "rq1_main.csv", index=False)
    rq2_top10.to_csv(OUT / "rq2_top10.csv", index=False)
    rq2_per_agent_all.to_csv(OUT / "rq2_per_agent_all.csv", index=False)
    rq2_summary.to_csv(OUT / "rq2_summary.csv", index=False)
    rq3.to_csv(OUT / "rq3_ablation_locked095.csv", index=False)
    rq3_paper.to_csv(OUT / "rq3_ablation_locked095_paper.csv", index=False)
    _price_template(rq2_per_agent_all).to_csv(OUT / "model_price_template.csv", index=False)

    swe_token = (
        EXP
        / "lightgbm_main/internal_review_swe16/fixed_symmetric_threshold_token_savings_model_tokenizers.csv"
    )
    if swe_token.exists():
        pd.read_csv(swe_token).to_csv(
            SUPPORTING / "swe_fixed_threshold_model_tokenizer_context_call_savings.csv",
            index=False,
        )

    _write_latex(rq1, threshold_compact, rq2_top10, rq2_summary, rq3)
    _write_readme(rq1, threshold_agg, rq2_summary, rq3, trajectory_prefix_counts)

    manifest_rows = []
    for p in sorted(OUT.rglob("*")):
        if p.is_file() and p.name != "manifest.json":
            manifest_rows.append(
                {
                    "path": str(p.relative_to(OUT)),
                    "bytes": p.stat().st_size,
                }
            )
    (OUT / "manifest.json").write_text(
        json.dumps({"files": manifest_rows}, indent=2), encoding="utf-8"
    )
    print(f"Wrote bundle to {OUT}")


if __name__ == "__main__":
    main()
