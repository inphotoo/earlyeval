from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from final3.core.contracts import PolicySpec
from final3.policies.safe_stop import apply_policy


DEFAULT_RUN_DIR = Path("paper/experiments/rq_final_lightgbm_17/lightgbm_main")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build latency/cost proxy audit for completed main LightGBM folds.")
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--benchmark-repeats", type=int, default=5)
    parser.add_argument("--price-per-million-input-tokens", type=float, default=None)
    parser.add_argument("--price-per-million-output-tokens", type=float, default=None)
    parser.add_argument("--price-per-million-total-tokens", type=float, default=None)
    parser.add_argument("--no-benchmark", action="store_true")
    return parser.parse_args()


def _fmt(value: Any, digits: int = 2) -> str:
    try:
        numeric = float(value)
    except Exception:
        return str(value)
    if math.isnan(numeric):
        return "n/a"
    return f"{numeric:.{digits}f}"


def _main_policy() -> PolicySpec:
    return PolicySpec(
        name="main_calibrated_dual_s095_f095_min0_k1",
        predictor="I_LightGBM_Dense_AF",
        score_mode="calibrated",
        policy_mode="dual",
        success_thr=0.95,
        failure_thr=0.95,
        min_step=0,
        consecutive=1,
    )


def _safe_sum(frame: pd.DataFrame, column: str) -> float:
    if column not in frame.columns:
        return float("nan")
    return float(pd.to_numeric(frame[column], errors="coerce").fillna(0).sum())


def _load_selected(run_dir: Path) -> pd.DataFrame:
    path = run_dir / "summary" / "per_fold_test_selected.csv"
    if path.exists():
        return pd.read_csv(path)
    rows = []
    for fold in sorted((run_dir / "folds").glob("*/_SUCCESS")):
        csv_path = fold.parent / "safe_stop_test_selected.csv"
        if csv_path.exists():
            part = pd.read_csv(csv_path)
            part.insert(0, "fold_id", fold.parent.name)
            rows.append(part)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _aggregate_step_summary(selected: pd.DataFrame) -> pd.DataFrame:
    total_steps = _safe_sum(selected, "total_steps")
    saved_steps = _safe_sum(selected, "total_saved_steps")
    original_total = _safe_sum(selected, "original_total")
    n_decided = _safe_sum(selected, "n_decided")
    tp = _safe_sum(selected, "true_positives")
    tn = _safe_sum(selected, "true_negatives")
    fp = _safe_sum(selected, "false_positives")
    fn = _safe_sum(selected, "false_negatives")
    return pd.DataFrame(
        [
            {
                "metric": "step_savings",
                "available": True,
                "trajectories": int(original_total),
                "decided_trajectories": int(n_decided),
                "coverage_pct": n_decided * 100.0 / original_total if original_total else float("nan"),
                "saved_steps": int(saved_steps),
                "total_steps": int(total_steps),
                "save_pct": saved_steps * 100.0 / total_steps if total_steps else float("nan"),
                "decision_accuracy_pct": (tp + tn) * 100.0 / n_decided if n_decided else float("nan"),
                "resolve_rate_change_pp": (fp - fn) * 100.0 / original_total if original_total else float("nan"),
                "note": "Audited from main fold selected-policy CSVs.",
            }
        ]
    )


def _token_proxy(run_dir: Path, args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    token_dir = run_dir / "internal_review_swe16"
    summary_path = token_dir / "selected_strategy_token_summary.csv"
    decisions_path = token_dir / "selected_strategy_decisions_with_tokens.csv"
    if not summary_path.exists():
        return (
            pd.DataFrame(
                [
                    {
                        "metric": "estimated_tokens",
                        "available": False,
                        "source": str(summary_path),
                        "note": "No internal-review token proxy file found.",
                    }
                ]
            ),
            pd.DataFrame(),
        )
    summary = pd.read_csv(summary_path).iloc[0].to_dict()
    rows = [
        {
            "metric": "estimated_final_transcript_tokens",
            "available": True,
            "source": str(summary_path),
            "baseline_tokens": summary.get("baseline_final_transcript_tokens_est"),
            "policy_tokens": summary.get("policy_transcript_tokens_spent_est"),
            "saved_tokens": summary.get("transcript_tokens_saved_est"),
            "save_pct": summary.get("transcript_token_save_pct_est"),
            "note": "Proxy only: post-hoc local tokenizer counts from internal_review_swe16/tokenizer_manifest.csv, not audited API usage.",
        },
        {
            "metric": "estimated_context_call_tokens",
            "available": True,
            "source": str(summary_path),
            "baseline_tokens": summary.get("baseline_context_call_tokens_est"),
            "policy_tokens": summary.get("policy_context_call_tokens_spent_est"),
            "saved_tokens": summary.get("future_context_call_tokens_saved_est"),
            "save_pct": summary.get("context_call_token_save_pct_est"),
            "note": "Proxy only: future rounds treated as resending full prefix context.",
        },
    ]
    price = args.price_per_million_total_tokens
    if price is None and args.price_per_million_input_tokens is not None:
        price = args.price_per_million_input_tokens
    if price is not None:
        for row in rows:
            row["price_per_million_tokens"] = float(price)
            row["estimated_baseline_cost"] = float(row["baseline_tokens"]) / 1_000_000.0 * float(price)
            row["estimated_policy_cost"] = float(row["policy_tokens"]) / 1_000_000.0 * float(price)
            row["estimated_saved_cost"] = float(row["saved_tokens"]) / 1_000_000.0 * float(price)
    else:
        rows.append(
            {
                "metric": "api_price",
                "available": False,
                "source": "",
                "note": "Set --price-per-million-total-tokens or --price-per-million-input-tokens to compute a proxy dollar cost.",
            }
        )
    decisions = pd.read_csv(decisions_path) if decisions_path.exists() else pd.DataFrame()
    return pd.DataFrame(rows), decisions


def _benchmark_apply_policy(run_dir: Path, repeats: int) -> pd.DataFrame:
    policy = _main_policy()
    rows: list[dict[str, Any]] = []
    for fold_marker in sorted((run_dir / "folds").glob("*/_SUCCESS")):
        fold_dir = fold_marker.parent
        pred_path = fold_dir / "test_predictions_safe_stop.parquet"
        if not pred_path.exists():
            continue
        frame = pd.read_parquet(pred_path)
        prefix_rows = len(frame)
        traj_count = frame["traj_id"].nunique()
        timings = []
        for _ in range(max(1, int(repeats))):
            started = time.perf_counter()
            apply_policy(frame, policy)
            timings.append(time.perf_counter() - started)
        arr = np.array(timings, dtype=float)
        rows.append(
            {
                "fold_id": fold_dir.name,
                "prefix_rows": int(prefix_rows),
                "trajectories": int(traj_count),
                "repeats": int(len(arr)),
                "mean_apply_policy_ms": float(arr.mean() * 1000.0),
                "median_apply_policy_ms": float(np.median(arr) * 1000.0),
                "p95_apply_policy_ms": float(np.percentile(arr, 95) * 1000.0),
                "prefixes_per_second": float(prefix_rows / arr.mean()) if arr.mean() > 0 else float("nan"),
                "trajectories_per_second": float(traj_count / arr.mean()) if arr.mean() > 0 else float("nan"),
                "note": "Measures Python policy scan over stored probabilities, not feature extraction or LightGBM predict_proba.",
            }
        )
    return pd.DataFrame(rows)


def _write_readme(out_dir: Path, step: pd.DataFrame, token: pd.DataFrame, bench: pd.DataFrame) -> None:
    step_row = step.iloc[0].to_dict() if len(step) else {}
    lines = [
        "# Main LightGBM Latency / Cost Audit",
        "",
        "This audit uses completed main SWEVerify fold artifacts only.",
        "",
        "Important boundary: the stored prediction parquets do not contain audited wall-clock, API token, or price logs. Token/cost values here are proxy estimates when available.",
        "",
        "## Step Savings",
        "",
        f"- trajectories: `{int(step_row.get('trajectories', 0))}`",
        f"- saved steps: `{int(step_row.get('saved_steps', 0))}` / `{int(step_row.get('total_steps', 0))}` (`{_fmt(step_row.get('save_pct'))}%`)",
        f"- decision accuracy: `{_fmt(step_row.get('decision_accuracy_pct'))}%`",
        f"- resolve-rate change: `{_fmt(step_row.get('resolve_rate_change_pp'))}pp`",
        "",
        "## Token / Cost Proxy",
        "",
    ]
    for row in token.to_dict("records"):
        metric = row.get("metric")
        available = row.get("available")
        lines.append(f"- `{metric}`: available=`{available}`; save=`{_fmt(row.get('save_pct'))}%`; {row.get('note', '')}")
    if not bench.empty:
        weighted_ms = float((bench["mean_apply_policy_ms"] * bench["prefix_rows"]).sum() / bench["prefix_rows"].sum())
        prefixes_per_sec = float(bench["prefix_rows"].sum() / (bench["mean_apply_policy_ms"].sum() / 1000.0))
        lines.extend(
            [
                "",
                "## Offline Policy-Scan Timing",
                "",
                f"- folds benchmarked: `{len(bench)}`",
                f"- weighted mean apply-policy time: `{weighted_ms:.2f} ms/fold`",
                f"- aggregate throughput: `{prefixes_per_sec:.1f} prefixes/sec`",
                "",
                "This timing does not include feature extraction, model inference, or external agent runtime.",
            ]
        )
    lines.extend(
        [
            "",
            "## Files",
            "",
            "- `step_savings_summary.csv`",
            "- `token_cost_proxy_summary.csv`",
            "- `policy_scan_latency.csv`",
            "- `manifest.json`",
            "",
        ]
    )
    (out_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    out_dir = args.output_dir or (args.run_dir / "latency_cost")
    out_dir.mkdir(parents=True, exist_ok=True)
    selected = _load_selected(args.run_dir)
    if selected.empty:
        raise SystemExit(f"No selected-policy rows found under {args.run_dir}")
    step = _aggregate_step_summary(selected)
    token, decisions = _token_proxy(args.run_dir, args)
    bench = pd.DataFrame() if args.no_benchmark else _benchmark_apply_policy(args.run_dir, args.benchmark_repeats)
    step.to_csv(out_dir / "step_savings_summary.csv", index=False)
    token.to_csv(out_dir / "token_cost_proxy_summary.csv", index=False)
    bench.to_csv(out_dir / "policy_scan_latency.csv", index=False)
    if not decisions.empty:
        by_model = (
            decisions.groupby("test_model", sort=True)
            .agg(
                trajectories=("traj_id", "count"),
                decided_trajectories=("decided", "sum"),
                saved_steps=("saved_steps", "sum"),
                transcript_tokens_saved_est=("transcript_tokens_saved_est", "sum"),
                future_context_call_tokens_saved_est=("future_context_call_tokens_saved_est", "sum"),
            )
            .reset_index()
        )
        by_model.to_csv(out_dir / "token_proxy_by_model.csv", index=False)
    payload = {
        "ok": True,
        "run_dir": str(args.run_dir),
        "output_dir": str(out_dir),
        "step_rows": int(len(step)),
        "token_rows": int(len(token)),
        "latency_rows": int(len(bench)),
        "benchmark_repeats": int(args.benchmark_repeats),
        "has_audited_wall_clock": False,
        "has_audited_api_cost": False,
    }
    (out_dir / "manifest.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_readme(out_dir, step, token, bench)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
